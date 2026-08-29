//! A paired forward/inverse affine transform.

use core::ops::Mul;

use thermite::{math::ScalarMath, prelude::*, simd::Simd3Vectors};

use super::{AosFloat, Bounds3, Matrix4, Quaternion, V3, vector::Vector3Ext as _};

/// A forward matrix paired with its inverse.
///
/// Keeping both means the hot path never inverts: a normal transform needs
/// `$(M^{-1})^T$` and an instance transform needs `$M^{-1}$` to bring a ray into
/// object space, and both are one field access away. Every constructor below
/// whose inverse has a closed form supplies it directly rather than calling
/// [`Matrix4::invert`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform<S: Simd3Vectors, E: AosFloat<S>> {
    pub forward: Matrix4<S, E>,
    pub inverse: Matrix4<S, E>,
}

impl<S: Simd3Vectors, E: AosFloat<S>> Default for Transform<S, E> {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Transform<S, E> {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        forward: Matrix4::IDENTITY,
        inverse: Matrix4::IDENTITY,
    };

    /// Pairs a forward matrix with an inverse that is already known.
    ///
    /// Nothing is checked: the two are **assumed** to be mutual inverses.
    #[inline(always)]
    pub const fn from_raw(forward: Matrix4<S, E>, inverse: Matrix4<S, E>) -> Self {
        Self { forward, inverse }
    }

    /// Pairs a forward matrix with its computed inverse, or `None` when the
    /// matrix is singular.
    #[inline(always)]
    pub fn new(forward: Matrix4<S, E>) -> Option<Self> {
        forward.invert().map(|inverse| Self { forward, inverse })
    }

    /// Pairs an inverse matrix with its computed forward. The mirror of
    /// [`new`](Self::new).
    #[inline(always)]
    pub fn new_inverse(inverse: Matrix4<S, E>) -> Option<Self> {
        inverse.invert().map(|forward| Self { forward, inverse })
    }

    /// The reverse transform, for free: the pair is already both directions.
    #[inline(always)]
    pub const fn invert(&self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// True when the forward matrix is the identity to within `tolerance`.
    #[inline(always)]
    pub fn is_identity(&self, tolerance: E) -> bool {
        self.forward.is_identity(tolerance)
    }

    /// A pure translation. The inverse is the negated translation, so nothing is
    /// inverted.
    #[inline(always)]
    pub fn translate(delta: V3<S, E>) -> Self {
        Self {
            forward: Matrix4::from_translation(delta),
            inverse: Matrix4::from_translation(-delta),
        }
    }

    /// A pure (non-uniform) scale about the origin. The inverse is the
    /// component-wise reciprocal, so nothing is inverted.
    ///
    /// A zero axis is not invertible, so that axis gets a zero in the inverse,
    /// which collapses it rather than poisoning the matrix with infinities.
    #[inline(always)]
    pub fn scale(axes: V3<S, E>) -> Self {
        let zero = <V3<S, E> as NumericVector>::ZERO;

        let rcp = axes
            .cmp_eq(zero)
            .select(zero, <V3<S, E> as NumericVector>::ONE / axes);

        Self {
            forward: Matrix4::from_scale(axes),
            inverse: Matrix4::from_scale(rcp),
        }
    }

    /// A pure rotation from a quaternion.
    ///
    /// A rotation matrix is orthogonal, so the inverse is exactly the transpose
    /// and nothing is inverted.
    #[inline(always)]
    pub fn from_quaternion(rotation: Quaternion<S, E>) -> Self {
        let forward = rotation.to_matrix();

        Self {
            inverse: forward.transpose(),
            forward,
        }
    }

    /// Transforms a point by the forward matrix (translation applies).
    #[inline(always)]
    pub fn transform_point(&self, p: V3<S, E>) -> V3<S, E> {
        self.forward.transform_point(p)
    }

    /// Transforms a direction by the forward matrix (translation does not apply).
    #[inline(always)]
    pub fn transform_vector(&self, v: V3<S, E>) -> V3<S, E> {
        self.forward.transform_vector(v)
    }

    /// Transforms a surface normal into world space by `$(M^{-1})^T$`.
    ///
    /// A normal is defined by the plane it is perpendicular to, and a
    /// non-uniform scale or shear tilts that plane the other way, so preserving
    /// `$\mathbf{n}\cdot\mathbf{v} = 0$` for every tangent needs the
    /// inverse-transpose, which the cached inverse makes free here, unlike
    /// [`Matrix4::transform_normal`] which has to build the cofactor matrix.
    ///
    /// The result is **not** renormalized.
    #[inline(always)]
    pub fn transform_normal(&self, n: V3<S, E>) -> V3<S, E> {
        // (M^-1)^T n == the 3x3 of M^-1 applied as ROWS, which is exactly what the
        // row-major form of the product does. No explicit transpose needed.
        n.mat3_vec3_product::<false>(&self.inverse.linear())
    }

    /// Transforms a surface normal from world space back into local space, by
    /// `$M^T$`, the inverse of the normal transform `$(M^{-1})^T$`.
    #[inline(always)]
    pub fn reverse_transform_normal(&self, n: V3<S, E>) -> V3<S, E> {
        n.mat3_vec3_product::<false>(&self.forward.linear())
    }

    /// The AABB of the transformed box.
    #[inline(always)]
    pub fn transform_bounds(&self, bounds: &Bounds3<S, E>) -> Bounds3<S, E> {
        bounds.transform(&self.forward)
    }

    /// The transformed X, Y and Z axes.
    ///
    /// Transforming a basis vector selects one column, so these are the columns
    /// of the linear part with no arithmetic at all.
    #[inline(always)]
    pub fn transform_axis(&self) -> (V3<S, E>, V3<S, E>, V3<S, E>) {
        let [x, y, z] = self.forward.linear();

        (x, y, z)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S> + ScalarMath> Transform<S, E> {
    /// Rotation by `angle` radians about a **pre-normalized** axis.
    ///
    /// Built through a quaternion, whose expansion to a matrix is trig-free
    /// after the one `sin_cos`, and whose inverse is the transpose.
    #[inline(always)]
    pub fn rotate_raw(axis: V3<S, E>, angle: E) -> Self {
        Self::from_quaternion(Quaternion::from_axis_angle_raw(axis, angle))
    }

    /// Rotation encoded as an axis-angle vector: the direction is the axis and
    /// the magnitude is the angle in radians. A zero vector gives the identity.
    #[inline(always)]
    pub fn rotate(axis_angle: V3<S, E>) -> Self {
        Self::from_quaternion(Quaternion::from_axis_angle(axis_angle))
    }

    /// A view transform from an eye position, a view direction and a world-up
    /// hint.
    ///
    /// `forward` maps world space into camera space and `inverse` maps back. The
    /// camera looks down its own `+Z`, with `+X` right and `+Y` up. A degenerate
    /// `dir`, or an `up` parallel to it, yields the identity.
    ///
    /// The camera-to-world basis is orthonormal, so world-to-camera is its
    /// transpose with the origin projected onto each axis. No inversion.
    #[inline(always)]
    pub fn look_at_dir(origin: V3<S, E>, dir: V3<S, E>, up: V3<S, E>) -> Self {
        let (dir, dir_len) = dir.normalize_norm();
        let (right, right_len) = up.cross3::<false>(dir).normalize_norm();

        // A zero-length view direction names no orientation, and an `up` parallel
        // to it leaves no right vector, so both degenerate to the identity.
        let usable = dir_len > E::ZERO && right_len > E::ZERO && dir.is_all_finite() && right.is_all_finite();

        if !usable {
            return Self::IDENTITY;
        }

        // Re-derived rather than taken from `up`, which is only a hint and need
        // not be perpendicular to `dir`.
        let new_up = dir.cross3::<false>(right);

        let cam_to_world = Matrix4::from_basis(right, new_up, dir, origin);

        // The transpose of an orthonormal basis is its inverse, and the translation
        // becomes the negated origin expressed in that basis.
        let world_to_cam = Matrix4::from_basis(
            V3::<S, E>::from_slice(&[right.extract::<0>(), new_up.extract::<0>(), dir.extract::<0>()]),
            V3::<S, E>::from_slice(&[right.extract::<1>(), new_up.extract::<1>(), dir.extract::<1>()]),
            V3::<S, E>::from_slice(&[right.extract::<2>(), new_up.extract::<2>(), dir.extract::<2>()]),
            V3::<S, E>::from_slice(&[-right.dot3(origin), -new_up.dot3(origin), -dir.dot3(origin)]),
        );

        Self {
            forward: world_to_cam,
            inverse: cam_to_world,
        }
    }

    /// A view transform aimed at a target point. See
    /// [`look_at_dir`](Self::look_at_dir).
    #[inline(always)]
    pub fn look_at(origin: V3<S, E>, target: V3<S, E>, up: V3<S, E>) -> Self {
        Self::look_at_dir(origin, target - origin, up)
    }

    /// Orthographic projection mapping `z` from `[near, far]` onto `[0, 1]`.
    ///
    /// The inverse is the affine map back, known in closed form.
    #[inline(always)]
    pub fn ortho(near: E, far: E) -> Self {
        let range = far - near;
        let inv_range = E::ONE / range;

        let mut forward = Matrix4::IDENTITY;
        forward.set(2, 2, inv_range);
        forward.set(2, 3, -near * inv_range);

        let mut inverse = Matrix4::IDENTITY;
        inverse.set(2, 2, range);
        inverse.set(2, 3, near);

        Self { forward, inverse }
    }

    /// Perspective projection with a vertical field of view (radians), mapping
    /// `z` from `[near, far]` onto `[0, 1]` and scaling `xy` by
    /// `$1/\tan(fov/2)$`.
    ///
    /// The forward matrix is block-diagonal, so its inverse is too: the `xy`
    /// block inverts to `$\tan(fov/2)$` (already computed) and the `zw` block is
    /// a 2x2 written out below. Cheaper and more accurate than a general 4x4
    /// inversion.
    #[inline(always)]
    pub fn perspective(fov: E, near: E, far: E) -> Self {
        let tan_half = (fov * E::from_ratio(1, 2)).scalar_tan();
        let inv_tan = E::ONE / tan_half;

        let range = far - near;

        let mut forward = Matrix4::ZERO;
        forward.set(0, 0, inv_tan);
        forward.set(1, 1, inv_tan);
        forward.set(2, 2, far / range);
        forward.set(3, 2, E::ONE);
        forward.set(2, 3, -(far * near) / range);

        let mut inverse = Matrix4::ZERO;
        inverse.set(0, 0, tan_half);
        inverse.set(1, 1, tan_half);
        // 1 / (-far * near / range), the reciprocal of the forward (2, 3) entry.
        inverse.set(3, 2, (near - far) / (far * near));
        inverse.set(2, 3, E::ONE);
        inverse.set(3, 3, E::ONE / near);

        Self { forward, inverse }
    }
}

/// Composition `$T_\text{self} \circ T_\text{rhs}$`: `rhs` applies first.
///
/// The inverses multiply in the opposite order, since `$(AB)^{-1} = B^{-1}A^{-1}$`.
impl<S: Simd3Vectors, E: AosFloat<S>> Mul for Transform<S, E> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            forward: self.forward * rhs.forward,
            inverse: rhs.inverse * self.inverse,
        }
    }
}

/// What [`Transform::decompose`] hands back: the translation, the rotation, and
/// the scale/shear as three columns (the same shape
/// [`Matrix4::linear`](super::Matrix4::linear) returns).
pub type Decomposed<S, E> = (V3<S, E>, Quaternion<S, E>, [V3<S, E>; 3]);

/// The orthogonal (rotation) factor of a 3x3, by polar decomposition.
///
/// See the SoA twin, [`soa::prim::Transform::decompose`](crate::soa::prim::Transform::decompose),
/// for the full derivation. In short: any invertible `$M$` factors uniquely as
/// `$M = RS$` with `$R$` orthogonal and `$S$` symmetric positive-definite, and
/// the iteration `$R_{k+1} = \tfrac{1}{2}(R_k + (R_k^{-1})^T)$` converges
/// quadratically to that `$R$`, since an orthogonal matrix has `$R^T = R^{-1}$` and
/// it is the fixed point of that average.
///
/// One matrix per call, so this exits the moment *this* matrix has converged,
/// where the SoA form has to keep the whole batch iterating until its slowest
/// lane is done.
#[inline(always)]
fn polar_rotation<S: Simd3Vectors, E: AosFloat<S>>(m: [V3<S, E>; 3]) -> [V3<S, E>; 3] {
    const MAX_ITERATIONS: usize = 100;

    let tolerance = E::from_ratio(1, 10_000);

    let mut r = m;

    for _ in 0..MAX_ITERATIONS {
        let Some(inv) = V3::<S, E>::mat3_inverse::<false>(&r) else {
            // Singular: no orthogonal factor to converge to. Hand back what we
            // have and let the caller's determinant check speak.
            break;
        };

        let inv_t = V3::<S, E>::mat3_transpose(&inv);

        let half = V3::<S, E>::splat(E::from_ratio(1, 2));

        let next: [V3<S, E>; 3] = core::array::from_fn(|i| (r[i] + inv_t[i]) * half);

        // Largest entry-wise change across the whole matrix.
        let mut delta = E::ZERO;

        for i in 0..3 {
            let d = (next[i] - r[i]).abs().max_element3();

            if d > delta {
                delta = d;
            }
        }

        r = next;

        if delta < tolerance {
            break;
        }
    }

    r
}

impl<S: Simd3Vectors, E: AosFloat<S>> Transform<S, E> {
    /// Splits an affine transform into translation, rotation and scale/shear.
    ///
    /// A 4x4 stores those three mixed together across 16 numbers, and this recovers
    /// them separately, as `$M = T \cdot R \cdot S$`. The rotation is the
    /// orthogonal polar factor, the closest true rotation to the matrix's
    /// linear part, and `$S = R^{-1}M$` is whatever remains, a pure scale for a
    /// well-behaved transform and a scale-plus-shear otherwise.
    ///
    /// # Why you would want this
    ///
    /// Chiefly to **interpolate transforms**. Two matrices cannot simply be
    /// lerped: the halfway point between two rotation matrices is not a rotation.
    /// It shrinks toward the origin, so an object animated that way visibly
    /// collapses and re-inflates. The correct path is to decompose both
    /// endpoints, [`slerp`](super::Quaternion::slerp) the rotations, lerp the
    /// translations and scales, and recompose, which is what a keyframed or
    /// motion-blurred transform does at every sample.
    ///
    /// It also answers questions a raw matrix cannot: whether an imported
    /// transform carries a scale, whether it **mirrors** (a negative determinant
    /// flips winding order and normal handedness), and what to display on the
    /// separate translate/rotate/scale handles of an editor gizmo.
    ///
    /// The scale comes back as three columns, the same shape
    /// [`Matrix4::linear`](super::Matrix4::linear) returns.
    #[inline(always)]
    pub fn decompose(&self) -> Decomposed<S, E> {
        let translation = self.forward.translation();

        let linear = self.forward.linear();
        let rotation = polar_rotation::<S, E>(linear);

        // A mirroring transform has a polar factor with determinant -1: an
        // improper rotation, i.e. a reflection, which no quaternion can
        // represent. Negating it flips the determinant (for a 3x3,
        // `det(-R) = -det(R)`) and makes it a genuine rotation. Because
        // `M = R S = (-R)(-S)`, the mirror simply moves into the scale factor
        // where a caller can actually see it.
        let rotation = if V3::<S, E>::mat3_det::<false>(&rotation) < E::ZERO {
            rotation.map(|c| -c)
        } else {
            rotation
        };

        // Whatever the rotation did not account for: S = R^-1 M.
        let scale = match V3::<S, E>::mat3_inverse::<false>(&rotation) {
            Some(inv) => V3::<S, E>::mat3_product::<true>(&inv, &linear),
            None => linear,
        };

        // Widen the rotation into a homogeneous matrix for Shepperd's extraction.
        let mut rot4 = Matrix4::<S, E>::IDENTITY;

        for (dst, src) in rot4.0.iter_mut().zip(&rotation) {
            *dst = src.extend::<super::V4<S, E>>().zero4();
        }

        (translation, Quaternion::from_matrix(&rot4), scale)
    }
}
