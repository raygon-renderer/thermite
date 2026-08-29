use core::ops::Mul;

use thermite::{
    mask::GenericMask,
    math::{RealMath, TranscendentalMath},
    vector::FloatVector,
};

use super::{Bounds, Matrix, Point, Quaternion, Vector, vector::VectorOps as _};

/// A paired forward/inverse affine transform.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform<V: FloatVector> {
    pub forward: Matrix<V, 4, 4>,
    pub inverse: Matrix<V, 4, 4>,
}

/// Lane-wise select between two matrices.
///
/// Every constructor below that would branch in a scalar implementation instead
/// computes both alternatives and blends them here: one lane of a
/// `Transform<f32x8>` may be degenerate while the other seven are not, so a
/// branch could not serve them all anyway.
#[inline(always)]
fn select_matrix<V: FloatVector>(mask: V::Mask, t: Matrix<V, 4, 4>, f: Matrix<V, 4, 4>) -> Matrix<V, 4, 4> {
    let mut result = Matrix::splat(V::ZERO);

    for c in 0..4 {
        for r in 0..4 {
            result.0[c][r] = mask.select(t.0[c][r], f.0[c][r]);
        }

        unsafe { result.0[c][0].block_autovectorization() };
    }

    result
}

impl<V: FloatVector> Default for Transform<V> {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<V: FloatVector> Transform<V> {
    /// The identity transform in every lane.
    pub const IDENTITY: Self = Self {
        forward: Matrix::IDENTITY,
        inverse: Matrix::IDENTITY,
    };

    /// Pair a forward matrix with an inverse that is already known. Nothing is
    /// checked: the two are assumed to be mutual inverses.
    #[inline(always)]
    pub const fn from_raw(forward: Matrix<V, 4, 4>, inverse: Matrix<V, 4, 4>) -> Self {
        Self { forward, inverse }
    }

    /// Pair a forward matrix with its computed inverse.
    ///
    /// Singular lanes come back as infinities or NaN rather than an error, which
    /// is what makes this branch-free. Use [`try_new`](Self::try_new) when the
    /// caller needs to know which lanes were invertible.
    #[inline(always)]
    pub fn new(forward: Matrix<V, 4, 4>) -> Self {
        Self {
            inverse: forward.invert(),
            forward,
        }
    }

    /// Like [`new`](Self::new), plus a mask that is set for the lanes whose
    /// forward matrix was actually invertible. The other lanes hold garbage.
    #[inline(always)]
    pub fn try_new(forward: Matrix<V, 4, 4>) -> (Self, V::Mask) {
        let (inverse, ok) = forward.try_invert();

        (Self { forward, inverse }, ok)
    }

    /// Pair an inverse matrix with its computed forward. The mirror of
    /// [`new`](Self::new), and equally tolerant of singular lanes.
    #[inline(always)]
    pub fn new_inverse(inverse: Matrix<V, 4, 4>) -> Self {
        Self {
            forward: inverse.invert(),
            inverse,
        }
    }

    /// The reverse transform, for free: the pair is already both directions.
    #[inline(always)]
    pub const fn invert(&self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// A mask set for the lanes whose forward matrix is the identity to within
    /// `tolerance`.
    #[inline(always)]
    pub fn is_identity(&self, tolerance: V) -> V::Mask {
        self.forward.is_identity(tolerance)
    }

    /// A pure translation. The inverse is the negated translation, so no
    /// inversion is performed.
    #[inline(always)]
    pub fn translate(delta: Vector<V, 3>) -> Self {
        Self {
            forward: Matrix::from_translation(delta),
            inverse: Matrix::from_translation(-delta),
        }
    }

    /// A pure (non-uniform) scale about the origin. The inverse is the
    /// component-wise reciprocal, so no inversion is performed.
    ///
    /// A zero axis is not invertible. Rather than assert (which cannot work when
    /// only some lanes are zero), that axis is given a zero in the inverse: the
    /// transform stays finite and collapses the axis instead of poisoning every
    /// later computation with an infinity.
    #[inline(always)]
    pub fn scale(axes: Vector<V, 3>) -> Self {
        let mut rcp = Vector::ZERO;

        for i in 0..3 {
            let s = axes[i];

            rcp[i] = s.cmp_eq(V::ZERO).select(V::ZERO, V::ONE / s);
        }

        Self {
            forward: Matrix::from_scale(axes),
            inverse: Matrix::from_scale(rcp),
        }
    }

    /// A pure rotation, from a quaternion.
    ///
    /// A rotation matrix is orthogonal, so the inverse is exactly the transpose
    /// and no inversion is performed.
    #[inline(always)]
    pub fn from_quaternion(rotation: Quaternion<V>) -> Self {
        let forward = rotation.to_matrix();

        Self {
            inverse: forward.transpose(),
            forward,
        }
    }

    /// Transform a point by the forward matrix (the translation applies).
    #[inline(always)]
    pub fn transform_point(&self, p: Point<V, 3>) -> Point<V, 3> {
        self.forward.transform_point(p)
    }

    /// Transform a direction by the forward matrix (the translation does not
    /// apply).
    #[inline(always)]
    pub fn transform_vector(&self, v: Vector<V, 3>) -> Vector<V, 3> {
        self.forward.transform_vector(v)
    }

    /// Transform a surface normal into world space.
    ///
    /// A normal is not a direction: it is defined by the plane it is
    /// perpendicular to, and a non-uniform scale or shear tilts that plane the
    /// other way. Preserving `$\mathbf{n} \cdot \mathbf{v} = 0$` for every
    /// tangent `$\mathbf{v}$` under `$\mathbf{v}' = M\mathbf{v}$` requires the
    /// inverse-transpose, `$\mathbf{n}' = (M^{-1})^T \mathbf{n}$`, which is what
    /// the cached inverse makes cheap here. The result is not renormalized.
    #[inline(always)]
    pub fn transform_normal(&self, n: Vector<V, 3>) -> Vector<V, 3> {
        self.inverse.transpose().transform_vector(n)
    }

    /// Transform a surface normal from world space back into local space, by
    /// `$M^T$`, the inverse of the normal transform `$(M^{-1})^T$`.
    #[inline(always)]
    pub fn reverse_transform_normal(&self, n: Vector<V, 3>) -> Vector<V, 3> {
        self.forward.transpose().transform_vector(n)
    }

    /// The AABB of the transformed box (the union of its eight transformed
    /// corners).
    #[inline(always)]
    pub fn transform_bounds(&self, bounds: &Bounds<V, 3>) -> Bounds<V, 3> {
        self.forward.transform_bounds(bounds)
    }

    /// The transformed X, Y and Z axes.
    ///
    /// Transforming a basis vector selects one column of the matrix, so the
    /// columns of the linear part are the answer with no arithmetic at all.
    #[inline(always)]
    pub fn transform_axis(&self) -> (Vector<V, 3>, Vector<V, 3>, Vector<V, 3>) {
        let m = &self.forward.0;

        (
            Vector([m[0][0], m[0][1], m[0][2]]),
            Vector([m[1][0], m[1][1], m[1][2]]),
            Vector([m[2][0], m[2][1], m[2][2]]),
        )
    }
}

impl<V: FloatVector + TranscendentalMath> Transform<V> {
    /// Rotation by `angle` radians about a **pre-normalized** axis.
    ///
    /// A rotation matrix is orthogonal, so its inverse is exactly its transpose
    /// and no inversion is performed.
    #[inline(always)]
    pub fn rotate_raw(axis: Vector<V, 3>, angle: V) -> Self {
        let forward = Matrix::from_axis_angle(axis, angle);

        Self {
            inverse: forward.transpose(),
            forward,
        }
    }

    /// Orthographic projection mapping `z` from `[near, far]` onto `[0, 1]`.
    ///
    /// The inverse is the affine map back, `$z \mapsto z(far - near) + near$`,
    /// known in closed form.
    #[inline(always)]
    pub fn ortho(near: V, far: V) -> Self {
        let range = far - near;

        let mut inverse = Matrix::IDENTITY;

        inverse.0[2][2] = range;
        inverse.0[3][2] = near;

        Self {
            forward: Matrix::orthographic(near, far),
            inverse,
        }
    }

    /// Perspective projection with a vertical field of view (radians), mapping
    /// `z` from `[near, far]` onto `[0, 1]` and scaling `xy` by
    /// `$1/\tan(fov/2)$`.
    ///
    /// The forward matrix is block-diagonal, so its inverse is too: the `xy`
    /// block inverts to `$\tan(fov/2)$` (the tangent is already computed) and the
    /// `zw` block is a 2x2 whose inverse is written out below. That is cheaper
    /// and more accurate than a general 4x4 inversion.
    #[inline(always)]
    pub fn perspective(fov: V, near: V, far: V) -> Self {
        let tan_half = (fov * V::HALF).tan();

        let inv_near = V::ONE / near;
        // 1 / (-far * near / (far - near)), the reciprocal of the forward (2, 3) entry.
        let inv_depth = (near - far) / (far * near);

        let inverse = Matrix([
            [tan_half, V::ZERO, V::ZERO, V::ZERO],
            [V::ZERO, tan_half, V::ZERO, V::ZERO],
            [V::ZERO, V::ZERO, V::ZERO, inv_depth],
            [V::ZERO, V::ZERO, V::ONE, inv_near],
        ]);

        Self {
            forward: Matrix::perspective(fov, near, far),
            inverse,
        }
    }
}

impl<V: RealMath> Transform<V> {
    /// Rotation encoded as an axis-angle vector: the direction is the axis and
    /// the magnitude is the angle in radians.
    ///
    /// A zero-magnitude vector names no axis. Its lanes take the identity through
    /// a select, because with one rotation per lane there is no branch that could
    /// answer for all of them. The arithmetic runs regardless and the NaN axis it
    /// produces is discarded by the blend.
    #[inline(always)]
    pub fn rotate(axis_angle: Vector<V, 3>) -> Self {
        let (axis, angle) = axis_angle.normalize_norm();

        let ok = angle.cmp_gt(V::ZERO) & angle.is_finite();

        let rotation = Self::rotate_raw(axis, angle);

        Self {
            forward: select_matrix(ok, rotation.forward, Matrix::IDENTITY),
            inverse: select_matrix(ok, rotation.inverse, Matrix::IDENTITY),
        }
    }

    /// A view transform from an eye position, a view direction, and a world-up
    /// hint.
    ///
    /// `forward` maps world space into camera space and `inverse` maps camera
    /// space back out into the world. The camera looks down its own `+Z`, with
    /// `+X` right and `+Y` up. Lanes whose `dir` is degenerate, or whose `up` is
    /// parallel to `dir` (so that no right vector exists), take the identity via
    /// a select rather than a branch.
    ///
    /// The camera-to-world basis is orthonormal, so the world-to-camera matrix is
    /// its transpose with the origin projected onto each axis, and no inversion
    /// is performed.
    #[inline(always)]
    pub fn look_at_dir(origin: Point<V, 3>, dir: Vector<V, 3>, up: Vector<V, 3>) -> Self {
        let (dir, dir_norm) = dir.normalize_norm();

        let (right, right_norm) = up.cross(dir).normalize_norm();

        let ok = dir_norm.cmp_gt(V::ZERO) & dir_norm.is_finite() & right_norm.cmp_gt(V::ZERO) & right_norm.is_finite();

        // Re-derived rather than taken from `up`, which is only a hint and need
        // not be perpendicular to `dir`.
        let new_up = dir.cross(right);

        let o = Vector::from(origin);

        let cam_to_world = Matrix::from_basis(right, new_up, dir, o);

        // The transpose of an orthonormal basis is its inverse, and the translation
        // becomes the negated origin expressed in that basis.
        let world_to_cam = Matrix::from_basis(
            Vector([right[0], new_up[0], dir[0]]),
            Vector([right[1], new_up[1], dir[1]]),
            Vector([right[2], new_up[2], dir[2]]),
            Vector([-right.dot(&o), -new_up.dot(&o), -dir.dot(&o)]),
        );

        Self {
            forward: select_matrix(ok, world_to_cam, Matrix::IDENTITY),
            inverse: select_matrix(ok, cam_to_world, Matrix::IDENTITY),
        }
    }

    /// A view transform aimed at a target point. See
    /// [`look_at_dir`](Self::look_at_dir).
    #[inline(always)]
    pub fn look_at(origin: Point<V, 3>, target: Point<V, 3>, up: Vector<V, 3>) -> Self {
        Self::look_at_dir(origin, target - origin, up)
    }
}

/// Composition, `$T_\text{self} \circ T_\text{rhs}$`: `rhs` applies first.
///
/// The inverses multiply in the opposite order because
/// `$(AB)^{-1} = B^{-1}A^{-1}$`.
impl<V: FloatVector> Mul for Transform<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            forward: self.forward * rhs.forward,
            inverse: rhs.inverse * self.inverse,
        }
    }
}

/// The orthogonal (rotation) factor of a 3x3 matrix, by polar decomposition.
///
/// Any invertible `$M$` factors uniquely as `$M = RS$` with `$R$` orthogonal and
/// `$S$` symmetric positive-definite. `$R$` is the *closest* rotation to `$M$` in
/// the least-squares sense, which is exactly what "strip the scale and shear,
/// keep the orientation" means.
///
/// The iteration is `$R_{k+1} = \tfrac{1}{2}(R_k + (R_k^{-1})^T)$` from
/// `$R_0 = M$`. An orthogonal matrix satisfies `$R^T = R^{-1}$`, so it is a fixed
/// point of that average, and anything else is dragged toward one. This is
/// Newton's method for the orthogonal polar factor and converges quadratically,
/// so a well-conditioned input settles in a handful of steps.
///
/// The whole batch shares one trip count, so the loop runs until *every* lane has
/// converged (or the cap is hit). Branching on `.all()` is legitimate here, since it
/// is a reduction over the batch, not a decision about one lane's data.
#[inline(always)]
fn polar_rotation<V: FloatVector>(m: Matrix<V, 3, 3>) -> Matrix<V, 3, 3> {
    // Convergence is quadratic, so reaching this cap means the input was
    // singular or wildly ill-conditioned, which `determinant` should have
    // caught before calling.
    const MAX_ITERATIONS: usize = 100;

    let tolerance: V = thermite::const_splat!(ratio <V::Element>: 1, 10_000);

    let mut r = m;

    for _ in 0..MAX_ITERATIONS {
        let next = (r + r.invert().transpose()) * V::HALF;

        // The largest entry-wise change: cheap, and monotone enough to stop on.
        let mut delta = (next.0[0][0] - r.0[0][0]).abs();

        for c in 0..3 {
            for row in 0..3 {
                delta = delta.max((next.0[c][row] - r.0[c][row]).abs());
            }
        }

        r = next;

        if delta.cmp_lt(tolerance).all() {
            break;
        }
    }

    r
}

impl<V: FloatVector> Transform<V> {
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
    /// collapses and re-inflates. The correct path is to decompose both endpoints,
    /// [`slerp`](Quaternion::slerp) the rotations, lerp the translations and
    /// scales, and recompose, which is what a keyframed or motion-blurred
    /// transform does at every sample.
    ///
    /// It also answers questions a raw matrix cannot: whether an imported
    /// transform carries a scale, whether it **mirrors** (a negative determinant
    /// flips winding order and normal handedness), and what to display on the
    /// separate translate/rotate/scale handles of an editor gizmo.
    ///
    /// Recompose with
    /// `Matrix::from_translation(t) * rotation.to_matrix() * scale.to_homogeneous()`.
    ///
    /// A singular linear part cannot be factored, so those lanes come back with
    /// whatever the iteration reached, so test
    /// [`determinant`](Matrix::determinant) first when the input is untrusted.
    #[inline(always)]
    pub fn decompose(&self) -> (Vector<V, 3>, Quaternion<V>, Matrix<V, 3, 3>) {
        let translation = self.forward.translation();

        let linear = self.forward.linear();
        let rotation = polar_rotation(linear);

        // A mirroring transform has a polar factor with determinant -1: an
        // improper rotation, i.e. a reflection, which no quaternion can
        // represent. Negating it flips the determinant (for a 3x3,
        // `det(-R) = -det(R)`) and makes it a genuine rotation. Negating the
        // scale in step keeps `R * S` unchanged, so the mirror lands in the
        // scale factor where a caller can actually see it.
        let flip = rotation.determinant().cmp_lt(V::ZERO);
        let sign = flip.select(V::NEG_ONE, V::ONE);

        let rotation = rotation * sign;

        // Whatever the rotation did not account for.
        let scale = rotation.invert() * linear;

        (translation, Quaternion::from_matrix(&rotation.to_homogeneous()), scale)
    }
}
