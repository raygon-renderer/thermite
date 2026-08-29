//! Column-major 4x4 matrix, one matrix per four SIMD registers.
//!
//! Every operation here delegates to a
//! [`LinAlg4Vector`](thermite::vector::LinAlg4Vector) primitive rather than
//! being open-coded, because those are what the backends specialize: the
//! transpose is a shuffle network, the inverse is a cofactor expansion tuned per
//! ISA, and the point transform is the FMA chain described on
//! [`transform_point`](Matrix4::transform_point).

use core::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use thermite::{
    element::Element,
    generic_array::{GenericArray, typenum::U4},
    prelude::*,
    simd::Simd3Vectors,
    vector::{NewConst, const_new},
};

use super::{AosFloat, V3, V4};

/// Compile-time carrier for column `I` of the identity matrix.
///
/// The [`const_new!`](thermite::const_new) macro only takes literal element
/// values, which a generic `E` cannot supply, so the carrier the macro would
/// have generated is written out by hand here. `Element` provides `ZERO`/`ONE`
/// as plain associated consts, so the whole column is built in a `const` block
/// and [`Matrix4::IDENTITY`] really is a constant, compiling to four aligned
/// loads from rodata, with no arithmetic at runtime.
struct IdentityColumn<E, const I: usize>(PhantomData<E>);

impl<E: Element, const I: usize> NewConst<E, U4> for IdentityColumn<E, I> {
    const VALUES: GenericArray<E, U4> = {
        let mut column = [E::ZERO; 4];

        column[I] = E::ONE;

        GenericArray::from_array(column)
    };
}

/// Column-major `$4 \times 4$` matrix, stored as four column registers.
///
/// `self.0[c]` is column `c`; element `$(r, c)$` is `self.0[c].extractv(r)`.
/// Column-major is the storage every `LinAlg4Vector` primitive prefers, since a
/// matrix-vector product becomes a broadcast-and-FMA over the columns with no
/// transpose, which is why the row-major forms of those primitives cost extra.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix4<S: Simd3Vectors, E: AosFloat<S>>(pub [V4<S, E>; 4]);

impl<S: Simd3Vectors, E: AosFloat<S>> Default for Matrix4<S, E> {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Matrix4<S, E> {
    /// The all-zero matrix.
    pub const ZERO: Self = Self([<V4<S, E> as NumericVector>::ZERO; 4]);

    /// The multiplicative identity.
    pub const IDENTITY: Self = Self([
        const_new::<V4<S, E>, U4, IdentityColumn<E, 0>>(),
        const_new::<V4<S, E>, U4, IdentityColumn<E, 1>>(),
        const_new::<V4<S, E>, U4, IdentityColumn<E, 2>>(),
        const_new::<V4<S, E>, U4, IdentityColumn<E, 3>>(),
    ]);

    /// From four explicit columns.
    #[inline(always)]
    pub const fn from_columns(columns: [V4<S, E>; 4]) -> Self {
        Self(columns)
    }

    /// From 16 scalars in **column-major** order: the first four are column 0.
    #[inline(always)]
    pub fn from_array(m: [E; 16]) -> Self {
        Self(core::array::from_fn(|c| V4::<S, E>::from_slice(&m[c * 4..c * 4 + 4])))
    }

    /// Element `$(row, col)$`.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> E {
        self.0[col].extractv(row)
    }

    /// Sets element `$(row, col)$`.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, value: E) {
        self.0[col] = self.0[col].insertv(row, value);
    }

    /// The transpose.
    #[inline(always)]
    pub fn transpose(&self) -> Self {
        Self(V4::<S, E>::mat4_transpose(&self.0))
    }

    /// The determinant.
    #[inline(always)]
    pub fn determinant(&self) -> E {
        V4::<S, E>::mat4_det::<false>(&self.0)
    }

    /// The inverse, or `None` when the matrix is exactly singular.
    ///
    /// One object per call means a real branch is the right answer here, unlike
    /// the SoA counterpart which must hand back a mask because its lanes can
    /// disagree.
    ///
    /// Only an *exactly* zero determinant is rejected, so an ill-conditioned matrix
    /// returns a finite but unreliable inverse. Use
    /// [`invert_det`](Self::invert_det) and test the determinant against your own
    /// tolerance when that matters.
    #[inline(always)]
    pub fn invert(&self) -> Option<Self> {
        V4::<S, E>::mat4_inverse::<false>(&self.0).map(Self)
    }

    /// The inverse **and** the determinant, which the inversion computes anyway.
    ///
    /// The matrix is returned untouched when the determinant is exactly zero.
    #[inline(always)]
    pub fn invert_det(&self) -> (Self, E) {
        let mut m = self.0;

        let det = V4::<S, E>::mat4_inverse_inplace::<false>(&mut m);

        (Self(m), det)
    }

    /// The upper-left 3x3 block as three 3-lane columns: the linear part of an
    /// affine transform, with the translation column dropped.
    #[inline(always)]
    pub fn linear(&self) -> [V3<S, E>; 3] {
        [
            GenericVector::narrow(self.0[0]),
            GenericVector::narrow(self.0[1]),
            GenericVector::narrow(self.0[2]),
        ]
    }

    /// The translation column of an affine transform.
    #[inline(always)]
    pub fn translation(&self) -> V3<S, E> {
        GenericVector::narrow(self.0[3])
    }

    /// True when the matrix is the identity to within `tolerance`.
    #[inline(always)]
    pub fn is_identity(&self, tolerance: E) -> bool {
        let eps = V4::<S, E>::splat(tolerance);
        let id = Self::IDENTITY;

        let mut ok = true;

        for c in 0..4 {
            ok &= (self.0[c] - id.0[c]).abs().cmp_le(eps).all();
        }

        ok
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Matrix4<S, E> {
    /// Transforms a **point** by an affine matrix: `$M[p, 1]^T$`, dropping `w`.
    ///
    /// This is the hot path and the reason the AoS layout earns its place. It
    /// forwards to
    /// [`mat4_point3_product`](thermite::vector::LinAlg4Vector::mat4_point3_product),
    /// which knows the input's `w` is 1 and the output's `w` is unwanted, so the
    /// translation column becomes the addend of the first FMA and `w` is never
    /// computed at all. On AVX2 the whole transform is three broadcasts and
    /// three FMAs, and the widen and narrow around it cost nothing.
    ///
    /// Correct only for affine matrices (last row `$[0,0,0,1]$`). Use
    /// [`project_point`](Self::project_point) for a projective one.
    #[inline(always)]
    pub fn transform_point(&self, p: V3<S, E>) -> V3<S, E> {
        let p4: V4<S, E> = p.extend();

        GenericVector::narrow(p4.mat4_point3_product::<true>(&self.0))
    }

    /// Transforms a **direction**: `$M[v, 0]^T$`, so translation does not apply.
    ///
    /// Uses the 3x3 block rather than a 4x4 product with `w = 0`, which skips the
    /// fourth column and the fourth row of work entirely.
    #[inline(always)]
    pub fn transform_vector(&self, v: V3<S, E>) -> V3<S, E> {
        v.mat3_vec3_product::<true>(&self.linear())
    }

    /// Transforms a **normal** by the inverse-transpose of the 3x3 block.
    ///
    /// A normal is defined by the plane it is perpendicular to, and a non-uniform
    /// scale or shear tilts that plane the other way, so a normal does not
    /// transform like a direction. `DIVIDE = false` uses the un-divided cofactor
    /// matrix, which is cheaper, never singular, and points the same direction, all of
    /// which is what matters when the result is renormalized. Set `DIVIDE = true` for
    /// the true `$(M^{-1})^T$` when the magnitude matters.
    ///
    /// The result is **not** renormalized.
    #[inline(always)]
    pub fn transform_normal<const DIVIDE: bool>(&self, n: V3<S, E>) -> V3<S, E> {
        n.mat3_vec3_product::<true>(&V3::<S, E>::mat3_normal::<DIVIDE, false>(&self.linear()))
    }

    /// Transforms a point by a **projective** matrix: `$M[p,1]^T$` then divides
    /// `xyz` by the resulting `w`.
    ///
    /// Correct for any matrix, including perspective. Costs a full 4-lane product
    /// plus a reciprocal more than [`transform_point`](Self::transform_point), so
    /// prefer that one whenever the matrix is known to be affine.
    #[inline(always)]
    pub fn project_point(&self, p: V3<S, E>) -> V3<S, E> {
        let p4: V4<S, E> = p.extend();
        let out = p4.one4().mat4_vec4_product::<true>(&self.0);

        // Broadcast w and divide once. The 4th lane is discarded by the narrow.
        let w = out.broadcast::<3>();

        GenericVector::narrow(out / w)
    }
}

/// `M * point`, the affine transform.
impl<S: Simd3Vectors, E: AosFloat<S>> Matrix4<S, E> {
    /// Transforms `N` points by this matrix in one call.
    ///
    /// Intended for **small, fixed** `N` (the eight corners of a box, a triangle's
    /// three vertices): the array is taken and returned by value and the loop
    /// fully unrolls. Backends with a double-width register transform two points
    /// per pass, which a hand-written loop over
    /// [`transform_point`](Self::transform_point) would not get.
    #[inline(always)]
    pub fn transform_points<const N: usize>(&self, points: [V3<S, E>; N]) -> [V3<S, E>; N] {
        let wide: [V4<S, E>; N] = points.map(GenericVector::extend);

        let out = V4::<S, E>::mat4_point3_product_array::<true, N>(&self.0, &wide);

        out.map(GenericVector::narrow)
    }

    /// Transforms `N` directions by this matrix in one call. See
    /// [`transform_points`](Self::transform_points).
    #[inline(always)]
    pub fn transform_vectors<const N: usize>(&self, vectors: [V3<S, E>; N]) -> [V3<S, E>; N] {
        let m3 = self.linear();

        V3::<S, E>::mat3_vec3_product_array::<true, N>(&m3, &vectors)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Matrix4<S, E> {
    /// A pure translation.
    #[inline(always)]
    pub fn from_translation(delta: V3<S, E>) -> Self {
        let mut m = Self::IDENTITY;

        m.0[3] = delta.extend::<V4<S, E>>().one4();

        m
    }

    /// A pure (non-uniform) scale about the origin.
    #[inline(always)]
    pub fn from_scale(scale: V3<S, E>) -> Self {
        let mut m = Self::IDENTITY;

        m.0[0] = m.0[0].insert::<0>(scale.extract::<0>());
        m.0[1] = m.0[1].insert::<1>(scale.extract::<1>());
        m.0[2] = m.0[2].insert::<2>(scale.extract::<2>());

        m
    }

    /// Assembles an affine transform from three basis columns and a translation.
    #[inline(always)]
    pub fn from_basis(x: V3<S, E>, y: V3<S, E>, z: V3<S, E>, translation: V3<S, E>) -> Self {
        Self([
            x.extend::<V4<S, E>>().zero4(),
            y.extend::<V4<S, E>>().zero4(),
            z.extend::<V4<S, E>>().zero4(),
            translation.extend::<V4<S, E>>().one4(),
        ])
    }

    /// The sum of the diagonal entries.
    #[inline(always)]
    pub fn trace(&self) -> E {
        (self.get(0, 0) + self.get(1, 1)) + (self.get(2, 2) + self.get(3, 3))
    }

    /// The AABB of a box transformed by this matrix.
    ///
    /// A rotated box is no longer axis-aligned, so this is the union of all
    /// eight transformed corners: conservative, and the tightest axis-aligned
    /// box that still contains the transformed geometry. Delegates to
    /// [`Bounds3::transform`](super::Bounds3::transform), which batches the
    /// corners through [`transform_points`](Self::transform_points).
    #[inline(always)]
    pub fn transform_bounds(&self, bounds: &super::Bounds3<S, E>) -> super::Bounds3<S, E> {
        bounds.transform(self)
    }

    /// A diagonal matrix.
    #[inline(always)]
    pub fn from_diagonal(diagonal: V4<S, E>) -> Self {
        let mut m = Self::ZERO;

        m.0[0] = m.0[0].insert::<0>(diagonal.extract::<0>());
        m.0[1] = m.0[1].insert::<1>(diagonal.extract::<1>());
        m.0[2] = m.0[2].insert::<2>(diagonal.extract::<2>());
        m.0[3] = m.0[3].insert::<3>(diagonal.extract::<3>());

        m
    }
}

impl<S: Simd3Vectors, E: AosFloat<S> + thermite::math::ScalarMath> Matrix4<S, E> {
    /// Rotation by `angle` radians about a **pre-normalized** axis.
    ///
    /// Routed through a quaternion rather than writing out Rodrigues: the
    /// expansion to a matrix is trig-free after the one `sin_cos`, and
    /// `quat_to_mat4` is a backend primitive.
    #[inline(always)]
    pub fn from_axis_angle(axis: V3<S, E>, angle: E) -> Self {
        super::Quaternion::from_axis_angle_raw(axis, angle).to_matrix()
    }

    /// Orthographic projection mapping `z` from `[near, far]` onto `[0, 1]`.
    #[inline(always)]
    pub fn orthographic(near: E, far: E) -> Self {
        let inv_range = E::ONE / (far - near);

        let mut m = Self::IDENTITY;

        m.set(2, 2, inv_range);
        m.set(2, 3, -near * inv_range);

        m
    }

    /// Perspective projection with a vertical field of view (radians), mapping
    /// `z` from `[near, far]` onto `[0, 1]` and scaling `xy` by
    /// `$1/\tan(fov/2)$`.
    ///
    /// The last row is **not** `[0, 0, 0, 1]`, so points must go through
    /// [`project_point`](Self::project_point) rather than
    /// [`transform_point`](Self::transform_point).
    #[inline(always)]
    pub fn perspective(fov: E, near: E, far: E) -> Self {
        let inv_tan = E::ONE / (fov * E::from_ratio(1, 2)).scalar_tan();
        let range = far - near;

        let mut m = Self::ZERO;

        m.set(0, 0, inv_tan);
        m.set(1, 1, inv_tan);
        m.set(2, 2, far / range);
        m.set(3, 2, E::ONE);
        m.set(2, 3, -(far * near) / range);

        m
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Add for Matrix4<S, E> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..4 {
            self.0[i] += rhs.0[i];
        }

        self
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Sub for Matrix4<S, E> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..4 {
            self.0[i] -= rhs.0[i];
        }

        self
    }
}

/// Matrix composition: `self * rhs` applies `rhs` first.
impl<S: Simd3Vectors, E: AosFloat<S>> Mul for Matrix4<S, E> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(V4::<S, E>::mat4_product::<true>(&self.0, &rhs.0))
    }
}

/// Scaling every entry by a scalar.
impl<S: Simd3Vectors, E: AosFloat<S>> Mul<E> for Matrix4<S, E> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: E) -> Self {
        let rhs = V4::<S, E>::splat(rhs);

        for i in 0..4 {
            self.0[i] *= rhs;
        }

        self
    }
}

#[doc(inline)]
pub use crate::gamma_scalar as gamma;
