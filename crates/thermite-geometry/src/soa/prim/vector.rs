use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use thermite::{
    mask::{GenericMask, GenericSelectable},
    math::policy::{DefaultPolicy, Policy},
    vector::FloatVector,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vector<V: FloatVector, const N: usize>(pub [V; N]);

impl<V: FloatVector, const N: usize> Vector<V, N> {
    pub const ZERO: Self = Self::splat(V::ZERO);
    pub const ONE: Self = Self::splat(V::ONE);

    #[inline(always)]
    pub const fn splat(value: V) -> Self {
        Self([value; N])
    }

    #[inline(always)]
    pub const fn new(coords: [V; N]) -> Self {
        Self(coords)
    }

    #[inline(always)]
    pub const fn basis<const I: usize>() -> Self {
        let mut coords = [V::ZERO; N];
        coords[I] = V::ONE;
        Self(coords)
    }

    /// Load `V::LANES` vectors from an interleaved (array-of-structures) span,
    /// a `&[[f32; N]]`, the layout every mesh, buffer and file format actually
    /// uses, transposing to SoA on the way in.
    ///
    /// `self.0[c]` ends up holding component `c` of all `LANES` input vectors:
    /// `out[c].extract(lane) == ptr[lane * N + c]`. For `N == 3` on NEON this is
    /// a single `LD3` (the transpose happens in the load unit) and a shuffle
    /// network elsewhere.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn load_interleaved(ptr: *const V::Element) -> Self {
        Self(unsafe { V::load_deinterleaved::<N>(ptr) })
    }

    /// Store `V::LANES` vectors back to an interleaved span, the exact inverse
    /// of [`load_interleaved`](Self::load_interleaved) (`ST3` on NEON).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn store_interleaved(self, ptr: *mut V::Element) {
        unsafe { V::store_interleaved::<N>(ptr, self.0) }
    }

    #[inline(always)]
    pub fn min(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].min(other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    #[inline(always)]
    pub fn max(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].max(other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    #[inline(always)]
    pub fn clamp(mut self, min: Self, max: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].clamp(min.0[i], max.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Component-wise absolute value.
    #[inline(always)]
    pub fn abs(mut self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].abs();

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Component-wise sign (`+1`/`-1`, matching `FloatVector::signum`).
    #[inline(always)]
    pub fn signum(mut self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].signum();

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    // TODO: Use the MulAddExt trait maybe?

    /// Fused `self * scalar + acc`, component-wise. Uses FMA where the hardware
    /// supports it, otherwise a separate multiply and add.
    #[inline(always)]
    pub fn mul_adde(mut self, scalar: V, acc: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].mul_adde(scalar, acc.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Fused `acc - self * scalar`, component-wise. Uses FMA where the hardware
    /// supports it, otherwise a separate multiply and subtract.
    #[inline(always)]
    pub fn nmul_adde(mut self, scalar: V, acc: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].nmul_adde(scalar, acc.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Component-wise linear interpolation from `self` to `other` by `t`.
    ///
    /// `t` is a full vector, so each lane may interpolate by a different amount.
    #[inline(always)]
    pub fn mix(mut self, other: Self, t: V) -> Self {
        for i in 0..N {
            self.0[i] = t.mix(self.0[i], other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// A mask that is set for the lanes whose every component is finite.
    #[inline(always)]
    pub fn is_finite(&self) -> V::Mask {
        let mut mask = self.0[0].is_finite();

        for i in 1..N {
            mask &= self.0[i].is_finite();
        }

        mask
    }

    /// A mask that is set for the lanes where any component is NaN.
    #[inline(always)]
    pub fn is_nan(&self) -> V::Mask {
        let mut mask = self.0[0].is_nan();

        for i in 1..N {
            mask |= self.0[i].is_nan();
        }

        mask
    }
}

impl<V: FloatVector> Vector<V, 2> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();

    /// 2D cross product (perp-dot): `self.x * other.y - self.y * other.x`.
    #[inline(always)]
    pub fn cross(self, other: Self) -> V {
        self.0[0].mul_sube(other.0[1], self.0[1] * other.0[0])
    }

    /// The vector rotated a quarter turn counter-clockwise: `(-y, x)`.
    #[inline(always)]
    pub fn perp(self) -> Self {
        Vector([-self.0[1], self.0[0]])
    }
}

impl<V: FloatVector> Vector<V, 3> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();
    pub const Z: Self = Self::basis::<2>();

    /// 3D cross product `self x other`.
    #[inline(always)]
    pub fn cross(self, other: Self) -> Self {
        Vector([
            self.0[1].mul_sube(other.0[2], self.0[2] * other.0[1]),
            self.0[2].mul_sube(other.0[0], self.0[0] * other.0[2]),
            self.0[0].mul_sube(other.0[1], self.0[1] * other.0[0]),
        ])
    }
}

impl<V: FloatVector> Vector<V, 4> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();
    pub const Z: Self = Self::basis::<2>();
    pub const W: Self = Self::basis::<3>();
}

impl<V: FloatVector, const N: usize> Neg for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self::Output {
        for i in 0..N {
            self.0[i] = -self.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Default for Vector<V, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

/// Generates the by-value operator and its `*Assign` counterpart from one
/// component-wise expression, for both `Self` and scalar-`V` right-hand sides.
macro_rules! impl_binop {
    ($($op:ident::$method:ident, $assign:ident::$assign_method:ident, $sym:tt;)*) => {$(
        impl<V: FloatVector, const N: usize> $assign<Self> for Vector<V, N> {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Self) {
                for i in 0..N {
                    self.0[i] = self.0[i] $sym rhs.0[i];

                    unsafe { self.0[i].block_autovectorization() };
                }
            }
        }

        impl<V: FloatVector, const N: usize> $assign<V> for Vector<V, N> {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: V) {
                for i in 0..N {
                    self.0[i] = self.0[i] $sym rhs;

                    unsafe { self.0[i].block_autovectorization() };
                }
            }
        }

        impl<V: FloatVector, const N: usize> $op<Self> for Vector<V, N> {
            type Output = Self;

            #[inline(always)]
            fn $method(mut self, rhs: Self) -> Self::Output {
                $assign::$assign_method(&mut self, rhs);
                self
            }
        }

        impl<V: FloatVector, const N: usize> $op<V> for Vector<V, N> {
            type Output = Self;

            #[inline(always)]
            fn $method(mut self, rhs: V) -> Self::Output {
                $assign::$assign_method(&mut self, rhs);
                self
            }
        }
    )*};
}

impl_binop! {
    Add::add, AddAssign::add_assign, +;
    Sub::sub, SubAssign::sub_assign, -;
    Mul::mul, MulAssign::mul_assign, *;
    Div::div, DivAssign::div_assign, /;
}

impl<V: FloatVector, const N: usize> Vector<V, N> {
    #[inline(always)]
    pub fn scale(mut self, rhs: V::Element) -> Self {
        let rhs = V::splat(rhs);

        for x in &mut self.0 {
            *x *= rhs;

            unsafe { x.block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize, I> Index<I> for Vector<V, N>
where
    [V; N]: Index<I>,
{
    type Output = <[V; N] as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.0, index)
    }
}

impl<V: FloatVector, const N: usize, I> IndexMut<I> for Vector<V, N>
where
    [V; N]: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.0, index)
    }
}

use thermite::math::SpatialMathWithPolicy;

pub trait VectorOpsWithPolicy<V: SpatialMathWithPolicy>: Sized {
    fn dot_p<P: Policy>(&self, other: &Self) -> V;
    /// Reflects `self` about the (unit) normal `n`.
    fn reflect_p<P: Policy>(self, n: &Self) -> Self;
    /// Refracts `self` through a surface with unit normal `n`.
    fn refract_p<P: Policy>(self, n: &Self, eta: V) -> Self;
    /// [`refract_p`](Self::refract_p), plus the mask of lanes that actually refracted.
    fn refract_mask_p<P: Policy>(self, n: &Self, eta: V) -> (Self, V::Mask);
    /// Flips `n` to lie in the hemisphere opposing `incident`.
    fn faceforward_p<P: Policy>(self, incident: &Self, reference: &Self) -> Self;
    /// Computes the Euclidean norm (hypotenuse) of the vector. If you need the squared norm, use [`norm_sqr`](VectorOps::norm_sqr) instead.
    fn l2_norm_p<P: Policy>(&self) -> V;
    fn l1_norm_p<P: Policy>(&self) -> V;
    /// The Chebyshev (L-infinity) norm: the largest absolute component.
    fn linf_norm_p<P: Policy>(&self) -> V;
    fn normalize_p<P: Policy>(self) -> Self;
    /// Returns the normalized vector *and* the original norm in one pass,
    /// sharing the reciprocal-square-root between them.
    fn normalize_norm_p<P: Policy>(self) -> (Self, V);
    /// Like [`normalize_p`](Self::normalize_p), but lanes whose norm is zero (or
    /// non-finite) come back as the zero vector instead of `NaN`.
    fn try_normalize_p<P: Policy>(self) -> Self;
    fn approx_reciprocal_p<P: Policy>(&self) -> Self;

    /// The component-wise reciprocal by **exact IEEE division**, so `1/0` is a
    /// signed infinity rather than `NaN`. Takes no policy, since exactness is the
    /// entire point.
    ///
    /// [`reciprocal`](VectorOps::reciprocal) goes through the policy system,
    /// which on any backend with `HAS_APPROX_RCP` (every x86 one) computes `rcp`
    /// plus a Newton refinement below `Best` precision. That step is
    /// `$y(2 - dy)$`, and at `$d = 0$` it evaluates to `inf * (2 - 0 * inf)` =
    /// `NaN`, so the approximate reciprocal cannot represent the axis-aligned
    /// case at all, and a `NaN` there fails every subsequent comparison.
    ///
    /// Use this wherever a division by zero must yield an infinity that later
    /// min/max operations act on, above all the ray-AABB slab test, whose
    /// empty-or-full interval per axis depends on exactly that behaviour.
    fn reciprocal_exact(&self) -> Self;
}

#[rustfmt::skip]
pub trait VectorOps<V: SpatialMathWithPolicy>: VectorOpsWithPolicy<V> {
    #[inline(always)] fn dot(&self, other: &Self) -> V { self.dot_p::<DefaultPolicy>(other) }
    /// The squared Euclidean norm, `self . self`. Cheaper than [`l2_norm`](Self::l2_norm)
    /// (no square root) and enough whenever you only compare lengths.
    #[inline(always)] fn norm_sqr(&self) -> V { self.dot_p::<DefaultPolicy>(self) }
    #[inline(always)] fn l2_norm(&self) -> V { self.l2_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn l1_norm(&self) -> V { self.l1_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn linf_norm(&self) -> V { self.linf_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn normalize(self) -> Self { self.normalize_p::<DefaultPolicy>() }
    #[inline(always)] fn normalize_norm(self) -> (Self, V) { self.normalize_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn try_normalize(self) -> Self { self.try_normalize_p::<DefaultPolicy>() }
    #[inline(always)] fn approx_reciprocal(&self) -> Self { self.approx_reciprocal_p::<DefaultPolicy>() }

    /// Reflects `self` about the unit normal `n`: `$\mathbf{i} - 2(\mathbf{n}\cdot\mathbf{i})\mathbf{n}$`.
    ///
    /// Follows the GLSL convention, in which the incident vector points *into*
    /// the surface, so the result points away from it. Shading code that keeps
    /// its directions pointing *away* from the surface (the usual `wo`/`wi`
    /// convention) wants `(-wo).reflect(&n)`, or equivalently
    /// `n * (V::TWO * n.dot(&wo)) - wo`.
    #[inline(always)] fn reflect(self, n: &Self) -> Self { self.reflect_p::<DefaultPolicy>(n) }

    /// Refracts `self` through a surface with unit normal `n` and relative index
    /// of refraction `eta`. Total internal reflection gives the zero vector.
    #[inline(always)] fn refract(self, n: &Self, eta: V) -> Self { self.refract_p::<DefaultPolicy>(n, eta) }

    /// [`refract`](Self::refract), plus the mask of lanes that actually refracted
    /// (clear where the lane totally internally reflected).
    #[inline(always)] fn refract_mask(self, n: &Self, eta: V) -> (Self, V::Mask) { self.refract_mask_p::<DefaultPolicy>(n, eta) }

    /// Returns `self` flipped, if needed, to lie in the hemisphere opposing `incident`.
    #[inline(always)] fn faceforward(self, incident: &Self, reference: &Self) -> Self { self.faceforward_p::<DefaultPolicy>(incident, reference) }
}

impl<T, V: SpatialMathWithPolicy> VectorOps<V> for T where T: VectorOpsWithPolicy<V> {}

impl<V: SpatialMathWithPolicy, const N: usize> VectorOpsWithPolicy<V> for Vector<V, N> {
    #[inline(always)]
    fn dot_p<P: Policy>(&self, other: &Self) -> V {
        if N <= 3 {
            let mut result = self[0] * other[0];

            // Loop with FMA for small-dimensioned vectors for better performance/accuracy
            for i in 1..N {
                result = self[i].mul_adde(other[i], result);
            }

            return result;
        }

        let mut tmp = self.0;

        // parallel element-wise multiplication
        for (t, o) in tmp.iter_mut().zip(&other.0) {
            *t *= *o;

            unsafe { t.block_autovectorization() };
        }

        // Log2(N) reduction
        thermite::math::algorithms::reduce_in_place(&mut tmp, |a, b| a + b);

        tmp[0]
    }

    #[inline(always)]
    fn reflect_p<P: Policy>(mut self, n: &Self) -> Self {
        // 2 * (n . i), subtracted by the negated-multiply FMA per component below.
        let coeff = self.dot_p::<P>(n) * V::TWO;

        for i in 0..N {
            self.0[i] = n.0[i].nmul_adde(coeff, self.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    #[inline(always)]
    fn refract_p<P: Policy>(self, n: &Self, eta: V) -> Self {
        self.refract_mask_p::<P>(n, eta).0
    }

    #[inline(always)]
    fn refract_mask_p<P: Policy>(mut self, n: &Self, eta: V) -> (Self, V::Mask) {
        let d = self.dot_p::<P>(n);

        // k = 1 - eta^2 * (1 - d^2). Negative k is total internal reflection:
        // the refracted ray does not exist and Snell's law has no solution.
        let omd2 = d.nmul_adde(d, V::ONE);
        let k = (eta * eta).nmul_adde(omd2, V::ONE);

        let refracted = k.cmp_ge(V::ZERO);

        // sqrt_z instead of sqrt-then-select: the masked variant zeroes the
        // TIR lanes with the AND the mask already provides, and never feeds
        // sqrt a negative (which would manufacture a NaN for the blend below
        // to discard). Those lanes are zeroed wholesale at the end anyway.
        let coeff = eta.mul_adde(d, k.sqrt_z(refracted));

        for i in 0..N {
            self.0[i] = n.0[i].nmul_adde(coeff, eta * self.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        (Self::select(refracted, self, Self::ZERO), refracted)
    }

    #[inline(always)]
    fn faceforward_p<P: Policy>(self, incident: &Self, reference: &Self) -> Self {
        // GLSL's rule: keep `self` when it already opposes the incident vector.
        // The test is against `reference` (usually the geometric normal), which
        // is what lets a shading normal be flipped by the geometric one.
        let flip = incident.dot_p::<P>(reference).cmp_lt(V::ZERO);

        Self::select(flip, self, -self)
    }

    #[inline(always)]
    fn l2_norm_p<P: Policy>(&self) -> V {
        SpatialMathWithPolicy::hypot_n_p::<P, N>(self.0)
    }

    #[inline(always)]
    fn l1_norm_p<P: Policy>(&self) -> V {
        let mut tmp = self.0;

        for x in &mut tmp {
            *x = x.abs();

            unsafe { x.block_autovectorization() };
        }

        thermite::math::algorithms::reduce_in_place(&mut tmp, |a, b| a + b);

        tmp[0]
    }

    #[inline(always)]
    fn linf_norm_p<P: Policy>(&self) -> V {
        let mut tmp = self.0;

        for x in &mut tmp {
            *x = x.abs();

            unsafe { x.block_autovectorization() };
        }

        thermite::math::algorithms::reduce_in_place(&mut tmp, V::max);

        tmp[0]
    }

    #[inline(always)]
    fn normalize_p<P: Policy>(mut self) -> Self {
        let inv_norm = SpatialMathWithPolicy::inv_hypot_n_p::<P, N>(self.0);

        for x in &mut self.0 {
            *x *= inv_norm;

            unsafe { x.block_autovectorization() };
        }

        self
    }

    #[inline(always)]
    fn normalize_norm_p<P: Policy>(self) -> (Self, V) {
        // hypot_n is the accurate (non-overflowing) norm, and the division below is a
        // single reciprocal shared by every component.
        let norm = SpatialMathWithPolicy::hypot_n_p::<P, N>(self.0);

        (self / norm, norm)
    }

    #[inline(always)]
    fn try_normalize_p<P: Policy>(self) -> Self {
        let norm = SpatialMathWithPolicy::hypot_n_p::<P, N>(self.0);

        // A zero-length vector has no direction, and dividing would give 0/0 = NaN.
        // Divide those lanes by one and zero the result instead. Non-finite norms
        // (an infinite or NaN input) are rejected by the same test.
        let ok = norm.cmp_gt(V::ZERO) & norm.is_finite();

        Self::select(ok, self / ok.select(norm, V::ONE), Self::ZERO)
    }

    #[inline(always)]
    fn approx_reciprocal_p<P: Policy>(&self) -> Self {
        let mut result = *self;

        for x in &mut result.0 {
            *x = x.approx_reciprocal_p::<P>();

            unsafe { x.block_autovectorization() };
        }

        result
    }

    #[inline(always)]
    fn reciprocal_exact(&self) -> Self {
        let mut result = *self;

        for x in &mut result.0 {
            // A true divide, never `approx_reciprocal_p`. See the trait docs.
            *x = V::ONE / *x;

            unsafe { x.block_autovectorization() };
        }

        result
    }
}

impl<V: FloatVector, const N: usize> GenericSelectable for Vector<V, N> {
    type SelectableMask = V::Mask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: thermite::prelude::CastMask<M>,
    {
        use thermite::prelude::GenericMask as _;

        let mask = <Self::SelectableMask as thermite::prelude::CastMask<M>>::mask_from(mask);

        Vector(core::array::from_fn(|i| mask.select(t.0[i], f.0[i])))
    }
}
