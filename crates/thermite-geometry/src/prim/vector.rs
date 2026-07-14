use core::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use thermite::{
    mask::GenericSelectable,
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

    /// Load `V::LANES` vectors from an interleaved (array-of-structures) span -
    /// a `&[[f32; N]]`, the layout every mesh, buffer and file format actually
    /// uses - transposing to SoA on the way in.
    ///
    /// `self.0[c]` ends up holding component `c` of all `LANES` input vectors:
    /// `out[c].extract(lane) == ptr[lane * N + c]`. For `N == 3` on NEON this is
    /// a single `LD3` - the transpose happens in the load unit - and a shuffle
    /// network elsewhere.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn load_interleaved(ptr: *const V::Element) -> Self {
        Self(unsafe { V::load_deinterleaved::<N>(ptr) })
    }

    /// Store `V::LANES` vectors back to an interleaved span - the exact inverse
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
}

impl<V: FloatVector> Vector<V, 2> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();

    /// 2D cross product (perp-dot): `self.x * other.y - self.y * other.x`.
    #[inline(always)]
    pub fn cross(self, other: Self) -> V {
        self.0[0].mul_sube(other.0[1], self.0[1] * other.0[0])
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

impl<V: FloatVector, const N: usize> Add<Self> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] += rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Sub<Self> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] -= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Mul<Self> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] *= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Mul<V> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: V) -> Self::Output {
        for i in 0..N {
            self.0[i] *= rhs;

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Div<Self> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] /= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Div<V> for Vector<V, N> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, rhs: V) -> Self::Output {
        for i in 0..N {
            self.0[i] /= rhs;

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
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
    /// Computes the Euclidean norm (hypotenuse) of the vector. If you need the squared norm, use `dot(x, x)` instead.
    fn l2_norm_p<P: Policy>(&self) -> V;
    fn l1_norm_p<P: Policy>(&self) -> V;
    fn normalize_p<P: Policy>(self) -> Self;
    fn reciprocal_p<P: Policy>(&self) -> Self;
}

#[rustfmt::skip]
pub trait VectorOps<V: SpatialMathWithPolicy>: VectorOpsWithPolicy<V> {
    #[inline(always)] fn dot(&self, other: &Self) -> V { self.dot_p::<DefaultPolicy>(other) }
    #[inline(always)] fn l2_norm(&self) -> V { self.l2_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn l1_norm(&self) -> V { self.l1_norm_p::<DefaultPolicy>() }
    #[inline(always)] fn normalize(self) -> Self { self.normalize_p::<DefaultPolicy>() }
    #[inline(always)] fn reciprocal(&self) -> Self { self.reciprocal_p::<DefaultPolicy>() }
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
    fn normalize_p<P: Policy>(mut self) -> Self {
        let inv_norm = SpatialMathWithPolicy::inv_hypot_n_p::<P, N>(self.0);

        for x in &mut self.0 {
            *x *= inv_norm;

            unsafe { x.block_autovectorization() };
        }

        self
    }

    #[inline(always)]
    fn reciprocal_p<P: Policy>(&self) -> Self {
        let mut result = *self;

        for x in &mut result.0 {
            *x = x.reciprocal_p::<P>();

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
