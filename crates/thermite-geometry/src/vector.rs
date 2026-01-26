use core::ops::{Add, Index, IndexMut, Mul, Sub};

use thermite::{
    generic::FloatVector,
    math::policy::{DefaultPolicy, Policy},
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
}

impl<V: FloatVector> Vector<V, 2> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();
}

impl<V: FloatVector> Vector<V, 3> {
    pub const X: Self = Self::basis::<0>();
    pub const Y: Self = Self::basis::<1>();
    pub const Z: Self = Self::basis::<2>();
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
