use core::ops::{Add, Index, IndexMut, Sub};

use thermite::generic::FloatVector;

use crate::Vector;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Point<V: FloatVector, const N: usize>(pub [V; N]);

impl<V: FloatVector, const N: usize> Point<V, N> {
    pub const ZERO: Self = Self::splat(V::ZERO);

    #[inline(always)]
    pub const fn splat(value: V) -> Self {
        Self([value; N])
    }

    #[inline(always)]
    pub const fn new(coords: [V; N]) -> Self {
        Self(coords)
    }
}

impl<V: FloatVector, const N: usize> Add<Vector<V, N>> for Point<V, N> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Vector<V, N>) -> Self::Output {
        for i in 0..N {
            self.0[i] += rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Sub<Self> for Point<V, N> {
    type Output = Vector<V, N>;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] -= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        Vector(self.0)
    }
}

impl<V: FloatVector, const N: usize> Sub<Vector<V, N>> for Point<V, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Vector<V, N>) -> Self::Output {
        for i in 0..N {
            self.0[i] -= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector, const N: usize, I> Index<I> for Point<V, N>
where
    [V; N]: Index<I>,
{
    type Output = <[V; N] as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.0, index)
    }
}

impl<V: FloatVector, const N: usize, I> IndexMut<I> for Point<V, N>
where
    [V; N]: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.0, index)
    }
}
