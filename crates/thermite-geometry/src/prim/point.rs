use core::ops::{Add, Index, IndexMut, Sub};

use thermite::generic::FloatVector;

use super::Vector;

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

    #[inline(always)]
    pub fn abs(mut self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].abs();
        }

        self
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

impl<V: FloatVector, const N: usize> From<Point<V, N>> for Vector<V, N> {
    #[inline(always)]
    fn from(point: Point<V, N>) -> Self {
        Vector(point.0)
    }
}

impl<V: FloatVector, const N: usize> From<Vector<V, N>> for Point<V, N> {
    #[inline(always)]
    fn from(vector: Vector<V, N>) -> Self {
        Point(vector.0)
    }
}

impl<V: FloatVector, const N: usize> thermite::generic::GenericSelectable for Point<V, N> {
    type SelectableMask = V::Mask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: thermite::prelude::CastMask<M>,
    {
        use thermite::prelude::GenericMask as _;

        let mask = <Self::SelectableMask as thermite::prelude::CastMask<M>>::mask_from(mask);

        Point(core::array::from_fn(|i| mask.select(t.0[i], f.0[i])))
    }
}
