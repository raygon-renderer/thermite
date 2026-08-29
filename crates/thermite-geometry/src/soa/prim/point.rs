use core::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

use thermite::{
    math::{SpatialMathWithPolicy, policy::DefaultPolicy},
    vector::FloatVector,
};

use super::{Vector, vector::VectorOps as _};

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

    /// Component-wise minimum of two points. Building an AABB from a batch of
    /// points is a fold of this and [`max`](Self::max).
    #[inline(always)]
    pub fn min(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].min(other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Component-wise maximum of two points.
    #[inline(always)]
    pub fn max(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i] = self.0[i].max(other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// Component-wise linear interpolation from `self` to `other` by `t`.
    #[inline(always)]
    pub fn mix(mut self, other: Self, t: V) -> Self {
        for i in 0..N {
            self.0[i] = t.mix(self.0[i], other.0[i]);

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }

    /// The point halfway between `self` and `other`.
    #[inline(always)]
    pub fn midpoint(self, other: Self) -> Self {
        self.mix(other, V::HALF)
    }

    /// A mask that is set for the lanes whose every coordinate is finite.
    #[inline(always)]
    pub fn is_finite(&self) -> V::Mask {
        Vector(self.0).is_finite()
    }
}

impl<V: SpatialMathWithPolicy, const N: usize> Point<V, N> {
    /// The Euclidean distance between two points.
    #[inline(always)]
    pub fn distance(self, other: Self) -> V {
        self.distance_p::<DefaultPolicy>(other)
    }

    /// The squared Euclidean distance between two points. Cheaper than
    /// [`distance`](Self::distance) (no square root) and enough whenever you only
    /// compare distances.
    #[inline(always)]
    pub fn distance_sqr(self, other: Self) -> V {
        (self - other).norm_sqr()
    }

    /// [`distance`](Self::distance) under an explicit precision policy.
    #[inline(always)]
    pub fn distance_p<P: thermite::math::policy::Policy>(self, other: Self) -> V {
        use super::vector::VectorOpsWithPolicy as _;

        (self - other).l2_norm_p::<P>()
    }

    /// Load `V::LANES` points from an interleaved (array-of-structures) span, a
    /// `&[[f32; N]]`, the layout meshes and buffers actually use, transposing to
    /// SoA on the way in. `N == 3` on NEON is a single `LD3`.
    ///
    /// See [`Vector::load_interleaved`](super::Vector::load_interleaved).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn load_interleaved(ptr: *const V::Element) -> Self {
        Self(unsafe { V::load_deinterleaved::<N>(ptr) })
    }

    /// Store `V::LANES` points back to an interleaved span, the exact inverse of
    /// [`load_interleaved`](Self::load_interleaved) (`ST3` on NEON).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn store_interleaved(self, ptr: *mut V::Element) {
        unsafe { V::store_interleaved::<N>(ptr, self.0) }
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

impl<V: FloatVector, const N: usize> AddAssign<Vector<V, N>> for Point<V, N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<V, N>) {
        for i in 0..N {
            self.0[i] += rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }
    }
}

impl<V: FloatVector, const N: usize> SubAssign<Vector<V, N>> for Point<V, N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<V, N>) {
        for i in 0..N {
            self.0[i] -= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }
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

impl<V: FloatVector, const N: usize> thermite::mask::GenericSelectable for Point<V, N> {
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
