use core::ops::{BitOr, BitOrAssign, Mul, MulAssign};

use thermite::vector::FloatVector;

use super::{Point, Vector};

/// Axis-aligned bounds in N dimensions, where each dimension has a [min, max] coordinate pair.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Bounds<V: FloatVector, const N: usize>(pub [[V; 2]; N]);

impl<V: FloatVector, const N: usize> Bounds<V, N> {
    /// Create bounds that encompass no points.
    #[inline(always)]
    pub const fn empty() -> Self {
        let max = V::MIN;
        let min = V::MAX;

        Self([[max, min]; N])
    }

    /// Bounds from explicit minimum and maximum corners.
    #[inline(always)]
    pub fn from_corners(min: Vector<V, N>, max: Vector<V, N>) -> Self {
        Bounds(core::array::from_fn(|i| [min[i], max[i]]))
    }

    /// Bounds symmetric about the origin with the given (positive) half-extents,
    /// i.e. `[-half_extent, half_extent]` per axis.
    #[inline(always)]
    pub fn symmetric(half_extent: Vector<V, N>) -> Self {
        Bounds(core::array::from_fn(|i| [-half_extent[i], half_extent[i]]))
    }

    /// Grow the bounds outward by `amount` on every side.
    #[inline(always)]
    pub fn expand(mut self, amount: V) -> Self {
        for i in 0..N {
            self.0[i][0] -= amount;
            self.0[i][1] += amount;
        }
        self
    }

    /// Intersection of two bounds (may be empty/inverted if they are disjoint).
    #[inline(always)]
    pub fn intersection(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i][0] = self.0[i][0].max(other.0[i][0]);
            self.0[i][1] = self.0[i][1].min(other.0[i][1]);
        }
        self
    }

    /// Returns one of the vertices of the bounds, specified by the given index,
    /// where the index can be between 0 and 2^N - 1. Index 0 is the minimum corner,
    /// and index 2^N - 1 is the maximum corner, though if you need those specifically,
    /// use `min_point` and `max_point`.
    #[inline(always)]
    pub const fn vertex(&self, index: usize) -> Point<V, N> {
        let mut result = [V::ZERO; N];

        let mut i = 0;

        while i < N {
            let bit = (index >> i) & 1;
            result[i] = self.0[i][bit];
            i += 1;
        }

        Point(result)
    }

    #[inline(always)]
    pub fn min_point(&self) -> Point<V, N> {
        Point(self.0.map(|p| p[0]))
    }

    #[inline(always)]
    pub fn max_point(&self) -> Point<V, N> {
        Point(self.0.map(|p| p[1]))
    }
}

impl<V: FloatVector, const N: usize> From<Point<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn from(point: Point<V, N>) -> Self {
        let mut bounds = [[V::ZERO; 2]; N];

        for i in 0..N {
            bounds[i][0] = point[i];
            bounds[i][1] = point[i];
        }

        Bounds(bounds)
    }
}

impl<V: FloatVector, const N: usize> BitOrAssign<Point<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Point<V, N>) {
        for i in 0..N {
            self.0[i][0] = self.0[i][0].min(rhs[i]);
            self.0[i][1] = self.0[i][1].max(rhs[i]);
        }
    }
}

impl<V: FloatVector, const N: usize> BitOrAssign for Bounds<V, N> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i][0] = self.0[i][0].min(rhs.0[i][0]);
            self.0[i][1] = self.0[i][1].max(rhs.0[i][1]);
        }
    }
}

impl<V: FloatVector, const N: usize> MulAssign<Vector<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Vector<V, N>) {
        for i in 0..N {
            self.0[i][0] *= rhs[i];
            self.0[i][1] *= rhs[i];
        }
    }
}

impl<V: FloatVector, const N: usize> BitOr<Point<V, N>> for Bounds<V, N> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Point<V, N>) -> Self::Output {
        self |= rhs;
        self
    }
}

impl<V: FloatVector, const N: usize> BitOr for Bounds<V, N> {
    type Output = Self;

    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl<V: FloatVector, const N: usize> Mul<Vector<V, N>> for Bounds<V, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: Vector<V, N>) -> Self::Output {
        self *= rhs;
        self
    }
}
