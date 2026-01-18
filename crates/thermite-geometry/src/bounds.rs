use thermite::generic::FloatVector;

use crate::Point;

/// Axis-aligned bounds in N dimensions, where each dimension has a [min, max] pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds<V: FloatVector, const N: usize>(pub [[V; 2]; N]);

impl<V: FloatVector, const N: usize> Bounds<V, N> {
    /// Create bounds that encompass no points.
    #[inline(always)]
    pub const fn empty() -> Self {
        let max = V::MIN;
        let min = V::MAX;

        Self([[max, min]; N])
    }

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
}
