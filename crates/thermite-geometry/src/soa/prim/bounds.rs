use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, MulAssign};

use thermite::{
    element::Element,
    mask::{GenericMask, GenericSelectable},
    math::{SpatialMathWithPolicy, policy::Policy},
    vector::{FloatVector, GenericVector, NumericVector},
};

use super::{Point, Vector, vector::VectorOpsWithPolicy as _};

/// Axis-aligned bounds in N dimensions, where each dimension has a [min, max] coordinate pair.
///
/// `self.0[axis][0]` is the minimum coordinate on that axis and `self.0[axis][1]`
/// the maximum. As everywhere in [`soa`](crate::soa), a `Bounds<f32x8, 3>` is
/// **eight independent boxes**, one per lane, so the queries below return a
/// `V::Mask` (one answer per lane) rather than a `bool`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Bounds<V: FloatVector, const N: usize>(pub [[V; 2]; N]);

impl<V: FloatVector, const N: usize> Bounds<V, N> {
    /// The inverted sentinel box (`min = +inf`, `max = -inf`), which encompasses
    /// no points and is the **identity for union**: `Bounds::EMPTY | b == b`.
    ///
    /// Start every incremental box-building fold here.
    pub const EMPTY: Self = Self([[V::INFINITY, V::NEG_INFINITY]; N]);

    /// The box containing everything (`min = -inf`, `max = +inf`), which is the
    /// **identity for intersection**: `Bounds::UNIVERSE & b == b`.
    pub const UNIVERSE: Self = Self([[V::NEG_INFINITY, V::INFINITY]; N]);

    /// Create bounds that encompass no points. See [`EMPTY`](Self::EMPTY).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Bounds from explicit minimum and maximum corners.
    #[inline(always)]
    pub fn from_corners(min: Vector<V, N>, max: Vector<V, N>) -> Self {
        Bounds(core::array::from_fn(|i| [min[i], max[i]]))
    }

    /// The degenerate box containing exactly one point.
    #[inline(always)]
    pub fn from_point(point: Point<V, N>) -> Self {
        Bounds(core::array::from_fn(|i| [point[i], point[i]]))
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

    /// Grow the bounds outward by a per-axis amount on every side.
    #[inline(always)]
    pub fn expand_axes(mut self, amount: Vector<V, N>) -> Self {
        for i in 0..N {
            self.0[i][0] -= amount[i];
            self.0[i][1] += amount[i];
        }
        self
    }

    /// Intersection of two bounds. Disjoint inputs give an inverted (empty) box -
    /// test the result with [`is_empty`](Self::is_empty).
    #[inline(always)]
    pub fn intersection(mut self, other: Self) -> Self {
        for i in 0..N {
            self.0[i][0] = self.0[i][0].max(other.0[i][0]);
            self.0[i][1] = self.0[i][1].min(other.0[i][1]);
        }
        self
    }

    /// The smallest box containing both `self` and `other`.
    #[inline(always)]
    pub fn union(mut self, other: Self) -> Self {
        self |= other;
        self
    }

    /// The smallest box containing both `self` and `point`.
    #[inline(always)]
    pub fn union_point(mut self, point: Point<V, N>) -> Self {
        self |= point;
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

    /// All `2^N` corners at once, in [`vertex`](Self::vertex) order.
    ///
    /// `N` is a const generic, so the array length is fixed at compile time, and
    /// this is the 3D case a box transform iterates.
    #[inline(always)]
    pub fn vertices(&self) -> [Point<V, N>; 8] {
        debug_assert!(N == 3, "vertices() returns the 8 corners of a 3D box");

        core::array::from_fn(|i| self.vertex(i))
    }

    #[inline(always)]
    pub fn min_point(&self) -> Point<V, N> {
        Point(self.0.map(|p| p[0]))
    }

    #[inline(always)]
    pub fn max_point(&self) -> Point<V, N> {
        Point(self.0.map(|p| p[1]))
    }

    /// The extent of the box on each axis, `max - min`. Negative on any axis
    /// where the box is inverted (empty).
    #[inline(always)]
    pub fn diagonal(&self) -> Vector<V, N> {
        Vector(core::array::from_fn(|i| self.0[i][1] - self.0[i][0]))
    }

    /// The center of the box.
    #[inline(always)]
    pub fn centroid(&self) -> Point<V, N> {
        Point(core::array::from_fn(|i| V::HALF.mix(self.0[i][0], self.0[i][1])))
    }

    /// A mask set for the lanes whose box is empty, meaning inverted on at least one
    /// axis, so it contains no points. True of [`EMPTY`](Self::EMPTY) and of the
    /// [`intersection`](Self::intersection) of two disjoint boxes.
    #[inline(always)]
    pub fn is_empty(&self) -> V::Mask {
        let mut mask = self.0[0][1].cmp_lt(self.0[0][0]);

        for i in 1..N {
            mask |= self.0[i][1].cmp_lt(self.0[i][0]);
        }

        mask
    }

    /// A mask set for the lanes whose box contains `point` (faces inclusive).
    #[inline(always)]
    pub fn contains(&self, point: Point<V, N>) -> V::Mask {
        let mut mask = self.0[0][0].cmp_le(point[0]) & self.0[0][1].cmp_ge(point[0]);

        for i in 1..N {
            mask &= self.0[i][0].cmp_le(point[i]) & self.0[i][1].cmp_ge(point[i]);
        }

        mask
    }

    /// A mask set for the lanes where `self` fully contains `other`.
    #[inline(always)]
    pub fn contains_bounds(&self, other: &Self) -> V::Mask {
        let mut mask = self.0[0][0].cmp_le(other.0[0][0]) & self.0[0][1].cmp_ge(other.0[0][1]);

        for i in 1..N {
            mask &= self.0[i][0].cmp_le(other.0[i][0]) & self.0[i][1].cmp_ge(other.0[i][1]);
        }

        mask
    }

    /// A mask set for the lanes whose boxes overlap (touching faces count).
    ///
    /// Two boxes overlap only when their projections overlap on **every** axis,
    /// so this is a conjunction across the axes, not a disjunction.
    #[inline(always)]
    pub fn overlaps(&self, other: &Self) -> V::Mask {
        let mut mask = self.0[0][1].cmp_ge(other.0[0][0]) & self.0[0][0].cmp_le(other.0[0][1]);

        for i in 1..N {
            mask &= self.0[i][1].cmp_ge(other.0[i][0]) & self.0[i][0].cmp_le(other.0[i][1]);
        }

        mask
    }

    /// The position of `point` within the box as a fraction of its extent: the
    /// min corner maps to 0 and the max corner to 1.
    ///
    /// This is the normalized coordinate a spatial hash or a Morton/Hilbert code
    /// is built from. An axis with zero extent would give `0/0`; those lanes
    /// return the raw offset from the min corner instead of `NaN`.
    #[inline(always)]
    pub fn offset(&self, point: Point<V, N>) -> Vector<V, N> {
        Vector(core::array::from_fn(|i| {
            let offset = point[i] - self.0[i][0];
            let extent = self.0[i][1] - self.0[i][0];

            extent.cmp_gt(V::ZERO).select(offset / extent, offset)
        }))
    }

    /// The N-dimensional volume (area when `N == 2`) of the box. Meaningless for
    /// an empty box, whose diagonal is negative.
    #[inline(always)]
    pub fn volume(&self) -> V {
        let d = self.diagonal();

        let mut volume = d[0];

        for i in 1..N {
            volume *= d[i];
        }

        volume
    }

    /// The surface area of the box, the total measure of its 2N faces,
    /// `2 * sum_i prod_{j != i} d_j`.
    ///
    /// For `N == 3` this is the familiar `2 * (dx*dy + dx*dz + dy*dz)`, the cost
    /// metric behind the surface-area heuristic. For `N == 2` it degenerates to
    /// the perimeter.
    #[inline(always)]
    pub fn surface_area(&self) -> V {
        let d = self.diagonal();

        let mut total = V::ZERO;

        for i in 0..N {
            // The face perpendicular to axis `i` spans every *other* axis.
            let mut face = V::ONE;

            for j in 0..N {
                if i != j {
                    face *= d[j];
                }
            }

            total += face;
        }

        total + total
    }

    /// The index of the longest axis, **per lane**, so lane `k` gets the split axis
    /// of box `k`. This is the axis a BVH builder splits on.
    #[inline(always)]
    pub fn max_extent_axis(&self) -> V::Unsigned {
        let d = self.diagonal();

        let mut longest = d[0];
        let mut axis = V::Unsigned::ZERO;

        for i in 1..N {
            let is_longer = d[i].cmp_gt(longest);

            longest = is_longer.select(d[i], longest);
            axis = is_longer.select(V::Unsigned::splat(Element::from_u16(i as u16)), axis);
        }

        axis
    }
}

impl<V: SpatialMathWithPolicy, const N: usize> Bounds<V, N> {
    /// The center and radius of the smallest sphere enclosing the box.
    #[inline(always)]
    pub fn bounding_sphere(&self) -> (Point<V, N>, V) {
        self.bounding_sphere_p::<thermite::math::policy::DefaultPolicy>()
    }

    /// [`bounding_sphere`](Self::bounding_sphere) under an explicit precision policy.
    #[inline(always)]
    pub fn bounding_sphere_p<P: Policy>(&self) -> (Point<V, N>, V) {
        let center = self.centroid();
        let radius = (self.max_point() - center).l2_norm_p::<P>();

        (center, radius)
    }

    /// The point of the box closest to `point`; `point` itself when it is inside.
    #[inline(always)]
    pub fn closest_point(&self, point: Point<V, N>) -> Point<V, N> {
        Point(core::array::from_fn(|i| point[i].clamp(self.0[i][0], self.0[i][1])))
    }

    /// The squared distance from `point` to the box, zero when it is inside.
    ///
    /// Squared, because that is what a nearest-neighbour or sphere-vs-box test
    /// compares against. No square root needed.
    #[inline(always)]
    pub fn distance_sqr(&self, point: Point<V, N>) -> V {
        use super::vector::VectorOps as _;

        (self.closest_point(point) - point).norm_sqr()
    }
}

impl<V: FloatVector, const N: usize> Default for Bounds<V, N> {
    /// [`EMPTY`](Self::EMPTY), so `Bounds::default()` is a valid fold accumulator.
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<V: FloatVector, const N: usize> From<Point<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn from(point: Point<V, N>) -> Self {
        Self::from_point(point)
    }
}

impl<V: FloatVector, const N: usize> FromIterator<Point<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Point<V, N>>>(iter: I) -> Self {
        iter.into_iter().fold(Self::EMPTY, Self::union_point)
    }
}

impl<V: FloatVector, const N: usize> FromIterator<Bounds<V, N>> for Bounds<V, N> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Bounds<V, N>>>(iter: I) -> Self {
        iter.into_iter().fold(Self::EMPTY, Self::union)
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

impl<V: FloatVector, const N: usize> BitAndAssign for Bounds<V, N> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(rhs);
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

impl<V: FloatVector, const N: usize> BitAnd for Bounds<V, N> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
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

impl<V: FloatVector, const N: usize> GenericSelectable for Bounds<V, N> {
    type SelectableMask = V::Mask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: thermite::prelude::CastMask<M>,
    {
        let mask = <Self::SelectableMask as thermite::prelude::CastMask<M>>::mask_from(mask);

        Bounds(core::array::from_fn(|i| {
            [mask.select(t.0[i][0], f.0[i][0]), mask.select(t.0[i][1], f.0[i][1])]
        }))
    }
}
