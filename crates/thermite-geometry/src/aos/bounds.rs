//! Axis-aligned bounding box, one box per pair of 3-lane registers.

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use thermite::{prelude::*, simd::Simd3Vectors};

use super::{AosFloat, Matrix4, V3, vector::Vector3Ext as _};

/// An axis-aligned box `$[\mathbf{p}_{\min},\, \mathbf{p}_{\max}]$` in 3D.
///
/// One box, so the queries return plain `bool`s. The SoA counterpart's
/// [`Bounds`](crate::soa::prim::Bounds) answers with a mask because its lanes
/// hold independent boxes.
///
/// `|` unions, `&` intersects. [`EMPTY`](Self::EMPTY) is the inverted sentinel
/// and the identity for union, so it is what an incremental fold starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Bounds3<S: Simd3Vectors, E: AosFloat<S>> {
    pub min: V3<S, E>,
    pub max: V3<S, E>,
}

impl<S: Simd3Vectors, E: AosFloat<S>> Bounds3<S, E> {
    /// The inverted sentinel box (`min = +inf`, `max = -inf`), which contains no
    /// points and is the **identity for union**: `EMPTY | b == b`.
    pub const EMPTY: Self = Self {
        min: <V3<S, E> as FloatVector>::INFINITY,
        max: <V3<S, E> as FloatVector>::NEG_INFINITY,
    };

    /// The box containing everything, the **identity for intersection**.
    pub const UNIVERSE: Self = Self {
        min: <V3<S, E> as FloatVector>::NEG_INFINITY,
        max: <V3<S, E> as FloatVector>::INFINITY,
    };

    #[inline(always)]
    pub const fn new(min: V3<S, E>, max: V3<S, E>) -> Self {
        Self { min, max }
    }

    /// The degenerate box containing exactly one point.
    #[inline(always)]
    pub const fn from_point(p: V3<S, E>) -> Self {
        Self { min: p, max: p }
    }

    /// Bounds symmetric about the origin with the given (positive) half-extents,
    /// i.e. `[-half_extent, half_extent]` per axis.
    #[inline(always)]
    pub fn symmetric(half_extent: V3<S, E>) -> Self {
        Self {
            min: -half_extent,
            max: half_extent,
        }
    }

    /// The smallest box containing both.
    #[inline(always)]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The box bounding a stream of boxes.
    ///
    /// The counterpart to the [`FromIterator`] impl, which takes points. See
    /// there for why only one of the two can be a trait impl.
    #[inline(always)]
    pub fn union_all<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::EMPTY, Self::union)
    }

    /// The smallest box containing `self` and `p`.
    #[inline(always)]
    pub fn union_point(self, p: V3<S, E>) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    /// The intersection. Disjoint inputs give an inverted box, so test it with
    /// [`is_empty`](Self::is_empty).
    #[inline(always)]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Grows the box by `amount` on every side.
    #[inline(always)]
    pub fn expand(self, amount: E) -> Self {
        let amount = V3::<S, E>::splat(amount);

        Self {
            min: self.min - amount,
            max: self.max + amount,
        }
    }

    /// Grows the box by a per-axis amount on every side.
    #[inline(always)]
    pub fn expand_axes(self, amount: V3<S, E>) -> Self {
        Self {
            min: self.min - amount,
            max: self.max + amount,
        }
    }

    /// True when the box is inverted on at least one axis, so it contains no
    /// points.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.max.cmp_lt(self.min).any()
    }

    /// True when `p` is inside the box (faces inclusive).
    #[inline(always)]
    pub fn contains(&self, p: V3<S, E>) -> bool {
        (self.min.cmp_le(p) & self.max.cmp_ge(p)).all()
    }

    /// True when `self` fully contains `other`.
    #[inline(always)]
    pub fn contains_bounds(&self, other: &Self) -> bool {
        (self.min.cmp_le(other.min) & self.max.cmp_ge(other.max)).all()
    }

    /// True when the boxes overlap (touching faces count).
    ///
    /// Overlap requires the projections to overlap on **every** axis, hence the
    /// `all`. A per-axis test that used `any` would report crossing boxes that
    /// miss each other entirely.
    #[inline(always)]
    pub fn overlaps(&self, other: &Self) -> bool {
        (self.max.cmp_ge(other.min) & self.min.cmp_le(other.max)).all()
    }

    /// `$\mathbf{d} = \mathbf{p}_{\max} - \mathbf{p}_{\min}$`. Negative on any
    /// axis where the box is inverted.
    #[inline(always)]
    pub fn diagonal(&self) -> V3<S, E> {
        self.max - self.min
    }

    /// The center of the box.
    #[inline(always)]
    pub fn centroid(&self) -> V3<S, E> {
        <V3<S, E> as FloatVector>::HALF.mix(self.min, self.max)
    }

    /// `$V = d_x d_y d_z$`.
    #[inline(always)]
    pub fn volume(&self) -> E {
        self.diagonal().prod_elements3()
    }

    /// `$SA = 2(d_x d_y + d_x d_z + d_y d_z)$`, the cost metric behind the
    /// surface-area heuristic.
    ///
    /// The three products are formed by one swizzle pair and a `dot3`, rather
    /// than by extracting the components to scalars.
    #[inline(always)]
    pub fn surface_area(&self) -> E {
        let d = self.diagonal();

        let a = thermite::swizzle!(d, [0, 0, 1]).dot3(thermite::swizzle!(d, [1, 2, 2]));

        a + a
    }

    /// The index of the longest axis, the one a BVH builder splits on.
    #[inline(always)]
    pub fn max_extent_axis(&self) -> usize {
        self.diagonal().arg_minmax().1
    }

    /// The position of `p` within the box as a fraction of its extent: the min
    /// corner maps to 0 and the max corner to 1.
    ///
    /// This is the normalized coordinate a Morton or Hilbert code is built from.
    /// A zero-extent axis would give `0/0`, so those axes return the raw offset
    /// instead of `NaN`.
    #[inline(always)]
    pub fn offset(&self, p: V3<S, E>) -> V3<S, E> {
        let o = p - self.min;
        let d = self.diagonal();

        d.cmp_gt(<V3<S, E> as NumericVector>::ZERO).select(o / d, o)
    }

    /// One of the eight corners. Bit `i` of `index` picks `max` on axis `i`.
    ///
    /// Corner 0 is the min corner and corner 7 the max corner.
    #[inline(always)]
    pub fn vertex(&self, index: u8) -> V3<S, E> {
        debug_assert!(index < 8, "a 3D box has 8 corners");

        // The corner index is already a per-lane bitmask (bit i selects axis i),
        // so it converts straight into a mask instead of being unpacked into a
        // vector and compared.
        let pick_max = <V3<S, E> as GenericVector>::Mask::from_native_bitmask(index as u64);

        pick_max.select(self.max, self.min)
    }

    /// All eight corners at once, in [`vertex`](Self::vertex) order.
    #[inline(always)]
    pub fn vertices(&self) -> [V3<S, E>; 8] {
        core::array::from_fn(|i| self.vertex(i as u8))
    }

    /// The point of the box closest to `p`; `p` itself when it is inside.
    #[inline(always)]
    pub fn closest_point(&self, p: V3<S, E>) -> V3<S, E> {
        p.clamp(self.min, self.max)
    }

    /// The squared distance from `p` to the box, zero when it is inside.
    #[inline(always)]
    pub fn distance_sqr(&self, p: V3<S, E>) -> E {
        (self.closest_point(p) - p).norm_sqr()
    }

    /// The center and radius of the smallest sphere enclosing the box.
    #[inline(always)]
    pub fn bounding_sphere(&self) -> (V3<S, E>, E) {
        let center = self.centroid();

        (center, (self.max - center).norm())
    }

    /// The AABB of this box transformed by `m`, the union of its eight
    /// transformed corners.
    ///
    /// A rotated box is no longer axis-aligned, so this is conservative: the
    /// tightest axis-aligned box that still contains the transformed geometry.
    /// The corners go through
    /// [`transform_points`](Matrix4::transform_points) in one batch.
    #[inline(always)]
    pub fn transform(&self, m: &Matrix4<S, E>) -> Self {
        let corners = m.transform_points(self.vertices());

        let mut out = Self::from_point(corners[0]);

        for &c in &corners[1..] {
            out = out.union_point(c);
        }

        out
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Bounds3<S, E> {
    /// Slab test against a ray, returning the `$[t_\text{min}, t_\text{max}]$`
    /// overlap, or `None` when the ray misses.
    ///
    /// `inv_direction` is the component-wise reciprocal of the ray direction,
    /// taken as a parameter so that one ray tested against many boxes (the
    /// entire point of the AoS layout) computes it once. Axis-aligned
    /// components give signed infinities there, which is exactly what makes the
    /// min/max below produce the correct empty-or-full interval per axis.
    ///
    /// `t_max` is widened by 4 ulps
    /// (<https://jcgt.org/published/0002/02/02/>): the rounding of the slab
    /// products can push it below the true exit point, which drops a ray
    /// grazing a thin box or lying exactly on a shared BVH split plane and
    /// punches a hole in the image. Widening is conservative, so it can only add
    /// false positives, which the leaf test rejects for free.
    #[inline(always)]
    pub fn intersect_ray(&self, origin: V3<S, E>, inv_direction: V3<S, E>) -> Option<(E, E)> {
        let t0 = (self.min - origin) * inv_direction;
        let t1 = (self.max - origin) * inv_direction;

        let t_min = t0.min(t1).max_element3();
        let t_max = t0.max(t1).min_element3();

        // ONE + 2 * EPSILON is that 4-ulp bound: EPSILON is the gap between 1 and
        // the next float, i.e. 2 ulps at the midpoint of a binade. Correct for
        // f32 and f64 alike, unlike a hardcoded 1.000_000_24.
        let t_far = t_max * (E::ONE + E::EPSILON * (E::ONE + E::ONE));

        if t_far >= t_min && t_far >= E::ZERO {
            Some((t_min, t_far))
        } else {
            None
        }
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Default for Bounds3<S, E> {
    /// [`EMPTY`](Self::EMPTY), so `default()` is a valid fold accumulator.
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Builds the box bounding a stream of **points**.
///
/// There is deliberately no matching `FromIterator<Bounds3>`, and no
/// `BitOr<V3>`: `V3<S, E>` is an associated-type projection, so the coherence
/// checker cannot prove it differs from `Bounds3` and refuses the second impl of
/// either trait. Points get `FromIterator` (they have no operator alternative
/// once `BitOr<V3>` is gone) and boxes get `|`, which makes folding a stream of
/// boxes a one-liner. See [`union_all`](Bounds3::union_all).
impl<S: Simd3Vectors, E: AosFloat<S>> FromIterator<V3<S, E>> for Bounds3<S, E> {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = V3<S, E>>>(iter: T) -> Self {
        iter.into_iter().fold(Self::EMPTY, Self::union_point)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> BitOr for Bounds3<S, E> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> BitOrAssign for Bounds3<S, E> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> BitAnd for Bounds3<S, E> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> BitAndAssign for Bounds3<S, E> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(rhs);
    }
}
