use thermite::{
    math::{
        SpatialMathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
    vector::FloatVector,
};

use super::{Bounds, Point, Vector};

pub struct Ray<V: FloatVector, const N: usize> {
    pub origin: Point<V, N>,
    pub direction: Vector<V, N>,
}

/// One ray in the interleaved (array-of-structures) layout a renderer actually
/// stores: origin then direction, `2 * N` contiguous scalars, no padding.
///
/// This is the element type of the span [`Ray::load_interleaved`] reads. It is
/// deliberately a plain scalar record, not a SIMD type: it is what is on disk and
/// in the vertex buffer.
///
/// A `&[RayRecord<f32, 3>]` in memory, and what one load pulls out of it
/// (`LANES = 4` here, so four rays per batch):
///
/// ```text
///           ray 0                 ray 1                 ray 2         ...
///   |---------------------|--------------------|---------------------|-- more rays...
///     ox oy oz   dx dy dz   ox oy oz  dx dy dz    ox oy oz  dx dy dz  ...
///   \__origin_/ \__dir___/                           AoS in memory
///
///   load_interleaved  =>  two records of N components, NOT 2N streams
///
///   origin.0[0] = [ ox0 ox1 ox2 ox3 ]   \
///   origin.0[1] = [ oy0 oy1 oy2 oy3 ]    > one LD3 (transposed in the load unit)
///   origin.0[2] = [ oz0 oz1 oz2 oz3 ]   /
///
///   direction.0[0] = [ dx0 dx1 dx2 dx3 ]   \
///   direction.0[1] = [ dy0 dy1 dy2 dy3 ]    > a second LD3
///   direction.0[2] = [ dz0 dz1 dz2 dz3 ]   /
/// ```
///
/// Element `c` of record `j` of ray `lane` sits at flat offset
/// `lane * 2N + j * N + c`, which is exactly the `M = 2, C = N` contract of
/// [`GenericVector::load_deinterleaved_arrays`](thermite::vector::GenericVector::load_deinterleaved_arrays).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RayRecord<E, const N: usize> {
    pub origin: [E; N],
    pub direction: [E; N],
}

impl<V: FloatVector, const N: usize> Ray<V, N> {
    #[inline(always)]
    pub const fn new(origin: Point<V, N>, direction: Vector<V, N>) -> Self {
        Self { origin, direction }
    }

    /// Load `V::LANES` rays from an interleaved span of [`RayRecord`]s,
    /// transposing AoS -> SoA on the way in.
    ///
    /// A ray is not `2 * N` independent streams; it is **two records of `N`
    /// components**, and saying so is what lets a backend use its structural
    /// loads: on NEON this becomes two `LD3`s per chunk (the transpose happens in
    /// the load unit), where a flat `2 * N`-stream view would fall back to a
    /// register shuffle network - no `LDn` covers 6 streams. Measured on NEON
    /// `f32x4`: 12 instructions this way, ~25 flat.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `2 * N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn load_interleaved(ptr: *const RayRecord<V::Element, N>) -> Self {
        let [origin, direction] = unsafe { V::load_deinterleaved_arrays::<2, N>(ptr as *const V::Element) };

        Self {
            origin: Point::new(origin),
            direction: Vector::new(direction),
        }
    }

    /// Store `V::LANES` rays back to an interleaved span - the exact inverse of
    /// [`load_interleaved`](Self::load_interleaved) (two `ST3`s on NEON).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `2 * N * V::LANES` elements.
    #[inline(always)]
    pub unsafe fn store_interleaved(self, ptr: *mut RayRecord<V::Element, N>) {
        unsafe {
            V::store_interleaved_arrays::<2, N>(ptr as *mut V::Element, [self.origin.0, self.direction.0]);
        }
    }
}

pub trait RayOpsWithPolicy<V: SpatialMathWithPolicy, const N: usize>: Sized {
    /// Computes the intersection of this ray with the given axis-aligned bounding box.
    /// Returns a tuple (t_min, t_max) representing the entry and exit points along the ray.
    ///
    /// If t_min > t_max, there is no intersection.
    ///
    /// The `inverse_direction` parameter should be the component-wise reciprocal of the ray's direction. This
    /// is manually given to allow avoiding redundant computations when testing multiple AABBs against the same ray.
    fn intersects_aabb_p<P: Policy>(&self, aabb: &Bounds<V, N>, inverse_direction: &Vector<V, N>) -> (V, V);
}

#[rustfmt::skip]
pub trait RayOps<V: SpatialMathWithPolicy, const N: usize>: RayOpsWithPolicy<V, N> {
    #[inline(always)]
    fn intersects_aabb(&self, aabb: &Bounds<V, N>, inverse_direction: &Vector<V, N>) -> (V, V) { self.intersects_aabb_p::<DefaultPolicy>(aabb, inverse_direction) }
}

impl<R, V: SpatialMathWithPolicy, const N: usize> RayOps<V, N> for R where R: RayOpsWithPolicy<V, N> {}

impl<V: SpatialMathWithPolicy, const N: usize> RayOpsWithPolicy<V, N> for Ray<V, N> {
    #[inline(always)]
    fn intersects_aabb_p<P: Policy>(&self, aabb: &Bounds<V, N>, inverse_direction: &Vector<V, N>) -> (V, V) {
        let t_min_walls = (aabb.min_point() - self.origin) * *inverse_direction;
        let t_max_walls = (aabb.max_point() - self.origin) * *inverse_direction;

        let mut start = t_min_walls.min(t_max_walls);
        let mut end = t_min_walls.max(t_max_walls);

        thermite::math::algorithms::reduce_in_place(&mut start.0, V::max);
        thermite::math::algorithms::reduce_in_place(&mut end.0, V::min);

        (start[0], end[0]) // t_min, t_max
    }
}
