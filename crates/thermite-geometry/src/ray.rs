use thermite::{
    generic::FloatVector,
    math::{
        SpatialMathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
};

use crate::{Bounds, Point, Vector};

pub struct Ray<V: FloatVector, const N: usize> {
    pub origin: Point<V, N>,
    pub direction: Vector<V, N>,
}

impl<V: FloatVector, const N: usize> Ray<V, N> {
    #[inline(always)]
    pub const fn new(origin: Point<V, N>, direction: Vector<V, N>) -> Self {
        Self { origin, direction }
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
