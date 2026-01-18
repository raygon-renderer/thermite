use thermite::{generic::FloatVector, math::SpatialMath};

use crate::{Bounds, Point, Vector, vector::VectorOps};

pub struct Ray<V: FloatVector, const N: usize> {
    pub origin: Point<V, N>,
    pub direction: Vector<V, N>,
}

impl<V: SpatialMath, const N: usize> Ray<V, N> {
    #[inline(always)]
    pub const fn new(origin: Point<V, N>, direction: Vector<V, N>) -> Self {
        Self { origin, direction }
    }

    /// Computes the intersection of this ray with the given axis-aligned bounding box.
    /// Returns a tuple (t_min, t_max) representing the entry and exit points along the ray.
    ///
    /// If t_min > t_max, there is no intersection.
    #[inline(always)]
    pub fn intersects(&self, aabb: &Bounds<V, N>) -> (V, V) {
        let min_walls = Point(aabb.0.map(|pair| pair[0]));
        let max_walls = Point(aabb.0.map(|pair| pair[1]));

        let id = self.direction.reciprocal();

        let t_min_walls = (min_walls - self.origin) * id;
        let t_max_walls = (max_walls - self.origin) * id;

        let mut start = t_min_walls.min(t_max_walls).0;
        let mut end = t_min_walls.max(t_max_walls).0;

        thermite::math::algorithms::reduce_in_place(&mut start, V::max);
        thermite::math::algorithms::reduce_in_place(&mut end, V::min);

        (start[0], end[0])
    }
}
