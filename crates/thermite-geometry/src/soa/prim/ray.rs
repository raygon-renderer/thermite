use core::ops::Neg;

use thermite::{
    mask::GenericMask,
    math::{
        SpatialMathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
    vector::FloatVector,
};

use super::{
    Bounds, Matrix, Point, Vector,
    matrix::gamma,
    vector::{VectorOps as _, VectorOpsWithPolicy as _},
};

/// A batch of rays `$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}$`, one ray per lane.
///
/// There is deliberately no `tmax` field: the `2 * N` interleaved layout of
/// [`RayRecord`] is the wire format, and a per-ray `t` range is the traversal
/// state of whatever is consuming these rays, not part of the geometry. Carry it
/// alongside as a plain `V`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// A ray is not `2 * N` independent streams. It is **two records of `N`
    /// components**, and saying so is what lets a backend use its structural
    /// loads: on NEON this becomes two `LD3`s per chunk (the transpose happens in
    /// the load unit), where a flat `2 * N`-stream view would fall back to a
    /// register shuffle network, since no `LDn` covers 6 streams. Measured on NEON
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

    /// Store `V::LANES` rays back to an interleaved span, the exact inverse of
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

    /// Evaluates `$\mathbf{o} + t\,\mathbf{d}$`, one `t` per lane.
    #[inline(always)]
    pub fn at(&self, t: V) -> Point<V, N> {
        Point::from(self.direction.mul_adde(t, Vector::from(self.origin)))
    }

    /// Advances the origin to `$\mathbf{r}(t)$`.
    ///
    /// Any `t` range the caller keeps for this ray must be shifted by `-t` to stay
    /// in the same parameterization. The ray itself carries none.
    #[inline(always)]
    pub fn move_to(mut self, t: V) -> Self {
        self.origin = self.at(t);
        self
    }
}

impl<V: SpatialMathWithPolicy, const N: usize> Ray<V, N> {
    /// The component-wise reciprocal of the direction, the precomputed input to
    /// [`RayOps::intersects_aabb`].
    ///
    /// Kept out of the ray so that one ray tested against many boxes pays for it
    /// once. Axis-aligned lanes divide by zero here on purpose: the resulting
    /// signed infinity is exactly what makes the slab test's min/max produce the
    /// correct empty-or-full interval for that axis.
    ///
    /// Uses [`reciprocal_exact`](super::vector::VectorOpsWithPolicy::reciprocal_exact)
    /// rather than the policy-based `reciprocal`, which is **not** merely less
    /// accurate here but outright wrong: on a backend with `HAS_APPROX_RCP` its
    /// Newton step turns `1/0` into `NaN` instead of an infinity, and a `NaN`
    /// fails every comparison in the slab test, so every axis-aligned ray was
    /// reported as a miss.
    #[inline(always)]
    pub fn inv_direction(&self) -> Vector<V, N> {
        self.direction.reciprocal_exact()
    }

    /// Offsets a hit point off the surface so a secondary ray cannot
    /// re-intersect the geometry it came from.
    ///
    /// This is the consumer the rest of the error-bound machinery exists for.
    /// [`RayError`], [`Matrix::transform_point_with_error`] and
    /// [`gamma`](super::gamma) all produce conservative bounds on where a
    /// computed point really is, and this turns such a bound into a ray origin that
    /// is *provably* outside the surface, which is what removes
    /// self-intersection acne without the magic epsilon the crate docs
    /// otherwise disparage.
    ///
    /// PBRT's `OffsetRayOrigin`: push along the normal by the normal's
    /// projection onto the error bound, `$\delta = |\mathbf{n}| \cdot
    /// \mathbf{e}$`, then nudge each component one ulp further in the direction
    /// of travel, because the rounding of that very addition can land back on
    /// the surface. Because `$\delta$` scales with the actual error at the
    /// point, this stays correct arbitrarily far from the origin, where a fixed
    /// epsilon does not.
    ///
    /// `normal` is the surface normal and `direction` the direction the new ray
    /// leaves in. The offset flips to whichever side `direction` points, so a
    /// transmitted ray needs no special handling at the call site. All of that
    /// is per-lane: the side test is a mask, not a branch, so one batch may
    /// carry reflected and transmitted rays at once.
    #[inline(always)]
    pub fn offset_origin(
        p: Point<V, N>,
        p_error: Vector<V, N>,
        normal: Vector<V, N>,
        direction: Vector<V, N>,
    ) -> Point<V, N> {
        use super::vector::VectorOps as _;

        let delta = normal.abs().dot(&p_error);

        // Leaving on the far side of the surface flips the offset. A masked
        // negate rather than a select: one op per lane, no blend.
        let leaving_back = direction.dot(&normal).cmp_lt(V::ZERO);
        let delta = delta.neg_c(leaving_back);

        let offset = normal * delta;
        let out = Point::from(Vector::from(p) + offset);

        // One ulp further out per component. The add above rounds, and rounding
        // *toward* the surface is exactly the failure this guards against, so
        // each component steps away along its own offset's sign. A zero offset
        // on an axis means no error to escape there, and is left alone.
        //
        // Both steps are computed from the same input and blended, rather than
        // chained as `next_up_c(..).next_down_c(..)`: chaining serializes them
        // and feeds the second a different value, so the two share nothing.
        // Side by side they pipeline, and the NaN, infinity and signed-zero
        // guards inside `next_up`/`next_down` become common subexpressions the
        // optimizer folds once instead of twice. Measured on the AoS twin of
        // this function, that alone was 66 instructions against 58.
        let mut result = out;

        for i in 0..N {
            let up = out[i].next_up();
            let down = out[i].next_down();

            let stepped = offset[i].cmp_lt(V::ZERO).select(down, out[i]);

            result[i] = offset[i].cmp_gt(V::ZERO).select(up, stepped);
        }

        result
    }
}

impl<V: FloatVector, const N: usize> Neg for Ray<V, N> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self::Output {
        self.direction = -self.direction;
        self
    }
}

/// Conservative floating-point error bounds on a transformed ray.
///
/// Each component `$e_i$` satisfies `$|\tilde{x}_i - x_i| \le e_i$`, where
/// `$\tilde{x}_i$` is the computed value and `$x_i$` the exact one. A renderer uses
/// these to offset ray origins past the surface they came from, killing
/// self-intersection acne without a magic epsilon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayError<V: FloatVector, const N: usize> {
    pub pos: Vector<V, N>,
    pub dir: Vector<V, N>,
}

impl<V: FloatVector, const N: usize> RayError<V, N> {
    pub const ZERO: Self = Self {
        pos: Vector::ZERO,
        dir: Vector::ZERO,
    };

    #[inline(always)]
    pub const fn new(pos: Vector<V, N>, dir: Vector<V, N>) -> Self {
        Self { pos, dir }
    }

    #[inline(always)]
    pub const fn position(error: Vector<V, N>) -> Self {
        Self {
            pos: error,
            dir: Vector::ZERO,
        }
    }

    #[inline(always)]
    pub const fn direction(error: Vector<V, N>) -> Self {
        Self {
            pos: Vector::ZERO,
            dir: error,
        }
    }
}

impl<V: FloatVector, const N: usize> Default for RayError<V, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<V: SpatialMathWithPolicy> Ray<V, 3> {
    /// Transforms the ray by `m`: the origin as a point, the direction as a vector.
    ///
    /// The direction is **not** renormalized, so it absorbs the matrix's scale and
    /// every `t` stays valid unchanged: `$\mathbf{r}'(t) = M\,\mathbf{r}(t)$` for
    /// all `t`. This is the usual convention for instance transforms in a renderer:
    /// a `t` found in object space is immediately a `t` in world space. Use
    /// [`transform_normalized`](Self::transform_normalized) when a unit-length
    /// direction matters more than that.
    #[inline(always)]
    pub fn transform(self, m: &Matrix<V, 4, 4>) -> Self {
        Ray {
            origin: m.transform_point(self.origin),
            direction: m.transform_vector(self.direction),
        }
    }

    /// Transforms the ray by `m` and renormalizes the direction, returning the
    /// scale factor `$\|M\mathbf{d}\|$` that was divided out.
    ///
    /// Because the direction is rescaled, `t` values do **not** carry across: a
    /// caller holding a `t` range must multiply it by the returned length (that is
    /// what raygon's `Ray::tmax` does internally, and why the length is handed back
    /// rather than dropped).
    ///
    /// The origin is nudged forward by
    /// `$\delta t = |\mathbf{d}| \cdot \mathbf{e}_o / \|\mathbf{d}\|^2$` to absorb the
    /// positional error of the transformed origin into the ray parameter.
    ///
    /// Lanes whose direction degenerates to zero under `m` cannot be normalized. The
    /// scalar original panics there, and a SoA batch cannot, because the other lanes are
    /// perfectly good rays, so those lanes come back with a zero direction, an
    /// un-nudged origin, and a returned length of zero, all finite and never NaN.
    #[inline(always)]
    pub fn transform_normalized(self, m: &Matrix<V, 4, 4>) -> (Self, V) {
        let d = m.transform_vector(self.direction);
        let (o, oe) = m.transform_point_with_error(self.origin);

        nudge_and_normalize(d, o, oe)
    }

    /// [`transform`](Self::transform), plus `$\gamma_3$` error bounds on the result,
    /// with the origin's positional error absorbed into the ray parameter.
    ///
    /// Like `transform` and unlike raygon's version, the direction keeps the scale of
    /// `m`, so `t` values survive the transform untouched.
    #[inline(always)]
    pub fn transform_with_error(self, m: &Matrix<V, 4, 4>) -> (Self, RayError<V, 3>) {
        let (d, de) = m.transform_vector_with_error(self.direction);
        let (o, oe) = m.transform_point_with_error(self.origin);

        (nudge(d, o, oe), RayError { pos: oe, dir: de })
    }

    /// [`transform_with_error`](Self::transform_with_error) for a ray that already
    /// carries error from an earlier transform, propagating it through `m`.
    #[inline(always)]
    pub fn transform_propagate_error(self, m: &Matrix<V, 4, 4>, error: RayError<V, 3>) -> (Self, RayError<V, 3>) {
        let (d, de) = m.transform_vector_with_error(self.direction);
        let (o, oe) = m.transform_point_propagate_error(self.origin, error.pos);

        // The incoming direction error is itself transformed by |M|, and the transform
        // of it rounds too, hence the extra (1 + gamma_3) factor. Same shape as
        // `Matrix::transform_point_propagate_error`, minus the translation column.
        let mut from_e = [V::ZERO; 3];

        for r in 0..3 {
            from_e[r] = m.0[0][r].abs().mul_adde(
                error.dir[0].abs(),
                m.0[1][r]
                    .abs()
                    .mul_adde(error.dir[1].abs(), m.0[2][r].abs() * error.dir[2].abs()),
            );
        }

        let from_e = Vector(from_e);
        let g3 = gamma::<V>(3);

        let de = de + from_e.mul_adde(g3, from_e);

        (nudge(d, o, oe), RayError { pos: oe, dir: de })
    }
}

/// Shifts the origin along `d` by just enough to swallow the transformed origin's
/// error bound `oe`, turning a positional uncertainty into an uncertainty in `t`
/// (which the intersector already tolerates).
///
/// `$\delta t = |\mathbf{d}| \cdot \mathbf{e}_o / \|\mathbf{d}\|^2$`
#[inline(always)]
fn nudge<V: SpatialMathWithPolicy>(d: Vector<V, 3>, o: Point<V, 3>, oe: Vector<V, 3>) -> Ray<V, 3> {
    let length_sq = d.norm_sqr();

    // A zero-length direction has no parameterization to push the error into. The
    // scalar original panics, and a lane cannot, so it selects a denominator of one and
    // a numerator of zero, which leaves the origin exactly where it was.
    let ok = length_sq.cmp_gt(V::ZERO) & length_sq.is_finite();

    let dt = d.abs().dot(&oe) / ok.select(length_sq, V::ONE);
    let dt = ok.select(dt, V::ZERO);

    Ray {
        origin: Point::from(d.mul_adde(dt, Vector::from(o))),
        direction: d,
    }
}

/// [`nudge`], then scale the direction to unit length, also returning that length so
/// the caller can rescale its own `t` values.
#[inline(always)]
fn nudge_and_normalize<V: SpatialMathWithPolicy>(d: Vector<V, 3>, o: Point<V, 3>, oe: Vector<V, 3>) -> (Ray<V, 3>, V) {
    let ray = nudge(d, o, oe);

    let length_sq = d.norm_sqr();
    let ok = length_sq.cmp_gt(V::ZERO) & length_sq.is_finite();

    let length = ok.select(length_sq.sqrt(), V::ZERO);

    // Dividing a degenerate (zero) direction by its zero length is 0/0 = NaN, which
    // would poison every lane of a later reduction. Divide by one instead and let the
    // lane keep its zero direction. The returned length of zero flags it.
    let ray = Ray {
        origin: ray.origin,
        direction: ray.direction / ok.select(length, V::ONE),
    };

    (ray, length)
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

    /// [`intersects_aabb_p`](Self::intersects_aabb_p), plus the hit mask.
    ///
    /// The mask is set for the lanes that actually hit the box: `t_max >= t_min`
    /// (the slabs overlap) and `t_max >= 0` (the overlap is not entirely behind the
    /// origin). It is the SoA form of the scalar test's `Option`; the `t` values in
    /// the missing lanes are still returned, and are meaningless.
    fn intersects_aabb_mask_p<P: Policy>(
        &self,
        aabb: &Bounds<V, N>,
        inverse_direction: &Vector<V, N>,
    ) -> (V, V, V::Mask);
}

#[rustfmt::skip]
pub trait RayOps<V: SpatialMathWithPolicy, const N: usize>: RayOpsWithPolicy<V, N> {
    #[inline(always)]
    fn intersects_aabb(&self, aabb: &Bounds<V, N>, inverse_direction: &Vector<V, N>) -> (V, V) { self.intersects_aabb_p::<DefaultPolicy>(aabb, inverse_direction) }

    #[inline(always)]
    fn intersects_aabb_mask(&self, aabb: &Bounds<V, N>, inverse_direction: &Vector<V, N>) -> (V, V, V::Mask) { self.intersects_aabb_mask_p::<DefaultPolicy>(aabb, inverse_direction) }
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

        // <https://jcgt.org/published/0002/02/02/> shows that the rounding of the
        // slab products can push t_max below the true exit point, so a ray grazing a
        // thin box (or one that lies exactly on a shared BVH split plane) is reported
        // as a miss and a hole appears in the image. Widening t_max by 4 ulps is the
        // paper's recommended fix, and conservative, so it can only ever add false
        // positives, which the leaf test then rejects for free.
        //
        // ONE + 2 * EPSILON is that 4-ulp bound: EPSILON is the gap between 1 and the
        // next float, i.e. 2 ulps at the midpoint of a binade, so this is the same
        // 1.000_000_24 the scalar f32 code hardcodes, and stays correct for f64.
        let t_far_mul = V::ONE + <V as FloatVector>::EPSILON * V::TWO;

        (start[0], end[0] * t_far_mul) // t_min, t_max
    }

    #[inline(always)]
    fn intersects_aabb_mask_p<P: Policy>(
        &self,
        aabb: &Bounds<V, N>,
        inverse_direction: &Vector<V, N>,
    ) -> (V, V, V::Mask) {
        let (t_min, t_max) = self.intersects_aabb_p::<P>(aabb, inverse_direction);

        // The scalar original branches to an `Option` here. One ray per lane means the
        // answer differs per lane, so the branch becomes the mask itself.
        (t_min, t_max, t_max.cmp_ge(t_min) & t_max.cmp_ge(V::ZERO))
    }
}
