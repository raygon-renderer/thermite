//! A single ray, and conservative floating-point error bounds on it.

use core::ops::Neg;

use thermite::{prelude::*, simd::Simd3Vectors};

use super::{AosFloat, Matrix4, V3, matrix::gamma, vector::Vector3Ext as _};

/// A ray `$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}$`.
///
/// There is no `t_max` field: a `t` range is the traversal state of whatever is
/// consuming the ray, not part of the geometry. Carry it alongside, and note
/// that [`transform_normalized`](Self::transform_normalized) hands back the
/// scale factor precisely so that a caller holding a range can rescale it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Ray3<S: Simd3Vectors, E: AosFloat<S>> {
    pub origin: V3<S, E>,
    pub direction: V3<S, E>,
}

/// Conservative floating-point error bounds on a transformed ray.
///
/// Each component `$e_i$` satisfies `$|\tilde{x}_i - x_i| \le e_i$`, with
/// `$\tilde{x}_i$` the computed value and `$x_i$` the exact one. A renderer uses
/// these to offset ray origins past the surface they came from, which kills
/// self-intersection acne without a magic epsilon.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RayError3<S: Simd3Vectors, E: AosFloat<S>> {
    pub pos: V3<S, E>,
    pub dir: V3<S, E>,
}

impl<S: Simd3Vectors, E: AosFloat<S>> RayError3<S, E> {
    pub const ZERO: Self = Self {
        pos: <V3<S, E> as NumericVector>::ZERO,
        dir: <V3<S, E> as NumericVector>::ZERO,
    };

    #[inline(always)]
    pub const fn new(pos: V3<S, E>, dir: V3<S, E>) -> Self {
        Self { pos, dir }
    }

    /// Error on the position only, with the direction taken as exact.
    #[inline(always)]
    pub const fn position(error: V3<S, E>) -> Self {
        Self {
            pos: error,
            dir: <V3<S, E> as NumericVector>::ZERO,
        }
    }

    /// Error on the direction only, with the position taken as exact.
    #[inline(always)]
    pub const fn direction(error: V3<S, E>) -> Self {
        Self {
            pos: <V3<S, E> as NumericVector>::ZERO,
            dir: error,
        }
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Default for RayError3<S, E> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Ray3<S, E> {
    #[inline(always)]
    pub const fn new(origin: V3<S, E>, direction: V3<S, E>) -> Self {
        Self { origin, direction }
    }

    /// Evaluates `$\mathbf{o} + t\,\mathbf{d}$`.
    #[inline(always)]
    pub fn at(&self, t: E) -> V3<S, E> {
        self.direction.mul_adde(V3::<S, E>::splat(t), self.origin)
    }

    /// Advances the origin to `$\mathbf{r}(t)$`.
    ///
    /// Any `t` range the caller keeps must be shifted by `-t` to stay in the
    /// same parameterization. The ray itself carries none.
    #[inline(always)]
    pub fn move_to(mut self, t: E) -> Self {
        self.origin = self.at(t);
        self
    }

    /// The component-wise reciprocal of the direction: the precomputed input to
    /// [`Bounds3::intersect_ray`](super::Bounds3::intersect_ray).
    ///
    /// Kept out of the ray so one ray tested against many boxes pays for it
    /// once, which is the whole reason the AoS layout exists. Axis-aligned
    /// components divide by zero on purpose, and the signed infinity is what makes
    /// the slab test's min/max produce the correct interval for that axis.
    ///
    /// Deliberately [`reciprocal_exact`](super::Vector3Ext::reciprocal_exact)
    /// and not `CoreMath::reciprocal`. See there for why the approximate one
    /// produces `NaN` at zero and silently drops every axis-aligned ray.
    #[inline(always)]
    pub fn inv_direction(&self) -> V3<S, E> {
        self.direction.reciprocal_exact()
    }

    /// Offsets a hit point off the surface so a secondary ray cannot
    /// re-intersect the geometry it came from.
    ///
    /// This is PBRT's `OffsetRayOrigin`: push along the normal by the normal's
    /// projection onto the error bound, then nudge each component one ulp
    /// further in the direction of travel so the rounding of the addition itself
    /// cannot land back on the surface. Where a fixed epsilon fails, this
    /// scales with the actual error at that point, so it works at any distance
    /// from the origin.
    ///
    /// `normal` should point to the side the new ray leaves on. Flip it (or pass
    /// `direction` through
    /// [`faceforward`](super::Vector3Ext::faceforward)) for a transmitted ray.
    #[inline(always)]
    pub fn offset_origin(p: V3<S, E>, p_error: V3<S, E>, normal: V3<S, E>, direction: V3<S, E>) -> V3<S, E> {
        let d = normal.abs().dot3(p_error);

        // The side test collapses to one scalar, so a real branch is right here -
        // there is one point, not a batch.
        let offset = normal * V3::<S, E>::splat(if direction.dot3(normal) < E::ZERO { -d } else { d });

        let out = p + offset;

        // One ulp further out per component: the add above rounds, and rounding
        // toward the surface is exactly the failure this guards against.
        //
        // Stays in the register. The lanes *are* the components here, so the
        // per-component sign test is a vector compare rather than a scalar
        // branch. Pulling the components out with `extractv` and writing them
        // back with `insertv` would be a dozen-odd ops on a runtime index and
        // typically round-trips the register through memory.
        //
        // Both steps are computed unconditionally from the *same* `out` and then
        // blended, rather than chained as `next_up_c(..).next_down_c(..)`.
        // Chaining would serialize them and hand the second a different input,
        // so the two would share nothing. Computed side by side they are
        // independent (they pipeline) and every internal NaN, infinity and
        // signed-zero guard is a common subexpression the optimizer folds once.
        // The wasted work is only the blend, since the steps themselves would each
        // have been paid anyway.
        //
        // Measured on AVX2, the whole function: 230 instructions and 17 stack
        // references for the scalar `extractv`/`insertv` loop this replaced,
        // 66 and 11 for the chained masked form, 58 and 9 for the blend below.
        let zero = <V3<S, E> as NumericVector>::ZERO;

        let up = out.next_up();
        let down = out.next_down();

        // A zero offset means no error to escape on that axis, so it falls
        // through both masks and keeps `out` exactly.
        let stepped = offset.cmp_lt(zero).select(down, out);

        offset.cmp_gt(zero).select(up, stepped)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Ray3<S, E> {
    /// Transforms the ray by `m`: the origin as a point, the direction as a
    /// vector.
    ///
    /// The direction is **not** renormalized, so it absorbs the matrix's scale
    /// and every `t` stays valid unchanged: `$\mathbf{r}'(t) = M\mathbf{r}(t)$`
    /// for all `t`. This is the usual convention for instance transforms: a `t`
    /// found in object space is immediately a `t` in world space.
    #[inline(always)]
    pub fn transform(self, m: &Matrix4<S, E>) -> Self {
        Self {
            origin: m.transform_point(self.origin),
            direction: m.transform_vector(self.direction),
        }
    }

    /// [`transform`](Self::transform), plus `$\gamma_3$` error bounds, with the
    /// origin's positional error absorbed into the ray parameter.
    #[inline(always)]
    pub fn transform_with_error(self, m: &Matrix4<S, E>) -> (Self, RayError3<S, E>) {
        let (d, de) = transform_vector_with_error(m, self.direction);
        let (o, oe) = transform_point_with_error(m, self.origin);

        let ray = Self {
            origin: nudged_origin(d, o, oe),
            direction: d,
        };

        (ray, RayError3 { pos: oe, dir: de })
    }

    /// [`transform_with_error`](Self::transform_with_error) for a ray that
    /// already carries error from an earlier transform, propagating it through
    /// `m`.
    ///
    /// Chaining instance transforms without this would understate the error at
    /// every step after the first, which is exactly when a spawned ray starts
    /// re-hitting its own surface.
    #[inline(always)]
    pub fn transform_propagate_error(self, m: &Matrix4<S, E>, error: RayError3<S, E>) -> (Self, RayError3<S, E>) {
        let (d, de) = transform_vector_propagate_error(m, self.direction, error.dir);
        let (o, oe) = transform_point_propagate_error(m, self.origin, error.pos);

        let ray = Self {
            origin: nudged_origin(d, o, oe),
            direction: d,
        };

        (ray, RayError3 { pos: oe, dir: de })
    }

    /// Transforms the ray by `m` and renormalizes the direction, returning the
    /// scale factor `$\|M\mathbf{d}\|$` that was divided out.
    ///
    /// Because the direction is rescaled, `t` values do **not** carry across: a
    /// caller holding a `t` range must multiply it by the returned length.
    #[inline(always)]
    pub fn transform_normalized(self, m: &Matrix4<S, E>) -> (Self, E) {
        let d = m.transform_vector(self.direction);
        let (o, oe) = transform_point_with_error(m, self.origin);

        let origin = nudged_origin(d, o, oe);
        let (direction, length) = d.normalize_norm();

        (Self { origin, direction }, length)
    }
}

/// Shifts the origin along `d` by just enough to swallow the transformed
/// origin's error bound, turning a positional uncertainty into an uncertainty in
/// `t` (which the intersector already tolerates).
///
/// `$\delta t = |\mathbf{d}| \cdot \mathbf{e}_o / \|\mathbf{d}\|^2$`
///
/// Returns just the nudged origin, and is generic over the vector type rather
/// than over `(S, E)`: inference cannot see through the `V3<S, E>` projection to
/// recover `S`, so the type has to come from the argument.
#[inline(always)]
fn nudged_origin<V: LinAlg3Vector>(d: V, o: V, oe: V) -> V {
    let zero = <V::Element as thermite::element::Element>::ZERO;

    let length_sq = d.dot3(d);

    // A zero-length direction has no parameterization to push the error into.
    let dt = if length_sq > zero {
        d.abs().dot3(oe) / length_sq
    } else {
        zero
    };

    d.mul_adde(V::splat(dt), o)
}

/// Transforms a direction and bounds the floating-point error of the result.
///
/// Each row of the product is a dot product of 3 terms, so its error is bounded
/// by `$\gamma_3 \sum_j |M_{rj}||v_j|$`.
#[inline(always)]
fn transform_vector_with_error<S: Simd3Vectors, E: AosFloat<S>>(
    m: &Matrix4<S, E>,
    v: V3<S, E>,
) -> (V3<S, E>, V3<S, E>) {
    let abs_m = m.linear().map(|c| c.abs());

    let error = v.abs().mat3_vec3_product::<true>(&abs_m);

    (m.transform_vector(v), error * V3::<S, E>::splat(gamma::<E>(3)))
}

/// Transforms a point and bounds the floating-point error of the result.
///
/// The translation column contributes because the point's implicit `w = 1`
/// multiplies it.
#[inline(always)]
fn transform_point_with_error<S: Simd3Vectors, E: AosFloat<S>>(m: &Matrix4<S, E>, p: V3<S, E>) -> (V3<S, E>, V3<S, E>) {
    let abs_p = p.abs();
    let [c0, c1, c2] = m.linear().map(|c| c.abs());
    let c3 = m.translation().abs();

    let bound = c0.mul_adde(
        abs_p.broadcast::<0>(),
        c1.mul_adde(abs_p.broadcast::<1>(), c2.mul_adde(abs_p.broadcast::<2>(), c3)),
    );

    (m.transform_point(p), bound * V3::<S, E>::splat(gamma::<E>(3)))
}

/// Transforms a point that already carries an error interval, propagating it.
///
/// The incoming error is itself transformed by `$|M|$`, and that transform
/// rounds too, hence the extra `$(1 + \gamma_3)$` factor:
///
/// `$e' = \gamma_3 \sum_j |M_{rj}||p_j| + (1 + \gamma_3)\sum_j |M_{rj}| e_j$`
#[inline(always)]
fn transform_point_propagate_error<S: Simd3Vectors, E: AosFloat<S>>(
    m: &Matrix4<S, E>,
    p: V3<S, E>,
    error: V3<S, E>,
) -> (V3<S, E>, V3<S, E>) {
    let (out, from_p) = transform_point_with_error(m, p);

    let abs_e = error.abs();
    let [c0, c1, c2] = m.linear().map(|c| c.abs());

    // No translation column here: the incoming error is a displacement, so its
    // implicit w is 0.
    let from_e = c0.mul_adde(
        abs_e.broadcast::<0>(),
        c1.mul_adde(abs_e.broadcast::<1>(), c2 * abs_e.broadcast::<2>()),
    );

    let g3 = V3::<S, E>::splat(gamma::<E>(3));

    (out, from_p + from_e.mul_adde(g3, from_e))
}

/// Transforms a direction that already carries an error interval, propagating it.
#[inline(always)]
fn transform_vector_propagate_error<S: Simd3Vectors, E: AosFloat<S>>(
    m: &Matrix4<S, E>,
    v: V3<S, E>,
    error: V3<S, E>,
) -> (V3<S, E>, V3<S, E>) {
    let (out, from_v) = transform_vector_with_error(m, v);

    let abs_e = error.abs();
    let [c0, c1, c2] = m.linear().map(|c| c.abs());

    let from_e = c0.mul_adde(
        abs_e.broadcast::<0>(),
        c1.mul_adde(abs_e.broadcast::<1>(), c2 * abs_e.broadcast::<2>()),
    );

    let g3 = V3::<S, E>::splat(gamma::<E>(3));

    (out, from_v + from_e.mul_adde(g3, from_e))
}

impl<S: Simd3Vectors, E: AosFloat<S>> Matrix4<S, E> {
    /// Transforms a direction and bounds the floating-point error of the result.
    /// See [`gamma`](super::matrix::gamma).
    #[inline(always)]
    pub fn transform_vector_with_error(&self, v: V3<S, E>) -> (V3<S, E>, V3<S, E>) {
        transform_vector_with_error(self, v)
    }

    /// Transforms a point and bounds the floating-point error of the result.
    #[inline(always)]
    pub fn transform_point_propagate_error(&self, p: V3<S, E>, error: V3<S, E>) -> (V3<S, E>, V3<S, E>) {
        transform_point_propagate_error(self, p, error)
    }

    /// Transforms a direction carrying an error interval, propagating it.
    #[inline(always)]
    pub fn transform_vector_propagate_error(&self, v: V3<S, E>, error: V3<S, E>) -> (V3<S, E>, V3<S, E>) {
        transform_vector_propagate_error(self, v, error)
    }

    /// Transforms a point and bounds the floating-point error of the result.
    #[inline(always)]
    pub fn transform_point_with_error(&self, p: V3<S, E>) -> (V3<S, E>, V3<S, E>) {
        transform_point_with_error(self, p)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Neg for Ray3<S, E> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self {
        self.direction = -self.direction;
        self
    }
}
