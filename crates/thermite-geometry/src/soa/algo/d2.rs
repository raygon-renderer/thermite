use thermite::{math::SpatialMath, prelude::*};

use crate::soa::prim::{Point2, Vector, Vector2, vector::VectorOps};

/// Computes the point on the ellipse defined by the radii `e` that is closest
/// to the given point `p`.
///
/// Based on https://github.com/0xfaded/ellipse_demo/issues/1
#[inline(always)]
pub fn point_on_ellipse<V: FloatVector + SpatialMath>(p: Point2<V>, e: Vector2<V>) -> Point2<V> {
    let p_abs: Vector2<V> = p.abs().into();
    let ei = e.approx_reciprocal();
    let e2 = e * e;
    let ve = ei * Vector2::new([e2[0] - e2[1], e2[1] - e2[0]]);

    let mut t = Vector2::splat(V::FRAC_1_SQRT_2);

    // if V is a float vector with known bit size, we can determine a
    // max iteration count to avoid convergence checks.
    let precision = thermite::with_bits!([V::ZERO]: [V; 1] as fn(v: [W; _]) -> u32 {
        // determine max iterations based on float size, f32 -> 3, f64 -> 4
        size_of::<<W::Bits as GenericVector>::Element>().ilog2() + 1
    });

    let mut i = 0;

    loop {
        let prev_t = t;
        let v = ve * t * t * t;
        let u = (p_abs - v).normalize() * (e * t - v).l2_norm();
        let w = ei * (v + u);
        let nt = w.clamp(Vector::ZERO, Vector::ONE).normalize();

        if let Some(max_iters) = precision {
            i += 1;

            if i >= max_iters {
                break;
            }

            t = nt;
        } else {
            let mask = (t - prev_t).l1_norm().cmp_lt(<V as FloatVector>::EPSILON);

            if mask.all() {
                break;
            }

            t = mask.select(t, nt); // only update non-converged lanes
        }
    }

    Point2::from(t * e)
}
