use crate::{math::policy::PrecisionPolicy, vector::generic::GenericMask as _};

use super::*;

#[inline(always)]
pub fn newtons_method<V: CoreMath, P: Policy, F>(
    mut x: V,
    tolerance: V,
    bounds: Option<(V, V)>,
    mut f: F,
) -> Result<V, V>
where
    F: FnMut(V) -> (V, V),
{
    for _ in 0..P::POLICY.max_iterations {
        let (y, y_prime) = f(x);
        let delta = y / y_prime;

        let mut stop = delta.abs().cmp_le(tolerance);

        if P::POLICY.check_overflow {
            stop |= y_prime.abs().cmp_le(tolerance);
        }

        if stop.all() {
            return Ok(x);
        }

        x = stop.select(x, x - delta);

        if let Some((min, max)) = bounds {
            x = x.clamp(min, max);
        }
    }

    Err(x)
}

#[inline(always)]
fn sum_f<V: CoreMath, P: Policy, F>(start: i64, end: i64, mut f: F) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let mut sum = V::ZERO;
    let mut c = V::ZERO; // Kahan summation compensation
    let mut n = start;

    let tolerance = V::tolerance_p::<P>();

    let mut converged = false;

    for _ in 0..P::POLICY.max_iterations {
        if n >= end {
            break;
        }

        let mut delta = f(n);
        let abs_delta = delta.abs();

        if abs_delta.cmp_le(tolerance).all() {
            converged = true;
            break;
        }

        let t = sum + delta;

        if P::POLICY.precision.ge(PrecisionPolicy::Best) {
            // if |sum| >= |input[i]| then
            //     c += (sum - t) + input[i] // If sum is bigger, low-order digits of input[i] are lost.
            // else
            //     c += (input[i] - t) + sum // Else low-order digits of sum are lost.
            // endif
            sum.abs().cmp_lt(abs_delta).swap(&mut sum, &mut delta);

            c += (sum - t) + delta;
        }

        sum = t;
        n += 1;
    }

    if P::POLICY.precision.ge(PrecisionPolicy::Best) {
        sum += c; // apply any remaining compensation
    }

    match converged {
        true => Ok(sum),
        false => Err(sum),
    }
}

#[inline(always)]
fn prod_f<V: CoreMath, P: Policy, F>(start: i64, end: i64, mut f: F) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let mut prod = V::ONE;
    let mut n = start;

    let tolerance = V::tolerance_p::<P>();

    for _ in 0..P::POLICY.max_iterations {
        if n >= end {
            break;
        }

        let new_prod = prod * f(n);

        let delta = new_prod - prod;

        if delta.abs().cmp_le(tolerance).all() {
            return Ok(prod);
        }

        prod = new_prod;
        n += 1;
    }

    Err(prod)
}
