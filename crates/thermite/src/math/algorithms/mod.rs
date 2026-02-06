use std::ops::BitAnd as _;

use crate::generic::{GenericMask as _, ops::BitAndNot as _};

use super::*;

/// Newton's method for finding roots of a function.
///
/// Returns `Ok(root)` if convergence was achieved within the maximum number of iterations,
/// otherwise returns `Err(approximation)` with the best approximation found.
///
/// The given function `f` should return a tuple `(f(x), f'(x))`, where `f(x)` is the function value
/// and `f'(x)` is its derivative at point `x`.
///
/// If bounds are provided, this will use a hybrid approach with bisection to ensure
/// the root remains within the specified bounds. If there are points where the derivative is zero,
/// this will help avoid divergence.
#[inline(always)]
pub fn newtons_method<V: FloatVector, P: Policy, F>(
    mut x: V,
    tolerance: V,
    mut bounds: Option<(V, V)>,
    mut f: F,
) -> Result<V, V>
where
    F: FnMut(V) -> (V, V),
{
    for _i in 0..P::POLICY.max_iterations {
        let (y, y_prime) = f(x);

        // If y=0 within tolerance, we're done.
        let stop = y.abs().cmp_le(tolerance);

        if stop.all() {
            // println!("Converged in {} iterations", _i);
            return Ok(x);
        }

        // We compute this speculatively. If y_prime is 0, this yields +/- Inf.
        // SIMD handles Inf correctly (non-trapping), so we don't need to branch.
        let x_newton = x - (y / y_prime);

        let next_x = if let Some((ref mut min, ref mut max)) = bounds {
            // Update Bounds, "Shrink-wrap" logic
            let is_negative = y.is_negative();

            // Optimization: These selects are independent and pipeline well
            *min = x.cmp_gt(*min).bitand(is_negative).select(x, *min);
            *max = x.cmp_lt(*max).bitandnot(is_negative).select(x, *max); // x.cmp_lt(*max) & !is_negative

            let x_bisection = (*min + *max) * V::HALF; // Midpoint Calculation

            // Hybrid Logic, If x_newton is Inf, NaN, or Overshot, "inside_bounds" is False.
            // This implicitly handles the "derivative is zero" case.
            let inside_bounds = x_newton.cmp_gt(*min) & x_newton.cmp_lt(*max);

            // If Newton behaved, keep it. If it exploded, use Bisection.
            inside_bounds.select(x_newton, x_bisection)
        } else {
            // No bounds provided? We must trust Newton, even if it explodes.
            x_newton
        };

        x = stop.select(x, next_x);
    }

    Err(x)
}

/// Computes the sum of a function `f` evaluated over the range `[start, end)`.
///
/// Returns `Ok(sum)` if convergence was achieved within the maximum number of iterations,
/// otherwise returns `Err(partial_sum)` with the best partial sum computed.
///
/// The function `f` is expected to return a value at each provided iteration index.
///
/// If using a precision policy of `Best` or higher, modified Kahan summation is employed to reduce numerical error.
#[inline(always)]
pub fn sum_f<V: FloatVector, P: Policy, F>(tolerance: V, start: i64, end: i64, mut f: F) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let mut sum = V::ZERO;
    let mut c = V::ZERO; // Kahan summation compensation
    let mut n = start;

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

        if P::POLICY.use_compensation {
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

    if P::POLICY.use_compensation {
        sum += c; // apply any remaining compensation
    }

    match converged {
        true => Ok(sum),
        false => Err(sum),
    }
}

/// Computes the sum of a function `f` evaluated over the range `[start, end)`.
///
/// Returns `Ok(sum)` if convergence was achieved within the maximum number of iterations,
/// otherwise returns `Err(partial_sum)` with the best partial sum computed.
///
/// The function `f` is expected to return a value at each provided iteration index.
#[inline(always)]
pub fn prod_f<V: FloatVector, P: Policy, F>(tolerance: V, start: i64, end: i64, mut f: F) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let mut prod = V::ONE;
    let mut n = start;

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

/// Reduces the elements of `values` in place using the binary operation `op` in O(n) steps, but
/// with a dependency depth of O(log n), allowing for better instruction-level parallelism.
///
/// The end result is stored in `values[0]`.
#[inline(always)]
pub fn reduce_in_place<V: Copy, F>(values: &mut [V], mut op: F)
where
    F: FnMut(V, V) -> V,
{
    let mut stride = 1;

    while stride < values.len() {
        let mut i = 0;
        let next_stride = stride << 1; // x2

        while i + stride < values.len() {
            values[i] = op(values[i], values[i + stride]);
            i += next_stride;
        }

        stride = next_stride;
    }
}

/// Reduces the elements of `values` using the binary operation `op` in O(n) steps, but
/// with a dependency depth of O(log n), allowing for better instruction-level parallelism.
///
/// The end result is returned.
#[inline(always)]
pub fn reduce_array<V: Copy, const N: usize, F>(mut values: [V; N], op: F) -> V
where
    F: FnMut(V, V) -> V,
{
    reduce_in_place(&mut values, op);

    values[0]
}
