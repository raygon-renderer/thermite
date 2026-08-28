use crate::mask::GenericMask as _;

use super::*;

/// Newton's method for finding roots of a function.
///
/// Returns `(root, converged)`, where `converged` is a per-lane mask: a lane is `true` when it
/// reached `tolerance` -- in function space (`|f(x)| <= tolerance`) or, when bounds are given,
/// once the bracket shrank to within `tolerance` or collapsed to floating-point resolution --
/// within `P::POLICY.max_iterations`. Use `converged.all()` to test for full convergence.
///
/// `root` always holds the best per-lane estimate; lanes that did not converge contain the latest
/// iterate. Each lane is tracked independently, so partial convergence is reported precisely.
///
/// The given function `f` should return a tuple `(f(x), f'(x))`, where `f(x)` is the function value
/// and `f'(x)` is its derivative at point `x`.
///
/// If bounds are provided, this will use a hybrid approach with bisection to ensure
/// the root remains within the specified bounds. If there are points where the derivative is zero,
/// this will help avoid divergence.
///
/// The bounds must form a bracket: `f(min)` and `f(max)` must have opposite signs. No assumption
/// is made about whether the function is increasing or decreasing. When `P::POLICY.check_overflow`
/// is set, a non-finite `f(x)` will not move the bracket, preventing a NaN/Inf from discarding the
/// root; otherwise `f` is assumed finite on the bracket.
#[inline(always)]
pub fn newtons_method<V: FloatVector, P: Policy, F>(
    mut x: V,
    tolerance: V,
    mut bounds: Option<(V, V)>,
    mut f: F,
) -> (V, V::Mask)
where
    F: FnMut(V) -> (V, V),
{
    // Determine which side of the bracket has f < 0. This is evaluated once and used
    // throughout to shrink-wrap the bounds without assuming function monotonicity.
    let min_is_negative = match bounds.as_ref() {
        None => V::Mask::FALSY,
        Some((min, max)) => {
            // Clamp initial guess into the bracket to prevent breaking the invariant.
            x = x.clamp(*min, *max);

            let min_neg = f(*min).0.is_negative();

            // Precondition: f(min) and f(max) must have opposite signs so the bracket
            // actually contains a root. We otherwise never evaluate f(max) (its sign is
            // inferred from this invariant), so this extra call is debug-only.
            debug_assert!(
                (min_neg ^ f(*max).0.is_negative()).all(),
                "newtons_method: bounds do not bracket a root (f(min) and f(max) must have opposite signs)"
            );

            min_neg
        }
    };

    let mut converged = V::Mask::FALSY;

    for _ in 0..P::POLICY.max_iterations {
        V::_loop_hint();

        let (y, y_prime) = f(x);

        // Lanes within tolerance in function space (|f(x)| <= tolerance) have converged.
        let mut stop = y.abs().cmp_le(tolerance);

        let next_x = if let Some((ref mut min, ref mut max)) = bounds {
            // Shrink-wrap: replace the bound whose sign matches f(x), preserving the bracket.
            // `x` is always inside [min, max] (clamped guess; every later step is bisection or an
            // in-bracket Newton step), so the update can never widen the bracket -- no guard needed.
            // XOR: true where y has the same sign as f(max), so x replaces max; else x replaces min.
            let same_sign_as_max = y.is_negative() ^ min_is_negative;

            if const { P::POLICY.check_overflow } {
                // A non-finite f(x) has no meaningful sign; leave the bracket untouched on those
                // lanes so a transient NaN/Inf cannot misclassify and discard the root.
                let live = y.is_finite();
                *min = ((!same_sign_as_max) & live).select(x, *min);
                *max = (same_sign_as_max & live).select(x, *max);
            } else {
                *min = same_sign_as_max.select(*min, x);
                *max = same_sign_as_max.select(x, *max);
            }

            let width = *max - *min;

            // Midpoint: min + (max - min) * 0.5 avoids overflow and cancellation.
            let x_bisection = width.mul_adde(V::HALF, *min);

            // Converged where the bracket is within tolerance, or where it has collapsed to FP
            // resolution -- the midpoint no longer lands strictly inside, so it cannot shrink
            // further. The latter guarantees termination even when tolerance is below the ULP.
            stop |= width.cmp_le(tolerance);
            stop |= x_bisection.cmp_le(*min) | x_bisection.cmp_ge(*max);

            if stop.all() {
                return (x, stop);
            }

            // Newton step, computed only once at least one lane is still live -- the division is
            // the costliest op in the loop. If y_prime is 0 this yields +/-Inf; SIMD handles it
            // non-trapping and `inside_bounds` rejects it, falling back to bisection. This also
            // implicitly handles overshoot and the zero-derivative case.
            let x_newton = x - (y / y_prime);
            let inside_bounds = x_newton.cmp_gt(*min) & x_newton.cmp_lt(*max);
            inside_bounds.select(x_newton, x_bisection)
        } else {
            if stop.all() {
                return (x, stop);
            }

            // No bounds provided? We must trust Newton, even if it explodes.
            x - (y / y_prime)
        };

        converged = stop;
        x = stop.select(x, next_x);
    }

    (x, converged)
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

    let mut _iter = 0usize;
    while _iter < P::POLICY.max_iterations {
        V::_loop_hint();

        _iter += 1;
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

        if const { P::POLICY.use_compensation } {
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

    if const { P::POLICY.use_compensation } {
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

    let mut _iter = 0usize;
    while _iter < P::POLICY.max_iterations {
        V::_loop_hint();

        _iter += 1;
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

/// Accelerates a linearly converging series using Aitken's Δ^2 process.
///
/// Given a term-generating function `f(n)` that produces the n-th term of a series,
/// this computes partial sums and applies Aitken's delta-squared extrapolation to
/// accelerate convergence. For a series converging at geometric rate r, the accelerated
/// sequence converges at rate r^2.
///
/// Returns `Ok(sum)` when the extrapolated estimate converges within `tolerance`,
/// or `Err(best)` with the best estimate if the iteration limit is reached.
///
/// The extrapolation formula is:
/// ```text
///     S'_n = S_n - (S_{n+1} - S_n)^2 / (S_{n+2} - 2*S_{n+1} + S_n)
/// ```
///
/// When the denominator (second forward difference) is near zero, the raw partial sum
/// is used instead, as this indicates the sequence has already converged or is not
/// amenable to acceleration.
///
/// If the policy enables compensation (`use_compensation`), Kahan summation is used
/// for the underlying partial sum accumulation.
///
/// # Examples
///
/// Accelerating the slowly-converging Leibniz series
/// `$\frac{\pi}{4} = \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$`,
/// which needs on the order of `1/tolerance` terms when summed naively:
///
/// ```
/// use thermite::prelude::*;
/// use thermite::math::algorithms::aitken_sum;
/// use thermite::math::policy::policies::Precision;
///
/// type V = Vector<f64>;
///
/// let leibniz = |n: i64| {
///     let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
///     V::splat(sign / (2 * n + 1) as f64)
/// };
///
/// let sum = match aitken_sum::<V, Precision, _>(V::splat(1e-12), 0, 100_000, leibniz) {
///     Ok(v) | Err(v) => v.extract::<0>(),
/// };
/// assert!((sum - core::f64::consts::FRAC_PI_4).abs() < 1e-10);
/// ```
#[inline(always)]
pub fn aitken_sum<V: FloatVector, P: Policy, F>(tolerance: V, start: i64, end: i64, mut f: F) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let mut sum = V::ZERO;
    let mut c = V::ZERO; // Kahan compensation

    // Sliding window of three consecutive partial sums for Δ^2 extrapolation
    let mut s0 = V::ZERO;
    let mut s1 = V::ZERO;
    let mut s2;

    let mut n = start;
    let mut best = V::ZERO;
    let mut phase = 0u32; // counts how many partial sums we've accumulated in the current window

    let mut _iter = 0usize;
    while _iter < P::POLICY.max_iterations {
        V::_loop_hint();

        _iter += 1;
        if n >= end {
            break;
        }

        // Accumulate next term
        let mut term = f(n);
        let t = sum + term;

        if const { P::POLICY.use_compensation } {
            let abs_term = term.abs();
            sum.abs().cmp_lt(abs_term).swap(&mut sum, &mut term);
            c += (sum - t) + term;
        }

        sum = t;
        n += 1;

        let res = if const { P::POLICY.use_compensation } {
            sum + c
        } else {
            sum
        };

        // Fill the sliding window
        match phase {
            0 => {
                s0 = res;
                phase = 1;
                continue;
            }
            1 => {
                s1 = res;
                phase = 2;
                continue;
            }
            _ => {
                s2 = res;
            }
        }

        // Aitken's Δ^2 extrapolation
        let d1 = s1 - s0; // ΔS_n
        let d2 = s2 - s1; // ΔS_{n+1}
        let denom = d2 - d1; // Δ^2S_n = second forward difference

        // Where |denom| is too small, the sequence has effectively converged
        // or the extrapolation is numerically unstable -- fall back to raw sum.
        let denom_ok = denom.abs().cmp_gt(tolerance);
        let a0 = (d2 * d2) / denom;
        let accelerated = s2 - a0;

        best = denom_ok.select(accelerated, s2);

        // Check convergence: |accelerated - s1_accelerated_prev| <= tolerance
        // We use the simpler check: |d2| <= tolerance (the raw sequence has converged)
        // OR the extrapolated value is stable (|s2 - accelerated| <= tolerance when denom is healthy)
        if a0.zz(denom_ok).abs().cmp_le(tolerance).all() {
            return Ok(best);
        }

        // Slide the window
        s0 = s1;
        s1 = s2;
    }

    // If we never filled the window, just return the raw sum
    if phase < 2 {
        best = if const { P::POLICY.use_compensation } {
            sum + c
        } else {
            sum
        };
    }

    Err(best)
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
