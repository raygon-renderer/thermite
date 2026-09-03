//! The phi-functions of exponential integrators, `phi_N(z) = sum z^n/(n+N)!`, shared
//! by every element type. `ps.rs`/`pd.rs` supply a compile-time series length, while
//! the element-agnostic default iterates to the element's own epsilon.

use thermite::{
    LargeInt,
    element::FloatElement,
    math::{FloatConsts, policy::Policy, specialized::SpecializedTranscendentalMath},
    prelude::*,
};

/// Series terms `phi_N` needs on `|z| < N` to converge to a relative `eps`.
///
/// The n-th term of `sum z^n/(n+N)!` relative to the leading `1/N!` is `z^n N!/(N+n)!`.
/// At `z = N` every factor `N/(N+n)` is below one, so the terms fall monotonically and
/// the first one under `eps` bounds the whole tail (the ratio there is small, so the
/// tail is barely more than that term). Bounded input, so this is a compile-time count.
pub const fn phi_series_terms(n: usize, eps: f64) -> usize {
    let t = n as f64;
    let mut term = 1.0;
    let mut k = 0;
    while term > eps {
        k += 1;
        term *= t / (n + k) as f64;
    }
    k
}

/// How many orders the runtime-order `phi` has its series length precomputed for. See
/// [`phi_terms_table`].
pub const PHI_TABLE_ORDERS: usize = 33;

/// [`phi_series_terms`] for every order below [`PHI_TABLE_ORDERS`], capped at
/// `max_iterations`: the table the f32/f64 runtime-order entries build in a `const` block
/// per policy, so the term count is an index per call rather than a search.
pub const fn phi_terms_table(eps: f64, max_iterations: usize) -> [usize; PHI_TABLE_ORDERS] {
    let mut t = [0; PHI_TABLE_ORDERS];
    let mut n = 0;
    while n < PHI_TABLE_ORDERS {
        let needed = phi_series_terms(n, eps);
        t[n] = if needed < max_iterations {
            needed
        } else {
            max_iterations
        };
        n += 1;
    }
    t
}

/// The runtime-order twin of [`phi_internal_n`]: the same two arms with `N` as a value. The
/// `1/N!` prefactor and the per-term ratios are the same running products, so the two forms
/// agree to the bit for the same `terms`.
#[inline(always)]
pub fn phi_internal<V, E, P, const ADAPTIVE: bool>(z: V, n: u32, terms: usize) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if n == 0 {
        return V::exp::<P>(z);
    }

    if n == 1 {
        let mut r = V::approx_div::<P>(V::exp_m1::<P>(z), z);

        if const { P::POLICY.check_overflow } {
            r = z.is_zero().select(V::ONE, r);
            r = z.cmp_eq(V::INFINITY).select(V::INFINITY, r);
        }

        return r;
    }

    let n = n as usize;

    let mut inv_fact = E::ONE;
    let mut k = 2;
    while k <= n {
        inv_fact = inv_fact * E::from_ratio(1, k as LargeInt);
        k += 1;
    }

    let near = z.abs().cmp_lt(V::splat(E::from_int(n as LargeInt)));

    // Series arm.
    let mut s = V::ZERO;
    if const { P::POLICY.avoid_branching } || thermite::unlikely(near.any()) {
        let tol = <V as FloatConsts>::EPSILON * V::HALF;
        let mut term = V::splat(inv_fact);
        s = term;
        let mut k = 1;
        while k <= terms {
            term *= z * V::splat(E::from_ratio(1, (n + k) as LargeInt));
            s += term;
            if const { ADAPTIVE } && term.abs().cmp_le(tol * s.abs()).all() {
                break;
            }
            k += 1;
        }
    }

    // Recurrence arm.
    let mut p = V::ZERO;
    if const { P::POLICY.avoid_branching } || thermite::unlikely(!near.all()) {
        let inv = V::ONE / z;
        p = V::exp_m1::<P>(z) * inv;
        let mut inv_kfact = E::ONE;
        let mut k = 1;
        while k < n {
            p = (p - V::splat(inv_kfact)) * inv;
            k += 1;
            inv_kfact = inv_kfact * E::from_ratio(1, k as LargeInt);
        }
    }

    let mut r = near.select(s, p);

    if const { P::POLICY.check_overflow } {
        r = z.cmp_eq(V::INFINITY).select(V::INFINITY, r);
    }

    r
}

/// `phi_N(z) = sum_{n>=0} z^n/(n+N)!`, the exponential-integrator functions.
///
/// `N = 0` is `exp` and `N = 1` is `expm1(z)/z`. Beyond that, two arms split at `|z| = N`:
///
/// * Below, the series, summed forward from `1/N!` with each term the previous times
///   `z/(N+k)`. Every coefficient is one small ratio, so this stays exact for any element
///   type. `terms` bounds the loop, and with `ADAPTIVE` it also stops as soon as the term
///   it just added is under half an ulp of the sum, which is how an element whose precision
///   is not known statically (`Compensated`) converges to its own epsilon.
/// * Above, the recurrence `phi_{k+1} = (phi_k - 1/k!)/z` upward from
///   `phi_1 = expm1(z)/z`. Each step subtracts a constant from something that is only
///   just larger than it while `|z|` is small (that is the cancellation the series
///   exists to avoid), but the amplification per step is `phi_k/(phi_k - 1/k!)`, which
///   is bounded once `|z| >= k`. `|z| >= N` covers every step, and measured against
///   mpmath the recurrence stays under 6 ulp for `N <= 8` in both f32 and f64. The same
///   bound is why the series arm stops at `N`: its terms are monotone there, so the
///   alternating negative side does not cancel either.
///
/// Both arms overflow gracefully: `expm1` saturates to `+inf` and each division by `z`
/// leaves it there, and `-inf` gives `-1 * -0` and then a run of `+0`s, the limit. Only
/// `+inf` itself, `inf * (1/inf)`, and the `0/0` of `N = 1` at the origin need patching.
#[inline(always)]
pub fn phi_internal_n<V, E, P, const N: usize, const ADAPTIVE: bool>(z: V, terms: usize) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { N == 0 } {
        return V::exp::<P>(z);
    }

    if const { N == 1 } {
        let mut r = V::approx_div::<P>(V::exp_m1::<P>(z), z);

        if const { P::POLICY.check_overflow } {
            // 0/0 at the origin, where the limit is 1, and inf/inf at +inf. -inf needs
            // nothing: expm1 gives -1, and -1 / -inf = 0 is already the limit. A
            // large finite z overflows expm1 to inf, and inf/z is likewise right.
            r = z.is_zero().select(V::ONE, r);
            r = z.cmp_eq(V::INFINITY).select(V::INFINITY, r);
        }

        return r;
    }

    // 1/N! as a running product of small ratios, exact-ish for any element and never
    // an integer overflow.
    let mut inv_fact = E::ONE;
    let mut k = 2;
    while k <= N {
        inv_fact = inv_fact * E::from_ratio(1, k as LargeInt);
        k += 1;
    }

    let near = z.abs().cmp_lt(V::splat(E::from_int(N as LargeInt)));

    // Series arm.
    let mut s = V::ZERO;
    if const { P::POLICY.avoid_branching } || thermite::unlikely(near.any()) {
        // FloatConsts, not FloatVector: `Compensated`'s FloatVector::EPSILON is the
        // single-width one, and its consts table carries the real 2^-105.
        let tol = <V as FloatConsts>::EPSILON * V::HALF;
        let mut term = V::splat(inv_fact);
        s = term;
        let mut k = 1;
        while k <= terms {
            term *= z * V::splat(E::from_ratio(1, (N + k) as LargeInt));
            s += term;
            if const { ADAPTIVE } && term.abs().cmp_le(tol * s.abs()).all() {
                break;
            }
            k += 1;
        }
    }

    // Recurrence arm.
    let mut p = V::ZERO;
    if const { P::POLICY.avoid_branching } || thermite::unlikely(!near.all()) {
        let inv = V::ONE / z;
        p = V::exp_m1::<P>(z) * inv;
        let mut inv_kfact = E::ONE; // 1/k!
        let mut k = 1;
        while k < N {
            p = (p - V::splat(inv_kfact)) * inv;
            k += 1;
            inv_kfact = inv_kfact * E::from_ratio(1, k as LargeInt);
        }
    }

    let mut r = near.select(s, p);

    if const { P::POLICY.check_overflow } {
        r = z.cmp_eq(V::INFINITY).select(V::INFINITY, r);
    }

    r
}
