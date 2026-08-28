use thermite::{
    element::FloatElement,
    math::policy::{Policy, PrecisionPolicy},
    prelude::*,
};

/// Shared Chebyshev series summation for all element types, all four kinds, and both the
/// compile-time and runtime coefficient counts.
///
/// Evaluates `$\sum_{k=0}^{N-1} c_k P_k(x)$` where `P_k` is `T_k`, `U_k`, `V_k`, or `W_k`
/// for `K` of 1, 2, 3, or 4. All four share the recurrence
/// `$P_{k+1}(x) = 2x P_k(x) - P_{k-1}(x)$` with `P_0 = 1`, differing only in `P_1`, so
/// the `b_k` loop below is common to every kind.
///
/// `REINSCH` says whether the *arithmetic* admits the endpoint form (see below): it needs
/// a real `copysign` and a meaningful nearest endpoint, so real vectors pass `true` and
/// `Complex` and the composites pass `false`. It is a capability, not a request: the
/// form is taken only when the policy also asks for `Best` precision or better.
///
/// # The `N` parameter
///
/// `N` is the coefficient count when the caller knows it and **`0` when it does not**, the
/// same sentinel `fast_polynomial::poly_f_internal` uses. At a nonzero `N` the length is
/// handed to LLVM as an `assert_unchecked`, so the `n == 1`/`n == 2` shortcuts fold away
/// and the loop unrolls exactly as it did when the bound was the const generic itself. At
/// `N = 0` every one of those becomes an ordinary runtime branch.
///
/// This replaces a hand-ported `chebyshev_series_slice` that duplicated the whole
/// recurrence, Reinsch arm included, under a doc comment reading "both forms must be edited
/// together". A series is still not a reduction, since it carries `k`-dependent state and
/// cannot be folded over chunks the way a norm can, so sharing the *body* is the only way
/// to share anything here, and it is what removes the drift.
///
/// Chebyshev is the merge's safe case on purpose: the only per-step quantity is
/// `coeffs[k]`, so nothing here depends on `k` becoming a literal. The Legendre, Hermite
/// and Laguerre series do (a division or a square root per step folds away only if the
/// loop unrolls), which is why they have not been merged.
///
/// # Safety
///
/// `N != 0` promises `coeffs.len() == N`. The const-length entry point is the only caller
/// that passes a nonzero `N`, and it takes a `&[E; N]`, so the promise is the array's.
///
/// The empty series is `0`. `N = 0` is therefore both "unknown length" and "empty", which
/// agree: an empty slice returns `V::ZERO` down the runtime path. The rejection of an empty
/// *const* count lives on the `chebyshev_n` entry point, where it is still a compile error.
#[inline(always)]
pub fn chebyshev_series<P, E, V, const K: usize, const N: usize, const REINSCH: bool>(x: V, coeffs: &[E]) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    const {
        assert!(K >= 1 && K <= 4, "chebyshev: K must be 1, 2, 3, or 4");
    }

    let n = coeffs.len();

    // SAFETY: IFF N != 0, `n` is guaranteed to be == N by this function's contract, so this
    // is an optimization hint rather than a check. It is what keeps the const-length caller
    // generating the code it did when `N` was the loop bound directly.
    if const { N != 0 } {
        unsafe { core::hint::assert_unchecked(n == N) };
    }

    if n == 0 {
        return V::ZERO;
    }

    // S = Σ c_k P_0 = c_0 when n = 1; skip the whole recurrence.
    if n == 1 {
        return V::splat(coeffs[0]);
    }

    let x2 = x + x;

    // P_1: T_1 = x, U_1 = 2x, V_1 = 2x - 1, W_1 = 2x + 1.
    let p1 = if const { K == 1 } {
        x
    } else if const { K == 2 } {
        x2
    } else if const { K == 3 } {
        x2 - V::ONE
    } else if const { K == 4 } {
        x2 + V::ONE
    } else {
        unsafe { core::hint::unreachable_unchecked() }
    };

    let cn1 = V::splat(coeffs[n - 1]);
    let cn2 = V::splat(coeffs[n - 2]);

    // S = c_0 + c_1*P_1(x) when n = 2.
    if n == 2 {
        return p1.mul_adde(cn1, cn2);
    }

    // Reinsch's modification. The plain recurrence below forms `2x*b - b` with consecutive
    // b_k of nearly equal magnitude as x -> +-1, and cancels. Measured against a 60-digit
    // oracle that costs up to 37 ulp on a sum whose own condition number is ~1. Recurring
    // instead on the differences (near +1) or the sums (near -1) forms the small quantity
    // directly:
    //
    //     d_k = b_k - b_{k+1} = 2(x-1)*b_{k+1} + d_{k+1} + c_k,  b_k = b_{k+1} + d_k
    //     d_k = b_k + b_{k+1} = 2(x+1)*b_{k+1} - d_{k+1} + c_k,  b_k = d_k - b_{k+1}
    //
    // The two differ only in the sign of d_{k+1} and of b_{k+1}, so s = copysign(1, x) folds
    // them into one branchless recurrence, which matters because the endpoint is a per-lane
    // property and a scalar branch is not available. x - s is exact by Sterbenz for
    // |x| >= 1/2, so the cancellation happens once, exactly, instead of once per step.
    //
    // Costs roughly 2x on the dependency chain (two FMAs deep per step instead of one), hence
    // the policy gate. Always using the +1 form to dodge the copysign was measured and is
    // WORSE than plain Clenshaw at x -> -1 (q 85 vs 53). Do not "simplify" it away.
    if const { REINSCH && P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let s = V::ONE.copysign(x);
        let step = (x - s) + (x - s);

        let mut b_1 = V::ZERO; // b_{k+1}
        let mut b_2 = V::ZERO; // b_{k+2}
        let mut d_1 = V::ZERO; // d_{k+1}

        // k = n-1 down to 1. The first two steps fold away against the zero seeds.
        let mut k = n - 1;
        while k >= 1 {
            let d = step.mul_adde(b_1, s.mul_adde(d_1, V::splat(coeffs[k])));
            let b = s.mul_adde(b_1, d);
            b_2 = b_1;
            b_1 = b;
            d_1 = d;
            k -= 1;
        }

        return b_1.mul_adde(p1, V::splat(coeffs[0]) - b_2);
    }

    // Clenshaw's backward recurrence:
    //
    //     b_{n+1} = b_n = 0
    //     for k = n-1 down to 1:  b_k = 2x*b_{k+1} - b_{k+2} + c_k
    //     S = (c_0 - b_2) + b_1 * P_1(x)
    //
    // This is more numerically stable than the forward sum (especially when the partial sums
    // of Σ c_k P_k are much smaller than max|c_k P_k|) and uses only two running scalars
    // instead of three.
    //
    // Hoist the first two iterations to eliminate the b_2 = 0 subtraction in the loop:
    //     k = n-1:  b_{n-1} = 2x*0 + c_{n-1} - 0          = c_{n-1}
    //     k = n-2:  b_{n-2} = 2x*c_{n-1} + c_{n-2} - 0    = 2x*c_{n-1} + c_{n-2}
    let mut b1 = x2.mul_adde(cn1, cn2); // b_{k+1} = b_{n-2}
    let mut b2 = cn1; // b_{k+2} = b_{n-1}

    // Iterate k = n-3, n-4, ..., 1.
    let mut k = n - 2;
    while k > 1 {
        k -= 1;
        // b_k = (2x*b_{k+1} + c_k) - b_{k+2}
        let bk = x2.mul_adde(b1, V::splat(coeffs[k]) - b2);
        b2 = b1;
        b1 = bk;
    }

    // S = b_1 * P_1(x) + (c_0 - b_2)
    b1.mul_adde(p1, V::splat(coeffs[0]) - b2)
}
