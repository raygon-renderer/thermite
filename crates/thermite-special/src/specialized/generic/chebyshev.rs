use thermite::{
    element::FloatElement,
    math::policy::{Policy, PrecisionPolicy},
    prelude::*,
};

/// Shared Chebyshev series summation for all element types and all four kinds.
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
#[inline(always)]
pub fn chebyshev_series<P, E, V, const K: usize, const N: usize, const REINSCH: bool>(x: V, coeffs: &[E; N]) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    const {
        assert!(K >= 1 && K <= 4, "chebyshev: K must be 1, 2, 3, or 4");
        assert!(N >= 1, "chebyshev: N must be at least 1");
    }

    // S = Σ c_k P_0 = c_0 when N = 1; skip the whole recurrence.
    if const { N == 1 } {
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

    let cn1 = V::splat(coeffs[N - 1]);
    let cn2 = V::splat(coeffs[N - 2]);

    // S = c_0 + c_1*P_1(x) when N = 2.
    if const { N == 2 } {
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

        // k = N-1 down to 1. The first two steps fold away against the zero seeds.
        let mut k = N - 1;
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
    //     b_{N+1} = b_N = 0
    //     for k = N-1 down to 1:  b_k = 2x*b_{k+1} - b_{k+2} + c_k
    //     S = (c_0 - b_2) + b_1 * P_1(x)
    //
    // This is more numerically stable than the forward sum (especially when the partial sums
    // of Σ c_k P_k are much smaller than max|c_k P_k|) and uses only two running scalars
    // instead of three.
    //
    // Hoist the first two iterations to eliminate the b_2 = 0 subtraction in the loop:
    //     k = N-1:  b_{N-1} = 2x*0 + c_{N-1} - 0          = c_{N-1}
    //     k = N-2:  b_{N-2} = 2x*c_{N-1} + c_{N-2} - 0    = 2x*c_{N-1} + c_{N-2}
    let mut b1 = x2.mul_adde(cn1, cn2); // b_{k+1} = b_{N-2}
    let mut b2 = cn1; // b_{k+2} = b_{N-1}

    // Iterate k = N-3, N-4, ..., 1.
    let mut k = N - 2;
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
