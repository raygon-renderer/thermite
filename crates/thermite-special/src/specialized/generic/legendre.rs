use thermite::{element::FloatElement, prelude::*};

/// Clenshaw summation of a Legendre series, `$\sum_{k=0}^{N-1} c_k P_k(x)$`.
///
/// The Legendre recurrence `(k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}` is
/// `P_{k+1} = a_k x P_k + b_k P_{k-1}` with `a_k = (2k+1)/(k+1)` and `b_k = -k/(k+1)`, so
/// Clenshaw's adjoint recurrence is
///
/// ```text
/// y_k = c_k + a_k x y_{k+1} + b_{k+1} y_{k+2}      k = N-1 down to 1,  y_N = y_{N+1} = 0
/// S   = c_0 + x y_1 + b_1 y_2 = c_0 + x y_1 - y_2 / 2
/// ```
///
/// Both ratios depend only on `k`, which is a compile-time constant at every step of the
/// unrolled loop, so they fold to literals. The per-step critical path is the one FMA
/// that carries `y_{k+1}`, exactly as in the Chebyshev kernel. Unlike that kernel there is
/// no endpoint-cancellation variant here: `P_n(1) = 1` for every `n` makes the same
/// degeneracy exist at `$x = \pm 1$`, but its Reinsch-style rewrite has not been derived
/// or measured, so this is plain Clenshaw at every policy, which is why, unlike its
/// siblings, this kernel takes no policy parameter.
#[inline(always)]
pub fn legendre_series<E, V, const N: usize>(x: V, coeffs: &[E; N]) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    const {
        assert!(N >= 1, "legendre_series: N must be at least 1");
    }

    // S = c_0 P_0 = c_0.
    if const { N == 1 } {
        return V::splat(coeffs[0]);
    }

    let cn1 = V::splat(coeffs[N - 1]);

    // S = c_0 + c_1 x.
    if const { N == 2 } {
        return x.mul_adde(cn1, V::splat(coeffs[0]));
    }

    // Hoist the top two steps, whose y_{k+2} (and y_{k+1}) terms are zero:
    //     k = N-1:  y = c_{N-1}
    //     k = N-2:  y = c_{N-2} + a_{N-2} x c_{N-1}
    let mut y2 = cn1;
    let mut y1 = (x * V::splat(a::<E>(N - 2))).mul_adde(cn1, V::splat(coeffs[N - 2]));

    // k = N-3 down to 1.
    let mut k = N - 2;
    while k > 1 {
        k -= 1;
        // y_k = a_k x y_{k+1} + (c_k + b_{k+1} y_{k+2}). b_{k+1} is negative, so the sign
        // lives in the constant and the addend is a single FMA off the critical path.
        let ax = x * V::splat(a::<E>(k));
        let yk = ax.mul_adde(y1, y2.mul_adde(V::splat(b_next::<E>(k)), V::splat(coeffs[k])));
        y2 = y1;
        y1 = yk;
    }

    // S = x y_1 + (c_0 - y_2 / 2)
    x.mul_adde(
        y1,
        y2.mul_adde(
            V::splat(<E as FloatElement>::ConstRatio::<{ -1 }, 2>::VALUE),
            V::splat(coeffs[0]),
        ),
    )
}

/// `a_k = (2k+1)/(k+1)`.
#[inline(always)]
fn a<E: FloatElement>(k: usize) -> E {
    E::from_ratio((2 * k + 1) as thermite::LargeInt, (k + 1) as thermite::LargeInt)
}

/// `b_{k+1} = -(k+1)/(k+2)`.
#[inline(always)]
fn b_next<E: FloatElement>(k: usize) -> E {
    E::from_ratio(-((k + 1) as thermite::LargeInt), (k + 2) as thermite::LargeInt)
}
