use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{
            Policy, PrecisionPolicy,
            policies::{ExtraPrecision, LessPrecision},
        },
        specialized::FlushDenormals,
    },
    register::{Element, FloatElement},
    vector::{FloatVectorWithBits, NumericVector, PartialOrdVector},
};

#[inline(always)]
pub fn expint_generic<P: Policy, E, V, const N: usize>(x: V) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + crate::specialized::SpecializedSpecialMath<E>,
{
    let exp_neg_x = (-x).exp_p::<P>();
    let x_ex = exp_neg_x / x;

    if const { N == 0 } {
        let mut result = x_ex;

        if const { P::POLICY.check_overflow } {
            result = x.is_zero().select(V::INFINITY, result);
            result = x.cmp_lt(V::ZERO).select(V::NAN, result);
            result = x.is_nan().select(V::NAN, result);
        }

        return result;
    }

    let is_large = x.cmp_gt(V::ONE);

    let inv_x = x.reciprocal_p::<P>();

    // Coefficients from Boost.Math expint_1_rational<double> (John Maddock, BSL-1.0)
    let mut e_n = x.poly_p::<P, _>(&[
        E::from_f64(0.0865197248079397976498),
        E::from_f64(0.0320913665303559189999),
        E::from_f64(-0.245088216639761496153),
        E::from_f64(-0.0368031736257943745142),
        E::from_f64(-0.00399167106081113256961),
        E::from_f64(-0.000111507792921197858394),
    ]) / x.poly_p::<P, _>(&[
        E::from_f64(1.0),
        E::from_f64(0.37091387659397013215),
        E::from_f64(0.056770677104207528384),
        E::from_f64(0.00427347600017103698101),
        E::from_f64(0.000131049900798434683324),
        E::from_f64(-0.528611029520217142048e-6),
    ]);

    // Coefficients from Boost.Math expint_1_rational<double> (John Maddock, BSL-1.0)
    let large_e1 = inv_x.poly_p::<P, _>(&[
        E::from_f64(-0.121013190657725568138e-18),
        E::from_f64(-0.999999999999998811143),
        E::from_f64(-43.3058660811817946037),
        E::from_f64(-724.581482791462469795),
        E::from_f64(-6046.8250112711035463),
        E::from_f64(-27182.6254466733970467),
        E::from_f64(-66598.2652345418633509),
        E::from_f64(-86273.1567711649528784),
        E::from_f64(-54844.4587226402067411),
        E::from_f64(-14751.4895786128450662),
        E::from_f64(-1185.45720315201027667),
    ]) / inv_x.poly_p::<P, _>(&[
        E::from_f64(1.0),
        E::from_f64(45.3058660811801465927),
        E::from_f64(809.193214954550328455),
        E::from_f64(7417.37624454689546708),
        E::from_f64(38129.5594484818471461),
        E::from_f64(113057.05869159631492),
        E::from_f64(192104.047790227984431),
        E::from_f64(180329.498380501819718),
        E::from_f64(86722.3403467334749201),
        E::from_f64(18455.4124737722049515),
        E::from_f64(1229.20784182403048905),
        E::from_f64(-0.776491285282330997549),
    ]);

    // Equation and constant from Boost.Math expint_1_rational<double> (John Maddock, BSL-1.0)
    e_n += x - x.ln_p::<P>() - V::splat(E::from_f64(0.66373538970947265625));

    e_n = is_large.select((V::ONE + large_e1) * x_ex, e_n);

    // --- Recurrence E_1 -> E_N for x < 2.5·N ---
    // E_{n+1}(x) = (e^{-x} - x·E_n(x)) / n

    for n in 1..N as i64 {
        e_n = x.nmul_adde(e_n, exp_neg_x);

        if n > 1 {
            if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || N < 4 } {
                // For best precision, do the division at each step to avoid error accumulation.
                e_n /= V::splat(FloatElement::from_i64(n));
            } else {
                // For lower precision, multiply by reciprocal since this will be unrolled.
                e_n *= V::splat(FloatElement::from_ratio(1, n));
            }
        }
    }

    // Computes the largest x for which the forward recurrence E_1 -> E_N is reliable.
    //
    // The recurrence E_{n+1}(x) = (e^{-x} - x·E_n(x)) / n has a homogeneous growing solution:
    // any error δ in E_1 is amplified after N steps to δ · x^(N-1) / (N-1)!
    //
    // The mantissa budget is 2^mantissa_bits, so precision is lost once:
    //   x^(N-1) / (N-1)! > 2^mantissa_bits
    //
    // Solving for x gives the recurrence limit: x_rec = ((N-1)! · 2^mantissa_bits)^(1/(N-1))
    //
    // However, x_rec converges toward (N-1)/e ≈ 0.368·N as N → ∞ and eventually falls
    // below N. The asymptotic series only starts shrinking when x > N (since the first-term
    // ratio N/x < 1 requires x > N), so below N it diverges immediately and gives wrong
    // results. The threshold is therefore clamped to at least N to ensure the asymptotic
    // path is always valid when taken. For f32 this matters around N ≥ 20; for f64, N ≥ 37.
    const fn recurrence_threshold(n: usize, mantissa_bits: u32) -> f64 {
        if n <= 1 {
            return f64::MAX;
        }

        // Binary exponentiation, used for bisection below.
        const fn const_powi_f64(mut x: f64, mut n: u32) -> f64 {
            let mut r = 1.0f64;
            while n > 0 {
                if n & 1 == 1 {
                    r *= x;
                }
                n >>= 1;
                if n > 0 {
                    x *= x;
                }
            }
            r
        }

        let k = (n - 1) as u32;

        // target = (n-1)! · 2^mantissa_bits
        let target: f64 = {
            let mut f = (1u64 << mantissa_bits) as f64;
            let mut i = 2usize;
            while i < n {
                f *= i as f64;
                i += 1;
            }
            f
        };

        // Bisect for the k-th root of target. The maximum threshold for k >= 2 occurs at k=2:
        // sqrt(2 * 2^mantissa_bits) = 2^((mantissa_bits+1)/2). Using mantissa_bits/2 + 2 as
        // the shift gives a safe ceiling (e.g. f32: 2^13=8192, f64: 2^28=268M) while keeping
        // intermediate powers well within f64 range for all practical N.
        let mut lo = 0.0f64;
        let mut hi = if k == 1 {
            target
        } else {
            (1u64 << (mantissa_bits / 2 + 2)) as f64
        };

        let mut i = 0;
        while i < 64 {
            let mid = (lo + hi) * 0.5;
            if const_powi_f64(mid, k) < target {
                lo = mid;
            } else {
                hi = mid;
            }
            i += 1;
        }

        let result = (lo + hi) * 0.5;

        // Clamp: asymptotic series diverges immediately for x < N, so never switch below N.
        if result < n as f64 { n as f64 } else { result }
    }

    let is_very_large = x.cmp_ge(V::splat(E::from_f64(
        const { recurrence_threshold(N, <E as FloatElementWithBits>::MANTISSA_BITS) },
    )));

    // Asymptotic expansion for large x where forward recurrence loses precision.
    // E_n(x) ~ (e^{-x}/x) * sum_{k=0}^{inf} (-1)^k * n(n+1)...(n+k) / x^k
    // This is a divergent series — terms eventually grow. We sum while terms shrink,
    // freezing each lane once its terms start increasing. The check is amortized
    // every 4 iterations to allow the compiler to unroll the inner loop body.
    if const { N > 1 } && thermite::unlikely(is_very_large.any()) {
        let mut term = V::ONE;
        let mut partial_sum = V::ONE;
        let mut prev_abs = V::INFINITY;

        for k in 0u32..(2 * N as u32 + 8) {
            term *= -V::splat(FloatElement::from_i64(N as i64 + k as i64)) * inv_x;
            let abs_term = term.abs();
            let still_shrinking = abs_term.cmp_le(prev_abs);

            if k % 4 == 3 && still_shrinking.none() {
                break;
            }

            partial_sum = partial_sum.add_c(still_shrinking, term);
            prev_abs = still_shrinking.select(abs_term, prev_abs);
        }

        // E_n is always positive for x > 0; abs() clamps truncation artifacts.
        e_n = is_very_large.select(x_ex * partial_sum.abs(), e_n);
    }

    if const { P::POLICY.check_overflow } {
        // E_1(0) = +inf, E_n(0) = 1/(n-1) for n > 1
        let x_is_zero = x.is_zero();

        if const { N == 1 } {
            e_n = x_is_zero.select(V::INFINITY, e_n);
        } else if const { N > 1 } {
            e_n = x_is_zero.select(V::splat(FloatElement::from_ratio(1, N as i64 - 1)), e_n);
        }

        // Negative x: NaN
        e_n = x.cmp_lt(V::ZERO).select(V::NAN, e_n);

        // NaN in, NaN out
        e_n = x.is_nan().select(V::NAN, e_n);
    }

    e_n
}
