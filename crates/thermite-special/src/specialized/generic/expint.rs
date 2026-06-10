use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{
            Policy, PrecisionPolicy,
            policies::{ExtraPrecision, LessPrecision},
        },
        specialized::FlushDenormals,
    },
    register::{Element, FloatElement},
    vector::{FloatVectorWithBits, NumericVector, PartialOrdVector, SplatConst},
};

// Computes the largest x for which the forward recurrence E_1 -> E_N is reliable.
//
// The recurrence E_{n+1}(x) = (e^{-x} - x*E_n(x)) / n has a homogeneous growing solution:
// any error δ in E_1 is amplified after N steps to δ * x^(N-1) / (N-1)!
//
// The mantissa budget is 2^mantissa_bits, so precision is lost once:
//   x^(N-1) / (N-1)! > 2^mantissa_bits
//
// Solving for x gives the recurrence limit: x_rec = ((N-1)! * 2^mantissa_bits)^(1/(N-1))
//
// However, x_rec converges toward (N-1)/e ≈ 0.368*N as N -> ∞ and eventually falls
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

    // target = (n-1)! * 2^mantissa_bits
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

// NOTE: When const-generics are more mature, we can have these polynomials by dynamic in size based on the
// type of `Self`. For now, it's only the double-precision (f64) polynomials.
pub trait ExpIntConsts<const N: usize>: FloatConsts + Sized {
    // polynomial coefficients
    const SMALL_N: [Self; 6];
    const SMALL_D: [Self; 6];
    const LARGE_N: [Self; 11];
    const LARGE_D: [Self; 12];
    const ASYMPTOTIC_CONST: Self;
    const RECURRENCE_THRESHOLD: Self;
    const ONE_OVER_N_MINUS_1: Self;
    const FACTORS: [Self; N]; // n+2
    const RECIPROCALS: [Self; N]; // reciprocal of factors
}

macro_rules! impl_expint_consts {
    (
        SMALL_N [ $($sn_value:literal),* $(,)? ],
        SMALL_D [ $($sd_value:literal),* $(,)? ],
        LARGE_N [ $($ln_value:literal),* $(,)? ],
        LARGE_D [ $($ld_value:literal),* $(,)? ]
    ) => {
        impl<const N: usize> ExpIntConsts<N> for f32 {
            const SMALL_N: [Self; 6] = [$($sn_value),*];
            const SMALL_D: [Self; 6] = [$($sd_value),*];
            const LARGE_N: [Self; 11] = [$($ln_value),*];
            const LARGE_D: [Self; 12] = [$($ld_value),*];
            const ASYMPTOTIC_CONST: Self = 0.66373538970947265625;
            const RECURRENCE_THRESHOLD: Self = const { recurrence_threshold(N, Self::MANTISSA_BITS) as f32 };
            const ONE_OVER_N_MINUS_1: Self = if N > 1 { 1.0 / (N as Self - 1.0) } else { Self::INFINITY };
            const FACTORS: [Self; N] = {
                let mut facts = [0.0; N]; let mut i = 0;
                while i < N { facts[i] = (2 + i) as Self; i += 1; }
                facts
            };
            const RECIPROCALS: [Self; N] = {
                let mut r = Self::FACTORS; let mut i = 0;
                while i < N { r[i] = 1.0 / r[i]; i += 1; }
                r
            };
        }

        impl<const N: usize> ExpIntConsts<N> for f64 {
            const SMALL_N: [Self; 6] = [$($sn_value),*];
            const SMALL_D: [Self; 6] = [$($sd_value),*];
            const LARGE_N: [Self; 11] = [$($ln_value),*];
            const LARGE_D: [Self; 12] = [$($ld_value),*];
            const ASYMPTOTIC_CONST: Self = 0.66373538970947265625;
            const RECURRENCE_THRESHOLD: Self = const { recurrence_threshold(N, Self::MANTISSA_BITS) };
            const ONE_OVER_N_MINUS_1: Self = if N > 1 { 1.0 / (N as Self - 1.0) } else { Self::INFINITY };
            const FACTORS: [Self; N] = {
                let mut facts = [0.0; N]; let mut i = 0;
                while i < N { facts[i] = (2 + i) as Self; i += 1; }
                facts
            };
            const RECIPROCALS: [Self; N] = {
                let mut r = Self::FACTORS; let mut i = 0;
                while i < N { r[i] = 1.0 / r[i]; i += 1; }
                r
            };
        }
    };
}

impl_expint_consts! {
    SMALL_N [
        -0.000111507792921197858394,
        -0.00399167106081113256961,
        -0.0368031736257943745142,
        -0.245088216639761496153,
        0.0320913665303559189999,
        0.0865197248079397976498,
    ],
    SMALL_D [
        0.528611029520217142048e-6,
        0.000131049900798434683324,
        0.00427347600017103698101,
        0.056770677104207528384,
        0.37091387659397013215,
        1.0,
    ],
    LARGE_N [
        -1185.45720315201027667,
        -14751.4895786128450662,
        -54844.4587226402067411,
        -86273.1567711649528784,
        -66598.2652345418633509,
        -27182.6254466733970467,
        -6046.8250112711035463,
        -724.581482791462469795,
        -43.3058660811817946037,
        -0.999999999999998811143,
        -0.121013190657725568138e-18,
    ],
    LARGE_D [
        -0.776491285282330997549,
        1229.20784182403048905,
        18455.4124737722049515,
        86722.3403467334749201,
        180329.498380501819718,
        192104.047790227984431,
        113057.05869159631492,
        38129.5594484818471461,
        7417.37624454689546708,
        809.193214954550328455,
        45.3058660811801465927,
        1.0,
    ]
}

#[inline(always)]
pub fn expint_double<P: Policy, E, V, const N: usize>(x: V) -> V
where
    E: FloatElementWithBits + ExpIntConsts<N>,
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
    let mut e_n = x
        .poly_rev_p::<P, _>(&E::SMALL_N)
        .approx_div_p::<P>(x.poly_rev_p::<P, _>(&E::SMALL_D));

    // Coefficients from Boost.Math expint_1_rational<double> (John Maddock, BSL-1.0)
    let large_e1 = inv_x
        .poly_rev_p::<P, _>(&E::LARGE_N)
        .approx_div_p::<P>(inv_x.poly_rev_p::<P, _>(&E::LARGE_D));

    // Equation and constant from Boost.Math expint_1_rational<double> (John Maddock, BSL-1.0)
    e_n += x - x.ln_p::<P>() - V::splat(E::ASYMPTOTIC_CONST);

    e_n = is_large.select((V::ONE + large_e1) * x_ex, e_n);

    // --- Recurrence E_1 -> E_N for x < 2.5*N ---
    // E_{n+1}(x) = (e^{-x} - x*E_n(x)) / n
    //
    // The n=1 step has no division (divides by 1), so it is peeled out to avoid a
    // runtime `if n > 1` check inside the loop.
    if const { N > 1 } {
        e_n = x.nmul_adde(e_n, exp_neg_x);

        if const { N > 2 } {
            if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
                let mut n = 0;
                while n < (N - 2) {
                    e_n = x.nmul_adde(e_n, exp_neg_x) / V::splat(E::FACTORS[n]);
                    n += 1;
                }
            } else {
                use crunchy::unroll;

                macro_rules! unroll_recurrence {
                    ($($len:tt),*) => {
                        $( if const { N == ($len + 2) } {
                            unroll! { for n in 0..$len {
                                e_n = x.nmul_adde(e_n, exp_neg_x)
                                    .scale(const { if n < N { E::RECIPROCALS[n] } else { E::ONE } });
                            }}
                        } else )* {
                            let mut n = 0;
                            while n < const { if N > 2 { N - 2 } else { 0 } } {
                                e_n = x.nmul_adde(e_n, exp_neg_x).scale(E::RECIPROCALS[n]);
                                n += 1;
                            }
                        }
                    };
                }

                unroll_recurrence!(1, 2, 3, 4, 5, 6); // up to N=8
            }
        }
    }

    let is_very_large = x.cmp_ge(V::splat(E::RECURRENCE_THRESHOLD));

    // Asymptotic expansion for large x where forward recurrence loses precision.
    // E_n(x) ~ (e^{-x}/x) * sum_{k=0}^{inf} (-1)^k * n(n+1)...(n+k) / x^k
    // This is a divergent series - terms eventually grow. We sum while terms shrink,
    // freezing each lane once its terms start increasing. The check is amortized
    // every 4 iterations to allow the compiler to unroll the inner loop body.
    if const { N > 1 } && thermite::unlikely(is_very_large.any()) {
        let mut term = V::ONE;
        let mut partial_sum = V::ONE;
        let mut prev_abs = V::INFINITY;

        let mut k = 0u32;
        while k < 2 * N as u32 + 8 {
            term *= inv_x.scale(FloatElement::from_int(
                const { -(N as thermite::LargeInt) } - k as thermite::LargeInt,
            ));

            let abs_term = term.abs();
            let still_shrinking = abs_term.cmp_le(prev_abs);

            // Check convergence every 4 iterations (amortized to allow loop unrolling).
            // Uses the loop condition instead of break to keep SPIR-V CFG well-structured.
            if k % 4 == 3 && still_shrinking.none() {
                break;
            }

            partial_sum = partial_sum.add_c(still_shrinking, term);
            prev_abs = still_shrinking.select(abs_term, prev_abs);
            k += 1;
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
            e_n = x_is_zero.select(V::splat(E::ONE_OVER_N_MINUS_1), e_n);
        }

        // Negative x: NaN
        e_n = x.cmp_lt(V::ZERO).select(V::NAN, e_n);

        // NaN in, NaN out
        e_n = x.is_nan().select(V::NAN, e_n);
    }

    e_n
}
