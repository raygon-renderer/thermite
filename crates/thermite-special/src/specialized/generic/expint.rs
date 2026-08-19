use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
    },
    vector::FloatVectorWithBits,
};

/// How much error amplification the forward recurrence is allowed before the continued
/// fraction takes over, in ulps of the seed.
///
/// The resulting worst-case relative error on the recurrence path is `AMP_CAP * eps`, i.e.
/// about 1.4e-14 in binary64 and 7.6e-6 in binary32 - proportional in either format, which is
/// why this is a pure count and not a function of the mantissa width.
const AMP_CAP: f64 = 64.0;

// Computes the largest x for which the forward recurrence E_1 -> E_N holds full precision.
//
// The recurrence E_{n+1}(x) = (e^{-x} - x*E_n(x)) / n has a homogeneous growing solution:
// any error δ in E_1 is amplified after N-1 steps to δ * x^(N-1) / (N-1)!
//
// Requiring that amplification to stay under AMP_CAP gives
//
//   x_cf = (AMP_CAP * (N-1)!)^(1/(N-1))
//
// which is 64 at N = 2, 11.3 at N = 3, and settles into the 6-to-17 range for everything
// above that (it grows like (N-1)/e). Past it, `expint_fraction` runs instead.
//
// The cap has to be a fixed number of ulp, not the whole mantissa. Solving
// x^(N-1)/(N-1)! = 2^mantissa_bits instead runs the recurrence until the amplification has
// consumed every bit, which leaves the answer with no correct digits at all just below the
// threshold for any N >= 6 (8.2e-3 relative at N = 12, x = 90; 24% at N = 20, x = 51). The
// asymptotic series on the far side needs x of 45 to 110 to converge, so that rule also
// leaves a band for every N >= 4 where neither method works. Boost.Math avoids the question
// entirely: it takes a continued fraction for essentially all x >= 1 and has no forward
// recurrence.
const fn recurrence_threshold(n: usize) -> f64 {
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

    // target = AMP_CAP * (n-1)!
    let target: f64 = {
        let mut f = AMP_CAP;
        let mut i = 2usize;
        while i < n {
            f *= i as f64;
            i += 1;
        }
        f
    };

    // Bisect for the k-th root of target. The largest value it can take is at k = 1
    // (target itself, AMP_CAP); above that the root pulls it into the single digits, so a
    // ceiling of AMP_CAP bounds every case.
    let mut lo = 0.0f64;
    let mut hi = if k == 1 { target } else { AMP_CAP };

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

    // No lower clamp is needed: the continued fraction converges for every x > 0 (more
    // slowly as x falls, but it converges), unlike the asymptotic series this replaced,
    // which diverged outright below x = N and forced a clamp there.
    (lo + hi) * 0.5
}

/// Iteration cap for [`expint_fraction`].
///
/// The fraction is only entered above [`recurrence_threshold`], and the slowest case at any
/// threshold needs 26 iterations (N = 5..10, where the threshold bottoms out near x = 6);
/// it falls to 8 by x = 60 and 6 by x = 90. Lanes freeze as they converge and the loop
/// exits once all of them have, so this bound is a backstop rather than a trip count.
const CF_MAX_ITER: u32 = 48;

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
            const RECURRENCE_THRESHOLD: Self = const { recurrence_threshold(N) as f32 };
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
            const RECURRENCE_THRESHOLD: Self = const { recurrence_threshold(N) };
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
        -0.528611029520217142048e-6,
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

/// `$E_N(x)$` by its continued fraction, evaluated in modified Lentz form.
///
/// ```text
/// E_n(x) = e^{-x} / (x + n - 1*n/(x + n + 2 - 2*(n+1)/(x + n + 4 - ...)))
/// ```
///
/// This is what Boost.Math uses for `$E_n$` at essentially all `$x \ge 1$`, and what this
/// kernel now uses above [`recurrence_threshold`]. It replaced an asymptotic series that
/// could not converge in the range it was being asked to cover: measured against a 45-digit
/// reference, the fraction holds **~1e-16 for every order from 1 to 20 at every `$x \ge 2$`**,
/// where the series it replaced left an unreachable band for every `$N \ge 4$`.
///
/// Lentz's formulation is the one to use here because it never forms the convergents
/// directly. It carries the *ratios* `c` and `d`, so nothing overflows even where the
/// numerator and denominator separately would. The two `is_zero` guards are Lentz's own: a
/// vanishing denominator is substituted with a tiny value, which perturbs the result by less
/// than an ulp and keeps the recurrence going.
///
/// Divisions here are exact rather than `approx_div` at every tier. The policy enters through
/// the *convergence tolerance* instead, which is the knob that actually pays: iterations, not
/// the cost of each one.
///
/// # The tolerance is where the policy lives
///
/// The loop stops a lane once `|delta - 1|` falls under the tolerance, and each tier names
/// **a fraction of the mantissa to keep** rather than an absolute figure, so it means the
/// same thing in either format. Measured against a 40-digit reference at `N = 8`, `x = 6.5`:
///
/// | tier | tolerance | binary64 | binary32 |
/// |---|---|---|---|
/// | `Average` and up | `EPSILON` | 1.5e-16 | 1.3e-07 |
/// | `Medium` | `eps^(3/4)`, three quarters | 4.4e-13 | 5.4e-07 |
/// | `Worst` | `sqrt(eps)`, half | 1.1e-09 | 6.8e-05 |
///
/// Those savings land exactly where the cost is. The fraction is dearest near the threshold
/// and converges in 6 to 8 iterations by `$x = 60$` whatever the tier, so the tiers collapse
/// on their own where the function is easy.
///
/// **`Average` and above are untouched**, matching `poisson`'s `stirlerr_terms`: only the two
/// tiers that exist to trade accuracy for speed do so, and the default policy keeps full
/// precision. The tolerance is also floored at the format's `EPSILON`, so asking binary32 for
/// `1e-8` does not spin the loop chasing digits it cannot represent.
///
/// An asymptotic series was considered for the low tiers and rejected. It is *anti-correlated
/// with need*: at `N = 8, x = 6.1` - the same worst case above - the best it can reach is
/// 100% relative error, because its terms grow rather than shrink until `$x > N$`. By the
/// time it is accurate (`$x \approx 90$`) the fraction already converges in 7 iterations. It
/// can only help where help is least needed.
#[inline(always)]
fn expint_fraction<P, E, V, const N: usize>(x: V, exp_neg_x: V) -> V
where
    P: Policy,
    E: FloatElementWithBits + ExpIntConsts<N>,
    V: FloatVectorWithBits<Element = E> + crate::specialized::SpecializedSpecialMath<E>,
{
    // Stated as a fraction of the format's mantissa rather than as an absolute figure, so
    // the tiers mean the same thing in binary32 and binary64. An absolute constant does not
    // survive the format change: 1e-8 is a real relaxation against a binary64 epsilon of
    // 2.2e-16 and is *below* a binary32 one of 1.2e-7, so binary32 would clamp straight back
    // to full precision and the tier would buy nothing at all.
    let tol = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // Half the mantissa: 1.5e-8 in binary64, 3.4e-4 in binary32.
        <V as FloatConsts>::SQRT_EPSILON
    } else if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // Three quarters of it: eps^(3/4) is eps^(1/2) * eps^(1/4), so the two constants
        // FloatConsts already carries give it as a plain multiply rather than a root.
        // 3.4e-12 in binary64, 3.6e-6 in binary32.
        <V as FloatConsts>::SQRT_EPSILON * <V as FloatConsts>::FOURTH_ROOT_EPSILON
    } else {
        <V as FloatConsts>::EPSILON
    };

    let tiny = V::MIN_POSITIVE;
    let two = V::ONE + V::ONE;
    let n_large = const { N as thermite::LargeInt };

    // b_0 = x + n, and the first convergent is 1/b_0. Both are positive for x > 0, so the
    // opening reciprocal needs no guard.
    let mut b = x + V::splat(E::from_int(n_large));
    let mut c = V::MAX;
    let mut d = V::ONE / b;
    let mut h = d;

    let mut active = <V::Mask as GenericMask>::TRUTHY;
    let mut i = 1u32;

    while i <= CF_MAX_ITER {
        // a_i = -i(n + i - 1)
        let a = V::splat(E::from_int(
            -(i as thermite::LargeInt) * (n_large - 1 + i as thermite::LargeInt),
        ));
        b += two;

        let den = a.mul_adde(d, b);
        d = V::ONE / den.is_zero().select(tiny, den);

        let num = b + a / c;
        c = num.is_zero().select(tiny, num);

        let delta = c * d;

        // Frozen lanes keep the value they converged to. Continuing to multiply a converged
        // lane by a delta that is only approximately one would walk it back off the answer.
        h = active.select(h * delta, h);

        // Converged once delta reaches one. Checked every fourth iteration so the reduction
        // is amortized.
        active &= (delta - V::ONE).abs().cmp_gt(tol);

        if i.is_multiple_of(4) && active.none() {
            break;
        }

        i += 1;
    }

    h * exp_neg_x
}

#[inline(always)]
/// `$E_N(x)$` only. See [`expint_double_primal`] for the shape of the computation.
pub fn expint_double<P: Policy, E, V, const N: usize>(x: V) -> V
where
    E: FloatElementWithBits + ExpIntConsts<N>,
    V: FloatVectorWithBits<Element = E> + crate::specialized::SpecializedSpecialMath<E>,
{
    expint_double_primal::<P, E, V, N>(x).0
}

/// `$E_N(x)$` together with the adjacent lower order `$E_{N-1}(x)$`, which is
/// `$-E_N'(x)$` by differentiation under the integral sign.
///
/// The lower order comes from whichever direction is stable in the regime the value
/// itself was computed in: below [`ExpIntConsts::RECURRENCE_THRESHOLD`] the forward
/// recurrence is running anyway, so `E_{N-1}` is just its previous iterate. Above it,
/// where the asymptotic series takes over, the recurrence is inverted instead --
/// `$E_{N-1}(x) = (e^{-x} - (N-1) E_N(x)) / x$`. Inverting is the *stable* direction
/// (it damps by `1/x` where the forward one amplifies by `x`) and its only weakness,
/// the cancellation as `x -> 0`, is unreachable here because that branch only runs for
/// very large `x`.
pub fn expint_double_primal<P: Policy, E, V, const N: usize>(x: V) -> (V, V)
where
    E: FloatElementWithBits + ExpIntConsts<N>,
    V: FloatVectorWithBits<Element = E> + crate::specialized::SpecializedSpecialMath<E>,
{
    let exp_neg_x = (-x).exp_p::<P>();
    let x_ex = exp_neg_x / x;

    if const { N == 0 } {
        let mut result = x_ex;
        // E_{-1}(x) = e^-x (1 + 1/x) / x
        let mut prev = x_ex * (V::ONE + x.reciprocal_p::<P>());

        if const { P::POLICY.check_overflow } {
            let x_is_zero = x.is_zero();
            result = x_is_zero.select(V::INFINITY, result);
            prev = x_is_zero.select(V::INFINITY, prev);

            let bad = x.cmp_lt(V::ZERO) | x.is_nan();
            result = bad.select(V::NAN, result);
            prev = bad.select(V::NAN, prev);
        }

        return (result, prev);
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
    // One order below whatever `e_n` currently holds. The rational path above produced
    // E_1, so before any recurrence step that is E_0 = e^-x / x.
    let mut e_prev = x_ex;

    if const { N > 1 } {
        e_prev = e_n;
        e_n = x.nmul_adde(e_n, exp_neg_x);

        if const { N > 2 } {
            if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
                let mut n = 0;
                while n < (N - 2) {
                    e_prev = e_n;
                    e_n = x.nmul_adde(e_n, exp_neg_x) / V::splat(E::FACTORS[n]);
                    n += 1;
                }
            } else {
                use crunchy::unroll;

                macro_rules! unroll_recurrence {
                    ($($len:tt),*) => {
                        $( if const { N == ($len + 2) } {
                            unroll! { for n in 0..$len {
                                e_prev = e_n;
                                e_n = x.nmul_adde(e_n, exp_neg_x)
                                    .scale(const { if n < N { E::RECIPROCALS[n] } else { E::ONE } });
                            }}
                        } else )* {
                            let mut n = 0;
                            while n < const { if N > 2 { N - 2 } else { 0 } } {
                                e_prev = e_n;
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

    // Past the point where the forward recurrence still holds its digits, take the
    // continued fraction instead. See [`expint_fraction`] and [`recurrence_threshold`];
    // this replaced an asymptotic series that left an unreachable band for every N >= 4.
    if const { N > 1 } && thermite::unlikely(is_very_large.any()) {
        e_n = is_very_large.select(expint_fraction::<P, E, V, N>(x, exp_neg_x), e_n);

        // The forward-carried `e_prev` came from a recurrence this branch just rejected
        // as unreliable, so re-derive it by inverting that recurrence instead:
        //   E_N = (e^-x - x*E_{N-1}) / (N-1)  =>  E_{N-1} = (e^-x - (N-1)*E_N) / x
        // Backward is the stable direction (it damps by 1/x where the forward one
        // amplifies by x), and this branch only runs above the threshold, far from the
        // x -> 0 cancellation that would otherwise spoil it.
        let back = (exp_neg_x - e_n.scale(E::from_int(const { N as thermite::LargeInt - 1 }))) / x;
        e_prev = is_very_large.select(back, e_prev);
    }

    if const { P::POLICY.check_overflow } {
        // E_1(0) = +inf, E_n(0) = 1/(n-1) for n > 1
        let x_is_zero = x.is_zero();

        if const { N == 1 } {
            e_n = x_is_zero.select(V::INFINITY, e_n);
        } else if const { N > 1 } {
            e_n = x_is_zero.select(V::splat(E::ONE_OVER_N_MINUS_1), e_n);
        }

        // Same rule one order down: E_0 and E_1 both diverge at zero, E_n (n >= 2) does not.
        if const { N <= 2 } {
            e_prev = x_is_zero.select(V::INFINITY, e_prev);
        } else {
            e_prev = x_is_zero.select(
                V::splat(E::ONE / E::from_int(const { N as thermite::LargeInt - 2 })),
                e_prev,
            );
        }

        // Negative x: NaN, and NaN in, NaN out.
        let bad = x.cmp_lt(V::ZERO) | x.is_nan();
        e_n = bad.select(V::NAN, e_n);
        e_prev = bad.select(V::NAN, e_prev);
    }

    (e_n, e_prev)
}
