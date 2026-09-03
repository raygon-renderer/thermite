//! The Riemann zeta function, as `$\zeta(s) - 1$` with `$\zeta$` built on top.
//!
//! # Which one is the primitive
//!
//! `$\zeta(s) \to 1$` fast: `$\zeta(40) - 1$` is about `$9.1\times10^{-13}$`, already far below
//! the mantissa of `$\zeta$` itself, and by `$s = 80$` the complement is `8.3e-25`. So a caller
//! who wants the complement cannot get it by subtracting: measured at `s = 80`, forming
//! `zeta(s)` and taking away 1 is **100% wrong**, and by `s = 200` it returns a flat zero
//! where the true value is `1e-61`.
//!
//! The Euler-Maclaurin sum below opens with the `$n = 1$` term, which _is_ that 1, so the
//! complement comes from **omitting** it rather than cancelling it (exact, with no subtraction
//! anywhere), and still carries digits at `s = 700` where the value is around `1e-211`. That
//! makes [`zetac`](Self) the primitive here and `$\zeta = 1 + \zeta_c$` the derived form, the
//! same relationship `exp_m1` has to `exp`.
//!
//! # Algorithm
//!
//! Euler-Maclaurin, truncated at [`N`] direct terms with [`bernoulli_terms`] correction terms:
//!
//! ```math
//! \zeta(s) = \sum_{n=1}^{N-1} n^{-s} + \frac{N^{1-s}}{s-1} + \frac{N^{-s}}{2}
//!          + \sum_{k\ge1} \frac{B_{2k}}{(2k)!}\,(s)_{2k-1}\,N^{-(s+2k-1)}
//! ```
//!
//! The usual alternative is a table of minimax rationals over five or six intervals in `s`
//! (this is what Boost does, in about a thousand lines). That is excellent scalar code and the
//! wrong shape for a vector unit, where selecting a coefficient _table_ per lane means either a
//! gather or evaluating every interval and discarding all but one. Euler-Maclaurin is one
//! straight-line expression for the whole positive axis instead.
//!
//! Borwein's accelerated eta series was the other candidate, and is the more famous one because
//! it converges in the critical strip and over the complex plane. Measured, it needs **22 terms
//! to match this at 10**, more than twice the transcendental calls for the same answer, so it
//! lost on cost. If complex `s` is ever wanted, it becomes interesting again.
//!
//! # Four exponentials, not nine
//!
//! Every direct term is `$n^{-s} = 2^{-s\log_2 n}$` with `$\log_2 n$` a compile-time constant,
//! which reads as one `exp2` per term. But the Dirichlet terms **factor over the primes**: with
//! `$p_n = n^{-s}$` evaluated for `n` in 2, 3, 5, 7, the rest are products:
//! `$p_4 = p_2^2$`, `$p_6 = p_2p_3$`, `$p_8 = p_2^3$`, `$p_9 = p_3^2$`, and
//! `$N^{-s} = p_2p_5$` needs no call of its own. Four transcendentals and five multiplies cover
//! all of `n = 2..10`, at measured accuracy indistinguishable from nine separate calls
//! (4.89e-16 against 4.41e-16).
//!
//! The count is `$\pi(N)$`, the prime-counting function, not `N`, so raising `N` to tighten
//! the critical strip is cheaper than it looks: `N = 16` costs six, `N = 20` costs eight.
//!
//! The correction sum needs **no transcendentals at all**. `$N^{-(s+2k-1)}$` is `$N^{-s}$` times
//! a constant, and `$(s)_{2k-1}/(2k)!$` advances by a two-factor recurrence whose denominator is
//! a compile-time integer, so the whole tail is one multiply-accumulate ladder over the shipped
//! Bernoulli table.
//!
//! # Accuracy
//!
//! Against mpmath at 40 digits, worst relative error with `N = 10` and 8 correction terms:
//! 4.4e-16 for `s` in `[1.5, 5]`, 4.3e-16 for `[5, 40]`, 2.3e-15 through the critical strip
//! `[0.1, 0.9]`, and 4.6e-16 approaching the pole. The strip is the weak region, and `N` is the
//! lever if it ever matters.
//!
//! Negative `s` is **not** reachable by adding terms. The expansion is asymptotic, and its
//! error there gets _worse_ with larger `N` (measured 3.7e-9 at `N = 10`, 5.4e-8 at `N = 16`).
//! It takes the functional equation instead, which lands at `$1 - s > 1$`, back in the region
//! where the series is at its best.

use thermite::{
    const_splat,
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;
use crate::tables::bernoulli::BernoulliNumbers;

/// The truncation point of the direct sum. Ten is the knee: eight direct terms and six
/// correction terms leave 2.4e-14, ten and eight reach 4.4e-16, and more of either buys nothing
/// (12 and 8 measured 2.7e-16). Because the terms factor over the primes, the transcendental
/// cost is `pi(10) = 4` rather than 9.
#[allow(dead_code)] // named in the docs as the truncation point; the value is inlined below
pub const N: usize = 10;

/// Correction terms by precision tier. The dropped term bounds the error directly, and the
/// series is convergent-then-asymptotic in this range, so the tiers are: 8 terms is 4.4e-16, 4
/// is around 1e-11, and 2 is around 1e-7, which straddles f32's floor, where the whole tail is
/// nearly free anyway.
#[inline(always)]
pub const fn bernoulli_terms(precision: PrecisionPolicy) -> usize {
    match precision {
        PrecisionPolicy::Worst => 2,
        PrecisionPolicy::Medium => 4,
        _ => 8,
    }
}

/// Per-element constants: the base-2 logarithms of the primes under `N`. `log2(pi)` for
/// the functional equation comes from `FloatConsts`. Declared for `f32`/`f64`. Add more as needed.
pub trait ZetaConsts {
    /// `log2(3)`, for `3^-s = exp2(-s log2 3)`.
    const LOG2_3: Self;
    /// `log2(5)`.
    const LOG2_5: Self;
    /// `log2(7)`.
    const LOG2_7: Self;
}

impl ZetaConsts for f32 {
    const LOG2_3: f32 = 1.5849624872207642;
    const LOG2_5: f32 = 2.321928024291992;
    const LOG2_7: f32 = 2.8073549270629883;
}

impl ZetaConsts for f64 {
    const LOG2_3: f64 = 1.584962500721156;
    const LOG2_5: f64 = 2.321928094887362;
    const LOG2_7: f64 = 2.807354922057604;
}

/// `zeta(s) - 1` by Euler-Maclaurin for `s > 0`, and optionally `zeta'(s)` alongside it. The
/// leading `n = 1` term is simply never added, which is what makes the complement exact rather
/// than a cancellation. Since the two functions differ by a constant, one derivative
/// serves both.
///
/// `DERIV` is a compile-time flag, so the whole derivative half folds away when it is not
/// wanted. It shares every transcendental with the value: the `log n` weights are constants
/// (`log n = log2 n * ln 2`, and the composite ones are sums of the prime ones), and the
/// correction sum's derivative rides the same ladder with a second accumulator.
///
/// That second accumulator is not optional. Dropping the correction sum's own
/// `d/ds (s)_{2k-1}` term (the tempting simplification, since the correction is already tiny)
/// was measured at **8.9e-5** relative against **2.4e-15** for the full form. The tail is small
/// but its derivative is not small in the same way.
#[inline(always)]
fn zetac_positive<P, E, V, const DERIV: bool>(s: V) -> (V, V)
where
    E: FloatElement + ZetaConsts + BernoulliNumbers,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E> + SpecializedSpecialMath<E>,
    P: Policy,
{
    let neg_s = -s;

    // The four primes below N. Everything else is a product of these.
    let p2 = neg_s.exp2_p::<P>();
    let p3 = (neg_s * V::splat(<E as ZetaConsts>::LOG2_3)).exp2_p::<P>();
    let p5 = (neg_s * V::splat(<E as ZetaConsts>::LOG2_5)).exp2_p::<P>();
    let p7 = (neg_s * V::splat(<E as ZetaConsts>::LOG2_7)).exp2_p::<P>();

    let p4 = p2 * p2;
    let p6 = p2 * p3;
    let p8 = p4 * p2;
    let p9 = p3 * p3;
    let n_s = p2 * p5; // 10^-s, and the fifth call it saves

    // n = 2..9, ordered small-to-large so the accumulation adds like magnitudes together.
    let direct = ((p9 + p8) + (p7 + p6)) + ((p5 + p4) + (p3 + p2));

    // N^{1-s}/(s-1) + N^-s/2, sharing the single N^-s.
    let ten: V = const_splat!(int <E>: 10);
    let boundary = n_s * (ten.approx_div_p::<P>(s - V::ONE) + V::HALF);

    // The correction sum. `u` carries (s)_{2k-1}/(2k)! * N^-(s+2k-1) and advances by
    //   u_{k+1} = u_k (s + 2k - 1)(s + 2k) / ((2k+1)(2k+2) N^2),
    // whose denominator is a compile-time integer, so no transcendental appears here at all.
    let recur: [E; 8] = [
        <E as FloatElement>::ConstRatio::<1, 1200>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 3000>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 5600>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 9000>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 13200>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 18200>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 24000>::VALUE,
        <E as FloatElement>::ConstRatio::<1, 30600>::VALUE,
    ];
    let terms = const { bernoulli_terms(P::POLICY.precision) };

    // u_1 = (s)_1/2! * N^-(s+1) = s N^-s / 20, and its derivative.
    let ln_n = V::LN_10;
    let twentieth: V = const_splat!(ratio <E>: 1 / 20);
    let mut u = s * n_s * twentieth;
    let mut du = (n_s - s * ln_n * n_s) * twentieth;
    let mut a = s + V::ONE; // s + 2k - 1 at k = 1
    let mut tail = V::ZERO;
    let mut dtail = V::ZERO;

    let mut k = 0;
    while k < terms {
        V::_loop_hint();

        let b = V::splat(E::B2N[k]);
        tail = u.mul_adde(b, tail);
        if const { DERIV } {
            dtail = du.mul_adde(b, dtail);
        }

        // u_{k+1} = u_k a(a+1) r_k, so du_{k+1} = [du_k a(a+1) + u_k (2a+1)] r_k. The r_k are
        // pure constants (every bit of the s-dependence lives in u_1), so nothing else enters.
        let step = a * (a + V::ONE);
        let r = V::splat(recur[k]);
        if const { DERIV } {
            du = du.mul_adde(step, u * (a + a + V::ONE)) * r;
        }
        u *= step * r;
        a += V::TWO;
        k += 1;
    }

    let value = (direct + boundary) + tail;

    let deriv = if const { DERIV } {
        // d/ds n^-s = -(ln n) n^-s. Every log is a constant, and the composite ones are sums of
        // the prime ones: ln 4 = 2 ln 2, ln 6 = ln 2 + ln 3, and so on.
        let l2 = V::LN_2;
        let l3 = V::splat(<E as ZetaConsts>::LOG2_3) * l2;
        let l5 = V::splat(<E as ZetaConsts>::LOG2_5) * l2;
        let l7 = V::splat(<E as ZetaConsts>::LOG2_7) * l2;

        let d_direct = -((l3 * p3 + l2 * p2)
            + ((l2 + l2) * p4 + l5 * p5)
            + ((l2 + l3) * p6 + l7 * p7)
            + ((l2 + l2 + l2) * p8 + (l3 + l3) * p9));

        // d/ds [N^-s (N/(s-1) + 1/2)] = -ln(N) * boundary - N^-s N/(s-1)^2.
        let sm1 = s - V::ONE;
        let d_boundary = -(ln_n * boundary) - n_s * ten.approx_div_p::<P>(sm1 * sm1);

        (d_direct + d_boundary) + dtail
    } else {
        V::ZERO
    };

    (value, deriv)
}

/// `zeta(s)` (`ZETAC = false`) or `zeta(s) - 1` (`ZETAC = true`), and its derivative when
/// `DERIV` is set. `zeta` and `zetac` differ by a constant, so the one derivative serves both.
#[inline(always)]
pub fn zeta_core<P, E, V, const ZETAC: bool, const DERIV: bool>(s: V) -> (V, V)
where
    E: FloatElement + ZetaConsts + BernoulliNumbers,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E> + SpecializedSpecialMath<E>,
    P: Policy,
{
    // Negative arguments reflect. The expansion is asymptotic, so this is not a matter of
    // taking more terms. Its error there grows with N. The functional equation maps s < 0 to
    // 1 - s > 1, which is where the series is at its best.
    let reflect = s.cmp_lt(V::ZERO);
    let arg = reflect.select(V::ONE - s, s);

    let (zc, dz_arg) = zetac_positive::<P, E, V, DERIV>(arg);
    let full = V::ONE + zc;

    let mut result = if const { ZETAC } { zc } else { full };
    // The reflected lanes hold d/d(arg), and arg = 1 - s there, so the chain rule's -1 is
    // applied inside the reflected branch rather than here.
    let mut deriv = dz_arg;

    if const { P::POLICY.avoid_branching } || thermite::unlikely(reflect.any()) {
        // zeta(s) = chi(s) zeta(1-s),  chi(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s).
        //
        // 2^s pi^(s-1) is one exp2, not two powers: 2^(s + (s-1) log2 pi).
        let base = (s - V::ONE).mul_adde(V::LOG2_PI, s).exp2_p::<P>()
            * <V as SpecializedSpecialMath<E>>::tgamma::<P>(V::ONE - s);
        let (sin_h, cos_h) = s
            .scale(<E as FloatElement>::ConstRatio::<1, 2>::VALUE)
            .sincos_pi_p::<P>();

        let reflected = base * sin_h * full;

        // Away from the positive axis zeta is nowhere near 1, so taking the complement here is
        // an ordinary subtraction rather than the cancellation `zetac` exists to avoid.
        let out = if const { ZETAC } { reflected - V::ONE } else { reflected };
        result = reflect.select(out, result);

        if const { DERIV } {
            // Differentiating chi(s) zeta(1-s) gives
            //   zeta'(s) = chi(s)[ln 2 + ln pi + (pi/2)cot(pi s/2) - psi(1-s)] zeta(1-s)
            //              - chi(s) zeta'(1-s).
            //
            // Written that way the cotangent blows up at every even negative integer: exactly
            // the trivial zeros, where zeta(s) is 0, so the product is 0 * inf. Folding chi's
            // own sine into it instead leaves `base * (pi/2) * cos(pi s/2)`, which is finite
            // there and needs no guard.
            let logs = V::LN_2 + V::LN_PI - <V as SpecializedSpecialMath<E>>::digamma::<P>(V::ONE - s);
            // `dz_arg` is zeta'(1-s), the derivative with respect to its own argument. The
            // chain rule's d(1-s)/ds = -1 is what makes this term subtract.
            let d_reflected = base * ((logs * sin_h + V::FRAC_PI_2 * cos_h) * full - sin_h * dz_arg);
            deriv = reflect.select(d_reflected, deriv);
        }
    }

    if const { P::POLICY.check_overflow } {
        // The simple pole. The boundary term already divides by s - 1 and produces the correct
        // infinity from the right. This pins the exact hit, where the two-sided limit does not
        // exist and the sign would otherwise come from the zero's.
        let pole = s.cmp_eq(V::ONE);
        result = pole.select(V::INFINITY, result);
        if const { DERIV } {
            deriv = pole.select(V::NEG_INFINITY, deriv);
        }
    }

    (result, deriv)
}

/// `zeta(s)` (`ZETAC = false`) or `zeta(s) - 1` (`ZETAC = true`), for real `s`.
#[inline(always)]
pub fn zeta_impl<P, E, V, const ZETAC: bool>(s: V) -> V
where
    E: FloatElement + ZetaConsts + BernoulliNumbers,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E> + SpecializedSpecialMath<E>,
    P: Policy,
{
    zeta_core::<P, E, V, ZETAC, false>(s).0
}
