//! The standard normal CDF `ndtr`, its logarithm `log_ndtr`, and `logerfc = ln erfc`.
//!
//! # Motivation
//!
//! `ndtr(x) = erfc(-x/sqrt 2)/2` is one line, and is here so the forward CDF exists
//! beside its inverse `probit`. The two log forms are not re-expressions: `ndtr`
//! underflows to zero near `x = -38.6` (binary64) and `-14.4` (binary32), and `erfc`
//! near `27` / `9.3`, so `ln(ndtr(x))` and `ln(erfc(x))` return `-inf` exactly where a
//! log-likelihood, a censored-data model or a Bayesian-optimization acquisition
//! function needs them most. Both logs are perfectly ordinary numbers there
//! (`log_ndtr(-100) = -5004.6`). These kernels carry them.
//!
//! # Algorithm
//!
//! Three arms per function, chosen by where each spelling is accurate. For `log_ndtr`,
//! with `u = |x|/sqrt 2`:
//!
//! ```text
//! x >  0:       ln_1p(-erfc(u)/2)           ndtr -> 1, the complement is the small side
//! -5.7 < x <= 0: ln(erfc(u)/2)               bit-identical to ln(ndtr(x))
//! x <= -5.7:    ln(erfcx(u)/2) - u^2        the tail; erfcx has no underflow
//! ```
//!
//! The moderate region runs on `erfc` deliberately. `erfcx` is a Weideman rational whose
//! term count follows the policy, and at the default tier it is 4.2e-10 relative. The
//! log turns relative error into absolute, so `ln(erfcx(u)/2)` at `x = -1` would be
//! 1e6 ulp off the `ln(ndtr(x))` a caller could write by hand. `erfc` is a rational-times-
//! exp fit that holds a few ulp at every tier, and its one weakness (the `x^2` under the
//! exp amplifies the argument's rounding by `x^2`) is exactly what the log absorbs: an
//! error of `c x^2 epsilon` relative to `erfc` is `c x^2 epsilon` absolute in the log,
//! against a result of `-x^2/2`.
//!
//! In the tail the same absorption is what makes `erfcx` affordable: its error goes into
//! the log as an absolute `delta`, against a result dominated by `-u^2`, so the tail arm
//! runs `erfcx` two rungs above the caller's tier ([`LogTailPolicy`], N = 40 from the
//! default tier) and starts at `u = 4`, where `8.7e-16 / 16` is a quarter ulp. The
//! threshold is where `erfc` still has a hundred orders of magnitude of headroom in
//! either format. The arms share one `ln`: the argument is lane-selected between
//! `erfc/2` and `erfcx/2`, and only the tail subtracts `u^2`.
//!
//! `u^2` is formed as `(|x|/2)|x|` rather than `x*x/2`, so the intermediate does not
//! overflow before the result does: `log_ndtr(-1.3e154)` is a representable `-8.45e307`.
//!
//! `logerfc` is the same shape with the tail on the right (`ln(erfcx(x)) - x^2` above
//! `x = 4`, `ln(erfc(x))` between `1/2` and `4`) and a bounded left side. There
//! `erfc(-|x|) = 1 + erf(|x|)`, so the arm is `ln_1p(erf(|x|))`, which is `2|x|/sqrt(pi)`
//! near zero and must not be formed from `erfc`. The same cancellation sits in
//! `erfc(x) = 1 - erf(x)` for small positive `x` (it rounds `1.128e-8` to `1.1e-8` at
//! `x = 1e-8`), so below `x = 1/2` the right side is `ln_1p(-erf(x))` as well. A python
//! model of the seam puts both spellings at 1-2 ulp on either side of it.
//!
//! Each transcendental is evaluated only when some lane's arm needs it, or
//! unconditionally under `avoid_branching`. A packet of one sign in the moderate region,
//! the common case, pays one `erfc` and one log.

use thermite::{
    element::FloatElement,
    math::{
        TranscendentalMathWithPolicy as _,
        algorithms::newtons_method,
        policy::{
            Policy, PrecisionPolicy,
            policies::{ExtraPrecision, LessPrecision, MaxIterations},
        },
    },
    prelude::*,
};

use crate::specialized::{SpecializedRealSpecialMath, SpecializedSpecialMath};

/// The policy the tail arms evaluate `erfcx` under: two precision rungs above the
/// caller's, so the default tier takes the full N = 40 Weideman table.
///
/// The log absorbs `erfcx`'s relative error as an absolute one against a result of
/// `-x^2/2`, which is why the bump is affordable and why the tail can start as early as
/// `u = 4`. Public so that `Dual`'s derivative factor, an `erfcx` quotient, can match.
pub type LogTailPolicy<P> = ExtraPrecision<ExtraPrecision<P>>;

/// Where the log forms hand over from `erfc` to `erfcx`: `u = |x|/sqrt 2` for
/// `log_ndtr`, `|x|` for `logerfc`.
#[inline(always)]
fn tail_start<E: FloatElement, V: FloatVector<Element = E>>() -> V {
    V::splat(<E as FloatElement>::ConstRatio::<4, 1>::VALUE)
}

/// `ndtr(x) = erfc(-x/sqrt 2)/2`, the standard normal CDF.
///
/// The `erfc` kernel handles the reflection to the right side itself, so this is the
/// whole function. The `1/sqrt 2` scaling costs one rounding in the argument, which the
/// tail amplifies by `x^2`. That is the function's own condition number, not the
/// kernel's.
#[inline(always)]
pub fn ndtr_impl<P, E, V>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    <V as SpecializedSpecialMath<E>>::erfc::<P>(x * -V::FRAC_1_SQRT_2) * V::HALF
}

/// `ln(ndtr(x))`, finite for every finite `x`.
#[inline(always)]
pub fn log_ndtr_impl<P, E, V>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    log_ndtr_with_deriv_impl::<P, E, V, false>(x).0
}

/// `(ln ndtr(x), phi(x)/ndtr(x))`: the value and its derivative, the inverse Mills ratio.
///
/// The ratio costs one `exp` in the moderate and right arms (`phi` needs `e^{-x^2/2}`,
/// which `erfc` keeps inside itself) and nothing in the tail, where it is
/// `1/(sqrt(2 pi) a)` from the `erfcx` already in hand. With `DERIV = false` the second
/// element is zero and no extra work is done.
#[inline(always)]
pub fn log_ndtr_with_deriv_impl<P, E, V, const DERIV: bool>(x: V) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let ax = x.abs();
    let u = ax * V::FRAC_1_SQRT_2;
    // (|x|/2)|x|, not x*x/2: the product overflows only where the result does.
    let u2 = (ax * V::HALF) * ax;

    let neg = x.cmp_lt(V::ZERO);
    let tail = neg & u.cmp_gt(tail_start());

    // `arg` is the argument of the one `ln` shared by the two left arms. `sub` is the
    // `u^2` only the tail subtracts.
    let mut arg = V::ZERO;
    let mut sub = V::ZERO;
    let mut right = V::ZERO;
    let mut mills = V::ZERO;

    if const { P::POLICY.avoid_branching } || !tail.all() {
        // ndtr(-|x|), the same bits `ndtr` itself returns.
        let c = <V as SpecializedSpecialMath<E>>::erfc::<P>(u) * V::HALF;
        arg = c;

        if const { P::POLICY.avoid_branching } || !neg.all() {
            right = (-c).ln_1p_p::<P>();
        }

        if const { DERIV } {
            // phi/Phi with Phi = c on the left and 1 - c on the right.
            let phi = (-u2).exp_p::<P>() * V::FRAC_1_SQRT_TAU;
            mills = phi / neg.select(c, V::ONE - c);
        }
    }

    if const { P::POLICY.avoid_branching } || tail.any() {
        let a = <V as SpecializedSpecialMath<E>>::erfcx::<LogTailPolicy<P>>(u) * V::HALF;

        arg = tail.select(a, arg);
        sub = tail.select(u2, V::ZERO);

        if const { DERIV } {
            // Phi = e^{-u^2} a and phi = e^{-u^2}/sqrt(2 pi): the exponential cancels.
            mills = tail.select(V::FRAC_1_SQRT_TAU / a, mills);
        }
    }

    let mut left = V::ZERO;
    if const { P::POLICY.avoid_branching } || neg.any() {
        left = arg.ln_p::<P>() - sub;
    }

    (neg.select(left, right), mills)
}

/// The inverse of [`log_ndtr_impl`]: the `x` with `ln ndtr(x) = y`, for `y <= 0`.
///
/// Newton on `log_ndtr` with the inverse Mills ratio as the derivative, from a `probit(e^y)`
/// seed one tier down wherever `e^y` is a normal number, and from the tail asymptotic
/// `x^2 = -2y - 2 ln(-x) - ln 2 pi` (one substitution) below `y = -700`. `ln ndtr` is concave
/// and increasing, so every Newton step lands left of the root and the iteration is
/// monotone from there. No bracket is needed. The residual tolerance is a few ulp of `y`,
/// which is the forward's own noise floor and, through the ratio `|y| / (x phi/Phi)`,
/// under two ulp of `x` everywhere.
#[inline(always)]
pub fn inv_log_ndtr_impl<P, E, V>(y: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>,
{
    // The domain is (-inf, 0]. y = 0 and the two infinities are selected in at the end.
    let active = y.cmp_lt(V::ZERO) & y.is_finite();

    // Three seeds. On the right (x > 0, y > -ln 2) it is the complement that carries the
    // information: e^y rounds to 1 for |y| under an ulp and probit(1) is nothing, while
    // -expm1(y) is 1 - e^y to full precision, so x0 = -probit(1 - e^y). In the middle it
    // is probit(e^y), which wants e^y as a normal number. The seam is y < -708 in f64 and
    // y < -87 in f32, and is read off e^y rather than spelled per format.
    let p = y.exp_p::<LessPrecision<P>>();
    let far = p.cmp_lt(V::MIN_POSITIVE);
    let right = y.cmp_gt(-V::LN_2);

    let mut x0 = V::ZERO;
    if const { P::POLICY.avoid_branching } || !(far | right).all() {
        x0 = <V as SpecializedRealSpecialMath<E>>::probit::<LessPrecision<P>>(p);
    }
    if const { P::POLICY.avoid_branching } || right.any() {
        // `exp_m1` at the caller's tier, not one down: below `Average` it is `exp(y) - 1`,
        // which is exactly zero for |y| under an ulp, and probit(0) seeds nothing.
        let q = -y.exp_m1_p::<P>();
        x0 = right.select(-<V as SpecializedRealSpecialMath<E>>::probit::<LessPrecision<P>>(q), x0);
    }
    if const { P::POLICY.avoid_branching } || far.any() {
        // x^2 = -2y - 2 ln(-x) - ln 2pi, with -x = sqrt(-2y) inside the log.
        let m2y = -(y + y);
        let ln_m2y = m2y.ln_p::<LessPrecision<P>>();
        let xa = -(m2y - ln_m2y - (V::LN_2 + V::LN_PI)).sqrt();
        x0 = far.select(xa, x0);
    }

    let tol = residual_tolerance::<P, E, V>(y.abs());
    let (x, _) = newtons_method::<V, MaxIterations<P, 8>, _>(x0, tol, active, None, |x| {
        let (v, m) = log_ndtr_with_deriv_impl::<P, E, V, true>(x);
        (v - y, m)
    });

    let x = y.cmp_eq(V::NEG_INFINITY).select(V::NEG_INFINITY, x);
    let x = y.cmp_eq(V::ZERO).select(V::INFINITY, x);
    (active | y.cmp_eq(V::NEG_INFINITY) | y.cmp_eq(V::ZERO)).select(x, V::NAN)
}

/// The residual tolerance the Newton inverses stop at: a tier-dependent number of ulps of
/// `scale`, which the caller sets to the size of the quantity the residual is measured in.
///
/// The floor is the forward kernel's own rounding (a few ulp), below which the residual
/// is noise and the loop would run to its cap for nothing.
#[inline(always)]
pub fn residual_tolerance<P, E, V>(scale: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let ulps: V = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        V::splat(<E as FloatElement>::ConstInt::<65536>::VALUE)
    } else if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        V::splat(<E as FloatElement>::ConstInt::<256>::VALUE)
    } else if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
        V::splat(<E as FloatElement>::ConstInt::<8>::VALUE)
    } else {
        V::splat(<E as FloatElement>::ConstInt::<4>::VALUE)
    };

    scale * (<V as FloatVector>::EPSILON * ulps)
}

/// `ln(erfc(x))`, finite for every finite `x`.
#[inline(always)]
pub fn logerfc_impl<P, E, V>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let ax = x.abs();

    let neg = x.cmp_lt(V::ZERO);
    // ln_1p(+-erf(|x|)): the whole left side, and the right side below 1/2.
    let bounded = neg | ax.cmp_lt(V::HALF);
    // ln(erfcx(x)) - x^2, the right tail. Everything else on the right is ln(erfc(x)).
    let tail = ax.cmp_gt(tail_start()) & !bounded;

    let mut bounded_arm = V::ZERO;
    if const { P::POLICY.avoid_branching } || bounded.any() {
        let e = <V as SpecializedSpecialMath<E>>::erf::<P>(ax);
        bounded_arm = e.neg_c(!neg).ln_1p_p::<P>();
    }

    let mut arg = V::ZERO;
    let mut sub = V::ZERO;
    let mut log_arm = V::ZERO;
    if const { P::POLICY.avoid_branching } || !bounded.all() {
        if const { P::POLICY.avoid_branching } || !(bounded | tail).all() {
            arg = <V as SpecializedSpecialMath<E>>::erfc::<P>(ax);
        }

        if const { P::POLICY.avoid_branching } || tail.any() {
            // The result there is -x^2 - ln(sqrt(pi) x) + ..., so x^2 overflowing means
            // the result does too. No rescue is needed or possible.
            let c = <V as SpecializedSpecialMath<E>>::erfcx::<LogTailPolicy<P>>(ax);
            arg = tail.select(c, arg);
            sub = tail.select(ax * ax, V::ZERO);
        }

        log_arm = arg.ln_p::<P>() - sub;
    }

    bounded.select(bounded_arm, log_arm)
}
