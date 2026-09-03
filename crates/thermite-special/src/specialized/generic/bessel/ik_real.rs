//! Modified Bessel `$I_\nu$` and `$K_\nu$` at **arbitrary real order**.
//!
//! The modified twin of [`bessel_nu`](super::jy_real), and built for the same reason that
//! one was: Boost's Airy functions reach `$\mathrm{Ai}$` and `$\mathrm{Bi}$` for `$x > 0$`
//! through `cyl_bessel_k(1/3, p)` and `cyl_bessel_i(\pm 1/3, p)`, not through `$J$`. Only the
//! `$x < 0$` branch is `$J_{\pm 1/3}$`, which [`bessel_nu`](super::jy_real) already covers.
//!
//! Port target: Boost.Math's `temme_ik`, `CF1_ik` and `CF2_ik`, assembled in its
//! `bessel_ik`. Modelled first in
//! `notes/special/tools/model_bessel_ik.py`, which is where the three departures below were
//! measured rather than argued.
//!
//! # One body, two arithmetics
//!
//! Every function here takes the **order in a real vector `R`** and the **argument in `C`**,
//! with `C: PrimalProjection<Primal = R>`. On the real line `C = R` and nothing changes. Over
//! C, `thermite-complex` instantiates the same body with `C = Complex<R>`: the order stays
//! real (as it does in Amos, whose `zbknu` / `zwrsk` / `zasyi` are exactly the three arms
//! below), every `z`-dependent quantity becomes complex, and the `nu`-only quantities
//! (Temme's `$\Gamma(1\pm\nu)$` pieces, the `$(2k+1)^2 - 4\nu^2$` numerators, the recurrence
//! coefficients) stay real and enter through `C: Mul<R>`, which is two real multiplies
//! rather than a complex one. Every _decision_ the kernel makes about `z` (region, domain,
//! overflow corner) goes through [`BesselDetails`], because on `Complex` the plain
//! comparisons mean something else.
//!
//! # Three regions, and one fewer than Boost has
//!
//! | `x` | `K` | `I` |
//! |---|---|---|
//! | `<= 2` | [`temme_ik`] + upward recurrence | Wronskian, from `K` and [`cf1_i_ratio`] |
//! | `2 .. max(40, nu^2/3)` | [`cf2_ik`] + upward recurrence | same |
//! | above that | [`cf2_ik`] + upward recurrence | [`asymptotic_series_g`](super::ik::asymptotic_series_g) |
//!
//! **Boost's fourth arm is deleted.** It takes an ascending series for `$I$` whenever
//! `$x/\nu < 0.25$`. Measured against the continued fraction alone over
//! orders to 100 and `$x$` to 10 (the whole region where that test can fire), the series is
//! 0.0 to 2.8 eps and the fraction 0.5 to **5.2 eps**. Both are inside the crate's contract, so
//! the arm buys nothing a vector packet would not pay for anyway. It also needs a `powf` and a
//! `tgamma`, and its own prefactor overflows at order 200 where the fraction does not care.
//!
//! **`CF2_ik`'s renormalisation is deleted too.** Boost rescales `q`, `prev`, `current` and `C`
//! whenever `$q < \varepsilon$`, and its comment says why: "particularly an issue for types
//! which have many digits precision but a narrow exponent range. A typical example being a
//! double double type." Measured in binary64 over `$u \in [-1/2, 1/2]$` and `$x$` from 2.001 to
//! `$10^5$`, with and without: **7.72 eps either way**, identical. It is dead code at this
//! precision, and a per-lane select if kept.
//!
//! # What each arm costs
//!
//! `$K$` gets cheaper as `$x$` grows and needs **no asymptotic arm at all**: [`cf2_ik`] takes
//! 9 iterations at `$x = 100$` and **2 at `$x = 10^8$`**, at 1 eps throughout. `$I$` is the
//! opposite: [`cf1_i_ratio`] grows like `$\sqrt{x}$` (39 iterations at 40, 428 at 5000, and
//! simply fails to converge by `$x = 10^6$`), which is what the asymptotic handover is for.
//! Boost's own comment calls that growth `$O(x)$`. Measured here it is `$O(\sqrt{x})$`.
//!
//! # Scaling
//!
//! Everything is carried in the crate's `SCALED` convention, `$(e^{-x}I_\nu,\; e^{x}K_\nu)$`,
//! because that is the form the algorithm _natively produces_: [`cf2_ik`] has the `$e^{-x}$` as
//! an explicit factor, and once `$K$` is scaled the Wronskian returns `$I$` **already scaled,
//! with no exponential anywhere**. The unscaled form is the one paying for a transcendental,
//! the reverse of the small-`$x$` arm where Temme's series is naturally unscaled.

use core::ops::{Add, Div, Mul, Sub};

use thermite::{
    math::{PrimalProjection, TranscendentalMathWithPolicy, algorithms::sum_pair, policy::Policy},
    prelude::*,
};

use thermite::const_splat;
use thermite::element::FloatElement;

use crate::specialized::BesselDetails;
use crate::specialized::generic::lgamma1p::tgamma1pm1_pair;
use crate::tables::lgamma1p::LogGamma1p;

/// `$(K_\nu(x), K_{\nu+1}(x))$` **unscaled**, by Temme's series, for `$|x| \le 2$` and
/// `$\lvert\nu\rvert \le 1/2$`.
///
/// Temme, _Journal of Computational Physics_ vol 19, 324 (1975). Boost's `temme_ik`.
/// The structural twin of
/// [`temme_y_nu`](super::jy_real::temme_y_nu): the same `gamma1`/`gamma2` limits, the same
/// `coef` chain, the same paired-Additive shape through
/// [`sum_pair`](thermite::math::algorithms::sum_pair), combined differently and with the
/// `coef` multiplier positive rather than negative, since `$I$`/`$K$` do not oscillate.
///
/// # Two of the four limits are shipped functions, not guards
///
/// `$c = \sin(\pi\nu)/(\pi\nu)$` is exactly [`sinc_pi`](thermite::math::TranscendentalMath::sinc_pi)
/// and `$d = \sinh\sigma/\sigma$` is exactly [`sinhc`](thermite::math::RealMath::sinhc), so
/// neither needs the `$0/0$` select Boost writes for it. Only `gamma1` keeps one, and its
/// limit is `$-\gamma$`.
///
/// # Precondition
///
/// `$\lvert\nu\rvert \le 1/2$` is not a suggestion. The series is built around
/// `$\Gamma(1\pm\nu)$` near one. The caller reduces the order and walks `$K$` up, which is
/// stable because `$K$` is the dominant solution.
#[inline(always)]
pub fn temme_ik<P, E, R, C, const NE: usize, const NO: usize>(
    nu: R,
    z: C,
    needed: C::Mask,
    t: &LogGamma1p<E, NE, NO>,
) -> (C, C)
where
    E: FloatElement,
    R: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    C: FloatVector<Mask = R::Mask>
        + TranscendentalMathWithPolicy
        + PrimalProjection<Primal = R>
        + Mul<R, Output = C>
        + Div<R, Output = C>
        + Add<R, Output = C>
        + Sub<R, Output = C>,
    P: Policy,
{
    if needed.none() {
        return (C::ZERO, C::ZERO);
    }

    // Everything on the order alone is real.
    let (gp, gm) = tgamma1pm1_pair::<P, E, R, NE, NO>(nu, t);
    let c = nu.sinc_pi_p::<P>();

    let at_zero = nu.abs().cmp_lt(<R as FloatConsts>::EPSILON);
    let gamma1 = at_zero.select(-<R as FloatConsts>::EULER_GAMMA, (R::HALF / nu) * (gp - gm) * c);
    let gamma2 = (R::TWO + gp + gm) * c * R::HALF;

    // Everything on the argument is `C`. `(z/2)^nu` with a real exponent.
    //
    // The real coefficients that multiply a `C` inside an FMA are lifted with
    // `from_primal` rather than written as `C * R`: on the real line `from_primal` is the
    // identity and the FMA fuses exactly as it did before this body was shared, so the
    // real instantiation is bit-identical to the pre-generic kernel.
    let log_half_z = (z * C::HALF).ln_p::<P>();
    let b = (z * C::HALF).powf_p::<P>(C::from_primal(nu));
    let sigma = -(log_half_z * nu);
    let d = sigma.sinhc_p::<P>();

    let mut p = C::from_primal(gp + R::ONE) / (b + b);
    let mut q = b * (R::ONE + gm) * R::HALF;
    let mut f = sigma
        .cosh_p::<P>()
        .mul_sube(C::from_primal(gamma1), d * log_half_z * gamma2)
        / c;

    let f0 = f;
    let h0 = p;
    let mut coef = C::ONE;

    let nu2 = nu * nu;
    // `+z^2/4` where the oscillating twin has `-z^2/4`. That single sign is the difference
    // between a series whose terms alternate and one whose terms do not.
    let coef_mult = z * z * C::from_primal(const_splat!(ratio <E>: 1 / 4));
    let tol = <C as FloatConsts>::EPSILON;

    let mut kf = R::ONE;
    let step = move || {
        // One real reciprocal serves all three: `k^2 - nu^2` is `(k - nu)(k + nu)`.
        let inv = R::ONE / kf.mul_sube(kf, nu2);
        f = f.mul_adde(C::from_primal(kf), p + q) * inv;
        p = p * ((kf + nu) * inv);
        q = q * ((kf - nu) * inv);
        let h = p - f * kf;
        coef *= coef_mult / kf;
        kf += R::ONE;

        (coef * f, coef * h)
    };

    // Non-convergence is only reachable outside the documented domain, and the partial pair is
    // still the best available answer there.
    let (sum, sum1) = match sum_pair::<C, P, _>(tol, needed, (f0, h0), step) {
        Ok(v) | Err(v) => v,
    };

    (sum, (sum1 + sum1) / z)
}

/// `$I_{\nu+1}(x)/I_\nu(x)$` by modified Lentz. Boost's `CF1_ik`.
///
/// The same continued fraction as [`cf1_j_ratio`](super::jy_real) with `$a_j = +1$` instead
/// of `$-1$`, and **no sign chain**: every convergent is positive because `$I_\nu$` does not
/// oscillate. That is one fewer piece of state than the oscillating twin needs, and is why
/// the magnitude here arrives without a separate parity to carry.
///
/// # Converged lanes freeze, and `needed` is not an optimization
///
/// The running value is built by multiplication at the noise floor, so extra steps past
/// convergence walk a lane off its answer, the strict form of the freeze rule in
/// [`iterate`](thermite::math::algorithms::iterate). And the trip count grows like
/// `$\sqrt{x}$`, so a lane bound for the asymptotic arm would otherwise set the packet's cost:
/// 39 iterations at `$x = 40$`, 428 at 5000, and no convergence at all by `$10^6$`.
///
/// The `tiny` sentinel is `sqrt(MIN_POSITIVE)`, which survives being squared by a complex
/// reciprocal, and `MIN_POSITIVE` itself would not.
#[inline(always)]
pub(crate) fn cf1_i_ratio<P, E, R, C>(nu: R, z: C, needed: C::Mask) -> C
where
    E: FloatElement,
    R: FloatVector<Element = E>,
    C: FloatVector<Mask = R::Mask> + PrimalProjection<Primal = R> + Mul<R, Output = C>,
    P: Policy,
{
    if needed.none() {
        return C::ZERO;
    }

    let tol = <C as FloatConsts>::EPSILON.scale(<C::Element as FloatElement>::from_int(2));

    // Boost uses `sqrt(min)` rather than `min` so squaring a substituted value cannot
    // underflow to zero further down the recurrence.
    let tiny = C::MIN_POSITIVE.sqrt();
    let two_over_z = C::TWO / z;

    let mut c = tiny;
    let mut f = tiny;
    let mut d = C::ZERO;

    let mut active = needed;
    let mut kf = R::ONE;
    let mut i = 0usize;

    while i < P::POLICY.max_iterations {
        C::_loop_hint();

        let b = two_over_z * (nu + kf);

        let cn = b + C::ONE / c;
        c = cn.is_zero().select(tiny, cn);

        let dn = b + d;
        d = C::ONE / dn.is_zero().select(tiny, dn);

        let delta = c * d;
        f = f.mul_c(active, delta);

        active &= (delta - C::ONE).abs().cmp_gt(tol);
        if active.none() {
            break;
        }

        kf += R::ONE;
        i += 1;
    }

    f
}

/// `$(e^{x}K_\nu(x),\; e^{x}K_{\nu+1}(x))$` for `$|x| > 2$` and `$\lvert\nu\rvert \le 1/2$`.
///
/// Thompson and Barnett's `$z_1/z_0 = U(\nu+3/2,\,2\nu+1,\,2x)/U(\nu+1/2,\,2\nu+1,\,2x)$`
/// (_Computer Physics Communications_ vol 47, 245). Boost's `CF2_ik`.
///
/// Unlike the oscillating twin's [`cf2_pq`](super::jy_real), this is **entirely real
/// arithmetic** on the real line: no complex Lentz, no six accumulators. It carries a
/// fraction `f` and a series `S` side by side, and the series is the slower of the two to
/// converge, so `S` sets the stopping test.
///
/// # The scaled form is the native one
///
/// Boost writes `$K_\nu = \sqrt{\pi/2x}\;e^{-x}/S$`, with the exponential as an explicit
/// factor rather than something the algorithm computes. Dropping it gives `$e^{x}K_\nu$` for
/// free, which is the crate's `SCALED` convention, which is why nothing in this file's large-
/// `$x$` path evaluates an exponential at all.
///
/// # Cost falls with `x`
///
/// 9 iterations at `$x = 100$`, 6 at 745, 4 at 5000, **2 at `$10^8$`**, at or under 1 eps
/// throughout. So `$K$` needs no large-`$x$` asymptotic arm, unlike every other member of the
/// family.
#[inline(always)]
fn cf2_ik<P, E, R, C>(nu: R, z: C, needed: C::Mask) -> (C, C)
where
    E: FloatElement,
    R: FloatVector<Element = E>,
    C: FloatVector<Mask = R::Mask>
        + PrimalProjection<Primal = R>
        + Mul<R, Output = C>
        + Div<R, Output = C>
        + Add<R, Output = C>,
    P: Policy,
{
    if needed.none() {
        return (C::ZERO, C::ZERO);
    }

    let tol = <C as FloatConsts>::EPSILON;

    // The `a_k` chain depends on the order and the step alone, so it stays real.
    let nu2m = nu.mul_sube(nu, const_splat!(ratio <E>: 1 / 4));
    let mut a = nu2m;
    let mut b = (z + C::ONE) * C::TWO;

    let mut d = C::ONE / b;
    let mut delta = d;
    let mut f = d;

    let mut prev = C::ZERO;
    let mut current = C::ONE;
    let mut cc = -a;
    let mut qq = C::from_primal(-a);
    let mut s = qq.mul_adde(delta, C::ONE);

    let mut active = needed;
    let mut kf = R::TWO;
    let mut i = 0usize;

    while i < P::POLICY.max_iterations {
        C::_loop_hint();

        a -= (kf - R::ONE) * R::TWO;
        b += C::TWO;

        d = C::ONE / d.mul_adde(C::from_primal(a), b);
        delta *= b.mul_sube(d, C::ONE);
        f = f.add_c(active, delta);

        // The `q` recurrence and the series that rides on it. Boost renormalises this trio
        // when `q` gets small. Measured, that is worth exactly nothing in binary64, see the
        // module docs.
        let q = (prev - (b - C::TWO) * current) / a;
        prev = current;
        current = q;
        cc *= -a / kf;
        qq = q.mul_adde(C::from_primal(cc), qq);
        s = qq.mul_adde(delta, s);

        active &= (qq * delta).abs().cmp_gt(s.abs() * tol);
        if active.none() {
            break;
        }

        kf += R::ONE;
        i += 1;
    }

    // sqrt(pi/(2z)) / S. The `e^{-z}` Boost multiplies in here is exactly what SCALED drops.
    let kv = (C::FRAC_PI_2 / z).sqrt() / s;
    let kv1 = kv * ((z + (R::HALF + nu)) + f * nu2m) / z;

    (kv, kv1)
}

/// `$(I_\nu(z), K_\nu(z))$` at **arbitrary real order**, over the positive axis (or, in a
/// complex arithmetic, the right half-plane), scaled by `$(e^{-z}, e^{z})$` when `SCALED`.
///
/// The region select over the arms above. `far_threshold` is where the unscaled form switches
/// to a halved exponential. It comes from the `BesselI` table so that both asymptotic paths in
/// the crate use one constant.
///
/// # Order reduction, unlike the oscillating twin
///
/// `bessel_jy_real` reduces the order only in its small-`x`
/// region, because Steed and the Hankel expansion take `$\nu$` directly. Here **both** `$K$`
/// arms want `$\lvert u\rvert \le 1/2$`, so the split `$\nu = n + u$` and the upward walk are
/// unconditional. That is cheap: `$K$` is the dominant solution, so the walk is its stable
/// direction and costs exactly `$n$` steps with no trip count and no `$x$` dependence, the
/// same shape the integer-order `$K$` kernel ships.
///
/// # `I` comes out of the Wronskian, which is the whole trick
///
/// `$I_\nu K_{\nu+1} + I_{\nu+1}K_\nu = 1/x$` with `$f = I_{\nu+1}/I_\nu$` from
/// `cf1_i_ratio` gives `$I_\nu = (1/x)/(K_\nu f + K_{\nu+1})$`. Both `$K$` values are already
/// in hand from the walk, so the whole first kind costs one continued fraction and a divide,
/// **and no exponential**, because two scaled `$K$`s in the denominator make the quotient
/// scaled too.
///
/// # Negative order
///
/// `$K_{-\nu} = K_\nu$` at every order, so `$K$` needs nothing. `$I$` does:
/// `$I_{-\nu} = I_\nu + \tfrac{2}{\pi}\sin(\nu\pi)K_\nu$`, which in the scaled domain picks up
/// an `$e^{-2x}$` because the two families are scaled in opposite directions. That factor is a
/// second exponential and **must not** be recovered from an `expm1` already in hand, see the
/// measured account on `bessel_ik_half`.
///
/// # Off the domain
///
/// Lanes outside [`BesselDetails::valid`] are kept out of every convergence mask: a NaN term
/// never passes a tolerance test, so such a lane would otherwise hold a series open to
/// `max_iterations` (measured 10.3 ms against 4 us per packet). The origin gets its limits
/// (`$I_a(0) = 0$`, `$K_a(0) = +\infty$`, the reflection's signed infinity at `$-a$`) and the
/// rest is NaN.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_ik_real<P, E, R, C, const NE: usize, const NO: usize, const SCALED: bool, const NEED_I: bool>(
    nu: R,
    z: C,
    t: &LogGamma1p<E, NE, NO>,
    far_threshold: E,
) -> (C, C)
where
    E: FloatElement,
    R: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    C: FloatVector<Mask = R::Mask>
        + TranscendentalMathWithPolicy
        + PrimalProjection<Primal = R>
        + BesselDetails<C>
        + Mul<R, Output = C>
        + Div<R, Output = C>
        + Add<R, Output = C>
        + Sub<R, Output = C>,
    P: Policy,
{
    let a = nu.abs();

    // nu = n + u with n whole and |u| <= 1/2, which both K arms require.
    let n = a.round();
    let u = a - n;

    let valid = C::valid(z);
    let zero = z.is_zero();
    let near = C::near(z) & valid;

    // ---- K at the reduced order ----------------------------------------------------------
    let (mut kp, mut kc) = cf2_ik::<P, E, R, C>(u, z, valid & !near);

    if const { P::POLICY.avoid_branching } || near.any() {
        let (t0, t1) = temme_ik::<P, E, R, C, NE, NO>(u, z, near, t);
        // Temme's series is unscaled and this arm is `|z| <= 2`, so the exponential is bounded
        // by `e^2` and cannot cost range. It is the only `exp` on the small-`z` path.
        let e = z.exp_p::<P>();
        kp = near.select(t0 * e, kp);
        kc = near.select(t1 * e, kc);
    }

    // ---- K upward to the wanted order ----------------------------------------------------
    //
    // The seed pair starts AT the base order, so after `n` steps the answer is in `prev`,
    // unlike `bessel_ik_half`, whose pair starts one below it.
    let two_over_z = C::TWO / z;
    let mut k = R::ONE;
    let mut step = R::ONE;
    let mut i = 0usize;

    while i < P::POLICY.max_iterations {
        let live = step.cmp_le(n);
        if live.none() {
            break;
        }
        C::_loop_hint();

        let next = (two_over_z * kc).mul_adde(C::from_primal(u + k), kp);
        kp = live.select(kc, kp);
        kc = live.select(next, kc);

        k += R::ONE;
        step += R::ONE;
        i += 1;
    }

    // `K_a(0) = +inf` in either scaling, and the walk above reaches it as `inf * 0`.
    let k_a = zero.select(C::INFINITY, kp);
    let k_a1 = zero.select(C::INFINITY, kc);

    // ---- I: the Wronskian below the handover, the asymptotic series above ------------------
    //
    // `NEED_I` gates the expensive half: `cf2_ik` costs 2 iterations at `x = 1e8`, while
    // `cf1_i_ratio` runs 39 at the crossover and 428 by `x = 5000`. A caller wanting only `K`
    // (`Ai` on the positive axis) skips it explicitly rather than trusting dead-code
    // elimination of an unused loop, the distinction Boost's `need_i` / `need_k` flags make.
    let i_a = match const { NEED_I } {
        false => C::ZERO,
        true => {
            let third: R = const_splat!(ratio <E>: 1 / 3);
            let thresh = (a * a * third).max(const_splat!(int <E>: 40));
            let use_asym = C::beyond(z, thresh);

            let fv = cf1_i_ratio::<P, E, R, C>(a, z, valid & !use_asym);
            // `I_a(0) = 0` for every `a > 0` and `1` at `a = 0`, and the Wronskian is
            // `inf / inf` there. Whole orders never reach here on the real line (they have
            // their own kernels), but a complex instantiation routes every order this way.
            let at_origin = a.is_zero().select(C::ONE, C::ZERO);
            // One division: `z (K f + K_1)` cannot overflow where the quotient is finite
            // (for `z < 1` the product is below `K`, and for `z > 1` the scaled `K` is bounded).
            let mut i_a = zero.select(at_origin, C::ONE / (z * k_a.mul_adde(fv, k_a1)));

            if const { P::POLICY.avoid_branching } || use_asym.any() {
                let far = C::exp_far(z, R::splat(far_threshold));
                i_a = use_asym.select(super::ik::asymptotic_series_g::<P, E, R, C, true>(z, a, far), i_a);
            }

            // Negative order. `K` is even and needs nothing. `I` picks up a `K` term, and in
            // the scaled domain an `e^{-2z}` with it.
            let reflected = nu.is_negative();
            if const { P::POLICY.avoid_branching } || reflected.any() {
                let sp = a.sin_pi_p::<P>();
                let refl = i_a + (k_a * (R::FRAC_2_PI * sp)) * (-(z + z)).exp_p::<P>();
                i_a = reflected.select(refl, i_a);

                // At the origin the reflection is `0 + inf * sin(a pi)`: a signed infinity at
                // non-integer order, and `I_{-n}(0) = I_n(0)` where the sine vanishes. The
                // formula above reaches it as `inf * 0` in one component.
                let origin = at_origin + C::from_primal(R::INFINITY.copysign(sp).nz(sp.is_zero()));
                i_a = (zero & reflected).select(origin, i_a);
            }

            i_a
        }
    };

    let (i_out, k_out) = match const { SCALED } {
        true => (i_a, k_a),
        false => {
            let i_out = match const { NEED_I } {
                false => i_a,
                true => {
                    let far = C::exp_far(z, R::splat(far_threshold));
                    super::ik::unscale_i_pair_masked::<P, C>(i_a, i_a, z, far).0
                }
            };

            (i_out, k_a * (-z).exp_p::<P>())
        }
    };

    // Off the domain and not at the origin: undefined, as for the integer `K`.
    let bad = !valid & !zero;
    (bad.select(C::NAN, i_out), bad.select(C::NAN, k_out))
}
