//! The Airy functions `$\mathrm{Ai}$`, `$\mathrm{Bi}$` and their derivatives.
//!
//! Airy is Bessel at order `$\pm 1/3$` and `$\pm 2/3$`, with
//! `$\zeta = \tfrac{2}{3}\lvert x\rvert^{3/2}$`. That is not a shortcut anyone invented here.
//! It is how Boost.Math's Airy functions are written, and the reason the fractional-order machinery in
//! [`bessel_nu`](super::jy_real) and [`bessel_ik_nu`](super::ik_real) had to exist
//! first.
//!
//! ```math
//! \begin{aligned}
//! x < 0:\quad \mathrm{Ai} &= \tfrac{\sqrt{-x}}{3}\left(J_{1/3} + J_{-1/3}\right), &
//!             \mathrm{Bi} &= \sqrt{\tfrac{-x}{3}}\left(J_{-1/3} - J_{1/3}\right) \\
//! x > 0:\quad \mathrm{Ai} &= \tfrac{1}{\pi}\sqrt{\tfrac{x}{3}}\,K_{1/3}, &
//!             \mathrm{Bi} &= \sqrt{\tfrac{x}{3}}\left(I_{-1/3} + I_{1/3}\right)
//! \end{aligned}
//! ```
//!
//! with the derivatives the same shapes at order `$2/3$`. Note `$\mathrm{Ai}$` on the positive
//! axis goes through `$K$` rather than the `$I$` **difference**, and Boost's comment says why:
//! "the accuracy is horrible as we're subtracting two very large values". `$\mathrm{Bi}$` uses
//! the `$I$` **sum**, which has no such problem.
//!
//! # Order `1/3` is where the rotation stops costing anything
//!
//! Both branches need the negative order as well as the positive one, and a second Bessel
//! evaluation is the expensive way to get it. One pass suffices:
//! `$J_{-\nu} = J_\nu\cos\nu\pi - Y_\nu\sin\nu\pi$` and
//! `$I_{-\nu} = I_\nu + \tfrac{2}{\pi}\sin(\nu\pi)K_\nu$`, and both kernels return the pair.
//!
//! At thirds those trigonometric factors are **exact**: `$\cos(\pi/3) = 1/2$`,
//! `$\cos(2\pi/3) = -1/2$`, and `$\sin(\pi/3) = \sin(2\pi/3) = \sqrt3/2$`. So the rotation
//! costs two multiplies and no transcendental at all. This is the whole of what a `Thirds`
//! specialisation can buy (there is no cheaper _algorithm_ for a third-order Bessel function
//! in any library), and it is bought here rather than in the order dispatch.
//!
//! # Accuracy is set by `zeta`, which is why the scaled form exists
//!
//! Measured against mpmath (`notes/special/tools/model_airy.py`), the error of the unscaled
//! functions grows **linearly in `$\zeta$`** on both sides of the origin, and for two different
//! reasons that arrive at the same number:
//!
//! | `x` | mechanism | measured |
//! |---|---|---|
//! | `-10` | phase: `$\mathrm{ulp}(\zeta)$` of argument error becomes phase error | 21 eps |
//! | `-300` | same | 3.7e3 eps |
//! | `-1e4` | same | 3.2e5 eps |
//! | `+10` | the `$e^{-\zeta}$` factor costs `$\zeta\varepsilon/2$` relative | 3.2 eps |
//! | `+100` | same | 684 eps |
//!
//! On the negative axis that is irreducible without a two-word `$\zeta$`, and Boost has the
//! same exposure. On the **positive** axis it is not: with `SCALED` the exponential is never
//! formed, because [`bessel_ik_real`](super::ik_real::bessel_ik_real) already returns
//! `$e^{\zeta}K$` natively. The scaled pair holds 1-3 eps everywhere and, being free of the
//! exponential, also has no range limit. `$\zeta$` passes 710 at `$x \approx 104$`, where the
//! unscaled `$\mathrm{Ai}$` goes subnormal (zero by 108) and `$\mathrm{Bi}$` overflows.
//!
//! Boost ships no scaled Airy. SciPy does, as `airye`, and for exactly this reason.

use thermite::{
    LargeInt,
    math::{
        PrimalProjection, TranscendentalMathWithPolicy,
        policy::{Policy, PrecisionPolicy},
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use thermite::const_splat;
use thermite::element::FloatElement;

use crate::specialized::{BesselDetails, SpecializedSpecialMath};
use crate::tables::bessel::airy::AiryZero;
use crate::tables::lgamma1p::LogGamma1p;

/// `$(\mathrm{Ai}(x),\; \mathrm{Ai}'(x),\; \mathrm{Bi}(x),\; \mathrm{Bi}'(x))$`.
///
/// SciPy's `airy` returns this tuple in this order, and so does this.
///
/// With `SCALED`, returns
/// `$(e^{\zeta}\mathrm{Ai},\; e^{\zeta}\mathrm{Ai}',\; e^{-\zeta}\mathrm{Bi},\;
/// e^{-\zeta}\mathrm{Bi}')$` for `$x > 0$` and the unscaled values for `$x \le 0$`, where they
/// oscillate and there is nothing to scale (SciPy's `airye` convention).
///
/// # The four `WANT_*` flags, and why the single-function entry points are not wrappers
///
/// The four outputs split across **two independent Bessel evaluations**: `$\mathrm{Ai}$` and
/// `$\mathrm{Bi}$` come from order `$1/3$`, the two derivatives from order `$2/3$`. Nothing is
/// shared between them, so a caller who wants one value should not pay for both passes, which
/// is the whole reason `airy_ai` is its own entry point rather than `airy(x).0`.
///
/// Each pass is behind a `const` test on the flags, and on the positive axis `WANT_BI` /
/// `WANT_BIP` additionally decide `bessel_ik_real`'s `NEED_I`: `$\mathrm{Ai}$` is `$K_{1/3}$`
/// alone, so asking only for it skips the continued fraction _and_ the asymptotic series that
/// produce `$I$`. So `airy_ai` costs roughly a quarter of `airy`, not a half.
///
/// They are four separate `bool` parameters rather than one bitmask because a **derived**
/// const cannot be a const-generic argument on stable (`generic_const_exprs`), and
/// `NEED_I` has to be passed on. A standalone const parameter can be forwarded. `WANT & BI`
/// cannot. Same wall the Bessel order parameter hit, recorded in the landmines.
///
/// Slots the flags exclude come back as zero. That is why the public entry points take one
/// field each and the tuple ones set all four: the impl is the same function.
///
/// # Cost, and why the two branches are guarded
///
/// A packet spanning the origin runs both branches: `$J/Y$` for the negative lanes, `$I/K$`
/// for the positive ones. Each is behind an `any()` guard, so a packet that does not straddle
/// zero pays for one, which is the common case and worth the two branches.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn airy_impl<
    P,
    E,
    V,
    const NH: usize,
    const NE: usize,
    const NO: usize,
    const FN: LargeInt,
    const FD: LargeInt,
    const SCALED: bool,
    const WANT_AI: bool,
    const WANT_AIP: bool,
    const WANT_BI: bool,
    const WANT_BIP: bool,
>(
    x: V,
    t: &LogGamma1p<E, NE, NO>,
    z: &AiryZero<E>,
    far_threshold: E,
) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>
        + TranscendentalMathWithPolicy
        + SpecializedTranscendentalMath<E>
        + SpecializedSpecialMath<E>
        + PrimalProjection<Primal = V>
        + BesselDetails<V>,
    P: Policy,
{
    // Which Bessel pass each half of the request needs. Order 1/3 carries the values, order
    // 2/3 the derivatives, and the two share nothing.
    let values = const { WANT_AI || WANT_BI };
    let derivs = const { WANT_AIP || WANT_BIP };

    let ax = x.abs();
    let root = ax.sqrt();

    let third: V = const_splat!(ratio <E>: 1 / 3);
    let two_third: V = const_splat!(ratio <E>: 2 / 3);

    // zeta = (2/3) |x|^{3/2}. The single most accuracy-critical line in the file: on the
    // negative axis this is a phase, so its rounding is the error floor.
    let zeta = (ax * root) * two_third;

    // The origin, and everything that rounds to it. `zeta` is subnormal for
    // `3e-216 < |x| < 8e-206` and zero below, and a subnormal `zeta` carries only a few bits,
    // which `Ai ~ zeta^{-1/3}` hands straight back (2.2e-2 relative at `x = 1e-215` with the
    // guard at `zeta == 0`). Every Airy value rounds to its value at the origin there, so the
    // substitution is exact. The same compare catches a denormal-flushing policy's zero, and
    // the Bessel passes get a harmless `zeta` on those lanes so a NaN cannot hold a series
    // open to `max_iterations` (10.5 ms against 3.3 us per packet).
    let at_zero = zeta.cmp_lt(V::MIN_POSITIVE);

    // ---- a second word of zeta, at `Best` -----------------------------------------------
    //
    // The error of everything downstream is `zeta`'s rounding: on the negative axis it is a
    // phase (measured 21 eps at x = -10, 3.7e3 at -300), and on the positive axis it is the
    // exponent of `e^{-zeta}` (684 eps at x = 100). Neither is in the Bessel kernels, which
    // hold a few eps. It is `zeta` itself, one rounded number, being handed to a sine or an
    // exponential of it. So carry the rounding: `zeta = hi + lo` with `lo` from three exact
    // residuals (the square root's, the product's, the constant's), and let the two
    // consumers that are sensitive to it (the Hankel arm's phase and the unscaled
    // exponentials) apply it to first order. Every other consumer is insensitive: the
    // scaled `K`/`I` see `lo / (2 zeta)`, and the small-`zeta` arms see `lo` against their
    // own several eps.
    //
    // Correctly-rounded FMAs (`mul_add`), since the residuals are only exact when fused;
    // `Best` pays the emulation where there is no hardware FMA, the lower tiers pay nothing.
    let zeta_lo = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        // sqrt(ax) = root + r / (2 root), r = ax - root^2 exactly.
        let r = root.nmul_add(root, ax);
        let root_lo = r / (root + root);
        // ax * sqrt(ax) = p + (p_err + ax * root_lo), p_err exact.
        let p = ax * root;
        let p_lo = ax.mul_sub(root, p) + ax * root_lo;
        // 2/3 = c + c_lo, with 2 - 3c exact.
        let c_lo = two_third.nmul_add(const_splat!(int <E>: 3), V::TWO) * third;
        // zeta = c * p: the rounding of that product, plus the two carried terms.
        let lo = two_third.mul_sub(p, zeta) + two_third * p_lo + c_lo * p;
        at_zero.select(V::ZERO, lo)
    } else {
        V::ZERO
    };

    let zeta = at_zero.select(V::ONE, zeta);

    let neg = x.is_negative();
    let pos = !neg;

    // The exact rotation factors at thirds. `sin(pi/3) = sin(2pi/3) = sqrt(3)/2`, and the
    // cosines are +-1/2, so no transcendental is evaluated for the negative order anywhere.
    let half_sqrt3 = V::SQRT_3 * V::HALF;

    // `root` was formed on the way to `zeta`. `Best` pays a second square root for
    // `sqrt(x/3)`. The lower tiers scale the one in hand, at about half an ulp.
    let root_third = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        (ax * third).sqrt()
    } else {
        root * V::FRAC_1_SQRT_3
    };

    let mut ai = V::ZERO;
    let mut aip = V::ZERO;
    let mut bi = V::ZERO;
    let mut bip = V::ZERO;

    // ---- x < 0: the oscillating branch --------------------------------------------------
    //
    // Both `Ai` and `Bi` need `J` **and** `Y` here, because the negative order comes from the
    // rotation, so unlike the positive branch there is nothing the mask can drop inside a
    // pass, only whole passes.
    if neg.any() {
        if values {
            let (j1, y1) = super::jy_real::bessel_jy_real::<P, E, V, NH, NE, NO, FN, FD>(third, zeta, zeta_lo, t);

            // J_{-1/3} = J_{1/3}/2 - (sqrt3/2) Y_{1/3}. Written folded, so `J + J_{-}` and
            // `J - J_{-}` are each one FMA rather than two roundings and a subtract.
            ai = root * (j1.mul_sube(const_splat!(ratio <E>: 3 / 2), half_sqrt3 * y1)) * third;
            bi = -root_third * (j1.mul_adde(V::HALF, half_sqrt3 * y1));
        }

        if derivs {
            let (j2, y2) = super::jy_real::bessel_jy_real::<P, E, V, NH, NE, NO, FN, FD>(two_third, zeta, zeta_lo, t);

            // At 2/3 the cosine flips sign, so the two combinations swap which one is the sum.
            // Ai' = |x| (J_{2/3} - J_{-2/3})/3, Bi' = |x| (J_{2/3} + J_{-2/3})/sqrt3.
            aip = ax * (j2.mul_adde(const_splat!(ratio <E>: 3 / 2), half_sqrt3 * y2)) * third;
            bip = ax * (j2.mul_sube(V::HALF, half_sqrt3 * y2)) * V::FRAC_1_SQRT_3;
        }
    }

    // ---- x > 0: the exponential branch ---------------------------------------------------
    if pos.any() {
        // The exponentials the unscaled form pays for, hoisted so the two orders share them.
        // Each costs `zeta eps / 2` relative (684 eps at x = 100), which is the whole
        // argument for the scaled twin.
        //
        // `e^{-(hi + lo)} = e^{-hi} (1 - lo)` to first order: the second word of `zeta`
        // is what turns the `zeta eps / 2` of the unscaled forms into a few eps at `Best`.
        let (em, ep) = match const { SCALED } {
            true => (V::ONE, V::ONE),
            false => (
                (-zeta).exp_p::<P>() * (V::ONE - zeta_lo),
                zeta.exp_p::<P>() * (V::ONE + zeta_lo),
            ),
        };

        // `e^{-2 zeta}`, the factor the scaled reflection needs, and only `Bi`/`Bi'` need it.
        // It must be its own exponential. Reconstructing it from an `expm1` in hand loses
        // everything past zeta ~ 8, measured on `bessel_ik_half`. `sqrt(3)/pi` is
        // `(2/pi) sin(nu pi)` at both thirds.
        let refl = match const { WANT_BI || WANT_BIP } {
            false => V::ZERO,
            true => V::SQRT_3 * V::FRAC_1_PI * (-(zeta + zeta)).exp_p::<P>() * (V::ONE - (zeta_lo + zeta_lo)),
        };

        if values {
            // `Ai` is `K_{1/3}` alone, so a caller asking only for it skips `I` entirely,
            // which is the continued fraction and the asymptotic series both.
            let (i1, k1) =
                super::ik_real::bessel_ik_real::<P, E, V, V, NE, NO, true, WANT_BI>(third, zeta, t, far_threshold);

            ai = pos.select(root_third * V::FRAC_1_PI * k1 * em, ai);
            bi = pos.select(root_third * refl.mul_adde(k1, i1 + i1) * ep, bi);
        }

        if derivs {
            let (i2, k2) =
                super::ik_real::bessel_ik_real::<P, E, V, V, NE, NO, true, WANT_BIP>(two_third, zeta, t, far_threshold);

            aip = pos.select(-(ax * V::FRAC_1_SQRT_3 * V::FRAC_1_PI) * k2 * em, aip);
            bip = pos.select((ax * V::FRAC_1_SQRT_3) * refl.mul_adde(k2, i2 + i2) * ep, bip);
        }
    }

    // ---- the origin, see `at_zero` above -------------------------------------------------
    (
        at_zero.select(V::splat(z.ai), ai),
        at_zero.select(V::splat(z.aip), aip),
        at_zero.select(V::splat(z.bi), bi),
        at_zero.select(V::splat(z.bip), bip),
    )
}
