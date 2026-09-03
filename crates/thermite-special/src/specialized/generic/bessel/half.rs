//! Bessel functions at **half-integer order**, where all four families are elementary.
//!
//! `$J_{1/2}(x) = \sqrt{2/\pi x}\,\sin x$` and `$J_{-1/2}(x) = \sqrt{2/\pi x}\,\cos x$`. The
//! modified pair swaps the circular functions for hyperbolic ones and `$K$` is a bare
//! exponential. Every other half-integer order follows from the same three-term recurrence the
//! whole family obeys, so this arm needs **no continued fraction, no `$\Gamma$`, and no
//! series**: one `sin_cos` (or one `exp_m1`), one `sqrt`, and a bounded walk.
//!
//! This is the spherical Bessel family wearing different clothes:
//! `$j_n(x) = \sqrt{\pi/2x}\,J_{n+1/2}(x)$` and `$y_n(x) = \sqrt{\pi/2x}\,Y_{n+1/2}(x)$`.
//!
//! # Boost does not do this, and that is not an oversight to copy
//!
//! Boost special-cases `$\nu = 1/2$` for `cyl_bessel_i` only. Its
//! `cyl_bessel_j` runs the full Steed machinery at every half-integer order, and its spherical
//! functions are thin wrappers that call straight back into it. So the
//! elementary route below is _not_ a port. It is the identity Boost declines to exploit,
//! presumably because a scalar library gains little from it. Under SIMD it is the difference
//! between two continued fractions and a `sin_cos`.
//!
//! What Boost's caution _is_ about is real: **the unstable recurrence direction is still
//! unstable at half-integer order.** `$J$` and `$I$` are the minimal solutions and `$Y$` and
//! `$K$` the dominant ones, exactly as at whole order, so the two arms here reuse the two
//! shapes the integer-order kernels measured, see
//! [`super::jy::bessel_jn_pair_impl`] and [`super::ik::bessel_in_pair_impl`].
//!
//! # Domain
//!
//! `$x > 0$`. At half-integer order these functions carry a `$\sqrt{x}$` and are genuinely
//! complex for negative `$x$`, so unlike the integer-order entry points there is no sign to
//! fold. The caller gets a NaN out of the `sqrt`.

use thermite::{
    math::{TranscendentalMathWithPolicy, policy::Policy},
    prelude::*,
};

use thermite::element::FloatElement;

/// The walk both oscillating half-integer families share, given their four seeds.
///
/// Returns `$(J_{a-1}, J_a, Y_{a-1}, Y_a)$`, the neighbour as well as the wanted order,
/// because every derivative identity in this family reaches **down** one and the walk passes
/// through it anyway.
///
/// # Why the spherical functions reuse this unchanged
///
/// `$j_n(x) = \sqrt{\pi/2x}\,J_{n+1/2}(x)$`, and that factor **does not depend on the order**.
/// So the spherical family satisfies the same three-term recurrence with the same
/// coefficients, and differs only in its seeds. Passing spherically-normalised seeds in gives
/// spherically-normalised values out, with no rescaling anywhere and, more usefully, without
/// ever forming the `$\sqrt{2/\pi x}$` that would then have to be cancelled against a
/// `$\sqrt{\pi/2x}$`. See [`super::spherical`].
///
/// `a` is the order on the **cylindrical** grid, `n + 1/2`, in both cases.
#[inline(always)]
pub(super) fn walk_jy<P, E, V>(x: V, a: V, j_lo: V, j_hi: V, y_lo: V, y_hi: V) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    // `m` is whole and non-negative: the number of recurrence steps up from order 1/2.
    let m = a - V::HALF;
    let two_over_x = V::TWO / x;

    // ---- upward, for Y always and for J while the order stays under x --------------------
    //
    // `Y` is the dominant solution, so upward is its stable direction at every order. `J` is
    // the minimal one and upward is safe only below `x`. The lanes where it is not are
    // overwritten by the downward arm below.
    let use_fwd = x.cmp_gt(a);

    let mut jp = j_lo;
    let mut jc = j_hi;
    let mut yp = y_lo;
    let mut yc = y_hi;

    // `h` is the order the step is taken _at_: 1/2, 3/2, ... The recurrence is
    // `f_{h+1} = (2h/x) f_h - f_{h-1}`.
    let mut h = V::HALF;
    let mut step = V::ONE;

    loop {
        let live = step.cmp_le(m);
        if live.none() {
            break;
        }
        V::_loop_hint();

        let coeff = two_over_x * h;

        let jn = coeff.mul_sube(jc, jp);
        jp = live.select(jc, jp);
        jc = live.select(jn, jc);

        let yn = coeff.mul_sube(yc, yp);
        yp = live.select(yc, yp);
        yc = live.select(yn, yc);

        h += V::ONE;
        step += V::ONE;
    }

    // The seed pair is `(f_{-1/2}, f_{1/2})`, so the wanted order is the _second_ slot: after
    // `m` steps `cur` holds order `m + 1/2`, and at `m = 0` the loop never ran and it is still
    // the seed. (`bessel_jy_real`'s walk reads its `prev` instead, since its Temme seed pair starts
    // _at_ the base order rather than one below it.)
    let (mut j_prev, mut j_a) = (jp, jc);

    // ---- downward on ratios, where forward is unstable -----------------------------------
    //
    // Identical in shape to the integer-order arm, with the order grid offset by 1/2:
    // `r_h = J_h/J_{h-1}` satisfies `r_h = 1/(2h/x - r_{h+1})`, seeded at zero well above the
    // wanted order and walked down. The trip count and its tier come from the same two
    // constants the integer kernel measured.
    if !use_fwd.all() {
        let (cn, cd) = const { super::ik::recurrence_x_coeff(P::POLICY.precision) };
        let coeff = V::splat(E::from_ratio(cn, cd));
        let margin = V::splat(E::from_int(super::ik::RECURRENCE_MARGIN as _));

        // Start above the wanted order by the same margin, on the half-integer grid.
        let mut k = (!use_fwd).select(x.mul_adde(coeff, a + margin).ceil() + V::HALF, V::ZERO);

        let mut r = V::ZERO;
        let mut prod = V::ONE;
        let mut prod_prev = V::ONE;
        let mut r_half = V::ONE;

        let three_halves = thermite::const_splat!(ratio <E>: 3 / 2);
        let a_prev = a - V::ONE;

        loop {
            let active = k.cmp_ge(V::HALF);
            if active.none() {
                break;
            }
            V::_loop_hint();

            r = active.select(V::ONE / two_over_x.mul_sube(k, r), r);

            let in_range = active & k.cmp_ge(three_halves);
            prod = (in_range & k.cmp_le(a)).select(prod * r, prod);
            // One factor short of `prod`, which is order `a - 1`. The walk visits it either
            // way, so carrying it costs one select rather than a second pass.
            prod_prev = (in_range & k.cmp_le(a_prev)).select(prod_prev * r, prod_prev);

            r_half = (active & k.cmp_le(V::HALF)).select(r, r_half);

            k -= V::ONE;
        }

        // `J_{1/2}` vanishes at every multiple of pi and `J_{-1/2}` at every odd multiple of
        // pi/2, and they share no zero, so normalising by whichever is larger is always safe.
        // Seeding from `J_{-1/2}` multiplies by `r_{1/2} = J_{1/2}/J_{-1/2}`, which cancels
        // the small value rather than dividing by it. Same trap, same fix, as the integer
        // kernel's `J_0`/`J_1` choice.
        let use_lo = j_lo.abs().cmp_ge(j_hi.abs());
        let base = use_lo.select(j_lo * r_half, j_hi);

        j_a = use_fwd.select(j_a, base * prod);
        // At `a = 1/2` there is no order below `1/2` on this grid except the seed itself, and
        // `prod_prev` is then the empty product, so this is `base`, which is `J_{1/2}` and not
        // `J_{-1/2}`. The `m = 0` case is therefore the caller's to handle, and both callers
        // do: it is exactly where the derivative identity folds back onto a seed.
        j_prev = use_fwd.select(j_prev, base * prod_prev);
    }

    (j_prev, j_a, yp, yc)
}

/// `$(J_\nu(x), Y_\nu(x))$` at half-integer `$\nu$`, both signs of `$\nu$`, for `$x > 0$`.
///
/// `nu` must be exactly a half-odd-integer (`$\pm 1/2, \pm 3/2, \ldots$`). Whole orders do not
/// belong here and are not detected: [`BesselOrder::simplify`](crate::BesselOrder::simplify)
/// narrows `HalfInteger(2m)` to `Integer(m)` before any kernel sees it, which is why the tag
/// stores a numerator.
///
/// # Negative order is a swap, not a rotation
///
/// The general rule at non-integer order is the rotation
/// `$J_{-\nu} = J_\nu\cos\nu\pi - Y_\nu\sin\nu\pi$`. At `$\nu = m + 1/2$` the cosine vanishes
/// **exactly** and the sine is `$(-1)^m$`, so the rotation degenerates into an exchange:
///
/// ```math
/// J_{-(m+1/2)} = (-1)^{m+1}\,Y_{m+1/2}, \qquad Y_{-(m+1/2)} = (-1)^m\,J_{m+1/2}
/// ```
///
/// No trigonometry is evaluated for it, and no cancellation is possible in it, which is the
/// second reason this order class is worth its own kernel, since the general path pays a
/// `sincos_pi` and a pair of products to reach the same answer less exactly.
#[inline(always)]
pub fn bessel_jy_half<P, E, V>(nu: V, x: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = nu.abs();
    let m = a - V::HALF;

    // sqrt(2/(pi x)), the envelope every order here rides on.
    let amp = (V::FRAC_2_PI / x).sqrt();
    let (sin_x, cos_x) = x.sin_cos_p::<P>();

    // The four seeds. `J_{-1/2}` and `Y_{-1/2}` are not extra work. They are the second
    // element each recurrence needs, and they cost a negation apiece from the pair already
    // in hand.
    let j_lo = amp * cos_x; // J_{-1/2}
    let j_hi = amp * sin_x; // J_{+1/2}
    let y_lo = j_hi; // Y_{-1/2} =  J_{1/2}
    let y_hi = -j_lo; // Y_{+1/2} = -J_{-1/2}

    let (_, j_a, _, y_a) = walk_jy::<P, E, V>(x, a, j_lo, j_hi, y_lo, y_hi);

    // The origin: `J_a(0) = 0` and `Y_a(0) = -inf` at every positive half-integer order, and
    // the exchange below turns those into the right signed infinities at negative order. The
    // seeds are `inf * 0` there, so this is a select rather than something that falls out.
    let zero = x.is_zero();
    let j_a = zero.select(V::ZERO, j_a);
    let y_a = zero.select(V::NEG_INFINITY, y_a);

    // Both vanish at infinity, where the seeds are `0 * NaN`.
    let inf = x.cmp_eq(V::INFINITY);
    let j_a = inf.select(V::ZERO, j_a);
    let y_a = inf.select(V::ZERO, y_a);

    // ---- negative order: an exchange with a sign, no trigonometry -------------------------
    let m_odd = (m * V::HALF).fract().cmp_gt(V::ZERO);
    let reflected = nu.is_negative();

    (
        reflected.select(y_a.neg_c(!m_odd), j_a),
        reflected.select(j_a.neg_c(m_odd), y_a),
    )
}

/// The walk both modified half-integer families share, given their seeds.
///
/// Returns `$(I_{a-1}, I_a, K_{a-1}, K_a)$`. `K`'s two seeds are equal (`$K_{-1/2} =
/// K_{1/2}$`), so it takes one value where `$I$` takes two.
///
/// Reused unchanged by the modified **spherical** family for the same reason
/// [`walk_jy`] is: `$i_n(x) = \sqrt{\pi/2x}\,I_{n+1/2}(x)$` and that factor does not depend on
/// the order, so spherically-normalised seeds give spherically-normalised values with no
/// rescaling. `a` is the order on the cylindrical grid, `n + 1/2`.
///
/// Everything here is in the scaled domain, `$(e^{-x}I,\; e^{x}K)$`. The recurrences are
/// homogeneous, so a uniform scaling passes straight through both.
#[inline(always)]
pub(super) fn walk_ik<P, E, V>(x: V, a: V, i_lo: V, i_hi: V, k_seed: V, asym_scale: V) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let m = a - V::HALF;
    let two_over_x = V::TWO / x;

    // ---- K upward, always stable ---------------------------------------------------------
    //
    // The two seeds are equal, so the pair starts degenerate and the first step is what
    // separates them.
    let mut kp = k_seed;
    let mut kc = k_seed;

    let mut h = V::HALF;
    let mut step = V::ONE;

    loop {
        let live = step.cmp_le(m);
        if live.none() {
            break;
        }
        V::_loop_hint();

        // K_{h+1} = K_{h-1} + (2h/x) K_h: an addition, where J/Y had a subtraction. That
        // sign is the whole difference between a dominant solution and a minimal one.
        let kn = (two_over_x * h).mul_adde(kc, kp);
        kp = live.select(kc, kp);
        kc = live.select(kn, kc);

        h += V::ONE;
        step += V::ONE;
    }

    // ---- I: the asymptotic series at large x, the ratio recurrence below -------------------
    //
    // The handover is what makes the tiered trip count safe. The recurrence starts at
    // `a + 24 + c*x` with `c` the tier's coefficient (0.35 at `Best`, 0.25 at `Average`), so
    // below the top tier its accuracy decays with `x`, and past this threshold nothing calls
    // it. Without the arm, `sph_bessel_i_scaled(n = 2)` measured 14131 ULP at `Average`
    // against 4.32 at `Best`. Low orders suffer most, since a larger `a` buys its own
    // headroom. Threshold and series are the integer kernel's own (`asymptotic_series_v`
    // takes a per-lane order): 0.99 eps for `nu = 1/3` at `x = 40`, where it takes over.
    let thresh = (a * a * V::splat(E::from_ratio(1, 3))).max(V::splat(E::from_int(40)));
    let use_asym = x.cmp_ge(thresh);

    let (cn, cd) = const { super::ik::recurrence_x_coeff(P::POLICY.precision) };
    let coeff = V::splat(E::from_ratio(cn, cd));
    let margin = V::splat(E::from_int(super::ik::RECURRENCE_MARGIN as _));

    // Asymptotic lanes start at zero so they cannot drag the packet's trip count, the same
    // guard `bessel_iv_impl` uses.
    let mut k = (!use_asym).select(x.mul_adde(coeff, a + margin).ceil() + V::HALF, V::ZERO);
    let mut r = V::ZERO;
    let mut prod = V::ONE;
    let mut prod_prev = V::ONE;

    let a_prev = a - V::ONE;

    loop {
        let active = k.cmp_ge(V::HALF);
        if active.none() {
            break;
        }
        V::_loop_hint();

        // `+ r` here, `- r` in the J arm: the modified equation flips it, and with it the
        // guarantee that every `r_h` lands in `(0, 1)` so nothing needs rescaling.
        r = active.select(V::ONE / two_over_x.mul_adde(k, r), r);
        prod = (active & k.cmp_le(a)).select(prod * r, prod);
        prod_prev = (active & k.cmp_le(a_prev)).select(prod_prev * r, prod_prev);

        k -= V::ONE;
    }

    // `prod` spans `h = 1/2 ..= a`, so it is `I_a / I_{-1/2}`. At `a = 1/2` the direct seed is
    // better than one continued-fraction step reproducing it, the same reason the integer
    // entry point selects its order-0 and order-1 closed forms out of the ladder.
    let mut i_a = a.cmp_le(V::HALF).select(i_hi, i_lo * prod);

    // One order lower. At `a = 1/2` that is `I_{-1/2}`, which is the seed and where
    // `prod_prev` is the empty product, so the same expression covers it.
    let mut i_prev = i_lo * prod_prev;

    if use_asym.any() {
        // `far_threshold` is only read on the unscaled path, and this one is scaled.
        let far = E::from_int(50);

        // The recurrence is normalization-agnostic, which is what lets the spherical family
        // reuse this walk. An absolute series is not: `asymptotic_series_v` computes the
        // CYLINDRICAL `e^-x I_nu`, so a spherical caller passes `sqrt(pi/2x)` as
        // `asym_scale` to land back in its own convention.
        i_a = use_asym.select(
            asym_scale * super::ik::asymptotic_series_v::<P, E, V, true>(x, a, far),
            i_a,
        );
        i_prev = use_asym.select(
            asym_scale * super::ik::asymptotic_series_v::<P, E, V, true>(x, a_prev, far),
            i_prev,
        );
    }

    (i_prev, i_a, kp, kc)
}

/// `$(I_\nu(x), K_\nu(x))$` at half-integer `$\nu$`, both signs of `$\nu$`, for `$x > 0$`.
///
/// With `SCALED`, returns `$(e^{-x}I_\nu(x),\; e^{x}K_\nu(x))$`, the same convention the
/// integer-order entry points use, and the one this kernel works in **internally regardless**,
/// because the seeds are otherwise unrepresentable: `$\sinh x$` overflows at `$x = 710$` while
/// `$e^{-x}\sinh x$` is `$1/2$` forever. The unscaled form is the scaled one times an
/// exponential, and pays that exponential's `$x\,\varepsilon/2$` relative error, which is the
/// documented reason to prefer the scaled twin on accuracy grounds, not only on range.
///
/// # Seeds
///
/// ```math
/// I_{1/2} = \sqrt{\tfrac{2}{\pi x}}\sinh x,\quad
/// I_{-1/2} = \sqrt{\tfrac{2}{\pi x}}\cosh x,\quad
/// K_{1/2} = K_{-1/2} = \sqrt{\tfrac{\pi}{2x}}\,e^{-x}
/// ```
///
/// Scaled, `$e^{-x}\sinh x = -\mathrm{expm1}(-2x)/2$` and `$e^{-x}\cosh x = (1 + e^{-2x})/2$`,
/// so **one `exp_m1` supplies both** and neither loses a bit to cancellation at small `$x$`,
/// which the algebraically equal `$(1 - e^{-2x})/2$` would.
///
/// # Directions
///
/// `$K$` is the dominant solution and walks **upward**, `$n$` steps, no trip count and no `$x$`
/// dependence. `$I$` is the minimal one and cannot: its upward recurrence subtracts nearly
/// equal terms for `$k \ll x$` and loses bits every step whatever the order. So `$I$` takes the
/// downward **ratio** recurrence `$r_h = 1/(2h/x + r_{h+1})$`, seeded at zero above the wanted
/// order, exactly as the integer-order `$I$` kernel does and with the same two tier constants.
///
/// Unlike `$J$` there is no zero to trip over: `$I_{-1/2} = \sqrt{2/\pi x}\cosh x$` is positive
/// everywhere, so the normalization needs no choice between two seeds.
///
/// # Negative order
///
/// `$K$` is even in `$\nu$` at every order and needs nothing. `$I$` is not, at non-integer
/// order, and the reflection brings `$K$` in:
///
/// ```math
/// I_{-(m+1/2)}(x) = I_{m+1/2}(x) + \tfrac{2}{\pi}(-1)^m K_{m+1/2}(x)
/// ```
///
/// This is a genuine subtraction when `$m$` is odd, and `$I_{-(m+1/2)}$` really does have
/// zeros: `$I_{-3/2}$` vanishes near `$x = 1.1997$`, where `$\tanh x = 1/x$`. The contract is
/// absolute against the larger term, not relative, for the same reason it is for `$J$` at its
/// zeros. Boost's `bessel_ik` carries the same formula with the same exposure.
///
/// `far_threshold` is where the unscaled form halves its exponential, see
/// [`unscale_i`](super::ik::unscale_i). It comes from the `BesselI` table so every `$I$`
/// arm in the crate turns that corner at the same `x`.
#[inline(always)]
pub fn bessel_ik_half<P, E, V, const SCALED: bool>(nu: V, x: V, far_threshold: E) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = nu.abs();
    let m = a - V::HALF;

    let amp = (V::FRAC_2_PI / x).sqrt();

    // `e^{-2x} - 1`, from which both hyperbolic seeds follow without cancellation.
    let e2m1 = (-(x + x)).exp_m1_p::<P>();

    let i_lo = amp * (V::TWO + e2m1) * V::HALF; // e^{-x} I_{-1/2} = amp (1 + e^{-2x})/2
    let i_hi = amp * (-e2m1) * V::HALF; //     e^{-x} I_{+1/2} = amp (1 - e^{-2x})/2
    let k_seed = (V::FRAC_PI_2 / x).sqrt(); // e^{ x} K_{\pm 1/2}

    // `asym_scale` is one: this kernel's seeds are already in the cylindrical normalization
    // the asymptotic series produces.
    let (_, i_a, _, k_a) = walk_ik::<P, E, V>(x, a, i_lo, i_hi, k_seed, V::ONE);

    // The origin: `I_a(0) = 0` and `K_a(0) = +inf`, in either scaling. The reflection below
    // then gives `I_{-a}(0)` its signed infinity through the `K` term.
    let zero = x.is_zero();
    let i_a = zero.select(V::ZERO, i_a);
    let k_a = zero.select(V::INFINITY, k_a);

    // ---- negative order ------------------------------------------------------------------
    let m_odd = (m * V::HALF).fract().cmp_gt(V::ZERO);
    let reflected = nu.is_negative();

    // In the scaled domain the reflection's `K` term carries an extra `e^{-2x}`, since the two
    // families are scaled in opposite directions.
    //
    // That factor is a **second** exponential and cannot be recovered as `1 + expm1(-2x)`:
    // past about `x = 8` the `expm1` sits within an ulp of `-1`, so adding one back leaves
    // `eps/2` absolute on a quantity of size `e^{-2x}` (`I_{-21/2}(15)` at 3.72e-14 against
    // 3.85e-16 for `K`). The `any()` guard keeps the extra call off packets with no negative
    // order, which is most of them.
    let i_out = match reflected.any() {
        false => i_a,
        true => {
            let k_term = (V::FRAC_2_PI * k_a * (-(x + x)).exp_p::<P>()).neg_c(m_odd);
            reflected.select(i_a + k_term, i_a)
        }
    };

    match const { SCALED } {
        true => (i_out, k_a),
        false => (
            super::ik::unscale_i::<P, E, V>(i_out, x, far_threshold),
            k_a * (-x).exp_p::<P>(),
        ),
    }
}
