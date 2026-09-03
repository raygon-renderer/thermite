//! Bessel functions at **arbitrary real order**, starting with the large-`x` Hankel arm.
//!
//! Separate from [`bessel_jy`](super::jy), which is the integer-order family and is
//! built from fitted minimax rationals. Nothing here uses a coefficient table at any order:
//! every term comes from a running ratio, which is what makes an arbitrary `$\nu$` possible.
//!
//! # The Hankel expansion, and why it needs no convergence test
//!
//! ```math
//! J_\nu(x) \sim \sqrt{\frac{2}{\pi x}}\left(P(\nu,x)\cos\omega - Q(\nu,x)\sin\omega\right),
//! \qquad \omega = x - \left(\tfrac{\nu}{2} + \tfrac{1}{4}\right)\pi
//! ```
//!
//! with `$P$` the even and `$Q$` the odd part of `$\sum_k a_k(\nu)/x^k$`, and
//!
//! ```math
//! \frac{a_k}{a_{k-1}} = \frac{\mu - (2k-1)^2}{8kx}, \qquad \mu = 4\nu^2
//! ```
//!
//! This series **diverges**. The ratio is about `$k/2x$`, so terms shrink while `$k < 2x$` and
//! grow forever after, and the least term (the floor on achievable accuracy) sits at
//! `$k^{*} = x + \sqrt{x^2 + \nu^2}$` with magnitude around `$e^{-2x}$`.
//!
//! So there is nothing to converge to, and the stopping point is an index rather than a
//! tolerance: the Counted discipline from
//! [`iterate`](thermite::math::algorithms::iterate), not a convergence test.
//!
//! It does **not** use [`sum_counted`](thermite::math::algorithms::sum_counted), though, and
//! the reason is worth recording. Producing `$Y_\nu$` as well as `$J_\nu$` needs `$P$` and
//! `$Q$` kept apart, which is **two accumulators**, and every driver in `iterate` carries one.
//! An earlier `$J$`-only version did fold the two into a single sum by rotating the trig
//! factor through `$\cos, -\sin, -\cos, \sin$`, and that worked, but it cannot produce `$Y$`.
//! Running the ratio chain twice to get both is worse than carrying one extra accumulator.
//!
//! That makes four places wanting a paired-accumulator driver: thermite-compensated's
//! `sin_cos`, this, `cf2_pq` below, and `sum_counted`'s now-vacant slot.
//!
//! Boost instead _tries_ the series and returns a `bool` (`hankel_PQ`),
//! bailing when consecutive terms stop halving, which happens at `$k \approx x$`, only
//! halfway to the least term. Measured, that costs it about two units of `$x$` at small order
//! and makes the arm unreachable entirely for `$\nu \ge 8$`, where its guard trips on the very
//! first term. A try-and-fail arm is also the one shape a packet cannot do cheaply, since
//! every lane would have to agree on whether the attempt worked.
//!
//! # Where it is usable, measured rather than assumed
//!
//! Smallest `$x$` reaching 1 eps under optimal truncation, from
//! `notes/special/tools/model_hankel_divergence.py`:
//!
//! | `$\nu$` | 0 | 1/3 | 1/2 | 1 | 3 | 5 | 8 | 12 |
//! |---|---|---|---|---|---|---|---|---|
//! | binary64 | 17.0 | 16.5 | 1.0 | 17.0 | 17.0 | 17.0 | 18.0 | 25.0 |
//! | binary32 | 6.5 | 6.5 | 1.0 | 6.5 | 7.5 | 7.5 | 11.0 | 24.5 |
//!
//! Flat in `$\nu$` up to about 5, then rising roughly `$1.75\nu$` (see
//! [`hankel_usable_from`]). `$\nu = 1/2$` is exact at any `$x$` because `$\mu = (2\nu)^2$` with
//! `$2\nu$` an odd integer makes `$\mu - (2k-1)^2$` vanish at `$k = \nu - 1/2$` and the series
//! **terminates**. That is the same fact as "half-integer order is elementary", seen from the
//! asymptotic side.
//!
//! # Term counts
//!
//! Terms needed to reach tolerance, worst case **at the gate** and falling from there, since
//! the stop is at tolerance rather than at the floor:
//!
//! | `$x$` | 17 | 20 | 30 | 60 | 300 | 1000 |
//! |---|---|---|---|---|---|---|
//! | binary64 | **29** | 20 | 14 | 10 | 7 | 5 |
//! | binary32 | **5** | 5 | 4 | 4 | 3 | 2 |
//!
//! `N` is a const generic so the loop unrolls, and the caller picks it from that table.
//! Carrying the worst case everywhere costs terms at large `$x$` that are already below
//! epsilon. Harmless numerically, and an open optimization rather than a correctness
//! question.

use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        algorithms::{sum_pair, sum_ratio},
        policy::Policy,
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use thermite::element::FloatElement;
use thermite::{LargeInt, const_element, const_splat};

use crate::specialized::SpecializedSpecialMath;
use crate::specialized::generic::lgamma1p::tgamma1pm1_pair;
use crate::tables::lgamma1p::LogGamma1p;

/// The smallest `x` at which the Hankel arm reaches full precision for order `nu`.
///
/// Only ever used as a _gate_, so being slightly conservative is free and being optimistic is
/// not. `nu` was swept to 12 when this was fitted. Past that the uniform Debye regime takes
/// over and this form should not be trusted. See the table in the
/// [module documentation](self).
///
/// `FLOOR_N / FLOOR_D` is the format's constant floor, which comes from `$e^{-2x} <
/// \varepsilon$` and is therefore per-format: **`<17, 1>` for binary64 and `<13, 2>` for
/// binary32**, both measured rather than derived. It is a const generic for the same reason
/// `N` is: the caller knows its format and the value should be materialized, not computed.
#[inline(always)]
pub fn hankel_usable_from<E, V, const FLOOR_N: LargeInt, const FLOOR_D: LargeInt>(nu: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let floor = V::splat(<E as FloatElement>::ConstRatio::<FLOOR_N, FLOOR_D>::VALUE);

    // The order term, `1.75|nu| + 4`, fitted to the measured rise past `nu ~ 5`.
    nu.abs()
        .mul_adde(const_splat!(ratio <E>: 7 / 4), const_splat!(int <E>: 4))
        .max(floor)
}

/// `$J_\nu(x)$` by the Hankel expansion, at arbitrary real order, for `x` past
/// [`hankel_usable_from`].
///
/// `N` is the term count. See the [module documentation](self) for the measured table. The
/// caller is responsible for the gate. This does not check it, and below the gate the answer
/// is simply the best a divergent series can do, which is not enough.
///
/// # The phase is never formed directly
///
/// `$\omega = x - (\nu/2 + 1/4)\pi$` cannot be computed that way: subtracting an irrational
/// from a large `$x$` destroys exactly the low-order bits that set the phase. The addition
/// formulae are used instead, with `$\sin$` and `$\cos$` of the `$\nu$`-dependent part taken
/// once through [`sincos_pi`](thermite::math::RealMath::sincos_pi) so `$\pi$` never multiplies
/// anything large. Boost's own comment says the same thing about the
/// same expansion.
///
/// # `x_lo`: a second word of the argument, for the phase alone
///
/// Past the gate the error of this arm is the **phase**: `sin x` and `cos x` are as accurate
/// as `x` itself, and a caller whose `x` was computed (Airy's `zeta = (2/3)|z|^{3/2}`)
/// has already lost `x eps / 2` of it to rounding. `x_lo` is that rounding, when the caller
/// has it, and enters here through the addition formulae to first order:
/// `sin(x + lo) = sin x + lo cos x`, `cos(x + lo) = cos x - lo sin x`. Two FMAs. The
/// amplitude series does not need it: its sensitivity to `x` is `O(1/x)`. Callers without a
/// second word pass zero, which is exact.
#[inline(always)]
pub fn hankel_jy_nu<P, E, V, const N: usize>(nu: V, x: V, x_lo: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let (sin_c, cos_c) = nu.mul_adde(V::HALF, const_splat!(ratio <E>: 1 / 4)).sincos_pi_p::<P>();
    let (sin_x0, cos_x0) = x.sin_cos_p::<P>();
    let sin_x = x_lo.mul_adde(cos_x0, sin_x0);
    let cos_x = x_lo.nmul_adde(sin_x0, cos_x0);

    // cos(w) and sin(w) with w = x - c*pi, through the addition formulae.
    let cos_w = cos_x.mul_adde(cos_c, sin_x * sin_c);
    let sin_w = sin_x.mul_sube(cos_c, cos_x * sin_c);

    let mu = (nu + nu) * (nu + nu);
    let inv_x = V::ONE / x;

    // `a` is `a_k / x^k`. The ratio already carries the `1/x`. `2k-1` rides alongside as a
    // vector stepped by a constant, and `1/(8k)` is a compile-time scalar once the loop
    // unrolls, so no term pays a vector division or an integer-to-float convert.
    let mut a = V::ONE;
    let mut j = V::ONE;

    // `P` takes the even `k`, `Q` the odd, and the sign `sigma_k` runs `+ + - - + + - -`,
    // flipping on every EVEN `k`. Unrolling the loop two at a time makes the parity static,
    // so neither the destination nor the flip costs a branch.
    let mut p = V::ONE;
    let mut q = V::ZERO;
    let mut sgn = V::ONE;

    let mut k = 1usize;
    while k <= N {
        V::_loop_hint();

        // Odd k -> Q, sign unchanged.
        a *= j.nmul_adde(j, mu) * (inv_x * V::splat(E::from_ratio(1, 8 * k as LargeInt)));
        j += V::TWO;
        q = sgn.mul_adde(a, q);
        k += 1;

        if k > N {
            break;
        }

        // Even k -> P, and the sign flips first.
        a *= j.nmul_adde(j, mu) * (inv_x * V::splat(E::from_ratio(1, 8 * k as LargeInt)));
        j += V::TWO;
        sgn = -sgn;
        p = sgn.mul_adde(a, p);
        k += 1;
    }

    // sqrt(2/(pi x)), the envelope both kinds ride on.
    let amp = (V::FRAC_2_PI / x).sqrt();

    (amp * p.mul_sube(cos_w, q * sin_w), amp * p.mul_adde(sin_w, q * cos_w))
}

/// `$J_\nu(x)$` by its ascending series, at arbitrary real order, for small `x`.
///
/// ```math
/// J_\nu(x) = \sum_{k\ge 0} \frac{(-1)^k}{k!\,\Gamma(\nu+k+1)}\left(\frac{x}{2}\right)^{\nu+2k}
/// ```
///
/// advanced by the ratio `$t_{k+1}/t_k = -\frac{(x/2)^2}{(k+1)(\nu+k+1)}$`, so the only
/// per-order quantity is the seed. No table at any order, like the rest of this module.
///
/// # Domain: roughly `$x \lesssim 6$`, and the limit is cancellation not convergence
///
/// The series converges everywhere, and Boost's own comment says so ("this series will
/// actually converge rapidly for all small x - say up to x < 20") before adding "but the
/// first few terms are large and divergent which leads to large errors :-(".
/// That is the real bound.
///
/// The terms peak near `$k \approx x$` at a magnitude around `$e^{x}/(\pi x)$` while the
/// answer is `$O(x^{-1/2})$`, so summing them loses about
/// `$1.4427x - \log_2\sqrt{2\pi x}$` bits. Measured envelope-relative at `$\nu = 1/3$` in
/// binary64: **0.00 eps at `x = 8`, 158 at `x = 10`, 6.1e3 at `x = 12`**. So it is usable to
/// about 8 and comfortable to 6, and the `(6, 16)` hole between here and
/// [`hankel_j_nu`] is a real gap that neither arm covers
/// (`notes/special/tools/model_fractional_arms.py`).
///
/// Nothing here detects that. The caller gates on `x`.
///
/// # The seed sets the accuracy floor
///
/// Every term is proportional to `$t_0 = (x/2)^\nu / \Gamma(\nu+1)$`, so the seed's relative
/// error passes straight through to the result and nothing later can recover it. That is one
/// `powf` and one `tgamma`, so the floor is roughly their combined error (about 2 ulp), and
/// no amount of extra terms improves it.
///
/// `$\Gamma(\nu+1)$` has poles at negative integer `$\nu$`, where the seed becomes zero rather
/// than infinite. Negative **integer** orders never arrive here: they reflect through
/// `$J_{-n} = (-1)^n J_n$` before any kernel sees them.
#[inline(always)]
pub fn series_j_nu<P, E, V>(nu: V, x: V, needed: V::Mask) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E> + SpecializedSpecialMath<E>,
    P: Policy,
{
    if needed.none() {
        return V::ZERO;
    }

    let half_x = x * V::HALF;

    let seed = <V as SpecializedTranscendentalMath<E>>::powf::<P>(half_x, nu)
        / <V as SpecializedSpecialMath<E>>::tgamma::<P>(nu + V::ONE);

    let neg_half_sq = -(half_x * half_x);

    // The same ladder `SpecializedRealMath::tolerance` uses: a multiple of EPSILON, so it
    // means the same thing in binary32 and binary64.
    let tol = <V as FloatConsts>::EPSILON.scale(E::from_int(const { P::POLICY.precision.tolerance() }));

    // `k` is carried as a vector and stepped by one rather than converted from the driver's
    // runtime index, for the same reason as in `hankel_j_nu`.
    let mut kf = V::ONE;

    // Additive discipline: the terms go to zero, so converged lanes freeze themselves and
    // `sum_ratio` needs no per-lane select. Its tolerance is relative to the LARGEST term,
    // which is what this series needs. The sum passes through zero at every root of
    // `J_nu`, so a test relative to the running sum would be meaningless there.
    // `needed` keeps lanes bound for another region from setting the trip count here: the
    // terms of a large-`x` lane peak at `k ~ x`, so one stray lane can multiply the packet's
    // work several times over for a value the region select then discards.
    let sum = sum_ratio::<V, P, _>(tol, needed, seed, move |_k, term| {
        let next = term * neg_half_sq / (kf * (nu + kf));
        kf += V::ONE;
        next
    });

    // Non-convergence is only reachable outside the documented domain, and the partial sum is
    // still the best available answer there.
    match sum {
        Ok(v) | Err(v) => v,
    }
}

/// The convergence tolerance both continued fractions below run at.
///
/// **Deliberately not the policy ladder.** `PrecisionPolicy::tolerance` gives 100x `EPSILON`
/// even at `Average`, and that is the right knob when the loop's output _is_ the answer, as
/// in `expint_fraction`. Here both fractions feed a Wronskian normalization that divides by
/// `q + gamma(p - t)`, so slack in either one is amplified rather than passed through, and the
/// modelled 7.6 eps result was measured at this tolerance. Tiering it is an open item, and one
/// that has to be measured end to end rather than reasoned about.
#[inline(always)]
fn cf_tolerance<E, V>() -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    <V as FloatConsts>::EPSILON.scale(const_element!(int <E>: 2))
}

/// `$J_{\nu+1}(x)/J_\nu(x)$` by modified Lentz, **and the sign of `$J_\nu$`**.
///
/// `b_0 = 0`, `a_j = -1`, `b_j = 2(\nu+j)/x` (A&S 9.1.73). Boost's `CF1_jy`.
///
/// # Why the sign has to come from here
///
/// Steed recovers `$\lvert J_\nu\rvert$` from a square root, so the sign has to arrive
/// separately. This fraction is the only place it exists: it is the parity of the number
/// of sign changes in the denominator chain. A magnitude-only Lentz (core's
/// [`lentz`](thermite::math::algorithms::lentz) included) throws it away, which is why this
/// is written out locally rather than delegating.
///
/// # Converged lanes are frozen, and counting instead would not work
///
/// The running value is built by **multiplication** and the fraction runs at the noise
/// floor, `$2\varepsilon$`, so the freeze rule in
/// [`iterate`](thermite::math::algorithms::iterate) applies in its strict form: extra steps
/// past convergence multiply by a `$\Delta$` that is only approximately one and walk the lane
/// off its answer. Since the trip count varies from about 20 to 35 across this arm's range, a
/// _counted_ loop would give the early lanes 15 extra multiplies, worth roughly 15 eps of
/// drift against a 7.6 eps target. So this converges and freezes. It does not count.
/// # `needed` is not an optimization
///
/// A packet spans regions, so this runs whenever **any** lane is in the Steed band, and the
/// lanes that are not must not be allowed to hold the loop open. `needed` seeds the active
/// mask, so an out-of-band lane never delays convergence and never contributes an iteration.
///
/// This matters more here than almost anywhere else in the crate, because the trip count is
/// not merely different out of band. It explodes. A lane at `x = 0.01` heading for Temme
/// would drag CF2 to about **5400** iterations, for a value that is then discarded.
#[inline(always)]
fn cf1_j_ratio<P, E, V>(nu: V, x: V, needed: V::Mask) -> (V, V::Mask)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    let tol = cf_tolerance::<E, V>();

    // Boost uses `sqrt(min)` rather than `min` so that squaring a substituted value cannot
    // underflow to zero further down the recurrence.
    let tiny = V::MIN_POSITIVE.sqrt();
    let two_over_x = V::TWO / x;

    let mut c = tiny;
    let mut f = tiny;
    let mut d = V::ZERO;

    let mut negative = <V::Mask as GenericMask>::FALSY;
    // Seeded from `needed`, not TRUTHY (see the note above).
    let mut active = needed;

    let mut kf = V::ONE;
    let mut i = 0usize;

    while i < P::POLICY.max_iterations {
        V::_loop_hint();

        let b = (nu + kf) * two_over_x;

        // a = -1 throughout, so `b + a/c` and `b + a*d` are a subtract apiece.
        let cn = b - V::ONE / c;
        c = cn.is_zero().select(tiny, cn);

        let dn = b - d;
        d = V::ONE / dn.is_zero().select(tiny, dn);

        let delta = c * d;
        f = f.mul_c(active, delta);

        // The sign chain must stop when the lane does: a frozen lane whose `d` is
        // still flipping would accumulate sign changes its answer never saw.
        negative ^= active & d.is_negative();

        active &= (delta - V::ONE).abs().cmp_gt(tol);
        if active.none() {
            break;
        }

        kf += V::ONE;
        i += 1;
    }

    (-f, negative)
}

/// `$(p, q)$` where `$p + iq$` is the logarithmic derivative of `$H^{(1)}_\nu(x) = J_\nu + iY_\nu$`.
///
/// Boost's `CF2_jy`. This **is** complex arithmetic (a complex Lentz)
/// written out into real components rather than carried in a complex type, which is what Boost
/// does and for the reason its own comment gives: the `std::complex` version measured about
/// ten times slower. Six accumulators (`cr, ci, dr, di, fr, fi`) instead of three, and no
/// complex type anywhere.
///
/// The first step is special because `$a_1$` is **purely imaginary**, `$i(1/4 - \nu^2)/x$`.
/// Expanding `$i\alpha/(c_r + ic_i)$` with `$c_i = 1$` gives the two lines that look wrong
/// next to the loop body and are not.
///
/// Converged lanes are frozen for the same reason as in [`cf1_j_ratio`]: `fr`/`fi` accumulate
/// by complex multiplication at the noise floor.
#[inline(always)]
fn cf2_pq<P, E, V>(nu: V, x: V, needed: V::Mask) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    let tol = cf_tolerance::<E, V>();
    let tiny = V::MIN_POSITIVE.sqrt();

    let nu2 = nu * nu;
    let br = x + x;
    let mut bi = V::TWO;

    let mut fr = -(V::HALF / x);
    let mut fi = V::ONE;

    // First step: `a` is purely imaginary, exactly once.
    let quarter: V = const_splat!(ratio <E>: 1 / 4);
    let a1 = (quarter - nu2) / x;
    let temp = fr.mul_adde(fr, V::ONE);
    let mut ci = bi + a1 * fr / temp;
    let mut cr = br + a1 / temp;
    let mut dr = br;
    let mut di = bi;

    // Seeded from `needed`, for the reason given on `cf1_j_ratio`.
    let mut active = needed;
    let mut kf: V = const_splat!(ratio <E>: 3 / 2);
    let mut i = 0usize;

    loop {
        // Guard both against collapsing to zero before either is inverted.
        let c_small = (cr.abs() + ci.abs()).cmp_lt(tiny);
        cr = c_small.select(tiny, cr);
        let d_small = (dr.abs() + di.abs()).cmp_lt(tiny);
        dr = d_small.select(tiny, dr);

        // 1/D, as a complex reciprocal: one division, two multiplies.
        let rn = V::ONE / dr.mul_adde(dr, di * di);
        dr *= rn;
        di = -(di * rn);

        let delta_r = cr.mul_sube(dr, ci * di);
        let delta_i = ci.mul_adde(dr, cr * di);

        // f *= delta, complex, frozen per lane.
        let next_r = fr.mul_sube(delta_r, fi * delta_i);
        let next_i = fr.mul_adde(delta_i, fi * delta_r);
        fr = active.select(next_r, fr);
        fi = active.select(next_i, fi);

        active &= ((delta_r - V::ONE).abs() + delta_i.abs()).cmp_gt(tol);
        i += 1;
        if active.none() || i >= P::POLICY.max_iterations {
            break;
        }

        // a_k = (k - 1/2)^2 - nu^2 for k >= 2, real from here on.
        let a = kf.mul_sube(kf, nu2);
        bi += V::TWO;

        // `a / |C|^2` once. The same real factor scales both components.
        let at = a / cr.mul_adde(cr, ci * ci);
        cr = at.mul_adde(cr, br);
        ci = at.nmul_adde(ci, bi);
        dr = a.mul_adde(dr, br);
        di = bi + a * di;

        kf += V::ONE;
    }

    (fr, fi)
}

/// `$(J_\nu, Y_u, Y_{u+1})$` by Steed's method at the **reduced order** `$u = \nu - n$`,
/// `$\lvert u\rvert \le 1/2$`, for the band between the two other arms.
///
/// # The order reduction is not optional
///
/// Boost runs CF1 at `$\nu$`, recurs the `$J$` ratio **down** to `$u$`, runs CF2 and the
/// Wronskian at `$u$`, and walks `$Y$` back up (its `x > 2` branch).
/// Running everything at `$\nu$` directly is fine for `$\nu \in [-1/3, 2]$` and catastrophic
/// above: at `$\nu = 7.4$`, `$x = 2.07$`, `$Y$` measured 4.3 million ULP and `$J$` 678
/// against mpmath.
///
/// The reason is CF2. The Thompson-Barnett fraction for `$H'/H$` converges for every
/// `$x > 0$`, but its accuracy collapses once `$\nu$` is well above `$x$`, the same fact
/// that makes the modified twin reduce its order for _both_ `$K$` arms. CF1 has no such
/// problem, so the ratio it delivers at `$\nu$` is walked down instead, and the walk is the
/// stable direction for `$J$`. Below `$\lvert\nu\rvert \le 1/2$` the reduction is the
/// identity and this is exactly the six-line arm.
///
/// # The pieces
///
/// CF1 at `$\nu$` gives `$f_\nu = J_{\nu+1}/J_\nu$` and the sign of `$J_\nu$`. The
/// three-term recurrence walked down from `$\nu$` with a tiny seed (`prev`, `cur` proportional
/// to `$J_{k+1}$`, `$J_k$`) reaches `$u$` with two things in hand: `$f_u = J_{u+1}/J_u$` and
/// the scaling `$J_\nu/J_u$`. CF2 at `$u$` and the Wronskian
/// `$J_u Y'_u - J'_u Y_u = 2/\pi x$` then give `$\lvert J_u\rvert$`, `$Y_u$` and `$Y_{u+1}$`.
/// The sign of `$J_u$` is the sign of `$J_\nu$` times the sign the walk ended on. The
/// caller walks `$Y$` up, which it does for the Temme arm anyway.
///
/// # Measured
///
/// At small order the arithmetic is unchanged: `notes/special/tools/model_steed.py`,
/// envelope-relative against mpmath at 60 digits, over `x` in `[4, 30]` and `nu` in
/// `[-1/3, 2]`, **worst 7.62 eps for `J`, 5.74 for `Y`**. The iteration counts move in
/// opposite directions: CF1 grows with `x` (20 to 35 across the gap), CF2 shrinks (16 to 8),
/// so their sum is nearly flat over the region this covers.
#[inline(always)]
pub fn steed_jy_nu<P, E, V>(nu: V, n: V, u: V, x: V, needed: V::Mask) -> (V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    // Region skip: if no lane is in the Steed band, none of this runs at all. The two
    // continued fractions are the most expensive thing in the file, so this is the difference
    // between a packet of small `x` paying for Steed and not.
    if needed.none() {
        return (V::ZERO, V::ZERO, V::ZERO);
    }

    let (fv, negative) = cf1_j_ratio::<P, E, V>(nu, x, needed);

    // Walk the ratio down from `nu` to `u`, each lane its own `n` steps:
    // `J_{k-1} = (2(u+k)/x) J_k - J_{k+1}`. Boost's tiny seed keeps the chain, which grows
    // like `J_u / J_nu`, away from overflow. At `n = 0` it is untouched and the ratio is one.
    let init = V::MIN_POSITIVE.sqrt();
    let two_over_x = V::TWO / x;
    let half_eps = <V as FloatConsts>::EPSILON * V::HALF;

    let mut prev = fv * init;
    let mut cur = init;
    let mut k = n;
    let mut i = 0usize;

    while i < P::POLICY.max_iterations {
        let active = needed & k.cmp_ge(V::ONE);
        if active.none() {
            break;
        }
        V::_loop_hint();

        let next = ((u + k) * two_over_x).mul_sube(cur, prev);
        // Boost: an exact cancellation to zero breaks the ratio below, so pretend a bit survived.
        let next = next.is_zero().select(prev * half_eps, next);
        prev = active.select(cur, prev);
        cur = active.select(next, cur);

        k -= V::ONE;
        i += 1;
    }

    // Boost's `over` branch: a chain that left the range gives nothing usable, so the ratio
    // is zero and `f_u` a harmless one rather than NaN.
    let over = !cur.is_finite();
    let ratio = over.select(V::ZERO, init / cur);
    let fu = over.select(V::ONE, prev / cur);

    let (p, q) = cf2_pq::<P, E, V>(u, x, needed);

    // t = J'_u / J_u, from J'_u = (u/x) J_u - J_{u+1}.
    let t = u / x - fu;

    // Boost's guard: gamma cancelling exactly to zero breaks everything below it, so pretend
    // one bit survived. Its only known trigger is v = 8.5, x = 4*pi.
    let gamma = (p - t) / q;
    let gamma = gamma.is_zero().select(u * <V as FloatConsts>::EPSILON / x, gamma);

    // The Wronskian supplies the magnitude of `J_u`. The sign is `J_nu`'s times the walk's.
    let w = V::FRAC_2_PI / x;
    let magnitude = (w / gamma.mul_adde(p - t, q)).sqrt();
    let j_u = magnitude.neg_c(negative ^ cur.is_negative());

    let y_u = gamma * j_u;
    let y_u1 = y_u * (u / x - p - q / gamma);

    (j_u * ratio, y_u, y_u1)
}

/// `$(Y_\nu(x), Y_{\nu+1}(x))$` by Temme's series, for **small `x` and `$\lvert\nu\rvert \le
/// 1/2$`**.
///
/// Temme, _Journal of Computational Physics_ vol 21, 343 (1976). Boost's `temme_jy`.
///
/// # Why this arm exists at all
///
/// Steed handles `$Y_\nu$` from about `x = 0.5` upward, so this is not filling a hole in
/// accuracy so much as one in **cost**, and then a hole in accuracy underneath it. Measured,
/// Steed's CF2 needs 22 iterations at `x = 4`, 150 at `0.5`, and **5392** at `0.01` (its trip
/// count grows like `$1/x$`), and below `0.5` it stops being accurate at all (2400 eps at
/// `x = 0.1`, 261000 at `0.01`).
///
/// Temme, over the same range, is **at most 12 terms and 2.19 eps**
/// (`notes/special/tools/model_temme.py`). So it is both cheaper and better below `x = 2`,
/// which is exactly where Boost switches.
///
/// # `|nu| <= 1/2` is a precondition
///
/// Not a suggestion: the series is built around `$\Gamma(1\pm\nu)$` near one. The caller
/// reduces the order and recurs `$Y$` upward, which is stable because `$Y$` is the dominant
/// solution.
///
/// # Four limits, and why the guards are wide
///
/// `d`, `e`, `g1` and `vspv` are each `$0/0$` at `$\nu = 0$` with a finite limit. `d` is
/// exactly [`sinhc`](thermite::math::RealMath::sinhc) and needs no guard. The other three
/// are `select`s on `$\lvert\nu\rvert < \varepsilon$`, wide rather than `== 0`, matching
/// Boost. A wide guard costs nothing here: the substituted limit is correct to several
/// digits well before `$\nu$` reaches `$\varepsilon$`.
#[inline(always)]
pub fn temme_y_nu<P, E, V, const NE: usize, const NO: usize>(
    nu: V,
    x: V,
    needed: V::Mask,
    t: &LogGamma1p<E, NE, NO>,
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if needed.none() {
        return (V::ZERO, V::ZERO);
    }

    let (gp, gm) = tgamma1pm1_pair::<P, E, V, NE, NO>(nu, t);

    let half = nu * V::HALF;
    let spv = nu.sin_pi_p::<P>();
    let spv2 = half.sin_pi_p::<P>();

    let log_half_x = (x * V::HALF).ln_p::<P>();
    let sigma = -(log_half_x * nu);

    // `sinh(sigma)/sigma`, which is what `sinhc` is, so this limit needs no select.
    let d = sigma.sinhc_p::<P>();

    let tiny = <V as FloatConsts>::EPSILON;
    let at_zero = nu.abs().cmp_lt(tiny);

    // The three remaining 0/0 limits. The unused arm may be NaN, but `select` is bitwise, so it
    // does not propagate.
    let e = at_zero.select(
        nu * <V as FloatConsts>::PI_SQUARED * V::HALF,
        (spv2 * spv2 + spv2 * spv2) / nu,
    );

    let denom = (V::ONE + gp) * (V::ONE + gm) * V::TWO;
    let g1 = at_zero.select(-<V as FloatConsts>::EULER_GAMMA, (gp - gm) / (denom * nu));
    let g2 = (V::TWO + gp + gm) / denom;
    let vspv = at_zero.select(<V as FloatConsts>::FRAC_1_PI, nu / spv);

    let mut f = (g1 * sigma.cosh_p::<P>() - g2 * log_half_x * d) * (vspv + vspv);

    let xp = <V as SpecializedTranscendentalMath<E>>::powf::<P>(x * V::HALF, nu);
    let mut p = vspv / (xp * (V::ONE + gm));
    let mut q = vspv * xp / (V::ONE + gp);

    let g0 = f + e * q;
    let h0 = p;
    let mut coef = V::ONE;

    let nu2 = nu * nu;
    let quarter: V = const_splat!(ratio <E>: 1 / 4);
    let coef_mult = -(x * x * quarter);
    let tol = <V as FloatConsts>::EPSILON;

    // `Y_v` and `Y_{v+1}` ride one `coef` chain, which is the paired-Additive shape, so this
    // is `sum_pair` rather than a hand-rolled loop with two accumulators.
    let mut kf = V::ONE;
    let step = move || {
        // One reciprocal serves all three: `k^2 - nu^2` is `(k - nu)(k + nu)`.
        let inv = V::ONE / kf.mul_sube(kf, nu2);
        f = kf.mul_adde(f, p + q) * inv;
        p *= (kf + nu) * inv;
        q *= (kf - nu) * inv;
        let g = f + e * q;
        let h = p - kf * g;
        coef *= coef_mult / kf;
        kf += V::ONE;

        (coef * g, coef * h)
    };

    // Non-convergence is only reachable outside the documented domain, and the partial pair is
    // still the best available answer there.
    let (sum, sum1) = match sum_pair::<V, P, _>(tol, needed, (g0, h0), step) {
        Ok(v) | Err(v) => v,
    };

    (-sum, -(sum1 + sum1) / x)
}

/// `$(J_\nu(x), Y_\nu(x))$` at **arbitrary real order**, over the whole positive axis.
///
/// The region select over the four arms. Everything above this line is a piece. This is the
/// function.
///
/// # Three regions, chosen to avoid overlap rather than to minimise cost
///
/// | `x` | `J` | `Y` |
/// |---|---|---|
/// | `<= 2` | [`series_j_nu`] | [`temme_y_nu`] + upward recurrence |
/// | `2 .. gate` | [`steed_jy_nu`] | same pass |
/// | `>= gate` | [`hankel_jy_nu`] | same pass |
///
/// `gate` is [`hankel_usable_from`]. The ascending series is usable to about `x = 6` and Steed
/// from about `0.5`, so the `2` boundary sits inside both their ranges. It is where Steed
/// stops being cheap (its CF2 needs 42 iterations at `x = 2` and 5392 at `0.01`) rather than
/// where it stops being right. Boost splits at the same place for the same reason.
///
/// **Every lane pays for every region any lane is in.** Each arm therefore receives the mask
/// of lanes that actually want it, so a lane bound elsewhere cannot extend an iteration. See
/// [`iterate`](thermite::math::algorithms::iterate).
///
/// # Negative order, and why it is a rotation rather than a sign
///
/// At non-integer `$\nu$`, `$J_\nu$` and `$J_{-\nu}$` are **linearly independent** (not a sign
/// apart, as they are at whole orders), so the pair rotates:
///
/// ```math
/// J_{-\nu} = J_\nu\cos\nu\pi - Y_\nu\sin\nu\pi, \qquad
/// Y_{-\nu} = J_\nu\sin\nu\pi + Y_\nu\cos\nu\pi
/// ```
///
/// Everything is computed at `$\lvert\nu\rvert$` and rotated once at the end. At whole orders
/// `$\sin\nu\pi$` vanishes and this collapses to the familiar `$(-1)^n$`.
///
/// # Order reduction, and only where it is needed
///
/// [`temme_y_nu`] requires `$\lvert\nu\rvert \le 1/2$`. So in the small-`x` region the order is
/// split as `$\nu = m + u$` with `$m$` whole and `$\lvert u\rvert \le 1/2$`, Temme evaluated at
/// `$u$`, and `$Y$` walked up `$m$` steps by `$Y_{k+1} = (2k/x)Y_k - Y_{k-1}$`. That direction
/// is stable because `$Y$` is the dominant solution, the mirror of `$J$`, where the same
/// direction is the unstable one.
///
/// The other two regions need no reduction. Steed and the Hankel expansion take `$\nu$`
/// directly.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_jy_real<
    P,
    E,
    V,
    const NH: usize,
    const NE: usize,
    const NO: usize,
    const FN: LargeInt,
    const FD: LargeInt,
>(
    nu: V,
    x: V,
    x_lo: V,
    t: &LogGamma1p<E, NE, NO>,
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>
        + TranscendentalMathWithPolicy
        + SpecializedTranscendentalMath<E>
        + SpecializedSpecialMath<E>,
    P: Policy,
{
    // Work at |nu| throughout and rotate once at the end.
    let a = nu.abs();

    let gate = hankel_usable_from::<E, V, FN, FD>(a);

    // Lanes off the positive axis (zero, negative, NaN) are kept out of every convergence
    // mask. A NaN term never passes a tolerance test, so such a lane would otherwise hold a
    // series open to `max_iterations`: measured 11.5 ms against 5 us for a packet with one
    // `x = 0` lane. The origin gets its limits below. The rest is NaN.
    let valid = x.cmp_gt(V::ZERO);
    let zero = x.is_zero();
    let near = x.cmp_le(V::TWO) & valid;
    let far = x.cmp_ge(gate);
    let mid = !near & !far & valid;

    // ---- x <= 2: the ascending series for J, Temme for Y --------------------------------
    let j_near = series_j_nu::<P, E, V>(a, x, near);

    // Both `Y` arms below the Hankel gate work at the reduced order, `|u| <= 1/2`: Temme
    // because its series is built around `Gamma(1 +- u)`, Steed because CF2 collapses
    // once the order is well above `x`. `m` is whole, `u` is the remainder.
    let m = a.round();
    let u = a - m;
    let (yu_near, yu1_near) = temme_y_nu::<P, E, V, NE, NO>(u, x, near, t);

    // ---- 2 < x < gate: Steed at the reduced order gives J_a and the Y pair at u ---------
    let (j_mid, yu_mid, yu1_mid) = steed_jy_nu::<P, E, V>(a, m, u, x, mid);

    // ---- Y upward from `u` to `a`, for both arms at once --------------------------------
    //
    // `Y` is the dominant solution, so this is its stable direction. Bounded by `m`, and
    // every lane freezes at its own order the way `hermitev` does.
    let below_gate = near | mid;
    let mut y_prev = near.select(yu_near, yu_mid);
    let mut y_cur = near.select(yu1_near, yu1_mid);

    let two_over_x = V::TWO / x;
    let mut k = V::ONE;
    let mut step = V::ONE;
    let mut i = 0usize;
    while i < P::POLICY.max_iterations {
        let live = below_gate & step.cmp_le(m);
        if live.none() {
            break;
        }
        V::_loop_hint();

        let next = (u + k).mul_sube(two_over_x * y_cur, y_prev);
        y_prev = live.select(y_cur, y_prev);
        y_cur = live.select(next, y_cur);

        k += V::ONE;
        step += V::ONE;
        i += 1;
    }
    // After exactly `m` steps `y_prev` holds `Y_{u+m} = Y_a`, and at `m = 0` the loop never
    // ran so it still holds the arm's own `Y_u`. Both cases are the same variable.
    let y_low = y_prev;

    // ---- x >= gate: the Hankel expansion gives both ------------------------------------
    let (j_far, y_far) = hankel_jy_nu::<P, E, V, NH>(a, x, x_lo);

    let j_abs = near.select(j_near, mid.select(j_mid, j_far));
    let y_abs = far.select(y_far, y_low);

    // The origin: `J_a(0) = 0` and `Y_a(0) = -inf` for every `a > 0` (whole orders never
    // reach here), and the rotation below turns those into the right signed infinities at
    // negative order.
    let j_abs = zero.select(V::ZERO, j_abs);
    let y_abs = zero.select(V::NEG_INFINITY, y_abs);

    // ---- the reflection, as a rotation --------------------------------------------------
    let (sin_a, cos_a) = a.sincos_pi_p::<P>();
    let reflected = nu.is_negative();

    let j = reflected.select(j_abs.mul_sube(cos_a, y_abs * sin_a), j_abs);
    let y = reflected.select(j_abs.mul_adde(sin_a, y_abs * cos_a), y_abs);

    // Both vanish at infinity, where the Hankel arm's `sin_cos` is NaN.
    let inf = x.cmp_eq(V::INFINITY);
    let j = inf.select(V::ZERO, j);
    let y = inf.select(V::ZERO, y);

    // Off the positive axis and not at the origin: complex at non-integer order, so NaN.
    let bad = !valid & !zero;
    (bad.select(V::NAN, j), bad.select(V::NAN, y))
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use super::*;

    use thermite::Vector;
    use thermite::backend::x86_v2::X86V2;
    use thermite::math::policy::policies::{Precision, Reference};

    use crate::SpecialMathWithPolicy;
    use crate::bessel::{J, Y};

    type V = Vector<f64>;

    const N64: usize = 29;

    /// The pre-reduction shape, `(J_nu, Y_nu)`: the reduced-order arm plus the upward
    /// `Y` walk the kernel does for it.
    fn steed_full<W: FloatVector<Element = f64>>(nu: W, x: W, needed: W::Mask) -> (W, W) {
        let n = nu.round();
        let u = nu - n;
        let (j, mut yp, mut yc) = steed_jy_nu::<Precision, f64, W>(nu, n, u, x, needed);
        let two_over_x = W::TWO / x;
        let mut k = W::ONE;
        let mut step = W::ONE;
        loop {
            let live = step.cmp_le(n);
            if live.none() {
                break;
            }
            let next = (u + k).mul_sube(two_over_x * yc, yp);
            yp = live.select(yc, yp);
            yc = live.select(next, yc);
            k += W::ONE;
            step += W::ONE;
        }
        (j, yp)
    }

    /// One lane is enough here, deliberately. This kernel has no lane-divergent behaviour at
    /// all: `N` is const, there are no masks, and no lane can take a different path, so a
    /// wider register would exercise nothing a single lane does not.
    fn j_nu(nu: f64, x: f64) -> f64 {
        hankel_jy_nu::<Precision, f64, V, N64>(V::splat(nu), V::splat(x), V::ZERO)
            .0
            .extract::<0>()
    }

    /// Plain relative error, for `Y` at small `x` where the function diverges and there is no
    /// nearby zero to make it meaningless.
    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            return if got == 0.0 { 0.0 } else { f64::INFINITY };
        }
        ((got - want) / want).abs()
    }

    /// Envelope-relative, the contract for a function with infinitely many zeros: dividing by
    /// the true value alone is meaningless at one.
    fn env_rel(got: f64, want: f64, x: f64) -> f64 {
        let env = (2.0 / (core::f64::consts::PI * x)).sqrt();
        (got - want).abs() / env.max(want.abs())
    }

    /// `J_{1/2}(x) = sqrt(2/(pi x)) sin(x)`, exactly: the one order with a closed form, and
    /// the one where the asymptotic series terminates rather than diverging.
    ///
    /// This is the strongest available check because the reference has no error of its own.
    #[test]
    fn half_integer_order_matches_its_closed_form() {
        for &x in &[20.0f64, 30.0, 55.0, 100.0, 400.0] {
            let want = (2.0 / (core::f64::consts::PI * x)).sqrt() * x.sin();
            let got = j_nu(0.5, x);
            assert!(
                env_rel(got, want, x) <= 4e-16,
                "J_1/2({x}): got {got}, want {want}, env-rel {:e}",
                env_rel(got, want, x)
            );
        }
    }

    /// Whole orders against the shipped integer kernel at `Reference`, which is libm
    /// bit-for-bit. Two completely independent implementations (this one a table-free
    /// asymptotic series, that one a fitted minimax rational), so agreement is meaningful.
    #[test]
    fn whole_orders_match_the_integer_kernel() {
        for &x in &[20.0f64, 30.0, 55.0, 100.0] {
            for n in [0i32, 1, 2, 5] {
                let want = match n {
                    0 => V::splat(x).bessel_n_p::<Reference, J, 0>(),
                    1 => V::splat(x).bessel_n_p::<Reference, J, 1>(),
                    2 => V::splat(x).bessel_n_p::<Reference, J, 2>(),
                    _ => V::splat(x).bessel_n_p::<Reference, J, 5>(),
                }
                .extract::<0>();

                let got = j_nu(n as f64, x);
                assert!(
                    env_rel(got, want, x) <= 8e-16,
                    "J_{n}({x}): hankel {got}, libm {want}, env-rel {:e}",
                    env_rel(got, want, x)
                );
            }
        }
    }

    /// Thirds: the Airy orders, and the reason this kernel exists.
    ///
    /// `libm` has no fractional-order Bessel, so there is nothing to differentially test
    /// against. These are mpmath at 60 digits. **Both signs are included**, because
    /// `$J_\nu$` and `$J_{-\nu}$` are linearly independent at non-integer order. They are
    /// genuinely different functions, not a sign apart, and Airy needs both.
    #[test]
    fn airy_orders_match_a_high_precision_reference() {
        // (nu, x, J_nu(x)) from mpmath, dps = 60.
        const ROWS: &[(f64, f64, f64)] = &[
            (0.3333333333333333, 20.0, 0.176060580012939),
            (0.3333333333333333, 30.0, -0.13334053387426162),
            (0.3333333333333333, 55.0, -0.10331600929280815),
            (0.3333333333333333, 100.0, -0.02127124485370254),
            (0.3333333333333333, 400.0, -0.03821195249289747),
            (0.6666666666666666, 20.0, 0.1390482612211654),
            (0.6666666666666666, 30.0, -0.1448985197420506),
            (0.6666666666666666, 55.0, -0.10455814523765926),
            (0.6666666666666666, 100.0, -0.056778819380529484),
            (0.6666666666666666, 400.0, -0.027373238316694467),
            (-0.3333333333333333, 20.0, 0.11295251588168025),
            (-0.3333333333333333, 30.0, -0.01588153828630604),
            (-0.3333333333333333, 55.0, -0.0256708694632734),
            (-0.3333333333333333, 100.0, 0.05596216843421023),
            (-0.3333333333333333, 400.0, -0.02903303875870895),
            (-0.6666666666666666, 20.0, 0.02731702987415274),
            (-0.6666666666666666, 30.0, 0.059390728756765904),
            (-0.6666666666666666, 55.0, 0.03032108324903861),
            (-0.6666666666666666, 100.0, 0.07693648950955431),
            (-0.6666666666666666, 400.0, -0.011446867797140254),
        ];

        let mut worst = 0.0f64;
        for &(nu, x, want) in ROWS {
            let got = j_nu(nu, x);
            let e = env_rel(got, want, x);
            assert!(e <= 8e-16, "J_{nu}({x}): got {got}, want {want}, env-rel {e:e}");
            worst = worst.max(e);
        }
        let _ = worst;
    }

    /// What the ascending series can possibly deliver at `x`, envelope-relative.
    ///
    /// Its terms peak near `k ~ x` at about `e^x/(pi x)` while the answer rides an envelope of
    /// `sqrt(2/pi x)`, so the summation loses that ratio, in eps, to cancellation. **A flat
    /// gate here would be wrong in both directions**: it fails at the top of the range, or it
    /// gets loosened until it hides the growth and stops testing anything.
    ///
    /// The constant floor is the seed. Every term is proportional to
    /// `(x/2)^nu / Gamma(nu+1)`, so one `powf` and one `tgamma` set a floor no number of terms
    /// can improve on.
    fn series_bound(x: f64) -> f64 {
        let env = (2.0 / (core::f64::consts::PI * x)).sqrt();
        let peak = x.exp() / (core::f64::consts::PI * x);

        8e-16 + f64::EPSILON * peak / env
    }

    /// The ascending series, checked the same three independent ways as the Hankel arm but at
    /// small `x`, where the series is the arm that works.
    #[test]
    fn ascending_series_matches_three_references() {
        let s = |nu: f64, x: f64| {
            series_j_nu::<Precision, f64, V>(V::splat(nu), V::splat(x), GenericMask::TRUTHY).extract::<0>()
        };

        // 1. `J_{1/2}` in closed form (exact, no reference error of its own).
        for &x in &[0.25f64, 1.0, 2.5, 4.0, 6.0] {
            let want = (2.0 / (core::f64::consts::PI * x)).sqrt() * x.sin();
            let got = s(0.5, x);
            assert!(
                env_rel(got, want, x) <= series_bound(x),
                "series J_1/2({x}): got {got}, want {want}, env-rel {:e}",
                env_rel(got, want, x)
            );
        }

        // 2. Whole orders against libm, through the shipped integer kernel.
        for &x in &[0.25f64, 1.0, 2.5, 4.0, 6.0] {
            for n in [0i32, 1, 2, 5] {
                let want = match n {
                    0 => V::splat(x).bessel_n_p::<Reference, J, 0>(),
                    1 => V::splat(x).bessel_n_p::<Reference, J, 1>(),
                    2 => V::splat(x).bessel_n_p::<Reference, J, 2>(),
                    _ => V::splat(x).bessel_n_p::<Reference, J, 5>(),
                }
                .extract::<0>();

                let got = s(n as f64, x);
                assert!(
                    env_rel(got, want, x) <= series_bound(x),
                    "series J_{n}({x}): got {got}, libm {want}, env-rel {:e}",
                    env_rel(got, want, x)
                );
            }
        }

        // 3. Thirds against mpmath at 60 digits, both signs.
        const ROWS: &[(f64, f64, f64)] = &[
            (0.3333333333333333, 0.25, 0.5533835954964775),
            (0.3333333333333333, 1.0, 0.730876402169448),
            (0.3333333333333333, 2.5, 0.19832093341860813),
            (0.3333333333333333, 4.0, -0.355427373454576),
            (0.3333333333333333, 6.0, -0.010674739474189045),
            (0.6666666666666666, 0.25, 0.2743443899886516),
            (0.6666666666666666, 1.0, 0.5979499736736285),
            (0.6666666666666666, 2.5, 0.3872124247708436),
            (0.6666666666666666, 4.0, -0.2325440850267039),
            (0.6666666666666666, 6.0, -0.16459872936403688),
            (-0.3333333333333333, 0.25, 1.4425215418779371),
            (-0.3333333333333333, 1.0, 0.6068875050465293),
            (-0.3333333333333333, 2.5, -0.3004751607573633),
            (-0.3333333333333333, 4.0, -0.33309316424600427),
            (-0.3333333333333333, 6.0, 0.2763443142062459),
            (-0.6666666666666666, 0.25, 1.4235474737365985),
            (-0.6666666666666666, 1.0, 0.18834029212239412),
            (-0.6666666666666666, 2.5, -0.47837308180342863),
            (-0.6666666666666666, 4.0, -0.1656584296075688),
            (-0.6666666666666666, 6.0, 0.32615507556979645),
        ];
        for &(nu, x, want) in ROWS {
            let got = s(nu, x);
            let e = env_rel(got, want, x);
            assert!(
                e <= series_bound(x),
                "series J_{nu}({x}): got {got}, want {want}, env-rel {e:e}, bound {:e}",
                series_bound(x)
            );
        }
    }

    /// The two arms do **not** overlap. This test pins that rather than pretending otherwise.
    ///
    /// The first attempt at this test compared them where both were "valid" and failed,
    /// correctly: there is no such `x`. The series dies to cancellation around 8 and the
    /// Hankel arm is not usable until 17, so `(6, 16)` is covered by **neither**. That is the
    /// measured gap from `notes/special/tools/model_fractional_arms.py`, and the reason a third
    /// arm is still owed.
    ///
    /// Locking it down matters because a later change that appears to extend either arm's
    /// range is far more likely to be a broken test than a real result.
    #[test]
    fn neither_arm_covers_the_measured_gap() {
        const NU: f64 = 1.0 / 3.0;
        // J_{1/3} at x = 12, mpmath at 60 digits (inside the gap).
        const X: f64 = 12.0;
        const TRUE: f64 = -0.0703213677045818;

        let series = series_j_nu::<Precision, f64, V>(V::splat(NU), V::splat(X), GenericMask::TRUTHY).extract::<0>();
        let hankel = j_nu(NU, X);

        // Both are far outside their ranges here. "Far" means hundreds of eps or worse, so a
        // gate at 1e-13 (roughly 450 eps) is generous to both and still fails loudly if
        // either one ever genuinely reaches into the gap.
        let s_err = env_rel(series, TRUE, X);
        let h_err = env_rel(hankel, TRUE, X);

        assert!(
            s_err > 1e-13 && h_err > 1e-13,
            "the gap at x = {X} appears to have closed: series {s_err:e}, hankel {h_err:e}. \
             If that is real, the third arm may no longer be needed - re-measure before \
             deleting this test."
        );
    }

    /// `Y_nu` from the Hankel arm at large `x`, which now returns both kinds from one `P`/`Q`
    /// pass. Same three references as its `J` half.
    #[test]
    fn hankel_y_matches_three_references() {
        let y_nu = |nu: f64, x: f64| {
            hankel_jy_nu::<Precision, f64, V, N64>(V::splat(nu), V::splat(x), V::ZERO)
                .1
                .extract::<0>()
        };

        // 1. `Y_{1/2}(x) = -sqrt(2/pi x) cos x`, exact.
        for &x in &[20.0f64, 30.0, 55.0, 100.0, 400.0] {
            let want = -(2.0 / (core::f64::consts::PI * x)).sqrt() * x.cos();
            let got = y_nu(0.5, x);
            assert!(env_rel(got, want, x) <= 4e-16, "Y_1/2({x}): got {got}, want {want}");
        }

        // 2. Whole orders against libm.
        for &x in &[20.0f64, 30.0, 55.0, 100.0] {
            for n in [0i32, 1, 2, 5] {
                let want = match n {
                    0 => V::splat(x).bessel_n_p::<Reference, Y, 0>(),
                    1 => V::splat(x).bessel_n_p::<Reference, Y, 1>(),
                    2 => V::splat(x).bessel_n_p::<Reference, Y, 2>(),
                    _ => V::splat(x).bessel_n_p::<Reference, Y, 5>(),
                }
                .extract::<0>();
                let got = y_nu(n as f64, x);
                assert!(env_rel(got, want, x) <= 8e-16, "Y_{n}({x}): got {got}, libm {want}");
            }
        }

        // 3. Thirds against mpmath at 60 digits, both signs.
        const ROWS: &[(f64, f64, f64)] = &[
            (0.3333333333333333, 20.0, -0.028777707635715168),
            (0.3333333333333333, 30.0, -0.05864577231670508),
            (0.3333333333333333, 55.0, -0.030007358986895383),
            (0.3333333333333333, 100.0, -0.0769005049621365),
            (0.6666666666666666, 20.0, -0.11182254014899551),
            (0.6666666666666666, 30.0, 0.015078692908077526),
            (0.6666666666666666, 55.0, 0.025354902147023572),
            (0.6666666666666666, 100.0, -0.056057339204074165),
            (-0.3333333333333333, 20.0, 0.1380840810783704),
            (-0.3333333333333333, 30.0, -0.14479917584764257),
            (-0.3333333333333333, 55.0, -0.1044779681586487),
            (-0.3333333333333333, 100.0, -0.05687169089449366),
        ];
        for &(nu, x, want) in ROWS {
            let got = y_nu(nu, x);
            assert!(
                env_rel(got, want, x) <= 8e-16,
                "Y_{nu}({x}): got {got}, want {want}, env-rel {:e}",
                env_rel(got, want, x)
            );
        }
    }

    /// Steed across the gap, `J` and `Y` together, checked the same three independent ways.
    ///
    /// This is the arm that bridges `(6, 16)`, so the grid sits squarely inside it, exactly
    /// where `neither_arm_covers_the_measured_gap` asserts the other two fail.
    #[test]
    fn steed_matches_three_references_across_the_gap() {
        let st = |nu: f64, x: f64| {
            let (j, y) = steed_full::<V>(V::splat(nu), V::splat(x), GenericMask::TRUTHY);
            (j.extract::<0>(), y.extract::<0>())
        };

        // 1. `nu = 1/2` in closed form, for both kinds (exact references).
        for &x in &[6.0f64, 8.0, 11.0, 14.0, 16.0] {
            let amp = (2.0 / (core::f64::consts::PI * x)).sqrt();
            let (j, y) = st(0.5, x);
            assert!(
                env_rel(j, amp * x.sin(), x) <= 2e-15,
                "steed J_1/2({x}): got {j}, want {}",
                amp * x.sin()
            );
            assert!(
                env_rel(y, -amp * x.cos(), x) <= 2e-15,
                "steed Y_1/2({x}): got {y}, want {}",
                -amp * x.cos()
            );
        }

        // 2. Whole orders against libm, both kinds.
        for &x in &[6.0f64, 8.0, 11.0, 14.0, 16.0] {
            for n in [0i32, 1, 2] {
                let (wj, wy) = match n {
                    0 => (
                        V::splat(x).bessel_n_p::<Reference, J, 0>(),
                        V::splat(x).bessel_n_p::<Reference, Y, 0>(),
                    ),
                    1 => (
                        V::splat(x).bessel_n_p::<Reference, J, 1>(),
                        V::splat(x).bessel_n_p::<Reference, Y, 1>(),
                    ),
                    _ => (
                        V::splat(x).bessel_n_p::<Reference, J, 2>(),
                        V::splat(x).bessel_n_p::<Reference, Y, 2>(),
                    ),
                };
                let (j, y) = st(n as f64, x);
                assert!(
                    env_rel(j, wj.extract::<0>(), x) <= 4e-15,
                    "steed J_{n}({x}): got {j}, libm {}",
                    wj.extract::<0>()
                );
                assert!(
                    env_rel(y, wy.extract::<0>(), x) <= 4e-15,
                    "steed Y_{n}({x}): got {y}, libm {}",
                    wy.extract::<0>()
                );
            }
        }

        // 3. Thirds against mpmath at 60 digits, both signs and both kinds.
        const ROWS: &[(f64, f64, f64, f64)] = &[
            (0.3333333333333333, 6.0, -0.010674739474189045, -0.3252579921009493),
            (0.3333333333333333, 8.0, 0.25977616110834967, 0.10958779463360625),
            (0.3333333333333333, 10.0, -0.18614516704869577, 0.1702011178826876),
            (0.3333333333333333, 12.0, -0.0703213677045818, -0.2192743582206475),
            (0.3333333333333333, 14.0, 0.21168092934398272, 0.02545667339212697),
            (0.3333333333333333, 16.0, -0.10416268410664775, 0.17008275621757885),
            (0.6666666666666666, 6.0, -0.16459872936403688, -0.28158032064897237),
            (0.6666666666666666, 8.0, 0.2807877136273063, -0.02922717844123111),
            (0.6666666666666666, 10.0, -0.08014960330431577, 0.23937232657540727),
            (0.6666666666666666, 12.0, -0.1684756369795518, -0.15717219617399347),
            (0.6666666666666666, 14.0, 0.1971137944823384, -0.0814947648718344),
            (0.6666666666666666, 16.0, -0.007241052782211041, 0.19937736879861026),
            (-0.3333333333333333, 6.0, 0.2763443142062459, -0.17187359161390292),
            (-0.3333333333333333, 8.0, 0.03498226645675983, 0.279766652134233),
            (-0.3333333333333333, 10.0, -0.24047107536326526, -0.07610588451452473),
            (-0.3333333333333333, 12.0, 0.15473648076531898, -0.17053726997135818),
            (-0.3333333333333333, 14.0, 0.08379433881856603, 0.19604939900465135),
            (-0.3333333333333333, 16.0, -0.19937732968342284, -0.005166152453941126),
        ];
        for &(nu, x, wj, wy) in ROWS {
            let (j, y) = st(nu, x);
            assert!(
                env_rel(j, wj, x) <= 4e-15,
                "steed J_{nu}({x}): got {j}, want {wj}, env-rel {:e}",
                env_rel(j, wj, x)
            );
            assert!(
                env_rel(y, wy, x) <= 4e-15,
                "steed Y_{nu}({x}): got {y}, want {wy}, env-rel {:e}",
                env_rel(y, wy, x)
            );
        }
    }

    /// The sign of `J_nu` comes from CF1's denominator chain, not from the square root that
    /// produces its magnitude. `J_{1/3}` changes sign between these two points, so a lost or
    /// stuck sign chain shows up here and nowhere in a magnitude-only check.
    #[test]
    fn steed_recovers_the_sign_from_cf1() {
        let sign_at = |x: f64| {
            steed_full::<V>(V::splat(1.0 / 3.0), V::splat(x), GenericMask::TRUTHY)
                .0
                .extract::<0>()
        };

        // J_{1/3} is negative at 6 and 12, positive at 8 and 14, straddling two of its roots.
        assert!(sign_at(6.0) < 0.0, "J_1/3(6) should be negative, got {}", sign_at(6.0));
        assert!(sign_at(8.0) > 0.0, "J_1/3(8) should be positive, got {}", sign_at(8.0));
        assert!(
            sign_at(12.0) < 0.0,
            "J_1/3(12) should be negative, got {}",
            sign_at(12.0)
        );
        assert!(
            sign_at(14.0) > 0.0,
            "J_1/3(14) should be positive, got {}",
            sign_at(14.0)
        );
    }

    /// Temme's series at small `x`, both returned orders, against mpmath at 60 digits.
    ///
    /// This is the arm that covers where Steed gets expensive and then fails: at `x = 0.01`
    /// Steed's CF2 needs 5392 iterations and is 261000 eps wrong, and Temme is 12 terms and a
    /// couple of eps.
    #[test]
    fn temme_matches_a_high_precision_reference() {
        use crate::tables::lgamma1p::LGAMMA1P_F64;

        let ty = |nu: f64, x: f64| {
            let (a, b) =
                temme_y_nu::<Precision, f64, V, 25, 25>(V::splat(nu), V::splat(x), GenericMask::TRUTHY, &LGAMMA1P_F64);
            (a.extract::<0>(), b.extract::<0>())
        };

        // (nu, x, Y_nu(x), Y_{nu+1}(x))
        const ROWS: &[(f64, f64, f64, f64)] = &[
            (0.0, 0.01, -3.005455637083646, -63.67859628206066),
            (0.0, 0.1, -1.5342386513503667, -6.4589510947020266),
            (0.0, 0.5, -0.44451873350670656, -1.471472392670243),
            (0.0, 1.0, 0.08825696421567696, -0.7812128213002887),
            (0.0, 2.0, 0.5103756726497451, -0.10703243154093754),
            (0.3333333333333333, 0.01, -4.876068267087222, -332.47855994042806),
            (0.3333333333333333, 0.1, -2.0682565649661906, -15.537743860478967),
            (0.3333333333333333, 0.5, -0.8406278260433777, -2.0532379702305654),
            (0.3333333333333333, 1.0, -0.2788016412759921, -0.9850592357315765),
            (0.3333333333333333, 2.0, 0.3431999662603444, -0.3080031737866188),
            (-0.3333333333333333, 0.01, -2.2722011190011107, -14.758583491338225),
            (-0.3333333333333333, 0.1, -0.6775147311988818, -3.23872328913618),
            (-0.3333333333333333, 0.5, 0.16237467777288853, -1.1316060101031433),
            (-0.3333333333333333, 1.0, 0.4935567106673179, -0.562703214974633),
            (-0.3333333333333333, 2.0, 0.5551971179944987, 0.1198934536190353),
            (0.5, 0.01, -7.97844666907276, -797.9244540335553),
            (0.5, 0.1, -2.5105273689585093, -25.357166629911095),
            (0.5, 0.5, -0.9902458802434049, -2.521465550421338),
            (0.5, 1.0, -0.4310988680183761, -1.1024955751601793),
            (0.5, 2.0, 0.23478571040624846, -0.3956232813587035),
            (0.25, 0.01, -4.046477065077802, -217.02001233018106),
            (0.25, 0.1, -1.9117683212071752, -12.303757510699864),
            (0.25, 0.5, -0.756843545694496, -1.8715902300683556),
            (0.25, 1.0, -0.19442175367716438, -0.9319659251969881),
            (0.25, 2.0, 0.39273839961538504, -0.2609445010948933),
        ];

        for &(nu, x, w0, w1) in ROWS {
            let (y0, y1) = ty(nu, x);
            // `Y` diverges as `x -> 0`, so plain relative error is the right contract here:
            // there is no zero nearby to make it meaningless.
            assert!(rel(y0, w0) <= 8e-15, "Y_{nu}({x}): got {y0}, want {w0}");
            assert!(rel(y1, w1) <= 8e-15, "Y_{nu}+1({x}): got {y1}, want {w1}");
        }
    }

    /// The four `0/0` limits at `nu = 0` must be the finite ones, not NaN.
    ///
    /// `d`, `e`, `g1` and `vspv` are each `0/0` there. `d` is [`sinhc`] and needs no guard.
    /// The other three are selects, and a select whose live arm is the NaN gives NaN.
    #[test]
    fn temme_is_finite_at_and_around_zero_order() {
        use crate::tables::lgamma1p::LGAMMA1P_F64;

        // Each order gets its OWN reference. `Y` is not flat near zero. It varies linearly in
        // `nu` with slope about 13.6 at `x = 1`, so `Y_{1e-8}` differs from `Y_0` in the
        // seventh digit. An earlier version of this test compared everything against the
        // `nu = 0` value and failed the kernel for being correct.
        const ROWS: &[(f64, f64, f64)] = &[
            (0.0, 0.08825696421567696, -0.7812128213002887),
            (1e-300, 0.08825696421567696, -0.7812128213002887),
            (1e-30, 0.08825696421567696, -0.7812128213002887),
            (1e-17, 0.08825696421567694, -0.7812128213002887),
            (1e-16, 0.08825696421567684, -0.7812128213002888),
            (1e-08, 0.08825695219597983, -0.7812128273300175),
            (-1e-08, 0.08825697623537414, -0.78121281527056),
        ];

        for &(nu, w0, w1) in ROWS {
            let (a, b) = temme_y_nu::<Precision, f64, V, 25, 25>(
                V::splat(nu),
                V::splat(1.0),
                GenericMask::TRUTHY,
                &LGAMMA1P_F64,
            );
            let (y0, y1) = (a.extract::<0>(), b.extract::<0>());

            assert!(y0.is_finite() && y1.is_finite(), "nu = {nu} gave ({y0}, {y1})");
            assert!(rel(y0, w0) <= 8e-15, "Y_{nu}(1): got {y0}, want {w0}");
            assert!(rel(y1, w1) <= 8e-15, "Y_{nu}+1(1): got {y1}, want {w1}");
        }
    }

    /// End to end: `bessel_jy_real` across **all three regions**, both kinds, positive and
    /// negative order, whole and fractional, small and large `|nu|`.
    ///
    /// This is the test that says the pieces compose. Each arm has its own check above, but
    /// only this one exercises the region select, the order reduction and the negative-order
    /// rotation together.
    #[test]
    fn bessel_jy_real_covers_the_whole_axis() {
        use crate::tables::lgamma1p::LGAMMA1P_F64;

        let jy = |nu: f64, x: f64| {
            let (j, y) = bessel_jy_real::<Precision, f64, V, N64, 25, 25, 17, 1>(
                V::splat(nu),
                V::splat(x),
                V::ZERO,
                &LGAMMA1P_F64,
            );
            (j.extract::<0>(), y.extract::<0>())
        };

        // (nu, x, J, Y) from mpmath at 50 digits. `x` spans every region boundary: 0.05 and
        // 0.5 are Temme, 1.5 straddles the 2 cut, 4 and 9 are Steed, 25 and 80 are Hankel.
        const ROWS: &[(f64, f64, f64, f64)] = &[
            (0.3333333333333333, 0.05, 0.32729164001955063, -2.724609099171694),
            (0.3333333333333333, 0.5, 0.672830829497946, -0.8406278260433777),
            (0.3333333333333333, 1.5, 0.6371326370648923, 0.09661008776662783),
            (0.3333333333333333, 4.0, -0.355427373454576, 0.17941676634394849),
            (0.3333333333333333, 9.0, 0.04514673992769786, 0.2619881509685795),
            (0.3333333333333333, 25.0, 0.020097162141383115, -0.1582974186494417),
            (0.3333333333333333, 80.0, -0.08819978440003455, -0.013358849535984083),
            (2.25, 0.05, 9.746930842421557e-05, -1451.8894167512065),
            (2.25, 0.5, 0.01700515517725076, -8.601107604647282),
            (2.25, 1.5, 0.17207040140276186, -1.0952365333165308),
            (2.25, 4.0, 0.4150977707888228, 0.11743330302206845),
            (2.25, 9.0, 0.06283886940664354, -0.2625685257863794),
            (2.25, 25.0, -0.055753132743452054, 0.14984908706204303),
            (2.25, 80.0, 0.08491093735158971, 0.027402058043983494),
            (-0.6666666666666666, 0.05, 4.357750582173945, 2.6252688861482163),
            (-0.6666666666666666, 0.5, 0.7683441764822306, 0.9324008688393952),
            (-0.6666666666666666, 1.5, -0.1623262567895261, 0.641516073554232),
            (-0.6666666666666666, 4.0, -0.16565842960756885, -0.36416171910470596),
            (-0.6666666666666666, 9.0, -0.2630406542532977, 0.04035719142945025),
            (-0.6666666666666666, 25.0, 0.1581810722120303, 0.021154005791818097),
            (-0.6666666666666666, 80.0, 0.013542732047879152, -0.08817291207412467),
            (5.5, 1.5, 0.0006543566107377901, -92.08800019920933),
            (5.5, 4.0, 0.08260584990805442, -1.0576777947628146),
            (5.5, 9.0, 0.08438779749107019, 0.2848318597461538),
            (5.5, 25.0, -0.14408915895213564, -0.07304429387418315),
            (5.5, 80.0, -0.006865341718278464, 0.08904676635224425),
            (-2.25, 1.5, 0.896121327384749, -0.6527770320379809),
            (-2.25, 4.0, 0.2104805636761565, 0.37655633348423506),
            (-2.25, 9.0, 0.2300977757892373, -0.1412301944301702),
            (-2.25, 25.0, -0.14538272385147266, 0.06653588738089529),
            (-2.25, 80.0, 0.040664918536847075, 0.0794172806595833),
        ];

        let mut worst = 0.0f64;
        for &(nu, x, wj, wy) in ROWS {
            let (j, y) = jy(nu, x);
            let ej = env_rel(j, wj, x);
            let ey = env_rel(y, wy, x);
            assert!(ej <= 4e-14, "J_{nu}({x}): got {j}, want {wj}, env-rel {ej:e}");
            assert!(ey <= 4e-14, "Y_{nu}({x}): got {y}, want {wy}, env-rel {ey:e}");
            worst = worst.max(ej).max(ey);
        }
        assert!(worst < 4e-14, "worst {worst:e}");
    }

    /// A single packet spanning every region at once, against the same lanes computed alone.
    ///
    /// The region select runs all three arms whenever any lane needs one, so this is the case
    /// the `needed` masks exist for, and the one where a mask threaded to the wrong arm shows
    /// up as a wrong answer rather than merely as wasted work.
    #[test]
    fn a_packet_spanning_every_region_agrees_with_single_lanes() {
        use crate::tables::lgamma1p::LGAMMA1P_F64;
        type W = Vector<<X86V2 as Simd>::f64x2>;

        // Lane 0 in Temme's region, lane 1 past the Hankel gate.
        let nu = W::splat(1.0 / 3.0);
        let x = W::splat(0.5).insert::<1>(25.0);

        let (j, y) = bessel_jy_real::<Precision, f64, W, N64, 25, 25, 17, 1>(nu, x, W::ZERO, &LGAMMA1P_F64);

        for (lane, xv) in [(0usize, 0.5f64), (1, 25.0)] {
            let (j1, y1) = bessel_jy_real::<Precision, f64, V, N64, 25, 25, 17, 1>(
                V::splat(1.0 / 3.0),
                V::splat(xv),
                V::ZERO,
                &LGAMMA1P_F64,
            );
            assert_eq!(
                j.extractv(lane).to_bits(),
                j1.extract::<0>().to_bits(),
                "lane {lane} (x = {xv}) J differs from the same lane alone"
            );
            assert_eq!(
                y.extractv(lane).to_bits(),
                y1.extract::<0>().to_bits(),
                "lane {lane} (x = {xv}) Y differs from the same lane alone"
            );
        }
    }

    /// Out-of-band lanes must not reach the in-band answers, and an all-masked call must not
    /// run at all.
    ///
    /// A packet spans regions, so Steed runs whenever _any_ lane is in its band. A lane at
    /// `x = 0.01` heading for a different arm would otherwise hold CF2 open for about 5400
    /// iterations for a value that is discarded.
    ///
    /// **What this cannot check is the saving.** A lane that is masked but still iterating
    /// produces the same answers (it is frozen once converged either way), so a missing mask
    /// costs time and changes nothing observable. Catching _that_ needs instrumentation or a
    /// benchmark, not an assertion. What is checked here is the part that can be: masked lanes
    /// do not corrupt, and a fully masked call short-circuits.
    #[test]
    fn out_of_band_lanes_do_not_reach_the_answer() {
        type WS = Vector<<X86V2 as Simd>::f64x2>;

        let nu = WS::splat(1.0 / 3.0);
        // Lane 0 is in the Steed band, lane 1 far below it and masked off.
        let x = WS::splat(10.0).insert::<1>(0.01);
        let needed = x.cmp_gt(WS::splat(2.0));

        let (j, y) = steed_full::<WS>(nu, x, needed);

        // The in-band lane must match the value it has on its own, bit for bit.
        let (j1, y1) = steed_full::<V>(V::splat(1.0 / 3.0), V::splat(10.0), GenericMask::TRUTHY);
        assert_eq!(
            j.extractv(0).to_bits(),
            j1.extract::<0>().to_bits(),
            "an out-of-band lane changed the in-band J"
        );
        assert_eq!(
            y.extractv(0).to_bits(),
            y1.extract::<0>().to_bits(),
            "an out-of-band lane changed the in-band Y"
        );

        // Nothing in band at all: the whole thing is skipped.
        let (jz, yz) = steed_full::<WS>(nu, WS::splat(0.01), GenericMask::FALSY);
        assert_eq!(jz.extractv(0), 0.0, "a fully masked call must not compute");
        assert_eq!(yz.extractv(0), 0.0, "a fully masked call must not compute");
    }

    /// The gate must be conservative, never optimistic: below what it returns, the divergent
    /// series has not yet reached full precision and no term count fixes that.
    #[test]
    fn the_gate_matches_the_measured_thresholds() {
        // <17, 1> is the binary64 floor from the module's table.
        let g = |nu: f64| hankel_usable_from::<f64, V, 17, 1>(V::splat(nu)).extract::<0>();

        // Measured: 16.5 to 17.0 for orders 0 through 5, rising after.
        for &nu in &[0.0f64, 1.0 / 3.0, 1.0, 3.0, 5.0] {
            assert!(
                g(nu) >= 17.0,
                "gate at nu = {nu} is {} , below the measured 17.0",
                g(nu)
            );
        }
        // Measured 18.0 at nu = 8 and 25.0 at nu = 12.
        assert!(g(8.0) >= 18.0, "gate at nu = 8 is {}", g(8.0));
        assert!(g(12.0) >= 25.0, "gate at nu = 12 is {}", g(12.0));

        // Symmetric in the sign of the order.
        assert_eq!(g(-3.0), g(3.0), "the gate must not depend on the sign of nu");
    }
}
