//! Loop drivers for series and continued fractions, on top of the helpers in the parent
//! module.
//!
//! # Why three drivers and not one
//!
//! Iterative kernels in this workspace stop in one of three ways, and the difference is
//! structural rather than stylistic. Choosing the wrong one is a correctness bug in one
//! direction and pure wasted work in the other.
//!
//! | Discipline | Stops when | Converged lanes | Driver |
//! |---|---|---|---|
//! | Additive | the terms stop mattering | freeze themselves | [`sum_f`], [`sum_ratio`] |
//! | Counted | a predicted index is reached | n/a (no test) | [`sum_counted`] |
//! | Multiplicative | the factor reaches one | **depends**, see below | [`prod_f`], [`lentz`] |
//!
//! # Whether to freeze a converged lane
//!
//! This is the one thing here that is easy to get wrong, and the obvious rule:
//! "additive accumulators are safe, multiplicative ones must freeze", is **not** the rule.
//! [`prod_f`] and [`lentz`] are both multiplicative and they differ.
//!
//! A packet exits when its _slowest_ lane converges, so every other lane keeps iterating past
//! its own convergence. The question is only what those extra steps are made of.
//!
//! **An additive accumulator never needs freezing.** [`sum_ratio`] adds a term that has gone
//! to zero, and its rounding went with it. Extra steps are exact no-ops, and a per-lane
//! `select` in that inner loop would be pure cost.
//!
//! **A multiplicative accumulator depends on where its tolerance sits.** The extra factors are
//! within `$\tau$` of one, so continuing costs `$m\,\varepsilon$` of accumulated rounding over
//! `$m$` steps, while freezing discards their true contribution, up to `$m\,\tau$`.
//!
//! * [`lentz`] is normally run at `$\tau \approx \varepsilon$`, where those two are equal and
//!   the computed `$\Delta$` is pure rounding noise carrying no signal. It freezes.
//! * [`prod_f`] is normally run far above the noise floor:
//!   [`PrecisionPolicy::tolerance`](crate::math::policy::PrecisionPolicy::tolerance) is 100
//!   times `EPSILON` even at `Average`, so its extra factors still carry real signal, and
//!   `$m\,\tau \gg m\,\varepsilon$`. It does not freeze.
//!
//! Since `$\tau \ge \varepsilon$` always, **not freezing is never the worse option**. Freezing
//! only wins by saving the rounding when `$\tau$` is already at the floor.
//!
//! [`sum_f`] is the same Additive discipline as
//! [`sum_ratio`], reached differently: it evaluates `$t_k$` from the index, where
//! [`sum_ratio`] advances `$t_k \to t_{k+1}$` by a ratio. Prefer [`sum_f`] when
//! the term has a closed form and [`sum_ratio`] when it does not, or when recomputing it would
//! cost more than one multiply.
//!
//! # Tolerances come from the caller
//!
//! None of these derive a tolerance. The ladder already exists as
//! [`PrecisionPolicy::tolerance`](crate::math::policy::PrecisionPolicy::tolerance), which
//! gives a multiple of `EPSILON` and therefore already means the same thing in binary32 and
//! binary64. Callers pass the result in, exactly as [`sum_f`] requires.
//!
//! # `active` says which lanes the loop is actually for
//!
//! **A packet spans regions.** A kernel that splits its domain runs _every_ region whose
//! predicate any lane satisfies, so a lane heading for a different arm is present in this
//! one's loop whether it wants to be or not, and if it is allowed to vote on convergence it
//! sets the trip count for the whole packet.
//!
//! That is not a small effect. A continued fraction whose trip count scales with the argument
//! can want twenty iterations in its own band and **thousands** outside it, so one stray lane
//! can cost two orders of magnitude for a value that is discarded the moment the region select
//! runs.
//!
//! So every driver here takes `active`, and lanes outside it are treated as **already
//! converged**: they cannot extend the loop and cannot keep it from finishing. Pass
//! `GenericMask::TRUTHY` when the whole packet is in range.
//!
//! Two consequences to know:
//!
//! * **Inactive lanes still accumulate**, because freezing them would cost a `select` per
//!   iteration for a value the caller throws away. Their contents are **unspecified**: read
//!   only the lanes you asked for.
//! * **`active.none()` short-circuits**, so a region no lane needs costs one reduction rather
//!   than a loop. That is the same guard as the `if r1.none()` region skips already used in
//!   `thermite-special`'s Bessel kernels. This is the per-lane half of the same idea.
//!
//! [`sum_counted`] takes no `active`, deliberately: it has no convergence test to mask, and
//! its trip count is a const generic, so a mask could not shorten it.

use crate::mask::GenericMask;

use super::super::*;

/// How often the all-lanes-converged reduction runs, in iterations.
///
/// A horizontal reduction is not free, and checking it every pass costs more than the handful
/// of extra iterations that amortizing it can add. Four is the stride already shipped in
/// `thermite-special`'s `expint_fraction`, which is the most tuned loop of this shape in the
/// workspace.
///
/// Overshooting is harmless in all three drivers that use it: [`lentz`] has already frozen
/// the converged lanes, an additive sum is adding terms that no longer register, and
/// [`prod_f`] is multiplying by factors within `$\tau$` of one, which it would have kept
/// doing anyway until its slowest lane finished.
const CHECK_STRIDE: usize = 4;

/// Sums a series whose terms are produced by a running ratio, `$t_{k+1} = \mathrm{advance}(k, t_k)$`.
///
/// Returns `Ok(sum)` once every lane has converged, or `Err(sum)` with the best partial sum if
/// `P::POLICY.max_iterations` runs out first, the same convention as
/// [`sum_f`].
///
/// `advance` receives the index `k` of the term it is being asked to produce (so the first
/// call receives `1`) and the previous term, and returns the next one. `first` is `$t_0$`, and
/// is included in the sum.
///
/// If `P::POLICY.use_compensation` is set, modified Kahan summation is used, matching
/// [`sum_f`].
///
/// # The convergence test is relative to the largest term, not to the sum
///
/// This is the one design decision here that is not obvious, and getting it wrong produces a
/// driver that works on every test case anyone thinks to write and fails on the interesting
/// ones.
///
/// The obvious test is `$\lvert t_k\rvert \le \tau\,\lvert S\rvert$`, relative to the running
/// sum. It breaks for any series whose sum passes near zero, which for the oscillatory
/// functions this exists to serve is not an edge case but the normal situation, since they
/// have infinitely many zeros. Near one, `$\lvert S \rvert$` collapses and the test can never
/// be satisfied, or is satisfied immediately by accident.
///
/// So the scale is the largest `$\lvert t_k \rvert$` seen so far. That quantity is also
/// exactly the one that governs how much accuracy the summation can possibly deliver: a
/// cancelling series loses about `$\log_2(\max_k \lvert t_k\rvert / \lvert S\rvert)$` bits, so
/// testing against it says "we have extracted everything this arithmetic can give" rather than
/// "the answer looks small".
#[inline(always)]
pub fn sum_ratio<V: FloatVector, P: Policy, F>(tolerance: V, active: V::Mask, first: V, mut advance: F) -> Result<V, V>
where
    F: FnMut(i64, V) -> V,
{
    let mut term = first;
    let mut k = 0i64;

    // `first` seeds the accumulator rather than being counted, so `max_iterations` bounds the
    // number of _advances_ (the expensive part), not the terms including the free one.
    sum_core::<V, P, _>(tolerance, active, first, P::POLICY.max_iterations, move || {
        k += 1;
        term = advance(k, term);
        term
    })
}

/// Sums `f` over the half-open range `[start, end)`.
///
/// Returns `Ok(sum)` once every lane has converged, or `Err(sum)` with the best partial sum if
/// the range or `P::POLICY.max_iterations` runs out first.
///
/// This is the Additive discipline reached by index rather than by ratio. See
/// [`sum_ratio`] for the other spelling and the [module documentation](self) for which to
/// pick.
///
/// If `P::POLICY.use_compensation` is set, modified Kahan summation is used.
///
/// # `tolerance` is RELATIVE, and was not always
///
/// A lane is converged once `$\lvert t_k \rvert \le \tau \cdot \max_j \lvert t_j \rvert$`.
/// The earlier version of this function compared `$\lvert t_k \rvert$` against `$\tau$`
/// **absolutely**, which did not match how anybody could supply `$\tau$`: the natural source
/// is [`PrecisionPolicy::tolerance`](crate::math::policy::PrecisionPolicy::tolerance), which
/// is a _multiple of `EPSILON`_ and therefore already a relative quantity. Used absolutely it
/// meant "stop below 2e-14" regardless of whether the sum was `$10^{10}$` or `$10^{-20}$`:
/// unreachable in one case and instantly true in the other.
///
/// Scaling by the largest term rather than by the running sum is deliberate. The reasoning is
/// on [`sum_ratio`]. It matters for any series whose sum passes near zero.
#[inline(always)]
pub fn sum_f<V: FloatVector, P: Policy, F>(
    tolerance: V,
    active: V::Mask,
    start: i64,
    end: i64,
    mut f: F,
) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    let span = end.saturating_sub(start).max(0) as usize;
    let mut n = start;

    sum_core::<V, P, _>(
        tolerance,
        active,
        V::ZERO,
        span.min(P::POLICY.max_iterations),
        move || {
            let v = f(n);
            n += 1;
            v
        },
    )
}

/// Multiplies `f` over the half-open range `[start, end)`.
///
/// Returns `Ok(prod)` once every lane has converged, or `Err(prod)` with the best partial
/// product if the range or `P::POLICY.max_iterations` runs out first.
///
/// # `tolerance` applies to the factor, not to the change in the product
///
/// A lane is converged once `$\lvert f(n) - 1 \rvert \le \tau$`. That is the textbook
/// criterion for an infinite product: `$\prod (1 + a_n)$` converges exactly when
/// `$\sum a_n$` does, so the factors approaching one _is_ convergence, and is the same
/// quantity that `$\lvert \Delta \rvert \le \tau \lvert \Pi \rvert$` would test, reached
/// without forming `$\Delta$` at all, since `$\Pi_{new} - \Pi = \Pi\,(f - 1)$`.
///
/// The earlier version compared `new_prod - prod` against `tolerance` **absolutely**, which
/// made convergence depend on how large the product happened to be rather than on whether it
/// had stopped moving. It also returned the product from _before_ the converging factor,
/// discarding a factor it had already paid to compute.
///
/// # Converged lanes are deliberately NOT frozen
///
/// [`lentz`] freezes its converged lanes and this does not, which looks inconsistent and is
/// not. The question is whether the per-step quantity past convergence is **noise or signal**.
///
/// [`lentz`] runs at a tolerance on the order of `EPSILON`, so once
/// `$\lvert \Delta - 1 \rvert$` is below it the computed `$\Delta$` is rounding noise with no
/// remaining signal, and multiplying by it is a random walk away from the answer.
///
/// A product's tolerance is normally far above the noise floor:
/// [`PrecisionPolicy::tolerance`](crate::math::policy::PrecisionPolicy::tolerance) is 100
/// times `EPSILON` even at `Average`. The factors between one lane converging and the last
/// lane converging therefore still carry real signal, and the true product includes them.
/// Freezing would truncate that, costing up to `$m\,\tau$` over `$m$` further iterations,
/// where continuing costs only the `$m\,\varepsilon$` of accumulated rounding. Since
/// `$\tau \ge \varepsilon$` always, **continuing is never the worse option**.
#[inline(always)]
pub fn prod_f<V: FloatVector, P: Policy, F>(
    tolerance: V,
    active: V::Mask,
    start: i64,
    end: i64,
    mut f: F,
) -> Result<V, V>
where
    F: FnMut(i64) -> V,
{
    if active.none() {
        return Ok(V::ONE);
    }

    let span = end.saturating_sub(start).max(0) as usize;
    let count = span.min(P::POLICY.max_iterations);

    let mut prod = V::ONE;
    let mut n = start;

    let mut converged = false;
    let mut k = 1usize;

    while k <= count {
        V::_loop_hint();

        let factor = f(n);
        n += 1;

        // Multiplied in BEFORE the test, so the factor that trips convergence is kept.
        prod *= factor;

        // Amortized over `CHECK_STRIDE`, with the final iteration always tested so a product
        // shorter than one stride can still report convergence.
        // Inactive lanes count as converged (see `sum_core`).
        if (k.is_multiple_of(CHECK_STRIDE) || k == count) && ((factor - V::ONE).abs().cmp_le(tolerance) | !active).all()
        {
            converged = true;
            break;
        }

        k += 1;
    }

    match converged {
        true => Ok(prod),
        false => Err(prod),
    }
}

/// The one Additive loop, behind both [`sum_f`] and [`sum_ratio`].
///
/// `next` yields successive terms and owns whatever state that takes. `first` seeds the
/// accumulator and the scale, and `count` is the number of calls to `next`.
///
/// Kept private and shared so the two public spellings cannot drift apart: they differ only
/// in how a term is produced, never in how one is accumulated or tested.
#[inline(always)]
fn sum_core<V: FloatVector, P: Policy, F>(
    tolerance: V,
    active: V::Mask,
    first: V,
    count: usize,
    mut next: F,
) -> Result<V, V>
where
    F: FnMut() -> V,
{
    if active.none() {
        return Ok(first);
    }

    let mut sum = first;
    let mut c = V::ZERO; // Kahan compensation
    let mut scale = first.abs();

    let mut converged = false;
    let mut k = 1usize;

    while k <= count {
        V::_loop_hint();

        let mut delta = next();
        let abs_delta = delta.abs();
        scale = scale.max(abs_delta);

        let t = sum + delta;

        if const { P::POLICY.use_compensation } {
            // The larger magnitude keeps its low-order bits.
            sum.abs().cmp_lt(abs_delta).swap(&mut sum, &mut delta);
            c += (sum - t) + delta;
        }

        // Accumulated BEFORE the test, so the term that trips convergence is kept. The
        // previous version of `sum_f` computed that term and then discarded it, which threw
        // away work already paid for and biased every result low by up to one tolerance.
        sum = t;

        // The reduction is amortized over `CHECK_STRIDE`, since a horizontal `all()` costs
        // more than the few extra terms overshooting can add, and those terms are
        // additively negligible by construction, so they cannot move the answer. The final
        // iteration is always tested, or a series shorter than one stride could converge and
        // still be reported as a failure.
        // An inactive lane is treated as already converged, so it can neither extend the loop
        // nor keep it from finishing. Its accumulator is still updated: freezing it would
        // cost a select for a value the caller discards, so its contents are unspecified.
        if (k.is_multiple_of(CHECK_STRIDE) || k == count) && (abs_delta.cmp_le(tolerance * scale) | !active).all() {
            converged = true;
            break;
        }

        k += 1;
    }

    if const { P::POLICY.use_compensation } {
        sum += c;
    }

    match converged {
        true => Ok(sum),
        false => Err(sum),
    }
}

/// Sums exactly `N` terms of a ratio-advanced series, with no convergence test at all.
///
/// For **divergent** asymptotic expansions, where there is nothing to converge to and the
/// stopping point is an index computed up front rather than a tolerance met along the way.
/// The Hankel expansion of `$J_\nu$` is the motivating case: its terms shrink until
/// `$k \approx 2x$` and grow forever after, so the useful truncation is a closed form in
/// `$(\nu, x)$` and any convergence test would be measuring the wrong thing.
///
/// `N` is a const generic so the loop unrolls and the whole sum folds into a straight-line
/// dependency chain. There is no `Result`: nothing here can fail to converge, because nothing
/// is trying to.
///
/// `advance` and `first` mean exactly what they do in [`sum_ratio`].
#[inline(always)]
pub fn sum_counted<V: FloatVector, P: Policy, const N: usize, F>(first: V, mut advance: F) -> V
where
    F: FnMut(i64, V) -> V,
{
    let mut term = first;
    let mut sum = first;
    let mut c = V::ZERO; // Kahan compensation

    let mut k = 1i64;
    while (k as usize) <= N {
        V::_loop_hint();

        term = advance(k, term);

        let mut delta = term;
        let t = sum + delta;

        if const { P::POLICY.use_compensation } {
            sum.abs().cmp_lt(delta.abs()).swap(&mut sum, &mut delta);
            c += (sum - t) + delta;
        }

        sum = t;
        k += 1;
    }

    if const { P::POLICY.use_compensation } {
        sum += c;
    }

    sum
}

/// Evaluates a continued fraction in modified Lentz form.
///
/// ```math
/// f = b_0 + \cfrac{a_1}{b_1 + \cfrac{a_2}{b_2 + \cfrac{a_3}{b_3 + \dotsb}}}
/// ```
///
/// Returns `Ok(f)` once every lane has converged, or `Err(f)` with the best partial value if
/// `P::POLICY.max_iterations` runs out first.
///
/// `coeffs` is called with `j = 1, 2, 3, ...` and returns the pair `$(a_j, b_j)$`. `b0` is the
/// leading term, passed separately because it has no `$a$` partner.
///
/// # Why Lentz rather than evaluating the convergents
///
/// Lentz's formulation never forms the convergents themselves. It carries only the _ratios_
/// `c` and `d`, so nothing overflows even where the numerator and denominator separately
/// would, which is the usual failure mode of the direct evaluation and the reason this
/// formulation exists.
///
/// The two `is_zero` substitutions are Lentz's own: a vanishing denominator is replaced with
/// the smallest positive normal, which perturbs the result by less than an ulp and keeps the
/// recurrence alive. They are written as `select` rather than as branches, so a single
/// unlucky lane does not cost the packet a detour.
///
/// # Freezing converged lanes is not an optimization
///
/// The running value is built by **multiplication**, so a lane that has converged and keeps
/// iterating gets multiplied by a `$\Delta$` that is only approximately one, and drifts back
/// off the answer it had already reached. The `select` that holds it is load-bearing.
///
/// See the [module documentation](self) for when a multiplicative accumulator should NOT
/// freeze: [`prod_f`] is the counter-example.
#[inline(always)]
pub fn lentz<V: FloatVector, P: Policy, F>(tolerance: V, active: V::Mask, b0: V, mut coeffs: F) -> Result<V, V>
where
    F: FnMut(i64) -> (V, V),
{
    let tiny = V::MIN_POSITIVE;

    // `f0 = b0`, substituted if zero so the first reciprocal is finite.
    let mut f = b0.is_zero().select(tiny, b0);
    let mut c = f;
    let mut d = V::ZERO;

    if active.none() {
        return Ok(b0);
    }

    let mut active = active;
    let mut converged = false;
    let mut j = 1i64;

    while (j as usize) <= P::POLICY.max_iterations {
        V::_loop_hint();

        let (a, b) = coeffs(j);

        let den = a.mul_adde(d, b);
        d = V::ONE / den.is_zero().select(tiny, den);

        let num = b + a / c;
        c = num.is_zero().select(tiny, num);

        let delta = c * d;

        // Frozen lanes keep the value they converged to (see the note above).
        f = f.mul_c(active, delta);

        active &= (delta - V::ONE).abs().cmp_gt(tolerance);

        // Amortized, and overshooting is free because the lanes in question are already frozen.
        // The final iteration is always tested, as in `sum_core`, so a fraction that converges
        // on it is not reported as a failure.
        if ((j as usize).is_multiple_of(CHECK_STRIDE) || j as usize == P::POLICY.max_iterations) && active.none() {
            converged = true;
            break;
        }

        j += 1;
    }

    match converged {
        true => Ok(f),
        false => Err(f),
    }
}

/// Sums **two** series advanced together, sharing whatever the caller's closure shares.
///
/// Returns `Ok((a, b))` once every active lane has converged in **both** components, or
/// `Err((a, b))` with the best partial pair if `P::POLICY.max_iterations` runs out.
///
/// # Why a second driver rather than two calls
///
/// The single-accumulator drivers cannot express a pair that shares a chain, and forcing the
/// shape produces either two full evaluations of a chain that is the expensive part, or a
/// closure that has to return one component and stash the other. Five kernels in this
/// workspace carry two accumulators by hand for exactly this reason:
///
/// * `thermite-compensated`'s `sin_cos`: two Taylor series off one `-r^2`;
/// * `thermite-special`'s Hankel expansion: `P` and `Q` off one coefficient chain;
/// * its Temme series: `Y_v` and `Y_{v+1}` off one `coef` chain;
/// * its `cf2_pq`, which does **not** fit this, being multiplicative and six-wide;
/// * and [`sum_counted`], written for a caller that then needed two.
///
/// So this covers the _additive pair_, which is four of those five. A multiplicative pair is a
/// different driver and is not attempted here.
///
/// # Convergence
///
/// Each component is tested against its **own** running largest term, and both must pass: the
/// same relative rule as [`sum_ratio`], applied twice. Testing only the first component is
/// what Boost's Temme series does and is not safe in general: nothing makes the second
/// component's terms shrink at the same rate.
///
/// `active`, the amortized reduction, and the "inactive lanes count as converged" rule are all
/// exactly as in [`sum_f`]. See the [module documentation](self).
#[inline(always)]
pub fn sum_pair<V: FloatVector, P: Policy, F>(
    tolerance: V,
    active: V::Mask,
    first: (V, V),
    next: F,
) -> Result<(V, V), (V, V)>
where
    F: FnMut() -> (V, V),
{
    sum_pair_core::<V, P, _>(tolerance, active, first, P::POLICY.max_iterations, next)
}

/// Sums exactly `N` steps of a pair, with no convergence test at all.
///
/// The Counted discipline of [`sum_counted`], for two accumulators. The motivating case is the
/// Hankel expansion's `P` and `Q`: a divergent asymptotic series whose truncation index is a
/// closed form in its parameters, so there is nothing to converge to in either component.
#[inline(always)]
pub fn sum_pair_counted<V: FloatVector, P: Policy, const N: usize, F>(first: (V, V), mut next: F) -> (V, V)
where
    F: FnMut() -> (V, V),
{
    let (mut a, mut b) = first;

    let mut k = 0usize;
    while k < N {
        V::_loop_hint();

        let (da, db) = next();
        a += da;
        b += db;

        k += 1;
    }

    (a, b)
}

/// The one paired Additive loop, behind [`sum_pair`].
///
/// Kahan compensation is applied to each component independently under
/// `P::POLICY.use_compensation`, matching [`sum_f`].
#[inline(always)]
fn sum_pair_core<V: FloatVector, P: Policy, F>(
    tolerance: V,
    active: V::Mask,
    first: (V, V),
    count: usize,
    mut next: F,
) -> Result<(V, V), (V, V)>
where
    F: FnMut() -> (V, V),
{
    if active.none() {
        return Ok(first);
    }

    let (mut sum_a, mut sum_b) = first;
    let (mut ca, mut cb) = (V::ZERO, V::ZERO);
    let (mut scale_a, mut scale_b) = (sum_a.abs(), sum_b.abs());

    let mut converged = false;
    let mut k = 1usize;

    while k <= count {
        V::_loop_hint();

        let (mut da, mut db) = next();
        let (abs_a, abs_b) = (da.abs(), db.abs());
        scale_a = scale_a.max(abs_a);
        scale_b = scale_b.max(abs_b);

        let ta = sum_a + da;
        let tb = sum_b + db;

        if const { P::POLICY.use_compensation } {
            sum_a.abs().cmp_lt(abs_a).swap(&mut sum_a, &mut da);
            ca += (sum_a - ta) + da;
            sum_b.abs().cmp_lt(abs_b).swap(&mut sum_b, &mut db);
            cb += (sum_b - tb) + db;
        }

        sum_a = ta;
        sum_b = tb;

        // Both components, each against its own scale, with inactive lanes counting as
        // converged (see `sum_core`, whose rule this is applied twice).
        if (k.is_multiple_of(CHECK_STRIDE) || k == count)
            && ((abs_a.cmp_le(tolerance * scale_a) & abs_b.cmp_le(tolerance * scale_b)) | !active).all()
        {
            converged = true;
            break;
        }

        k += 1;
    }

    if const { P::POLICY.use_compensation } {
        sum_a += ca;
        sum_b += cb;
    }

    match converged {
        true => Ok((sum_a, sum_b)),
        false => Err((sum_a, sum_b)),
    }
}
