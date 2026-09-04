//! The oscillatory Bessel functions `$J_0$`, `$J_1$`, `$Y_0$`, `$Y_1$`.
//!
//! # What "accuracy" means for a function with zeros
//!
//! This is the one decision to make before any tolerance is written, and getting it wrong
//! produces a test that is either impossible to pass or meaningless.
//!
//! `$J_\nu$` and `$Y_\nu$` oscillate through zero forever. At a zero the _relative_ error of
//! any implementation is unbounded (the true value is 0 and the computed one is not), so a
//! relative-error contract is not merely hard to meet, it is not a statement about anything.
//! What every implementation actually delivers, and what this kernel promises, is accuracy
//! **relative to the envelope**:
//!
//! ```math
//! \left|\,\hat{f}(x) - f(x)\,\right| \;\lesssim\; C\,\varepsilon\,\sqrt{\frac{2}{\pi x}}
//! ```
//!
//! since `$\sqrt{2/\pi x}$` is the amplitude the oscillation rides on. Equivalently: absolute
//! error scaled by `$\sqrt{x}$` is bounded. Tests here compare on that basis.
//!
//! Below `x = 8` there is a stronger guarantee. It is why the fits are shaped the way they
//! are. Each sub-8 region carries one zero of the function, factored out as
//! `$(x + x_k)\left((x - x_{k1}/256) - x_{k2}\right)$`: `$x_{k1}/256$` is a power-of-two-scaled
//! integer and therefore exact, so the subtraction near the root loses nothing and full
//! _relative_ accuracy survives at the first two or three zeros. Nobody does this above 8
//! (Boost included) because the number of zeros to factor grows without bound.
//!
//! # Regions
//!
//! `$J$` splits at 4 and 8, `$Y_0$` at 3, 5.5 and 8, `$Y_1$` at 4 and 8. Above 8 all four
//! share the Hankel form: one amplitude pair in `$(8/x)^2$` against `sin x` and `cos x`.
//!
//! That last point is the whole reason this file exists rather than a port of fdlibm, whose
//! `j0f` splits the asymptotic envelope alone into **four** sub-intervals with a rational
//! apiece. A branch picks one and skips three. A vector unit evaluates all four and discards
//! three. Boost's single Hankel region is higher degree and strictly cheaper here.
//!
//! # `Y` calls `J`
//!
//! `$Y_\nu$` is singular at the origin, and the singularity is carried by a
//! `$\frac{2}{\pi}\ln(x/x_k)\,J_\nu(x)$` term rather than by the rational, so these kernels
//! call the `$J$` kernels, exactly as `$K$` calls `$I$`. The log is taken about the region's
//! own root, not as a bare `$\ln x$`, which is what stops that term from swamping the rational
//! near the zero.

use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        policy::{Policy, PolicyParameters, PrecisionPolicy},
    },
    prelude::*,
};

/// Compensated Horner for the rationals in this file, at `Best` and above only.
///
/// No standard policy sets `use_compensation`, so something has to turn it on. This is that
/// something, **tier-gated** rather than unconditional. Both halves of that matter.
///
/// It is on at all because these fits were ill-conditioned in a way the measurement made
/// specific: `J_1`/`Y_1` region 2 evaluated at `y = x^2` up to 64 against coefficients reaching
/// 1.7e18, and real Boost (compiled and graded) measures 35-37 ULP there against a fit whose
/// exact-arithmetic error is 0.002 ULP.
///
/// It is off below `Best` because it is **not cheap and no longer load-bearing**. Measured
/// against the pre-compensation baseline it cost about **4x on f64x4** and, before the non-FMA
/// arm existed, 27x on the 1-lane seed, against a tier spread of only 23-43%. Meanwhile the
/// bounded-variable substitution in the tables now attacks the same conditioning from the other
/// side, so the default tier does not need both. A caller who wants the old behavior asks for
/// `Precision`.
///
/// Spelled as a `Policy` impl rather than `UseCompensation<P, true>` because the flag has to be
/// _computed_ from `P`, and a const-generic bool in a type alias cannot depend on `P` on stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Comp<P: Policy>(core::marker::PhantomData<P>);

impl<P: Policy> Policy for Comp<P> {
    const POLICY: PolicyParameters = PolicyParameters {
        use_compensation: P::POLICY.precision.ge(PrecisionPolicy::Best),
        ..P::POLICY
    };
}

use thermite::element::FloatElement;

use crate::tables::bessel::{BesselJ, BesselY};

/// `(x + root) * ((x - root_hi/256) - root_lo)`, the exactly-split root factor.
///
/// `root_hi/256` is exact by construction, so the inner subtraction is exact whenever `x` is
/// near the root, which is the entire point. Writing this as `x*x - root*root`, or even as
/// `(x + root) * (x - root)`, throws that away and costs every digit at the zero.
#[inline(always)]
fn root_factor<E, V>(x: V, root: V, hi: V, lo: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    (x + root) * ((x - hi * V::splat(E::from_ratio(1, 256))) - lo)
}

/// `$J_0(x)$`. Even in `x`.
#[inline(always)]
pub fn bessel_j0_impl<P, E, V, const N1: usize, const N2: usize, const NH: usize>(x: V, t: &BesselJ<E, N1, N2, NH>) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let four = V::splat(E::from_int(4));
    let eight = V::splat(E::from_int(8));

    let r1 = ax.cmp_le(four);
    let r2 = ax.cmp_le(eight);

    let mut value = V::ZERO;

    if r2.any() {
        let y = ax * ax;
        // Region 1 is a rational in x^2, region 2 in `1 - x^2/64`, which is the same
        // interval mapped to [0, 1] so the fit does not have to span two decades.
        // Each region's rational is skipped when no lane wants it, the way the `x > 8` branch
        // already guards itself. A packet spanning both still pays for both (that is the
        // standing trade), but a packet inside one region now pays for one.
        let lo = if r1.none() {
            V::ZERO
        } else {
            root_factor::<E, V>(ax, V::splat(t.root1), V::splat(t.root1_hi), V::splat(t.root1_lo))
                * y.poly_rational_n_p::<Comp<P>, N1, N1>(&t.p1, &t.q1)
        };
        let mid = if r1.all() {
            V::ZERO
        } else {
            root_factor::<E, V>(ax, V::splat(t.root2), V::splat(t.root2_hi), V::splat(t.root2_lo))
                * (V::ONE - y * V::splat(E::from_ratio(1, 64))).poly_rational_n_p::<Comp<P>, N2, N2>(&t.p2, &t.q2)
        };
        value = r1.select(lo, mid);
    }

    if !r2.all() {
        value = r2.select(value, hankel::<P, E, V, NH>(ax, t.pc, t.qc, t.ps, t.qs, false));
    }

    value
}

/// `$J_1(x)$`. Odd in `x`.
#[inline(always)]
pub fn bessel_j1_impl<P, E, V, const N1: usize, const N2: usize, const NH: usize>(x: V, t: &BesselJ<E, N1, N2, NH>) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let four = V::splat(E::from_int(4));
    let eight = V::splat(E::from_int(8));

    let r1 = ax.cmp_le(four);
    let r2 = ax.cmp_le(eight);

    let mut value = V::ZERO;

    if r2.any() {
        let y = ax * ax;
        // The extra leading `x` is `J_1`'s odd factor, and also supplies the zero at the
        // origin exactly rather than through the rational.
        // See `bessel_j0_impl`: skip a region's rational when no lane is in it.
        let lo = if r1.none() {
            V::ZERO
        } else {
            ax * root_factor::<E, V>(ax, V::splat(t.root1), V::splat(t.root1_hi), V::splat(t.root1_lo))
                * y.poly_rational_n_p::<Comp<P>, N1, N1>(&t.p1, &t.q1)
        };
        // Region 2 in `1 - x^2/64`, as `J_0` does. Boost leaves `J_1` on the raw `x^2`, which
        // runs to 64 against coefficients reaching 1.7e18. Compiled and graded, that costs real
        // Boost 35.60 eps here against 3.20 for `j0`. The table carries the same rational
        // re-expressed by exact algebra (see `scripts/bessel_jy_tables.py`), coefficients
        // topping out near 7.4e5.
        let mid = if r1.all() {
            V::ZERO
        } else {
            ax * root_factor::<E, V>(ax, V::splat(t.root2), V::splat(t.root2_hi), V::splat(t.root2_lo))
                * (V::ONE - y * V::splat(E::from_ratio(1, 64))).poly_rational_n_p::<Comp<P>, N2, N2>(&t.p2, &t.q2)
        };
        value = r1.select(lo, mid);
    }

    if !r2.all() {
        value = r2.select(value, hankel::<P, E, V, NH>(ax, t.pc, t.qc, t.ps, t.qs, true));
    }

    // NOT `copysign(x)`: `J_1` is odd but also oscillates, so past its first zero at 3.83 it
    // is negative for positive `x` and `copysign` would force it positive. `I_1` gets away
    // with `copysign` only because it is positive on the whole positive axis.
    value.neg_c(x.is_negative())
}

/// `(sin x + cos x, sin x - cos x)`, with whichever one is near zero rebuilt so it is not.
///
/// Both combinations vanish periodically (`sin x + cos x` at `x = 3pi/4 + k pi`, `sin x - cos x`
/// at `pi/4 + k pi`), and at those points the subtraction of two same-magnitude numbers loses
/// every digit that sets the Bessel function's phase. The repair is an exact identity:
///
/// ```math
/// (\sin x + \cos x)(\sin x - \cos x) = \sin^2 x - \cos^2 x = -\cos 2x
/// ```
///
/// so the small one equals `-cos(2x)` divided by the large one, and the large one never
/// cancels. They cannot both be small: their squares sum to 2.
///
/// `sin x * cos x < 0` is exactly the condition for `sin x + cos x` being the small one, which
/// is why the sign of the product picks the branch.
///
/// fdlibm does this and Boost does not. Boost writes the addition formulae out flat. It is the
/// larger part of libm's remaining accuracy advantage in the asymptotic region, and the trick
/// was already in this crate, in the dormant fdlibm `bessel_j0` port at
/// `crates/thermite-special/src/specialized/ps.rs`. It did not survive the move to the
/// Boost-shaped kernel.
///
/// Gated at `Average` and above, which includes the default policy. Below that the pair is
/// returned unrepaired. See the comment in the body for why that beats the cheap algebraic
/// `-cos 2x` it replaced.
#[inline(always)]
fn sum_diff_repaired<P, E, V>(ax: V, sx: V, cx: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let cc = sx + cx;
    let ss = sx - cx;

    // Below `Average` the pair is returned unrepaired, which is the measured second-best form.
    // The cheap algebraic repair, `-cos 2x = 1 - 2cos^2 x`, cancels exactly where it is needed
    // (`cx` near `1/sqrt2`, so `cx*cx` carries ~1.3e-16 absolute, ~2.6e-16 in the result),
    // while the plain `sx - cx` is already Sterbenz-exact there and limited only by sin/cos's
    // own half-ulp: about 2.4x worse, for two extra ops and two divides.
    //
    // The gate is `ge(Average)`, so the DEFAULT policy repairs. `gt` would have excluded every
    // plain, non-`_p` call.
    if const { !P::POLICY.precision.ge(PrecisionPolicy::Average) } {
        return (cc, ss);
    }

    // `-cos(2x)`, from a fresh call at the doubled argument. `ax + ax` is exact, so this is the
    // only form with RELATIVE accuracy near its own zero, which is where the repair is used.
    let neg_cos2x = -(ax + ax).cos_p::<P>();

    // The replacement is formed from the ORIGINAL large one, then selected in. Computing it
    // from an already-repaired other would feed the repair its own rounding. Only one of the
    // pair is ever rebuilt per lane, so the divisor is selected and the division happens once.
    let fix_cc = (sx * cx).is_negative();
    let rebuilt = neg_cos2x / fix_cc.select(ss, cc);

    (fix_cc.select(rebuilt, cc), fix_cc.select(ss, rebuilt))
}

/// The shared Hankel asymptotic for `x > 8`.
///
/// ```math
/// J_\nu(x) = \sqrt{\frac{1}{\pi x}}\left(R_c \cos z - \tfrac{8}{x} R_s \sin z\right),
/// \qquad z = x - \left(\tfrac{\nu}{2} + \tfrac14\right)\pi
/// ```
///
/// Written out through the sin/cos addition formulae instead of forming `z`. Two reasons, and
/// the second is the important one: it saves a subtraction, and more to the point `x - z_0`
/// for a large `x` and an irrational `z_0` loses exactly the low bits that set the phase. The
/// `$\sin(\pi/4) = \cos(\pi/4) = 1/\sqrt2$` factors then cancel against the `$1/\sqrt{\pi x}$`
/// out front, which is why no `$1/\sqrt2$` appears anywhere below.
#[inline(always)]
fn hankel<P, E, V, const NH: usize>(ax: V, pc: [E; NH], qc: [E; NH], ps: [E; NH], qs: [E; NH], order_one: bool) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let y = V::splat(E::from_int(8)) / ax;
    let y2 = y * y;
    let rc = y2.poly_rational_n_p::<Comp<P>, NH, NH>(&pc, &qc);
    let rs = y2.poly_rational_n_p::<Comp<P>, NH, NH>(&ps, &qs);
    let (sx, cx) = ax.sin_cos_p::<P>();
    // `cc = sin+cos`, `ss = sin-cos`, each repaired where it cancels.
    let (cc, ss) = sum_diff_repaired::<P, E, V>(ax, sx, cx);
    // `sqrt(1/(pi x))`, not `(1/sqrt pi) / sqrt(x)`. Same op count (one divide, one sqrt), but
    // the sqrt HALVES the relative error entering it instead of adding to it, so the constant's
    // and the divide's roundings cost half as much: ~1 ulp against ~1.5.
    let factor = (V::FRAC_1_PI / ax).sqrt();

    let yrs = y * rs;
    let value = if order_one {
        factor * yrs.mul_adde(cc, rc * ss)
    } else {
        factor * yrs.nmul_adde(ss, rc * cc)
    };
    // Zero at infinity, where `sin_cos` is NaN and `factor` is 0.
    ax.cmp_eq(V::INFINITY).select(V::ZERO, value)
}

/// `$Y_0(x)$` and `$Y_1(x)$`, selected by `ORDER_ONE`.
///
/// Both are `$\frac{2}{\pi}\ln(x/x_k)J_\nu(x) + \text{(root-factored rational)}$` below 8, and
/// the shared Hankel above it. Undefined for `x <= 0`: NaN there, and `$-\infty$` at 0.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_y_impl<
    P,
    E,
    V,
    const N1: usize,
    const N2: usize,
    const N3: usize,
    const NH: usize,
    const J1: usize,
    const J2: usize,
    const JH: usize,
    const ORDER_ONE: bool,
>(
    x: V,
    t: &BesselY<E, N1, N2, N3, NH>,
    tj: &BesselJ<E, J1, J2, JH>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let eight = V::splat(E::from_int(8));
    let small = x.cmp_le(eight);

    let mut value = V::ZERO;

    if small.any() {
        let y = x * x;
        let j = if const { ORDER_ONE } {
            bessel_j1_impl::<P, E, V, J1, J2, JH>(x, tj)
        } else {
            bessel_j0_impl::<P, E, V, J1, J2, JH>(x, tj)
        };
        let two_over_pi = V::FRAC_2_PI;

        // Every region shares the shape. Only the rational, the root and the log's base move.
        // The three are evaluated and selected rather than branched, which is the standing
        // trade here, but the `$\ln$` and the `$J$` are computed once for all of them.
        // `Y_1`'s upper region spans (4, 8], so its argument runs to 64 against coefficients
        // reaching 1.15e19, the same defect `J_1` had, and Boost has it in both. The table
        // carries that rational re-expressed in `1 - x^2/64` (exact algebra, see
        // `scripts/bessel_jy_tables.py`), so the argument here must match.
        //
        // `Y_0` stays on `x^2`: it bounds its argument the other way, by splitting (0, 8] into
        // three narrow regions, and measures 4.43 against Boost's 37.49 for `Y_1`.
        //
        // Region 3 is `Y_1`'s duplicate of region 2 and is unreachable for it (`threshold2` is
        // 8.0, so `in2` is always true below 8), but it shares the argument anyway rather than
        // evaluating a `u`-basis rational at a `y`-basis point.
        let arg2 = if const { ORDER_ONE } {
            V::ONE - y * V::splat(E::from_ratio(1, 64))
        } else {
            y
        };

        let in1 = x.cmp_le(V::splat(t.threshold1));
        let in2 = x.cmp_le(V::splat(t.threshold2));

        let a = root_factor::<E, V>(x, V::splat(t.root1), V::splat(t.root1_hi), V::splat(t.root1_lo))
            * y.poly_rational_n_p::<Comp<P>, N1, N1>(&t.p1, &t.q1);
        let b = root_factor::<E, V>(x, V::splat(t.root2), V::splat(t.root2_hi), V::splat(t.root2_lo))
            * arg2.poly_rational_n_p::<Comp<P>, N2, N2>(&t.p2, &t.q2);

        let mut rat = if const { ORDER_ONE } {
            // `Y_1` has only TWO regions. It fills the third slot with a copy of the second so
            // the table keeps one shape. `threshold2` is 8.0, so `in2` is true across this
            // whole branch and region 3 is unreachable. Evaluating it anyway cost a full
            // degree-9/9 rational per call, discarded.
            b
        } else {
            // `Y_0` genuinely has three. Skip region 3's rational when no lane is in it. The
            // `x > 8` branch already guards itself this way.
            let c = if in2.all() {
                V::ZERO
            } else {
                root_factor::<E, V>(x, V::splat(t.root3), V::splat(t.root3_hi), V::splat(t.root3_lo))
                    * arg2.poly_rational_n_p::<Comp<P>, N3, N3>(&t.p3, &t.q3)
            };
            in2.select(b, c)
        };
        rat = in1.select(a, rat);

        let root = in1.select(V::splat(t.root1), in2.select(V::splat(t.root2), V::splat(t.root3)));
        // `ln(x / x_k)`, about the region's own root: a bare `ln x` would make this term
        // dominate the rational near the zero and take the accuracy with it.
        //
        // Not `ln_1p(delta / x_k)` off the exactly-split delta (LOG 2026-08-30): near the root
        // the quotient's rounding vanishes with the term (under 0.4 eps envelope-relative),
        // and at the far end of region 1 (`x ~ 1e-3`) `delta / x_k` approaches -1 and cancels
        // INSIDE `ln_1p`, 142x worse at `x = 0.00114` and `Y_0` at 10.85 eps against libm's 0.13.
        let z = two_over_pi * (x / root).ln_p::<P>() * j;

        // `J_1`'s factored form carries an extra `1/x` on the Y side. Folding the addend into
        // an FMA measured zero change on all three compensation arms (LOG 2026-08-31): the
        // 1.58 of 2.20 eps here is the `ln` and the rounded `2/pi`.
        value = if const { ORDER_ONE } { z + rat / x } else { z + rat };
    }

    if !small.all() {
        // Y's Hankel is J's with sin and cos exchanged (the two are a quarter-period apart).
        let y = eight / x;
        let y2 = y * y;
        let rc = y2.poly_rational_n_p::<Comp<P>, NH, NH>(&t.pc, &t.qc);
        let rs = y2.poly_rational_n_p::<Comp<P>, NH, NH>(&t.ps, &t.qs);
        let (sx, cx) = x.sin_cos_p::<P>();
        let (cc, ss) = sum_diff_repaired::<P, E, V>(x, sx, cx);
        // See `hankel`: the sqrt halves the incoming relative error rather than adding to it.
        let factor = (V::FRAC_1_PI / x).sqrt();
        let yrs = y * rs;
        let hi = if const { ORDER_ONE } {
            factor * yrs.mul_sube(ss, rc * cc)
        } else {
            factor * yrs.mul_adde(cc, rc * ss)
        };
        // Zero at infinity, as for `J`.
        let hi = x.cmp_eq(V::INFINITY).select(V::ZERO, hi);
        value = small.select(value, hi);
    }

    // No reflection: Y has a branch cut on the negative axis.
    x.cmp_lt(V::ZERO).select(V::NAN, value)
}

/// `(Y_{N-1}, Y_N)` by **upward** recurrence from the two closed forms.
///
/// ```math
/// Y_{n+1}(x) = \frac{2n}{x} Y_n(x) - Y_{n-1}(x)
/// ```
///
/// `$Y_\nu$` is the dominant solution of Bessel's equation, so upward is stable and costs
/// exactly `N - 1` steps, with no trip count, `x` dependence or precision tier. Same argument
/// and same shape as [`bessel_kn_recur`](super::ik::bessel_kn_recur). The only difference
/// is the sign, since this is the unmodified equation.
///
/// Measured against mpmath over orders 2..50 and `x` in 0.1..300: worst **2.8e-14** relative.
/// Looser than `K`'s 1.4e-15 because this recurrence _subtracts_ where `K`'s adds, so it does
/// accumulate a little cancellation, but `Y_n` grows with `n`, which keeps it bounded.
#[inline(always)]
pub fn bessel_yn_recur<E, V, const N: i32>(x: V, y0: V, y1: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let mut prev = y0;
    let mut cur = y1;
    let two_over_x = V::TWO / x;

    // `N` is signed and the loop walks to `|N|`: the reflection `Y_{-n} = (-1)^n Y_n` is a
    // sign the CALLER applies, because the recurrence itself has no notion of a negative
    // order. Taking the absolute value here rather than in a const-generic argument is
    // forced: `foo::<{ N.unsigned_abs() }>` needs `generic_const_exprs`, which is not
    // stable. A `const` block folds identically, so trip counts still unroll.
    let m = const { N.unsigned_abs() as usize };

    let mut n = 1usize;
    while n < m {
        let next = two_over_x.mul_sube(V::splat(E::from_int(n as _)) * cur, prev);
        prev = cur;
        cur = next;
        n += 1;
    }

    (prev, cur)
}

/// `(J_{N-1}, J_N)` for `N >= 2`.
///
/// # Two arms, and why `J` gets a crossover that `I` did not
///
/// `$J_\nu$` is the minimal solution, so upward recurrence is unstable _in general_, and yet
/// measured against mpmath it is **exact whenever `N < x`**: worst 3.08e-16 envelope-relative
/// over orders 2..50 and `x` to 300.
///
/// That is the reverse of what happened for `$I_\nu$`, where forward recurrence failed even
/// far into the region the textbook rule blesses, and the reason is worth writing down because
/// the two look like the same recurrence. `I` accumulates **cancellation**:
/// `I_{k+1} = I_{k-1} - (2k/x)I_k` subtracts two nearly equal numbers for `k << x`, losing bits
/// every step regardless of which solution dominates. `J` oscillates, so its terms are not
/// systematically close and there is nothing to cancel. What is left is dominant-solution
/// admixture, and `Y_n/J_n` is `O(1)` precisely while `N < x`.
///
/// So the split is on `N < x`, and **both arms are bounded by `N` alone**: forward costs
/// `N - 1` steps, and the downward arm is only reached where `x <= N`, which caps its
/// `x`-scaled trip count at `0.35N`. No `O(x)` tail, and therefore no asymptotic arm needed.
///
/// # The normalization, and the trap in it
///
/// The downward arm produces ratios `r_k = J_k/J_{k-1}` and recovers `J_N = J_0 \prod r_k`.
/// That normalization is fine for `$I$`, whose order-0 value is positive everywhere, and
/// **wrong for `$J$`**, whose `J_0` vanishes at 2.405, 5.520, ... Boost normalizes by `J_0`
/// regardless. Here the seed is whichever of `J_0`, `J_1` is larger in magnitude. They have no
/// common zero, so one of them is always well away from zero. `J_1` seeding divides the
/// product by `r_1`, which is exactly `J_0/J_1` and cancels the bad factor rather than
/// carrying it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_jn_pair_impl<
    P,
    E,
    V,
    const A1: usize,
    const A2: usize,
    const AH: usize,
    const B1: usize,
    const B2: usize,
    const BH: usize,
    const N: i32,
>(
    x: V,
    t0: &BesselJ<E, A1, A2, AH>,
    t1: &BesselJ<E, B1, B2, BH>,
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let j0 = bessel_j0_impl::<P, E, V, A1, A2, AH>(ax, t0);
    let j1 = bessel_j1_impl::<P, E, V, B1, B2, BH>(ax, t1);

    // Evaluated at `|N|`. `J_{-n} = (-1)^n J_n` is the caller's to apply. See
    // `bessel_yn_recur` for why the absolute value cannot live in the generic argument.
    let m = const { N.unsigned_abs() as usize };

    let n_f = V::splat(E::from_int(m as _));
    let two_over_x = V::TWO / ax;

    // `N < x`: forward is exact here, and costs N-1 FMAs.
    let use_fwd = ax.cmp_gt(n_f);

    let mut value = V::ZERO;
    let mut prev_out = V::ZERO;

    if use_fwd.any() {
        let mut prev = j0;
        let mut cur = j1;
        let mut n = 1usize;
        while n < m {
            let next = two_over_x.mul_sube(V::splat(E::from_int(n as _)) * cur, prev);
            prev = cur;
            cur = next;
            n += 1;
        }
        value = cur;
        prev_out = prev;
    }

    if !use_fwd.all() {
        // Downward ratio recurrence, `r_k = 1/(2k/x - r_{k+1})`. The minus is the whole
        // difference from the `I` version. The ratios are no longer confined to (0,1), but
        // they still cannot overflow, because a near-pole in one is followed by a near-zero
        // in the next and the running product telescopes.
        let (cn, cd) = const { super::ik::recurrence_x_coeff(P::POLICY.precision) };
        let coeff = V::splat(E::from_ratio(cn, cd));
        let start_f = V::splat(E::from_int((m + super::ik::RECURRENCE_MARGIN) as _));
        let nm1_f = V::splat(E::from_int((m as i64) - 1));

        let mut k = (!use_fwd).select(ax.mul_adde(coeff, start_f).ceil(), V::ZERO);
        let mut r = V::ZERO;
        // Both products EXCLUDE `r_1`, and `r_1` is carried separately. Folding it in and
        // dividing it back out looks equivalent and is not: at a zero of `J_0`, `r_1 = J_1/J_0`
        // overflows to infinity, and then `(j1 / inf) * inf` is NaN rather than the right
        // answer. Keeping `r_1` out of the product means the `J_1` seed never has to undo it.
        let mut prod = V::ONE;
        let mut prod_prev = V::ONE;
        let mut r1 = V::ONE;
        let two_f = V::splat(E::from_int(2));

        loop {
            let active = k.cmp_ge(V::ONE);
            if active.none() {
                break;
            }
            r = active.select(V::ONE / two_over_x.mul_sube(k, r), r);
            let in_prod = active & k.cmp_ge(two_f);
            prod = (in_prod & k.cmp_le(n_f)).select(prod * r, prod);
            prod_prev = (in_prod & k.cmp_le(nm1_f)).select(prod_prev * r, prod_prev);
            // The last rung of the descent is r_1 = J_1/J_0.
            r1 = (active & k.cmp_le(V::ONE)).select(r, r1);
            k -= V::ONE;
        }

        // Seed from whichever closed form is further from its own zeros. `J_0` and `J_1` have
        // no common zero, so one of them is always healthy, and `J_0` vanishes at 2.405,
        // 5.520, ... where Boost's unconditional `J_0` normalization has nothing to divide by.
        // The unused arm may evaluate to NaN. `select` is bitwise, so it does not propagate.
        let use_j0 = j0.abs().cmp_ge(j1.abs());
        let base = use_j0.select(j0 * r1, j1);
        value = use_fwd.select(value, base * prod);
        prev_out = use_fwd.select(prev_out, base * prod_prev);
    }

    // J_N has the parity of N, and J_{N-1} the opposite.
    let odd = x.is_negative();
    let v = if const { N % 2 == 0 } { value } else { value.neg_c(odd) };
    let p = if const { N % 2 == 0 } {
        prev_out.neg_c(odd)
    } else {
        prev_out
    };
    (p, v)
}

/// `Y_n(x)` with a **per-lane** order.
///
/// Upward recurrence with each lane freezing at its own order, the `hermitev` shape. `Y` is
/// the dominant solution so there is only ever one arm, which makes this the simplest of the
/// four runtime-order entry points.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_yv_impl<
    P,
    E,
    V,
    const A1: usize,
    const A2: usize,
    const A3: usize,
    const AH: usize,
    const B1: usize,
    const B2: usize,
    const B3: usize,
    const BH: usize,
    const J1: usize,
    const J2: usize,
    const JH: usize,
    const K1: usize,
    const K2: usize,
    const KH: usize,
>(
    x: V,
    n: V,
    t0: &BesselY<E, A1, A2, A3, AH>,
    t1: &BesselY<E, B1, B2, B3, BH>,
    tj0: &BesselJ<E, J1, J2, JH>,
    tj1: &BesselJ<E, K1, K2, KH>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let y0 = bessel_y_impl::<P, E, V, A1, A2, A3, AH, J1, J2, JH, false>(x, t0, tj0);
    let y1 = bessel_y_impl::<P, E, V, B1, B2, B3, BH, K1, K2, KH, true>(x, t1, tj1);

    let two_over_x = V::TWO / x;
    let mut prev = y0;
    let mut cur = y1;
    let mut step = V::ONE;

    loop {
        let cont = step.cmp_lt(n);
        if cont.none() {
            break;
        }
        let next = two_over_x.mul_sube(step * cur, prev);
        prev = cont.select(cur, prev);
        cur = cont.select(next, cur);
        step += V::ONE;
    }

    n.cmp_le(V::ZERO).select(y0, cur)
}

/// `J_n(x)` with a **per-lane** order.
///
/// Both arms, selected per lane as in the const form: forward where `n < x`, downward ratio
/// otherwise. The forward arm freezes each lane at its own order. The downward arm was already
/// masked on `k <= N`, so a vector `n` slots straight in.
///
/// The `J_0`-zero guard carries over unchanged and matters just as much: `r_1` is kept out of
/// the running product and folded into the `J_0` seed, so a lane sitting on a zero of `J_0`
/// takes the `J_1` seed without ever forming `inf * 0`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn bessel_jv_impl<
    P,
    E,
    V,
    const A1: usize,
    const A2: usize,
    const AH: usize,
    const B1: usize,
    const B2: usize,
    const BH: usize,
>(
    x: V,
    n: V,
    t0: &BesselJ<E, A1, A2, AH>,
    t1: &BesselJ<E, B1, B2, BH>,
) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let ax = x.abs();
    let j0 = bessel_j0_impl::<P, E, V, A1, A2, AH>(ax, t0);
    let j1 = bessel_j1_impl::<P, E, V, B1, B2, BH>(ax, t1);
    let two_over_x = V::TWO / ax;

    let use_fwd = ax.cmp_gt(n);
    let mut value = V::ZERO;

    if use_fwd.any() {
        let mut prev = j0;
        let mut cur = j1;
        let mut step = V::ONE;
        loop {
            let cont = step.cmp_lt(n) & use_fwd;
            if cont.none() {
                break;
            }
            let next = two_over_x.mul_sube(step * cur, prev);
            prev = cont.select(cur, prev);
            cur = cont.select(next, cur);
            step += V::ONE;
        }
        value = n.cmp_le(V::ZERO).select(j0, cur);
    }

    if !use_fwd.all() {
        let (cn, cd) = const { super::ik::recurrence_x_coeff(P::POLICY.precision) };
        let coeff = V::splat(E::from_ratio(cn, cd));
        let margin = V::splat(E::from_int(super::ik::RECURRENCE_MARGIN as _));
        let two_f = V::splat(E::from_int(2));

        let mut k = (!use_fwd).select(ax.mul_adde(coeff, n + margin).ceil(), V::ZERO);
        let mut r = V::ZERO;
        let mut prod = V::ONE;
        let mut r1 = V::ONE;

        loop {
            let active = k.cmp_ge(V::ONE);
            if active.none() {
                break;
            }
            r = active.select(V::ONE / two_over_x.mul_sube(k, r), r);
            prod = (active & k.cmp_ge(two_f) & k.cmp_le(n)).select(prod * r, prod);
            r1 = (active & k.cmp_le(V::ONE)).select(r, r1);
            k -= V::ONE;
        }

        let use_j0 = j0.abs().cmp_ge(j1.abs());
        let base = use_j0.select(j0 * r1, j1);
        // Order 0 has an empty product AND no `r_1` factor, so it is the seed itself.
        let down = n.cmp_le(V::ZERO).select(j0, base * prod);
        value = use_fwd.select(value, down);
    }

    let odd_order = (n * V::HALF).fract().cmp_gt(V::ZERO);
    value.neg_c(odd_order & x.is_negative())
}
