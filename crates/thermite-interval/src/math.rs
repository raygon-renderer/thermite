//! The math library for `Interval<V, W>`: implementations of Thermite's
//! `Specialized*Math` traits, which light up the whole public `CoreMath` /
//! `TranscendentalMath` / `SpatialMath` / `RealMath` API through the blanket
//! impls in `thermite::math`.
//!
//! # The two policies
//!
//! Every method here takes thermite's math policy `P` _and_ carries the
//! type's widening policy `W`. They do different jobs:
//!
//! - `P` selects the inner vector's approximation algorithm, exactly as it
//!   does for a plain float vector. A cheaper `P` means a looser _value_.
//! - `W` governs the interval bookkeeping around it.
//! - Bridging them: after evaluating the inner kernel at each endpoint, the
//!   result is widened by that kernel's algorithm error at `P`
//!   ([`algo_widen`]), then by `W`'s rounding strategy.
//!
//! So `sin_p::<Performance>` on a `Tightest` interval is a cheap value with
//! an honest (if algorithm-error-dominated) enclosure, while `Precision` on
//! the same type gives the tight one. Containment holds for every
//! combination. That is ground rule 1.
//!
//! # The algorithm-error bound is MEASURED, not proven
//!
//! [`algo_widen`] widens by [`ULP_DEFAULT`] ulps (or a per-function override),
//! derived from a sweep of every kernel against a `Compensated<Vector<f64>>`
//! reference over its full domain (`bin/ulp_sweep`, 2026-08-15). Nearly every
//! thermite kernel measured at or below 4 ulp, and the margins are that
//! maximum times a 4x safety factor.
//!
//! Measured, not proven: a sweep can miss the worst input, so these are
//! defensible engineering bounds rather than certificates. A rigorous crate
//! wants per-function proven bounds (a Gappa-style proof pass) or
//! correctly-rounded kernels with directed rounding (which is
//! how IntervalArithmetic.jl gets zero-slack enclosures out of CRlibm). Until
//! one of those lands, treat transcendental enclosures as "high-confidence"
//! and the arithmetic/`sqrt` ones (correctly-rounded primitives plus
//! error-free transforms) as rigorous.
//!
//! # The kernel policy has a floor
//!
//! Thermite's `Medium` and `Worst` precision tiers deliberately trade accuracy
//! for speed in ways that abandon _relative_ error bounds entirely: at
//! `Medium`, `ln1m_expnx(40)` returns exactly `0.0` where the true value is
//! `-4.2e-18`. No relative margin can cover that, so no enclosure built on
//! those kernels is sound. This was measured, and it did violate containment
//! before this floor existed.
//!
//! [`KernelPolicy`] therefore raises the _inner kernel's_ precision to at
//! least `Average` whatever the caller passes. The caller's speed knob is `W`
//! (interval bookkeeping), which is unaffected. What they lose is the option
//! to run a kernel so inaccurate that the enclosure around it would be
//! meaningless anyway.
//!
//! # Structure
//!
//! - Monotone increasing functions map endpoint-wise: `[f(lo), f(hi)]`.
//! - Monotone decreasing ones swap: `[f(hi), f(lo)]`.
//! - Non-monotone ones need real interval reasoning: `sin`/`cos` use
//!   quadrant analysis, even powers go through the mignitude/magnitude pair,
//!   and anything whose interval form is not yet worked out is left to the
//!   trait default (which composes out of the enclosing arithmetic above and
//!   is therefore valid, if wide).

use thermite::math::policy::{Policy, PrecisionPolicy};
use thermite::math::specialized::{
    SpecializedCoreMath, SpecializedRealMath, SpecializedSpatialMath, SpecializedTranscendentalMath,
};
use thermite::math::{PrimalProjection, RealMathWithPolicy};
use thermite::prelude::*;

use crate::consts::BoundedFloatConsts;
use crate::element::IntervalElem;
use crate::round::{bump_down, bump_up};
use crate::widen::WideningPolicy;
use crate::{Interval, IntervalFloatVector};

/// Inner-vector requirements for the interval math library.
pub trait IntervalMathVector: IntervalFloatVector + BoundedFloatConsts<Self> + RealMathWithPolicy {}
impl<V> IntervalMathVector for V where V: IntervalFloatVector + BoundedFloatConsts<V> + RealMathWithPolicy {}

/// Raises a caller's math policy to the minimum precision at which thermite's
/// kernels still have a usable relative error bound (`Average`). See the
/// module docs: below this, some kernels return zero for a nonzero result and
/// no enclosure built on them can be sound.
///
/// Everything except the precision tier is passed through untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelPolicy<P: Policy>(core::marker::PhantomData<P>);

impl<P: Policy> Policy for KernelPolicy<P> {
    const POLICY: thermite::math::policy::PolicyParameters = {
        let p = P::POLICY;
        thermite::math::policy::PolicyParameters {
            precision: if p.precision.lt(PrecisionPolicy::Average) {
                PrecisionPolicy::Average
            } else {
                p.precision
            },
            check_overflow: p.check_overflow,
            unroll_loops: p.unroll_loops,
            avoid_branching: p.avoid_branching,
            max_iterations: p.max_iterations,
            use_compensation: p.use_compensation,
            denormal_behavior: p.denormal_behavior,
        }
    };
}

/// Default algorithm-error margin, in ulps of the computed bound.
///
/// The 2026-08-15 sweep (`bin/ulp_sweep`) measured every kernel below against
/// a double-double reference across its domain. All but the three overridden
/// below came in at or under 4 ulp at every tier from `Average` up. 4x safety.
pub const ULP_DEFAULT: u32 = 16;

/// `powf`: measured 355 ulp at extreme base/exponent combinations
/// (`6.5e18 ^ -9.15`). This is the known-weak kernel, see the double-double
/// `ln` arc in the math precision audit.
pub const ULP_POWF: u32 = 2048;

/// `compound`: measured 35.7 ulp, and the precision audit records 247 at
/// `n = 1e4`, beyond this sweep's range, hence the wide margin.
pub const ULP_COMPOUND: u32 = 1024;

/// `powf_m1`: measured 9.1 ulp.
pub const ULP_POWF_M1: u32 = 64;

/// `compound_m1`: shares `compound`'s `ln_1p * n` exponent, so it inherits the
/// same wide margin rather than `powf_m1`'s narrow one.
pub const ULP_COMPOUND_M1: u32 = 1024;

/// Largest `|x|` at which thermite's trig kernels still deliver their nominal
/// accuracy, by tier. Past this the enclosure degrades to `[-1, 1]`.
///
/// Measured 2026-08-15 (`bin/ulp_sweep`, `trigmag`): the `Average`-tier range
/// reduction holds to 1.1e-16 absolute out to |x| ~ 1e8, then collapses
/// (1.2e-7 at 1e9, 1.9e-6 at 1e10, a useless 1.0 by 1e14). `Best` and up
/// carry a Payne-Hanek-class reduction and stay at 2.2e-16 through 1e16.
/// Past that even a double-double reference stops being trustworthy, so the
/// cap stays inside the verified range.
///
/// Expressed through `EPSILON` so it adapts to the element format rather than
/// hard-coding f64 magnitudes: `1/sqrt(eps)` is 6.7e7 in f64 (just inside the
/// measured 1e8 cliff) and 2.9e3 in f32, and `1/(8 eps)` is 2.8e14 / 5.2e5.
/// Both sit inside the verified range, and erring low is always sound, since
/// the guard only ever widens the result to `[-1, 1]`.
#[inline(always)]
pub(crate) fn trig_safe_magnitude<V: IntervalMathVector, P: Policy>() -> V {
    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        <V as FloatConsts>::EPSILON
            .scale(<V::Element as thermite::element::FloatElement>::from_int(8))
            .reciprocal_p::<P>()
    } else {
        <V as FloatConsts>::SQRT_EPSILON.reciprocal_p::<P>()
    }
}

/// Widens `[lo, hi]` outward by the algorithm error of the inner kernel at
/// policy `P`, then by one bookkeeping step. Relative widening (scaled by
/// the magnitude of each bound) plus an absolute floor of `MIN_POSITIVE`, so
/// results near zero still widen.
#[inline(always)]
pub(crate) fn algo_widen<V: IntervalMathVector, P: Policy>(lo: V, hi: V) -> (V, V) {
    algo_widen_n::<V, P>(lo, hi, ULP_DEFAULT)
}

/// [`algo_widen`] with an explicit per-function margin. See [`ULP_POWF`] and
/// friends for the measured overrides.
#[inline(always)]
pub(crate) fn algo_widen_n<V: IntervalMathVector, P: Policy>(lo: V, hi: V, n: u32) -> (V, V) {
    // n ulps relative: n * eps * |x|, plus the denormal floor.
    let scale = <V as FloatVector>::EPSILON
        * V::splat(<V::Element as thermite::element::FloatElement>::from_int(
            n as thermite::LargeInt,
        ));

    let lo_w = lo - (lo.abs().mul_adde(scale, V::MIN_POSITIVE));
    let hi_w = hi + (hi.abs().mul_adde(scale, V::MIN_POSITIVE));

    // Non-finite bounds pass through unchanged (already saturated).
    (
        lo.is_finite().select(bump_down(lo_w), lo),
        hi.is_finite().select(bump_up(hi_w), hi),
    )
}

impl<V: IntervalMathVector, W: WideningPolicy> Interval<V, W> {
    /// Enclosure of a _monotonically increasing_ function: evaluate the inner
    /// kernel at both endpoints and widen for algorithm + rounding error.
    #[inline(always)]
    pub(crate) fn monotone_inc<P: Policy>(self, f: impl Fn(V) -> V) -> Self {
        let poison = self.is_empty();
        let (lo, hi) = algo_widen::<V, P>(f(self.lo), f(self.hi));
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Enclosure of a _monotonically decreasing_ function: the endpoints swap.
    #[inline(always)]
    pub(crate) fn monotone_dec<P: Policy>(self, f: impl Fn(V) -> V) -> Self {
        let poison = self.is_empty();
        let (lo, hi) = algo_widen::<V, P>(f(self.hi), f(self.lo));
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Enclosure of an _even_ function that is increasing in `|x|` (`cosh`,
    /// `x^2n`): `[f(mig), f(mag)]`.
    ///
    /// `NONNEG` clamps the lower bound at zero for functions whose range is
    /// non-negative (even powers). Without it, `algo_widen`'s absolute
    /// `MIN_POSITIVE` floor pushes `f(mig) == 0` to a denormal _negative_
    /// lower bound. That is valid as an enclosure, but a lie about the sign,
    /// and exactly the thing `square_interval` exists to avoid.
    #[inline(always)]
    pub(crate) fn even_inc<P: Policy, const NONNEG: bool>(self, f: impl Fn(V) -> V) -> Self {
        let poison = self.is_empty();
        let (lo, hi) = algo_widen::<V, P>(f(self.mignitude()), f(self.magnitude()));
        let lo = if const { NONNEG } { lo.max(V::ZERO) } else { lo };
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Intersects the input with `[d_lo, d_hi]` before evaluating, returning
    /// empty where the intersection is empty. The standard interval
    /// convention for partial functions (`asin`, `acos`, `ln`, `sqrt`).
    #[inline(always)]
    pub(crate) fn restrict(self, d_lo: V, d_hi: V) -> Self {
        Self::from_bounds_unchecked(self.lo.max(d_lo), self.hi.min(d_hi))
    }
}

// --- SpecializedCoreMath ------------------------------------------------------

impl<V: IntervalMathVector, W: WideningPolicy> SpecializedCoreMath<IntervalElem<V::Element>> for Interval<V, W> {
    /// Horner over primal (degenerate, unaugmented) coefficients: the
    /// interval multiply is genuine, but the addend is a point, so only the
    /// endpoints shift. Inherits the enclosing `mul`/`add`.
    #[inline(always)]
    fn mul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        self.mul_interval(m).add_interval(Self::from_primal(a))
    }

    /// Plain Horner, always. The trait default switches to an Estrin/ILP
    /// scheme at some policies, which forms `x^2`, `x^4`, ... explicitly,
    /// each a dependency-problem product that a zero-straddling interval
    /// pays for. Horner never squares `x`, so it is the tighter form here,
    /// and the ILP the default was buying does not exist for a serial
    /// interval accumulator anyway.
    #[inline(always)]
    fn poly<P: Policy, const N: usize>(self, coeffs: &[IntervalElem<V::Element>; N]) -> Self {
        let x = self;
        let mut res = Self::splat(coeffs[N - 1]);
        let mut i = const { N - 1 };
        while i > 0 {
            i -= 1;
            unsafe { core::hint::assert_unchecked(i < N) };
            res = res.mul_interval(x).add_interval(Self::splat(coeffs[i]));
        }
        res
    }

    #[inline(always)]
    fn poly_rev<P: Policy, const N: usize>(self, coeffs: &[IntervalElem<V::Element>; N]) -> Self {
        let x = self;
        let mut res = Self::splat(coeffs[0]);
        let mut i = 1;
        while i < N {
            unsafe { core::hint::assert_unchecked(i < N) };
            res = res.mul_interval(x).add_interval(Self::splat(coeffs[i]));
            i += 1;
        }
        res
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(self) -> Self {
        self.recip_interval()
    }

    #[inline(always)]
    fn approx_div<P: Policy>(self, divisor: Self) -> Self {
        // No approximate reciprocal: it has no per-lane error bound
        // (ground rule 2). Exact endpoint division.
        self.div_interval(divisor)
    }

    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        self.sqrt_interval().recip_interval()
    }

    /// `x^e` for a runtime scalar exponent: even exponents are even
    /// functions of `x` (mignitude/magnitude), odd ones are monotone.
    #[inline(always)]
    fn powi<P: Policy>(self, e: i32) -> Self {
        let f = |x: V| x.powi_p::<KernelPolicy<P>>(e);

        if e % 2 == 0 {
            self.even_inc::<P, true>(f)
        } else {
            self.monotone_inc::<P>(f)
        }
    }

    #[inline(always)]
    fn powiv<P: Policy>(self, e: Self::Signed) -> Self {
        // The exponent varies per lane, so evaluate both endpoints and take
        // the hull: valid for either parity, and the zero-crossing case is
        // covered by including 0^e via the mignitude endpoint.
        let a = self.lo.powiv_p::<KernelPolicy<P>>(e);
        let b = self.hi.powiv_p::<KernelPolicy<P>>(e);
        let m = self.mignitude().powiv_p::<KernelPolicy<P>>(e);

        let poison = self.is_empty();
        let (lo, hi) = algo_widen::<V, P>(a.min(b).min(m), a.max(b).max(m));
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }
}

// --- SpecializedTranscendentalMath --------------------------------------------

impl<V: IntervalMathVector, W: WideningPolicy> SpecializedTranscendentalMath<IntervalElem<V::Element>>
    for Interval<V, W>
{
    /// Sine and cosine over an interval, by quadrant analysis.
    ///
    /// If the interval spans a full period (or more) the answer is `[-1, 1]`
    /// exactly. Otherwise the extrema of the enclosure are the endpoint
    /// values, plus `+-1` wherever a critical point (`pi/2 + k*pi` for sine,
    /// `k*pi` for cosine) falls inside, detected by comparing the reduced
    /// argument's quadrant indices, all branchless.
    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        let poison = self.is_empty();

        // Full-period check: width >= 2pi means every value is attained.
        let width = self.hi - self.lo;
        let full = width.cmp_ge(<V as FloatConsts>::TAU) | !width.is_finite();

        let (s_lo, c_lo) = self.lo.sin_cos_p::<KernelPolicy<P>>();
        let (s_hi, c_hi) = self.hi.sin_cos_p::<KernelPolicy<P>>();

        // Quadrant index of each endpoint: floor(x / (pi/2)). Sine peaks at
        // q = 1 (mod 4) boundaries and troughs at q = 3 (mod 4). Cosine peaks
        // at q = 0 (mod 4) and troughs at q = 2 (mod 4). If the endpoints'
        // quadrant indices differ by enough to contain such a boundary, the
        // corresponding extremum is attained.
        //
        // The reduction runs through an ENCLOSURE of 2/pi, not a rounded
        // point value: `x * fl(2/pi)` rounds, and past |x| ~ 1e8 that rounding
        // is large enough to move the computed index across an integer, hiding
        // a crossed extremum and producing an enclosure that misses the true
        // value outright (measured: violations from |x| ~ 1e9, and a full
        // sign flip (2.0 wide) by 1e14). Widening q first can only ever
        // report EXTRA crossings, which is the safe direction.
        let (f2pi_lo, f2pi_hi) = <V as BoundedFloatConsts<V>>::FRAC_2_PI;
        let q_iv = self.mul_interval(Self::from_bounds_unchecked(f2pi_lo, f2pi_hi));
        let q_lo = q_iv.lo.floor();
        let q_hi = q_iv.hi.floor();

        // Number of quadrant boundaries crossed.
        let crossed = q_hi - q_lo;

        // Which residue classes are crossed: a boundary at index k is
        // crossed iff q_lo < k <= q_hi. Checking each of the four residues
        // reduces to "is there an integer in (q_lo, q_hi] congruent to r
        // mod 4", which is true iff floor((q_hi - r)/4) >= ceil((q_lo + 1 -
        // r)/4). Computed branchlessly below via the crossing count.
        let four = V::splat(<V::Element as thermite::element::FloatElement>::from_int(4));
        let has_residue = |r: i64| -> V::Mask {
            let rv = V::splat(<V::Element as thermite::element::FloatElement>::from_int(r));
            // Largest k <= q_hi with k = r (mod 4):
            let k = ((q_hi - rv) * four.reciprocal_p::<KernelPolicy<P>>())
                .floor()
                .mul_adde(four, rv);
            k.cmp_gt(q_lo) & crossed.cmp_ge(V::ZERO)
        };

        let sin_max = has_residue(1); // x = pi/2 + 2k*pi
        let sin_min = has_residue(3); // x = 3pi/2 + 2k*pi
        let cos_max = has_residue(0); // x = 2k*pi
        let cos_min = has_residue(2); // x = pi + 2k*pi

        let (s_l, s_h) = algo_widen::<V, P>(s_lo.min(s_hi), s_lo.max(s_hi));
        let (c_l, c_h) = algo_widen::<V, P>(c_lo.min(c_hi), c_lo.max(c_hi));

        let one = V::ONE;
        let neg_one = V::NEG_ONE;

        // Beyond the magnitude where the kernel itself keeps its accuracy, no
        // amount of correct quadrant reasoning helps, because the endpoint
        // values are simply wrong. `[-1, 1]` is always a valid enclosure of
        // sine and cosine, so that is the honest answer there.
        let full = full | self.magnitude().cmp_gt(trig_safe_magnitude::<V, KernelPolicy<P>>());

        let s_l = (sin_min | full).select(neg_one, s_l).max(neg_one);
        let s_h = (sin_max | full).select(one, s_h).min(one);
        let c_l = (cos_min | full).select(neg_one, c_l).max(neg_one);
        let c_h = (cos_max | full).select(one, c_h).min(one);

        let mk = |lo: V, hi: V| {
            Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
        };

        (mk(s_l, s_h), mk(c_l, c_h))
    }

    // Monotone increasing on their whole domains.
    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp_p::<KernelPolicy<P>>())
    }

    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        // Domain (0, inf): clamp the lower bound at zero, empty if entirely
        // non-positive.
        let all_np = self.hi.cmp_le(V::ZERO);
        let r = self
            .restrict(V::ZERO, V::INFINITY)
            .monotone_inc::<P>(|x| x.ln_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(all_np.select(V::INFINITY, r.lo), all_np.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        let all_np = self.hi.cmp_le(V::ZERO);
        let r = self
            .restrict(V::ZERO, V::INFINITY)
            .monotone_inc::<P>(|x| x.log_n_p::<KernelPolicy<P>, N>());
        Self::from_bounds_unchecked(all_np.select(V::INFINITY, r.lo), all_np.select(V::NEG_INFINITY, r.hi))
    }

    /// `x^e` over intervals: via `exp(e * ln(x))` on the positive domain,
    /// which is the composition the trait default would build anyway, but
    /// stated here so the domain restriction is explicit.
    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        let all_np = self.hi.cmp_le(V::ZERO);
        let base = self.restrict(V::ZERO, V::INFINITY);

        // Monotone in each argument separately (for base > 0), so the four
        // corner evaluations bracket the result.
        let c0 = base.lo.powf_p::<KernelPolicy<P>>(e.lo);
        let c1 = base.lo.powf_p::<KernelPolicy<P>>(e.hi);
        let c2 = base.hi.powf_p::<KernelPolicy<P>>(e.lo);
        let c3 = base.hi.powf_p::<KernelPolicy<P>>(e.hi);

        let lo = c0.min(c1).min(c2.min(c3));
        let hi = c0.max(c1).max(c2.max(c3));
        let (lo, hi) = algo_widen_n::<V, P>(lo, hi, ULP_POWF);

        let poison = self.is_empty() | e.is_empty() | all_np;
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    // --- monotone increasing on their whole domains ---
    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exph_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp2_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp10_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp_m1_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp2_m1_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.exp10_m1_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.cbrt_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.tanh_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.asinh_p::<KernelPolicy<P>>())
    }
    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        self.monotone_inc::<P>(|x| x.atan_p::<KernelPolicy<P>>())
    }

    // --- monotone increasing on a restricted domain ---
    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        // Domain (-1, inf).
        let empty = self.hi.cmp_le(V::NEG_ONE);
        let r = self
            .restrict(V::NEG_ONE, V::INFINITY)
            .monotone_inc::<P>(|x| x.ln_1p_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_le(V::ZERO);
        let r = self
            .restrict(V::ZERO, V::INFINITY)
            .monotone_inc::<P>(|x| x.log2_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_le(V::ZERO);
        let r = self
            .restrict(V::ZERO, V::INFINITY)
            .monotone_inc::<P>(|x| x.log10_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        // Domain [-1, 1], increasing.
        let empty = self.hi.cmp_lt(V::NEG_ONE) | self.lo.cmp_gt(V::ONE);
        let r = self
            .restrict(V::NEG_ONE, V::ONE)
            .monotone_inc::<P>(|x| x.asin_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        // Domain [-1, 1], DEcreasing.
        let empty = self.hi.cmp_lt(V::NEG_ONE) | self.lo.cmp_gt(V::ONE);
        let r = self
            .restrict(V::NEG_ONE, V::ONE)
            .monotone_dec::<P>(|x| x.acos_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        // Domain [1, inf), increasing.
        let empty = self.hi.cmp_lt(V::ONE);
        let r = self
            .restrict(V::ONE, V::INFINITY)
            .monotone_inc::<P>(|x| x.acosh_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        // Domain (-1, 1), increasing.
        let empty = self.hi.cmp_le(V::NEG_ONE) | self.lo.cmp_ge(V::ONE);
        let r = self
            .restrict(V::NEG_ONE, V::ONE)
            .monotone_inc::<P>(|x| x.atanh_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    /// `atanhc(x) = atanh(x)/x` on the same domain `(-1, 1)`, even with its
    /// minimum `1` at zero and increasing in `|x|` to `+inf` at the ends, so
    /// the mignitude/magnitude pair bounds it, unlike the parent `atanh`,
    /// which is monotone on the signed value.
    ///
    /// The domain handling is `atanh`'s: restrict first, and report empty for
    /// an interval that misses `(-1, 1)` entirely. Restricting before the
    /// mignitude matters, because an interval straddling an endpoint has its
    /// magnitude clamped to `1` rather than running off to a finite value
    /// outside the domain.
    #[inline(always)]
    fn atanhc<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_le(V::NEG_ONE) | self.lo.cmp_ge(V::ONE);
        let r = self
            .restrict(V::NEG_ONE, V::ONE)
            .even_inc::<P, true>(|x| x.atanhc_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    /// `sinh` is odd and increasing, `cosh` is even with its minimum at 0.
    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        (
            self.monotone_inc::<P>(|x| x.sinh_p::<KernelPolicy<P>>()),
            self.even_inc::<P, true>(|x| x.cosh_p::<KernelPolicy<P>>()),
        )
    }

    /// `sinhc(x) = sinh(x)/x` is even with its minimum `1` at zero and
    /// increasing in `|x|` over the whole line, the `cosh` shape, so the
    /// mignitude/magnitude pair bounds it directly. Contrast [`sinc`](Self::sinc)
    /// just below, whose oscillation makes the same pair wrong.
    #[inline(always)]
    fn sinhc<P: Policy>(self) -> Self {
        self.even_inc::<P, true>(|x| x.sinhc_p::<KernelPolicy<P>>())
    }

    /// `sinc(x) = sin(x)/x` is even with a global max of 1 at x = 0, but is
    /// NOT monotone in |x|. Enclosure: the endpoint values, hulled with the
    /// conservative global bounds whenever the interval is wide enough to
    /// contain an extremum (width >= pi) and with 1 whenever it straddles
    /// zero.
    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        let poison = self.is_empty();

        let a = self.lo.sinc_p::<KernelPolicy<P>>();
        let b = self.hi.sinc_p::<KernelPolicy<P>>();
        let (lo, hi) = algo_widen::<V, P>(a.min(b), a.max(b));

        // Global bounds of sinc: [-0.21723, 1]. Used wherever the interval
        // may contain an interior extremum (any interval wide enough to
        // cross a stationary point, conservatively: width >= pi).
        let wide = (self.hi - self.lo).cmp_ge(<V as FloatConsts>::PI) | !(self.hi - self.lo).is_finite();
        let straddles_zero = self.lo.cmp_le(V::ZERO) & self.hi.cmp_ge(V::ZERO);

        let g_lo = V::splat(<V::Element as thermite::element::FloatElement>::try_from_ratio(-2173, 10000).unwrap());
        let lo = wide.select(g_lo.min(lo), lo);
        let hi = (wide | straddles_zero).select(V::ONE.max(hi), hi).min(V::ONE);

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// `ln(1 - e^-x)`: monotone increasing in `x` on `(0, inf)`.
    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self {
        let _ = lnx;
        <Self as SpecializedTranscendentalMath<IntervalElem<V::Element>>>::ln1m_expnx::<P>(self)
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_lt(V::ZERO);
        let r = self
            .restrict(V::ZERO, V::INFINITY)
            .monotone_inc::<P>(|x| x.ln1m_expnx_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    /// Tangent: monotone increasing between consecutive poles, so an
    /// interval that stays on one branch maps to `[tan(lo), tan(hi)]`, and
    /// one that crosses a pole is the entire line.
    ///
    /// Pole crossing is detected three ways, any of which forces `entire`:
    /// the width is at least `pi` (some pole is inside), the branch indices
    /// `floor((x - pi/2)/pi)` of the endpoints differ, or the endpoint
    /// tangents come out of order or non-finite. That last one is the
    /// backstop that does not depend on the rounded pole positions, since
    /// `tan(lo) > tan(hi)` can only happen across a pole (tan is increasing
    /// on a branch).
    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let poison = self.is_empty();

        let width = self.hi - self.lo;
        let wide = width.cmp_ge(<V as BoundedFloatConsts<V>>::PI.0) | !width.is_finite();

        let inv_pi = <V as FloatConsts>::FRAC_1_PI;
        let half_pi = <V as FloatConsts>::FRAC_PI_2;
        let k_lo = ((self.lo - half_pi) * inv_pi).floor();
        let k_hi = ((self.hi - half_pi) * inv_pi).floor();
        let branch_differs = !k_lo.cmp_eq(k_hi);

        let t_lo = self.lo.tan_p::<KernelPolicy<P>>();
        let t_hi = self.hi.tan_p::<KernelPolicy<P>>();
        let disordered = t_lo.cmp_gt(t_hi) | !(t_lo.is_finite() & t_hi.is_finite());

        let entire = wide | branch_differs | disordered;

        let (lo, hi) = algo_widen::<V, P>(t_lo, t_hi);
        let lo = entire.select(V::NEG_INFINITY, lo);
        let hi = entire.select(V::INFINITY, hi);

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// `sin^2(x/2)`: the sine enclosure squared _as a set_ (mignitude /
    /// magnitude), never `s * s`, which goes negative for a zero-straddling
    /// sine. Range-clamped to `[0, 1]`.
    #[inline(always)]
    fn haversin<P: Policy>(self) -> Self {
        let s = <Self as SpecializedTranscendentalMath<IntervalElem<V::Element>>>::sin::<P>(
            self.mul_interval(Self::from_bounds_unchecked(V::HALF, V::HALF)),
        );
        let sq = s.square_interval();
        Self::from_bounds_unchecked(sq.lo.max(V::ZERO), sq.hi.min(V::ONE))
    }

    /// Monotone increasing on `[-1, inf)`.
    #[inline(always)]
    fn sqrt1pm1<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_lt(V::NEG_ONE);
        let r = self
            .restrict(V::NEG_ONE, V::INFINITY)
            .monotone_inc::<P>(|x| x.sqrt1pm1_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn log2_p1<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_le(V::NEG_ONE);
        let r = self
            .restrict(V::NEG_ONE, V::INFINITY)
            .monotone_inc::<P>(|x| x.log2_p1_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    #[inline(always)]
    fn log10_p1<P: Policy>(self) -> Self {
        let empty = self.hi.cmp_le(V::NEG_ONE);
        let r = self
            .restrict(V::NEG_ONE, V::INFINITY)
            .monotone_inc::<P>(|x| x.log10_p1_p::<KernelPolicy<P>>());
        Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
    }

    /// The N-th root: odd `N` is monotone increasing on the whole line, even
    /// `N` on `[0, inf)`.
    ///
    /// Overridden for CONTAINMENT, not just tightness: the trait default
    /// tests `is_negative()` (certainly-negative) to restore the sign of an
    /// odd root, which is false on a zero-straddling lane, so it would take
    /// `abs()` (`[0, mag]`) and silently drop every negative root.
    #[inline(always)]
    fn nth_root<P: Policy, const N: usize>(self) -> Self {
        if const { N == 0 } {
            return Self::from_bounds_unchecked(V::NEG_INFINITY, V::INFINITY);
        }
        if const { N == 1 } {
            return self;
        }

        let f = |x: V| x.nth_root_p::<KernelPolicy<P>, N>();

        if const { N & 1 == 1 } {
            self.monotone_inc::<P>(f)
        } else {
            let empty = self.hi.cmp_lt(V::ZERO);
            let r = self.restrict(V::ZERO, V::INFINITY).monotone_inc::<P>(f);
            Self::from_bounds_unchecked(empty.select(V::INFINITY, r.lo), empty.select(V::NEG_INFINITY, r.hi))
        }
    }

    /// `(1 + x)^n` on `x >= -1`: monotone in each argument separately, so
    /// the four corners bracket the box (same argument as `powf`).
    #[inline(always)]
    fn compound<P: Policy>(self, n: Self) -> Self {
        let empty = self.hi.cmp_lt(V::NEG_ONE);
        let base = self.restrict(V::NEG_ONE, V::INFINITY);
        let r = corners4::<V, P>(base, n, ULP_COMPOUND, |x, e| x.compound_p::<KernelPolicy<P>>(e));
        let poison = self.is_empty() | n.is_empty() | empty;
        Self::from_bounds_unchecked(poison.select(V::INFINITY, r.0), poison.select(V::NEG_INFINITY, r.1))
    }

    /// `(1 + x)^n - 1` on `x >= -1`: same monotonicity as `compound`, since
    /// subtracting a constant does not change it.
    #[inline(always)]
    fn compound_m1<P: Policy>(self, n: Self) -> Self {
        let empty = self.hi.cmp_lt(V::NEG_ONE);
        let base = self.restrict(V::NEG_ONE, V::INFINITY);
        let r = corners4::<V, P>(base, n, ULP_COMPOUND_M1, |x, e| x.compound_m1_p::<KernelPolicy<P>>(e));
        let poison = self.is_empty() | n.is_empty() | empty;
        Self::from_bounds_unchecked(poison.select(V::INFINITY, r.0), poison.select(V::NEG_INFINITY, r.1))
    }

    /// `x^e - 1` on `x >= 0`: four corners, like `powf`.
    #[inline(always)]
    fn powf_m1<P: Policy>(self, e: Self) -> Self {
        let empty = self.hi.cmp_lt(V::ZERO);
        let base = self.restrict(V::ZERO, V::INFINITY);
        let r = corners4::<V, P>(base, e, ULP_POWF_M1, |x, e| x.powf_m1_p::<KernelPolicy<P>>(e));
        let poison = self.is_empty() | e.is_empty() | empty;
        Self::from_bounds_unchecked(poison.select(V::INFINITY, r.0), poison.select(V::NEG_INFINITY, r.1))
    }
}

/// Hull of a two-argument function over the four corners of a box, widened.
/// Valid whenever `f` is monotone in each argument separately on the box
/// (the extrema of such a function lie at vertices).
#[inline(always)]
fn corners4<V: IntervalMathVector, P: Policy>(
    a: Interval<V, impl WideningPolicy>,
    b: Interval<V, impl WideningPolicy>,
    ulps: u32,
    f: impl Fn(V, V) -> V,
) -> (V, V) {
    let c0 = f(a.lo, b.lo);
    let c1 = f(a.lo, b.hi);
    let c2 = f(a.hi, b.lo);
    let c3 = f(a.hi, b.hi);
    algo_widen_n::<V, P>(c0.min(c1).min(c2.min(c3)), c0.max(c1).max(c2.max(c3)), ulps)
}

// --- SpecializedSpatialMath ---------------------------------------------------

impl<V: IntervalMathVector, W: WideningPolicy> SpecializedSpatialMath<IntervalElem<V::Element>> for Interval<V, W> {
    /// `|x|^2` as a set: the dependency-correct square.
    #[inline(always)]
    fn l2_norm_squared<P: Policy>(self) -> Self {
        self.square_interval()
    }

    /// `|x|` as a set: `[mig, mag]`, exact.
    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        self.abs_interval()
    }

    /// For a 1-D value the L1 norm is the absolute value, exact.
    #[inline(always)]
    fn l1_norm<P: Policy>(self) -> Self {
        self.abs_interval()
    }

    /// `hypot` is increasing in `|x|` and `|y|`: `[hypot(mig), hypot(mag)]`.
    ///
    /// The trait default's max/scale scheme is built on certainly-compares
    /// (`max.cmp_eq(ZERO)`) that go wide the moment an interval touches
    /// zero, ending in `[0, inf]`. This form is tight.
    #[inline(always)]
    fn hypot<P: Policy>(self, other: Self) -> Self {
        let poison = self.is_empty() | other.is_empty();
        let (lo, hi) = algo_widen::<V, P>(
            self.mignitude().hypot_p::<KernelPolicy<P>>(other.mignitude()),
            self.magnitude().hypot_p::<KernelPolicy<P>>(other.magnitude()),
        );
        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, lo.max(V::ZERO)),
            poison.select(V::NEG_INFINITY, hi),
        )
    }

    #[inline(always)]
    fn hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        let (migs, mags, poison) = split_mig_mag(values);
        let (lo, hi) = algo_widen::<V, P>(
            V::hypot_n_p::<KernelPolicy<P>, N>(migs),
            V::hypot_n_p::<KernelPolicy<P>, N>(mags),
        );
        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, lo.max(V::ZERO)),
            poison.select(V::NEG_INFINITY, hi),
        )
    }

    /// Decreasing in every `|x_i|`: `[inv_hypot(mag), inv_hypot(mig)]`.
    #[inline(always)]
    fn inv_hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        let (migs, mags, poison) = split_mig_mag(values);
        let (lo, hi) = algo_widen::<V, P>(
            V::inv_hypot_n_p::<KernelPolicy<P>, N>(mags),
            V::inv_hypot_n_p::<KernelPolicy<P>, N>(migs),
        );
        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, lo.max(V::ZERO)),
            poison.select(V::NEG_INFINITY, hi),
        )
    }
}

/// Mignitude and magnitude arrays of an interval array, plus the union of
/// the empty masks. Hand-rolled loops (no `array::map` in SIMD code).
#[inline(always)]
fn split_mig_mag<V: IntervalMathVector, W: WideningPolicy, const N: usize>(
    values: [Interval<V, W>; N],
) -> ([V; N], [V; N], V::Mask) {
    let mut migs = [V::ZERO; N];
    let mut mags = [V::ZERO; N];
    let mut poison: V::Mask = thermite::mask::GenericMask::FALSY;
    let mut i = 0;
    while i < N {
        migs[i] = values[i].mignitude();
        mags[i] = values[i].magnitude();
        poison = poison | values[i].is_empty();
        i += 1;
    }
    (migs, mags, poison)
}

// --- SpecializedRealMath ------------------------------------------------------

impl<V: IntervalMathVector, W: WideningPolicy> SpecializedRealMath<IntervalElem<V::Element>> for Interval<V, W> {
    /// Four-quadrant arctangent over intervals.
    ///
    /// Conservative: evaluates the four corner combinations and takes their
    /// hull, then widens. Correct wherever `atan2` is continuous on the
    /// input box. Boxes straddling the negative real axis (where `atan2`
    /// jumps from `+pi` to `-pi`) are detected and given the full
    /// `[-pi, pi]` enclosure rather than a wrong-but-narrow one.
    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        let poison = self.is_empty() | x.is_empty();

        // Branch cut: x may be negative while y straddles zero.
        let straddles_cut = x.lo.cmp_lt(V::ZERO) & self.lo.cmp_le(V::ZERO) & self.hi.cmp_ge(V::ZERO);

        let c0 = self.lo.atan2_p::<KernelPolicy<P>>(x.lo);
        let c1 = self.lo.atan2_p::<KernelPolicy<P>>(x.hi);
        let c2 = self.hi.atan2_p::<KernelPolicy<P>>(x.lo);
        let c3 = self.hi.atan2_p::<KernelPolicy<P>>(x.hi);

        let lo = c0.min(c1).min(c2.min(c3));
        let hi = c0.max(c1).max(c2.max(c3));
        let (lo, hi) = algo_widen::<V, P>(lo, hi);

        let pi_lo = <V as BoundedFloatConsts<V>>::PI.0;
        let pi_hi = <V as BoundedFloatConsts<V>>::PI.1;

        let lo = straddles_cut.select(-pi_hi, lo);
        let hi = straddles_cut.select(pi_hi, hi);
        let _ = pi_lo;

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// The Heaviside step over intervals: `1` where `self` is certainly at
    /// or above `edge`, `0` where certainly below, and `[0, 1]` where the
    /// comparison is uncertain.
    ///
    /// Overridden for CONTAINMENT: the trait default is `ONE.zz(cmp_ge)`,
    /// and certainly-`ge` is false on an uncertain lane, so it would answer
    /// `0` where the true image is `{0, 1}`.
    #[inline(always)]
    fn step<P: Policy>(self, edge: Self) -> Self {
        let poison = self.is_empty() | edge.is_empty();
        let certainly_ge = self.lo.cmp_ge(edge.hi);
        let certainly_lt = self.hi.cmp_lt(edge.lo);
        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, certainly_ge.select(V::ONE, V::ZERO)),
            poison.select(V::NEG_INFINITY, certainly_lt.select(V::ZERO, V::ONE)),
        )
    }

    /// Increasing in both arguments: `[f(a.lo, b.lo), f(a.hi, b.hi)]`.
    #[inline(always)]
    fn logaddexp<P: Policy>(self, other: Self) -> Self {
        let poison = self.is_empty() | other.is_empty();
        let (lo, hi) = algo_widen::<V, P>(
            self.lo.logaddexp_p::<KernelPolicy<P>>(other.lo),
            self.hi.logaddexp_p::<KernelPolicy<P>>(other.hi),
        );
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Increasing in `a`, decreasing in `b`, defined for `a >= b`. A box that
    /// touches the diagonal `a == b` reaches `-inf`, and one entirely below
    /// it (`a.hi < b.lo`) is out of domain and empty.
    #[inline(always)]
    fn logsubexp<P: Policy>(self, other: Self) -> Self {
        let touches_diag = self.lo.cmp_lt(other.hi);
        let below = self.hi.cmp_lt(other.lo);
        let poison = self.is_empty() | other.is_empty() | below;

        let (lo, hi) = algo_widen::<V, P>(
            self.lo.logsubexp_p::<KernelPolicy<P>>(other.hi),
            self.hi.logsubexp_p::<KernelPolicy<P>>(other.lo),
        );
        let lo = touches_diag.select(V::NEG_INFINITY, lo);

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Increasing in every argument.
    #[inline(always)]
    fn logsumexp_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        let mut los = [V::ZERO; N];
        let mut his = [V::ZERO; N];
        let mut poison: V::Mask = thermite::mask::GenericMask::FALSY;
        let mut i = 0;
        while i < N {
            los[i] = values[i].lo;
            his[i] = values[i].hi;
            poison = poison | values[i].is_empty();
            i += 1;
        }
        let (lo, hi) = algo_widen::<V, P>(
            V::logsumexp_n_p::<KernelPolicy<P>, N>(los),
            V::logsumexp_n_p::<KernelPolicy<P>, N>(his),
        );
        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Smoothstep is the _clamped_ polynomial here, always: `t` is formed
    /// from the edges with enclosing arithmetic (an edge span containing
    /// zero gives `t` = entire, hence `[0, 1]`), set-clamped to `[0, 1]`,
    /// and the polynomial (increasing on `[0, 1]`) maps its endpoints. The
    /// inner kernel is called with `None` edges on in-range values, so its
    /// clamp policy no longer matters.
    #[inline(always)]
    fn smoothstep<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;
        let mut poison = self.is_empty();
        if let Some((a, b)) = edges {
            poison = poison | a.is_empty() | b.is_empty();
            t = t.sub_interval(a).div_interval(b.sub_interval(a));
        }
        // Set-clamp to [0, 1]: `max_interval`/`min_interval` are the endpoint
        // maps, so a `t` entirely below 0 becomes [0, 0] and entirely above 1
        // becomes [1, 1] (the saturated points, not empty).
        let t = t
            .max_interval(Self::from_bounds_unchecked(V::ZERO, V::ZERO))
            .min_interval(Self::from_bounds_unchecked(V::ONE, V::ONE));

        let (lo, hi) = algo_widen::<V, P>(
            t.lo.smoothstep_p::<KernelPolicy<P>, N>(None),
            t.hi.smoothstep_p::<KernelPolicy<P>, N>(None),
        );
        // p(0) = 0 and p(1) = 1 exactly for every smoothstep polynomial, so
        // saturated endpoints are pinned rather than smeared by the widening
        // floor. A fully clamped lane is exactly [0, 0] or [1, 1].
        let lo = t.lo.cmp_eq(V::ONE).select(V::ONE, lo.max(V::ZERO));
        let hi = t.hi.cmp_eq(V::ZERO).select(V::ZERO, hi.min(V::ONE));

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }
}
