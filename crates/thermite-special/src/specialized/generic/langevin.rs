//! Langevin function `L(x) = coth(x) - 1/x` and its inverse, shared by every real
//! element type. The per-precision pieces (the polynomial tables, and Newton vs
//! Halley for the inverse) come in from `ps.rs`/`pd.rs`.
//!
//! # Forward
//!
//! `coth(x)` and `1/x` both grow like `1/x` while their difference is only `x/3`, so
//! the direct form loses relative accuracy as `3u/x^2`. That is not a corner case, at
//! `x = 0.1` it is already 300 ulp. Below the crossover `X0 = 2` the function is therefore an odd
//! polynomial `x * p(x^2)` (minimax, fitted to `L(x)/x`), and above it
//!
//! ```math
//! L(x) = 1 - \frac{1}{x} + \frac{2q}{1 - q}, \qquad q = e^{-2x}
//! ```
//!
//! which is `1 - small` and cancels nothing. `q` rather than `expm1(2x)` because it
//! never overflows (`q -> 0` is the correct limit and `L(inf) = 1` falls out), and it
//! also gives the derivative for free: `csch^2(x) = 4q/(1-q)^2`.
//!
//! # Inverse
//!
//! `L^-1` has a simple pole at `y = 1`, and near it `x = 1/(1-y) - 2x^2 e^{-2x}`, so for
//! `y >= 0.85` the seed is `1/(1-y)` itself (relative error `2.2e-5` at 0.85, `4e-8` at
//! 0.9, below `u` past 0.95). Below 0.85 the seed is `y * q(y^2) / (1 - y^2)` with `q`
//! a minimax fit of `L^-1(y)(1-y^2)/y`, the same shape as Cohen's Pade `(3-y^2)/(1-y^2)`,
//! which is what the vMF literature calls the Banerjee estimator. Both seeds share the
//! one division `1/(1-y^2)`, since `1/(1-y) = (1+y)/(1-y^2)`.
//!
//! The seed is then polished with one step. f32 takes Newton, `x <- x - f/f'` for
//! `f = L(x) - y`: the error squares with constant ~1, so the `8e-5` seed lands at
//! `~6e-9`, past f32. f64 takes Halley, `x <- x - 2ff'/(2f'^2 - ff'')`: `L''` is
//! nearly free on both branches (`2 csch^2 coth - 2/x^3` from the same `q`, and the
//! differentiated identity below 2), the step still has one division, and the error
//! cubes with constant `f'''/(6f') - (f''/(2f'))^2` at most ~0.07 (small `y`) and
//! ~2e-4 at the tail crossover. So f64's `1.1e-6` seed (deg 8 rather than f32's deg 4
//! exactly for this) reaches full precision in one step where Newton needed two, i.e.
//! a second exp and division. `L` is monotone and concave on `x > 0`, so from any
//! positive seed either iteration is safe without safeguards.
//!
//! The residual is formed as `((1-y) - 1/x) + 2q/(1-q)` on the large branch rather
//! than `L(x) - y`: `1 - y` is exact for `y >= 0.5`, and that keeps the step accurate
//! to `u` even where `L(x)` is within an ulp of 1, which `L(x) - y` cannot do (its
//! error, `u`, divided by `L' ~ 1/x^2`, would grow as `u x`).
//!
//! `L^-1` itself is ill-conditioned near 1 (a relative error `u` in `y` moves the
//! result by `u/(1-y)`), so callers with `1 - y` in hand should compute it exactly
//! before rounding, exactly as they would for `acos` near 1.

use thermite::{
    element::{FloatElement, FloatElementWithBits},
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{
            Policy, PrecisionPolicy,
            policies::{AveragePrecision, CheckOverflow, CmpLessPrecision},
        },
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;

/// Crossover between the odd polynomial and the `1 - 1/x + 2q/(1-q)` form.
///
/// The tables in `ps.rs`/`pd.rs` (and the compensated one) are fitted on `[0, X0]`.
const X0_NUM: thermite::LargeInt = 2;

/// Seed crossover for the inverse: `1/(1-y)` above, `y q(y^2)/(1-y^2)` below.
const Y1_NUM: thermite::LargeInt = 85;
const Y1_DEN: thermite::LargeInt = 100;

/// Refinement steps for a given precision policy. One step (Newton from f32's ~8e-5
/// seed, Halley from f64's ~1e-6 seed, see the module docs) reaches the type's full
/// precision. `Reference` takes a second for good measure, `Worst` ships the seed.
#[inline(always)]
pub const fn refine_steps(precision: PrecisionPolicy) -> usize {
    match precision {
        PrecisionPolicy::Worst => 0,
        PrecisionPolicy::Reference => 2,
        _ => 1,
    }
}

/// The large-branch pieces for `x >= X0`: `q = e^{-2x}`, `1 - q`, and the one
/// reciprocal `r = 1/(x(1-q))` that everything else is a product with:
///
/// ```text
/// L  = 1 - 1/x + 2q/(1-q)      = (x - 1 + q(x+1)) r
/// L' = 1/x^2 - 4q/(1-q)^2      = ((1-q)^2 - 4q x^2) r^2
/// ```
///
/// Both numerators are free of cancellation on `x >= 2` (`q <= e^-4`, so `4qx^2` is at
/// most 7% of `(1-q)^2` at the crossover and vanishes beyond it), and the whole branch
/// costs the exp, the division and a handful of FMAs.
#[inline(always)]
fn large_parts<P, E, V>(x: V) -> (V, V, V)
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    // The exp never needs the Best tier: `x >= 2` keeps `-2x` inside the single-scale
    // range of the Average kernel (its low end saturates to exactly 0, which is what
    // `q` wants there anyway), and Average is already 1-2 ulp, which is below what
    // `L`'s own rounding contributes. Best's two-part scaling and range gate cost more
    // than the whole rest of the branch (measured 2x on f32x8), for nothing here.
    // Overflow checks are off for the same reason: the caller owns the edges.
    type ExpPolicy<P> = CheckOverflow<CmpLessPrecision<P, AveragePrecision<P>>, false>;

    // Argument clamped where q has long underflowed to 0 (x > 52 in f32, 372 in f64):
    // without the range gate, a huge argument's reduction leaves garbage in the
    // polynomial that then multiplies the zeroed scale.
    let xc = x.min(V::splat(<E as FloatElement>::ConstInt::<400>::VALUE));
    let q = (-(xc + xc)).exp_p::<ExpPolicy<P>>();
    let omq = V::ONE - q;
    let r = (x * omq).reciprocal_p::<P>();
    (q, omq, r)
}

/// `L'(x)` and `L''(x)` on the large branch from [`large_parts`]:
///
/// ```text
/// L'  = 1/x^2 - csch^2 x
/// L'' = 2 csch^2 x coth x - 2/x^3,   coth x = 1 + 2q/(1-q)
/// ```
///
/// with `1/x = (1-q) r`, `1/(1-q) = x r`, `csch^2 = 4q/(1-q)^2`.
#[inline(always)]
fn large_derivs<V: FloatVector>(x: V, q: V, omq: V, r: V) -> (V, V) {
    let rcp = omq * r;
    let d = x * r;
    let w = (q + q) * d;
    let csch2 = w * (d + d);
    let rcp2 = rcp * rcp;
    let dl = rcp2 - csch2;
    let d2l = rcp2.nmul_adde(rcp, csch2.mul_adde(w, csch2));
    (dl, d2l + d2l)
}

/// `L(x)` (or, with `ONE_MINUS`, `1 - L(x)`) and `L'(x)` together.
///
/// `L'` costs no transcendental of its own: on the small branch it is `1 - L^2 - 2L/x`
/// (which is exact algebra, and cancels only ~2 bits there since `L ~ x/3`), on the
/// large branch `1/x^2 - csch^2(x)` from the same `q`.
///
/// The complement is not a second kernel: on the large branch `1 - L = (1 - q(2x+1)) r`
/// with the same `q` and division (no cancellation, `q(2x+1) <= 0.092` at `x = 2`), on
/// the small one `1 - x p` where `1 - L >= 0.46`. It exists because `1 - L(x)` is what
/// sits against `L^-1`'s pole in the vMF convolution, and forming it from `L` loses
/// every digit once `L` rounds to 1 (`x > 1/u`, i.e. sharpness ~1e7 in f32).
#[inline(always)]
pub fn langevin_primal<P, E, V, const N: usize, const ONE_MINUS: bool>(x: V, small: &[E; N]) -> (V, V)
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let ax = x.abs();
    let is_small = ax.cmp_le(V::splat(<E as FloatElement>::ConstInt::<X0_NUM>::VALUE));

    // Odd polynomial on the signed input, so the sign rides along for free.
    let p = (x * x).poly_p::<P, _>(small);
    let l_small = x * p;
    // 1 - L*(L + 2/x) with L = x p: 2L/x = 2p, no division needed, as -L*L + (1 - 2p).
    let mut dl = l_small.nmul_adde(l_small, p.nmul_adde(V::TWO, V::ONE));
    let mut l = if const { ONE_MINUS } { V::ONE - l_small } else { l_small };

    if const { P::POLICY.avoid_branching } || !is_small.all() {
        let (q, omq, r) = large_parts::<P, E, V>(ax);
        let lpos = q.mul_adde(ax + V::ONE, ax - V::ONE) * r; // L(|x|)
        let big = if const { ONE_MINUS } {
            // 1 - L(x): (1 - q(2x+1)) r for x > 0, and 1 + L(|x|) for x < 0 (no
            // cancellation either way, so the sign only picks a form).
            let onem = q.nmul_adde(ax.mul_adde(V::TWO, V::ONE), V::ONE) * r;
            x.select_negative(V::ONE + lpos, onem)
        } else {
            lpos.copysign(x)
        };
        l = is_small.select(l, big);
        dl = is_small.select(dl, large_derivs(ax, q, omq, r).0);

        // x = inf is inf * 0 above (r = 0 against infinite numerators). The limits are
        // L = ±1 (so 1 - L = 0 or 2), L' = 0.
        let is_inf = ax.cmp_eq(V::INFINITY);
        let l_inf = if const { ONE_MINUS } {
            x.select_negative(V::TWO, V::ZERO)
        } else {
            V::ONE.copysign(x)
        };
        l = is_inf.select(l_inf, l);
        dl = dl.nz(is_inf);
    }

    (l, dl)
}

/// `L^-1(y)` (or, with `ONE_MINUS`, `L^-1(1 - t)` from `t` directly): seed plus
/// [`refine_steps`] of Newton (`HALLEY = false`) or Halley, see the module docs. Halley's
/// `L''` costs a reciprocal and a few FMAs on top of Newton, which buys f64 a whole
/// second step. f32 is already done after one Newton and would only pay.
///
/// The complement is a re-entry point, not a second implementation: the kernel already
/// works in `t = 1 - y` (the tail seed is `1/t`, the large-branch residual consumes `t`),
/// so `ONE_MINUS` only changes where `t` comes from, exact from the caller instead of
/// rounded from `y`. That is the whole difference between a result conditioned by
/// `1/(1-y)` and one accurate to `u` at any sharpness.
#[inline(always)]
pub fn inv_langevin<P, E, V, const NF: usize, const NI: usize, const HALLEY: bool, const ONE_MINUS: bool>(
    input: V,
    small: &[E; NF],
    seed_poly: &[E; NI],
) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    // (signed y, |y|, 1 - |y|). In complement mode t is the input. y = 1 - t is exact
    // for t in [0.5, 2] and its rounding is harmless below (the result is O(1) there),
    // and a negative y (t > 1) simply falls back to the rounded 1 - |y|.
    let (y_in, y, t) = if const { ONE_MINUS } {
        let y_in = V::ONE - input;
        let y = y_in.abs();
        (y_in, y, y_in.select_negative(V::ONE - y, input))
    } else {
        let y = input.abs();
        (input, y, V::ONE - y) // exact for y >= 0.5 (Sterbenz), which is where it matters
    };
    let opy = V::ONE + y;
    // 1/(1-y^2) as the product of the two exact-ish factors, never as 1 - y*y: without
    // FMA that would lose the whole low half near y = 1.
    let inv = (t * opy).reciprocal_p::<P>();

    let use_tail = y.cmp_ge(V::splat(<E as FloatElement>::ConstRatio::<Y1_NUM, Y1_DEN>::VALUE));
    let s = y * y;
    let num = use_tail.select(opy, y * s.poly_p::<P, _>(seed_poly));
    let mut x = num * inv;

    let x0 = V::splat(<E as FloatElement>::ConstInt::<X0_NUM>::VALUE);

    let steps = const { refine_steps(P::POLICY.precision) };
    let mut i = 0;
    while i < steps {
        let is_small = x.cmp_le(x0);

        // Small branch: L = x p, L' = 1 - L(L + 2p), residual L - y (no cancellation
        // issue: both are ~y and the quotient is against L' ~ 1/3). L'' from
        // differentiating that identity: L'' = -2 L L' - 2 (L' - p)/x. It cancels near
        // 0 (L'' ~ -2x/15), which Halley's correction term does not mind. The clamp
        // only keeps x = 0 (y = 0, whose step is exactly zero anyway) finite.
        let p = (x * x).poly_p::<P, _>(small);
        let l = x * p;
        let mut r = l - y;
        let mut dl = l.nmul_adde(l, p.nmul_adde(V::TWO, V::ONE));
        let mut d2l = V::ZERO;
        if const { HALLEY } {
            let rcp = x.max(V::MIN_POSITIVE).reciprocal_p::<P>();
            d2l = (l + l).nmul_sube(dl, (rcp + rcp) * (dl - p));
        }

        if const { P::POLICY.avoid_branching } || !is_small.all() {
            let (q, omq, rr) = large_parts::<P, E, V>(x);
            // The accurate form of L(x) - y (module docs), over the shared denominator:
            //   ((1-y) - 1/x) + 2q/(1-q) = ((1-q)(tx - 1) + 2qx) / (x(1-q))
            // `tx - 1` is one FMA and both terms are O(q x), so the residual carries no
            // rounding of an O(1) quantity, which is what keeps the step at ~u even
            // where the seed is already within an ulp.
            let rbig = omq.mul_adde(t.mul_sube(x, V::ONE), (q + q) * x) * rr;
            let (dbig, d2big) = large_derivs(x, q, omq, rr);
            r = is_small.select(r, rbig);
            dl = is_small.select(dl, dbig);
            if const { HALLEY } {
                d2l = is_small.select(d2l, d2big);
            }
        }

        if const { HALLEY } {
            // One division: x - 2 f f' / (2 f'^2 - f f'').
            let two_dl = dl + dl;
            x -= (r * two_dl) / dl.mul_sube(two_dl, r * d2l);
        } else {
            x -= r / dl;
        }
        i += 1;
    }

    // y = 1 (t = 0) is the pole (the seed already gives +inf there, but a step on it is
    // 0/0), and y > 1 (t < 0) is out of the domain. In complement mode these are read off
    // `t` itself: a `t` below u/2 rounds `1 - t` to exactly 1, and is a perfectly good
    // finite input. The negative branch (t > 1) went through the rounded `1 - |y|`, so
    // its pole (t = 2) is read off `y` as before.
    let (at_pole, out_of_domain) = if const { ONE_MINUS } {
        let neg = y_in.is_negative();
        (
            input.cmp_eq(V::ZERO) | (neg & y.cmp_ge(V::ONE)),
            input.cmp_lt(V::ZERO) | (neg & y.cmp_gt(V::ONE)),
        )
    } else {
        (y.cmp_ge(V::ONE), y.cmp_gt(V::ONE))
    };
    x = at_pole.select(V::INFINITY, x);
    if const { P::POLICY.check_overflow } {
        x = (out_of_domain | input.is_nan()).select(V::NAN, x);
    }

    x.copysign(y_in)
}
