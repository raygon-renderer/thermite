//! The Fresnel integrals `C(x) = int_0^x cos(pi t^2/2) dt` and
//! `S(x) = int_0^x sin(pi t^2/2) dt`.
//!
//! # Two regions
//!
//! Below a crossover (2.5265 at both precisions) the integrands are summed
//! directly: `C(x)/x` and `S(x)/x^3` are both smooth functions of `w = x^4`, fitted
//! as Chebyshev series and summed by Clenshaw. Above it the standard auxiliary
//! form
//!
//! ```text
//! C = 1/2 + f sin t - g cos t,   S = 1/2 - f cos t - g sin t,   t = pi x^2 / 2
//! ```
//!
//! with `f = P(u)/(pi x)` and `g = Q(u)/(pi^2 x^3)` for `u = 1/(pi x^2)^2`, both
//! `P` and `Q` plain Horner polynomials tending to 1.
//!
//! # Why Chebyshev below and Horner above
//!
//! Measurement, not symmetry. The small-argument fit in the monomial basis has an
//! error amplification (`sum |c_k| |w|^k / |f|`) of 3482 at this crossover, and
//! 1.26e5 if the crossover moves to 3. The same fit in the Chebyshev basis summed
//! by Clenshaw sits at 5.8 and 7.0. Monomial Horner would cap the f64 kernel at
//! about 4e-13. Clenshaw costs two operations per term against Horner's one and
//! buys three orders of magnitude, which is also what lets the crossover sit far
//! enough out for the auxiliaries to be well conditioned. Their own amplification
//! is 1.02 there, so they keep Horner.
//!
//! # The phase
//!
//! `t = pi x^2 / 2` computed as `x*x*0.5` is worthless long before the function is:
//! measured against a 45-digit oracle, `sin(pi*(x*x*0.5))` in binary64 is 5.2e-13
//! off at `x = 123`, 9.8e-11 at 1234, **5.3e-6 at 98765**, and returns the wrong
//! sign by `x ~ 1e9`. Since `C` and `S` are `1/2` plus a term of size `1/(pi x)`,
//! that error lands directly on the result.
//!
//! [`phase_half_x2`] fixes it in about ten operations, and the fix is exact: `x*x`
//! splits as `p + e` with `e` always representable, halving is exact, and each half
//! reduces mod 2 exactly by Sterbenz. Both halves must be reduced _before_ being
//! added. `|e/2|` reaches `ulp(x^2)/4`, which is 32 at `x = 1e9`, and adding that
//! to an already-reduced `p/2` rounds the latter's low bits straight off. Measured
//! 2.80 ulp (`C`) and 2.64 (`S`) in f64 over `x` from 1e-4 to 1e15, and 2.14 / 3.40
//! in f32 out to 1e7. The naive phase alone is worth thousands of ulp there.
//!
//! Above `x = 1.147e16` (f64) / `2.136e7` (f32) the correction has fallen under
//! half an ulp of `1/2` and both functions are exactly `1/2`. Below that `x^2`
//! cannot overflow, so the phase needs no range guard.

use thermite::{
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use super::chebyshev::chebyshev_series;

/// `x^2/2 mod 2`, the argument for `sincos_pi`, to full precision for every `x`
/// whose square is finite.
///
/// See the module docs for why the two words are reduced separately. Below
/// `Average` the residual is dropped entirely and this is the naive `x*x/2`,
/// which is accurate only while `x^2` is exact.
#[inline(always)]
pub fn phase_half_x2<P, E, V>(x: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    // v - 2*round(v/2). Exact: `round` is ties-to-even and the subtraction is
    // Sterbenz-exact whenever |v - 2k| <= 1 <= |v|/2.
    let rem2 = |v: V| -> V { (v * V::HALF).round().nmul_adde(V::TWO, v) };

    let p = x * x;

    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        return rem2(p * V::HALF);
    }

    // `mul_sub`, deliberately, not `mul_sube`. The residual has to be EXACT: it is
    // multiplied by nothing and added to a quantity of size 1, so any error in it is
    // an error in the phase, and only a fused (or correctly-rounded emulated)
    // multiply-add produces it. On a backend with hardware FMA the two spellings emit
    // the same instruction. On one without, this is the one place in the kernel that
    // pays for the emulation, which buys the entire large-argument accuracy claim.
    // A Veltkamp split would be the same result in about eight ordinary operations,
    // but needs a per-format splitting constant, and the emulation is already
    // correctly rounded here.
    let e = x.mul_sub(x, p);

    rem2(rem2(p * V::HALF) + rem2(e * V::HALF))
}

/// `(S(x), C(x))`, in SciPy's order.
///
/// `cheb_c` and `cheb_s` are Chebyshev coefficients for `C(x)/x` and `S(x)/x^3` in
/// `w = x^4` mapped onto `[-1, 1]` by `w*map - 1`. `aux_p` and `aux_q` are ascending
/// monomial coefficients in `u = 1/(pi x^2)^2`. `x0` is the crossover and `cutoff`
/// the point above which both functions are `1/2`.
#[inline(always)]
pub fn fresnel_with<P, E, V, const NC: usize, const NS: usize, const NP: usize, const NQ: usize>(
    x: V,
    x0: E,
    map: E,
    cutoff: E,
    cheb_c: &[E; NC],
    cheb_s: &[E; NS],
    aux_p: &[E; NP],
    aux_q: &[E; NQ],
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let ax = x.abs();
    let q = ax * ax;
    let is_small = ax.cmp_le(V::splat(x0));

    if !const { P::POLICY.avoid_branching } && is_small.all() {
        let (c, s) = fresnel_small::<P, E, V, NC, NS>(ax, q, map, cheb_c, cheb_s);
        // Both functions are odd and both are positive for x > 0, so the sign is a
        // copysign rather than a branch. It carries -0.0 through unchanged.
        return (s.copysign(x), c.copysign(x));
    }

    // One division for the whole branch: r = 1/(pi x^2), and then
    // f = P(u) r x = P(u)/(pi x) and g = Q(u) u x = Q(u)/(pi^2 x^3).
    let r = V::FRAC_1_PI.approx_div_p::<P>(q);
    let u = r * r;
    let f = (u.poly_n_p::<P, NP>(aux_p) * r) * ax;
    let g = (u.poly_n_p::<P, NQ>(aux_q) * u) * ax;

    let (sin_t, cos_t) = phase_half_x2::<P, E, V>(ax).sincos_pi_p::<P>();

    // Two chained FMAs apiece: 1/2 + f sin t - g cos t and 1/2 - f cos t - g sin t.
    let mut c = g.nmul_adde(cos_t, f.mul_adde(sin_t, V::HALF));
    let mut s = g.nmul_adde(sin_t, f.nmul_adde(cos_t, V::HALF));

    if const { P::POLICY.avoid_branching } || thermite::unlikely(is_small.any()) {
        let (cs, ss) = fresnel_small::<P, E, V, NC, NS>(ax, q, map, cheb_c, cheb_s);
        c = is_small.select(cs, c);
        s = is_small.select(ss, s);
    }

    // Past the cutoff `f` and `g` have fallen under half an ulp of 1/2. Naming it is
    // a shortcut and also what keeps the infinities right, since `q` is
    // +inf there and the auxiliary branch would otherwise take the trig of a NaN.
    let done = ax.cmp_gt(V::splat(cutoff));
    c = done.select(V::HALF, c);
    s = done.select(V::HALF, s);

    (s.copysign(x), c.copysign(x))
}

/// `(C, S)` from the Chebyshev series in `w = x^4`, for `|x|` under the crossover.
#[inline(always)]
fn fresnel_small<P, E, V, const NC: usize, const NS: usize>(
    ax: V,
    q: V,
    map: E,
    cheb_c: &[E; NC],
    cheb_s: &[E; NS],
) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    P: Policy,
{
    // w = x^4, mapped onto [-1, 1] by a single FMA.
    let w = q * q;
    let t = w.mul_sube(V::splat(map), V::ONE);
    let a = chebyshev_series::<P, E, V, 1, NC, true>(t, cheb_c);
    let b = chebyshev_series::<P, E, V, 1, NS, true>(t, cheb_s);
    (ax * a, (ax * q) * b)
}
