//! Elliptic integrals.
//!
//! The complete integrals K and E use the arithmetic-geometric mean (AGM), which is
//! branchless and needs only one `sqrt` per iteration - far cheaper for SIMD than the
//! Carlson duplication algorithm used for the incomplete forms (see [`carlson_rf`] etc.).
//!
//! The algorithms are classical (Gauss/Legendre AGM; Carlson's symmetric forms); this is
//! an original SIMD implementation, not a port of any particular source.

#![allow(clippy::extra_unused_type_parameters)]

use thermite::{
    math::{TranscendentalMathWithPolicy as _, policy::Policy},
    prelude::*,
    register::FloatElement,
};

use crate::specialized::SpecializedSpecialMath;
use thermite::math::policy::DenormalBehavior;
use thermite::{const_element, const_splat};

/// Legendre integral kinds for [`ellint_impl`]'s `KIND` const parameter.
pub const KIND_F: u8 = 1; // first kind: F(phi, k) / K(k)
pub const KIND_E: u8 = 2; // second kind: E(phi, k) / E(k)
pub const KIND_PI: u8 = 3; // third kind: Pi(n, phi, k)
pub const KIND_D: u8 = 4; // D(phi, k) = (F - E) / k^2

// `v.scale(const_element!(ratio <E>: 1 / 3))` is `v * const_splat!(ratio <E>: 1 / 3)` on CPU
// and a single `OpVectorTimesScalar` on SPIR-V. `const_splat!` is used where the constant is
// an FMA or vector operand and `scale` does not fit.

/// Complete elliptic integrals of the first and second kind, `(K(k), E(k))`, evaluated
/// together from a single AGM pass (they share the iteration).
///
/// ```text
/// K(k) = pi / (2 * AGM(1, k')),     k' = sqrt(1 - k^2)
/// E(k) = K(k) * (1 - sum_{n>=0} 2^{n-1} c_n^2)
/// ```
///
/// Valid for `|k| <= 1`; `|k| > 1` gives NaN. The endpoint `|k| = 1` is pinned to
/// `(inf, 1)` under `check_overflow`. See the note at the end of the function.
#[inline(always)]
pub fn agm_complete_ke<P, E, V>(k: V) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    // k' = sqrt(1 - k^2); one_minus_sq is cancellation-free as |k| -> 1 (FMA or factored form).
    let mut a = V::ONE;
    let b0 = k.one_minus_sq().sqrt();
    let mut b = b0;
    let mut c = k;

    // sum starts with the n = 0 term: 2^{-1} c_0^2 = k^2 / 2.
    let mut sum = (c * c).scale(const_element!(ratio <E>: 1 / 2));
    let mut pow2 = V::ONE; // 2^{n-1} for the first in-loop term (n = 1) is 2^0 = 1

    // AGM converges quadratically, so this is a handful of iterations; the masked break
    // stops once every lane's remaining contribution is below the rounding threshold.
    let thresh = V::SQRT_EPSILON; // c_n^2 ~ eps once |c_n| ~ sqrt(eps)
    let mut iter = 0;
    loop {
        V::_loop_hint();

        let an = (a + b).scale(const_element!(ratio <E>: 1 / 2));
        let bn = (a * b).sqrt();
        c = (a - b).scale(const_element!(ratio <E>: 1 / 2));
        a = an;
        b = bn;

        sum = pow2.mul_adde(c * c, sum); // sum += pow2 * c^2
        pow2 = pow2 + pow2;

        iter += 1;
        if iter >= 24 || c.abs().cmp_le(a * thresh).all() {
            break;
        }
    }

    let k_int = V::FRAC_PI_2 / a;
    let e_int = k_int.nmul_adde(sum, k_int); // k_int * (1 - sum)

    if const { P::POLICY.check_overflow } {
        // |k| = 1 makes k' = 0, and AGM(1, 0) = 0, a limit the iteration cannot reach, since
        // `a` merely halves every pass and stops at the cap (K comes out as (pi/2) * 2^24, and
        // E as the inf*0 of that, pi/4). K really does diverge there, but E(1) = 1 exactly, so
        // both are pinned rather than left to the loop. |k| > 1 makes k' NaN, not zero, so the
        // out-of-domain NaN still propagates.
        let deg = b0.is_zero();
        (deg.select(V::INFINITY, k_int), deg.select(V::ONE, e_int))
    } else {
        (k_int, e_int)
    }
}

/// The arithmetic-geometric mean `$\mathrm{AGM}(a, b)$` of two non-negative arguments.
///
/// The same recurrence [`agm_complete_ke`] runs, without that function's `E` accumulator
/// and without its `a = 1` starting pin:
///
/// ```math
/// a_{n+1} = \frac{a_n + b_n}{2}, \qquad b_{n+1} = \sqrt{a_n b_n}
/// ```
///
/// Both sequences converge to the common limit quadratically. From an extreme starting
/// ratio the logarithm of that ratio roughly halves each pass until the two arguments are
/// within a factor of a few, after which the correct digits double per pass, so the
/// iteration cap covers the whole representable range with room to spare, and the loop
/// stays uniform across lanes rather than data-dependent.
///
/// Kept beside `agm_complete_ke` on purpose: they are one recurrence, and a change to
/// either is nearly always a change to both.
///
/// # Domain
///
/// Defined for `a, b >= 0`, and symmetric in its arguments. A negative argument makes the
/// geometric mean's sign ambiguous after the first pass (`(a + b)/2` can be negative
/// while `sqrt(ab)` is not), so negatives give NaN under `check_overflow` rather than a
/// plausible wrong value. `AGM(a, 0) = 0` for every `a`, a limit the iteration approaches
/// but cannot reach (`b` is pinned at zero while `a` merely halves), so it is pinned too,
/// as is `AGM(inf, b) = inf`, which otherwise leaves the loop as `inf - inf`. The one
/// pairing with no limit at all, a zero against an infinity, is NaN.
///
/// # Range
///
/// `sqrt(ab)` is formed as a single product, so two arguments both above `sqrt(MAX)`
/// (about 1.3e154 in binary64, 1.8e19 in binary32) overflow to infinity even though the
/// mean itself is perfectly representable. The AGM is homogeneous,
/// `$\mathrm{AGM}(ca, cb) = c\,\mathrm{AGM}(a, b)$`, so a caller working up there should
/// scale both arguments by a common power of two, which is exact.
#[inline(always)]
pub fn agm<P, E, V>(a: V, b: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let mut x = a;
    let mut y = b;

    // The limit sits between the two sequences, roughly at their midpoint, so `x` is off
    // by about half the _current_ gap, linearly, not quadratically. The test is therefore
    // on the gap going in, exactly as `agm_complete_ke` tests its `c`: a gap of
    // `sqrt(eps)` before the pass leaves `eps/2` after it, since one pass squares it
    // (`x' - y' = (sqrt x - sqrt y)^2 / 2`). Testing the gap coming out instead stops a
    // whole pass early and costs half the mantissa: measured 5.8e-11 at AGM(1, sqrt 2).
    let thresh = V::SQRT_EPSILON;
    let mut iter = 0;
    loop {
        V::_loop_hint();

        let gap = x - y;

        let xn = (x + y).scale(const_element!(ratio <E>: 1 / 2));
        let yn = (x * y).sqrt();
        x = xn;
        y = yn;

        iter += 1;
        if iter >= 24 || gap.abs().cmp_le(x * thresh).all() {
            break;
        }
    }

    if const { P::POLICY.check_overflow } {
        let zero = a.is_zero() | b.is_zero();
        let inf = a.is_infinite() | b.is_infinite();

        x = zero.select(V::ZERO, x);
        x = inf.select(V::INFINITY, x);

        // Negative arguments, and the one indeterminate pairing (a zero against an
        // infinity), where neither pin above is the limit.
        let nan = a.cmp_lt(V::ZERO) | b.cmp_lt(V::ZERO) | (zero & inf);
        x = nan.select(V::NAN, x);
    }

    x
}

/// True where at least two of three non-negative arguments are zero, the interior
/// singularity shared by `R_F`, `R_D` and `R_J`, all of which diverge there while the
/// duplication loop only walks toward the pole until it hits its iteration cap.
///
/// Phrased as "the second-smallest argument is zero", which is exact. The cheaper-looking
/// product test `x*y + x*z + y*z == 0` is not: three denormal arguments underflow every
/// product to zero and would be misread as the singular case.
#[inline(always)]
fn two_or_more_zero<V: FloatVector>(x: V, y: V, z: V) -> V::Mask {
    x.min(y).max(z.min(x.max(y))).is_zero()
}

/// Compile-time per-element constants for the elliptic routines, so values that are otherwise a
/// few runtime ops (e.g. the Carlson convergence threshold, three sequential `sqrt`s) become a
/// constant splat. Declared for `f32`/`f64`; add more element types as needed.
pub trait EllipticConsts {
    /// `(3 * eps)^(1/8)`: the relative-deviation threshold at which the Carlson 7th-order Taylor
    /// tail (`~deviation^8`) drops below rounding. Equals `sqrt(sqrt(sqrt(eps + eps + eps)))`.
    const CARLSON_THRESH: Self;

    /// `|t|` below which `carlson_rc` takes its 8-term series in `t = (y - x)/x` instead of
    /// the closed `atan`/`ln` forms. The tail is `t^8/17`, so this is where that drops below
    /// `eps`: `1/128` at binary32/64 (`1.5e-17`). A wider element type needs it smaller
    /// (double-double takes `2^-14`), or every `R_J`, which calls `R_C(1, 1 + E_n)` with a
    /// small `E_n` on every step, is capped at the series' truncation.
    const RC_SERIES_THRESH: Self;
}

impl EllipticConsts for f32 {
    const CARLSON_THRESH: f32 = 0.15637917816638947;
    const RC_SERIES_THRESH: f32 = 0.0078125;
}

impl EllipticConsts for f64 {
    const CARLSON_THRESH: f64 = 0.012674918778210762;
    const RC_SERIES_THRESH: f64 = 0.0078125;
}

/// The Carlson convergence threshold splatted to the vector type `V`.
#[inline(always)]
fn carlson_thresh<V: FloatVector<Element: EllipticConsts>>() -> V {
    V::splat(<V::Element as EllipticConsts>::CARLSON_THRESH)
}

/// Carlson symmetric integral of the first kind, `R_F(x, y, z)`, via the duplication
/// algorithm. All Boost special cases are omitted: the duplication converges for any
/// valid input (e.g. a single zero argument becomes positive after one step), so the
/// loop runs uniformly across lanes and stops once every lane has converged.
///
/// Two or more zero arguments is the one exception. `R_F` diverges there, and the
/// duplication only walks toward the pole, so it is pinned to infinity under
/// `check_overflow`.
#[inline(always)]
pub fn carlson_rf<P, E, V>(x: V, y: V, z: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E>,
{
    let quarter: V = const_splat!(ratio <E>: 1 / 4);
    let thresh = carlson_thresh::<V>();

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut an = (x + y + z).scale(const_element!(ratio <E>: 1 / 3));
    let a0 = an;
    let mut fmn = V::ONE; // 4^-n
    // Convergence bound. The deviation identity |An - vn| = fmn |A0 - v0| means the current max
    // deviation is fmn * q0, so `fmn * q0 <= An * thresh` is the stop test. Fold thresh in once
    // (q = q0/thresh) and the per-iteration test becomes just `fmn * q <= An` - no recomputing
    // |An - vn| every step.
    let q = (a0 - x).abs().max((a0 - y).abs()).max((a0 - z).abs()) / thresh;

    let mut iter = 0;
    loop {
        V::_loop_hint();

        let rx = xn.sqrt();
        let ry = yn.sqrt();
        let rz = zn.sqrt();
        let lambda = rx.mul_adde(ry + rz, ry * rz); // rx*(ry+rz) + ry*rz; 2-deep vs serial FMA chain
        // (v + lambda)/4 == v/4 + lambda/4; with true FMA, premultiplying lambda lets each update
        // be a single fused `v.mul_adde(1/4, lambda/4)`. Without FMA that extra mul is wasted, so
        // keep the plain add-then-scale form there.
        if const { matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) } {
            let lq = lambda * quarter;
            an = an.mul_add(quarter, lq);
            xn = xn.mul_add(quarter, lq);
            yn = yn.mul_add(quarter, lq);
            zn = zn.mul_add(quarter, lq);
        } else {
            an = (an + lambda) * quarter;
            xn = (xn + lambda) * quarter;
            yn = (yn + lambda) * quarter;
            zn = (zn + lambda) * quarter;
        }
        fmn *= quarter;
        iter += 1;
        if iter >= 30 || (fmn * q).cmp_le(an).all() {
            break;
        }
    }

    // Deviation identity: (An - xn)/An = 4^-n (A0 - x)/An. We use the right-hand form: the
    // initial deviation (A0 - x) is at full precision, whereas (An - xn) is catastrophic
    // cancellation once xn -> An (it costs many digits in the converged, e.g. x = 0, case).
    let scale = fmn / an; // one division, shared by all deviations (vs one div each)
    let xd = (a0 - x) * scale;
    let yd = (a0 - y) * scale;
    let zd = -xd - yd;
    let e2 = xd.mul_sube(yd, zd * zd); // X*Y - Z*Z
    let e3 = xd * yd * zd;

    // 7th-order Taylor expansion (Carlson 2015). Like rdj_poly, split by total degree into three
    // *independent* FMA chains run in parallel and summed - a ~3-deep critical path instead of the
    // serial `1 + e3*A + e2*B` form (~5 deep, B being a 4-deep chain). The e's are tiny deviations,
    // so the regrouped sum is as accurate as the serial form (corrections never cancel against 1).
    let e2_2 = e2 * e2;
    // Linear:    1 - 1/10 e2 + 1/14 e3
    let lin = e2.mul_adde(const_splat!(ratio <E>: -1 / 10), V::ONE);
    let lin = e3.mul_adde(const_splat!(ratio <E>: 1 / 14), lin);
    // Quadratic: 1/24 e2^2 - 3/44 e2 e3 + 3/104 e3^2
    let quad = (e2 * e3).mul_adde(
        const_splat!(ratio <E>: -3 / 44),
        e2_2.scale(const_element!(ratio <E>: 1 / 24)),
    );
    let quad = (e3 * e3).mul_adde(const_splat!(ratio <E>: 3 / 104), quad);
    // Cubic:     -5/208 e2^3 + 1/16 e2^2 e3
    let cub = (e2_2 * e3).mul_adde(
        const_splat!(ratio <E>: 1 / 16),
        (e2_2 * e2).scale(const_element!(ratio <E>: -5 / 208)),
    );

    let poly = lin + (quad + cub);
    let rf = poly / an.sqrt();

    if const { P::POLICY.check_overflow } {
        two_or_more_zero(x, y, z).select(V::INFINITY, rf)
    } else {
        rf
    }
}

/// Carlson symmetric integral `R_D(x, y, z) = R_J(x, y, z, z)` (degenerate third kind,
/// used for the second-kind incomplete integral). Same duplication scheme as [`carlson_rf`],
/// plus an accumulated sum term.
#[inline(always)]
pub fn carlson_rd<P, E, V>(x: V, y: V, z: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E>,
{
    let quarter: V = const_splat!(ratio <E>: 1 / 4);
    let thresh = carlson_thresh::<V>();

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut an = ((x + y) + (z + z + z)).scale(const_element!(ratio <E>: 1 / 5)); // (x + y + 3z) / 5; grouped for ILP
    let a0 = an;
    let mut sum = V::ZERO;
    let mut fac = V::ONE; // 4^-n
    // Convergence bound q = q0 / thresh; the loop tests `fac * q <= An` (see carlson_rf).
    let q = (a0 - x).abs().max((a0 - y).abs()).max((a0 - z).abs()) / thresh;

    let mut iter = 0;
    loop {
        V::_loop_hint();

        let rx = xn.sqrt();
        let ry = yn.sqrt();
        let rz = zn.sqrt();
        let lambda = rx.mul_adde(ry + rz, ry * rz); // rx*(ry+rz) + ry*rz; 2-deep vs serial FMA chain
        sum += fac / (rz * (zn + lambda));
        // (v + lambda)/4 as a fused FMA when available (see carlson_rf).
        if const { matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) } {
            let lq = lambda * quarter;
            an = an.mul_add(quarter, lq);
            xn = xn.mul_add(quarter, lq);
            yn = yn.mul_add(quarter, lq);
            zn = zn.mul_add(quarter, lq);
        } else {
            an = (an + lambda) * quarter;
            xn = (xn + lambda) * quarter;
            yn = (yn + lambda) * quarter;
            zn = (zn + lambda) * quarter;
        }
        fac *= quarter;
        iter += 1;
        if iter >= 30 || (fac * q).cmp_le(an).all() {
            break;
        }
    }

    // Reconstruct from initial deviations (see carlson_rf) to avoid An - xn cancellation.
    let scale = fac / an; // one division, shared by the deviations
    let xd = (a0 - x) * scale;
    let yd = (a0 - y) * scale;
    let zd = (xd + yd).scale(const_element!(ratio <E>: -1 / 3));
    let xy = xd * yd;
    let zz = zd * zd;
    let xy3 = xy.scale(const_element!(ratio <E>: 3 / 1)); // 3 xy, shared by e3 and e4
    let e2 = zz.mul_adde(const_splat!(int <E>: -6), xy); // xy - 6 zz
    let e3 = zz.mul_adde(const_splat!(int <E>: -8), xy3) * zd; // (3 xy - 8 zz) zd
    let e4 = zz.mul_adde(const_splat!(int <E>: -3), xy3) * zz; // (3 xy - 3 zz) zz = 3 (xy - zz) zz
    let e5 = xy * (zz * zd);

    let taylor = fac * rdj_poly_n::<E, V>(e2, e3, e4, e5) / (an * an.sqrt()); // fac * An^(-3/2) * poly
    let rd = sum.mul_adde(const_splat!(int <E>: 3), taylor); // taylor + 3 * sum

    if const { P::POLICY.check_overflow } {
        // `z == 0` alone also diverges, but that one arrives on its own: the accumulated
        // `fac / (rz * (zn + lambda))` term divides by zero and carries the infinity out.
        two_or_more_zero(x, y, z).select(V::INFINITY, rd)
    } else {
        rd
    }
}

/// Shared 5th-order Taylor tail polynomial for R_D and R_J (Carlson 2015) - they use the
/// same form in the deviation variables E2..E5.
#[inline(always)]
fn rdj_poly_n<E, V>(e2: V, e3: V, e4: V, e5: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    // e2^2 feeds several higher terms; compute once. Then three *independent* FMA chains (by
    // total degree) run in parallel and are summed - a ~7-deep critical path instead of one
    // 12-deep serial Horner chain. The e's are tiny deviations, so the regrouped sum is as
    // accurate as the serial form (the corrections never cancel against the leading 1).
    let e2_2 = e2 * e2;
    // Linear:    1 - 3/14 e2 + 1/6 e3 - 3/22 e4 + 3/26 e5
    let lin = e2.mul_adde(const_splat!(ratio <E>: -3 / 14), V::ONE);
    let lin = e3.mul_adde(const_splat!(ratio <E>: 1 / 6), lin);
    let lin = e4.mul_adde(const_splat!(ratio <E>: -3 / 22), lin);
    let lin = e5.mul_adde(const_splat!(ratio <E>: 3 / 26), lin);
    // Quadratic: 9/88 e2^2 - 9/52 e2 e3 + 3/40 e3^2 + 3/20 e2 e4
    let quad = (e2 * e3).mul_adde(
        const_splat!(ratio <E>: -9 / 52),
        e2_2.scale(const_element!(ratio <E>: 9 / 88)),
    );
    let quad = (e3 * e3).mul_adde(const_splat!(ratio <E>: 3 / 40), quad);
    let quad = (e2 * e4).mul_adde(const_splat!(ratio <E>: 3 / 20), quad);
    // Cubic+:    -1/16 e2^3 + 45/272 e2^2 e3 - 9/68 (e3 e4 + e2 e5)
    let cub = (e2_2 * e3).mul_adde(
        const_splat!(ratio <E>: 45 / 272),
        (e2_2 * e2).scale(const_element!(ratio <E>: -1 / 16)),
    );
    let cub = (e3 * e4 + e2 * e5).mul_adde(const_splat!(ratio <E>: -9 / 68), cub);

    lin + (quad + cub)
}

/// Carlson symmetric integral of the second kind, `R_G(x, y, z)`, as a combination of
/// [`carlson_rf`] and [`carlson_rd`] (Carlson 2015):
///
/// ```text
/// R_G = (z * R_F(x,y,z) - (x-z)(y-z) * R_D(x,y,z) / 3 + sqrt(x*y/z)) / 2
/// ```
///
/// The arguments are sorted to `hi >= mid >= lo` and substituted as `x = hi, z = mid, y = lo`,
/// the ordering that keeps `(x-z)(y-z)` from cancelling and puts the middle value (the divisor)
/// in `z`. That form needs `mid > 0`; two or more zero arguments divide by it, so under
/// `check_overflow` they take the closed form `R_G(x, 0, 0) = sqrt(x)/2` instead (which also
/// covers `R_G(0,0,0) = 0`). Unlike `R_F`/`R_D`/`R_J`, `R_G` stays finite there. Not wired into
/// the public `ellint` surface - no Legendre form needs it; provided for direct use
/// (e.g. `E(k) = 2 R_G(0, 1-k^2, 1)`).
#[inline(always)]
pub fn carlson_rg<P, E, V>(x: V, y: V, z: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E>,
{
    let lo = x.min(y).min(z);
    let hi = x.max(y).max(z);
    let mid = (x + y + z) - (lo + hi); // grouped: (x+y+z) and (lo+hi) form in parallel
    // R_F is fully symmetric; R_D's third argument must be the middle value.
    let rf = carlson_rf::<P, E, V>(hi, lo, mid);
    let rd = carlson_rd::<P, E, V>(hi, lo, mid);
    let root = (hi * lo / mid).sqrt();
    let prod = (hi - mid) * (lo - mid) * rd;
    // (mid*rf + sqrt(xy/z) - (x-z)(y-z) rd / 3) / 2
    let rg = prod
        .mul_adde(const_splat!(ratio <E>: -1 / 3), mid.mul_adde(rf, root))
        .scale(const_element!(ratio <E>: 1 / 2));

    if const { P::POLICY.check_overflow } {
        // Sorted, so `mid == 0` is exactly "two or more arguments are zero". There the general
        // form is 0/0 and `rf`/`rd` are themselves infinite, but the integral is not: it
        // collapses to sqrt(hi)/2.
        mid.is_zero()
            .select(hi.sqrt().scale(const_element!(ratio <E>: 1 / 2)), rg)
    } else {
        rg
    }
}

/// Carlson degenerate integral `R_C(x, y) = R_F(x, y, y)`, closed form. Assumes `y > 0`
/// (the only cases that arise inside R_J and the third-kind reductions); the `y < 0`
/// Cauchy-principal-value branch is not handled here.
///
/// Writing `t = (y - x)/x`, `R_C(x, y) = S(t)/sqrt(x)` where
/// `S(t) = atan(sqrt(t))/sqrt(t) = 1 - t/3 + t^2/5 - t^3/7 + ...` is smooth and the *same*
/// series for either sign of `t` (`atanh` for `t < 0`). The closed `atan`/`ln` forms lose
/// ~`sqrt(eps)` precision as `t -> 0` (e.g. `ln(1 + sqrt|t|)` with tiny `sqrt|t|`), which
/// matters because R_J calls this with `y -> x` every iteration once `p` nears an argument.
/// So for small `|t|` we use the series instead; the two agree to full precision at the
/// crossover. This keeps R_C (and hence R_J near the `p == arg` degeneracy) accurate without
/// resorting to extended precision.
#[inline(always)]
pub fn carlson_rc<P, E, V>(x: V, y: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: SpecializedSpecialMath<E>,
{
    let d = y - x;
    let absd = d.abs();

    // The closed form divides by sad = sqrt(|y-x|) (both branches) and sqrt(y) (neg branch).
    // With hardware rsqrt (f32) compute each 1/sqrt directly and recover the roots by multiply -
    // no full sqrt or divide in the hot path. Without it (f64, where rsqrt = rcp(sqrt)) a plain
    // a/sqrt(b) is a single divide, so sqrt + div stays optimal. `irx = 1/sqrt(x)` also scales
    // the series (poly * irx) in both paths. The carried `scale` is 1/sad on the rsqrt path
    // (multiply) and sad on the divide path; the combine below picks the matching op.
    // The rsqrt arm only pays when the hardware estimate is BOTH present and permitted.
    // Under `Preserve` the estimate is forbidden (`rsqrtps` treats a subnormal operand as
    // zero in hardware whatever MXCSR says), so `inverse_sqrt_p` becomes an exact
    // `1/sqrt` and this arm would spend three of them where the other spends one divide.
    let (s, neg_arg, irx, scale) =
        if const { V::HAS_APPROX_RSQRT && !matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) } {
            let isad = absd.inverse_sqrt_p::<P>(); // 1/sqrt(|y-x|)
            let irx = x.inverse_sqrt_p::<P>(); // 1/sqrt(x)
            let iry = y.inverse_sqrt_p::<P>(); // 1/sqrt(y)
            let sad = absd * isad; // sqrt(|y-x|)
            let rx = x * irx; // sqrt(x)
            (sad * irx, (rx + sad) * iry, irx, isad)
        } else {
            let sad = absd.sqrt(); // sqrt(|y-x|)
            let rx = x.sqrt();
            let irx = rx.approx_reciprocal_p::<P>(); // 1/sqrt(x)
            (sad * irx, (rx + sad) / y.sqrt(), irx, sad)
        };

    // x < y: atan(s)  ;  x > y: ln((sqrt(x) + sqrt(x-y))/sqrt(y)) = atanh(s). Both scaled by 1/sad.
    let num = d.cmp_gt(V::ZERO).select(s.atan_p::<P>(), neg_arg.ln_p::<P>());
    // Must match the arm chosen above, since `scale` is 1/sad on one path and sad on the other.
    let closed = if const { V::HAS_APPROX_RSQRT && !matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) }
    {
        num * scale
    } else {
        num / scale
    };

    // Series for small |t|, where the closed forms cancel: S(t)/sqrt(x). S is univariate in t,
    // S(t) = 1 - t/3 + t^2/5 - ... ; evaluated leading-coefficient-first via poly_rev (Estrin + FMA).
    let t = d / x;

    let mut res = closed;

    // |t| < 1/128 ~ 0.0078 at binary64: series is accurate to <1e-16 with these 8 terms, and
    // the closed forms are already degrading there. Above it the closed forms are accurate.
    // t == 0 (x == y) is covered by the series limit S(0) = 1. The threshold is per element
    // type (`EllipticConsts`) because the tail is the series', not the type's.

    let small = t.abs().cmp_lt(V::splat(E::RC_SERIES_THRESH));

    if const { P::POLICY.avoid_branching } || small.any() {
        let series = t.poly_rev_n_p::<P, _>(&[
            <E as FloatElement>::ConstRatio::<-1, 15>::VALUE,
            <E as FloatElement>::ConstRatio::<1, 13>::VALUE,
            <E as FloatElement>::ConstRatio::<-1, 11>::VALUE,
            <E as FloatElement>::ConstRatio::<1, 9>::VALUE,
            <E as FloatElement>::ConstRatio::<-1, 7>::VALUE,
            <E as FloatElement>::ConstRatio::<1, 5>::VALUE,
            <E as FloatElement>::ConstRatio::<-1, 3>::VALUE,
            <E as FloatElement>::ConstRatio::<1, 1>::VALUE,
        ]) * irx;

        res = small.select(series, res);
    }

    res
}

/// Carlson symmetric integral of the third kind, `R_J(x, y, z, p)`, via duplication.
/// Handles `p < 0` (a Cauchy principal value) through Carlson's transform to a positive
/// parameter. Each step accumulates an `R_C` term, so this is the most expensive Carlson
/// primitive.
///
/// Accuracy note: when `p` coincides with one of `x, y, z` (so `(p-x)(p-y)(p-z) -> 0`), the
/// per-step `R_C(1, b)` term has `b -> 1` every iteration. That used to lose ~7 digits, but
/// [`carlson_rc`] now switches to its small-argument series there, so this case holds full
/// precision without the `R_D` special-case or extended precision Boost resorts to.
#[inline(always)]
pub fn carlson_rj<P, E, V>(x: V, y: V, z: V, p: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: SpecializedSpecialMath<E>,
{
    let quarter: V = const_splat!(ratio <E>: 1 / 4);
    let thresh = carlson_thresh::<V>();

    // R_J is symmetric in (x, y, z); sort so `hi` is the largest. For p < 0 the integral is
    // a Cauchy principal value, mapped to a positive parameter p' via Carlson's transform.
    let lo = x.min(y).min(z);
    let hi = x.max(y).max(z);
    let mid = (x + y + z) - (lo + hi); // grouped: (x+y+z) and (lo+hi) form in parallel

    let neg = p.cmp_lt(V::ZERO);
    let q = -p; // |p| on the p < 0 lanes
    // p' = (hi(lo + mid + q) - lo*mid) / (hi + q)  (> 0 for sorted lo<=mid<=hi, p < 0)
    let p_new = hi.mul_sube(lo + mid + q, lo * mid) / (hi + q);
    let p_eff = neg.select(p_new, p); // R_J parameter: p' where p<0, else p (both > 0)

    let mut xn = lo;
    let mut yn = mid;
    let mut zn = hi;
    let mut pn = p_eff;
    let mut an = ((lo + mid + hi) + (p_eff + p_eff)).scale(const_element!(ratio <E>: 1 / 5)); // (x + y + z + 2p) / 5; grouped for ILP
    let a0 = an;
    let mut rc_sum = V::ZERO;
    let mut fmn = V::ONE; // 4^-n
    // Convergence bound qb = q0 / thresh; the loop tests `fmn * qb <= An` (see carlson_rf).
    let qb = (a0 - lo)
        .abs()
        .max((a0 - mid).abs())
        .max((a0 - hi).abs().max((a0 - p_eff).abs()))
        / thresh;

    let mut iter = 0;
    loop {
        V::_loop_hint();

        let rx = xn.sqrt();
        let ry = yn.sqrt();
        let rz = zn.sqrt();
        let rp = pn.sqrt();
        let dn = (rp + rx) * (rp + ry) * (rp + rz);
        // b = 1 + E_n computed stably (avoids the E_n ~ -1 cancellation); R_C(1, 1+E_n).
        let inner = ry.mul_adde(rz, rx.mul_adde(ry + rz, pn)); // pn + rx(ry+rz) + ry rz
        // 2 * rp * inner / dn, balanced so numerator (rp*inner) and denominator (dn/2) form in
        // parallel before the divide, shortening the critical path.
        let b = (rp * inner) / dn.scale(const_element!(ratio <E>: 1 / 2));
        rc_sum = (fmn / dn).mul_adde(carlson_rc::<P, E, V>(V::ONE, b), rc_sum); // += (fmn/dn) R_C

        let lambda = rx.mul_adde(ry + rz, ry * rz); // rx*(ry+rz) + ry*rz; 2-deep vs serial FMA chain
        // (v + lambda)/4 as a fused FMA when available (see carlson_rf).
        if const { matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) } {
            let lq = lambda * quarter;
            an = an.mul_add(quarter, lq);
            xn = xn.mul_add(quarter, lq);
            yn = yn.mul_add(quarter, lq);
            zn = zn.mul_add(quarter, lq);
            pn = pn.mul_add(quarter, lq);
        } else {
            an = (an + lambda) * quarter;
            xn = (xn + lambda) * quarter;
            yn = (yn + lambda) * quarter;
            zn = (zn + lambda) * quarter;
            pn = (pn + lambda) * quarter;
        }
        fmn *= quarter;
        iter += 1;
        if iter >= 30 || (fmn * qb).cmp_le(an).all() {
            break;
        }
    }

    // Reconstruct from initial deviations (see carlson_rf) to avoid An - xn cancellation.
    let scale = fmn / an; // one division, shared by all three deviations
    let xd = (a0 - lo) * scale;
    let yd = (a0 - mid) * scale;
    let zd = (a0 - hi) * scale;
    let pd = (xd + yd + zd).scale(const_element!(ratio <E>: -1 / 2));
    let xyz = xd * yd * zd;
    let pp = pd * pd;
    let ppd = pp * pd; // pd^3, shared by e3 and e4
    let sym = yd.mul_adde(zd, xd * (yd + zd)); // xd*yd + xd*zd + yd*zd
    let e2 = pp.mul_adde(const_splat!(int <E>: -3), sym); // sym - 3 pd^2
    // e2 is the latest-arriving input (it trails the fmn/an divide through pd/pp/sym). Precompute
    // the e2-independent parts of e3/e4 - which LLVM can't hoist itself, FP adds don't reassociate
    // without fast-math - so each e-term is a single FMA past e2 instead of a 2-3 deep chain.
    // e3 = xyz + 2 e2 pd + 4 pd^3
    let pre3 = ppd.mul_adde(const_splat!(int <E>: 4), xyz); // 4 pd^3 + xyz
    let e3 = e2.mul_adde(pd + pd, pre3); // 2 pd e2 + pre3
    // e4 = (2 xyz + e2 pd + 3 pd^3) pd = e2 pp + (3 pd^3 + 2 xyz) pd
    let pre4 = ppd.mul_adde(const_splat!(int <E>: 3), xyz + xyz) * pd; // (3 pd^3 + 2 xyz) pd = 3 pp^2 + 2 xyz pd
    let e4 = e2.mul_adde(pp, pre4); // e2 pp + pre4
    let e5 = xyz * pp;

    let taylor = fmn * rdj_poly_n::<E, V>(e2, e3, e4, e5) / (an * an.sqrt());
    let rj = rc_sum.mul_adde(const_splat!(int <E>: 6), taylor); // taylor + 6 * rc_sum

    let out = if const { P::POLICY.avoid_branching } || neg.any() {
        // Cauchy PV recombination for p < 0 (Carlson):
        //   R_J = ((p'-z) R_J(x,y,z,p') - 3 R_F(x,y,z) + 3 sqrt(xyz/(xy+p'q)) R_C(xy+p'q, p'q)) / (z+q)
        let rf = carlson_rf::<P, E, V>(lo, mid, hi);
        let xy = lo * mid;
        let xyz = xy * hi;
        let pq = p_new * q;
        let rc = carlson_rc::<P, E, V>(xy + pq, pq);
        // ((p'-z) R_J - 3 R_F + 3 sqrt(xyz/(xy+p'q)) R_C) / (z+q)
        //   = ((p'-z) R_J + 3 (sqrt(..) R_C - R_F)) / (z+q)
        let root = (xyz / (xy + pq)).sqrt();
        let val_neg =
            (p_new - hi).mul_adde(rj, root.mul_sube(rc, rf).scale(const_element!(ratio <E>: 3 / 1))) / (hi + q);
        neg.select(val_neg, rj)
    } else {
        rj
    };

    if const { P::POLICY.check_overflow } {
        // Two singular sets. `p == 0` mostly arrives on its own (the per-step `R_C(1, 0)` is
        // infinite), but not when an argument is zero too: `dn` is then zero as well and the
        // `R_C` argument becomes 0/0. Since `mid` is already sorted, both tests are a compare.
        (mid.is_zero() | p.is_zero()).select(V::INFINITY, out)
    } else {
        out
    }
}

/// Legendre elliptic integral, const-generic over kind and completeness.
///
/// `KIND` is one of [`KIND_F`], [`KIND_E`], [`KIND_D`], [`KIND_PI`]. `COMPLETE` selects
/// `phi = pi/2` (the AGM path for F/E/D, Carlson R_F+R_J for Pi); otherwise the incomplete
/// form is evaluated via Carlson at a range-reduced amplitude. Argument is the modulus `k`;
/// `n` is the characteristic, used only by the third kind ([`KIND_PI`]).
///
/// Incomplete amplitudes are reduced into `[-pi/2, pi/2]` using the quasi-period identity
/// `I(phi + m*pi) = I(phi) + 2m * I_complete` (the integrands have period pi, and one full
/// period equals twice the complete value). Reduction by `phi - m*pi` loses precision for
/// very large `|phi|` (argument cancellation); a Cody-Waite split would extend the range.
#[inline(always)]
pub fn ellint_impl<P, E, V, const KIND: u8, const COMPLETE: bool>(phi: V, k: V, n: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: SpecializedSpecialMath<E>, // : FloatVector + (via blanket) TranscendentalMathWithPolicy
{
    const {
        assert!(
            KIND == KIND_F || KIND == KIND_E || KIND == KIND_D || KIND == KIND_PI,
            "ellint_impl: KIND must be KIND_F, KIND_E, KIND_D, or KIND_PI"
        );
    }
    if const { COMPLETE } {
        if const { KIND == KIND_PI } {
            // Pi(n, k) = R_F(0, 1-k^2, 1) + (n/3) R_J(0, 1-k^2, 1, 1-n)  (the AGM does not
            // cover the third kind, so the complete Pi still goes through Carlson).
            let w = k.one_minus_sq(); // 1 - k^2
            let rf = carlson_rf::<P, E, V>(V::ZERO, w, V::ONE);
            let rj = carlson_rj::<P, E, V>(V::ZERO, w, V::ONE, V::ONE - n);
            n.scale(const_element!(ratio <E>: 1 / 3)).mul_adde(rj, rf) // (n/3) rj + rf
        } else {
            let (kk, ee) = agm_complete_ke::<P, E, V>(k);
            if const { KIND == KIND_F } {
                kk
            } else if const { KIND == KIND_E } {
                ee
            } else {
                // D(k) = (K - E) / k^2
                let d = (kk - ee) / (k * k);
                if const { P::POLICY.check_overflow } {
                    // K(0) == E(0), so k = 0 is 0/0 here. The limit is finite: D(k) -> pi/4.
                    k.is_zero().select(V::FRAC_PI_4, d)
                } else {
                    d
                }
            }
        }
    } else {
        // Range-reduce phi into [-pi/2, pi/2]; m counts the half-periods stripped off.
        //
        // A single `m*PI` deliberately, rather than a Cody-Waite split of pi. Such a
        // split exists because a naive reduction leaves an absolute error of about
        // |phi|*eps in the reduced angle - but here
        // the result is `F(phi_red) + 2m*K`, which grows with |phi| in the same
        // proportion, so that error stays at roughly one ULP of the returned value no
        // matter how large phi gets. Measured at m = 1e9: reduction error ~7e-7 against
        // a result of ~3.4e9 whose ULP is ~4.8e-7.
        //
        // Reduction precision pays off when the output does *not* grow with the input -
        // sin and cos, whose range is fixed, are where it is worth the extra products.
        let m = (phi * V::FRAC_1_PI).round();
        let phi_red = m.nmul_adde(V::PI, phi);

        let (s, cphi) = phi_red.sin_cos_p::<P>();
        let c2 = cphi * cphi;
        let k2 = k * k;
        // w = 1 - k^2 sin^2(phi), rewritten as (1 - k^2) + k^2 cos^2(phi).
        //
        // The direct form subtracts two quantities that both approach 1 at the domain
        // boundary |k sin(phi)| = 1, and `k * sin(phi)` rounds to exactly 1 before the
        // subtraction ever runs, so w collapses to 0 while the true value is the (tiny,
        // nonzero) cos^2 term. This form reads that term straight off cos(phi) instead.
        // For |k| <= 1 both addends are non-negative, so it cannot cancel at all.
        let w = k2.mul_adde(c2, k.one_minus_sq());
        let rf = carlson_rf::<P, E, V>(c2, w, V::ONE);
        let core = if const { KIND == KIND_F } {
            // F(phi, k) = sin(phi) * R_F(cos^2 phi, 1 - k^2 sin^2 phi, 1)
            s * rf
        } else if const { KIND == KIND_PI } {
            // Pi(n, phi, k) = sin(phi) R_F + (n/3) sin^3(phi) R_J(c2, w, 1, 1 - n sin^2 phi)
            let pp = n.nmul_adde(s * s, V::ONE);
            let rj = carlson_rj::<P, E, V>(c2, w, V::ONE, pp);
            let s3 = s * s * s;
            (n.scale(const_element!(ratio <E>: 1 / 3)) * s3).mul_adde(rj, s * rf) // (n/3) s^3 rj + s rf
        } else {
            let rd = carlson_rd::<P, E, V>(c2, w, V::ONE);
            let s3 = s * s * s;
            if const { KIND == KIND_E } {
                // E(phi, k) = sin(phi) R_F - (k^2 / 3) sin^3(phi) R_D
                (k2.scale(const_element!(ratio <E>: 1 / 3)) * s3).nmul_adde(rd, s * rf)
            } else {
                // D(phi, k) = (1/3) sin^3(phi) R_D
                s3.scale(const_element!(ratio <E>: 1 / 3)) * rd
            }
        };

        // Quasi-period correction: I(phi) = core + 2m * I_complete. Skip the (expensive)
        // recursive complete evaluation when no lane was reduced, unless the policy forbids
        // branching. Must use `select`, not a bare `2m * complete` add: for |k| > 1 lanes the
        // complete value is NaN, but |k| > 1 forces phi < pi/2 so m = 0 there - the select
        // keeps `core` for those lanes and avoids 0 * NaN poisoning them.
        if const { P::POLICY.avoid_branching } || !m.is_zero().all() {
            let complete = ellint_impl::<P, E, V, KIND, true>(phi, k, n);
            m.is_zero().select(core, (m + m).mul_adde(complete, core))
        } else {
            core
        }
    }
}

/// A Carlson symmetric elliptic integral request. The implementors are small structs that carry
/// the integral's arguments as named fields ([`CarlsonRf`], [`CarlsonRc`], [`CarlsonRd`],
/// [`CarlsonRj`], [`CarlsonRg`]), so each kind has exactly its own arguments - no dummy slots, and
/// the special roles (`R_J`'s parameter `p`, `R_D`'s repeated `z`) are named at the call site.
pub trait CarlsonKind {
    /// The vector type returned (and the field type of the request struct).
    type Output;
    /// Evaluate the integral under precision policy `P`, consuming the request.
    fn eval<P: Policy>(self) -> Self::Output;
}

/// The Jacobi zeta function `$Z(\varphi, k)$`.
///
/// The oscillating part of the incomplete integral of the second kind (what is left of
/// `$E(\varphi, k)$` once its linear growth is removed):
///
/// ```math
/// Z(\varphi, k) = E(\varphi, k) - \frac{E(k)}{K(k)} F(\varphi, k)
/// ```
///
/// Odd in `phi`, `pi`-periodic, and exactly zero at every multiple of `pi/2`.
///
/// That defining difference is **not** how it is evaluated. Both terms grow with `phi` while
/// `Z` does not, so the subtraction cancels wherever `Z` is small, which is near the zeros,
/// i.e. everywhere the function is most delicate. The Carlson form used instead has no
/// subtraction in it at all:
///
/// ```math
/// Z(\varphi, k) = \frac{k^2 \sin\varphi \cos\varphi \sqrt{1 - k^2\sin^2\varphi}}{3 K(k)}
///                 R_J(0,\ k'^2,\ 1,\ 1 - k^2\sin^2\varphi)
/// ```
///
/// and `$1 - k^2\sin^2\varphi$` is itself formed as `$k'^2 + k^2\cos^2\varphi$`, a sum of two
/// non-negative terms, so it cannot cancel either. Measured against the defining difference
/// at 40 digits, worst relative error 3.1e-15 over `k` to 0.999 and `|phi|` to 4.5.
///
/// No sign fixup is needed: `sin` is odd and every other factor is even in `phi`, so the
/// oddness falls out. `k = 1` is the one modulus with no Carlson form (`$k'^2 = 0$` gives
/// `R_J` two zero arguments and `K` is infinite) and takes the limit
/// `$\sin\varphi\,\operatorname{sign}(\cos\varphi)$` instead.
#[inline(always)]
pub fn jacobi_zeta<P, E, V>(phi: V, k: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let k2 = k * k;
    let kp = k.one_minus_sq();
    let (sin_phi, cos_phi) = phi.sin_cos_p::<P>();
    let c2 = cos_phi * cos_phi;

    // 1 - k^2 sin^2(phi), written as a sum of non-negative terms so it never cancels.
    let one_minus_ks2 = k2.mul_adde(c2, kp);

    let (k_complete, _) = agm_complete_ke::<P, E, V>(k);
    let rj = carlson_rj::<P, E, V>(V::ZERO, kp, V::ONE, one_minus_ks2);

    let num = (k2 * sin_phi * cos_phi * one_minus_ks2.sqrt() * rj).scale(const_element!(ratio <E>: 1 / 3));
    let mut z = num.approx_div_p::<P>(k_complete);

    if const { P::POLICY.check_overflow } {
        // k = 1 leaves R_J with two zero arguments and K infinite, so the quotient is NaN
        // rather than the limit. Mathematica's simplification of Z(phi, 1) is the signed sine.
        let unit = k.abs().cmp_eq(V::ONE);
        if const { P::POLICY.avoid_branching } || thermite::unlikely(unit.any()) {
            z = unit.select(sin_phi.copysign(cos_phi * sin_phi), z);
        }
    }

    z
}

/// Heuman's lambda function `$\Lambda_0(\varphi, k)$`.
///
/// ```math
/// \Lambda_0(\varphi, k) = \frac{2}{\pi}\Big[E(k) F(\varphi, k') + K(k) E(\varphi, k')
///                                          - K(k) F(\varphi, k')\Big]
/// ```
///
/// with `$k' = \sqrt{1-k^2}$` the complementary modulus. `$\Lambda_0(0, k) = 0$` and
/// `$\Lambda_0(\pi/2, k) = 1$`, which is what makes it the natural companion to the complete
/// integral of the third kind. It is also the standard closed form for the off-axis field of a
/// circular current loop.
///
/// Inside `$|\varphi| \le \pi/2$` a Carlson form avoids the three-term difference above:
///
/// ```math
/// \Lambda_0 = \frac{2}{\pi}\frac{k'^2 \sin\varphi\cos\varphi}{\delta}
///             \left[R_F(0, k'^2, 1) + \frac{k^2}{3\delta^2} R_J(0, k'^2, 1, p)\right],
/// \qquad \delta^2 = 1 - k'^2\sin^2\varphi
/// ```
///
/// **The parameter `p` is the delicate part.** Its textbook spelling is `$1 - k^2/\delta^2$`,
/// which is exactly zero at `$\varphi = \pi/2$` and therefore rounds _negative_ just before
/// it, and a negative fourth argument sends `R_J` into its Cauchy-principal-value branch,
/// which is a different function. Since `$\delta^2 = k^2 + k'^2\cos^2\varphi$`, the parameter
/// is identically `$k'^2\cos^2\varphi/\delta^2$`, a ratio of non-negative quantities that is
/// correct at the endpoint and cannot go negative. Measured against the defining form at 40
/// digits: 3.6e-15 worst with that spelling, versus a NaN with the textbook one.
///
/// Beyond `$|\varphi| > \pi/2$` the Carlson form no longer applies and the identity
/// `$\Lambda_0 = F(\varphi,k')/K(k') + \tfrac{2}{\pi}K(k) Z(\varphi, k')$` takes over. That
/// arm costs three more elliptic evaluations, so it is gated on a lane actually needing it:
/// the function's usual domain is `$[0, \pi/2]$`.
#[inline(always)]
pub fn heuman_lambda<P, E, V>(phi: V, k: V) -> V
where
    P: Policy,
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let k2 = k * k;
    let kp = k.one_minus_sq();
    let (sin_phi, cos_phi) = phi.sin_cos_p::<P>();
    let c2 = cos_phi * cos_phi;

    // delta^2 = 1 - k'^2 sin^2(phi) = k^2 + k'^2 cos^2(phi). The same trick as `jacobi_zeta`'s,
    // with k^2 and k'^2 swapped: two non-negative terms, so no cancellation.
    let d2 = kp.mul_adde(c2, k2);
    let delta = d2.sqrt();

    // p = 1 - k^2/delta^2, in the one form that stays non-negative through phi = pi/2. The
    // subtraction rounds negative just short of the endpoint, and a negative fourth argument
    // is a different R_J (the Cauchy principal value).
    let p = (kp * c2).approx_div_p::<P>(d2);

    let rf = carlson_rf::<P, E, V>(V::ZERO, kp, V::ONE);
    let rj = carlson_rj::<P, E, V>(V::ZERO, kp, V::ONE, p);

    let bracket = rf + (k2 * rj).scale(const_element!(ratio <E>: 1 / 3)).approx_div_p::<P>(d2);
    let scale = (kp * sin_phi * cos_phi).approx_div_p::<P>(delta * V::FRAC_PI_2);
    let mut result = scale * bracket;

    // Outside [-pi/2, pi/2] the Carlson form does not hold and the Legendre identity takes
    // over. Three more elliptic evaluations, so only when a lane is actually out there.
    let far = phi.abs().cmp_gt(V::FRAC_PI_2);
    if const { P::POLICY.avoid_branching } || thermite::unlikely(far.any()) {
        let k_prime = kp.sqrt();
        let f_inc = ellint_impl::<P, E, V, KIND_F, false>(phi, k_prime, k);
        let (k_prime_complete, _) = agm_complete_ke::<P, E, V>(k_prime);
        let (k_complete, _) = agm_complete_ke::<P, E, V>(k);

        let ratio = f_inc.approx_div_p::<P>(k_prime_complete);
        let zeta = jacobi_zeta::<P, E, V>(phi, k_prime);
        result = far.select(ratio + (k_complete * zeta).approx_div_p::<P>(V::FRAC_PI_2), result);
    }

    result
}

/// Legendre elliptic integral request, dual to [`CarlsonKind`] (see it for the named-field
/// rationale). The implementors are [`EllintK`]/[`EllintF`] (1st kind, complete/incomplete),
/// [`EllintE`]/[`EllintEInc`] (2nd kind), [`EllintD`]/[`EllintDInc`], and
/// [`EllintPi`]/[`EllintPiInc`] (3rd kind). Completeness is encoded by the *fields*: a complete
/// integral has no `phi`.
pub trait EllipticKind {
    /// The vector (or scalar) type returned.
    type Output;
    /// Evaluate the integral under precision policy `P`, consuming the request.
    fn eval<P: Policy>(self) -> Self::Output;
}

use thermite::math::scalar::Unwrap;

/// The wider (vector) form of a scalar request struct - the inverse of [`Unwrap`] for these structs.
/// `Unwrap` only maps vector -> scalar (`type Unwrapped`), and Rust cannot invert an associated type,
/// so this names the forward (scalar -> vector) direction. It is what lets the `Scalar*` math-trait
/// layer take a scalar request (`CarlsonRf<f64>`), wrap it into a width-1 vector request
/// (`CarlsonRf<Vector<f64>>`), evaluate on the (vector-only) backend, then unwrap the scalar result -
/// without implementing the backend twice. Implemented only for scalar-element request structs.
pub trait WrapTo {
    /// The vector request struct whose `Unwrapped` is `Self`.
    type Wrapped: Unwrap<Unwrapped = Self>;
}

/// Generates a request struct (`$name<V>` with named fields) plus its `Unwrap` (field-wise
/// vector<->scalar) and `WrapTo` (scalar -> width-1 vector) impls. The `CarlsonKind` / `EllipticKind`
/// impl is added by the caller, bounded on `FloatVector` - so the compute backend is vector-only.
macro_rules! request_struct {
    ($(#[$meta:meta])* $name:ident { $($field:ident),* }) => {
        #[derive(Debug, Clone, Copy)]
        $(#[$meta])* pub struct $name<V> {
            $(pub $field: V,)*
        }

        impl<V: Unwrap> Unwrap for $name<V> {
            type Unwrapped = $name<<V as Unwrap>::Unwrapped>;
            #[inline(always)]
            fn wrap(value: Self::Unwrapped) -> Self {
                $name { $($field: Unwrap::wrap(value.$field),)* }
            }
            #[inline(always)]
            fn unwrap(self) -> Self::Unwrapped {
                $name { $($field: self.$field.unwrap(),)* }
            }
        }

        impl<E> WrapTo for $name<E>
        where
            E: FloatElement + thermite::register::FloatRegister<Storage = E>,
            thermite::Vector<E>: Unwrap<Unwrapped = E>,
        {
            type Wrapped = $name<thermite::Vector<E>>;
        }
    };
}

/// Carlson request structs. `$func` is the free function the struct evaluates (all take `<P, E, V>`).
macro_rules! decl_carlson {
    ($( $(#[$meta:meta])* struct $name:ident { $($field:ident),* } => $func:ident; )*) => {$(
        request_struct! { $(#[$meta])* $name { $($field),* } }

        impl<E, V> CarlsonKind for $name<V>
        where
            E: FloatElement + EllipticConsts,
            V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
        {
            type Output = V;
            #[inline(always)]
            fn eval<P: Policy>(self) -> V {
                $func::<P, E, V>($(self.$field),*)
            }
        }
    )*};
}

decl_carlson! {
    /// Carlson `R_F(x, y, z)` - symmetric integral of the first kind. See [`CarlsonKind`].
    struct CarlsonRf { x, y, z } => carlson_rf;

    /// Carlson `R_C(x, y) = R_F(x, y, y)` - degenerate first kind. See [`CarlsonKind`].
    struct CarlsonRc { x, y } => carlson_rc;

    /// Carlson `R_D(x, y, z) = R_J(x, y, z, z)` - degenerate third kind; `z` is the repeated argument.
    /// See [`CarlsonKind`].
    struct CarlsonRd { x, y, z } => carlson_rd;

    /// Carlson `R_J(x, y, z, p)` - symmetric integral of the third kind; `p` is the parameter (a
    /// Cauchy principal value when `p < 0`). See [`CarlsonKind`].
    struct CarlsonRj { x, y, z, p } => carlson_rj;

    /// Carlson `R_G(x, y, z)` - symmetric integral of the second kind. See [`CarlsonKind`].
    struct CarlsonRg { x, y, z } => carlson_rg;
}

/// Legendre request structs. Each maps to `ellint_impl::<KIND, COMPLETE>(phi, k, n)`; the three
/// trailing field names give the (phi, k, n) arguments - absent slots reuse `k` (ignored: COMPLETE
/// drops phi, non-Pi drops n).
macro_rules! decl_ellint {
    ($( $(#[$meta:meta])* struct $name:ident { $($field:ident),* } = [$kind:expr, $complete:expr]($phi:ident, $k:ident, $n:ident); )*) => {$(
        request_struct! { $(#[$meta])* $name { $($field),* } }

        impl<E, V> EllipticKind for $name<V>
        where
            E: FloatElement + EllipticConsts,
            V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
        {
            type Output = V;
            #[inline(always)]
            fn eval<P: Policy>(self) -> V {
                ellint_impl::<P, E, V, { $kind }, { $complete }>(self.$phi, self.$k, self.$n)
            }
        }
    )*};
}

decl_ellint! {
    /// Complete elliptic integral of the first kind, `K(k)`.
    struct EllintK { k } = [KIND_F, true](k, k, k);

    /// Incomplete elliptic integral of the first kind, `F(phi, k)`.
    struct EllintF { phi, k } = [KIND_F, false](phi, k, k);

    /// Complete elliptic integral of the second kind, `E(k)`.
    struct EllintE { k } = [KIND_E, true](k, k, k);

    /// Incomplete elliptic integral of the second kind, `E(phi, k)`.
    struct EllintEInc { phi, k } = [KIND_E, false](phi, k, k);

    /// Complete `D(k) = (K - E) / k^2`.
    struct EllintD { k } = [KIND_D, true](k, k, k);

    /// Incomplete `D(phi, k)`.
    struct EllintDInc { phi, k } = [KIND_D, false](phi, k, k);

    /// Complete elliptic integral of the third kind, `Pi(n, k)`.
    struct EllintPi { n, k } = [KIND_PI, true](k, k, n);

    /// Incomplete elliptic integral of the third kind, `Pi(n, phi, k)`.
    struct EllintPiInc { n, phi, k } = [KIND_PI, false](phi, k, n);
}

/// The members of the elliptic family that are _not_ Legendre integrals and so do not route
/// through [`ellint_impl`]. Same request-struct shape as [`decl_carlson`], evaluating a free
/// function of the struct's own fields, but implementing [`EllipticKind`] so they reach callers
/// through the same `ellint` entry point as their siblings.
macro_rules! decl_ellint_fn {
    ($( $(#[$meta:meta])* struct $name:ident { $($field:ident),* } => $func:ident; )*) => {$(
        request_struct! { $(#[$meta])* $name { $($field),* } }

        impl<E, V> EllipticKind for $name<V>
        where
            E: FloatElement + EllipticConsts,
            V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
        {
            type Output = V;
            #[inline(always)]
            fn eval<P: Policy>(self) -> V {
                $func::<P, E, V>($(self.$field),*)
            }
        }
    )*};
}

decl_ellint_fn! {
    /// Jacobi zeta `Z(phi, k)`: the oscillating part of `E(phi, k)`. See `jacobi_zeta`.
    struct JacobiZeta { phi, k } => jacobi_zeta;

    /// Heuman's lambda `Lambda_0(phi, k)`, the complementary-modulus companion to the complete
    /// integral of the third kind. See `heuman_lambda`.
    struct HeumanLambda { phi, k } => heuman_lambda;
}
