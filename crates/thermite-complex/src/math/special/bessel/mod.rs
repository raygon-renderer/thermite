//! Bessel functions of complex argument.
//!
//! The kernels are `thermite-special`'s own, run in complex arithmetic. What this module
//! owns is the per-arithmetic _decisions_ those kernels make ([`BesselDetails`] for
//! `Complex`), the reduction of a complex-typed order to the real one the kernels take,
//! and, as the port proceeds, the rotations and continuations that build the rest of
//! the family out of `I` and `K` on the right half-plane. Reference: Amos (TOMS 644).
//! The plan is `notes/complex-bessel/PLAN.md`.

use thermite::element::FloatElement;
use thermite::math::policy::Policy;
use thermite::math::{PrimalProjection, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::BesselOrder;
use thermite_special::specialized::{BesselDetails, kernels};
use thermite_special::tables::lgamma1p::LogGamma1p;

use crate::Complex;
use crate::vector::RealFloatVector;

pub mod airy;

/// `$H^{(1)}_\nu$`, the Hankel function that decays in the upper half-plane. A marker for
/// [`hankel`](crate::math::special::ComplexSpecialMath::hankel). Under
/// [`Scaled`](thermite_special::bessel::Scaled) it is SciPy's `hankel1e`, `$e^{-iz}H^{(1)}_\nu$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H1;

/// `$H^{(2)}_\nu$`, decaying in the lower half-plane. Scaled: `$e^{iz}H^{(2)}_\nu$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H2;

/// A Hankel selector: [`H1`], [`H2`], or either under
/// [`Scaled`](thermite_special::bessel::Scaled). The marker's impl is the dispatch into the
/// four-output `(J, Y, H1, H2)` kernel, at the scaling it names.
pub trait HankelFn: Copy {
    #[doc(hidden)]
    fn eval<P: Policy, E, V>(z: V, order: BesselOrder<V::Real, <V::Real as GenericVector>::Signed>) -> V
    where
        V: super::SpecializedComplexSpecialMath<E>;
}

macro_rules! impl_hankel_fn {
    ($($marker:ty => $slot:tt / $scaled:literal;)*) => {$(
        impl HankelFn for $marker {
            #[inline(always)]
            fn eval<P: Policy, E, V>(z: V, order: BesselOrder<V::Real, <V::Real as GenericVector>::Signed>) -> V
            where
                V: super::SpecializedComplexSpecialMath<E>,
            {
                z.complex_bessel_jyh::<P, $scaled>(order).$slot
            }
        }
    )*};
}

impl_hankel_fn! {
    H1 => 2 / false;
    H2 => 3 / false;
    thermite_special::bessel::Scaled<H1> => 2 / true;
    thermite_special::bessel::Scaled<H2> => 3 / true;
}

/// The real-order modified Bessel kernel's decisions over C (see
/// [`BesselDetails`]). The arithmetic is inherited unchanged from `thermite-special`. Only
/// what the kernel _compares_ is restated here, because on `Complex` the default
/// comparisons are a lexicographic order and not a modulus.
///
/// Phase 0 of the port: the closed right half-plane, `Re z >= 0`, which is where Amos's own
/// `zbknu` / `zwrsk` / `zasyi` run and where every arm below is valid as written. The left
/// half-plane needs the analytic continuation (`zacon`), and is not here yet.
impl<V: RealFloatVector + PrimalProjection<Primal = V>> BesselDetails<Complex<V>> for Complex<V> {
    /// `|z| <= 2`, compared as `norm_sqr <= 4` to skip the root.
    #[inline(always)]
    fn near(z: Complex<V>) -> V::Mask {
        z.norm_sqr().cmp_le(V::TWO * V::TWO)
    }

    /// `|z| >= threshold`, likewise squared. Squaring is monotone and the threshold is
    /// positive, so this is the same test.
    #[inline(always)]
    fn beyond(z: Complex<V>, threshold: V) -> V::Mask {
        z.norm_sqr().cmp_ge(threshold * threshold)
    }

    /// The closed right half-plane less the origin. `-0.0` on the real part counts as
    /// on the axis, which `cmp_ge` gives and a sign-bit test would not.
    #[inline(always)]
    fn valid(z: Complex<V>) -> V::Mask {
        z.re.cmp_ge(V::ZERO) & !z.is_zero() & !z.im.is_nan()
    }

    /// The exponential's overflow is governed by `Re z` alone.
    #[inline(always)]
    fn exp_far(z: Complex<V>, threshold: V) -> V::Mask {
        z.re.cmp_ge(threshold)
    }

    /// Off the real axis the second exponential of the large-`|z|` expansion is
    /// `e^{-2 Re z}` relative, which on the imaginary axis is one. Kept.
    const ASYM_TWO_TERMS: bool = true;

    /// `-2z + (nu + 1/2) pi i` with the sign of `Im z` (DLMF 10.40.5, upper sign for
    /// `ph z` in `(-pi/2, 3pi/2)`, lower for its mirror. On the overlap the term is
    /// exponentially small and either sign is correct).
    #[inline(always)]
    fn asym_second_exponent(z: Complex<V>, nu: V) -> Complex<V> {
        let phase = ((nu + V::HALF) * V::PI).copysign(z.im);
        Complex::new(-(z.re + z.re), phase - (z.im + z.im))
    }
}

/// A complex-typed [`BesselOrder`] reduced to the real one the kernels take.
///
/// The order is real by construction in every variant but `Real`, whose payload is the
/// vector type (`Complex<V>` here). Complex _order_ is not implemented (Amos does not do it
/// either), so the imaginary part of a `Real` order is not silently dropped: the lanes that
/// carry one are reported back and the caller makes them NaN.
#[inline(always)]
pub(crate) fn real_order<V: RealFloatVector>(
    order: BesselOrder<Complex<V>, V::Signed>,
) -> (BesselOrder<V, V::Signed>, V::Mask) {
    match order {
        BesselOrder::Integer(k) => (BesselOrder::Integer(k), <V::Mask as GenericMask>::FALSY),
        BesselOrder::HalfInteger(k) => (BesselOrder::HalfInteger(k), <V::Mask as GenericMask>::FALSY),
        BesselOrder::Thirds(k) => (BesselOrder::Thirds(k), <V::Mask as GenericMask>::FALSY),
        BesselOrder::Real(c) => (BesselOrder::Real(c.re), !c.im.is_zero()),
    }
}

/// `$e^{i\theta}$` as a unit complex number, from one `sin_cos`.
#[inline(always)]
fn phasor<P: Policy, V: RealFloatVector>(theta: V) -> Complex<V> {
    let (s, c) = theta.sin_cos_p::<P>();
    Complex::new(c, s)
}

/// `$i z$`.
#[inline(always)]
fn times_i<V: RealFloatVector>(z: Complex<V>) -> Complex<V> {
    Complex::new(-z.im, z.re)
}

/// `$c\, e^{|x|}$` for a real `x`, with the exponential halved past `far_threshold` so a
/// value that is representable is not lost to a single overflowing `e^{|x|}`.
#[inline(always)]
fn unscale_by_real<P: Policy, V: RealFloatVector>(c: Complex<V>, x: V, far_threshold: V::Element) -> Complex<V> {
    let ax = x.abs();
    let far = ax.cmp_ge(V::splat(far_threshold));
    if thermite::unlikely(far.any()) {
        let half = (ax * V::HALF).exp_p::<P>();
        far.select((c * half) * half, c * ax.exp_p::<P>())
    } else {
        c * ax.exp_p::<P>()
    }
}

/// `$(I_\nu(z), K_\nu(z))$` over the **whole plane**, at real order, from the right-half-plane
/// kernel by analytic continuation.
///
/// With `SCALED`, the scalings are SciPy's `ive` / `kve`: `$e^{-|\mathrm{Re}\,z|}I_\nu(z)$`
/// and `$e^{z}K_\nu(z)$`. Both are bounded over the whole plane, which is what makes the
/// continuation safe to write down without Amos's overflow bookkeeping.
///
/// # The continuation, and why `I` is free and `K` is not
///
/// For `$\mathrm{Re}\,z < 0$` write `$z = w e^{\pm i\pi}$` with `$w = -z$` in the right
/// half-plane, the upper sign when `$\mathrm{Im}\,z \ge 0$` (the sign bit, so `$-0$` takes
/// the lower side: that is the branch cut on the negative real axis, and lands where the
/// principal `$z^\nu$` puts it). Then (DLMF 10.34.1-2)
///
/// ```math
/// I_\nu(w e^{\pm i\pi}) = e^{\pm i\nu\pi} I_\nu(w), \qquad
/// K_\nu(w e^{\pm i\pi}) = e^{\mp i\nu\pi} K_\nu(w) \mp i\pi\, I_\nu(w)
/// ```
///
/// `I` is a rotation, exact at whole orders because the phase comes from `sincos_pi`. `K`
/// picks up an `I` term, which off the imaginary axis dominates it exponentially. In
/// the scaled domain that is `$e^{-2w}$` on the `K` term against a bounded `$e^{-w}I$`, so
/// nothing overflows and nothing cancels catastrophically. Amos's `zacon` / `zs1s2` spend
/// their length on exactly the overflow this scaling removes.
///
/// A lane wanting `K` on the left half-plane therefore needs `I` as well, so `NEED_I` is
/// promoted to `true` whenever any lane is there.
#[inline(always)]
pub(crate) fn ik_whole_plane<P, E, V, const NE: usize, const NO: usize, const SCALED: bool, const NEED_I: bool>(
    nu: V,
    z: Complex<V>,
    t: &LogGamma1p<E, NE, NO>,
    far_threshold: E,
) -> (Complex<V>, Complex<V>)
where
    E: FloatElement,
    V: RealFloatVector<Element = E> + PrimalProjection<Primal = V> + FloatVectorWithBits,
    P: Policy,
{
    let left = z.re.cmp_lt(V::ZERO);
    // The half-plane by the SIGN BIT of `Im z`, so `-0.0` takes the lower side: that is the
    // branch cut on the negative real axis, landing where the principal `z^nu` puts it.
    let s = V::ONE.copysign(z.im);
    let w = z.neg_c(left);

    let (is, ks) = if left.any() {
        kernels::bessel_ik_real::<P, E, V, Complex<V>, NE, NO, true, true>(nu, w, t, far_threshold)
    } else {
        kernels::bessel_ik_real::<P, E, V, Complex<V>, NE, NO, true, NEED_I>(nu, w, t, far_threshold)
    };

    // `e^{-w}I(w)` to `e^{-Re w}I(w)`: a unit rotation by `Im w`. On the left `Im w = -Im z`.
    let mut i_out = match const { NEED_I } {
        false => Complex::ZERO,
        true => is * phasor::<P, V>(w.im),
    };
    let mut k_out = ks;

    if left.any() {
        // `e^{+- i nu pi}` from `sincos_pi`, so a whole order rotates by exactly `+-1`.
        let (sn, cs) = nu.sincos_pi_p::<P>();
        let rot = Complex::new(cs, sn * s);
        let rot_c = Complex::new(cs, -(sn * s));

        if const { NEED_I } {
            i_out = left.select(i_out * rot, i_out);
        }

        // `e^{z}K(z) = e^{-+ i nu pi} (e^{w}K(w)) e^{-2w} -+ i pi (e^{-w}I(w))`.
        let k_left = (rot_c * ks) * (-(w + w)).exp_p::<P>();
        let pi_i = is * (V::PI * s);
        k_out = left.select(k_left - times_i(pi_i), k_out);
    }

    match const { SCALED } {
        true => (i_out, k_out),
        false => {
            let i_out = match const { NEED_I } {
                false => i_out,
                true => unscale_by_real::<P, V>(i_out, z.re, far_threshold),
            };
            (i_out, k_out * (-z).exp_p::<P>())
        }
    }
}

/// `$(J_\nu, Y_\nu, H^{(1)}_\nu, H^{(2)}_\nu)$` at `z`, real order, over the whole plane.
///
/// With `SCALED`, SciPy's `jve` / `yve` / `hankel1e` / `hankel2e`: `$e^{-|\mathrm{Im}\,z|}$`
/// on `J` and `Y`, `$e^{-iz}$` on `$H^{(1)}$` and `$e^{iz}$` on `$H^{(2)}$`, each the
/// factor that keeps its function bounded.
///
/// # Everything is one `I`/`K` evaluation at a rotated argument
///
/// Amos's `zbesj` and `zbesh` are exactly this. With `$w = -isz$` for `$s = \pm 1$` (upper
/// sign for `$\mathrm{Im}\,z \ge 0$`, so `$w$` lands in the right half-plane),
///
/// ```math
/// J_\nu(z) = e^{is\nu\pi/2} I_\nu(w), \qquad
/// H^{(1)}_\nu(z) = -\tfrac{2i}{\pi} e^{-i\nu\pi/2} K_\nu(w) \;(s = +1), \qquad
/// H^{(2)}_\nu(z) = \tfrac{2i}{\pi} e^{i\nu\pi/2} K_\nu(w) \;(s = -1).
/// ```
///
/// The Hankel function _native_ to a half-plane is the one that decays there, and comes
/// straight from `K`. The other one comes from `$H^{(1)} + H^{(2)} = 2J$`, and `Y` from
/// `$H^{(1)} = J + iY$` (upper) or `$H^{(2)} = J - iY$` (lower). In the scaled domain every
/// one of those steps is bounded: the subtracted term carries `$e^{2isz}$`, which is small
/// exactly where the dominant one is large.
///
/// The scaling `$e^{-w}$` the kernel returns is `$e^{isz}$`, so `J`'s SciPy scaling
/// `$e^{-|\mathrm{Im}\,z|}$` is one unit rotation by `$-s\,\mathrm{Re}\,z$` away. That
/// same rotation (conjugated) is what the `Y` and other-`H` steps need. One `sin_cos` of
/// `Re z`, one `sincos_pi` of `nu/2`, one complex `exp`.
#[inline(always)]
pub(crate) fn jyh_whole_plane<P, E, V, const NE: usize, const NO: usize, const SCALED: bool>(
    nu: V,
    z: Complex<V>,
    t: &LogGamma1p<E, NE, NO>,
    far_threshold: E,
) -> (Complex<V>, Complex<V>, Complex<V>, Complex<V>)
where
    E: FloatElement,
    V: RealFloatVector<Element = E> + PrimalProjection<Primal = V> + FloatVectorWithBits,
    P: Policy,
{
    // Sign bit, as in `ik_whole_plane`: `-0.0` is the lower side of the cut.
    let s = V::ONE.copysign(z.im);
    let upper = s.cmp_gt(V::ZERO);

    // Everything below is at `a = |nu|`. A negative order is rotated at the end, exactly as
    // the real kernel does it. Going through the kernel's own `I` reflection instead would
    // hand `Y_{-a}` over as `H - J`, which at a negative half-integer is two large terms
    // cancelling to `J_a`, measured 11 relative at `nu = -5/2, z = 0.001`.
    let a = nu.abs();
    let reflected = nu.is_negative();

    // w = -i s z: (Im z, -Re z) for the upper half, (-Im z, Re z) for the lower.
    let w = Complex::new(z.im * s, -(z.re * s));
    let (is, ks) = kernels::bessel_ik_real::<P, E, V, Complex<V>, NE, NO, true, true>(a, w, t, far_threshold);

    let (sn, cs) = (a * V::HALF).sincos_pi_p::<P>();
    let rot = Complex::new(cs, sn * s); // e^{ i s nu pi/2}
    let rot_c = Complex::new(cs, -(sn * s)); // e^{-i s nu pi/2}

    let (sx, cx) = z.re.sin_cos_p::<P>();
    let e_neg = Complex::new(cx, -(sx * s)); // e^{-i s Re z}
    let e_pos = Complex::new(cx, sx * s); // e^{+i s Re z}

    // J e^{-|Im z|}.
    let js = (rot * is) * e_neg;

    // The Hankel function native to this half-plane, scaled by its own exponential:
    // (-s 2i/pi) e^{-i s nu pi/2} (e^{w} K(w)).
    let own = {
        let tk = rot_c * ks;
        let c = V::FRAC_2_PI * s;
        Complex::new(tk.im * c, -(tk.re * c))
    };

    // The other one: H_other = 2J - H_own, each scaled by the other's exponential, which
    // puts e^{2 i s z} on the subtracted term, small wherever it matters.
    let e2 = Complex::new(-(z.im + z.im) * s, (z.re + z.re) * s).exp_p::<P>();
    let other = (js + js) * e_pos - own * e2;

    // Y e^{-|Im z|} = -i s (H_own e^{-|Im z|} - J e^{-|Im z|}), and H_own carries its own
    // scaling e^{-isz}, so bringing it to J's costs e^{isz} e^{-|Im z|} = e^{2isz} e^{-is Re z}.
    let ys = {
        let tt = (own * e2) * e_neg - js;
        Complex::new(tt.im * s, -(tt.re * s))
    };

    let h1s = upper.select(own, other);
    let h2s = upper.select(other, own);

    // ---- negative order: the rotation, at every z (DLMF 10.4.6, 10.2.3-4) --------------
    //
    //     J_{-a} = J_a cos(a pi) - Y_a sin(a pi)      H1_{-a} = e^{ i a pi} H1_a
    //     Y_{-a} = J_a sin(a pi) + Y_a cos(a pi)      H2_{-a} = e^{-i a pi} H2_a
    //
    // Constant coefficients, so every scaling passes through. The identity rotation is
    // substituted on the other lanes rather than selected around, so the arithmetic below
    // is uniform, and `js * 1 - ys * 0` is `js` to the bit.
    let (sn, cs) = a.sincos_pi_p::<P>();
    let sn = reflected.select(sn, V::ZERO);
    let cs = reflected.select(cs, V::ONE);

    let j_out = js * cs - ys * sn;
    let y_out = js * sn + ys * cs;
    let h1_out = h1s * Complex::new(cs, sn);
    let h2_out = h2s * Complex::new(cs, -sn);

    // ---- the origin --------------------------------------------------------------------
    //
    // `J_a(0)` is 1 at a = 0 and 0 above it, `Y_a(0)` is `-inf`, and the rotation turns
    // those into `+-inf` where its sine / cosine is non-zero and into `0` where it is: the
    // half-integer exchange, `Y_{-5/2}(0) = J_{5/2}(0) = 0`. The formulas above reach every
    // one of these as `inf * 0`, so they are written out. The Hankel pair is `J -+ iY`.
    let zero = z.is_zero();
    let j0 = a.is_zero().select(V::ONE, V::ZERO);
    let jr = cs * j0 + V::INFINITY.copysign(sn).nz(sn.is_zero());
    let yr = sn * j0 - V::INFINITY.copysign(cs).nz(cs.is_zero());
    let js = zero.select(Complex::real(jr), j_out);
    let ys = zero.select(Complex::real(yr), y_out);
    let h1s = zero.select(Complex::new(jr, yr), h1_out);
    let h2s = zero.select(Complex::new(jr, -yr), h2_out);

    match const { SCALED } {
        true => (js, ys, h1s, h2s),
        false => {
            let j = unscale_by_real::<P, V>(js, z.im, far_threshold);
            let y = unscale_by_real::<P, V>(ys, z.im, far_threshold);
            let e_iz = times_i(z).exp_p::<P>();
            let e_miz = (-times_i(z)).exp_p::<P>();
            (j, y, h1s * e_iz, h2s * e_miz)
        }
    }
}
