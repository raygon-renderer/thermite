//! The Airy functions of complex argument, `$\mathrm{Ai}$`, `$\mathrm{Bi}$` and their
//! derivatives, over the whole plane.
//!
//! Airy is Bessel at order `$\pm 1/3$` and `$\pm 2/3$`, with `$\zeta = \tfrac{2}{3}z^{3/2}$`,
//! the same decomposition the real kernel makes (`thermite-special`'s
//! `generic/bessel/airy.rs`) and Amos's `zairy`. Over C the plane splits into **two sectors**
//! by which Bessel pair keeps its principal branch:
//!
//! | sector | route | DLMF |
//! |---|---|---|
//! | `$\lvert\mathrm{ph}\,z\rvert \le 2\pi/3$` | `$K_{1/3}$`, `$K_{2/3}$`, `$I_{\pm 1/3}$`, `$I_{\pm 2/3}$` at `$\zeta$`, where `$\lvert\mathrm{ph}\,\zeta\rvert \le \pi$` | 9.6.1-9.6.4 |
//! | the wedge around the negative axis | `$J_{\pm 1/3}$`, `$J_{\pm 2/3}$` at `$\zeta_u = \tfrac{2}{3}u^{3/2}$`, `$u = -z$`, where `$\lvert\mathrm{ph}\,u\rvert < \pi/3$` | 9.6.6-9.6.9 |
//!
//! Past `$2\pi/3$` the principal `$\zeta$` has `$\lvert\mathrm{ph}\,\zeta\rvert > \pi$` and
//! `$K_{1/3}(\zeta)$` is no longer the principal `$K$`. That is what Amos's `zacai` is for,
//! and what the second sector avoids by going through `$-z$` instead. Both sectors reuse the
//! whole-plane `I`/`K` and `J`/`Y` from the parent module and the real kernel's exact
//! rotation constants at thirds (`$\cos\pi/3 = 1/2$`, `$\sin\pi/3 = \sqrt3/2$`).
//!
//! # Scalings
//!
//! SciPy's `airye`: `$e^{\zeta}\mathrm{Ai}$`, `$e^{\zeta}\mathrm{Ai}'$`,
//! `$e^{-\lvert\mathrm{Re}\,\zeta\rvert}\mathrm{Bi}$`, `$e^{-\lvert\mathrm{Re}\,\zeta\rvert}\mathrm{Bi}'$`,
//! with the principal `$\zeta$` everywhere. In the first sector those are exactly the
//! `kve` / `ive` scalings of the Bessel values they are built from. In the second they are a
//! unit rotation away from the `jve` / `yve` ones (`$\zeta = \mp i\zeta_u$` on the two sides
//! of the axis, so `$e^{\zeta} = e^{\mp i\,\mathrm{Re}\,\zeta_u}e^{-\lvert\mathrm{Im}\,\zeta_u\rvert}$`).
//! Nothing overflows in either sector.

use thermite::element::FloatElement;
use thermite::math::policy::Policy;
use thermite::math::{PrimalProjection, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::tables::bessel::airy::AiryZero;
use thermite_special::tables::lgamma1p::LogGamma1p;

use super::{ik_whole_plane, jyh_whole_plane, phasor, unscale_by_real};
use crate::Complex;
use crate::vector::RealFloatVector;

/// `$(\mathrm{Ai}, \mathrm{Ai}', \mathrm{Bi}, \mathrm{Bi}')$` at complex `z`, scaled as
/// SciPy's `airye` when `SCALED`. See the [module documentation](self).
#[inline(always)]
pub(crate) fn airy_whole_plane<P, E, V, const NE: usize, const NO: usize, const SCALED: bool>(
    z: Complex<V>,
    t: &LogGamma1p<E, NE, NO>,
    at_origin: &AiryZero<E>,
    far_threshold: E,
) -> (Complex<V>, Complex<V>, Complex<V>, Complex<V>)
where
    E: FloatElement,
    V: RealFloatVector<Element = E> + PrimalProjection<Primal = V> + FloatVectorWithBits,
    P: Policy,
{
    let third = V::splat(E::from_ratio(1, 3));
    let two_third = V::splat(E::from_ratio(2, 3));
    let three_half = V::splat(E::from_ratio(3, 2));
    let half_sqrt3 = V::SQRT_3 * V::HALF;

    // The principal zeta, and the origin. A zeta with both parts below the smallest normal
    // rounds every Airy value to its value at the origin, and a subnormal one would hand its
    // few bits back as relative error (measured on the real line). The Bessel passes get a
    // harmless argument on those lanes so no series is held open by a NaN.
    let sqrt_z = z.sqrt();
    let zeta = (z * sqrt_z) * two_third;
    let at_zero = zeta.is_zero_or_subnormal();
    let zeta = at_zero.select(Complex::ONE, zeta);

    // |ph z| <= 2pi/3  <=>  Re z >= -|Im z| / sqrt3.
    let sector_k = z.re.cmp_ge(-(z.im.abs() * V::FRAC_1_SQRT_3));

    let mut ai = Complex::ZERO;
    let mut aip = Complex::ZERO;
    let mut bi = Complex::ZERO;
    let mut bip = Complex::ZERO;

    // ---- |ph z| <= 2pi/3: K and I at zeta --------------------------------------------------
    //
    //     Ai  = (1/pi) sqrt(z/3) K_{1/3}         Bi  = sqrt(z/3) (I_{-1/3} + I_{1/3})
    //     Ai' = -(z/(pi sqrt3)) K_{2/3}          Bi' = (z/sqrt3) (I_{-2/3} + I_{2/3})
    //
    // with I_{-nu} = I_nu + (2/pi) sin(nu pi) K_nu and sin(pi/3) = sin(2pi/3) = sqrt3/2, so
    // each sum is 2I + (sqrt3/pi) K. In the scaled domain `kve = e^zeta K` is Ai's own
    // scaling and `ive = e^{-|Re zeta|} I` is Bi's. The K term inside Bi needs
    // e^{-zeta - |Re zeta|} to reach Bi's, which is bounded wherever this sector runs.
    if sector_k.any() {
        // On the rays themselves `zeta` sits on `K`'s cut, and rounding decides which side
        // of it `Im zeta` lands on, while `sqrt z` has already committed to a side. Inside
        // the sector `ph zeta = (3/2) ph z` has the sign of `Im z`, so pinning the sign is a
        // no-op there and, on the rays, keeps the continuation on the side `sqrt z` chose.
        // Measured before this: `Ai` on the `-2pi/3` ray came back negated.
        let zeta_k = Complex::new(zeta.re, zeta.im.copysign(z.im));
        let (i1, k1) = ik_whole_plane::<P, E, V, NE, NO, true, true>(third, zeta_k, t, far_threshold);
        let (i2, k2) = ik_whole_plane::<P, E, V, NE, NO, true, true>(two_third, zeta_k, t, far_threshold);

        let root_third = sqrt_z * V::FRAC_1_SQRT_3;
        let z_over_sqrt3 = z * V::FRAC_1_SQRT_3;
        let k_to_i = (-(zeta + Complex::real(zeta.re.abs()))).exp_p::<P>();
        let refl = V::SQRT_3 * V::FRAC_1_PI;

        ai = root_third * k1 * V::FRAC_1_PI;
        aip = -(z_over_sqrt3 * k2 * V::FRAC_1_PI);
        bi = root_third * ((i1 + i1) + (k1 * k_to_i) * refl);
        bip = z_over_sqrt3 * ((i2 + i2) + (k2 * k_to_i) * refl);
    }

    // ---- the wedge around the negative axis: J and Y at zeta_u, u = -z --------------------
    //
    //     Ai(-u)  = (sqrt u / 3) (J_{1/3} + J_{-1/3})        Bi(-u)  = sqrt(u/3) (J_{-1/3} - J_{1/3})
    //     Ai'(-u) = (u/3) (J_{2/3} - J_{-2/3})               Bi'(-u) = (u/sqrt3) (J_{-2/3} + J_{2/3})
    //
    // with J_{-nu} = J_nu cos(nu pi) - Y_nu sin(nu pi), exact at thirds. `jve`/`yve` carry
    // e^{-|Im zeta_u|}, which is `Bi`'s scaling here (|Re zeta| = |Im zeta_u|) and one unit
    // rotation short of `Ai`'s: zeta = -i s zeta_u with s the sign of Im z, so
    // e^zeta = e^{-i s Re zeta_u} e^{-|Im zeta_u|}.
    if !sector_k.all() {
        let u = -z;
        let sqrt_u = u.sqrt();
        let zeta_u = (u * sqrt_u) * two_third;
        // A lane at the origin is in the K sector, but keep this sector's argument sane anyway.
        let zeta_u = at_zero.select(Complex::ONE, zeta_u);

        let (j1, y1, _, _) = jyh_whole_plane::<P, E, V, NE, NO, true>(third, zeta_u, t, far_threshold);
        let (j2, y2, _, _) = jyh_whole_plane::<P, E, V, NE, NO, true>(two_third, zeta_u, t, far_threshold);

        let s = V::ONE.copysign(z.im);
        let rot = phasor::<P, V>(-(zeta_u.re * s));

        let ai_w = (sqrt_u * third) * (j1 * three_half - y1 * half_sqrt3);
        let bi_w = -((sqrt_u * V::FRAC_1_SQRT_3) * (j1 * V::HALF + y1 * half_sqrt3));
        let aip_w = (u * third) * (j2 * three_half + y2 * half_sqrt3);
        let bip_w = (u * V::FRAC_1_SQRT_3) * (j2 * V::HALF - y2 * half_sqrt3);

        ai = sector_k.select(ai, ai_w * rot);
        aip = sector_k.select(aip, aip_w * rot);
        bi = sector_k.select(bi, bi_w);
        bip = sector_k.select(bip, bip_w);
    }

    // ---- the origin ----------------------------------------------------------------------
    let ai = at_zero.select(Complex::real(V::splat(at_origin.ai)), ai);
    let aip = at_zero.select(Complex::real(V::splat(at_origin.aip)), aip);
    let bi = at_zero.select(Complex::real(V::splat(at_origin.bi)), bi);
    let bip = at_zero.select(Complex::real(V::splat(at_origin.bip)), bip);

    match const { SCALED } {
        true => (ai, aip, bi, bip),
        false => {
            // The scaled zeta on origin lanes is the substitute 1. The true one is 0, and
            // the unscaled value there is the tabled one already.
            let e_mz = at_zero.select(Complex::ONE, (-zeta).exp_p::<P>());
            let re_z = at_zero.select(V::ZERO, zeta.re);
            (
                ai * e_mz,
                aip * e_mz,
                unscale_by_real::<P, V>(bi, re_z, far_threshold),
                unscale_by_real::<P, V>(bip, re_z, far_threshold),
            )
        }
    }
}
