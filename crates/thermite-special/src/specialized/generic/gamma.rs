use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{DenormalBehavior, Policy, PrecisionPolicy, policies::ExtraPrecision},
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;
use crate::tables::{LANCZOS_F32, LANCZOS_F64, LN_MAX_F32, LN_MAX_F64, Lanczos};

/// Shared `tgamma` implementation for all real element types.
///
/// Covers only the `precision >= Average` path; the low-precision shortcut through
/// `lgamma_r` stays at the call site, because on f32 `lgamma_r` has its own
/// low-precision Pade branch that this module deliberately does not know about.
///
/// * `int_cap`: the largest integer whose factorial is finite in `E` (36 for f32,
///   172 for f64). Bounds the integer fast-path loop.
/// * `ln_max`: `ln(E::MAX)`, the overflow threshold for the `pow` in the main term.
#[inline(always)]
pub fn tgamma_impl<P, E, V, const N: usize>(z_in: V, l: &Lanczos<E, N>, int_cap: E, ln_max: E) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let mut z = z_in.flush_denormals_p::<P>();

    let orig_z = z;

    let is_negative = z.is_negative();
    let mut reflected = GenericMask::FALSY;

    let mut res = V::ONE;

    // Reflect ALL negative values via Γ(z) = -π / (z*sin(πz)*Γ(|z|))
    // This avoids the repeated-division recurrence which accumulates rounding error.
    if const { P::POLICY.avoid_branching } || is_negative.any() {
        reflected = is_negative;
        let refl_res = z * z.sin_pi_p::<P>(); // z * sin(πz)
        res = reflected.select(refl_res, res);
        z = z.abs();
    }

    // Negative integer poles and ±0
    let is_neg_int = is_negative & orig_z.cmp_eq(orig_z.floor()) & orig_z.cmp_ne(V::ZERO);
    let is_zero = orig_z.cmp_eq(V::ZERO);

    // Shift z ∈ (SQRT_EPSILON, 1) up by 1 via Γ(z) = Γ(z+1)/z.
    // The Lanczos polynomial is fit for z >= 1; evaluating below that is the
    // primary source of error in the (0, 1) range.
    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let needs_shift = z.cmp_lt(V::ONE) & z.cmp_ge(V::SQRT_EPSILON);
        res = needs_shift.select(res / z, res);
        z = needs_shift.select(z + V::ONE, z);
    }

    // Integers (positive, after reflection)

    let mut is_int = GenericMask::FALSY;
    let mut int_res = V::ONE;

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let zf = z.floor();
        // Capped at int_cap - Γ overflows beyond that, and this bounds the loop.
        is_int = zf.cmp_eq(z) & zf.cmp_lt(V::splat(int_cap)) & !is_neg_int & !is_zero;

        if thermite::unlikely(is_int.any()) {
            let mut j = V::ONE;
            // Mask with is_int so non-integer lanes with large zf can't keep the loop alive.
            let mut k = j.cmp_lt(zf) & is_int;

            while k.any() {
                int_res = k.select(int_res * j, int_res);
                j += V::ONE;
                k = j.cmp_lt(zf) & is_int;
            }

            if thermite::unlikely(is_int.all()) {
                return int_res;
            }
        }
    }

    // Full

    let gh = V::splat(l.g) - V::HALF;

    // Uses the leading-term-first (reversed) Lanczos arrays - see `Lanczos`.
    let lanczos_sum = z.poly_rev_p::<P, _>(&l.p_rev) / z.poly_rev_p::<P, _>(&l.q_rev);

    let zgh = z + gh;
    let lzgh = zgh.ln_p::<P>();

    // (z * lzfg) > ln(E::MAX)
    let very_large = (z * lzgh).cmp_gt(V::splat(ln_max));

    // only compute powf once
    let h = zgh.powf_p::<P>(very_large.select(z.mul_sube(V::HALF, V::splat(E::from_f64(0.25))), z - V::HALF));

    // save a couple cycles by avoiding this division, but worst-case precision is slightly worse
    let denom = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        lanczos_sum / zgh.exp_p::<P>()
    } else {
        lanczos_sum * (-zgh).exp_p::<P>()
    };

    let normal_res = very_large.select(h * h, h) * denom;

    // Tiny
    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let is_tiny = z.cmp_lt(V::SQRT_EPSILON);
        let tiny_res = z.reciprocal_p::<P>() - V::EULER_GAMMA;
        res *= is_tiny.select(tiny_res, normal_res);
    } else {
        res *= normal_res;
    }

    // Edge cases: Γ(-int) = NaN, Γ(±0) = ±∞
    let zero_res = is_negative.select(V::NEG_INFINITY, V::INFINITY);
    let result = reflected.select(-V::PI / res, is_int.select(int_res, res));
    let mut result = is_neg_int.select(V::NAN, result);

    if const {
        P::POLICY.precision.ge(PrecisionPolicy::Best)
            && matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
    } {
        let is_subnormal = z.is_subnormal();

        if thermite::unlikely(is_subnormal.any()) {
            result = is_subnormal.select(V::ONE / orig_z, result);
        }
    }

    is_zero.select(zero_res, result)
}

/// Shared `lgamma_r` implementation (log-gamma with its separate sign) for all real
/// element types, via the `exp(g)`-scaled Lanczos sum.
///
/// f32 short-circuits to a cheaper Pade approximant below `Average` precision; that
/// branch lives at the call site and never reaches here.
#[inline(always)]
pub fn lgamma_r_impl<P, E, V, const N: usize>(z_in: V, l: &Lanczos<E, N>) -> (V, V)
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let mut z = z_in.flush_denormals_p::<P>();
    let mut signum = V::ONE;

    let reflect = z.is_negative();

    let mut t = V::ONE;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        let pix = z * z.sin_pi_p::<P>(); // z * sin(pi * z)

        signum |= reflect.select(pix.signed_zero(), signum);

        t = reflect.select(pix.abs(), t);
        z = z.abs();
    }

    let b = z - V::HALF;
    let g = V::splat(l.g);

    let mut lanczos_sum = z.poly_rational_p::<P, _, _>(&l.p_expg_scaled, &l.q);

    // Full A term
    let mut a = (b + g).ln_p::<P>() - V::ONE;

    // tiny value handling
    if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
        let is_not_tiny = z.cmp_ge(V::SQRT_EPSILON);

        // shove the tiny result into the log down below
        lanczos_sum = is_not_tiny.select(lanczos_sum, z.reciprocal_p::<P>() - V::EULER_GAMMA);

        // force multiplier to zero for tiny case, allowing the modified
        // lanczos sum and ln(t) to be combined for cheap
        a = a.zz(is_not_tiny);
    }

    let c = (lanczos_sum * t).ln_p::<P>();

    let res = a.mul_adde(b, c);

    let y = reflect.select(V::LN_PI - res, res);

    (y, signum)
}

/// Shared `beta` implementation for all real element types.
///
/// `B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)`, evaluated from the `exp(g)`-scaled
/// Lanczos sums directly rather than through three `tgamma` calls, so the large
/// common factors cancel symbolically instead of overflowing.
///
/// Only defined for `a, b > 0`; anything else is NaN under `check_overflow`.
#[inline(always)]
pub fn beta_impl<P, E, V, const N: usize>(a: V, b: V, l: &Lanczos<E, N>) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let (a, b) = (a.flush_denormals_p::<P>(), b.flush_denormals_p::<P>());

    let is_valid = a.cmp_gt(V::ZERO) & b.cmp_gt(V::ZERO);

    if const { P::POLICY.check_overflow && !P::POLICY.avoid_branching } && is_valid.none() {
        return V::NAN;
    }

    let c = a + b;

    // if a < b then swap
    let (a, b) = (a.max(b), a.min(b));

    let mut result = a.poly_rational_p::<P, _, _>(&l.p_expg_scaled, &l.q)
        * (b.poly_rational_p::<P, _, _>(&l.p_expg_scaled, &l.q) / c.poly_rational_p::<P, _, _>(&l.p_expg_scaled, &l.q));

    let gh = V::splat(l.g) - V::HALF;

    let agh = a + gh;
    let bgh = b + gh;
    let cgh = c + gh;

    let agh_d_cgh = agh / cgh;
    let bgh_d_cgh = bgh / cgh;
    let agh_p_bgh = agh * bgh;
    let cgh_p_cgh = cgh * cgh;

    let base = cgh
        .cmp_gt(V::splat(E::from_f64(1e10)))
        .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

    let denom = if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
        V::SQRT_E / bgh.sqrt()
    } else {
        // bump up the precision a little to improve beta function accuracy
        V::SQRT_E * bgh.inverse_sqrt_p::<ExtraPrecision<P>>()
    };

    // encourage instruction-level parallelism
    result *= agh_d_cgh.powf_p::<P>(a - V::HALF - b) * (base.powf_p::<P>(b) * denom);

    if const { P::POLICY.check_overflow } {
        result = is_valid.select(result, V::NAN);
    }

    result
}
