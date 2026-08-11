use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy,
        specialized::FlushDenormals,
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;
use crate::tables::Digamma;

/// Shared digamma (`psi`) implementation for all real element types.
///
/// `psi(x) = d/dx ln(Gamma(x))`. The element-specific rational/asymptotic
/// coefficients are passed in so the f32 and f64 specializations can share this
/// body:
///
/// * `y` / `roots` / `p_12` / `q_12`: the `[1, 2]` rational `psi(x) = (x - root)(Y + R(x-1))`,
///   where `root` is summed from `roots` via staged subtraction to preserve bits.
/// * `p_large`: the `x >= 10` asymptotic expansion in `1/(x-1)^2`.
#[inline(always)]
pub fn digamma_impl<P, E, V, const NR: usize, const NL: usize, const NP: usize, const NQ: usize>(
    x_in: V,
    t: &Digamma<E, NR, NL, NP, NQ>,
) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let mut x0 = x_in;

    #[cfg(not(target_arch = "spirv"))]
    if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x0]) {
        x0 = new_x[0];
    }

    let mut result = V::ZERO;
    let mut x = x0;

    // --- Reflection for x <= -1: psi(x) = psi(1-x) + pi*cot(pi*(1-x)) ---
    let reflect = x0.cmp_le(V::NEG_ONE);
    let mut refl_pole = GenericMask::FALSY;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        let xr = V::ONE - x0; // 1 - x, >= 2 for reflected lanes
        // fractional part shifted to (-1/2, 1/2] for tan argument reduction
        let mut rem = xr - xr.floor();
        rem = rem.sub_c(rem.cmp_gt(V::HALF), V::ONE);
        // pi * cot(pi*rem) = pi / tan(pi*rem); tan_pi is accurate near the poles
        let refl_term = V::PI / rem.tan_pi_p::<P>();
        result = refl_term.zz(reflect); // result is still zero here
        x = reflect.select(xr, x);
        refl_pole = reflect & rem.is_zero(); // reflected negative integer is a pole
    }

    // Large lanes (x >= 10) use the asymptotic expansion directly; smaller lanes are
    // reduced into [1, 2] via the recurrence psi(x) = psi(x+1) - 1/x.
    let large = x.cmp_ge(V::splat(E::from_int(10)));

    // Reduce into [1, 2]: lanes above 2 walk down (x -= 1, result += 1/x), lanes
    // below 1 walk up (result -= 1/x, x += 1). The two directions are disjoint per
    // lane, so one loop handles both with a single division per iteration.
    let mut active = (x.cmp_gt(V::TWO) | x.cmp_lt(V::ONE)) & !large;
    while active.any() {
        // sign(x-1) is +1 above the interval (walk down: x -= 1, add +1/(x-1)) and
        // -1 below it (walk up: x += 1, add -1/x). The reciprocal point is the smaller
        // of {x, x-sign}: x-1 when walking down, x when walking up.
        let sign = (x - V::ONE).signum();
        let xs = x.sub_c(active, sign); // step toward [1, 2]; inactive lanes keep x
        let term = sign * x.min(xs).reciprocal_p::<P>();
        result = result.add_c(active, term);
        x = xs;
        active = (x.cmp_gt(V::TWO) | x.cmp_lt(V::ONE)) & !large;
    }

    // x - 1 is shared by both the [1, 2] rational and the asymptotic expansion.
    let xm1 = x - V::ONE;

    // --- Rational approximation on [1, 2] (small lanes) ---
    // staged subtraction preserves bits: root = sum(roots)
    let mut g = x;
    let mut i = 0;
    while i < NR {
        g -= V::splat(t.roots[i]);
        i += 1;
    }
    let r = xm1.poly_p::<P, _>(&t.p_12) / xm1.poly_p::<P, _>(&t.q_12);
    let rational = g * (V::splat(t.y) + r);

    // --- Asymptotic expansion for x >= 10 (large lanes) ---
    // ln(x-1) + 1/(2(x-1)) - z*P(z), with the trailing product fused into an FMA.
    let z = (xm1 * xm1).reciprocal_p::<P>();
    let asymptotic = z.nmul_adde(
        z.poly_p::<P, _>(&t.p_large),
        xm1.ln_p::<P>() + (xm1 + xm1).reciprocal_p::<P>(),
    );

    // both paths share the accumulated recurrence term
    let mut res = result + large.select(asymptotic, rational);

    // --- Poles: x == 0 and the negative integers -> NaN ---
    if const { P::POLICY.check_overflow } {
        let pole = x0.is_zero() | refl_pole;
        res = pole.select(V::NAN, res);
        res = x0.is_nan().select(V::NAN, res);
    }

    res
}
