use thermite::{
    element::FloatElementWithBits,
    mask::GenericMask,
    math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy},
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;
use crate::tables::Trigamma;

/// Shared trigamma (`psi_1`) implementation for all real element types.
///
/// `psi_1(x) = d/dx psi(x)`, the second derivative of `ln Gamma`. Structurally much
/// cheaper than `digamma`: the reduction to the fitted range is a *single* step
/// rather than a masked walk, because the `[1, 2]` rational and the `x > 4` rational
/// between them already cover everything from 1 upward.
///
/// The poles at zero and the negative integers evaluate to `+inf`, which is the
/// correct two-sided limit (`psi_1` has a double pole there, so unlike `psi` the two
/// one-sided limits agree) and falls out of the reflection term for free.
#[inline(always)]
pub fn trigamma_impl<P, E, V>(x_in: V, t: &Trigamma<E>) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let x0 = x_in.flush_denormals_p::<P>();

    // --- Reflection for x <= 0: psi_1(x) = -psi_1(1 - x) + pi^2 / sin^2(pi*x) ---
    let reflect = x0.cmp_le(V::ZERO);
    let mut refl = V::ZERO;
    let mut x = x0;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        // Boost takes sin_pi of whichever of {x, 1 - x} is smaller in magnitude; on
        // this path that is always x, since |x| < |1 - x| for every x <= 0.
        let s = x0.sin_pi_p::<P>();

        // sin_pi is exactly zero at the negative integers, so those lanes divide by
        // zero and yield +inf - the intended pole value, no special case needed.
        refl = (V::PI_SQUARED / (s * s)).zz(reflect);
        x = reflect.select(V::ONE - x0, x);
    }

    // --- One recurrence step for 0 < x < 1: psi_1(x) = 1/x^2 + psi_1(x + 1) ---
    // One step suffices: the rational below is fit from 1 up, and the reflected lanes
    // already satisfy x >= 1.
    let mut acc = V::ZERO;
    let below_one = x.cmp_lt(V::ONE);

    if const { P::POLICY.avoid_branching } || below_one.any() {
        acc = (x * x).reciprocal_p::<P>().zz(below_one);
        x = x.add_c(below_one, V::ONE);
    }

    // --- Three minimax rational regions ---
    let small = x.cmp_le(V::TWO);
    let mid = x.cmp_le(V::splat(E::from_int(4))) & !small;
    let large = !(small | mid);

    // A single reciprocal covers both uses: it is the trailing 1/x^2 (small) or 1/x
    // (mid, large) scale factor, and on the x > 2 lanes it is *also* the polynomial
    // argument y = 1/x. Note the small lanes take 1/(x*x) directly rather than
    // squaring a reciprocal, so they keep the accuracy of a single rounding.
    let y = small.select(x * x, x).reciprocal_p::<P>();

    let mut num = V::EMPTY;
    let mut den = V::EMPTY;
    let mut base = V::ONE;

    if const { P::POLICY.avoid_branching } || small.any() {
        num = x.poly_p::<P, _>(&t.p_1_2);
        den = x.poly_p::<P, _>(&t.q_1_2);
        base = small.select(V::splat(t.offset), V::ONE);
    }

    if const { P::POLICY.avoid_branching } || mid.any() {
        num = mid.select(y.poly_p::<P, _>(&t.p_2_4), num);
        den = mid.select(y.poly_p::<P, _>(&t.q_2_4), den);
    }

    if const { P::POLICY.avoid_branching } || large.any() {
        num = large.select(y.poly_p::<P, _>(&t.p_4_inf), num);
        den = large.select(y.poly_p::<P, _>(&t.q_4_inf), den);
    }

    // Selecting the numerator and denominator before dividing keeps this to one
    // division for all three regions - and means an unselected region's overflow can
    // never turn into an inf/inf NaN in a lane that survives.
    let main = (base + num / den) * y;

    reflect.select(refl - main, acc + main)
}
