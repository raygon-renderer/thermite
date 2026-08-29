//! Reciprocal square root shared by the f32 and f64 backends.

use super::super::*;

use crate::math::policy::DenormalBehavior;

#[inline(always)]
pub fn inverse_sqrt_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    // At `Best` and above, take the exact route rather than refining an estimate.
    // This mirrors `reciprocal` (specialized/mod.rs), which has always had the
    // same escape, and closes an asymmetry that made the two functions disagree
    // on identical input at identical policy.
    //
    // The approximate instruction is the reason. `rsqrtps` treats a denormal
    // operand as zero in hardware regardless of MXCSR, so the estimate comes back
    // `+inf`; the Newton step below then evaluates `inf * (inf * -tiny + 1.5)`,
    // which is `inf * -inf`, and the result is `-inf`. In f32, `inverse_sqrt(1e-38)`
    // returned `-inf` at every tier where `1.0e19` is the answer and is perfectly
    // representable. `ultra_performance` was the only tier with even the right sign,
    // because its `gt(Worst)` gate skips the refinement.
    //
    // f64 was never affected, since x86 has no approximate rsqrt for it before
    // AVX-512, so `rsqrt()` is already exact there and `HAS_APPROX_RSQRT` is false.
    //
    // So two escapes to the exact form: `Best` wanting full precision, and `Preserve`
    // forbidding the estimate outright. Neither needs `HAS_APPROX_RSQRT`, since with no
    // estimate `rsqrt()` IS `V::ONE / x.sqrt()` and the path below already lands here.
    if const {
        P::POLICY.precision.ge(PrecisionPolicy::Best)
            || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
    } {
        return V::ONE / x.sqrt();
    }

    let y0 = x.rsqrt();
    let mut y = y0;

    // The capability DOES gate the refinement: with no estimate `y` is already exact, and
    // a Newton step on an exact value is pure cost.
    if const { V::HAS_APPROX_RSQRT && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
        let nx2 = x.scale(const { E::ConstRatio::<{ -1 }, { 2 }>::VALUE }); // -0.5*x
        let threehalfs = V::splat(const { E::ConstRatio::<{ 3 }, { 2 }>::VALUE }); // 1.5

        // one iteration of Newton's method
        y = y0 * y0.square().mul_adde(nx2, threehalfs);

        if const { P::POLICY.check_overflow } {
            // The step is only valid where the estimate is finite and nonzero, which is
            // exactly the interior of the domain. At either end it manufactures a NaN out
            // of an answer that was already right:
            //
            //   x = 0    -> y0 = +inf, and the step is inf * (inf * -0.0 + 1.5)
            //   x = inf  -> y0 = 0,    and the step is 0   * (0   * -inf + 1.5)
            //
            // Both are `inf * NaN`. `inverse_sqrt(0.0)` and `inverse_sqrt(inf)` both
            // returned NaN at `Performance`, where `+inf` and `0` are the answers and both
            // are exactly what `rsqrt` already produced.
            //
            // Sibling of the denormal case documented above, one step further out. There
            // the estimate itself is wrong, so the only fix is to avoid it. Here the
            // estimate is right and the refinement is what breaks it, so keeping `y0` is
            // both the cheapest fix and the exactly correct one.
            y = y0.is_finite().bitandnot(y0.is_zero()).select(y, y0);
        }
    }

    y
}
