//! `a*b - c*d` and `a*b + c*d` shared by the f32 and f64 backends.
//!
//! Kahan's compensated form recovers the rounding of one product with a second FMA.
//! That only works when the multiply-add is a single rounding, which is not true of
//! `Complex` or `Dual`, hence the `FloatVectorWithBits` bound. Composites build their
//! own out of the inner type's version.

use super::super::*;

/// Shared body of the real-vector [`SpecializedCoreMath::difference_of_products`]
/// override. See that method's docs for the three lowerings and the reasoning.
#[inline(always)]
pub fn difference_of_products_internal<V, E, P>(a: V, b: V, c: V, d: V) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
    P: Policy,
{
    let cd = c * d;

    if const { !matches!(V::HAS_NATIVE_FMA, tribool::True) } {
        // Without a fused multiply the naive form is the only one with the exactness
        // property, so it runs at every precision.
        a * b - cd
    } else if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
        a.mul_sub(b, cd)
    } else {
        a.mul_sub(b, cd) + c.nmul_add(d, cd) // value + recovered rounding of `cd`
    }
}

/// Shared body of the real-vector [`SpecializedCoreMath::sum_of_products`] override.
#[inline(always)]
pub fn sum_of_products_internal<V, E, P>(a: V, b: V, c: V, d: V) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
    P: Policy,
{
    let cd = c * d;

    if const { !matches!(V::HAS_NATIVE_FMA, tribool::True) } {
        a * b + cd
    } else if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
        a.mul_add(b, cd)
    } else {
        a.mul_add(b, cd) - c.nmul_add(d, cd)
    }
}
