//! Per-element-type constants for primitives that need precomputed irrational
//! values (regular polygons, stars, ...).
//!
//! The constants live on the scalar element types ([`f32`]/[`f64`]) via the
//! [`SdfConsts`] trait, so a generic primitive can splat them into whatever
//! vector type it is evaluated on with `V::splat(<V::Element>::CONST[i])`.

use thermite::element::FloatElement;
use thermite::vector::SplatConst;

use crate::SdfVector;

/// Compile-time rational constant `N/D`, splatted into the vector type `V`
/// (const-folded via [`FloatElement::ConstRatio`]).
#[inline(always)]
pub(crate) fn frac<V: SdfVector, const N: i64, const D: i64>() -> V {
    V::splat(const { <V::Element as FloatElement>::ConstRatio::<N, D>::VALUE })
}

/// Compile-time integer constant `N`, splatted into the vector type `V`
/// (const-folded via [`FloatElement::ConstInt`]).
#[inline(always)]
pub(crate) fn cint<V: SdfVector, const N: i64>() -> V {
    V::splat(const { <V::Element as FloatElement>::ConstInt::<N>::VALUE })
}

/// Compile-time constants used by the closed-form regular-polygon / star SDFs.
///
/// Implemented on the scalar element types; splat the values into a vector with
/// [`GenericVector::splat`](thermite::vector::GenericVector::splat).
pub trait SdfConsts: FloatElement {
    /// Regular pentagon (apothem `r`): `(cos(pi/5), sin(pi/5), tan(pi/5))`.
    const PENTAGON: [Self; 3];
    /// Regular octagon (apothem `r`): `(-cos(pi/8), sin(pi/8), tan(pi/8))`.
    const OCTAGON: [Self; 3];
    /// Hexagram: `(-1/2, sqrt(3)/2, 1/sqrt(3), sqrt(3))`.
    const HEXAGRAM: [Self; 4];
    /// Pentagram: `(cos(pi/5), sin(pi/10), sin(pi/5), cos(pi/10), tan(pi/5))`.
    const PENTAGRAM: [Self; 5];
}

impl SdfConsts for f32 {
    const PENTAGON: [f32; 3] = [0.809_017, 0.587_785_2, 0.726_542_5];
    const OCTAGON: [f32; 3] = [-0.923_879_5, 0.382_683_43, 0.414_213_6];
    const HEXAGRAM: [f32; 4] = [-0.5, 0.866_025_4, 0.577_350_26, 1.732_050_8];
    const PENTAGRAM: [f32; 5] = [0.809_017, 0.309_017, 0.587_785_2, 0.951_056_5, 0.726_542_5];
}

impl SdfConsts for f64 {
    const PENTAGON: [f64; 3] = [
        0.809_016_994_374_947_4,
        0.587_785_252_292_473_1,
        0.726_542_528_005_360_9,
    ];
    const OCTAGON: [f64; 3] = [
        -0.923_879_532_511_286_7,
        0.382_683_432_365_089_8,
        0.414_213_562_373_095_1,
    ];
    const HEXAGRAM: [f64; 4] = [
        -0.5,
        0.866_025_403_784_438_6,
        0.577_350_269_189_625_7,
        1.732_050_807_568_877_2,
    ];
    const PENTAGRAM: [f64; 5] = [
        0.809_016_994_374_947_4,
        0.309_016_994_374_947_45,
        0.587_785_252_292_473_1,
        0.951_056_516_295_153_5,
        0.726_542_528_005_360_9,
    ];
}
