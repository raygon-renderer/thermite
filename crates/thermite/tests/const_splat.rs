#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//! Regression tests for the `const_splat!` macro arms.
//!
//! The `int`/`ratio` arms are not exercised anywhere in-tree, and that let two
//! latent bugs slip in: they referenced `FloatElement` associated types by an
//! outdated name (`IntSplat`/`RatioSplat` instead of `ConstInt`/`ConstRatio`),
//! and they were listed after the generic `($ty:ty: $value:expr)` arm, which
//! shadowed them entirely (`int <E>` parses as the type `int<E>`). Keep at
//! least one instantiation of every arm here so they stay compilable.

use thermite::Vector;
use thermite::backend::scalar::Scalar;
use thermite::prelude::*;
use thermite::simd::Simd;

type VF32 = Vector<<Scalar as Simd>::f32x4>;
type VF64 = Vector<<Scalar as Simd>::f64x2>;

// The int/ratio arms only make sense in a generic context, where the element
// type is not yet known.
fn int_splat<V: FloatVector>() -> V {
    thermite::const_splat!(int <V::Element>: 7i64)
}

fn ratio_splat<V: FloatVector>() -> V {
    thermite::const_splat!(ratio <V::Element>: 1i64, 4i64)
}

#[test]
fn const_splat_int_ratio_arms() {
    assert_eq!(int_splat::<VF32>().to_array(), [7.0f32; 4].into());
    assert_eq!(int_splat::<VF64>().to_array(), [7.0f64; 2].into());
    assert_eq!(ratio_splat::<VF32>().to_array(), [0.25f32; 4].into());
    assert_eq!(ratio_splat::<VF64>().to_array(), [0.25f64; 2].into());
}

#[test]
fn const_splat_static_and_assoc_arms() {
    let v: VF32 = thermite::const_splat!(f32: 1.5);
    assert_eq!(v.to_array(), [1.5f32; 4].into());

    let v: VF32 = thermite::const_splat!(<f32>::INFINITY);
    assert!(v.to_array().iter().all(|x| x.is_infinite() && x.is_sign_positive()));
}
