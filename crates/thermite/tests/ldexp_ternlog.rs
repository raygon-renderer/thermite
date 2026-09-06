//! Verify the `HAS_NATIVE_TERNLOG` arm of `ldexp`'s checked tail.
//!
//! No shipping backend sets `HAS_NATIVE_TERNLOG` yet (it is for AVX-512's
//! `vpternlog{d,q}`), so that arm is dead code today and would rot silently
//! until the x86_v4 backend lands. This test re-implements it verbatim over
//! the public vector API and differentially checks it against the shipped
//! blend arm, so both lowerings stay in agreement.
//!
//! DELETE THIS FILE once x86_v4 sets `HAS_NATIVE_TERNLOG`: the normal
//! differential suites then exercise that arm directly (under SDE), and this
//! becomes a hand-maintained copy of kernel source that can silently drift
//! from the original. It is scaffolding with a defined end, not a permanent
//! test - though it earned its keep by catching `ldexp(0.0, 300) == inf` in
//! the *shipped* arm, which no other test covered.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::element::FloatElementWithBits;
use thermite::math::policy::{DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};
use thermite::prelude::*;
use thermite::simd::Simd;

/// The two lowerings being compared are the two arms of `ldexp`'s **flush**
/// tail, so the policy has to pin `FlushToZero`: under `preserve_denormals` /
/// `strict_ieee754` the default policy takes the product-chain path instead,
/// which legitimately produces subnormals the flush tail cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlushPolicy;

impl Policy for FlushPolicy {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: true,
        precision: PrecisionPolicy::Average,
        avoid_branching: false,
        max_iterations: 10000,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

/// The ternlog bit-assembly tail, transcribed from
/// `SpecializedFloatMath::ldexp`'s `HAS_NATIVE_TERNLOG` branch.
fn ldexp_ternlog<V>(x: V, exp: V::SignedBits) -> V
where
    V: FloatVectorWithBits,
    V::Element: FloatElementWithBits,
{
    let mantissa_bits = <V::Element as FloatElementWithBits>::MANTISSA_BITS;
    let exp_lsb_mask: V::Bits = thermite::const_splat!(<V> = <S: FloatVectorWithBits>
        <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_LSB_MASK);
    let sign_mantissa_mask: V::Bits = thermite::const_splat!(<V> = <S: FloatVectorWithBits>
        <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::SIGN_MANTISSA_MASK);
    let max_biased_exp: V::SignedBits = thermite::const_splat!(<V> = <S: FloatVectorWithBits>
        <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::MAX_BIASED_EXP);
    let exp_bias: V::SignedBits = thermite::const_splat!(<V> = <S: FloatVectorWithBits>
        <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_BIAS);

    let bits: V::Bits = x.into_bits();
    let biased_exp = V::SignedBits::from_bits((bits >> mantissa_bits) & exp_lsb_mask);

    let exp_limit = exp_bias.shli::<2>();
    let exp = exp.max(-exp_limit).min(exp_limit);

    let new_exp = biased_exp + exp;
    let clamped_exp = new_exp.max(V::SignedBits::ZERO).min(max_biased_exp);

    let out_of_range = new_exp.cmp_le(V::SignedBits::ZERO) | new_exp.cmp_ge(max_biased_exp);
    let zero_sub = biased_exp.cmp_eq(V::SignedBits::ZERO);
    let non_finite = biased_exp.cmp_eq(max_biased_exp);

    let keep_mantissa =
        GenericMask::ternlog::<{ thermite::ternlog_imm!(!(A | B) | C) }>(out_of_range, zero_sub, non_finite);

    let ibits = V::SignedBits::from_bits(bits);
    let sign_bit = V::SignedBits::from_bits(<V as FloatVector>::NEG_ZERO);
    let sign_mantissa = V::SignedBits::from_bits(sign_mantissa_mask);
    let exp_field = max_biased_exp << mantissa_bits;

    let mantissa = V::SignedBits::ternlog::<{ thermite::ternlog_imm!(A & B & !C) }>(ibits, sign_mantissa, sign_bit)
        .zz(keep_mantissa);

    let field = V::SignedBits::ternlog::<{ thermite::ternlog_imm!(A | B | C) }>(
        (clamped_exp << mantissa_bits).nz(zero_sub),
        exp_field.zz(non_finite),
        mantissa,
    );

    V::from_bits(V::SignedBits::ternlog::<{ thermite::ternlog_imm!((A & B) | C) }>(
        ibits, sign_bit, field,
    ))
}

#[inline(always)]
fn check_f32<S: Simd>(name: &str) {
    let cases: [f32; 14] = [
        1.0,
        1.5,
        -1.5,
        0.0,
        -0.0,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1.0e-40, // subnormal
        -1.0e-40,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        core::f32::consts::PI,
    ];
    let exps: [i32; 13] = [0, 1, -1, 10, -10, 100, -100, 127, -126, 300, -300, i32::MAX, i32::MIN];

    for &x in &cases {
        for &e in &exps {
            let v = Vector::<S::f32x8>::splat(x);
            let ev = Vector::<S::i32x8>::splat(e);

            let blend = v.ldexp_p::<FlushPolicy>(ev).extract::<0>();
            let tern = ldexp_ternlog(v, ev).extract::<0>();

            assert_eq!(
                blend.to_bits(),
                tern.to_bits(),
                "[{name}] ldexp({x:e}, {e}): blend={blend:e} ternlog={tern:e}"
            );
        }
    }
}

#[inline(always)]
fn check_f64<S: Simd>(name: &str) {
    let cases: [f64; 12] = [
        1.0,
        1.5,
        -1.5,
        0.0,
        -0.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        5.0e-324, // subnormal
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        core::f64::consts::PI,
    ];
    let exps: [i64; 11] = [0, 1, -1, 100, -100, 1023, -1022, 3000, -3000, i64::MAX, i64::MIN];

    for &x in &cases {
        for &e in &exps {
            let v = Vector::<S::f64x4>::splat(x);
            let ev = Vector::<S::i64x4>::splat(e);

            let blend = v.ldexp_p::<FlushPolicy>(ev).extract::<0>();
            let tern = ldexp_ternlog(v, ev).extract::<0>();

            assert_eq!(
                blend.to_bits(),
                tern.to_bits(),
                "[{name}] ldexp({x:e}, {e}): blend={blend:e} ternlog={tern:e}"
            );
        }
    }
}

for_each_backend_concrete! {

    fn f32_arms_agree() {
        check_f32::<S>(&harness::label::<S>(""));
    }

    fn f64_arms_agree() {
        check_f64::<S>(&harness::label::<S>(""));
    }
}
