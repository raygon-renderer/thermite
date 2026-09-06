//! [`PackedFloatRegister`] for the 16-bit registers on x86-v4.
//!
//! binary16 ([`Fp16`] / [`Fp16Fast`]) is hardware at every width: F16C's `vcvtph2ps` /
//! `vcvtps2ph` are part of the v4 floor (`Avx512Features::F16C`), and AVX-512F adds the zmm
//! forms, so `u16x16 <-> f32x16` is one instruction each way instead of v3's two halves.
//! The rounding immediates differ by encoding: the F16C xmm/ymm forms take a 2-bit round
//! mode (`_MM_FROUND_TO_NEAREST_INT` = 0), the zmm form takes the extended
//! `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC` (= 8) and rejects a bare 0 at const-eval.
//! `vcvtps2ph` rounds to nearest-even and matches the scalar [`Fp16`] oracle bit-for-bit
//! (v3's `tests/packed_float_f16c.rs`).
//!
//! bfloat16 ([`Bf16`]) keeps the generic branchless default: the AVX512BF16 `vcvtneps2bf16`
//! is tier 3 only and the default is already a shift-and-round. A `F::AVX512BF16` fork is
//! an M4 item.

use crate::element::float::spec::{Bf16, Fp16, Fp16Fast};
use crate::register::array::ArrayRegister;
use crate::register::reduced::ReducedRegister;
use crate::register::{BitwiseRegister, PackedFloatRegister, PartialOrdRegister, Register, Storage};

use super::arch;
use super::half16::U16x4V4;
use super::{F32x4V4, F32x8V4, F32x16V4, U16x8V4, U16x16V4};

// ---- bf16: generic default at every width -------------------------------------------------
impl PackedFloatRegister<Bf16, F32x4V4> for U16x4V4 {}
impl PackedFloatRegister<Bf16, F32x8V4> for U16x8V4 {}
impl PackedFloatRegister<Bf16, F32x16V4> for U16x16V4 {}

/// binary16 exponent field (all ones = inf/NaN code point).
const F16_EXP: u16 = 0x7C00;
/// binary16 sign bit.
const F16_SIGN: u16 = 0x8000;

/// Replace every all-ones-exponent (inf/NaN) half with _signed_ zero, matching the
/// `Unchecked` oracle's flush-to-zero. Trait ops only, so it works at any width.
#[inline(always)]
fn flush_nonfinite<R>(h: Storage<R>) -> Storage<R>
where
    R: BitwiseRegister + PartialOrdRegister + Register<Element = u16>,
{
    let nonfinite = R::eq(R::bitand(h, R::splat(F16_EXP)), R::splat(F16_EXP));
    let signed_zero = R::bitand(h, R::splat(F16_SIGN));
    R::blendv(nonfinite, h, signed_zero)
}

/// F16C xmm/ymm `vcvtps2ph` round mode: nearest, ties to even.
const RNE: i32 = 0;
/// zmm `vcvtps2ph` extended rounding: nearest-even, exceptions suppressed.
const RNE_ZMM: i32 = 0x08;

// ---- binary16, checked ([`Fp16`]) -------------------------------------------------------

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16, F32x4V4> for U16x4V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x4V4> {
        unsafe { arch::_mm_cvtph_ps(values.0) }
    }

    fn pack(values: Storage<F32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtps_ph::<RNE>(values) })
    }
}

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16, F32x8V4> for U16x8V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x8V4> {
        unsafe { arch::_mm256_cvtph_ps(values) }
    }

    fn pack(values: Storage<F32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtps_ph::<RNE>(values) }
    }
}

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16, F32x16V4> for U16x16V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x16V4> {
        unsafe { arch::_mm512_cvtph_ps(values) }
    }

    fn pack(values: Storage<F32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtps_ph::<RNE_ZMM>(values) }
    }
}

// ---- binary16, unchecked ([`Fp16Fast`]): plain decode, encode + flush-to-signed-zero -------

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16Fast, F32x4V4> for U16x4V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x4V4> {
        unsafe { arch::_mm_cvtph_ps(values.0) }
    }

    fn pack(values: Storage<F32x4V4>) -> Storage<Self> {
        // The zeroed upper lanes are finite, so the 8-lane flush leaves them alone.
        ReducedRegister::new(flush_nonfinite::<U16x8V4>(unsafe { arch::_mm_cvtps_ph::<RNE>(values) }))
    }
}

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16Fast, F32x8V4> for U16x8V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x8V4> {
        unsafe { arch::_mm256_cvtph_ps(values) }
    }

    fn pack(values: Storage<F32x8V4>) -> Storage<Self> {
        flush_nonfinite::<U16x8V4>(unsafe { arch::_mm256_cvtps_ph::<RNE>(values) })
    }
}

#[thermite_macros::inline_always]
impl PackedFloatRegister<Fp16Fast, F32x16V4> for U16x16V4 {
    fn unpack(values: Storage<Self>) -> Storage<F32x16V4> {
        unsafe { arch::_mm512_cvtph_ps(values) }
    }

    fn pack(values: Storage<F32x16V4>) -> Storage<Self> {
        flush_nonfinite::<U16x16V4>(unsafe { arch::_mm512_cvtps_ph::<RNE_ZMM>(values) })
    }
}

// Keep the array import referenced for the doc links above.
#[allow(dead_code)]
type _F64x16 = ArrayRegister<super::F64x8V4, 2>;
