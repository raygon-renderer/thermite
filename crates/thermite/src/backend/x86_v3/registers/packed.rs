//! [`PackedFloatRegister`] for the native 16-bit registers on x86-v3.
//!
//! The binary16 formats ([`Fp16`] / [`Fp16Fast`]) get an **F16C hardware override** when the
//! `avx2-f16c` crate feature is on - `vcvtph2ps` / `vcvtps2ph`, one instruction per 128-bit half
//! - and the **generic branchless defaults** otherwise. bfloat16 ([`Bf16`]) always uses the
//! generic default (F16C is a binary16<->binary32 converter only). The two are `#[cfg]`-exclusive
//! per `(format, register)` so they never collide.
//!
//! `avx2-f16c` also adds `f16c` to the `#[target_feature]` set the dispatch macros emit for this
//! backend (see [`X86V3_TARGET_FEATURE`] in `thermite-macros`); every AVX2 CPU also has F16C (it
//! shipped one generation earlier, on Ivy Bridge), so assuming it is safe. The F16C `vcvtph2ps`
//! decodes exactly and `vcvtps2ph` (round-to-nearest-ties-to-even) matches the [`Fp16`] oracle
//! bit-for-bit, validated in `tests/packed_float_f16c.rs`; the generic defaults are validated on
//! the native registers in `tests/packed_float_native.rs`.

use crate::element::float::spec::Bf16;
use crate::register::PackedFloatRegister;
use crate::register::array::ArrayRegister;

use super::{F32x4V3, F32x8V3, U16x8V3, U16x16V3};

// f32 partners (16-lane uses the native 256-bit f32x8 doubled).
type F32x16 = ArrayRegister<F32x8V3, 2>;

// ---- bf16: always the generic default (no F16C path) ------------------------------------
impl PackedFloatRegister<Bf16, F32x4V3> for super::half16::U16x4V3 {}
impl PackedFloatRegister<Bf16, F32x8V3> for U16x8V3 {}
impl PackedFloatRegister<Bf16, F32x16> for U16x16V3 {}

// ---- binary16: generic default when F16C is unavailable ---------------------------------
#[cfg(not(feature = "avx2-f16c"))]
mod fallback {
    use super::*;
    use crate::element::float::spec::{Fp16, Fp16Fast};

    macro_rules! impl_fp16 {
        ($u16:ty, $f32:ty) => {
            impl PackedFloatRegister<Fp16, $f32> for $u16 {}
            impl PackedFloatRegister<Fp16Fast, $f32> for $u16 {}
        };
    }
    impl_fp16!(super::super::half16::U16x4V3, F32x4V3);
    impl_fp16!(U16x8V3, F32x8V3);
    impl_fp16!(U16x16V3, F32x16);
}

// ---- binary16: F16C hardware override ---------------------------------------------------
#[cfg(feature = "avx2-f16c")]
mod f16c {
    use crate::element::float::spec::{Fp16, Fp16Fast};
    use crate::register::array::ArrayRegister;
    use crate::register::reduced::ReducedRegister;
    use crate::register::{BitwiseRegister, PackedFloatRegister, PartialOrdRegister, Register, Storage};

    use super::super::arch;
    use super::super::half16::U16x4V3;
    use super::super::{F32x4V3, F32x8V3, U16x8V3, U16x16V3};
    use super::F32x16;

    /// binary16 exponent field (all ones = inf/NaN code point).
    const F16_EXP: u16 = 0x7C00;
    /// binary16 sign bit.
    const F16_SIGN: u16 = 0x8000;

    /// Replace every all-ones-exponent (inf/NaN) half in a u16 lane register with *signed* zero,
    /// matching the `Unchecked` oracle's flush-to-zero. Built from the register's own trait ops
    /// (`bitand`/`eq`/`blendv`/`splat`) rather than raw intrinsics, so it works for any width.
    #[inline(always)]
    fn flush_nonfinite<R>(h: Storage<R>) -> Storage<R>
    where
        R: BitwiseRegister + PartialOrdRegister + Register<Element = u16>,
    {
        let nonfinite = R::eq(R::bitand(h, R::splat(F16_EXP)), R::splat(F16_EXP));
        let signed_zero = R::bitand(h, R::splat(F16_SIGN));
        R::blendv(nonfinite, h, signed_zero)
    }

    /// `vcvtps2ph` rounding-mode immediate: round to nearest, ties to even (matches the scalar
    /// [`Fp16`] oracle).
    ///
    /// Intel's docs recommend `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC` (= `0x08`), but Rust's
    /// stdarch guards this immediate with `static_assert_uimm_bits!(_, 3)`, so any value with bit 3
    /// set fails const-eval when `_mm256_cvtps_ph::<8>` is actually monomorphized (verified on the
    /// pinned `nightly-2026-04-11`: a dead, never-instantiated call elides the check, but the real
    /// build does not). `_MM_FROUND_NO_EXC` only suppresses x87/SSE exception flags, which we never
    /// read, so it is a no-op for us; the rounding field we need is just `_MM_FROUND_TO_NEAREST_INT`
    /// (0). See the upstream doc/assert inconsistency in `core_arch/src/x86/f16c.rs`.
    const RNE: i32 = arch::_MM_FROUND_TO_NEAREST_INT;

    // ---- 4-lane: u16x4 (4x binary16, low 64 bits) <-> f32x4 (4x f32, 128-bit) ---------------
    //
    // `u16x4` is a `ReducedRegister` over the 128-bit `U16x8V3`, so its data lives in the low 4
    // lanes of a `__m128i` - exactly the layout `_mm_cvtph_ps` reads and `_mm_cvtps_ph` writes (the
    // upper 64 bits are don't-care on input, zeroed on output).

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16, F32x4V3> for U16x4V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x4V3> {
            // 4x binary16 in the low 64 bits of a __m128i -> 4x f32 in a __m128.
            unsafe { arch::_mm_cvtph_ps(values.0) }
        }

        fn pack(values: Storage<F32x4V3>) -> Storage<Self> {
            // 4x f32 -> 4x binary16 in the low 64 bits, round to nearest even (upper 64 bits zeroed).
            ReducedRegister::new(unsafe { arch::_mm_cvtps_ph::<RNE>(values) })
        }
    }

    // ---- 8-lane: u16x8 (8x binary16, 128-bit) <-> f32x8 (8x f32, 256-bit) -------------------

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16, F32x8V3> for U16x8V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x8V3> {
            // 8x binary16 in a __m128i -> 8x f32 in a __m256.
            unsafe { arch::_mm256_cvtph_ps(values) }
        }

        fn pack(values: Storage<F32x8V3>) -> Storage<Self> {
            // 8x f32 -> 8x binary16, round to nearest even.
            unsafe { arch::_mm256_cvtps_ph::<RNE>(values) }
        }
    }

    // ---- 16-lane: u16x16 (16x binary16, 256-bit) <-> f32x16 (ArrayRegister<f32x8, 2>) -------

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16, F32x16> for U16x16V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x16> {
            // Split the 256-bit register into its two 128-bit halves (8 halves each) and widen each.
            unsafe {
                let lo = arch::_mm256_castsi256_si128(values);
                let hi = arch::_mm256_extracti128_si256(values, 1);
                ArrayRegister([arch::_mm256_cvtph_ps(lo), arch::_mm256_cvtph_ps(hi)])
            }
        }

        fn pack(values: Storage<F32x16>) -> Storage<Self> {
            // Narrow each 8-wide f32 half to 8 halves, then pack the two __m128i into one __m256i.
            unsafe {
                let lo = arch::_mm256_cvtps_ph::<RNE>(values.0[0]);
                let hi = arch::_mm256_cvtps_ph::<RNE>(values.0[1]);
                arch::_mm256_set_m128i(hi, lo)
            }
        }
    }

    // =========================================================================================
    // Fast / unchecked binary16 ([`Fp16Fast`], `SpecialEncoding::Unchecked`).
    //
    // `unpack` is the plain `vcvtph2ps` (the fast contract promises finite inputs, so we don't care
    // that the hardware would decode a stray inf/NaN code point as inf/NaN rather than the generic
    // fallback's "large normal" - both are unspecified for the inputs this mode rules out).
    //
    // `pack` must match the `Unchecked` oracle, which flushes any non-finite or overflowing input to
    // *signed* zero. `vcvtps2ph` already produces a finite half for in-range finite inputs and the
    // inf code point for inf/overflow (and a NaN half for NaN), so it suffices to run the result
    // through `flush_nonfinite`, which zeroes the magnitude (keeping the sign) of any all-ones-exponent
    // half.
    // =========================================================================================

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16Fast, F32x4V3> for U16x4V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x4V3> {
            unsafe { arch::_mm_cvtph_ps(values.0) }
        }

        fn pack(values: Storage<F32x4V3>) -> Storage<Self> {
            // cvtps_ph writes 4 halves in the low 64 bits, upper 64 zeroed; flush at the 8-lane level
            // (the zero upper lanes are finite, so they pass through untouched) and re-wrap as reduced.
            ReducedRegister::new(flush_nonfinite::<U16x8V3>(unsafe { arch::_mm_cvtps_ph::<RNE>(values) }))
        }
    }

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16Fast, F32x8V3> for U16x8V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x8V3> {
            unsafe { arch::_mm256_cvtph_ps(values) }
        }

        fn pack(values: Storage<F32x8V3>) -> Storage<Self> {
            flush_nonfinite::<U16x8V3>(unsafe { arch::_mm256_cvtps_ph::<RNE>(values) })
        }
    }

    #[thermite_macros::inline_always]
    impl PackedFloatRegister<Fp16Fast, F32x16> for U16x16V3 {
        fn unpack(values: Storage<Self>) -> Storage<F32x16> {
            unsafe {
                let lo = arch::_mm256_castsi256_si128(values);
                let hi = arch::_mm256_extracti128_si256(values, 1);
                ArrayRegister([arch::_mm256_cvtph_ps(lo), arch::_mm256_cvtph_ps(hi)])
            }
        }

        fn pack(values: Storage<F32x16>) -> Storage<Self> {
            unsafe {
                let lo = arch::_mm256_cvtps_ph::<RNE>(values.0[0]);
                let hi = arch::_mm256_cvtps_ph::<RNE>(values.0[1]);
                flush_nonfinite::<U16x16V3>(arch::_mm256_set_m128i(hi, lo))
            }
        }
    }
} // mod f16c
