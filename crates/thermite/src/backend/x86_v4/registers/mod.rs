//! Native AVX-512 registers, generic over the compiled tier `F`.
//!
//! Only [`X86V4Default`](super::X86V4Default) is ever instantiated (one tier
//! per build), but everything here stays generic over [`Avx512Features`] so
//! tier forks are `if const { F::FEATURE }` reads, not `cfg` -- see the
//! backend module docs for why.

use crate::{
    element::FindUSize,
    isa::InstructionSet,
    register::{Storage, array::ArrayRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A},
};

use super::{Avx512Features, X86V4, arch};

pub mod kmask;

pub mod f32x16;
pub mod f64x8;
pub mod i32x16;
pub mod i64x8;
pub mod u32x16;
pub mod u64x8;

pub mod f32x8;
pub mod f64x4;
pub mod i32x8;
pub mod i64x4;
pub mod u32x8;
pub mod u64x4;

pub mod f32x4;
pub mod f64x2;
pub mod i32x4;
pub mod i64x2;
pub mod u32x4;
pub mod u64x2;

pub mod half;

pub mod half16;
pub mod half8;
pub mod i16x16;
pub mod i16x32;
pub mod i16x8;
pub mod i8x16;
pub mod i8x32;
pub mod i8x64;
pub mod packed;
pub mod u16x16;
pub mod u16x32;
pub mod u16x8;
pub mod u8x16;
pub mod u8x32;
pub mod u8x64;

pub use half::{F32x2V4, I32x2V4, KMask2Half, U32x2V4};
pub use kmask::{KMask2, KMask4, KMask8, KMask16, KMask32, KMask64};

pub use half8::{I8x4V4, I8x8V4, KMask4Of16, KMask8Of16, U8x4V4, U8x8V4};
pub use half16::{I16x4V4, KMask4Of8, U16x4V4};
pub use i8x16::I8x16V4;
pub use i8x32::I8x32V4;
pub use i8x64::I8x64V4;
pub use i16x8::I16x8V4;
pub use i16x16::I16x16V4;
pub use i16x32::I16x32V4;
pub use u8x16::U8x16V4;
pub use u8x32::U8x32V4;
pub use u8x64::U8x64V4;
pub use u16x8::U16x8V4;
pub use u16x16::U16x16V4;
pub use u16x32::U16x32V4;

pub use f32x16::F32x16V4;
pub use f64x8::F64x8V4;
pub use i32x16::I32x16V4;
pub use i64x8::I64x8V4;
pub use u32x16::U32x16V4;
pub use u64x8::U64x8V4;

pub use f32x8::F32x8V4;
pub use f64x4::F64x4V4;
pub use i32x8::I32x8V4;
pub use i64x4::I64x4V4;
pub use u32x8::U32x8V4;
pub use u64x4::U64x4V4;

pub use f32x4::F32x4V4;
pub use f64x2::F64x2V4;
pub use i32x4::I32x4V4;
pub use i64x2::I64x2V4;
pub use u32x4::U32x4V4;
pub use u64x2::U64x2V4;

impl_newregister!(
    F32x16V4, F64x8V4, I32x16V4, I64x8V4, U32x16V4, U64x8V4, F32x8V4, F64x4V4, I32x8V4, I64x4V4, U32x8V4, U64x4V4,
    F32x4V4, F64x2V4, I32x4V4, I64x2V4, U32x4V4, U64x2V4, I16x32V4, U16x32V4, I8x64V4, U8x64V4, I16x16V4, U16x16V4,
    I16x8V4, U16x8V4, I8x32V4, U8x32V4, I8x16V4, U8x16V4
);

impl_has_isa!(
    crate::backend::x86_v4::X86V4Default: F32x16V4, F64x8V4, I32x16V4, I64x8V4, U32x16V4, U64x8V4, F32x8V4, F64x4V4,
    I32x8V4, I64x4V4, U32x8V4, U64x4V4, F32x4V4, F64x2V4, I32x4V4, I64x2V4, U32x4V4, U64x2V4, I16x32V4, U16x32V4,
    I8x64V4, U8x64V4, I16x16V4, U16x16V4, I16x8V4, U16x8V4, I8x32V4, U8x32V4, I8x16V4, U8x16V4
);

// Same-width bit reinterprets. The integer matrix shares `__m512i` storage,
// so those are identity. Float <-> int route through the cast intrinsics.
impl_bit_casts! {
    F32x16V4 as I32x16V4 => _mm512_castps_si512,
    F32x16V4 as U32x16V4 => _mm512_castps_si512,
    I32x16V4 as F32x16V4 => _mm512_castsi512_ps,
    U32x16V4 as F32x16V4 => _mm512_castsi512_ps,

    F64x8V4 as I64x8V4 => _mm512_castpd_si512,
    F64x8V4 as U64x8V4 => _mm512_castpd_si512,
    I64x8V4 as F64x8V4 => _mm512_castsi512_pd,
    U64x8V4 as F64x8V4 => _mm512_castsi512_pd,

    F32x8V4 as I32x8V4 => _mm256_castps_si256,
    F32x8V4 as U32x8V4 => _mm256_castps_si256,
    I32x8V4 as F32x8V4 => _mm256_castsi256_ps,
    U32x8V4 as F32x8V4 => _mm256_castsi256_ps,

    F64x4V4 as I64x4V4 => _mm256_castpd_si256,
    F64x4V4 as U64x4V4 => _mm256_castpd_si256,
    I64x4V4 as F64x4V4 => _mm256_castsi256_pd,
    U64x4V4 as F64x4V4 => _mm256_castsi256_pd,

    F32x4V4 as I32x4V4 => _mm_castps_si128,
    F32x4V4 as U32x4V4 => _mm_castps_si128,
    I32x4V4 as F32x4V4 => _mm_castsi128_ps,
    U32x4V4 as F32x4V4 => _mm_castsi128_ps,

    F64x2V4 as I64x2V4 => _mm_castpd_si128,
    F64x2V4 as U64x2V4 => _mm_castpd_si128,
    I64x2V4 as F64x2V4 => _mm_castsi128_pd,
    U64x2V4 as F64x2V4 => _mm_castsi128_pd,

    // integer <-> integer: same storage
    I32x16V4 as U32x16V4 => identity,
    U32x16V4 as I32x16V4 => identity,
    I64x8V4 as U64x8V4 => identity,
    U64x8V4 as I64x8V4 => identity,
    I32x8V4 as U32x8V4 => identity,
    U32x8V4 as I32x8V4 => identity,
    I64x4V4 as U64x4V4 => identity,
    U64x4V4 as I64x4V4 => identity,
    I32x4V4 as U32x4V4 => identity,
    U32x4V4 as I32x4V4 => identity,
    I64x2V4 as U64x2V4 => identity,
    U64x2V4 as I64x2V4 => identity,

    I16x32V4 as U16x32V4 => identity,
    U16x32V4 as I16x32V4 => identity,
    I8x64V4 as U8x64V4 => identity,
    U8x64V4 as I8x64V4 => identity,
    I16x16V4 as U16x16V4 => identity,
    U16x16V4 as I16x16V4 => identity,
    I16x8V4 as U16x8V4 => identity,
    U16x8V4 as I16x8V4 => identity,
    I8x32V4 as U8x32V4 => identity,
    U8x32V4 as I8x32V4 => identity,
    I8x16V4 as U8x16V4 => identity,
    U8x16V4 as I8x16V4 => identity,

    // identity casts to self
    I16x32V4 as I16x32V4 => identity,
    U16x32V4 as U16x32V4 => identity,
    I8x64V4 as I8x64V4 => identity,
    U8x64V4 as U8x64V4 => identity,
    I16x16V4 as I16x16V4 => identity,
    U16x16V4 as U16x16V4 => identity,
    I16x8V4 as I16x8V4 => identity,
    U16x8V4 as U16x8V4 => identity,
    I8x32V4 as I8x32V4 => identity,
    U8x32V4 as U8x32V4 => identity,
    I8x16V4 as I8x16V4 => identity,
    U8x16V4 as U8x16V4 => identity,
    F32x16V4 as F32x16V4 => identity,
    F64x8V4 as F64x8V4 => identity,
    I32x16V4 as I32x16V4 => identity,
    U32x16V4 as U32x16V4 => identity,
    I64x8V4 as I64x8V4 => identity,
    U64x8V4 as U64x8V4 => identity,
    F32x8V4 as F32x8V4 => identity,
    I32x8V4 as I32x8V4 => identity,
    U32x8V4 as U32x8V4 => identity,
    F64x4V4 as F64x4V4 => identity,
    I64x4V4 as I64x4V4 => identity,
    U64x4V4 as U64x4V4 => identity,
    F32x4V4 as F32x4V4 => identity,
    I32x4V4 as I32x4V4 => identity,
    U32x4V4 as U32x4V4 => identity,
    F64x2V4 as F64x2V4 => identity,
    I64x2V4 as I64x2V4 => identity,
    U64x2V4 as U64x2V4 => identity,
}

impl_type_casts! {
    // self casts
    I16x32V4 as I16x32V4 => identity,
    U16x32V4 as U16x32V4 => identity,
    I16x32V4 as U16x32V4 => identity,
    U16x32V4 as I16x32V4 => identity,
    I8x64V4 as I8x64V4 => identity,
    U8x64V4 as U8x64V4 => identity,
    I8x64V4 as U8x64V4 => identity,
    U8x64V4 as I8x64V4 => identity,
    I16x16V4 as I16x16V4 => identity,
    U16x16V4 as U16x16V4 => identity,
    I16x16V4 as U16x16V4 => identity,
    U16x16V4 as I16x16V4 => identity,
    I16x8V4 as I16x8V4 => identity,
    U16x8V4 as U16x8V4 => identity,
    I16x8V4 as U16x8V4 => identity,
    U16x8V4 as I16x8V4 => identity,
    I8x32V4 as I8x32V4 => identity,
    U8x32V4 as U8x32V4 => identity,
    I8x32V4 as U8x32V4 => identity,
    U8x32V4 as I8x32V4 => identity,
    I8x16V4 as I8x16V4 => identity,
    U8x16V4 as U8x16V4 => identity,
    I8x16V4 as U8x16V4 => identity,
    U8x16V4 as I8x16V4 => identity,
    F32x16V4 as F32x16V4 => identity,
    F64x8V4 as F64x8V4 => identity,
    I32x16V4 as I32x16V4 => identity,
    U32x16V4 as U32x16V4 => identity,
    I64x8V4 as I64x8V4 => identity,
    U64x8V4 as U64x8V4 => identity,
    F32x8V4 as F32x8V4 => identity,
    I32x8V4 as I32x8V4 => identity,
    U32x8V4 as U32x8V4 => identity,
    F64x4V4 as F64x4V4 => identity,
    I64x4V4 as I64x4V4 => identity,
    U64x4V4 as U64x4V4 => identity,
    F32x4V4 as F32x4V4 => identity,
    I32x4V4 as I32x4V4 => identity,
    U32x4V4 as U32x4V4 => identity,
    F64x2V4 as F64x2V4 => identity,
    I64x2V4 as I64x2V4 => identity,
    U64x2V4 as U64x2V4 => identity,

    // int -> float: all four directions are native at v4 (epu32 is F,
    // the 64-bit pair is DQ = floor), no magic-number polyfills.
    I32x16V4 as F32x16V4 => _mm512_cvtepi32_ps,
    U32x16V4 as F32x16V4 => _mm512_cvtepu32_ps,
    I64x8V4 as F64x8V4 => _mm512_cvtepi64_pd,
    U64x8V4 as F64x8V4 => _mm512_cvtepu64_pd,
    I32x8V4 as F32x8V4 => _mm256_cvtepi32_ps,
    U32x8V4 as F32x8V4 => _mm256_cvtepu32_psx_v4,
    I64x4V4 as F64x4V4 => _mm256_cvtepi64_pd,
    U64x4V4 as F64x4V4 => _mm256_cvtepu64_pd,
    I32x4V4 as F32x4V4 => _mm_cvtepi32_ps,
    U32x4V4 as F32x4V4 => _mm_cvtepu32_psx_v4,
    I64x2V4 as F64x2V4 => _mm_cvtepi64_pd,
    U64x2V4 as F64x2V4 => _mm_cvtepu64_pd,

    // 8-lane cross-width int <-> float (ymm <-> zmm), all single converts.
    I32x8V4 as F64x8V4 => _mm512_cvtepi32_pd,
    U32x8V4 as F64x8V4 => _mm512_cvtepu32_pd,
    I64x8V4 as F32x8V4 => _mm512_cvtepi64_ps,
    U64x8V4 as F32x8V4 => _mm512_cvtepu64_ps,

    // 4-lane cross-width int <-> float (xmm <-> ymm).
    I32x4V4 as F64x4V4 => _mm256_cvtepi32_pd,
    U32x4V4 as F64x4V4 => _mm256_cvtepu32_pd,
    I64x4V4 as F32x4V4 => _mm256_cvtepi64_ps,
    U64x4V4 as F32x4V4 => _mm256_cvtepu64_ps,

    // integer sign changes: bit casts
    I32x16V4 as U32x16V4 => identity,
    U32x16V4 as I32x16V4 => identity,
    I64x8V4 as U64x8V4 => identity,
    U64x8V4 as I64x8V4 => identity,
    I32x8V4 as U32x8V4 => identity,
    U32x8V4 as I32x8V4 => identity,
    I64x4V4 as U64x4V4 => identity,
    U64x4V4 as I64x4V4 => identity,
    I32x4V4 as U32x4V4 => identity,
    U32x4V4 as I32x4V4 => identity,
    I64x2V4 as U64x2V4 => identity,
    U64x2V4 as I64x2V4 => identity,
}

// 32-bit int -> f64 at 16 lanes (the x16 grid slot is two zmm): one
// `vcvtdq2pd`/`vcvtudq2pd` per ymm index half. The `ArrayRegister` cast
// ladder derives everything else reaching this slot.
impl crate::register::CastRegister<I32x16V4> for crate::register::array::ArrayRegister<F64x8V4, 2> {
    #[inline(always)]
    fn cast_from(value: Storage<I32x16V4>) -> Storage<Self> {
        let (lo, hi) = <I32x16V4 as crate::register::ConcatRegister<I32x8V4>>::split(value);

        unsafe { crate::register::array::ArrayRegister([arch::_mm512_cvtepi32_pd(lo), arch::_mm512_cvtepi32_pd(hi)]) }
    }
}

impl crate::register::CastRegister<U32x16V4> for crate::register::array::ArrayRegister<F64x8V4, 2> {
    #[inline(always)]
    fn cast_from(value: Storage<U32x16V4>) -> Storage<Self> {
        let (lo, hi) = <U32x16V4 as crate::register::ConcatRegister<U32x8V4>>::split(value);

        unsafe { crate::register::array::ArrayRegister([arch::_mm512_cvtepu32_pd(lo), arch::_mm512_cvtepu32_pd(hi)]) }
    }
}

// The native 512-bit byte/word registers as the SAD sources (`u8xN`/`u16xN`):
// same-width reinterprets to the accumulator widths are identity (`__m512i`),
// and `vpsadbw`/`vpmaddubsw`/`vpmaddwd` all have zmm forms under BW.
impl_bit_casts_identity! {
    U8x64V4 as U16x32V4,
    U8x64V4 as U32x16V4,
    U8x64V4 as U64x8V4,
    U16x32V4 as U32x16V4,
    U16x32V4 as U64x8V4,
}

const _: () = {
    use crate::register::{Sad16Register, Sad32Register, Sad64Register, UnsignedIntegerRegister};

    impl Sad16Register<U16x32V4> for U8x64V4 {
        #[inline(always)]
        fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<U16x32V4> {
            unsafe { arch::_mm512_maddubs_epi16(Self::abs_diff(a, b), arch::_mm512_set1_epi8(1)) }
        }
    }

    impl Sad32Register<U32x16V4> for U8x64V4 {
        #[inline(always)]
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x16V4> {
            unsafe {
                arch::_mm512_madd_epi16(
                    <Self as Sad16Register<U16x32V4>>::sad16(a, b),
                    arch::_mm512_set1_epi16(1),
                )
            }
        }
    }

    impl Sad64Register<U64x8V4> for U8x64V4 {
        #[inline(always)]
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x8V4> {
            unsafe { arch::_mm512_sad_epu8(a, b) }
        }
    }
};

impl_sad_u16!(@swar U16x32V4 => (U32x16V4, U64x8V4));

// The fixed-width `u8x16` slot: `psadbw` + the SSSE3 pair/quad forms, all
// xmm, and the 128/256-bit u16/u32 SWAR rungs the `Simd` grid names.
impl_bit_casts_identity! {
    U8x16V4 as U16x8V4,
    U8x16V4 as U32x4V4,
    U8x16V4 as U64x2V4,
    U16x8V4 as U32x4V4,
    U16x8V4 as U64x2V4,
    U32x4V4 as U64x2V4,
    U16x16V4 as U32x8V4,
    U16x16V4 as U64x4V4,
    U32x8V4 as U64x4V4,
    U32x16V4 as U64x8V4,
}

impl_sad_native_u64!(@ssse3 U8x16V4 => (U16x8V4, U32x4V4, U64x2V4) via _mm_sad_epu8);

impl_sad_u16!(@swar U16x8V4 => (U32x4V4, U64x2V4), U16x16V4 => (U32x8V4, U64x4V4));
impl_sad_u32!(@swar U32x4V4 => U64x2V4, U32x8V4 => U64x4V4, U32x16V4 => U64x8V4);

// Sub-native rungs: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    U8x8V4 => (U16x4V4, U32x2V4, u64),
    U8x4V4 => (crate::register::array::ArrayRegister<u16, 2>, u32, u64),
}
impl_sad_u16!(@scalar U16x4V4 => (U32x2V4, u64));
impl_sad_u32!(@scalar U32x2V4 => u64);

// fp8 pack/unpack (generic branchless defaults) on the u8 ladder -> matching f32 widths.
impl_packed_fp8!(
    crate::register::array::ArrayRegister<u8, 2> => F32x2V4,
    U8x4V4 => F32x4V4,
    U8x8V4 => F32x8V4,
    U8x16V4 => F32x16V4,
);

// The 2-lane array rungs indexed by the 2-lane u32/u64 index types: lane-wise defaults.
const _: () = {
    use crate::register::{IndexableRegister, array::ArrayRegister};

    impl IndexableRegister<U32x2V4> for ArrayRegister<i16, 2> {}
    impl IndexableRegister<U32x2V4> for ArrayRegister<u16, 2> {}
    impl IndexableRegister<U32x2V4> for ArrayRegister<i8, 2> {}
    impl IndexableRegister<U32x2V4> for ArrayRegister<u8, 2> {}
    impl IndexableRegister<U64x2V4> for ArrayRegister<i16, 2> {}
    impl IndexableRegister<U64x2V4> for ArrayRegister<u16, 2> {}
    impl IndexableRegister<U64x2V4> for ArrayRegister<i8, 2> {}
    impl IndexableRegister<U64x2V4> for ArrayRegister<u8, 2> {}
};

// Extend-from-scalar for every native width. The 64-bit x2 files carry
// explicit `ExtendRegister<f64>`-class impls (with the scalar `ConcatRegister`
// beside them). The 32-bit x2 rungs arrive with `half.rs`.
impl_native_extend_from_scalar!(
    F32x4V4 => f32, F32x8V4 => f32, F32x16V4 => f32,
    I32x4V4 => i32, I32x8V4 => i32, I32x16V4 => i32,
    U32x4V4 => u32, U32x8V4 => u32, U32x16V4 => u32,
    F64x4V4 => f64, F64x8V4 => f64,
    I64x4V4 => i64, I64x8V4 => i64,
    U64x4V4 => u64, U64x8V4 => u64,
);

impl_float_to_int_casts! {
    // NOTE: `cvtt` (truncate toward zero) everywhere. `cast` is "like `as`"
    // for in-range inputs. `sat` = the `as`-exact saturating variant (also
    // `cast` under strict_ieee754). The v4 polyfills lean on the hardware's
    // own out-of-range behavior, so they are 2-5 instructions.
    F32x16V4 as I32x16V4 => _mm512_cvttps_epi32 sat _mm512_cvtps_epi32_satx_v4,
    F32x16V4 as U32x16V4 => _mm512_cvttps_epu32 sat _mm512_cvtps_epu32_satx_v4,
    F64x8V4 as I64x8V4 => _mm512_cvttpd_epi64 sat _mm512_cvtpd_epi64_satx_v4,
    F64x8V4 as U64x8V4 => _mm512_cvttpd_epu64 sat _mm512_cvtpd_epu64_satx_v4,
    F32x8V4 as I32x8V4 => _mm256_cvttps_epi32 sat _mm256_cvtps_epi32_satx_v4,
    F32x8V4 as U32x8V4 => _mm256_cvttps_epu32 sat _mm256_cvtps_epu32_satx_v4,
    F64x4V4 as I64x4V4 => _mm256_cvttpd_epi64 sat _mm256_cvtpd_epi64_satx_v4,
    F64x4V4 as U64x4V4 => _mm256_cvttpd_epu64 sat _mm256_cvtpd_epu64_satx_v4,
    F32x4V4 as I32x4V4 => _mm_cvttps_epi32 sat _mm_cvtps_epi32_satx_v4,
    F32x4V4 as U32x4V4 => _mm_cvttps_epu32 sat _mm_cvtps_epu32_satx_v4,
    F64x2V4 as I64x2V4 => _mm_cvttpd_epi64 sat _mm_cvtpd_epi64_satx_v4,
    F64x2V4 as U64x2V4 => _mm_cvttpd_epu64 sat _mm_cvtpd_epu64_satx_v4,

    // 4-lane cross-width float -> int (xmm <-> ymm).
    F32x4V4 as I64x4V4 => _mm256_cvttps_epi64 sat _mm256_cvtps_epi64_satx_v4,
    F32x4V4 as U64x4V4 => _mm256_cvttps_epu64 sat _mm256_cvtps_epu64_satx_v4,
    F64x4V4 as I32x4V4 => _mm256_cvttpd_epi32 sat _mm256_cvtpd_epi32_satx_v4,
    F64x4V4 as U32x4V4 => _mm256_cvttpd_epu32 sat _mm256_cvtpd_epu32_satx_v4,

    // 8-lane cross-width float -> int (ymm <-> zmm).
    F32x8V4 as I64x8V4 => _mm512_cvttps_epi64 sat _mm512_cvtps_epi64_satx_v4,
    F32x8V4 as U64x8V4 => _mm512_cvttps_epu64 sat _mm512_cvtps_epu64_satx_v4,
    F64x8V4 as I32x8V4 => _mm512_cvttpd_epi32 sat _mm512_cvtpd_epi32_satx_v4,
    F64x8V4 as U32x8V4 => _mm512_cvttpd_epu32 sat _mm512_cvtpd_epu32_satx_v4,
}

// ---------------------------------------------------------------------------
// Cross-width cast matrices, one row per `Simd` lane count. v4 differs from v3
// in what is already native: `f32xK -> i64xK` at x4/x8 and `f64xK -> i32xK` at
// x4/x8 are single converts stamped in `impl_float_to_int_casts!` above, so the
// generic `impl_float_cast_matrix!` (which would restate them) is not used.
// Its pieces are stamped by hand below.
// ---------------------------------------------------------------------------

// float -> 8/16-bit narrows: same-width float -> int, then the vpmov narrow.
// Both strengths thread through (`impl_cast_via!`), so the saturating path
// clamps at the destination width.
impl_cast_via! {
    F32x2V4 as ArrayRegister<i16, 2> => via I32x2V4,
    F32x2V4 as ArrayRegister<i8, 2> => via I32x2V4,
    F32x2V4 as ArrayRegister<u16, 2> => via U32x2V4,
    F32x2V4 as ArrayRegister<u8, 2> => via U32x2V4,
    F64x2V4 as ArrayRegister<i16, 2> => via I64x2V4,
    F64x2V4 as ArrayRegister<i8, 2> => via I64x2V4,
    F64x2V4 as ArrayRegister<u16, 2> => via U64x2V4,
    F64x2V4 as ArrayRegister<u8, 2> => via U64x2V4,

    F32x4V4 as I16x4V4 => via I32x4V4,
    F32x4V4 as I8x4V4 => via I32x4V4,
    F32x4V4 as U16x4V4 => via U32x4V4,
    F32x4V4 as U8x4V4 => via U32x4V4,
    F64x4V4 as I16x4V4 => via I64x4V4,
    F64x4V4 as I8x4V4 => via I64x4V4,
    F64x4V4 as U16x4V4 => via U64x4V4,
    F64x4V4 as U8x4V4 => via U64x4V4,

    F32x8V4 as I16x8V4 => via I32x8V4,
    F32x8V4 as I8x8V4 => via I32x8V4,
    F32x8V4 as U16x8V4 => via U32x8V4,
    F32x8V4 as U8x8V4 => via U32x8V4,
    F64x8V4 as I16x8V4 => via I64x8V4,
    F64x8V4 as I8x8V4 => via I64x8V4,
    F64x8V4 as U16x8V4 => via U64x8V4,
    F64x8V4 as U8x8V4 => via U64x8V4,

    F32x16V4 as I16x16V4 => via I32x16V4,
    F32x16V4 as I8x16V4 => via I32x16V4,
    F32x16V4 as U16x16V4 => via U32x16V4,
    F32x16V4 as U8x16V4 => via U32x16V4,
    ArrayRegister<F64x8V4, 2> as I16x16V4 => via ArrayRegister<I64x8V4, 2>,
    ArrayRegister<F64x8V4, 2> as I8x16V4 => via ArrayRegister<I64x8V4, 2>,
    ArrayRegister<F64x8V4, 2> as U16x16V4 => via ArrayRegister<U64x8V4, 2>,
    ArrayRegister<F64x8V4, 2> as U8x16V4 => via ArrayRegister<U64x8V4, 2>,

    // f64 -> 32-bit at the two widths with no single convert (x4/x8 are
    // `vcvttpd2{,u}dq` above).
    F64x2V4 as I32x2V4 => via I64x2V4,
    F64x2V4 as U32x2V4 => via U64x2V4,
    ArrayRegister<F64x8V4, 2> as I32x16V4 => via ArrayRegister<I64x8V4, 2>,
    ArrayRegister<F64x8V4, 2> as U32x16V4 => via ArrayRegister<U64x8V4, 2>,
}

// i64x16 -> f32x16: one `vcvt{,u}qq2ps` per zmm half (each yields a ymm).
const _: () = {
    use crate::register::CastRegister;

    impl CastRegister<ArrayRegister<I64x8V4, 2>> for F32x16V4 {
        #[inline(always)]
        fn cast_from(value: Storage<ArrayRegister<I64x8V4, 2>>) -> Storage<Self> {
            let ArrayRegister([lo, hi]) = value;
            unsafe {
                arch::_mm512_insertf32x8::<1>(
                    arch::_mm512_zextps256_ps512(arch::_mm512_cvtepi64_ps(lo)),
                    arch::_mm512_cvtepi64_ps(hi),
                )
            }
        }
    }

    impl CastRegister<ArrayRegister<U64x8V4, 2>> for F32x16V4 {
        #[inline(always)]
        fn cast_from(value: Storage<ArrayRegister<U64x8V4, 2>>) -> Storage<Self> {
            let ArrayRegister([lo, hi]) = value;
            unsafe {
                arch::_mm512_insertf32x8::<1>(
                    arch::_mm512_zextps256_ps512(arch::_mm512_cvtepu64_ps(lo)),
                    arch::_mm512_cvtepu64_ps(hi),
                )
            }
        }
    }
};

// f32 -> 64-bit int at the two widths with no single convert: widen exactly to
// f64 first so the clamp happens at the destination's range.
impl_cast_via_widen! {
    F32x2V4 as I64x2V4 => via F64x2V4,
    F32x2V4 as U64x2V4 => via F64x2V4,
    F32x16V4 as ArrayRegister<I64x8V4, 2> => via ArrayRegister<F64x8V4, 2>,
    F32x16V4 as ArrayRegister<U64x8V4, 2> => via ArrayRegister<F64x8V4, 2>,
}

// Sign-changing integer casts: same-signedness width change, then the free
// reinterpret. The x16 row's 64-bit slots are arrays but its 32-bit slots are
// native, so the array cast ladder derives none of its crossings, so state all.
impl_sign_cast_matrix! {
    [F32x2V4, F64x2V4, I32x2V4, U32x2V4, I64x2V4, U64x2V4,
        ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>],
    [F32x4V4, F64x4V4, I32x4V4, U32x4V4, I64x4V4, U64x4V4, I16x4V4, U16x4V4, I8x4V4, U8x4V4],
    [F32x8V4, F64x8V4, I32x8V4, U32x8V4, I64x8V4, U64x8V4, I16x8V4, U16x8V4, I8x8V4, U8x8V4],
    [F32x16V4, ArrayRegister<F64x8V4, 2>, I32x16V4, U32x16V4, ArrayRegister<I64x8V4, 2>, ArrayRegister<U64x8V4, 2>,
        I16x16V4, U16x16V4, I8x16V4, U8x16V4],
}

// 8 <-> 16 sign crossings (the x2 row gets these from the scalar array impls).
impl_sign_cast_matrix_8_16! {
    [I16x4V4, U16x4V4, I8x4V4, U8x4V4],
    [I16x8V4, U16x8V4, I8x8V4, U8x8V4],
    [I16x16V4, U16x16V4, I8x16V4, U8x16V4],
}

// ---------------------------------------------------------------------------
// The grid.
// ---------------------------------------------------------------------------

impl NativeSimd for super::X86V4Default {
    type f32xN = F32x16V4;
    type i32xN = I32x16V4;
    type u32xN = U32x16V4;

    type f64xN = F64x8V4;
    type i64xN = I64x8V4;
    type u64xN = U64x8V4;

    type i16xN = I16x32V4;
    type u16xN = U16x32V4;

    type i8xN = I8x64V4;
    type u8xN = U8x64V4;
}

impl Simd for super::X86V4Default {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = F32x2V4;
    type i32x2 = I32x2V4;
    type u32x2 = U32x2V4;

    type f32x4 = F32x4V4;
    type i32x4 = I32x4V4;
    type u32x4 = U32x4V4;

    type f32x8 = F32x8V4;
    type i32x8 = I32x8V4;
    type u32x8 = U32x8V4;

    type f32x16 = F32x16V4;
    type i32x16 = I32x16V4;
    type u32x16 = U32x16V4;

    type f64x2 = F64x2V4;
    type i64x2 = I64x2V4;
    type u64x2 = U64x2V4;

    type f64x4 = F64x4V4;
    type i64x4 = I64x4V4;
    type u64x4 = U64x4V4;

    type f64x8 = F64x8V4;
    type i64x8 = I64x8V4;
    type u64x8 = U64x8V4;

    type f64x16 = ArrayRegister<F64x8V4, 2>;
    type i64x16 = ArrayRegister<I64x8V4, 2>;
    type u64x16 = ArrayRegister<U64x8V4, 2>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = I16x4V4;
    type u16x4 = U16x4V4;

    type i16x8 = I16x8V4;
    type u16x8 = U16x8V4;

    type i16x16 = I16x16V4;
    type u16x16 = U16x16V4;

    type i8x16 = I8x16V4;
    type u8x16 = U8x16V4;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = I8x4V4;
    type u8x4 = U8x4V4;
    type i8x8 = I8x8V4;
    type u8x8 = U8x8V4;
}

impl Simd3 for super::X86V4Default {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

#[thermite_macros::inline_always]
impl<F: Avx512Features> HasIsa for X86V4<F> {
    type Native = Self;

    const ISA: InstructionSet = InstructionSet::X86V4;
}

#[thermite_macros::inline_always]
impl<F: Avx512Features> NativeIsa for X86V4<F> {
    type Registers = generic_array::typenum::U32;

    type Native32Width = generic_array::typenum::U16;
    type Native64Width = generic_array::typenum::U8;
    type Native16Width = generic_array::typenum::U32;
    type Native8Width = generic_array::typenum::U64;

    type NativeAlignment = crate::simd::Align64; // 512-bit vectors = 64 bytes

    const HAS_PREFETCH: bool = arch::HAS_PREFETCH;

    fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
        arch::prefetch::<LOCALITY, WRITE>(ptr);
    }

    unsafe fn disable_denormals() -> Result<bool, crate::simd::UnsupportedError> {
        unsafe { Ok(arch::disable_denormals()) }
    }

    #[allow(clippy::unit_arg)]
    unsafe fn enable_denormals() -> Result<(), crate::simd::UnsupportedError> {
        unsafe { Ok(arch::enable_denormals()) }
    }

    unsafe fn zeroupper() -> bool {
        // `vzeroupper` only touches the upper halves of ymm0-15, while zmm16-31 are
        // unaffected but also carry no SSE-transition penalty, so this is
        // still the right (and complete) transition fence on AVX-512 parts.
        unsafe { arch::_mm256_zeroupper() };

        true
    }
}
