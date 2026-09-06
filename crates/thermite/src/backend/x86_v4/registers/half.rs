//! 2-lane 32-bit rungs: `f32x2`/`i32x2`/`u32x2` as the low half of the xmm
//! registers (`HalfRegister2`), plus everything the `Simd` grid demands of
//! them that the `ReducedRegister` blankets cannot derive:
//!
//! - scalar and half concat (`ConcatRegister<f32>` / `<F32x2V4> for F32x4V4`),
//!   and the SAME on the mask side: the reduced mask is `ReducedRegister<KMask4,
//!   U2>`, a distinct type from the register (unlike v3, where a register is its
//!   own mask), so `FullConcatRegister` needs explicit opmask-half impls here.
//! - the 2-lane 32 <-> 64 casts (all single EVEX converts at v4: `vcvtudq2pd`,
//!   `vpmovqd`/`vpmovsqd`/`vpmovusqd`).
//! - 64-bit-index gathers/scatters (two qword indices in one xmm).
//!
//! Layout mirrors the v3 `half.rs`. The extend-from-scalar impls come from the
//! `impl_reduced_extend_from_scalar!` blanket in `register/reduced.rs`.

use generic_array::typenum::U2;

use super::arch;
use super::kmask::KMask4;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Register, Storage,
    reduced::{HalfRegister2, ReducedRegister},
};

pub type F32x2V4 = HalfRegister2<super::F32x4V4>;
pub type I32x2V4 = HalfRegister2<super::I32x4V4>;
pub type U32x2V4 = HalfRegister2<super::U32x4V4>;

/// The opmask of every 2-lane 32-bit rung: the low two bits of a `KMask4`.
pub type KMask2Half = ReducedRegister<KMask4, U2>;

// ===========================================================================================
// Mask side of the ladder. The reduced mask's dead upper bits are don't-cares
// (a `not` on it sets them), so every read scrubs to the low two bits.
// ===========================================================================================

#[thermite_macros::inline_always]
impl ConcatRegister<bool> for KMask2Half {
    fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
        ReducedRegister::new((lo as u8) | ((hi as u8) << 1))
    }

    fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
        (value.0 & 0b01 != 0, value.0 & 0b10 != 0)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<bool> for KMask2Half {
    fn extend(value: Storage<bool>) -> Storage<Self> {
        ReducedRegister::new(value as u8)
    }

    fn narrow(value: Storage<Self>) -> Storage<bool> {
        value.0 & 0b01 != 0
    }
}

// `ExtendRegister<KMask2Half> for KMask4` is the `ExtendRegister<ReducedRegister<R, N>>
// for R` blanket in `register/reduced.rs` (`zeroupper` on extend, wrap on narrow).
#[thermite_macros::inline_always]
impl ConcatRegister<KMask2Half> for KMask4 {
    fn concat(lo: Storage<KMask2Half>, hi: Storage<KMask2Half>) -> Storage<Self> {
        (lo.0 & 0b11) | ((hi.0 & 0b11) << 2)
    }

    fn split(value: Storage<Self>) -> (Storage<KMask2Half>, Storage<KMask2Half>) {
        (ReducedRegister::new(value & 0b11), ReducedRegister::new(value >> 2))
    }
}

// ===========================================================================================
// f32x2
// ===========================================================================================

#[thermite_macros::inline_always]
impl ConcatRegister<f32> for F32x2V4 {
    fn concat(lo: Storage<f32>, hi: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_ps(lo, hi, 0.0, 0.0) })
    }

    fn split(value: Storage<Self>) -> (Storage<f32>, Storage<f32>) {
        let mut arr = [0.0f32; 4];
        unsafe { arch::_mm_storeu_ps(arr.as_mut_ptr(), value.0) };
        (arr[0], arr[1])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<F32x2V4> for super::F32x4V4 {
    fn concat(lo: Storage<F32x2V4>, hi: Storage<F32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_movelh_ps(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<F32x2V4>, Storage<F32x2V4>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_movehl_ps(value, value) }),
        )
    }
}

// ===========================================================================================
// i32x2
// ===========================================================================================

#[thermite_macros::inline_always]
impl ConcatRegister<i32> for I32x2V4 {
    fn concat(lo: Storage<i32>, hi: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo, hi, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<i32>, Storage<i32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0], arr[1])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<I32x2V4> for super::I32x4V4 {
    fn concat(lo: Storage<I32x2V4>, hi: Storage<I32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I32x2V4>, Storage<I32x2V4>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// ===========================================================================================
// u32x2
// ===========================================================================================

#[thermite_macros::inline_always]
impl ConcatRegister<u32> for U32x2V4 {
    fn concat(lo: Storage<u32>, hi: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo as i32, hi as i32, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<u32>, Storage<u32>) {
        let mut arr = [0u32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0], arr[1])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<U32x2V4> for super::U32x4V4 {
    fn concat(lo: Storage<U32x2V4>, hi: Storage<U32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U32x2V4>, Storage<U32x2V4>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// ===========================================================================================
// 2-lane 32 <-> 64 casts. Every one is a single EVEX convert. The widening
// converts read only the low two lanes, the narrowing ones write only the low
// two (upper lanes are zero, which is fine for a reduced register).
// ===========================================================================================

#[thermite_macros::inline_always]
impl CastRegister<F32x2V4> for super::F64x2V4 {
    fn cast_from(value: Storage<F32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V4> for F32x2V4 {
    fn cast_from(value: Storage<super::F64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_ps(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V4> for super::F64x2V4 {
    fn cast_from(value: Storage<I32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2V4> for super::F64x2V4 {
    fn cast_from(value: Storage<U32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu32_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2V4> for super::U64x2V4 {
    fn cast_from(value: Storage<U32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V4> for U32x2V4 {
    fn cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi64_epi32(value) })
    }

    // vpmovusqd saturates in hardware.
    fn saturating_cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtusepi64_epi32(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V4> for super::I64x2V4 {
    fn cast_from(value: Storage<I32x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V4> for I32x2V4 {
    fn cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi64_epi32(value) })
    }

    // vpmovsqd saturates in hardware.
    fn saturating_cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtsepi64_epi32(value) })
    }
}

// The two 32 -> 64 casts the x2 rungs need in the other direction: single
// `vcvt{,u}qq2ps` into the low two lanes.
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V4> for F32x2V4 {
    fn cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi64_ps(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V4> for F32x2V4 {
    fn cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepu64_ps(value) })
    }
}

// ===========================================================================================
// 32-bit-index gathers/scatters on the 64-bit x2 registers: two dword indices
// in the low half of the reduced xmm. AVX2 gather form is `(ptr, idx)`.
// ===========================================================================================

macro_rules! x2_gather_i32 {
    ($($ty:ty => $gather:ident, $mgather:ident, $scatter:ident, $mscatter:ident;)*) => {$(
        #[thermite_macros::inline_always]
        impl IndexableRegister<U32x2V4> for $ty {
            unsafe fn gather(ptr: *const <$ty as Register>::Element, indices: Storage<U32x2V4>) -> Storage<$ty> {
                unsafe { arch::$gather::<8>(ptr as *const _, indices.0) }
            }

            unsafe fn gather_m(
                src: Storage<$ty>,
                mask: Storage<<$ty as crate::register::CoreRegister>::Mask>,
                ptr: *const <$ty as Register>::Element,
                indices: Storage<U32x2V4>,
            ) -> Storage<$ty> {
                unsafe { arch::$mgather::<8>(src, mask, indices.0, ptr as *const _) }
            }

            unsafe fn scatter(value: Storage<$ty>, ptr: *mut <$ty as Register>::Element, indices: Storage<U32x2V4>) {
                unsafe { arch::$scatter::<8>(ptr as *mut _, indices.0, value) }
            }

            unsafe fn scatter_m(
                value: Storage<$ty>,
                mask: Storage<<$ty as crate::register::CoreRegister>::Mask>,
                ptr: *mut <$ty as Register>::Element,
                indices: Storage<U32x2V4>,
            ) {
                unsafe { arch::$mscatter::<8>(ptr as *mut _, mask, indices.0, value) }
            }
        }
    )*};
}

x2_gather_i32! {
    super::F64x2V4 => _mm_i32gather_pd, _mm_mmask_i32gather_pd, _mm_i32scatter_pd, _mm_mask_i32scatter_pd;
    super::I64x2V4 => _mm_i32gather_epi64, _mm_mmask_i32gather_epi64, _mm_i32scatter_epi64, _mm_mask_i32scatter_epi64;
    super::U64x2V4 => _mm_i32gather_epi64, _mm_mmask_i32gather_epi64, _mm_i32scatter_epi64, _mm_mask_i32scatter_epi64;
}

// ===========================================================================================
// 64-bit-index gathers/scatters: two qword indices in one xmm. The AVX2 gather
// form is `(ptr, idx)`. The EVEX masked forms are `(src, k, idx, ptr)` and
// `(ptr, k, idx, value)`. Only the low two `k` bits are read, so the reduced
// mask's dirty upper bits are harmless here.
// ===========================================================================================

macro_rules! half_gather_i64 {
    ($($ty:ty => $gather:ident, $mgather:ident, $mscatter:ident;)*) => {$(
        #[thermite_macros::inline_always]
        impl IndexableRegister<super::U64x2V4> for $ty {
            unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V4>) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::$gather::<4>(ptr as *const _, indices) })
            }

            unsafe fn gather_m(
                src: Storage<Self>,
                mask: Storage<Self::Mask>,
                ptr: *const Self::Element,
                indices: Storage<super::U64x2V4>,
            ) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::$mgather::<4>(src.0, mask.0, indices, ptr as *const _) })
            }

            unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x2V4>) {
                unsafe { arch::$mscatter::<4>(ptr as *mut _, 0b11, indices, value.0) }
            }

            unsafe fn scatter_m(
                value: Storage<Self>,
                mask: Storage<Self::Mask>,
                ptr: *mut Self::Element,
                indices: Storage<super::U64x2V4>,
            ) {
                unsafe { arch::$mscatter::<4>(ptr as *mut _, mask.0 & 0b11, indices, value.0) }
            }
        }
    )*};
}

// `scatter` (unmasked) goes through the masked form with `k = 0b11`: only two
// indices exist, and the plain `_mm_i64scatter_ps` form would still only write
// two lanes, but stating the mask keeps the intent explicit.
half_gather_i64! {
    F32x2V4 => _mm_i64gather_ps, _mm_mmask_i64gather_ps, _mm_mask_i64scatter_ps;
    I32x2V4 => _mm_i64gather_epi32, _mm_mmask_i64gather_epi32, _mm_mask_i64scatter_epi32;
    U32x2V4 => _mm_i64gather_epi32, _mm_mmask_i64gather_epi32, _mm_mask_i64scatter_epi32;
}
