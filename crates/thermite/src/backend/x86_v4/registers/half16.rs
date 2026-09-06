//! Reduced (sub-native-width) 16-bit registers for x86-v4 and the concat/cast glue
//! bridging the scalar/array 16-bit halves, the native 128-bit `I16x8V4`/`U16x8V4`, and
//! the 32/64-bit registers they widen into. Mirrors the v3 `half16.rs`. Every narrow is
//! a single `vpmov{d,q}w` (`vpmovs*`/`vpmovus*` for the saturating forms) and every
//! widen a single `vpmov{s,z}x`, so the pshufb tables and clamp cascades are gone.
//!
//! The reduced register's mask is `ReducedRegister<KMask8, U4>`: a distinct type from
//! the register, so the mask side of the ladder (`bool` pairs at the bottom, `KMask8`
//! at the top) is written out here.

use generic_array::typenum::U4;

use super::arch;
use super::kmask::KMask8;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage, array::ArrayRegister,
    reduced::ReducedRegister,
};

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8V4`.
pub type I16x4V4 = ReducedRegister<super::I16x8V4, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8V4`.
pub type U16x4V4 = ReducedRegister<super::U16x8V4, U4>;
/// The opmask of the 4-lane 16-bit rungs: the low four bits of a `KMask8`.
pub type KMask4Of8 = ReducedRegister<KMask8, U4>;

// --- x4 <- x2 (two scalar-array halves into a 4-lane reduced register) ---

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4V4 {
    fn concat(lo: Storage<ArrayRegister<i16, 2>>, hi: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<i16, 2>>, Storage<ArrayRegister<i16, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4V4 {
    fn extend(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0) })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<i16, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4V4 {
    fn concat(lo: Storage<ArrayRegister<u16, 2>>, hi: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe {
            arch::_mm_setr_epi16(
                lo.0[0] as i16,
                lo.0[1] as i16,
                hi.0[0] as i16,
                hi.0[1] as i16,
                0,
                0,
                0,
                0,
            )
        })
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<u16, 2>>, Storage<ArrayRegister<u16, 2>>) {
        let mut arr = [0u16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4V4 {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi16(value.0[0] as i16, value.0[1] as i16, 0, 0, 0, 0, 0, 0) })
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        let mut arr = [0u16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

// --- x8 <- x4 (two 4-lane reduced halves into the native 8-lane register) ---
// `ExtendRegister<I16x4V4> for I16x8V4` is the `register/reduced.rs` blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4V4> for super::I16x8V4 {
    fn concat(lo: Storage<I16x4V4>, hi: Storage<I16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4V4>, Storage<I16x4V4>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4V4> for super::U16x8V4 {
    fn concat(lo: Storage<U16x4V4>, hi: Storage<U16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4V4>, Storage<U16x4V4>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

// --- mask side of the ladder ---
// The reduced mask's upper bits are don't-cares (`not` sets them), so every read scrubs.

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for KMask4Of8 {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new((lo.0[0] as u8) | ((lo.0[1] as u8) << 1) | ((hi.0[0] as u8) << 2) | ((hi.0[1] as u8) << 3))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let m = value.0;
        (
            ArrayRegister([m & 0b0001 != 0, m & 0b0010 != 0]),
            ArrayRegister([m & 0b0100 != 0, m & 0b1000 != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for KMask4Of8 {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new((value.0[0] as u8) | ((value.0[1] as u8) << 1))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        ArrayRegister([value.0 & 0b01 != 0, value.0 & 0b10 != 0])
    }
}

// `ExtendRegister<KMask4Of8> for KMask8` is the reduced.rs blanket.
#[thermite_macros::inline_always]
impl ConcatRegister<KMask4Of8> for KMask8 {
    fn concat(lo: Storage<KMask4Of8>, hi: Storage<KMask4Of8>) -> Storage<Self> {
        (lo.0 & 0x0F) | ((hi.0 & 0x0F) << 4)
    }

    fn split(value: Storage<Self>) -> (Storage<KMask4Of8>, Storage<KMask4Of8>) {
        (ReducedRegister::new(value & 0x0F), ReducedRegister::new(value >> 4))
    }
}

// --- 16 <-> 32 ---
// x4: I16x4V4 <-> I32x4V4 (native). x2: ArrayRegister<i16, 2> <-> I32x2V4 (reduced).

#[thermite_macros::inline_always]
impl CastRegister<I16x4V4> for super::I32x4V4 {
    fn cast_from(value: Storage<I16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi16_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4V4> for super::U32x4V4 {
    fn cast_from(value: Storage<U16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu16_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V4> for I16x4V4 {
    fn cast_from(value: Storage<super::I32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_epi16(value) })
    }

    fn saturating_cast_from(value: Storage<super::I32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtsepi32_epi16(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V4> for U16x4V4 {
    fn cast_from(value: Storage<super::U32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_epi16(value) })
    }

    fn saturating_cast_from(value: Storage<super::U32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtusepi32_epi16(value) })
    }
}

/// Low two words of an xmm as a scalar pair.
#[inline(always)]
fn low_words(v: arch::__m128i) -> [i16; 2] {
    let mut arr = [0i16; 8];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    [arr[0], arr[1]]
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2V4 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2V4 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V4> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2V4>) -> Storage<Self> {
        ArrayRegister(low_words(unsafe { arch::_mm_cvtepi32_epi16(value.0) }))
    }

    fn saturating_cast_from(value: Storage<super::half::I32x2V4>) -> Storage<Self> {
        ArrayRegister(low_words(unsafe { arch::_mm_cvtsepi32_epi16(value.0) }))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V4> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2V4>) -> Storage<Self> {
        let [a, b] = low_words(unsafe { arch::_mm_cvtepi32_epi16(value.0) });
        ArrayRegister([a as u16, b as u16])
    }

    fn saturating_cast_from(value: Storage<super::half::U32x2V4>) -> Storage<Self> {
        let [a, b] = low_words(unsafe { arch::_mm_cvtusepi32_epi16(value.0) });
        ArrayRegister([a as u16, b as u16])
    }
}

// --- 16 <-> 64 ---
// x2: ArrayRegister<i16, 2> <-> I64x2V4. x4: I16x4V4 <-> I64x4V4. (x8/x16 live in
// `i16x8.rs` / `i16x16.rs`.)

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2V4 {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2V4 {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V4> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ArrayRegister(low_words(unsafe { arch::_mm_cvtepi64_epi16(value) }))
    }

    fn saturating_cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ArrayRegister(low_words(unsafe { arch::_mm_cvtsepi64_epi16(value) }))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V4> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        let [a, b] = low_words(unsafe { arch::_mm_cvtepi64_epi16(value) });
        ArrayRegister([a as u16, b as u16])
    }

    fn saturating_cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        let [a, b] = low_words(unsafe { arch::_mm_cvtusepi64_epi16(value) });
        ArrayRegister([a as u16, b as u16])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I16x4V4> for super::I64x4V4 {
    fn cast_from(value: Storage<I16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4V4> for super::U64x4V4 {
    fn cast_from(value: Storage<U16x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu16_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x4V4> for I16x4V4 {
    fn cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi16(value) })
    }

    fn saturating_cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtsepi64_epi16(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x4V4> for U16x4V4 {
    fn cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi16(value) })
    }

    fn saturating_cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtusepi64_epi16(value) })
    }
}

// --- 16 -> f32 / f64 widens: the 16 -> 32 widen above, then the native int -> float
// convert (exact: every i16/u16 fits an i32/u32). The float -> 16 narrows come from
// `impl_float_cast_matrix!` in `registers/mod.rs` (float -> i32/i64, then the vpmov narrow).
impl_cast_from_via! {
    ArrayRegister<i16, 2> as super::half::F32x2V4 => via super::half::I32x2V4,
    ArrayRegister<u16, 2> as super::half::F32x2V4 => via super::half::U32x2V4,
    ArrayRegister<i16, 2> as super::F64x2V4 => via super::half::I32x2V4,
    ArrayRegister<u16, 2> as super::F64x2V4 => via super::half::U32x2V4,
    I16x4V4 as super::F32x4V4 => via super::I32x4V4,
    U16x4V4 as super::F32x4V4 => via super::U32x4V4,
    I16x4V4 as super::F64x4V4 => via super::I32x4V4,
    U16x4V4 as super::F64x4V4 => via super::U32x4V4,
    super::I16x8V4 as super::F32x8V4 => via super::I32x8V4,
    super::U16x8V4 as super::F32x8V4 => via super::U32x8V4,
    super::I16x8V4 as super::F64x8V4 => via super::I32x8V4,
    super::U16x8V4 as super::F64x8V4 => via super::U32x8V4,
    super::I16x16V4 as super::F32x16V4 => via super::I32x16V4,
    super::U16x16V4 as super::F32x16V4 => via super::U32x16V4,
    super::I16x16V4 as ArrayRegister<super::F64x8V4, 2> => via super::I32x16V4,
    super::U16x16V4 as ArrayRegister<super::F64x8V4, 2> => via super::U32x16V4,
}

// --- gather/scatter index markers (no 16-bit gather, lane-wise defaults) ---
impl IndexableRegister<super::U32x4V4> for I16x4V4 {}
impl IndexableRegister<super::U32x4V4> for U16x4V4 {}
impl IndexableRegister<super::U64x4V4> for I16x4V4 {}
impl IndexableRegister<super::U64x4V4> for U16x4V4 {}
