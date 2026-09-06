//! Reduced (sub-native-width) 8-bit registers for x86-v4. Mirrors [`half16`](super::half16)
//! one element size down: `vpmov{s,z}x` widens and `vpmov*` narrows everywhere, and the
//! reduced masks (`ReducedRegister<KMask16, U12 | U8>`) get their own concat ladder.

use generic_array::typenum::{U8, U12};

use super::arch;
use super::kmask::KMask16;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage, array::ArrayRegister,
    reduced::ReducedRegister,
};

/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16V4`.
pub type I8x4V4 = ReducedRegister<super::I8x16V4, U12>;
/// 4-lane unsigned 8-bit register.
pub type U8x4V4 = ReducedRegister<super::U8x16V4, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16V4`.
pub type I8x8V4 = ReducedRegister<super::I8x16V4, U8>;
/// 8-lane unsigned 8-bit register.
pub type U8x8V4 = ReducedRegister<super::U8x16V4, U8>;
/// The opmask of the 4-lane byte rungs: the low four bits of a `KMask16`.
pub type KMask4Of16 = ReducedRegister<KMask16, U12>;
/// The opmask of the 8-lane byte rungs: the low eight bits of a `KMask16`.
pub type KMask8Of16 = ReducedRegister<KMask16, U8>;

#[inline(always)]
fn store_bytes(v: arch::__m128i) -> [i8; 16] {
    let mut arr = [0i8; 16];
    unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// ===========================================================================================
// x4 <- x2 (scalar-array) concat / extend.
// ===========================================================================================

macro_rules! impl_concat_x4_from_x2 {
    ($red:ty, $elem:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<$elem, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<$elem, 2>>, hi: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_cvtsi32_si128(
                        (lo.0[0] as u8 as i32)
                            | ((lo.0[1] as u8 as i32) << 8)
                            | ((hi.0[0] as u8 as i32) << 16)
                            | ((hi.0[1] as u8 as i32) << 24),
                    )
                })
            }

            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<$elem, 2>>, Storage<ArrayRegister<$elem, 2>>) {
                let a = store_bytes(value.0);
                (
                    ArrayRegister([a[0] as $elem, a[1] as $elem]),
                    ArrayRegister([a[2] as $elem, a[3] as $elem]),
                )
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<$elem, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new(unsafe {
                    arch::_mm_cvtsi32_si128((value.0[0] as u8 as i32) | ((value.0[1] as u8 as i32) << 8))
                })
            }

            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<$elem, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] as $elem, a[1] as $elem])
            }
        }
    };
}

impl_concat_x4_from_x2!(I8x4V4, i8);
impl_concat_x4_from_x2!(U8x4V4, u8);

// ===========================================================================================
// x8 <- x4 (both reduced over the same native register).
// ===========================================================================================

macro_rules! impl_concat_x8_from_x4 {
    ($x8:ty, $x4:ty) => {
        #[thermite_macros::inline_always]
        impl ExtendRegister<$x4> for $x8 {
            fn extend(value: Storage<$x4>) -> Storage<Self> {
                // The dead lanes 4..16 must read as zero at x8's lanes 4..8.
                ReducedRegister::new(unsafe { arch::_mm_maskz_mov_epi8(0x000F, value.0) })
            }

            fn narrow(value: Storage<Self>) -> Storage<$x4> {
                ReducedRegister::new(value.0)
            }
        }

        #[thermite_macros::inline_always]
        impl ConcatRegister<$x4> for $x8 {
            fn concat(lo: Storage<$x4>, hi: Storage<$x4>) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::_mm_unpacklo_epi32(lo.0, hi.0) })
            }

            fn split(value: Storage<Self>) -> (Storage<$x4>, Storage<$x4>) {
                (
                    ReducedRegister::new(value.0),
                    ReducedRegister::new(unsafe { arch::_mm_bsrli_si128::<4>(value.0) }),
                )
            }
        }
    };
}

impl_concat_x8_from_x4!(I8x8V4, I8x4V4);
impl_concat_x8_from_x4!(U8x8V4, U8x4V4);

// ===========================================================================================
// x16 (native) <- x8 (reduced). Extend is the reduced.rs blanket.
// ===========================================================================================

macro_rules! impl_concat_x16_from_x8 {
    ($native:ty, $x8:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<$x8> for $native {
            fn concat(lo: Storage<$x8>, hi: Storage<$x8>) -> Storage<Self> {
                unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
            }

            fn split(value: Storage<Self>) -> (Storage<$x8>, Storage<$x8>) {
                (
                    ReducedRegister::new(value),
                    ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
                )
            }
        }
    };
}

impl_concat_x16_from_x8!(super::I8x16V4, I8x8V4);
impl_concat_x16_from_x8!(super::U8x16V4, U8x8V4);

// ===========================================================================================
// Mask side of the ladder: bool pairs -> KMask4Of16 -> KMask8Of16 -> KMask16. Upper bits of
// a reduced mask are don't-cares, so every read scrubs to its lane count.
// ===========================================================================================

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for KMask4Of16 {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(
            (lo.0[0] as u16) | ((lo.0[1] as u16) << 1) | ((hi.0[0] as u16) << 2) | ((hi.0[1] as u16) << 3),
        )
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
impl ExtendRegister<ArrayRegister<bool, 2>> for KMask4Of16 {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new((value.0[0] as u16) | ((value.0[1] as u16) << 1))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        ArrayRegister([value.0 & 0b01 != 0, value.0 & 0b10 != 0])
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<KMask4Of16> for KMask8Of16 {
    fn extend(value: Storage<KMask4Of16>) -> Storage<Self> {
        ReducedRegister::new(value.0 & 0x0F)
    }

    fn narrow(value: Storage<Self>) -> Storage<KMask4Of16> {
        ReducedRegister::new(value.0 & 0x0F)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<KMask4Of16> for KMask8Of16 {
    fn concat(lo: Storage<KMask4Of16>, hi: Storage<KMask4Of16>) -> Storage<Self> {
        ReducedRegister::new((lo.0 & 0x0F) | ((hi.0 & 0x0F) << 4))
    }

    fn split(value: Storage<Self>) -> (Storage<KMask4Of16>, Storage<KMask4Of16>) {
        (
            ReducedRegister::new(value.0 & 0x0F),
            ReducedRegister::new((value.0 >> 4) & 0x0F),
        )
    }
}

// `ExtendRegister<KMask8Of16> for KMask16` is the reduced.rs blanket.
#[thermite_macros::inline_always]
impl ConcatRegister<KMask8Of16> for KMask16 {
    fn concat(lo: Storage<KMask8Of16>, hi: Storage<KMask8Of16>) -> Storage<Self> {
        (lo.0 & 0xFF) | ((hi.0 & 0xFF) << 8)
    }

    fn split(value: Storage<Self>) -> (Storage<KMask8Of16>, Storage<KMask8Of16>) {
        (ReducedRegister::new(value & 0xFF), ReducedRegister::new(value >> 8))
    }
}

// ===========================================================================================
// 8 <-> 16 (x4 both reduced, x8 native word register). x16 lives in i8x16.rs.
// ===========================================================================================

macro_rules! impl_cast_8_16 {
    ($($i8:ty, $i16:ty => $widen:ident, $narrow:ident, $sat:ident, $wrap16:expr;)*) => {$(
        #[thermite_macros::inline_always]
        impl CastRegister<$i8> for $i16 {
            fn cast_from(value: Storage<$i8>) -> Storage<Self> {
                let w = unsafe { arch::$widen(value.0) };
                $wrap16(w)
            }
        }

        #[thermite_macros::inline_always]
        impl CastRegister<$i16> for $i8 {
            fn cast_from(value: Storage<$i16>) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::$narrow(reduced_or_native(value)) })
            }

            fn saturating_cast_from(value: Storage<$i16>) -> Storage<Self> {
                ReducedRegister::new(unsafe { arch::$sat(reduced_or_native(value)) })
            }
        }
    )*};
}

/// Both the reduced and the native 128-bit word registers hand an `__m128i` to the narrow.
trait Xmm {
    fn xmm(self) -> arch::__m128i;
}

impl Xmm for arch::__m128i {
    #[inline(always)]
    fn xmm(self) -> arch::__m128i {
        self
    }
}

impl<R: crate::register::CoreRegister<Storage = arch::__m128i>, N: generic_array::typenum::Unsigned> Xmm
    for ReducedRegister<R, N>
where
    R: crate::register::reduced::CoreReducible<N>,
{
    #[inline(always)]
    fn xmm(self) -> arch::__m128i {
        self.0
    }
}

#[inline(always)]
fn reduced_or_native<T: Xmm>(value: T) -> arch::__m128i {
    value.xmm()
}

impl_cast_8_16! {
    I8x4V4, super::half16::I16x4V4 => _mm_cvtepi8_epi16, _mm_cvtepi16_epi8, _mm_cvtsepi16_epi8, ReducedRegister::new;
    U8x4V4, super::half16::U16x4V4 => _mm_cvtepu8_epi16, _mm_cvtepi16_epi8, _mm_cvtusepi16_epi8, ReducedRegister::new;
    I8x8V4, super::I16x8V4 => _mm_cvtepi8_epi16, _mm_cvtepi16_epi8, _mm_cvtsepi16_epi8, core::convert::identity;
    U8x8V4, super::U16x8V4 => _mm_cvtepu8_epi16, _mm_cvtepi16_epi8, _mm_cvtusepi16_epi8, core::convert::identity;
}

// ===========================================================================================
// 8 <-> 32 (x4: xmm dwords, x8: ymm dwords, x2: scalar). x16 lives in i8x16.rs.
// ===========================================================================================

#[thermite_macros::inline_always]
impl CastRegister<I8x4V4> for super::I32x4V4 {
    fn cast_from(value: Storage<I8x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi8_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x4V4> for super::U32x4V4 {
    fn cast_from(value: Storage<U8x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu8_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x4V4> for I8x4V4 {
    fn cast_from(value: Storage<super::I32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::I32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtsepi32_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4V4> for U8x4V4 {
    fn cast_from(value: Storage<super::U32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtepi32_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::U32x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtusepi32_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I8x8V4> for super::I32x8V4 {
    fn cast_from(value: Storage<I8x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x8V4> for super::U32x8V4 {
    fn cast_from(value: Storage<U8x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi32(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x8V4> for I8x8V4 {
    fn cast_from(value: Storage<super::I32x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::I32x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtsepi32_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x8V4> for U8x8V4 {
    fn cast_from(value: Storage<super::U32x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi32_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::U32x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtusepi32_epi8(value) })
    }
}

/// Low two bytes of an xmm as a scalar pair.
#[inline(always)]
fn low_bytes(v: arch::__m128i) -> [i8; 2] {
    let a = store_bytes(v);
    [a[0], a[1]]
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2V4 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2V4 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value.0[0] as i32, value.0[1] as i32, 0, 0) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2V4> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2V4>) -> Storage<Self> {
        ArrayRegister(low_bytes(unsafe { arch::_mm_cvtepi32_epi8(value.0) }))
    }

    fn saturating_cast_from(value: Storage<super::half::I32x2V4>) -> Storage<Self> {
        ArrayRegister(low_bytes(unsafe { arch::_mm_cvtsepi32_epi8(value.0) }))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2V4> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2V4>) -> Storage<Self> {
        let [a, b] = low_bytes(unsafe { arch::_mm_cvtepi32_epi8(value.0) });
        ArrayRegister([a as u8, b as u8])
    }

    fn saturating_cast_from(value: Storage<super::half::U32x2V4>) -> Storage<Self> {
        let [a, b] = low_bytes(unsafe { arch::_mm_cvtusepi32_epi8(value.0) });
        ArrayRegister([a as u8, b as u8])
    }
}

// ===========================================================================================
// 8 <-> 64 (x2: scalar, x4: ymm qwords, x8: zmm qwords). x16 lives in i8x16.rs.
// ===========================================================================================

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2V4 {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2V4 {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(value.0[1] as i64, value.0[0] as i64) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V4> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ArrayRegister(low_bytes(unsafe { arch::_mm_cvtepi64_epi8(value) }))
    }

    fn saturating_cast_from(value: Storage<super::I64x2V4>) -> Storage<Self> {
        ArrayRegister(low_bytes(unsafe { arch::_mm_cvtsepi64_epi8(value) }))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V4> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        let [a, b] = low_bytes(unsafe { arch::_mm_cvtepi64_epi8(value) });
        ArrayRegister([a as u8, b as u8])
    }

    fn saturating_cast_from(value: Storage<super::U64x2V4>) -> Storage<Self> {
        let [a, b] = low_bytes(unsafe { arch::_mm_cvtusepi64_epi8(value) });
        ArrayRegister([a as u8, b as u8])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I8x4V4> for super::I64x4V4 {
    fn cast_from(value: Storage<I8x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x4V4> for super::U64x4V4 {
    fn cast_from(value: Storage<U8x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x4V4> for I8x4V4 {
    fn cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtsepi64_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x4V4> for U8x4V4 {
    fn cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtepi64_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm256_cvtusepi64_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I8x8V4> for super::I64x8V4 {
    fn cast_from(value: Storage<I8x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi8_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x8V4> for super::U64x8V4 {
    fn cast_from(value: Storage<U8x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepu8_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x8V4> for I8x8V4 {
    fn cast_from(value: Storage<super::I64x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm512_cvtepi64_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::I64x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm512_cvtsepi64_epi8(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x8V4> for U8x8V4 {
    fn cast_from(value: Storage<super::U64x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm512_cvtepi64_epi8(value) })
    }

    fn saturating_cast_from(value: Storage<super::U64x8V4>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm512_cvtusepi64_epi8(value) })
    }
}

// --- 8 -> f32 / f64 widens: the 8 -> 32 widen, then the native int -> float convert.
// Float -> 8 narrows come from `impl_float_cast_matrix!` in `registers/mod.rs`.
impl_cast_from_via! {
    ArrayRegister<i8, 2> as super::half::F32x2V4 => via super::half::I32x2V4,
    ArrayRegister<u8, 2> as super::half::F32x2V4 => via super::half::U32x2V4,
    ArrayRegister<i8, 2> as super::F64x2V4 => via super::half::I32x2V4,
    ArrayRegister<u8, 2> as super::F64x2V4 => via super::half::U32x2V4,
    I8x4V4 as super::F32x4V4 => via super::I32x4V4,
    U8x4V4 as super::F32x4V4 => via super::U32x4V4,
    I8x4V4 as super::F64x4V4 => via super::I32x4V4,
    U8x4V4 as super::F64x4V4 => via super::U32x4V4,
    I8x8V4 as super::F32x8V4 => via super::I32x8V4,
    U8x8V4 as super::F32x8V4 => via super::U32x8V4,
    I8x8V4 as super::F64x8V4 => via super::I32x8V4,
    U8x8V4 as super::F64x8V4 => via super::U32x8V4,
    super::I8x16V4 as super::F32x16V4 => via super::I32x16V4,
    super::U8x16V4 as super::F32x16V4 => via super::U32x16V4,
    super::I8x16V4 as ArrayRegister<super::F64x8V4, 2> => via super::I32x16V4,
    super::U8x16V4 as ArrayRegister<super::F64x8V4, 2> => via super::U32x16V4,
}

// --- gather/scatter index markers (no byte gather, lane-wise defaults) ---
macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(super::U32x4V4 => I8x4V4, U8x4V4);
impl_indexable8!(super::U64x4V4 => I8x4V4, U8x4V4);
impl_indexable8!(super::U32x8V4 => I8x8V4, U8x8V4);
impl_indexable8!(super::U64x8V4 => I8x8V4, U8x8V4);
