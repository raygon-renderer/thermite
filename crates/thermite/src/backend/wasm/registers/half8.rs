//! Reduced (sub-native-width) 8-bit registers for the WASM backend. Mirrors `half16.rs` one
//! element size down; widen casts go byte -> word -> dword via the wasm `*_extend_low_*` ops, and
//! narrows use `i8x16_shuffle` to gather the low byte of each lane.

use generic_array::typenum::{U8, U12};

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, Storage,
    array::ArrayRegister, reduced::ReducedRegister,
};

// --- saturating narrows into 8-bit ---

#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Wasm> for I8x8Wasm {
    // Signed `i16 -> i8` via `i8x16.narrow_i16x8_s`.
    fn saturating_cast_from(value: Storage<super::I16x8Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_narrow_i16x8(value, value))
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::I16x8Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 2, 4, 6, 8, 10, 12, 14, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::I16x4Wasm> for I8x4Wasm {
    fn saturating_cast_from(value: Storage<super::half16::I16x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_narrow_i16x8(value.0, value.0))
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::half16::I16x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 2, 4, 6, 8, 10, 12, 14, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value.0, value.0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Wasm> for U8x8Wasm {
    // Unsigned `u16 -> u8`: clamp the high end (`u8x16.narrow_i16x8_u` reads a signed source).
    fn saturating_cast_from(value: Storage<super::U16x8Wasm>) -> Storage<Self> {
        let c = arch::u16x8_min(value, arch::u16x8_splat(0xFF));
        ReducedRegister::new(arch::u8x16_narrow_i16x8(c, c))
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::U16x8Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 2, 4, 6, 8, 10, 12, 14, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::U16x4Wasm> for U8x4Wasm {
    fn saturating_cast_from(value: Storage<super::half16::U16x4Wasm>) -> Storage<Self> {
        let c = arch::u16x8_min(value.0, arch::u16x8_splat(0xFF));
        ReducedRegister::new(arch::u8x16_narrow_i16x8(c, c))
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::half16::U16x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 2, 4, 6, 8, 10, 12, 14, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value.0, value.0))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x4Wasm> for I8x4Wasm {
    // Skip-level 32 -> 8: compose the 32 -> 16 and 16 -> 8 saturating narrows (idempotent).
    fn saturating_cast_from(value: Storage<super::I32x4Wasm>) -> Storage<Self> {
        let w = <super::half16::I16x4Wasm as CastRegister<super::I32x4Wasm>>::saturating_cast_from(value);
        <Self as CastRegister<super::half16::I16x4Wasm>>::saturating_cast_from(w)
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::I32x4Wasm>) -> Storage<Self> {
        // low byte of each of the 4 i32 lanes -> low 4 bytes
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 4, 8, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        >(value, value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x4Wasm> for U8x4Wasm {
    fn saturating_cast_from(value: Storage<super::U32x4Wasm>) -> Storage<Self> {
        let w = <super::half16::U16x4Wasm as CastRegister<super::U32x4Wasm>>::saturating_cast_from(value);
        <Self as CastRegister<super::half16::U16x4Wasm>>::saturating_cast_from(w)
    }

    #[rustfmt::skip]
    fn cast_from(value: Storage<super::U32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 4, 8, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        >(value, value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4Wasm, 2>> for I8x8Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4Wasm, 2>>) -> Storage<Self> {
        let w = <super::I16x8Wasm as CastRegister<ArrayRegister<super::I32x4Wasm, 2>>>::saturating_cast_from(value);
        <Self as CastRegister<super::I16x8Wasm>>::saturating_cast_from(w)
    }

    fn cast_from(value: Storage<ArrayRegister<super::I32x4Wasm, 2>>) -> Storage<Self> {
        let lo = store_dwords(value.0[0]);
        let hi = store_dwords(value.0[1]);
        ReducedRegister::new(arch::i8x16(
            lo[0] as i8,
            lo[1] as i8,
            lo[2] as i8,
            lo[3] as i8,
            hi[0] as i8,
            hi[1] as i8,
            hi[2] as i8,
            hi[3] as i8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4Wasm, 2>> for U8x8Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 2>>) -> Storage<Self> {
        let w = <super::U16x8Wasm as CastRegister<ArrayRegister<super::U32x4Wasm, 2>>>::saturating_cast_from(value);
        <Self as CastRegister<super::U16x8Wasm>>::saturating_cast_from(w)
    }

    fn cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 2>>) -> Storage<Self> {
        let lo = store_dwords(value.0[0]);
        let hi = store_dwords(value.0[1]);
        ReducedRegister::new(arch::i8x16(
            lo[0] as i8,
            lo[1] as i8,
            lo[2] as i8,
            lo[3] as i8,
            hi[0] as i8,
            hi[1] as i8,
            hi[2] as i8,
            hi[3] as i8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }
}

// WASM has no 64-bit narrow, so `i64 -> i8` and the 2-lane `i32/i64 -> i8` combos clamp + narrow.
macro_rules! sat_clamp_narrow8 {
    ($from:ty, $fe:ty, $ie:ty) => {
        #[inline(always)]
        fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
            let lo = <$from as Register>::splat(<$ie>::MIN as $fe);
            let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
            let clamped = <$from as NumericRegister>::min(<$from as NumericRegister>::max(value, lo), hi);
            <Self as CastRegister<$from>>::cast_from(clamped)
        }
    };
}

// `ReducedRegister<R, N>` removes `N` lanes (lane count = R::Lanes - N); over the native 16-lane
// register the 4-lane form removes 12 and the 8-lane form removes 8.
/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16Wasm`.
pub type I8x4Wasm = ReducedRegister<super::I8x16Wasm, U12>;
/// 4-lane unsigned 8-bit register.
pub type U8x4Wasm = ReducedRegister<super::U8x16Wasm, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16Wasm`.
pub type I8x8Wasm = ReducedRegister<super::I8x16Wasm, U8>;
/// 8-lane unsigned 8-bit register.
pub type U8x8Wasm = ReducedRegister<super::U8x16Wasm, U8>;

#[inline(always)]
fn store_bytes(v: arch::v128) -> [i8; 16] {
    let mut arr = [0i8; 16];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
    arr
}

#[inline(always)]
fn store_dwords(v: arch::v128) -> [i32; 4] {
    let mut arr = [0i32; 4];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
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
                ReducedRegister::new(arch::i8x16(
                    lo.0[0] as i8,
                    lo.0[1] as i8,
                    hi.0[0] as i8,
                    hi.0[1] as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ))
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
                ReducedRegister::new(arch::i8x16(
                    value.0[0] as i8,
                    value.0[1] as i8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ))
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<$elem, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] as $elem, a[1] as $elem])
            }
        }
    };
}

impl_concat_x4_from_x2!(I8x4Wasm, i8);
impl_concat_x4_from_x2!(U8x4Wasm, u8);

// ===========================================================================================
// x8 <- x4 (both reduced over the same native register).
// ===========================================================================================

macro_rules! impl_concat_x8_from_x4 {
    ($x8:ty, $x4:ty) => {
        #[thermite_macros::inline_always]
        impl ExtendRegister<$x4> for $x8 {
            fn extend(value: Storage<$x4>) -> Storage<Self> {
                ReducedRegister::new(value.0)
            }
            fn narrow(value: Storage<Self>) -> Storage<$x4> {
                ReducedRegister::new(value.0)
            }
        }

        #[thermite_macros::inline_always]
        impl ConcatRegister<$x4> for $x8 {
            fn concat(lo: Storage<$x4>, hi: Storage<$x4>) -> Storage<Self> {
                // Each x4 keeps its data in the low 4 bytes; interleave the low 32 bits of each.
                ReducedRegister::new(arch::i32x4_shuffle::<0, 4, 1, 5>(lo.0, hi.0))
            }
            fn split(value: Storage<Self>) -> (Storage<$x4>, Storage<$x4>) {
                (
                    ReducedRegister::new(value.0),
                    // shift the high 4 bytes (lanes 4..8) down into the low 4 bytes.
                    ReducedRegister::new(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0)),
                )
            }
        }
    };
}

impl_concat_x8_from_x4!(I8x8Wasm, I8x4Wasm);
impl_concat_x8_from_x4!(U8x8Wasm, U8x4Wasm);

// ===========================================================================================
// x16 (native) <- x8 (reduced). Both 8-lane halves live in the low 64 bits; merge via i64x2.
// ===========================================================================================

macro_rules! impl_concat_x16_from_x8 {
    ($native:ty, $x8:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<$x8> for $native {
            fn concat(lo: Storage<$x8>, hi: Storage<$x8>) -> Storage<Self> {
                arch::i64x2_shuffle::<0, 2>(lo.0, hi.0)
            }
            fn split(value: Storage<Self>) -> (Storage<$x8>, Storage<$x8>) {
                (
                    ReducedRegister::new(value),
                    ReducedRegister::new(arch::i64x2_shuffle::<1, 1>(value, value)),
                )
            }
        }
    };
}

impl_concat_x16_from_x8!(super::I8x16Wasm, I8x8Wasm);
impl_concat_x16_from_x8!(super::U8x16Wasm, U8x8Wasm);

// ===========================================================================================
// 8 <-> 16 widen/narrow casts.
//   widen 8 -> 16 via `i16x8_extend_low_*` (low 8 bytes -> 8x i16).
//   narrow 16 -> 8 via `i8x16_shuffle` gathering byte 0 of each i16 (even byte positions).
//   x4: I8x4Wasm <-> half16::I16x4Wasm.  x8: I8x8Wasm <-> I16x8Wasm (native).
//   x16: I8x16Wasm <-> ArrayRegister<I16x8Wasm, 2>.
// ===========================================================================================

// --- x4 (both reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Wasm> for super::half16::I16x4Wasm {
    fn cast_from(value: Storage<I8x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8_extend_low_i8x16(value.0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Wasm> for super::half16::U16x4Wasm {
    fn cast_from(value: Storage<U8x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8_extend_low_u8x16(value.0))
    }
}

// --- x8 (i16x8 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Wasm> for super::I16x8Wasm {
    fn cast_from(value: Storage<I8x8Wasm>) -> Storage<Self> {
        arch::i16x8_extend_low_i8x16(value.0)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Wasm> for super::U16x8Wasm {
    fn cast_from(value: Storage<U8x8Wasm>) -> Storage<Self> {
        arch::i16x8_extend_low_u8x16(value.0)
    }
}

// --- x16 (i16x16 = ArrayRegister<I16x8Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Wasm> for ArrayRegister<super::I16x8Wasm, 2> {
    fn cast_from(value: Storage<super::I8x16Wasm>) -> Storage<Self> {
        ArrayRegister([
            arch::i16x8_extend_low_i8x16(value),
            arch::i16x8_extend_high_i8x16(value),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Wasm> for ArrayRegister<super::U16x8Wasm, 2> {
    fn cast_from(value: Storage<super::U8x16Wasm>) -> Storage<Self> {
        ArrayRegister([
            arch::i16x8_extend_low_u8x16(value),
            arch::i16x8_extend_high_u8x16(value),
        ])
    }
}

// ===========================================================================================
// Widen casts to 32-bit (byte -> word -> dword via *_extend_low_*; narrows via i8x16_shuffle).
//   x4: I8x4Wasm <-> I32x4Wasm.  x8: I8x8Wasm <-> ArrayRegister<I32x4Wasm, 2>.
//   x2: ArrayRegister<i8,2> <-> I32x2Wasm.
// ===========================================================================================

// --- x4 ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Wasm> for super::I32x4Wasm {
    fn cast_from(value: Storage<I8x4Wasm>) -> Storage<Self> {
        unsafe { arch::widen_i8_to_i32(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Wasm> for super::U32x4Wasm {
    fn cast_from(value: Storage<U8x4Wasm>) -> Storage<Self> {
        unsafe { arch::widen_u8_to_u32(value.0) }
    }
}

// --- x8 (i32x8 = ArrayRegister<I32x4Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Wasm> for ArrayRegister<super::I32x4Wasm, 2> {
    fn cast_from(value: Storage<I8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_i8_to_i32(value.0);
            let hi = arch::widen_i8_to_i32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            ArrayRegister([lo, hi])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Wasm> for ArrayRegister<super::U32x4Wasm, 2> {
    fn cast_from(value: Storage<U8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_u8_to_u32(value.0);
            let hi = arch::widen_u8_to_u32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            ArrayRegister([lo, hi])
        }
    }
}

// --- x16 widen i8 -> i32 (native 16 bytes -> ArrayRegister<I32x4Wasm, 4>) ---
// Move each 4-byte group into the low 4 bytes (via a dword shuffle), then widen the low 4.
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Wasm> for ArrayRegister<super::I32x4Wasm, 4> {
    fn cast_from(value: Storage<super::I8x16Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::widen_i8x16_to_4xi32x4(value)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Wasm> for ArrayRegister<super::U32x4Wasm, 4> {
    fn cast_from(value: Storage<super::U8x16Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::widen_u8x16_to_4xu32x4(value)) }
    }
}


// --- x2 (ArrayRegister<i8,2> <-> I32x2Wasm reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u32x4(value.0[0] as u32, value.0[1] as u32, 0, 0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2Wasm> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2Wasm>) -> Storage<Self> {
        let a = store_dwords(value.0);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }

    sat_clamp_narrow8!(super::half::I32x2Wasm, i32, i8);
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2Wasm> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2Wasm>) -> Storage<Self> {
        let a = store_dwords(value.0);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }

    sat_clamp_narrow8!(super::half::U32x2Wasm, u32, u8);
}

// ===========================================================================================
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 8 -> 64: store the source bytes and rebuild each I64x2Wasm via `i64x2(a, b)`; `as i64`
//     sign-extends (signed source) or zero-extends (unsigned source).
//   narrow 64 -> 8: store each I64x2Wasm's lanes and rebuild via `i8x16(...)` truncating `as i8`.
//   x2:  ArrayRegister<i8,2> <-> I64x2Wasm (native 2-lane).
//   x4:  I8x4Wasm <-> ArrayRegister<I64x2Wasm, 2> (the wasm i64x4).
//   x8:  I8x8Wasm <-> ArrayRegister<I64x2Wasm, 4> (the wasm i64x8).
//   x16: I8x16Wasm <-> ArrayRegister<I64x2Wasm, 8> (the wasm i64x16).
// ===========================================================================================

#[inline(always)]
pub(super) fn store_qwords(v: arch::v128) -> [i64; 2] {
    let mut arr = [0i64; 2];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// --- x2 widen i8 -> i64 (ArrayRegister<i8,2> -> I64x2Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        arch::i64x2(value.0[0] as i64, value.0[1] as i64)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        arch::u64x2(value.0[0] as u64, value.0[1] as u64)
    }
}

// --- x2 narrow i64 -> i8 (I64x2Wasm native -> ArrayRegister<i8,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Wasm> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2Wasm>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }

    sat_clamp_narrow8!(super::I64x2Wasm, i64, i8);
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Wasm> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2Wasm>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }

    sat_clamp_narrow8!(super::U64x2Wasm, u64, u8);
}

// --- x4 widen i8 -> i64 (low 4 bytes -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Wasm> for ArrayRegister<super::I64x2Wasm, 2> {
    fn cast_from(value: Storage<I8x4Wasm>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister([
            arch::i64x2(a[0] as i64, a[1] as i64),
            arch::i64x2(a[2] as i64, a[3] as i64),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Wasm> for ArrayRegister<super::U64x2Wasm, 2> {
    fn cast_from(value: Storage<U8x4Wasm>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister([
            arch::u64x2(a[0] as u8 as u64, a[1] as u8 as u64),
            arch::u64x2(a[2] as u8 as u64, a[3] as u8 as u64),
        ])
    }
}

// --- x4 narrow i64 -> i8 (byte 0 of each of 4 lanes -> low 4 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 2>> for I8x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 2>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        ReducedRegister::new(arch::i8x16(
            a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ))
    }

    sat_clamp_narrow8!(ArrayRegister<super::I64x2Wasm, 2>, i64, i8);
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 2>> for U8x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 2>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        ReducedRegister::new(arch::i8x16(
            a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ))
    }

    sat_clamp_narrow8!(ArrayRegister<super::U64x2Wasm, 2>, u64, u8);
}

// --- x8 widen i8 -> i64 (low 8 bytes -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Wasm> for ArrayRegister<super::I64x2Wasm, 4> {
    fn cast_from(value: Storage<I8x8Wasm>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister([
            arch::i64x2(a[0] as i64, a[1] as i64),
            arch::i64x2(a[2] as i64, a[3] as i64),
            arch::i64x2(a[4] as i64, a[5] as i64),
            arch::i64x2(a[6] as i64, a[7] as i64),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Wasm> for ArrayRegister<super::U64x2Wasm, 4> {
    fn cast_from(value: Storage<U8x8Wasm>) -> Storage<Self> {
        let a = store_bytes(value.0);
        ArrayRegister([
            arch::u64x2(a[0] as u8 as u64, a[1] as u8 as u64),
            arch::u64x2(a[2] as u8 as u64, a[3] as u8 as u64),
            arch::u64x2(a[4] as u8 as u64, a[5] as u8 as u64),
            arch::u64x2(a[6] as u8 as u64, a[7] as u8 as u64),
        ])
    }
}

// --- x8 narrow i64 -> i8 (byte 0 of each of 8 lanes -> low 8 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 4>> for I8x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 4>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        let c = store_qwords(value.0[2]);
        let d = store_qwords(value.0[3]);
        ReducedRegister::new(arch::i8x16(
            a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8, 0, 0, 0, 0,
            0, 0, 0, 0,
        ))
    }

    sat_clamp_narrow8!(ArrayRegister<super::I64x2Wasm, 4>, i64, i8);
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 4>> for U8x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 4>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        let c = store_qwords(value.0[2]);
        let d = store_qwords(value.0[3]);
        ReducedRegister::new(arch::i8x16(
            a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8, 0, 0, 0, 0,
            0, 0, 0, 0,
        ))
    }

    sat_clamp_narrow8!(ArrayRegister<super::U64x2Wasm, 4>, u64, u8);
}

// --- x16 widen i8 -> i64 (native 16 bytes -> ArrayRegister<I64x2Wasm, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Wasm> for ArrayRegister<super::I64x2Wasm, 8> {
    fn cast_from(value: Storage<super::I8x16Wasm>) -> Storage<Self> {
        let a = store_bytes(value);
        ArrayRegister([
            arch::i64x2(a[0] as i64, a[1] as i64),
            arch::i64x2(a[2] as i64, a[3] as i64),
            arch::i64x2(a[4] as i64, a[5] as i64),
            arch::i64x2(a[6] as i64, a[7] as i64),
            arch::i64x2(a[8] as i64, a[9] as i64),
            arch::i64x2(a[10] as i64, a[11] as i64),
            arch::i64x2(a[12] as i64, a[13] as i64),
            arch::i64x2(a[14] as i64, a[15] as i64),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Wasm> for ArrayRegister<super::U64x2Wasm, 8> {
    fn cast_from(value: Storage<super::U8x16Wasm>) -> Storage<Self> {
        let a = store_bytes(value);
        ArrayRegister([
            arch::u64x2(a[0] as u8 as u64, a[1] as u8 as u64),
            arch::u64x2(a[2] as u8 as u64, a[3] as u8 as u64),
            arch::u64x2(a[4] as u8 as u64, a[5] as u8 as u64),
            arch::u64x2(a[6] as u8 as u64, a[7] as u8 as u64),
            arch::u64x2(a[8] as u8 as u64, a[9] as u8 as u64),
            arch::u64x2(a[10] as u8 as u64, a[11] as u8 as u64),
            arch::u64x2(a[12] as u8 as u64, a[13] as u8 as u64),
            arch::u64x2(a[14] as u8 as u64, a[15] as u8 as u64),
        ])
    }
}


// ===========================================================================================
// 8 <-> f32/f64 direct casts (widen int -> i32 then native i32<->float converts; narrow via
// truncating-saturating float -> i32 then the existing store-rebuild byte narrows).
//   WIDEN  i8/u8 -> f32 = widen low bytes to i32x4 then `f32x4_convert_i32x4` (exact).
//   NARROW f32 -> i8/u8 = `i32x4_trunc_sat_f32x4` (truncate-saturate) then rebuild low byte/lane.
//   WIDEN  i8/u8 -> f64 = widen to i32 then `f64x2_convert_low_i32x4` (2 lanes per F64x2, fan out).
//   NARROW f64 -> i8/u8 = `i32x4_trunc_sat_f64x2_zero` (truncate, low 2 lanes) then rebuild bytes.
// Unsigned 8-bit values fit in positive i32, so the signed i32->float convert is exact for them.
// The trunc_sat narrow matches the existing 32-bit wasm float->int cast path.
// ===========================================================================================

// --- x2 (ArrayRegister<i8,2> <-> ReducedRegister<F32x4Wasm,2> = F32x2Wasm) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::F32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        ReducedRegister::new(arch::f32x4_convert_i32x4(ints))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::F32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        ReducedRegister::new(arch::f32x4_convert_i32x4(ints))
    }
}

// --- x2 (ArrayRegister<i8,2> <-> F64x2Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::F64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        arch::f64x2_convert_low_i32x4(ints)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::F64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        arch::f64x2_convert_low_i32x4(ints)
    }
}

// --- x4 (I8x4Wasm <-> F32x4Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Wasm> for super::F32x4Wasm {
    fn cast_from(value: Storage<I8x4Wasm>) -> Storage<Self> {
        unsafe { arch::f32x4_convert_i32x4(arch::widen_i8_to_i32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Wasm> for super::F32x4Wasm {
    fn cast_from(value: Storage<U8x4Wasm>) -> Storage<Self> {
        unsafe { arch::f32x4_convert_i32x4(arch::widen_u8_to_u32(value.0)) }
    }
}

// --- x4 (I8x4Wasm <-> ArrayRegister<F64x2Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Wasm> for ArrayRegister<super::F64x2Wasm, 2> {
    fn cast_from(value: Storage<I8x4Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::i32x4_to_2xf64x2(arch::widen_i8_to_i32(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Wasm> for ArrayRegister<super::F64x2Wasm, 2> {
    fn cast_from(value: Storage<U8x4Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::i32x4_to_2xf64x2(arch::widen_u8_to_u32(value.0))) }
    }
}

// --- x8 (I8x8Wasm <-> ArrayRegister<F32x4Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Wasm> for ArrayRegister<super::F32x4Wasm, 2> {
    fn cast_from(value: Storage<I8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_i8_to_i32(value.0);
            let hi = arch::widen_i8_to_i32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            ArrayRegister([arch::f32x4_convert_i32x4(lo), arch::f32x4_convert_i32x4(hi)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Wasm> for ArrayRegister<super::F32x4Wasm, 2> {
    fn cast_from(value: Storage<U8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_u8_to_u32(value.0);
            let hi = arch::widen_u8_to_u32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            ArrayRegister([arch::f32x4_convert_i32x4(lo), arch::f32x4_convert_i32x4(hi)])
        }
    }
}

// --- x8 (I8x8Wasm <-> ArrayRegister<F64x2Wasm, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Wasm> for ArrayRegister<super::F64x2Wasm, 4> {
    fn cast_from(value: Storage<I8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_i8_to_i32(value.0);
            let hi = arch::widen_i8_to_i32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            let a = arch::i32x4_to_2xf64x2(lo);
            let b = arch::i32x4_to_2xf64x2(hi);
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Wasm> for ArrayRegister<super::F64x2Wasm, 4> {
    fn cast_from(value: Storage<U8x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_u8_to_u32(value.0);
            let hi = arch::widen_u8_to_u32(arch::i32x4_shuffle::<1, 0, 0, 0>(value.0, value.0));
            let a = arch::i32x4_to_2xf64x2(lo);
            let b = arch::i32x4_to_2xf64x2(hi);
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}

// --- x16 (I8x16Wasm native <-> ArrayRegister<F32x4Wasm, 4>) ---
// Move each 4-byte group into the low 4 bytes (via a dword shuffle), then widen the low 4.
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Wasm> for ArrayRegister<super::F32x4Wasm, 4> {
    fn cast_from(value: Storage<super::I8x16Wasm>) -> Storage<Self> {
        unsafe {
            let q = arch::widen_i8x16_to_4xi32x4(value);
            ArrayRegister([
                arch::f32x4_convert_i32x4(q[0]),
                arch::f32x4_convert_i32x4(q[1]),
                arch::f32x4_convert_i32x4(q[2]),
                arch::f32x4_convert_i32x4(q[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Wasm> for ArrayRegister<super::F32x4Wasm, 4> {
    fn cast_from(value: Storage<super::U8x16Wasm>) -> Storage<Self> {
        unsafe {
            let q = arch::widen_u8x16_to_4xu32x4(value);
            ArrayRegister([
                arch::f32x4_convert_i32x4(q[0]),
                arch::f32x4_convert_i32x4(q[1]),
                arch::f32x4_convert_i32x4(q[2]),
                arch::f32x4_convert_i32x4(q[3]),
            ])
        }
    }
}

// --- x16 (I8x16Wasm native <-> ArrayRegister<F64x2Wasm, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Wasm> for ArrayRegister<super::F64x2Wasm, 8> {
    fn cast_from(value: Storage<super::I8x16Wasm>) -> Storage<Self> {
        unsafe {
            let q = arch::widen_i8x16_to_4xi32x4(value);
            let a = arch::i32x4_to_2xf64x2(q[0]);
            let b = arch::i32x4_to_2xf64x2(q[1]);
            let c = arch::i32x4_to_2xf64x2(q[2]);
            let d = arch::i32x4_to_2xf64x2(q[3]);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Wasm> for ArrayRegister<super::F64x2Wasm, 8> {
    fn cast_from(value: Storage<super::U8x16Wasm>) -> Storage<Self> {
        unsafe {
            let q = arch::widen_u8x16_to_4xu32x4(value);
            let a = arch::i32x4_to_2xf64x2(q[0]);
            let b = arch::i32x4_to_2xf64x2(q[1]);
            let c = arch::i32x4_to_2xf64x2(q[2]);
            let d = arch::i32x4_to_2xf64x2(q[3]);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}

// ===========================================================================================
// Mask-side concat (x4 <- bool2).
// ===========================================================================================

#[inline(always)]
fn bool_to_i8_mask(b: bool) -> i8 {
    if b { !0 } else { 0 }
}

macro_rules! impl_mask_concat_x4_from_bool2 {
    ($red:ty) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<bool, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new(arch::i8x16(
                    bool_to_i8_mask(lo.0[0]),
                    bool_to_i8_mask(lo.0[1]),
                    bool_to_i8_mask(hi.0[0]),
                    bool_to_i8_mask(hi.0[1]),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ))
            }
            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
                let a = store_bytes(value.0);
                (
                    ArrayRegister([a[0] != 0, a[1] != 0]),
                    ArrayRegister([a[2] != 0, a[3] != 0]),
                )
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<bool, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new(arch::i8x16(
                    bool_to_i8_mask(value.0[0]),
                    bool_to_i8_mask(value.0[1]),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ))
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
                let a = store_bytes(value.0);
                ArrayRegister([a[0] != 0, a[1] != 0])
            }
        }
    };
}

impl_mask_concat_x4_from_bool2!(I8x4Wasm);
impl_mask_concat_x4_from_bool2!(U8x4Wasm);

// ===========================================================================================
// Cross-type gather/scatter index markers (scalar fallback).
// ===========================================================================================

macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(<super::super::Wasm as crate::simd::Simd>::u32x4 => I8x4Wasm, U8x4Wasm);
impl_indexable8!(<super::super::Wasm as crate::simd::Simd>::u64x4 => I8x4Wasm, U8x4Wasm);
impl_indexable8!(<super::super::Wasm as crate::simd::Simd>::u32x8 => I8x8Wasm, U8x8Wasm);
impl_indexable8!(<super::super::Wasm as crate::simd::Simd>::u64x8 => I8x8Wasm, U8x8Wasm);
