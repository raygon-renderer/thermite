//! Reduced (4-lane) 16-bit registers for the WASM backend and the cast/concat glue bridging
//! the scalar/array 16-bit halves, the native 128-bit `I16x8Wasm`/`U16x8Wasm`, and the 32-bit
//! registers they widen into. Mirrors the x86-v2 `half16.rs` structure with WASM intrinsics.

use generic_array::typenum::U4;

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// --- saturating narrows into 16-bit ---

// i32x4 -> i16x4 via `i16x8.narrow_i32x4_s` (low half holds the result).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4Wasm> for I16x4Wasm {
    fn saturating_cast_from(value: Storage<super::I32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8_narrow_i32x4(value, value))
    }
}
// u32x4 -> u16x4: `u16x8.narrow_i32x4_u` reads a signed source, so clamp the high end first.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4Wasm> for U16x4Wasm {
    fn saturating_cast_from(value: Storage<super::U32x4Wasm>) -> Storage<Self> {
        let c = arch::u32x4_min(value, arch::u32x4_splat(0xFFFF));
        ReducedRegister::new(arch::u16x8_narrow_i32x4(c, c))
    }
}

// WASM has no 64-bit narrow, so `i64 -> i16` and the 2-lane `i32/i64 -> i16` combos with a
// scalar `ArrayRegister` destination clamp into range (register `min`/`max`) + truncating narrow.
macro_rules! sat_clamp_narrow16 {
    ($(($from:ty, $fe:ty, $into:ty, $ie:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl SaturatingCastRegister<$from> for $into {
            fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                let lo = <$from as Register>::splat(<$ie>::MIN as $fe);
                let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
                let clamped = <$from as NumericRegister>::min(<$from as NumericRegister>::max(value, lo), hi);
                <Self as CastRegister<$from>>::cast_from(clamped)
            }
        }
    )*};
}
sat_clamp_narrow16! {
    (ArrayRegister<super::I64x2Wasm, 2>, i64, I16x4Wasm, i16),
    (ArrayRegister<super::U64x2Wasm, 2>, u64, U16x4Wasm, u16),
    (super::half::I32x2Wasm, i32, ArrayRegister<i16, 2>, i16),
    (super::half::U32x2Wasm, u32, ArrayRegister<u16, 2>, u16),
    (super::I64x2Wasm, i64, ArrayRegister<i16, 2>, i16),
    (super::U64x2Wasm, u64, ArrayRegister<u16, 2>, u16),
    (ArrayRegister<super::I64x2Wasm, 8>, i64, ArrayRegister<super::I16x8Wasm, 2>, i16),
    (ArrayRegister<super::U64x2Wasm, 8>, u64, ArrayRegister<super::U16x8Wasm, 2>, u16),
}

/// 4-lane signed 16-bit register, backed by the low 4 lanes of a 128-bit `I16x8Wasm`.
pub type I16x4Wasm = ReducedRegister<super::I16x8Wasm, U4>;
/// 4-lane unsigned 16-bit register, backed by the low 4 lanes of a 128-bit `U16x8Wasm`.
pub type U16x4Wasm = ReducedRegister<super::U16x8Wasm, U4>;

// ---------------------------------------------------------------------------------------
// x4 <- x2 : build a 4-lane reduced register from two 2-lane (scalar-array) halves.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<i16, 2>> for I16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<i16, 2>>, hi: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<i16, 2>>, Storage<ArrayRegister<i16, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<i16, 2>> for I16x4Wasm {
    fn extend(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<i16, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<u16, 2>> for U16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<u16, 2>>, hi: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u16x8(lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<u16, 2>>, Storage<ArrayRegister<u16, 2>>) {
        let mut arr = [0u16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (ArrayRegister([arr[0], arr[1]]), ArrayRegister([arr[2], arr[3]]))
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<u16, 2>> for U16x4Wasm {
    fn extend(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u16x8(value.0[0], value.0[1], 0, 0, 0, 0, 0, 0))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<u16, 2>> {
        let mut arr = [0u16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0], arr[1]])
    }
}

// ---------------------------------------------------------------------------------------
// x8 <- x4 : build the native 8-lane register from two 4-lane reduced halves.
// Both halves live in the low 64 bits of their v128; an i64x2 shuffle merges them.
// ---------------------------------------------------------------------------------------

#[thermite_macros::inline_always]
impl ConcatRegister<I16x4Wasm> for super::I16x8Wasm {
    fn concat(lo: Storage<I16x4Wasm>, hi: Storage<I16x4Wasm>) -> Storage<Self> {
        arch::i64x2_shuffle::<0, 2>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<I16x4Wasm>, Storage<I16x4Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i64x2_shuffle::<1, 1>(value, value)),
        )
    }
}

// NOTE: `ExtendRegister<I16x4Wasm> for I16x8Wasm` is provided by the ReducedRegister blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U16x4Wasm> for super::U16x8Wasm {
    fn concat(lo: Storage<U16x4Wasm>, hi: Storage<U16x4Wasm>) -> Storage<Self> {
        arch::i64x2_shuffle::<0, 2>(lo.0, hi.0)
    }

    fn split(value: Storage<Self>) -> (Storage<U16x4Wasm>, Storage<U16x4Wasm>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(arch::i64x2_shuffle::<1, 1>(value, value)),
        )
    }
}

// NOTE: `ExtendRegister<U16x4Wasm> for U16x8Wasm` is provided by the ReducedRegister blanket.

// ---------------------------------------------------------------------------------------
// Widen casts to 32-bit. x4: I16x4Wasm <-> I32x4Wasm.  x2: ArrayRegister<i16,2> <-> I32x2Wasm.
// ---------------------------------------------------------------------------------------

// --- x4 widen i16 -> i32 (low 4 lanes of the reduced register) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Wasm> for super::I32x4Wasm {
    fn cast_from(value: Storage<I16x4Wasm>) -> Storage<Self> {
        arch::i32x4_extend_low_i16x8(value.0)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x4Wasm> for super::U32x4Wasm {
    fn cast_from(value: Storage<U16x4Wasm>) -> Storage<Self> {
        arch::i32x4_extend_low_u16x8(value.0)
    }
}

// --- x4 narrow i32 -> i16 (truncate low 16 bits per lane into the low 4 i16 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4Wasm> for I16x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<super::I32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x4Wasm> for U16x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<super::U32x4Wasm>) -> Storage<Self> {
        ReducedRegister::new(arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7,
        >(value, value))
    }
}

// --- x2 widen i16 -> i32 (ArrayRegister<i16,2> -> I32x2Wasm reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::I32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0))
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::U32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::u32x4(value.0[0] as u32, value.0[1] as u32, 0, 0))
    }
}

// --- x2 narrow i32 -> i16 (I32x2Wasm reduced -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2Wasm> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::I32x2Wasm>) -> Storage<Self> {
        let mut arr = [0i32; 4];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as i16, arr[1] as i16])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2Wasm> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::U32x2Wasm>) -> Storage<Self> {
        let mut arr = [0u32; 4];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] as u16, arr[1] as u16])
    }
}

// ---------------------------------------------------------------------------------------
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 16 -> 64: store the source words and rebuild each I64x2Wasm via `i64x2(a, b)`;
//     `as i64` sign-extends (signed source) or zero-extends (unsigned source).
//   narrow 64 -> 16: store each I64x2Wasm's lanes and rebuild via `i16x8(...)` truncating `as i16`.
//   x2:  ArrayRegister<i16,2> <-> I64x2Wasm (native 2-lane).
//   x4:  I16x4Wasm <-> ArrayRegister<I64x2Wasm, 2> (the wasm i64x4).
//   x8:  I16x8Wasm <-> ArrayRegister<I64x2Wasm, 4> (the wasm i64x8).
//   x16: ArrayRegister<I16x8Wasm, 2> <-> ArrayRegister<I64x2Wasm, 8> (i16x16 / i64x16).
// ---------------------------------------------------------------------------------------

#[inline(always)]
fn store_words(v: arch::v128) -> [i16; 8] {
    let mut arr = [0i16; 8];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
    arr
}

#[inline(always)]
fn store_qwords(v: arch::v128) -> [i64; 2] {
    let mut arr = [0i64; 2];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
    arr
}

// --- x2 widen i16 -> i64 (ArrayRegister<i16,2> -> I64x2Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::I64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        arch::i64x2(value.0[0] as i64, value.0[1] as i64)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::U64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        arch::u64x2(value.0[0] as u64, value.0[1] as u64)
    }
}

// --- x2 narrow i64 -> i16 (I64x2Wasm native -> ArrayRegister<i16,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Wasm> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::I64x2Wasm>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Wasm> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::U64x2Wasm>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x4 widen i16 -> i64 (low 4 words -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Wasm> for ArrayRegister<super::I64x2Wasm, 2> {
    fn cast_from(value: Storage<I16x4Wasm>) -> Storage<Self> {
        let a = store_words(value.0);
        ArrayRegister([
            arch::i64x2(a[0] as i64, a[1] as i64),
            arch::i64x2(a[2] as i64, a[3] as i64),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Wasm> for ArrayRegister<super::U64x2Wasm, 2> {
    fn cast_from(value: Storage<U16x4Wasm>) -> Storage<Self> {
        let a = store_words(value.0);
        ArrayRegister([
            arch::u64x2(a[0] as u16 as u64, a[1] as u16 as u64),
            arch::u64x2(a[2] as u16 as u64, a[3] as u16 as u64),
        ])
    }
}

// --- x4 narrow i64 -> i16 (word 0 of each of 4 lanes -> low 4 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 2>> for I16x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 2>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        ReducedRegister::new(arch::i16x8(
            a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16, 0, 0, 0, 0,
        ))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 2>> for U16x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 2>>) -> Storage<Self> {
        let a = store_qwords(value.0[0]);
        let b = store_qwords(value.0[1]);
        ReducedRegister::new(arch::i16x8(
            a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16, 0, 0, 0, 0,
        ))
    }
}

// --- x8 widen i16 -> i64 (8 words -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Wasm> for ArrayRegister<super::I64x2Wasm, 4> {
    fn cast_from(value: Storage<super::I16x8Wasm>) -> Storage<Self> {
        let a = store_words(value);
        ArrayRegister([
            arch::i64x2(a[0] as i64, a[1] as i64),
            arch::i64x2(a[2] as i64, a[3] as i64),
            arch::i64x2(a[4] as i64, a[5] as i64),
            arch::i64x2(a[6] as i64, a[7] as i64),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Wasm> for ArrayRegister<super::U64x2Wasm, 4> {
    fn cast_from(value: Storage<super::U16x8Wasm>) -> Storage<Self> {
        let a = store_words(value);
        ArrayRegister([
            arch::u64x2(a[0] as u16 as u64, a[1] as u16 as u64),
            arch::u64x2(a[2] as u16 as u64, a[3] as u16 as u64),
            arch::u64x2(a[4] as u16 as u64, a[5] as u16 as u64),
            arch::u64x2(a[6] as u16 as u64, a[7] as u16 as u64),
        ])
    }
}

// --- x8 narrow i64 -> i16 (word 0 of each of 8 lanes -> 8 words) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 4>> for super::I16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 4>>) -> Storage<Self> {
        unsafe { arch::narrow_4xi64x2_to_words(value.0) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 4>> for super::U16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 4>>) -> Storage<Self> {
        unsafe { arch::narrow_4xi64x2_to_words(value.0) }
    }
}

// --- x16 widen i16 -> i64 (ArrayRegister<I16x8Wasm, 2> -> ArrayRegister<I64x2Wasm, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8Wasm, 2>> for ArrayRegister<super::I64x2Wasm, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8Wasm, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_i16x8_to_4xi64x2(value.0[0]);
            let hi = arch::widen_i16x8_to_4xi64x2(value.0[1]);
            ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8Wasm, 2>> for ArrayRegister<super::U64x2Wasm, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8Wasm, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::widen_u16x8_to_4xu64x2(value.0[0]);
            let hi = arch::widen_u16x8_to_4xu64x2(value.0[1]);
            ArrayRegister([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
        }
    }
}

// --- x16 narrow i64 -> i16 (ArrayRegister<I64x2Wasm, 8> -> ArrayRegister<I16x8Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 8>> for ArrayRegister<super::I16x8Wasm, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::narrow_4xi64x2_to_words([v[0], v[1], v[2], v[3]]),
                arch::narrow_4xi64x2_to_words([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 8>> for ArrayRegister<super::U16x8Wasm, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            ArrayRegister([
                arch::narrow_4xi64x2_to_words([v[0], v[1], v[2], v[3]]),
                arch::narrow_4xi64x2_to_words([v[4], v[5], v[6], v[7]]),
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// 16 <-> f32/f64 direct casts (widen low words to i32 then native i32<->float converts;
// narrow via truncating-saturating float -> i32 then the existing store-rebuild word narrows).
//   WIDEN  i16/u16 -> f32 = widen low 4 words to i32x4 then `f32x4_convert_i32x4` (exact).
//   NARROW f32 -> i16/u16 = `i32x4_trunc_sat_f32x4` (truncate-saturate) then rebuild low words.
//   WIDEN  i16/u16 -> f64 = widen to i32 then `f64x2_convert_low_i32x4` (2 lanes per F64x2, fan out).
//   NARROW f64 -> i16/u16 = `i32x4_trunc_sat_f64x2_zero` (truncate, low 2 lanes) then rebuild words.
// Unsigned 16-bit values fit in positive i32, so the signed i32->float convert is exact for them.
// The trunc_sat narrow matches the existing 32-bit wasm float->int cast path.
// ---------------------------------------------------------------------------------------

#[inline(always)]
fn store_dwords(v: arch::v128) -> [i32; 4] {
    let mut arr = [0i32; 4];
    unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, v) };
    arr
}
// --- x2 (ArrayRegister<i16,2> <-> ReducedRegister<F32x4Wasm,2> = F32x2Wasm) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::half::F32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        ReducedRegister::new(arch::f32x4_convert_i32x4(ints))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::half::F32x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        ReducedRegister::new(arch::f32x4_convert_i32x4(ints))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Wasm> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::half::F32x2Wasm>) -> Storage<Self> {
        let a = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0));
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Wasm> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::half::F32x2Wasm>) -> Storage<Self> {
        let a = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0));
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x2 (ArrayRegister<i16,2> <-> F64x2Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i16, 2>> for super::F64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<i16, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        arch::f64x2_convert_low_i32x4(ints)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u16, 2>> for super::F64x2Wasm {
    fn cast_from(value: Storage<ArrayRegister<u16, 2>>) -> Storage<Self> {
        let ints = arch::i32x4(value.0[0] as i32, value.0[1] as i32, 0, 0);
        arch::f64x2_convert_low_i32x4(ints)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Wasm> for ArrayRegister<i16, 2> {
    fn cast_from(value: Storage<super::F64x2Wasm>) -> Storage<Self> {
        let a = unsafe { arch::f64x2_to_2xi32(value) };
        ArrayRegister([a[0] as i16, a[1] as i16])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Wasm> for ArrayRegister<u16, 2> {
    fn cast_from(value: Storage<super::F64x2Wasm>) -> Storage<Self> {
        let a = unsafe { arch::f64x2_to_2xi32(value) };
        ArrayRegister([a[0] as u16, a[1] as u16])
    }
}

// --- x4 (I16x4Wasm <-> F32x4Wasm native) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Wasm> for super::F32x4Wasm {
    fn cast_from(value: Storage<I16x4Wasm>) -> Storage<Self> {
        arch::f32x4_convert_i32x4(arch::i32x4_extend_low_i16x8(value.0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Wasm> for super::F32x4Wasm {
    fn cast_from(value: Storage<U16x4Wasm>) -> Storage<Self> {
        arch::f32x4_convert_i32x4(arch::i32x4_extend_low_u16x8(value.0))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Wasm> for I16x4Wasm {
    fn cast_from(value: Storage<super::F32x4Wasm>) -> Storage<Self> {
        let a = store_dwords(arch::i32x4_trunc_sat_f32x4(value));
        ReducedRegister::new(arch::i16x8(
            a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0,
        ))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Wasm> for U16x4Wasm {
    fn cast_from(value: Storage<super::F32x4Wasm>) -> Storage<Self> {
        let a = store_dwords(arch::i32x4_trunc_sat_f32x4(value));
        ReducedRegister::new(arch::i16x8(
            a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, 0, 0, 0, 0,
        ))
    }
}

// --- x4 (I16x4Wasm <-> ArrayRegister<F64x2Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I16x4Wasm> for ArrayRegister<super::F64x2Wasm, 2> {
    fn cast_from(value: Storage<I16x4Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::i32x4_to_2xf64x2(arch::i32x4_extend_low_i16x8(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U16x4Wasm> for ArrayRegister<super::F64x2Wasm, 2> {
    fn cast_from(value: Storage<U16x4Wasm>) -> Storage<Self> {
        unsafe { ArrayRegister(arch::i32x4_to_2xf64x2(arch::i32x4_extend_low_u16x8(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 2>> for I16x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::f64x2_to_2xi32(value.0[0]);
            let hi = arch::f64x2_to_2xi32(value.0[1]);
            ReducedRegister::new(arch::i16x8(
                lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0,
            ))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 2>> for U16x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::f64x2_to_2xi32(value.0[0]);
            let hi = arch::f64x2_to_2xi32(value.0[1]);
            ReducedRegister::new(arch::i16x8(
                lo[0] as i16, lo[1] as i16, hi[0] as i16, hi[1] as i16, 0, 0, 0, 0,
            ))
        }
    }
}

// --- x8 (I16x8Wasm native <-> ArrayRegister<F32x4Wasm, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Wasm> for ArrayRegister<super::F32x4Wasm, 2> {
    fn cast_from(value: Storage<super::I16x8Wasm>) -> Storage<Self> {
        let lo = arch::i32x4_extend_low_i16x8(value);
        let hi = arch::i32x4_extend_low_i16x8(arch::i64x2_shuffle::<1, 1>(value, value));
        ArrayRegister([arch::f32x4_convert_i32x4(lo), arch::f32x4_convert_i32x4(hi)])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Wasm> for ArrayRegister<super::F32x4Wasm, 2> {
    fn cast_from(value: Storage<super::U16x8Wasm>) -> Storage<Self> {
        let lo = arch::i32x4_extend_low_u16x8(value);
        let hi = arch::i32x4_extend_low_u16x8(arch::i64x2_shuffle::<1, 1>(value, value));
        ArrayRegister([arch::f32x4_convert_i32x4(lo), arch::f32x4_convert_i32x4(hi)])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Wasm, 2>> for super::I16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Wasm, 2>>) -> Storage<Self> {
        let lo = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0[0]));
        let hi = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0[1]));
        arch::i16x8(
            lo[0] as i16, lo[1] as i16, lo[2] as i16, lo[3] as i16,
            hi[0] as i16, hi[1] as i16, hi[2] as i16, hi[3] as i16,
        )
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Wasm, 2>> for super::U16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Wasm, 2>>) -> Storage<Self> {
        let lo = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0[0]));
        let hi = store_dwords(arch::i32x4_trunc_sat_f32x4(value.0[1]));
        arch::i16x8(
            lo[0] as i16, lo[1] as i16, lo[2] as i16, lo[3] as i16,
            hi[0] as i16, hi[1] as i16, hi[2] as i16, hi[3] as i16,
        )
    }
}

// --- x8 (I16x8Wasm native <-> ArrayRegister<F64x2Wasm, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Wasm> for ArrayRegister<super::F64x2Wasm, 4> {
    fn cast_from(value: Storage<super::I16x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::i32x4_extend_low_i16x8(value);
            let hi = arch::i32x4_extend_low_i16x8(arch::i64x2_shuffle::<1, 1>(value, value));
            let a = arch::i32x4_to_2xf64x2(lo);
            let b = arch::i32x4_to_2xf64x2(hi);
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Wasm> for ArrayRegister<super::F64x2Wasm, 4> {
    fn cast_from(value: Storage<super::U16x8Wasm>) -> Storage<Self> {
        unsafe {
            let lo = arch::i32x4_extend_low_u16x8(value);
            let hi = arch::i32x4_extend_low_u16x8(arch::i64x2_shuffle::<1, 1>(value, value));
            let a = arch::i32x4_to_2xf64x2(lo);
            let b = arch::i32x4_to_2xf64x2(hi);
            ArrayRegister([a[0], a[1], b[0], b[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 4>> for super::I16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::f64x2_to_2xi32(value.0[0]);
            let b = arch::f64x2_to_2xi32(value.0[1]);
            let c = arch::f64x2_to_2xi32(value.0[2]);
            let d = arch::f64x2_to_2xi32(value.0[3]);
            arch::i16x8(
                a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
            )
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 4>> for super::U16x8Wasm {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::f64x2_to_2xi32(value.0[0]);
            let b = arch::f64x2_to_2xi32(value.0[1]);
            let c = arch::f64x2_to_2xi32(value.0[2]);
            let d = arch::f64x2_to_2xi32(value.0[3]);
            arch::i16x8(
                a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
            )
        }
    }
}

// --- x16 16 <-> f32: ArrayRegister<I16x8Wasm, 2> <-> ArrayRegister<F32x4Wasm, 4> is provided
//     for free by the ArrayRegister 2<->4 reshape cast blanket (array.rs), so no explicit impl. ---

// --- x16 (ArrayRegister<I16x8Wasm, 2> <-> ArrayRegister<F64x2Wasm, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I16x8Wasm, 2>> for ArrayRegister<super::F64x2Wasm, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::I16x8Wasm, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::i32x4_extend_low_i16x8(v[0]);
            let q1 = arch::i32x4_extend_low_i16x8(arch::i64x2_shuffle::<1, 1>(v[0], v[0]));
            let q2 = arch::i32x4_extend_low_i16x8(v[1]);
            let q3 = arch::i32x4_extend_low_i16x8(arch::i64x2_shuffle::<1, 1>(v[1], v[1]));
            let a = arch::i32x4_to_2xf64x2(q0);
            let b = arch::i32x4_to_2xf64x2(q1);
            let c = arch::i32x4_to_2xf64x2(q2);
            let d = arch::i32x4_to_2xf64x2(q3);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8Wasm, 2>> for ArrayRegister<super::F64x2Wasm, 8> {
    fn cast_from(value: Storage<ArrayRegister<super::U16x8Wasm, 2>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let q0 = arch::i32x4_extend_low_u16x8(v[0]);
            let q1 = arch::i32x4_extend_low_u16x8(arch::i64x2_shuffle::<1, 1>(v[0], v[0]));
            let q2 = arch::i32x4_extend_low_u16x8(v[1]);
            let q3 = arch::i32x4_extend_low_u16x8(arch::i64x2_shuffle::<1, 1>(v[1], v[1]));
            let a = arch::i32x4_to_2xf64x2(q0);
            let b = arch::i32x4_to_2xf64x2(q1);
            let c = arch::i32x4_to_2xf64x2(q2);
            let d = arch::i32x4_to_2xf64x2(q3);
            ArrayRegister([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 8>> for ArrayRegister<super::I16x8Wasm, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = arch::f64x2_to_2xi32(v[0]);
            let b = arch::f64x2_to_2xi32(v[1]);
            let c = arch::f64x2_to_2xi32(v[2]);
            let d = arch::f64x2_to_2xi32(v[3]);
            let e = arch::f64x2_to_2xi32(v[4]);
            let f = arch::f64x2_to_2xi32(v[5]);
            let g = arch::f64x2_to_2xi32(v[6]);
            let h = arch::f64x2_to_2xi32(v[7]);
            ArrayRegister([
                arch::i16x8(
                    a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                    c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
                ),
                arch::i16x8(
                    e[0] as i16, e[1] as i16, f[0] as i16, f[1] as i16,
                    g[0] as i16, g[1] as i16, h[0] as i16, h[1] as i16,
                ),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Wasm, 8>> for ArrayRegister<super::U16x8Wasm, 2> {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Wasm, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let a = arch::f64x2_to_2xi32(v[0]);
            let b = arch::f64x2_to_2xi32(v[1]);
            let c = arch::f64x2_to_2xi32(v[2]);
            let d = arch::f64x2_to_2xi32(v[3]);
            let e = arch::f64x2_to_2xi32(v[4]);
            let f = arch::f64x2_to_2xi32(v[5]);
            let g = arch::f64x2_to_2xi32(v[6]);
            let h = arch::f64x2_to_2xi32(v[7]);
            ArrayRegister([
                arch::i16x8(
                    a[0] as i16, a[1] as i16, b[0] as i16, b[1] as i16,
                    c[0] as i16, c[1] as i16, d[0] as i16, d[1] as i16,
                ),
                arch::i16x8(
                    e[0] as i16, e[1] as i16, f[0] as i16, f[1] as i16,
                    g[0] as i16, g[1] as i16, h[0] as i16, h[1] as i16,
                ),
            ])
        }
    }
}

// ---------------------------------------------------------------------------------------
// Mask-side concat: the reduced i16x4 register is its own Mask, and `FullConcatRegister`
// requires that Mask to concat from the i16x2 half's Mask (`ArrayRegister<bool, 2>`).
// ---------------------------------------------------------------------------------------

#[inline(always)]
fn bool_to_i16_mask(b: bool) -> i16 {
    if b { !0 } else { 0 }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for I16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(lo.0[0]),
            bool_to_i16_mask(lo.0[1]),
            bool_to_i16_mask(hi.0[0]),
            bool_to_i16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for I16x4Wasm {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(value.0[0]),
            bool_to_i16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<ArrayRegister<bool, 2>> for U16x4Wasm {
    fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(lo.0[0]),
            bool_to_i16_mask(lo.0[1]),
            bool_to_i16_mask(hi.0[0]),
            bool_to_i16_mask(hi.0[1]),
            0,
            0,
            0,
            0,
        ))
    }

    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        (
            ArrayRegister([arr[0] != 0, arr[1] != 0]),
            ArrayRegister([arr[2] != 0, arr[3] != 0]),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<ArrayRegister<bool, 2>> for U16x4Wasm {
    fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
        ReducedRegister::new(arch::i16x8(
            bool_to_i16_mask(value.0[0]),
            bool_to_i16_mask(value.0[1]),
            0,
            0,
            0,
            0,
            0,
            0,
        ))
    }

    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
        let mut arr = [0i16; 8];
        unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value.0) };
        ArrayRegister([arr[0] != 0, arr[1] != 0])
    }
}

// Gather/scatter for the reduced 16-bit registers (indexed by the native 4-lane index types)
// falls back to scalar; only the cross-type index markers are needed.
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u32x4> for I16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u32x4> for U16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u64x4> for I16x4Wasm {}
impl IndexableRegister<<super::super::Wasm as crate::simd::Simd>::u64x4> for U16x4Wasm {}
