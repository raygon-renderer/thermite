//! Reduced (sub-native-width) 8-bit registers for the NEON backend. Mirrors `half16.rs` one
//! element size down; widen casts go byte -> word -> dword via the `vmovl_*` ladder, and
//! narrows use the `vmovn_*`/`vqmovn_*` narrowing ladder (NEON has native narrows at every
//! element size, so the wasm backend's shuffle/store-rebuild dances collapse into real
//! narrowing instructions here).

use generic_array::typenum::{U8, U12};

use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister,
    Storage, array::ArrayRegister, reduced::ReducedRegister,
};

// --- saturating narrows into 8-bit ---

// Signed `i16 -> i8` via native `vqmovn_s16`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I16x8Neon> for I8x8Neon {
    fn saturating_cast_from(value: Storage<super::I16x8Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s8(arch::vqmovn_s16(value), arch::vdup_n_s8(0))) }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::I16x4Neon> for I8x4Neon {
    fn saturating_cast_from(value: Storage<super::half16::I16x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s8(arch::vqmovn_s16(value.0), arch::vdup_n_s8(0))) }
    }
}
// Unsigned `u16 -> u8`: native `vqmovn_u16` reads a genuinely unsigned source, so the
// pre-clamp the wasm backend needs (its narrow reads a signed source) is dropped here.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U16x8Neon> for U8x8Neon {
    fn saturating_cast_from(value: Storage<super::U16x8Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u8(arch::vqmovn_u16(value), arch::vdup_n_u8(0))) }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::half16::U16x4Neon> for U8x4Neon {
    fn saturating_cast_from(value: Storage<super::half16::U16x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u8(arch::vqmovn_u16(value.0), arch::vdup_n_u8(0))) }
    }
}

// Skip-level 32 -> 8: compose the 32 -> 16 and 16 -> 8 saturating narrows (idempotent).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x4Neon> for I8x4Neon {
    fn saturating_cast_from(value: Storage<super::I32x4Neon>) -> Storage<Self> {
        let w = <super::half16::I16x4Neon as SaturatingCastRegister<super::I32x4Neon>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::I16x4Neon>>::saturating_cast_from(w)
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U32x4Neon> for U8x4Neon {
    fn saturating_cast_from(value: Storage<super::U32x4Neon>) -> Storage<Self> {
        let w = <super::half16::U16x4Neon as SaturatingCastRegister<super::U32x4Neon>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::half16::U16x4Neon>>::saturating_cast_from(w)
    }
}
// x8 skip-level: direct `vqmovn` ladder (the wasm backend routes through the i16x8
// SaturatingCastRegister<ArrayRegister<I32x4, 2>> impl; that impl is not available in the NEON
// half16 module, so the two saturating narrows are spelled inline here).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4Neon, 2>> for I8x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            let w = arch::vqmovn_high_s32(arch::vqmovn_s32(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_s8(arch::vqmovn_s16(w), arch::vdup_n_s8(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U32x4Neon, 2>> for U8x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            let w = arch::vqmovn_high_u32(arch::vqmovn_u32(value.0[0]), value.0[1]);
            ReducedRegister::new(arch::vcombine_u8(arch::vqmovn_u16(w), arch::vdup_n_u8(0)))
        }
    }
}

// Skip-level 64 -> 8, vector outputs: NEON has native 64-bit saturating narrows, so chain
// `vqmovn_s64 -> vqmovn_s32 -> vqmovn_s16` instead of the wasm backend's generic
// clamp-then-truncate (saturating narrows compose: min/max clamping at each step is idempotent).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 2>> for I8x4Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let d32 = arch::vqmovn_high_s64(arch::vqmovn_s64(value.0[0]), value.0[1]);
            let d16 = arch::vcombine_s16(arch::vqmovn_s32(d32), arch::vdup_n_s16(0));
            ReducedRegister::new(arch::vcombine_s8(arch::vqmovn_s16(d16), arch::vdup_n_s8(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 2>> for U8x4Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            let d32 = arch::vqmovn_high_u64(arch::vqmovn_u64(value.0[0]), value.0[1]);
            let d16 = arch::vcombine_u16(arch::vqmovn_u32(d32), arch::vdup_n_u16(0));
            ReducedRegister::new(arch::vcombine_u8(arch::vqmovn_u16(d16), arch::vdup_n_u8(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 4>> for I8x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::vqmovn_high_s64(arch::vqmovn_s64(value.0[0]), value.0[1]);
            let b = arch::vqmovn_high_s64(arch::vqmovn_s64(value.0[2]), value.0[3]);
            let w = arch::vqmovn_high_s32(arch::vqmovn_s32(a), b);
            ReducedRegister::new(arch::vcombine_s8(arch::vqmovn_s16(w), arch::vdup_n_s8(0)))
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 4>> for U8x8Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::vqmovn_high_u64(arch::vqmovn_u64(value.0[0]), value.0[1]);
            let b = arch::vqmovn_high_u64(arch::vqmovn_u64(value.0[2]), value.0[3]);
            let w = arch::vqmovn_high_u32(arch::vqmovn_u32(a), b);
            ReducedRegister::new(arch::vcombine_u8(arch::vqmovn_u16(w), arch::vdup_n_u8(0)))
        }
    }
}

// Skip-level saturating narrows into the NATIVE 16-lane byte registers (required by the
// `Simd::i8x16`/`u8x16` bounds; the wasm backend implements these in its `i8x16.rs`, and the
// i16x16 -> i8x16 step is already stamped by `neon_widen_casts!` there).

// Saturating narrow i32x16 -> i8x16: three `vqmovn` levels (saturating narrows compose).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4Neon, 4>> for super::I8x16Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4Neon, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::vqmovn_high_s32(arch::vqmovn_s32(value.0[0]), value.0[1]);
            let b = arch::vqmovn_high_s32(arch::vqmovn_s32(value.0[2]), value.0[3]);
            arch::vqmovn_high_s16(arch::vqmovn_s16(a), b)
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U32x4Neon, 4>> for super::U8x16Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4Neon, 4>>) -> Storage<Self> {
        unsafe {
            let a = arch::vqmovn_high_u32(arch::vqmovn_u32(value.0[0]), value.0[1]);
            let b = arch::vqmovn_high_u32(arch::vqmovn_u32(value.0[2]), value.0[3]);
            arch::vqmovn_high_u16(arch::vqmovn_u16(a), b)
        }
    }
}

// Saturating narrow i64x16 -> i8x16: native `vqmovn_s64` ladder (the wasm backend has no
// 64-bit narrow and must clamp + truncate instead).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Neon, 8>> for super::I8x16Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 8>>) -> Storage<Self> {
        unsafe {
            let v = value.0;
            let a = arch::vqmovn_high_s64(arch::vqmovn_s64(v[0]), v[1]);
            let b = arch::vqmovn_high_s64(arch::vqmovn_s64(v[2]), v[3]);
            let c = arch::vqmovn_high_s64(arch::vqmovn_s64(v[4]), v[5]);
            let d = arch::vqmovn_high_s64(arch::vqmovn_s64(v[6]), v[7]);
            let lo = arch::vqmovn_high_s32(arch::vqmovn_s32(a), b);
            let hi = arch::vqmovn_high_s32(arch::vqmovn_s32(c), d);
            arch::vqmovn_high_s16(arch::vqmovn_s16(lo), hi)
        }
    }
}
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Neon, 8>> for super::U8x16Neon {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 8>>) -> Storage<Self> {
        unsafe {
            let v = value.0;
            let a = arch::vqmovn_high_u64(arch::vqmovn_u64(v[0]), v[1]);
            let b = arch::vqmovn_high_u64(arch::vqmovn_u64(v[2]), v[3]);
            let c = arch::vqmovn_high_u64(arch::vqmovn_u64(v[4]), v[5]);
            let d = arch::vqmovn_high_u64(arch::vqmovn_u64(v[6]), v[7]);
            let lo = arch::vqmovn_high_u32(arch::vqmovn_u32(a), b);
            let hi = arch::vqmovn_high_u32(arch::vqmovn_u32(c), d);
            arch::vqmovn_high_u16(arch::vqmovn_u16(lo), hi)
        }
    }
}

// The 2-lane scalar-array outputs keep the backend-agnostic clamp + truncating cast (the
// output is a scalar array either way, so there is no vector narrow to exploit).
macro_rules! sat_clamp_narrow8 {
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
sat_clamp_narrow8! {
    (super::half::I32x2Neon, i32, ArrayRegister<i8, 2>, i8),
    (super::half::U32x2Neon, u32, ArrayRegister<u8, 2>, u8),
    (super::I64x2Neon, i64, ArrayRegister<i8, 2>, i8),
    (super::U64x2Neon, u64, ArrayRegister<u8, 2>, u8),
}

// `ReducedRegister<R, N>` removes `N` lanes (lane count = R::Lanes - N); over the native 16-lane
// register the 4-lane form removes 12 and the 8-lane form removes 8.
/// 4-lane signed 8-bit register, backed by the low 4 lanes of a 128-bit `I8x16Neon`.
pub type I8x4Neon = ReducedRegister<super::I8x16Neon, U12>;
/// 4-lane unsigned 8-bit register, backed by the low 4 lanes of a 128-bit `U8x16Neon`.
pub type U8x4Neon = ReducedRegister<super::U8x16Neon, U12>;
/// 8-lane signed 8-bit register, backed by the low 8 lanes of a 128-bit `I8x16Neon`.
pub type I8x8Neon = ReducedRegister<super::I8x16Neon, U8>;
/// 8-lane unsigned 8-bit register, backed by the low 8 lanes of a 128-bit `U8x16Neon`.
pub type U8x8Neon = ReducedRegister<super::U8x16Neon, U8>;

// ===========================================================================================
// Scalar spill/rebuild helpers. NEON vector types transmute directly to/from element arrays
// (lane order == memory order on aarch64); same mechanism as `polyfills/casts.rs`.
// ===========================================================================================

#[inline(always)]
fn store_bytes(v: arch::int8x16_t) -> [i8; 16] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn store_bytes_u(v: arch::uint8x16_t) -> [u8; 16] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn from_bytes(arr: [i8; 16]) -> arch::int8x16_t {
    unsafe { core::mem::transmute(arr) }
}
#[inline(always)]
fn from_bytes_u(arr: [u8; 16]) -> arch::uint8x16_t {
    unsafe { core::mem::transmute(arr) }
}
#[inline(always)]
fn store_dwords(v: arch::int32x4_t) -> [i32; 4] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn store_dwords_u(v: arch::uint32x4_t) -> [u32; 4] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn from_dwords(arr: [i32; 4]) -> arch::int32x4_t {
    unsafe { core::mem::transmute(arr) }
}
#[inline(always)]
fn from_dwords_u(arr: [u32; 4]) -> arch::uint32x4_t {
    unsafe { core::mem::transmute(arr) }
}
#[inline(always)]
fn store_qwords(v: arch::int64x2_t) -> [i64; 2] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn store_qwords_u(v: arch::uint64x2_t) -> [u64; 2] {
    unsafe { core::mem::transmute(v) }
}
#[inline(always)]
fn from_qwords(arr: [i64; 2]) -> arch::int64x2_t {
    unsafe { core::mem::transmute(arr) }
}
#[inline(always)]
fn from_qwords_u(arr: [u64; 2]) -> arch::uint64x2_t {
    unsafe { core::mem::transmute(arr) }
}

// ===========================================================================================
// Widen/narrow ladder helpers (byte <-> word <-> dword <-> qword via `vmovl`/`vmovn`).
// ===========================================================================================

/// Sign-extend the low 4 bytes to 4x i32.
#[inline(always)]
fn widen_i8_to_i32(v: arch::int8x16_t) -> arch::int32x4_t {
    unsafe { arch::vmovl_s16(arch::vget_low_s16(arch::vmovl_s8(arch::vget_low_s8(v)))) }
}
/// Zero-extend the low 4 bytes to 4x u32.
#[inline(always)]
fn widen_u8_to_u32(v: arch::uint8x16_t) -> arch::uint32x4_t {
    unsafe { arch::vmovl_u16(arch::vget_low_u16(arch::vmovl_u8(arch::vget_low_u8(v)))) }
}

/// Widen all 16 bytes to 4x `int32x4_t` (sign-extending).
#[inline(always)]
fn widen_i8x16_to_4xi32x4(v: arch::int8x16_t) -> [arch::int32x4_t; 4] {
    unsafe {
        let lo = arch::vmovl_s8(arch::vget_low_s8(v));
        let hi = arch::vmovl_high_s8(v);
        [
            arch::vmovl_s16(arch::vget_low_s16(lo)),
            arch::vmovl_high_s16(lo),
            arch::vmovl_s16(arch::vget_low_s16(hi)),
            arch::vmovl_high_s16(hi),
        ]
    }
}
/// Widen all 16 bytes to 4x `uint32x4_t` (zero-extending).
#[inline(always)]
fn widen_u8x16_to_4xu32x4(v: arch::uint8x16_t) -> [arch::uint32x4_t; 4] {
    unsafe {
        let lo = arch::vmovl_u8(arch::vget_low_u8(v));
        let hi = arch::vmovl_high_u8(v);
        [
            arch::vmovl_u16(arch::vget_low_u16(lo)),
            arch::vmovl_high_u16(lo),
            arch::vmovl_u16(arch::vget_low_u16(hi)),
            arch::vmovl_high_u16(hi),
        ]
    }
}

/// Widen 4x i32 to 2x `int64x2_t` (sign-extending).
#[inline(always)]
fn widen_i32x4_to_2xi64x2(v: arch::int32x4_t) -> [arch::int64x2_t; 2] {
    unsafe { [arch::vmovl_s32(arch::vget_low_s32(v)), arch::vmovl_high_s32(v)] }
}
/// Widen 4x u32 to 2x `uint64x2_t` (zero-extending).
#[inline(always)]
fn widen_u32x4_to_2xu64x2(v: arch::uint32x4_t) -> [arch::uint64x2_t; 2] {
    unsafe { [arch::vmovl_u32(arch::vget_low_u32(v)), arch::vmovl_high_u32(v)] }
}

/// Truncate 4x i32 into the low 4 bytes (upper 12 bytes zero).
#[inline(always)]
fn narrow_i32_to_low_bytes(v: arch::int32x4_t) -> arch::int8x16_t {
    unsafe {
        let w = arch::vcombine_s16(arch::vmovn_s32(v), arch::vdup_n_s16(0));
        arch::vcombine_s8(arch::vmovn_s16(w), arch::vdup_n_s8(0))
    }
}
/// Truncate 4x u32 into the low 4 bytes (upper 12 bytes zero).
#[inline(always)]
fn narrow_u32_to_low_bytes(v: arch::uint32x4_t) -> arch::uint8x16_t {
    unsafe {
        let w = arch::vcombine_u16(arch::vmovn_u32(v), arch::vdup_n_u16(0));
        arch::vcombine_u8(arch::vmovn_u16(w), arch::vdup_n_u8(0))
    }
}
/// Truncate 8x i32 (two registers) into the low 8 bytes (upper 8 bytes zero).
#[inline(always)]
fn narrow_2xi32_to_low_bytes(a: arch::int32x4_t, b: arch::int32x4_t) -> arch::int8x16_t {
    unsafe {
        let w = arch::vmovn_high_s32(arch::vmovn_s32(a), b);
        arch::vcombine_s8(arch::vmovn_s16(w), arch::vdup_n_s8(0))
    }
}
/// Truncate 8x u32 (two registers) into the low 8 bytes (upper 8 bytes zero).
#[inline(always)]
fn narrow_2xu32_to_low_bytes(a: arch::uint32x4_t, b: arch::uint32x4_t) -> arch::uint8x16_t {
    unsafe {
        let w = arch::vmovn_high_u32(arch::vmovn_u32(a), b);
        arch::vcombine_u8(arch::vmovn_u16(w), arch::vdup_n_u8(0))
    }
}
/// Truncate 16x i32 (four registers) into all 16 bytes.
#[inline(always)]
fn narrow_4xi32x4_to_bytes(v: [arch::int32x4_t; 4]) -> arch::int8x16_t {
    unsafe {
        let a = arch::vmovn_high_s32(arch::vmovn_s32(v[0]), v[1]);
        let b = arch::vmovn_high_s32(arch::vmovn_s32(v[2]), v[3]);
        arch::vmovn_high_s16(arch::vmovn_s16(a), b)
    }
}
/// Truncate 16x u32 (four registers) into all 16 bytes.
#[inline(always)]
fn narrow_4xu32x4_to_bytes(v: [arch::uint32x4_t; 4]) -> arch::uint8x16_t {
    unsafe {
        let a = arch::vmovn_high_u32(arch::vmovn_u32(v[0]), v[1]);
        let b = arch::vmovn_high_u32(arch::vmovn_u32(v[2]), v[3]);
        arch::vmovn_high_u16(arch::vmovn_u16(a), b)
    }
}

/// Truncate 4x i64 (two registers) to 4x i32 (wrapping, Rust `as`).
#[inline(always)]
fn narrow_2xi64_to_i32x4(a: arch::int64x2_t, b: arch::int64x2_t) -> arch::int32x4_t {
    unsafe { arch::vmovn_high_s64(arch::vmovn_s64(a), b) }
}
/// Truncate 4x u64 (two registers) to 4x u32 (wrapping, Rust `as`).
#[inline(always)]
fn narrow_2xu64_to_u32x4(a: arch::uint64x2_t, b: arch::uint64x2_t) -> arch::uint32x4_t {
    unsafe { arch::vmovn_high_u64(arch::vmovn_u64(a), b) }
}

// ===========================================================================================
// x4 <- x2 (scalar-array) concat / extend.
// ===========================================================================================

macro_rules! impl_concat_x4_from_x2 {
    ($red:ty, $elem:ty, $ctor:ident, $store:ident) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<$elem, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<$elem, 2>>, hi: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new($ctor([
                    lo.0[0], lo.0[1], hi.0[0], hi.0[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]))
            }
            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<$elem, 2>>, Storage<ArrayRegister<$elem, 2>>) {
                let a = $store(value.0);
                (ArrayRegister([a[0], a[1]]), ArrayRegister([a[2], a[3]]))
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<$elem, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<$elem, 2>>) -> Storage<Self> {
                ReducedRegister::new($ctor([
                    value.0[0], value.0[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]))
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<$elem, 2>> {
                let a = $store(value.0);
                ArrayRegister([a[0], a[1]])
            }
        }
    };
}

impl_concat_x4_from_x2!(I8x4Neon, i8, from_bytes, store_bytes);
impl_concat_x4_from_x2!(U8x4Neon, u8, from_bytes_u, store_bytes_u);

// ===========================================================================================
// x8 <- x4 (both reduced over the same native register).
// ===========================================================================================

macro_rules! impl_concat_x8_from_x4 {
    ($x8:ty, $x4:ty, $vext:ident, $to_u32:ident, $from_u32:ident) => {
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
                // Each x4 keeps its data in the low 4 bytes; zip the low 32-bit words:
                // [lo0, hi0, ...] puts lo's 4 bytes at 0..4 and hi's at 4..8.
                unsafe {
                    ReducedRegister::new(arch::$from_u32(arch::vzip1q_u32(
                        arch::$to_u32(lo.0),
                        arch::$to_u32(hi.0),
                    )))
                }
            }
            fn split(value: Storage<Self>) -> (Storage<$x4>, Storage<$x4>) {
                (
                    ReducedRegister::new(value.0),
                    // shift the high 4 bytes (lanes 4..8) down into the low 4 bytes.
                    ReducedRegister::new(unsafe { arch::$vext::<4>(value.0, value.0) }),
                )
            }
        }
    };
}

impl_concat_x8_from_x4!(I8x8Neon, I8x4Neon, vextq_s8, vreinterpretq_u32_s8, vreinterpretq_s8_u32);
impl_concat_x8_from_x4!(U8x8Neon, U8x4Neon, vextq_u8, vreinterpretq_u32_u8, vreinterpretq_u8_u32);

// ===========================================================================================
// x16 (native) <- x8 (reduced). Both 8-lane halves live in the low 64 bits; merge the
// low d-registers via `vcombine`.
// ===========================================================================================

macro_rules! impl_concat_x16_from_x8 {
    ($native:ty, $x8:ty, $combine:ident, $get_low:ident, $vext:ident) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<$x8> for $native {
            fn concat(lo: Storage<$x8>, hi: Storage<$x8>) -> Storage<Self> {
                unsafe { arch::$combine(arch::$get_low(lo.0), arch::$get_low(hi.0)) }
            }
            fn split(value: Storage<Self>) -> (Storage<$x8>, Storage<$x8>) {
                (
                    ReducedRegister::new(value),
                    // shift the high 8 bytes down into the low 8 bytes.
                    ReducedRegister::new(unsafe { arch::$vext::<8>(value, value) }),
                )
            }
        }
    };
}

impl_concat_x16_from_x8!(super::I8x16Neon, I8x8Neon, vcombine_s8, vget_low_s8, vextq_s8);
impl_concat_x16_from_x8!(super::U8x16Neon, U8x8Neon, vcombine_u8, vget_low_u8, vextq_u8);

// ===========================================================================================
// 8 <-> 16 widen/narrow casts.
//   widen 8 -> 16 via `vmovl_s8`/`vmovl_u8` (low 8 bytes -> 8x i16).
//   narrow 16 -> 8 via `vmovn_s16`/`vmovn_u16` (the wasm backend's even-byte shuffle is a
//   truncating narrow; NEON has it as a real instruction).
//   x4: I8x4Neon <-> half16::I16x4Neon.  x8: I8x8Neon <-> I16x8Neon (native).
//   x16: I8x16Neon <-> ArrayRegister<I16x8Neon, 2>.
// ===========================================================================================

// --- x4 (both reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Neon> for super::half16::I16x4Neon {
    fn cast_from(value: Storage<I8x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vmovl_s8(arch::vget_low_s8(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Neon> for super::half16::U16x4Neon {
    fn cast_from(value: Storage<U8x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vmovl_u8(arch::vget_low_u8(value.0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::I16x4Neon> for I8x4Neon {
    fn cast_from(value: Storage<super::half16::I16x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s8(arch::vmovn_s16(value.0), arch::vdup_n_s8(0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half16::U16x4Neon> for U8x4Neon {
    fn cast_from(value: Storage<super::half16::U16x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u8(arch::vmovn_u16(value.0), arch::vdup_n_u8(0))) }
    }
}

// --- x8 (i16x8 native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Neon> for super::I16x8Neon {
    fn cast_from(value: Storage<I8x8Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_s8(arch::vget_low_s8(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Neon> for super::U16x8Neon {
    fn cast_from(value: Storage<U8x8Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_u8(arch::vget_low_u8(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I16x8Neon> for I8x8Neon {
    fn cast_from(value: Storage<super::I16x8Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s8(arch::vmovn_s16(value), arch::vdup_n_s8(0))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U16x8Neon> for U8x8Neon {
    fn cast_from(value: Storage<super::U16x8Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u8(arch::vmovn_u16(value), arch::vdup_n_u8(0))) }
    }
}

// --- x16 (i16x16 = ArrayRegister<I16x8Neon, 2>) ---
// The wasm backend implements the x16 8 <-> 16 widen/narrow casts in its half8 module; on NEON
// they are already stamped by `neon_widen_casts!` in `i8x16.rs`/`u8x16.rs` (along with the
// saturating i16x16 -> i8x16 narrow), so nothing to do here.

// ===========================================================================================
// Widen casts to 32-bit (byte -> word -> dword via the `vmovl` ladder; narrows via `vmovn`).
//   x4: I8x4Neon <-> I32x4Neon.  x8: I8x8Neon <-> ArrayRegister<I32x4Neon, 2>.
//   x2: ArrayRegister<i8,2> <-> I32x2Neon.
// ===========================================================================================

// --- x4 ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Neon> for super::I32x4Neon {
    fn cast_from(value: Storage<I8x4Neon>) -> Storage<Self> {
        widen_i8_to_i32(value.0)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Neon> for super::U32x4Neon {
    fn cast_from(value: Storage<U8x4Neon>) -> Storage<Self> {
        widen_u8_to_u32(value.0)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::I32x4Neon> for I8x4Neon {
    fn cast_from(value: Storage<super::I32x4Neon>) -> Storage<Self> {
        // low byte of each of the 4 i32 lanes -> low 4 bytes
        ReducedRegister::new(narrow_i32_to_low_bytes(value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U32x4Neon> for U8x4Neon {
    fn cast_from(value: Storage<super::U32x4Neon>) -> Storage<Self> {
        ReducedRegister::new(narrow_u32_to_low_bytes(value))
    }
}

// --- x8 (i32x8 = ArrayRegister<I32x4Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Neon> for ArrayRegister<super::I32x4Neon, 2> {
    fn cast_from(value: Storage<I8x8Neon>) -> Storage<Self> {
        unsafe {
            // widen all 8 low bytes to i16 once, then split (cheaper than the wasm
            // backend's shuffle-then-rewiden of each 4-byte group).
            let w = arch::vmovl_s8(arch::vget_low_s8(value.0));
            ArrayRegister([arch::vmovl_s16(arch::vget_low_s16(w)), arch::vmovl_high_s16(w)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Neon> for ArrayRegister<super::U32x4Neon, 2> {
    fn cast_from(value: Storage<U8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_u8(arch::vget_low_u8(value.0));
            ArrayRegister([arch::vmovl_u16(arch::vget_low_u16(w)), arch::vmovl_high_u16(w)])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4Neon, 2>> for I8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4Neon, 2>>) -> Storage<Self> {
        // native truncating narrow ladder (wasm spills to scalars here).
        ReducedRegister::new(narrow_2xi32_to_low_bytes(value.0[0], value.0[1]))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4Neon, 2>> for U8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4Neon, 2>>) -> Storage<Self> {
        ReducedRegister::new(narrow_2xu32_to_low_bytes(value.0[0], value.0[1]))
    }
}

// --- x16 widen i8 -> i32 (native 16 bytes -> ArrayRegister<I32x4Neon, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Neon> for ArrayRegister<super::I32x4Neon, 4> {
    fn cast_from(value: Storage<super::I8x16Neon>) -> Storage<Self> {
        ArrayRegister(widen_i8x16_to_4xi32x4(value))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Neon> for ArrayRegister<super::U32x4Neon, 4> {
    fn cast_from(value: Storage<super::U8x16Neon>) -> Storage<Self> {
        ArrayRegister(widen_u8x16_to_4xu32x4(value))
    }
}

// --- x16 narrow i32 -> i8 (ArrayRegister<I32x4Neon, 4> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4Neon, 4>> for super::I8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4Neon, 4>>) -> Storage<Self> {
        narrow_4xi32x4_to_bytes(value.0)
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4Neon, 4>> for super::U8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4Neon, 4>>) -> Storage<Self> {
        narrow_4xu32x4_to_bytes(value.0)
    }
}

// --- x2 (ArrayRegister<i8,2> <-> I32x2Neon reduced) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::I32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        ReducedRegister::new(from_dwords([value.0[0] as i32, value.0[1] as i32, 0, 0]))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::U32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        ReducedRegister::new(from_dwords_u([value.0[0] as u32, value.0[1] as u32, 0, 0]))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::I32x2Neon> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::I32x2Neon>) -> Storage<Self> {
        let a = store_dwords(value.0);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::U32x2Neon> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::U32x2Neon>) -> Storage<Self> {
        let a = store_dwords_u(value.0);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// ===========================================================================================
// Widen/narrow casts to 64-bit (CastRegister = numeric widen / `as`-style narrow).
//
//   widen 8 -> 64: the full `vmovl` ladder (8 -> 16 -> 32 -> 64); the wasm backend spills to
//     scalars because it has no 64-bit extends, NEON does not need to.
//   narrow 64 -> 8: `vmovn` ladder (64 -> 32 -> 16 -> 8, wrapping `as` semantics); the 2-lane
//     scalar-array endpoints keep the scalar store/rebuild.
//   x2:  ArrayRegister<i8,2> <-> I64x2Neon (native 2-lane).
//   x4:  I8x4Neon <-> ArrayRegister<I64x2Neon, 2> (the backend i64x4).
//   x8:  I8x8Neon <-> ArrayRegister<I64x2Neon, 4> (the backend i64x8).
//   x16: I8x16Neon <-> ArrayRegister<I64x2Neon, 8> (the backend i64x16).
// ===========================================================================================

// --- x2 widen i8 -> i64 (ArrayRegister<i8,2> -> I64x2Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::I64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        from_qwords([value.0[0] as i64, value.0[1] as i64])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::U64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        from_qwords_u([value.0[0] as u64, value.0[1] as u64])
    }
}

// --- x2 narrow i64 -> i8 (I64x2Neon native -> ArrayRegister<i8,2>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Neon> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::I64x2Neon>) -> Storage<Self> {
        let a = store_qwords(value);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Neon> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::U64x2Neon>) -> Storage<Self> {
        let a = store_qwords_u(value);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x4 widen i8 -> i64 (low 4 bytes -> two 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Neon> for ArrayRegister<super::I64x2Neon, 2> {
    fn cast_from(value: Storage<I8x4Neon>) -> Storage<Self> {
        ArrayRegister(widen_i32x4_to_2xi64x2(widen_i8_to_i32(value.0)))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Neon> for ArrayRegister<super::U64x2Neon, 2> {
    fn cast_from(value: Storage<U8x4Neon>) -> Storage<Self> {
        ArrayRegister(widen_u32x4_to_2xu64x2(widen_u8_to_u32(value.0)))
    }
}

// --- x4 narrow i64 -> i8 (byte 0 of each of 4 lanes -> low 4 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 2>> for I8x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 2>>) -> Storage<Self> {
        ReducedRegister::new(narrow_i32_to_low_bytes(narrow_2xi64_to_i32x4(value.0[0], value.0[1])))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 2>> for U8x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 2>>) -> Storage<Self> {
        ReducedRegister::new(narrow_u32_to_low_bytes(narrow_2xu64_to_u32x4(value.0[0], value.0[1])))
    }
}

// --- x8 widen i8 -> i64 (low 8 bytes -> four 2x i64 lanes) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Neon> for ArrayRegister<super::I64x2Neon, 4> {
    fn cast_from(value: Storage<I8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_s8(arch::vget_low_s8(value.0));
            let [a, b] = widen_i32x4_to_2xi64x2(arch::vmovl_s16(arch::vget_low_s16(w)));
            let [c, d] = widen_i32x4_to_2xi64x2(arch::vmovl_high_s16(w));
            ArrayRegister([a, b, c, d])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Neon> for ArrayRegister<super::U64x2Neon, 4> {
    fn cast_from(value: Storage<U8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_u8(arch::vget_low_u8(value.0));
            let [a, b] = widen_u32x4_to_2xu64x2(arch::vmovl_u16(arch::vget_low_u16(w)));
            let [c, d] = widen_u32x4_to_2xu64x2(arch::vmovl_high_u16(w));
            ArrayRegister([a, b, c, d])
        }
    }
}

// --- x8 narrow i64 -> i8 (byte 0 of each of 8 lanes -> low 8 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 4>> for I8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 4>>) -> Storage<Self> {
        let a = narrow_2xi64_to_i32x4(value.0[0], value.0[1]);
        let b = narrow_2xi64_to_i32x4(value.0[2], value.0[3]);
        ReducedRegister::new(narrow_2xi32_to_low_bytes(a, b))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 4>> for U8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 4>>) -> Storage<Self> {
        let a = narrow_2xu64_to_u32x4(value.0[0], value.0[1]);
        let b = narrow_2xu64_to_u32x4(value.0[2], value.0[3]);
        ReducedRegister::new(narrow_2xu32_to_low_bytes(a, b))
    }
}

// --- x16 widen i8 -> i64 (native 16 bytes -> ArrayRegister<I64x2Neon, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Neon> for ArrayRegister<super::I64x2Neon, 8> {
    fn cast_from(value: Storage<super::I8x16Neon>) -> Storage<Self> {
        let q = widen_i8x16_to_4xi32x4(value);
        let [a, b] = widen_i32x4_to_2xi64x2(q[0]);
        let [c, d] = widen_i32x4_to_2xi64x2(q[1]);
        let [e, f] = widen_i32x4_to_2xi64x2(q[2]);
        let [g, h] = widen_i32x4_to_2xi64x2(q[3]);
        ArrayRegister([a, b, c, d, e, f, g, h])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Neon> for ArrayRegister<super::U64x2Neon, 8> {
    fn cast_from(value: Storage<super::U8x16Neon>) -> Storage<Self> {
        let q = widen_u8x16_to_4xu32x4(value);
        let [a, b] = widen_u32x4_to_2xu64x2(q[0]);
        let [c, d] = widen_u32x4_to_2xu64x2(q[1]);
        let [e, f] = widen_u32x4_to_2xu64x2(q[2]);
        let [g, h] = widen_u32x4_to_2xu64x2(q[3]);
        ArrayRegister([a, b, c, d, e, f, g, h])
    }
}

// --- x16 narrow i64 -> i8 (ArrayRegister<I64x2Neon, 8> -> native 16 bytes) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Neon, 8>> for super::I8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        narrow_4xi32x4_to_bytes([
            narrow_2xi64_to_i32x4(v[0], v[1]),
            narrow_2xi64_to_i32x4(v[2], v[3]),
            narrow_2xi64_to_i32x4(v[4], v[5]),
            narrow_2xi64_to_i32x4(v[6], v[7]),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Neon, 8>> for super::U8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        narrow_4xu32x4_to_bytes([
            narrow_2xu64_to_u32x4(v[0], v[1]),
            narrow_2xu64_to_u32x4(v[2], v[3]),
            narrow_2xu64_to_u32x4(v[4], v[5]),
            narrow_2xu64_to_u32x4(v[6], v[7]),
        ])
    }
}

// ===========================================================================================
// 8 <-> f32/f64 direct casts (widen int -> i32 then native i32<->float converts; narrow via
// truncating-saturating float -> i32 then the truncating byte-narrow ladder).
//   WIDEN  i8/u8 -> f32 = widen low bytes to i32x4/u32x4 then `vcvtq_f32_*` (exact).
//   NARROW f32 -> i8/u8 = `vcvtq_s32_f32` (NEON converts truncate toward zero AND saturate,
//     matching the wasm `trunc_sat` semantics) then truncating narrow to bytes.
//   WIDEN  i8/u8 -> f64 = widen to i32 then extend to i64 and `vcvtq_f64_s64` (exact).
//   NARROW f64 -> i8/u8 = f64 -> i64 (`vcvtq_s64_f64`, truncate-saturate) -> saturating
//     narrow to i32 (together exactly f64-as-i32) then truncating narrow to bytes.
// Unsigned 8-bit values fit in positive i32, so the signed i32->float convert is exact for
// them; the float -> u8 path deliberately mirrors the wasm backend (signed i32 trunc_sat,
// then wrapping byte truncation) so all backends agree.
// ===========================================================================================

/// i32x4 -> two f64x2 (sign-extend to i64, then exact i64 -> f64 convert of i32-range values).
#[inline(always)]
fn i32x4_to_2xf64x2(v: arch::int32x4_t) -> [arch::float64x2_t; 2] {
    unsafe {
        let [a, b] = widen_i32x4_to_2xi64x2(v);
        [arch::vcvtq_f64_s64(a), arch::vcvtq_f64_s64(b)]
    }
}

/// f64x2 -> two i32 with Rust `as` (f64 -> i32) semantics: truncate-saturate to i64,
/// then saturating narrow to i32.
#[inline(always)]
fn f64x2_to_2xi32(v: arch::float64x2_t) -> [i32; 2] {
    unsafe { core::mem::transmute(arch::vqmovn_s64(arch::vcvtq_s64_f64(v))) }
}

/// Two f64x2 -> i32x4 with Rust `as` (f64 -> i32) semantics, fully in-register.
#[inline(always)]
fn f64x4_to_i32x4(a: arch::float64x2_t, b: arch::float64x2_t) -> arch::int32x4_t {
    unsafe {
        arch::vqmovn_high_s64(
            arch::vqmovn_s64(arch::vcvtq_s64_f64(a)),
            arch::vcvtq_s64_f64(b),
        )
    }
}

// --- x2 (ArrayRegister<i8,2> <-> ReducedRegister<F32x4Neon,2> = F32x2Neon) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::half::F32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        let ints = from_dwords([value.0[0] as i32, value.0[1] as i32, 0, 0]);
        unsafe { ReducedRegister::new(arch::vcvtq_f32_s32(ints)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::half::F32x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        let ints = from_dwords([value.0[0] as i32, value.0[1] as i32, 0, 0]);
        unsafe { ReducedRegister::new(arch::vcvtq_f32_s32(ints)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Neon> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::half::F32x2Neon>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::vcvtq_s32_f32(value.0) });
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::half::F32x2Neon> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::half::F32x2Neon>) -> Storage<Self> {
        let a = store_dwords(unsafe { arch::vcvtq_s32_f32(value.0) });
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x2 (ArrayRegister<i8,2> <-> F64x2Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<i8, 2>> for super::F64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<i8, 2>>) -> Storage<Self> {
        unsafe { arch::vcvtq_f64_s64(from_qwords([value.0[0] as i64, value.0[1] as i64])) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<u8, 2>> for super::F64x2Neon {
    fn cast_from(value: Storage<ArrayRegister<u8, 2>>) -> Storage<Self> {
        unsafe { arch::vcvtq_f64_s64(from_qwords([value.0[0] as i64, value.0[1] as i64])) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Neon> for ArrayRegister<i8, 2> {
    fn cast_from(value: Storage<super::F64x2Neon>) -> Storage<Self> {
        let a = f64x2_to_2xi32(value);
        ArrayRegister([a[0] as i8, a[1] as i8])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Neon> for ArrayRegister<u8, 2> {
    fn cast_from(value: Storage<super::F64x2Neon>) -> Storage<Self> {
        let a = f64x2_to_2xi32(value);
        ArrayRegister([a[0] as u8, a[1] as u8])
    }
}

// --- x4 (I8x4Neon <-> F32x4Neon native) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Neon> for super::F32x4Neon {
    fn cast_from(value: Storage<I8x4Neon>) -> Storage<Self> {
        unsafe { arch::vcvtq_f32_s32(widen_i8_to_i32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Neon> for super::F32x4Neon {
    fn cast_from(value: Storage<U8x4Neon>) -> Storage<Self> {
        unsafe { arch::vcvtq_f32_u32(widen_u8_to_u32(value.0)) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Neon> for I8x4Neon {
    fn cast_from(value: Storage<super::F32x4Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(narrow_i32_to_low_bytes(arch::vcvtq_s32_f32(value))) }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::F32x4Neon> for U8x4Neon {
    fn cast_from(value: Storage<super::F32x4Neon>) -> Storage<Self> {
        // signed i32 trunc-sat then wrapping byte truncation (matches the wasm backend).
        unsafe {
            ReducedRegister::new(arch::vreinterpretq_u8_s8(narrow_i32_to_low_bytes(arch::vcvtq_s32_f32(
                value,
            ))))
        }
    }
}

// --- x4 (I8x4Neon <-> ArrayRegister<F64x2Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x4Neon> for ArrayRegister<super::F64x2Neon, 2> {
    fn cast_from(value: Storage<I8x4Neon>) -> Storage<Self> {
        ArrayRegister(i32x4_to_2xf64x2(widen_i8_to_i32(value.0)))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x4Neon> for ArrayRegister<super::F64x2Neon, 2> {
    fn cast_from(value: Storage<U8x4Neon>) -> Storage<Self> {
        // u8 values fit in positive i32, so the signed i32 -> f64 path is exact.
        ArrayRegister(i32x4_to_2xf64x2(unsafe {
            arch::vreinterpretq_s32_u32(widen_u8_to_u32(value.0))
        }))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 2>> for I8x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 2>>) -> Storage<Self> {
        ReducedRegister::new(narrow_i32_to_low_bytes(f64x4_to_i32x4(value.0[0], value.0[1])))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 2>> for U8x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 2>>) -> Storage<Self> {
        unsafe {
            ReducedRegister::new(arch::vreinterpretq_u8_s8(narrow_i32_to_low_bytes(f64x4_to_i32x4(
                value.0[0],
                value.0[1],
            ))))
        }
    }
}

// --- x8 (I8x8Neon <-> ArrayRegister<F32x4Neon, 2>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Neon> for ArrayRegister<super::F32x4Neon, 2> {
    fn cast_from(value: Storage<I8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_s8(arch::vget_low_s8(value.0));
            ArrayRegister([
                arch::vcvtq_f32_s32(arch::vmovl_s16(arch::vget_low_s16(w))),
                arch::vcvtq_f32_s32(arch::vmovl_high_s16(w)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Neon> for ArrayRegister<super::F32x4Neon, 2> {
    fn cast_from(value: Storage<U8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_u8(arch::vget_low_u8(value.0));
            ArrayRegister([
                arch::vcvtq_f32_u32(arch::vmovl_u16(arch::vget_low_u16(w))),
                arch::vcvtq_f32_u32(arch::vmovl_high_u16(w)),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 2>> for I8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::vcvtq_s32_f32(value.0[0]);
            let hi = arch::vcvtq_s32_f32(value.0[1]);
            ReducedRegister::new(narrow_2xi32_to_low_bytes(lo, hi))
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 2>> for U8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 2>>) -> Storage<Self> {
        unsafe {
            let lo = arch::vcvtq_s32_f32(value.0[0]);
            let hi = arch::vcvtq_s32_f32(value.0[1]);
            ReducedRegister::new(arch::vreinterpretq_u8_s8(narrow_2xi32_to_low_bytes(lo, hi)))
        }
    }
}

// --- x8 (I8x8Neon <-> ArrayRegister<F64x2Neon, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<I8x8Neon> for ArrayRegister<super::F64x2Neon, 4> {
    fn cast_from(value: Storage<I8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_s8(arch::vget_low_s8(value.0));
            let [a, b] = i32x4_to_2xf64x2(arch::vmovl_s16(arch::vget_low_s16(w)));
            let [c, d] = i32x4_to_2xf64x2(arch::vmovl_high_s16(w));
            ArrayRegister([a, b, c, d])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<U8x8Neon> for ArrayRegister<super::F64x2Neon, 4> {
    fn cast_from(value: Storage<U8x8Neon>) -> Storage<Self> {
        unsafe {
            let w = arch::vmovl_u8(arch::vget_low_u8(value.0));
            let [a, b] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_u16(arch::vget_low_u16(w))));
            let [c, d] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(arch::vmovl_high_u16(w)));
            ArrayRegister([a, b, c, d])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 4>> for I8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 4>>) -> Storage<Self> {
        let a = f64x4_to_i32x4(value.0[0], value.0[1]);
        let b = f64x4_to_i32x4(value.0[2], value.0[3]);
        ReducedRegister::new(narrow_2xi32_to_low_bytes(a, b))
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 4>> for U8x8Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 4>>) -> Storage<Self> {
        let a = f64x4_to_i32x4(value.0[0], value.0[1]);
        let b = f64x4_to_i32x4(value.0[2], value.0[3]);
        unsafe { ReducedRegister::new(arch::vreinterpretq_u8_s8(narrow_2xi32_to_low_bytes(a, b))) }
    }
}

// --- x16 (I8x16Neon native <-> ArrayRegister<F32x4Neon, 4>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Neon> for ArrayRegister<super::F32x4Neon, 4> {
    fn cast_from(value: Storage<super::I8x16Neon>) -> Storage<Self> {
        unsafe {
            let q = widen_i8x16_to_4xi32x4(value);
            ArrayRegister([
                arch::vcvtq_f32_s32(q[0]),
                arch::vcvtq_f32_s32(q[1]),
                arch::vcvtq_f32_s32(q[2]),
                arch::vcvtq_f32_s32(q[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Neon> for ArrayRegister<super::F32x4Neon, 4> {
    fn cast_from(value: Storage<super::U8x16Neon>) -> Storage<Self> {
        unsafe {
            let q = widen_u8x16_to_4xu32x4(value);
            ArrayRegister([
                arch::vcvtq_f32_u32(q[0]),
                arch::vcvtq_f32_u32(q[1]),
                arch::vcvtq_f32_u32(q[2]),
                arch::vcvtq_f32_u32(q[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 4>> for super::I8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            narrow_4xi32x4_to_bytes([
                arch::vcvtq_s32_f32(v[0]),
                arch::vcvtq_s32_f32(v[1]),
                arch::vcvtq_s32_f32(v[2]),
                arch::vcvtq_s32_f32(v[3]),
            ])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F32x4Neon, 4>> for super::U8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F32x4Neon, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::vreinterpretq_u8_s8(narrow_4xi32x4_to_bytes([
                arch::vcvtq_s32_f32(v[0]),
                arch::vcvtq_s32_f32(v[1]),
                arch::vcvtq_s32_f32(v[2]),
                arch::vcvtq_s32_f32(v[3]),
            ]))
        }
    }
}

// --- x16 (I8x16Neon native <-> ArrayRegister<F64x2Neon, 8>) ---
#[thermite_macros::inline_always]
impl CastRegister<super::I8x16Neon> for ArrayRegister<super::F64x2Neon, 8> {
    fn cast_from(value: Storage<super::I8x16Neon>) -> Storage<Self> {
        let q = widen_i8x16_to_4xi32x4(value);
        let [a, b] = i32x4_to_2xf64x2(q[0]);
        let [c, d] = i32x4_to_2xf64x2(q[1]);
        let [e, f] = i32x4_to_2xf64x2(q[2]);
        let [g, h] = i32x4_to_2xf64x2(q[3]);
        ArrayRegister([a, b, c, d, e, f, g, h])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<super::U8x16Neon> for ArrayRegister<super::F64x2Neon, 8> {
    fn cast_from(value: Storage<super::U8x16Neon>) -> Storage<Self> {
        unsafe {
            let q = widen_u8x16_to_4xu32x4(value);
            let [a, b] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(q[0]));
            let [c, d] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(q[1]));
            let [e, f] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(q[2]));
            let [g, h] = i32x4_to_2xf64x2(arch::vreinterpretq_s32_u32(q[3]));
            ArrayRegister([a, b, c, d, e, f, g, h])
        }
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 8>> for super::I8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        narrow_4xi32x4_to_bytes([
            f64x4_to_i32x4(v[0], v[1]),
            f64x4_to_i32x4(v[2], v[3]),
            f64x4_to_i32x4(v[4], v[5]),
            f64x4_to_i32x4(v[6], v[7]),
        ])
    }
}
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 8>> for super::U8x16Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 8>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            arch::vreinterpretq_u8_s8(narrow_4xi32x4_to_bytes([
                f64x4_to_i32x4(v[0], v[1]),
                f64x4_to_i32x4(v[2], v[3]),
                f64x4_to_i32x4(v[4], v[5]),
                f64x4_to_i32x4(v[6], v[7]),
            ]))
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

#[inline(always)]
fn bool_to_u8_mask(b: bool) -> u8 {
    if b { !0 } else { 0 }
}

macro_rules! impl_mask_concat_x4_from_bool2 {
    ($red:ty, $ctor:ident, $store:ident, $mask:ident) => {
        #[thermite_macros::inline_always]
        impl ConcatRegister<ArrayRegister<bool, 2>> for $red {
            fn concat(lo: Storage<ArrayRegister<bool, 2>>, hi: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new($ctor([
                    $mask(lo.0[0]),
                    $mask(lo.0[1]),
                    $mask(hi.0[0]),
                    $mask(hi.0[1]),
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
                ]))
            }
            fn split(value: Storage<Self>) -> (Storage<ArrayRegister<bool, 2>>, Storage<ArrayRegister<bool, 2>>) {
                let a = $store(value.0);
                (
                    ArrayRegister([a[0] != 0, a[1] != 0]),
                    ArrayRegister([a[2] != 0, a[3] != 0]),
                )
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<ArrayRegister<bool, 2>> for $red {
            fn extend(value: Storage<ArrayRegister<bool, 2>>) -> Storage<Self> {
                ReducedRegister::new($ctor([
                    $mask(value.0[0]),
                    $mask(value.0[1]),
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
                ]))
            }
            fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<bool, 2>> {
                let a = $store(value.0);
                ArrayRegister([a[0] != 0, a[1] != 0])
            }
        }
    };
}

impl_mask_concat_x4_from_bool2!(I8x4Neon, from_bytes, store_bytes, bool_to_i8_mask);
impl_mask_concat_x4_from_bool2!(U8x4Neon, from_bytes_u, store_bytes_u, bool_to_u8_mask);

// ===========================================================================================
// Cross-type gather/scatter index markers (scalar fallback).
// `Neon` does not implement `crate::simd::Simd` yet, so the index register types are spelled
// concretely (mirroring the wasm backend's Simd choices: u64x4 = ArrayRegister<U64x2, 2>,
// u32x8 = ArrayRegister<U32x4, 2>, u64x8 = ArrayRegister<U64x2, 4>); switch these to
// `<super::super::Neon as crate::simd::Simd>::u32x4`-style paths once that impl lands.
// ===========================================================================================

macro_rules! impl_indexable8 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable8!(super::U32x4Neon => I8x4Neon, U8x4Neon);
impl_indexable8!(ArrayRegister<super::U64x2Neon, 2> => I8x4Neon, U8x4Neon);
impl_indexable8!(ArrayRegister<super::U32x4Neon, 2> => I8x8Neon, U8x8Neon);
impl_indexable8!(ArrayRegister<super::U64x2Neon, 4> => I8x8Neon, U8x8Neon);
