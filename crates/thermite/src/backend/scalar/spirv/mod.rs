pub mod float;
pub mod mask;
pub mod signed;
pub mod unsigned;

use crate::backend::spirv::arch::{self as arch, glsl};
use crate::{
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastMaskRegister, CoreRegister, Element, InterleaveRegister, MaskElement, MaskRegister,
        Storage, ZeroUpper,
    },
    simd::HasIsa,
};
use generic_array::{GenericArray, typenum};

impl HasIsa for super::Scalar {
    const ISA: InstructionSet = InstructionSet::SPIRV;
}

// ----------------------------------------------------------------------------
// Shared SPIRV bit-manipulation helpers (pub(super) so submodules can import)
// ----------------------------------------------------------------------------

// Software byte-swap — SPIRV has no native bswap instruction.
#[inline(always)]
pub(super) fn spirv_swap_bytes_u32(x: u32) -> u32 {
    ((x & 0xFF000000u32) >> 24) | ((x & 0x00FF0000u32) >> 8) | ((x & 0x0000FF00u32) << 8) | ((x & 0x000000FFu32) << 24)
}

#[inline(always)]
pub(super) fn spirv_swap_bytes_u64(x: u64) -> u64 {
    ((x & 0xFF00000000000000u64) >> 56)
        | ((x & 0x00FF000000000000u64) >> 40)
        | ((x & 0x0000FF0000000000u64) >> 24)
        | ((x & 0x000000FF00000000u64) >> 8)
        | ((x & 0x00000000FF000000u64) << 8)
        | ((x & 0x0000000000FF0000u64) << 24)
        | ((x & 0x000000000000FF00u64) << 40)
        | ((x & 0x00000000000000FFu64) << 56)
}

// OpBitCount result type must be the same width as the input per SPIRV spec.
// For u32 inputs this is straightforward; for u64 we split into two u32 halves.
#[inline(always)]
pub(super) unsafe fn spirv_count_ones_u32(x: u32) -> u32 {
    unsafe { arch::op_opbitcount::<u32>(x) }
}

#[inline(always)]
pub(super) unsafe fn spirv_count_ones_u64(x: u64) -> u64 {
    let lo = unsafe { arch::op_opbitcount::<u32>(x as u32) } as u64;
    let hi = unsafe { arch::op_opbitcount::<u32>((x >> 32) as u32) } as u64;
    lo + hi
}

// GLSLstd450 FindUMsb is only defined for 32-bit types.
// For leading_zeros on u32: use the wrapping-subtract trick so zero -> 32.
#[inline(always)]
pub(super) unsafe fn spirv_leading_zeros_u32(x: u32) -> u32 {
    let msb = unsafe { arch::glsl_op1::<u32, u32, { glsl::FIND_U_MSB }, false>(x) };
    // FindUMsb(0) = 0xFFFFFFFF; 31u32.wrapping_sub(0xFFFFFFFF) = 32 ✓
    31u32.wrapping_sub(msb)
}

// For u64 we split into two u32 halves and combine with OpSelect (branchless).
#[inline(always)]
pub(super) unsafe fn spirv_leading_zeros_u64(x: u64) -> u64 {
    let hi = (x >> 32) as u32;
    let hi_lz = unsafe { spirv_leading_zeros_u32(hi) };
    let lo_lz = unsafe { spirv_leading_zeros_u32(x as u32) };
    // if hi != 0: result = hi_lz, else result = 32 + lo_lz
    unsafe { arch::op_opselect::<u32, bool>(hi != 0, hi_lz, 32u32.wrapping_add(lo_lz)) as u64 }
}

#[inline(always)]
pub(super) unsafe fn spirv_trailing_zeros_u32(x: u32) -> u32 {
    // trailing_zeros(x) = leading_zeros(reverse_bits(x))
    let reversed = unsafe { arch::op_opbitreverse::<u32>(x) };
    unsafe { spirv_leading_zeros_u32(reversed) }
}

#[inline(always)]
pub(super) unsafe fn spirv_trailing_zeros_u64(x: u64) -> u64 {
    let lo = x as u32;
    let lo_tz = unsafe { spirv_trailing_zeros_u32(lo) };
    let hi_tz = unsafe { spirv_trailing_zeros_u32((x >> 32) as u32) };
    // if lo != 0: result = lo_tz, else result = 32 + hi_tz
    unsafe { arch::op_opselect::<u32, bool>(lo != 0, lo_tz, 32u32.wrapping_add(hi_tz)) as u64 }
}

// ----------------------------------------------------------------------------
// Bool mask — CoreRegister + MaskRegister (uses OpSelect instead of
// core::hint::select_unpredictable)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Bit-cast and numeric cast impls for SPIRV scalar types
// ----------------------------------------------------------------------------

macro_rules! impl_spirv_bitcast {
    ($($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::BitCastRegister<$from> for $to {
            fn from_bits(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opbitcast::<$to, $from>(value) }
            }
        }
    )*};
}

impl_spirv_bitcast! {
    f32 => i32, f32 => u32,
    i32 => f32, i32 => u32,
    u32 => f32, u32 => i32,
    f64 => i64, f64 => u64,
    i64 => f64, i64 => u64,
    u64 => f64, u64 => i64,
    // Same-type identity bitcasts
    f32 => f32, i32 => i32, u32 => u32,
    f64 => f64, i64 => i64, u64 => u64,
}

macro_rules! impl_spirv_cast {
    // float -> signed int
    (f->i: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opconvertftos::<$to, $from>(value) }
            }
        }
    )*};
    // float -> unsigned int
    (f->u: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opconvertftou::<$to, $from>(value) }
            }
        }
    )*};
    // signed int -> float
    (i->f: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opconvertstof::<$to, $from>(value) }
            }
        }
    )*};
    // unsigned int -> float
    (u->f: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opconvertutof::<$to, $from>(value) }
            }
        }
    )*};
    // float -> float (width-converting)
    (f->f: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opfconvert::<$to, $from>(value) }
            }
        }
    )*};
    // signed -> signed (width-converting)
    (i->i: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opsconvert::<$to, $from>(value) }
            }
        }
    )*};
    // unsigned -> unsigned (width-converting)
    (u->u: $($from:ty => $to:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$from> for $to {
            fn cast_from(value: Storage<$from>) -> Storage<Self> {
                unsafe { arch::op_opuconvert::<$to, $from>(value) }
            }
        }
    )*};
    // same type — identity
    (id: $($ty:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl crate::register::CastRegister<$ty> for $ty {
            fn cast_from(value: Storage<$ty>) -> Storage<Self> { value }
        }
    )*};
}

impl_spirv_cast!(id:  f32, i32, u32, f64, i64, u64);
impl_spirv_cast!(f->i: f32 => i32, f64 => i64, f32 => i64, f64 => i32);
impl_spirv_cast!(f->u: f32 => u32, f64 => u64, f32 => u64, f64 => u32);
impl_spirv_cast!(i->f: i32 => f32, i64 => f64, i32 => f64, i64 => f32);
impl_spirv_cast!(u->f: u32 => f32, u64 => f64, u32 => f64, u64 => f32);
impl_spirv_cast!(f->f: f32 => f64, f64 => f32);
impl_spirv_cast!(i->i: i32 => i64, i64 => i32);
impl_spirv_cast!(u->u: u32 => u64, u64 => u32);
// Cross-sign same-width: OpSConvert/OpUConvert work for sign changes too
impl_spirv_cast!(i->i: i32 => u32, i64 => u64);
impl_spirv_cast!(u->u: u32 => i32, u64 => i64);
