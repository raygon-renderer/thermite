pub mod float;
pub mod mask;
pub mod signed;
pub mod unsigned;

use crate::{
    isa::InstructionSet,
    register::{BitCastRegister, CastRegister, Storage},
    simd::HasIsa,
};

impl HasIsa for super::Scalar {
    const ISA: InstructionSet = InstructionSet::Scalar;
}

macro_rules! impl_easy_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            impl BitCastRegister<$from> for $to {
                #[inline(always)]
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { core::mem::transmute(value) }
                }
            }

            impl CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { value as _ } // built-in cast
                }
            }
        )+
    )*};
}

macro_rules! impl_nontrivial_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            impl CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { value as _ } // built-in cast
                }
            }
        )+
    )*};
}

impl_easy_casts! {
    i8  as (i8,  u8),
    u8  as (u8,  i8),
    i16 as (i16, u16),
    u16 as (u16, i16),

    f32 as (f32, i32, u32),
    i32 as (f32, u32, i32),
    u32 as (f32, i32, u32),

    f64 as (f64, i64, u64),
    i64 as (f64, u64, i64),
    u64 as (f64, i64, u64),
}

// all different-sized casts
impl_nontrivial_casts! {
    //     (f32, f64, i8, i16, i32, i64, u8, u16, u32, u64)
    i8  as (f32, f64,     i16, i32, i64,     u16, u32, u64),
    u8  as (f32, f64,     i16, i32, i64,     u16, u32, u64),
    u16 as (f32, f64, i8,      i32, i64, u8,      u32, u64),
    i16 as (f32, f64, i8,      i32, i64, u8,      u32, u64),
    f32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    i32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    u32 as (     f64, i8, i16,      i64, u8, u16,      u64),
    f64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
    i64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
    u64 as (f32,      i8, i16, i32,      u8, u16, u32     ),
}
