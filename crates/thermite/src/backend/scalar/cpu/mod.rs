pub mod float;
pub mod mask;
pub mod signed;
pub mod unsigned;

use crate::{
    isa::InstructionSet,
    register::{BitCastRegister, CastRegister, SaturatingCastRegister, Storage},
    simd::HasIsa,
};

impl HasIsa for super::Scalar {
    type Native = Self;

    const ISA: InstructionSet = InstructionSet::Scalar;
}

macro_rules! impl_easy_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            #[thermite_macros::inline_always]
            impl BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { core::mem::transmute(value) }
                }
            }

            #[thermite_macros::inline_always]
            impl CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    value as _ // built-in cast
                }
            }
        )+
    )*};
}

macro_rules! impl_nontrivial_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            #[thermite_macros::inline_always]
            impl CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    value as _ // built-in cast
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

// Reference oracle for `SaturatingCastRegister`: clamp the source value into the
// destination element range, then convert. Narrowing, same-signedness only.
// For unsigned destinations `<$to>::MIN` is 0, so only the high end clamps.
macro_rules! impl_saturating_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            #[thermite_macros::inline_always]
            impl SaturatingCastRegister<$from> for $to {
                fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                    value.clamp(<$to>::MIN as $from, <$to>::MAX as $from) as $to
                }
            }
        )+
    )*};
}

impl_saturating_casts! {
    // signed narrowing (incl. skip-level pairs)
    i64 as (i32, i16, i8),
    i32 as (i16, i8),
    i16 as (i8),
    // unsigned narrowing (incl. skip-level pairs)
    u64 as (u32, u16, u8),
    u32 as (u16, u8),
    u16 as (u8),
}

// Float -> int saturating casts: Rust `as` already has exactly the saturating
// semantics (NaN -> 0, out-of-range clamps to MIN/MAX), so the plain cast is
// the reference implementation the SIMD backends must match.
macro_rules! impl_saturating_float_casts {
    ($($from:ty as ($($to:ty),+)),* $(,)?) => {$(
        $(
            #[thermite_macros::inline_always]
            impl SaturatingCastRegister<$from> for $to {
                fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                    value as $to
                }
            }
        )+
    )*};
}

impl_saturating_float_casts! {
    // same-width
    f32 as (i32, u32),
    f64 as (i64, u64),
    // cross-width (`as` saturates at the DESTINATION's range regardless of width)
    f32 as (i8, i16, i64, u8, u16, u64),
    f64 as (i8, i16, i32, u8, u16, u32),
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
