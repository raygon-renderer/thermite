use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub mod i16x8;
pub mod u16x8;

pub mod i8x16;
pub mod u8x16;

pub mod half;
pub mod half16;
pub mod packed; // PackedFloatRegister (16-bit float) generic-default impls for native u16 regs

pub use f32x4::F32x4V2;
pub use i32x4::I32x4V2;
pub use u32x4::U32x4V2;

pub use f64x2::F64x2V2;
pub use i64x2::I64x2V2;
pub use u64x2::U64x2V2;

pub use i16x8::I16x8V2;
pub use u16x8::U16x8V2;

pub use i8x16::I8x16V2;
pub use u8x16::U8x16V2;

impl_newregister!(
    F32x4V2, I32x4V2, U32x4V2, F64x2V2, I64x2V2, U64x2V2, I16x8V2, U16x8V2, I8x16V2, U8x16V2
);

use crate::{
    backend::scalar::Scalar,
    element::FindUSize,
    isa::InstructionSet,
    register::{
        IndexableRegister, Storage,
        array::ArrayRegister,
        reduced::{HalfRegister2, ReducedRegister},
    },
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A, SimdExperimental},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86V2;

impl HasIsa for X86V2 {
    const ISA: InstructionSet = InstructionSet::X86V2;
}

impl NativeIsa for X86V2 {
    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes

    unsafe fn disable_denormals() -> Result<bool, crate::simd::UnsupportedError> {
        unsafe { Ok(arch::disable_denormals()) }
    }

    #[allow(clippy::unit_arg)]
    unsafe fn enable_denormals() -> Result<(), crate::simd::UnsupportedError> {
        unsafe { Ok(arch::enable_denormals()) }
    }
}

impl NativeSimd for X86V2 {
    type f32xN = F32x4V2;
    type i32xN = I32x4V2;
    type u32xN = U32x4V2;

    type f64xN = F64x2V2;
    type i64xN = I64x2V2;
    type u64xN = U64x2V2;
}

// Scatter/Gather is not available in x86v2, so we must use fallback impls
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable!(<X86V2 as Simd>::u32x2 => F64x2V2, I64x2V2, U64x2V2);
impl_indexable!(<X86V2 as Simd>::u64x2 => F64x2V2, I64x2V2, U64x2V2);
impl_indexable!(<X86V2 as Simd>::u32x4 => F32x4V2, I32x4V2, U32x4V2, <X86V2 as Simd>::u64x4, <X86V2 as Simd>::i64x4, <X86V2 as Simd>::f64x4);
impl_indexable!(<X86V2 as Simd>::u64x4 => F32x4V2, I32x4V2, U32x4V2);

// 16-bit gather/scatter falls back to scalar (no hardware support on x86v2): marker impls only.
// Native 8-lane (i16x8/u16x8) indexed by same-width u16 and by the 8-lane u32/u64 index types.
impl_indexable!(U16x8V2 => I16x8V2, U16x8V2);
impl_indexable!(<X86V2 as Simd>::u32x8 => I16x8V2, U16x8V2);
impl_indexable!(<X86V2 as Simd>::u64x8 => I16x8V2, U16x8V2);
// 2-lane (ArrayRegister<_,2>) indexed by the 2-lane u32/u64 index types (the reduced/native
// u32x2 and U64x2V2). The 16-lane array forms inherit u32x16 indexing from the array blanket,
// but the 16-lane u64 index (ArrayRegister<U64x2V2, 8>) has a mismatched chunking, so mark it.
impl_indexable!(<X86V2 as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>);
impl_indexable!(<X86V2 as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>);
impl_indexable!(<X86V2 as Simd>::u64x16 => ArrayRegister<I16x8V2, 2>, ArrayRegister<U16x8V2, 2>);

// Native 8-bit (u8x16/i8x16) gather/scatter falls back to scalar; the native slot only needs
// same-width self-indexing by u8x16.
impl_indexable!(U8x16V2 => I8x16V2, U8x16V2);

impl Simd for X86V2 {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = half::F32x2V2;
    type i32x2 = half::I32x2V2;
    type u32x2 = half::U32x2V2;

    type f32x4 = F32x4V2;
    type i32x4 = I32x4V2;
    type u32x4 = U32x4V2;

    type f64x2 = F64x2V2;
    type i64x2 = I64x2V2;
    type u64x2 = U64x2V2;

    type f32x8 = ArrayRegister<F32x4V2, 2>;
    type i32x8 = ArrayRegister<I32x4V2, 2>;
    type u32x8 = ArrayRegister<U32x4V2, 2>;

    type f64x4 = ArrayRegister<F64x2V2, 2>;
    type i64x4 = ArrayRegister<I64x2V2, 2>;
    type u64x4 = ArrayRegister<U64x2V2, 2>;

    type f64x8 = ArrayRegister<F64x2V2, 4>;
    type i64x8 = ArrayRegister<I64x2V2, 4>;
    type u64x8 = ArrayRegister<U64x2V2, 4>;

    type f32x16 = ArrayRegister<F32x4V2, 4>;
    type i32x16 = ArrayRegister<I32x4V2, 4>;
    type u32x16 = ArrayRegister<U32x4V2, 4>;

    type f64x16 = ArrayRegister<F64x2V2, 8>;
    type i64x16 = ArrayRegister<I64x2V2, 8>;
    type u64x16 = ArrayRegister<U64x2V2, 8>;
}

impl Simd3 for X86V2 {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

impl SimdExperimental for X86V2 {
    type Native16Width = generic_array::typenum::U8;

    type i16xN = I16x8V2;
    type u16xN = U16x8V2;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4V2;
    type u16x4 = half16::U16x4V2;

    type i16x8 = I16x8V2;
    type u16x8 = U16x8V2;

    type i16x16 = ArrayRegister<I16x8V2, 2>;
    type u16x16 = ArrayRegister<U16x8V2, 2>;

    type Native8Width = generic_array::typenum::U16;
    type i8xN = I8x16V2;
    type u8xN = U8x16V2;

    type i8x16 = I8x16V2;
    type u8x16 = U8x16V2;
}

impl_concat_bool_register2!(f32, half::F32x2V2);
impl_concat_bool_register2!(u32, half::U32x2V2);
impl_concat_bool_register2!(i32, half::I32x2V2);

impl_concat_bool_register2!(f64, F64x2V2);
impl_concat_bool_register2!(u64, U64x2V2);
impl_concat_bool_register2!(i64, I64x2V2);

impl_bit_casts! {
    F64x2V2 as I64x2V2 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V2 as U64x2V2 => _mm_castpd_si128, // f64x2 -> u64x2
    I64x2V2 as F64x2V2 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V2 as F64x2V2 => _mm_castsi128_pd, // u64x2 -> f64x2

    F32x4V2 as I32x4V2 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V2 as U32x4V2 => _mm_castps_si128, // f32x4 -> u32x4
    I32x4V2 as F32x4V2 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V2 as F32x4V2 => _mm_castsi128_ps, // u32x4 -> f32x4

    // integer casts use the same underlying storage, so identity casts
    U32x4V2 as I32x4V2 => identity, // u32x4 -> i32x4
    I32x4V2 as U32x4V2 => identity, // i32x4 -> u32x4
    U64x2V2 as I64x2V2 => identity, // u64x2 -> i64x2
    I64x2V2 as U64x2V2 => identity, // i64x2 -> u64x2

    // all the identity casts to self
    U32x4V2 as U32x4V2 => identity, // u32x4 -> u32x4
    I32x4V2 as I32x4V2 => identity, // i32x4 -> i32x4
    I64x2V2 as I64x2V2 => identity, // i64x2 -> i64x2
    F32x4V2 as F32x4V2 => identity, // f32x4 -> f32x4
    F64x2V2 as F64x2V2 => identity, // f64x2 -> f64x2
    U64x2V2 as U64x2V2 => identity, // u64x2 -> u64x2

    // 16-bit integer casts (same 128-bit storage)
    U16x8V2 as I16x8V2 => identity, // u16x8 -> i16x8
    I16x8V2 as U16x8V2 => identity, // i16x8 -> u16x8
    I16x8V2 as I16x8V2 => identity, // i16x8 -> i16x8
    U16x8V2 as U16x8V2 => identity, // u16x8 -> u16x8

    // 8-bit integer casts (same 128-bit storage)
    U8x16V2 as I8x16V2 => identity, // u8x16 -> i8x16
    I8x16V2 as U8x16V2 => identity, // i8x16 -> u8x16
    I8x16V2 as I8x16V2 => identity, // i8x16 -> i8x16
    U8x16V2 as U8x16V2 => identity, // u8x16 -> u8x16
}

impl_type_casts! {
    // self casts
    F32x4V2 as F32x4V2 => identity, // f32x4 -> f32x4
    F64x2V2 as F64x2V2 => identity, // f64x2 -> f64x2
    I32x4V2 as I32x4V2 => identity, // i32x4 -> i32x4
    I64x2V2 as I64x2V2 => identity, // i64x2 -> i64x2
    U32x4V2 as U32x4V2 => identity, // u32x4 -> u32x4
    U64x2V2 as U64x2V2 => identity, // u64x2 -> u64x2

    // f32x4 casts (truncate toward zero - `cast` is "like `as`")
    F32x4V2 as I32x4V2 => _mm_cvttps_epi32, // f32x4 -> i32x4
    F32x4V2 as U32x4V2 => _mm_cvtps_epu32x_v2, // f32x4 -> u32x4
    I32x4V2 as F32x4V2 => _mm_cvtepi32_ps, // i32x4 -> f32x4
    U32x4V2 as F32x4V2 => _mm_cvtepu32_psx_v2, // u32x4 -> f32x4

    // f64x2 casts
    F64x2V2 as I64x2V2 => _mm_cvtpd_epi64x_v2 | _mm_cvtpd_epi64x_limited_v1, // f64x2 -> i64x2
    F64x2V2 as U64x2V2 => _mm_cvtpd_epu64x_limited_v1, // f64x2 -> u64x2
    I64x2V2 as F64x2V2 => _mm_cvtepi64_pdx_v2 | _mm_cvtepi64_pdx_limited_v1, // i64x2 -> f64x2
    U64x2V2 as F64x2V2 => _mm_cvtepu64_pdx_v2 | _mm_cvtepu64_pdx_limited_v1, // u64x2 -> f64x2

    // for integer casts we don't do anything, basically bit casting, same as Rust
    I32x4V2 as U32x4V2 => identity, // i32x4 -> u32x4
    U32x4V2 as I32x4V2 => identity, // u32x4 -> i32x4
    I64x2V2 as U64x2V2 => identity, // i64x2 -> u64x2
    U64x2V2 as I64x2V2 => identity, // u64x2 -> i64x2

    // 16-bit
    I16x8V2 as I16x8V2 => identity, // i16x8 -> i16x8
    U16x8V2 as U16x8V2 => identity, // u16x8 -> u16x8
    I16x8V2 as U16x8V2 => identity, // i16x8 -> u16x8
    U16x8V2 as I16x8V2 => identity, // u16x8 -> i16x8

    // 8-bit
    I8x16V2 as I8x16V2 => identity, // i8x16 -> i8x16
    U8x16V2 as U8x16V2 => identity, // u8x16 -> u8x16
    I8x16V2 as U8x16V2 => identity, // i8x16 -> u8x16
    U8x16V2 as I8x16V2 => identity, // u8x16 -> i8x16
}

impl_mask_casts! {
    // self casts
    I32x4V2 as I32x4V2 => identity, // i32x4 -> i32x4
    U32x4V2 as U32x4V2 => identity, // u32x4 -> u32x4
    I64x2V2 as I64x2V2 => identity, // i64x2 -> i64x2
    U64x2V2 as U64x2V2 => identity, // u64x2 -> u64x2
    F32x4V2 as F32x4V2 => identity, // f32x4 -> f32x4
    F64x2V2 as F64x2V2 => identity, // f64x2 -> f64x2

    // same-size integer casts
    I32x4V2 as U32x4V2 => identity, // i32x4 -> u32x4
    U32x4V2 as I32x4V2 => identity, // u32x4 -> i32x4
    I64x2V2 as U64x2V2 => identity, // i64x2 -> u64x2
    U64x2V2 as I64x2V2 => identity, // u64x2 -> i64x2
    I16x8V2 as I16x8V2 => identity, // i16x8 -> i16x8
    U16x8V2 as U16x8V2 => identity, // u16x8 -> u16x8
    I16x8V2 as U16x8V2 => identity, // i16x8 -> u16x8
    U16x8V2 as I16x8V2 => identity, // u16x8 -> i16x8
    I8x16V2 as I8x16V2 => identity, // i8x16 -> i8x16
    U8x16V2 as U8x16V2 => identity, // u8x16 -> u8x16
    I8x16V2 as U8x16V2 => identity, // i8x16 -> u8x16
    U8x16V2 as I8x16V2 => identity, // u8x16 -> i8x16

    // same-size float/integer casts
    I32x4V2 as F32x4V2 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V2 as F32x4V2 => _mm_castsi128_ps, // u32x4 -> f32x4
    I64x2V2 as F64x2V2 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V2 as F64x2V2 => _mm_castsi128_pd, // u64x2 -> f64x2
    F32x4V2 as I32x4V2 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V2 as U32x4V2 => _mm_castps_si128, // f32x4 -> u32x4
    F64x2V2 as I64x2V2 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V2 as U64x2V2 => _mm_castpd_si128, // f64x2 -> u64x2
}

macro_rules! impl_extend_same {
    ($($ty:ty),* $(,)?) => {$( impl crate::register::ExtendRegister<$ty> for $ty {
        #[inline(always)]
        fn extend(value: Storage<$ty>) -> Storage<Self> {
            value
        }

        #[inline(always)]
        fn narrow(value: Storage<Self>) -> Storage<$ty> {
            value
        }
    } )*};
}

impl_extend_same!(
    F32x4V2, I32x4V2, U32x4V2, F64x2V2, I64x2V2, U64x2V2, I16x8V2, U16x8V2, I8x16V2, U8x16V2
);
