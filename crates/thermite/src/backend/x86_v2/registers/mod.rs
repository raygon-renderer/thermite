use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub mod half;

pub use f32x4::F32x4V2;
pub use i32x4::I32x4V2;
pub use u32x4::U32x4V2;

pub use f64x2::F64x2V2;
pub use i64x2::I64x2V2;
pub use u64x2::U64x2V2;

use crate::{
    backend::scalar::Scalar,
    element::FindUSize,
    isa::InstructionSet,
    register::{IndexableRegister, Storage, dp::DoublePumpRegister, reduced::HalfRegister2},
    simd::{NativeIsa, NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86V2;

impl NativeIsa for X86V2 {
    const ISA: InstructionSet = InstructionSet::X86V2;

    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes
}

impl NativeSimd for X86V2 {
    type f32xN = F32x4V2;
    type i32xN = I32x4V2;
    type u32xN = U32x4V2;

    type f64xN = F64x2V2;
    type i64xN = I64x2V2;
    type u64xN = U64x2V2;
}

type F64x4V2 = DoublePumpRegister<F64x2V2>;
type I64x4V2 = DoublePumpRegister<I64x2V2>;
type U64x4V2 = DoublePumpRegister<U64x2V2>;

// Scatter/Gather is not available in x86v2, so we must use fallback impls
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable!(<X86V2 as Simd>::u32x2 => F64x2V2, I64x2V2, U64x2V2);
impl_indexable!(<X86V2 as Simd>::u64x2 => F64x2V2, I64x2V2, U64x2V2);
impl_indexable!(<X86V2 as Simd>::u32x4 => <X86V2 as Simd>::f32x4, <X86V2 as Simd>::i32x4, <X86V2 as Simd>::u32x4, <X86V2 as Simd>::u64x4, <X86V2 as Simd>::i64x4, <X86V2 as Simd>::f64x4);
impl_indexable!(<X86V2 as Simd>::u64x4 => <X86V2 as Simd>::f32x4, <X86V2 as Simd>::i32x4, <X86V2 as Simd>::u32x4);

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

    type f32x8 = DoublePumpRegister<Self::f32x4>;
    type i32x8 = DoublePumpRegister<Self::i32x4>;
    type u32x8 = DoublePumpRegister<Self::u32x4>;

    type f64x4 = F64x4V2;
    type i64x4 = I64x4V2;
    type u64x4 = U64x4V2;

    type f64x8 = DoublePumpRegister<Self::f64x4>;
    type i64x8 = DoublePumpRegister<Self::i64x4>;
    type u64x8 = DoublePumpRegister<Self::u64x4>;

    type f32x16 = DoublePumpRegister<Self::f32x8>;
    type i32x16 = DoublePumpRegister<Self::i32x8>;
    type u32x16 = DoublePumpRegister<Self::u32x8>;

    type f64x16 = DoublePumpRegister<Self::f64x8>;
    type i64x16 = DoublePumpRegister<Self::i64x8>;
    type u64x16 = DoublePumpRegister<Self::u64x8>;
}

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
}

impl_type_casts! {
    // self casts
    F32x4V2 as F32x4V2 => identity, // f32x4 -> f32x4
    F64x2V2 as F64x2V2 => identity, // f64x2 -> f64x2
    I32x4V2 as I32x4V2 => identity, // i32x4 -> i32x4
    I64x2V2 as I64x2V2 => identity, // i64x2 -> i64x2
    U32x4V2 as U32x4V2 => identity, // u32x4 -> u32x4
    U64x2V2 as U64x2V2 => identity, // u64x2 -> u64x2

    // f32x4 casts
    F32x4V2 as I32x4V2 => _mm_cvtps_epi32, // f32x4 -> i32x4
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

impl_extend_same!(F32x4V2, I32x4V2, U32x4V2, F64x2V2, I64x2V2, U64x2V2);
