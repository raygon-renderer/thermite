use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub use f32x4::F32x4V2;
pub use i32x4::I32x4V2;
pub use u32x4::U32x4V2;

pub use f64x2::F64x2V2;
pub use i64x2::I64x2V2;
pub use u64x2::U64x2V2;

use crate::{
    register::{Storage, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

pub struct X86V2;

impl NativeSimd for X86V2 {
    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;

    type f32xN = F32x4V2;
    type i32xN = I32x4V2;
    type u32xN = U32x4V2;

    type f64xN = F64x2V2;
    type i64xN = I64x2V2;
    type u64xN = U64x2V2;
}

impl Simd for X86V2 {
    type f32x4 = F32x4V2;
    type i32x4 = I32x4V2;
    type u32x4 = U32x4V2;

    type f64x2 = F64x2V2;
    type i64x2 = I64x2V2;
    type u64x2 = U64x2V2;

    type f32x8 = DoublePumpRegister<Self::f32x4>;
    type i32x8 = DoublePumpRegister<Self::i32x4>;
    type u32x8 = DoublePumpRegister<Self::u32x4>;

    type f64x4 = DoublePumpRegister<Self::f64x2>;
    type i64x4 = DoublePumpRegister<Self::i64x2>;
    type u64x4 = DoublePumpRegister<Self::u64x2>;

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
