//! Native AVX2 registers.

use super::arch;

pub mod f32x4;
pub mod f32x8;
pub mod f64x2;
pub mod f64x4;
pub mod i32x4;
pub mod i32x8;
pub mod i64x2;
pub mod i64x4;
pub mod u32x4;
pub mod u32x8;
pub mod u64x2;
pub mod u64x4;

pub use f32x4::F32x4V3;
pub use f32x8::F32x8V3;
pub use f64x2::F64x2V3;
pub use f64x4::F64x4V3;
pub use i32x4::I32x4V3;
pub use i32x8::I32x8V3;
pub use i64x2::I64x2V3;
pub use i64x4::I64x4V3;
pub use u32x4::U32x4V3;
pub use u32x8::U32x8V3;
pub use u64x2::U64x2V3;
pub use u64x4::U64x4V3;

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{Storage, dp::DoublePumpRegister},
    simd::{NativeSimd, Simd},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86V3;

impl NativeSimd for X86V3 {
    const ISA: InstructionSet = InstructionSet::X86V3;

    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U8;
    type Native64Width = generic_array::typenum::U4;

    type NativeAlignment = crate::simd::Align32; // 256-bit vectors = 32 bytes

    type f32xN = F32x8V3;
    type i32xN = I32x8V3;
    type u32xN = U32x8V3;

    type f64xN = F64x4V3;
    type i64xN = I64x4V3;
    type u64xN = U64x4V3;
}

impl Simd for X86V3 {
    type f32x2 = <Scalar as Simd>::f32x2;
    type i32x2 = <Scalar as Simd>::i32x2;
    type u32x2 = <Scalar as Simd>::u32x2;

    type f32x4 = F32x4V3;
    type i32x4 = I32x4V3;
    type u32x4 = U32x4V3;

    type f32x8 = F32x8V3;
    type i32x8 = I32x8V3;
    type u32x8 = U32x8V3;

    type f64x2 = F64x2V3;
    type i64x2 = I64x2V3;
    type u64x2 = U64x2V3;

    type f64x4 = F64x4V3;
    type i64x4 = I64x4V3;
    type u64x4 = U64x4V3;

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

const fn shuffle_to_m256i(bitmask: i32) -> arch::__m256i {
    let mut masks = [0i32; 8];

    let mut i = 0;

    while i < 8 {
        let k = i as i32;
        masks[i] = (bitmask >> (k * 3)) & 0b111;
        i += 1;
    }

    unsafe { generic_array::const_transmute(masks) }
}

impl_bit_casts! {
    F64x2V3 as I64x2V3 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_castpd_si128, // f64x2 -> u64x2
    I64x2V3 as F64x2V3 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_castsi128_pd, // u64x2 -> f64x2

    F64x4V3 as I64x4V3 => _mm256_castpd_si256, // f64x4 -> i64x4
    I64x4V3 as F64x4V3 => _mm256_castsi256_pd, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_castsi256_pd, // u64x4 -> f64x4
    F64x4V3 as U64x4V3 => _mm256_castpd_si256, // f64x4 -> u64x4

    F32x4V3 as I32x4V3 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_castps_si128, // f32x4 -> u32x4
    I32x4V3 as F32x4V3 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_castsi128_ps, // u32x4 -> f32x4

    F32x8V3 as I32x8V3 => _mm256_castps_si256, // f32x8 -> i32x8
    F32x8V3 as U32x8V3 => _mm256_castps_si256, // f32x8 -> u32x8
    I32x8V3 as F32x8V3 => _mm256_castsi256_ps, // i32x8 -> f32x8
    U32x8V3 as F32x8V3 => _mm256_castsi256_ps, // u32x8 -> f32x8

    // integer casts use the same underlying storage, so identity casts
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4

    // all the identity casts to self
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4
}

impl_type_casts! {
    // self casts
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4

    // f32x4 casts
    F32x4V3 as I32x4V3 => _mm_cvtps_epi32, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_cvtps_epu32x_v2, // f32x4 -> u32x4
    I32x4V3 as F32x4V3 => _mm_cvtepi32_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_cvtepu32_psx_v2, // u32x4 -> f32x4

    // f32x8 casts
    F32x8V3 as I32x8V3 => _mm256_cvtps_epi32, // i32x8 -> f32x8
    F32x8V3 as U32x8V3 => _mm256_cvtps_epu32x_v3, // f32x8 -> u32x8
    I32x8V3 as F32x8V3 => _mm256_cvtepi32_ps, // i32x4 -> f32x4
    U32x8V3 as F32x8V3 => _mm256_cvtepu32_psx_v3, // i32x8 -> f32x8

    // f64x2 casts
    F64x2V3 as I64x2V3 => _mm_cvtpd_epi64x_v2 | _mm_cvtpd_epi64x_limited_v1, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_cvtpd_epu64x_limited_v1, // f64x2 -> u64x2
    I64x2V3 as F64x2V3 => _mm_cvtepi64_pdx_v2 | _mm_cvtepi64_pdx_limited_v1, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_cvtepu64_pdx_v2 | _mm_cvtepu64_pdx_limited_v1, // u64x2 -> f64x2

    // f64x4 casts
    F64x4V3 as I64x4V3 => _mm256_cvtpd_epi64x_v3 | _mm256_cvtpd_epi64x_limited_v3, // f64x4 -> i64x4
    F64x4V3 as U64x4V3 => _mm256_cvtpd_epu64x_limited_v3, // f64x4 -> u64x4
    I64x4V3 as F64x4V3 => _mm256_cvtepi64_pdx_v3 | _mm256_cvtepi64_pdx_limited_v3, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_cvtepu64_pdx_v3 | _mm256_cvtepu64_pdx_limited_v3, // u64x4 -> f64x4

    // for integer casts we don't do anything, basically bit casting, same as Rust
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4

    // simple precision casts, others are implemented in-module
    F32x4V3 as F64x4V3 => _mm256_cvtps_pd, // f32x4 -> f64x4
    F64x4V3 as F32x4V3 => _mm256_cvtpd_ps, // f64x4 -> f32x4
    U32x4V3 as U64x4V3 => _mm256_cvtepu32_epi64, // u32x4 -> u64x4
    U64x4V3 as U32x4V3 => _mm256_cvtepi64_epi32_v3, // u64x4 -> u32x4
    I32x4V3 as I64x4V3 => _mm256_cvtepi32_epi64, // i32x4 -> i64x4
    I64x4V3 as I32x4V3 => _mm256_cvtepi64_epi32_v3, // i64x2 -> i32x4
}

impl_mask_casts! {
    // self casts
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4

    // same-size integer casts
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4

    // same-size float/integer casts
    I32x4V3 as F32x4V3 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_castsi128_ps, // u32x4 -> f32x4
    I32x8V3 as F32x8V3 => _mm256_castsi256_ps, // i32x8 -> f32x8
    U32x8V3 as F32x8V3 => _mm256_castsi256_ps, // u32x8 -> f32x8
    I64x2V3 as F64x2V3 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_castsi128_pd, // u64x2 -> f64x2
    I64x4V3 as F64x4V3 => _mm256_castsi256_pd, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_castsi256_pd, // u64x4 -> f64x4
    F32x4V3 as I32x4V3 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_castps_si128, // f32x4 -> u32x4
    F32x8V3 as I32x8V3 => _mm256_castps_si256, // f32x8 -> i32x8
    F32x8V3 as U32x8V3 => _mm256_castps_si256, // f32x8 -> u32x8
    F64x2V3 as I64x2V3 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_castpd_si128, // f64x2 -> u64x2
    F64x4V3 as I64x4V3 => _mm256_castpd_si256, // f64x4 -> i64x4
    F64x4V3 as U64x4V3 => _mm256_castpd_si256, // f64x4 -> u64x4
}

#[cfg(test)]
mod tests {
    use crate::register::{FloatRegister, IntegerRegister, SignedRegister, UnsignedIntegerRegister};

    use super::*;

    fn assert_is_float<R: FloatRegister>() {}
    fn assert_is_integer<R: IntegerRegister>() {}
    fn assert_is_unsigned<R: UnsignedIntegerRegister>() {}
    fn assert_is_signed<R: SignedRegister>() {}

    #[test]
    fn test_compiles() {
        assert_is_float::<F32x4V3>();
        assert_is_float::<F32x8V3>();
        assert_is_float::<F64x2V3>();
        assert_is_float::<F64x4V3>();

        assert_is_integer::<I32x4V3>();
        assert_is_integer::<I32x8V3>();
        assert_is_integer::<I64x2V3>();
        assert_is_integer::<I64x4V3>();
        assert_is_integer::<U32x4V3>();
        assert_is_integer::<U32x8V3>();
        assert_is_integer::<U64x2V3>();
        assert_is_integer::<U64x4V3>();

        assert_is_unsigned::<U32x4V3>();
        assert_is_unsigned::<U32x8V3>();
        assert_is_unsigned::<U64x2V3>();
        assert_is_unsigned::<U64x4V3>();

        assert_is_signed::<I32x4V3>();
        assert_is_signed::<I32x8V3>();
        assert_is_signed::<I64x2V3>();
        assert_is_signed::<I64x4V3>();

        assert_is_signed::<F32x4V3>();
        assert_is_signed::<F32x8V3>();
        assert_is_signed::<F64x2V3>();
        assert_is_signed::<F64x4V3>();
    }
}
