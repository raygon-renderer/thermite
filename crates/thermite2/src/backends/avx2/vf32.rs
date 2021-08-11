use crate::backends::polyfills::float_rem;

use super::*;

#[rustfmt::skip]
unsafe impl Register for AVX2F32Register<4> {
    type Element = f32;
    type Storage = __m128;

    #[inline(always)] unsafe fn set1(x: f32) -> __m128 { _mm_set1_ps(x) }
}

#[rustfmt::skip]
unsafe impl Register for AVX2F32Register<8> {
    type Element = f32;
    type Storage = __m256;

    #[inline(always)] unsafe fn set1(x: f32) -> __m256 { _mm256_set1_ps(x) }
}

unsafe impl FixedRegister<4> for AVX2F32Register<4> {
    #[inline(always)]
    unsafe fn setr(values: [f32; 4]) -> __m128 {
        core::mem::transmute(values)
    }
}

unsafe impl FixedRegister<8> for AVX2F32Register<8> {
    #[inline(always)]
    unsafe fn setr(values: [f32; 8]) -> __m256 {
        core::mem::transmute(values)
    }
}

unsafe impl<const N: usize> UnaryRegisterOps for AVX2F32Register<N>
where
    Self: BinaryRegisterOps<Element = f32>,
{
    #[inline(always)]
    unsafe fn bit_not(r: Self::Storage) -> Self::Storage {
        Self::bitxor(r, Self::set1(f32::from_bits(!0)))
    }
}

#[rustfmt::skip]
unsafe impl BinaryRegisterOps for AVX2F32Register<4> {
    #[inline(always)] unsafe fn bitand(lhs: __m128, rhs: __m128) -> __m128 { _mm_and_ps(lhs, rhs) }
    #[inline(always)] unsafe fn bitor(lhs: __m128, rhs: __m128) -> __m128  { _mm_or_ps(lhs, rhs)  }
    #[inline(always)] unsafe fn bitxor(lhs: __m128, rhs: __m128) -> __m128 { _mm_xor_ps(lhs, rhs) }
    #[inline(always)] unsafe fn and_not(lhs: __m128, rhs: __m128) -> __m128 { _mm_andnot_ps(lhs, rhs) }
    #[inline(always)] unsafe fn add(lhs: __m128, rhs: __m128) -> __m128 { _mm_add_ps(lhs, rhs) }
    #[inline(always)] unsafe fn sub(lhs: __m128, rhs: __m128) -> __m128 { _mm_sub_ps(lhs, rhs) }
    #[inline(always)] unsafe fn mul(lhs: __m128, rhs: __m128) -> __m128 { _mm_mul_ps(lhs, rhs) }
    #[inline(always)] unsafe fn div(lhs: __m128, rhs: __m128) -> __m128 { _mm_div_ps(lhs, rhs) }
    #[inline(always)] unsafe fn rem(lhs: __m128, rhs: __m128) -> __m128 { float_rem::<Self>(lhs, rhs) }
}

#[rustfmt::skip]
unsafe impl BinaryRegisterOps for AVX2F32Register<8> {
    #[inline(always)] unsafe fn bitand(lhs: __m256, rhs: __m256) -> __m256 { _mm256_and_ps(lhs, rhs) }
    #[inline(always)] unsafe fn bitor(lhs: __m256, rhs: __m256) -> __m256  { _mm256_or_ps(lhs, rhs)  }
    #[inline(always)] unsafe fn bitxor(lhs: __m256, rhs: __m256) -> __m256 { _mm256_xor_ps(lhs, rhs) }
    #[inline(always)] unsafe fn and_not(lhs: __m256, rhs: __m256) -> __m256 { _mm256_andnot_ps(lhs, rhs) }
    #[inline(always)] unsafe fn add(lhs: __m256, rhs: __m256) -> __m256 { _mm256_add_ps(lhs, rhs) }
    #[inline(always)] unsafe fn sub(lhs: __m256, rhs: __m256) -> __m256 { _mm256_sub_ps(lhs, rhs) }
    #[inline(always)] unsafe fn mul(lhs: __m256, rhs: __m256) -> __m256 { _mm256_mul_ps(lhs, rhs) }
    #[inline(always)] unsafe fn div(lhs: __m256, rhs: __m256) -> __m256 { _mm256_div_ps(lhs, rhs) }
    #[inline(always)] unsafe fn rem(lhs: __m256, rhs: __m256) -> __m256 { float_rem::<Self>(lhs, rhs) }
}

unsafe impl<const N: usize> SignedRegisterOps for AVX2F32Register<N>
where
    Self: BinaryRegisterOps<Element = f32>,
{
    #[inline(always)]
    unsafe fn neg(x: Self::Storage) -> Self::Storage {
        Self::bitxor(x, Self::set1(-0.0))
    }

    #[inline(always)]
    unsafe fn abs(x: Self::Storage) -> Self::Storage {
        Self::bitand(x, Self::set1(f32::from_bits(0x7fffffff)))
    }
}

#[rustfmt::skip]
unsafe impl FloatRegisterOps for AVX2F32Register<4> {
    #[inline(always)] unsafe fn ceil(x: __m128) -> __m128 { _mm_ceil_ps(x) }
    #[inline(always)] unsafe fn floor(x: __m128) -> __m128 { _mm_floor_ps(x) }
    #[inline(always)] unsafe fn round(x: __m128) -> __m128 { _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }
    #[inline(always)] unsafe fn trunc(x: __m128) -> __m128 { _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }

    #[inline(always)] unsafe fn sqrt(x: __m128) -> __m128 { _mm_sqrt_ps(x) }
    #[inline(always)] unsafe fn rsqrt(x: __m128) -> __m128 { _mm_rsqrt_ps(x) }
    #[inline(always)] unsafe fn rcp(x: __m128) -> __m128 { _mm_rcp_ps(x) }

    #[inline(always)] unsafe fn mul_add(x: __m128, m: __m128, a: __m128) -> __m128 { _mm_fmadd_ps(x, m, a) }
    #[inline(always)] unsafe fn mul_sub(x: __m128, m: __m128, a: __m128) -> __m128 { _mm_fmsub_ps(x, m, a) }
    #[inline(always)] unsafe fn nmul_add(x: __m128, m: __m128, a: __m128) -> __m128 { _mm_fnmadd_ps(x, m, a) }
    #[inline(always)] unsafe fn nmul_sub(x: __m128, m: __m128, a: __m128) -> __m128 { _mm_fnmsub_ps(x, m, a) }
}

#[rustfmt::skip]
unsafe impl FloatRegisterOps for AVX2F32Register<8> {
    #[inline(always)] unsafe fn ceil(x: __m256) -> __m256 { _mm256_ceil_ps(x) }
    #[inline(always)] unsafe fn floor(x: __m256) -> __m256 { _mm256_floor_ps(x) }
    #[inline(always)] unsafe fn round(x: __m256) -> __m256 { _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }
    #[inline(always)] unsafe fn trunc(x: __m256) -> __m256 { _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }

    #[inline(always)] unsafe fn sqrt(x: __m256) -> __m256 { _mm256_sqrt_ps(x) }
    #[inline(always)] unsafe fn rsqrt(x: __m256) -> __m256 { _mm256_rsqrt_ps(x) }
    #[inline(always)] unsafe fn rcp(x: __m256) -> __m256 { _mm256_rcp_ps(x) }

    #[inline(always)] unsafe fn mul_add(x: __m256, m: __m256, a: __m256) -> __m256 { _mm256_fmadd_ps(x, m, a) }
    #[inline(always)] unsafe fn mul_sub(x: __m256, m: __m256, a: __m256) -> __m256 { _mm256_fmsub_ps(x, m, a) }
    #[inline(always)] unsafe fn nmul_add(x: __m256, m: __m256, a: __m256) -> __m256 { _mm256_fnmadd_ps(x, m, a) }
    #[inline(always)] unsafe fn nmul_sub(x: __m256, m: __m256, a: __m256) -> __m256 { _mm256_fnmsub_ps(x, m, a) }
}
