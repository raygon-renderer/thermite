use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    FloatRegister, LinAlg3Register, NumericRegister, PermuteRegister, Register, ShiftRegister, ShuffleRegister,
    SignedRegister, SwizzleRegister, dp::DoublePumpRegister,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F32x4SSE41;

impl Register for F32x4SSE41 {
    type Lanes = generic_array::typenum::U4;

    type Element = f32;
    type Storage = arch::__m128;

    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    #[inline(always)]
    fn new(value: generic_array::GenericArray<f32, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_ps(value) }
    }

    #[inline(always)]
    fn empty() -> Self::Storage {
        unsafe { arch::_mm_undefined_ps() }
    }

    #[inline(always)]
    fn xor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn and(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_and_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn andnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_andnot_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn or(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_or_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_ps(value, arch::_mm_set1_ps(f32::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_ps(lhs, rhs, mask) }
    }

    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_sll_epi32(
                arch::_mm_castps_si128(value),
                arch::_mm_set_epi32(0, 0, 0, shift as i32),
            ))
        }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_srl_epi32(
                arch::_mm_castps_si128(value),
                arch::_mm_set_epi32(0, 0, 0, shift as i32),
            ))
        }
    }

    #[inline(always)]
    fn shlv(mut value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        // TODO: use _mm_sllv_epi32 doesn't exist on SSE4.1, so we may want to emulate it
        // more intelligently later.
        Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(shifts.into())
            .for_each(|(v, s)| {
                let mut vi = v.to_bits();
                vi <<= s;
                *v = f32::from_bits(vi);
            });

        value
    }

    #[inline(always)]
    fn shrv(mut value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(shifts.into())
            .for_each(|(v, s)| {
                let mut vi = v.to_bits();
                vi >>= s;
                *v = f32::from_bits(vi);
            });

        value
    }
}

impl ShiftRegister for F32x4SSE41 {
    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_slli_epi32(arch::_mm_castps_si128(value), IMM8)) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_srli_epi32(arch::_mm_castps_si128(value), IMM8)) }
    }
}

impl ShuffleRegister for F32x4SSE41 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F32x4SSE41 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_ps(value, value, IMM8) }
    }
}

impl SwizzleRegister for F32x4SSE41 {
    fn permutev(value: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe {
            arch::_mm_castsi128_ps(crate::backend::sse41::polyfills::_mm_permutevarx_epi32(
                arch::_mm_castps_si128(value),
                core::mem::transmute(idxs.into()),
            ))
        }
    }

    #[inline(always)]
    fn swizzle2<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
        a: Self::Storage,
        b: Self::Storage,
    ) -> Self::Storage {
        unsafe {
            arch::_mm_blend_ps(
                arch::_mm_shuffle_ps(a, a, AIMM8),
                arch::_mm_shuffle_ps(b, b, BIMM8),
                BLEND,
            )
        }
    }
}

impl NumericRegister for F32x4SSE41 {
    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps!(value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps!(value; _mm_max_ps _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps!(value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps!(value; _mm_mul_ps _mm_mul_ss)
    }

    #[inline(always)]
    fn max_value() -> Self::Storage {
        Self::splat(f32::MAX)
    }

    #[inline(always)]
    fn min_value() -> Self::Storage {
        Self::splat(f32::MIN)
    }

    #[inline(always)]
    fn one() -> Self::Storage {
        Self::splat(1.0)
    }

    #[inline(always)]
    fn zero() -> Self::Storage {
        Self::splat(0.0)
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_add_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sub_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_mul_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_div_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_min_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_ps(lhs, rhs) }
    }
}

impl SignedRegister for F32x4SSE41 {
    #[inline(always)]
    fn neg_one() -> Self::Storage {
        Self::splat(-1.0)
    }

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        Self::xor(value, Self::splat(f32::from_bits(0x8000_0000)))
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        Self::and(value, Self::splat(f32::from_bits(0x7fffffff)))
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        let sign_mask = Self::splat(f32::from_bits(0x8000_0000));

        // take everything but the sign from lhs, and copy the sign from rhs
        Self::or(Self::andnot(sign_mask, lhs), Self::and(sign_mask, rhs))
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        // copy sign bit to 1.0
        Self::or(
            Self::splat(1.0),
            Self::and(value, Self::splat(f32::from_bits(0x8000_0000))),
        )
    }
}

impl FloatRegister for F32x4SSE41 {
    #[inline(always)]
    fn neg_zero() -> Self::Storage {
        Self::splat(-0.0)
    }

    #[inline(always)]
    fn epsilon() -> Self::Storage {
        Self::splat(f32::EPSILON)
    }

    #[inline(always)]
    fn infinity() -> Self::Storage {
        Self::splat(f32::INFINITY)
    }

    #[inline(always)]
    fn neg_infinity() -> Self::Storage {
        Self::splat(f32::NEG_INFINITY)
    }

    #[inline(always)]
    fn nan() -> Self::Storage {
        Self::splat(f32::NAN)
    }

    #[inline(always)]
    fn sqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sqrt_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_rsqrt_ps(value) }
    }

    #[inline(always)]
    fn rcp(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_rcp_ps(value) }
    }

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_floor_ps(value) }
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_ceil_ps(value) }
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }
}

impl LinAlg3Register for F32x4SSE41 {
    #[inline(always)]
    fn dot3(lhs: Self::Storage, rhs: Self::Storage) -> f32 {
        unsafe { crate::backend::sse41::polyfills::dot3_sse41(lhs, rhs) }
    }

    #[inline(always)]
    fn cross3(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { crate::backend::sse41::polyfills::cross3_sse41(lhs, rhs) }
    }

    #[inline(always)]
    fn zero4(value: Self::Storage) -> Self::Storage {
        unsafe { crate::backend::sse41::polyfills::zero4_sse41(value) }
    }

    #[inline(always)]
    fn one4(value: Self::Storage) -> Self::Storage {
        unsafe { crate::backend::sse41::polyfills::one4_sse41(value) }
    }
}
