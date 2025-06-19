use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    CastRegister, FloatRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
    ShiftRegister, ShuffleRegister, SignedRegister, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F32x8V3;

impl Register for F32x8V3 {
    type Lanes = generic_array::typenum::U8;

    type Element = f32;
    type Storage = arch::__m256;
    type HalfRegister = super::f32x4::F32x4V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Self::Storage
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128(lo, hi) }
    }

    #[inline(always)]
    fn split(
        value: Self::Storage,
    ) -> (
        <Self::HalfRegister as Register>::Storage,
        <Self::HalfRegister as Register>::Storage,
    )
    where
        Self::HalfRegister: Register,
    {
        let lo = unsafe { arch::_mm256_castps256_ps128(value) };
        let hi = unsafe { arch::_mm256_extractf128_ps(value, 1) };
        (lo, hi)
    }

    #[inline(always)]
    fn new(value: generic_array::GenericArray<f32, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_set1_ps(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_load_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_loadu_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_store_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_storeu_ps(ptr, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_and_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_andnot_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_or_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_ps(value, arch::_mm256_set1_ps(f32::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_blendv_ps(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe {
            arch::_mm256_castsi256_ps(arch::_mm256_sll_epi32(
                arch::_mm256_castps_si256(value),
                arch::_mm_cvtsi32_si128(shift as i32),
            ))
        }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe {
            arch::_mm256_castsi256_ps(arch::_mm256_srl_epi32(
                arch::_mm256_castps_si256(value),
                arch::_mm_cvtsi32_si128(shift as i32),
            ))
        }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe {
            arch::_mm256_castsi256_ps(arch::_mm256_sllv_epi32(
                arch::_mm256_castps_si256(value),
                core::mem::transmute(shifts.into()),
            ))
        }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe {
            arch::_mm256_castsi256_ps(arch::_mm256_srlv_epi32(
                arch::_mm256_castps_si256(value),
                core::mem::transmute(shifts.into()),
            ))
        }
    }
}

impl ShiftRegister for F32x8V3 {
    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_slli_epi32(arch::_mm256_castps_si256(value), IMM8)) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_srli_epi32(arch::_mm256_castps_si256(value), IMM8)) }
    }
}

impl ShuffleRegister for F32x8V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_shuffle_ps(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F32x8V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_permutevar8x32_ps(value, const { super::shuffle_to_m256i(IMM8) }) }
    }
}

impl SwizzleRegister for F32x8V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm256_permutevar8x32_ps(value, core::mem::transmute(idxs.into())) }
    }

    // #[inline(always)]
    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     a: Self::Storage,
    //     b: Self::Storage,
    // ) -> Self::Storage {
    //     unsafe {
    //         let tmp_a = arch::_mm256_permutevar8x32_ps(a, const { super::shuffle_to_m256i(AIMM8) });
    //         let tmp_b = arch::_mm256_permutevar8x32_ps(b, const { super::shuffle_to_m256i(BIMM8) });

    //         arch::_mm256_blend_ps(tmp_a, tmp_b, BLEND)
    //     }
    // }

    #[inline(always)]
    fn swizzle(a: Self::Storage, b: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe {
            let idxs: arch::__m256i = core::mem::transmute(idxs.into());

            let blend = arch::_mm256_cmpgt_epi32(idxs, arch::_mm256_set1_epi32(7));
            let a_idxs = arch::_mm256_and_si256(idxs, arch::_mm256_set1_epi32(0b111));
            let b_idxs = arch::_mm256_sub_epi32(idxs, arch::_mm256_set1_epi32(8));

            let tmp_a = arch::_mm256_permutevar8x32_ps(a, a_idxs);
            let tmp_b = arch::_mm256_permutevar8x32_ps(b, b_idxs);

            arch::_mm256_blendv_ps(tmp_a, tmp_b, arch::_mm256_castsi256_ps(blend))
        }
    }
}

impl MaskRegister for F32x8V3 {
    const FALSY: Self::Storage = reg::<Self, 8>([f32::from_bits(0); 8]);
    const TRUTHY: Self::Storage = reg::<Self, 8>([f32::from_bits(!0); 8]);

    #[inline(always)]
    fn new_mask(value: impl Into<GenericArray<bool, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_cvtboolx8_to_epi32_mask_v3(value.into())) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0xff }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0 }
    }
}

impl PartialOrdRegister for F32x8V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

impl NumericRegister for F32x8V3 {
    const ZERO: Self::Storage = reg::<Self, 8>([0.0; 8]);
    const ONE: Self::Storage = reg::<Self, 8>([1.0; 8]);
    const TWO: Self::Storage = reg::<Self, 8>([2.0; 8]);

    const MIN: Self::Storage = reg::<Self, 8>([f32::MIN; 8]);
    const MAX: Self::Storage = reg::<Self, 8>([f32::MAX; 8]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_ps!(value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_ps!(value; _mm_max_ps _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_ps!(value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_ps!(value; _mm_mul_ps _mm_mul_ss)
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
        unsafe { arch::_mm256_add_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sub_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_mul_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_div_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_min_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_max_ps(lhs, rhs) }
    }
}

impl SignedRegister for F32x8V3 {
    const NEG_ONE: Self::Storage = reg::<Self, 8>([-1.0; 8]);

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        Self::bitandnot(Self::NEG_ZERO, value)
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // take everything but the sign from lhs, and copy the sign from rhs
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

impl FloatRegister for F32x8V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U32x8V3;
    type Signed = super::I32x8V3;

    const HALF: Self::Storage = reg::<Self, 8>([0.5; 8]);
    const NEG_ZERO: Self::Storage = reg::<Self, 8>([-0.0; 8]);
    const EPSILON: Self::Storage = reg::<Self, 8>([f32::EPSILON; 8]);
    const INFINITY: Self::Storage = reg::<Self, 8>([f32::INFINITY; 8]);
    const NEG_INFINITY: Self::Storage = reg::<Self, 8>([f32::NEG_INFINITY; 8]);
    const NAN: Self::Storage = reg::<Self, 8>([f32::NAN; 8]);

    #[inline(always)] #[rustfmt::skip]
    fn is_subnormal(value: Self::Storage) -> Self::Storage {
        let m = Self::splat(f32::from_bits(0xFF000000));
        let u = Self::shli::<1>(value);

        Self::bitand(
            Self::eq(Self::ZERO, Self::bitand(u, m)),
            Self::ne(Self::ZERO, Self::bitandnot(m, u))
        )
    }

    #[inline(always)]
    fn is_zero_or_subnormal(value: Self::Storage) -> Self::Storage {
        Self::eq(Self::ZERO, Self::bitand(value, Self::splat(f32::from_bits(0x7F800000))))
    }

    #[inline(always)]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fmsub_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fnmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fnmsub_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::mul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::mul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::nmul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::nmul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn sqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sqrt_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_rsqrt_ps(value) }
    }

    #[inline(always)]
    fn rcp(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_rcp_ps(value) }
    }

    const HAS_APPROX_RSQRT: bool = true;
    const HAS_APPROX_RCP: bool = true;

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_floor_ps(value) }
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_ceil_ps(value) }
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_nextupps_v3(value) }
    }

    #[inline(always)]
    fn next_down(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_nextdownps_v3(value) }
    }
}

impl CastRegister<F32x8V3> for DoublePumpRegister<super::F64x4V3> {
    #[inline(always)]
    fn cast_from(value: <F32x8V3 as Register>::Storage) -> Self::Storage {
        let (lo, hi) = F32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtps_pd(lo);
            let hi = arch::_mm256_cvtps_pd(hi);

            DoublePumpRegister::join(lo, hi)
        }
    }
}
