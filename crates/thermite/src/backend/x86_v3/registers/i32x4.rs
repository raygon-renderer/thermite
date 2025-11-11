use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::register::{
    IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShiftRegister,
    ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister,
    empty_reg, reg,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct I32x4V3;

impl Register for I32x4V3 {
    type Lanes = typenum::U4;

    type Element = i32;
    type Storage = arch::__m128i;
    type HalfRegister = ();
    type DoubleRegister = super::I32x8V3;

    type SCOUNT = super::I32x4V3;
    type UCOUNT = super::U32x4V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_epi32(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_epi8(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }
}

impl ShiftRegister for I32x4V3 {
    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_sllv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_srlv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_slli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_srli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi32x_v3(value, shifts) }
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi32x_v3(value, shifts) }
    }

    #[inline(always)]
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi32x_v3(value, arch::_mm_set1_epi32(shift as i32)) }
    }

    #[inline(always)]
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi32x_v3(value, arch::_mm_set1_epi32(shift as i32)) }
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_reverse_bits_epi32x_v2(value) }
    }
}

impl ShuffleRegister for I32x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_shuffle_ps(
                arch::_mm_castsi128_ps(lhs),
                arch::_mm_castsi128_ps(rhs),
                IMM8,
            ))
        }
    }
}

impl PermuteRegister for I32x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_epi32(value, IMM8) }
    }
}

impl SwizzleRegister for I32x4V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_permutevar_ps(
                arch::_mm_castsi128_ps(value),
                core::mem::transmute(idxs),
            ))
        }
    }

    // #[inline(always)]
    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     a: Self::Storage,
    //     b: Self::Storage,
    // ) -> Self::Storage {
    //     unsafe {
    //         arch::_mm_blend_epi16(
    //             arch::_mm_shuffle_epi32(a, AIMM8),
    //             arch::_mm_shuffle_epi32(b, BIMM8),
    //             BLEND,
    //         )
    //     }
    // }
}

impl MaskRegister for I32x4V3 {
    const FALSY: Self::Storage = reg::<Self, 4>([0; 4]);
    const TRUTHY: Self::Storage = reg::<Self, 4>([-1; 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_cvtboolx4_to_epi32_mask_v2(value) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }
}

impl PartialOrdRegister for I32x4V3 {
    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpgt_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpeq_epi32(lhs, rhs) }
    }
}

impl NumericRegister for I32x4V3 {
    const ZERO: Self::Storage = reg::<Self, 4>([0; 4]);
    const ONE: Self::Storage = reg::<Self, 4>([1; 4]);
    const TWO: Self::Storage = reg::<Self, 4>([2; 4]);

    const MIN: Self::Storage = reg::<Self, 4>([i32::MIN; 4]);
    const MAX: Self::Storage = reg::<Self, 4>([i32::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_min_epi32 _mm_min_epi32)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_max_epi32 _mm_max_epi32)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_add_epi32 _mm_add_epi32)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_mullo_epi32 _mm_mullo_epi32)
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I32)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as i32))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_add_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sub_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_min_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_epi32(lhs, rhs) }
    }
}

impl SignedRegister for I32x4V3 {
    const NEG_ONE: Self::Storage = reg::<Self, 4>([-1; 4]);

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sign_epi32(value, Self::NEG_ONE) }
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_abs_epi32(value) }
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // sign_epi32 negates if b is negative, but also sets lhs to zero
        // if rhs is zero, so we OR it with 1 to prevent that behavior
        unsafe { arch::_mm_sign_epi32(lhs, arch::_mm_or_si128(rhs, arch::_mm_set1_epi32(1))) }
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        // same thing as above, but negating 1 instead of an input value
        unsafe {
            arch::_mm_sign_epi32(
                arch::_mm_set1_epi32(1),
                arch::_mm_or_si128(value, arch::_mm_set1_epi32(1)),
            )
        }
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self::add(Self::bitxor(value, mask), Self::shri::<31>(mask))
    }
}

impl IntegerRegister for I32x4V3 {
    #[inline(always)]
    fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_adds_epi32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_subs_epi32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn wrapping_sum(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_add_epi32 _mm_add_epi32) as i32
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_mullo_epi32 _mm_mullo_epi32) as i32
    }

    #[inline(always)]
    fn div_branched(value: Self::Storage, divider: crate::divider::Divider<Self::Element>) -> Self::Storage {
        unsafe { arch::_mm_div_epi32x(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn div_branchfree(
        value: Self::Storage,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Self::Storage {
        unsafe { arch::_mm_div_epi32x_bf(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_popcnt_epi32x_v2(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        // treat as unsigned
        super::U32x4V3::leading_zeros(value)
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }

    #[inline(always)]
    fn leading_ones(value: Self::Storage) -> Self::Storage {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Self::Storage) -> Self::Storage {
        Self::trailing_zeros(Self::not(value))
    }
}

impl SignedIntegerRegister for I32x4V3 {
    #[inline(always)]
    fn srai<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_srai_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn sra(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_sra_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn srav(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_srav_epi32(value, shifts) }
    }
}
