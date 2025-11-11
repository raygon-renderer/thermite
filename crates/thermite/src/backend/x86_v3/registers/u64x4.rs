use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::register::{
    CastRegister, IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
    ShiftRegister, ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, dp::DoublePumpRegister,
    empty_reg, reg,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct U64x4V3;

impl Register for U64x4V3 {
    type Lanes = typenum::U4;

    type Element = u64;
    type Storage = arch::__m256i;
    type HalfRegister = super::U64x2V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    type SCOUNT = super::I64x4V3;
    type UCOUNT = super::U64x4V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_set1_epi64x(value as i64) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
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
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };

        (lo, hi)
    }

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Self::Storage
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_blendv_epi8(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_permute4x64_epi64::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }
}

impl ShiftRegister for U64x4V3 {
    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_srlv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_sllv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_slli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_srli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_rolv_epi64x_v3(value, shifts) }
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_rorv_epi64x_v3(value, shifts) }
    }

    #[inline(always)]
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_rolv_epi64x_v3(value, arch::_mm256_set1_epi64x(shift as i64)) }
    }

    #[inline(always)]
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_rorv_epi64x_v3(value, arch::_mm256_set1_epi64x(shift as i64)) }
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_reverse_bits_epi64x_v3(value) }
    }
}

impl MaskRegister for U64x4V3 {
    const FALSY: Self::Storage = reg::<Self, 4>([0; 4]);
    const TRUTHY: Self::Storage = reg::<Self, 4>([!0; 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_cvtboolx4_to_epi64_mask_v3(value) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) == 0 }
    }
}

impl PartialOrdRegister for U64x4V3 {
    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpgt_epu64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpeq_epi64(lhs, rhs) }
    }
}

impl NumericRegister for U64x4V3 {
    const ZERO: Self::Storage = reg::<Self, 4>([0; 4]);
    const ONE: Self::Storage = reg::<Self, 4>([1; 4]);
    const TWO: Self::Storage = reg::<Self, 4>([2; 4]);

    const MIN: Self::Storage = reg::<Self, 4>([u64::MIN; 4]);
    const MAX: Self::Storage = reg::<Self, 4>([u64::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_min_epu64x_v2 _mm_min_epu64x_v2) as u64
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_max_epu64x_v2 _mm_max_epu64x_v2) as u64
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_add_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sub_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
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
        unsafe { arch::_mm256_min_epu64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_max_epu64x_v3(lhs, rhs) }
    }
}

impl IntegerRegister for U64x4V3 {
    #[inline(always)]
    fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

    #[inline(always)]
    fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi64!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    #[inline(always)]
    fn div_branched(value: Self::Storage, divider: crate::divider::Divider<Self::Element>) -> Self::Storage {
        unsafe { arch::_mm256_div_epu64x(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn div_branchfree(
        value: Self::Storage,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Self::Storage {
        unsafe { arch::_mm256_div_epu64x_bf(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_popcnt_epi64x_v3(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        super::I64x4V3::count_ones(value)
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

impl UnsignedIntegerRegister for U64x4V3 {
    #[inline(always)]
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_np2_m1_epu64x_v3(value) }
    }

    #[inline(always)]
    fn is_power_of_two(value: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm256_cmpeq_epi64(
                value,
                arch::_mm256_and_si256(value, arch::_mm256_sub_epi64(value, arch::_mm256_set1_epi64x(1))),
            )
        }
    }

    #[inline(always)]
    fn parity(mut value: Self::Storage) -> Self::Storage {
        unsafe {
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi64(value, 32));
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi64(value, 16));
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi64(value, 8));
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi64(value, 4));
            value = arch::_mm256_and_si256(value, arch::_mm256_set1_epi64x(0x0F));

            arch::_mm256_and_si256(
                arch::_mm256_srlv_epi64(arch::_mm256_set1_epi64x(0x6996), value),
                arch::_mm256_set1_epi64x(1),
            )
        }
    }
}

impl CastRegister<DoublePumpRegister<U64x4V3>> for super::U32x8V3 {
    #[inline(always)]
    fn cast_from(value: <DoublePumpRegister<U64x4V3> as Register>::Storage) -> Self::Storage {
        let (lo, hi) = DoublePumpRegister::<U64x4V3>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepi64_epi32_v3(lo);
            let hi = arch::_mm256_cvtepi64_epi32_v3(hi);

            arch::_mm256_setr_m128i(lo, hi)
        }
    }
}
