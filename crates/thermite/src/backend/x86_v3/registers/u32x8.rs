use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
        dp::DoublePumpRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct U32x8V3;

impl Register for U32x8V3 {
    type Lanes = typenum::U8;

    type Element = u32;
    type Storage = arch::__m256i;
    type HalfRegister = super::U32x4V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type SCOUNT = super::I32x8V3;
    type UCOUNT = super::U32x8V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_set1_epi32(value as i32) }
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
        let (lo, hi) = Self::split(value);
        Self::join(Self::HalfRegister::reverse(hi), Self::HalfRegister::reverse(lo))
    }
}

impl BitshiftRegister for U32x8V3 {
    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_sllv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_srlv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_slli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_srli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_rolv_epi32x_v3(value, shifts) }
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm256_rorv_epi32x_v3(value, shifts) }
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_reverse_bits_epi32x_v3(value) }
    }

    #[inline(always)]
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_rolv_epi32x_v3(value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    #[inline(always)]
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm256_rorv_epi32x_v3(value, arch::_mm256_set1_epi32(shift as i32)) }
    }
}

impl ShuffleRegister for U32x8V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm256_castps_si256(arch::_mm256_shuffle_ps(
                arch::_mm256_castsi256_ps(lhs),
                arch::_mm256_castsi256_ps(rhs),
                IMM8,
            ))
        }

        // unsafe {
        //     arch::_mm256_blend_epi32(
        //         arch::_mm256_shuffle_epi32(lhs, IMM8),
        //         arch::_mm256_shuffle_epi32(rhs, IMM8),
        //         0xF0,
        //     )
        // }
    }
}

impl PermuteRegister for U32x8V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_shuffle_epi32(value, IMM8) }
    }
}

impl SwizzleRegister for U32x8V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        unsafe {
            arch::_mm256_castps_si256(arch::_mm256_permutevar_ps(
                arch::_mm256_castsi256_ps(value),
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
    //         arch::_mm256_blend_epi16(
    //             arch::_mm256_shuffle_epi32(a, AIMM8),
    //             arch::_mm256_shuffle_epi32(b, BIMM8),
    //             BLEND,
    //         )
    //     }
    // }
}

impl MaskRegister for U32x8V3 {
    const FALSY: Self::Storage = reg::<Self, 8>([0; 8]);
    const TRUTHY: Self::Storage = reg::<Self, 8>([!0; 8]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_cvtboolx8_to_epi32_mask_v3(value) }
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

impl PartialOrdRegister for U32x8V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpgt_epu32x_v3(rhs, lhs) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpgt_epu32x_v3(rhs, lhs) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpgt_epu32x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpgt_epu32x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmpeq_epi32(lhs, rhs) }
    }
}

impl NumericRegister for U32x8V3 {
    const ZERO: Self::Storage = reg::<Self, 8>([0; 8]);
    const ONE: Self::Storage = reg::<Self, 8>([1; 8]);
    const TWO: Self::Storage = reg::<Self, 8>([2; 8]);

    const MIN: Self::Storage = reg::<Self, 8>([u32::MIN; 8]);
    const MAX: Self::Storage = reg::<Self, 8>([u32::MAX; 8]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_min_epu32 _mm_min_epu32) as u32
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_max_epu32 _mm_max_epu32) as u32
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U32)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as u32))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_add_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sub_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_mullo_epi32(lhs, rhs) }
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
        unsafe { arch::_mm256_min_epu32(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_max_epu32(lhs, rhs) }
    }
}

impl IntegerRegister for U32x8V3 {
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
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    #[inline(always)]
    fn div_branched(value: Self::Storage, divider: crate::divider::Divider<Self::Element>) -> Self::Storage {
        unsafe { arch::_mm256_div_epu32x(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn div_branchfree(
        value: Self::Storage,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Self::Storage {
        unsafe { arch::_mm256_div_epu32x_bf(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_popcnt_epi32x_v3(value) }
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        // treat as unsigned
        super::I32x8V3::trailing_zeros(value)
    }
}

impl UnsignedIntegerRegister for U32x8V3 {
    #[inline(always)]
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_np2_m1_epu32x_v3(value) }
    }

    #[inline(always)]
    fn is_power_of_two(value: Self::Storage) -> Self::Storage {
        unsafe {
            // f = (v & (v - 1)) == 0
            arch::_mm256_cmpeq_epi32(
                value,
                arch::_mm256_and_si256(value, arch::_mm256_sub_epi32(value, arch::_mm256_set1_epi32(1))),
            )
        }
    }

    #[inline(always)]
    fn parity(mut value: Self::Storage) -> Self::Storage {
        unsafe {
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi32(value, 16));
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi32(value, 8));
            value = arch::_mm256_xor_si256(value, arch::_mm256_srli_epi32(value, 4));
            value = arch::_mm256_and_si256(value, arch::_mm256_set1_epi32(0x0F));

            arch::_mm256_and_si256(
                arch::_mm256_srlv_epi32(arch::_mm256_set1_epi32(0x6996), value),
                arch::_mm256_set1_epi32(1),
            )
        }
    }
}

impl CastRegister<U32x8V3> for DoublePumpRegister<super::U64x4V3> {
    fn cast_from(value: <U32x8V3 as Register>::Storage) -> Self::Storage {
        let (lo, hi) = U32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepu32_epi64(lo);
            let hi = arch::_mm256_cvtepu32_epi64(hi);

            Self::join(lo, hi)
        }
    }
}
