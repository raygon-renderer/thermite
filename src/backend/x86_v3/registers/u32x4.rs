use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::register::{
    IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShiftRegister,
    ShuffleRegister, SwizzleRegister, UnsignedIntegerRegister, empty_reg, reg,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct U32x4V3;

impl Register for U32x4V3 {
    type Lanes = typenum::U4;

    type Element = u32;
    type Storage = arch::__m128i;
    type HalfRegister = ();
    type DoubleRegister = super::U32x8V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_epi32(value as i32) }
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
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm_sllv_epi32(value, core::mem::transmute(shifts.into())) }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm_srlv_epi32(value, core::mem::transmute(shifts.into())) }
    }
}

impl ShiftRegister for U32x4V3 {
    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_slli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_srli_epi32(value, IMM8) }
    }
}

impl ShuffleRegister for U32x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_shuffle_ps(
                arch::_mm_castsi128_ps(lhs),
                arch::_mm_castsi128_ps(rhs),
                IMM8,
            ))
        }

        // unsafe {
        //     arch::_mm_blend_epi32(
        //         arch::_mm_shuffle_epi32(lhs, IMM8),
        //         arch::_mm_shuffle_epi32(rhs, IMM8),
        //         0xF0,
        //     )
        // }
    }
}

impl PermuteRegister for U32x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_epi32(value, IMM8) }
    }
}

impl SwizzleRegister for U32x4V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_permutevar_ps(
                arch::_mm_castsi128_ps(value),
                core::mem::transmute(idxs.into()),
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

impl MaskRegister for U32x4V3 {
    const FALSY: Self::Storage = reg::<Self, 4>([0; 4]);
    const TRUTHY: Self::Storage = reg::<Self, 4>([!0; 4]);

    #[inline(always)]
    fn new_mask(value: impl Into<GenericArray<bool, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm_cvtboolx4_to_epi32_mask_v2(value.into()) }
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

impl PartialOrdRegister for U32x4V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmplt_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmple_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpgt_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpge_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpeq_epi32(lhs, rhs) }
    }
}

impl NumericRegister for U32x4V3 {
    const ZERO: Self::Storage = reg::<Self, 4>([0; 4]);
    const ONE: Self::Storage = reg::<Self, 4>([1; 4]);
    const TWO: Self::Storage = reg::<Self, 4>([2; 4]);

    const MIN: Self::Storage = reg::<Self, 4>([u32::MIN; 4]);
    const MAX: Self::Storage = reg::<Self, 4>([u32::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_min_epu32 _mm_min_epu32) as u32
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_max_epu32 _mm_max_epu32) as u32
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
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
        unsafe { arch::_mm_min_epu32(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_epu32(lhs, rhs) }
    }
}

impl IntegerRegister for U32x4V3 {
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
        _mm_reduce_epi32!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        _mm_reduce_epi32!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi32x_v3(value, core::mem::transmute(shifts.into())) }
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi32x_v3(value, core::mem::transmute(shifts.into())) }
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_reverse_bits_epi32x_v2(value) }
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_popcnt_epi32x_v2(value) }
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        // treat as unsigned
        super::I32x4V3::trailing_zeros(value)
    }

    #[inline(always)]
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi32x_v3(value, arch::_mm_set1_epi32(shift as i32)) }
    }

    #[inline(always)]
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi32x_v3(value, arch::_mm_set1_epi32(shift as i32)) }
    }
}

impl UnsignedIntegerRegister for U32x4V3 {
    #[inline(always)]
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_np2_m1_epu32x_v2(value) }
    }

    #[inline(always)]
    fn is_power_of_two(value: Self::Storage) -> Self::Storage {
        unsafe {
            // f = (v & (v - 1)) == 0
            arch::_mm_cmpeq_epi32(
                value,
                arch::_mm_and_si128(value, arch::_mm_sub_epi32(value, arch::_mm_set1_epi32(1))),
            )
        }
    }

    #[inline(always)]
    fn parity(mut value: Self::Storage) -> Self::Storage {
        unsafe {
            value = arch::_mm_xor_si128(value, arch::_mm_srli_epi32(value, 16));
            value = arch::_mm_xor_si128(value, arch::_mm_srli_epi32(value, 8));
            value = arch::_mm_xor_si128(value, arch::_mm_srli_epi32(value, 4));
            value = arch::_mm_and_si128(value, arch::_mm_set1_epi32(0x0F));

            arch::_mm_and_si128(
                arch::_mm_srlv_epi32(arch::_mm_set1_epi32(0x6996), value),
                arch::_mm_set1_epi32(1),
            )
        }
    }
}
