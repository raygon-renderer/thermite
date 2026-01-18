use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, CoreRegister, IntegerRegister, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister,
        dp::DoublePumpRegister, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I32x8V3;

impl CoreRegister for I32x8V3 {
    type Lanes = typenum::U8;
    type Element = i32;
    type Storage = arch::__m256i;
}

impl Register for I32x8V3 {
    type HalfRegister = super::I32x4V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type ISize = super::I32x8V3;
    type USize = super::U32x8V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_epi32(value, 0, 0, 0, 0, 0, 0, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi32(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_stream_load_si256(ptr as _) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_si256(ptr as _, value) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };

        (lo, hi)
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    const FALSY: Storage<Self> = reg::<Self, 8>([0; 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([-1; 8]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtboolx8_to_epi32_mask_v3(value) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_ps(arch::_mm256_castsi256_ps(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_ps(arch::_mm256_castsi256_ps(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        let (lo, hi) = Self::split(value);
        Self::join(Self::HalfRegister::reverse(hi), Self::HalfRegister::reverse(lo))
    }

    const HAS_SIMPLE_UNPACK: bool = false;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // 1. Group 128-bit lanes (Integer Domain)
            // t1 = [a_lo, b_lo]
            let t1 = arch::_mm256_permute2x128_si256(a, b, 0x20);
            // t2 = [a_hi, b_hi]
            let t2 = arch::_mm256_permute2x128_si256(a, b, 0x31);

            // 2. Shuffle locally using Float instructions
            // We cast to f32 to access _mm256_shuffle_ps.
            // This is a "zero-cost" cast (reinterpretation), though on some older
            // CPUs it might incur a 1-cycle domain crossing penalty, which is
            // still cheaper than the 3-4 integer instructions needed to replace it.
            let t1_ps = arch::_mm256_castsi256_ps(t1);
            let t2_ps = arch::_mm256_castsi256_ps(t2);

            // Mask 0x88 (10 00 10 00): Selects indices 0, 2 from both inputs
            // Effectively gathers all even indices (the 'a's)
            let res_a_ps = arch::_mm256_shuffle_ps(t1_ps, t2_ps, 0x88);

            // Mask 0xDD (11 01 11 01): Selects indices 1, 3 from both inputs
            // Effectively gathers all odd indices (the 'b's)
            let res_b_ps = arch::_mm256_shuffle_ps(t1_ps, t2_ps, 0xDD);

            // 3. Cast back to Integer
            (arch::_mm256_castps_si256(res_a_ps), arch::_mm256_castps_si256(res_b_ps))
        }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi32x_v3(value) }
    }
}

impl BitshiftRegister for I32x8V3 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi32(value, IMM8) }
    }
}

impl ShuffleRegister for I32x8V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm256_castps_si256(arch::_mm256_shuffle_ps(
                arch::_mm256_castsi256_ps(lhs),
                arch::_mm256_castsi256_ps(rhs),
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

impl PermuteRegister for I32x8V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_shuffle_epi32(value, IMM8) }
    }
}

impl SwizzleRegister for I32x8V3 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            arch::_mm256_castps_si256(arch::_mm256_permutevar_ps(
                arch::_mm256_castsi256_ps(value),
                core::mem::transmute(idxs),
            ))
        }
    }

    // #[inline(always)]
    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     a: Storage<Self>,
    //     b: Storage<Self>,
    // ) -> Storage<Self> {
    //     unsafe {
    //         arch::_mm256_blend_epi16(
    //             arch::_mm256_shuffle_epi32(a, AIMM8),
    //             arch::_mm256_shuffle_epi32(b, BIMM8),
    //             BLEND,
    //         )
    //     }
    // }
}

impl PartialOrdRegister for I32x8V3 {
    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi32(lhs, rhs) }
    }
}

impl NumericRegister for I32x8V3 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([i32::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([i32::MAX; 8]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_min_epi32 _mm_min_epi32)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_max_epi32 _mm_max_epi32)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i32))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epi32(lhs, rhs) }
    }
}

impl SignedRegister for I32x8V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1; 8]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sign_epi32(value, Self::NEG_ONE) }
    }

    #[inline(always)]
    fn is_negative(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi32::<31>(value) }
    }

    #[inline(always)]
    fn is_positive(value: Storage<Self>) -> Storage<Self> {
        Self::not(Self::is_negative(value))
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_abs_epi32(value) }
    }

    #[inline(always)]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // sign_epi32 negates if b is negative, but also sets lhs to zero
        // if rhs is zero, so we OR it with 1 to prevent that behavior
        unsafe { arch::_mm256_sign_epi32(lhs, arch::_mm256_or_si256(rhs, arch::_mm256_set1_epi32(1))) }
    }

    #[inline(always)]
    fn signum(value: Storage<Self>) -> Storage<Self> {
        // same thing as above, but negating 1 instead of an input value
        unsafe {
            arch::_mm256_sign_epi32(
                arch::_mm256_set1_epi32(1),
                arch::_mm256_or_si256(value, arch::_mm256_set1_epi32(1)),
            )
        }
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<31>(mask))
    }
}

impl IntegerRegister for I32x8V3 {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullhi_epi32x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epi32x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epi32x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32) as i32
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32) as i32
    }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[inline(always)]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_popcnt_epi32x_v3(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        // treat as unsigned
        super::U32x8V3::leading_zeros(value)
    }

    #[inline(always)]
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }

    #[inline(always)]
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

impl SignedIntegerRegister for I32x8V3 {
    #[inline(always)]
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sra_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn srav(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm256_srav_epi32(value, shifts) }
    }
}

impl CastRegister<I32x8V3> for DoublePumpRegister<super::I64x4V3> {
    #[inline(always)]
    fn cast_from(value: Storage<I32x8V3>) -> Storage<Self> {
        let (lo, hi) = I32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepi32_epi64(lo);
            let hi = arch::_mm256_cvtepi32_epi64(hi);

            DoublePumpRegister::join(lo, hi)
        }
    }
}
