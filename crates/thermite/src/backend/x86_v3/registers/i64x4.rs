use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        IndexableRegister, IntegerRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister,
        ZeroUpper, dp::DoublePumpRegister, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x4V3;

impl CoreRegister for I64x4V3 {
    type Lanes = typenum::U4;
    type Storage = arch::__m256i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(on_false, on_true, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(mask, value) }
    }

    #[inline(always)]
    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 2 } {
            unsafe { arch::_mm256_zextsi128_si256(arch::_mm256_castsi256_si128(value)) }
        } else {
            Self::EMPTY // N == 0, so zero everything
        }
    }
}

impl MaskRegister for I64x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([-1; 4]);

    #[inline(always)]
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    #[inline(always)]
    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtboolx4_to_epi64_mask_v3(value) }
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
        Some(unsafe { arch::_mm256_movemask_pd(arch::_mm256_castsi256_pd(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_pd(arch::_mm256_castsi256_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::bitand_z]
impl BitwiseRegister for I64x4V3 {
    #[masked] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    #[masked] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    #[masked] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }

    #[masked] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    #[masked] fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }
}

impl ConcatRegister<super::I64x2V3> for I64x4V3 {
    #[inline(always)]
    fn concat(lo: Storage<super::I64x2V3>, hi: Storage<super::I64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<super::I64x2V3>, Storage<super::I64x2V3>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };
        (lo, hi)
    }
}

impl ExtendRegister<super::I64x2V3> for I64x4V3 {
    #[inline(always)]
    fn extend(value: Storage<super::I64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<super::I64x2V3> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

impl Register for I64x4V3 {
    type Element = i64;

    type Signed = super::I64x4V3;
    type Unsigned = super::U64x4V3;

    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)]
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    #[inline(always)]
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    #[inline(always)]
    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }

    #[inline(always)]
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::is_negative(value)
    }

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set_epi64x(0, 0, 0, value) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi64x(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    #[inline(always)]
    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskload_epi64(ptr, mask) }
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
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_epi64::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    const HAS_SIMPLE_UNPACK: bool = false;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // 1. Unpack to de-interleave locally within 128-bit lanes
            // v0 becomes [a0, a2, a1, a3]
            let v0 = arch::_mm256_unpacklo_epi64(a, b);
            // v1 becomes [b0, b2, b1, b3]
            let v1 = arch::_mm256_unpackhi_epi64(a, b);

            // 2. Permute to cross lanes and fix the order
            // We want indices 0, 2, 1, 3 (swap middle two elements)
            // _MM_SHUFFLE(3, 1, 2, 0) = 0b11_01_10_00 = 0xD8
            let real_lo = arch::_mm256_permute4x64_epi64(v0, 0xD8);
            let real_hi = arch::_mm256_permute4x64_epi64(v1, 0xD8);

            (real_lo, real_hi)
        }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi64x_v3(value) }
    }
}

impl SwizzleRegister for I64x4V3 {
    const HAS_PERMUTEV: bool = false;
}

impl IndexableRegister<super::U64x4V3> for I64x4V3 {
    #[inline(always)]
    unsafe fn gather(ptr: *const <I64x4V3 as Register>::Element, indices: Storage<super::U64x4V3>) -> Storage<I64x4V3> {
        unsafe { arch::_mm256_i64gather_epi64::<8>(ptr as *const _, indices) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<I64x4V3>,
        mask: Storage<<I64x4V3 as CoreRegister>::Mask>,
        ptr: *const <I64x4V3 as Register>::Element,
        indices: Storage<super::U64x4V3>,
    ) -> Storage<I64x4V3> {
        unsafe { arch::_mm256_mask_i64gather_epi64::<8>(src, ptr as *const _, indices, mask) }
    }
}

impl IndexableRegister<super::U32x4V3> for I64x4V3 {
    #[inline(always)]
    unsafe fn gather(ptr: *const <I64x4V3 as Register>::Element, indices: Storage<super::U32x4V3>) -> Storage<I64x4V3> {
        unsafe { arch::_mm256_i32gather_epi64::<8>(ptr as *const _, indices) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<I64x4V3>,
        mask: Storage<<I64x4V3 as CoreRegister>::Mask>,
        ptr: *const <I64x4V3 as Register>::Element,
        indices: Storage<super::U32x4V3>,
    ) -> Storage<I64x4V3> {
        unsafe { arch::_mm256_mask_i32gather_epi64::<8>(src, ptr as *const _, indices, mask) }
    }
}

impl IndexableRegister<super::U32x8V3> for DoublePumpRegister<I64x4V3> {
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V3>) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_i32gather_epi64::<8>(ptr as *const _, lo);
            let hi = arch::_mm256_i32gather_epi64::<8>(ptr as *const _, hi);

            Self::concat(lo, hi)
        }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V3>,
    ) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_mask_i32gather_epi64::<8>(src.0, ptr as *const _, lo, mask.0);
            let hi = arch::_mm256_mask_i32gather_epi64::<8>(src.1, ptr as *const _, hi, mask.1);

            Self::concat(lo, hi)
        }
    }
}

impl BitshiftRegister for I64x4V3 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi64(value, IMM8) }
    }
}

impl PartialOrdRegister for I64x4V3 {
    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi64(lhs, rhs) }
    }
}

impl NumericRegister for I64x4V3 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([i64::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([i64::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_min_epi64x_v2 _mm_min_epi64x_v2)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_max_epi64x_v2 _mm_max_epi64x_v2)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i64))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
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
        unsafe { arch::_mm256_min_epi64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epi64x_v3(lhs, rhs) }
    }
}

#[thermite_macros::bitand_z]
impl SignedRegister for I64x4V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1; 4]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    #[masked]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm256_add_epi64(
                arch::_mm256_xor_si256(value, arch::_mm256_set1_epi64x(-1)),
                arch::_mm256_set1_epi64x(1),
            )
        }
    }

    #[inline(always)]
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_signbits_epi64x_v3(value) }
    }

    #[masked]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let sign = arch::_mm256_signbits_epi64x_v3(value);
            arch::_mm256_xor_si256(sign, arch::_mm256_add_epi64(value, sign))
        }
    }

    #[masked]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_copysign_epi64x_v3(lhs, rhs) }
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(Self::NEG_ONE, Self::ONE, arch::_mm256_cmpgt_epi64(value, Self::NEG_ONE)) }
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<63>(mask))
    }
}

impl IntegerRegister for I64x4V3 {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullhi_epi64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epi64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epi64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2)
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
        unsafe { arch::_mm256_popcnt_epi64x_v3(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        use crate::register::UnsignedIntegerRegister;

        Self::sub(Self::splat(64), super::U64x4V3::ilog2p1(value))
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

#[thermite_macros::bitand_z]
impl SignedIntegerRegister for I64x4V3 {
    #[masked]
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi64x_v3(value, IMM8) }
    }

    #[masked]
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi64x_v3(value, shift as i32) }
    }

    #[masked]
    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srav_epi64x_v3(value, shifts) }
    }
}

impl CastRegister<DoublePumpRegister<I64x4V3>> for super::I32x8V3 {
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<I64x4V3>>) -> Storage<Self> {
        let (lo, hi) = <DoublePumpRegister<I64x4V3>>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepi64_epi32_v3(lo);
            let hi = arch::_mm256_cvtepi64_epi32_v3(hi);

            arch::_mm256_setr_m128i(lo, hi)
        }
    }
}
