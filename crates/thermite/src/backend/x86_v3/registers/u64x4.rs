use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element,
        ExtendRegister, IndexableRegister, IntegerRegister, MaskElement, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        UnsignedIntegerRegister, dp::DoublePumpRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x4V3;

impl CoreRegister for U64x4V3 {
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
    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I64x4V3::zeroupper_z::<Z>(value)
    }
}

impl MaskRegister for U64x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([!0; 4]);

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
impl BitwiseRegister for U64x4V3 {
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

impl ConcatRegister<super::U64x2V3> for U64x4V3 {
    #[inline(always)]
    fn concat(lo: Storage<super::U64x2V3>, hi: Storage<super::U64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<super::U64x2V3>, Storage<super::U64x2V3>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };

        (lo, hi)
    }
}

impl ExtendRegister<super::U64x2V3> for U64x4V3 {
    #[inline(always)]
    fn extend(value: Storage<super::U64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<super::U64x2V3> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

impl Register for U64x4V3 {
    type Element = u64;

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
        super::I64x4V3::is_negative(value) // reuse signed implementation
    }

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_epi64x(value as i64, 0, 0, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi64x(value as i64) }
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
        unsafe { arch::_mm256_maskload_epi64(ptr as *const _, mask) }
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

    const HAS_SIMPLE_UNPACK: bool = super::I64x4V3::HAS_SIMPLE_UNPACK;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I64x4V3::unpack(a, b) // reuse signed implementation
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi64x_v3(value) }
    }
}

impl SwizzleRegister for U64x4V3 {
    const HAS_PERMUTEV: bool = false;
}

impl<I> IndexableRegister<I> for U64x4V3
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
    super::I64x4V3: IndexableRegister<I>,
{
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<I>) -> Storage<Self> {
        unsafe { super::I64x4V3::gather(ptr as *const _, indices) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<I>,
    ) -> Storage<Self> {
        unsafe { super::I64x4V3::gather_m(src, mask, ptr as *const _, indices) }
    }

    #[inline(always)]
    unsafe fn gather_z(mask: Storage<Self::Mask>, ptr: *const Self::Element, indices: Storage<I>) -> Storage<Self> {
        unsafe { super::I64x4V3::gather_z(mask, ptr as *const _, indices) }
    }
}

impl IndexableRegister<super::U32x8V3> for DoublePumpRegister<U64x4V3> {
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

#[thermite_macros::bitand_z]
impl BitshiftRegister for U64x4V3 {
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

impl PartialOrdRegister for U64x4V3 {
    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epu64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi64(lhs, rhs) }
    }
}

#[thermite_macros::bitand_z]
impl NumericRegister for U64x4V3 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([u64::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([u64::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_min_epu64x_v2 _mm_min_epu64x_v2) as u64
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_max_epu64x_v2 _mm_max_epu64x_v2) as u64
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    #[masked]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi64(lhs, rhs) }
    }

    #[masked]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi64(lhs, rhs) }
    }

    #[masked]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
    }

    #[masked]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[masked]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[masked]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu64x_v3(lhs, rhs) }
    }

    #[masked]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu64x_v3(lhs, rhs) }
    }
}

impl IntegerRegister for U64x4V3 {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullhi_epu64x_v3(lhs, rhs) }
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        todo!("arch::_mm_mullo_epu64x_v2(lhs, rhs)")
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
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
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I64x4V3::count_ones(value)
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

impl UnsignedIntegerRegister for U64x4V3 {}

impl CastRegister<DoublePumpRegister<U64x4V3>> for super::U32x8V3 {
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<U64x4V3>>) -> Storage<Self> {
        let (lo, hi) = DoublePumpRegister::<U64x4V3>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepi64_epi32_v3(lo);
            let hi = arch::_mm256_cvtepi64_epi32_v3(hi);

            arch::_mm256_setr_m128i(lo, hi)
        }
    }
}
