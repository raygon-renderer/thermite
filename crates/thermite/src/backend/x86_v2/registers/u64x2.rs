use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        IntegerRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, array::ArrayRegister,
        empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2V2;

impl CoreRegister for U64x2V2 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V2;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8(lhs, rhs, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }

    #[inline(always)]
    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I64x2V2::zeroupper_z::<Z>(value) // just reuse the i64 version since it's the same mask
    }
}

impl MaskRegister for U64x2V2 {
    const FALSY: Storage<Self> = reg::<Self, 2>([0; 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([!0; 2]);

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
        unsafe { arch::_mm_cvtboolx2_to_epi64_mask_v2(value) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0xFFFF }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::bitand_z]
impl BitwiseRegister for U64x2V2 {
    #[masked] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    #[masked] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    #[masked] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    #[masked] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    #[masked] fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }
}

impl ConcatRegister<u64> for U64x2V2 {
    fn concat(lo: Storage<u64>, hi: Storage<u64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(lo as i64, hi as i64) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<u64>, Storage<u64>) {
        unsafe {
            let mut arr = [0; 2];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (arr[0], arr[1])
        }
    }
}

impl ExtendRegister<u64> for U64x2V2 {
    #[inline(always)]
    fn extend(value: Storage<u64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value as i64, 0) }
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<u64> {
        unsafe { arch::_mm_cvtsi128_si64(value) as u64 }
    }
}

impl Register for U64x2V2 {
    type Element = u64;

    type Signed = super::I64x2V2;
    type Unsigned = super::U64x2V2;

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
        super::I64x2V2::is_negative(value) // reuse signed implementation
    }

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value as i64, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi64x(value as i64) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE_R!(2, 3, 0, 1) }>(value) }
    }

    #[inline(always)]
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    #[inline(always)]
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi64x_v2(value) }
    }

    #[inline(always)]
    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

#[rustfmt::skip] #[thermite_macros::bitand_z]
impl BitshiftRegister for U64x2V2 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    #[masked] fn bshli<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }

    #[masked] fn bshri<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }

    #[masked] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[masked] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[masked] fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi64x_v1(value, shifts) }
    }

    #[masked] fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi64x_v1(value, shifts) }
    }

    #[masked] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi64(value, IMM8) }
    }

    #[masked] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi64(value, IMM8) }
    }
}

impl ShuffleRegister for U64x2V2 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castpd_si128(arch::_mm_shuffle_pd(
                arch::_mm_castsi128_pd(lhs),
                arch::_mm_castsi128_pd(rhs),
                IMM8,
            ))
        }
    }
}

// impl PermuteRegister for U64x2V2 {
//     fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
//         unsafe { arch::_mm_permute_epi64x(value, IMM8) }
//     }
// }

impl SwizzleRegister for U64x2V2 {
    const HAS_PERMUTEV: bool = false;
}

impl PartialOrdRegister for U64x2V2 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu64x_v2(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi64x_v1(lhs, rhs) }
    }
}

#[thermite_macros::bitand_z]
impl NumericRegister for U64x2V2 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([u64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([u64::MAX; 2]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.min(hi)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.max(hi)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    #[masked]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    #[masked]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    #[masked]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
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
        unsafe { arch::_mm_min_epu64x_v2(lhs, rhs) }
    }

    #[masked]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu64x_v2(lhs, rhs) }
    }
}

#[thermite_macros::bitand_z]
impl IntegerRegister for U64x2V2 {
    #[masked]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullhi_epu64x_v1(lhs, rhs) }
    }

    #[masked]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
    }

    #[masked]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

    #[masked]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    #[masked]
    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[masked]
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[masked]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[masked]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi64x_v2(value) }
    }

    #[masked]
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    #[masked]
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[masked]
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I64x2V2::count_ones(value)
    }
}

impl UnsignedIntegerRegister for U64x2V2 {}

impl CastRegister<ArrayRegister<U64x2V2, 2>> for super::U32x4V2 {
    #[inline(always)]
    fn cast_from(value: Storage<ArrayRegister<U64x2V2, 2>>) -> Storage<Self> {
        let (lo, hi) = <ArrayRegister<U64x2V2, 2> as ConcatRegister<U64x2V2>>::split(value);
        unsafe { arch::_mm_cvtepi64_epi32x_v2(lo, hi) }
    }
}

impl CastRegister<<Scalar as Simd>::u32x2> for U64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<<Scalar as Simd>::u32x2>) -> Storage<Self> {
        // zero-extend lower two u32 lanes to u64 lanes
        unsafe { arch::_mm_setr_epi32(0, value.0[0] as i32, 0, value.0[1] as i32) }
    }
}

impl CastRegister<U64x2V2> for <Scalar as Simd>::u32x2 {
    #[inline(always)]
    fn cast_from(value: Storage<U64x2V2>) -> Storage<<Scalar as Simd>::u32x2> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtsi128_si32(value) as u32,      // lowest 32 bits
                arch::_mm_extract_epi32::<2>(value) as u32, // next 32 bits after 64 bits
            ])
        }
    }
}
