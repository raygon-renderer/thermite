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
        IntegerRegister, InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
        array::ArrayRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2V1;

#[thermite_macros::inline_always]
impl CoreRegister for U64x2V1 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V1;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8x_v1(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }

    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I64x2V1::zeroupper_z::<Z>(value) // just reuse the i64 version since it's the same mask
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U64x2V1 {
    const FALSY: Storage<Self> = reg::<Self, 2>([0; 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([!0; 2]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_cvtboolx2_to_epi64_mask_v1(value) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0xFFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U64x2V1 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<u64> for U64x2V1 {
    fn concat(lo: Storage<u64>, hi: Storage<u64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(lo as i64, hi as i64) }
    }

    fn split(value: Storage<Self>) -> (Storage<u64>, Storage<u64>) {
        unsafe {
            let mut arr = [0; 2];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (arr[0], arr[1])
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u64> for U64x2V1 {
    fn extend(value: Storage<u64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value as i64, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<u64> {
        unsafe { arch::_mm_cvtsi128_si64(value) as u64 }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U64x2V1 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }
}

#[thermite_macros::inline_always]
impl Register for U64x2V1 {
    type Element = u64;

    type Signed = super::I64x2V1;
    type Unsigned = super::U64x2V1;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        super::I64x2V1::is_negative(value) // reuse signed implementation
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value as i64, 0) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi64x(value as i64) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        // no non-temporal load before SSE4.1 (`movntdqa`); a regular aligned load is the best SSE2 can do
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE_R!(2, 3, 0, 1) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi64x_v1(value) }
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U64x2V1 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi64x_v1(value, shifts) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi64x_v1(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi64(value, IMM8) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi64(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U64x2V1 {
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

#[thermite_macros::inline_always]
impl SwizzleRegister for U64x2V1 {
    const HAS_PERMUTEV: bool = false;
}

impl PartialOrdRegister for U64x2V1 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu64x_v1(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi64x_v1(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U64x2V1 {
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

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        _mm_pairwise_sum_epi64_v1!(lo, hi)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v1(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epu64x_v1(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu64x_v1(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U64x2V1 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullhi_epu64x_v1(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v1(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

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

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi64x_v1(value) }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(64), Self::ilog2p1(value))
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I64x2V1::trailing_zeros(value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U64x2V1 {}

impl CastRegister<ArrayRegister<U64x2V1, 2>> for super::U32x4V1 {
    fn cast_from(value: Storage<ArrayRegister<U64x2V1, 2>>) -> Storage<Self> {
        let (lo, hi) = <ArrayRegister<U64x2V1, 2> as ConcatRegister<U64x2V1>>::split(value);
        unsafe { arch::_mm_cvtepi64_epi32x_v1(lo, hi) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<<Scalar as Simd>::u32x2> for U64x2V1 {
    fn cast_from(value: Storage<<Scalar as Simd>::u32x2>) -> Storage<Self> {
        // zero-extend the two u32 lanes into the low dword of each u64 lane
        unsafe { arch::_mm_setr_epi32(value.0[0] as i32, 0, value.0[1] as i32, 0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U64x2V1> for <Scalar as Simd>::u32x2 {
    fn cast_from(value: Storage<U64x2V1>) -> Storage<<Scalar as Simd>::u32x2> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtsi128_si32(value) as u32,          // lowest 32 bits
                arch::_mm_extract_epi32x_v1::<2>(value) as u32, // next 32 bits after 64 bits
            ])
        }
    }
}
