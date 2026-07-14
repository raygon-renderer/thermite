//! Native 256-bit unsigned 16-bit register for x86-v3 (AVX2). Native-width unsigned 16-bit.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        IntegerRegister, InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SaturatingCastRegister, Storage, UnsignedIntegerRegister, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x16V3;

#[thermite_macros::inline_always]
impl CoreRegister for U16x16V3 {
    type Lanes = typenum::U16;
    type Storage = arch::__m256i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::X86V3;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(mask, value) }
    }

    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I16x16V3::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::U16x8V3> for U16x16V3 {
    fn concat(lo: Storage<super::U16x8V3>, hi: Storage<super::U16x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U16x8V3>, Storage<super::U16x8V3>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::U16x8V3> for U16x16V3 {
    fn extend(value: Storage<super::U16x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U16x8V3> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U16x16V3 {
    const FALSY: Storage<Self> = reg::<Self, 16>([0; 16]);
    const TRUTHY: Storage<Self> = reg::<Self, 16>([!0; 16]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) == 0 }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        unsafe {
            let packed = arch::_mm256_packs_epi16(value, arch::_mm256_setzero_si256());
            let fixed = arch::_mm256_permute4x64_epi64(packed, 0b00_00_10_00);
            let lo128 = arch::_mm256_castsi256_si128(fixed);
            Some((arch::_mm_movemask_epi8(lo128) as u64) & 0xFFFF)
        }
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U16x16V3 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u16> for U16x16V3 {
    fn extend(value: Storage<u16>) -> Storage<Self> {
        let mut arr = [0u16; 16];
        arr[0] = value;
        unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
    }

    fn narrow(value: Storage<Self>) -> Storage<u16> {
        Self::as_slice(&value)[0]
    }
}

#[thermite_macros::inline_always]
impl Register for U16x16V3 {
    type Element = u16;

    type Signed = super::I16x16V3;
    type Unsigned = super::U16x16V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_srai_epi16(value, 15) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        let mut arr = [0u16; 16];
        arr[0] = value;
        unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi16(value as i16) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_stream_load_si256(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_si256(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        super::I16x16V3::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi16x_v3(value) }
    }

    const HAS_PERMUTEV: bool = <super::I16x16V3 as Register>::HAS_PERMUTEV;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        super::I16x16V3::permutev(value, idxs)
    }

    compress_via_wide!();

    // Byte-identical to the signed register (both raw `__m256i`); reuse it.
    fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        super::I16x16V3::align::<OFFSET>(a, b)
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U16x16V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x16V3::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x16V3::deinterleave(a, b)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U16x16V3 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bslli_epi128(value, IMM8) }
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bsrli_epi128(value, IMM8) }
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi16(value, IMM8) }
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi16(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U16x16V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epu16x_v3(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epu16x_v3(rhs, lhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U16x16V3 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([u16::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([u16::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_min_epu16) as u16
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_max_epu16) as u16
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_add_epi16) as u16
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_mullo_epi16) as u16
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u16))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi16(lhs, rhs) }
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi16(lhs, rhs) }
    }
    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi16(lhs, arch::_mm256_and_si256(rhs, mask)) }
    }
    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi16(lhs, arch::_mm256_and_si256(rhs, mask)) }
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi16(lhs, rhs) }
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu16(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U16x16V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhi_epu16(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi16(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epu16(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epu16(lhs, rhs) }
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
        unsafe { arch::_mm256_popcnt_epi16x_v3(value) }
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(16), Self::ilog2p1(value))
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I16x16V3::trailing_zeros(value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U16x16V3 {
    /// 2D Morton via a 256-bit PSHUFB nibble-LUT; every other `N` uses the
    /// generic cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm256_morton2_epu16x_v3(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    /// 2D Morton decode via the 256-bit PSHUFB compress; other `N` use the cascade.
    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            unsafe {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::_mm256_morton2_compress_epu16x_v3(code),
                    arch::_mm256_morton2_compress_epu16x_v3(arch::_mm256_srli_epi16(code, 1)),
                )
            }
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }
}

// Widen u16x16 -> u32x16 (= ArrayRegister<U32x8V3, 2>): zero-extend the two 128-bit halves.
#[thermite_macros::inline_always]
impl CastRegister<U16x16V3> for ArrayRegister<super::U32x8V3, 2> {
    fn cast_from(value: Storage<U16x16V3>) -> Storage<Self> {
        let (lo, hi) = U16x16V3::split(value);
        unsafe { ArrayRegister([arch::_mm256_cvtepu16_epi32(lo), arch::_mm256_cvtepu16_epi32(hi)]) }
    }
}

// Narrow u32x16 -> u16x16: truncate each 32-bit lane, then concat the two halves.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x8V3, 2>> for U16x16V3 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x8V3, 2>>) -> Storage<Self> {
        let lo = <super::U16x8V3 as CastRegister<super::U32x8V3>>::cast_from(value.0[0]);
        let hi = <super::U16x8V3 as CastRegister<super::U32x8V3>>::cast_from(value.0[1]);
        <Self as ConcatRegister<super::U16x8V3>>::concat(lo, hi)
    }
}

// Saturating narrow u32x16 -> u16x16: clamp each half (`vpminud`) then a two-source `vpackusdw` +
// `vpermq` restitch (the `_u` pack reads a signed source, so the clamp keeps lanes in range).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U32x8V3, 2>> for U16x16V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x8V3, 2>>) -> Storage<Self> {
        unsafe {
            let max = arch::_mm256_set1_epi32(0xFFFF);
            let a = arch::_mm256_min_epu32(value.0[0], max);
            let b = arch::_mm256_min_epu32(value.0[1], max);
            arch::_mm256_permute4x64_epi64(arch::_mm256_packus_epi32(a, b), 0b11_01_10_00)
        }
    }
}

// Saturating narrow u64x16 -> u16x16: clamp down to u32x16 then the pack above.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x4V3, 4>> for U16x16V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x4V3, 4>>) -> Storage<Self> {
        let words = <ArrayRegister<super::U32x8V3, 2> as SaturatingCastRegister<ArrayRegister<super::U64x4V3, 4>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<ArrayRegister<super::U32x8V3, 2>>>::saturating_cast_from(words)
    }
}
