//! Native 128-bit signed 16-bit register for x86-v3. Identical SSE intrinsics to the v2
//! `I16x8V2`; it exists separately so it can serve as the 128-bit half of the 256-bit
//! `I16x16V3` (the native-width 16-bit register on AVX2).

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, Element, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SaturatingCastRegister, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister,
        empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I16x8V3;

#[thermite_macros::inline_always]
impl CoreRegister for I16x8V3 {
    type Lanes = typenum::U8;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::X86V3;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            let mut arr = [0i16; 8];
            unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 8 - Z::N };
            while i < 8 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::_mm_loadu_si128(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I16x8V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi16(a, b), arch::_mm_unpackhi_epi16(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // One `pshufb` per input groups even words low / odd words high, then `unpacklo/hi`
        // merges: evens = [a0 a2 a4 a6 | b0 b2 b4 b6], odds = [a1 a3 a5 a7 | b1 b3 b5 b7].
        unsafe {
            let shuf = arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
            let a_s = arch::_mm_shuffle_epi8(a, shuf);
            let b_s = arch::_mm_shuffle_epi8(b, shuf);
            (arch::_mm_unpacklo_epi64(a_s, b_s), arch::_mm_unpackhi_epi64(a_s, b_s))
        }
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I16x8V3 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 8>([0; 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([-1; 8]);

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        let packed = unsafe { arch::_mm_packs_epi16(value, arch::_mm_setzero_si128()) };
        Some(unsafe { (arch::_mm_movemask_epi8(packed) as u64) & 0xFF })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I16x8V3 {
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
impl ExtendRegister<i16> for I16x8V3 {
    fn extend(value: Storage<i16>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi16(value, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i16> {
        unsafe { arch::_mm_extract_epi16::<0>(value) as i16 }
    }
}

#[thermite_macros::inline_always]
impl Register for I16x8V3 {
    type Element = i16;

    type Signed = super::I16x8V3;
    type Unsigned = super::U16x8V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi16(value, 15) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi16(value, 0, 0, 0, 0, 0, 0, 0) }
    }

    impl_native_extract!(@epi16x8);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi16(value) }
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
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_shuffle_epi8(
                value,
                arch::_mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
            )
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi16x_v2(value) }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let p = idxs.as_ptr() as *const arch::__m128i;
            arch::_mm_permutev_epi16x_v2(value, arch::_mm_loadu_si128(p), arch::_mm_loadu_si128(p.add(1)))
        }
    }

    compress_via_table!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I16x8V3 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi16(value, IMM8) }
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi16(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I16x8V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epi16(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I16x8V3 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([i16::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([i16::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi16_v2!(value; _mm_min_epi16)
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi16_v2!(value; _mm_max_epi16)
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi16_v2!(value; _mm_add_epi16)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi16_v2!(value; _mm_mullo_epi16)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i16))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi16(lhs, rhs) }
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi16(lhs, rhs) }
    }
    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi16(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi16(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi16(lhs, rhs) }
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi16(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I16x8V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1; 8]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sign_epi16(value, Self::NEG_ONE) }
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi16(value, 15) }
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi16(value) }
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sign_epi16(arch::_mm_set1_epi16(1), value) }
    }
    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<15>(mask))
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I16x8V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epi16(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi16(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi16(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi16(lhs, rhs) }
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi16x_v2(value) }
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        super::U16x8V3::leading_zeros(value)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I16x8V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi16(value, IMM8) }
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhrs_epi16(a, b) }
    }
}

// Widen i16x8 -> i32x8 (= native I32x8V3, 256-bit): sign-extend all 8 lanes at once.
#[thermite_macros::inline_always]
impl CastRegister<I16x8V3> for super::I32x8V3 {
    fn cast_from(value: Storage<I16x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi32(value) }
    }
}

// Narrow i32x8 -> i16x8: truncate the low 16 bits of each lane (wrapping, like `as`).
#[thermite_macros::inline_always]
impl CastRegister<super::I32x8V3> for I16x8V3 {
    fn cast_from(value: Storage<super::I32x8V3>) -> Storage<Self> {
        unsafe {
            // pack the two 128-bit halves' low-16 bits; vpackssdw saturates, so mask first
            // is not needed because we then re-narrow via shuffle. Use a byte shuffle per
            // 128-bit lane to pick the low 2 bytes, then combine the two lanes.
            let lo = arch::_mm256_castsi256_si128(value);
            let hi = arch::_mm256_extracti128_si256(value, 1);
            let pick = arch::_mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
            let lo = arch::_mm_shuffle_epi8(lo, pick);
            let hi = arch::_mm_shuffle_epi8(hi, pick);
            arch::_mm_unpacklo_epi64(lo, hi)
        }
    }
}

// Saturating narrow i32x8 -> i16x8 via `vpackssdw`. AVX2 packs interleave the two
// 128-bit lanes, so pack `(v, v)` and pull the populated 64-bit groups (positions 0
// and 2) back into sequence with `vpermq` before truncating to 128 bits.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I32x8V3> for I16x8V3 {
    fn saturating_cast_from(value: Storage<super::I32x8V3>) -> Storage<Self> {
        unsafe {
            let packed = arch::_mm256_packs_epi32(value, value);
            arch::_mm256_castsi256_si128(arch::_mm256_permute4x64_epi64(packed, 0b00_00_10_00))
        }
    }
}

// i64x8 -> i16x8: clamp down to i32x8 (no 64-bit pack), then `vpackssdw`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x4V3, 2>> for I16x8V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x4V3, 2>>) -> Storage<Self> {
        let words =
            <super::I32x8V3 as SaturatingCastRegister<ArrayRegister<super::I64x4V3, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I32x8V3>>::saturating_cast_from(words)
    }
}

// Identity casts to self and the unsigned sibling are provided by the cast macros in mod.rs.
