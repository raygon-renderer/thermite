//! Native 256-bit signed 16-bit register for x86-v3 (AVX2). This is the native-width 16-bit
//! register (`Native16Width = U16`), with the 128-bit `I16x8V3` as its half.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I16x16V3;

#[thermite_macros::inline_always]
impl CoreRegister for I16x16V3 {
    type NativeIsa = crate::backend::x86_v3::X86V3;
    type Lanes = typenum::U16;
    type Storage = arch::__m256i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
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

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 16 } {
            value
        } else if const { Z::N == 8 } {
            unsafe { arch::_mm256_zextsi128_si256(arch::_mm256_castsi256_si128(value)) }
        } else {
            let mut arr = [0i16; 16];
            unsafe { arch::_mm256_storeu_si256(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 16 - Z::N };
            while i < 16 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::I16x8V3> for I16x16V3 {
    fn concat(lo: Storage<super::I16x8V3>, hi: Storage<super::I16x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::I16x8V3>, Storage<super::I16x8V3>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::I16x8V3> for I16x16V3 {
    fn extend(value: Storage<super::I16x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::I16x8V3> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I16x16V3 {
    const FALSY: Storage<Self> = reg::<Self, 16>([0; 16]);
    const TRUTHY: Storage<Self> = reg::<Self, 16>([-1; 16]);

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

    fn count_set<const N: usize>(values: [Storage<Self>; N]) -> usize {
        unsafe { arch::_mm256_count_mask_epi16x_v3(values) }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi16x_v3(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        // Pack the 16x16-bit mask to 16x8-bit. `vpacksswb` interleaves the two 128-bit
        // lanes ([a0..a7 b0..b7 a8..a15 b8..b15] order), so permute the 64-bit groups back
        // into sequence before taking one bit per byte.
        unsafe {
            let packed = arch::_mm256_packs_epi16(value, arch::_mm256_setzero_si256());
            // packed (per 128-bit lane) = [lane_lo8 | 0]; after permute4x64 we gather the
            // two populated 64-bit halves (positions 0 and 2) into the low 128 bits.
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
impl BitwiseRegister for I16x16V3 {
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
impl Register for I16x16V3 {
    type Element = i16;

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
        let mut arr = [0i16; 16];
        arr[0] = value;
        unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
    }

    impl_native_extract!(@epi16x16);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi16(value) }
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
        let (lo, hi) = Self::split(value);
        Self::concat(super::I16x8V3::reverse(hi), super::I16x8V3::reverse(lo))
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi16x_v3(value) }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let p = idxs.as_ptr() as *const arch::__m256i;
            arch::_mm256_permutev_epi16x_v3(value, arch::_mm256_loadu_si256(p), arch::_mm256_loadu_si256(p.add(1)))
        }
    }

    compress_via_wide!();

    impl_byte_align_alignr256!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I16x16V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let u_lo = arch::_mm256_unpacklo_epi16(a, b);
            let u_hi = arch::_mm256_unpackhi_epi16(a, b);
            let res_lo = arch::_mm256_permute2x128_si256(u_lo, u_hi, 0x20);
            let res_hi = arch::_mm256_permute2x128_si256(u_lo, u_hi, 0x31);
            (res_lo, res_hi)
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // Within each 128-bit lane, gather even 16-bit words to the low 64 bits and odd words
        // to the high 64 bits (one `pshufb` per input). `unpacklo/hi_epi64` then merges the
        // two inputs' even/odd groups, and `permute4x64` fixes the 128-bit-lane ordering.
        unsafe {
            let shuf = arch::_mm256_setr_epi8(
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // lane 0
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // lane 1
            );
            let a_s = arch::_mm256_shuffle_epi8(a, shuf); // [Ea0|Oa0 | Ea1|Oa1]  (64-bit groups)
            let b_s = arch::_mm256_shuffle_epi8(b, shuf); // [Eb0|Ob0 | Eb1|Ob1]

            // unpacklo/hi work within 128-bit lanes -> [Ea0,Eb0, Ea1,Eb1] / [Oa0,Ob0, Oa1,Ob1]
            let even_pre = arch::_mm256_unpacklo_epi64(a_s, b_s);
            let odd_pre = arch::_mm256_unpackhi_epi64(a_s, b_s);

            // reorder 64-bit lanes [0,2,1,3] -> [Ea0,Ea1,Eb0,Eb1] / [Oa0,Oa1,Ob0,Ob1]
            (
                arch::_mm256_permute4x64_epi64(even_pre, 0b11_01_10_00),
                arch::_mm256_permute4x64_epi64(odd_pre, 0b11_01_10_00),
            )
        }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I16x16V3 {
    const HAS_TRUE_SHIFTV: bool = false; // no _mm256_sllv_epi16 before AVX512BW
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
impl PartialOrdRegister for I16x16V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epi16(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I16x16V3 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([i16::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([i16::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_min_epi16)
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_max_epi16)
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_add_epi16)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_mullo_epi16)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i16))
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
        unsafe { arch::_mm256_min_epi16(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I16x16V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 16>([-1; 16]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sign_epi16(value, Self::NEG_ONE) }
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_srai_epi16(value, 15) }
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_abs_epi16(value) }
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sign_epi16(arch::_mm256_set1_epi16(1), value) }
    }
    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<15>(mask))
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I16x16V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhi_epi16(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi16(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epi16(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epi16(lhs, rhs) }
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
        unsafe { arch::_mm256_popcnt_epi16x_v3(value) }
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        super::U16x16V3::leading_zeros(value)
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
impl SignedIntegerRegister for I16x16V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi16(value, IMM8) }
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sra_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhrs_epi16(a, b) }
    }
}

// Widen i16x16 -> i32x16 (= ArrayRegister<I32x8V3, 2>): sign-extend the two 128-bit halves.
#[thermite_macros::inline_always]
impl CastRegister<I16x16V3> for ArrayRegister<super::I32x8V3, 2> {
    fn cast_from(value: Storage<I16x16V3>) -> Storage<Self> {
        let (lo, hi) = I16x16V3::split(value);
        unsafe { ArrayRegister([arch::_mm256_cvtepi16_epi32(lo), arch::_mm256_cvtepi16_epi32(hi)]) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x8V3, 2>> for I16x16V3 {
    // Narrow i32x16 -> i16x16: truncate each 32-bit lane, then concat the two halves.
    fn cast_from(value: Storage<ArrayRegister<super::I32x8V3, 2>>) -> Storage<Self> {
        let lo = <super::I16x8V3 as CastRegister<super::I32x8V3>>::cast_from(value.0[0]);
        let hi = <super::I16x8V3 as CastRegister<super::I32x8V3>>::cast_from(value.0[1]);
        <Self as ConcatRegister<super::I16x8V3>>::concat(lo, hi)
    }

    // Saturating narrow i32x16 -> i16x16 via a two-source `vpackssdw` over the two 256-bit halves,
    // then `vpermq` to restitch the interleaved 64-bit groups ([a.lo, b.lo, a.hi, b.hi] -> sequence).
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x8V3, 2>>) -> Storage<Self> {
        unsafe {
            let packed = arch::_mm256_packs_epi32(value.0[0], value.0[1]);
            arch::_mm256_permute4x64_epi64(packed, 0b11_01_10_00)
        }
    }
}

// Saturating narrow i64x16 -> i16x16: clamp down to i32x16 (no 64-bit pack) then the pack above.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x4V3, 4>> for I16x16V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x4V3, 4>>) -> Storage<Self> {
        let words =
            <ArrayRegister<super::I32x8V3, 2> as CastRegister<ArrayRegister<super::I64x4V3, 4>>>::saturating_cast_from(
                value,
            );
        <Self as CastRegister<ArrayRegister<super::I32x8V3, 2>>>::saturating_cast_from(words)
    }

    fn cast_from(value: Storage<ArrayRegister<super::I64x4V3, 4>>) -> Storage<Self> {
        let v = value.0;
        unsafe {
            let lo = arch::_mm_cvt2epi64x4_epi16x_v3([v[0], v[1]]); // 8 words (lanes 0..8)
            let hi = arch::_mm_cvt2epi64x4_epi16x_v3([v[2], v[3]]); // 8 words (lanes 8..16)
            arch::_mm256_set_m128i(hi, lo)
        }
    }
}
