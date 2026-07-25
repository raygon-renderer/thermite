//! Native 128-bit signed 8-bit register for x86-v3 (AVX2). This is the fixed-width `i8x16`
//! slot (the 128-bit half of the native 256-bit [`super::I8x32V3`]). It reuses the 128-bit byte
//! polyfills and the `_mm_reduce_epi8_v2!` macro inherited through the v1/v2 polyfill chain.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, Element, ExtendRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register, SaturatingCastRegister,
        SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I8x16V3;

#[thermite_macros::inline_always]
impl CoreRegister for I8x16V3 {
    type Lanes = typenum::U16;
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
        if const { Z::N >= 16 } {
            value
        } else {
            let mut arr = [0i8; 16];
            unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 16 - Z::N };
            while i < 16 {
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
impl InterleaveRegister for I8x16V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi8(a, b), arch::_mm_unpackhi_epi8(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let shuf = arch::_mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
            let a_s = arch::_mm_shuffle_epi8(a, shuf);
            let b_s = arch::_mm_shuffle_epi8(b, shuf);
            (arch::_mm_unpacklo_epi64(a_s, b_s), arch::_mm_unpackhi_epi64(a_s, b_s))
        }
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I8x16V3 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 16>([0; 16]);
    const TRUTHY: Storage<Self> = reg::<Self, 16>([-1; 16]);

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
        Some(unsafe { (arch::_mm_movemask_epi8(value) as u32) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I8x16V3 {
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
impl ExtendRegister<i8> for I8x16V3 {
    fn extend(value: Storage<i8>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi8(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i8> {
        unsafe { arch::_mm_extract_epi8::<0>(value) as i8 }
    }
}

#[thermite_macros::inline_always]
impl Register for I8x16V3 {
    type Element = i8;

    type Signed = super::I8x16V3;
    type Unsigned = super::U8x16V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epi8(arch::_mm_setzero_si128(), value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi8(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
    }

    impl_native_extract!(@epi8x16);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi8(value) }
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
                arch::_mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            )
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let p = idxs.as_ptr() as *const arch::__m128i;
            arch::_mm_permutev_epi8x_v2(
                value,
                arch::_mm_loadu_si128(p),
                arch::_mm_loadu_si128(p.add(1)),
                arch::_mm_loadu_si128(p.add(2)),
                arch::_mm_loadu_si128(p.add(3)),
            )
        }
    }

    compress_via_wide!();

    impl_byte_align_alignr!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I8x16V3 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi8x_v1(value, shift) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi8x_v1(value, shift) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi8x_v1::<IMM8>(value) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi8x_v1::<IMM8>(value) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I8x16V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epi8(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I8x16V3 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([i8::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([i8::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_min_epi8)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_max_epi8)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_add_epi8)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1i8, i8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i8))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi8(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi8(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi8(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi8(lhs, arch::_mm_and_si128(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v1(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi8(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I8x16V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 16>([-1; 16]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sign_epi8(value, Self::NEG_ONE) }
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi8(value) }
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sign_epi8(arch::_mm_set1_epi8(1), value) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I8x16V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epi8x_v1(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v1(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi8(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi8(lhs, rhs) }
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
        unsafe { arch::_mm_popcnt_epi8x_v2(value) }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        super::U8x16V3::leading_zeros(value)
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
impl SignedIntegerRegister for I8x16V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi8x_v1::<IMM8>(value) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi8x_v1(value, shift) }
    }
}

// Saturating narrow i16x16 -> i8x16 via `vpacksswb`. Pack `(v, v)` and restitch the
// two populated 64-bit groups with `vpermq` (see `i16x8.rs` for the lane-crossing note).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I16x16V3> for I8x16V3 {
    fn saturating_cast_from(value: Storage<super::I16x16V3>) -> Storage<Self> {
        unsafe {
            let packed = arch::_mm256_packs_epi16(value, value);
            arch::_mm256_castsi256_si128(arch::_mm256_permute4x64_epi64(packed, 0b00_00_10_00))
        }
    }
}

// i32x16 -> i8x16: compose two packs (i32x16 -> i16x16 -> i8x16).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x8V3, 2>> for I8x16V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x8V3, 2>>) -> Storage<Self> {
        let words =
            <super::I16x16V3 as SaturatingCastRegister<ArrayRegister<super::I32x8V3, 2>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x16V3>>::saturating_cast_from(words)
    }
}

// i64x16 -> i8x16: clamp down to i32x16, then the two packs above.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x4V3, 4>> for I8x16V3 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x4V3, 4>>) -> Storage<Self> {
        let words =
            <super::I16x16V3 as SaturatingCastRegister<ArrayRegister<super::I64x4V3, 4>>>::saturating_cast_from(value);
        <Self as SaturatingCastRegister<super::I16x16V3>>::saturating_cast_from(words)
    }
}
