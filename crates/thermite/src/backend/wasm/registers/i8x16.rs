//! Native 128-bit signed 8-bit register for the WASM SIMD128 backend. WASM has native byte
//! shifts/popcount/compares, so most ops map directly; byte multiply (no `i8x16_mul`),
//! reductions, leading/trailing zeros and signum fall back to scalar.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SaturatingCastRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, ZeroUpper,
        array::ArrayRegister, empty_reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I8x16Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for I8x16Wasm {
    type Lanes = typenum::U16;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = arch::ISA;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = empty_reg::<Self>();

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 16 } {
            value
        } else {
            let mut arr = [0i8; 16];
            unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 16 - Z::N };
            while i < 16 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::v128_load(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl InterleaveRegister for I8x16Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
        let high = arch::i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a, b);
        let odds = arch::i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I8x16Wasm {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = arch::i8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    fn all(value: Storage<Self>) -> bool {
        arch::i8x16_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn none(value: Storage<Self>) -> bool {
        !arch::v128_any_true(value)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i8x16_bitmask(value) as u16 as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i8x16_bitmask(value) as u16 as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I8x16Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs)
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl ExtendRegister<i8> for I8x16Wasm {
    fn extend(value: Storage<i8>) -> Storage<Self> {
        arch::i8x16(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<i8> {
        arch::i8x16_extract_lane::<0>(value)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl Register for I8x16Wasm {
    type Element = i8;
    type Signed = super::I8x16Wasm;
    type Unsigned = super::U8x16Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i8x16_shr(value, 7) // arithmetic shift propagates the sign bit
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::v128_load(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::i8x16(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i8x16_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::u8x16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        )
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value // one byte per lane
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I8x16Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i8x16_shl(value, shift)
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u8x16_shr(value, shift) // logical
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i8x16_shl(value, IMM8 as u32)
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_shr(value, IMM8 as u32) // logical
    }
}

#[thermite_macros::inline_always]
impl SwizzleRegister for I8x16Wasm {
    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        // For 8-bit lanes the index IS the byte index; pack to bytes and swizzle.
        let mut bytes = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            bytes[i] = idxs[i] as u8;
            i += 1;
        }
        let ctrl = unsafe { arch::v128_load(bytes.as_ptr() as *const _) };
        arch::u8x16_relaxed_swizzle(value, ctrl)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for I8x16Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_eq(lhs, rhs) }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl NumericRegister for I8x16Wasm {
    const ZERO: Storage<Self> = arch::i8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const ONE: Storage<Self> = arch::i8x16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    const TWO: Storage<Self> = arch::i8x16(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);

    const MIN: Storage<Self> = arch::i8x16(i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
    const MAX: Storage<Self> = arch::i8x16(i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX, i8::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        Self::as_array(&value).iter().copied().min().unwrap()
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        Self::as_array(&value).iter().copied().max().unwrap()
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        Self::as_array(&value).iter().copied().fold(0i8, i8::wrapping_add)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_array(&value).iter().copied().fold(1i8, i8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        arch::i8x16_splat(<Self::Lanes as Unsigned>::USIZE as i8)
    }

    fn indexed() -> Storage<Self> {
        arch::i8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_add(lhs, rhs)
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_sub(lhs, rhs)
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, i8::wrapping_mul)
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_min(lhs, rhs)
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I8x16Wasm {
    const NEG_ONE: Storage<Self> = arch::i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i8x16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i8x16_neg(value)
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i8x16_shr(value, 7)
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i8x16_ge(value, Self::ZERO)
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i8x16_abs(value)
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.signum())
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I8x16Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| ((a as i16 * b as i16) >> 8) as i8)
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, i8::wrapping_mul)
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_add_sat(lhs, rhs)
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_sub_sat(lhs, rhs)
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
        arch::i8x16_popcnt(value)
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_zeros() as i8)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_zeros() as i8)
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_ones() as i8)
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_ones() as i8)
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I8x16Wasm {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i8x16_shr(value, IMM8 as u32) // arithmetic
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i8x16_shr(value, shift) // arithmetic
    }
}

// Saturating narrow i16x16 -> i8x16 via a single two-source `i8x16.narrow_i16x8_s`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I16x8Wasm, 2>> for I8x16Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I16x8Wasm, 2>>) -> Storage<Self> {
        arch::i8x16_narrow_i16x8(value.0[0], value.0[1])
    }
}

// Saturating narrow i32x16 -> i8x16: three narrows (two `i16x8.narrow_i32x4_s`, one `i8x16.narrow_i16x8_s`).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4Wasm, 4>> for I8x16Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4Wasm, 4>>) -> Storage<Self> {
        let w0 = arch::i16x8_narrow_i32x4(value.0[0], value.0[1]);
        let w1 = arch::i16x8_narrow_i32x4(value.0[2], value.0[3]);
        arch::i8x16_narrow_i16x8(w0, w1)
    }
}

// Saturating narrow i64x16 -> i8x16: no 64-bit narrow, so clamp + truncating narrow.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2Wasm, 8>> for I8x16Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 8>>) -> Storage<Self> {
        type Src = ArrayRegister<super::I64x2Wasm, 8>;
        let lo = <Src as Register>::splat(i8::MIN as i64);
        let hi = <Src as Register>::splat(i8::MAX as i64);
        let clamped = <Src as NumericRegister>::min(<Src as NumericRegister>::max(value, lo), hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }
}
