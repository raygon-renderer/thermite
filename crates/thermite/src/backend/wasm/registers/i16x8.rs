//! Native 128-bit signed 16-bit register for the WASM SIMD128 backend. Most 16-bit ops map to
//! native `i16x8_*` instructions; the few without a direct instruction (mulhi, reductions,
//! leading/trailing zeros, signum, truncating narrow) fall back to scalar.

use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I16x8Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for I16x8Wasm {
    type Lanes = typenum::U8;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = empty_reg::<Self>();

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            let mut arr = [0i16; 8];
            unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 8 - Z::N };
            while i < 8 {
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

#[thermite_macros::inline_always]
impl InterleaveRegister for I16x8Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b);
        let high = arch::i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b);
        let odds = arch::i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I16x8Wasm {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = arch::i16x8(0, 0, 0, 0, 0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::i16x8(-1, -1, -1, -1, -1, -1, -1, -1);

    fn all(value: Storage<Self>) -> bool {
        arch::i16x8_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn none(value: Storage<Self>) -> bool {
        !arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i16x8x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i16x8_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i16x8_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I16x8Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // WASM andnot has operands reversed vs. the trait
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<i16> for I16x8Wasm {
    fn extend(value: Storage<i16>) -> Storage<Self> {
        arch::i16x8(value, 0, 0, 0, 0, 0, 0, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<i16> {
        arch::i16x8_extract_lane::<0>(value)
    }
}

#[thermite_macros::inline_always]
impl Register for I16x8Wasm {
    type Element = i16;
    type Signed = super::I16x8Wasm;
    type Unsigned = super::U16x8Wasm;

    // Hardware extend ladder for the compress/expand byte index rows, plus a
    // direct `i8x16.swizzle` by the raw row.
    impl_widen_index_bytes_wasm!(x8);

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i16x8_shr(value, 15) // arithmetic shift propagates the sign bit
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::v128_load(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::i16x8(value, 0, 0, 0, 0, 0, 0, 0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i16x8_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1))
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14))
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        // Per-byte swizzle control: lane w -> bytes [2w, 2w+1].
        arch::u8x16_relaxed_swizzle(value, arch::wasm_ctrl_x8(idxs))
    }

    compress_via_table!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I16x8Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i16x8_shl(value, shift)
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u16x8_shr(value, shift) // logical
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i16x8_shl(value, IMM8 as u32)
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u16x8_shr(value, IMM8 as u32) // logical
    }
    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        arch::wasm_shlv_i16x8(value, shifts)
    }
    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        arch::wasm_shrv_u16x8(value, shifts) // logical
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for I16x8Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for I16x8Wasm {
    sort_via_network!(8);

    const ZERO: Storage<Self> = arch::i16x8(0, 0, 0, 0, 0, 0, 0, 0);
    const ONE: Storage<Self> = arch::i16x8(1, 1, 1, 1, 1, 1, 1, 1);
    const TWO: Storage<Self> = arch::i16x8(2, 2, 2, 2, 2, 2, 2, 2);

    const MIN: Storage<Self> = arch::i16x8(
        i16::MIN,
        i16::MIN,
        i16::MIN,
        i16::MIN,
        i16::MIN,
        i16::MIN,
        i16::MIN,
        i16::MIN,
    );
    const MAX: Storage<Self> = arch::i16x8(
        i16::MAX,
        i16::MAX,
        i16::MAX,
        i16::MAX,
        i16::MAX,
        i16::MAX,
        i16::MAX,
        i16::MAX,
    );

    fn min_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().min().unwrap()
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().max().unwrap()
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(0i16, i16::wrapping_add)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1i16, i16::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        arch::i16x8_splat(<Self::Lanes as Unsigned>::USIZE as i16)
    }

    fn indexed() -> Storage<Self> {
        arch::i16x8(0, 1, 2, 3, 4, 5, 6, 7)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_add(lhs, rhs)
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_sub(lhs, rhs)
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_mul(lhs, rhs)
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_min(lhs, rhs)
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I16x8Wasm {
    const NEG_ONE: Storage<Self> = arch::i16x8(-1, -1, -1, -1, -1, -1, -1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i16x8(1, 1, 1, 1, 1, 1, 1, 1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i16x8_neg(value)
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i16x8_shr(value, 15)
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i16x8_ge(value, Self::ZERO)
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i16x8_abs(value)
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.signum())
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I16x8Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| ((a as i32 * b as i32) >> 16) as i16)
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_mul(lhs, rhs)
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_add_sat(lhs, rhs)
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_sub_sat(lhs, rhs)
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
        arch::i16x8_extadd_pairwise_u8x16(arch::i8x16_popcnt(value))
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_zeros() as i16)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_zeros() as i16)
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_ones() as i16)
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_ones() as i16)
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I16x8Wasm {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i16x8_shr(value, IMM8 as u32) // arithmetic
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i16x8_shr(value, shift) // arithmetic
    }
    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        arch::wasm_shrv_i16x8(value, shifts) // arithmetic
    }
}

// Widen i16x8 -> i32x8 (= ArrayRegister<I32x4Wasm, 2>): sign-extend the low/high 4 lanes.
#[thermite_macros::inline_always]
impl CastRegister<I16x8Wasm> for ArrayRegister<super::I32x4Wasm, 2> {
    fn cast_from(value: Storage<I16x8Wasm>) -> Storage<Self> {
        let lo = arch::i32x4_extend_low_i16x8(value);
        let hi = arch::i32x4_extend_high_i16x8(value);
        ArrayRegister([lo, hi])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4Wasm, 2>> for I16x8Wasm {
    // Narrow i32x8 -> i16x8: truncate the low 16 bits of each 32-bit lane (wrapping, like `as`).
    #[rustfmt::skip]
    fn cast_from(value: Storage<ArrayRegister<super::I32x4Wasm, 2>>) -> Storage<Self> {
        // Gather the low 2 bytes of each 32-bit lane (4 from lo, 4 from hi) into 8 i16 lanes.
        arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13,        // lo vector lanes 0..3
            16, 17, 20, 21, 24, 25, 28, 29,  // hi vector lanes 0..3
        >(value.0[0], value.0[1])
    }

    // Saturating narrow i32x8 -> i16x8 via a single two-source `i16x8.narrow_i32x4_s`.
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4Wasm, 2>>) -> Storage<Self> {
        arch::i16x8_narrow_i32x4(value.0[0], value.0[1])
    }
}

// Saturating narrow i64x8 -> i16x8: no 64-bit narrow, so clamp + truncating narrow.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 4>> for I16x8Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 4>>) -> Storage<Self> {
        type Src = ArrayRegister<super::I64x2Wasm, 4>;
        let lo = <Src as Register>::splat(i16::MIN as i64);
        let hi = <Src as Register>::splat(i16::MAX as i64);
        let clamped = <Src as NumericRegister>::min(<Src as NumericRegister>::max(value, lo), hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }

    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 4>>) -> Storage<Self> {
        unsafe { arch::narrow_4xi64x2_to_words(value.0) }
    }
}
