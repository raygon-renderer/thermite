use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, ZeroUpper,
        array::ArrayRegister, empty_reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I32x4Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for I32x4Wasm {
    type Lanes = typenum::U4;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = arch::ISA;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::i32x4(0, 0, 0, 0);

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 3 } {
            arch::v128_and(value, arch::u32x4(!0, !0, !0, 0))
        } else if const { Z::N == 2 } {
            arch::v128_and(value, arch::u32x4(!0, !0, 0, 0))
        } else if const { Z::N == 1 } {
            arch::v128_and(value, arch::u32x4(!0, 0, 0, 0))
        } else {
            Self::EMPTY
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I32x4Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: WASM andnot has operands reversed vs. the trait
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

impl InterleaveRegister for I32x4Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i32x4_shuffle::<0, 2, 4, 6>(a, b);
        let odds = arch::i32x4_shuffle::<1, 3, 5, 7>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I32x4Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shr(value, shift) // Non-arithmetic
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shl(value, shift)
    }

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I32x4Wasm {
    const FALSY: Storage<Self> = arch::i32x4(0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::i32x4(-1, -1, -1, -1);

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    fn all(value: Storage<Self>) -> bool {
        arch::i32x4_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i32x4_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for I32x4Wasm {
    type Element = i32;
    type Signed = super::I32x4Wasm;
    type Unsigned = super::U32x4Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // Arithmetic shift right by 31 propagates sign bit to all bits
        arch::i32x4_shr(value, 31)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::i32x4(value[0], value[1], value[2], value[3])
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i32x4_splat(value)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::i32x4(value, 0, 0, 0)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 32-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            3, 2, 1, 0,
            7, 6, 5, 4,
            11, 10, 9, 8,
            15, 14, 13, 12,
        ))
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::i32x4_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::i32x4_replace_lane::<I>(value, element)
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for I32x4Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for I32x4Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

#[thermite_macros::inline_always]
impl SwizzleRegister for I32x4Wasm {
    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::x4indices(idxs[0] as u8, idxs[1] as u8, idxs[2] as u8, idxs[3] as u8),
        )
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for I32x4Wasm {
 fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_ge(lhs, rhs) }
 fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_lt(lhs, rhs) }
 fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_le(lhs, rhs) }
 fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_ne(lhs, rhs) }
 fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_gt(lhs, rhs) }
 fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for I32x4Wasm {
    const ZERO: Storage<Self> = arch::i32x4(0, 0, 0, 0);
    const ONE: Storage<Self> = arch::i32x4(1, 1, 1, 1);
    const TWO: Storage<Self> = arch::i32x4(2, 2, 2, 2);

    const MIN: Storage<Self> = arch::i32x4(i32::MIN, i32::MIN, i32::MIN, i32::MIN);
    const MAX: Storage<Self> = arch::i32x4(i32::MAX, i32::MAX, i32::MAX, i32::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_min i32x4_min)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_max i32x4_max)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_add i32x4_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_mul i32x4_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i32x4_shuffle::<0, 2, 4, 6>(lo, hi); // [a0,a2,b0,b2]
        let odd = arch::i32x4_shuffle::<1, 3, 5, 7>(lo, hi); // [a1,a3,b1,b3]
        arch::i32x4_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::i32x4_splat(<Self::Lanes as Unsigned>::USIZE as i32)
    }

    fn indexed() -> Storage<Self> {
        arch::i32x4(0, 1, 2, 3)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_min(lhs, rhs)
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I32x4Wasm {
    const NEG_ONE: Storage<Self> = arch::i32x4(-1, -1, -1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i32x4(1, 1, 1, 1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i32x4_neg(value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        // Arithmetic shift right by 31 to propagate the sign bit
        arch::i32x4_shr(value, 31)
    }

    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i32x4_gt(value, Self::ZERO)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i32x4_abs(value)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I32x4Wasm {
    #[cfg(target_arch = "wasm64")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Extract high 32 bits from the intermediate 64-bit lanes
        arch::i32x4_shuffle::<1, 3, 5, 7>(
            arch::i64x2_extmul_low_i32x4(lhs, rhs),
            arch::i64x2_extmul_high_i32x4(lhs, rhs),
        )
    }

    #[cfg(target_arch = "wasm32")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // 1. Prepare Low Lanes (0 and 1)
        // Shuffle to place the values into the HIGH 32-bits of each 64-bit lane.
        // Layout: [a0, a0, a1, a1]
        let a_lo_dup = arch::i32x4_shuffle::<0, 0, 1, 1>(lhs, lhs);
        let b_lo_dup = arch::i32x4_shuffle::<0, 0, 1, 1>(rhs, rhs);

        // Arithmetic Shift Right by 32.
        // This moves the value to the low half and fills the top with sign bits.
        // Result: [ (i64)a0, (i64)a1 ]
        let a_lo = arch::i64x2_shr(a_lo_dup, 32);
        let b_lo = arch::i64x2_shr(b_lo_dup, 32);

        // Multiply 64-bit integers
        let prod_lo = arch::i64x2_mul(a_lo, b_lo);

        // 2. Prepare High Lanes (2 and 3)
        // Layout: [a2, a2, a3, a3]
        let a_hi_dup = arch::i32x4_shuffle::<2, 2, 3, 3>(lhs, lhs);
        let b_hi_dup = arch::i32x4_shuffle::<2, 2, 3, 3>(rhs, rhs);

        let a_hi = arch::i64x2_shr(a_hi_dup, 32);
        let b_hi = arch::i64x2_shr(b_hi_dup, 32);

        let prod_hi = arch::i64x2_mul(a_hi, b_hi);

        // 3. Shuffle to extract the high 32 bits from each 64-bit product.
        // Indices 1 and 3 are the high halves of the i64 lanes.
        arch::i32x4_shuffle::<1, 3, 5, 7>(prod_lo, prod_hi)
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_mul(lhs, rhs)
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_saturating_add(lhs, rhs)
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_saturating_sub(lhs, rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_add i32x4_add)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_mul i32x4_mul)
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
        // use unsigned version's popcnt implementation
        super::U32x4Wasm::count_ones(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.leading_zeros() as i32)
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.trailing_zeros() as i32)
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I32x4Wasm {
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i32x4_shr(value, shift) // Arithmetic shift right
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x4Wasm> for ArrayRegister<super::I64x2Wasm, 2> {
    fn cast_from(value: Storage<I32x4Wasm>) -> Storage<Self> {
        // sign-extend each pair of i32 to i64
        let lo = arch::i64x2_extend_low_i32x4(value);
        let hi = arch::i64x2_extend_high_i32x4(value);
        ArrayRegister([lo, hi])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x2Wasm, 2>> for I32x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<ArrayRegister<super::I64x2Wasm, 2>>) -> Storage<Self> {
        // Selects bytes 0-3 (lane 0 low) and 8-11 (lane 1 low) from lo
        // Selects bytes 16-19 (lane 0 low) and 24-27 (lane 1 low) from hi
        arch::i8x16_shuffle::<
            0, 1, 2, 3,     // Lo Vec, Lane 0 (Low bits)
            8, 9, 10, 11,   // Lo Vec, Lane 1 (Low bits)
            16, 17, 18, 19, // Hi Vec, Lane 0 (Low bits)
            24, 25, 26, 27  // Hi Vec, Lane 1 (Low bits)
        >(value.0[0], value.0[1])
    }
}
