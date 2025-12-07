use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, CastRegister, IntegerRegister, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage,
        SwizzleRegister, dp::DoublePumpRegister,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x2Wasm;

impl Register for I64x2Wasm {
    type Lanes = typenum::U2;

    type Element = i64;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = arch::ISA;

    type ISize = super::I64x2Wasm;
    type USize = super::U64x2Wasm;

    const EMPTY: Storage<Self> = arch::i64x2(0, 0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::i64x2(value[0], value[1])
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i64x2_splat(value)
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::i64x2(value, 0)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::i64x2_extract_lane::<I>(value)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: arguments are reversed
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(1, 0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = Self::swizzle(a, b, GenericArray::from_array([0, 2]));
        let high = Self::swizzle(a, b, GenericArray::from_array([1, 3]));

        (low, high)
    }

    #[inline(always)]
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
}

impl BitshiftRegister for I64x2Wasm {
    const HAS_TRUE_SHIFTV: bool = false;

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shr(value, shift) // Non-arithmetic
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shl(value, shift)
    }
}

impl MaskRegister for I64x2Wasm {
    const FALSY: Storage<Self> = arch::i64x2(0, 0);
    const TRUTHY: Storage<Self> = arch::i64x2(-1, -1);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx2_to_i64x2x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::i64x2_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i64x2_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for I64x2Wasm {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for I64x2Wasm {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for I64x2Wasm {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(idxs[0] as u8, idxs[1] as u8))
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for I64x2Wasm {
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_ge(lhs, rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_lt(lhs, rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_le(lhs, rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_ne(lhs, rhs) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_gt(lhs, rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_eq(lhs, rhs) }
}

impl NumericRegister for I64x2Wasm {
    const ZERO: Storage<Self> = arch::i64x2(0, 0);
    const ONE: Storage<Self> = arch::i64x2(1, 1);
    const TWO: Storage<Self> = arch::i64x2(2, 2);

    const MIN: Storage<Self> = arch::i64x2(i64::MIN, i64::MIN);
    const MAX: Storage<Self> = arch::i64x2(i64::MAX, i64::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_min i64x2_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_max i64x2_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_add i64x2_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_mul i64x2_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::i64x2_splat(<Self::Lanes as Unsigned>::USIZE as i64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::i64x2(0, 1)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_mul(lhs, rhs)
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_max(lhs, rhs)
    }
}

impl SignedRegister for I64x2Wasm {
    const NEG_ONE: Storage<Self> = arch::i64x2(-1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i64x2(1, 1);

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i64x2_neg(value)
    }

    #[inline(always)]
    fn is_negative(value: Storage<Self>) -> Storage<Self> {
        // Arithmetic shift right by 31 to propagate the sign bit
        arch::i64x2_shr(value, 31)
    }

    #[inline(always)]
    fn is_positive(value: Storage<Self>) -> Storage<Self> {
        arch::i64x2_gt(value, Self::ZERO)
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i64x2_abs(value)
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<31>(mask))
    }
}

impl IntegerRegister for I64x2Wasm {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let a0 = arch::i64x2_extract_lane::<0>(lhs);
        let a1 = arch::i64x2_extract_lane::<1>(lhs);
        let b0 = arch::i64x2_extract_lane::<0>(rhs);
        let b1 = arch::i64x2_extract_lane::<1>(rhs);

        arch::u64x2(
            (((a0 as i128) * (b0 as i128)) >> 64) as u64,
            (((a1 as i128) * (b1 as i128)) >> 64) as u64,
        )
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_mul(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_saturating_add(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_saturating_sub(lhs, rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_add i64x2_add)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_mul i64x2_mul)
    }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[inline(always)]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        // use unsigned version's popcnt implementation
        super::U64x2Wasm::count_ones(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }
}

impl SignedIntegerRegister for I64x2Wasm {
    #[inline(always)]
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i64x2_shr(value, shift) // Arithmetic shift right
    }
}

impl CastRegister<DoublePumpRegister<I64x2Wasm>> for super::I32x4Wasm {
    #[rustfmt::skip]
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<I64x2Wasm>>) -> Storage<Self> {
        // Selects bytes 0-3 (lane 0 low) and 8-11 (lane 1 low) from 'lo'
        // Selects bytes 16-19 (lane 0 low) and 24-27 (lane 1 low) from 'hi'
        arch::i8x16_shuffle::<
            0, 1, 2, 3,     // Lo Vec, Lane 0 (Low bits)
            8, 9, 10, 11,   // Lo Vec, Lane 1 (Low bits)
            16, 17, 18, 19, // Hi Vec, Lane 0 (Low bits)
            24, 25, 26, 27  // Hi Vec, Lane 1 (Low bits)
        >(value.0, value.1)
    }
}
