use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, CastRegister, IntegerRegister, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        UnsignedIntegerRegister, dp::DoublePumpRegister,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2Wasm32;

impl Register for U64x2Wasm32 {
    type Lanes = typenum::U2;

    type Element = u64;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::WASM32;

    type ISize = super::I64x2Wasm32;
    type USize = super::U64x2Wasm32;

    const EMPTY: Storage<Self> = arch::u64x2(0, 0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::u64x2(value[0], value[1])
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u64x2_splat(value)
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::u64x2(value, 0)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::u64x2_extract_lane::<I>(value)
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
        // within each 64-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            7, 6, 5, 4,3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8,
        ))
    }
}

impl BitshiftRegister for U64x2Wasm32 {
    const HAS_TRUE_SHIFTV: bool = false;

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shr(value, shift) // Non-arithmetic
    }

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shl(value, shift)
    }
}

impl MaskRegister for U64x2Wasm32 {
    const FALSY: Storage<Self> = arch::u64x2(0, 0);
    const TRUTHY: Storage<Self> = arch::u64x2(!0, !0);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx2_to_i64x2x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::u64x2_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::u64x2_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::u64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for U64x2Wasm32 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for U64x2Wasm32 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for U64x2Wasm32 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(idxs[0] as u8, idxs[1] as u8))
    }
}

impl PartialOrdRegister for U64x2Wasm32 {
    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_ne(lhs, rhs)
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_eq(lhs, rhs)
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_ge(lhs, rhs)
    }

    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_lt(lhs, rhs)
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_le(lhs, rhs)
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_gt(lhs, rhs)
    }
}

impl NumericRegister for U64x2Wasm32 {
    const ZERO: Storage<Self> = arch::u64x2(0, 0);
    const ONE: Storage<Self> = arch::u64x2(1, 1);
    const TWO: Storage<Self> = arch::u64x2(2, 2);

    const MIN: Storage<Self> = arch::u64x2(u64::MIN, u64::MIN);
    const MAX: Storage<Self> = arch::u64x2(u64::MAX, u64::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_min u64x2_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_max u64x2_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_add u64x2_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_mul u64x2_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::u64x2_splat(<Self::Lanes as Unsigned>::USIZE as u64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::u64x2(0, 1)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_mul(lhs, rhs)
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
        arch::u64x2_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_max(lhs, rhs)
    }
}

impl IntegerRegister for U64x2Wasm32 {
    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_saturating_add(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_saturating_sub(lhs, rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_add u64x2_add)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_mul u64x2_mul)
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        todo!()
    }

    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        todo!()
    }

    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        todo!()
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[inline(always)]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        // 1. Get 32-bit counts: [c0, c1, c2, c3]
        let counts_32 = super::U32x4Wasm32::count_ones(value);

        // 2. We need [c0+c1, c2+c3].
        // Since max bit count for 64-bits is 64, this fits comfortably in i32
        // without overflow, so we stay in i32 domain for the add.

        // Move odd lanes to even positions: [c1, c1, c3, c3]
        let shifted = arch::i32x4_shuffle::<1, 1, 3, 3>(counts_32, counts_32);

        // Add: [c0+c1, ..., c2+c3, ...]
        let sums = arch::u32x4_add(counts_32, shifted);

        // 3. Arrange for 64-bit extension: [SumLo, SumHi, SumLo, SumHi]
        let arranged = arch::i32x4_shuffle::<0, 2, 0, 2>(sums, sums);

        // Extend to i64 (Zero extend is fine as count is positive)
        arch::u64x2_extend_low_u32x4(arranged)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }
}

impl UnsignedIntegerRegister for U64x2Wasm32 {}

impl CastRegister<DoublePumpRegister<U64x2Wasm32>> for super::U32x4Wasm32 {
    #[rustfmt::skip]
    #[inline(always)]
    fn cast_from(value: <DoublePumpRegister<U64x2Wasm32> as Register>::Storage) -> Storage<Self> {
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
