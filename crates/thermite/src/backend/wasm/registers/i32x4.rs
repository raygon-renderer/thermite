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
pub struct I32x4Wasm;

impl Register for I32x4Wasm {
    type Lanes = typenum::U4;

    type Element = i32;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = arch::ISA;

    type ISize = super::I32x4Wasm;
    type USize = super::U32x4Wasm;

    const EMPTY: Storage<Self> = arch::i32x4(0, 0, 0, 0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::i32x4(value[0], value[1], value[2], value[3])
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i32x4_splat(value)
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::i32x4(value, 0, 0, 0)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::i32x4_extract_lane::<I>(value)
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
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);

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

impl BitshiftRegister for I32x4Wasm {
    const HAS_TRUE_SHIFTV: bool = false;

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shr(value, shift) // Non-arithmetic
    }

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shl(value, shift)
    }
}

impl MaskRegister for I32x4Wasm {
    const FALSY: Storage<Self> = arch::i32x4(0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::i32x4(-1, -1, -1, -1);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::i32x4_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i32x4_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for I32x4Wasm {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for I32x4Wasm {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for I32x4Wasm {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::x4indices(idxs[0] as u8, idxs[1] as u8, idxs[2] as u8, idxs[3] as u8),
        )
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for I32x4Wasm {
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_ge(lhs, rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_lt(lhs, rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_le(lhs, rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_ne(lhs, rhs) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_gt(lhs, rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i32x4_eq(lhs, rhs) }
}

impl NumericRegister for I32x4Wasm {
    const ZERO: Storage<Self> = arch::i32x4(0, 0, 0, 0);
    const ONE: Storage<Self> = arch::i32x4(1, 1, 1, 1);
    const TWO: Storage<Self> = arch::i32x4(2, 2, 2, 2);

    const MIN: Storage<Self> = arch::i32x4(i32::MIN, i32::MIN, i32::MIN, i32::MIN);
    const MAX: Storage<Self> = arch::i32x4(i32::MAX, i32::MAX, i32::MAX, i32::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_min i32x4_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_max i32x4_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_add i32x4_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_mul i32x4_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::i32x4_splat(<Self::Lanes as Unsigned>::USIZE as i32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::i32x4(0, 1, 2, 3)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_mul(lhs, rhs)
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
        arch::i32x4_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_max(lhs, rhs)
    }
}

impl SignedRegister for I32x4Wasm {
    const NEG_ONE: Storage<Self> = arch::i32x4(-1, -1, -1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i32x4(1, 1, 1, 1);

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i32x4_neg(value)
    }

    #[inline(always)]
    fn is_negative(value: Storage<Self>) -> Storage<Self> {
        // Arithmetic shift right by 31 to propagate the sign bit
        arch::i32x4_shr(value, 31)
    }

    #[inline(always)]
    fn is_positive(value: Storage<Self>) -> Storage<Self> {
        arch::i32x4_gt(value, Self::ZERO)
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i32x4_abs(value)
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<31>(mask))
    }
}

impl IntegerRegister for I32x4Wasm {
    #[inline(always)]
    #[cfg(target_arch = "wasm64")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Extract high 32 bits from the intermediate 64-bit lanes
        arch::i32x4_shuffle::<1, 3, 5, 7>(
            arch::i64x2_extmul_low_i32x4(a, b), //
            arch::i64x2_extmul_high_i32x4(a, b),
        )
    }

    #[inline(always)]
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

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_mul(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_saturating_add(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_saturating_sub(lhs, rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_add i32x4_add)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(i value; i32x4_mul i32x4_mul)
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
        super::U32x4Wasm::count_ones(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }
}

impl SignedIntegerRegister for I32x4Wasm {
    #[inline(always)]
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i32x4_shr(value, shift) // Arithmetic shift right
    }
}

impl CastRegister<I32x4Wasm> for DoublePumpRegister<super::I64x2Wasm> {
    #[inline(always)]
    fn cast_from(value: Storage<I32x4Wasm>) -> Storage<Self> {
        // sign-extend each pair of u32 to u64
        let lo = arch::i64x2_extend_low_i32x4(value);
        let hi = arch::i64x2_extend_high_i32x4(value);

        DoublePumpRegister(lo, hi)
    }
}
