use generic_array::{GenericArray, typenum};

use super::{
    spirv_count_ones_u32, spirv_count_ones_u64, spirv_leading_zeros_u32, spirv_leading_zeros_u64, spirv_swap_bytes_u32,
    spirv_swap_bytes_u64, spirv_trailing_zeros_u32, spirv_trailing_zeros_u64,
};
use crate::backend::spirv::arch::{self as arch, glsl};
use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, Element, IndexableRegister, IntegerRegister,
        InterleaveRegister, MaskElement, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, ZeroUpper,
    },
};

#[rustfmt::skip]
macro_rules! decl_spirv_unsigned_scalar { ($u:ty => $width:literal) => { paste::paste! {

#[thermite_macros::inline_always]
impl CoreRegister for [<u $width>] {
    type Lanes   = typenum::U1;
    type Storage = $u;
    type Mask    = bool;

    const IS_EMULATED:         bool = false;
    const ISA:                 InstructionSet = InstructionSet::SPIRV;
    const EMPTY:               Storage<Self> = 0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: bool, lhs: Self, rhs: Self) -> Self {
        unsafe { arch::op_opselect::<Self, bool>(mask, rhs, lhs) }
    }
 fn z (mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, value, 0) } }
 fn nz(mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, 0, value) } }
 fn zeroupper_z<Z: ZeroUpper>(value: Self) -> Self {
        if const { Z::N >= 1 } { value } else { Self::EMPTY }
    }

 fn from_mask(mask: bool) -> Self { Self::from_bool(mask) }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl BitwiseRegister for [<u $width>] {
 fn bitxor(lhs: Self, rhs: Self) -> Self { lhs ^ rhs }
 fn bitand(lhs: Self, rhs: Self) -> Self { lhs & rhs }
 fn bitor (lhs: Self, rhs: Self) -> Self { lhs | rhs }
 fn not(value: Self) -> Self { !value }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for [<u $width>] {
 fn interleave (a: Self, b: Self) -> (Self, Self) { (a, b) }
 fn deinterleave(a: Self, b: Self) -> (Self, Self) { (a, b) }
}

impl Register for [<u $width>] {
    type Element  = $u;
    type Signed   = [<i $width>];
    type Unsigned = [<u $width>];

 fn into_mask(value: Self) -> bool { value.to_bool() }
 fn msb_to_mask(value: Self) -> bool {
        // Reinterpret as signed then arithmetic-shift to propagate the MSB
        let signed = unsafe { arch::op_opbitcast::<[<i $width>], Self>(value) };
        <[<i $width>]>::into_mask(signed >> (<$u>::BITS - 1))
    }

 fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self { value[0] }
 fn single(value: Self::Element) -> Self { value }
 fn splat (value: Self::Element) -> Self { value }
 fn broadcast<const I: usize>(value: Self) -> Self { value }
 fn reverse(value: Self) -> Self { value }

    fn swap_bytes(value: Self) -> Self {
        [<spirv_swap_bytes_u $width>](value)
    }
}

#[thermite_macros::inline_always]
impl<I: UnsignedIntegerRegister<Lanes = Self::Lanes>> IndexableRegister<I> for [<u $width>] {}

impl BitshiftRegister for [<u $width>] {
    const HAS_TRUE_SHIFTV:      bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

 fn bshli<const IMM8: i32>(value: Self) -> Self { value << (8 * IMM8) }
 fn bshri<const IMM8: i32>(value: Self) -> Self { value >> (8 * IMM8) }
 fn shl (value: Self, shift: u32) -> Self { value << shift }
 fn shr (value: Self, shift: u32) -> Self { value >> shift }
 fn shlv(value: Self, shifts: $u) -> Self { value << shifts }
 fn shrv(value: Self, shifts: $u) -> Self { value >> shifts }
 fn shli<const IMM8: i32>(value: Self) -> Self { value << IMM8 }
 fn shri<const IMM8: i32>(value: Self) -> Self { value >> IMM8 }

 fn rol (value: Self, shift: u32) -> Self { value.rotate_left (shift) }
 fn ror (value: Self, shift: u32) -> Self { value.rotate_right(shift) }
 fn rolv(value: Self, shifts: $u) -> Self { Self::rol(value, shifts as _) }
 fn rorv(value: Self, shifts: $u) -> Self { Self::ror(value, shifts as _) }
    // OpBitReverse: native SPIR-V instruction — override the default swap+shift chain.
 fn reverse_bits(value: Self) -> Self { unsafe { arch::op_opbitreverse::<Self>(value) } }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for [<u $width>] {
    fn shuffle<const IMM8: i32>(lhs: Self, rhs: Self) -> Self {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for [<u $width>] {
 fn permute<const IMM8: i32>(value: Self) -> Self { value }
}

impl SwizzleRegister for [<u $width>] {
    const HAS_PERMUTEV: bool = false;
 fn permutev(value: Self, _idxs: GenericArray<u32, Self::Lanes>) -> Self { value }
 fn swizzle(a: Self, b: Self, idxs: GenericArray<u32, Self::Lanes>) -> Self {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl PartialOrdRegister for [<u $width>] {
 fn gt(lhs: Self, rhs: Self) -> bool { lhs > rhs }
 fn eq(lhs: Self, rhs: Self) -> bool { lhs == rhs }
 fn ge(lhs: Self, rhs: Self) -> bool { lhs >= rhs }
 fn lt(lhs: Self, rhs: Self) -> bool { lhs < rhs }
 fn le(lhs: Self, rhs: Self) -> bool { lhs <= rhs }
 fn ne(lhs: Self, rhs: Self) -> bool { lhs != rhs }
}

#[thermite_macros::inline_always]
impl NumericRegister for [<u $width>] {
    const ZERO: Self = 0;
    const ONE:  Self = 1;
    const TWO:  Self = 2;
    const MIN:  Self = <$u>::MIN;
    const MAX:  Self = <$u>::MAX;

 fn min_element (value: Self) -> Self::Element { value }
 fn max_element (value: Self) -> Self::Element { value }
 fn sum_elements(value: Self) -> Self::Element { value }
 fn prod_elements(value: Self) -> Self::Element { value }
 fn pairwise_sum(lo: Self, hi: Self) -> Self { lo.wrapping_add(hi) }
 fn offset() -> Self { 1 }
 fn indexed() -> Self { 0 }
 fn add(lhs: Self, rhs: Self) -> Self { lhs + rhs }
 fn sub(lhs: Self, rhs: Self) -> Self { lhs - rhs }
 fn mul(lhs: Self, rhs: Self) -> Self { lhs * rhs }
 fn div(lhs: Self, rhs: Self) -> Self { lhs / rhs }
 fn rem(lhs: Self, rhs: Self) -> Self { lhs % rhs }
 fn sort(value: Self) -> Self { value }
 fn min(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::U_MIN}, false>(lhs, rhs) } }
 fn max(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::U_MAX}, false>(lhs, rhs) } }
}

#[thermite_macros::inline_always]
impl IntegerRegister for [<u $width>] {
    fn mulhi(lhs: Self, rhs: Self) -> Self {
        // OpUMulExtended returns (lo, hi) pair; we want only hi
        unsafe { arch::spirv_umul_extended(lhs, rhs).1 }
    }

    fn mullo(lhs: Self, rhs: Self) -> Self { lhs.wrapping_mul(rhs) }

    // OpIAddCarry provides the carry bit at no extra cost; avoids widening to u64/u128.
    fn saturating_add(lhs: Self, rhs: Self) -> Self {
        let (sum, carry) = unsafe { arch::spirv_iadd_carry(lhs, rhs) };
        let overflow = carry != 0;
        unsafe { arch::op_opselect::<Self, bool>(overflow, sum, <$u>::MAX) }
    }
    // OpISubBorrow provides the borrow bit at no extra cost.
    fn saturating_sub(lhs: Self, rhs: Self) -> Self {
        let (diff, borrow) = unsafe { arch::spirv_isub_borrow(lhs, rhs) };
        let underflow = borrow != 0;
        unsafe { arch::op_opselect::<Self, bool>(underflow, diff, 0) }
    }

 fn wrapping_sum (value: Self) -> Self::Element { value }
 fn wrapping_product(value: Self) -> Self::Element { value }

    fn div_branched(value: Self, divider: crate::divider::Divider<Self::Element>) -> Self {
        divider.divide(value)
    }
    fn div_branchfree(value: Self, divider: crate::divider::BranchfreeDivider<Self::Element>) -> Self {
        divider.divide(value)
    }
    fn divv_branchfree(value: Self, dividers: crate::divider::vector::VectorDivider<Self>) -> Self {
        crate::divider::BranchfreeDivider::<$u>::new(dividers.multipliers.0, dividers.shifts.0 as i8 as u8).divide(value)
    }

    const HAS_HARDWARE_POPCNT: bool = true;

    // Use width-appropriate helpers; for 64-bit, OpBitCount is 32-bit-only so
    // we split into two halves (see spirv_count_ones_u64 in mod.rs).
    fn count_ones(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](value) as $u }
    }
    fn count_zeros(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](!value) as $u }
    }
    fn leading_zeros(value: Self) -> Self {
        unsafe { [<spirv_leading_zeros_u $width>](value) as $u }
    }
    fn trailing_zeros(value: Self) -> Self {
        unsafe { [<spirv_trailing_zeros_u $width>](value) as $u }
    }
    // Use BitwiseRegister::not so the inverted value stays the right type
    fn leading_ones(value: Self) -> Self { <Self as IntegerRegister>::leading_zeros(<Self as BitwiseRegister>::not(value)) }
    fn trailing_ones(value: Self) -> Self { <Self as IntegerRegister>::trailing_zeros(<Self as BitwiseRegister>::not(value)) }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for [<u $width>] {
    fn next_power_of_two_m1(value: Self) -> Self {
        if value == 0 {
            return 0;
        }
        // Fill all bits below and including the MSB: MAX >> leading_zeros(value - 1)
        let lz = Self::leading_zeros(value - 1);
        <$u>::MAX >> lz
    }

    fn is_power_of_two(value: Self) -> bool {
        value != 0 && (value & (value - 1)) == 0
    }

    fn parity(value: Self) -> Self {
        <Self as IntegerRegister>::count_ones(value) & 1
    }

    // GLSLstd450 FindUMsb(x) = floor(log2(x)) for x > 0, typically 0xFFFFFFFF for x = 0.
    // ilog2p1(x) = FindUMsb(x) + 1; for x = 0, 0xFFFFFFFF + 1 wraps to 0 correctly.
    fn ilog2p1(value: Self) -> Self {
        let msb = unsafe { arch::glsl_op1::<Self, Self, { glsl::FIND_U_MSB }, false>(value) };
        msb.wrapping_add(1)
    }
}

}}} // end macro

decl_spirv_unsigned_scalar!(u32 => 32);
decl_spirv_unsigned_scalar!(u64 => 64);
