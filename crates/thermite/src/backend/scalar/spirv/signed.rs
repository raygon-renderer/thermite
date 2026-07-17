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
        ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, UnsignedIntegerRegister, ZeroUpper,
    },
};

#[rustfmt::skip]
macro_rules! decl_spirv_signed_scalar { ($i:ty: $u:ty => $width:literal) => { paste::paste! {

#[thermite_macros::inline_always]
impl CoreRegister for [<i $width>] {
    type Lanes   = typenum::U1;
    type Storage = $i;
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
impl BitwiseRegister for [<i $width>] {
    fn bitxor(lhs: Self, rhs: Self) -> Self { lhs ^ rhs }
    fn bitand(lhs: Self, rhs: Self) -> Self { lhs & rhs }
    fn bitor (lhs: Self, rhs: Self) -> Self { lhs | rhs }
    fn not(value: Self) -> Self { !value }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for [<i $width>] {
    fn interleave (a: Self, b: Self) -> (Self, Self) { (a, b) }
    fn deinterleave(a: Self, b: Self) -> (Self, Self) { (a, b) }
}

#[thermite_macros::inline_always]
impl Register for [<i $width>] {
    type Element  = $i;
    type Signed   = [<i $width>];
    type Unsigned = [<u $width>];

    fn into_mask(value: Self) -> bool { value.to_bool() }
    fn msb_to_mask(value: Self) -> bool {
        // Arithmetic shift-right propagates sign bit to all positions
        Self::into_mask(value >> (<$i>::BITS - 1))
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self { value[0] }
    fn single(value: Self::Element) -> Self { value }
    fn splat (value: Self::Element) -> Self { value }
    fn broadcast<const I: usize>(value: Self) -> Self { value }
    fn reverse(value: Self) -> Self { value }

    fn swap_bytes(value: Self) -> Self {
        [<spirv_swap_bytes_u $width>](value as $u) as $i
    }

    const HAS_PERMUTEV: bool = false;
    fn permutev(value: Self, _idxs: GenericArray<u32, Self::Lanes>) -> Self { value }
    fn swizzle(a: Self, b: Self, idxs: GenericArray<u32, Self::Lanes>) -> Self {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[thermite_macros::inline_always]
impl<I: UnsignedIntegerRegister<Lanes = Self::Lanes>> IndexableRegister<I> for [<i $width>] {}

#[thermite_macros::inline_always]
impl BitshiftRegister for [<i $width>] {
    const HAS_TRUE_SHIFTV:      bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false; // no whole-vector byte-lane shift on SPIRV

    // Logical byte-granularity shifts (cast through unsigned to avoid arithmetic shift)
    fn bshli<const IMM8: i32>(value: Self) -> Self { (value as $u).unbounded_shl((8 * IMM8) as u32) as $i }
    fn bshri<const IMM8: i32>(value: Self) -> Self { (value as $u).unbounded_shr((8 * IMM8) as u32) as $i }

    // Logical bit shifts (cast through unsigned)
    fn shl (value: Self, shift: u32) -> Self { (value as $u).unbounded_shl(shift) as $i }
    fn shr (value: Self, shift: u32) -> Self { (value as $u).unbounded_shr(shift) as $i }
    fn shlv(value: Self, shifts: $u) -> Self { (value as $u).unbounded_shl(shifts as u32) as $i }
    fn shrv(value: Self, shifts: $u) -> Self { (value as $u).unbounded_shr(shifts as u32) as $i }
    fn shli<const IMM8: i32>(value: Self) -> Self { (value as $u).unbounded_shl(IMM8 as u32) as $i }
    fn shri<const IMM8: i32>(value: Self) -> Self { (value as $u).unbounded_shr(IMM8 as u32) as $i }

    fn rol (value: Self, shift: u32) -> Self { value.rotate_left (shift) }
    fn ror (value: Self, shift: u32) -> Self { value.rotate_right(shift) }
    fn rolv(value: Self, shifts: $u) -> Self { Self::rol(value, shifts as _) }
    fn rorv(value: Self, shifts: $u) -> Self { Self::ror(value, shifts as _) }

    // OpBitReverse: native SPIR-V instruction - override the default swap+shift chain.
    fn reverse_bits(value: Self) -> Self { unsafe { arch::op_opbitreverse::<Self>(value) } }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for [<i $width>] {
    fn shuffle<const IMM8: i32>(lhs: Self, rhs: Self) -> Self {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for [<i $width>] {
    fn permute<const IMM8: i32>(value: Self) -> Self { value }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl PartialOrdRegister for [<i $width>] {
    fn gt(lhs: Self, rhs: Self) -> bool { lhs > rhs }
    fn eq(lhs: Self, rhs: Self) -> bool { lhs == rhs }
    fn ge(lhs: Self, rhs: Self) -> bool { lhs >= rhs }
    fn lt(lhs: Self, rhs: Self) -> bool { lhs < rhs }
    fn le(lhs: Self, rhs: Self) -> bool { lhs <= rhs }
    fn ne(lhs: Self, rhs: Self) -> bool { lhs != rhs }
}

#[thermite_macros::inline_always]
impl NumericRegister for [<i $width>] {
    const ZERO: Self = 0;
    const ONE:  Self = 1;
    const TWO:  Self = 2;
    const MIN:  Self = <$i>::MIN;
    const MAX:  Self = <$i>::MAX;

    fn min_element (value: Self) -> Self::Element { value }
    fn max_element (value: Self) -> Self::Element { value }
    fn sum_elements(value: Self) -> Self::Element { value }
    fn prod_elements(value: Self) -> Self::Element { value }
    fn pairwise_sum(lo: Self, hi: Self) -> Self { lo.wrapping_add(hi) }
    fn offset() -> Self { 1 }
    fn indexed() -> Self { 0 }
    fn add(lhs: Self, rhs: Self) -> Self { lhs.wrapping_add(rhs) }
    fn sub(lhs: Self, rhs: Self) -> Self { lhs.wrapping_sub(rhs) }
    fn mul(lhs: Self, rhs: Self) -> Self { lhs.wrapping_mul(rhs) }
    fn div(lhs: Self, rhs: Self) -> Self { lhs.wrapping_div(rhs) }
    fn rem(lhs: Self, rhs: Self) -> Self { lhs.wrapping_rem(rhs) }
    fn sort(value: Self) -> Self { value }
    fn min(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::S_MIN}, false>(lhs, rhs) } }
    fn max(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::S_MAX}, false>(lhs, rhs) } }
}

#[thermite_macros::inline_always]
impl SignedRegister for [<i $width>] {
    const NEG_ONE:      Self = -1;
    const MIN_POSITIVE: Self =  1;

    // OpSNegate: canonical SPIR-V signed negation.
    fn neg(value: Self) -> Self { unsafe { arch::op_opsnegate::<Self>(value) } }

    // GLSLstd450 SAbs: cheaper than the default abs+select approach on GPU.
    fn abs(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::S_ABS }, false>(value) }
    }

    fn signum(value: Self) -> Self { value.signum() }

    // Branchless copysign via abs+blend - avoids branch divergence across SIMD lanes.
    // Edge case: lhs = MIN -> abs overflows to MIN; behavior matches scalar.
    fn copysign(lhs: Self, rhs: Self) -> Self {
        let abs_lhs = Self::abs(lhs);
        let neg_abs = Self::neg(abs_lhs);
        let rhs_neg = Self::is_negative(rhs);
        unsafe { arch::op_opselect::<Self, bool>(rhs_neg, neg_abs, abs_lhs) }
    }

    fn neg_c(mask: bool, value: Self) -> Self {
        unsafe { arch::op_opselect::<Self, bool>(mask, Self::neg(value), value) }
    }

    // MSB set <-> negative for two's-complement; compare-to-zero via OpSLessThan.
    fn is_negative(value: Self) -> bool {
        unsafe { arch::op_opslessthan::<bool, Self>(value, 0) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for [<i $width>] {
    fn mulhi(lhs: Self, rhs: Self) -> Self {
        // OpSMulExtended returns (lo, hi) pair; we want only hi
        unsafe { arch::spirv_smul_extended(lhs, rhs).1 }
    }

    fn mullo(lhs: Self, rhs: Self) -> Self { lhs.wrapping_mul(rhs) }

    // Bitwise overflow detection - avoids widening to i64/i128 which may be unsupported on GPU.
    // Overflow when inputs have the same sign but the sum has a different sign.
    fn saturating_add(lhs: Self, rhs: Self) -> Self {
        let sum      = lhs.wrapping_add(rhs);
        let overflow = (lhs ^ sum) & (rhs ^ sum);
        let ovf_mask = Self::is_negative(overflow);
        // Clamp to MIN on negative overflow (lhs < 0), MAX on positive overflow (lhs >= 0).
        let clamped  = if Self::is_negative(lhs) { <$i>::MIN } else { <$i>::MAX };
        unsafe { arch::op_opselect::<Self, bool>(ovf_mask, clamped, sum) }
    }
    // Overflow when inputs have opposite signs and result sign matches rhs.
    fn saturating_sub(lhs: Self, rhs: Self) -> Self {
        let diff     = lhs.wrapping_sub(rhs);
        let overflow = (lhs ^ rhs) & (lhs ^ diff);
        let ovf_mask = Self::is_negative(overflow);
        let clamped  = if Self::is_negative(lhs) { <$i>::MIN } else { <$i>::MAX };
        unsafe { arch::op_opselect::<Self, bool>(ovf_mask, clamped, diff) }
    }

 fn wrapping_sum (value: Self) -> Self::Element { value }
 fn wrapping_product(value: Self) -> Self::Element { value }

    // On GPU, integer division is always natively available
    fn div_branched(value: Self, divider: crate::divider::Divider<Self::Element>) -> Self {
        divider.divide(value)
    }
    fn div_branchfree(value: Self, divider: crate::divider::BranchfreeDivider<Self::Element>) -> Self {
        divider.divide(value)
    }
    fn divv_branchfree(value: Self, dividers: crate::divider::vector::VectorDivider<Self>) -> Self {
        crate::divider::BranchfreeDivider::<$i>::new(dividers.multipliers.0, dividers.shifts.0 as i8 as u8).divide(value)
    }

    const HAS_HARDWARE_POPCNT: bool = true;

    // Use width-appropriate helpers; for 64-bit, OpBitCount is 32-bit-only so
    // we split into two halves (see spirv_count_ones_u64 in mod.rs).
    fn count_ones(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](value as $u) as $i }
    }
    fn count_zeros(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](!(value as $u)) as $i }
    }
    fn leading_zeros(value: Self) -> Self {
        unsafe { [<spirv_leading_zeros_u $width>](value as $u) as $i }
    }
    fn trailing_zeros(value: Self) -> Self {
        unsafe { [<spirv_trailing_zeros_u $width>](value as $u) as $i }
    }
    // Use BitwiseRegister::not so the inverted value stays the right type
    fn leading_ones(value: Self) -> Self { <Self as IntegerRegister>::leading_zeros(<Self as BitwiseRegister>::not(value)) }
    fn trailing_ones(value: Self) -> Self { <Self as IntegerRegister>::trailing_zeros(<Self as BitwiseRegister>::not(value)) }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for [<i $width>] {
    // Rust `>>` on signed integers is arithmetic shift - compiles to OpShiftRightArithmetic
    fn srai<const IMM8: i32>(value: Self) -> Self { value.unbounded_shr(IMM8 as u32) }
    fn sra (value: Self, shift: u32) -> Self { value.unbounded_shr(shift) }
    fn srav(value: Self, shifts: $u) -> Self { value.unbounded_shr(shifts as u32) }
}

}}} // end macro

decl_spirv_signed_scalar!(i32: u32 => 32);
decl_spirv_signed_scalar!(i64: u64 => 64);
