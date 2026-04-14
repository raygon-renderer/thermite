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
        ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
        ZeroUpper,
    },
};

#[rustfmt::skip]
macro_rules! decl_spirv_signed_scalar { ($i:ty: $u:ty => $width:literal) => { paste::paste! {

impl CoreRegister for [<i $width>] {
    type Lanes   = typenum::U1;
    type Storage = $i;
    type Mask    = bool;

    const IS_EMULATED:         bool = false;
    const ISA:                 InstructionSet = InstructionSet::SPIRV;
    const EMPTY:               Storage<Self> = 0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    #[inline(always)]
    fn blendv(mask: bool, lhs: Self, rhs: Self) -> Self {
        unsafe { arch::op_opselect::<Self, bool>(mask, rhs, lhs) }
    }
    #[inline(always)] fn z (mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, value, 0) } }
    #[inline(always)] fn nz(mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, 0, value) } }
    #[inline(always)] fn zeroupper_z<Z: ZeroUpper>(value: Self) -> Self {
        if const { Z::N >= 1 } { value } else { Self::EMPTY }
    }
}

#[rustfmt::skip]
impl BitwiseRegister for [<i $width>] {
    #[inline(always)] fn bitxor(lhs: Self, rhs: Self) -> Self { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: Self, rhs: Self) -> Self { lhs & rhs }
    #[inline(always)] fn bitor (lhs: Self, rhs: Self) -> Self { lhs | rhs }
    #[inline(always)] fn not(value: Self) -> Self { !value }
}

#[rustfmt::skip]
impl InterleaveRegister for [<i $width>] {
    #[inline(always)] fn interleave  (a: Self, b: Self) -> (Self, Self) { (a, b) }
    #[inline(always)] fn deinterleave(a: Self, b: Self) -> (Self, Self) { (a, b) }
}

impl Register for [<i $width>] {
    type Element  = $i;
    type Signed   = [<i $width>];
    type Unsigned = [<u $width>];

    #[inline(always)] fn from_mask(mask: bool) -> Self { Self::from_bool(mask) }
    #[inline(always)] fn into_mask(value: Self) -> bool { value.to_bool() }
    #[inline(always)] fn msb_to_mask(value: Self) -> bool {
        // Arithmetic shift-right propagates sign bit to all positions
        Self::into_mask(value >> (<$i>::BITS - 1))
    }

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self { value[0] }
    #[inline(always)] fn single(value: Self::Element) -> Self { value }
    #[inline(always)] fn splat (value: Self::Element) -> Self { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Self) -> Self { value }
    #[inline(always)] fn reverse(value: Self) -> Self { value }

    #[inline(always)]
    fn swap_bytes(value: Self) -> Self {
        [<spirv_swap_bytes_u $width>](value as $u) as $i
    }
}

impl<I: UnsignedIntegerRegister<Lanes = Self::Lanes>> IndexableRegister<I> for [<i $width>] {}

impl BitshiftRegister for [<i $width>] {
    const HAS_TRUE_SHIFTV:      bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false; // no whole-vector byte-lane shift on SPIRV

    // Logical byte-granularity shifts (cast through unsigned to avoid arithmetic shift)
    #[inline(always)] fn bshli<const IMM8: i32>(value: Self) -> Self { ((value as $u) << (8 * IMM8)) as $i }
    #[inline(always)] fn bshri<const IMM8: i32>(value: Self) -> Self { ((value as $u) >> (8 * IMM8)) as $i }

    // Logical bit shifts (cast through unsigned)
    #[inline(always)] fn shl (value: Self, shift: u32) -> Self { ((value as $u) << shift) as $i }
    #[inline(always)] fn shr (value: Self, shift: u32) -> Self { ((value as $u) >> shift) as $i }
    #[inline(always)] fn shlv(value: Self, shifts: $u) -> Self { ((value as $u) << shifts) as $i }
    #[inline(always)] fn shrv(value: Self, shifts: $u) -> Self { ((value as $u) >> shifts) as $i }
    #[inline(always)] fn shli<const IMM8: i32>(value: Self) -> Self { ((value as $u) << IMM8 as u32) as $i }
    #[inline(always)] fn shri<const IMM8: i32>(value: Self) -> Self { ((value as $u) >> IMM8 as u32) as $i }

    #[inline(always)] fn rol (value: Self, shift: u32) -> Self { value.rotate_left (shift) }
    #[inline(always)] fn ror (value: Self, shift: u32) -> Self { value.rotate_right(shift) }
    #[inline(always)] fn rolv(value: Self, shifts: $u) -> Self { Self::rol(value, shifts as _) }
    #[inline(always)] fn rorv(value: Self, shifts: $u) -> Self { Self::ror(value, shifts as _) }
    // OpBitReverse: native SPIR-V instruction — override the default swap+shift chain.
    #[inline(always)] fn reverse_bits(value: Self) -> Self { unsafe { arch::op_opbitreverse::<Self>(value) } }
}

impl ShuffleRegister for [<i $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self, rhs: Self) -> Self {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<i $width>] {
    #[inline(always)] fn permute<const IMM8: i32>(value: Self) -> Self { value }
}

impl SwizzleRegister for [<i $width>] {
    const HAS_PERMUTEV: bool = false;
    #[inline(always)] fn permutev(value: Self, _idxs: GenericArray<u32, Self::Lanes>) -> Self { value }
    #[inline(always)] fn swizzle(a: Self, b: Self, idxs: GenericArray<u32, Self::Lanes>) -> Self {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for [<i $width>] {
    #[inline(always)] fn gt(lhs: Self, rhs: Self) -> bool { lhs > rhs }
    #[inline(always)] fn eq(lhs: Self, rhs: Self) -> bool { lhs == rhs }
    #[inline(always)] fn ge(lhs: Self, rhs: Self) -> bool { lhs >= rhs }
    #[inline(always)] fn lt(lhs: Self, rhs: Self) -> bool { lhs < rhs }
    #[inline(always)] fn le(lhs: Self, rhs: Self) -> bool { lhs <= rhs }
    #[inline(always)] fn ne(lhs: Self, rhs: Self) -> bool { lhs != rhs }
}

impl NumericRegister for [<i $width>] {
    const ZERO: Self = 0;
    const ONE:  Self = 1;
    const TWO:  Self = 2;
    const MIN:  Self = <$i>::MIN;
    const MAX:  Self = <$i>::MAX;

    #[inline(always)] fn min_element (value: Self) -> Self::Element { value }
    #[inline(always)] fn max_element (value: Self) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Self) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Self) -> Self::Element { value }
    #[inline(always)] fn offset()  -> Self { 1 }
    #[inline(always)] fn indexed() -> Self { 0 }
    #[inline(always)] fn add(lhs: Self, rhs: Self) -> Self { lhs + rhs }
    #[inline(always)] fn sub(lhs: Self, rhs: Self) -> Self { lhs - rhs }
    #[inline(always)] fn mul(lhs: Self, rhs: Self) -> Self { lhs * rhs }
    #[inline(always)] fn div(lhs: Self, rhs: Self) -> Self { lhs / rhs }
    #[inline(always)] fn rem(lhs: Self, rhs: Self) -> Self { lhs % rhs }
    #[inline(always)] fn sort(value: Self) -> Self { value }
    #[inline(always)] fn min(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::S_MIN}, false>(lhs, rhs) } }
    #[inline(always)] fn max(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::S_MAX}, false>(lhs, rhs) } }
}

impl SignedRegister for [<i $width>] {
    const NEG_ONE:      Self = -1;
    const MIN_POSITIVE: Self =  1;

    // OpSNegate: canonical SPIR-V signed negation.
    #[inline(always)] fn neg(value: Self) -> Self { unsafe { arch::op_opsnegate::<Self>(value) } }

    // GLSLstd450 SAbs: cheaper than the default abs+select approach on GPU.
    #[inline(always)] fn abs(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::S_ABS }, false>(value) }
    }

    #[inline(always)] fn signum(value: Self) -> Self { value.signum() }

    // Branchless copysign via abs+blend — avoids branch divergence across SIMD lanes.
    // Edge case: lhs = MIN -> abs overflows to MIN; behavior matches scalar.
    #[inline(always)]
    fn copysign(lhs: Self, rhs: Self) -> Self {
        let abs_lhs = Self::abs(lhs);
        let neg_abs = Self::neg(abs_lhs);
        let rhs_neg = Self::is_negative(rhs);
        unsafe { arch::op_opselect::<Self, bool>(rhs_neg, neg_abs, abs_lhs) }
    }

    #[inline(always)]
    fn neg_c(mask: bool, value: Self) -> Self {
        unsafe { arch::op_opselect::<Self, bool>(mask, Self::neg(value), value) }
    }

    // MSB set ↔ negative for two's-complement; compare-to-zero via OpSLessThan.
    #[inline(always)]
    fn is_negative(value: Self) -> bool {
        unsafe { arch::op_opslessthan::<bool, Self>(value, 0) }
    }
}

impl IntegerRegister for [<i $width>] {
    #[inline(always)]
    fn mulhi(lhs: Self, rhs: Self) -> Self {
        // OpSMulExtended returns (lo, hi) pair; we want only hi
        unsafe { arch::spirv_smul_extended(lhs, rhs).1 }
    }

    #[inline(always)]
    fn mullo(lhs: Self, rhs: Self) -> Self { lhs.wrapping_mul(rhs) }

    // Bitwise overflow detection — avoids widening to i64/i128 which may be unsupported on GPU.
    // Overflow when inputs have the same sign but the sum has a different sign.
    #[inline(always)]
    fn saturating_add(lhs: Self, rhs: Self) -> Self {
        let sum      = lhs.wrapping_add(rhs);
        let overflow = (lhs ^ sum) & (rhs ^ sum);
        let ovf_mask = Self::is_negative(overflow);
        // Clamp to MIN on negative overflow (lhs < 0), MAX on positive overflow (lhs >= 0).
        let clamped  = if Self::is_negative(lhs) { <$i>::MIN } else { <$i>::MAX };
        unsafe { arch::op_opselect::<Self, bool>(ovf_mask, clamped, sum) }
    }
    // Overflow when inputs have opposite signs and result sign matches rhs.
    #[inline(always)]
    fn saturating_sub(lhs: Self, rhs: Self) -> Self {
        let diff     = lhs.wrapping_sub(rhs);
        let overflow = (lhs ^ rhs) & (lhs ^ diff);
        let ovf_mask = Self::is_negative(overflow);
        let clamped  = if Self::is_negative(lhs) { <$i>::MIN } else { <$i>::MAX };
        unsafe { arch::op_opselect::<Self, bool>(ovf_mask, clamped, diff) }
    }

    #[inline(always)] fn wrapping_sum    (value: Self) -> Self::Element { value }
    #[inline(always)] fn wrapping_product(value: Self) -> Self::Element { value }

    // On GPU, integer division is always natively available
    #[inline(always)]
    fn div_branched(value: Self, divider: crate::divider::Divider<Self::Element>) -> Self {
        divider.divide(value)
    }
    #[inline(always)]
    fn div_branchfree(value: Self, divider: crate::divider::BranchfreeDivider<Self::Element>) -> Self {
        divider.divide(value)
    }
    #[inline(always)]
    fn divv_branchfree(value: Self, dividers: crate::divider::vector::VectorDivider<Self>) -> Self {
        crate::divider::BranchfreeDivider::<$i>::new(dividers.multipliers.0, dividers.shifts.0 as i8 as u8).divide(value)
    }

    const HAS_HARDWARE_POPCNT: bool = true;

    // Use width-appropriate helpers; for 64-bit, OpBitCount is 32-bit-only so
    // we split into two halves (see spirv_count_ones_u64 in mod.rs).
    #[inline(always)]
    fn count_ones(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](value as $u) as $i }
    }
    #[inline(always)]
    fn count_zeros(value: Self) -> Self {
        unsafe { [<spirv_count_ones_u $width>](!(value as $u)) as $i }
    }
    #[inline(always)]
    fn leading_zeros(value: Self) -> Self {
        unsafe { [<spirv_leading_zeros_u $width>](value as $u) as $i }
    }
    #[inline(always)]
    fn trailing_zeros(value: Self) -> Self {
        unsafe { [<spirv_trailing_zeros_u $width>](value as $u) as $i }
    }
    // Use BitwiseRegister::not so the inverted value stays the right type
    #[inline(always)]
    fn leading_ones(value: Self) -> Self { <Self as IntegerRegister>::leading_zeros(<Self as BitwiseRegister>::not(value)) }
    #[inline(always)]
    fn trailing_ones(value: Self) -> Self { <Self as IntegerRegister>::trailing_zeros(<Self as BitwiseRegister>::not(value)) }
}

impl SignedIntegerRegister for [<i $width>] {
    // Rust `>>` on signed integers is arithmetic shift — compiles to OpShiftRightArithmetic
    #[inline(always)] fn srai<const IMM8: i32>(value: Self) -> Self { value >> IMM8  }
    #[inline(always)] fn sra (value: Self, shift: u32)      -> Self { value >> shift }
    #[inline(always)] fn srav(value: Self, shifts: $u)      -> Self { value >> shifts }
}

}}} // end macro

decl_spirv_signed_scalar!(i32: u32 => 32);
decl_spirv_signed_scalar!(i64: u64 => 64);
