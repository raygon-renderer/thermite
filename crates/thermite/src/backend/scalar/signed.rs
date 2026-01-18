use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitsRegister, BitshiftRegister, CoreRegister, Element, FloatRegister, IntegerRegister, LinAlg3Register,
    NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister,
    SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_signed_scalar { ($i:ty: $u:ty: $ei:ty => $width:literal) => {paste::paste! {

impl CoreRegister for [<i $width>] {
    type Lanes = typenum::U1;
    type Element = $i;
    type Storage = $i;
}

impl Register for [<i $width>] {
    type HalfRegister = Self;
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::Scalar;

    type ISize = [<i $width>];
    type USize = [<u $width>];

    const EMPTY: Storage<Self> = 0;

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    #[inline(always)] fn single(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn splat(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    #[inline(always)] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
    #[inline(always)] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }
    #[inline(always)] fn not(value: Storage<Self>) -> Storage<Self> { !value }
    #[inline(always)] fn reverse(value: Storage<Self>) -> Storage<Self> { value }

    const TRUTHY: Storage<Self> = Element::TRUTHY;
    const FALSY: Storage<Self> = Element::FALSY;

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        if value[0] { <Self as Register>::TRUTHY } else { <Self as Register>::FALSY }
    }

    #[inline(always)] fn all(value: Storage<Self>) -> bool { value.to_bool() }
    #[inline(always)] fn any(value: Storage<Self>) -> bool { value.to_bool() }
    #[inline(always)] fn none(value: Storage<Self>) -> bool { value == 0 }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some((value & 1) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value.to_bool());
    }

    // use msb + cmov/csel on x86/x86_64/ARM/AArch64
    const HAS_MSB_BLENDV: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable((mask >> (<$i>::BITS - 1)) != 0, rhs, lhs)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if mask != 0 { rhs } else { lhs }
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        (a, b) // no-op for scalar
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value.swap_bytes()
    }
}

impl BitshiftRegister for [<i $width>] {
    const HAS_TRUE_SHIFTV: bool = true; // Technically true!
    const HAS_WIDE_BYTE_SHIFTS: bool = true; // Also technically true!

    // NOTE: We do _NOT_ want arithmetic shift here, so we cast to unsigned first
    #[inline(always)] fn bshli<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> { ((value as $u) << (8 * IMM8)) as $i }
    #[inline(always)] fn bshri<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> { ((value as $u) >> (8 * IMM8)) as $i }
    #[inline(always)] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> { ((value as $u) << shift) as $i }
    #[inline(always)] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> { ((value as $u) >> shift) as $i }
    #[inline(always)] fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { ((value as $u) << shifts) as $i }
    #[inline(always)] fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { ((value as $u) >> shifts) as $i }
    #[inline(always)] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { ((value as $u) << IMM8 as u32) as $i }
    #[inline(always)] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { ((value as $u) >> IMM8 as u32) as $i }

    #[inline(always)] fn rolv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self::rol(value, shifts as _) }
    #[inline(always)] fn rorv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self::ror(value, shifts as _) }
    #[inline(always)] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_left(shift) }
    #[inline(always)] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_right(shift) }

    #[inline(always)] fn reverse_bits(value: Storage<Self>) -> Storage<Self> { value.reverse_bits() }
}

impl ShuffleRegister for [<i $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<i $width>] {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        value
    }
}

impl SwizzleRegister for [<i $width>] {
    const HAS_PERMUTEV: bool = false;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        value
    }

    #[inline(always)]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

impl PartialOrdRegister for [<i $width>] {
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs != rhs) }
}

impl NumericRegister for [<i $width>] {
    const ZERO: Storage<Self> = 0;
    const ONE: Storage<Self> = 1;
    const TWO: Storage<Self> = 2;

    const MIN: Storage<Self> = <$i>::MIN;
    const MAX: Storage<Self> = <$i>::MAX;

    #[inline(always)] fn min_element(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn max_element(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn offset() -> Storage<Self> { 1 }
    #[inline(always)] fn indexed() -> Storage<Self> { 0 }
    #[inline(always)] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs + rhs }
    #[inline(always)] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs - rhs }
    #[inline(always)] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs * rhs }
    #[inline(always)] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs / rhs }
    #[inline(always)] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs % rhs }
    #[inline(always)] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.min(rhs) }
    #[inline(always)] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.max(rhs) }
    #[inline(always)] fn sort(value: Storage<Self>) -> Storage<Self> { value } // no-op for scalar
}

impl SignedRegister for [<i $width>] {
    const NEG_ONE: Storage<Self> = -1;
    const MIN_POSITIVE: Storage<Self> = 1;

    #[inline(always)] fn neg(value: Storage<Self>) -> Storage<Self> { -value }
    #[inline(always)] fn abs(value: Storage<Self>) -> Storage<Self> { value.abs() }
    #[inline(always)] fn signum(value: Storage<Self>) -> Storage<Self> { value.signum() }

    #[inline(always)] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        match (lhs >= 0, rhs >= 0) {
            (true, false) | (false, true) => -lhs,
            (true, true) | (false, false) => lhs,
        }
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), mask >> const { <$i>::BITS - 1 })
    }
}

impl IntegerRegister for [<i $width>] {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        (((lhs as $ei) * (rhs as $ei)) >> <$i>::BITS) as $i
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        ((lhs as $ei) * (rhs as $ei)) as $i
    }

    #[inline(always)] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.saturating_add(rhs) }
    #[inline(always)] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.saturating_sub(rhs) }
    #[inline(always)] fn wrapping_sum(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn wrapping_product(value: Storage<Self>) -> Self::Element { value }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::divider::Divider<Self::Element>) -> Storage<Self> {
        divider.divide(value)
    }

    #[inline(always)]
    fn div_branchfree(
        value: Storage<Self>,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Storage<Self> {
        divider.divide(value)
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        let m = dividers.multipliers;
        let s = dividers.shifts;

        // reconstitute scalar branchfree divider
        crate::divider::BranchfreeDivider::<$i>::new(m.0, s.0 as i8 as u8).divide(value)
    }

    const HAS_HARDWARE_POPCNT: bool = true;

    #[inline(always)] fn count_ones(value: Storage<Self>) -> Storage<Self> { value.count_ones() as _ }
    #[inline(always)] fn count_zeros(value: Storage<Self>) -> Storage<Self> { value.count_ones() as _ }
    #[inline(always)] fn leading_zeros(value: Storage<Self>) -> Storage<Self> { value.leading_zeros() as _ }
    #[inline(always)] fn trailing_zeros(value: Storage<Self>) -> Storage<Self> { value.trailing_zeros() as _ }
    #[inline(always)] fn leading_ones(value: Storage<Self>) -> Storage<Self> { value.leading_ones() as _ }
    #[inline(always)] fn trailing_ones(value: Storage<Self>) -> Storage<Self> { value.trailing_ones() as _ }
}

impl SignedIntegerRegister for [<i $width>] {
    // NOTE: These _do_ use arithmetic shift
    #[inline(always)] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value >> IMM8 }
    #[inline(always)] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> { value >> shift }
    #[inline(always)] fn srav(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { value >> shifts }
}

}}} // end macro

decl_signed_scalar!(i32: u32: i64 => 32);
decl_signed_scalar!(i64: u64: i128 => 64);
