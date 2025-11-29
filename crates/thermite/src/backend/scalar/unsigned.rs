use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitsRegister, BitshiftRegister, Element, FloatRegister, IntegerRegister, LinAlg3Register, MaskRegister,
    NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, Storage, SwizzleRegister,
    UnsignedIntegerRegister, dp::DoublePumpRegister, empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_unsigned_scalar { ($i:ty => $width:literal) => {paste::paste! {

impl Register for [<u $width>] {
    type Lanes = typenum::U1;

    type Element = $i;
    type Storage = $i;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::Scalar;

    type ISize = [<i $width>];
    type USize = [<u $width>];

    const EMPTY: Self::Storage = 0;

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage { value[0] }
    #[inline(always)] fn splat(value: Self::Element) -> Self::Storage { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Self::Storage) -> Self::Storage { value }
    #[inline(always)] fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs & rhs }
    #[inline(always)] fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs | rhs }
    #[inline(always)] fn not(value: Self::Storage) -> Self::Storage { !value }
    #[inline(always)] fn reverse(value: Self::Storage) -> Self::Storage { value }

    // use msb + cmov/csel on x86/x86_64/ARM/AArch64
    const HAS_MSB_BLENDV: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        core::hint::select_unpredictable((mask >> (<$i>::BITS - 1)) != 0, rhs, lhs)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if mask != 0 { rhs } else { lhs }
    }

    #[inline(always)]
    fn unpack(a: Self::Storage, b: Self::Storage) -> (Self::Storage, Self::Storage) {
        (a, b) // no-op for scalar
    }

    #[inline(always)]
    fn swap_bytes(value: Self::Storage) -> Self::Storage {
        value.swap_bytes()
    }
}

impl BitshiftRegister for [<u $width>] {
    #[inline(always)] fn shl(value: Self::Storage, shift: u32) -> Self::Storage { value << shift }
    #[inline(always)] fn shr(value: Self::Storage, shift: u32) -> Self::Storage { value >> shift }
    #[inline(always)] fn shlv(value: Self::Storage, shifts: Storage<Self::USize>) -> Self::Storage { value << shifts }
    #[inline(always)] fn shrv(value: Self::Storage, shifts: Storage<Self::USize>) -> Self::Storage { value >> shifts }
    #[inline(always)] fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage { value << IMM8 }
    #[inline(always)] fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage { value >> IMM8 }

    #[inline(always)] fn rolv(value: Self::Storage, shifts: Storage<Self::USize>) -> Self::Storage { Self::rol(value, shifts as _) }
    #[inline(always)] fn rorv(value: Self::Storage, shifts: Storage<Self::USize>) -> Self::Storage { Self::ror(value, shifts as _) }
    #[inline(always)] fn rol(value: Self::Storage, shift: u32) -> Self::Storage { value.rotate_left(shift) }
    #[inline(always)] fn ror(value: Self::Storage, shift: u32) -> Self::Storage { value.rotate_right(shift) }

    #[inline(always)] fn reverse_bits(value: Self::Storage) -> Self::Storage { value.reverse_bits() }
}

impl ShuffleRegister for [<u $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<u $width>] {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        value
    }
}

impl SwizzleRegister for [<u $width>] {
    const HAS_PERMUTEV: bool = false;

    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        value
    }

    #[inline(always)]
    fn swizzle(a: Self::Storage, b: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

impl MaskRegister for [<u $width>] {
    const TRUTHY: Self::Storage = Element::TRUTHY;
    const FALSY: Self::Storage = Element::FALSY;

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        if value[0] { <Self as MaskRegister>::TRUTHY } else { <Self as MaskRegister>::FALSY }
    }

    #[inline(always)] fn all(value: Self::Storage) -> bool { value != 0 }
    #[inline(always)] fn any(value: Self::Storage) -> bool { value != 0 }
    #[inline(always)] fn none(value: Self::Storage) -> bool { value == 0 }

    #[inline(always)]
    fn native_bitmask(value: Self::Storage) -> Option<u64> {
        Some((value & 1) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Self::Storage, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value.to_bool());
    }
}

impl PartialOrdRegister for [<u $width>] {
    #[inline(always)] fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs != rhs) }
}

impl NumericRegister for [<u $width>] {
    const ZERO: Self::Storage = 0;
    const ONE: Self::Storage = 1;
    const TWO: Self::Storage = 2;

    const MIN: Self::Storage = <$i>::MIN;
    const MAX: Self::Storage = <$i>::MAX;

    #[inline(always)] fn min_element(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn max_element(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn offset() -> Self::Storage { 1 }
    #[inline(always)] fn indexed() -> Self::Storage { 0 }
    #[inline(always)] fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs + rhs }
    #[inline(always)] fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs - rhs }
    #[inline(always)] fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs * rhs }
    #[inline(always)] fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs / rhs }
    #[inline(always)] fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs % rhs }
    #[inline(always)] fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.min(rhs) }
    #[inline(always)] fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.max(rhs) }
}

impl IntegerRegister for [<u $width>] {
    #[inline(always)] fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.saturating_add(rhs) }
    #[inline(always)] fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.saturating_sub(rhs) }
    #[inline(always)] fn wrapping_sum(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn wrapping_product(value: Self::Storage) -> Self::Element { value }

    #[inline(always)]
    fn div_branched(value: Self::Storage, divider: crate::divider::Divider<Self::Element>) -> Self::Storage {
        divider.divide(value)
    }

    #[inline(always)]
    fn div_branchfree(
        value: Self::Storage,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Self::Storage {
        divider.divide(value)
    }

    #[inline(always)]
    fn divv_branchfree(value: Self::Storage, dividers: crate::divider::vector::VectorDivider<Self>) -> Self::Storage {
        let m = dividers.multipliers;
        let s = dividers.shifts;

        // reconstitute scalar branchfree divider
        crate::divider::BranchfreeDivider::<$i>::new(m.0, s.0 as i8 as u8).divide(value)
    }

    #[inline(always)] fn count_ones(value: Self::Storage) -> Self::Storage { value.count_ones() as _ }
    #[inline(always)] fn count_zeros(value: Self::Storage) -> Self::Storage { value.count_ones() as _ }
    #[inline(always)] fn leading_zeros(value: Self::Storage) -> Self::Storage { value.leading_zeros() as _ }
    #[inline(always)] fn trailing_zeros(value: Self::Storage) -> Self::Storage { value.trailing_zeros() as _ }
    #[inline(always)] fn leading_ones(value: Self::Storage) -> Self::Storage { value.leading_ones() as _ }
    #[inline(always)] fn trailing_ones(value: Self::Storage) -> Self::Storage { value.trailing_ones() as _ }
}

impl UnsignedIntegerRegister for [<u $width>] {
    #[inline(always)]
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage {
        if value == 0 { 0 } else { value.next_power_of_two() - 1 }
    }

    #[inline(always)]
    fn is_power_of_two(value: Self::Storage) -> Self::Storage {
        Element::from_bool(value.is_power_of_two())
    }

    #[inline(always)]
    fn parity(value: Self::Storage) -> Self::Storage {
        Element::from_bool(value.count_ones() % 2 == 1)
    }
}

}}} // end macro

decl_unsigned_scalar!(u32 => 32);
decl_unsigned_scalar!(u64 => 64);
