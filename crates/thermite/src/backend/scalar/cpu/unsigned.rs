use generic_array::{
    GenericArray,
    typenum::{self},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitshiftRegister, BitwiseRegister, CoreRegister, IndexableRegister,
    IntegerRegister, InterleaveRegister, MaskElement, NumericRegister,
    PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, Storage, UnsignedIntegerRegister, ZeroUpper,
};

#[rustfmt::skip]
macro_rules! decl_unsigned_scalar { ($i:ty: $ei:ty => $width:literal) => {paste::paste! {

#[thermite_macros::inline_always]
impl CoreRegister for [<u $width>] {
    type NativeIsa = crate::backend::scalar::Scalar;
    type Lanes = typenum::U1;
    type Storage = $i;
    type Mask = bool;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = 0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, rhs, lhs)
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, value, 0)
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, 0, value)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 1 } { value } else { Self::EMPTY } // if N == 0 zero everything
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> { Self::from_bool(mask) }
}

#[thermite_macros::inline_always]
impl BitwiseRegister for [<u $width>] {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }
    fn not(value: Storage<Self>) -> Storage<Self> { !value }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for [<u $width>] {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
}

#[thermite_macros::inline_always]
impl Register for [<u $width>] {
    type Element = $i;

    type Signed = [<i $width>];
    type Unsigned = [<u $width>];

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> { value.to_bool() }
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::into_mask(value & (core::mem::size_of::<$i>() as Self * 8 - 1))
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    fn single(value: Self::Element) -> Storage<Self> { value }
    fn splat(value: Self::Element) -> Storage<Self> { value }
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    fn reverse(value: Storage<Self>) -> Storage<Self> { value }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value.swap_bytes()
    }

    const HAS_PERMUTEV: bool = false;

    fn permutev(value: Storage<Self>, _idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        value
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[thermite_macros::inline_always]
impl<I> IndexableRegister<I> for [<u $width>]
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
{
}

#[thermite_macros::inline_always]
impl BitshiftRegister for [<u $width>] {
    const HAS_TRUE_SHIFTV: bool = true; // Technically true!
    const HAS_WIDE_BYTE_SHIFTS: bool = true; // Also technically true!

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value.unbounded_shl((8 * IMM8) as u32) }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value.unbounded_shr((8 * IMM8) as u32) }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> { value.unbounded_shl(shift) }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> { value.unbounded_shr(shift) }
    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { value.unbounded_shl(shifts as u32) }
    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { value.unbounded_shr(shifts as u32) }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value.unbounded_shl(IMM8 as u32) }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value.unbounded_shr(IMM8 as u32) }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { Self::rol(value, shifts as _) }
    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { Self::ror(value, shifts as _) }
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_left(shift) }
    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_right(shift) }

    fn reverse_bits(value: Storage<Self>) -> Storage<Self> { value.reverse_bits() }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for [<u $width>] {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for [<u $width>] {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        value
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for [<u $width>] {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs > rhs }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs == rhs }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs >= rhs }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs < rhs }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs <= rhs }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs != rhs }
}

#[thermite_macros::inline_always]
impl NumericRegister for [<u $width>] {
    const ZERO: Storage<Self> = 0;
    const ONE: Storage<Self> = 1;
    const TWO: Storage<Self> = 2;

    const MIN: Storage<Self> = <$i>::MIN;
    const MAX: Storage<Self> = <$i>::MAX;

    fn min_element(value: Storage<Self>) -> Self::Element { value }
    fn max_element(value: Storage<Self>) -> Self::Element { value }
    fn sum_elements(value: Storage<Self>) -> Self::Element { value }
    fn prod_elements(value: Storage<Self>) -> Self::Element { value }
    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> { lo.wrapping_add(hi) }
    fn offset() -> Storage<Self> { 1 }
    fn indexed() -> Storage<Self> { 0 }
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.wrapping_add(rhs) }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.wrapping_sub(rhs) }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.wrapping_mul(rhs) }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.wrapping_div(rhs) }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.wrapping_rem(rhs) }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.min(rhs) }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.max(rhs) }
    fn sort(value: Storage<Self>) -> Storage<Self> { value } // no-op for scalar
}

#[thermite_macros::inline_always]
impl IntegerRegister for [<u $width>] {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        (((lhs as $ei) * (rhs as $ei)) >> <$i>::BITS) as $i
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        ((lhs as $ei) * (rhs as $ei)) as $i
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.saturating_add(rhs) }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.saturating_sub(rhs) }
    fn wrapping_sum(value: Storage<Self>) -> Self::Element { value }
    fn wrapping_product(value: Storage<Self>) -> Self::Element { value }

    fn div_branched(value: Storage<Self>, divider: crate::divider::Divider<Self::Element>) -> Storage<Self> {
        divider.divide(value)
    }

    fn div_branchfree(
        value: Storage<Self>,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Storage<Self> {
        divider.divide(value)
    }

    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        let m = dividers.multipliers;
        let s = dividers.shifts;

        // reconstitute scalar branchfree divider
        crate::divider::BranchfreeDivider::<$i>::new(m.0, s.0 as i8 as u8).divide(value)
    }

    const HAS_HARDWARE_POPCNT: bool = true;

    fn count_ones(value: Storage<Self>) -> Storage<Self> { value.count_ones() as _ }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> { value.count_zeros() as _ }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> { value.leading_zeros() as _ }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> { value.trailing_zeros() as _ }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> { value.leading_ones() as _ }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> { value.trailing_ones() as _ }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for [<u $width>] {
    fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {
        if value == 0 { 0 } else { value.next_power_of_two() - 1 }
    }

    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {
        value.is_power_of_two()
    }

    fn parity(value: Storage<Self>) -> Storage<Self> {
        value.count_ones() as Self & 1
    }
}

}}} // end macro

decl_unsigned_scalar!(u8: u16 => 8);
decl_unsigned_scalar!(u16: u32 => 16);
decl_unsigned_scalar!(u32: u64 => 32);
decl_unsigned_scalar!(u64: u128 => 64);
