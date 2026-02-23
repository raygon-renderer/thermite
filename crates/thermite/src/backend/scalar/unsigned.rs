use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitCastRegister, BitshiftRegister, BitwiseRegister, CoreRegister, Element, FloatRegister, IndexableRegister,
    IntegerRegister, LinAlg3Register, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister,
    Register, ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, ZeroUpper, dp::DoublePumpRegister,
    empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_unsigned_scalar { ($i:ty: $ei:ty => $width:literal) => {paste::paste! {

impl CoreRegister for [<u $width>] {
    type Lanes = typenum::U1;
    type Storage = $i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::Scalar;

    const EMPTY: Storage<Self> = 0;

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask != 0, rhs, lhs)
    }

    #[inline(always)] fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask != 0, value, 0)
    }

    #[inline(always)] fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask == 0, value, 0)
    }

    #[inline(always)] fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 1 } { value } else { Self::EMPTY } // if N == 0 zero everything
    }
}

impl BitwiseRegister for [<u $width>] {
    #[inline(always)] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
    #[inline(always)] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }
    #[inline(always)] fn not(value: Storage<Self>) -> Storage<Self> { !value }
}

impl MaskRegister for [<u $width>] {
    const TRUTHY: Storage<Self> = MaskElement::TRUTHY;
    const FALSY: Storage<Self> = MaskElement::FALSY;

    #[inline(always)]
    fn set(mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        if value { <Self as MaskRegister>::TRUTHY } else { <Self as MaskRegister>::FALSY }
    }

    #[inline(always)]
    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool { mask.to_bool() }

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        if value[0] { <Self as MaskRegister>::TRUTHY } else { <Self as MaskRegister>::FALSY }
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
}

impl Register for [<u $width>] {
    type Element = $i;

    type Signed = [<i $width>];
    type Unsigned = [<u $width>];

    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)] fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> { value }
    #[inline(always)] fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> { mask }
    #[inline(always)] fn into_mask(value: Storage<Self>) -> Storage<Self> {
        if value.to_bool() { <Self as MaskRegister>::TRUTHY } else { <Self as MaskRegister>::FALSY }
    }
    #[inline(always)] fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        ((value as [<i $width>]) >> (core::mem::size_of::<$i>() * 8 - 1)) as _ // arithmetic shift to propagate sign bit
    }

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    #[inline(always)] fn single(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn splat(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    #[inline(always)] fn reverse(value: Storage<Self>) -> Storage<Self> { value }

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

impl<I> IndexableRegister<I> for [<u $width>]
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
{
}

impl BitshiftRegister for [<u $width>] {
    const HAS_TRUE_SHIFTV: bool = true; // Technically true!
    const HAS_WIDE_BYTE_SHIFTS: bool = true; // Also technically true!

    #[inline(always)] fn bshli<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> { value << (8 * IMM8) }
    #[inline(always)] fn bshri<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> { value >> (8 * IMM8) }
    #[inline(always)] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> { value << shift }
    #[inline(always)] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> { value >> shift }
    #[inline(always)] fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { value << shifts }
    #[inline(always)] fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { value >> shifts }
    #[inline(always)] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value << IMM8 }
    #[inline(always)] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value >> IMM8 }

    #[inline(always)] fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { Self::rol(value, shifts as _) }
    #[inline(always)] fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> { Self::ror(value, shifts as _) }
    #[inline(always)] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_left(shift) }
    #[inline(always)] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> { value.rotate_right(shift) }

    #[inline(always)] fn reverse_bits(value: Storage<Self>) -> Storage<Self> { value.reverse_bits() }
}

impl ShuffleRegister for [<u $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<u $width>] {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        value
    }
}

impl SwizzleRegister for [<u $width>] {
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

impl PartialOrdRegister for [<u $width>] {
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { MaskElement::from_bool(lhs != rhs) }
}

impl NumericRegister for [<u $width>] {
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

impl IntegerRegister for [<u $width>] {
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

impl UnsignedIntegerRegister for [<u $width>] {
    #[inline(always)]
    fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {
        if value == 0 { 0 } else { value.next_power_of_two() - 1 }
    }

    #[inline(always)]
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {
        MaskElement::from_bool(value.is_power_of_two())
    }

    #[inline(always)]
    fn parity(value: Storage<Self>) -> Storage<Self> {
        MaskElement::from_bool(value.count_ones() % 2 == 1)
    }
}

}}} // end macro

decl_unsigned_scalar!(u8: u16 => 8);
decl_unsigned_scalar!(u16: u32 => 16);
decl_unsigned_scalar!(u32: u64 => 32);
decl_unsigned_scalar!(u64: u128 => 64);
