use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitsRegister, BitshiftRegister, Element, FloatRegister, IntegerRegister, LinAlg3Register, MaskRegister,
    NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister,
    SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_signed_scalar { ($i:ty: $u:ty => $width:literal) => {paste::paste! {

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct [<I $width x1Scalar>];

impl Register for [<I $width x1Scalar>] {
    type Lanes = typenum::U1;

    type Element = $i;
    type Storage = $i;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    type SCOUNT = super::[<I $width x1Scalar>];
    type UCOUNT = super::[<U $width x1Scalar>];

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
}

impl BitshiftRegister for [<I $width x1Scalar>] {
    // NOTE: We do _NOT_ want arithmetic shift here, so we cast to unsigned first
    #[inline(always)] fn shl(value: Self::Storage, shift: u32) -> Self::Storage { ((value as $u) << shift) as $i }
    #[inline(always)] fn shr(value: Self::Storage, shift: u32) -> Self::Storage { ((value as $u) >> shift) as $i }
    #[inline(always)] fn shlv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage { ((value as $u) << shifts) as $i }
    #[inline(always)] fn shrv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage { ((value as $u) >> shifts) as $i }
    #[inline(always)] fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage { ((value as $u) << IMM8 as u32) as $i }
    #[inline(always)] fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage { ((value as $u) >> IMM8 as u32) as $i }

    #[inline(always)] fn rolv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage { Self::rol(value, shifts as _) }
    #[inline(always)] fn rorv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage { Self::ror(value, shifts as _) }
    #[inline(always)] fn rol(value: Self::Storage, shift: u32) -> Self::Storage { value.rotate_left(shift) }
    #[inline(always)] fn ror(value: Self::Storage, shift: u32) -> Self::Storage { value.rotate_right(shift) }

    #[inline(always)] fn reverse_bits(value: Self::Storage) -> Self::Storage { value.reverse_bits() }
}

impl ShuffleRegister for [<I $width x1Scalar>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<I $width x1Scalar>] {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        value
    }
}

impl SwizzleRegister for [<I $width x1Scalar>] {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        value
    }

    #[inline(always)]
    fn swizzle(a: Self::Storage, b: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

impl MaskRegister for [<I $width x1Scalar>] {
    const TRUTHY: Self::Storage = Element::TRUTHY;
    const FALSY: Self::Storage = Element::FALSY;

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        if value[0] { Self::TRUTHY } else { Self::FALSY }
    }

    #[inline(always)] fn all(value: Self::Storage) -> bool { value.to_bool() }
    #[inline(always)] fn any(value: Self::Storage) -> bool { value.to_bool() }
    #[inline(always)] fn none(value: Self::Storage) -> bool { value == 0 }
}

impl PartialOrdRegister for [<I $width x1Scalar>] {
    #[inline(always)] fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs != rhs) }
}

impl NumericRegister for [<I $width x1Scalar>] {
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

impl SignedRegister for [<I $width x1Scalar>] {
    const NEG_ONE: Self::Storage = -1;

    #[inline(always)] fn neg(value: Self::Storage) -> Self::Storage { -value }
    #[inline(always)] fn abs(value: Self::Storage) -> Self::Storage { value.abs() }
    #[inline(always)] fn signum(value: Self::Storage) -> Self::Storage { value.signum() }

    #[inline(always)] fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        match (lhs >= 0, rhs >= 0) {
            (true, false) | (false, true) => -lhs,
            (true, true) | (false, false) => lhs,
        }
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self::add(Self::bitxor(value, mask), mask >> const { <$i>::BITS - 1 })
    }
}

impl IntegerRegister for [<I $width x1Scalar>] {
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
        todo!()
    }

    #[inline(always)] fn count_ones(value: Self::Storage) -> Self::Storage { value.count_ones() as _ }
    #[inline(always)] fn count_zeros(value: Self::Storage) -> Self::Storage { value.count_ones() as _ }
    #[inline(always)] fn leading_zeros(value: Self::Storage) -> Self::Storage { value.leading_zeros() as _ }
    #[inline(always)] fn trailing_zeros(value: Self::Storage) -> Self::Storage { value.trailing_zeros() as _ }
    #[inline(always)] fn leading_ones(value: Self::Storage) -> Self::Storage { value.leading_ones() as _ }
    #[inline(always)] fn trailing_ones(value: Self::Storage) -> Self::Storage { value.trailing_ones() as _ }
}

impl SignedIntegerRegister for [<I $width x1Scalar>] {
    // NOTE: These _do_ use arithmetic shift
    #[inline(always)] fn srai<const IMM8: i32>(value: Self::Storage) -> Self::Storage { value >> IMM8 }
    #[inline(always)] fn sra(value: Self::Storage, shift: u32) -> Self::Storage { value >> shift }
    #[inline(always)] fn srav(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage { value >> shifts }
}

}}} // end macro

decl_signed_scalar!(i32: u32 => 32);
decl_signed_scalar!(i64: u64 => 64);
