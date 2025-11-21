use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitsRegister, BitshiftRegister, Element, FloatElement, FloatRegister, LinAlg3Register, MaskRegister,
    NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
    SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_float_scalar { ($f:ty $(: $s:ident)? => $width:literal) => {paste::paste! {

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct [<F $width x1Scalar>];

impl Register for [<F $width x1Scalar>] {
    type Lanes = typenum::U1;

    type Element = [<f $width>];
    type Storage = [<f $width>];
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::Scalar;

    type SCOUNT = super::[<I $width x1Scalar>];
    type UCOUNT = super::[<U $width x1Scalar>];

    const EMPTY: Self::Storage = 0.0;

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage { value[0] }
    #[inline(always)] fn splat(value: Self::Element) -> Self::Storage { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Self::Storage) -> Self::Storage { value }
    #[inline(always)] fn reverse(value: Self::Storage) -> Self::Storage { value }

    #[inline(always)] fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        $f::from_bits(lhs.to_bits() ^ rhs.to_bits())
    }

    #[inline(always)] fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        $f::from_bits(lhs.to_bits() & rhs.to_bits())
    }

    #[inline(always)] fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        $f::from_bits(lhs.to_bits() | rhs.to_bits())
    }

    #[inline(always)] fn not(value: Self::Storage) -> Self::Storage {
        $f::from_bits(!value.to_bits())
    }

    // use msb + cmov/csel on x86/x86_64/ARM/AArch64
    const HAS_MSB_BLENDV: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    #[inline(always)] fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        core::hint::select_unpredictable((mask.to_bits() >> 31) != 0, rhs, lhs)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)] fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if mask != 0.0 { rhs } else { lhs }
    }

    #[inline(always)]
    fn unpack(a: Self::Storage, b: Self::Storage) -> (Self::Storage, Self::Storage) {
        (a, b) // no-op for scalar
    }
}

impl ShuffleRegister for [<F $width x1Scalar>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<F $width x1Scalar>] {
    #[inline(always)] fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage { value }
}

impl SwizzleRegister for [<F $width x1Scalar>] {
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

impl MaskRegister for [<F $width x1Scalar>] {
    const TRUTHY: Self::Storage = Element::TRUTHY;
    const FALSY: Self::Storage = Element::FALSY;

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        if value[0] { Self::TRUTHY } else { Self::FALSY }
    }

    #[inline(always)] fn all(value: Self::Storage) -> bool { value.to_bool() }
    #[inline(always)] fn any(value: Self::Storage) -> bool { value.to_bool() }

    #[inline(always)]
    fn native_bitmask(value: Self::Storage) -> Option<u64> {
        Some(value.to_bool() as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Self::Storage, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value.to_bool());
    }
}

impl PartialOrdRegister for [<F $width x1Scalar>] {
    #[inline(always)] fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { Element::from_bool(lhs != rhs) }
}

impl NumericRegister for [<F $width x1Scalar>] {
    const ZERO: Self::Storage = 0.0;
    const ONE: Self::Storage = 1.0;
    const TWO: Self::Storage = 2.0;

    const MIN: Self::Storage = $f::MIN;
    const MAX: Self::Storage = $f::MAX;

    #[inline(always)] fn min_element(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn max_element(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Self::Storage) -> Self::Element { value }
    #[inline(always)] fn offset() -> Self::Storage { 1.0 }
    #[inline(always)] fn indexed() -> Self::Storage { 0.0 }
    #[inline(always)] fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs + rhs }
    #[inline(always)] fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs - rhs }
    #[inline(always)] fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs * rhs }
    #[inline(always)] fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs / rhs }
    #[inline(always)] fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs % rhs }
    #[inline(always)] fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.min(rhs) }
    #[inline(always)] fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.max(rhs) }
}

impl SignedRegister for [<F $width x1Scalar>] {
    const NEG_ONE: Self::Storage = -1.0;
    const MIN_POSITIVE: Self::Storage = <$f>::MIN_POSITIVE;

    #[inline(always)] fn neg(value: Self::Storage) -> Self::Storage { -value }
    #[inline(always)] fn abs(value: Self::Storage) -> Self::Storage { value.abs() }
    #[inline(always)] fn signum(value: Self::Storage) -> Self::Storage { value.signum() }
    #[inline(always)] fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage { lhs.copysign(rhs) }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        if mask.to_bool() { -value } else { value }
    }
}

impl FloatRegister for [<F $width x1Scalar>] {
    type Bits = super::[<U $width x1Scalar>];
    type Signed = super::[<I $width x1Scalar>];
    type ExtendedPrecision = super::F64x1Scalar;

    // best guess we can do
    const HAS_TRUE_FMA: bool = cfg!(any(target_feature = "fma", target_feature = "avx2", target_feature = "avxifma", target_feature = "avx512ifma"));

    const HALF: Self::Storage = 0.5;
    const NEG_ZERO: Self::Storage = -0.0;
    const INFINITY: Self::Storage = $f::INFINITY;
    const NEG_INFINITY: Self::Storage = $f::NEG_INFINITY;
    const NAN: Self::Storage = $f::NAN;
    const EPSILON: Self::Storage = $f::EPSILON;

    const EXP_MASK: Storage<Self::Bits> = $f::INFINITY.to_bits(); // all exponent bits set

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    #[inline(always)] fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage { FloatElement::scalar_mul_add(lhs, rhs, acc) }
    #[inline(always)] fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage { FloatElement::scalar_mul_sub(lhs, rhs, acc) }
    #[inline(always)] fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage { FloatElement::scalar_nmul_add(lhs, rhs, acc) }
    #[inline(always)] fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage { FloatElement::scalar_nmul_sub(lhs, rhs, acc) }

    #[inline(always)] fn sqrt(value: Self::Storage) -> Self::Storage { FloatElement::sqrt(value) }
    #[inline(always)] fn floor(value: Self::Storage) -> Self::Storage { FloatElement::floor(value) }
    #[inline(always)] fn ceil(value: Self::Storage) -> Self::Storage { FloatElement::ceil(value) }
    #[inline(always)] fn round(value: Self::Storage) -> Self::Storage { FloatElement::round(value) }
    #[inline(always)] fn trunc(value: Self::Storage) -> Self::Storage { FloatElement::trunc(value) }
    #[inline(always)] fn fract(value: Self::Storage) -> Self::Storage { FloatElement::fract(value) }
    #[inline(always)] fn next_up(value: Self::Storage) -> Self::Storage { FloatElement::next_up(value) }
    #[inline(always)] fn next_down(value: Self::Storage) -> Self::Storage { FloatElement::next_down(value) }
}

}}} // end macro

decl_float_scalar!(f32 => 32);
decl_float_scalar!(f64 => 64);
