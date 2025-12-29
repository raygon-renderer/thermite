use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::isa::InstructionSet;
use crate::register::{
    BitsRegister, BitshiftRegister, Element, FloatElement, FloatRegister, LinAlg3Register, MaskRegister,
    NumericRegister, PartialMaskRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
    SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
};

#[rustfmt::skip]
macro_rules! decl_float_scalar { ($f:ty $(: $s:ident)? => $width:literal) => {paste::paste! {

impl Register for [<f $width>] {
    type Lanes = typenum::U1;

    type Element = [<f $width>];
    type Storage = [<f $width>];
    type HalfRegister = Self;
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::Scalar;

    type ISize = [<i $width>];
    type USize = [<u $width>];

    const EMPTY: Storage<Self> = 0.0;

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    #[inline(always)] fn single(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn splat(value: Self::Element) -> Storage<Self> { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    #[inline(always)] fn reverse(value: Storage<Self>) -> Storage<Self> { value }

    #[inline(always)] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() ^ rhs.to_bits())
    }

    #[inline(always)] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() & rhs.to_bits())
    }

    #[inline(always)] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() | rhs.to_bits())
    }

    #[inline(always)] fn not(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(!value.to_bits())
    }

    // use msb + cmov/csel on x86/x86_64/ARM/AArch64
    const HAS_MSB_BLENDV: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    #[inline(always)] fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable((mask.to_bits() >> 31) != 0, rhs, lhs)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)] fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if mask != 0.0 { rhs } else { lhs }
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        (a, b) // no-op for scalar
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(value.to_bits().swap_bytes())
    }
}

impl ShuffleRegister for [<f $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<f $width>] {
    #[inline(always)] fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value }
}

impl SwizzleRegister for [<f $width>] {
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

impl PartialMaskRegister for [<f $width>] {
    const TRUTHY: Storage<Self> = Element::TRUTHY;
    const FALSY: Storage<Self> = Element::FALSY;

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        if value[0] { <Self as PartialMaskRegister>::TRUTHY } else { <Self as PartialMaskRegister>::FALSY }
    }

    #[inline(always)] fn all(value: Storage<Self>) -> bool { value.to_bool() }
    #[inline(always)] fn any(value: Storage<Self>) -> bool { value.to_bool() }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(value.to_bool() as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value.to_bool());
    }
}

impl PartialOrdRegister for [<f $width>] {
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs > rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs == rhs) }
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs >= rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs < rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs <= rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Element::from_bool(lhs != rhs) }
}

impl NumericRegister for [<f $width>] {
    const ZERO: Storage<Self> = 0.0;
    const ONE: Storage<Self> = 1.0;
    const TWO: Storage<Self> = 2.0;

    const MIN: Storage<Self> = $f::MIN;
    const MAX: Storage<Self> = $f::MAX;

    #[inline(always)] fn min_element(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn max_element(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Storage<Self>) -> Self::Element { value }
    #[inline(always)] fn offset() -> Storage<Self> { 1.0 }
    #[inline(always)] fn indexed() -> Storage<Self> { 0.0 }
    #[inline(always)] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs + rhs }
    #[inline(always)] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs - rhs }
    #[inline(always)] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs * rhs }
    #[inline(always)] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs / rhs }
    #[inline(always)] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs % rhs }
    #[inline(always)] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.min(rhs) }
    #[inline(always)] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.max(rhs) }
    #[inline(always)] fn sort(value: Storage<Self>) -> Storage<Self> { value } // no-op for scalar
}

impl SignedRegister for [<f $width>] {
    const NEG_ONE: Storage<Self> = -1.0;
    const MIN_POSITIVE: Storage<Self> = <$f>::MIN_POSITIVE;

    #[inline(always)] fn neg(value: Storage<Self>) -> Storage<Self> { -value }
    #[inline(always)] fn abs(value: Storage<Self>) -> Storage<Self> { value.abs() }
    #[inline(always)] fn signum(value: Storage<Self>) -> Storage<Self> { value.signum() }
    #[inline(always)] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.copysign(rhs) }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        if mask.to_bool() { -value } else { value }
    }
}

impl FloatRegister for [<f $width>] {
    type Bits = [<u $width>];
    type Signed = [<i $width>];
    type ExtendedPrecision = f64;

    // best guess we can do
    const HAS_TRUE_FMA: bool = cfg!(any(target_feature = "fma", target_feature = "avx2", target_feature = "avxifma", target_feature = "avx512ifma"));

    const HALF: Storage<Self> = 0.5;
    const NEG_ZERO: Storage<Self> = -0.0;
    const INFINITY: Storage<Self> = $f::INFINITY;
    const NEG_INFINITY: Storage<Self> = $f::NEG_INFINITY;
    const NAN: Storage<Self> = $f::NAN;
    const EPSILON: Storage<Self> = $f::EPSILON;

    const EXP_MASK: Storage<Self::Bits> = $f::INFINITY.to_bits(); // all exponent bits set

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    #[inline(always)] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { FloatElement::scalar_mul_add(lhs, rhs, acc) }
    #[inline(always)] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { FloatElement::scalar_mul_sub(lhs, rhs, acc) }
    #[inline(always)] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { FloatElement::scalar_nmul_add(lhs, rhs, acc) }
    #[inline(always)] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { FloatElement::scalar_nmul_sub(lhs, rhs, acc) }

    #[inline(always)] fn sqrt(value: Storage<Self>) -> Storage<Self> { FloatElement::sqrt(value) }
    #[inline(always)] fn floor(value: Storage<Self>) -> Storage<Self> { FloatElement::floor(value) }
    #[inline(always)] fn ceil(value: Storage<Self>) -> Storage<Self> { FloatElement::ceil(value) }
    #[inline(always)] fn round(value: Storage<Self>) -> Storage<Self> { FloatElement::round(value) }
    #[inline(always)] fn trunc(value: Storage<Self>) -> Storage<Self> { FloatElement::trunc(value) }
    #[inline(always)] fn fract(value: Storage<Self>) -> Storage<Self> { FloatElement::fract(value) }
    #[inline(always)] fn next_up(value: Storage<Self>) -> Storage<Self> { FloatElement::next_up(value) }
    #[inline(always)] fn next_down(value: Storage<Self>) -> Storage<Self> { FloatElement::next_down(value) }

    // TODO: maybe at some point?
    const HAS_NATIVE_LDEXP: bool = false;
    const HAS_NATIVE_FREXP: bool = false;
}

}}} // end macro

decl_float_scalar!(f32 => 32);
decl_float_scalar!(f64 => 64);
