use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitCastRegister, BitshiftRegister, BitwiseRegister, CoreRegister, Element, FloatElement, FloatRegister,
    IndexableRegister, InterleaveRegister, LinAlg3Register, MaskElement, MaskRegister, NativeCapability,
    NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
    UnsignedIntegerRegister, ZeroUpper, empty_reg, reg,
};

use crate::isa::InstructionSet;
use crate::vector::ops::MulAddExt;

#[rustfmt::skip]
macro_rules! decl_float_scalar { ($f:ty $(: $s:ident)? => $width:literal) => {paste::paste! {

#[thermite_macros::inline_always]
impl CoreRegister for [<f $width>] {
    type Lanes = typenum::U1;
    type Storage = [<f $width>];
    type Mask = bool;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::Scalar;
    const EMPTY: Storage<Self> = 0.0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, rhs, lhs)
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, value, 0.0)
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, 0.0, value)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 1 } { value } else { Self::EMPTY } // if N == 0 zero everything
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> { Self::from_bool(mask) }
}

#[thermite_macros::inline_always]
impl BitwiseRegister for [<f $width>] {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() ^ rhs.to_bits())
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() & rhs.to_bits())
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() | rhs.to_bits())
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(!value.to_bits())
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for [<f $width>] {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
}

impl Register for [<f $width>] {
    type Element = [<f $width>];

    type Signed = [<i $width>];
    type Unsigned = [<u $width>];

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> { value.to_bool() }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // scalars don't use MSB for mask conversion, but we can use NEG_ZERO to extract the sign bit
        // and convert to a mask
        Self::into_mask(Self::bitand(value, Self::NEG_ZERO))
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    fn single(value: Self::Element) -> Storage<Self> { value }
    fn splat(value: Self::Element) -> Storage<Self> { value }
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    fn reverse(value: Storage<Self>) -> Storage<Self> { value }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(value.to_bits().swap_bytes())
    }

    const HAS_PERMUTEV: bool = false;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        value
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[thermite_macros::inline_always]
impl<I> IndexableRegister<I> for [<f $width>]
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
{
}

#[thermite_macros::inline_always]
impl ShuffleRegister for [<f $width>] {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for [<f $width>] {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for [<f $width>] {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs > rhs }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs == rhs }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs >= rhs }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs < rhs }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs <= rhs }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs != rhs }
}

#[thermite_macros::inline_always]
impl NumericRegister for [<f $width>] {
    const ZERO: Storage<Self> = 0.0;
    const ONE: Storage<Self> = 1.0;
    const TWO: Storage<Self> = 2.0;

    const MIN: Storage<Self> = $f::MIN;
    const MAX: Storage<Self> = $f::MAX;

    fn min_element(value: Storage<Self>) -> Self::Element { value }
    fn max_element(value: Storage<Self>) -> Self::Element { value }
    fn sum_elements(value: Storage<Self>) -> Self::Element { value }
    fn prod_elements(value: Storage<Self>) -> Self::Element { value }
    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> { lo + hi }
    fn offset() -> Storage<Self> { 1.0 }
    fn indexed() -> Storage<Self> { 0.0 }
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs + rhs }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs - rhs }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs * rhs }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs / rhs }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs % rhs }
    fn sort(value: Storage<Self>) -> Storage<Self> { value } // no-op for scalar

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { core::hint::select_unpredictable(lhs < rhs, lhs, rhs) }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64"))]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { core::hint::select_unpredictable(lhs < rhs, rhs, lhs) }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { if lhs < rhs { lhs } else { rhs } }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { if lhs < rhs { rhs } else { lhs } }
}

#[thermite_macros::inline_always]
impl SignedRegister for [<f $width>] {
    const NEG_ONE: Storage<Self> = -1.0;
    const MIN_POSITIVE: Storage<Self> = <$f>::MIN_POSITIVE;

    fn neg(value: Storage<Self>) -> Storage<Self> { -value }
    fn abs(value: Storage<Self>) -> Storage<Self> { value.abs() }
    fn signum(value: Storage<Self>) -> Storage<Self> { value.signum() }
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.copysign(rhs) }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if mask { -value } else { value }
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for [<f $width>] {
    type Bits = [<u $width>];
    type SignedBits = [<i $width>];
    type ExtendedPrecision = f64;

    // best guess we can do
    const HAS_TRUE_FMA: bool = cfg!(any(
        all(feature = "spirv", target_arch = "spirv"),
        all(feature = "std", any(target_feature = "fma", target_feature = "avx2", target_feature = "avxifma", target_feature = "avx512ifma"))
    ));

    const HALF: Storage<Self> = 0.5;
    const NEG_ZERO: Storage<Self> = -0.0;
    const INFINITY: Storage<Self> = $f::INFINITY;
    const NEG_INFINITY: Storage<Self> = $f::NEG_INFINITY;
    const NAN: Storage<Self> = $f::NAN;
    const EPSILON: Storage<Self> = $f::EPSILON;

    const EXP_MASK: Storage<Self::Bits> = $f::INFINITY.to_bits(); // all exponent bits set

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_add(lhs, rhs, acc) }
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_sub(lhs, rhs, acc) }
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_add(lhs, rhs, acc) }
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_sub(lhs, rhs, acc) }

    fn sqrt(value: Storage<Self>) -> Storage<Self> { FloatElement::sqrt(value) }
    fn floor(value: Storage<Self>) -> Storage<Self> { FloatElement::floor(value) }
    fn ceil(value: Storage<Self>) -> Storage<Self> { FloatElement::ceil(value) }
    fn round(value: Storage<Self>) -> Storage<Self> { FloatElement::round(value) }
    fn trunc(value: Storage<Self>) -> Storage<Self> { FloatElement::trunc(value) }
    fn fract(value: Storage<Self>) -> Storage<Self> { FloatElement::fract(value) }
    fn next_up(value: Storage<Self>) -> Storage<Self> { FloatElement::next_up(value) }
    fn next_down(value: Storage<Self>) -> Storage<Self> { FloatElement::next_down(value) }

    // TODO: maybe at some point?
    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    unsafe fn block_autovectorization(value: &mut Storage<Self>) {
        unsafe {
            // x86_64: Use "xmm_reg" to keep it in the float/vector registers.
            // aarch64: Use "vreg" (or "reg" often works as floats are standard).
            #[cfg(target_arch = "x86_64")]
            core::arch::asm!(
                "/* {0} */",
                inout(xmm_reg) *value, // tied operand: reads xmmN, writes xmmN
                options(nomem, nostack, preserves_flags)
            );

            // `:v` names the whole vector register; the template is only a
            // comment (this is a pure compiler barrier), so the formatting is
            // cosmetic - the modifier just silences `asm_sub_register`.
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!(
                "/* {0:v} */",
                inout(vreg) *value,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}

}}} // end macro

decl_float_scalar!(f32 => 32);
decl_float_scalar!(f64 => 64);
