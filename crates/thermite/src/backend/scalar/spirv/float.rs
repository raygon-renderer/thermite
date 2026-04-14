use generic_array::{GenericArray, typenum};

use super::{spirv_swap_bytes_u32, spirv_swap_bytes_u64};
use crate::backend::spirv::arch::{self as arch, glsl};
use crate::{
    isa::InstructionSet,
    math::policy::{Policy, PrecisionPolicy},
    register::{
        BitwiseRegister, CoreRegister, Element, FloatElement, FloatRegister, IndexableRegister, InterleaveRegister,
        MaskElement, NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, ZeroUpper,
    },
    vector::ops::MulAddExt,
};

#[rustfmt::skip]
macro_rules! decl_spirv_float_scalar { ($f:ty => $width:literal) => { paste::paste! {

impl CoreRegister for [<f $width>] {
    type Lanes   = typenum::U1;
    type Storage = [<f $width>];
    type Mask    = bool;

    const IS_EMULATED:         bool = false;
    const ISA:                 InstructionSet = InstructionSet::SPIRV;
    const EMPTY:               Storage<Self> = 0.0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    #[inline(always)]
    fn blendv(mask: bool, lhs: Self, rhs: Self) -> Self {
        // OpSelect: true -> rhs, false -> lhs  (matches blendv semantics)
        unsafe { arch::op_opselect::<Self, bool>(mask, rhs, lhs) }
    }
    #[inline(always)] fn z (mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, value, 0.0) } }
    #[inline(always)] fn nz(mask: bool, value: Self) -> Self { unsafe { arch::op_opselect::<Self, bool>(mask, 0.0, value) } }
    #[inline(always)] fn zeroupper_z<Z: ZeroUpper>(value: Self) -> Self {
        if const { Z::N >= 1 } { value } else { Self::EMPTY }
    }
}

impl BitwiseRegister for [<f $width>] {
    // f32::to_bits / f32::from_bits compile to OpBitcast on SPIRV
    #[inline(always)] fn bitxor(lhs: Self, rhs: Self) -> Self { <$f>::from_bits(lhs.to_bits() ^ rhs.to_bits()) }
    #[inline(always)] fn bitand(lhs: Self, rhs: Self) -> Self { <$f>::from_bits(lhs.to_bits() & rhs.to_bits()) }
    #[inline(always)] fn bitor (lhs: Self, rhs: Self) -> Self { <$f>::from_bits(lhs.to_bits() | rhs.to_bits()) }
    #[inline(always)] fn not(value: Self) -> Self             { <$f>::from_bits(!value.to_bits()) }
}

#[rustfmt::skip]
impl InterleaveRegister for [<f $width>] {
    #[inline(always)] fn interleave  (a: Self, b: Self) -> (Self, Self) { (a, b) }
    #[inline(always)] fn deinterleave(a: Self, b: Self) -> (Self, Self) { (a, b) }
}

impl Register for [<f $width>] {
    type Element  = [<f $width>];
    type Signed   = [<i $width>];
    type Unsigned = [<u $width>];

    #[inline(always)] fn from_mask(mask: bool) -> Self { Self::from_bool(mask) }
    #[inline(always)] fn into_mask(value: Self) -> bool { value.to_bool() }
    #[inline(always)] fn msb_to_mask(value: Self) -> bool {
        Self::into_mask(Self::bitand(value, Self::NEG_ZERO))
    }

    #[inline(always)] fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self { value[0] }
    #[inline(always)] fn single(value: Self::Element) -> Self { value }
    #[inline(always)] fn splat (value: Self::Element) -> Self { value }
    #[inline(always)] fn broadcast<const I: usize>(value: Self) -> Self { value }
    #[inline(always)] fn reverse(value: Self) -> Self { value }

    #[inline(always)]
    fn swap_bytes(value: Self) -> Self {
        <$f>::from_bits([<spirv_swap_bytes_u $width>](value.to_bits()))
    }
}

impl<I: UnsignedIntegerRegister<Lanes = Self::Lanes>> IndexableRegister<I> for [<f $width>] {}

impl ShuffleRegister for [<f $width>] {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self, rhs: Self) -> Self {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

impl PermuteRegister for [<f $width>] {
    #[inline(always)] fn permute<const IMM8: i32>(value: Self) -> Self { value }
}

impl SwizzleRegister for [<f $width>] {
    const HAS_PERMUTEV: bool = false;
    #[inline(always)] fn permutev(value: Self, _idxs: GenericArray<u32, Self::Lanes>) -> Self { value }
    #[inline(always)] fn swizzle(a: Self, b: Self, idxs: GenericArray<u32, Self::Lanes>) -> Self {
        if idxs[0] & 0b1 == 0 { a } else { b }
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for [<f $width>] {
    #[inline(always)] fn gt(lhs: Self, rhs: Self) -> bool { lhs > rhs }
    #[inline(always)] fn eq(lhs: Self, rhs: Self) -> bool { lhs == rhs }
    #[inline(always)] fn ge(lhs: Self, rhs: Self) -> bool { lhs >= rhs }
    #[inline(always)] fn lt(lhs: Self, rhs: Self) -> bool { lhs < rhs }
    #[inline(always)] fn le(lhs: Self, rhs: Self) -> bool { lhs <= rhs }
    #[inline(always)] fn ne(lhs: Self, rhs: Self) -> bool { lhs != rhs }
}

impl NumericRegister for [<f $width>] {
    const ZERO: Self = 0.0;
    const ONE:  Self = 1.0;
    const TWO:  Self = 2.0;
    const MIN:  Self = <$f>::MIN;
    const MAX:  Self = <$f>::MAX;

    #[inline(always)] fn min_element (value: Self) -> Self::Element { value }
    #[inline(always)] fn max_element (value: Self) -> Self::Element { value }
    #[inline(always)] fn sum_elements(value: Self) -> Self::Element { value }
    #[inline(always)] fn prod_elements(value: Self) -> Self::Element { value }
    #[inline(always)] fn offset()  -> Self { 1.0 }
    #[inline(always)] fn indexed() -> Self { 0.0 }
    #[inline(always)] fn add(lhs: Self, rhs: Self) -> Self { lhs + rhs }
    #[inline(always)] fn sub(lhs: Self, rhs: Self) -> Self { lhs - rhs }
    #[inline(always)] fn mul(lhs: Self, rhs: Self) -> Self { lhs * rhs }
    #[inline(always)] fn div(lhs: Self, rhs: Self) -> Self { lhs / rhs }
    #[inline(always)] fn rem(lhs: Self, rhs: Self) -> Self { lhs % rhs }
    #[inline(always)] fn sort(value: Self) -> Self { value }
    // Use GLSLstd450 FMin/FMax for correct NaN semantics on GPU
    #[inline(always)] fn min(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::F_MIN}, false>(lhs, rhs) } }
    #[inline(always)] fn max(lhs: Self, rhs: Self) -> Self { unsafe { arch::glsl_op2::<Self, Self, Self, {glsl::F_MAX}, false>(lhs, rhs) } }
}

impl SignedRegister for [<f $width>] {
    const NEG_ONE:      Self = -1.0;
    const MIN_POSITIVE: Self = <$f>::MIN_POSITIVE;

    // OpFNegate is the canonical SPIR-V float negation.
    #[inline(always)] fn neg(value: Self) -> Self { unsafe { arch::op_opfnegate::<Self>(value) } }

    // GLSLstd450 FAbs: cheaper than the default abs+blend approach on GPU.
    #[inline(always)] fn abs(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::F_ABS }, false>(value) }
    }

    #[inline(always)] fn signum(value: Self) -> Self { value.signum() }

    // Bitwise copysign: (|lhs| bits) | (sign bit of rhs). f32::to_bits / from_bits
    // compile to OpBitcast on SPIRV — zero-cost type reinterpretation.
    #[inline(always)] fn copysign(lhs: Self, rhs: Self) -> Self { lhs.copysign(rhs) }

    #[inline(always)] fn neg_c(mask: bool, value: Self) -> Self {
        unsafe { arch::op_opselect::<Self, bool>(mask, Self::neg(value), value) }
    }

    // OpSignBitSet is the direct MSB read — cheaper than comparing to zero.
    // Fall back to OpFOrdLessThan in non-kernel SPIRV where OpSignBitSet may not be valid.
    #[inline(always)] fn is_negative(value: Self) -> bool {
        if const { cfg!(target_feature = "Kernel") } {
            unsafe { arch::op_opsignbitset::<bool, Self>(value) }
        } else {
            unsafe { arch::op_opfordlessthan::<bool, Self>(value, 0.0) }
        }
    }
}

impl FloatRegister for [<f $width>] {
    type Bits             = [<u $width>];
    type SignedBits       = [<i $width>];
    type ExtendedPrecision = f64;

    // GPU hardware always has FMA via GLSLstd450.
    const HAS_TRUE_FMA: bool = true;

    const HALF:         Self = 0.5;
    const NEG_ZERO:     Self = -0.0;
    const INFINITY:     Self = <$f>::INFINITY;
    const NEG_INFINITY: Self = <$f>::NEG_INFINITY;
    const NAN:          Self = <$f>::NAN;
    const EPSILON:      Self = <$f>::EPSILON;
    const EXP_MASK:     Storage<Self::Bits> = <$f>::INFINITY.to_bits();

    // GLSLstd450 InverseSqrt is a native GPU instruction.
    const HAS_APPROX_RSQRT: bool = true;
    const HAS_APPROX_RCP:   bool = false;

    // GLSLstd450 provides all transcendentals natively on GPU.
    const NATIVE_CAP: NativeCapability = NativeCapability(
        NativeCapability::LDEXP
        | NativeCapability::FREXP
        | NativeCapability::SIN
        | NativeCapability::COS
        | NativeCapability::TAN
        | NativeCapability::EXP2
        | NativeCapability::LOG2
        | NativeCapability::EXP
        | NativeCapability::LN
        | NativeCapability::POWF,
    );

    // True FMA via GLSLstd450.
    #[inline(always)] fn mul_add (lhs: Self, rhs: Self, acc: Self) -> Self {
        unsafe { arch::glsl_op3::<Self, Self, Self, Self, { glsl::FMA }, false>(lhs, rhs, acc) }
    }
    // -(lhs * rhs) + acc = FMA(-lhs, rhs, acc)
    #[inline(always)] fn nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self {
        unsafe { arch::glsl_op3::<Self, Self, Self, Self, { glsl::FMA }, false>(Self::neg(lhs), rhs, acc) }
    }
    // lhs * rhs - acc = FMA(lhs, rhs, -acc)
    #[inline(always)] fn mul_sub (lhs: Self, rhs: Self, acc: Self) -> Self {
        unsafe { arch::glsl_op3::<Self, Self, Self, Self, { glsl::FMA }, false>(lhs, rhs, Self::neg(acc)) }
    }
    // -(lhs * rhs) - acc = FMA(-lhs, rhs, -acc)
    #[inline(always)] fn nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self {
        unsafe { arch::glsl_op3::<Self, Self, Self, Self, { glsl::FMA }, false>(Self::neg(lhs), rhs, Self::neg(acc)) }
    }

    #[inline(always)] fn sqrt (value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::SQRT }, false>(value) }
    }
    // GLSLstd450 InverseSqrt: 1/sqrt(x), native GPU instruction.
    #[inline(always)] fn rsqrt(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::INVERSE_SQRT }, false>(value) }
    }
    #[inline(always)] fn floor(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::FLOOR }, false>(value) }
    }
    #[inline(always)] fn ceil (value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::CEIL }, false>(value) }
    }
    // GLSLstd450 Round: tie-breaking is implementation-defined per GPU vendor.
    #[inline(always)] fn round(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::ROUND }, false>(value) }
    }
    #[inline(always)] fn trunc(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::TRUNC }, false>(value) }
    }
    // GLSLstd450 Fract: native; avoids the sub(v, trunc(v)) default chain.
    #[inline(always)] fn fract(value: Self) -> Self {
        unsafe { arch::glsl_op1::<Self, Self, { glsl::FRACT }, false>(value) }
    }

    #[inline(always)] fn next_up  (value: Self) -> Self { FloatElement::next_up  (value) }
    #[inline(always)] fn next_down(value: Self) -> Self { FloatElement::next_down(value) }

    // GLSLstd450 FMix: a + (b - a) * t, with hardware guarantees.
    #[inline(always)] fn mix(a: Self, b: Self, t: Self) -> Self {
        unsafe { arch::glsl_op3::<Self, Self, Self, Self, { glsl::F_MIX }, false>(a, b, t) }
    }

    // Native SPIR-V classification ops; cheaper than the bitwise-trick defaults on GPU.
    #[inline(always)] fn is_nan     (value: Self) -> bool { unsafe { arch::op_opisnan    ::<bool, Self>(value) } }
    #[inline(always)] fn is_infinite(value: Self) -> bool { unsafe { arch::op_opisinf    ::<bool, Self>(value) } }
    #[inline(always)] fn is_finite  (value: Self) -> bool { unsafe { arch::op_opisfinite ::<bool, Self>(value) } }
    #[inline(always)] fn is_normal  (value: Self) -> bool { unsafe { arch::op_opisnormal ::<bool, Self>(value) } }

    // GLSLstd450 Ldexp: x * 2^exp, with int exponent.
    #[inline(always)] unsafe fn native_ldexp(value: Self, exp: [<i $width>]) -> Self {
        unsafe { arch::glsl_op2::<Self, Self, [<i $width>], { glsl::LDEXP }, false>(value, exp) }
    }

    // GLSLstd450 FrexpStruct: splits x into (significand in [0.5, 1), exponent).
    #[inline(always)] unsafe fn native_frexp(value: Self) -> (Self, [<i $width>]) {
        unsafe { arch::glsl_frexp::<Self, [<i $width>]>(value) }
    }

    #[inline(always)] unsafe fn native_sin<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::SIN }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::SIN }, false>(value) }
        }
    }

    #[inline(always)] unsafe fn native_cos<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::COS }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::COS }, false>(value) }
        }
    }

    // Emit both with a single call each rather than sharing a combined instruction;
    // the GPU scheduler can still issue them in parallel.
    #[inline(always)] unsafe fn native_sin_cos<P: Policy>(value: Self) -> (Self, Self) {
        unsafe { (Self::native_sin::<P>(value), Self::native_cos::<P>(value)) }
    }

    #[inline(always)] unsafe fn native_tan<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::TAN }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::TAN }, false>(value) }
        }
    }

    #[inline(always)] unsafe fn native_exp2<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::EXP2 }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::EXP2 }, false>(value) }
        }
    }

    #[inline(always)] unsafe fn native_log2<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::LOG2 }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::LOG2 }, false>(value) }
        }
    }

    #[inline(always)] unsafe fn native_exp<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::EXP }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::EXP }, false>(value) }
        }
    }

    // GLSLstd450 Log is the natural logarithm (ln).
    #[inline(always)] unsafe fn native_ln<P: Policy>(value: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::LOG }, true>(value) }
        } else {
            unsafe { arch::glsl_op1::<Self, Self, { glsl::LOG }, false>(value) }
        }
    }

    #[inline(always)] unsafe fn native_powf<P: Policy>(base: Self, exp: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            unsafe { arch::glsl_op2::<Self, Self, Self, { glsl::POW }, true>(base, exp) }
        } else {
            unsafe { arch::glsl_op2::<Self, Self, Self, { glsl::POW }, false>(base, exp) }
        }
    }

    // No register-file constraints in SPIRV — this is a no-op.
    #[inline(always)] unsafe fn block_autovectorization(_value: &mut Self) {}
}

}}} // end macro

decl_spirv_float_scalar!(f32 => 32);
decl_spirv_float_scalar!(f64 => 64);
