use super::{SignedElement, SignedIntegerElement, UnsignedIntegerElement};
use crate::LargeInt;
use crate::register::FloatRegister;
use crate::vector::SplatConst;
use crate::vector::ops::MulAddExt;

pub mod spec;

/// Marker type for a compile-time integer constant cast to a float element type.
///
/// Implements [`SplatConst<f32>`] and [`SplatConst<f64>`], enabling use with
/// [`const_splat!`](crate::const_splat) and [`FloatElement::ConstInt`].
pub struct IntConst<const N: crate::LargeInt>;

/// Marker type for a compile-time rational constant (N/D) cast to a float element type.
///
/// Implements [`SplatConst<f32>`] and [`SplatConst<f64>`], enabling use with
/// [`const_splat!`](crate::const_splat) and [`FloatElement::ConstRatio`].
pub struct RatioConst<const N: crate::LargeInt, const D: crate::LargeInt>;

impl<const N: crate::LargeInt> SplatConst<f32> for IntConst<N> {
    const VALUE: f32 = N as f32;
}

impl<const N: crate::LargeInt> SplatConst<f64> for IntConst<N> {
    const VALUE: f64 = N as f64;
}

impl<const N: crate::LargeInt, const D: crate::LargeInt> SplatConst<f32> for RatioConst<N, D> {
    const VALUE: f32 = {
        assert!(D != 0, "RatioConst: denominator must not be zero");
        let (q, r) = (N / D, N % D);
        (q as f32) + (r as f32) / (D as f32)
    };
}

impl<const N: crate::LargeInt, const D: crate::LargeInt> SplatConst<f64> for RatioConst<N, D> {
    const VALUE: f64 = {
        assert!(D != 0, "RatioConst: denominator must not be zero");
        let (q, r) = (N / D, N % D);
        (q as f64) + (r as f64) / (D as f64)
    };
}

/// A trait for float element types that can be used in SIMD operations.
///
/// This provides common scalar fallbacks, as well as float specifications for
/// non-IEE 754 floating point formats
pub trait FloatElement:
    SignedElement
    + crate::math::FloatConsts
    + num_traits::NumOps
    + core::ops::Neg<Output = Self>
    + MulAddExt<Self, Self, Output = Self>
{
    /// Marker type for splatting a compile-time integer constant as this float type.
    ///
    /// Satisfies `SplatConst<Self>`, enabling const-folded splats via
    /// [`const_splat!`](crate::const_splat).
    type ConstInt<const N: crate::LargeInt>: SplatConst<Self>;

    /// Marker type for splatting a compile-time rational constant (N/D) as this float type.
    ///
    /// Satisfies `SplatConst<Self>`, enabling const-folded splats via
    /// [`const_splat!`](crate::const_splat).
    type ConstRatio<const N: crate::LargeInt, const D: crate::LargeInt>: SplatConst<Self>;

    /// Try to represent this LargeInt value as this float type,
    /// returning None if it cannot be represented exactly.
    fn try_from_int(value: LargeInt) -> Option<Self>;
    fn try_from_ratio(n: LargeInt, d: LargeInt) -> Option<Self>;

    cfg_if::cfg_if! {
        if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
            #[inline(always)]
            fn from_int(value: LargeInt) -> Self {
                Self::try_from_int(value).unwrap_or(Self::ZERO)
            }

            #[inline(always)]
            fn from_ratio(n: LargeInt, d: LargeInt) -> Self {
                Self::try_from_ratio(n, d).unwrap_or(Self::ZERO)
            }
        } else {
            #[inline(always)]
            fn from_int(value: LargeInt) -> Self {
                #[cold]
                fn _panic_int_overflow() -> ! {
                    panic!("LargeInt value exceeds maximum exact representable value for this float type")
                }

                Self::try_from_int(value).unwrap_or_else(|| _panic_int_overflow())
            }

            #[inline(always)]
            fn from_ratio(n: LargeInt, d: LargeInt) -> Self {
                #[cold]
                fn _panic_ratio_overflow() -> ! {
                    panic!("LargeInt ratio exceeds maximum exact representable value for this float type")
                }

                Self::try_from_ratio(n, d).unwrap_or_else(|| _panic_ratio_overflow())
            }
        }
    }

    fn sqrt(value: Self) -> Self;
    fn floor(value: Self) -> Self;
    fn ceil(value: Self) -> Self;
    fn round(value: Self) -> Self;
    fn trunc(value: Self) -> Self;

    #[inline(always)]
    fn fract(value: Self) -> Self {
        value - FloatElement::trunc(value) // fallback implementation
    }

    fn next_up(value: Self) -> Self;
    fn next_down(value: Self) -> Self;

    /// Does the format support Infinity?
    /// If FALSE, overflow saturates to MAX_FINITE instead of INF.
    /// (e.g., E4M3 = false, E5M2 = true)
    const HAS_INFINITY: bool;

    /// Does the format distinguish between +0 and -0?
    /// (Usually true, but some integer-like quantizations might not)
    const HAS_SIGNED_ZERO: bool;

    /// Does the format support subnormal numbers?
    /// If FALSE, any value smaller than MinNormal is flushed to zero (FTZ).
    const HAS_SUBNORMALS: bool;
}

pub trait FloatElementWithBits: FloatElement {
    type Bits: UnsignedIntegerElement<Unsigned = Self::Bits> + TryFrom<u32>;
    type SignedBits: SignedIntegerElement<Signed = Self::SignedBits> + TryFrom<u32>;

    const EXP_BITS: u32;
    const MANTISSA_BITS: u32;
    const EXP_BIAS: Self::SignedBits;

    /// The specific bit pattern for NaN.
    /// IEEE formats have a *range* of NaNs, but E4M3 has only *one* (0x7F).
    const NAN_PATTERN: Option<Self::Bits>;

    /// If !HAS_INFINITY, what is the max finite bit pattern?
    /// Used for clamping overflow.
    const MAX_FINITE_PATTERN: Self::Bits;

    /// Largest positive subnormal value
    const MAX_SUBNORMAL: Self::Bits;

    /// Magic value for crushing denormals
    const DENORMAL_TRICK: Self::Bits;

    // /// Is there an implicit leading bit (1.xxx)?
    // /// Almost always TRUE.
    // /// Exception: x87 80-bit float (FALSE).
    // const IMPLICIT_LEAD_BIT: bool = true;

    // maximum unsigned integer that can be exactly represented in this float type without loss of precision
    const MAX_LARGE_UINT: crate::LargeUInt;

    const MAX_BIASED_EXP: Self::SignedBits;
    const EXP_LSB_MASK: Self::Bits;
    const SIGN_MANTISSA_MASK: Self::Bits;

    const HALF_EXP_BITS: Self::Bits;
    const FREXP_BIAS_OFFSET: Self::SignedBits;

    /// Convert from f64 to this float type, potentially losing precision.
    fn from_f64(value: f64) -> Self;

    fn from_signed(value: Self::SignedBits) -> Self;
}

trait FloatElementInternal: FloatElement {
    fn try_from_int(value: crate::LargeInt) -> Option<Self>;
    fn try_from_ratio(n: crate::LargeInt, d: crate::LargeInt) -> Option<Self>;
}

macro_rules! impl_float_element {
    (CONSTS $($const:ident: $const_ty:ty = $value:expr;)+) => {paste::paste! {
        $(const $const: $const_ty = $value;)+

        const FREXP_BIAS_OFFSET: Self::SignedBits = Self::EXP_BIAS - 1;
        const HALF_EXP_BITS: Self::Bits = (Self::FREXP_BIAS_OFFSET << Self::MANTISSA_BITS) as _;
        const MAX_LARGE_UINT: crate::LargeUInt = ((1 as crate::LargeUInt) << (Self::MANTISSA_BITS + 1)) as _;
    }};

    (COMMON) => {
        #[inline(always)] fn try_from_int(value: crate::LargeInt) -> Option<Self> { FloatElementInternal::try_from_int(value) }
        #[inline(always)] fn try_from_ratio(n: crate::LargeInt, d: crate::LargeInt) -> Option<Self> { FloatElementInternal::try_from_ratio(n, d) }

        type ConstInt<const N: crate::LargeInt> = IntConst<N>;
        type ConstRatio<const N: crate::LargeInt, const D: crate::LargeInt> = RatioConst<N, D>;

        const HAS_INFINITY: bool = true;
        const HAS_SIGNED_ZERO: bool = true;
        const HAS_SUBNORMALS: bool = cfg!(not(feature = "ignore-denormals"));
    };

    (MUL_ADDE) => {
        #[inline(always)] fn mul_adde(self, rhs: Self, acc: Self) -> Self { self * rhs + acc }
        #[inline(always)] fn mul_sube(self, rhs: Self, acc: Self) -> Self { self * rhs - acc }
        #[inline(always)] fn nmul_adde(self, rhs: Self, acc: Self) -> Self { acc - self * rhs }
        #[inline(always)] fn nmul_sube(self, rhs: Self, acc: Self) -> Self { self * -rhs - acc }
    };

    ($t:ty $(: $f:ident)? => $bits:ty, $signed:ty { $($const:ident: $const_ty:ty = $value:expr;)* }) => {paste::paste! {
        impl FloatElementInternal for $t {
            #[inline(always)]
            fn try_from_int(value: crate::LargeInt) -> Option<Self> {
                if crate::likely(value.unsigned_abs() < Self::MAX_LARGE_UINT) {
                    Some(value as $t) // safe to convert directly
                } else {
                    None
                }
            }

            // This implementation is more accurate than simply doing n as f64 / d as f64,
            // since that can lose precision when n and d are large but their ratio is small.
            // Instead, we do integer division first, then add the fractional part. Although
            // this is non-trivial, LLVM should optimize it down to a constant value when
            // the inputs are known at compile time.
            #[inline(always)]
            fn try_from_ratio(n: crate::LargeInt, d: crate::LargeInt) -> Option<Self> {
                if d == 0 {
                    return None;
                }

                // fast path for values that both fit in the float exactly
                if let (Some(n), Some(d)) = (<Self as FloatElementInternal>::try_from_int(n), <Self as FloatElementInternal>::try_from_int(d)) {
                    return Some(n / d);
                }

                let (q, r) = (n / d, n % d);

                let mut result = FloatElementInternal::try_from_int(q)?;

                // d may not be exactly representable, but this will still scale it correctly
                result += (r as $t) / (d as $t);

                Some(result)
            }
        }

        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                impl FloatElement for $t {
                    #[inline(always)] fn sqrt(value: Self) -> Self { value.sqrt() }
                    #[inline(always)] fn floor(value: Self) -> Self { value.floor() }
                    #[inline(always)] fn ceil(value: Self) -> Self { value.ceil() }
                    #[inline(always)] fn round(value: Self) -> Self { value.round() }
                    #[inline(always)] fn trunc(value: Self) -> Self { value.trunc() }
                    #[inline(always)] fn fract(value: Self) -> Self { value.fract() }
                    #[inline(always)] fn next_up(value: Self) -> Self { value.next_up() }
                    #[inline(always)] fn next_down(value: Self) -> Self { value.next_down() }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    // trust the register implementation
                    const HAS_TRUE_FMA: bool = <$t as FloatRegister>::HAS_TRUE_FMA;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { <$t>::mul_add(self, rhs, acc) }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { <$t>::mul_add(self, rhs, -acc) }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { <$t>::mul_add(self, -rhs, acc) }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { <$t>::mul_add(self, -rhs, -acc) }

                    #[inline(always)] fn mul_adde(self, rhs: Self, acc: Self) -> Self { if !<Self as MulAddExt>::HAS_TRUE_FMA { self * rhs + acc } else { <$t>::mul_add(self, rhs, acc) } }
                    #[inline(always)] fn mul_sube(self, rhs: Self, acc: Self) -> Self { if !<Self as MulAddExt>::HAS_TRUE_FMA { self * rhs - acc } else { <$t>::mul_add(self, rhs, -acc) } }
                    #[inline(always)] fn nmul_adde(self, rhs: Self, acc: Self) -> Self { if !<Self as MulAddExt>::HAS_TRUE_FMA { acc - self * rhs } else { <$t>::mul_add(self, -rhs, acc) } }
                    #[inline(always)] fn nmul_sube(self, rhs: Self, acc: Self) -> Self { if !<Self as MulAddExt>::HAS_TRUE_FMA { self * -rhs - acc } else { <$t>::mul_add(self, -rhs, -acc) } }
                }
            } else if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
                impl FloatElement for $t {
                    #[inline(always)] fn sqrt(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::SQRT}, false>(value) } }
                    #[inline(always)] fn floor(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::FLOOR}, false>(value) } }
                    #[inline(always)] fn ceil(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::CEIL}, false>(value) } }
                    #[inline(always)] fn round(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::ROUND}, false>(value) } }
                    #[inline(always)] fn trunc(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::TRUNC}, false>(value) } }
                    #[inline(always)] fn fract(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::FRACT}, false>(value) } }
                    #[inline(always)] fn next_up(value: Self) -> Self { value.next_up() }
                    #[inline(always)] fn next_down(value: Self) -> Self { value.next_down() }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    // GPU hardware always has FMA
                    const HAS_TRUE_FMA: bool = true;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, rhs, acc) } }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, rhs, -acc) } }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, -rhs, acc) } }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, -rhs, -acc) } }

                    #[inline(always)] fn mul_adde(self, rhs: Self, acc: Self) -> Self { self.mul_add(rhs, acc) }
                    #[inline(always)] fn mul_sube(self, rhs: Self, acc: Self) -> Self { self.mul_sub(rhs, acc) }
                    #[inline(always)] fn nmul_adde(self, rhs: Self, acc: Self) -> Self { self.nmul_add(rhs, acc) }
                    #[inline(always)] fn nmul_sube(self, rhs: Self, acc: Self) -> Self { self.nmul_sub(rhs, acc) }
                }
            } else if #[cfg(all(feature = "nightly", feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")))] {
                impl FloatElement for $t {
                    // WASM has native scalar float ops for these
                    #[inline(always)] fn sqrt(value: Self) -> Self { crate::backend::wasm::arch::[<$t _sqrt>](value) }
                    #[inline(always)] fn floor(value: Self) -> Self { crate::backend::wasm::arch::[<$t _floor>](value) }
                    #[inline(always)] fn ceil(value: Self) -> Self { crate::backend::wasm::arch::[<$t _ceil>](value) }
                    #[inline(always)] fn trunc(value: Self) -> Self { crate::backend::wasm::arch::[<$t _trunc>](value) }
                    #[inline(always)] fn fract(value: Self) -> Self { value - crate::backend::wasm::arch::[<$t _trunc>](value) }
                    // WASM nearest() is banker's rounding (ties-to-even), not half-away-from-zero
                    #[inline(always)] fn round(value: Self) -> Self { libm::[<round $($f)?>](value) }
                    // No WASM scalar nextafter; fall back to libm
                    #[inline(always)] fn next_up(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::INFINITY) }
                    #[inline(always)] fn next_down(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::NEG_INFINITY) }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    // No hardware scalar FMA on WASM; use libm for exact, separate ops for estimating
                    const HAS_TRUE_FMA: bool = false;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, rhs, acc) }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, rhs, -acc) }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, -rhs, acc) }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, -rhs, -acc) }

                    impl_float_element!(MUL_ADDE);
                }
            } else {
                impl FloatElement for $t {
                    #[inline(always)] fn sqrt(value: Self) -> Self { libm::[<sqrt $($f)?>](value) }
                    #[inline(always)] fn floor(value: Self) -> Self { libm::[<floor $($f)?>](value) }
                    #[inline(always)] fn ceil(value: Self) -> Self { libm::[<ceil $($f)?>](value) }
                    #[inline(always)] fn round(value: Self) -> Self { libm::[<round $($f)?>](value) }
                    #[inline(always)] fn trunc(value: Self) -> Self { libm::[<trunc $($f)?>](value) }
                    #[inline(always)] fn next_up(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::INFINITY) }
                    #[inline(always)] fn next_down(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::NEG_INFINITY) }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    const HAS_TRUE_FMA: bool = false;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, rhs, acc) }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, rhs, -acc) }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, -rhs, acc) }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](self, -rhs, -acc) }

                    impl_float_element!(MUL_ADDE);
                }
            }
        }

        impl FloatElementWithBits for $t {
            type Bits = $bits;
            type SignedBits = $signed;

            impl_float_element!(CONSTS $($const: $const_ty = $value;)*);

            #[inline(always)] fn from_f64(value: f64) -> Self { value as $t }
            #[inline(always)] fn from_signed(value: Self::SignedBits) -> Self { value as $t }
        }
    }};
}

impl_float_element!(f32: f => u32, i32 {
    EXP_BITS: u32 = 8;
    MANTISSA_BITS: u32 = 23;
    EXP_BIAS: i32 = 127;
    MAX_BIASED_EXP: i32 = 255;

    // 8 bits of exponent
    EXP_LSB_MASK: u32 = 0xFF;

    // Clear bits 23-30
    SIGN_MANTISSA_MASK: u32 = 0x807F_FFFF;

    // Canonical Quiet NaN: Sign=0, Exp=All 1s, Mantissa=100...0
    // (Note: IEEE 754 allows many NaN patterns; this is just the standard "default")
    NAN_PATTERN: Option<u32> = None; //0x7FC0_0000, but we don't want to use it

    // Max Finite: Sign=0, Exp=254 (0xFE), Mantissa=All 1s
    MAX_FINITE_PATTERN: u32 = 0x7F7F_FFFF;

    MAX_SUBNORMAL: u32 = 0x007F_FFFF;
    DENORMAL_TRICK: u32 = 0x0C800001;

    // IMPLICIT_LEAD_BIT: bool = true;
});

impl_float_element!(f64 => u64, i64 {
    EXP_BITS: u32 = 11;
    MANTISSA_BITS: u32 = if cfg!(not(all(feature = "spirv", target_arch = "spirv", not(target_feature = "ext:Float64")))) { 52 } else { 0 };
    EXP_BIAS: i64 = 1023;
    MAX_BIASED_EXP: i64 = 2047;

    // 11 bits of exponent
    EXP_LSB_MASK: u64 = 0x7FF;

    // Clear bits 52-62
    SIGN_MANTISSA_MASK: u64 = 0x800F_FFFF_FFFF_FFFF;

    // Canonical Quiet NaN: Sign=0, Exp=All 1s, Mantissa=100...0
    NAN_PATTERN: Option<u64> = None; //0x7FF8_0000_0000_0000, but we don't want to use it

    // Max Finite: Sign=0, Exp=2046 (0x7FE), Mantissa=All 1s
    MAX_FINITE_PATTERN: u64 = 0x7FEF_FFFF_FFFF_FFFF;

    MAX_SUBNORMAL: u64 = 0x000F_FFFF_FFFF_FFFF;
    DENORMAL_TRICK: u64 = 0x0360000000000001;

    // IMPLICIT_LEAD_BIT: bool = true;
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    NearestTiesToEven,
    Truncate,

    #[cfg(feature = "rand")]
    Stochastic(u64),
}
