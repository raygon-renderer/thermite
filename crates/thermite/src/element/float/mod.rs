use super::{SignedElement, SignedIntegerElement, UnsignedIntegerElement};
use crate::LargeInt;
use crate::vector::SplatConst;
use crate::vector::ops::MulAddExt;

pub mod spec;

pub(crate) mod algebraic;
pub(crate) mod arch;

/// Marker type for a compile-time integer constant cast to a float element type.
///
/// Implements [`SplatConst<f32>`] and [`SplatConst<f64>`], so it works with
/// [`const_splat!`](crate::const_splat) and [`FloatElement::ConstInt`].
pub struct IntConst<const N: crate::LargeInt>;

/// Marker type for a compile-time rational constant (N/D) cast to a float element type.
///
/// Implements [`SplatConst<f32>`] and [`SplatConst<f64>`], so it works with
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
    /// Satisfies `SplatConst<Self>`, so const-folded splats go through
    /// [`const_splat!`](crate::const_splat).
    type ConstInt<const N: crate::LargeInt>: SplatConst<Self>;

    /// Marker type for splatting a compile-time rational constant (N/D) as this float type.
    ///
    /// Satisfies `SplatConst<Self>`, so const-folded splats go through
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
    /// Round to nearest, **ties to even**.
    ///
    /// This deliberately differs from `f32::round`/`f64::round`, which break ties away
    /// from zero. Ties-to-even is what every SIMD backend's nearest-integer instruction
    /// does (`roundps`/`roundpd` with `_MM_FROUND_TO_NEAREST_INT`, NEON `frintn`, WASM
    /// `nearest`), so matching it here keeps a one-lane `Vector<f64>` in agreement with
    /// a `f64x4` and keeps this a single instruction on every target. Synthesizing
    /// ties-away would cost five instructions and disagree with the vector layer.
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

    /// Veltkamp's splitting constant, `$2^{\lceil p/2 \rceil} + 1$` for a format with `p`
    /// bits of precision (so `MANTISSA_BITS + 1`): `$2^{12} + 1$` for binary32,
    /// `$2^{27} + 1$` for binary64.
    ///
    /// Splits `a` into two halves of about `p/2` bits whose products are exact, which is
    /// what makes Dekker's `two_prod` possible without an FMA. Lives here rather than in
    /// `thermite-compensated` because core's emulated FMA and `compensated_horner` need it.
    ///
    /// `a * VELTKAMP_SPLITTER` overflows above roughly `MAX / VELTKAMP_SPLITTER`, so a
    /// general `two_prod` scales large operands down first; see
    /// [`VELTKAMP_SPLIT_THRESH`](Self::VELTKAMP_SPLIT_THRESH).
    const VELTKAMP_SPLITTER: Self;

    /// `|a|` above this overflows `a * VELTKAMP_SPLITTER`, so a split must scale it down
    /// first.
    ///
    /// The real limit is `MAX / VELTKAMP_SPLITTER` (8.3056e34 for binary32, 1.3394e300 for
    /// binary64); this sits at or below it: `$2^{115}$` for binary32, and the largest
    /// double under `$2^{996}$` for binary64. Not `$2^{116}$` for binary32, which is
    /// 8.3077e34 and lands just above the limit. Pinned by the const assertions below.
    const VELTKAMP_SPLIT_THRESH: Self;

    /// Exact power of two to scale an operand past
    /// [`VELTKAMP_SPLIT_THRESH`](Self::VELTKAMP_SPLIT_THRESH) down by before splitting.
    ///
    /// The factor moves onto the other operand, leaving the product unchanged. That cannot
    /// overflow: if `|a| > THRESH` and `a * b` is finite then `|b| < MAX / THRESH`, which
    /// is this factor. Scaling the split's output back up does not work, since for `a`
    /// near `MAX` the split rounds `hi` past `MAX / scale`.
    const VELTKAMP_SPLIT_DOWN: Self;

    /// Exact reciprocal of [`VELTKAMP_SPLIT_DOWN`](Self::VELTKAMP_SPLIT_DOWN), also an
    /// exact power of two. Applied to the operand that did *not* need scaling down.
    const VELTKAMP_SPLIT_UP: Self;

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
        const HAS_SUBNORMALS: bool = cfg!(not(feature = "ignore_denormals"));
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
            if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
                impl FloatElement for $t {
                    #[inline(always)] fn sqrt(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::SQRT}, false>(value) } }
                    #[inline(always)] fn floor(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::FLOOR}, false>(value) } }
                    #[inline(always)] fn ceil(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::CEIL}, false>(value) } }
                    #[inline(always)] fn round(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::ROUND_EVEN}, false>(value) } }
                    #[inline(always)] fn trunc(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::TRUNC}, false>(value) } }
                    #[inline(always)] fn fract(value: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op1::<Self, Self, {crate::backend::spirv::arch::glsl::FRACT}, false>(value) } }
                    #[inline(always)] fn next_up(value: Self) -> Self { value.next_up() }
                    #[inline(always)] fn next_down(value: Self) -> Self { value.next_down() }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    // GPU hardware always has FMA
                    const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, rhs, acc) } }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, rhs, -acc) } }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, -rhs, acc) } }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { unsafe { crate::backend::spirv::arch::glsl_op3::<Self, Self, Self, Self, {crate::backend::spirv::arch::glsl::FMA}, false>(self, -rhs, -acc) } }

                    #[inline(always)] fn mul_adde(self, rhs: Self, acc: Self) -> Self { self.mul_add(rhs, acc) }
                    #[inline(always)] fn mul_sube(self, rhs: Self, acc: Self) -> Self { self.mul_sub(rhs, acc) }
                    #[inline(always)] fn nmul_adde(self, rhs: Self, acc: Self) -> Self { self.nmul_add(rhs, acc) }
                    #[inline(always)] fn nmul_sube(self, rhs: Self, acc: Self) -> Self { self.nmul_sub(rhs, acc) }
                }
            } else {
                // `arch` walks its own ladder: `core::intrinsics` under `nightly`, then
                // the explicit per-ISA intrinsics the baseline proves are available, then
                // std's methods (or `libm` without std). See its module docs.
                impl FloatElement for $t {
                    #[inline(always)] fn sqrt(value: Self) -> Self { arch::[<sqrt $($f)?>](value) }
                    #[inline(always)] fn floor(value: Self) -> Self { arch::[<floor $($f)?>](value) }
                    #[inline(always)] fn ceil(value: Self) -> Self { arch::[<ceil $($f)?>](value) }
                    #[inline(always)] fn round(value: Self) -> Self { arch::[<round $($f)?>](value) }
                    #[inline(always)] fn trunc(value: Self) -> Self { arch::[<trunc $($f)?>](value) }
                    #[inline(always)] fn next_up(value: Self) -> Self { value.next_up() }
                    #[inline(always)] fn next_down(value: Self) -> Self { value.next_down() }

                    impl_float_element!(COMMON);
                }

                impl MulAddExt for $t {
                    type Output = Self;

                    const HAS_NATIVE_FMA: tribool::Tribool = arch::HAS_NATIVE_FMA;

                    #[inline(always)] fn mul_add(self, rhs: Self, acc: Self) -> Self { arch::[<fma $($f)?>](self, rhs, acc) }
                    #[inline(always)] fn mul_sub(self, rhs: Self, acc: Self) -> Self { arch::[<fma $($f)?>](self, rhs, -acc) }
                    #[inline(always)] fn nmul_add(self, rhs: Self, acc: Self) -> Self { arch::[<fma $($f)?>](self, -rhs, acc) }
                    #[inline(always)] fn nmul_sub(self, rhs: Self, acc: Self) -> Self { arch::[<fma $($f)?>](self, -rhs, -acc) }

                    // Only worth the FMA when it is a single instruction; without one the
                    // estimating forms must stay as separate multiply and add.
                    #[inline(always)] fn mul_adde(self, rhs: Self, acc: Self) -> Self { if !matches!(<Self as MulAddExt>::HAS_NATIVE_FMA, tribool::True) { self * rhs + acc } else { <Self as MulAddExt>::mul_add(self, rhs, acc) } }
                    #[inline(always)] fn mul_sube(self, rhs: Self, acc: Self) -> Self { if !matches!(<Self as MulAddExt>::HAS_NATIVE_FMA, tribool::True) { self * rhs - acc } else { <Self as MulAddExt>::mul_sub(self, rhs, acc) } }
                    #[inline(always)] fn nmul_adde(self, rhs: Self, acc: Self) -> Self { if !matches!(<Self as MulAddExt>::HAS_NATIVE_FMA, tribool::True) { acc - self * rhs } else { <Self as MulAddExt>::nmul_add(self, rhs, acc) } }
                    #[inline(always)] fn nmul_sube(self, rhs: Self, acc: Self) -> Self { if !matches!(<Self as MulAddExt>::HAS_NATIVE_FMA, tribool::True) { self * -rhs - acc } else { <Self as MulAddExt>::nmul_sub(self, rhs, acc) } }
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

    // Veltkamp: 2^ceil(24/2) + 1. Written out rather than derived from MANTISSA_BITS,
    // which is deliberately zeroed for f64 on a SPIR-V target without Float64 and so is
    // not a safe base for arithmetic in a shared macro.
    VELTKAMP_SPLITTER: f32 = 4097.0;

    // 2^115, NOT 2^116 - see the doc comment. The obvious power lands above the limit.
    VELTKAMP_SPLIT_THRESH: f32 = 4.153_837_5e34;
    VELTKAMP_SPLIT_DOWN: f32 = 1.220_703_1e-4; // 2^-13
    VELTKAMP_SPLIT_UP: f32 = 8192.0; // 2^13

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

    // Veltkamp: 2^ceil(53/2) + 1. See the f32 entry for why this is a literal.
    VELTKAMP_SPLITTER: f64 = 134217729.0;

    // Largest f64 strictly below 2^996.
    VELTKAMP_SPLIT_THRESH: f64 = 6.69692879491417e299;
    VELTKAMP_SPLIT_DOWN: f64 = 3.725_290_298_461_914e-9; // 2^-28
    VELTKAMP_SPLIT_UP: f64 = 268435456.0; // 2^28

    // IMPLICIT_LEAD_BIT: bool = true;
});

// Checked at compile time because a wrong value produces plausible numbers, not a failure.
// The thresholds need not be powers of two (the f64 value is Bailey's QD literal); they
// only need to keep the split from overflowing. An earlier f32 threshold of 2^116 failed
// exactly this check: `2^116 * 4097` exceeds `f32::MAX`.
const _: () = {
    // 1. The scale factors are exact reciprocals, so rebalancing preserves the product.
    assert!(f32::VELTKAMP_SPLIT_DOWN * f32::VELTKAMP_SPLIT_UP == 1.0);
    assert!(f64::VELTKAMP_SPLIT_DOWN * f64::VELTKAMP_SPLIT_UP == 1.0);

    // 2. Both are powers of two, so scaling is lossless. A power of two has an all-zero
    //    significand field.
    assert!(f32::VELTKAMP_SPLIT_DOWN.to_bits() & ((1 << 23) - 1) == 0);
    assert!(f64::VELTKAMP_SPLIT_DOWN.to_bits() & ((1 << 52) - 1) == 0);
    assert!(f32::VELTKAMP_SPLIT_UP.to_bits() & ((1 << 23) - 1) == 0);
    assert!(f64::VELTKAMP_SPLIT_UP.to_bits() & ((1 << 52) - 1) == 0);

    // 3. Splitting is safe on both sides of the branch: at the threshold unscaled, and at
    //    the largest finite value once scaled down.
    assert!(f32::VELTKAMP_SPLIT_THRESH * f32::VELTKAMP_SPLITTER < f32::MAX);
    assert!(f64::VELTKAMP_SPLIT_THRESH * f64::VELTKAMP_SPLITTER < f64::MAX);
    assert!(f32::MAX * f32::VELTKAMP_SPLIT_DOWN * f32::VELTKAMP_SPLITTER < f32::MAX);
    assert!(f64::MAX * f64::VELTKAMP_SPLIT_DOWN * f64::VELTKAMP_SPLITTER < f64::MAX);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    NearestTiesToEven,
    Truncate,

    #[cfg(feature = "rand")]
    Stochastic(u64),
}
