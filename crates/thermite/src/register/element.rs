/// Common trait for types that can be used as elements in SIMD registers.
pub trait Element:
    Sized + Copy + Default + PartialEq + PartialOrd + core::fmt::Debug + 'static + num_traits::NumOps
{
    /// Unsigned integer type to be used with operations that require unsigned counts, such as shifts.
    type USize: Element;
    /// Signed integer type to be used with operations that require signed counts, such as shifts.
    type ISize: Element;

    /// When used as a mask, represents "true"
    const TRUTHY: Self;
    /// When used as a mask, represents "false"
    const FALSY: Self;

    const ZERO: Self;

    /// Convert the element, as a mask, to a boolean value.
    fn to_bool(self) -> bool;

    /// Create the element, as a mask, from a boolean value.
    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    fn from_i8(value: i8) -> Self;
    fn from_u16(value: u16) -> Self;
}

macro_rules! impl_element {
    ($(($t:ty, $u:ty, $s:ty)),+) => {$(
        impl Element for $t {
            type USize = $u;
            type ISize = $s;

            const TRUTHY: Self = !0;
            const FALSY: Self = 0;
            const ZERO: Self = 0;

            #[inline(always)] fn to_bool(self) -> bool { self != 0 }
            #[inline(always)] fn from_i8(value: i8) -> Self { value as $t }
            #[inline(always)] fn from_u16(value: u16) -> Self { value as $t }
        }
    )+};

    (F $f:ty, $u:ty, $s:ty) => {
        impl Element for $f {
            type USize = $u;
            type ISize = $s;

            const TRUTHY: Self = <$f>::from_bits(!0);
            const FALSY: Self = <$f>::from_bits(0);
            const ZERO: Self = 0.0;

            #[inline(always)] fn to_bool(self) -> bool { self.to_bits() != 0 }
            #[inline(always)] fn from_i8(value: i8) -> Self { value as $f }
            #[inline(always)] fn from_u16(value: u16) -> Self { value as $f }
        }
    }
}

impl_element! {
    //(u8, u8, i8),
    (u16, u16, i16),
    (u32, u32, i32),
    (u64, u64, i64),
    //(i8, u8, i8),
    (i16, u16, i16),
    (i32, u32, i32),
    (i64, u64, i64)
    //(f32, u32, i32),
    //(f64, u64, i64)
}

impl_element!(F f32, u32, i32);
impl_element!(F f64, u64, i64);

/// A trait for float element types that can be used in SIMD operations.
///
/// Notably, this trait provides scalar fallback methods for true fused multiply-add (FMA) operations,
/// when they aren't available in the target architecture. Sometimes it's essential to have these
/// fallbacks for correctness, given FMAs rounding behavior.
pub trait FloatElement: Element + num_traits::float::FloatCore + From<i8> + core::fmt::Display
// + crate::math::FloatConsts
{
    type Bits: Element;
    type Signed: Element;

    // maximum u32 that can be exactly represented in this float type without loss of precision
    const MAX_U64: u64;
    const MANTISSA: u32;
    const EXP_BIAS: Self::Signed;
    const MAX_BIASED_EXP: Self::Signed;
    const EXP_LSB_MASK: Self::Bits;
    const SIGN_MANTISSA_MASK: Self::Bits;

    const HALF_EXP_BITS: Self::Bits;
    const FREXP_BIAS_OFFSET: Self::Signed;

    fn from_f64(value: f64) -> Self;
    fn from_i64(value: i64) -> Self;
    fn from_signed(value: Self::Signed) -> Self;

    fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self;

    fn sqrt(value: Self) -> Self;
    fn floor(value: Self) -> Self;
    fn ceil(value: Self) -> Self;
    fn round(value: Self) -> Self;
    fn trunc(value: Self) -> Self;

    #[inline(always)]
    fn fract(value: Self) -> Self {
        value - value.trunc() // fallback implementation
    }

    fn next_up(value: Self) -> Self;
    fn next_down(value: Self) -> Self;
}

macro_rules! impl_float_element {
    (CONSTS $($const:ident: $const_ty:ty = $value:expr;)+) => {paste::paste! {
        $(const $const: $const_ty = $value;)+

        const FREXP_BIAS_OFFSET: Self::Signed = Self::EXP_BIAS - 1;
        const HALF_EXP_BITS: Self::Bits = (Self::FREXP_BIAS_OFFSET << Self::MANTISSA) as _;
    }};

    ($t:ty $(: $f:ident)? => $bits:ty, $signed:ty { $($const:ident: $const_ty:ty = $value:expr;)* }) => {paste::paste! {
        #[cfg(feature = "std")]
        impl FloatElement for $t {
            type Bits = $bits;
            type Signed = $signed;

            impl_float_element!(CONSTS $($const: $const_ty = $value;)*);

            #[inline(always)]
            fn from_i64(value: i64) -> Self {
                if value.unsigned_abs() < Self::MAX_U64 {
                    value as $t // safe to convert directly
                } else {
                    panic!("Value exceeds maximum exact representable i64 in this float type");
                }
            }

            #[inline(always)] fn from_f64(value: f64) -> Self { value as $t }
            #[inline(always)] fn from_signed(value: Self::Signed) -> Self { value as $t }

            #[inline(always)] fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self { lhs.mul_add(rhs, acc) }
            #[inline(always)] fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { lhs.mul_add(rhs, -acc) }
            #[inline(always)] fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self { lhs.mul_add(-rhs, acc) }
            #[inline(always)] fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { lhs.mul_add(-rhs, -acc) }

            #[inline(always)] fn sqrt(value: Self) -> Self { value.sqrt() }
            #[inline(always)] fn floor(value: Self) -> Self { value.floor() }
            #[inline(always)] fn ceil(value: Self) -> Self { value.ceil() }
            #[inline(always)] fn round(value: Self) -> Self { value.round() }
            #[inline(always)] fn trunc(value: Self) -> Self { value.trunc() }
            #[inline(always)] fn fract(value: Self) -> Self { value.fract() }
            #[inline(always)] fn next_up(value: Self) -> Self { value.next_up() }
            #[inline(always)] fn next_down(value: Self) -> Self { value.next_down() }
        }

        #[cfg(not(feature = "std"))]
        impl FloatElement for $t {
            type Bits = $bits;
            type Signed = $signed;

            impl_float_element!(CONSTS $($const: $const_ty = $value;)*);

            const MAX_U64: u64 = (1u64 << (Self::MANTISSA + 1));

            #[inline(always)]
            fn from_i64(value: i64) -> Self {
                if value.unsigned_abs() < Self::MAX_U64 {
                    value as $t // safe to convert directly
                } else {
                    panic!("Value exceeds maximum exact representable i64 in this float type");
                }
            }

            #[inline(always)] fn from_f64(value: f64) -> Self { value as $t }
            #[inline(always)] fn from_signed(value: Self::Signed) -> Self { value as $t }

            #[inline(always)] fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](lhs, rhs, acc) }
            #[inline(always)] fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](lhs, rhs, -acc) }
            #[inline(always)] fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](lhs, -rhs, acc) }
            #[inline(always)] fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::[<fma $($f)?>](lhs, -rhs, -acc) }

            #[inline(always)] fn sqrt(value: Self) -> Self { libm::[<sqrt $($f)?>](value) }
            #[inline(always)] fn floor(value: Self) -> Self { libm::[<floor $($f)?>](value) }
            #[inline(always)] fn ceil(value: Self) -> Self { libm::[<ceil $($f)?>](value) }
            #[inline(always)] fn round(value: Self) -> Self { libm::[<round $($f)?>](value) }
            #[inline(always)] fn trunc(value: Self) -> Self { libm::[<trunc $($f)?>](value) }
            #[inline(always)] fn next_up(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::INFINITY) }
            #[inline(always)] fn next_down(value: Self) -> Self { libm::[<nextafter $($f)?>](value, Self::NEG_INFINITY) }
        }
    }};
}

impl_float_element!(f32: f => u32, i32 {
    MANTISSA: u32 = 23;
    EXP_BIAS: i32 = 127;
    MAX_BIASED_EXP: i32 = 255;

    // 8 bits of exponent
    EXP_LSB_MASK: u32 = 0xFF;

    // Clear bits 23-30
    SIGN_MANTISSA_MASK: u32 = 0x807F_FFFF;
});

impl_float_element!(f64 => u64, i64 {
    MANTISSA: u32 = 52;
    EXP_BIAS: i64 = 1023;
    MAX_BIASED_EXP: i64 = 2047;

    // 11 bits of exponent
    EXP_LSB_MASK: u64 = 0x7FF;

    // Clear bits 52-62
    SIGN_MANTISSA_MASK: u64 = 0x800F_FFFF_FFFF_FFFF;
});
