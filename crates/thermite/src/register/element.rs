/// Common trait for types that can be used as elements in SIMD registers.
pub trait Element: Sized + Copy + Default + PartialEq + PartialOrd + core::fmt::Debug + 'static {
    /// Unsigned integer type to be used with operations that require unsigned counts, such as shifts.
    type UCOUNT: Element;
    /// Signed integer type to be used with operations that require signed counts, such as shifts.
    type SCOUNT: Element;

    /// When used as a mask, represents "true"
    const TRUTHY: Self;
    /// When used as a mask, represents "false"
    const FALSY: Self;

    /// Convert the element, as a mask, to a boolean value.
    fn to_bool(self) -> bool;

    /// Create the element, as a mask, from a boolean value.
    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    fn from_i8(value: i8) -> Self;
}

macro_rules! impl_element {
    ($(($t:ty, $u:ty, $s:ty)),+) => {$(
        impl Element for $t {
            type UCOUNT = $u;
            type SCOUNT = $s;

            const TRUTHY: Self = !0;
            const FALSY: Self = 0;

            #[inline(always)]
            fn to_bool(self) -> bool { self != 0 }

            #[inline(always)]
            fn from_i8(value: i8) -> Self { value as $t }
        }
    )+};
}

impl_element! {
    (u8, u8, i8),
    (u16, u16, i16),
    (u32, u32, i32),
    (u64, u64, i64),
    (i8, u8, i8),
    (i16, u16, i16),
    (i32, u32, i32),
    (i64, u64, i64)
    //(f32, u32, i32),
    //(f64, u64, i64)
}

impl Element for f32 {
    type UCOUNT = u32;
    type SCOUNT = i32;

    const TRUTHY: Self = f32::from_bits(!0);
    const FALSY: Self = f32::from_bits(0);

    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn from_i8(value: i8) -> Self {
        value as f32
    }
}

impl Element for f64 {
    type UCOUNT = u64;
    type SCOUNT = i64;

    const TRUTHY: Self = f64::from_bits(!0);
    const FALSY: Self = f64::from_bits(0);

    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn from_i8(value: i8) -> Self {
        value as f64
    }
}

/// A trait for float element types that can be used in SIMD operations.
///
/// Notably, this trait provides scalar fallback methods for true fused multiply-add (FMA) operations,
/// when they aren't available in the target architecture. Sometimes it's essential to have these
/// fallbacks for correctness, given FMAs rounding behavior.
pub trait FloatElement:
    Element + num_traits::float::FloatCore + From<i8> + core::fmt::Display + crate::math::FloatConsts
{
    type Bits: Element;
    type Signed: Element;

    // maximum u32 that can be exactly represented in this float type without loss of precision
    const MAX_U64: u64;

    fn from_f64(value: f64) -> Self;
    fn from_i64(value: i64) -> Self;

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
    ($t:ty $(: $f:ident)? => $bits:ty, $signed:ty, $max_u64:expr) => {paste::paste! {
        #[cfg(feature = "std")]
        impl FloatElement for $t {
            type Bits = $bits;
            type Signed = $signed;

            const MAX_U64: u64 = $max_u64;

            #[inline(always)]
            fn from_i64(value: i64) -> Self {
                if value.unsigned_abs() < Self::MAX_U64 {
                    value as $t // safe to convert directly
                } else {
                    panic!("Value exceeds maximum exact representable i64 in this float type");
                }
            }

            #[inline(always)] fn from_f64(value: f64) -> Self { value as $t }

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

            const MAX_U64: u64 = $max_u64;

            #[inline(always)]
            fn from_i64(value: i64) -> Self {
                if value.unsigned_abs() < Self::MAX_U64 {
                    value as $t // safe to convert directly
                } else {
                    panic!("Value exceeds maximum exact representable i64 in this float type");
                }
            }

            #[inline(always)] fn from_f64(value: f64) -> Self { value as $t }

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

impl_float_element!(f32: f => u32, i32, 1 << 23);
impl_float_element!(f64 => u64, i64, 1 << 53);
