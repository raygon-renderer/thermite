pub trait FindUSize<U16, U32, U64> {
    type Output;
}

impl<U16, U32, U64> FindUSize<U16, U32, U64> for () {
    #[cfg(target_pointer_width = "16")]
    type Output = U16;

    #[cfg(target_pointer_width = "32")]
    type Output = U32;

    #[cfg(target_pointer_width = "64")]
    type Output = U64;
}

/// The unsigned integer type corresponding to the pointer width of the target architecture.
///
/// It will be `u16` on 16-bit targets, `u32` on 32-bit targets, and `u64` on 64-bit targets.
pub type USize = <() as FindUSize<u16, u32, u64>>::Output;

/// Common trait for types that can be used as elements in SIMD registers.
pub trait Element: 'static + Sized + Copy + Default + PartialEq + PartialOrd + core::fmt::Debug {
    /// Unsigned integer type to be used with operations that require unsigned counts, such as shifts.
    type Unsigned: UnsignedIntegerElement<Signed = Self::Signed>;
    /// SignedBits integer type to be used with operations that require signed counts, such as shifts.
    type Signed: SignedIntegerElement<Unsigned = Self::Unsigned>;

    const ZERO: Self;
    const ONE: Self;

    fn from_i8(value: i8) -> Self;
    fn from_u8(value: u8) -> Self;
    fn from_u16(value: u16) -> Self;
}

pub trait MaskElement: 'static + Sized + Copy + Default + PartialEq + core::fmt::Debug {
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
}

pub trait SignedElement: Element + core::ops::Neg<Output = Self> {
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}

macro_rules! impl_element {
    ($(($t:ty, $u:ty, $s:ty)),+) => {$(
        impl MaskElement for $t {
            const TRUTHY: Self = !0;
            const FALSY: Self = 0;

            #[inline(always)] fn to_bool(self) -> bool { self != 0 }
        }

        impl Element for $t {
            type Unsigned = $u;
            type Signed = $s;

            const ZERO: Self = 0;
            const ONE: Self = 1;

            #[inline(always)] fn from_i8(value: i8) -> Self { value as $t }
            #[inline(always)] fn from_u8(value: u8) -> Self { value as $t }
            #[inline(always)] fn from_u16(value: u16) -> Self { value as $t }
        }
    )+};

    (F $f:ty, $u:ty, $s:ty) => {
        impl MaskElement for $f {
            const TRUTHY: Self = <$f>::from_bits(!0);
            const FALSY: Self = <$f>::from_bits(0);

            #[inline(always)] fn to_bool(self) -> bool { self.to_bits() != 0 }
        }

        impl SignedElement for $f {
            #[inline(always)] fn abs(self) -> Self { <$f>::abs(self) }
            #[inline(always)] fn signum(self) -> Self { <$f>::signum(self) }
        }

        impl Element for $f {
            type Unsigned = $u;
            type Signed = $s;

            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline(always)] fn from_i8(value: i8) -> Self { value as $f }
            #[inline(always)] fn from_u8(value: u8) -> Self { value as $f }
            #[inline(always)] fn from_u16(value: u16) -> Self { value as $f }
        }
    }
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

impl_element!(F f32, u32, i32);
impl_element!(F f64, u64, i64);

/// A trait for integer element types that can be used in SIMD operations.
///
/// This trait is implemented for all primitive integer types that also implement `Element`, and
/// wrapping addition and multiplication.
pub trait IntegerElement:
    Element
    + crate::divider::Denominator
    + num_traits::PrimInt
    + num_traits::WrappingAdd
    + num_traits::WrappingMul
    + num_traits::WrappingSub
    + core::ops::Shr<Output = Self>
    + core::ops::Shl<Output = Self>
    + core::ops::Shr<Self::Unsigned, Output = Self>
    + core::ops::Shl<Self::Unsigned, Output = Self>
{
}

impl<T> IntegerElement for T where
    T: Element
        + crate::divider::Denominator
        + num_traits::PrimInt
        + num_traits::WrappingAdd
        + num_traits::WrappingMul
        + num_traits::WrappingSub
        + core::ops::Shr<Output = Self>
        + core::ops::Shl<Output = Self>
        + core::ops::Shr<Self::Unsigned, Output = Self>
        + core::ops::Shl<Self::Unsigned, Output = Self>
{
}

pub trait SignedIntegerElement: IntegerElement<Signed = Self> + num_traits::Signed + TryInto<isize> {}
pub trait UnsignedIntegerElement:
    IntegerElement<Unsigned = Self> + num_traits::Unsigned + TryInto<usize> + TryFrom<usize>
{
}

impl<S> SignedIntegerElement for S where S: IntegerElement<Signed = S> + num_traits::Signed + TryInto<isize> {}
impl<U> UnsignedIntegerElement for U where
    U: IntegerElement<Unsigned = U> + num_traits::Unsigned + TryInto<usize> + TryFrom<usize>
{
}

pub mod float;
pub use float::{FloatElement, FloatElementWithBits};
