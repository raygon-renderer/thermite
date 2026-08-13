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

    /// The greatest value under this type's natural total order, and the least.
    ///
    /// **Not the same as the greatest finite value for floats**, where these are
    /// the infinities. That distinction is the entire reason they exist: a
    /// sorting network pads a partial register with a value that must sort past
    /// every real input, and `f32::MAX` does not sort past `f32::INFINITY`. The
    /// prefix-scan ladder was bitten by exactly this once, with a `+inf` input
    /// lane coming back as `f32::MAX`.
    ///
    /// NaN is deliberately not accounted for - it is unordered, so no value
    /// sorts past it and no sentinel can. A float sort has to handle NaN before
    /// the network sees it; see `thermite-sort`.
    const ORDER_MAX: Self;
    /// The least value under this type's natural total order. See
    /// [`ORDER_MAX`](Self::ORDER_MAX).
    const ORDER_MIN: Self;

    /// Whether values of this type can be *unordered* under [`PartialOrd`] -
    /// float NaN. `false` for every integer type.
    ///
    /// This is a compile-time gate, not a detector: sorting and searching
    /// algorithms use it to skip their NaN pre-pass entirely for types that
    /// cannot contain one, folding the code away at monomorphization. The
    /// runtime test for the lanes themselves is order-theoretic and needs no
    /// per-type code: a value is unordered iff `v != v`, so
    /// `v.cmp_eq(v)` masks the ordered lanes on any vector - including
    /// composite vectors, whose comparisons delegate to their value part.
    ///
    /// Composite element types (e.g. `Compensated<E>`) should forward their
    /// inner element's value rather than restate it.
    const HAS_UNORDERED: bool = false;

    /// Whether this is a floating-point element type. This is distinct
    /// from `HAS_UNORDERED` because some float types may
    /// not have NaN and so are ordered, but still have some special
    /// properties of floats.
    const IS_FLOAT: bool = false;

    fn from_i8(value: i8) -> Self;
    fn from_u8(value: u8) -> Self;
    fn from_u16(value: u16) -> Self;

    /// Returns a scalar `Vector<Self>`.
    #[inline(always)]
    fn as_vector(self) -> crate::Vector<Self>
    where
        Self: crate::register::Register<Storage = Self>,
    {
        crate::Vector(self)
    }
}

pub trait ElementExt: Element {
    type Element: Element<Unsigned = Self::Unsigned, Signed = Self::Signed>;
}

impl<T: Element> ElementExt for T {
    type Element = Self;
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

            const ORDER_MAX: Self = <$t>::MAX;
            const ORDER_MIN: Self = <$t>::MIN;

            #[inline(always)] fn from_i8(value: i8) -> Self { value as $t }
            #[inline(always)] fn from_u8(value: u8) -> Self { value as $t }
            #[inline(always)] fn from_u16(value: u16) -> Self { value as $t }
        }

        impl IntegerElement for $t {
            #[inline(always)] fn logical_shr(self, n: $u) -> Self { ((self as $u) >> n) as $t }
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

            const ORDER_MAX: Self = <$f>::INFINITY;
            const ORDER_MIN: Self = <$f>::NEG_INFINITY;

            const HAS_UNORDERED: bool = true;
            const IS_FLOAT: bool = true;

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
    /// Logical (zero-fill) right shift by `n` bits, independent of signedness.
    fn logical_shr(self, n: Self::Unsigned) -> Self;
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
pub use float::{FloatElement, FloatElementWithBits, IntConst, RatioConst};
