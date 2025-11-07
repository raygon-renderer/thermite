#![warn(missing_docs, clippy::missing_safety_doc)]

//! Vector type and operations, where each vector wraps a low-level SIMD register type.

use crate::{
    mask::Mask,
    register::{
        self, BitsRegister, CastRegister, FloatRegister, IntegerRegister, LinAlg3Register, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShiftRegister, ShuffleRegister, SignedRegister, SwizzleRegister,
        UnsignedIntegerRegister,
    },
};

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Index, IndexMut,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use num_traits::{MulAdd, MulAddAssign, Num, One, Saturating, SaturatingAdd, SaturatingSub, Zero};

/// SIMD Vector type.
///
/// This wraps a low-level register type and provides a vector-like interface, including
/// operator overloading and element-wise operations.
#[repr(transparent)]
pub struct Vector<R: Register>(pub(crate) R::Storage);

impl<R: Register> Clone for Vector<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Vector<R> {}

const _: () = {
    use core::fmt;

    impl<R: Register> fmt::Debug for Vector<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut t = f.debug_tuple("Vector");

            for v in R::as_array(&self.0) {
                t.field(&v);
            }

            t.finish()
        }
    }
};

use generic_array::{GenericArray, typenum::Unsigned};

#[cfg(feature = "const-default")]
impl<R: Register> const_default::ConstDefault for Vector<R> {
    const DEFAULT: Self = Self::EMPTY;
}

impl<R: Register> Vector<R> {
    /// Number of lanes in the vector.
    pub const LANES: usize = <R::Lanes as Unsigned>::USIZE;

    /// Create a new vector from a single element by splatting it across all lanes.
    ///
    /// If you **NEED** to use this in a const-context, use [`Vector::splat_const`] instead, but
    /// it has downsides if used in non-const contexts.
    #[inline(always)]
    pub fn splat(value: R::Element) -> Self {
        Self(R::splat(value))
    }

    /// **READ DOCS** Create a new vector from a single element by splatting it across all lanes.
    ///
    /// This is a const version of [`Vector::splat`]. However, if used with any dynamic value
    /// it will likely produce suboptimal code. Use this only if you need to use it in a const context
    /// that will be precalculated at compile time.
    ///
    /// Wrap this call in a `const { }` block to ensure it is evaluated at compile time. This function
    /// is marked as `#[inline(never)]` to intentionally disallow optimizations and make it easier to
    /// debug.
    #[inline(never)]
    pub const fn splat_const(value: R::Element) -> Self {
        Self(register::reg_splat::<R>(value))
    }

    /// Create a new vector from an array of elements.
    ///
    /// This is similar to [`Vector::from_array`], but `const` and more limited due to its use of
    /// const-generics. Unlike [`Vector::splat_const`], this is fine to use with dynamic values,
    /// it can just be more difficult to use in generic contexts.
    #[inline(always)]
    pub const fn new<const N: usize>(values: [R::Element; N]) -> Self
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength<ArrayLength = R::Lanes>,
    {
        Self(register::reg::<R, N>(values))
    }

    /// Broadcast the value of a single lane across all lanes of the vector.
    #[inline(always)]
    pub fn broadcast<const I: usize>(self) -> Self {
        Self(R::broadcast::<I>(self.0))
    }

    /// Broadcast the value of a single lane across all lanes of the vector.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the vector's lanes.
    #[inline(always)]
    pub fn broadcastv(self, idx: usize) -> Self {
        Self(R::broadcastv(self.0, idx))
    }

    /// Load a vector from an **aligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `R::Lanes` elements long.
    #[inline(always)]
    pub unsafe fn load(ptr: *const R::Element) -> Self {
        unsafe { Self(R::load(ptr)) }
    }

    /// Load a vector from an **unaligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid and points to a memory region
    /// that is at least `R::Lanes` elements long. Unaligned access may be slower on some architectures.
    #[inline(always)]
    pub unsafe fn load_unaligned(ptr: *const R::Element) -> Self {
        unsafe { Self(R::load_unaligned(ptr)) }
    }

    /// Store the vector to an **aligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `R::Lanes` elements long.
    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut R::Element) {
        // SAFETY: The caller must ensure that the pointer is valid and aligned.
        unsafe { R::store(ptr, self.0) }
    }

    /// Store the vector to an **unaligned** pointer to its elements.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid and points to a memory region
    /// that is at least `R::Lanes` elements long. Unaligned access may be slower on some architectures.
    #[inline(always)]
    pub unsafe fn store_unaligned(self, ptr: *mut R::Element) {
        // SAFETY: The caller must ensure that the pointer is valid.
        unsafe { R::store_unaligned(ptr, self.0) }
    }

    /// Transforms a slice of element values into a slice of vectors, with
    /// alignment and length checks. A prefix and/or suffix slice may be returned if the slice is
    /// not aligned or if the length is not a multiple of the number of lanes in the vector.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (&[], values, &[]) = i32x4::from_slice(&[1, 2, 3, 4]) else {
    ///     panic!("Slice is not aligned to the register type of the vector, or has remaining elements");
    /// };
    ///
    /// assert_eq!(values, &[i32x4::new([1, 2, 3, 4])]);
    /// ```
    #[inline(always)]
    pub fn from_slice(values: &[R::Element]) -> (&[R::Element], &[Self], &[R::Element]) {
        // SAFETY: This transmutes the slice to Self if and only if it was the correct length and alignment,
        // which is really all that's needed to consider it a slice of registers.
        unsafe { values.align_to::<Self>() }
    }

    /// Create a new vector from an array of elements.
    #[inline(always)]
    pub fn from_array(values: impl Into<GenericArray<R::Element, R::Lanes>>) -> Self {
        Self(R::new(values.into()))
    }

    /// A vector without specified values. This memory is not undefined in a safety sense,
    /// but it is unspecified in a logical sense. The values are not guaranteed to be zero or any other value,
    /// although in practice they are almost certainly zero.
    pub const EMPTY: Self = Self(R::EMPTY);

    /// Create a new vector without specified values. This memory is not undefined in a safety sense,
    /// but it is unspecified in a logical sense. The values are not guaranteed to be zero or any other value,
    /// although in practice they are almost certainly zero.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Join together low and high vectors to create a register of double the size.
    #[inline(always)]
    pub fn join(low: Vector<R::HalfRegister>, high: Vector<R::HalfRegister>) -> Self
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        Self(R::join(low.0, high.0))
    }

    /// Split the double pump register into two vectors, low and high.
    #[inline(always)]
    pub fn split(self) -> (Vector<R::HalfRegister>, Vector<R::HalfRegister>)
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        let (low, high) = R::split(self.0);
        (Vector(low), Vector(high))
    }

    /// Concatenate two Vectors into one vector of twice the size. If a native register of
    /// this size is available, it'll use that, otherwise it'll use a double pump register that
    /// is just two of the original registers working together. This can be nested.
    #[inline(always)]
    pub fn concat(self, other: Self) -> Vector<R::DoubleRegister>
    where
        R::DoubleRegister: Register<HalfRegister = R, Element = R::Element>,
    {
        Vector(R::concat(self.0, other.0))
    }

    /// Returns a reference to the vector's elements as an array.
    #[inline(always)]
    pub fn as_array(&self) -> &GenericArray<R::Element, R::Lanes> {
        R::as_array(&self.0)
    }

    /// Returns a mutable reference to the vector's elements as an array.
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut GenericArray<R::Element, R::Lanes> {
        R::as_array_mut(&mut self.0)
    }

    /// Returns a slice of the vector's elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[R::Element] {
        R::as_array(&self.0).as_slice()
    }

    /// Returns a mutable slice of the vector's elements.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [R::Element] {
        R::as_array_mut(&mut self.0).as_mut_slice()
    }

    /// Convert the vector to an array of elements.
    #[inline(always)]
    pub fn to_array(self) -> GenericArray<R::Element, R::Lanes> {
        R::as_array(&self.0).clone()
    }

    /// Extract a single element from the vector at the given index.
    #[inline(always)]
    pub fn extract<const I: usize>(&self) -> R::Element {
        const { assert!(I < Self::LANES, "Index out of bounds") };

        R::extract::<I>(self.0)
    }

    /// Replace a single element in the vector at the given index with a new value.
    #[inline(always)]
    pub fn insert<const I: usize>(self, element: R::Element) -> Self {
        const { assert!(I < Self::LANES, "Index out of bounds") };

        Self(R::insert::<I>(self.0, element))
    }

    /// Reverse the order of the elements in the vector.
    #[inline(always)]
    pub fn reverse(self) -> Self {
        Self(R::reverse(self.0))
    }

    #[inline(always)]
    fn fold<F>(self, init: R::Element, f: F) -> R::Element
    where
        F: Fn(R::Element, R::Element) -> R::Element,
    {
        R::fold(init, self.0, f)
    }

    #[inline(always)]
    fn reduce<F>(self, f: F) -> R::Element
    where
        F: Fn(R::Element, R::Element) -> R::Element,
    {
        R::reduce(self.0, f)
    }
}

impl<R: Register> Vector<R> {
    /// Cast the vector to a different type, converting the elements.
    ///
    /// This is not a bitwise cast, but a conversion of the elements to the new type,
    /// and therefore may lose precision or change the representation of the data.
    #[inline(always)]
    pub fn cast<INTO: CastRegister<R>>(self) -> Vector<INTO> {
        Vector(INTO::cast_from(self.0))
    }

    /// Like [`Vector::cast`], but potentially uses faster
    /// conversion methods if available, at the cost of
    /// potentially losing precision or only working with a subset of the
    /// values possible in the original type. If outside of that,
    /// junk may be returned.
    ///
    /// If you know your value is within the range of the target type,
    /// this is a good way to convert it without the overhead of
    /// fully conforming to the type's range.
    #[inline(always)]
    pub fn fast_cast<INTO: CastRegister<R>>(self) -> Vector<INTO> {
        Vector(INTO::fast_cast_from(self.0))
    }

    /// Cast the input vector to a different type, converting the elements
    /// as necessary.
    #[inline(always)]
    pub fn from<FROM: Register>(value: Vector<FROM>) -> Vector<R>
    where
        R: CastRegister<FROM>,
    {
        Vector(R::cast_from(value.0))
    }

    /// Like [`Vector::from`], but potentially uses faster
    /// conversion methods if available, at the cost of
    /// potentially losing precision or only working with a subset of the
    /// values possible in the original type. If outside of that,
    /// junk may be returned.
    ///
    /// If you know your value is within the range of the target type,
    /// this is a good way to convert it without the overhead of
    /// fully conforming to the type's range.
    #[inline(always)]
    pub fn fast_from<FROM: Register>(value: Vector<FROM>) -> Vector<R>
    where
        R: CastRegister<FROM>,
    {
        Vector(R::fast_cast_from(value.0))
    }

    /// Convert the vector to a different type, without changing the representation of the data.
    ///
    /// This is a bitwise cast, and therefore may not be safe if the types are not compatible.
    #[inline(always)]
    pub fn into_bits<INTO: BitsRegister<R>>(self) -> Vector<INTO> {
        Vector(INTO::from_bits(self.0))
    }

    /// Convert the vector to a different type, without changing the representation of the data.
    #[inline(always)]
    pub fn from_bits<FROM: Register>(value: Vector<FROM>) -> Vector<R>
    where
        R: BitsRegister<FROM>,
    {
        Vector(R::from_bits(value.0))
    }
}

impl<R: ShiftRegister> Vector<R> {
    /// For each lane in the vector, shift left by the immediate value.
    #[inline(always)]
    pub fn shli<const IMM8: i32>(self) -> Self {
        Self(R::shli::<IMM8>(self.0))
    }

    /// For each lane in the vector, shift right by the immediate value.
    #[inline(always)]
    pub fn shri<const IMM8: i32>(self) -> Self {
        Self(R::shri::<IMM8>(self.0))
    }
}

#[rustfmt::skip]
impl<R: PartialOrdRegister> Vector<R> {
    /// For each lane in the vector, return a mask indicating whether that lane is less than the other.
    #[inline(always)] pub fn cmp_lt(self, rhs: Self) -> Mask<R> { Mask(R::lt(self.0, rhs.0)) }
    /// For each lane in the vector, return a mask indicating whether that lane is less than or equal to the other.
    #[inline(always)] pub fn cmp_le(self, rhs: Self) -> Mask<R> { Mask(R::le(self.0, rhs.0)) }
    /// For each lane in the vector, return a mask indicating whether that lane is greater than the other.
    #[inline(always)] pub fn cmp_gt(self, rhs: Self) -> Mask<R> { Mask(R::gt(self.0, rhs.0)) }
    /// For each lane in the vector, return a mask indicating whether that lane is greater than or equal to the other.
    #[inline(always)] pub fn cmp_ge(self, rhs: Self) -> Mask<R> { Mask(R::ge(self.0, rhs.0)) }
    /// For each lane in the vector, return a mask indicating whether that lane is equal to the other.
    #[inline(always)] pub fn cmp_eq(self, rhs: Self) -> Mask<R> { Mask(R::eq(self.0, rhs.0)) }
    /// For each lane in the vector, return a mask indicating whether that lane is not equal to the other.
    #[inline(always)] pub fn cmp_ne(self, rhs: Self) -> Mask<R> { Mask(R::ne(self.0, rhs.0)) }

    /// For each lane in the vector, return a mask indicating whether that lane is less than the provided value.
    #[inline(always)] pub fn cmp_lt1(self, rhs: R::Element) -> Mask<R> { Mask(R::lt(self.0, R::splat(rhs))) }
    /// For each lane in the vector, return a mask indicating whether that lane is less than or equal to the provided value.
    #[inline(always)] pub fn cmp_le1(self, rhs: R::Element) -> Mask<R> { Mask(R::le(self.0, R::splat(rhs))) }
    /// For each lane in the vector, return a mask indicating whether that lane is greater than the provided value.
    #[inline(always)] pub fn cmp_gt1(self, rhs: R::Element) -> Mask<R> { Mask(R::gt(self.0, R::splat(rhs))) }
    /// For each lane in the vector, return a mask indicating whether that lane is greater than or equal to the provided value.
    #[inline(always)] pub fn cmp_ge1(self, rhs: R::Element) -> Mask<R> { Mask(R::ge(self.0, R::splat(rhs))) }
    /// For each lane in the vector, return a mask indicating whether that lane is equal to the provided value.
    #[inline(always)] pub fn cmp_eq1(self, rhs: R::Element) -> Mask<R> { Mask(R::eq(self.0, R::splat(rhs))) }
    /// For each lane in the vector, return a mask indicating whether that lane is not equal to the provided value.
    #[inline(always)] pub fn cmp_ne1(self, rhs: R::Element) -> Mask<R> { Mask(R::ne(self.0, R::splat(rhs))) }
}

impl<R: NumericRegister> Vector<R> {
    /// Return the minimum of two vectors, element-wise.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(R::min(self.0, rhs.0))
    }

    /// Return the maximum of two vectors, element-wise.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(R::max(self.0, rhs.0))
    }

    /// Clamps the elements of the vector between the given minimum and maximum values.
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// A vector of the minimum value the element type of this vector can represent.
    pub const MIN: Self = Self(R::MIN);

    /// A vector of the maximum value the element type of this vector can represent.
    pub const MAX: Self = Self(R::MAX);

    /// A vector of the value "0" in the element type.
    pub const ZERO: Self = Self(R::ZERO);
    /// A vector of the value "1" in the element type.
    pub const ONE: Self = Self(R::ONE);
    /// A vector of the value "2" in the element type.
    pub const TWO: Self = Self(R::TWO);

    /// Returns a vector where each element is the index of the lane as that element type.
    ///
    /// `[0, 1, 2, 3]`, etc.
    #[inline(always)]
    pub fn indexed() -> Self {
        Self(R::indexed())
    }

    /// Returns the minimum value in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    #[inline(always)]
    pub fn min_element(self) -> R::Element {
        R::min_element(self.0)
    }

    /// Returns the maximum value in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    #[inline(always)]
    pub fn max_element(self) -> R::Element {
        R::max_element(self.0)
    }

    /// Returns the sum of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    #[inline(always)]
    pub fn sum_elements(self) -> R::Element {
        R::sum_elements(self.0)
    }

    /// Returns the product of all elements in the vector.
    ///
    /// This operation has an `O(log2 n)` complexity to reduce.
    #[inline(always)]
    pub fn prod_elements(self) -> R::Element {
        R::prod_elements(self.0)
    }
}

impl<R: NumericRegister> core::iter::Sum for Vector<R> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector::ZERO, |acc, vec| acc + vec)
    }
}

impl<R: NumericRegister> core::iter::Product for Vector<R> {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector::ONE, |acc, vec| acc * vec)
    }
}

impl<R: NumericRegister> num_traits::Bounded for Vector<R> {
    #[inline(always)]
    fn max_value() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::MIN
    }
}

impl<R: PartialOrdRegister> PartialEq for Vector<R> {
    /// Compare two vectors for equality, returning true only if all elements are equal.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        R::all(R::eq(self.0, other.0))
    }

    /// Compare two vectors for inequality, returning true if any element is not equal.
    #[allow(clippy::partialeq_ne_impl)] // sometimes might have better underlying implementation
    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        R::any(R::ne(self.0, other.0))
    }
}

impl<R: NumericRegister> Num for Vector<R>
where
    R::Element: Num,
{
    type FromStrRadixErr = <R::Element as Num>::FromStrRadixErr;

    #[inline(always)]
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self(R::splat(Num::from_str_radix(s, radix)?)))
    }
}

impl<R: UnsignedIntegerRegister> num_traits::Unsigned for Vector<R> where R::Element: num_traits::Unsigned {}

impl<R: SignedRegister> Vector<R> {
    /// A vector of the value "-1" in the element type.
    pub const NEG_ONE: Self = Self(R::NEG_ONE);

    /// Take the absolute value of the vector, element-wise.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(R::abs(self.0))
    }

    /// For each element in the vector, set the sign of that
    /// element to the sign of the corresponding element in the other vector.
    #[inline(always)]
    pub fn copysign(self, rhs: Self) -> Self {
        Self(R::copysign(self.0, rhs.0))
    }

    /// For each element in the vector, return a new vector
    /// where each element is either -1 or +1 depending
    /// on the sign of the element.
    #[inline(always)]
    pub fn signum(self) -> Self {
        Self(R::signum(self.0))
    }

    /// For each element in the vector, return a mask indicating
    /// whether that element is negative.
    #[inline(always)]
    pub fn is_negative(self) -> Mask<R> {
        Mask(R::is_negative(self.0))
    }

    /// For each element in the vector, return a mask indicating
    /// whether that element is positive.
    #[inline(always)]
    pub fn is_positive(self) -> Mask<R> {
        Mask(R::is_positive(self.0))
    }

    /// Conditionally negate each lane in the vector based on the mask, where a `true` value
    /// indicates that the lane should be negated.
    #[inline(always)]
    pub fn conditional_negate(self, mask: Mask<R>) -> Self {
        Self(R::conditional_negate(self.0, mask.0))
    }

    /// Selects elements from `truthy` or `falsy` based on the mask,
    /// where `true` in the mask selects from `truthy` and `false` selects from `falsy`.
    #[inline(always)]
    pub fn select_negative(self, truthy: Self, falsy: Self) -> Self {
        Self(R::select_negative(self.0, falsy.0, truthy.0))
    }
}

impl<R: SignedRegister> num_traits::Signed for Vector<R>
where
    R::Element: num_traits::Signed,
{
    #[inline(always)]
    fn abs(&self) -> Self {
        Self(R::abs(self.0))
    }

    #[inline(always)]
    fn abs_sub(&self, other: &Self) -> Self {
        Self(R::sub(R::max(self.0, other.0), other.0))
    }

    #[inline(always)]
    fn signum(&self) -> Self {
        Self(R::signum(self.0))
    }

    /// Returns true if any element in the vector is negative.
    #[inline(always)]
    fn is_negative(&self) -> bool {
        // true if any element is negative
        R::any(R::is_negative(self.0))
    }

    /// Returns true if all elements in the vector are positive.
    #[inline(always)]
    fn is_positive(&self) -> bool {
        // true if all elements are positive
        R::all(R::is_positive(self.0))
    }
}

impl<R: FloatRegister> Vector<R> {
    /// A vector of the value "0.5" in the element type.
    pub const HALF: Self = Self(R::HALF);
    /// A vector of the value "-0.0" in the element type.
    pub const NEG_ZERO: Self = Self(R::NEG_ZERO);
    /// A vector of the positive infinity value in the element type.
    pub const INFINITY: Self = Self(R::INFINITY);
    /// A vector of the negative infinity value in the element type.
    pub const NEG_INFINITY: Self = Self(R::NEG_INFINITY);
    /// A vector of NaN values in the element type.
    pub const NAN: Self = Self(R::NAN);
    /// A vector of the smallest positive value in the element type.
    pub const EPSILON: Self = Self(R::EPSILON);

    /// Check if each element in the vector is infinite, returning a mask.
    #[inline(always)]
    pub fn is_infinite(self) -> Mask<R> {
        Mask(R::is_infinite(self.0))
    }

    /// Check if each element in the vector is finite, returning a mask.
    #[inline(always)]
    pub fn is_finite(self) -> Mask<R> {
        Mask(R::is_finite(self.0))
    }

    /// Check if each element in the vector is NaN, returning a mask.
    #[inline(always)]
    pub fn is_nan(self) -> Mask<R> {
        Mask(R::is_nan(self.0))
    }

    /// Check if each element in the vector is zero or subnormal, returning a mask.
    #[inline(always)]
    pub fn is_zero_or_subnormal(self) -> Mask<R> {
        Mask(R::is_zero_or_subnormal(self.0))
    }

    // TODO: Move to math library?
    // #[inline(always)]
    // pub fn lerp(self, a: Self, b: Self) -> Self {
    //     self.mul_adde(b - a, a) // t * (b - a) + a
    // }

    /// Maybe fused Multiply-Add operation.
    ///
    /// If the instruction set does not support fused-multiply-add
    /// instructions than this will fallback to regular operations.
    #[inline(always)]
    pub fn mul_adde(self, rhs: Self, acc: Self) -> Self {
        Self(R::mul_adde(self.0, rhs.0, acc.0))
    }

    /// Maybe fused Multiply-Subtract operation.
    ///
    /// If the instruction set does not support fused-multiply-add
    /// instructions than this will fallback to regular operations.
    #[inline(always)]
    pub fn mul_sube(self, rhs: Self, acc: Self) -> Self {
        Self(R::mul_sube(self.0, rhs.0, acc.0))
    }

    /// Maybe fused Negate-Multiply-Add operation.
    ///
    /// If the instruction set does not support fused-multiply-add
    /// instructions than this will fallback to regular operations.
    #[inline(always)]
    pub fn nmul_adde(self, rhs: Self, acc: Self) -> Self {
        Self(R::nmul_adde(self.0, rhs.0, acc.0))
    }

    /// Maybe fused Negate-Multiply-Subtract operation.
    ///
    /// If the instruction set does not support fused-multiply-add
    /// instructions than this will fallback to regular operations.
    #[inline(always)]
    pub fn nmul_sube(self, rhs: Self, acc: Self) -> Self {
        Self(R::nmul_sube(self.0, rhs.0, acc.0))
    }

    /// Guaranteed fused Multiply-Add operation.
    ///
    /// On platforms where FMA is not available this will
    /// fallback to `libm` element-wise FMA, which is very slow.
    #[inline(always)]
    pub fn mul_add(self, rhs: Self, acc: Self) -> Self {
        Self(R::mul_add(self.0, rhs.0, acc.0))
    }

    /// Guaranteed fused Multiply-Subtract operation.
    ///
    /// On platforms where FMA is not available this will
    /// fallback to `libm` element-wise FMA, which is very slow.
    #[inline(always)]
    pub fn mul_sub(self, rhs: Self, acc: Self) -> Self {
        Self(R::mul_sub(self.0, rhs.0, acc.0))
    }

    /// Guaranteed Negate-Multiply-Add operation.
    ///
    /// On platforms where FMA is not available this will
    /// fallback to `libm` element-wise FMA, which is very slow.
    #[inline(always)]
    pub fn nmul_add(self, rhs: Self, acc: Self) -> Self {
        Self(R::nmul_add(self.0, rhs.0, acc.0))
    }

    /// Guaranteed Negate-Multiply-Subtract operation.
    ///
    /// On platforms where FMA is not available this will
    /// fallback to `libm` element-wise FMA, which is very slow.
    #[inline(always)]
    pub fn nmul_sub(self, rhs: Self, acc: Self) -> Self {
        Self(R::nmul_sub(self.0, rhs.0, acc.0))
    }

    /// Square root operation.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(R::sqrt(self.0))
    }

    /// Approximate inverse square root operation.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(R::rsqrt(self.0))
    }

    /// Approximate reciprocal operation.
    #[inline(always)]
    pub fn rcp(self) -> Self {
        Self(R::rcp(self.0))
    }

    /// Floor operation.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(R::floor(self.0))
    }

    /// Ceiling operation.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(R::ceil(self.0))
    }

    /// Round to nearest int operation.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(R::round(self.0))
    }

    /// Truncate to int operation.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(R::trunc(self.0))
    }

    /// Fractional part operation.
    #[inline(always)]
    pub fn fract(self) -> Self {
        Self(R::fract(self.0))
    }

    /// Effectively `self * sign.signum()`, multiplying the sign bits.
    #[inline(always)]
    pub fn mul_sign(self, sign: Self) -> Self {
        Self(R::mul_sign(self.0, sign.0))
    }

    /// Returns zero with the sign of `self`, i.e.: only the sign bit
    /// is set.
    #[inline(always)]
    pub fn signed_zero(self) -> Self {
        Self(R::signed_zero(self.0))
    }

    /// Returns the next representable value greater than the current value.
    #[inline(always)]
    pub fn next_up(self) -> Self {
        Self(R::next_up(self.0))
    }

    /// Returns the next representable value less than the current value.
    #[inline(always)]
    pub fn next_down(self) -> Self {
        Self(R::next_down(self.0))
    }
}

impl<R: NumericRegister> num_traits::ConstZero for Vector<R> {
    const ZERO: Self = Self(R::ZERO);
}

impl<R: NumericRegister> num_traits::ConstOne for Vector<R> {
    const ONE: Self = Self(R::ONE);
}

#[rustfmt::skip]
impl<R: FloatRegister> num_traits::FloatConst for Vector<R>
where
    R::Element: num_traits::FloatConst
        + Add<R::Element, Output = R::Element>
        + Div<R::Element, Output = R::Element>,
{
    #[inline(always)] fn E() -> Self { Self::splat(num_traits::FloatConst::E()) }
    #[inline(always)] fn FRAC_1_PI() -> Self { Self::splat(num_traits::FloatConst::FRAC_1_PI()) }
    #[inline(always)] fn FRAC_1_SQRT_2() -> Self { Self::splat(num_traits::FloatConst::FRAC_1_SQRT_2()) }
    #[inline(always)] fn FRAC_2_PI() -> Self { Self::splat(num_traits::FloatConst::FRAC_2_PI()) }
    #[inline(always)] fn FRAC_2_SQRT_PI() -> Self { Self::splat(num_traits::FloatConst::FRAC_2_SQRT_PI()) }
    #[inline(always)] fn FRAC_PI_2() -> Self { Self::splat(num_traits::FloatConst::FRAC_PI_2()) }
    #[inline(always)] fn FRAC_PI_3() -> Self { Self::splat(num_traits::FloatConst::FRAC_PI_3()) }
    #[inline(always)] fn FRAC_PI_4() -> Self { Self::splat(num_traits::FloatConst::FRAC_PI_4()) }
    #[inline(always)] fn FRAC_PI_6() -> Self { Self::splat(num_traits::FloatConst::FRAC_PI_6()) }
    #[inline(always)] fn FRAC_PI_8() -> Self { Self::splat(num_traits::FloatConst::FRAC_PI_8()) }
    #[inline(always)] fn LN_10() -> Self { Self::splat(num_traits::FloatConst::LN_10()) }
    #[inline(always)] fn LN_2() -> Self { Self::splat(num_traits::FloatConst::LN_2()) }
    #[inline(always)] fn LOG10_E() -> Self { Self::splat(num_traits::FloatConst::LOG10_E()) }
    #[inline(always)] fn LOG2_E() -> Self { Self::splat(num_traits::FloatConst::LOG2_E()) }
    #[inline(always)] fn PI() -> Self { Self::splat(num_traits::FloatConst::PI()) }
    #[inline(always)] fn SQRT_2() -> Self { Self::splat(num_traits::FloatConst::SQRT_2()) }

    // the bounds on these three are dumb
    #[inline(always)]
    fn TAU() -> Self where Self: Sized + Add<Self, Output = Self> {
        Self::splat(num_traits::FloatConst::TAU())
    }

    #[inline(always)]
    fn LOG10_2() -> Self where Self: Sized + Div<Self, Output = Self> {
        Self::splat(num_traits::FloatConst::LOG10_2())
    }

    #[inline(always)]
    fn LOG2_10() -> Self where Self: Sized + Div<Self, Output = Self> {
        Self::splat(num_traits::FloatConst::LOG2_10())
    }
}

impl<R: LinAlg3Register> Vector<R> {
    /// Scalar Product using only the first three lanes of the register as a 3D vector.
    ///
    /// This is more efficient than a raw scalar product, as there is no need to
    /// zero out the last lane of the register.
    #[inline(always)]
    pub fn dot3(self, rhs: Self) -> R::Element {
        R::dot3(self.0, rhs.0)
    }

    /// Cross Product using only the first three lanes of the register as a 3D vector.
    ///
    /// This is more efficient than a raw cross product, as there is no need to
    /// zero out the last lane of the register.
    #[inline(always)]
    pub fn cross3(self, rhs: Self) -> Self {
        Self(R::cross3(self.0, rhs.0))
    }

    /// Efficiently set the 4th (last) lane of the register to 0.0.
    ///
    /// Useful for sanitizing 3D Homogeneous vectors.
    ///
    /// See [`Vector::one4`] for similar functionality for 3D points.
    #[inline(always)]
    pub fn zero4(self) -> Self {
        Self(R::zero4(self.0))
    }

    /// Efficiently set the 4th (last) lane of the register to 1.0.
    ///
    /// Useful for sanitizing 3D Homogeneous points.
    ///
    /// See [`Vector::zero4`] for similar functionality for 3D vectors.
    #[inline(always)]
    pub fn one4(self) -> Self {
        Self(R::one4(self.0))
    }

    /// Returns the minimum value in the first three lanes of the register.
    #[inline(always)]
    pub fn min_element3(self) -> R::Element {
        R::min_element3(self.0)
    }

    /// Returns the maximum value in the first three lanes of the register.
    #[inline(always)]
    pub fn max_element3(self) -> R::Element {
        R::max_element3(self.0)
    }

    /// Returns the sum of the first three elements of the register.
    #[inline(always)]
    pub fn sum_elements3(self) -> R::Element {
        R::sum_elements3(self.0)
    }

    /// Returns the product of the first three elements of the register.
    #[inline(always)]
    pub fn prod_elements3(self) -> R::Element {
        R::prod_elements3(self.0)
    }
}

impl<R: ShuffleRegister> Vector<R> {
    /// Shuffle vectors according to the mask in IMM8. Note that this
    /// instruction does not work like expected.
    ///
    /// For each 128-bit subvector, it interleaves A and B, selecting
    /// from A for imm8 bits 0:1 and 2:3, then from B using bits 4:5 and 6:7
    ///
    /// If the inputs are larger than 128 bits, it repeats this for each 128 bits
    /// **_using the same imm8 indices_**.
    #[inline(always)]
    pub fn interleave<const IMM8: i32>(self, other: Self) -> Self {
        Self(R::shuffle::<IMM8>(self.0, other.0))
    }
}

impl<R: Register> Index<usize> for Vector<R> {
    type Output = R::Element;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &R::as_array(&self.0)[index]
    }
}

impl<R: Register> IndexMut<usize> for Vector<R> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut R::as_array_mut(&mut self.0)[index]
    }
}

impl<R: NumericRegister> Zero for Vector<R> {
    /// Returns true if all elements in the vector are zero.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        R::all(R::eq(self.0, R::ZERO))
    }

    #[inline(always)]
    fn set_zero(&mut self) {
        self.0 = R::ZERO;
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }
}

impl<R: NumericRegister> One for Vector<R> {
    /// Returns true if all elements in the vector are one.
    #[inline(always)]
    fn is_one(&self) -> bool {
        R::all(R::eq(self.0, R::ONE))
    }

    #[inline(always)]
    fn set_one(&mut self) {
        self.0 = R::ONE;
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

#[rustfmt::skip]
macro_rules! impl_binary_op {
    ($R:ident; $($trait:ident::$op:ident),* $(,)?) => {paste::paste!{$(
        impl<R: $R> $trait<Self> for Vector<R> {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self::Output {
                Self(R::$op(self.0, rhs.0))
            }
        }

        impl <R: $R> [<$trait Assign>] for Vector<R> {
            #[inline(always)]
            fn [<$op _assign>](&mut self, rhs: Self) {
                self.0 = R::$op(self.0, rhs.0);
            }
        }
    )*}};
}

impl_binary_op!(Register; BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl<R: Register> Not for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(R::not(self.0))
    }
}

impl<R: Register> Shr<u32> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(R::shr(self.0, rhs))
    }
}

impl<R: Register> Shr<GenericArray<u32, R::Lanes>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: GenericArray<u32, R::Lanes>) -> Self::Output {
        Self(R::shrv(self.0, rhs))
    }
}

impl<R: Register> Shl<GenericArray<u32, R::Lanes>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: GenericArray<u32, R::Lanes>) -> Self::Output {
        Self(R::shlv(self.0, rhs))
    }
}

impl<R: Register> Shl<u32> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(R::shl(self.0, rhs))
    }
}

impl<R: Register> ShlAssign<u32> for Vector<R> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 = R::shl(self.0, rhs);
    }
}

impl<R: Register> ShrAssign<u32> for Vector<R> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 = R::shr(self.0, rhs);
    }
}

impl<R: Register> ShrAssign<GenericArray<u32, R::Lanes>> for Vector<R> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: GenericArray<u32, R::Lanes>) {
        self.0 = R::shrv(self.0, rhs);
    }
}

impl<R: Register> ShlAssign<GenericArray<u32, R::Lanes>> for Vector<R> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: GenericArray<u32, R::Lanes>) {
        self.0 = R::shlv(self.0, rhs);
    }
}

impl_binary_op!(NumericRegister; Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);

impl<R: SignedRegister> Neg for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(R::neg(self.0))
    }
}

impl<R: FloatRegister> MulAdd for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, rhs: Self, acc: Self) -> Self::Output {
        Self(R::mul_adde(self.0, rhs.0, acc.0))
    }
}

impl<R: FloatRegister> MulAddAssign for Vector<R> {
    #[inline(always)]
    fn mul_add_assign(&mut self, rhs: Self, acc: Self) {
        self.0 = R::mul_adde(self.0, rhs.0, acc.0);
    }
}

impl<R: IntegerRegister> SaturatingAdd for Vector<R> {
    #[inline(always)]
    fn saturating_add(&self, v: &Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }
}

impl<R: IntegerRegister> SaturatingSub for Vector<R> {
    #[inline(always)]
    fn saturating_sub(&self, v: &Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> Saturating for Vector<R> {
    #[inline(always)]
    fn saturating_add(self, v: Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }

    #[inline(always)]
    fn saturating_sub(self, v: Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> Vector<R> {
    /// Perform saturating addition for each element of the vectors.
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(R::saturating_add(self.0, rhs.0))
    }

    /// Perform saturating subtraction for each element of the vectors.
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(R::saturating_sub(self.0, rhs.0))
    }

    /// For each element in the vector, rotate the bits to the left by the given
    /// number of bits.
    #[inline(always)]
    pub fn rol(self, shift: u32) -> Self {
        Self(R::rol(self.0, shift))
    }

    /// For each element in the vector, rotate the bits to the right by the given
    /// number of bits.
    #[inline(always)]
    pub fn ror(self, shift: u32) -> Self {
        Self(R::ror(self.0, shift))
    }

    /// For each element in the vector, reverse the bits of that element.
    #[inline(always)]
    pub fn reverse_bits(self) -> Self {
        Self(R::reverse_bits(self.0))
    }

    /// For each element in the vector, count the number of bits that are set to 1.
    #[inline(always)]
    pub fn count_ones(self) -> Self {
        Self(R::count_ones(self.0))
    }

    /// For each element in the vector, count the number of bits that are set to 0.
    #[inline(always)]
    pub fn count_zeros(self) -> Self {
        Self(R::count_zeros(self.0))
    }
}

impl<R: UnsignedIntegerRegister> Vector<R> {
    /// Returns the next power of two minus one for each unsigned integer
    /// element in the vector.
    #[inline(always)]
    pub fn next_power_of_two_m1(self) -> Self {
        Self(R::next_power_of_two_m1(self.0))
    }

    /// Computes log2(x) + 1 for each unsigned integer element in the vector.
    #[inline(always)]
    pub fn ilog2p1(self) -> Self {
        Self(R::ilog2p1(self.0))
    }

    /// Determines if each unsigned integer element in the vector is a
    /// power of two, returning a mask indicating whether or not it is.
    #[inline(always)]
    pub fn is_power_of_two(self) -> Mask<R> {
        Mask(R::is_power_of_two(self.0))
    }

    /// Compute the parity of each unsigned integer lane in the vector.
    pub fn parity(self) -> Self {
        Self(R::parity(self.0))
    }
}

#[rustfmt::skip]
macro_rules! impl_swizzle4 {
    (@ x) => { 0 };
    (@ y) => { 1 };
    (@ z) => { 2 };
    (@ w) => { 3 };

    (IMPL $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c $d>](self) -> Self {
            const IMM8: i32 = MM_SHUFFLE!(
                impl_swizzle4!(@ $d),
                impl_swizzle4!(@ $c),
                impl_swizzle4!(@ $b),
                impl_swizzle4!(@ $a)
            );

            Self(R::permute::<IMM8>(self.0))
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        $(#[$meta])* fn [<$a $b $c $d>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident $d:ident]),*) => {
        /// Only available for 4-lane vectors, this allows human-readable swizzle/permutations
        /// of the vector.
        pub trait Swizzle4 { $(impl_swizzle4!(DECL $(#[$meta])* $a $b $c $d);)* }

        impl<R: PermuteRegister<Lanes = generic_array::typenum::consts::U4>> Swizzle4 for Vector<R> {
            $(impl_swizzle4!(IMPL $a $b $c $d);)*
        }
    }
}

#[rustfmt::skip]
macro_rules! impl_swizzle3 {
    (IMPL $a:ident $b:ident $c:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c>](self) -> Self {
            const IMM8: i32 = MM_SHUFFLE!(
                3, // 4th lane is unchanged
                impl_swizzle4!(@ $c),
                impl_swizzle4!(@ $b),
                impl_swizzle4!(@ $a)
            );

            Self(R::permute::<IMM8>(self.0))
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident) => {paste::paste! {
        $(#[$meta])* fn [<$a $b $c>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident]),*) => {
        /// Only available for "3-lane" (ignoring 4th lane) [`LinAlg3Register`] vectors,
        /// this allows human-readable swizzle/permutations of the vector. Permutations
        /// will ignore the 4th lane of the register, leaving it unchanged.
        pub trait Swizzle3 { $(impl_swizzle3!(DECL $(#[$meta])* $a $b $c);)* }

        impl<R: LinAlg3Register + PermuteRegister> Swizzle3 for Vector<R> {
            $(impl_swizzle3!(IMPL $a $b $c);)*
        }
    }
}

impl_swizzle3! {
    /// Identity
    [x y z],
    [x x x],
    [x x y],
    [x x z],
    [x y x],
    [x y y],
    [x z x],
    [x z y],
    [x z z],
    [y x x],
    [y x y],
    [y x z],
    [y y x],
    [y y y],
    [y y z],
    [y z x],
    [y z y],
    [y z z],
    [z x x],
    [z x y],
    [z x z],
    [z y x],
    [z y y],
    [z y z],
    [z z x],
    [z z y],
    [z z z]
}

impl_swizzle4! {
    /// Identity
    [x y z w],
    [x x x x],
    [x x x y],
    [x x x z],
    [x x x w],
    [x x y x],
    [x x y y],
    [x x y z],
    [x x y w],
    [x x z x],
    [x x z y],
    [x x z z],
    [x x z w],
    [x x w x],
    [x x w y],
    [x x w z],
    [x x w w],
    [x y x x],
    [x y x y],
    [x y x z],
    [x y x w],
    [x y y x],
    [x y y y],
    [x y y z],
    [x y y w],
    [x y z x],
    [x y z y],
    [x y z z],
    [x y w x],
    [x y w y],
    [x y w z],
    [x y w w],
    [x z x x],
    [x z x y],
    [x z x z],
    [x z x w],
    [x z y x],
    [x z y y],
    [x z y z],
    [x z y w],
    [x z z x],
    [x z z y],
    [x z z z],
    [x z z w],
    [x z w x],
    [x z w y],
    [x z w z],
    [x z w w],
    [x w x x],
    [x w x y],
    [x w x z],
    [x w x w],
    [x w y x],
    [x w y y],
    [x w y z],
    [x w y w],
    [x w z x],
    [x w z y],
    [x w z z],
    [x w z w],
    [x w w x],
    [x w w y],
    [x w w z],
    [x w w w],
    [y x x x],
    [y x x y],
    [y x x z],
    [y x x w],
    [y x y x],
    [y x y y],
    [y x y z],
    [y x y w],
    [y x z x],
    [y x z y],
    [y x z z],
    [y x z w],
    [y x w x],
    [y x w y],
    [y x w z],
    [y x w w],
    [y y x x],
    [y y x y],
    [y y x z],
    [y y x w],
    [y y y x],
    [y y y y],
    [y y y z],
    [y y y w],
    [y y z x],
    [y y z y],
    [y y z z],
    [y y z w],
    [y y w x],
    [y y w y],
    [y y w z],
    [y y w w],
    [y z x x],
    [y z x y],
    [y z x z],
    [y z x w],
    [y z y x],
    [y z y y],
    [y z y z],
    [y z y w],
    [y z z x],
    [y z z y],
    [y z z z],
    [y z z w],
    [y z w x],
    [y z w y],
    [y z w z],
    [y z w w],
    [y w x x],
    [y w x y],
    [y w x z],
    [y w x w],
    [y w y x],
    [y w y y],
    [y w y z],
    [y w y w],
    [y w z x],
    [y w z y],
    [y w z z],
    [y w z w],
    [y w w x],
    [y w w y],
    [y w w z],
    [y w w w],
    [z x x x],
    [z x x y],
    [z x x z],
    [z x x w],
    [z x y x],
    [z x y y],
    [z x y z],
    [z x y w],
    [z x z x],
    [z x z y],
    [z x z z],
    [z x z w],
    [z x w x],
    [z x w y],
    [z x w z],
    [z x w w],
    [z y x x],
    [z y x y],
    [z y x z],
    [z y x w],
    [z y y x],
    [z y y y],
    [z y y z],
    [z y y w],
    [z y z x],
    [z y z y],
    [z y z z],
    [z y z w],
    [z y w x],
    [z y w y],
    [z y w z],
    [z y w w],
    [z z x x],
    [z z x y],
    [z z x z],
    [z z x w],
    [z z y x],
    [z z y y],
    [z z y z],
    [z z y w],
    [z z z x],
    [z z z y],
    [z z z z],
    [z z z w],
    [z z w x],
    [z z w y],
    [z z w z],
    [z z w w],
    [z w x x],
    [z w x y],
    [z w x z],
    [z w x w],
    [z w y x],
    [z w y y],
    [z w y z],
    [z w y w],
    [z w z x],
    [z w z y],
    [z w z z],
    [z w z w],
    [z w w x],
    [z w w y],
    [z w w z],
    [z w w w],
    [w x x x],
    [w x x y],
    [w x x z],
    [w x x w],
    [w x y x],
    [w x y y],
    [w x y z],
    [w x y w],
    [w x z x],
    [w x z y],
    [w x z z],
    [w x z w],
    [w x w x],
    [w x w y],
    [w x w z],
    [w x w w],
    [w y x x],
    [w y x y],
    [w y x z],
    [w y x w],
    [w y y x],
    [w y y y],
    [w y y z],
    [w y y w],
    [w y z x],
    [w y z y],
    [w y z z],
    [w y z w],
    [w y w x],
    [w y w y],
    [w y w z],
    [w y w w],
    [w z x x],
    [w z x y],
    [w z x z],
    [w z x w],
    [w z y x],
    [w z y y],
    [w z y z],
    [w z y w],
    [w z z x],
    [w z z y],
    [w z z z],
    [w z z w],
    [w z w x],
    [w z w y],
    [w z w z],
    [w z w w],
    [w w x x],
    [w w x y],
    [w w x z],
    [w w x w],
    [w w y x],
    [w w y y],
    [w w y z],
    [w w y w],
    [w w z x],
    [w w z y],
    [w w z z],
    [w w z w],
    [w w w x],
    [w w w y],
    [w w w z],
    [w w w w]
}
