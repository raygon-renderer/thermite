#![warn(missing_docs, clippy::missing_safety_doc)]

//! Vector type and operations, where each vector wraps a low-level SIMD register type.

use crate::{
    divider::{BranchfreeDivider, Denominator, Divider, UnsupportedDivisor, vector::VectorDivider},
    generic::*,
    mask::Mask,
    math::FloatConsts,
    register::{
        self, BitCastRegister, BitshiftRegister, CastRegister, FloatRegister, IntegerRegister, LinAlg3Register,
        LinAlg4Register, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
    },
};

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Index, IndexMut,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use num_traits::{
    ConstOne, ConstZero, MulAdd, MulAddAssign, Num, One, Saturating, SaturatingAdd, SaturatingSub, Signed, WrappingAdd,
    WrappingMul, WrappingSub, Zero,
};

pub mod num;
pub mod streaming;
pub mod unaligned;

/// SIMD Vector type.
///
/// This wraps a low-level register type and provides a vector-like interface, including
/// operator overloading and element-wise operations.
#[repr(transparent)]
pub struct Vector<R: Register>(#[doc(hidden)] pub Storage<R>);

#[doc(hidden)]
pub trait IRegisterOf {
    type Register: Register;
}

impl<R: Register> IRegisterOf for Vector<R> {
    type Register = R;
}

impl<R: Register> IRegisterOf for Mask<R> {
    type Register = R;
}

/// The mask type corresponding to a given vector type.
///
/// # Example
/// ```
/// # use thermite::vector::{Vector, MaskOf};
/// # use thermite::backend::scalar::*;
/// fn example(x: f32x4) -> MaskOf<f32x4> {
///     x.is_negative()
/// }
/// ```
pub type MaskOf<V> = Mask<<V as IRegisterOf>::Register>;

/// The register type corresponding to a given vector or mask type.
pub type RegisterOf<V> = <V as IRegisterOf>::Register;

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

impl<R: Register> const_default::ConstDefault for Vector<R> {
    const DEFAULT: Self = Self::EMPTY;
}

impl<R: Register> Default for Vector<R> {
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<R: Register> Vector<R> {
    /// Number of lanes in the vector.
    pub const LANES: usize = <R::Lanes as Unsigned>::USIZE;

    /// Create a new vector from a single element by splatting it across all lanes.
    ///
    /// If you're seeing this documentation, you are using the `nightly` feature on the nightly branch of Rust.
    /// This version of `splat` uses `const_eval_select` to choose the best implementation
    /// based on whether it's used in a const context or not.
    #[cfg(feature = "nightly")]
    #[rustversion::nightly]
    #[inline(always)]
    pub const fn splat(value: R::Element) -> Self {
        // On nightly, we can use const_eval_select to choose the best implementation
        // based on whether we're in a const context or not.
        #[inline(always)]
        const fn splat_const_impl<R: Register>(value: R::Element) -> Vector<R> {
            Vector(register::reg_splat::<R>(value))
        }

        #[inline(always)]
        fn splat_runtime_impl<R: Register>(value: R::Element) -> Vector<R> {
            Vector(R::splat(value))
        }

        // SAFETY: This is safe because both branches return the same type.
        unsafe { core::intrinsics::const_eval_select((value,), splat_const_impl, splat_runtime_impl) }
    }

    /// **READ DOCS** Create a new vector from a single element by splatting it across all lanes.
    ///
    /// If you're seeing this documentation, you are using the `nightly` feature on the nightly branch of Rust,
    /// in which case this is the same as [`Vector::splat`], which is `const` and automatically chooses the
    /// best implementation based on whether it's used in a const context or not.
    ///
    /// Without the nightly features, this is still a `const` version of [`Vector::splat`]. However, if
    /// used with any dynamic value it will likely produce suboptimal code. Use this only if you
    /// need to use it in a const context that will be precalculated at compile time. You don't
    /// have to worry about that, though, since the nightly version of this function will
    /// just work as expected.
    #[cfg(feature = "nightly")]
    #[rustversion::nightly]
    #[inline(always)]
    pub const fn splat_const(value: R::Element) -> Self {
        Self::splat(value)
    }

    /// Create a new vector from a single element by splatting it across all lanes.
    ///
    /// If you **NEED** to use this in a const-context, use [`Vector::splat_const`] instead, but
    /// it has downsides if used in non-const contexts.
    ///
    /// If you using the nightly branch of Rust, _and_ the `nightly` crate feature,
    /// this version of `splat` will be `const` and automatically choose the
    /// best implementation based on whether it's used in a const context or not.
    #[cfg(not(feature = "nightly"))]
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
    #[cfg(not(feature = "nightly"))]
    #[inline(never)]
    pub const fn splat_const(value: R::Element) -> Self {
        unsafe {
            use core::mem::transmute_copy;

            // plain transmutes are faster than reg_splat for small lane counts
            match Self::LANES {
                0 => Self::EMPTY,
                1 => transmute_copy(&[value; 1]),
                2 => transmute_copy(&[value; 2]),
                4 => transmute_copy(&[value; 4]),
                8 => transmute_copy(&[value; 8]),
                16 => transmute_copy(&[value; 16]),
                32 => transmute_copy(&[value; 32]),
                64 => transmute_copy(&[value; 64]),
                _ => Self(register::reg_splat::<R>(value)),
            }
        }
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

    /// Create a new vector with the first lane set to the given value, and all other lanes set to zero.
    #[inline(always)]
    pub fn single(value: R::Element) -> Self {
        Self(R::single(value))
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

    /// Load a vector from a pointer to its elements using non-temporal (streaming) loads.
    ///
    /// The memory region should not be accessed frequently by the CPU,
    /// as non-temporal loads are intended for data that will not be reused soon.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `R::Lanes` elements long.
    #[inline(always)]
    pub unsafe fn load_streaming(ptr: *const R::Element) -> Self {
        unsafe { Self(R::load_stream(ptr)) }
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

    /// Store the vector to a pointer to its elements using non-temporal (streaming) stores.
    ///
    /// The memory region should not be accessed frequently by the CPU,
    /// as non-temporal stores are intended for data that will not be reused soon.
    ///
    /// # SAFETY
    /// The caller must ensure that the pointer is valid, aligned, and points to a memory region
    /// that is at least `R::Lanes` elements long.
    pub unsafe fn store_streaming(self, ptr: *mut R::Element) {
        // SAFETY: The caller must ensure that the pointer is valid and aligned.
        unsafe { R::store_stream(ptr, self.0) }
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

    /// Transforms a mutable slice of element values into a mutable slice of vectors, with
    /// alignment and length checks. A prefix and/or suffix slice may be returned if the slice is
    /// not aligned or if the length is not a multiple of the number of lanes in the vector.
    #[inline(always)]
    pub fn from_slice_mut(values: &mut [R::Element]) -> (&mut [R::Element], &mut [Self], &mut [R::Element]) {
        // SAFETY: This transmutes the slice to Self if and only if it was the correct length and alignment,
        // which is really all that's needed to consider it a slice of registers.
        unsafe { values.align_to_mut::<Self>() }
    }

    /// Transform a slice of element values into an unaligned iterator of vectors,
    /// returning any remaining elements as a suffix slice.
    #[inline(always)]
    pub fn from_slice_unaligned<'a>(values: &'a [R::Element]) -> (unaligned::Unaligned<'a, R>, &'a [R::Element]) {
        let num_vectors = values.len() / Self::LANES;
        let offset = num_vectors * Self::LANES;

        let head = &values[..offset];
        let tail = &values[offset..];

        (unaligned::Unaligned(head), tail)
    }

    /// Transform a mutable slice of element values into an unaligned iterator of vectors,
    /// returning any remaining elements as a suffix slice.
    #[inline(always)]
    pub fn from_slice_unaligned_mut<'a>(
        values: &'a mut [R::Element],
    ) -> (unaligned::UnalignedMut<'a, R>, &'a mut [R::Element]) {
        let num_vectors = values.len() / Self::LANES;
        let offset = num_vectors * Self::LANES;

        let (head, tail) = values.split_at_mut(offset);

        (unaligned::UnalignedMut(head), tail)
    }

    /// Like [`Vector::from_slice_unaligned`], but returns the remaining elements as a prefix slice.
    #[inline(always)]
    pub fn from_rslice_unaligned<'a>(values: &'a [R::Element]) -> (&'a [R::Element], unaligned::Unaligned<'a, R>) {
        let num_vectors = values.len() / Self::LANES;
        let offset = values.len() - num_vectors * Self::LANES;

        let head = &values[..offset];
        let tail = &values[offset..];

        (head, unaligned::Unaligned(tail))
    }

    /// Like [`Vector::from_slice_unaligned_mut`], but returns the remaining elements as a prefix slice.
    #[inline(always)]
    pub fn from_rslice_unaligned_mut<'a>(
        values: &'a mut [R::Element],
    ) -> (&'a mut [R::Element], unaligned::UnalignedMut<'a, R>) {
        let num_vectors = values.len() / Self::LANES;
        let offset = values.len() - num_vectors * Self::LANES;

        let (head, tail) = values.split_at_mut(offset);

        (head, unaligned::UnalignedMut(tail))
    }

    /// Iterate over a slice of element values as Vectors using non-temporal (streaming) loads.
    ///
    /// # Panics
    ///
    /// If the slice is not aligned to the register type of the vector, or has remaining elements.
    pub fn stream_slice<'a>(values: &'a [R::Element]) -> impl Iterator<Item = streaming::StreamingVector<'a, R>> {
        let (&[], values, &[]) = (unsafe { values.align_to::<R::Storage>() }) else {
            panic!("Slice is not aligned to the register type of the vector, or has remaining elements");
        };

        values.iter().map(|v| streaming::StreamingVector(v))
    }

    /// Iterate over a mutable slice of element values as Vectors using non-temporal (streaming) loads and stores.
    ///
    /// # Panics
    ///
    /// If the slice is not aligned to the register type of the vector, or has remaining elements.
    pub fn stream_slice_mut<'a>(
        values: &'a mut [R::Element],
    ) -> impl Iterator<Item = streaming::StreamingVectorMut<'a, R>> {
        let (&mut [], values, &mut []) = (unsafe { values.align_to_mut::<R::Storage>() }) else {
            panic!("Slice is not aligned to the register type of the vector, or has remaining elements");
        };

        values.iter_mut().map(|v| streaming::StreamingVectorMut(v))
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

    /// Widen the vector to a register of double the width, filling the high half with zeros.
    #[inline(always)]
    pub fn widen<INTO>(self) -> Vector<INTO>
    where
        INTO: Register<HalfRegister = R, Element = R::Element>,
    {
        Vector(INTO::join(self.0, R::EMPTY))
    }

    /// Narrow the vector to a register of half the width by taking the low half,
    /// and discarding the high half.
    #[inline(always)]
    pub fn narrow(self) -> Vector<R::HalfRegister>
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        Vector(R::split(self.0).0)
    }

    /// Narrow the vector to a register of half the width by taking the high half,
    /// and discarding the low half.
    #[inline(always)]
    pub fn narrow_high(self) -> Vector<R::HalfRegister>
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        Vector(R::split(self.0).1)
    }

    /// Join together low and high vectors to create a register of double the width.
    #[inline(always)]
    pub fn join(low: Vector<R::HalfRegister>, high: Vector<R::HalfRegister>) -> Self
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        Self(R::join(low.0, high.0))
    }

    /// Split the double-width register into two vectors, low and high.
    #[inline(always)]
    pub fn split(self) -> (Vector<R::HalfRegister>, Vector<R::HalfRegister>)
    where
        R::HalfRegister: Register<Element = R::Element, DoubleRegister = R>,
    {
        let (low, high) = R::split(self.0);
        (Vector(low), Vector(high))
    }

    /// Split the register into two vectors of half the width. This does not
    /// guarantee that the original register was double-width.
    #[inline(always)]
    pub fn split2(self) -> (Vector<R::HalfRegister>, Vector<R::HalfRegister>)
    where
        R::HalfRegister: Register<Element = R::Element>,
    {
        let (low, high) = R::split(self.0);
        (Vector(low), Vector(high))
    }

    /// Concatenate two Vectors into one vector of twice the width. If a native register of
    /// this width is available, it'll use that, otherwise it'll use a double-width register that
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
    pub fn as_mut_slice(&mut self) -> &mut [R::Element] {
        R::as_array_mut(&mut self.0).as_mut_slice()
    }

    /// Convert the vector to an array of elements.
    #[inline(always)]
    pub fn to_array(self) -> GenericArray<R::Element, R::Lanes> {
        R::as_array(&self.0).clone()
    }
}

impl<R: Register> Vector<R> {
    /// Cast the vector to a different type, converting the elements.
    ///
    /// This is not a bitwise cast, but a conversion of the elements to the new type,
    /// and therefore may lose precision or change the representation of the data.
    #[inline(always)]
    pub fn cast<INTO: Register + CastRegister<R>>(self) -> Vector<INTO> {
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
    pub fn fast_cast<INTO: Register + CastRegister<R>>(self) -> Vector<INTO> {
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
    pub fn into_bits<INTO: Register + BitCastRegister<R>>(self) -> Vector<INTO> {
        Vector(INTO::from_bits(self.0))
    }

    /// Convert the vector to a different type, without changing the representation of the data.
    #[inline(always)]
    pub fn from_bits<FROM: Register>(value: Vector<FROM>) -> Vector<R>
    where
        R: BitCastRegister<FROM>,
    {
        Vector(R::from_bits(value.0))
    }
}

impl<R: NumericRegister> core::iter::Sum for Vector<R> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ZERO), Add::add)
    }
}

impl<R: NumericRegister> core::iter::Product for Vector<R> {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ONE), Mul::mul)
    }
}

impl<R: NumericRegister> num_traits::Bounded for Vector<R> {
    #[inline(always)]
    fn max_value() -> Self {
        Vector(R::MAX)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Vector(R::MIN)
    }
}

impl<R: PartialOrdRegister> PartialEq for Vector<R> {
    /// Compare two vectors for equality, returning true only if all elements are equal.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        Mask::<R>(R::eq(self.0, other.0)).all()
    }

    /// Compare two vectors for inequality, returning true if any element is not equal.
    #[allow(clippy::partialeq_ne_impl)] // sometimes might have better underlying implementation
    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        Mask::<R>(R::ne(self.0, other.0)).any()
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

    /// A vector of the smallest positive (non-zero) value in the element type.
    pub const MIN_POSITIVE: Self = Self(R::MIN_POSITIVE);

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

impl<R: SignedRegister> Signed for Vector<R>
where
    R::Element: Signed,
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
        Mask::<R>(R::is_negative(self.0)).any()
    }

    /// Returns true if all elements in the vector are positive.
    #[inline(always)]
    fn is_positive(&self) -> bool {
        // true if all elements are positive
        Mask::<R>(R::is_positive(self.0)).all()
    }
}

impl<R: NumericRegister> ConstZero for Vector<R> {
    const ZERO: Self = Self(R::ZERO);
}

impl<R: NumericRegister> ConstOne for Vector<R> {
    const ONE: Self = Self(R::ONE);
}

#[rustfmt::skip]
impl<R: FloatRegister> num_traits::FloatConst for Vector<R> {
    #[inline(always)] fn E() -> Self                { const { Self::splat_const(FloatConsts::E) } }
    #[inline(always)] fn FRAC_1_PI() -> Self        { const { Self::splat_const(FloatConsts::FRAC_1_PI) } }
    #[inline(always)] fn FRAC_1_SQRT_2() -> Self    { const { Self::splat_const(FloatConsts::FRAC_1_SQRT_2) } }
    #[inline(always)] fn FRAC_2_PI() -> Self        { const { Self::splat_const(FloatConsts::FRAC_2_PI) } }
    #[inline(always)] fn FRAC_2_SQRT_PI() -> Self   { const { Self::splat_const(FloatConsts::FRAC_2_SQRT_PI) } }
    #[inline(always)] fn FRAC_PI_2() -> Self        { const { Self::splat_const(FloatConsts::FRAC_PI_2) } }
    #[inline(always)] fn FRAC_PI_3() -> Self        { const { Self::splat_const(FloatConsts::FRAC_PI_3) } }
    #[inline(always)] fn FRAC_PI_4() -> Self        { const { Self::splat_const(FloatConsts::FRAC_PI_4) } }
    #[inline(always)] fn FRAC_PI_6() -> Self        { const { Self::splat_const(FloatConsts::FRAC_PI_6) } }
    #[inline(always)] fn FRAC_PI_8() -> Self        { const { Self::splat_const(FloatConsts::FRAC_PI_8) } }
    #[inline(always)] fn LN_10() -> Self            { const { Self::splat_const(FloatConsts::LN_10) } }
    #[inline(always)] fn LN_2() -> Self             { const { Self::splat_const(FloatConsts::LN_2) } }
    #[inline(always)] fn LOG10_E() -> Self          { const { Self::splat_const(FloatConsts::LOG10_E) } }
    #[inline(always)] fn LOG2_E() -> Self           { const { Self::splat_const(FloatConsts::LOG2_E) } }
    #[inline(always)] fn PI() -> Self               { const { Self::splat_const(FloatConsts::PI) } }
    #[inline(always)] fn SQRT_2() -> Self           { const { Self::splat_const(FloatConsts::SQRT_2) } }

    // the bounds on these three are dumb
    #[inline(always)]
    fn TAU() -> Self where Self: Sized + Add<Self, Output = Self> { const { Self::splat_const(FloatConsts::TAU) } }

    #[inline(always)]
    fn LOG10_2() -> Self where Self: Sized + Div<Self, Output = Self> { const { Self::splat_const(FloatConsts::LOG10_2) } }

    #[inline(always)]
    fn LOG2_10() -> Self where Self: Sized + Div<Self, Output = Self> { const { Self::splat_const(FloatConsts::LOG2_10) } }
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
    /// Returns true if **all** elements in the vector are zero.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ZERO)).all()
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
    /// Returns true if **all** elements in the vector are one.
    #[inline(always)]
    fn is_one(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ONE)).all()
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

impl<R: IntegerRegister> WrappingAdd for Vector<R> {
    #[inline(always)]
    fn wrapping_add(&self, v: &Self) -> Self {
        Self(R::add(self.0, v.0))
    }
}

impl<R: IntegerRegister> WrappingSub for Vector<R> {
    #[inline(always)]
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self(R::sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> WrappingMul for Vector<R> {
    #[inline(always)]
    fn wrapping_mul(&self, v: &Self) -> Self {
        Self(R::mul(self.0, v.0))
    }
}

impl<R: IntegerRegister> Vector<R> {
    /// Wrapping addition for each element of the vectors.
    #[inline(always)]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self(R::add(self.0, rhs.0))
    }

    /// Wrapping subtraction for each element of the vectors.
    #[inline(always)]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self(R::sub(self.0, rhs.0))
    }

    /// Wrapping multiplication for each element of the vectors.
    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self(R::mul(self.0, rhs.0))
    }

    /// Try to use this vector as the denominators for a vectorized division operation.
    ///
    /// This creates a `VectorDivider` which can then be used to perform
    /// vectorized integer division with the `Div` trait. If any of the
    /// denominators are unsupported (such as `1` for unsigned integers),
    /// an error is returned.
    ///
    /// This operation itself is NOT vectorized and is `O(n)` in the number of lanes.
    /// It is designed to be calculated once and then reused for multiple division operations.
    #[inline(always)]
    pub fn try_to_divider(self) -> Result<VectorDivider<R>, UnsupportedDivisor>
    where
        R::Element: Denominator,
    {
        VectorDivider::try_new(self)
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
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c $d>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident $d:ident]),*) => {
        /// Only available for 4-lane vectors, this allows human-readable swizzle/permutations
        /// of the vector.
        pub trait Swizzle4 { $(impl_swizzle4!(DECL $(#[$meta])* $a $b $c $d);)* }

        /// Implements 4-lane swizzling for vectors.
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
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident]),*) => {
        /// Only available for "3-lane" (ignoring 4th lane) [`LinAlg3Register`] vectors,
        /// this allows human-readable swizzle/permutations of the vector. Permutations
        /// will ignore the 4th lane of the register, leaving it unchanged.
        pub trait Swizzle3 { $(impl_swizzle3!(DECL $(#[$meta])* $a $b $c);)* }

        /// Implements 3-lane swizzling for vectors support 3-lane linear algebra operations.
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

/// Implements conversion from `Vector<R>` to primitive types by extracting
/// the first lane and converting that. Other lanes are ignored.
#[rustfmt::skip]
impl<R: Register> num_traits::ToPrimitive for Vector<R>
where
    R::Element: num_traits::ToPrimitive,
{
    #[inline(always)] fn to_isize(&self) -> Option<isize> { self.extract::<0>().to_isize() }
    #[inline(always)] fn to_i8(&self) -> Option<i8> { self.extract::<0>().to_i8() }
    #[inline(always)] fn to_i16(&self) -> Option<i16> { self.extract::<0>().to_i16() }
    #[inline(always)] fn to_i32(&self) -> Option<i32> { self.extract::<0>().to_i32() }
    #[inline(always)] fn to_i128(&self) -> Option<i128> { self.extract::<0>().to_i128() }
    #[inline(always)] fn to_usize(&self) -> Option<usize> { self.extract::<0>().to_usize() }
    #[inline(always)] fn to_u8(&self) -> Option<u8> { self.extract::<0>().to_u8() }
    #[inline(always)] fn to_u16(&self) -> Option<u16> { self.extract::<0>().to_u16() }
    #[inline(always)] fn to_u32(&self) -> Option<u32> { self.extract::<0>().to_u32() }
    #[inline(always)] fn to_u128(&self) -> Option<u128> { self.extract::<0>().to_u128() }
    #[inline(always)] fn to_f32(&self) -> Option<f32> { self.extract::<0>().to_f32() }
    #[inline(always)] fn to_f64(&self) -> Option<f64> { self.extract::<0>().to_f64() }
    #[inline(always)] fn to_i64(&self) -> Option<i64> { self.extract::<0>().to_i64() }
    #[inline(always)] fn to_u64(&self) -> Option<u64> { self.extract::<0>().to_u64() }
}

impl<R: Register> num_traits::NumCast for Vector<R>
where
    R::Element: num_traits::NumCast,
{
    #[inline(always)]
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::splat(num_traits::NumCast::from(n)?))
    }
}

#[cfg(feature = "partial-ord")]
impl<R: PartialOrdRegister> PartialOrd for Vector<R> {
    /// Partial comparison between two vectors, returning `None` if
    /// the vectors are not fully ordered. Only returns `Some(Ordering)` if
    /// all lanes are less than, greater than, or equal.
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        let is_less = R::all(R::lt(self.0, other.0));
        let is_greater = R::all(R::gt(self.0, other.0));
        let is_equal = R::all(R::eq(self.0, other.0));

        match (is_less, is_greater, is_equal) {
            (true, false, false) => Some(core::cmp::Ordering::Less),
            (false, true, false) => Some(core::cmp::Ordering::Greater),
            (false, false, true) => Some(core::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

macro_rules! impl_unsigned_pow {
    ($($t:ty),* $(,)?) => {$(
        impl<R: NumericRegister> num_traits::Pow<$t> for Vector<R> {
            type Output = Self;

            #[inline(always)]
            fn pow(self, mut e: $t) -> Self::Output {
                let mut res = Self::ONE;
                let mut x = self;

                while e != 0 {
                    if e & 1 != 0 {
                        res *= x;
                    }

                    x *= x;
                    e >>= 1;
                }

                res
            }
        })*
    };
}

impl_unsigned_pow!(u8, u16, u32, u64, usize);

#[cfg(feature = "rand")]
const _: () = {
    use generic_array::sequence::GenericSequence;
    use rand::{Fill, Rng, distr::Distribution};

    impl<R: Register> Distribution<Vector<R>> for rand::distr::Uniform<R::Element>
    where
        rand::distr::Uniform<R::Element>: Distribution<R::Element>,
        R::Element: rand::distr::uniform::SampleUniform,
    {
        #[inline(always)]
        fn sample<Rng: rand::Rng + ?Sized>(&self, rng: &mut Rng) -> Vector<R> {
            Vector::from_array(GenericArray::generate(|_| self.sample(rng)))
        }
    }

    macro_rules! impl_distr {
        ($($distr:ident),* $(,)?) => {$(
            impl<R: Register> Distribution<Vector<R>> for rand::distr::$distr
            where
                rand::distr::$distr: Distribution<R::Element>,
            {
                #[inline(always)]
                fn sample<Rng: rand::Rng + ?Sized>(&self, rng: &mut Rng) -> Vector<R> {
                    Vector::from_array(GenericArray::generate(|_| self.sample(rng)))
                }
            }
        )*};
    }

    impl_distr!(Open01, OpenClosed01, StandardUniform);

    impl<R: Register> Fill for Vector<R>
    where
        [R::Element]: Fill,
    {
        #[inline(always)]
        fn fill<Rng: rand::Rng + ?Sized>(&mut self, rng: &mut Rng) {
            Fill::fill(self.as_mut_slice(), rng);
        }
    }
};
