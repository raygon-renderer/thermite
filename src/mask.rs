use crate::*;

use std::fmt;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask<S: Simd, V>(V, PhantomData<S>);

/// Small integer representation of a mask using individual bits
#[repr(transparent)]
pub struct BitMask<S: Simd, V> {
    mask: u16,
    vec: PhantomData<(S, V)>,
}

impl<S: Simd, V> Clone for BitMask<S, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Simd, V> Copy for BitMask<S, V> {}

impl<S: Simd, V> PartialEq<Self> for BitMask<S, V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.mask == other.mask
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        self.mask != other.mask
    }
}

impl<S: Simd, V> Eq for BitMask<S, V> {}

impl<S: Simd, V> BitMask<S, V>
where
    V: SimdBitwise<S>,
{
    /// Create a bitmask for this vector type of all true bits
    #[inline(always)]
    pub fn truthy() -> Self {
        Self {
            mask: <V as SimdBitwise<S>>::FULL_BITMASK,
            vec: PhantomData,
        }
    }

    /// Create a bitmask with all false bits
    #[inline(always)]
    pub fn falsey() -> Self {
        Self {
            mask: 0,
            vec: PhantomData,
        }
    }

    #[inline(always)]
    pub fn raw(self) -> u16 {
        self.mask
    }

    /// Return true if all lanes for this vector type are set to true
    #[inline(always)]
    pub fn all(self) -> bool {
        self.mask == <V as SimdBitwise<S>>::FULL_BITMASK
    }

    /// Return true if any lane for this vector type is set to true
    #[inline(always)]
    pub fn any(self) -> bool {
        self.mask != 0
    }

    /// Return true if no lanes for this vector type are set to true
    #[inline(always)]
    pub fn none(self) -> bool {
        !self.any()
    }

    /// Count the number of set bits
    #[inline(always)]
    pub fn count(self) -> u32 {
        self.mask.count_ones()
    }
}

impl<S: Simd, V> Debug for BitMask<S, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitmask({:b})", self.mask)
    }
}

impl<S: Simd, V> Not for BitMask<S, V>
where
    V: SimdBitwise<S>,
{
    type Output = Self;

    #[inline(always)]
    fn not(mut self) -> Self {
        self.mask = (!self.mask) & <V as SimdBitwise<S>>::FULL_BITMASK;
        self
    }
}

macro_rules! impl_bitmask_ops {
    (@BINARY $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl<S: Simd, V> $op_trait<Self> for BitMask<S, V> {
            type Output = Self;
            #[inline(always)] fn $op(mut self, rhs: Self) -> Self {
                self.mask = $op_trait::$op(self.mask, rhs.mask);
                self
            }
        }
        impl<S: Simd, V> [<$op_trait Assign>]<Self> for BitMask<S, V> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) {
                self.mask = $op_trait::$op(self.mask, rhs.mask);
            }
        }
    )*}}
}

impl_bitmask_ops!(@BINARY BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl<S: Simd, V> Debug for Mask<S, V>
where
    V: SimdVectorBase<S>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut t = f.debug_tuple("Mask");
        for i in 0..Self::NUM_ELEMENTS {
            t.field(unsafe { &self.extract_unchecked(i) });
        }
        t.finish()
    }
}

impl<S: Simd + ?Sized, V> Default for Mask<S, V>
where
    V: SimdVectorBase<S>,
{
    #[inline(always)]
    fn default() -> Self {
        Self::falsey()
    }
}

impl<S: Simd + ?Sized, V> Mask<S, V> {
    #[inline(always)]
    pub(crate) fn new(value: V) -> Self {
        Self(value, PhantomData)
    }

    #[inline(always)]
    pub fn value(self) -> V {
        self.0
    }
}

impl<S: Simd + ?Sized, V> Mask<S, V>
where
    V: SimdVectorBase<S>,
{
    /// Mask vector containing all true/non-zero lanes.
    #[inline(always)]
    pub fn truthy() -> Self {
        Self::new(V::splat(Truthy::truthy()))
    }

    /// Mask vector containing all zero/false lanes.
    #[inline(always)]
    pub fn falsey() -> Self {
        Self::new(V::splat(Truthy::falsey()))
    }
}

impl<S: Simd + ?Sized, V> Mask<S, V>
where
    V: SimdVector<S>,
{
    /// Creates a mask from a value. Any non-zero lanes are considered truthy.
    #[inline(always)]
    pub fn from_value(v: V) -> Self {
        v.ne(V::zero())
    }

    #[inline(always)]
    pub fn cast_to<U: SimdCastFrom<S, V>>(self) -> Mask<S, U> {
        U::from_cast_mask(self)
    }

    /// Bitmask indicating all truthy values for each lane
    pub const FULL_BITMASK: BitMask<S, V> = BitMask {
        mask: V::FULL_BITMASK,
        vec: PhantomData,
    };

    /// Computes `!self & other` for each lane of the mask.
    #[inline(always)]
    pub fn and_not(self, other: Self) -> Self {
        Self::new(self.0.and_not(other.0))
    }

    /// Returns a bitmask that can be trivially evaluated to all/any/none/etc.
    ///
    /// The can be useful for reusing a mask value for branching
    #[inline(always)]
    pub fn bitmask(self) -> BitMask<S, V> {
        BitMask {
            mask: self.0.bitmask(),
            vec: PhantomData,
        }
    }

    /// Returns `true` if all lanes are truthy
    ///
    /// **NOTE**: If you wish to use the same mask for multiply calls to all/any/none,
    /// consider using `bitmask()` instead to precompute the bitmask and reuse its methods.
    #[inline(always)]
    pub fn all(self) -> bool {
        unsafe { self.value()._mm_all() }
    }

    /// Returns `true` if any lanes are truthy
    ///
    /// **NOTE**: If you wish to use the same mask for multiply calls to all/any/none,
    /// consider using `bitmask()` instead to precompute the bitmask and reuse its methods.
    #[inline(always)]
    pub fn any(self) -> bool {
        unsafe { self.value()._mm_any() }
    }

    /// Returns `true` if all lanes are falsey
    ///
    /// **NOTE**: If you wish to use the same mask for multiply calls to all/any/none,
    /// consider using `bitmask()` instead to precompute the bitmask and reuse its methods.
    #[inline(always)]
    pub fn none(self) -> bool {
        unsafe { self.value()._mm_none() }
    }

    /// Counts the number of truthy lanes
    #[inline(always)]
    pub fn count(self) -> u32 {
        self.bitmask().count()
    }

    /// For each lane, selects from `t` if the mask lane is truthy, or `f` is falsey
    #[inline(always)]
    pub fn select<U>(self, t: U, f: U) -> U
    where
        V: SimdCastTo<S, U>,
        U: SimdVector<S>,
    {
        unsafe { V::cast_mask(self).value()._mm_blendv(t, f) }
    }
}

impl<S: Simd + ?Sized, V> SimdVectorBase<S> for Mask<S, V>
where
    V: SimdVectorBase<S>,
{
    type Element = bool;

    #[inline(always)]
    fn splat(value: bool) -> Self {
        Self::new(V::splat(Truthy::from_bool(value)))
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        let mut mask = V::default();
        for i in 0..Self::NUM_ELEMENTS {
            mask = mask.replace_unchecked(i, Truthy::from_bool(*ptr.add(i)));
        }
        Mask::new(mask)
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        for i in 0..Self::NUM_ELEMENTS {
            *ptr.add(i) = self.extract_unchecked(i);
        }
    }

    // The aligned versions of these two are scalar, so just call those

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        Self::load_aligned_unchecked(src)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        self.store_aligned_unchecked(dst);
    }

    #[inline(always)]
    unsafe fn extract_unchecked(self, index: usize) -> bool {
        self.value().extract_unchecked(index) != Truthy::falsey()
    }

    #[inline(always)]
    unsafe fn replace_unchecked(self, index: usize, value: bool) -> Self {
        Self::new(self.value().replace_unchecked(index, Truthy::from_bool(value)))
    }
}

macro_rules! impl_ops {
    (@UNARY => $($op_trait:ident::$op:ident),* ) => {$(
        impl<S: Simd, V> $op_trait for Mask<S, V> where V: SimdVector<S> {
            type Output = Self;
            #[inline(always)] fn $op(self) -> Self { Self::new($op_trait::$op(self.0)) }
        }
    )*};
    (@BINARY => $($op_trait:ident::$op:ident),* ) => {paste::paste! {$(
        impl<S: Simd, V> $op_trait<Self> for Mask<S, V> where V: SimdVector<S> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: Self) -> Self { Self::new($op_trait::$op(self.0, rhs.0)) }
        }
        impl<S: Simd, V> [<$op_trait Assign>]<Self> for Mask<S, V> where V: SimdVector<S> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) {
                self.0 = $op_trait::$op(self.0, rhs.0);
            }
        }
    )*}};
    (@SHIFTS => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl<S: Simd, V, U> $op_trait<U> for Mask<S, V> where V: SimdVector<S> + $op_trait<U, Output = V> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: U) -> Self { $op_trait::$op(self.0, rhs).ne(V::splat(Truthy::falsey())) }
        }
        impl<S: Simd, V, U> [<$op_trait Assign>]<U> for Mask<S, V> where V: SimdVector<S> + [<$op_trait Assign>]<U> + $op_trait<U, Output = V> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: U) {
                self.0 = $op_trait::$op(self.0, rhs).ne(V::splat(Truthy::falsey())).0;
            }
        }
    )*}};
}

impl_ops!(@UNARY => Not::not);
impl_ops!(@BINARY => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

pub trait Truthy: Sized + PartialEq {
    fn truthy() -> Self;
    fn falsey() -> Self;

    #[rustfmt::skip]
    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        if value { Self::truthy() } else { Self::falsey() }
    }
}

macro_rules! decl_truthy {
    (@INT => $($ty:ty),*) => {$(
        impl Truthy for $ty {
            #[inline(always)] fn truthy() -> $ty { !0 }
            #[inline(always)] fn falsey() -> $ty {  0 }
        }
    )*};
    (@FLOAT => $($ty:ty),*) => {$(
        impl Truthy for $ty {
            #[inline(always)] fn truthy() -> $ty { <$ty>::from_bits(!0) }
            #[inline(always)] fn falsey() -> $ty { <$ty>::from_bits( 0) }
        }
    )*};
}

decl_truthy!(@INT => i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
decl_truthy!(@FLOAT => f32, f64);

#[rustfmt::skip]
impl Truthy for bool {
    #[inline(always)] fn truthy() -> Self { true }
    #[inline(always)] fn falsey() -> Self { false }
}
