#![warn(missing_docs, clippy::missing_safety_doc)]

use crate::{
    Vector,
    register::{
        BitsRegister, CastMaskRegister, CastRegister, FloatRegister, IntegerRegister, LinAlg3Register, MaskElement,
        MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShiftRegister, ShuffleRegister,
        SignedRegister, SwizzleRegister, UnsignedIntegerRegister,
    },
};

/// SIMD Mask Vector, where each lane is a boolean value represented by
/// all '1's' or '0's' bits in the underlying register.
///
/// This is a wrapper around the underlying mask register type. It provides a way to create and manipulate masks
/// for SIMD operations. See [`new`](Mask::new), [`new_unchecked`](Mask::new_unchecked),
/// and [`From<bool>/From<Vector<R>>`](Mask::from) for creating masks from values.
#[repr(transparent)]
pub struct Mask<R: MaskRegister>(pub(crate) R::Storage);

impl<R: MaskRegister> Clone for Mask<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: MaskRegister> Copy for Mask<R> {}

const _: () = {
    use core::fmt;

    impl<R: MaskRegister> fmt::Debug for Mask<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut t = f.debug_tuple("Mask");

            for v in R::debug_iter_bool(&self.0) {
                t.field(&v);
            }

            t.finish()
        }
    }
};

use generic_array::{GenericArray, typenum::Unsigned};

#[cfg(feature = "const-default")]
impl<R: MaskRegister> const_default::ConstDefault for Mask<R> {
    const DEFAULT: Self = Self::FALSY;
}

impl<R: MaskRegister> Mask<R> {
    /// The number of lanes in the mask register.
    pub const LANES: usize = <R::Lanes as Unsigned>::USIZE;

    /// A mask with all bits set to `true`.
    pub const TRUTHY: Mask<R> = Mask(R::TRUTHY);

    /// A mask with all bits set to `false`.
    pub const FALSY: Mask<R> = Mask(R::FALSY);

    /// Create a new mask from an array of `bool` values.
    ///
    /// This is not zero-cost, but is sufficiently efficient on modern platforms.
    #[inline(always)]
    pub fn new(values: impl Into<GenericArray<bool, R::Lanes>>) -> Self {
        Self(R::new_mask(values.into()))
    }

    #[inline(always)]
    pub fn splat(value: bool) -> Self {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    /// Create a mask from an array of values of the underlying element type. It's best to
    /// have every bit in truthy values be `1` and every bit in falsy values be `0`.
    #[inline(always)]
    pub fn new_unchecked(values: impl Into<GenericArray<R::Element, R::Lanes>>) -> Self {
        Self(R::new(values.into()))
    }

    /// Broadcast the value of a single lane across all lanes of the mask.
    #[inline(always)]
    pub fn broadcast<const I: usize>(self) -> Self {
        Self(R::broadcast::<I>(self.0))
    }

    /// Broadcast the value of a single lane across all lanes of the mask.
    ///
    /// # Panics
    /// If `idx` is out of bounds for the mask's lanes.
    #[inline(always)]
    pub fn broadcastv(self, idx: usize) -> Self {
        Self(R::broadcastv(self.0, idx))
    }

    #[inline(always)]
    pub fn insert<const LANE: usize>(mut self, value: bool) -> Self {
        Self(R::insert::<LANE>(self.0, MaskElement::from_bool(value)))
    }

    #[inline(always)]
    pub fn extract<const LANE: usize>(self) -> bool {
        MaskElement::to_bool(R::extract::<LANE>(self.0))
    }

    #[inline(always)]
    pub fn cast<INTO: CastMaskRegister<R>>(self) -> Mask<INTO> {
        Mask(INTO::mask_from(self.0))
    }

    #[inline(always)]
    pub fn from_mask<FROM: MaskRegister>(mask: Mask<FROM>) -> Mask<R>
    where
        R: CastMaskRegister<FROM>,
    {
        Mask(R::mask_from(mask.0))
    }

    #[inline(always)]
    pub fn reverse(self) -> Self {
        Self(R::reverse(self.0))
    }

    /// Create a mask from a vector of the underlying element type, without
    /// verifying the values.
    #[inline(always)]
    pub const fn from_unchecked(value: Vector<R>) -> Self {
        Self(value.0)
    }

    /// Returns a Vector with the same bits as the mask.
    #[inline(always)]
    pub const fn value(self) -> Vector<R> {
        Vector(self.0)
    }

    /// Returns !self & value
    pub fn andnot(self, value: Vector<R>) -> Vector<R>
    where
        R: NumericRegister,
    {
        Vector(R::bitandnot(self.0, value.0))
    }

    /// Returns `true` if **all** bits in the mask are `true`.
    #[inline(always)]
    pub fn all(self) -> bool {
        R::all(self.0)
    }

    /// Returns `true` if **any** bit in the mask is `true`.
    #[inline(always)]
    pub fn any(self) -> bool {
        R::any(self.0)
    }

    /// Returns `true` if **none** of the bits in the mask are `true` (i.e. all bits are `false`).
    #[inline(always)]
    pub fn none(self) -> bool {
        R::none(self.0)
    }

    /// For each lane in mask, if the lane is `true`, the corresponding lane in `truthy` is selected,
    /// otherwise the corresponding lane in `falsy` is selected.
    #[inline(always)]
    pub fn select<S>(self, truthy: Vector<S>, falsy: Vector<S>) -> Vector<S>
    where
        S: CastMaskRegister<R, Lanes = R::Lanes>,
    {
        Vector(S::blendv(S::mask_from(self.0), falsy.0, truthy.0))
    }

    /// For each lane in mask, if the lane is `true`, the corresponding lane in `truthy` is selected,
    /// otherwise the corresponding lane in `falsy` is selected.
    #[inline(always)]
    pub fn select_mask<M>(self, truthy: Mask<M>, falsy: Mask<M>) -> Mask<M>
    where
        M: CastMaskRegister<R, Lanes = R::Lanes>,
    {
        Mask(M::blendv(M::mask_from(self.0), falsy.0, truthy.0))
    }

    /// For each lane of the mask, if the lane is `true`, swap the corresponding lanes in `a` and `b`.
    #[inline(always)]
    pub fn swap<S>(self, a: &mut Vector<S>, b: &mut Vector<S>)
    where
        S: CastMaskRegister<R, Lanes = R::Lanes>,
    {
        let mask = S::mask_from(self.0);

        let a2 = S::blendv(mask, a.0, b.0);
        let b2 = S::blendv(mask, b.0, a.0);

        a.0 = a2;
        b.0 = b2;
    }
}

impl<R: MaskRegister> From<bool> for Mask<R> {
    /// Sets all bits in the mask to `true` if `value` is `true`, and all bits to `false` if `value` is `false`.
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(R::boolean(value))
    }
}

impl<R: MaskRegister> From<Vector<R>> for Mask<R>
where
    R: NumericRegister,
{
    /// Converts a vector of `R` into a mask. The mask will have all bits set to `true` if the corresponding
    /// element in the vector is non-zero, and `false` otherwise.
    #[inline(always)]
    fn from(value: Vector<R>) -> Self {
        Self(R::ne(value.0, R::ZERO))
    }
}

use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

impl<R: MaskRegister> BitAnd for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(R::bitand(self.0, rhs.0))
    }
}

impl<R: MaskRegister> BitAndAssign for Mask<R> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = R::bitand(self.0, rhs.0);
    }
}

impl<R: MaskRegister> BitOr for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(R::bitor(self.0, rhs.0))
    }
}

impl<R: MaskRegister> BitOrAssign for Mask<R> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = R::bitor(self.0, rhs.0);
    }
}

impl<R: MaskRegister> BitXor for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(R::bitxor(self.0, rhs.0))
    }
}

impl<R: MaskRegister> BitXorAssign for Mask<R> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = R::bitxor(self.0, rhs.0);
    }
}

impl<R: MaskRegister> Not for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(R::not(self.0))
    }
}
