#![warn(missing_docs, clippy::missing_safety_doc)]

//! SIMD Mask Vector Type and Operations.
//!
//! These mask vectors are used to represent boolean values in SIMD operations, where each lane
//! of the vector corresponds to a boolean value (true or false). The underlying representation
//! uses all bits set to '1' for `true` and all bits set to '0' for `false`.

use crate::{
    Vector,
    register::{BitwiseRegister, CastMaskRegister, Lanes, MaskRegister, NumericRegister, Register, Storage},
};

/// SIMD Mask Vector, where each lane is a boolean value represented by
/// all '1's' or '0's' bits in the underlying register.
///
/// This is a wrapper around the underlying mask register type. It provides a way to create and manipulate masks
/// for SIMD operations. See [`new`](Mask::new), [`splat`](Mask::splat),
/// and [`From<bool>/From<Vector<R>>`](Mask::from) for creating masks from values.
///
/// Masks are created by certain operations on vectors, such as comparisons, and can be used
/// to select elements from vectors based on the mask values.
#[repr(transparent)]
pub struct Mask<R: Register>(#[doc(hidden)] pub Storage<R::Mask>);

impl<R: Register> Clone for Mask<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Mask<R> {}

const _: () = {
    use core::fmt;

    impl<R: Register> fmt::Debug for Mask<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut t = f.debug_tuple("Mask");

            // TODO: Check if this needs to be reversed?
            for v in self.bitmask() {
                t.field(&v);
            }

            t.finish()
        }
    }
};

use generic_array::{GenericArray, typenum::Unsigned};

impl<R: Register> const_default::ConstDefault for Mask<R> {
    const DEFAULT: Self = Self::FALSY;
}

impl<R: Register> Default for Mask<R> {
    #[inline(always)]
    fn default() -> Self {
        Self::FALSY
    }
}

impl<R: Register> Mask<R> {
    /// The number of lanes in the mask register.
    pub const LANES: usize = <R::Lanes as Unsigned>::USIZE;

    /// A mask with all bits set to `true`.
    pub const TRUTHY: Mask<R> = Mask(<R::Mask as MaskRegister>::TRUTHY);

    /// A mask with all bits set to `false`.
    pub const FALSY: Mask<R> = Mask(<R::Mask as MaskRegister>::FALSY);

    /// Create a new mask from an array of `bool` values.
    ///
    /// This is not zero-cost, but is sufficiently efficient on modern platforms.
    #[inline(always)]
    pub fn new(values: impl Into<GenericArray<bool, R::Lanes>>) -> Self {
        Self(R::Mask::new_mask(values.into()))
    }

    /// Create a mask with all bits set to the same boolean value.
    #[inline(always)]
    pub const fn splat(value: bool) -> Self {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    /// Cast this mask to another mask type.
    #[inline(always)]
    pub fn cast<INTO: Register<Mask: CastMaskRegister<R::Mask>>>(self) -> Mask<INTO> {
        Mask(<INTO::Mask as CastMaskRegister<R::Mask>>::mask_from(self.0))
    }

    /// Create a mask by casting from another mask type.
    #[inline(always)]
    pub fn from_mask<FROM: Register>(mask: Mask<FROM>) -> Mask<R>
    where
        R::Mask: CastMaskRegister<FROM::Mask>,
    {
        Mask(<R::Mask as CastMaskRegister<FROM::Mask>>::mask_from(mask.0))
    }

    /// Returns a bitmask representation of the mask as a native integer type, if supported.
    ///
    /// The bitmask will have one bit per lane in the register, with the least significant bit
    /// corresponding to lane 0.
    ///
    /// If the register does not support native bitmask extraction, or it exceeds 64 lanes,
    /// this will return `None`.
    #[inline(always)]
    pub fn native_bitmask(&self) -> Option<u64> {
        <R::Mask as MaskRegister>::native_bitmask(self.0)
    }

    /// Returns a bitmask representation of the mask as a [`GenericBitArray`](generic_array::GenericBitArray).
    ///
    /// The length of this bitmask is determined by the number of lanes in the register, as
    /// `ceil(LANES / 32)`, but with typenum's type-level integers. `u32` was chosen as the storage
    /// type for better compatibility with the actual SIMD operations on most platforms.
    #[inline(always)]
    pub fn bitmask(&self) -> bitvec::array::BitArray<<R::Lanes as Lanes>::BitmaskStorage> {
        <R::Mask as MaskRegister>::bitmask(self.0)
    }

    /// Returns `true` if **all** bits in the mask are `true`.
    #[inline(always)]
    pub fn all(self) -> bool {
        <R::Mask as MaskRegister>::all(self.0)
    }

    /// Returns `true` if **any** bit in the mask is `true`.
    #[inline(always)]
    pub fn any(self) -> bool {
        <R::Mask as MaskRegister>::any(self.0)
    }

    /// Returns `true` if **none** of the bits in the mask are `true` (i.e. all bits are `false`).
    #[inline(always)]
    pub fn none(self) -> bool {
        <R::Mask as MaskRegister>::none(self.0)
    }

    /// For each lane in mask, if the lane is `true`, the corresponding lane in `truthy` is selected,
    /// otherwise the corresponding lane in `falsy` is selected.
    #[inline(always)]
    pub fn select<T, S>(self, truthy: T, falsy: T) -> T
    where
        T: Selectable<S>,
        S: Register<Mask: CastMaskRegister<R::Mask, Lanes = R::Lanes>>,
    {
        T::select(self, truthy, falsy)
    }

    /// For each lane of the mask, if the lane is `true`, swap the corresponding lanes in `a` and `b`.
    #[inline(always)]
    pub fn swap<S>(self, a: &mut Vector<S>, b: &mut Vector<S>)
    where
        S: Register<Mask: CastMaskRegister<R::Mask, Lanes = R::Lanes>>,
    {
        let mask = <S::Mask as CastMaskRegister<R::Mask>>::mask_from(self.0);

        let a2 = S::blendv(mask, a.0, b.0);
        let b2 = S::blendv(mask, b.0, a.0);

        a.0 = a2;
        b.0 = b2;
    }
}

/// Trait for types that support selection based on a mask.
pub trait Selectable<R: Register> {
    /// For each lane in `mask`, if the lane is `true`, the corresponding lane in `truthy` is selected,
    /// otherwise the corresponding lane in `falsy` is selected.
    fn select<M: Register>(mask: Mask<M>, truthy: Self, falsy: Self) -> Self
    where
        R::Mask: CastMaskRegister<M::Mask, Lanes = M::Lanes>;
}

impl<R: Register> Selectable<R> for Vector<R> {
    #[inline(always)]
    fn select<M: Register>(mask: Mask<M>, truthy: Self, falsy: Self) -> Self
    where
        R::Mask: CastMaskRegister<M::Mask, Lanes = M::Lanes>,
    {
        Vector(R::blendv(
            <R::Mask as CastMaskRegister<M::Mask>>::mask_from(mask.0),
            falsy.0,
            truthy.0,
        ))
    }
}

impl<R: Register> From<bool> for Mask<R> {
    /// Sets all bits in the mask to `true` if `value` is `true`, and all bits to `false` if `value` is `false`.
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(<R::Mask as MaskRegister>::boolean(value))
    }
}

impl<R: Register> From<Vector<R>> for Mask<R>
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

use crate::generic::ops::{BitAndNot, BitAndNotAssign};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

impl<R: Register> BitAnd for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(<R::Mask as BitwiseRegister>::bitand(self.0, rhs.0))
    }
}

impl<R: Register> BitAndAssign for Mask<R> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = <R::Mask as BitwiseRegister>::bitand(self.0, rhs.0);
    }
}

impl<R: Register> BitAndNot for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitandnot(self, rhs: Self) -> Self::Output {
        // exposed logic is reversed from register operation
        Self(<R::Mask as BitwiseRegister>::bitandnot(rhs.0, self.0))
    }
}

impl<R: Register> BitAndNotAssign for Mask<R> {
    #[inline(always)]
    fn bitandnot_assign(&mut self, rhs: Self) {
        // exposed logic is reversed from register operation
        self.0 = <R::Mask as BitwiseRegister>::bitandnot(rhs.0, self.0);
    }
}

impl<R: Register> BitOr for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(<R::Mask as BitwiseRegister>::bitor(self.0, rhs.0))
    }
}

impl<R: Register> BitOrAssign for Mask<R> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = <R::Mask as BitwiseRegister>::bitor(self.0, rhs.0);
    }
}

impl<R: Register> BitXor for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(<R::Mask as BitwiseRegister>::bitxor(self.0, rhs.0))
    }
}

impl<R: Register> BitXorAssign for Mask<R> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = <R::Mask as BitwiseRegister>::bitxor(self.0, rhs.0);
    }
}

impl<R: Register> Not for Mask<R> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(<R::Mask as BitwiseRegister>::not(self.0))
    }
}
