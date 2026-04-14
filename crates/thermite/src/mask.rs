// #![warn(missing_docs, clippy::missing_safety_doc)]

//! SIMD Mask Vector Type and Operations.
//!
//! These mask vectors are used to represent boolean values in SIMD operations, where each lane
//! of the vector corresponds to a boolean value (true or false). The underlying representation
//! uses all bits set to '1' for `true` and all bits set to '0' for `false`.

use crate::{
    Vector,
    register::{
        BitwiseRegister, CastMaskRegister, InterleaveRegister, Lanes, MaskRegister, NumericRegister, Register, Storage,
    },
    vector::Interleave,
};

pub trait CastMask<FROM>: Sized {
    fn mask_from(from: FROM) -> Self;
}

pub trait GenericSelectable: Copy {
    type SelectableMask: Copy;

    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>;
}

#[rustfmt::skip]
pub trait GenericMask: 'static + Sized + Copy + Default + core::fmt::Debug
    + CastMask<Self>
    + BitAnd<Self, Output = Self> + BitAndAssign<Self>
    + crate::vector::ops::BitAndNot<Self, Output = Self>
    + crate::vector::ops::BitAndNotAssign<Self>
    + BitOr<Self, Output = Self> + BitOrAssign<Self>
    + BitXor<Self, Output = Self> + BitXorAssign<Self>
    + Not<Output = Self>
    + Interleave
{
    const TRUTHY: Self;
    const FALSY: Self;

    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;

    fn native_bitmask(&self) -> Option<u64>;

    #[cfg(feature = "bitvec")]
    fn bitmask(&self) -> bitvec::array::BitArray<impl bitvec::view::BitViewSized<Store = u32>>;

    #[inline(always)]
    fn select<S>(self, t: S, f: S) -> S
    where
        S: GenericSelectable<SelectableMask: CastMask<Self>>,
    {
        S::select(self, t, f)
    }

    #[inline(always)]
    fn cast<INTO>(self) -> INTO
    where
        INTO: CastMask<Self>,
    {
        INTO::mask_from(self)
    }

    #[inline(always)]
    fn swap<S>(self, a: &mut S, b: &mut S)
    where
        S: GenericSelectable<SelectableMask: CastMask<Self> + CastMask<S::SelectableMask>>,
    {
        let mask = S::SelectableMask::mask_from(self);

        let a2 = S::select(mask, *a, *b);
        let b2 = S::select(mask, *b, *a);

        *a = a2;
        *b = b2;
    }

    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self;
}

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

            #[cfg(feature = "bitvec")]
            for v in self.bitmask()[..R::Lanes::USIZE].iter() {
                t.field(&*v);
            }

            #[cfg(not(feature = "bitvec"))]
            {
                let Some(bitmask) = self.native_bitmask() else {
                    return t.field(&"<non-bitmaskable>").finish();
                };

                for i in 0..R::Lanes::USIZE {
                    let bit = (bitmask >> i) & 1 != 0;
                    t.field(&bit);
                }
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

impl<FROM, INTO> CastMask<Mask<FROM>> for Mask<INTO>
where
    FROM: Register,
    INTO: Register<Mask: CastMaskRegister<FROM::Mask>>,
{
    #[inline(always)]
    fn mask_from(from: Mask<FROM>) -> Self {
        Mask(<INTO::Mask as CastMaskRegister<FROM::Mask>>::mask_from(from.0))
    }
}

impl<R: Register> Interleave for Mask<R> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (a, b) = <R::Mask as InterleaveRegister>::interleave(self.0, other.0);
        (Mask(a), Mask(b))
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (a, b) = <R::Mask as InterleaveRegister>::deinterleave(self.0, other.0);
        (Mask(a), Mask(b))
    }
}

impl<R: Register> GenericMask for Mask<R> {
    const FALSY: Self = Mask(<R::Mask as MaskRegister>::FALSY);
    const TRUTHY: Self = Mask(<R::Mask as MaskRegister>::TRUTHY);

    #[inline(always)]
    fn all(self) -> bool {
        <R::Mask as MaskRegister>::all(self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        <R::Mask as MaskRegister>::any(self.0)
    }

    #[inline(always)]
    fn none(self) -> bool {
        <R::Mask as MaskRegister>::none(self.0)
    }

    #[inline(always)]
    fn native_bitmask(&self) -> Option<u64> {
        <R::Mask as MaskRegister>::native_bitmask(self.0)
    }

    #[cfg(feature = "bitvec")]
    #[inline(always)]
    fn bitmask(&self) -> bitvec::array::BitArray<impl bitvec::view::BitViewSized<Store = u32>> {
        <R::Mask as MaskRegister>::bitmask(self.0)
    }

    #[inline(always)]
    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        Mask(<R::Mask as BitwiseRegister>::ternlog::<IMM>(a.0, b.0, c.0))
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

use crate::vector::ops::{BitAndNot, BitAndNotAssign};
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
