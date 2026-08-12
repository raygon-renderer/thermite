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

/// Conversion between mask types of the same lane count but differing element
/// width or backing representation.
///
/// A mask produced from, say, an `f32x8` comparison and one from an `i32x8`
/// comparison are semantically the same eight booleans but may be different
/// concrete types. `CastMask` reinterprets one as the other so a mask computed
/// against one vector type can drive a [`select`](GenericMask::select) or
/// masked operation on another. Most users reach this through
/// [`GenericMask::cast`] rather than calling [`mask_from`](Self::mask_from)
/// directly.
pub trait CastMask<FROM>: Sized {
    /// Reinterpret the `from` mask as `Self`, preserving the per-lane boolean
    /// values.
    fn mask_from(from: FROM) -> Self;
}

/// Types whose lanes can be selected between by a mask.
///
/// Implemented by [`Vector`] (and any other lane-structured value). Given a
/// mask, [`select`](Self::select) chooses each lane from one of two candidates.
/// The associated [`SelectableMask`](Self::SelectableMask) names the mask type
/// natural to this value; any mask castable to it (via [`CastMask`]) can be
/// used, which is what lets a comparison on one element type select lanes of
/// another.
pub trait GenericSelectable: Copy {
    /// The mask type whose lane count and layout match `Self`.
    type SelectableMask: Copy;

    /// For each lane, take the value from `t` where `mask` is `true`, otherwise
    /// from `f`.
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>;
}

/// The boolean-vector trait: a per-lane mask supporting the logical and
/// selection operations the rest of the library is built on.
///
/// Every [`Mask`] implements this. A mask has the same lane count as the vector
/// it came from; each lane is either fully `true` or fully `false`. On
/// pre-AVX-512 backends the representation is a full-width vector (all-ones /
/// all-zeros per lane); on AVX-512 it is a dedicated `k` mask register.
///
/// Masks are produced by [`PartialOrdVector`](crate::vector::PartialOrdVector)
/// comparisons (and similar predicates), combined with the bitwise operators
/// (`&`, `|`, `^`, `!`, andnot), reduced with [`all`](Self::all) /
/// [`any`](Self::any) / [`none`](Self::none), and consumed by
/// [`select`](Self::select) and the `_c`/`_m`/`_z` masked operation variants.
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
    /// Number of lanes, matching the vector this mask came from.
    const LANES: usize;

    /// Number of lanes, as a runtime value.
    ///
    /// Prefer it over the constant in loop bounds and address arithmetic, for the
    /// same reason as [`GenericVector::lanes`](crate::vector::GenericVector::lanes):
    /// a scalable-vector backend can only report its lane count at runtime.
    #[inline(always)]
    fn lanes() -> usize {
        Self::LANES
    }

    /// Number of lanes, as a typenum.
    type Lanes: Lanes;

    /// A mask with every lane set to `true` (all bits set).
    const TRUTHY: Self;
    /// A mask with every lane set to `false` (all bits clear). This is also the
    /// [`Default`].
    const FALSY: Self;

    /// Returns `true` if every lane is `true`.
    fn all(self) -> bool;
    /// Returns `true` if at least one lane is `true`.
    fn any(self) -> bool;
    /// Returns `true` if every lane is `false`. Equivalent to `!self.any()`.
    fn none(self) -> bool;

    /// Index of the lowest lane set to `true`, or `None` if every lane is
    /// `false`.
    ///
    /// A SIMD find-first: combined with a comparison this is a vectorized
    /// `memchr`, e.g. `v.cmp_eq(Vector::splat(byte)).first_set()` gives the
    /// index of the first matching lane.
    fn first_set(self) -> Option<usize>;

    /// Index of the highest lane set to `true`, or `None` if every lane is
    /// `false` (a find-last).
    fn last_set(self) -> Option<usize>;

    /// Number of lanes set to `true` (population count of the mask).
    fn count_set(self) -> usize;

    /// [`first_set`](Self::first_set) over several masks at once, treating them
    /// as one concatenated mask (`masks[i]` occupying lanes
    /// `i * LANES .. (i + 1) * LANES`).
    fn first_set_many<const N: usize>(masks: [Self; N]) -> Option<usize> {
        let lanes = Self::LANES;

        let mut i = 0;
        while i < N {
            if let Some(idx) = masks[i].first_set() {
                return Some(i * lanes + idx);
            }
            i += 1;
        }

        None
    }

    /// [`last_set`](Self::last_set) over several masks at once; concatenation
    /// order is as described on [`first_set_many`](Self::first_set_many).
    fn last_set_many<const N: usize>(masks: [Self; N]) -> Option<usize> {
        let lanes = Self::LANES;

        let mut i = N;
        while i > 0 {
            i -= 1;
            if let Some(idx) = masks[i].last_set() {
                return Some(i * lanes + idx);
            }
        }

        None
    }

    /// Total lanes set to `true` across several masks at once.
    ///
    /// Prefer this over summing [`count_set`](Self::count_set) yourself: a
    /// population count cannot observe lane order, so backends merge the masks
    /// with a narrowing pack (x86) or a plain vector add (NEON) and reduce once,
    /// instead of extracting a bitmask and popcounting per mask. Counting a
    /// 64-byte block of 16 `i32` lanes on AVX2, for instance, is one `vpackssdw`
    /// + `vpmovmskb` + `popcnt` rather than two `vmovmskps` + `popcnt` pairs.
    fn count_set_many<const N: usize>(masks: [Self; N]) -> usize {
        let mut total = 0;

        let mut i = 0;
        while i < N {
            total += masks[i].count_set();
            i += 1;
        }

        total
    }

    /// Extract the mask as a packed integer bitmask, one bit per lane (lane 0 in
    /// the least-significant bit), if the backend can produce one directly.
    ///
    /// Returns `None` when there is no native bit-packing instruction for this
    /// mask representation (for example, very wide emulated masks, or backends
    /// where lanes are not movemask-extractable). For a representation-agnostic
    /// bitmask, enable the `bitvec` feature and use [`bitmask`](Self::bitmask).
    fn native_bitmask(&self) -> Option<u64>;

    /// Extract the mask as a [`bitvec`] bit array, one bit per lane.
    ///
    /// Unlike [`native_bitmask`](Self::native_bitmask) this always succeeds,
    /// falling back to a software pack when there is no native instruction.
    /// Only available with the `bitvec` feature.
    #[cfg(feature = "bitvec")]
    fn bitmask(&self) -> bitvec::array::BitArray<impl bitvec::view::BitViewSized<Store = u32>>;

    /// Build a mask from a packed integer bitmask, bit `i` driving lane `i`
    /// (lane 0 in the least-significant bit) - the inverse of
    /// [`native_bitmask`](Self::native_bitmask).
    ///
    /// Bits at or above [`LANES`](Self::LANES) are ignored. A mask wider than 64
    /// lanes takes its low 64 lanes from `bitmask` and leaves everything above
    /// lane 63 `false`; use [`from_bitmask`](Self::from_bitmask) for those.
    ///
    /// Lowers to a broadcast + AND + compare on most backends, a single
    /// `vpmovm2*` on AVX-512.
    fn from_native_bitmask(bitmask: u64) -> Self;

    /// Build a mask from a [`bitvec`] bit array, one bit per lane - the inverse
    /// of [`bitmask`](Self::bitmask), and unlike
    /// [`from_native_bitmask`](Self::from_native_bitmask) valid at any width.
    ///
    /// `bits` shorter than [`LANES`](Self::LANES) is allowed: the lanes it does
    /// not reach are `false`. Only available with the `bitvec` feature.
    #[cfg(feature = "bitvec")]
    fn from_bitmask(bits: &bitvec::slice::BitSlice<u32>) -> Self;

    /// Select between two lane-structured values: for each lane, take `t` where
    /// this mask is `true` and `f` where it is `false`.
    ///
    /// Typically lowers to a single blend instruction. The selectable type `S`
    /// may have a different element type, as long as its mask is
    /// [`CastMask`]-compatible with this one.
    #[inline(always)]
    fn select<S>(self, t: S, f: S) -> S
    where
        S: GenericSelectable<SelectableMask: CastMask<Self>>,
    {
        S::select(self, t, f)
    }

    /// Reinterpret this mask as another mask type of the same lane count.
    ///
    /// Convenience wrapper over [`CastMask::mask_from`].
    #[inline(always)]
    fn cast<INTO>(self) -> INTO
    where
        INTO: CastMask<Self>,
    {
        INTO::mask_from(self)
    }

    /// Conditionally swap corresponding lanes of `a` and `b`: lanes where this
    /// mask is `true` are exchanged, the rest are left in place.
    #[inline(always)]
    fn swap<S>(self, a: &mut S, b: &mut S)
    where
        S: GenericSelectable<SelectableMask: CastMask<Self> + CastMask<S::SelectableMask>>,
    {
        let mask = S::SelectableMask::mask_from(self);

        let a2 = S::select(mask, *b, *a);
        let b2 = S::select(mask, *a, *b);

        *a = a2;
        *b = b2;
    }

    /// Arbitrary 3-input bitwise function of masks `a`, `b`, `c` selected by the
    /// compile-time truth table `IMM`.
    ///
    /// The mask analogue of
    /// [`BitwiseVector::ternlog`](crate::vector::BitwiseVector::ternlog); see
    /// that method for how to compute `IMM`.
    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self;

    /// Whether [`ternlog`](Self::ternlog) is a single native instruction rather
    /// than the DNF polyfill, forwarded from the underlying mask register. See
    /// [`BitwiseVector::HAS_NATIVE_TERNLOG`](crate::vector::BitwiseVector::HAS_NATIVE_TERNLOG).
    const HAS_NATIVE_TERNLOG: bool;
}

/// SIMD Mask Vector, where each lane is a boolean value represented by
/// all '1's' or '0's' bits in the underlying register.
///
/// This is a `#[repr(transparent)]` wrapper around the underlying mask register
/// type ([`Storage<R::Mask>`](crate::register::Storage)), exposing the
/// [`GenericMask`] API for combining, reducing, and selecting with masks.
///
/// Masks are most often *produced* by predicate operations on vectors - e.g.
/// the [`PartialOrdVector`](crate::vector::PartialOrdVector) comparisons
/// (`cmp_lt`, `cmp_eq`, ...) - and then used to select elements via
/// [`select`](GenericMask::select) or to drive the `_c`/`_m`/`_z` masked
/// operation variants. They can also be built directly from a scalar `bool`
/// (splatting it to every lane) or from a [`Vector<R>`] (nonzero lanes become
/// `true`) via the [`From`] impls, or from the [`TRUTHY`](GenericMask::TRUTHY) /
/// [`FALSY`](GenericMask::FALSY) constants.
#[repr(transparent)]
pub struct Mask<R: Register>(#[doc(hidden)] pub Storage<R::Mask>);

impl<R: Register> Clone for Mask<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Mask<R> {}

impl<R: Register> Mask<R> {
    /// Strip the wrappers off a batch of masks for the register layer.
    #[inline(always)]
    fn raw<const N: usize>(masks: [Self; N]) -> [Storage<R::Mask>; N] {
        let mut raw = [<R::Mask as MaskRegister>::FALSY; N];

        let mut i = 0;
        while i < N {
            raw[i] = masks[i].0;
            i += 1;
        }

        raw
    }
}

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

use generic_array::typenum::Unsigned;

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
    const LANES: usize = <R::Lanes as Unsigned>::USIZE;
    type Lanes = R::Lanes;
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
    fn first_set(self) -> Option<usize> {
        <R::Mask as MaskRegister>::first_set([self.0])
    }

    #[inline(always)]
    fn last_set(self) -> Option<usize> {
        <R::Mask as MaskRegister>::last_set([self.0])
    }

    #[inline(always)]
    fn count_set(self) -> usize {
        <R::Mask as MaskRegister>::count_set([self.0])
    }

    // Hand the whole batch to the register layer in one call - that is where the
    // merged pack / horizontal-add reductions live.

    #[inline(always)]
    fn first_set_many<const N: usize>(masks: [Self; N]) -> Option<usize> {
        <R::Mask as MaskRegister>::first_set(Self::raw(masks))
    }

    #[inline(always)]
    fn last_set_many<const N: usize>(masks: [Self; N]) -> Option<usize> {
        <R::Mask as MaskRegister>::last_set(Self::raw(masks))
    }

    #[inline(always)]
    fn count_set_many<const N: usize>(masks: [Self; N]) -> usize {
        <R::Mask as MaskRegister>::count_set(Self::raw(masks))
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
    fn from_native_bitmask(bitmask: u64) -> Self {
        Mask(<R::Mask as MaskRegister>::from_native_bitmask(bitmask))
    }

    #[cfg(feature = "bitvec")]
    #[inline(always)]
    fn from_bitmask(bits: &bitvec::slice::BitSlice<u32>) -> Self {
        Mask(<R::Mask as MaskRegister>::from_bitmask(bits))
    }

    #[inline(always)]
    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        Mask(<R::Mask as BitwiseRegister>::ternlog::<IMM>(a.0, b.0, c.0))
    }

    const HAS_NATIVE_TERNLOG: bool = <R::Mask as BitwiseRegister>::HAS_NATIVE_TERNLOG;
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
