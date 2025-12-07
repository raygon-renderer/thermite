//! Low-level SIMD Register interface

pub mod dp;
pub mod element;
//pub mod tuple;

pub use element::{Element, FloatElement};

use generic_array::{
    ArrayLength, GenericArray, IntoArrayLength,
    typenum::{self, Unsigned},
};

use crate::{
    divider::{BranchfreeDivider, Divider, vector::VectorDivider},
    isa::InstructionSet,
    register::element::IntegerElement,
};

/// Helper type alias for double-pumped vectors.
pub type DoublePump<V> = <V as dp::DoublePumpVector>::DoublePumped;

#[inline(always)]
pub(crate) const fn reg<R: Register, const N: usize>(values: [R::Element; N]) -> R::Storage
where
    typenum::Const<N>: IntoArrayLength<ArrayLength = R::Lanes>,
{
    const {
        assert!(
            size_of::<R::Storage>() == size_of::<[R::Element; N]>(),
            "Size mismatch between register and array of elements"
        );
    }

    // SAFETY: The way const_transmute works
    // handles alignment automatically, so this
    // is valid so long as the size is the same.
    unsafe { generic_array::const_transmute(values) }
}

#[inline(always)]
pub(crate) const fn reg_splat<R: Register>(value: R::Element) -> R::Storage {
    let mut dst = R::EMPTY;

    // SAFETY: This is iterating over contiguous memory, just using a pointer
    unsafe {
        let dst = &mut dst as *mut R::Storage as *mut R::Element;

        let mut i = 0;
        while i < <R::Lanes as typenum::Unsigned>::USIZE {
            dst.add(i).write(value);

            i += 1;
        }
    }

    dst
}

#[inline(always)]
pub(crate) const fn empty_reg<R>() -> R::Storage
where
    R: Register,
{
    // SAFETY: Initialized memory but unset
    unsafe { core::mem::zeroed() }
}

pub trait Interoperable<A: MaskRegister<Lanes = Self::Lanes>, B: MaskRegister<Lanes = Self::Lanes>>: MaskRegister
    // bits
    + BitsRegister<Self>
    + BitsRegister<A>
    + BitsRegister<B>
    // casts
    + CastRegister<Self>
    + CastRegister<A>
    + CastRegister<B>
    // masks
    + CastMaskRegister<Self>
    + CastMaskRegister<A>
    + CastMaskRegister<B>
where
    A: BitsRegister<Self> + CastRegister<Self> + CastMaskRegister<Self>,
    B: BitsRegister<Self> + CastRegister<Self> + CastMaskRegister<Self>
{}

impl<R, A, B> Interoperable<A, B> for R
where
    R: MaskRegister
        + BitsRegister<Self>
        + BitsRegister<A>
        + BitsRegister<B>
        + CastRegister<Self>
        + CastRegister<A>
        + CastRegister<B>
        + CastMaskRegister<Self>
        + CastMaskRegister<A>
        + CastMaskRegister<B>,
    A: MaskRegister<Lanes = R::Lanes>,
    B: MaskRegister<Lanes = R::Lanes>,
    A: BitsRegister<R> + CastRegister<R> + CastMaskRegister<R>,
    B: BitsRegister<R> + CastRegister<R> + CastMaskRegister<R>,
{
}

// 1. Define our storage unit width
type BitsPerWord = typenum::U32; // We are storing bits in u32 chunks

// 2. Calculate the "Minus One" part of the ceiling formula: (y - 1)
// 32 - 1 = 31
type RoundUpConst = typenum::U31;

/// Defines the number of u32 words needed to hold a bitmask for a register with `Lanes` lanes.
pub type MaskWordCount<Lanes> = typenum::Quot<typenum::Sum<Lanes, RoundUpConst>, BitsPerWord>;

/// A trait for array length types representing the number of lanes in a SIMD register.
pub trait Lanes: ArrayLength + core::ops::Shl<typenum::B1> + core::ops::Add<RoundUpConst> {
    /// For a register of `Self` lanes, this is the number of `u32` words needed to hold a bitmask.
    ///
    /// Used in [`Mask::bitmask()`](crate::Mask::bitmask).
    type BitmaskLength: ArrayLength;
}

impl<T> Lanes for T
where
    T: ArrayLength + core::ops::Shl<typenum::B1> + core::ops::Add<RoundUpConst>,
    typenum::Sum<T, RoundUpConst>: core::ops::Div<BitsPerWord>,
    MaskWordCount<T>: ArrayLength,
{
    type BitmaskLength = MaskWordCount<T>;
}

pub(crate) type Storage<R> = <R as Register>::Storage;

pub trait Register: Sized + 'static {
    type Lanes: Lanes;
    type Element: Element;
    type Storage: Sized + Copy + core::fmt::Debug;

    const ISA: InstructionSet;

    // Note: These don't require :Register because it would introduce recursive type bounds.
    type HalfRegister;
    type DoubleRegister;

    /// Unsigned integer register type with the same number of lanes, used for
    /// variable shifts and other operations.
    type USize: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as Element>::USize>
        + CastRegister<Self::ISize>
        + BitsRegister<Self::ISize>;

    /// Signed integer register type with the same number of lanes.
    type ISize: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as Element>::ISize>
        + CastRegister<Self::USize>
        + BitsRegister<Self::USize>;

    const EMPTY: Storage<Self>;

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self>;
    fn single(value: Self::Element) -> Storage<Self>;
    fn splat(value: Self::Element) -> Storage<Self>;

    #[inline(always)]
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        Self::splat(Self::extract::<I>(value))
    }

    #[inline(always)]
    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        // NOTE: Slice indexing checks bounds, so this is safe.
        Self::splat(Self::as_array(&value)[idx])
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::read(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::read_unaligned(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        // Default to regular load if streaming loads are not supported.
        unsafe { Self::load(ptr) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::write(ptr as *mut Storage<Self>, value) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::write_unaligned(ptr as *mut Storage<Self>, value) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        // Default to regular store if streaming stores are not supported.
        unsafe { Self::store(ptr, value) }
    }

    // join/split fallbacks really shouldn't be used, but are here
    // for when HalfRegister is () (i.e., no smaller register type exists),
    // and if they _are_ used are at least a not-terrible fallback.

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        // NOTE: const_transmute will double-check sizes
        unsafe { generic_array::const_transmute((lo, hi)) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        // NOTE: const_transmute will double-check sizes
        unsafe { generic_array::const_transmute(value) }
    }

    #[inline(always)]
    fn concat(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self::DoubleRegister>
    where
        Self::DoubleRegister: Register<HalfRegister = Self>,
    {
        <Self::DoubleRegister as Register>::join(lo, hi)
    }

    #[inline(always)]
    fn as_array(storage: &Storage<Self>) -> &GenericArray<Self::Element, Self::Lanes> {
        unsafe { &*(storage as *const Storage<Self> as *const GenericArray<Self::Element, Self::Lanes>) }
    }

    #[inline(always)]
    fn as_array_mut(storage: &mut Storage<Self>) -> &mut GenericArray<Self::Element, Self::Lanes> {
        unsafe { &mut *(storage as *mut Storage<Self> as *mut GenericArray<Self::Element, Self::Lanes>) }
    }

    #[inline(always)]
    fn iter(storage: &Storage<Self>) -> core::slice::Iter<'_, Self::Element> {
        Self::as_array(storage).iter()
    }

    #[inline(always)]
    fn iter_mut(storage: &mut Storage<Self>) -> core::slice::IterMut<'_, Self::Element> {
        Self::as_array_mut(storage).iter_mut()
    }

    #[inline(always)]
    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        const {
            assert!(
                I < <Self::Lanes as Unsigned>::USIZE,
                "Index out of bounds for register lane extraction"
            );
        }

        Self::as_array(&value)[I]
    }

    #[inline(always)]
    fn insert<const I: usize>(mut value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        const {
            assert!(
                I < <Self::Lanes as Unsigned>::USIZE,
                "Index out of bounds for register lane insertion"
            );
        }

        Self::as_array_mut(&mut value)[I] = element;
        value
    }

    #[inline(always)]
    fn map<F>(mut value: Storage<Self>, mut f: F) -> Storage<Self>
    where
        F: FnMut(Self::Element) -> Self::Element,
    {
        for v in Self::as_array_mut(&mut value) {
            *v = f(*v);
        }

        value
    }

    #[inline(always)]
    fn zip<F>(mut lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for (a, b) in Self::as_array_mut(&mut lhs).iter_mut().zip(Self::as_array(&rhs)) {
            *a = f(*a, *b);
        }

        lhs
    }

    #[inline(always)]
    fn fold<F>(first: Self::Element, value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_array(&value).iter().fold(first, |acc, &v| f(acc, v))
    }

    #[inline(always)]
    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_array(&value)
            .iter()
            .skip(1)
            .fold(Self::extract::<0>(value), |acc, &v| f(acc, v))
    }

    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    /// !lhs & rhs
    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::bitand(Self::not(lhs), rhs)
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn not(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::bitandnot(mask, lhs), Self::bitand(mask, rhs))
    }

    /// Indicates if blendv only cares about the most significant bit (MSB) of the mask.
    const HAS_MSB_BLENDV: bool;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        Self::as_array_mut(&mut value).reverse();
        value
    }

    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>);

    /// Swap the byte order of each element in the register.
    fn swap_bytes(value: Storage<Self>) -> Storage<Self>;
}

pub trait ShuffleRegister: Register {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
}

pub trait PermuteRegister: Register {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self>;
}

pub trait BlendRegister: Register {
    fn blend<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
}

const fn is_power_of_2(n: u32) -> bool {
    (n & (n - 1)) == 0
}

pub trait SwizzleRegister: MaskRegister {
    const HAS_PERMUTEV: bool;

    #[inline(always)]
    fn scalar_permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        let value_array = Self::as_array(&value);
        let result_array = Self::as_array_mut(&mut result);

        let mask = Self::Lanes::U32 - 1;

        for (&idx, dst) in idxs.iter().zip(result_array.iter_mut()) {
            let idx = if const { is_power_of_2(Self::Lanes::U32) } {
                idx & mask // we can AND with the mask if power-of-two lane count
            } else {
                idx.min(mask) // otherwise clamp to the max index
            } as usize;

            unsafe { core::hint::assert_unchecked(idx < value_array.len()) };

            *dst = value_array[idx];
        }

        result
    }

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        Self::scalar_permutev(value, idxs)
    }

    #[inline(always)]
    fn scalar_swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        let a_array = Self::as_array(&a);
        let b_array = Self::as_array(&b);
        let result_array = Self::as_array_mut(&mut result);

        let mask = (<Self::Lanes as Unsigned>::U32 << 1) - 1;

        for (&idx, dst) in idxs.iter().zip(result_array.iter_mut()) {
            // NOTE: If Self is power of two, so is 2 * Self
            let mut idx = if const { is_power_of_2(Self::Lanes::U32) } {
                idx & mask // we can AND with the mask if power-of-two lane count
            } else {
                idx.min(mask) // otherwise clamp to the max index
            } as usize;

            *dst = if idx < Self::Lanes::USIZE {
                unsafe { core::hint::assert_unchecked(idx < a_array.len()) };

                a_array[idx]
            } else {
                idx -= Self::Lanes::USIZE;

                unsafe { core::hint::assert_unchecked(idx < b_array.len()) };

                b_array[idx]
            };
        }

        result
    }

    #[inline(always)]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        use typenum::Unsigned;

        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_swizzle(a, b, idxs);
        }

        let mut a_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();
        let mut b_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();

        let mut blend_mask = <Self as MaskRegister>::FALSY;

        let blend = Self::as_array_mut(&mut blend_mask);

        for (i, &idx) in idxs.iter().enumerate() {
            if idx < Self::Lanes::U32 {
                a_idxs[i] = idx;
                b_idxs[i] = i as u32;
                blend[i] = Element::FALSY;
            } else {
                a_idxs[i] = i as u32;
                b_idxs[i] = idx - Self::Lanes::U32;
                blend[i] = Element::TRUTHY;
            }
        }

        let tmp_a = Self::permutev(a, a_idxs);
        let tmp_b = Self::permutev(b, b_idxs);

        Self::blendv(blend_mask, tmp_a, tmp_b)
    }
}

pub trait BitshiftRegister: MaskRegister<Element: IntegerElement> {
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self>;
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self>;

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::shl(value, IMM8 as u32)
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::shr(value, IMM8 as u32)
    }

    /// Indicates if true variable shifts are supported, or `false` if it
    /// requires a scalar fallback.
    const HAS_TRUE_SHIFTV: bool;

    #[inline(always)]
    fn shrv(mut value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::USize as Register>::as_array(&shifts))
        {
            *r = *r >> *s;
        }

        value
    }

    #[inline(always)]
    fn shlv(mut value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::USize as Register>::as_array(&shifts))
        {
            *r = *r << *s;
        }

        value
    }

    /// Rotate bits left
    #[inline(always)]
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        Self::bitor(Self::shl(value, shift), Self::shr(value, width - shift))
    }

    /// Rotate bits right
    #[inline(always)]
    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        Self::bitor(Self::shr(value, shift), Self::shl(value, width - shift))
    }

    /// Rotate bits left by a constant amount
    #[inline(always)]
    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::rol(value, IMM8 as u32)
    }

    /// Rotate bits right by a constant amount
    #[inline(always)]
    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::ror(value, IMM8 as u32)
    }

    #[inline(always)]
    fn rolv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u16;
        let width_vec = Self::USize::splat(Element::from_u16(width));

        Self::bitor(
            Self::shlv(value, shifts),
            Self::shrv(value, Self::USize::sub(width_vec, shifts)),
        )
    }

    #[inline(always)]
    fn rorv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u16;
        let width_vec = Self::USize::splat(Element::from_u16(width));

        Self::bitor(
            Self::shrv(value, shifts),
            Self::shlv(value, Self::USize::sub(width_vec, shifts)),
        )
    }

    #[inline(always)]
    fn reverse_bits(mut value: Storage<Self>) -> Storage<Self> {
        // Use hardware byte swapping to handle bit reversals at the byte level and above.
        // This effectively handles s=32, s=16, s=8 for u64/u32/u16 in one go.
        value = Self::swap_bytes(value);

        let mut s = size_of::<Self::Element>() as u32 * 4; // Start with half the bit width
        let mut mask = Self::TRUTHY; // guaranteed to be all 1s

        // Update mask until it's at the byte level.
        // This is a separate loop because the compiler has an easier
        // time pre-computing the masks compared to a single complex loop.
        while s >= 8 {
            mask = Self::bitxor(mask, Self::shl(mask, s));
            s >>= 1;
        }

        // Perform the remaining sub-byte swaps (s=4, s=2, s=1)
        while s != 0 {
            mask = Self::bitxor(mask, Self::shl(mask, s));

            let left = Self::bitand(Self::shr(value, s), mask);
            let right = Self::bitand(Self::shl(value, s), Self::not(mask));

            value = Self::bitor(left, right);

            s >>= 1;
        }

        value
    }
}

pub trait MaskRegister: Register {
    const TRUTHY: Storage<Self>;
    const FALSY: Storage<Self>;

    #[inline(always)]
    fn boolean(value: bool) -> Storage<Self> {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        // NOTE: This is a fallback implementation.
        let mut result = Self::EMPTY;

        {
            let result = Self::as_array_mut(&mut result);
            for (i, v) in value.into_iter().enumerate() {
                result[i] = Self::Element::from_bool(v);
            }
        }
        result
    }

    #[inline(always)]
    fn debug_iter_bool(value: &Storage<Self>) -> impl Iterator<Item = bool> {
        Self::as_array(value).iter().map(|v| v.to_bool())
    }

    fn all(value: Storage<Self>) -> bool;
    fn any(value: Storage<Self>) -> bool;

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        !Self::any(value)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64>;

    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>);
}

pub trait WidenRegister<FROM>
where
    Self: Register<HalfRegister = FROM>,
    FROM: Register<Element = Self::Element>,
{
    /// Truncate the register from another register type into this register type.
    fn widen_from(value: FROM::Storage) -> Storage<Self>;

    #[inline(always)]
    fn join(lo: Storage<FROM>, hi: Storage<FROM>) -> Storage<Self> {
        <Self as Register>::join(lo, hi)
    }
}

impl<FROM, INTO> WidenRegister<FROM> for INTO
where
    INTO: Register<HalfRegister = FROM>,
    FROM: Register<Element = INTO::Element>,
{
    #[inline(always)]
    fn widen_from(value: <FROM as Register>::Storage) -> Storage<Self> {
        INTO::join(value, FROM::EMPTY)
    }
}

pub trait NarrowRegister<INTO>
where
    INTO: Register,
    Self: Register<Element = INTO::Element, HalfRegister = INTO>,
{
    /// Extend the register from another register type into this register type.
    fn narrow_from(value: Self::Storage) -> Storage<INTO>;

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<INTO>, Storage<INTO>) {
        <Self as Register>::split(value)
    }
}

impl<FROM, INTO> NarrowRegister<INTO> for FROM
where
    INTO: Register,
    FROM: Register<Element = INTO::Element, HalfRegister = INTO>,
{
    #[inline(always)]
    fn narrow_from(value: <FROM as Register>::Storage) -> Storage<INTO> {
        FROM::split(value).0
    }
}

/// A trait for registers that can be cast to/from other registers,
/// including of varying element types.
pub trait CastRegister<FROM: Register>: Register {
    /// Cast a register from another register type.
    fn cast_from(value: Storage<FROM>) -> Storage<Self>;

    /// Cast a register to another register type, potentially faster
    /// when the values are within a certain range, otherwise
    /// unspecified values are returned. This method is safe in the
    /// Rust sense, but may not be safe in the sense that values
    /// may not be preserved across the cast.
    #[inline(always)]
    fn fast_cast_from(value: Storage<FROM>) -> Storage<Self> {
        Self::cast_from(value)
    }
}

/// A trait for registers that can be reinterpreted as other registers,
/// though this is not a safe operation. This is only available for registers
/// of the same size in bytes. This is enforced simply by the fact that
/// it will only be implemented for registers of the same size.
pub trait BitsRegister<FROM: Register>: Register {
    fn from_bits(value: Storage<FROM>) -> Storage<Self>;
}

/// A trait for registers that can be reinterpreted as other registers, as masks,
/// such that the masks retain 0 or !0 values for the appropriate lanes.
pub trait CastMaskRegister<FROM: MaskRegister>: MaskRegister {
    fn mask_from(value: Storage<FROM>) -> Storage<Self>;
}

pub trait PartialOrdRegister: MaskRegister {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let gt = Self::gt(lhs, rhs);
        let eq = Self::eq(lhs, rhs);

        Self::bitor(gt, eq)
    }

    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::gt(rhs, lhs)
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ge(rhs, lhs)
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::not(Self::eq(lhs, rhs))
    }
}

// TODO: Replace `: Register` with `: PartialOrdRegister` when
// it's implemented for all registers.
pub trait NumericRegister: PartialOrdRegister {
    const ZERO: Storage<Self>;
    const ONE: Storage<Self>;
    const TWO: Storage<Self>;

    const MIN: Storage<Self>;
    const MAX: Storage<Self>;

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn min_element(value: Storage<Self>) -> Self::Element;
    fn max_element(value: Storage<Self>) -> Self::Element;

    fn sum_elements(value: Storage<Self>) -> Self::Element;
    fn prod_elements(value: Storage<Self>) -> Self::Element;

    /// Effectively the number of lanes in the register, splatted across the lanes.
    fn offset() -> Storage<Self>;
    /// 0, 1, 2, 3, 4, ... etc.
    fn indexed() -> Storage<Self>;
}

pub trait SignedRegister: NumericRegister {
    fn neg(value: Storage<Self>) -> Storage<Self>;
    fn abs(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn signum(value: Storage<Self>) -> Storage<Self> {
        let is_neg = Self::is_negative(value);
        let is_zero = Self::eq(value, Self::ZERO);

        Self::bitandnot(is_zero, Self::blendv(is_neg, Self::NEG_ONE, Self::ONE))
    }

    #[inline(always)]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let abs = Self::abs(lhs);

        Self::blendv(Self::is_negative(rhs), abs, Self::neg(abs))
    }

    const NEG_ONE: Storage<Self>;
    const MIN_POSITIVE: Storage<Self>;

    #[inline(always)]
    fn is_negative(value: Storage<Self>) -> Storage<Self> {
        Self::lt(value, Self::ZERO)
    }

    #[inline(always)]
    fn is_positive(value: Storage<Self>) -> Storage<Self> {
        Self::ge(value, Self::ZERO)
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::blendv(mask, value, Self::neg(value))
    }

    /// On platforms where blendv only checks the MSB, this can be optimized to avoid comparisons.
    #[inline(always)]
    fn select_negative(mut mask: Storage<Self>, falsy: Storage<Self>, truthy: Storage<Self>) -> Storage<Self> {
        if !Self::HAS_MSB_BLENDV {
            mask = Self::is_negative(mask);
        }

        Self::blendv(mask, falsy, truthy)
    }
}

#[inline(always)]
fn zip_ternary<R: FloatRegister, F>(mut lhs: R::Storage, rhs: R::Storage, acc: R::Storage, f: F) -> R::Storage
where
    F: Fn(&mut R::Element, R::Element, R::Element),
{
    let rhs = R::iter(&rhs);
    let acc = R::iter(&acc);

    for ((lhs, rhs), acc) in R::iter_mut(&mut lhs).zip(rhs).zip(acc) {
        f(lhs, *rhs, *acc);
    }

    lhs
}

pub trait FloatRegister:
    SignedRegister<Element: FloatElement> + Interoperable<Self::Bits, Self::Signed> + CastRegister<Self::ExtendedPrecision>
{
    type Bits: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElement>::Bits>
        + Interoperable<Self, Self::Signed>;
    type Signed: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElement>::Signed>
        + Interoperable<Self, Self::Bits>;

    /// Some algorithms may benefit from using a higher-precision float type for intermediate calculations,
    /// and this associated type provides that capability. If no higher-precision type is available,
    /// this type should be the same as `Self` as a safe fallback.
    type ExtendedPrecision: FloatRegister<Lanes = Self::Lanes> + CastRegister<Self>;

    const HAS_TRUE_FMA: bool;

    const HALF: Storage<Self>;
    const NEG_ZERO: Storage<Self>;
    const INFINITY: Storage<Self>;
    const NEG_INFINITY: Storage<Self>;
    const NAN: Storage<Self>;
    const EPSILON: Storage<Self>;

    const EXP_MASK: Storage<Self::Bits>;

    #[inline(always)]
    fn total_order(value: Storage<Self>) -> Storage<Self::Signed> {
        // value ^ (is_negative(value) >> 1), where is_negative produces all 1s for negative and all 0s for positive,
        // usually by shifting the sign bit to fill the register using an arithmetic shift right

        // Original algorithm from Rust's f32/f64 total_cmp implementation:
        //
        // In case of negatives, flip all the bits except the sign
        // to achieve a similar layout as two's complement integers
        //
        // Why does this work? IEEE 754 floats consist of three fields:
        // Sign bit, exponent and mantissa. The set of exponent and mantissa
        // fields as a whole have the property that their bitwise order is
        // equal to the numeric magnitude where the magnitude is defined.
        // The magnitude is not normally defined on NaN values, but
        // IEEE 754 totalOrder defines the NaN values also to follow the
        // bitwise order. This leads to order explained in the doc comment.
        // However, the representation of magnitude is the same for negative
        // and positive numbers – only the sign bit is different.
        // To easily compare the floats as signed integers, we need to
        // flip the exponent and mantissa bits in case of negative numbers.
        // We effectively convert the numbers to "two's complement" form.
        //
        // To do the flipping, we construct a mask and XOR against it.
        // We branchlessly calculate an "all-ones except for the sign bit"
        // mask from negative-signed values: right shifting sign-extends
        // the integer, so we "fill" the mask with sign bits, and then
        // convert to unsigned to push one more zero bit.
        // On positive values, the mask is all zeros, so it's a no-op.

        let signed_bits = <Self::Signed as BitsRegister<Self>>::from_bits(value);
        let is_negative = <Self::Signed as SignedRegister>::is_negative(signed_bits);
        let mask = <Self::Signed as BitshiftRegister>::shri::<1>(is_negative);

        <Self::Signed as Register>::bitxor(signed_bits, mask)
    }

    #[inline(always)]
    fn is_nan(value: Storage<Self>) -> Storage<Self> {
        // easiest way to check for NaN is to check if it's not equal to itself
        Self::ne(value, value)
    }

    #[inline(always)]
    fn is_infinite(value: Storage<Self>) -> Storage<Self> {
        Self::eq(Self::abs(value), Self::INFINITY)
    }

    #[inline(always)]
    fn is_finite(value: Storage<Self>) -> Storage<Self> {
        Self::lt(Self::abs(value), Self::INFINITY)
    }

    #[inline(always)]
    fn is_subnormal(value: Storage<Self>) -> Storage<Self> {
        // we're operating in the integer domain here
        let bits: Storage<Self::Bits> = <Self::Bits as BitsRegister<Self>>::from_bits(value);

        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits
        let rest = Self::Bits::bitandnot(Self::EXP_MASK, bits); // extract mantissa + sign bits

        // shift mantissa to remove sign bit, and even though it's offset
        // it'll still work since we're just checking for zero
        let mantissa = Self::Bits::shli::<1>(rest);

        // use eq here for both since there's always an instruction for that
        let exp_is_zero = Self::Bits::eq(exp, Self::Bits::ZERO);
        let mantissa_is_zero = Self::Bits::eq(mantissa, Self::Bits::ZERO);

        // float is subnormal if mantissa != 0 && exp == 0, and by using bitandnot we can avoid using ne above
        let is_subnormal = Self::Bits::bitandnot(mantissa_is_zero, exp_is_zero);

        // convert back to float register
        <Self as BitsRegister<Self::Bits>>::from_bits(is_subnormal)
    }

    #[inline(always)]
    fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self> {
        // we're operating in the integer domain here
        let bits: Storage<Self::Bits> = <Self::Bits as BitsRegister<Self>>::from_bits(value);

        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits

        // zero or subnormal if exp == 0, very simple
        let is_zero_or_subnormal = Self::Bits::eq(exp, Self::Bits::ZERO);

        // convert back to float register
        <Self as BitsRegister<Self::Bits>>::from_bits(is_zero_or_subnormal)
    }

    #[inline(always)]
    fn is_normal(value: Storage<Self>) -> Storage<Self> {
        let bits = <Self::Bits as BitsRegister<Self>>::from_bits(value);

        // "normal" is defined as not zero/subnormal, not infinite, and not NaN
        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits

        let exp_is_zero = Self::Bits::eq(exp, Self::Bits::ZERO);
        let exp_is_max = Self::Bits::eq(exp, Self::EXP_MASK);

        // normal if exp != 0 && exp != max, so 0 < exp < max is the normal range
        let is_normal = Self::Bits::bitandnot(exp_is_max, exp_is_zero);

        // convert back to float register
        <Self as BitsRegister<Self::Bits>>::from_bits(is_normal)
    }

    #[inline(always)]
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::mul_add(lhs, rhs, acc)
        } else {
            Self::add(Self::mul(lhs, rhs), acc)
        }
    }

    #[inline(always)]
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::mul_sub(lhs, rhs, acc)
        } else {
            Self::sub(Self::mul(lhs, rhs), acc)
        }
    }

    #[inline(always)]
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::nmul_add(lhs, rhs, acc)
        } else {
            Self::sub(acc, Self::mul(lhs, rhs))
        }
    }

    #[inline(always)]
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::nmul_sub(lhs, rhs, acc)
        } else {
            Self::mul_sube(Self::neg(lhs), rhs, acc)
        }
    }

    #[inline]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_mul_add(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_mul_sub(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_nmul_add(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_nmul_sub(*lhs, rhs, acc);
        })
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn rcp(value: Storage<Self>) -> Storage<Self> {
        Self::div(Self::ONE, value)
    }

    #[inline(always)]
    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        Self::rcp(Self::sqrt(value))
    }

    const HAS_APPROX_RSQRT: bool;
    const HAS_APPROX_RCP: bool;

    fn floor(value: Storage<Self>) -> Storage<Self>;
    fn ceil(value: Storage<Self>) -> Storage<Self>;
    fn round(value: Storage<Self>) -> Storage<Self>;
    fn trunc(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn fract(value: Storage<Self>) -> Storage<Self> {
        Self::sub(value, Self::trunc(value))
    }

    #[inline(always)]
    fn mul_sign(value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::signed_zero(sign))
    }

    #[inline(always)]
    fn signed_zero(value: Storage<Self>) -> Storage<Self> {
        Self::bitand(Self::NEG_ZERO, value)
    }

    #[inline(always)]
    fn next_up(value: Storage<Self>) -> Storage<Self> {
        let bits = <Self::Bits as BitsRegister<Self>>::from_bits(value);
        let abs = <Self::Bits as BitsRegister<Self>>::from_bits(Self::abs(value));

        let is_nan = Self::is_nan(value);
        let is_inf = Self::eq(value, Self::INFINITY);
        let unchanged = <Self::Bits as CastMaskRegister<Self>>::mask_from(Self::bitor(is_nan, is_inf));

        // Use bitwise comparison for positive/zero check to handle -0.0 correctly
        // (abs == bits) is true for positive numbers and +0.0, false for negative numbers and -0.0
        let is_positive = Self::Bits::eq(abs, bits);
        let is_zero = Self::Bits::eq(abs, <Self::Bits as BitsRegister<Self>>::from_bits(Self::ZERO));

        let add = Self::Bits::add(bits, Self::Bits::ONE);
        let sub = Self::Bits::sub(bits, Self::Bits::ONE);

        // If positive, add 1 (magnitude up). If negative, sub 1 (magnitude down towards -inf).
        // blendv(mask, lhs, rhs) -> if mask { rhs } else { lhs }
        let next_bits = Self::Bits::blendv(is_positive, sub, add);

        // If zero, return MIN_POSITIVE (0x1)
        let next_bits = Self::Bits::blendv(is_zero, next_bits, Self::Bits::ONE);

        <Self as BitsRegister<Self::Bits>>::from_bits(Self::Bits::blendv(unchanged, next_bits, bits))
    }

    #[inline(always)]
    fn next_down(value: Storage<Self>) -> Storage<Self> {
        let bits = <Self::Bits as BitsRegister<Self>>::from_bits(value);
        let abs = <Self::Bits as BitsRegister<Self>>::from_bits(Self::abs(value));

        let is_nan = Self::is_nan(value);
        let is_neg_inf = Self::eq(value, Self::NEG_INFINITY);
        let unchanged = <Self::Bits as CastMaskRegister<Self>>::mask_from(Self::bitor(is_nan, is_neg_inf));

        let is_positive = Self::Bits::eq(abs, bits);
        let is_zero = Self::Bits::eq(abs, <Self::Bits as BitsRegister<Self>>::from_bits(Self::ZERO));

        let add = Self::Bits::add(bits, Self::Bits::ONE);
        let sub = Self::Bits::sub(bits, Self::Bits::ONE);

        // If positive, sub 1 (magnitude down). If negative, add 1 (magnitude up towards -inf).
        // blendv(mask, lhs, rhs) -> if mask { rhs } else { lhs }
        let next_bits = Self::Bits::blendv(is_positive, add, sub);

        // If zero, return -MIN_POSITIVE (0x80...01)
        let sign_bit = <Self::Bits as BitsRegister<Self>>::from_bits(Self::NEG_ZERO);
        let min_neg = Self::Bits::bitor(Self::Bits::ONE, sign_bit);

        let next_bits = Self::Bits::blendv(is_zero, next_bits, min_neg);

        <Self as BitsRegister<Self::Bits>>::from_bits(Self::Bits::blendv(unchanged, next_bits, bits))
    }
}

/// Useful swizzle indices for 3D linear algebra operations, such as cross products,
/// for both 3-lane and 4-lane registers.
pub trait ValidLinAlg3Length<R: FloatRegister<Lanes = Self>>: Lanes {
    const ZXYW: GenericArray<u32, R::Lanes>;
    const YZXW: GenericArray<u32, R::Lanes>;
}

impl<R> ValidLinAlg3Length<R> for typenum::U4
where
    R: FloatRegister<Lanes = Self>,
{
    const ZXYW: GenericArray<u32, R::Lanes> = GenericArray::from_array([2, 0, 1, 3]);
    const YZXW: GenericArray<u32, R::Lanes> = GenericArray::from_array([1, 2, 0, 3]);
}

// Certain GPU vectors may actually have 3-lane registers, but this will probably
// never be implemented for CPU SIMD.
impl<R> ValidLinAlg3Length<R> for typenum::U3
where
    R: FloatRegister<Lanes = Self>,
{
    const ZXYW: GenericArray<u32, R::Lanes> = GenericArray::from_array([2, 0, 1]);
    const YZXW: GenericArray<u32, R::Lanes> = GenericArray::from_array([1, 2, 0]);
}

/// Extensions to the `FloatRegister` trait for the most common 3D linear algebra operations.
///
/// This is only available on 3 or 4-lane registers.
pub trait LinAlg3Register: FloatRegister<Lanes: ValidLinAlg3Length<Self>> + SwizzleRegister {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements3(Self::mul(lhs, rhs))
    }

    #[inline(always)]
    fn cross3<const DOP: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if DOP {
            // More accurate cross product using the "accurate difference of sums" method, but
            // requires fused multiply-add/subtract operations for best accuracy.
            let a = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::YZXW); // [y, z, x]
            let b = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW); // [z, x, y]
            let c = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW); // [z, x, y]
            let d = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::YZXW); // [y, z, x]

            let cd = Self::mul(c, d);

            let err = Self::nmul_add(c, d, cd);
            let dop = Self::mul_sub(a, b, cd);

            Self::add(dop, err)
        } else {
            let lhszxy = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW);
            let rhszxy = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW);

            let lhszxy_rhs = Self::mul(lhszxy, rhs);
            let rhszxy_lhs = Self::mul(rhszxy, lhs);

            let sub = Self::sub(lhszxy_rhs, rhszxy_lhs);

            Self::permutev(sub, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW)
        }
    }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        if Self::Lanes::USIZE == 4 {
            Self::insert::<3>(value, Element::ZERO)
        } else {
            value
        }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        if Self::Lanes::USIZE == 4 {
            Self::insert::<3>(value, Element::ONE)
        } else {
            value
        }
    }

    fn min_element3(value: Storage<Self>) -> Self::Element;
    fn max_element3(value: Storage<Self>) -> Self::Element;
    fn sum_elements3(value: Storage<Self>) -> Self::Element;
    fn prod_elements3(value: Storage<Self>) -> Self::Element;
}

/// Extensions to the `FloatRegister` trait for 4D linear algebra operations,
/// including quaternion operations.
pub trait LinAlg4Register: LinAlg3Register<Lanes = typenum::U4> {
    /// Quaternion multiplication.
    ///
    /// Corresponds to `lhs * rhs`.
    ///
    /// Method:
    /// 1. Calculate T1 = (lhs.w * rhs)
    /// 2. Calculate T2 = (lhs.x * rhs.wzyx) * {+,-,+,-}
    /// 3. Calculate T3 = (lhs.y * rhs.zwxy) * {+,+,-,-}
    /// 4. Calculate T4 = (lhs.z * rhs.yxwz) * {-,+,+,-}
    /// 5. Sum T1 + T2 + T3 + T4
    #[inline(always)]
    fn quat4_product(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        use crate::math::FloatConsts as C;

        let w = Self::broadcast::<3>(lhs);
        let x = Self::broadcast::<0>(lhs);
        let y = Self::broadcast::<1>(lhs);
        let z = Self::broadcast::<2>(lhs);

        // TODO: Alternative implementation when permutev is not available?
        let rhs_x = Self::permutev(rhs, GenericArray::from_array([3, 2, 1, 0]));
        let rhs_y = Self::permutev(rhs, GenericArray::from_array([2, 3, 0, 1]));
        let rhs_z = Self::permutev(rhs, GenericArray::from_array([1, 0, 3, 2]));

        // T2 Signs: (+, -, +, -) -> Negate indices 1 and 3
        let rhs_x_signed = Self::bitxor(
            rhs_x,
            const { reg::<Self, 4>([C::ZERO, C::NEG_ZERO, C::ZERO, C::NEG_ZERO]) },
        );

        // T3 Signs: (+, +, -, -) -> Negate indices 2 and 3
        let rhs_y_signed = Self::bitxor(
            rhs_y,
            const { reg::<Self, 4>([C::ZERO, C::ZERO, C::NEG_ZERO, C::NEG_ZERO]) },
        );

        // T4 Signs: (-, +, +, -) -> Negate indices 0 and 3
        let rhs_z_signed = Self::bitxor(
            rhs_z,
            const { reg::<Self, 4>([C::NEG_ZERO, C::ZERO, C::ZERO, C::NEG_ZERO]) },
        );

        // Pair 1: (w * rhs) + (x * rhs_x_signed)
        let sum12 = Self::mul_adde(x, rhs_x_signed, Self::mul(w, rhs));

        // Pair 2: (y * rhs_y_signed) + (z * rhs_z_signed)
        let sum34 = Self::mul_adde(z, rhs_z_signed, Self::mul(y, rhs_y_signed));

        Self::add(sum12, sum34)
    }

    #[inline(always)]
    fn quat4_vec3_product<const DOP: bool>(q: Storage<Self>, v: Storage<Self>) -> Storage<Self> {
        if const { Self::HAS_PERMUTEV } {
            // --- Fast SIMD Path (Giesen) ---
            // Formula: v + 2w(q x v) + 2(q x (q x v))

            let w = Self::broadcast::<3>(q);
            let q_xyz = q;

            // t = 2 * cross(q, v)
            let t = Self::cross3::<DOP>(q_xyz, v);
            let t = Self::add(t, t); // multiply by 2

            // result = v + w*t + cross(q, t)
            let w_t = Self::mul(w, t);
            let cross_q_t = Self::cross3::<DOP>(q_xyz, t);

            Self::add(v, Self::add(w_t, cross_q_t))
        } else {
            // --- Scalar/Fallback Path (Textbook) ---
            // Formula: 2(q.v)q + (w^2 - q.q)v + 2w(q x v)
            //
            // When shuffles are expensive (emulated), Cross Products are expensive.
            // This variant only uses 1 Cross Product, substituting the other with
            // 2 Dot Products (which are cheap purely vertical/scalar math).

            let u = q; // Vector part
            let s = Self::broadcast::<3>(q);

            // Term 1: 2 * dot(u, v) * u
            let dot_uv = Self::splat(Self::dot3(u, v));
            let t1 = Self::mul(u, Self::add(dot_uv, dot_uv));

            // Term 2: v * (s*s - dot(u, u))
            let t2 = Self::mul(v, Self::sub(Self::mul(s, s), Self::splat(Self::dot3(u, u))));

            // Term 3: 2s * cross(u, v)
            let t3 = Self::mul(Self::add(s, s), Self::cross3::<DOP>(u, v));

            // Summation: (Term 1 + Term 2) + Term 3
            Self::add(Self::add(t1, t2), t3)
        }
    }
}

use num_traits::{WrappingAdd, WrappingMul};

pub trait IntegerRegister: NumericRegister<Element: IntegerElement> + BitshiftRegister {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        Self::reduce(value, |a, b| a.wrapping_add(&b))
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        Self::reduce(value, |a, b| a.wrapping_mul(&b))
    }

    fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self>;
    fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self>;
    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self>;

    const HAS_HARDWARE_POPCNT: bool;

    fn count_ones(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self>;
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

pub trait UnsignedIntegerRegister: IntegerRegister<USize = Self> {
    /// Returns `floor(log2(x)) + 1`
    #[inline(always)]
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::next_power_of_two_m1(value))
    }

    /// Next power of two minus 1
    #[inline(always)]
    fn next_power_of_two_m1(mut value: Storage<Self>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u32;
        let mut s = 1;

        while s < width {
            value = Self::bitor(value, Self::shr(value, s));

            s <<= 1;
        }

        value
    }

    #[inline(always)]
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self> {
        // f = (v & (v - 1)) == 0
        Self::eq(Self::ZERO, Self::bitand(value, Self::sub(value, Self::ONE)))
    }

    #[inline(always)]
    fn parity(mut value: Storage<Self>) -> Storage<Self> {
        let mut shift = size_of::<Self::Element>() as u32 * 4; // Start with half the bit width

        if Self::HAS_HARDWARE_POPCNT {
            // If we have a hardware popcnt, we can just use that.
            value = Self::count_ones(value);
        } else if Self::HAS_TRUE_SHIFTV {
            // Slightly faster XOR reduction method that relies on variable shifts.
            // This is still O(log2(N)), but solves the last 4 bits with a lookup table.
            while shift >= 4 {
                value = Self::bitxor(value, Self::shr(value, shift));
                shift >>= 1;
            }

            value = Self::shrv(
                Self::splat(Element::from_u16(0x6996)),
                Self::bitand(value, Self::splat(Element::from_u16(0x0F))),
            );
        } else {
            // Generic exhaustive XOR reduction to compute parity, performs O(log2(N)) shifts and XORs.
            while shift > 0 {
                value = Self::bitxor(value, Self::shr(value, shift));
                shift >>= 1;
            }
        }

        Self::bitand(Self::ONE, value)
    }

    // TODO: Interleave bits?
}

pub trait SignedIntegerRegister: IntegerRegister<ISize = Self> + SignedRegister {
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self>;

    #[inline(always)]
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::sra(value, IMM8 as u32)
    }

    #[inline(always)]
    fn srav(mut value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::USize as Register>::as_array(&shifts))
        {
            *r = *r >> *s; // r in this context is signed, so this is an arithmetic shift
        }

        value
    }
}
