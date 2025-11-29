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
            core::mem::size_of::<R::Storage>() == core::mem::size_of::<[R::Element; N]>(),
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
    type USize: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as Element>::USize>;

    /// Signed integer register type with the same number of lanes.
    type ISize: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as Element>::ISize>;

    const EMPTY: Storage<Self>;

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self>;
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
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::read(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::read_unaligned(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        // Default to regular load if streaming loads are not supported.
        unsafe { Self::load(ptr) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::write(ptr as *mut Storage<Self>, value) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::write_unaligned(ptr as *mut Storage<Self>, value) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * core::mem::size_of::<Self::Element>()`.
    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        // Default to regular store if streaming stores are not supported.
        unsafe { Self::store(ptr, value) }
    }

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        let _ = (lo, hi);
        unimplemented!("Register::join() not implemented for this register type.")
    }

    #[inline(always)]
    fn split(
        value: Storage<Self>,
    ) -> (
        <Self::HalfRegister as Register>::Storage,
        <Self::HalfRegister as Register>::Storage,
    )
    where
        Self::HalfRegister: Register,
    {
        let _ = value;
        unimplemented!("Register::split() not implemented for this register type.")
    }

    #[inline(always)]
    fn concat(lo: Storage<Self>, hi: Storage<Self>) -> <Self::DoubleRegister as Register>::Storage
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

    fn reverse(value: Storage<Self>) -> Storage<Self>;

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

pub trait SwizzleRegister: MaskRegister {
    const HAS_PERMUTEV: bool;

    #[inline(always)]
    fn scalar_permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        let value_array = Self::as_array(&value);
        let result_array = Self::as_array_mut(&mut result);

        let mask = (<Self::Lanes as Unsigned>::U32) - 1;

        for (&idx, dst) in idxs.iter().zip(result_array.iter_mut()) {
            let idx = idx & mask;

            *dst = value_array[idx as usize];
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
            let idx = idx & mask;

            *dst = if idx < Self::Lanes::U32 {
                a_array[idx as usize]
            } else {
                b_array[(idx - Self::Lanes::U32) as usize]
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

pub trait BitshiftRegister: Register {
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self>;
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self>;

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self>;
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self>;

    fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self>;
    fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self>;

    /// Rotate bits left
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self>;
    /// Rotate bits right
    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self>;

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

    fn rolv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self>;
    fn rorv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self>;
    //fn rotatev(value: Storage<Self>, shifts: GenericArray<i32, Self::Lanes>) -> Storage<Self>;

    fn reverse_bits(value: Storage<Self>) -> Storage<Self>;
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

/// A trait for registers that can be cast to/from other registers,
/// including of varying element types.
pub trait CastRegister<FROM: Register>: Register {
    /// Cast a register from another register type.
    fn cast_from(value: FROM::Storage) -> Storage<Self>;

    /// Cast a register to another register type, potentially faster
    /// when the values are within a certain range, otherwise
    /// unspecified values are returned. This method is safe in the
    /// Rust sense, but may not be safe in the sense that values
    /// may not be preserved across the cast.
    #[inline(always)]
    fn fast_cast_from(value: FROM::Storage) -> Storage<Self> {
        Self::cast_from(value)
    }
}

/// A trait for registers that can be reinterpreted as other registers,
/// though this is not a safe operation. This is only available for registers
/// of the same size in bytes. This is enforced simply by the fact that
/// it will only be implemented for registers of the same size.
pub trait BitsRegister<FROM: Register>: Register {
    fn from_bits(value: FROM::Storage) -> Storage<Self>;
}

/// A trait for registers that can be reinterpreted as other registers, as masks,
/// such that the masks retain 0 or !0 values for the appropriate lanes.
pub trait CastMaskRegister<FROM: MaskRegister>: MaskRegister {
    fn mask_from(value: FROM::Storage) -> Storage<Self>;
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

    fn signum(value: Storage<Self>) -> Storage<Self>;

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

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

    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self>;

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
    fn total_order(value: Storage<Self>) -> <Self::Signed as Register>::Storage {
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

    fn next_up(value: Storage<Self>) -> Storage<Self>;
    fn next_down(value: Storage<Self>) -> Storage<Self>;
}

/// Extensions to the `FloatRegister` trait for the most common 3D linear algebra operations.
///
/// This is only available on 4-lane registers.
pub trait LinAlg3Register: FloatRegister<Lanes = generic_array::typenum::U4> + SwizzleRegister {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements3(Self::mul(lhs, rhs))
    }

    #[inline(always)]
    fn cross3(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let lhszxy = Self::permutev(lhs, GenericArray::from_array([2, 0, 1, 3]));
        let rhszxy = Self::permutev(rhs, GenericArray::from_array([2, 0, 1, 3]));

        let lhszxy_rhs = Self::mul(lhszxy, rhs);
        let rhszxy_lhs = Self::mul(rhszxy, lhs);

        let sub = Self::sub(lhszxy_rhs, rhszxy_lhs);

        Self::permutev(sub, GenericArray::from_array([2, 0, 1, 3]))
    }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        Self::insert::<3>(value, num_traits::Zero::zero())
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        Self::insert::<3>(value, num_traits::One::one())
    }

    fn min_element3(value: Storage<Self>) -> Self::Element;
    fn max_element3(value: Storage<Self>) -> Self::Element;
    fn sum_elements3(value: Storage<Self>) -> Self::Element;
    fn prod_elements3(value: Storage<Self>) -> Self::Element;

    // /// Quaternion multiplication.
    // fn quat4_product(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
}

pub trait IntegerRegister: NumericRegister + BitshiftRegister {
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn wrapping_sum(value: Storage<Self>) -> Self::Element;
    fn wrapping_product(value: Storage<Self>) -> Self::Element;

    fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self>;
    fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self>;
    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self>;

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

pub trait UnsignedIntegerRegister: IntegerRegister {
    /// Returns `floor(log2(x)) + 1`
    #[inline(always)]
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::next_power_of_two_m1(value))
    }

    /// Next power of two minus 1
    fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self>;

    fn is_power_of_two(value: Storage<Self>) -> Storage<Self>;

    fn parity(value: Storage<Self>) -> Storage<Self>;

    // TODO: Interleave bits?
}

pub trait SignedIntegerRegister: IntegerRegister + SignedRegister {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self>;
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self>;
    fn srav(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self>;
}
