//! Low-level SIMD Register interface

macro_rules! s {
    ($ty:ty: $a:expr, [$($idx:literal),* $(,)?]) => {{
        #[inline(always)]
        fn __do_permutev<R: Register>(a: Storage<R>) -> Storage<R> {
            struct Indices<N: generic_array::ArrayLength>(core::marker::PhantomData<N>);

            impl<N: generic_array::ArrayLength> SwizzleIndices<N> for Indices<N> {
                const INDICES: generic_array::GenericArray<u32, N> = const {
                    let idxs = [$($idx),*];
                    assert!(N::USIZE == idxs.len(), "Swizzle mask must be the same length as the register");
                    unsafe { generic_array::const_transmute::<_, generic_array::GenericArray<u32, N>>(idxs) }
                };
            }

            R::permutev_const::<Indices<R::Lanes>>(a)
        }

        __do_permutev::<$ty>($a)
    }};

    ($ty:ty: $a:expr, $b:expr, [$($idx:literal),* $(,)?]) => {{
        #[inline(always)]
        fn __do_swizzle<R: Register>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
            struct Indices<N: generic_array::ArrayLength>(core::marker::PhantomData<N>);

            impl<N: generic_array::ArrayLength> SwizzleIndices<N> for Indices<N> {
                const INDICES: generic_array::GenericArray<u32, N> = const {
                    let idxs = [$($idx),*];
                    assert!(N::USIZE == idxs.len(), "Swizzle mask must be the same length as the register");
                    unsafe { generic_array::const_transmute::<_, generic_array::GenericArray<u32, N>>(idxs) }
                };
            }

            R::swizzle_const::<Indices<R::Lanes>>(a, b)
        }

        __do_swizzle::<$ty>($a, $b)
    }};
}

pub mod array;
// pub mod dp;
pub mod linalg;
pub mod reduced;
pub mod well_formed;

use core::marker::PhantomData;

pub use crate::element::{Element, FloatElement, MaskElement};
pub use linalg::{LinAlg3Register, LinAlg4Register, ValidLinAlg3Length};

use generic_array::{
    ArrayLength, GenericArray, IntoArrayLength,
    typenum::{self, Unsigned},
};

use crate::{
    divider::{BranchfreeDivider, Divider, vector::VectorDivider},
    element::{FloatElementWithBits, IntegerElement, float::spec, float::spec::FloatSpec},
    isa::InstructionSet,
    math::policy::Policy,
    vector::{NewConst, ops::MulAddExt},
};

#[inline(always)]
pub(crate) const fn reg<R: Register, const N: usize>(values: [R::Element; N]) -> Storage<R>
where
    typenum::Const<N>: IntoArrayLength<ArrayLength = R::Lanes>,
{
    const {
        assert!(
            size_of::<Storage<R>>() == size_of::<[R::Element; N]>(),
            "Size mismatch between register and array of elements"
        );
    }

    // SAFETY: The way const_transmute works
    // handles alignment automatically, so this
    // is valid so long as the size is the same.
    unsafe { generic_array::const_transmute(values) }
}

#[inline(always)]
pub(crate) const fn reg_splat<R: Register>(value: R::Element) -> Storage<R> {
    let mut dst = R::EMPTY;

    // SAFETY: This is iterating over contiguous memory, just using a pointer
    unsafe {
        let dst = &mut dst as *mut Storage<R> as *mut R::Element;

        let mut i = 0;
        while i < <R::Lanes as typenum::Unsigned>::USIZE {
            dst.add(i).write(value);

            i += 1;
        }
    }

    dst
}

#[inline(always)]
pub(crate) const fn empty_reg<R>() -> Storage<R>
where
    R: CoreRegister,
{
    // SAFETY: Initialized memory but unset
    unsafe { core::mem::zeroed() }
}

pub trait MaskInteroperable<
    A: CoreRegister<Lanes = Self::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<B::Mask>>,
    B: CoreRegister<Lanes = Self::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<A::Mask>>,
>: CoreRegister<Mask: CastMaskRegister<A::Mask> + CastMaskRegister<B::Mask>>
{
}

impl<R, A, B> MaskInteroperable<A, B> for R
where
    R: CoreRegister<Mask: CastMaskRegister<A::Mask> + CastMaskRegister<B::Mask>>,
    A: CoreRegister<Lanes = R::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<B::Mask>>,
    B: CoreRegister<Lanes = R::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<A::Mask>>,
{
}

pub trait FullyInteroperable<
    A: Register<Lanes = Self::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<B::Mask>>,
    B: Register<Lanes = Self::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<A::Mask>>,
>: Register<Mask: CastMaskRegister<A::Mask> + CastMaskRegister<B::Mask>>
    // bits
    + BitCastRegister<Self>
    + BitCastRegister<A>
    + BitCastRegister<B>
    // casts
    + CastRegister<Self>
    + CastRegister<A>
    + CastRegister<B>
where
    A: BitCastRegister<Self> + CastRegister<Self>,
    B: BitCastRegister<Self> + CastRegister<Self>
{}

impl<R, A, B> FullyInteroperable<A, B> for R
where
    R: Register<Mask: CastMaskRegister<A::Mask> + CastMaskRegister<B::Mask>>
        + BitCastRegister<Self>
        + BitCastRegister<A>
        + BitCastRegister<B>
        + CastRegister<Self>
        + CastRegister<A>
        + CastRegister<B>,
    A: Register<Lanes = R::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<B::Mask>>,
    B: Register<Lanes = R::Lanes, Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<A::Mask>>,
    A: BitCastRegister<R> + CastRegister<R>,
    B: BitCastRegister<R> + CastRegister<R>,
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
pub trait Lanes:
    ArrayLength + core::ops::Shl<typenum::B1> + core::ops::Add<RoundUpConst> + core::ops::Shr<typenum::B1>
{
    /// For a register of `Self` lanes, this is the number of `u32` words needed to hold a bitmask.
    ///
    /// Used in [`GenericMask::bitmask()`](crate::mask::GenericMask::bitmask).
    type BitmaskLength: ArrayLength;

    #[cfg(feature = "bitvec")]
    type BitmaskStorage: bitvec::view::BitViewSized<Store = u32>;

    const IS_POWER_OF_TWO: bool;
}

#[cfg(feature = "bitvec")]
use bitvec::view::BitViewSized;

#[cfg(not(feature = "bitvec"))]
trait BitViewSized {
    type Store;
}

#[cfg(not(feature = "bitvec"))]
impl<N: ArrayLength> BitViewSized for GenericArray<u32, N> {
    type Store = u32;
}

impl<T> Lanes for T
where
    T: ArrayLength + core::ops::Shl<typenum::B1> + core::ops::Add<RoundUpConst> + core::ops::Shr<typenum::B1>,
    typenum::Sum<T, RoundUpConst>: core::ops::Div<BitsPerWord>,
    MaskWordCount<T>: ArrayLength,
    GenericArray<u32, MaskWordCount<T>>: BitViewSized<Store = u32>,
{
    type BitmaskLength = MaskWordCount<T>;

    #[cfg(feature = "bitvec")]
    type BitmaskStorage = GenericArray<u32, Self::BitmaskLength>;

    const IS_POWER_OF_TWO: bool = {
        let lanes = <T as Unsigned>::USIZE;
        lanes != 0 && (lanes & (lanes - 1)) == 0
    };
}

pub type Storage<R> = <R as CoreRegister>::Storage;

/// Value to give to `zeroupper_z`
pub trait ZeroUpper {
    /// The number of elements _remaining_ after zeroing.
    const N: usize;
}

pub(crate) struct OwnLanes<R: CoreRegister>(PhantomData<R>);

impl<R: CoreRegister> ZeroUpper for OwnLanes<R> {
    const N: usize = <R::Lanes as Unsigned>::USIZE;
}

/// Core data types for a given register. These are simple types
/// without any intertwining trait bounds.
pub trait CoreRegister: 'static + Sized {
    type Lanes: Lanes;
    type Storage: Sized + Copy + core::fmt::Debug;
    type Mask: MaskRegister<Lanes = Self::Lanes>;

    /// Indicates if the register is emulated in software.
    const IS_EMULATED: bool;

    const ISA: InstructionSet;

    /// If the associated mask type is equal in size to this register, which also implies it is
    /// trivially convertible to this register.
    const HAS_EQUAL_SIZE_MASK: bool;

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self>;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self>;

    /// Selects elements from `value` where `mask` is true, and zeroes elsewhere.
    #[inline(always)]
    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::blendv(mask, Self::EMPTY, value)
    }

    /// Selects elements from `value` where `mask` is false, and zeroes elsewhere.
    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if const { Self::HAS_EQUAL_SIZE_MASK } {
            // If the mask has the same size as the register, we can assume the
            // default behavior is `z` doing a bitwise AND, and we should NOT the mask.
            Self::zz(<Self::Mask as BitwiseRegister>::not(mask), value)
        } else {
            // Otherwise, we need to blend with zero.
            Self::blendv(mask, value, Self::EMPTY)
        }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self>;

    #[inline(always)]
    fn zeroupper(value: Storage<Self>) -> Storage<Self> {
        Self::zeroupper_z::<OwnLanes<Self>>(value)
    }

    const EMPTY: Storage<Self>;
}

#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait BitwiseRegister: CoreRegister {
    #[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    #[conditional] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn not(value: Storage<Self>) -> Storage<Self>;

    /// !lhs & rhs
    #[conditional] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::bitand(Self::not(lhs), rhs)
    }

    /// const A = 0xF0, B = 0xCC, C = 0xAA
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        let mut acc = Self::EMPTY;

        if IMM == 0xCA {
            // Special case for select pattern `a ? b : c` to improve debug builds
            return Self::bitor(Self::bitand(a, b), Self::bitandnot(a, c));
        }

        // Combine cases using Disjunctive Normal Form (DNF)
        macro_rules! case {
            (0,         $expr:expr) => { if (IMM & (1 << 0))    != 0 { acc = $expr; } };
            ($bit:expr, $expr:expr) => { if (IMM & (1 << $bit)) != 0 { acc = Self::bitor(acc, $expr); } };
        }

        case!(0, Self::bitandnot(a, Self::bitandnot(b, Self::not(c)))); // Case 0: inputs are 0, 0, 0
        case!(1, Self::bitandnot(a, Self::bitandnot(b, c)));            // Case 1: inputs are 0, 0, 1
        case!(2, Self::bitandnot(a, Self::bitandnot(c, b)));            // Case 2: inputs are 0, 1, 0; b, c swapped to save a NOT
        case!(3, Self::bitandnot(a, Self::bitand(b, c)));               // Case 3: inputs are 0, 1, 1
        case!(4, Self::bitandnot(c, Self::bitandnot(b, a)));            // Case 4: inputs are 1, 0, 0; a, c swapped to save a NOT
        case!(5, Self::bitand(a, Self::bitandnot(b, c)));               // Case 5: inputs are 1, 0, 1
        case!(6, Self::bitand(a, Self::bitandnot(c, b)));               // Case 6: inputs are 1, 1, 0; b, c swapped to save a NOT
        case!(7, Self::bitand(a, Self::bitand(b, c)));                  // Case 7: inputs are 1, 1, 1

        acc
    }

    /// const A = 0xC, B = 0xA
    #[conditional] fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        let mut acc = Self::EMPTY;

        // Disjunctive Normal Form (DNF) again
        if (IMM & (1 << 0)) != 0 { acc = Self::not(Self::bitor(a, b)); } // Case 0: inputs are 0, 0, simplified
        if (IMM & (1 << 1)) != 0 { acc = Self::bitor(acc, Self::bitandnot(a, b)); } // Case 1: inputs are 0, 1
        if (IMM & (1 << 2)) != 0 { acc = Self::bitor(acc, Self::bitandnot(b, a)); } // Case 2: inputs are 1, 0
        if (IMM & (1 << 3)) != 0 { acc = Self::bitor(acc, Self::bitand(a, b)); } // Case 3: inputs are 1, 1

        acc
    }
}

pub trait InterleaveRegister: CoreRegister {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>);
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>);
}

/// Bitmask of the low `lanes` bits set (the valid-lane window of a packed
/// bitmask). `lanes >= 64` yields all ones.
#[inline(always)]
const fn lane_bitmask(lanes: usize) -> u64 {
    if lanes >= 64 { u64::MAX } else { (1u64 << lanes) - 1 }
}

/// Mask registers, which operate on boolean values, though not necessarily
/// with `bool` storage.
///
/// Their storage type may differ from that of regular registers, or even between
/// similar vector types between architectures. E.g., AVX-512 mask registers
/// use 16-bit integers as storage, while AVX2 uses full SIMD registers with
/// all `0` and `1` bits to represent `false` and `true`, respectively.
#[thermite_macros::register_trait]
pub trait MaskRegister: BitwiseRegister<Mask = Self> + CastMaskRegister<Self> + InterleaveRegister {
    const TRUTHY: Storage<Self>;
    const FALSY: Storage<Self>;

    fn boolean(value: bool) -> Storage<Self> {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    fn set(mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self>;
    fn test(mask: Storage<Self>, lane: usize) -> bool;

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        // NOTE: This is a fallback implementation.
        let mut result = Self::FALSY;

        {
            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                if value[i] {
                    result = Self::set(result, i, true);
                }
            }
        }

        result
    }

    fn all(value: Storage<Self>) -> bool;
    fn any(value: Storage<Self>) -> bool;

    fn none(value: Storage<Self>) -> bool {
        !Self::any(value)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64>;

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>);

    #[cfg(feature = "bitvec")]
    fn bitmask(value: Storage<Self>) -> bitvec::array::BitArray<<Self::Lanes as Lanes>::BitmaskStorage> {
        let mut bitmask = bitvec::array::BitArray::ZERO;

        // try to use native bitmask if available
        if let Some(native) = Self::native_bitmask(value) {
            let bits = unsafe { core::mem::transmute::<u64, [u32; 2]>(native) };
            let bits = bitvec::slice::BitSlice::<u32>::from_slice(&bits);
            bitmask[..<Self::Lanes as Unsigned>::USIZE].copy_from_bitslice(&bits[..<Self::Lanes as Unsigned>::USIZE]);
        } else {
            // otherwise fill bitmask using the register's method
            Self::fill_bitmask(value, &mut bitmask[..<Self::Lanes as Unsigned>::USIZE]);
        }

        bitmask
    }

    // The `else` arms below are only reached by masks whose `native_bitmask`
    // returns `None` (wider than 64 lanes). The only such type today,
    // `ArrayRegister`, overrides all three with a sub-register scan, so these
    // fall back to the canonical `bitmask()` word-scan (the same packing
    // `bitmask()` itself uses) rather than poking one lane at a time - dropping
    // to the per-lane `test` loop only when `bitvec` is unavailable.

    /// Index of the lowest lane set to `true`, or `None` if every lane is
    /// `false`. A SIMD find-first: combined with a comparison this is `memchr`.
    fn first_set(value: Storage<Self>) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            let bm = bm & lane_bitmask(lanes);
            (bm != 0).then(|| bm.trailing_zeros() as usize)
        } else {
            #[cfg(feature = "bitvec")]
            { Self::bitmask(value).first_one() }
            #[cfg(not(feature = "bitvec"))]
            { (0..lanes).find(|&i| Self::test(value, i)) }
        }
    }

    /// Index of the highest lane set to `true`, or `None` if every lane is
    /// `false` (a find-last).
    fn last_set(value: Storage<Self>) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            let bm = bm & lane_bitmask(lanes);
            (bm != 0).then(|| 63 - bm.leading_zeros() as usize)
        } else {
            #[cfg(feature = "bitvec")]
            { Self::bitmask(value).last_one() }
            #[cfg(not(feature = "bitvec"))]
            { (0..lanes).rev().find(|&i| Self::test(value, i)) }
        }
    }

    /// Number of lanes set to `true` (population count of the mask).
    fn count_set(value: Storage<Self>) -> usize {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            (bm & lane_bitmask(lanes)).count_ones() as usize
        } else {
            #[cfg(feature = "bitvec")]
            { Self::bitmask(value).count_ones() }
            #[cfg(not(feature = "bitvec"))]
            { (0..lanes).filter(|&i| Self::test(value, i)).count() }
        }
    }
}

pub trait NewRegister<E, N, S> {
    type New<C: NewConst<E, N>>: crate::vector::VectorValue<C, S>;
}

/// SIMD Register trait where each Element implements the [`Element`] trait.
#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait Register:
    BitwiseRegister + InterleaveRegister +
    NewRegister<Self::Element, Self::Lanes, Storage<Self>> +
    CastRegister<Self> + BitCastRegister<Self> + MaskInteroperable<Self::Signed, Self::Unsigned>
{
    type Element: Element;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask>;

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::into_mask(value)
    }

    /// Convert the most significant bit of each element into a mask register. Only
    /// the MSB of each element is considered. The rest of the bits are ignored.
    ///
    /// This can skip some intermediate steps on some architectures and data types,
    /// and useful when dealing with sign bits.
    ///
    /// For floats this is often free, but for integers it'll have to effectively call `is_negative`.
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask>;

    /// Unsigned integer register type with the same number of lanes, used for
    /// variable shifts and other operations.
    type Unsigned: UnsignedIntegerRegister<
            Signed = Self::Signed,
            Unsigned = Self::Unsigned,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::Unsigned,
        > + CastRegister<Self::Signed>
        + BitCastRegister<Self::Signed>
        + MaskInteroperable<Self, Self::Signed>;

    /// SignedBits integer register type with the same number of lanes.
    type Signed: SignedIntegerRegister<
            Unsigned = Self::Unsigned,
            Signed = Self::Signed,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::Signed,
        > + CastRegister<Self::Unsigned>
        + BitCastRegister<Self::Unsigned>
        + MaskInteroperable<Self, Self::Unsigned>;

    #[masked]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self>;

    fn single(value: Self::Element) -> Storage<Self>;

    #[masked] fn splat(value: Self::Element) -> Storage<Self>;

    #[conditional]
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        Self::splat(Self::extract::<I>(value))
    }

    #[conditional]
    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        // NOTE: Slice indexing checks bounds, so this is safe.
        Self::splat(Self::as_array(&value)[idx])
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::read(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe {
            let mut result = src;
            let res = Self::as_array_mut(&mut result);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                if !<Self::Mask as MaskRegister>::test(mask, i) {
                    continue;
                }

                res[i] = ptr.add(i).read();
            }

            result
        }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self::load_m(Self::EMPTY, mask, ptr) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        const {
            assert!(
                size_of::<Storage<Self>>() == (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Size mismatch between register storage and array of elements"
            );
        }

        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::read_unaligned(ptr as *const Storage<Self>) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        // Default to regular load if streaming loads are not supported.
        unsafe { Self::load(ptr) }
    }

    // TODO: Masked stores? Would have to fallback to scalar on all but AVX-512,
    // but could still be useful for some patterns.

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::write(ptr as *mut Storage<Self>, value) }
    }

    /// # Safety
    ///
    /// The pointer must be valid, align, and point to a memory location where, when the mask is true,
    /// is valid for writing a value of type `Self::Element`.
    ///
    /// The memory locations where the mask is false are not accessed.
    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe {
            let res = Self::as_array(&value);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                if !<Self::Mask as MaskRegister>::test(mask, i) {
                    continue;
                }

                ptr.add(i).write(res[i]);
            }
        }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        const {
            assert!(
                size_of::<Storage<Self>>() == (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Size mismatch between register storage and array of elements"
            );
        }

        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        unsafe { core::ptr::write_unaligned(ptr as *mut Storage<Self>, value) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        // Default to regular store if streaming stores are not supported.
        unsafe { Self::store(ptr, value) }
    }

    /// # Safety
    ///
    /// Every lane of `indices` must be a valid index into `values` (i.e. `< values.len()`).
    /// The default implementation bounds-checks and panics on an out-of-range index, but
    /// hardware-gather overrides (e.g. `_mm256_permutevar8x32_ps`, `vpgatherdd`) do not -
    /// passing an out-of-range index there is undefined behavior.
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        let indices = <Self::Unsigned as Register>::as_array(&indices);

        let mut res = Self::EMPTY;
        let mut resa = Self::as_array_mut(&mut res);

        for i in 0..Self::Lanes::USIZE {
            let idx: usize = indices[i].try_into().unwrap_or_else(#[cold] |_| panic!("Invalid index given for lookup"));

            resa[i] = values[idx];
        }

        res
    }

    fn as_array(storage: &Storage<Self>) -> &GenericArray<Self::Element, Self::Lanes> {
        unsafe { &*(storage as *const Storage<Self> as *const GenericArray<Self::Element, Self::Lanes>) }
    }

    fn as_array_mut(storage: &mut Storage<Self>) -> &mut GenericArray<Self::Element, Self::Lanes> {
        unsafe { &mut *(storage as *mut Storage<Self> as *mut GenericArray<Self::Element, Self::Lanes>) }
    }

    fn iter(storage: &Storage<Self>) -> core::slice::Iter<'_, Self::Element> {
        Self::as_array(storage).iter()
    }

    fn iter_mut(storage: &mut Storage<Self>) -> core::slice::IterMut<'_, Self::Element> {
        Self::as_array_mut(storage).iter_mut()
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        const {
            assert!(
                I < <Self::Lanes as Unsigned>::USIZE,
                "Index out of bounds for register lane extraction"
            );
        }

        Self::as_array(&value)[I]
    }

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

    fn map<F>(mut value: Storage<Self>, mut f: F) -> Storage<Self>
    where
        F: FnMut(Self::Element) -> Self::Element,
    {
        for v in Self::as_array_mut(&mut value) {
            *v = f(*v);
        }

        value
    }

    fn zip<F>(mut lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for (a, b) in Self::as_array_mut(&mut lhs).iter_mut().zip(Self::as_array(&rhs)) {
            *a = f(*a, *b);
        }

        lhs
    }

    fn fold<F>(first: Self::Element, value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_array(&value).iter().fold(first, |acc, &v| f(acc, v))
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_array(&value)
            .iter()
            .skip(1)
            .fold(Self::extract::<0>(value), |acc, &v| f(acc, v))
    }

    // /// SIMD version of reduce, where the reduction is done in a tree-like fashion.
    // /// The result is still a full register, but the lowest lane contains the reduced value.
    // fn reduce_simd<F, L>(mut value: Storage<Self>, f: F, last: L) -> Self::Element
    // where
    //     F: Fn(Storage<Self>, Storage<Self>) -> Storage<Self>,
    //     L: FnOnce(Storage<Self>) -> Self::Element,
    // {
    //     let mut lanes = <Self::Lanes as Unsigned>::USIZE;

    //     while lanes > 1 {
    //         let half = lanes >> 1;

    //         // TODO

    //         lanes = half;
    //     }

    //     last(value)
    // }

    #[conditional]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        Self::as_array_mut(&mut value).reverse();
        value
    }

    /// Swap the byte order of each element in the register.
    #[conditional] fn swap_bytes(value: Storage<Self>) -> Storage<Self>;

    /// Left-pack (a.k.a. `compress`): gather the lanes where `mask` is set into
    /// the low lanes, preserving their relative order. The unselected lanes are
    /// *kept* (not zeroed) and packed into the high lanes, also in order - i.e. a
    /// stable partition of the register by `mask`.
    ///
    /// The number of low lanes that came from `mask` equals its population
    /// count. For `value = [a, b, c, d]` and `mask = [T, F, T, F]` the result is
    /// `[a, c, b, d]` (selected `a, c` first, then unselected `b, d`).
    ///
    /// This maps to AVX-512 `vpcompress*` (merge form). The default is a portable
    /// scalar stable partition that every register inherits; concrete backend
    /// registers override it with the table / wide / merge polyfills where those
    /// are a win. For the zero-filled tail variant matching AVX-512 zero-masking,
    /// see [`compress_z`](Self::compress_z).
    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        let n = <Self::Lanes as Unsigned>::USIZE;

        let src = Self::as_array(&value);
        let mut result = value;
        let dst = Self::as_array_mut(&mut result);

        let mut pos = 0;

        // Selected lanes first, in order.
        for i in 0..n {
            if <Self::Mask as MaskRegister>::test(mask, i) {
                dst[pos] = src[i];
                pos += 1;
            }
        }

        // Unselected lanes after, in order.
        for i in 0..n {
            if !<Self::Mask as MaskRegister>::test(mask, i) {
                dst[pos] = src[i];
                pos += 1;
            }
        }

        result
    }

    /// Zero-filling left-pack: like [`compress`](Self::compress), but the lanes
    /// beyond the population count are zeroed instead of holding the unselected
    /// elements. Matches AVX-512 zero-masking `vpcompress*`.
    ///
    /// For `value = [a, b, c, d]` and `mask = [T, F, T, F]` the result is
    /// `[a, c, 0, 0]`.
    ///
    /// The default is a single scalar pass - selected lanes to the front, the
    /// rest left zero - skipping the unselected-lane bookkeeping that
    /// [`compress`](Self::compress) needs. Concrete backend registers override
    /// it (the macros do so alongside `compress`) for the table / wide paths.
    fn compress_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        let n = <Self::Lanes as Unsigned>::USIZE;
        let src = Self::as_array(&value);

        // `EMPTY` is zero, so the tail is already filled - only place selected.
        let mut result = Self::EMPTY;
        let dst = Self::as_array_mut(&mut result);

        let mut pos = 0;
        for i in 0..n {
            if <Self::Mask as MaskRegister>::test(mask, i) {
                dst[pos] = src[i];
                pos += 1;
            }
        }

        result
    }

    const HAS_PERMUTEV: bool;

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

    #[masked]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        Self::scalar_permutev(value, idxs)
    }

    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        Self::permutev(value, I::INDICES)
    }

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

    #[masked]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        use typenum::Unsigned;

        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_swizzle(a, b, idxs);
        }

        let mut a_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();
        let mut b_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();

        let mut blend_mask = <Self::Mask as MaskRegister>::FALSY;

        for (i, &idx) in idxs.iter().enumerate() {
            if idx < Self::Lanes::U32 {
                a_idxs[i] = idx;
                b_idxs[i] = i as u32;
            } else {
                a_idxs[i] = i as u32;
                b_idxs[i] = idx - Self::Lanes::U32;
                blend_mask = <Self::Mask as MaskRegister>::set(blend_mask, i, true);
            }
        }

        let tmp_a = Self::permutev(a, a_idxs);
        let tmp_b = Self::permutev(b, b_idxs);

        Self::blendv(blend_mask, tmp_a, tmp_b)
    }

    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::swizzle(a, b, I::INDICES)
    }

    /// Two-register element align (the `palignr` family): the window of
    /// `Self::Lanes` lanes starting at lane `OFFSET` of the concatenation
    /// `[a, b]` (`a`'s lanes first, then `b`'s). `OFFSET == 0` returns `a`,
    /// `OFFSET == LANES` returns `b`; in between, lanes spill from the tail of
    /// `a` into the head of `b`.
    ///
    /// The cross-register sliding window used for multi-byte delimiter /
    /// substring scanning across a load boundary - the cross-register companion
    /// to the single-register [`bshli`](BitshiftRegister::bshli)/[`bshri`](BitshiftRegister::bshri).
    /// The default routes through [`swizzle_const`](Self::swizzle_const) with a
    /// compile-time [`AlignIndices`](crate::swizzle::AlignIndices) pattern, so it
    /// is correct on every backend, element type, and lane count. Integer
    /// backends override it with native byte aligns (`palignr`, whole-register
    /// byte shifts, or the AVX2 256-bit sequence).
    fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b)
    }

    /// Runtime permute of an `N`-chunk [`ArrayRegister<Self, N>`](array::ArrayRegister)
    /// by a full-width index slice (`idxs.len() == N * Self::LANES`).
    ///
    /// `ArrayRegister`'s `permutev` delegates here so a specific backend register
    /// can override the cross-chunk routing with a faster sequence. The default
    /// is branchless: for each output chunk it splits each global index into a
    /// local index (`idx % LANES`) and a source-chunk id (`idx / LANES`), then
    /// for each input chunk builds the blend mask with a single vector compare
    /// (`chunk_id == j`) rather than per-lane mask inserts.
    ///
    /// `#[inline(always)]` so that when called with compile-time-constant indices
    /// (via [`permutev_const`](Self::permutev_const)) the whole routing -
    /// local/chunk split and blend selectors - constant-folds.
    #[inline(always)]
    fn array_permutev<const N: usize>(value: [Storage<Self>; N], idxs: &[u32]) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;
        let total = N * l;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let base = i * l;

            // Branchless split of this output chunk's indices into local offsets
            // (for the per-chunk permute) and source-chunk ids (for the blend).
            let mut local: GenericArray<u32, Self::Lanes> = GenericArray::default();
            let mut chunk_ids: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();

            for lane in 0..l {
                let g = idxs[base + lane] as usize;
                let g = if const { (N * <Self::Lanes as Unsigned>::USIZE).is_power_of_two() } {
                    g & (total - 1)
                } else {
                    g.min(total - 1)
                };
                local[lane] = (g % l) as u32;
                chunk_ids[lane] = Element::from_u16((g / l) as u16);
            }

            let chunk_reg = Self::Unsigned::new(chunk_ids);

            let mut out = Self::EMPTY;
            for j in 0..N {
                let j_splat = Self::Unsigned::splat(Element::from_u16(j as u16));
                let eq = Self::Unsigned::eq(chunk_reg, j_splat);
                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(eq);
                let permuted = Self::permutev(value[j], local.clone());
                out = Self::blendv(blend, out, permuted);
            }

            result[i] = out;
        }

        result
    }

    /// Runtime swizzle of two `N`-chunk [`ArrayRegister<Self, N>`](array::ArrayRegister)
    /// values by a full-width index slice selecting across all `2N` input chunks
    /// (`a` then `b`). The two-source companion to [`array_permutev`](Self::array_permutev);
    /// same branchless default, overridable per register.
    #[inline(always)]
    fn array_swizzle<const N: usize>(a: [Storage<Self>; N], b: [Storage<Self>; N], idxs: &[u32]) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;
        let total = N * l;
        let span = 2 * total;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let base = i * l;

            let mut local: GenericArray<u32, Self::Lanes> = GenericArray::default();
            let mut chunk_ids: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();

            for lane in 0..l {
                let g = idxs[base + lane] as usize;
                let g = if const { (2 * N * <Self::Lanes as Unsigned>::USIZE).is_power_of_two() } {
                    g & (span - 1)
                } else {
                    g.min(span - 1)
                };
                local[lane] = (g % l) as u32;
                chunk_ids[lane] = Element::from_u16((g / l) as u16);
            }

            let chunk_reg = Self::Unsigned::new(chunk_ids);

            let mut out = Self::EMPTY;
            for j in 0..(2 * N) {
                let src = if j < N { a[j] } else { b[j - N] };
                let j_splat = Self::Unsigned::splat(Element::from_u16(j as u16));
                let eq = Self::Unsigned::eq(chunk_reg, j_splat);
                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(eq);
                let permuted = Self::permutev(src, local.clone());
                out = Self::blendv(blend, out, permuted);
            }

            result[i] = out;
        }

        result
    }
}

const fn is_power_of_2(n: u32) -> bool {
    (n & (n - 1)) == 0
}

pub trait SwizzleIndices<N: ArrayLength> {
    const INDICES: GenericArray<u32, N>;
}

/// Combine and split registers.
///
/// This is used for widening and narrowing operations, where we want to combine
/// two narrower registers into a wider one, or split a wider register into two narrower ones.
pub trait ConcatRegister<HALF: CoreRegister>: ExtendRegister<HALF> {
    fn concat(lo: Storage<HALF>, hi: Storage<HALF>) -> Storage<Self>;
    fn split(value: Storage<Self>) -> (Storage<HALF>, Storage<HALF>);
}

pub trait SplitRegister<WIDE: ConcatRegister<Self>>: CoreRegister {}
impl<HALF: CoreRegister, WIDE: CoreRegister> SplitRegister<WIDE> for HALF where WIDE: ConcatRegister<HALF> {}

/// Zero-extend a narrower register into a wider register.
pub trait ExtendRegister<FROM: CoreRegister>: CoreRegister {
    fn extend(value: Storage<FROM>) -> Storage<Self>;

    /// Narrow a wider register into a narrower register, discarding the upper bits.
    fn narrow(value: Storage<Self>) -> Storage<FROM>;
}

pub trait NarrowRegister<TO: ExtendRegister<Self>>: CoreRegister {}
impl<FROM: CoreRegister, TO: CoreRegister> NarrowRegister<TO> for FROM where TO: ExtendRegister<FROM> {}

/// Register trait implemented for registers with known wider registers available, such as on AVX2 we can concat two 128-bit
/// registers to one 256-bit register.
pub trait WideRegister: Register
where
    typenum::Double<Self::Lanes>: Lanes,
{
    /// 2x Wide register
    type Wide: ConcatRegister<Self> + Register<Element = Self::Element, Lanes = typenum::Double<Self::Lanes>>;
}

// pub trait IndexableFor<FOR: IndexableRegister<Self>>: UnsignedIntegerRegister<Lanes = FOR::Lanes> {}

// impl<IDX, FOR> IndexableFor<FOR> for IDX
// where
//     IDX: UnsignedIntegerRegister<Lanes = FOR::Lanes>,
//     FOR: IndexableRegister<Self>,
// {
// }

pub trait IndexableRegister<IDX: UnsignedIntegerRegister<Lanes = Self::Lanes>>: Register {
    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and pointing to memory locations that
    /// can be safely read from based on the register's requirements.
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<IDX>) -> Storage<Self> {
        let scale = size_of::<Self::Element>();

        unsafe {
            let mut result = Self::EMPTY;

            let res = Self::as_array_mut(&mut result);
            let indices = IDX::as_array(&indices);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                res[i] = ptr.add(indices[i].try_into().unwrap_unchecked()).read();
            }

            result
        }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and pointing to memory locations that
    /// can be safely read from based on the register's requirements.
    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<IDX>,
    ) -> Storage<Self> {
        let scale = size_of::<Self::Element>();

        unsafe {
            let mut result = src;

            let res = Self::as_array_mut(&mut result);
            let src = Self::as_array(&src);
            let indices = IDX::as_array(&indices);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                if !<Self::Mask as MaskRegister>::test(mask, i) {
                    continue;
                }

                res[i] = ptr.add(indices[i].try_into().unwrap_unchecked()).read();
            }

            result
        }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and pointing to memory locations that
    /// can be safely read from based on the register's requirements.
    #[inline(always)]
    unsafe fn gather_z(mask: Storage<Self::Mask>, ptr: *const Self::Element, indices: Storage<IDX>) -> Storage<Self> {
        unsafe { Self::gather_m(Self::EMPTY, mask, ptr, indices) }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and pointing to memory locations that
    /// can be safely written to based on the register's requirements.
    #[inline(always)]
    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<IDX>) {
        unsafe {
            let value = Self::as_array(&value);
            let indices = IDX::as_array(&indices);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                ptr.add(indices[i].try_into().unwrap_unchecked()).write(value[i]);
            }
        }
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and pointing to memory locations that
    /// can be safely written to based on the register's requirements.
    #[inline(always)]
    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<IDX>,
    ) {
        unsafe {
            let value = Self::as_array(&value);
            let indices = IDX::as_array(&indices);

            for i in 0..<Self::Lanes as Unsigned>::USIZE {
                if !<Self::Mask as MaskRegister>::test(mask, i) {
                    continue;
                }

                ptr.add(indices[i].try_into().unwrap_unchecked()).write(value[i]);
            }
        }
    }
}

/// Shuffle registers using an immediate value.
///
/// This MUST support the full lane count of the register,
/// not just 128-bit segments. As a result, this may not map directly
/// to hardware instructions on some ISAs.
pub trait ShuffleRegister: Register {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
}

/// Permute registers using an immediate value.
///
/// This MUST support the full lane count of the register,
/// not just 128-bit segments. As a result, this may not map directly
/// to hardware instructions on some ISAs.
pub trait PermuteRegister: Register {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self>;
}

pub trait BlendRegister: Register {
    fn blend<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
}

#[rustfmt::skip]
#[thermite_macros::register_trait]
pub trait BitshiftRegister: Register<Element: IntegerElement> {
    #[conditional] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self>;
    #[conditional] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self>;

    #[conditional] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self::shl(value, IMM8 as u32) }
    #[conditional] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self::shr(value, IMM8 as u32) }

    /// Indicates if bshli/bshri are supported natively.
    const HAS_WIDE_BYTE_SHIFTS: bool;

    /// Shifts the ENTIRE register left by a constant amount of BYTES,
    /// filling with zeros. This is different from lane-wise shifts, and effectively
    /// treats the register as one large integer.
    #[conditional] fn bshli<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        let arr = Self::as_array_mut(&mut value);
        let lane_width = core::mem::size_of::<Self::Element>() * 8;
        let lanes = <Self::Lanes as Unsigned>::USIZE;

        let skip = (8 * IMM8 as usize) / lane_width;
        let shift = (8 * IMM8 as u16) % lane_width as u16;

        if skip >= lanes {
            return Self::EMPTY;
        }

        if shift == 0 {
            for i in (skip..lanes).rev() {
                arr[i] = arr[i - skip];
            }
            for i in 0..skip {
                arr[i] = Self::Element::ZERO;
            }

            return value;
        }

        let inv_shift = lane_width as u16 - shift;

        let shift: Self::Element = Element::from_u16(shift);
        let inv_shift: Self::Element = Element::from_u16(inv_shift);

        for i in (skip + 1..lanes).rev() {
            arr[i] = (arr[i - skip] << shift) | (arr[i - skip - 1] >> inv_shift);
        }

        arr[skip] = arr[0] << shift;

        if skip > 0 {
            for i in 0..skip {
                arr[i] = Self::Element::ZERO;
            }
        }

        value
    }

    /// Shifts the ENTIRE register right by a constant amount of BYTES,
    /// filling with zeros. This is different from lane-wise shifts, and effectively
    /// treats the register as one large integer.
    #[conditional] fn bshri<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        let arr = Self::as_array_mut(&mut value);
        let lane_width = core::mem::size_of::<Self::Element>() * 8;
        let lanes = <Self::Lanes as Unsigned>::USIZE;

        let skip = (8 * IMM8 as usize) / lane_width;
        let shift = (8 * IMM8 as u16) % lane_width as u16;

        if skip >= lanes {
            return Self::EMPTY;
        }

        if shift == 0 {
            for i in 0..(lanes - skip) {
                arr[i] = arr[i + skip];
            }
            for i in (lanes - skip)..lanes {
                arr[i] = Self::Element::ZERO;
            }

            return value;
        }

        let inv_shift = lane_width as u16 - shift;

        let shift: Self::Element = Element::from_u16(shift);
        let inv_shift: Self::Element = Element::from_u16(inv_shift);

        for i in 0..(lanes - skip - 1) {
            arr[i] = (arr[i + skip] >> shift) | (arr[i + skip + 1] << inv_shift);
        }

        arr[lanes - skip - 1] = arr[lanes - 1] >> shift;

        if skip > 0 {
            for i in (lanes - skip)..lanes {
                arr[i] = Self::Element::ZERO;
            }
        }

        value
    }

    /// Indicates if true variable shifts are supported, or `false` if it
    /// requires a scalar fallback.
    const HAS_TRUE_SHIFTV: bool;

    #[conditional] fn shrv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // Scalar fallback. `shrv` is a *logical* (zero-fill) shift, so use
        // `unsigned_shr`: a plain `>>` on a signed element arithmetic-shifts, which
        // is `srav`, not `shrv`. (Backends with a hardware variable shift override this.)
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_array(&shifts))
        {
            *r = r.logical_shr(*s);
        }

        value
    }

    #[conditional] fn shlv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_array(&shifts))
        {
            *r = *r << *s;
        }

        value
    }

    /// Rotate bits left
    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        Self::bitor(Self::shl(value, shift), Self::shr(value, width - shift))
    }

    /// Rotate bits right
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        Self::bitor(Self::shr(value, shift), Self::shl(value, width - shift))
    }

    /// Rotate bits left by a constant amount
    #[conditional] fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::rol(value, IMM8 as u32)
    }

    /// Rotate bits right by a constant amount
    #[conditional] fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::ror(value, IMM8 as u32)
    }

    #[masked]
    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u16;
        let width_vec = Self::Unsigned::splat(Element::from_u16(width));

        Self::bitor(
            Self::shlv(value, shifts),
            Self::shrv(value, Self::Unsigned::sub(width_vec, shifts)),
        )
    }

    fn rolv_c(mask: Storage<Self::Mask>, value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // zero out the shifts where the mask is not set
        let mask = <<Self::Unsigned as CoreRegister>::Mask as CastMaskRegister<Self::Mask>>::mask_from(mask);
        Self::rolv(value, <Self::Unsigned as CoreRegister>::zz(mask, shifts))
    }

    #[masked]
    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u16;
        let width_vec = Self::Unsigned::splat(Element::from_u16(width));

        Self::bitor(
            Self::shrv(value, shifts),
            Self::shlv(value, Self::Unsigned::sub(width_vec, shifts)),
        )
    }

    fn rorv_c(mask: Storage<Self::Mask>, value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        let mask = <<Self::Unsigned as CoreRegister>::Mask as CastMaskRegister<Self::Mask>>::mask_from(mask);
        Self::rorv(value, <Self::Unsigned as CoreRegister>::zz(mask, shifts))
    }

    #[conditional] fn reverse_bits(mut value: Storage<Self>) -> Storage<Self> {
        // Use hardware byte swapping to handle bit reversals at the byte level and above.
        // This effectively handles s=32, s=16, s=8 for u64/u32/u16 in one go.
        value = Self::swap_bytes(value);

        let mut s = size_of::<Self::Element>() as u32 * 4; // Start with half the bit width
        let mut mask = Self::not(Self::EMPTY); // All bits set

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

            let a = mask;
            let b = Self::shr(value, s);
            let c = Self::shl(value, s);

            // standard select logic: (A & B) | (!A & C)
            value = Self::ternlog::<{ crate::ternlog_imm!((A & B) | (!A & C)) }>(a, b, c);

            s >>= 1;
        }

        // TODO: When implementing AVX-512, we can use the Galois field affine transformation
        // instructions to do sub-byte-level bit reversals more efficiently.

        value
    }
}

/// A trait for registers that can be cast to/from other registers,
/// including of varying element types.
pub trait CastRegister<FROM: CoreRegister>: CoreRegister {
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

/// A narrowing cast that clamps (saturates) out-of-range source values to the
/// destination element's representable range, instead of the wrapping
/// truncation [`CastRegister`] performs.
///
/// Implemented only in the **narrowing, same-signedness** direction
/// (`i64 -> i32 -> i16 -> i8`, `u64 -> u32 -> u16 -> u8`, including skip-level
/// pairs such as `i64 -> i8`). Widening conversions lose nothing and go through
/// [`CastRegister`]; sign-changing conversions are intentionally out of scope
/// (use [`CastRegister`], which wraps).
///
/// # Reference semantics
///
/// Defined by the scalar backend and matched lane-for-lane by every
/// hardware (`pack*`-based) implementation: clamp the source value into
/// `[INTO::MIN, INTO::MAX]`, then convert. For unsigned destinations
/// `INTO::MIN` is `0`, so only the high end is clamped. Saturation is
/// idempotent across nested ranges, so a direct `i64 -> i8` is bit-identical
/// to chaining `i64 -> i32 -> i16 -> i8`.
pub trait SaturatingCastRegister<FROM: CoreRegister>: CoreRegister {
    /// Narrow `value` into `Self`, clamping each lane to `Self`'s element range.
    fn saturating_cast_from(value: Storage<FROM>) -> Storage<Self>;
}

/// A trait for registers that can be reinterpreted as other registers,
/// though this is not a safe operation. This is only available for registers
/// of the same size in bytes. This is enforced simply by the fact that
/// it will only be implemented for registers of the same size.
pub trait BitCastRegister<FROM: CoreRegister>: CoreRegister {
    fn from_bits(value: Storage<FROM>) -> Storage<Self>;
}

/// A trait for registers that can be reinterpreted as other registers, as masks,
/// such that the masks retain 0 or !0 values for the appropriate lanes.
pub trait CastMaskRegister<FROM: CoreRegister>: CoreRegister {
    fn mask_from(value: Storage<FROM>) -> Storage<Self>;
}

#[rustfmt::skip]
#[thermite_macros::register_trait]
pub trait PartialOrdRegister: Register {
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask>;
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask>;

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        let gt = Self::gt(lhs, rhs);
        let eq = Self::eq(lhs, rhs);

        Self::Mask::bitor(gt, eq)
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { Self::gt(rhs, lhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { Self::ge(rhs, lhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { Self::Mask::not(Self::eq(lhs, rhs)) }
}

#[rustfmt::skip]
#[thermite_macros::register_trait]
pub trait NumericRegister:
    PartialOrdRegister<Signed: CastRegister<Self>, Unsigned: CastRegister<Self>, Element: num_traits::NumOps>
    + CastRegister<Self::Signed>
    + CastRegister<Self::Unsigned>
{
    const ZERO: Storage<Self>;
    const ONE: Storage<Self>;
    const TWO: Storage<Self>;

    const MIN: Storage<Self>;
    const MAX: Storage<Self>;

    fn is_all_zero(value: Storage<Self>) -> bool {
        <Self::Mask as MaskRegister>::all(Self::eq(value, Self::ZERO))
    }

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn square(lhs: Storage<Self>) -> Storage<Self> {
        Self::mul(lhs, lhs)
    }

    #[conditional] fn scale(value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        Self::mul(value, Self::splat(scalar))
    }

    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn arg_minmax(value: Storage<Self>) -> (usize, usize) {
        let (min_val, max_val) = Self::min_max_element(value);

        let min = Self::splat(min_val);
        let max = Self::splat(max_val);

        let min = Self::eq(min, value);
        let max = Self::eq(max, value);

        let min = match <Self::Mask as MaskRegister>::native_bitmask(min) {
            Some(mask) => mask.trailing_zeros() as usize,

            #[cfg(feature = "bitvec")]
            None => <Self::Mask as MaskRegister>::bitmask(min).trailing_zeros(),

            #[cfg(not(feature = "bitvec"))]
            None => unreachable!(),
        };

        let max = match <Self::Mask as MaskRegister>::native_bitmask(max) {
            Some(mask) => mask.trailing_zeros() as usize,

            #[cfg(feature = "bitvec")]
            None => <Self::Mask as MaskRegister>::bitmask(max).trailing_zeros(),

            #[cfg(not(feature = "bitvec"))]
            None => unreachable!(),
        };

        (min, max)
    }

    fn sort(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::sort::sort_any::<Self>(value)
    }

    fn min_element(value: Storage<Self>) -> Self::Element;
    fn max_element(value: Storage<Self>) -> Self::Element;

    #[inline(always)]
    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        (Self::min_element(value), Self::max_element(value))
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element;
    fn prod_elements(value: Storage<Self>) -> Self::Element;

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let half = const { <Self::Lanes as Unsigned>::USIZE / 2 };

        let lo = Self::as_array(&lo);
        let hi = Self::as_array(&hi);

        let mut result = Self::EMPTY;

        let out = Self::as_array_mut(&mut result);
        for i in 0..half {
            out[i] = lo[2 * i] + lo[2 * i + 1];
            out[i + half] = hi[2 * i] + hi[2 * i + 1];
        }

        result
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        Self::pairwise_sum(lo, hi)
    }

    /// Effectively the number of lanes in the register, splatted across the lanes.
    fn offset() -> Storage<Self>;
    /// 0, 1, 2, 3, 4, ... etc.
    fn indexed() -> Storage<Self>;
}

#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait SignedRegister: NumericRegister<Element: num_traits::Signed> {
    #[conditional] fn neg(value: Storage<Self>) -> Storage<Self>;
    #[conditional] fn abs(value: Storage<Self>) -> Storage<Self>;

    fn signum(value: Storage<Self>) -> Storage<Self> {
        let is_neg = Self::is_negative(value);
        let is_zero = Self::eq(value, Self::ZERO);

        // `blendv(mask, on_false, on_true)` selects `on_true` where the mask is set:
        // negative -> -1, otherwise +1 (the zero case is fixed up below).
        let sign = Self::blendv(is_neg, Self::ONE, Self::NEG_ONE);

        if const { Self::HAS_EQUAL_SIZE_MASK } {
            // this is almost certainly zero-cost on such platforms
            let is_zero = Self::from_mask(is_zero);

            // so use a bitandnot to zero out the result when is_zero is true
            Self::bitandnot(is_zero, sign)
        } else {
            Self::blendv(is_zero, sign, Self::ZERO)
        }
    }

    #[conditional]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let abs = Self::abs(lhs);

        Self::blendv(Self::is_negative(rhs), abs, Self::neg(abs))
    }

    const NEG_ONE: Storage<Self>;
    const MIN_POSITIVE: Storage<Self>;

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::lt(value, Self::ZERO)
    }

    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ge(value, Self::ZERO)
    }

    /// On platforms where blendv only checks the MSB, this can be optimized to avoid comparisons.
    fn select_negative(value: Storage<Self>, on_neg: Storage<Self>, on_pos: Storage<Self>) -> Storage<Self> {
        // no matter the element type, float or integer, MSB is the sign bit
        Self::blendv(Self::msb_to_mask(value), on_pos, on_neg)
    }
}

use num_traits::{WrappingAdd, WrappingMul};

#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait IntegerRegister: NumericRegister<Element: IntegerElement> + BitshiftRegister {
    #[conditional] fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    #[conditional] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        Self::reduce(value, |a, b| a.wrapping_add(&b))
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        Self::reduce(value, |a, b| a.wrapping_mul(&b))
    }

    #[conditional] fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self>;
    #[conditional] fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self>;
    #[conditional] fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self>;

    const HAS_HARDWARE_POPCNT: bool;

    #[conditional] fn count_ones(value: Storage<Self>) -> Storage<Self>;

    #[conditional] fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    #[conditional] fn leading_zeros(value: Storage<Self>) -> Storage<Self>;
    #[conditional] fn trailing_zeros(value: Storage<Self>) -> Storage<Self>;

    #[conditional] fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    #[conditional] fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

#[thermite_macros::register_trait]
pub trait UnsignedIntegerRegister:
    IntegerRegister<Unsigned = Self, Element: crate::element::UnsignedIntegerElement>
{
    /// Returns `floor(log2(x)) + 1`
    #[conditional]
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::next_power_of_two_m1(value))
    }

    /// Next power of two minus 1
    #[conditional]
    fn next_power_of_two_m1(mut value: Storage<Self>) -> Storage<Self> {
        let width = (size_of::<Self::Element>() * 8) as u32;
        let mut s = 1;

        while s < width {
            value = Self::bitor(value, Self::shr(value, s));

            s <<= 1;
        }

        value
    }

    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {
        // f = (v & (v - 1)) == 0
        Self::eq(Self::ZERO, Self::bitand(value, Self::sub(value, Self::ONE)))
    }

    /// Per-lane inclusive unsigned range test: a mask of `lo <= value <= hi`,
    /// assuming `lo <= hi`.
    ///
    /// Uses the branchless `(value - lo) <= (hi - lo)` trick with *wrapping*
    /// subtraction: when `value < lo` the subtraction wraps to a large value
    /// that fails the `<=` test. The win over the naive `value >= lo & value
    /// <= hi` is a single unsigned compare instead of two (plus an `and`) -
    /// which matters on ISAs that lack a native unsigned compare. It is two
    /// subtracts and one compare in general; when `lo`/`hi` are constants
    /// `hi - lo` folds away, leaving one subtract and one compare - the usual
    /// byte-classification case (digit/alpha/whitespace ranges).
    ///
    /// (Note: this needs wrapping, not saturating, sub - with saturating sub
    /// `value < lo` would give `0 <= hi - lo` and wrongly test true.)
    fn in_range(value: Storage<Self>, lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self::Mask> {
        Self::le(Self::sub(value, lo), Self::sub(hi, lo))
    }

    #[conditional]
    fn parity(mut value: Storage<Self>) -> Storage<Self> {
        let mut shift = size_of::<Self::Element>() as u32 * 4; // Start with half the bit width

        if const { Self::HAS_HARDWARE_POPCNT } {
            // If we have a hardware popcnt, we can just use that.
            value = Self::count_ones(value);
        } else if const { Self::HAS_TRUE_SHIFTV } {
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

    /// Ceiling average: `(a + b + 1) >> 1`, computed without overflow.
    ///
    /// Matches x86 `PAVGB`/`PAVGW` and ARM `vrhadd` semantics.
    #[conditional]
    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b)))
    }

    /// Per-lane unsigned absolute difference `|a - b|`, without overflow.
    ///
    /// Computed as `(a -| b) | (b -| a)` with saturating subtraction: exactly
    /// one of the two saturating subtractions is nonzero (whichever operand is
    /// larger wins), so the `OR` yields `|a - b|` for any unsigned width. On x86
    /// this lowers to the canonical `psubus`/`psubus`/`por` sequence, so no
    /// native override is needed.
    #[conditional]
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::saturating_sub(a, b), Self::saturating_sub(b, a))
    }

    /// Per-lane `N`-dimensional Morton code (Z-order curve index): interleave
    /// the low bits of `N` coordinate vectors into a single value, placing the
    /// bits of `values[d]` at output positions `d, d + N, d + 2N, ...`.
    ///
    /// Each coordinate contributes its low `floor(W / N)` bits, where `W` is the
    /// element bit width; higher input bits are discarded. `N = 2` is the
    /// classic 2D code, `N = 3` the 3D (voxel/octree) code; `N = 1` is the
    /// identity.
    ///
    /// ```math
    /// \mathrm{morton}(v_0, \dots, v_{N-1}) = \bigvee_{d=0}^{N-1} \mathrm{spread}_N(v_d) \ll d
    /// ```
    ///
    /// where `$\mathrm{spread}_N$` sends input bit `i` to output bit `N i`. The
    /// default is a portable `O(log W)` shift/mask bit-spread (the generalized
    /// "magic number" cascade); backends override with hardware bit-deposit
    /// (BMI2 `PDEP`), carryless multiply (`spread_2(x) = clmul(x, x)`), or GFNI
    /// affine transforms where available.
    ///
    /// [`reverse_morton`](Self::reverse_morton) is the inverse.
    ///
    /// The default delegates to the generic
    /// [`morton_cascade`](crate::backend::generic::polyfills::morton_cascade)
    /// bit-spread. A backend override should accelerate the dimensions it has
    /// hardware for (e.g. `N == 2` via carryless multiply) and delegate every
    /// other `N` back to `morton_cascade`, since there is no `super` for a trait
    /// default.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
    }

    /// Per-lane inverse of [`morton`](Self::morton): de-interleave an
    /// `N`-dimensional Morton code back into its `N` coordinate vectors, where
    /// `out[d]` gathers output bits `d, d + N, d + 2N, ...` back into the low
    /// `floor(W / N)` bits.
    ///
    /// The default delegates to
    /// [`reverse_morton_cascade`](crate::backend::generic::polyfills::reverse_morton_cascade);
    /// backends override with hardware bit-extract (BMI2 `PEXT`) or GFNI where
    /// available. Note carryless multiply does *not* invert, so the CLMUL
    /// `morton` fast path has no `reverse_morton` counterpart - de-interleaving
    /// stays on the cascade.
    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
    }
}

#[thermite_macros::register_trait]
pub trait SignedIntegerRegister:
    IntegerRegister<Signed = Self, Element: crate::element::SignedIntegerElement> + SignedRegister
{
    #[conditional]
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self>;

    #[conditional]
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self::sra(value, IMM8 as u32)
    }

    #[conditional]
    fn srav(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_array(&shifts))
        {
            *r = *r >> *s; // r in this context is signed, so this is an arithmetic shift
        }

        value
    }

    /// Floor average: `(a + b) >> 1` rounded toward -∞, computed without overflow.
    #[conditional]
    fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b)))
    }

    /// Ceiling average: `(a + b + 1) >> 1` rounded toward +∞, computed without overflow.
    #[conditional]
    fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b)))
    }

    /// Rounded high-half signed multiply: the fixed-point `Q(W-1)` product
    /// `(a * b + 2^(W-2)) >> (W-1)` keeping the low `W` bits, where `W` is the
    /// element bit width.
    ///
    /// For `i16` lanes this is the Q15 rounded multiply (x86 `PMULHRSW` /
    /// `_mm_mulhrs_epi16`), the workhorse for gain, fades, and window functions
    /// in fixed-point DSP. Unlike [`mulhi`](IntegerRegister::mulhi) it rounds to
    /// nearest rather than truncating, so it avoids the DC bias truncation
    /// introduces. The `MIN * MIN` corner wraps rather than saturating, matching
    /// `PMULHRSW`.
    ///
    /// The default reconstructs the double-width product from
    /// [`mulhi`](IntegerRegister::mulhi)/[`mullo`](IntegerRegister::mullo); ISAs
    /// with a native instruction (SSSE3+) override it for `i16`.
    #[conditional]
    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        let w = (size_of::<Self::Element>() * 8) as u32;
        let lo = Self::mullo(a, b);
        let hi = Self::mulhi(a, b);
        // (hi:lo) is the 2W-bit product P. `(hi << 1) | (lo >>u (W-1))` is
        // floor(P / 2^(W-1)) keeping the low W bits (note: shr is logical here).
        let shifted = Self::bitor(Self::shli::<1>(hi), Self::shr(lo, w - 1));
        // Round to nearest by adding the highest dropped bit (bit W-2 of P).
        let round = Self::bitand(Self::shr(lo, w - 2), Self::ONE);
        Self::add(shifted, round)
    }
}

#[inline(always)]
fn zip_ternary<R: FloatRegister, F>(mut lhs: Storage<R>, rhs: Storage<R>, acc: Storage<R>, f: F) -> Storage<R>
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

/// Native Capability bitflags
///
/// Some backends may provide "native" implementations of certain functions. For example,
/// BigInt/BigFloat may support native ldexp/frexp in ways that will be much faster than bitcasting
/// and manipulating that way.
///
/// GPUs may also have native instructions for transcendentals, which the Math library can take
/// advantage of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct NativeCapability(pub u64);

impl NativeCapability {
    pub const fn has(&self, cap: u64) -> bool {
        (self.0 & cap) == cap
    }

    pub const NONE: Self = Self(0);

    pub const LDEXP: u64 = 1 << 0;
    pub const FREXP: u64 = 1 << 1;

    pub const SIN: u64 = 1 << 2;
    pub const COS: u64 = 1 << 3;
    pub const TAN: u64 = 1 << 4;
    pub const EXP2: u64 = 1 << 5;
    pub const LOG2: u64 = 1 << 6;
    pub const EXP: u64 = 1 << 7;
    pub const LN: u64 = 1 << 8;
    pub const POWF: u64 = 1 << 9;
}

/// A trait for floating-point registers, which notably define associated types for their bitwise integer counterparts.
/// The `Bits` and `SignedBits` associated types allow for efficient bitwise manipulation of floating-point values by treating them as integers,
/// and are notably potentially different from the `Signed` and `Unsigned` associated types from `Register`. Consider a BigFloat register,
/// which may have `Signed` and `Unsigned` associated types be simple non-BigNum integer registers for shifts and whatnot,
/// bit `Bits` and `SignedBits` types would be BigInt registers of the same size as the BigFloat register,
/// allowing for efficient bitwise manipulation of the BigFloat values. This distinction is important.
#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait FloatRegister:
    SignedRegister<
        Element: FloatElementWithBits,

        Signed: CastRegister<Self::SignedBits> + MaskInteroperable<Self::SignedBits, Self::Bits>,
        Unsigned: CastRegister<Self::Bits> + MaskInteroperable<Self::SignedBits, Self::Bits>,
    >
    + FullyInteroperable<Self::Bits, Self::SignedBits>
    + CastRegister<Self::ExtendedPrecision>
{
    /// Bitwise-compatible unsigned integer register type, with the same lane count and element size as `Self`,
    /// where each lane's bits can be manipulated as an integer.
    type Bits: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElementWithBits>::Bits>
        + FullyInteroperable<Self, Self::SignedBits> + CastRegister<Self::Unsigned> + MaskInteroperable<Self::Signed, Self::Unsigned>;

    /// Bitwise-compatible signed integer register type, with the same lane count and element size as `Self`,
    /// where each lane's bits can be manipulated as an integer.
    type SignedBits: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElementWithBits>::SignedBits>
        + FullyInteroperable<Self, Self::Bits> + CastRegister<Self::Signed> + MaskInteroperable<Self::Signed, Self::Unsigned>;

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

    const NATIVE_CAP: NativeCapability;

    /// LLVM sometimes attempts to further autovectorize our vectorized code, and ends up making it far worse.
    /// Inserting this into a tight loop will prevent that from happening,
    /// and it has no effect on the generated code otherwise. No codegen is produced.
    ///
    /// # Safety
    ///
    /// This method is generally safe, but will drastically affect codegen. Use with caution.
    unsafe fn block_autovectorization(_value: &mut Storage<Self>) {}

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
        unreachable!("native_ldexp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        unreachable!("native_frexp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_sin_cos<P: Policy>(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unreachable!("native_sin_cos is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_sin<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_sin is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_cos<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_cos is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_tan<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_tan is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_exp2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp2 is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_log2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_ln2 is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_exp<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_ln<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_log is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_powf<P: Policy>(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {
        unreachable!("native_powf is not implemented for this FloatRegister");
    }

    fn total_order(value: Storage<Self>) -> Storage<Self::SignedBits> {
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
        // and positive numbers - only the sign bit is different.
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

        let shift = const { size_of::<Self::Element>() as u32 * 8 - 1 };
        let signed_bits = <Self::SignedBits as BitCastRegister<Self>>::from_bits(value);
        let is_negative = <Self::SignedBits as SignedIntegerRegister>::sra(signed_bits, shift);
        let mask = <Self::SignedBits as BitshiftRegister>::shri::<1>(is_negative);

        Self::SignedBits::bitxor(signed_bits, mask)
    }

    fn linear_order(value: Storage<Self>) -> Storage<Self::SignedBits> {
        let shift = const { size_of::<Self::Element>() as u32 * 8 - 1 };
        let signed_bits = <Self::SignedBits as BitCastRegister<Self>>::from_bits(value);
        let is_negative = <Self::SignedBits as SignedIntegerRegister>::sra(signed_bits, shift);
        let mask = <Self::SignedBits as BitshiftRegister>::shri::<1>(is_negative);

        Self::SignedBits::sub(Self::SignedBits::bitxor(signed_bits, mask), is_negative)
    }

    fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {
        if let Some(nan_pattern) = <Self::Element as FloatElementWithBits>::NAN_PATTERN {
            // If the type has a specific NaN pattern, we can check for that directly.
            let nan = <Self as BitCastRegister<Self::Bits>>::from_bits(Self::Bits::splat(nan_pattern));

            // This will also imply that comparing values is similar to comparing integers,
            // and it won't trigger a false positive.
            return Self::eq(value, nan);
        }

        // easiest way to check for NaN is to check if it's not equal to itself
        Self::ne(value, value)
    }

    fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {
        if const { !<Self::Element as FloatElement>::HAS_INFINITY } {
            // If the type doesn't support infinity, then there are no infinite values.
            return Self::Mask::FALSY;
        }

        Self::eq(Self::abs(value), Self::INFINITY)
    }

    fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {
        if const { !<Self::Element as FloatElement>::HAS_INFINITY } {
            // If the type doesn't support infinity, then all values are finite.
            return Self::Mask::TRUTHY;
        }

        Self::lt(Self::abs(value), Self::INFINITY)
    }

    fn is_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {
        if const { !<Self::Element as FloatElement>::HAS_SUBNORMALS } {
            // If the type doesn't support subnormals, then there are no subnormal values.
            return Self::Mask::FALSY;
        }

        // we're operating in the integer domain here
        let bits: Storage<Self::Bits> = <Self::Bits as BitCastRegister<Self>>::from_bits(value);

        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits
        let rest = Self::Bits::bitandnot(Self::EXP_MASK, bits); // extract mantissa + sign bits

        // shift mantissa to remove sign bit, and even though it's offset
        // it'll still work since we're just checking for zero
        let mantissa = Self::Bits::shli::<1>(rest);

        // use eq here for both since there's always an instruction for that
        let exp_is_zero = Self::Bits::eq(exp, Self::Bits::ZERO);
        let mantissa_is_zero = Self::Bits::eq(mantissa, Self::Bits::ZERO);

        // float is subnormal if mantissa != 0 && exp == 0, and by using bitandnot we can avoid using ne above
        let is_subnormal = <Self::Bits as CoreRegister>::Mask::bitandnot(mantissa_is_zero, exp_is_zero);

        // convert back to self mask register
        <Self::Mask as CastMaskRegister<<Self::Bits as CoreRegister>::Mask>>::mask_from(is_subnormal)
    }

    fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {
        if const { !<Self::Element as FloatElement>::HAS_SUBNORMALS } {
            // If the type doesn't support subnormals, then there are no subnormal values.
            return Self::eq(value, Self::ZERO);
        }

        // we're operating in the integer domain here
        let bits: Storage<Self::Bits> = <Self::Bits as BitCastRegister<Self>>::from_bits(value);

        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits

        // zero or subnormal if exp == 0, very simple
        let is_zero_or_subnormal = Self::Bits::eq(exp, Self::Bits::ZERO);

        // convert back to float register
        <Self::Mask as CastMaskRegister<<Self::Bits as CoreRegister>::Mask>>::mask_from(is_zero_or_subnormal)
    }

    fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {
        let bits = <Self::Bits as BitCastRegister<Self>>::from_bits(value);

        // "normal" is defined as not zero/subnormal, not infinite, and not NaN
        let exp = Self::Bits::bitand(Self::EXP_MASK, bits); // extract exponent bits

        // exp = 0 implies zero or subnormal
        let exp_is_zero = Self::Bits::eq(exp, Self::Bits::ZERO);

        let exp_is_max: Storage<<Self::Bits as CoreRegister>::Mask> =
            if const { <Self::Element as FloatElement>::HAS_INFINITY } {
                Self::Bits::eq(exp, Self::EXP_MASK) // exp is max implies infinity or NaN
            } else if let Some(nan_pattern) = <Self::Element as FloatElementWithBits>::NAN_PATTERN {
                let nan = <Self as BitCastRegister<Self::Bits>>::from_bits(Self::Bits::splat(nan_pattern));

                // if NaN is represented by a specific pattern
                <<Self::Bits as CoreRegister>::Mask as CastMaskRegister<Self::Mask>>::mask_from(Self::eq(value, nan))
            } else {
                // If the type doesn't support infinity, then it also doesn't have a max exponent pattern.
                // This value should be optimized out of the bitor below
                <<Self::Bits as CoreRegister>::Mask as MaskRegister>::FALSY
            };

        // normal if exp != 0 && exp != max, so 0 < exp < max is the normal range
        let is_not_normal = <Self::Bits as CoreRegister>::Mask::bitor(exp_is_max, exp_is_zero);

        let is_normal = <Self::Bits as CoreRegister>::Mask::not(is_not_normal);

        // convert back to float register
        <Self::Mask as CastMaskRegister<<Self::Bits as CoreRegister>::Mask>>::mask_from(is_normal)
    }

    #[conditional] fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::mul_add(lhs, rhs, acc)
        } else {
            Self::add(Self::mul(lhs, rhs), acc)
        }
    }

    #[conditional] fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::mul_sub(lhs, rhs, acc)
        } else {
            Self::sub(Self::mul(lhs, rhs), acc)
        }
    }

    #[conditional] fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::nmul_add(lhs, rhs, acc)
        } else {
            Self::sub(acc, Self::mul(lhs, rhs))
        }
    }

    #[conditional] fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::nmul_sub(lhs, rhs, acc)
        } else {
            Self::mul_sube(Self::neg(lhs), rhs, acc)
        }
    }

    #[conditional] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = MulAddExt::mul_add(*lhs, rhs, acc);
        })
    }

    #[conditional] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = MulAddExt::mul_sub(*lhs, rhs, acc);
        })
    }

    #[conditional] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = MulAddExt::nmul_add(*lhs, rhs, acc);
        })
    }

    #[conditional] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = MulAddExt::nmul_sub(*lhs, rhs, acc);
        })
    }

    #[conditional] fn sqrt(value: Storage<Self>) -> Storage<Self>;

    #[conditional] fn rcp(value: Storage<Self>) -> Storage<Self> {
        Self::div(Self::ONE, value)
    }

    #[conditional] fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        Self::rcp(Self::sqrt(value))
    }

    const HAS_APPROX_RSQRT: bool;
    const HAS_APPROX_RCP: bool;

    #[conditional] fn floor(value: Storage<Self>) -> Storage<Self>;
    #[conditional] fn ceil(value: Storage<Self>) -> Storage<Self>;
    #[conditional] fn round(value: Storage<Self>) -> Storage<Self>;
    #[conditional] fn trunc(value: Storage<Self>) -> Storage<Self>;

    #[conditional] fn fract(value: Storage<Self>) -> Storage<Self> {
        Self::sub(value, Self::trunc(value))
    }

    #[conditional] fn mul_sign(value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::signed_zero(sign))
    }

    /// Returns a signed zero with the same sign as the given value
    #[conditional] fn signed_zero(value: Storage<Self>) -> Storage<Self> {
        Self::bitand(Self::NEG_ZERO, value)
    }

    #[conditional] fn next_up(value: Storage<Self>) -> Storage<Self> {
        let bits = <Self::Bits as BitCastRegister<Self>>::from_bits(value);
        let abs = <Self::Bits as BitCastRegister<Self>>::from_bits(Self::abs(value));

        let is_nan = Self::is_nan(value);
        let is_inf = Self::eq(value, Self::INFINITY);
        let unchanged = Self::Mask::bitor(is_nan, is_inf);

        // Use bitwise comparison for positive/zero check to handle -0.0 correctly
        // (abs == bits) is true for positive numbers and +0.0, false for negative numbers and -0.0
        let is_positive = Self::Bits::eq(abs, bits);
        let is_zero = Self::Bits::eq(abs, <Self::Bits as BitCastRegister<Self>>::from_bits(Self::ZERO));

        let add = Self::Bits::add(bits, Self::Bits::ONE);
        let sub = Self::Bits::sub(bits, Self::Bits::ONE);

        // If positive, add 1 (magnitude up). If negative, sub 1 (magnitude down towards -inf).
        // blendv(mask, lhs, rhs) -> if mask { rhs } else { lhs }
        let next_bits = Self::Bits::blendv(is_positive, sub, add);

        // If zero, return MIN_POSITIVE (0x1)
        let next_bits = Self::Bits::blendv(is_zero, next_bits, Self::Bits::ONE);

        // cast mask from float mask to bits mask
        let unchanged = <<Self::Bits as CoreRegister>::Mask as CastMaskRegister<Self::Mask>>::mask_from(unchanged);

        <Self as BitCastRegister<Self::Bits>>::from_bits(Self::Bits::blendv(unchanged, next_bits, bits))
    }

    #[conditional] fn next_down(value: Storage<Self>) -> Storage<Self> {
        let bits = <Self::Bits as BitCastRegister<Self>>::from_bits(value);
        let abs = <Self::Bits as BitCastRegister<Self>>::from_bits(Self::abs(value));

        let is_nan = Self::is_nan(value);
        let is_neg_inf = Self::eq(value, Self::NEG_INFINITY);
        let unchanged = Self::Mask::bitor(is_nan, is_neg_inf);

        let is_positive = Self::Bits::eq(abs, bits);
        let is_zero = Self::Bits::eq(abs, <Self::Bits as BitCastRegister<Self>>::from_bits(Self::ZERO));

        let add = Self::Bits::add(bits, Self::Bits::ONE);
        let sub = Self::Bits::sub(bits, Self::Bits::ONE);

        // If positive, sub 1 (magnitude down). If negative, add 1 (magnitude up towards -inf).
        // blendv(mask, lhs, rhs) -> if mask { rhs } else { lhs }
        let next_bits = Self::Bits::blendv(is_positive, add, sub);

        // If zero, return -MIN_POSITIVE (0x80...01)
        let sign_bit = <Self::Bits as BitCastRegister<Self>>::from_bits(Self::NEG_ZERO);
        let min_neg = Self::Bits::bitor(Self::Bits::ONE, sign_bit);

        let next_bits = Self::Bits::blendv(is_zero, next_bits, min_neg);

        // cast mask from float mask to bits mask
        let unchanged = <<Self::Bits as CoreRegister>::Mask as CastMaskRegister<Self::Mask>>::mask_from(unchanged);

        <Self as BitCastRegister<Self::Bits>>::from_bits(Self::Bits::blendv(unchanged, next_bits, bits))
    }

    fn mix(a: Storage<Self>, b: Storage<Self>, t: Storage<Self>) -> Storage<Self> {
        if Self::HAS_TRUE_FMA {
            Self::mul_add(Self::sub(b, a), t, a) // a + (b - a) * t
        } else {
            let t0 = Self::sub(Self::ONE, t); // 1 - t
            Self::add(Self::mul(a, t0), Self::mul(b, t)) // a * (1 - t) + b * t
        }
    }
}

/// Generic branchless decode of a packed float format (`S`) into `f32`, operating entirely on
/// the f32 register's `Bits` (a `u32` lane register). The container is zero-extended to `u32`,
/// the fields are reconstructed with shifts/masks/selects, and the result is bit-cast back to
/// `f32`. Subnormals are decoded denormal-safe (a normal f32 intermediate minus its bias, which
/// is exact and independent of the FPU's flush-to-zero mode). This is the fallback every
/// backend gets; hardware paths (F16C, AVX512-BF16, ...) override `unpack` directly.
#[inline(always)]
fn unpack_packed<S, C, F, B>(values: Storage<C>) -> Storage<F>
where
    S: FloatSpec,
    C: UnsignedIntegerRegister,
    F: FloatRegister<Element = f32, Lanes = C::Lanes, Bits = B> + BitCastRegister<B>,
    B: UnsignedIntegerRegister<Lanes = C::Lanes, Unsigned = B, Element = u32>
        + BitshiftRegister
        + CastRegister<C>
        + BitCastRegister<F>,
{
    type M<B> = <B as CoreRegister>::Mask;

    // Zero-extend the container's bits into a u32 lane register, then split out the fields.
    let h = <B as CastRegister<C>>::cast_from(values);
    let e = B::bitand(B::shr(h, S::MANTISSA_BITS), B::splat(S::EXP_FIELD_MAX));
    let m = B::bitand(h, B::splat(S::MANTISSA_MASK));
    let mant = B::shl(m, S::MANTISSA_SHIFT); // mantissa aligned into f32's 23-bit field

    // Normal: (e + (127 - BIAS)) << 23 | mant.
    let normal = B::bitor(
        B::shl(B::add(e, B::splat(S::EXP_REBIAS as u32)), spec::F32_MANTISSA_BITS),
        mant,
    );

    // Subnormal / zero, denormal-safe: drop the aligned mantissa into a *normal* f32 with
    // exponent field K = 128 - BIAS, then subtract 2^(K-127). Exact (Sterbenz) and never forms a
    // denormal intermediate, so it is correct regardless of the FPU's flush-to-zero state.
    let k = B::splat(((128 - S::EXP_BIAS) as u32) << spec::F32_MANTISSA_BITS);
    let to_f = <F as BitCastRegister<B>>::from_bits;
    let sub_f = F::sub(to_f(B::bitor(mant, k)), to_f(k));
    let subnormal = <B as BitCastRegister<F>>::from_bits(sub_f);

    let mut out = B::blendv(B::eq(e, B::ZERO), normal, subnormal);

    // Non-finite code points. Only the schemes with an all-ones-exponent escape do anything here;
    // `Finite` and `Unchecked` decode every code point as the finite value computed above (the
    // latter deliberately, for speed - so `e_is_max` is never even computed for it).
    if const { matches!(S::SPECIAL, spec::SpecialEncoding::Ieee) } {
        let e_is_max = B::eq(e, B::splat(S::EXP_FIELD_MAX));
        // inf when mantissa is zero, quiet NaN (payload carried up) otherwise.
        let quiet = B::blendv(B::eq(m, B::ZERO), B::splat(spec::F32_IMPLICIT >> 1), B::ZERO);
        let inf_nan = B::bitor(
            B::bitor(B::splat(spec::F32_EXP_FIELD_MAX << spec::F32_MANTISSA_BITS), mant),
            quiet,
        );
        out = B::blendv(e_is_max, out, inf_nan);
    } else if const { matches!(S::SPECIAL, spec::SpecialEncoding::FiniteNanOnly) } {
        // The single NaN code point is the all-ones exponent *and* all-ones mantissa.
        let is_nan = M::<B>::bitand(
            B::eq(e, B::splat(S::EXP_FIELD_MAX)),
            B::eq(m, B::splat(S::MANTISSA_MASK)),
        );
        out = B::blendv(is_nan, out, B::splat(spec::F32_QUIET_NAN));
    }

    if const { S::HAS_SIGN } {
        out = B::bitor(out, B::shl(B::shr(h, S::SIGN_SHIFT), 31));
    }

    <F as BitCastRegister<B>>::from_bits(out)
}

/// Generic branchless encode of an `f32` register into a packed float format (`S`),
/// round-to-nearest-ties-to-even. The mantissa is rounded with a per-lane variable shift
/// (`shlv`/`shrv`) so normal and subnormal results share one path; overflow / non-finite inputs
/// become `±inf` (IEEE), saturate (no-inf schemes), or flush to signed zero
/// ([`Unchecked`](spec::SpecialEncoding::Unchecked)) - all pre-resolved into `S::OVERFLOW_BITS` /
/// `S::NAN_OUT_BITS` so the body has no per-scheme branch for them - and tiny values flush to
/// signed zero. Mirrors the scalar [`FloatSpec::pack`] oracle. Hardware paths override `pack`.
#[inline(always)]
fn pack_packed<S, C, F, B>(values: Storage<F>) -> Storage<C>
where
    S: FloatSpec,
    C: UnsignedIntegerRegister + CastRegister<B>,
    F: FloatRegister<Element = f32, Lanes = C::Lanes, Bits = B> + BitCastRegister<B>,
    B: UnsignedIntegerRegister<Lanes = C::Lanes, Unsigned = B, Element = u32>
        + BitshiftRegister
        + CastRegister<C>
        + BitCastRegister<F>,
{
    type M<B> = <B as CoreRegister>::Mask;

    let fb = <B as BitCastRegister<F>>::from_bits(values);
    let abs = B::bitand(fb, B::splat(0x7FFF_FFFF));
    let f32_exp = B::shr(abs, spec::F32_MANTISSA_BITS); // biased, 0..255
    let f32_mant = B::bitand(abs, B::splat(spec::F32_IMPLICIT - 1));
    let significand = B::bitor(B::splat(spec::F32_IMPLICIT), f32_mant); // 1.<23>, the implicit one set

    let one = B::ONE;
    let rebias = B::splat(S::EXP_REBIAS as u32); // 127 - BIAS, >= 0

    // Target (biased) packed exponent, and how many low significand bits to discard. When the
    // exponent would be <= 0 the result is subnormal: clamp `e` to 0 and discard `(1 - e)` extra
    // bits so normal and subnormal share the single rounding path below.
    let sub = B::le(f32_exp, rebias);
    let extra = B::sub(B::add(rebias, one), f32_exp); // = 1 - e, valid (>= 1) only where `sub`
    let shift = B::blendv(
        sub,
        B::splat(S::MANTISSA_SHIFT),
        B::add(B::splat(S::MANTISSA_SHIFT), extra),
    );
    let e = B::blendv(sub, B::sub(f32_exp, rebias), B::ZERO);

    // A shift of >= 32 discards the whole significand (input is below half the smallest
    // subnormal): flush to zero. Clamp the shift so the variable-shift ops stay well-defined.
    let tiny = B::ge(shift, B::splat(32));
    let shift = B::min(shift, B::splat(31));

    // Round to nearest, ties to even, on the discarded low `shift` bits.
    let keep = B::shrv(significand, shift);
    let rem = B::bitand(significand, B::sub(B::shlv(one, shift), one));
    let halfway = B::shlv(one, B::sub(shift, one));
    let tie_to_odd = M::<B>::bitand(B::eq(rem, halfway), B::eq(B::bitand(keep, one), one));
    let round_up = M::<B>::bitor(B::gt(rem, halfway), tie_to_odd);
    let q = B::add(keep, B::bitand(B::from_mask(round_up), one));

    // Normal magnitude. A rounding carry out of the implicit-bit position bumps the exponent and
    // clears the fraction. (Subnormal magnitude is just `q`: a carry there lands on the smallest
    // normal's bit pattern automatically.)
    let carry = B::ge(B::shr(q, S::MANTISSA_BITS), B::splat(2));
    let e_carried = B::add(e, B::bitand(B::from_mask(carry), one));
    let frac = B::nz(carry, B::bitand(q, B::splat(S::MANTISSA_MASK)));
    let mag_normal = B::bitor(B::shl(e_carried, S::MANTISSA_BITS), frac);
    let mut mag = B::blendv(sub, mag_normal, q);

    // Overflow of a normal (subnormals can't overflow) -> inf / saturate.
    let overflow = M::<B>::bitandnot(sub, B::gt(e_carried, B::splat(S::MAX_FINITE_EXP_FIELD)));
    mag = B::blendv(overflow, mag, B::splat(S::OVERFLOW_BITS));
    mag = B::blendv(tiny, mag, B::ZERO);

    // For the no-infinity schemes, a finite input must never land on the reserved NaN code point
    // (E4M3's S.1111.111); the saturation target is already the largest finite.
    if const { matches!(S::SPECIAL, spec::SpecialEncoding::FiniteNanOnly) } {
        mag = B::blendv(
            B::gt(mag, B::splat(S::MAX_FINITE_BITS)),
            mag,
            B::splat(S::MAX_FINITE_BITS),
        );
    }

    // f32 subnormals are below every target range -> signed zero.
    mag = B::blendv(B::eq(f32_exp, B::ZERO), mag, B::ZERO);

    // f32 inf / NaN. inf shares OVERFLOW_BITS with the overflow case; NaN uses NAN_OUT_BITS.
    let special = B::eq(f32_exp, B::splat(spec::F32_EXP_FIELD_MAX));
    let is_nan = M::<B>::bitandnot(B::eq(f32_mant, B::ZERO), special);
    let is_inf = M::<B>::bitand(special, B::eq(f32_mant, B::ZERO));
    mag = B::blendv(is_inf, mag, B::splat(S::OVERFLOW_BITS));
    mag = B::blendv(is_nan, mag, B::splat(S::NAN_OUT_BITS));

    if const { S::HAS_SIGN } {
        mag = B::bitor(mag, B::shr(B::bitand(fb, B::splat(0x8000_0000)), 31 - S::SIGN_SHIFT));
    }

    <C as CastRegister<B>>::cast_from(mag)
}

/// A `u32`/`u16`/`u8` integer register reinterpreted as a vector of packed floats (`S`:
/// fp16, bf16, fp8, ...), convertible to/from a wider `f32` register `F` of the same lane
/// count. `pack`/`unpack` have generic branchless defaults (see [`unpack_packed`]/
/// [`pack_packed`]); backends override them where hardware exists (e.g. F16C `vcvtph2ps`).
pub trait PackedFloatRegister<
    S: FloatSpec,
    F: FloatRegister<Element = f32, Lanes = Self::Lanes, Bits: CastRegister<Self>>,
>: UnsignedIntegerRegister<Unsigned = Self>
{
    #[inline(always)]
    fn pack(values: Storage<F>) -> Storage<Self>
    where
        Self: CastRegister<F::Bits>,
    {
        pack_packed::<S, Self, F, F::Bits>(values)
    }

    #[inline(always)]
    fn unpack(values: Storage<Self>) -> Storage<F> {
        unpack_packed::<S, Self, F, F::Bits>(values)
    }
}

// The emulated `ArrayRegister` container (`u16x8`, `u8x16`, ...) has no hardware transcoder, so
// it just takes the generic branchless defaults - for every format `S` and every width `N` at
// once. Native backends impl `PackedFloatRegister` for their own concrete register types (which
// are distinct types, so this blanket does not conflict), overriding `pack`/`unpack` with F16C
// etc. where the hardware exists.
impl<S, C, const N: usize> PackedFloatRegister<S, array::ArrayRegister<f32, N>> for array::ArrayRegister<C, N>
where
    S: FloatSpec,
    C: CoreRegister,
    array::ArrayRegister<C, N>: UnsignedIntegerRegister<Unsigned = Self>,
    array::ArrayRegister<f32, N>: FloatRegister<Element = f32, Lanes = Self::Lanes, Bits: CastRegister<Self>>,
{
}
