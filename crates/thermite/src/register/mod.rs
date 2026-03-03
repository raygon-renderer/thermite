//! Low-level SIMD Register interface

macro_rules! s {
    ($ty:ty: $a:expr, [$($idx:literal),* $(,)?]) => {
        <$ty as SwizzleRegister>::permutev($a, generic_array::arr![$($idx),*])
    };

    ($ty:ty: $a:expr, $b:expr, [$($idx:literal),* $(,)?]) => {
        <$ty as SwizzleRegister>::swizzle($a, $b, generic_array::arr![$($idx),*])
    };
}

pub mod dp;
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
    element::{FloatElementWithBits, IntegerElement},
    generic::ops::MulAddExt,
    isa::InstructionSet,
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
    /// Used in [`Mask::bitmask()`](crate::Mask::bitmask).
    type BitmaskLength: ArrayLength;

    type BitmaskStorage: bitvec::view::BitViewSized<Store = u32>;
}

impl<T> Lanes for T
where
    T: ArrayLength + core::ops::Shl<typenum::B1> + core::ops::Add<RoundUpConst> + core::ops::Shr<typenum::B1>,
    typenum::Sum<T, RoundUpConst>: core::ops::Div<BitsPerWord>,
    MaskWordCount<T>: ArrayLength,
    GenericArray<u32, MaskWordCount<T>>: bitvec::view::BitViewSized<Store = u32>,
{
    type BitmaskLength = MaskWordCount<T>;
    type BitmaskStorage = GenericArray<u32, Self::BitmaskLength>;
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
    type Storage: Sized + Copy;
    type Mask: MaskRegister<Lanes = Self::Lanes>;

    /// Indicates if the register is emulated in software.
    const IS_EMULATED: bool;

    const ISA: InstructionSet;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self>;

    /// Selects elements from `value` where `mask` is true, and zeroes elsewhere.
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self>;

    /// Selects elements from `value` where `mask` is false, and zeroes elsewhere.
    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::z(<Self::Mask as BitwiseRegister>::not(mask), value)
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

/// Mask registers, which operate on boolean values, though not necessarily
/// with `bool` storage.
///
/// Their storage type may differ from that of regular registers, or even between
/// similar vector types between architectures. E.g., AVX-512 mask registers
/// use 16-bit integers as storage, while AVX2 uses full SIMD registers with
/// all `0` and `1` bits to represent `false` and `true`, respectively.
#[thermite_macros::register_trait]
pub trait MaskRegister: BitwiseRegister<Mask = Self> + CastMaskRegister<Self> {
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

    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>);

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
}

/// SIMD Register trait where each Element implements the [`Element`] trait.
#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait Register:
    BitwiseRegister<
    Mask: CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>
              + CastMaskRegister<<Self::Signed as CoreRegister>::Mask>,
>
{
    type Element: Element;

    const HAS_EQUAL_SIZE_MASK: bool;

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self>;

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
            Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<<Self::Signed as CoreRegister>::Mask>,
        > + CastRegister<Self::Signed>
        + BitCastRegister<Self::Signed>;

    /// SignedBits integer register type with the same number of lanes.
    type Signed: SignedIntegerRegister<
            Unsigned = Self::Unsigned,
            Signed = Self::Signed,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::Signed,
            Mask: CastMaskRegister<Self::Mask> + CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>,
        > + CastRegister<Self::Unsigned>
        + BitCastRegister<Self::Unsigned>;

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

    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>);
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>);

    /// Swap the byte order of each element in the register.
    #[conditional] fn swap_bytes(value: Storage<Self>) -> Storage<Self>;
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

const fn is_power_of_2(n: u32) -> bool {
    (n & (n - 1)) == 0
}

#[thermite_macros::register_trait]
pub trait SwizzleRegister: Register {
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
        // Scalar fallback
        for (r, s) in Self::as_array_mut(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_array(&shifts))
        {
            *r = *r >> *s;
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
        Self::rolv(value, <Self::Unsigned as CoreRegister>::z(mask, shifts))
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
        Self::rorv(value, <Self::Unsigned as CoreRegister>::z(mask, shifts))
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
    PartialOrdRegister<Signed: CastRegister<Self>, Unsigned: CastRegister<Self>>
    + CastRegister<Self::Signed>
    + CastRegister<Self::Unsigned>
{
    const ZERO: Storage<Self>;
    const ONE: Storage<Self>;
    const TWO: Storage<Self>;

    const MIN: Storage<Self>;
    const MAX: Storage<Self>;

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn square(lhs: Storage<Self>) -> Storage<Self> {
        Self::mul(lhs, lhs)
    }

    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

    fn sort(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::sort::sort_any::<Self>(value)
    }

    fn min_element(value: Storage<Self>) -> Self::Element;
    fn max_element(value: Storage<Self>) -> Self::Element;
    fn sum_elements(value: Storage<Self>) -> Self::Element;
    fn prod_elements(value: Storage<Self>) -> Self::Element;

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

        if const { Self::HAS_EQUAL_SIZE_MASK } {
            // this is almost certainly zero-cost on such platforms
            let is_zero = Self::from_mask(is_zero);

            // so use a bitandnot to zero out the result when is_zero is true
            Self::bitandnot(is_zero, Self::blendv(is_neg, Self::NEG_ONE, Self::ONE))
        } else {
            Self::blendv(is_zero, Self::ZERO, Self::blendv(is_neg, Self::NEG_ONE, Self::ONE))
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

    #[conditional] fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        Self::reduce(value, |a, b| a.wrapping_add(&b))
    }

    #[conditional] fn wrapping_product(value: Storage<Self>) -> Self::Element {
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

    #[conditional]
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

#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait FloatRegister:
    SignedRegister<Element: FloatElementWithBits>
    + FullyInteroperable<Self::Bits, Self::SignedBits>
    + CastRegister<Self::ExtendedPrecision>
{
    type Bits: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElementWithBits>::Bits>
        + FullyInteroperable<Self, Self::SignedBits>;
    type SignedBits: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElementWithBits>::SignedBits>
        + FullyInteroperable<Self, Self::Bits>;

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

    unsafe fn block_autovectorization(_value: &mut Storage<Self>) {}

    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
        unreachable!("native_ldexp is not implemented for this FloatRegister");
    }

    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        unreachable!("native_frexp is not implemented for this FloatRegister");
    }

    unsafe fn native_sin_cos(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unreachable!("native_sin_cos is not implemented for this FloatRegister");
    }

    unsafe fn native_sin(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_sin is not implemented for this FloatRegister");
    }

    unsafe fn native_cos(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_cos is not implemented for this FloatRegister");
    }

    unsafe fn native_tan(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_tan is not implemented for this FloatRegister");
    }

    unsafe fn native_exp2(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp2 is not implemented for this FloatRegister");
    }

    unsafe fn native_log2(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_ln2 is not implemented for this FloatRegister");
    }

    unsafe fn native_exp(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp is not implemented for this FloatRegister");
    }

    unsafe fn native_ln(value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_log is not implemented for this FloatRegister");
    }

    unsafe fn native_powf(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {
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

        let shift = const { size_of::<Self::Element>() as u32 * 8 - 1 };
        let signed_bits = <Self::SignedBits as BitCastRegister<Self>>::from_bits(value);
        let is_negative = <Self::SignedBits as SignedIntegerRegister>::sra(signed_bits, shift);
        let mask = <Self::SignedBits as BitshiftRegister>::shri::<1>(is_negative);

        Self::SignedBits::bitxor(signed_bits, mask)
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
}
