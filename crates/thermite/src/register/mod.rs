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

pub use crate::backend::generic::polyfills::StreamGroup;

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

/// Build a lane-alternating sign-bit constant for a float register: a value that
/// is `-0.0` on lanes of one parity and `+0.0` on the other, so that
/// `bitxor(x, ...)` flips the sign of `x` on exactly those lanes.
///
/// With `neg_on_even = true` this yields `[-0.0, +0.0, -0.0, +0.0, ...]`
/// (`ALT_NEG` - even lanes flip); with `false` it yields `[+0.0, -0.0, ...]`
/// (`ALT_POS` - odd lanes flip). Materialized entirely at compile time (a plain
/// constant load at runtime), so `addsub`/`fmaddsub`/`fmsubadd` need no runtime
/// shuffle to construct their mask.
///
/// Works for every float register - including the emulated `ArrayRegister` and
/// `ReducedRegister` widths - because it treats the storage as a flat run of
/// `R::Element` lanes, exactly like [`reg_splat`].
#[inline(always)]
pub(crate) const fn alt_sign_reg<R: FloatRegister>(neg_on_even: bool) -> Storage<R> {
    // Start from all `+0.0` (zeroed storage) and copy the element-wise `-0.0`
    // out of NEG_ZERO into every lane of the selected parity.
    let neg = <R as FloatRegister>::NEG_ZERO;
    let mut dst = R::EMPTY;

    // SAFETY: contiguous element storage, same assumption as `reg_splat`.
    unsafe {
        let dstp = &mut dst as *mut Storage<R> as *mut R::Element;
        let negp = &neg as *const Storage<R> as *const R::Element;

        let mut i = 0;
        while i < <R::Lanes as typenum::Unsigned>::USIZE {
            if (i % 2 == 0) == neg_on_even {
                dstp.add(i).write(negp.add(i).read());
            }
            i += 1;
        }
    }

    dst
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
///
/// Every register advertises its owning backend via the [`HasIsa`](crate::simd::HasIsa)
/// supertrait (`impl HasIsa for MyReg { type Native = X86V3; }`, usually via the
/// `impl_has_isa!` macro). Emulated registers forward the register they are built
/// from, so `ArrayRegister<F32x4V1, 2>` reports `X86V1` while `ArrayRegister<i16, 2>`
/// reports `Scalar`. This is what lets `#[thermite::dispatch(R)]` work over bare
/// register types.
pub trait CoreRegister: 'static + Sized + crate::simd::HasIsa {
    type Lanes: Lanes;
    type Storage: Sized + Copy + core::fmt::Debug;
    type Mask: MaskRegister<Lanes = Self::Lanes>;

    /// Number of lanes in the register, as a runtime value.
    ///
    /// Today this is always `Self::Lanes::USIZE`; prefer it in slice lengths and
    /// loop bounds for the same forward-compatibility reasons as
    /// `GenericVector::lanes()`.
    #[inline(always)]
    fn lanes() -> usize {
        <Self::Lanes as Unsigned>::USIZE
    }

    /// Indicates if the register is emulated in software.
    const IS_EMULATED: bool;

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

    /// Whether [`ternlog`](Self::ternlog) is a single native instruction
    /// (AVX-512 `vpternlog{d,q}`) rather than the DNF polyfill below.
    ///
    /// Kernels with a choice between a mask + ternlog bit assembly and a
    /// `blendv`-style select chain should fork on this: the polyfill expands
    /// to up to 8 DNF terms, so below AVX-512 the blends win, while a native
    /// ternlog collapses the whole assembly into one instruction per term.
    /// Measured on znver3 (ldexp's checked tail): 3.8 cyc/iter for the blend
    /// form against 5.8 for the ternlog form.
    const HAS_NATIVE_TERNLOG: bool = false;

    /// const A = 0xF0, B = 0xCC, C = 0xAA
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        let mut acc = Self::EMPTY;

        if const { IMM == 0xCA } {
            // Special case for select pattern `a ? b : c` to improve debug builds
            return Self::bitor(Self::bitand(a, b), Self::bitandnot(a, c));
        }

        // Combine cases using Disjunctive Normal Form (DNF)
        macro_rules! case {
            (0,         $expr:expr) => { if const { (IMM & (1 << 0))    != 0 } { acc = $expr; } };
            ($bit:expr, $expr:expr) => { if const { (IMM & (1 << $bit)) != 0 } { acc = Self::bitor(acc, $expr); } };
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
        if const { (IMM & (1 << 0)) != 0 } { acc = Self::not(Self::bitor(a, b)); } // Case 0: inputs are 0, 0, simplified
        if const { (IMM & (1 << 1)) != 0 } { acc = Self::bitor(acc, Self::bitandnot(a, b)); } // Case 1: inputs are 0, 1
        if const { (IMM & (1 << 2)) != 0 } { acc = Self::bitor(acc, Self::bitandnot(b, a)); } // Case 2: inputs are 1, 0
        if const { (IMM & (1 << 3)) != 0 } { acc = Self::bitor(acc, Self::bitand(a, b)); } // Case 3: inputs are 1, 1

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
            for i in 0..Self::lanes() {
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

    /// Build a mask from a packed integer bitmask, bit `i` driving lane `i`
    /// (lane 0 in the least-significant bit) - the inverse of
    /// [`native_bitmask`](Self::native_bitmask).
    ///
    /// Bits at or above the lane count **must** be ignored by every
    /// implementation; callers rely on that to hand a wider bitmask straight
    /// through (`ArrayRegister` shifts one word per sub-register without
    /// re-masking). Masks wider than 64 lanes take their low 64 lanes from
    /// `bitmask` and leave everything above lane 63 `false` - for those, use
    /// [`from_bitmask`](Self::from_bitmask) instead.
    ///
    /// The default is a per-lane [`set`](Self::set) loop; every backend with a
    /// broadcast-and-compare sequence overrides it.
    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        let lanes = <Self::Lanes as Unsigned>::USIZE.min(64);

        let mut result = Self::FALSY;

        let mut i = 0;
        while i < lanes {
            if (bitmask >> i) & 1 != 0 {
                result = Self::set(result, i, true);
            }
            i += 1;
        }

        result
    }

    /// Build a mask from a [`bitvec`] bit array, one bit per lane - the inverse
    /// of [`bitmask`](Self::bitmask).
    ///
    /// Unlike [`from_native_bitmask`](Self::from_native_bitmask) this covers
    /// masks of any width. `bits` shorter than the lane count is allowed: the
    /// lanes it does not reach are `false`.
    #[cfg(feature = "bitvec")]
    fn from_bitmask(bits: &bitvec::slice::BitSlice<u32>) -> Storage<Self> {
        let lanes = <Self::Lanes as Unsigned>::USIZE.min(bits.len());

        if const { <Self::Lanes as Unsigned>::USIZE <= 64 } {
            // Pack into one word so a native `from_native_bitmask` is used.
            let mut bitmask = 0u64;
            for i in bits[..lanes].iter_ones() {
                bitmask |= 1 << i;
            }
            Self::from_native_bitmask(bitmask)
        } else {
            let mut result = Self::FALSY;
            for i in bits[..lanes].iter_ones() {
                result = Self::set(result, i, true);
            }
            result
        }
    }

    // The `else` arms below are only reached by masks whose `native_bitmask`
    // returns `None` (wider than 64 lanes). The only such type today,
    // `ArrayRegister`, overrides the `*_one` forms with a sub-register scan, so
    // these fall back to the canonical `bitmask()` word-scan (the same packing
    // `bitmask()` itself uses) rather than poking one lane at a time - dropping
    // to the per-lane `test` loop only when `bitvec` is unavailable.

    /// Single-register form of [`first_set`](Self::first_set).
    ///
    /// The N-ary forms below default to scanning with this; a backend that
    /// overrides them for some shapes still routes leftovers here, so this is
    /// the one that must always be correct.
    fn first_set_one(value: Storage<Self>) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            let bm = bm & lane_bitmask(lanes);
            (bm != 0).then(|| bm.trailing_zeros() as usize)
        } else {
            #[cfg(feature = "bitvec")]
            {
                Self::bitmask(value).first_one()
            }
            #[cfg(not(feature = "bitvec"))]
            {
                (0..lanes).find(|&i| Self::test(value, i))
            }
        }
    }

    /// Single-register form of [`last_set`](Self::last_set).
    fn last_set_one(value: Storage<Self>) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            let bm = bm & lane_bitmask(lanes);
            (bm != 0).then(|| 63 - bm.leading_zeros() as usize)
        } else {
            #[cfg(feature = "bitvec")]
            {
                Self::bitmask(value).last_one()
            }
            #[cfg(not(feature = "bitvec"))]
            {
                (0..lanes).rev().find(|&i| Self::test(value, i))
            }
        }
    }

    /// Single-register form of [`count_set`](Self::count_set).
    fn count_set_one(value: Storage<Self>) -> usize {
        let lanes = <Self::Lanes as Unsigned>::USIZE;
        if let Some(bm) = Self::native_bitmask(value) {
            (bm & lane_bitmask(lanes)).count_ones() as usize
        } else {
            #[cfg(feature = "bitvec")]
            {
                Self::bitmask(value).count_ones()
            }
            #[cfg(not(feature = "bitvec"))]
            {
                (0..lanes).filter(|&i| Self::test(value, i)).count()
            }
        }
    }

    /// Index of the lowest lane set to `true`, or `None` if every lane is
    /// `false`. A SIMD find-first: combined with a comparison this is `memchr`.
    ///
    /// `values` is treated as one concatenated mask, `values[i]` occupying
    /// lanes `i * LANES .. (i + 1) * LANES`. Unlike
    /// [`count_set`](Self::count_set) this is **order-preserving**, so an
    /// override may not use a lane-scrambling narrowing pack without
    /// restitching the lane order first.
    fn first_set<const N: usize>(values: [Storage<Self>; N]) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;

        let mut i = 0;
        while i < N {
            if let Some(idx) = Self::first_set_one(values[i]) {
                return Some(i * lanes + idx);
            }
            i += 1;
        }

        None
    }

    /// Index of the highest lane set to `true`, or `None` if every lane is
    /// `false` (a find-last). Concatenation order and the order-preservation
    /// requirement are as described on [`first_set`](Self::first_set).
    fn last_set<const N: usize>(values: [Storage<Self>; N]) -> Option<usize> {
        let lanes = <Self::Lanes as Unsigned>::USIZE;

        let mut i = N;
        while i > 0 {
            i -= 1;
            if let Some(idx) = Self::last_set_one(values[i]) {
                return Some(i * lanes + idx);
            }
        }

        None
    }

    /// Total number of lanes set to `true` across all of `values` (population
    /// count of the concatenated mask).
    ///
    /// **Lane order across `values` is unspecified and irrelevant.** A
    /// population count cannot observe it, which is what lets an implementation
    /// merge several masks with a saturating narrowing pack - scrambling the
    /// lane order in the process - and extract a single bitmask, instead of one
    /// bitmask extraction and popcount per register. Any override is free to
    /// exploit that; nothing here may depend on where a given lane lands.
    fn count_set<const N: usize>(values: [Storage<Self>; N]) -> usize {
        let mut total = 0;

        let mut i = 0;
        while i < N {
            total += Self::count_set_one(values[i]);
            i += 1;
        }

        total
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
        Self::splat(Self::as_slice(&value)[idx])
    }

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        // The default reads size_of::<Storage>() bytes, but the safety contract only
        // promises Lanes * size_of::<Element>() -- padded storage must override.
        const {
            assert!(
                size_of::<Storage<Self>>() == (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Size mismatch between register storage and array of elements"
            );
        }

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
            let res = Self::as_mut_slice(&mut result);

            for i in 0..Self::lanes() {
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

    /// # SAFETY
    ///
    /// The pointer must be valid, aligned, and point to a memory location
    /// of at least length `Self::Lanes::USIZE * size_of::<Self::Element>()`.
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        // The default writes size_of::<Storage>() bytes, but the safety contract only
        // promises Lanes * size_of::<Element>() -- padded storage must override.
        const {
            assert!(
                size_of::<Storage<Self>>() == (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Size mismatch between register storage and array of elements"
            );
        }

        // SAFETY: This is safe as long as the pointer is valid, aligned, and of the correct length.
        unsafe { core::ptr::write(ptr as *mut Storage<Self>, value) }
    }

    /// # Safety
    ///
    /// The pointer must be valid, aligned, and point to a memory location where, when the mask is true,
    /// is valid for writing a value of type `Self::Element`.
    ///
    /// The memory locations where the mask is false are not accessed.
    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe {
            let res = Self::as_slice(&value);

            for i in 0..Self::lanes() {
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

    /// Radix-`N` de-interleave: the generic sibling of
    /// [`InterleaveRegister::deinterleave`] (`N == 2`). Treats the `N` inputs as
    /// one contiguous `N * LANES` span and splits it by residue mod `N`:
    /// `out[r][lane] == concat(inputs)[lane * N + r]`.
    ///
    /// `N` is inferred from the array length, so radix-2/3 call sites need no
    /// turbofish: `R::deinterleave_radix([a, b])` is the 2-way split. The default
    /// forwards `N == 2` to the required [`deinterleave`](InterleaveRegister::deinterleave)
    /// primitive and sends every other `N` to the single-round permute+blend
    /// gather ([`deinterleave_any`](crate::backend::generic::polyfills::deinterleave_any)).
    /// Backends with a native radix-3 sequence (NEON `TBL3`, an x86 shuffle
    /// network) override this via `impl_native_radix3!` to add an `N == 3` arm;
    /// this radix-3 primitive is what the `2^a * 3^b` part of
    /// [`load_deinterleaved`](Self::load_deinterleaved) rides on.
    ///
    /// This is a primitive for small, fixed radices; the tuned mixed-radix engine
    /// for arbitrary `N` is [`load_deinterleaved`](Self::load_deinterleaved).
    fn deinterleave_radix<const N: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        crate::backend::generic::polyfills::deinterleave_radix_default::<Self, N>(inputs)
    }

    /// Radix-`N` interleave - the exact inverse of
    /// [`deinterleave_radix`](Self::deinterleave_radix):
    /// `concat(out)[q * N + r]` is lane `q` of the `r`-th input.
    ///
    /// Same dispatch as [`deinterleave_radix`](Self::deinterleave_radix): `N == 2`
    /// forwards to [`interleave`](InterleaveRegister::interleave), a native radix-3
    /// override (via `impl_native_radix3!`) handles `N == 3`, and any other `N`
    /// uses [`interleave_any`](crate::backend::generic::polyfills::interleave_any).
    fn interleave_radix<const N: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        crate::backend::generic::polyfills::interleave_radix_default::<Self, N>(inputs)
    }

    /// Group-granularity radix-`N` de-interleave: the two-axis unification of
    /// [`deinterleave_radix`](Self::deinterleave_radix) (`GROUP == 1`) and
    /// [`deinterleave_by`](Self::deinterleave_by) (`N == 2`). Each register is
    /// viewed as `LANES / GROUP` groups of `GROUP` consecutive elements, and the `N`
    /// inputs' group sequences are split by residue mod `N`:
    /// `out[r].group[q] == concat_groups(inputs)[q * N + r]`, where each group moves
    /// as a unit and is never split.
    ///
    /// The square case `N == LANES / GROUP` is a **register-array transpose** of
    /// `GROUP`-wide elements: `out[r].group[q] == inputs[q].group[r]`. In particular
    /// `deinterleave_radix_by::<4, 2>` on an 8-lane f32 register is the 4x4
    /// interleaved-complex transpose (four `unpacklo/hi_pd` + four `permute2f128` =
    /// 8 ops on AVX2), and `deinterleave_radix_by::<4, 1>` on f64x4 is the plain 4x4
    /// `f64` transpose - the natural primitives for FFT codelets and small matrices.
    ///
    /// The default forwards `GROUP == 1` to [`deinterleave_radix`](Self::deinterleave_radix)
    /// (inheriting its native radix-2/3 paths) and `N == 2` to
    /// [`deinterleave_by`](Self::deinterleave_by), and sends the general case to a
    /// lane-wise fallback. Backends override the `(N, GROUP)` shapes they do natively.
    /// `GROUP` must divide `LANES`.
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        crate::backend::generic::polyfills::deinterleave_radix_by_default::<Self, N, GROUP>(inputs)
    }

    /// The exact inverse of [`deinterleave_radix_by`](Self::deinterleave_radix_by):
    /// `concat_groups(out)[q * N + r]` is group `q` of the `r`-th input. For the
    /// square case it is the same register-array transpose (which is its own
    /// inverse). Same dispatch as [`deinterleave_radix_by`](Self::deinterleave_radix_by).
    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        crate::backend::generic::polyfills::interleave_radix_by_default::<Self, N, GROUP>(inputs)
    }

    /// Load `N` interleaved (array-of-structures) streams and de-interleave them
    /// into `N` registers: reads `N * LANES` contiguous elements starting at
    /// `ptr`, and returns `out` such that `out[j]` holds every `j`-th element,
    /// i.e. `out[j][lane] == ptr[lane * N + j]`.
    ///
    /// This is the AoS -> SoA load. `N == 3` over `f32` is the classic case:
    /// `xyzxyzxyz...` in memory becomes one register each of `xxx`, `yyy`, `zzz`.
    ///
    /// **Any `N >= 1`.** The pointer needs **no alignment** beyond that of
    /// `Element` - ARM's structural loads (`LD2`/`LD3`/`LD4`) have no alignment
    /// requirement on AArch64, and the portable path uses unaligned loads.
    ///
    /// The default loads `N` contiguous registers and hands them to
    /// [`deinterleave_n`](crate::backend::generic::polyfills::deinterleave_n),
    /// a mixed-radix stage engine: a radix-2 butterfly of the native 2-way
    /// `deinterleave` and radix-3 rounds of [`deinterleave_radix::<3>`](Self::deinterleave_radix)
    /// cover the `2^a * 3^b` part of `N`, and a permute+blend gather stage
    /// handles any leftover factor. A backend with true structural loads
    /// overrides this for the widths it supports.
    ///
    /// Whether [`load_deinterleaved`](Self::load_deinterleaved) /
    /// [`store_interleaved`](Self::store_interleaved) lower to true structural
    /// memory instructions (ARM `LD2`/`LD3`/`LD4` + `ST2`/`ST3`/`ST4`, where
    /// the transpose happens in the load/store unit) for small stream counts,
    /// rather than to plain loads plus a register shuffle network.
    ///
    /// The grouped memory ops use this to pick their strategy: with structural
    /// hardware, splitting a grouped load into per-chunk structural loads is a
    /// clear win; without it, the chunk loads are shuffles anyway and the flat
    /// full-width engine is measurably tighter (an x86 `Dual<f32x8, 2>` pair
    /// load: 47 instructions flat vs 55 per-chunk).
    const HAS_STRUCTURAL_MEMOPS: bool = false;

    /// # Safety
    ///
    /// `ptr` must be valid for reads of `N * LANES` elements.
    unsafe fn load_deinterleaved<const N: usize>(ptr: *const Self::Element) -> [Storage<Self>; N] {
        const { assert!(N >= 1) };

        let lanes = Self::lanes();

        let mut src = [Self::EMPTY; N];
        for (i, s) in src.iter_mut().enumerate() {
            *s = unsafe { Self::load_unaligned(ptr.add(i * lanes)) };
        }

        crate::backend::generic::polyfills::deinterleave_n::<Self, N>(src)
    }

    /// Interleave `N` registers and store them as a contiguous
    /// array-of-structures: writes `N * LANES` elements starting at `ptr` such
    /// that `ptr[lane * N + j] == values[j][lane]`.
    ///
    /// The SoA -> AoS store, and the exact inverse of
    /// [`load_deinterleaved`](Self::load_deinterleaved). Any `N >= 1`, same
    /// alignment freedom, same portable strategy
    /// ([`interleave_n`](crate::backend::generic::polyfills::interleave_n)).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `N * LANES` elements.
    unsafe fn store_interleaved<const N: usize>(ptr: *mut Self::Element, values: [Storage<Self>; N]) {
        const { assert!(N >= 1) };

        let lanes = Self::lanes();

        let out = crate::backend::generic::polyfills::interleave_n::<Self, N>(values);

        for (i, o) in out.iter().enumerate() {
            unsafe { Self::store_unaligned(ptr.add(i * lanes), *o) };
        }
    }

    /// Load `M` interleaved AoS records of `C` components each and de-interleave
    /// them: reads `M * C * LANES` contiguous elements, and `out[j][c]` holds
    /// component `c` of record `j`, i.e.
    /// `out[j][c][lane] == ptr[lane * M * C + j * C + c]`.
    ///
    /// The array sibling of
    /// [`load_deinterleaved_grouped`](Self::load_deinterleaved_grouped), keyed on
    /// the component COUNT rather than the count minus one. Both exist because
    /// stable Rust can compute neither `C = TAIL + 1` nor `TAIL = C - 1` as a
    /// const-generic argument, so each caller uses whichever its own const
    /// generic already spells - see
    /// [`deinterleave_arrays`](crate::backend::generic::polyfills::deinterleave_arrays).
    ///
    /// This is the natural spelling for geometry: an AoS `[[f32; 3]]` of points
    /// is `M = 1, C = 3`, and a ray (origin + direction) is `M = 2, C = 3`.
    ///
    /// Two strategies, chosen at compile time. With structural loads
    /// ([`HAS_STRUCTURAL_MEMOPS`](Self::HAS_STRUCTURAL_MEMOPS)), each chunk of
    /// `LANES` records is loaded by [`load_deinterleaved`](Self::load_deinterleaved)
    /// at radix `C` - an `LD2`/`LD3`/`LD4`, transposing in the load unit - and one
    /// radix-`M` register de-interleave per component re-sorts chunk order into
    /// stream order. (No dispatch ladder is needed here, unlike the grouped form:
    /// `C` IS the chunk radix, so it passes straight through as the const-generic
    /// argument.) Otherwise the whole `M * C`-stream problem goes to the flat
    /// shuffle engine in one go, which measures tighter when the chunk loads would
    /// be shuffles anyway.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `M * C * LANES` elements.
    unsafe fn load_deinterleaved_arrays<const M: usize, const C: usize>(
        ptr: *const Self::Element,
    ) -> [[Storage<Self>; C]; M] {
        const { assert!(M >= 1 && C >= 1) };

        let lanes = Self::lanes();

        if const { Self::HAS_STRUCTURAL_MEMOPS && C <= 4 } {
            // comp[c][k] = component c of chunk k (records k*LANES .. (k+1)*LANES).
            let mut comp = [[Self::EMPTY; M]; C];

            let mut k = 0;
            while k < M {
                let chunk = unsafe { Self::load_deinterleaved::<C>(ptr.add(k * C * lanes)) };

                let mut c = 0;
                while c < C {
                    comp[c][k] = chunk[c];
                    c += 1;
                }
                k += 1;
            }

            // Per-component radix-M de-interleave: chunk-order record
            // q = k * LANES + lane becomes stream-order q = lane * M + j.
            let mut out = [[Self::EMPTY; C]; M];

            let mut c = 0;
            while c < C {
                let streams = crate::backend::generic::polyfills::deinterleave_n::<Self, M>(comp[c]);

                let mut j = 0;
                while j < M {
                    out[j][c] = streams[j];
                    j += 1;
                }
                c += 1;
            }

            out
        } else {
            let mut buf = [[Self::EMPTY; C]; M];

            {
                let flat = crate::backend::generic::polyfills::flat_arrays_mut(&mut buf);
                let mut i = 0;
                while i < M * C {
                    flat[i] = unsafe { Self::load_unaligned(ptr.add(i * lanes)) };
                    i += 1;
                }
            }

            crate::backend::generic::polyfills::deinterleave_arrays::<Self, M, C>(buf)
        }
    }

    /// Interleave `M` records of `C` components and store them as a contiguous
    /// array-of-structures - the exact inverse of
    /// [`load_deinterleaved_arrays`](Self::load_deinterleaved_arrays), with the
    /// same two strategies replayed backwards.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `M * C * LANES` elements.
    unsafe fn store_interleaved_arrays<const M: usize, const C: usize>(
        ptr: *mut Self::Element,
        values: [[Storage<Self>; C]; M],
    ) {
        const { assert!(M >= 1 && C >= 1) };

        let lanes = Self::lanes();

        if const { Self::HAS_STRUCTURAL_MEMOPS && C <= 4 } {
            // Per-component radix-M interleave: stream-order back to chunk-order.
            let mut comp = [[Self::EMPTY; M]; C];

            let mut c = 0;
            while c < C {
                let mut streams = [Self::EMPTY; M];

                let mut j = 0;
                while j < M {
                    streams[j] = values[j][c];
                    j += 1;
                }

                comp[c] = crate::backend::generic::polyfills::interleave_n::<Self, M>(streams);
                c += 1;
            }

            let mut k = 0;
            while k < M {
                let mut chunk = [Self::EMPTY; C];

                let mut c = 0;
                while c < C {
                    chunk[c] = comp[c][k];
                    c += 1;
                }

                unsafe { Self::store_interleaved::<C>(ptr.add(k * C * lanes), chunk) };
                k += 1;
            }
        } else {
            let out = crate::backend::generic::polyfills::interleave_arrays::<Self, M, C>(values);
            let flat = crate::backend::generic::polyfills::flat_arrays(&out);

            let mut i = 0;
            while i < M * C {
                unsafe { Self::store_unaligned(ptr.add(i * lanes), flat[i]) };
                i += 1;
            }
        }
    }

    /// Load `M` interleaved composite records of `1 + TAIL` components each and
    /// de-interleave them into `M` [`StreamGroup`]s: reads
    /// `M * (TAIL + 1) * LANES` contiguous elements starting at `ptr`, and
    /// `out[j].head`/`out[j].tail[c]` hold the de-interleaved components of
    /// composite stream `j`, i.e.
    /// `out[j].head[lane] == ptr[lane * M * (TAIL + 1) + j * (TAIL + 1)]` and
    /// `out[j].tail[c][lane] == ptr[lane * M * (TAIL + 1) + j * (TAIL + 1) + 1 + c]`.
    ///
    /// The `TAIL` spelling of
    /// [`load_deinterleaved_arrays`](Self::load_deinterleaved_arrays), which is
    /// the real implementation - this only re-shapes `[[_; TAIL + 1]; M]` into
    /// `[StreamGroup<_, TAIL>; M]`. It exists because a composite type built as
    /// "a head plus `N` more" (`Dual<V, N>`: a primal and `N` derivatives) can
    /// spell `TAIL = N` but not `C = N + 1`, while a type built as "`N`
    /// components" (a geometric `Vector<V, N>`) is the reverse. Stable Rust can
    /// bridge neither direction: `arrays::<M, { TAIL + 1 }>` is a const-generic
    /// ARGUMENT computed from a generic parameter, which needs
    /// `generic_const_exprs`. Hence the small `TAIL -> C` dispatch below - three
    /// arms, naming the only component counts a structural load can serve
    /// anyway, and no duplicated transpose logic.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `M * (TAIL + 1) * LANES` elements.
    unsafe fn load_deinterleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *const Self::Element,
    ) -> [StreamGroup<Storage<Self>, TAIL>; M] {
        const { assert!(M >= 1) };

        let mut out = [StreamGroup { head: Self::EMPTY, tail: [Self::EMPTY; TAIL] }; M];

        // Re-shape `[[_; C]; M]` (C == TAIL + 1) into groups. `records[j][0]` is
        // the head; the rest is the tail, in order.
        macro_rules! reshape {
            ($c:literal) => {{
                let records = unsafe { Self::load_deinterleaved_arrays::<M, $c>(ptr) };

                let mut j = 0;
                while j < M {
                    out[j].head = records[j][0];

                    let mut c = 0;
                    while c < TAIL {
                        out[j].tail[c] = records[j][1 + c];
                        c += 1;
                    }
                    j += 1;
                }
            }};
        }

        if const { TAIL == 0 } {
            reshape!(1);
        } else if const { TAIL == 1 } {
            reshape!(2);
        } else if const { TAIL == 2 } {
            reshape!(3);
        } else if const { TAIL == 3 } {
            reshape!(4);
        } else {
            // Beyond the structural widths the array form has nothing extra to
            // offer, so take the flat engine directly and skip the re-shape.
            let lanes = Self::lanes();
            let empty = StreamGroup { head: Self::EMPTY, tail: [Self::EMPTY; TAIL] };
            let mut buf = [empty; M];

            {
                let flat = crate::backend::generic::polyfills::flat_groups_mut(&mut buf);
                let mut i = 0;
                while i < M * (TAIL + 1) {
                    flat[i] = unsafe { Self::load_unaligned(ptr.add(i * lanes)) };
                    i += 1;
                }
            }

            return crate::backend::generic::polyfills::deinterleave_grouped::<Self, M, TAIL>(buf);
        }

        out
    }

    /// Interleave `M` [`StreamGroup`]s and store them as a contiguous
    /// array-of-structures - the exact inverse of
    /// [`load_deinterleaved_grouped`](Self::load_deinterleaved_grouped), and the
    /// same thin re-shape over
    /// [`store_interleaved_arrays`](Self::store_interleaved_arrays).
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `M * (TAIL + 1) * LANES` elements.
    unsafe fn store_interleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *mut Self::Element,
        values: [StreamGroup<Storage<Self>, TAIL>; M],
    ) {
        const { assert!(M >= 1) };

        macro_rules! reshape {
            ($c:literal) => {{
                let mut records = [[Self::EMPTY; $c]; M];

                let mut j = 0;
                while j < M {
                    records[j][0] = values[j].head;

                    let mut c = 0;
                    while c < TAIL {
                        records[j][1 + c] = values[j].tail[c];
                        c += 1;
                    }
                    j += 1;
                }

                unsafe { Self::store_interleaved_arrays::<M, $c>(ptr, records) };
            }};
        }

        if const { TAIL == 0 } {
            reshape!(1);
        } else if const { TAIL == 1 } {
            reshape!(2);
        } else if const { TAIL == 2 } {
            reshape!(3);
        } else if const { TAIL == 3 } {
            reshape!(4);
        } else {
            let lanes = Self::lanes();
            let out = crate::backend::generic::polyfills::interleave_grouped::<Self, M, TAIL>(values);
            let flat = crate::backend::generic::polyfills::flat_groups(&out);

            let mut i = 0;
            while i < M * (TAIL + 1) {
                unsafe { Self::store_unaligned(ptr.add(i * lanes), flat[i]) };
                i += 1;
            }
        }
    }

    /// # Safety
    ///
    /// Every lane of `indices` must be a valid index into `values` (i.e. `< values.len()`).
    /// The default implementation bounds-checks and panics on an out-of-range index, but
    /// hardware-gather overrides (e.g. `_mm256_permutevar8x32_ps`, `vpgatherdd`) do not -
    /// passing an out-of-range index there is undefined behavior.
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        let indices = <Self::Unsigned as Register>::as_slice(&indices);

        let mut res = Self::EMPTY;
        let resa = Self::as_mut_slice(&mut res);

        for i in 0..Self::lanes() {
            let idx: usize = indices[i].try_into().unwrap_or_else(#[cold] |_| panic!("Invalid index given for lookup"));

            resa[i] = values[idx];
        }

        res
    }

    /// Borrow the register's storage as a slice of elements.
    ///
    /// The lane count travels as the slice length rather than in the type, which a
    /// future runtime-length backend can implement, while an array-typed borrow
    /// cannot. The length is constructed directly from [`lanes()`](CoreRegister::lanes), so
    /// LLVM sees it as a constant on fixed-width backends.
    #[inline(always)]
    fn as_slice(storage: &Storage<Self>) -> &[Self::Element] {
        // The default borrows the first `lanes()` elements of storage. Unlike the
        // whole-storage load/store defaults this only needs the elements to be a
        // contiguous PREFIX, so wider-than-lanes storage is fine (ReducedRegister
        // is a lanes-prefix view of a wider register); smaller is never sound.
        const {
            assert!(
                size_of::<Storage<Self>>() >= (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Register storage is smaller than its lane count implies"
            );
        }

        // SAFETY: asserted above; storage is exactly `lanes()` elements.
        unsafe { core::slice::from_raw_parts(storage as *const Storage<Self> as *const Self::Element, Self::lanes()) }
    }

    /// Mutably borrow the register's storage as a slice of elements.
    ///
    /// See [`as_slice`](Self::as_slice).
    #[inline(always)]
    fn as_mut_slice(storage: &mut Storage<Self>) -> &mut [Self::Element] {
        // The default borrows the first `lanes()` elements of storage. Unlike the
        // whole-storage load/store defaults this only needs the elements to be a
        // contiguous PREFIX, so wider-than-lanes storage is fine (ReducedRegister
        // is a lanes-prefix view of a wider register); smaller is never sound.
        const {
            assert!(
                size_of::<Storage<Self>>() >= (size_of::<Self::Element>() * <Self::Lanes as Unsigned>::USIZE),
                "Register storage is smaller than its lane count implies"
            );
        }

        // SAFETY: asserted above; storage is exactly `lanes()` elements.
        unsafe { core::slice::from_raw_parts_mut(storage as *mut Storage<Self> as *mut Self::Element, Self::lanes()) }
    }

    fn iter(storage: &Storage<Self>) -> core::slice::Iter<'_, Self::Element> {
        Self::as_slice(storage).iter()
    }

    fn iter_mut(storage: &mut Storage<Self>) -> core::slice::IterMut<'_, Self::Element> {
        Self::as_mut_slice(storage).iter_mut()
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        const {
            assert!(
                I < <Self::Lanes as Unsigned>::USIZE,
                "Index out of bounds for register lane extraction"
            );
        }

        Self::as_slice(&value)[I]
    }

    fn last_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value)[Self::lanes() - 1]
    }

    fn insert<const I: usize>(mut value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        const {
            assert!(
                I < <Self::Lanes as Unsigned>::USIZE,
                "Index out of bounds for register lane insertion"
            );
        }

        Self::as_mut_slice(&mut value)[I] = element;
        value
    }

    fn map<F>(mut value: Storage<Self>, mut f: F) -> Storage<Self>
    where
        F: FnMut(Self::Element) -> Self::Element,
    {
        for v in Self::as_mut_slice(&mut value) {
            *v = f(*v);
        }

        value
    }

    fn zip<F>(mut lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for (a, b) in Self::as_mut_slice(&mut lhs).iter_mut().zip(Self::as_slice(&rhs)) {
            *a = f(*a, *b);
        }

        lhs
    }

    fn fold<F>(first: Self::Element, value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_slice(&value).iter().fold(first, |acc, &v| f(acc, v))
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        Self::as_slice(&value)
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
        Self::as_mut_slice(&mut value).reverse();
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
        crate::backend::generic::polyfills::compress_default::<Self>(value, mask)
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
        crate::backend::generic::polyfills::compress_z_default::<Self>(value, mask)
    }

    /// Merge-masked left-pack: like [`compress`](Self::compress), but the lanes
    /// at and beyond the population count take their values from `src` (at their
    /// own positions) instead of holding the unselected elements. Matches
    /// AVX-512 merge-masked `vpcompress*`. This is the accumulator step of a
    /// buffered stream compactor: `src` holds the leftovers, `value` the
    /// incoming batch.
    ///
    /// For `src = [w, x, y, z]`, `value = [a, b, c, d]`, `mask = [T, F, T, F]`
    /// the result is `[a, c, y, z]`.
    ///
    /// The keep-lanes are *position*-addressed (`i >= popcount`), not
    /// mask-addressed, so this cannot be expressed with the usual masked-variant
    /// blend; the default builds a prefix mask of the count
    /// ([`from_native_bitmask`](MaskRegister::from_native_bitmask)) and blends
    /// over [`compress`](Self::compress), inheriting its fast path. Registers
    /// wider than 64 lanes (beyond `from_native_bitmask`) fall back to a scalar
    /// single pass.
    fn compress_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if const { <Self::Lanes as Unsigned>::USIZE > 64 } {
            return crate::backend::generic::polyfills::compress_m_default::<Self>(src, mask, value);
        }

        let cnt = <Self::Mask as MaskRegister>::count_set_one(mask);
        let bits = if cnt >= 64 { u64::MAX } else { (1u64 << cnt) - 1 };
        let prefix = <Self::Mask as MaskRegister>::from_native_bitmask(bits);

        Self::blendv(prefix, src, Self::compress(value, mask))
    }

    /// Inverse left-pack (`expand`): scatter the packed low lanes of `value`
    /// back out to the lanes where `mask` is set, preserving order. Defined as
    /// the **exact inverse permutation** of [`compress`](Self::compress) - the
    /// unselected lanes read the tail, so for every input
    /// `expand(compress(v, m), m) == v` and `compress(expand(v, m), m) == v`.
    ///
    /// For `value = [a, c, b, d]` and `mask = [T, F, T, F]` the result is
    /// `[a, b, c, d]` (lane 0 reads packed `a`, lane 2 reads packed `c`, the
    /// unselected lanes 1/3 read the tail `b, d`).
    ///
    /// This is the return trip of stream compaction - compact the active lanes,
    /// operate, expand the results back to their home lanes. AVX-512
    /// `vpexpand*` defines only the selected lanes (see
    /// [`expand_z`](Self::expand_z) / [`expand_m`](Self::expand_m)); the
    /// full-permutation form costs the same single permute.
    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        crate::backend::generic::polyfills::expand_default::<Self>(value, mask)
    }

    /// Zero-filling inverse left-pack: like [`expand`](Self::expand), but the
    /// unselected lanes are zeroed instead of reading the tail. Matches AVX-512
    /// zero-masking `vpexpand*`.
    ///
    /// For `value = [a, c, x, x]` and `mask = [T, F, T, F]` the result is
    /// `[a, 0, c, 0]`.
    ///
    /// Unlike [`compress_z`](Self::compress_z) (whose `zz` must compose *before*
    /// the permute), the zeroing here composes *after*, which is why the
    /// overriding macros implement it as `zz(mask, expand(value, mask))`.
    fn expand_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        crate::backend::generic::polyfills::expand_z_default::<Self>(value, mask)
    }

    /// Merge-masked inverse left-pack: like [`expand`](Self::expand), but the
    /// unselected lanes take their values from `src` instead of reading the
    /// tail. Matches AVX-512 merge-masked `vpexpand*`.
    ///
    /// For `src = [w, x, y, z]`, `value = [a, c, ..]`, `mask = [T, F, T, F]`
    /// the result is `[a, x, c, z]`.
    ///
    /// The keep-lanes here *are* mask-addressed, so the default is a blend over
    /// the plain [`expand`](Self::expand), inheriting its fast path. It still
    /// cannot ride the `#[masked]` macro: `expand` already takes the mask as a
    /// semantic argument, and the generated variant would insert a second one.
    fn expand_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::blendv(mask, src, Self::expand(value, mask))
    }

    const HAS_PERMUTEV: bool;

    /// Scalar reference lowering for [`permutev`](Self::permutev): a per-lane
    /// walk over the index register. Out-of-range indices wrap (power-of-two
    /// lane counts) or clamp, one legal instance of the "unspecified lane
    /// value" contract, kept branch-cheap and memory-safe.
    fn scalar_permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        let value_array = Self::as_slice(&value);
        let idxs_array = <Self::Unsigned as Register>::as_slice(&idxs);
        let result_array = Self::as_mut_slice(&mut result);

        let mask = (Self::Lanes::USIZE - 1) as usize;

        for (&idx, dst) in idxs_array.iter().zip(result_array.iter_mut()) {
            let idx: usize = idx.try_into().unwrap_or(usize::MAX);

            let idx = if const { is_power_of_2(Self::Lanes::U32) } {
                idx & mask // we can AND with the mask if power-of-two lane count
            } else {
                idx.min(mask) // otherwise clamp to the max index
            };

            unsafe { core::hint::assert_unchecked(idx < value_array.len()) };

            *dst = value_array[idx];
        }

        result
    }

    /// Group-granularity 2-way interleave: blocks of `GROUP` consecutive elements
    /// move as a unit, never split. It is [`InterleaveRegister::interleave`] on the
    /// register reinterpreted as `LANES / GROUP` elements of `GROUP *` the width.
    ///
    /// `GROUP == 1` is exactly [`interleave`](InterleaveRegister::interleave);
    /// `GROUP == 2` is the pair (complex) interleave -
    /// `lo == [a.G0, b.G0, a.G1, b.G1, ...]` over the low half of the groups, `hi`
    /// over the high half - the natural primitive for interleaved-complex SIMD
    /// (FFT transposes, complex gather/scatter). `GROUP` must divide `LANES`.
    ///
    /// The default forwards `GROUP == 1` to the required
    /// [`interleave`](InterleaveRegister::interleave) primitive and sends any other
    /// `GROUP` to the lane-wise [`interleave_by`](crate::backend::generic::polyfills::interleave_by)
    /// fallback. Backends override this for the group sizes they do natively (the
    /// doubled-element `_mm256_unpacklo_pd` + `permute2f128` for `GROUP == 2` on
    /// AVX2, one `zip` on NEON).
    fn interleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        crate::backend::generic::polyfills::interleave_by_default::<Self, GROUP>(a, b)
    }

    /// The inverse of [`interleave_by`](Self::interleave_by) - group-granularity
    /// de-interleave. `GROUP == 1` forwards to
    /// [`deinterleave`](InterleaveRegister::deinterleave).
    fn deinterleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        crate::backend::generic::polyfills::deinterleave_by_default::<Self, GROUP>(a, b)
    }

    /// Permute the lanes of `value` by a live index register: lane `i` of the
    /// result is `value[idxs[i]]`.
    ///
    /// Indices must be in `0..LANES`. An out-of-range index produces an
    /// UNSPECIFIED value in that lane, never a fault or UB, but backends
    /// differ (`vpermd` wraps, NEON `tbl` zeroes, the scalar walk wraps or
    /// clamps). Do not write code against any particular out-of-range result.
    /// No release-mode range checks are performed.
    #[masked]
    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        Self::scalar_permutev(value, idxs)
    }

    /// [`permutev`](Self::permutev) with compile-time indices. The index
    /// register construction constant-folds, and backends with immediate-operand
    /// shuffles override this (or `swizzle_const`) to pattern-match `I`.
    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        Self::permutev(value, index_register::<Self>(&I::INDICES))
    }

    /// Widen a `LANES`-byte index array into the unsigned index register
    /// [`permutev`](Self::permutev) consumes: lane `i` is `bytes[i]`
    /// zero-extended to the index element width. For byte-element registers
    /// this is the identity load.
    ///
    /// This portable default does NOT contract into a widening load. LLVM
    /// emits a `movzx` + insert chain per lane instead (measured at
    /// `GenericArray` and `[u8; 8]` shapes, 128- and 256-bit, costing ~8
    /// instructions per compress/expand with every correctness test still
    /// green). Backends with a hardware widening load (`vpmovzxb*`, NEON
    /// `vmovl`, wasm extends) override it, and with a real widening load the
    /// byte rows are one instruction cheaper than `u32` rows at 256-bit
    /// (`mov`/`vpmovzxbd`/`vpermd` vs `mov`/`shl`/`vmovups`/`vpermps`) and on
    /// the SSE4.2 `pshufb` path, and a tie at 128-bit AVX. The grouped
    /// compress kernel's index assembly is bounded by this method on
    /// 2-byte-and-wider elements.
    fn widen_index_bytes(bytes: &GenericArray<u8, Self::Lanes>) -> Storage<Self::Unsigned> {
        let mut idx: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();

        let mut i = 0;
        while i < Self::Lanes::USIZE {
            idx[i] = Element::from_u16(bytes[i] as u16);
            i += 1;
        }

        Self::Unsigned::new(idx)
    }

    /// Permute `value` directly by a compress/expand table row.
    ///
    /// Plumbing for the `<= 8`-lane compress/expand table paths
    /// ([`polyfills::compress`](crate::backend::generic::polyfills::compress)
    /// and [`polyfills::expand`](crate::backend::generic::polyfills::expand)),
    /// whose gather indices are stored as `u8` (every index is in `0..8`),
    /// keeping the two tables at ~4.6 KB of `.rodata` instead of ~18 KB. The
    /// row is ALWAYS 8 wide regardless of `LANES`, and entries past `LANES` are
    /// ignored, since a row's leading `LANES` entries are always `< LANES`.
    ///
    /// The default takes the leading `min(LANES, 8)` bytes through
    /// [`widen_index_bytes`](Self::widen_index_bytes) and defers to
    /// [`permutev`](Self::permutev), which is optimal where the permute
    /// control matches the element width (x86 32-bit lanes: one `pmovzxbd`
    /// feeding `vpermd`).
    ///
    /// Byte-shuffle backends override it: their `permutev` control wants raw
    /// bytes, so the widen would be a round trip. Feeding the row in as bytes
    /// skips it, and the clamp too.
    fn permutev_row(value: Storage<Self>, row: &GenericArray<u8, generic_array::typenum::U8>) -> Storage<Self> {
        if const { Self::Lanes::USIZE <= 8 } {
            // Reinterpret the row's leading `LANES` bytes in place rather than
            // staging a copy: the copy is a stack round trip LLVM does NOT
            // elide, and it costs the widening load its table-row memory operand
            // (measured 27 vs 11 instructions on a `f32x8` compress).
            //
            // SAFETY: `GenericArray<u8, N>` is exactly `N` bytes at align 1, so
            // for `LANES <= 8` the row's prefix is a valid instance, and
            // `widen_index_bytes` reads only those `LANES` bytes.
            let bytes = unsafe { &*(row.as_slice().as_ptr() as *const GenericArray<u8, Self::Lanes>) };

            return Self::permutev(value, Self::widen_index_bytes(bytes));
        }

        // No table path reaches this method above 8 lanes, so lanes past the
        // row's 8 entries keep their zero fill.
        let mut bytes: GenericArray<u8, Self::Lanes> = GenericArray::default();

        let mut i = 0;
        while i < 8 {
            bytes[i] = row[i];
            i += 1;
        }

        Self::permutev(value, Self::widen_index_bytes(&bytes))
    }

    /// Scalar reference lowering for [`swizzle`](Self::swizzle), with the same
    /// wrap-or-clamp handling of out-of-range indices as
    /// [`scalar_permutev`](Self::scalar_permutev), over the `2 * LANES` span.
    fn scalar_swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        let a_array = Self::as_slice(&a);
        let b_array = Self::as_slice(&b);
        let idxs_array = <Self::Unsigned as Register>::as_slice(&idxs);
        let result_array = Self::as_mut_slice(&mut result);

        let mask = (Self::Lanes::USIZE << 1) - 1;

        for (&idx, dst) in idxs_array.iter().zip(result_array.iter_mut()) {
            let idx: usize = idx.try_into().unwrap_or(usize::MAX);

            // NOTE: If Self is power of two, so is 2 * Self
            let mut idx = if const { is_power_of_2(Self::Lanes::U32) } {
                idx & mask // we can AND with the mask if power-of-two lane count
            } else {
                idx.min(mask) // otherwise clamp to the max index
            };

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

    /// Select lanes from the concatenation `[a, b]` by a live index register:
    /// index `i < LANES` takes `a[i]`, `LANES <= i < 2*LANES` takes
    /// `b[i - LANES]`. Same out-of-range contract as
    /// [`permutev`](Self::permutev): unspecified lane value, never UB.
    ///
    /// The default is branchless: both sources are permuted by the raw
    /// indices (`b` by `idxs - LANES`), and the compare-derived blend keeps
    /// the lane whose source was actually addressed. The other permute's
    /// lane held an unspecified value the blend discards.
    #[masked]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_swizzle(a, b, idxs);
        }

        let lanes = Self::Unsigned::splat(Element::from_u16(<Self::Lanes as Unsigned>::U16));

        let from_b = Self::Unsigned::ge(idxs, lanes);

        let tmp_a = Self::permutev(a, idxs);
        let tmp_b = Self::permutev(b, Self::Unsigned::sub(idxs, lanes));

        Self::blendv(
            <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(from_b),
            tmp_a,
            tmp_b,
        )
    }

    /// [`swizzle`](Self::swizzle) with compile-time indices, see
    /// [`permutev_const`](Self::permutev_const).
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::swizzle(a, b, index_register::<Self>(&I::INDICES))
    }

    /// Whether [`align`](Self::align) has a native cross-register implementation on
    /// this register, rather than the generic [`swizzle_const`](Self::swizzle_const)
    /// default.
    ///
    /// The default path is correct everywhere but its cost varies: one instruction
    /// with a real cross-register align (`palignr`, `vext`, `i8x16.shuffle`), two
    /// `permutev`s plus a `blendv` with only variable permutes, and a scalar memory
    /// round-trip where [`HAS_PERMUTEV`](Self::HAS_PERMUTEV) is false.
    ///
    /// Algorithms built out of an `align` ladder - the prefix-scan family in
    /// `backend::generic::polyfills::scan` - gate on this, since a ladder of spilled
    /// aligns loses to a scalar loop outright.
    ///
    /// Set by the `impl_*_align*!` macros alongside the `align` body they emit, so
    /// the flag cannot drift from the implementation.
    const HAS_NATIVE_ALIGN: bool = false;

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
    /// compile-time `AlignIndices` pattern (in `crate::swizzle`), so it
    /// is correct on every backend, element type, and lane count. Integer
    /// backends override it with native byte aligns (`palignr`, whole-register
    /// byte shifts, or the AVX2 256-bit sequence).
    fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b)
    }

    /// Runtime permute of an `N`-chunk [`ArrayRegister<Self, N>`](array::ArrayRegister)
    /// by per-chunk live index registers (global indices in `0..N*LANES`).
    ///
    /// `ArrayRegister`'s `permutev` delegates here so a specific backend register
    /// can override the cross-chunk routing with a faster sequence (NEON's
    /// multi-register `tbl`). The default is branchless and entirely
    /// in-register: for each (output chunk, source chunk `j`) pair,
    /// `local = idx - j*LANES` wraps below zero, so `local < LANES` is exactly
    /// "this lane addresses chunk `j`", so one sub, one compare, one permute
    /// and one blend per pair. Lanes addressed to other chunks feed `permutev` an
    /// out-of-range local index, whose unspecified result the blend discards.
    ///
    /// Same out-of-range contract as [`permutev`](Self::permutev): a global
    /// index `>= N*LANES` yields an unspecified lane value (here: whatever the
    /// last chunk's blend left), never UB.
    #[inline(always)]
    fn array_permutev<const N: usize>(
        value: [Storage<Self>; N],
        idxs: [Storage<Self::Unsigned>; N],
    ) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let mut out = Self::EMPTY;

            for j in 0..N {
                let local = Self::Unsigned::sub(idxs[i], Self::Unsigned::splat(Element::from_u16((j * l) as u16)));
                let here = Self::Unsigned::lt(local, Self::Unsigned::splat(Element::from_u16(l as u16)));

                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(here);
                let permuted = Self::permutev(value[j], local);

                out = Self::blendv(blend, out, permuted);
            }

            result[i] = out;
        }

        result
    }

    /// Compile-time-index companion of [`array_permutev`](Self::array_permutev),
    /// kept on the `&[u32]` form: the scalar local/chunk-id split below costs
    /// nothing when `idxs` is constant (everything folds, including the blend
    /// selectors), which is why `ArrayRegister::permutev_const` routes here
    /// instead of materializing an index register.
    #[inline(always)]
    fn array_permutev_indices<const N: usize>(value: [Storage<Self>; N], idxs: &[u32]) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;
        let total = N * l;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let base = i * l;

            // Branchless split of this output chunk's indices into local offsets
            // (for the per-chunk permute) and source-chunk ids (for the blend).
            let mut local: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();
            let mut chunk_ids: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();

            for lane in 0..l {
                let g = idxs[base + lane] as usize;
                let g = if const { (N * <Self::Lanes as Unsigned>::USIZE).is_power_of_two() } {
                    g & (total - 1)
                } else {
                    g.min(total - 1)
                };
                local[lane] = Element::from_u16((g % l) as u16);
                chunk_ids[lane] = Element::from_u16((g / l) as u16);
            }

            let local_reg = Self::Unsigned::new(local);
            let chunk_reg = Self::Unsigned::new(chunk_ids);

            let mut out = Self::EMPTY;
            for j in 0..N {
                let j_splat = Self::Unsigned::splat(Element::from_u16(j as u16));
                let eq = Self::Unsigned::eq(chunk_reg, j_splat);
                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(eq);
                let permuted = Self::permutev(value[j], local_reg);
                out = Self::blendv(blend, out, permuted);
            }

            result[i] = out;
        }

        result
    }

    /// Runtime swizzle of two `N`-chunk [`ArrayRegister<Self, N>`](array::ArrayRegister)
    /// values by per-chunk live index registers selecting across all `2N` input
    /// chunks (`a` then `b`). The two-source companion to
    /// [`array_permutev`](Self::array_permutev), with the same branchless
    /// sub/compare per (output, source) pair and the same out-of-range contract.
    #[inline(always)]
    fn array_swizzle<const N: usize>(
        a: [Storage<Self>; N],
        b: [Storage<Self>; N],
        idxs: [Storage<Self::Unsigned>; N],
    ) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let mut out = Self::EMPTY;

            for j in 0..(2 * N) {
                let src = if j < N { a[j] } else { b[j - N] };

                let local = Self::Unsigned::sub(idxs[i], Self::Unsigned::splat(Element::from_u16((j * l) as u16)));
                let here = Self::Unsigned::lt(local, Self::Unsigned::splat(Element::from_u16(l as u16)));

                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(here);
                let permuted = Self::permutev(src, local);

                out = Self::blendv(blend, out, permuted);
            }

            result[i] = out;
        }

        result
    }

    /// Compile-time-index companion of [`array_swizzle`](Self::array_swizzle),
    /// see [`array_permutev_indices`](Self::array_permutev_indices).
    #[inline(always)]
    fn array_swizzle_indices<const N: usize>(
        a: [Storage<Self>; N],
        b: [Storage<Self>; N],
        idxs: &[u32],
    ) -> [Storage<Self>; N] {
        let l = <Self::Lanes as Unsigned>::USIZE;
        let total = N * l;
        let span = 2 * total;

        let mut result = [Self::EMPTY; N];

        for i in 0..N {
            let base = i * l;

            let mut local: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();
            let mut chunk_ids: GenericArray<<Self::Unsigned as Register>::Element, Self::Lanes> = GenericArray::default();

            for lane in 0..l {
                let g = idxs[base + lane] as usize;
                let g = if const { (2 * N * <Self::Lanes as Unsigned>::USIZE).is_power_of_two() } {
                    g & (span - 1)
                } else {
                    g.min(span - 1)
                };
                local[lane] = Element::from_u16((g % l) as u16);
                chunk_ids[lane] = Element::from_u16((g / l) as u16);
            }

            let local_reg = Self::Unsigned::new(local);
            let chunk_reg = Self::Unsigned::new(chunk_ids);

            let mut out = Self::EMPTY;
            for j in 0..(2 * N) {
                let src = if j < N { a[j] } else { b[j - N] };
                let j_splat = Self::Unsigned::splat(Element::from_u16(j as u16));
                let eq = Self::Unsigned::eq(chunk_reg, j_splat);
                let blend = <Self::Mask as CastMaskRegister<<Self::Unsigned as CoreRegister>::Mask>>::mask_from(eq);
                let permuted = Self::permutev(src, local_reg);
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

/// Build the unsigned index register [`permutev`](Register::permutev) consumes
/// from a `u32` index array. With compile-time indices (the
/// `permutev_const`/`swizzle_const` defaults) the whole construction
/// constant-folds into a literal vector.
///
/// Indices are narrowed through `u16`, which every swizzle span fits
/// (`2 * LANES <= 128`). Values above `u16::MAX` would be out of range anyway
/// and land in the unspecified-lane contract.
#[inline(always)]
pub(crate) fn index_register<R: Register + ?Sized>(idxs: &GenericArray<u32, R::Lanes>) -> Storage<R::Unsigned> {
    let mut out: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> = GenericArray::default();

    let mut i = 0;
    while i < <R::Lanes as Unsigned>::USIZE {
        out[i] = Element::from_u16(idxs[i] as u16);
        i += 1;
    }

    R::Unsigned::new(out)
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
        // let scale = size_of::<Self::Element>();

        unsafe {
            let mut result = Self::EMPTY;

            let res = Self::as_mut_slice(&mut result);
            let indices = IDX::as_slice(&indices);

            for i in 0..Self::lanes() {
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
        // let scale = size_of::<Self::Element>();

        unsafe {
            let mut result = src;

            let res = Self::as_mut_slice(&mut result);
            let indices = IDX::as_slice(&indices);

            for i in 0..Self::lanes() {
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
            let value = Self::as_slice(&value);
            let indices = IDX::as_slice(&indices);

            for i in 0..Self::lanes() {
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
            let value = Self::as_slice(&value);
            let indices = IDX::as_slice(&indices);

            for i in 0..Self::lanes() {
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
        let arr = Self::as_mut_slice(&mut value);
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
        let arr = Self::as_mut_slice(&mut value);
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

    // ## Out-of-range shift counts
    //
    // A lane whose count is `>= the element bit width` produces an
    // **unspecified value**. Not undefined behaviour (it is always some
    // value, never a fault or a memory-safety problem), but which value
    // depends on the backend, and nothing here promises to make them agree.
    //
    // That is deliberate. Pinning a single answer would mean masking or
    // clamping the count on every variable shift, including the native ones
    // that already do something sensible on their own, so the cost would land
    // permanently on the fastest paths in order to tidy up input that callers
    // are not supposed to supply. Each backend therefore does whatever its
    // hardware does for free: x86 flushes to zero (sign fill for `srav`), NEON
    // reads the low 8 bits of the count as a signed value, and the emulated
    // paths generally wrap the count.
    //
    // The one thing that IS guaranteed is that no path panics, which is why
    // the fallbacks below use `wrapping_*` rather than plain `<<`/`>>`.

    #[conditional] fn shrv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // Scalar fallback. `shrv` is a *logical* (zero-fill) shift, so use
        // `unsigned_shr`: a plain `>>` on a signed element arithmetic-shifts, which
        // is `srav`, not `shrv`. (Backends with a hardware variable shift override this.)
        //
        // `logical_shr` is a plain `>>`, which PANICS under overflow checks for
        // an out-of-range count, a debug-only fault on a lane value that a
        // release build shifts happily. The count is masked first so the result
        // is merely unspecified, which is all this promises, rather than a
        // crash that only shows up in one profile.
        let mask: <Self::Element as Element>::Unsigned =
            Element::from_u16((core::mem::size_of::<Self::Element>() * 8 - 1) as u16);

        for (r, s) in Self::as_mut_slice(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_slice(&shifts))
        {
            *r = r.logical_shr(*s & mask);
        }

        value
    }

    #[conditional] fn shlv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        // Scalar fallback. See `shrv` for why the count is masked.
        let mask: <Self::Element as Element>::Unsigned =
            Element::from_u16((core::mem::size_of::<Self::Element>() * 8 - 1) as u16);

        for (r, s) in Self::as_mut_slice(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_slice(&shifts))
        {
            *r = *r << (*s & mask);
        }

        value
    }

    /// Rotate bits left
    ///
    /// The amount is reduced modulo the element bit width, matching
    /// [`u32::rotate_left`] (and hence the scalar backend, which *is*
    /// `rotate_left`). Without the mask, `width - shift` underflows for
    /// `shift >= width` and every vector backend returns zeros where the
    /// scalar oracle returns a rotation - a silent cross-backend divergence.
    /// `shift == 0` is unaffected: `shr(value, width)` is a defined zero on
    /// every backend, and `value | 0 == value`.
    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        let shift = shift & (width - 1); // widths are powers of two
        Self::bitor(Self::shl(value, shift), Self::shr(value, width - shift))
    }

    /// Rotate bits right
    ///
    /// The amount is reduced modulo the element bit width (see [`Self::rol`]).
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        let width = (core::mem::size_of::<Self::Element>() * 8) as u32;
        let shift = shift & (width - 1);
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
///
/// Three strengths of the same conversion, weakest domain first:
/// [`fast_cast_from`](Self::fast_cast_from) (unspecified outside a narrow
/// range), [`cast_from`](Self::cast_from) (`as`, but only guaranteed for
/// in-range finite inputs), and
/// [`saturating_cast_from`](Self::saturating_cast_from) (`as` on every input).
///
/// # Implementing
///
/// `cast_from` and `saturating_cast_from` **default to each other**, so a
/// backend provides whichever ones it has a distinct lowering for and gets the
/// rest for free. A pair whose hardware conversion is already `as`-exact
/// (AArch64 `FCVTZS`, wasm `trunc_sat`, anything in the scalar backend) needs
/// only one of them. A pair where they genuinely differ, which on x86 is every
/// float -> int conversion, provides both.
///
/// **Overriding neither is infinite mutual recursion.** It is not a compile
/// error, so it shows up as a hang or a stack overflow. The differential cast
/// suites are what catch it.
pub trait CastRegister<FROM: CoreRegister>: CoreRegister {
    /// Cast a register from another register type.
    ///
    /// For float -> int casts this behaves like `as` (truncation toward zero)
    /// **for in-range finite inputs only**: out-of-range or NaN lanes produce a
    /// backend-defined value (x86 returns the hardware "indefinite" integer,
    /// `INT::MIN`, where scalar `as` would saturate). For exact `as` semantics
    /// on every input (NaN -> 0, out-of-range clamps) use
    /// [`saturating_cast_from`](Self::saturating_cast_from).
    ///
    /// Integer narrowing wraps here, the same way `as` does.
    #[inline(always)]
    fn cast_from(value: Storage<FROM>) -> Storage<Self> {
        Self::saturating_cast_from(value)
    }

    /// Cast that clamps (saturates) out-of-range source values to the
    /// destination element's representable range, instead of the wrapping
    /// truncation (integer) or backend-defined indefinite value (float -> int)
    /// [`cast_from`](Self::cast_from) produces.
    ///
    /// Meaningful in two directions:
    ///
    /// - **narrowing, same-signedness integers** (`i64 -> i32 -> i16 -> i8`,
    ///   `u64 -> u32 -> u16 -> u8`, including skip-level pairs such as
    ///   `i64 -> i8`). Widening conversions lose nothing, so the default
    ///   (`cast_from`) is already exact for them. Sign-changing conversions
    ///   have no saturating lowering and fall through to the wrapping default,
    ///   which is what `as` does but not what the name promises: prefer
    ///   [`cast_from`](Self::cast_from) there and say what you meant.
    /// - **float -> int, every pair** at a given lane count (`f32`/`f64` into
    ///   any of `i8`/`i16`/`i32`/`i64` and their unsigned forms), with exact
    ///   Rust `as` semantics: NaN -> 0, out-of-range clamps to the destination
    ///   MIN/MAX.
    ///
    /// # Reference semantics
    ///
    /// Defined by the scalar backend and matched lane-for-lane by every
    /// hardware (`pack*`-based) implementation: clamp the source value into
    /// `[INTO::MIN, INTO::MAX]`, then convert. For unsigned destinations
    /// `INTO::MIN` is `0`, so only the high end is clamped. Saturation is
    /// idempotent across nested ranges, so a direct `i64 -> i8` is bit-identical
    /// to chaining `i64 -> i32 -> i16 -> i8`.
    ///
    /// Only the same-width float -> int casts (`f32 -> i32/u32`,
    /// `f64 -> i64/u64`) are primitive. The rest compose out of those. A
    /// narrowing destination goes through the same-width int and then the
    /// saturating integer narrow, and `f32` into a 64-bit int widens exactly to
    /// `f64` first. Either way the clamp lands at the destination's range rather
    /// than an intermediate one, which is what keeps the composition
    /// bit-identical to a direct `as`.
    #[inline(always)]
    fn saturating_cast_from(value: Storage<FROM>) -> Storage<Self> {
        Self::cast_from(value)
    }

    /// Cast a register to another register type, potentially faster
    /// when the values are within a certain range, otherwise
    /// unspecified values are returned. This method is safe in the
    /// Rust sense, but may not be safe in the sense that values
    /// may not be preserved across the cast.
    ///
    /// Float -> int keeps its narrow-domain shortcut in every configuration,
    /// `strict_ieee754` included. This is the "unspecified out of range"
    /// operation by definition, so the feature only redirects the vector
    /// layer's `cast`.
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

    /// Sort the lanes of this register in `O` order.
    ///
    /// Backed by a sorting network at every lane count up to 16. A register
    /// whose width has a hand-tuned in-lane network gets it via
    /// `sort_via_network!`; everything else gets the default
    /// [`sort_lanes`](crate::backend::generic::polyfills::sort::sort_lanes) -
    /// the one-chunk degenerate case of the array sort's widening merge, which
    /// is depth-minimal and needs no lane-count type equality, so it can serve
    /// as a default where `sort_8` and friends cannot.
    ///
    /// Past 16 lanes it is still a scalar compare-and-swap walk over the element
    /// slice, which is branchless and data-independent but quadratic and spills
    /// to memory.
    ///
    /// The direction is free on the network path: a layer emits the same
    /// permute + min + max + blend either way, with only the blend operands
    /// swapped. See [`crate::sort`] for the measurements and for why this is
    /// generic over a marker type rather than parameterized on the register.
    fn sort_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::sort::sort_lanes::<Self, O>(value)
    }

    /// Sort the lanes of a **bitonic** register in `O` order - one that rises
    /// then falls, or a rotation of one.
    ///
    /// `log2(LANES)` compare-exchange layers instead of a full sort, which is
    /// what a caller merging two already-sorted registers needs: reverse one,
    /// compare across the pair, then clean each side. That decomposition is how
    /// a multi-register sort gets its cross-register stages for free (whole
    /// register `min`/`max`, no shuffles) and confines shuffles to the cleanup.
    ///
    /// Note that "bitonic" is order-independent: two *descending* runs with the
    /// second reversed form a valley rather than a mountain, which is equally
    /// bitonic and equally cleanable. A merge therefore needs no direction
    /// handling of its own beyond passing `O` down.
    ///
    /// **Garbage in, garbage out**: on non-bitonic input the result is a
    /// permutation of the lanes but is not sorted. Use
    /// [`sort_by`](Self::sort_by) when the input is arbitrary. Past 16 lanes the
    /// default body *is* a full sort, so it happens to be correct for any input
    /// there, but that is an accident of the fallback - neither the default at
    /// 16 lanes and below nor an overriding backend's is.
    fn bitonic_clean_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::sort::bitonic_clean_lanes::<Self, O>(value)
    }

    /// Sort the lanes of this register ascending.
    ///
    /// Shorthand for [`sort_by::<Ascending>`](Self::sort_by); never override
    /// this one, override `sort_by`.
    #[inline(always)]
    fn sort(value: Storage<Self>) -> Storage<Self> {
        Self::sort_by::<crate::sort::Ascending>(value)
    }

    /// Sort the lanes of a **bitonic** register ascending.
    ///
    /// Shorthand for [`bitonic_clean_by::<Ascending>`](Self::bitonic_clean_by);
    /// never override this one, override `bitonic_clean_by`.
    #[inline(always)]
    fn bitonic_clean(value: Storage<Self>) -> Storage<Self> {
        Self::bitonic_clean_by::<crate::sort::Ascending>(value)
    }

    fn min_element(value: Storage<Self>) -> Self::Element;
    fn max_element(value: Storage<Self>) -> Self::Element;

    #[inline(always)]
    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        (Self::min_element(value), Self::max_element(value))
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element;
    fn prod_elements(value: Storage<Self>) -> Self::Element;

    /// Inclusive forward prefix sum: `out[i] = value[0] + .. + value[i]`.
    ///
    /// A `ceil(log2(LANES))`-stage [`align`](Register::align) ladder where the register
    /// has a native cross-register align, a sequential lane walk where it does not,
    /// chosen at compile time on [`HAS_NATIVE_ALIGN`](Register::HAS_NATIVE_ALIGN).
    /// See [`polyfills::scan`](crate::backend::generic::polyfills::scan) for the
    /// derivation, the fill values, and the NaN caveat on `min`/`max`.
    fn prefix_sum(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::prefix_sum::<Self>(value)
    }

    /// Inclusive forward prefix minimum: `out[i] = min(value[0], .., value[i])`.
    fn prefix_min(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::prefix_min::<Self>(value)
    }

    /// Inclusive forward prefix maximum: `out[i] = max(value[0], .., value[i])`.
    fn prefix_max(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::prefix_max::<Self>(value)
    }

    /// Inclusive reverse (suffix) sum: `out[i] = value[i] + .. + value[LANES-1]`.
    fn reverse_prefix_sum(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::reverse_prefix_sum::<Self>(value)
    }

    /// Inclusive reverse (suffix) minimum: `out[i] = min(value[i], .., value[LANES-1])`.
    fn reverse_prefix_min(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::reverse_prefix_min::<Self>(value)
    }

    /// Inclusive reverse (suffix) maximum: `out[i] = max(value[i], .., value[LANES-1])`.
    fn reverse_prefix_max(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::scan::reverse_prefix_max::<Self>(value)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let half = const { <Self::Lanes as Unsigned>::USIZE / 2 };

        let lo = Self::as_slice(&lo);
        let hi = Self::as_slice(&hi);

        let mut result = Self::EMPTY;

        let out = Self::as_mut_slice(&mut result);
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
    /// For each lane, how many *earlier* lanes hold the same value:
    /// `out[i] == |{ j < i : value[j] == value[i] }|`.
    ///
    /// AVX-512CD `vpconflict` followed by a population count. `== 0` is the
    /// first-occurrence mask, and the count is the round number for a
    /// conflicting read-modify-write (histogram / SAH-bin increment), where a
    /// plain scatter would silently drop duplicate writes.
    ///
    /// The default is the portable rotate ladder in
    /// [`polyfills::conflict`](crate::backend::generic::polyfills::conflict) -
    /// `LANES - 1` steps of ~4 vector ops. A backend with real conflict
    /// detection (AVX-512CD, behind `avx512-tier1`) should override it with
    /// `vpconflict` + `vpopcnt`, two instructions at any width.
    fn count_conflicts(value: Storage<Self>) -> Storage<Self> {
        crate::backend::generic::polyfills::conflict::count_conflicts_default::<Self>(value)
    }

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
        // Scalar fallback. The count is masked so an out-of-range one cannot
        // panic under overflow checks; see `BitshiftRegister::shrv`.
        let mask: <Self::Element as Element>::Unsigned =
            Element::from_u16((core::mem::size_of::<Self::Element>() * 8 - 1) as u16);

        for (r, s) in Self::as_mut_slice(&mut value)
            .iter_mut()
            .zip(<Self::Unsigned as Register>::as_slice(&shifts))
        {
            // r in this context is signed, so `>>` is an arithmetic shift
            *r = *r >> (*s & mask);
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

    /// Lane-alternating sign-bit mask `[-0.0, +0.0, -0.0, +0.0, ...]` (sign set on
    /// **even** lanes). `bitxor`ing a value with this negates its even lanes; it
    /// is the mask that turns [`addsub`](Self::addsub)/[`fmaddsub`](Self::fmaddsub)
    /// into a cheap `xor` on backends without a native alternating add/sub.
    ///
    /// Materialized at compile time (a constant load, never a runtime shuffle);
    /// the default fits every width including the emulated `ArrayRegister` /
    /// `ReducedRegister` ones. A backend with a native instruction just overrides
    /// the methods and leaves this untouched.
    const ALT_NEG: Storage<Self> = crate::register::alt_sign_reg::<Self>(true);

    /// Lane-alternating sign-bit mask `[+0.0, -0.0, +0.0, -0.0, ...]` (sign set on
    /// **odd** lanes) - the opposite parity of [`ALT_NEG`](Self::ALT_NEG), used by
    /// [`fmsubadd`](Self::fmsubadd).
    const ALT_POS: Storage<Self> = crate::register::alt_sign_reg::<Self>(false);

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
    unsafe fn native_ldexp(_value: Storage<Self>, _exp: Storage<Self::SignedBits>) -> Storage<Self> {
        unreachable!("native_ldexp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_frexp(_value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        unreachable!("native_frexp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_sin_cos<P: Policy>(_value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unreachable!("native_sin_cos is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_sin<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_sin is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_cos<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_cos is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_tan<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_tan is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_exp2<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp2 is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_log2<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_ln2 is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_exp<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_exp is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_ln<P: Policy>(_value: Storage<Self>) -> Storage<Self> {
        unreachable!("native_log is not implemented for this FloatRegister");
    }

    /// # Safety
    /// This method interfaces with underlying intrinsics and may produce undefined behavior on
    /// invalid inputs. Use with caution.
    unsafe fn native_powf<P: Policy>(_base: Storage<Self>, _exp: Storage<Self>) -> Storage<Self> {
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

    /// Lane-alternating subtract/add: **even lanes subtract, odd lanes add**.
    ///
    /// ```text
    /// [a0 - b0, a1 + b1, a2 - b2, a3 + b3, ...]
    /// ```
    ///
    /// This matches x86 `ADDSUBPS`/`ADDSUBPD` semantics exactly, so the native
    /// path is a single instruction. It is the building block for interleaved
    /// complex `[re, im, re, im, ...]` arithmetic; see [`fmaddsub`](Self::fmaddsub)
    /// for the complex-multiply lowering.
    ///
    /// The portable default flips the sign bit of `b` on even lanes with the
    /// materialized [`ALT_NEG`](Self::ALT_NEG) constant, then adds - so the even
    /// lanes compute `a + (-b) == a - b` exactly (single rounding).
    #[conditional] fn addsub(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::add(a, Self::bitxor(b, Self::ALT_NEG))
    }

    /// Fused multiply then [`addsub`](Self::addsub): **even lanes subtract, odd lanes add**.
    ///
    /// ```text
    /// [a0*b0 - c0, a1*b1 + c1, a2*b2 - c2, ...]
    /// ```
    ///
    /// Matches x86 `VFMADDSUB213PS`/`PD` (native path is one instruction). This
    /// is the core of an interleaved complex multiply of `a` by `w`:
    ///
    /// ```text
    /// wr = duplicate_even(w);  wi = duplicate_odd(w);  a_swap = swap_adjacent(a);
    /// result = fmaddsub(a, wr, a_swap * wi)   // [ar*wr - ai*wi, ar*wi + ai*wr, ...]
    /// ```
    ///
    /// The portable default uses the estimating [`mul_adde`](Self::mul_adde) (real
    /// FMA where available, otherwise a plain multiply-add) against a `c` whose
    /// even lanes are sign-flipped by the materialized [`ALT_NEG`](Self::ALT_NEG)
    /// constant, so non-FMA backends stay a cheap `xor` + `mul_adde`.
    #[conditional] fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        // even lanes: a*b - c ; odd lanes: a*b + c
        Self::mul_adde(a, b, Self::bitxor(c, Self::ALT_NEG))
    }

    /// Fused multiply then subadd - the opposite parity of [`fmaddsub`](Self::fmaddsub):
    /// **even lanes add, odd lanes subtract**.
    ///
    /// ```text
    /// [a0*b0 + c0, a1*b1 - c1, a2*b2 + c2, ...]
    /// ```
    ///
    /// Matches x86 `VFMSUBADD213PS`/`PD`. The portable default flips the sign of
    /// `c` on the *odd* lanes with the materialized [`ALT_POS`](Self::ALT_POS)
    /// constant and feeds it through [`mul_adde`](Self::mul_adde).
    #[conditional] fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        // even lanes: a*b + c ; odd lanes: a*b - c
        Self::mul_adde(a, b, Self::bitxor(c, Self::ALT_POS))
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
        if const { Self::HAS_TRUE_FMA } {
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
/// count. `pack`/`unpack` have generic branchless defaults (the private `unpack_packed` /
/// `pack_packed` in this module); backends override them where hardware exists (e.g. F16C
/// `vcvtph2ps`).
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

// =====================================================================================
// Sum of absolute differences (SAD).
//
// `SadN` sums `N / 8` consecutive byte-lanes of `|a - b|` into one `uN` lane, so the
// output register is always the SAME total width as the input (group bytes * 8 == output
// bits) and the lane count is `LANES / (N / 8)`. That same-width property is what lets
// the generic default simply bit-cast and run a SWAR cascade.
//
// The lane-count relation is deliberately NOT expressed in the bounds. Like
// [`BitCastRegister`], it is "enforced simply by the fact that it will only be
// implemented for" correctly-shaped pairs - the per-backend stamping macro and the
// `Simd` slot bounds pin the exact output register. Encoding `Lanes = Quot<Lanes, U8>`
// instead would drag a `typenum::Div` obligation through every emulated container and
// produce an unsatisfiable `U0` for the sub-native widths.
//
// The u64 form is the widest grouping and the one x86 does in a single `psadbw` (SSE2,
// so every tier); the narrower groupings are the single-instruction cases on NEON
// (`vpaddl`) and wasm (`extadd_pairwise`). Every backend gets a correct SWAR default and
// overrides where its hardware wins.
// =====================================================================================

/// Sum `2` consecutive byte-lanes of `|a - b|` into each `u16` lane of `W`.
///
/// Output lanes: `Self::LANES / 2`, same total width. Each result is at most `2 * 255 =
/// 510`, so no lane can overflow. There is no accumulating form: a `u16` lane saturates
/// after only ~128 accumulations, so callers that reduce over a long run should widen
/// deliberately or use [`Sad32Register`] / [`Sad64Register`].
pub trait Sad16Register<W>: UnsignedIntegerRegister<Unsigned = Self>
where
    W: UnsignedIntegerRegister<Element = u16>,
{
    fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<W>;
}

/// Sum `4` consecutive byte-lanes of `|a - b|` into each `u32` lane of `W`.
///
/// Output lanes: `Self::LANES / 4`, same total width. Each result is at most `4 * 255 =
/// 1020`, and [`sad32_accum`](Sad32Register::sad32_accum) can absorb roughly `4.2e6`
/// accumulations before a `u32` lane overflows.
pub trait Sad32Register<W>: UnsignedIntegerRegister<Unsigned = Self>
where
    W: UnsignedIntegerRegister<Element = u32>,
{
    fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<W>;

    /// `acc + sad32(a, b)`, the accumulate step of a blocked SAD loop.
    #[inline(always)]
    fn sad32_accum(acc: Storage<W>, a: Storage<Self>, b: Storage<Self>) -> Storage<W> {
        W::add(acc, Self::sad32(a, b))
    }
}

/// Sum `8` consecutive byte-lanes of `|a - b|` into each `u64` lane of `W` - x86
/// `PSADBW` semantics.
///
/// Output lanes: `Self::LANES / 8`, same total width. Each result is at most `8 * 255 =
/// 2040`; the `u64` lane is deliberate accumulation headroom, so
/// [`sad64_accum`](Sad64Register::sad64_accum) cannot overflow in any realistic loop
/// (~9e15 iterations). This is the form to reach for when reducing a large byte buffer:
/// accumulate in the `u64` lanes and reduce horizontally exactly once, at the end.
pub trait Sad64Register<W>: UnsignedIntegerRegister<Unsigned = Self>
where
    W: UnsignedIntegerRegister<Element = u64>,
{
    fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<W>;

    /// `acc + sad64(a, b)`, the accumulate step of a blocked SAD loop.
    #[inline(always)]
    fn sad64_accum(acc: Storage<W>, a: Storage<Self>, b: Storage<Self>) -> Storage<W> {
        W::add(acc, Self::sad64(a, b))
    }
}

/// Lane-wise SAD for registers narrower than 128 bits, where there is no SIMD win to be
/// had: sum `GROUP` consecutive byte lanes of `|a - b|` into each output lane, clamped at
/// the input lane count so a register holding fewer than one full group sums everything it
/// has into a single lane. The `ReducedRegister`/`ArrayRegister` sub-native ladder takes
/// this path on every backend.
macro_rules! decl_sad_scalar {
    ($name:ident, $ielem:ty, $oelem:ty, $group:expr) => {
        #[inline(always)]
        pub(crate) fn $name<C, W>(a: Storage<C>, b: Storage<C>) -> Storage<W>
        where
            C: UnsignedIntegerRegister<Element = $ielem>,
            W: UnsignedIntegerRegister<Element = $oelem>,
        {
            let d = C::abs_diff(a, b);
            let ds = C::as_slice(&d);
            let n = ds.len();

            let mut out = W::EMPTY;
            {
                let os = W::as_mut_slice(&mut out);
                let mut j = 0;
                while j < os.len() {
                    let start = j * $group;
                    let mut acc: $oelem = 0;
                    let mut k = 0;
                    while k < $group && start + k < n {
                        acc += ds[start + k] as $oelem;
                        k += 1;
                    }
                    os[j] = acc;
                    j += 1;
                }
            }
            out
        }
    };
}

decl_sad_scalar!(sad_scalar_u8_16, u8, u16, 2);
decl_sad_scalar!(sad_scalar_u8_32, u8, u32, 4);
decl_sad_scalar!(sad_scalar_u8_64, u8, u64, 8);
decl_sad_scalar!(sad_scalar_u16_32, u16, u32, 2);
decl_sad_scalar!(sad_scalar_u16_64, u16, u64, 4);
decl_sad_scalar!(sad_scalar_u32_64, u32, u64, 2);

// `u8x2` is `ArrayRegister<u8, 2>` on every backend, and its SAD outputs are the 1-lane
// scalar registers, so these three impls cover every backend at once. Two bytes is below
// any grouping, so `sad32`/`sad64` sum the whole register into one lane.
#[thermite_macros::inline_always]
impl Sad16Register<u16> for array::ArrayRegister<u8, 2> {
    fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<u16> {
        sad_scalar_u8_16::<Self, u16>(a, b)
    }
}

#[thermite_macros::inline_always]
impl Sad32Register<u32> for array::ArrayRegister<u8, 2> {
    fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<u32> {
        sad_scalar_u8_32::<Self, u32>(a, b)
    }
}

#[thermite_macros::inline_always]
impl Sad64Register<u64> for array::ArrayRegister<u8, 2> {
    fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<u64> {
        sad_scalar_u8_64::<Self, u64>(a, b)
    }
}

// `u16x2` is `ArrayRegister<u16, 2>` on every backend: two u16 lanes make exactly one
// 4-byte group and a partial 8-byte one, so both sum the whole register into one lane.
#[thermite_macros::inline_always]
impl Sad32Register<u32> for array::ArrayRegister<u16, 2> {
    fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<u32> {
        sad_scalar_u16_32::<Self, u32>(a, b)
    }
}

#[thermite_macros::inline_always]
impl Sad64Register<u64> for array::ArrayRegister<u16, 2> {
    fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<u64> {
        sad_scalar_u16_64::<Self, u64>(a, b)
    }
}

// An `ArrayRegister` composite (`u16x16` = `[u16x8; 2]`, `u32x16` = `[u32x8; 2]`, ...) just
// applies the inner register's SAD to each half: a group never spans two inner registers,
// so the result is exact and every emulated width above 128 bits comes for free. The
// output must be the array of the inner outputs, which is exactly how the wider `Simd`
// slots are defined.
macro_rules! impl_array_sad {
    ($trait:ident, $method:ident, $elem:ty) => {
        impl<C, W, const N: usize> $trait<array::ArrayRegister<W, N>> for array::ArrayRegister<C, N>
        where
            C: $trait<W>,
            W: UnsignedIntegerRegister<Element = $elem>,
            array::ArrayRegister<C, N>: UnsignedIntegerRegister<Unsigned = Self, Storage = array::ArrayRegister<C, N>>,
            array::ArrayRegister<W, N>: UnsignedIntegerRegister<Element = $elem, Storage = array::ArrayRegister<W, N>>,
        {
            #[inline(always)]
            fn $method(a: Storage<Self>, b: Storage<Self>) -> Storage<array::ArrayRegister<W, N>> {
                let mut out = <array::ArrayRegister<W, N> as CoreRegister>::EMPTY;
                let mut i = 0;
                // Hand-rolled: `array::map`/`zip` do not inline in target_feature code.
                while i < N {
                    out.0[i] = C::$method(a.0[i], b.0[i]);
                    i += 1;
                }
                out
            }
        }
    };
}

impl_array_sad!(Sad16Register, sad16, u16);
impl_array_sad!(Sad32Register, sad32, u32);
impl_array_sad!(Sad64Register, sad64, u64);

/// One SWAR fold step: `(x & mask) + ((x >> shift) & mask)`, summing adjacent
/// `shift`-bit fields into `2 * shift`-bit fields. Written purely in register-trait ops,
/// so it compiles for every backend.
#[inline(always)]
pub(crate) fn swar_fold<W: UnsignedIntegerRegister>(x: Storage<W>, shift: u32, mask: W::Element) -> Storage<W> {
    let mask = W::splat(mask);
    W::add(W::bitand(x, mask), W::bitand(W::shr(x, shift), mask))
}

/// Generic byte-pair sum: one fold on `u16` lanes. Backends with a widening pairwise add
/// (NEON `vpaddlq_u8`, wasm `i16x8.extadd_pairwise_i8x16_u`) override with one instruction.
#[inline(always)]
pub(crate) fn sad_cascade_u8_16<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u8>,
    W: UnsignedIntegerRegister<Element = u16> + BitCastRegister<C>,
{
    swar_fold::<W>(<W as BitCastRegister<C>>::from_bits(diffs), 8, 0x00ff)
}

/// Generic 4-byte group sum: fold to `u16` fields, then to `u32` fields.
#[inline(always)]
pub(crate) fn sad_cascade_u8_32<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u8>,
    W: UnsignedIntegerRegister<Element = u32> + BitCastRegister<C>,
{
    let x = <W as BitCastRegister<C>>::from_bits(diffs);
    let x = swar_fold::<W>(x, 8, 0x00ff_00ff);
    swar_fold::<W>(x, 16, 0x0000_ffff)
}

/// Generic 8-byte group sum: fold to `u16`, `u32`, then `u64` fields. The final step
/// needs no pre-mask - the high half is garbage that the trailing mask discards.
#[inline(always)]
pub(crate) fn sad_cascade_u8_64<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u8>,
    W: UnsignedIntegerRegister<Element = u64> + BitCastRegister<C>,
{
    let x = <W as BitCastRegister<C>>::from_bits(diffs);
    let x = swar_fold::<W>(x, 8, 0x00ff_00ff_00ff_00ff);
    let x = swar_fold::<W>(x, 16, 0x0000_ffff_0000_ffff);
    W::bitand(W::add(x, W::shr(x, 32)), W::splat(0x0000_0000_ffff_ffff))
}

/// Sum adjacent `u16` lane pairs of `|a - b|` into `u32` lanes: one fold.
#[inline(always)]
pub(crate) fn sad_cascade_u16_32<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u16>,
    W: UnsignedIntegerRegister<Element = u32> + BitCastRegister<C>,
{
    swar_fold::<W>(<W as BitCastRegister<C>>::from_bits(diffs), 16, 0x0000_ffff)
}

/// Sum groups of four `u16` lanes into `u64` lanes. After the first fold each 32-bit field
/// holds at most `2 * 65535`, so the cheap add-then-mask final step cannot overflow.
#[inline(always)]
pub(crate) fn sad_cascade_u16_64<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u16>,
    W: UnsignedIntegerRegister<Element = u64> + BitCastRegister<C>,
{
    let x = <W as BitCastRegister<C>>::from_bits(diffs);
    let x = swar_fold::<W>(x, 16, 0x0000_ffff_0000_ffff);
    W::bitand(W::add(x, W::shr(x, 32)), W::splat(0x0000_0000_ffff_ffff))
}

/// Sum adjacent `u32` lane pairs of `|a - b|` into `u64` lanes.
///
/// Unlike the narrower cascades this MUST mask before adding: two `u32` absolute
/// differences can each reach `u32::MAX`, so their sum needs 33 bits and the cheap
/// add-then-mask form used above would truncate it.
#[inline(always)]
pub(crate) fn sad_cascade_u32_64<C, W>(diffs: Storage<C>) -> Storage<W>
where
    C: UnsignedIntegerRegister<Element = u32>,
    W: UnsignedIntegerRegister<Element = u64> + BitCastRegister<C>,
{
    swar_fold::<W>(<W as BitCastRegister<C>>::from_bits(diffs), 32, 0x0000_0000_ffff_ffff)
}
