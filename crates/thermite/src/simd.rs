#![allow(non_camel_case_types)]

use core::hash::Hash;

use generic_array::{
    ArrayLength,
    typenum::{U1, U2, U3, U4, U8, U16},
};

use crate::{
    Vector,
    element::{FloatElementWithBits, USize},
    isa::InstructionSet,
    register::{
        CastRegister, ConcatRegister, ExtendRegister, FloatRegister, FullyInteroperable, IndexableRegister, Lanes,
        LinAlg3Register, LinAlg4Register, Register, SignedIntegerRegister, SwizzleRegister, UnsignedIntegerRegister,
        reduced::ReducedRegister,
        well_formed::{
            WellFormedFloatElement, WellFormedFloatRegister, WellFormedSignedIntegerElement,
            WellFormedSignedIntegerRegister, WellFormedUnsignedIntegerElement, WellFormedUnsignedIntegerRegister,
        },
    },
};

#[doc(hidden)]
#[repr(align(16))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align16;

#[doc(hidden)]
#[repr(align(32))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align32;

#[doc(hidden)]
#[repr(align(64))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align64;

#[rustfmt::skip]
pub trait NativeIsa: Clone + Copy + PartialEq + Eq + Hash {
    const ISA: InstructionSet;

    type Registers: ArrayLength;

    /// Largest native 32-bit SIMD width
    type Native32Width: Lanes;

    /// Largest native 64-bit SIMD width
    type Native64Width: Lanes;

    /// Opaque type with the minimum required alignment for native SIMD types.
    ///
    /// Include this as a field in structs that contain native SIMD types to ensure
    /// proper alignment of the containing struct.
    ///
    /// Because this cannot be instantiated easier (despite the `Default` impl),
    /// I'd recommend using a zero-length array of this type, like `[<S as NativeSimd>::NativeAlignment; 0]`,
    /// then it's easy to construction with `[]` and it doesn't take up any real space in the struct.
    ///
    /// You may also want `#[repr(C)]` to ensure the order of the fields is preserved. Note that only the first field
    /// is guaranteed to have the correct alignment, so the SIMD fields should be placed first in the struct, before any other fields.
    type NativeAlignment: Sized + Default + Copy + Ord + Hash + Send + Sync + Unpin + core::panic::UnwindSafe + core::panic::RefUnwindSafe + core::fmt::Debug + 'static;
}

/// Native-width SIMD types supported directly by the target architecture.
#[rustfmt::skip]
pub trait NativeSimd: NativeIsa {
    // Largest Native 32-bit SIMD types
    type f32xN: FullyInteroperable<Self::i32xN, Self::u32xN, Lanes = Self::Native32Width, Element = f32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + WellFormedFloatRegister<Bits = Self::u32xN, SignedBits = Self::i32xN> + IndexableRegister<Self::u32xN> + SwizzleRegister;
    type i32xN: FullyInteroperable<Self::f32xN, Self::u32xN, Lanes = Self::Native32Width, Element = i32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + WellFormedSignedIntegerRegister + IndexableRegister<Self::u32xN> + SwizzleRegister;
    type u32xN: FullyInteroperable<Self::f32xN, Self::i32xN, Lanes = Self::Native32Width, Element = u32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + WellFormedUnsignedIntegerRegister + IndexableRegister<Self::u32xN> + SwizzleRegister;

    // Largest Native 64-bit SIMD types
    type f64xN: FullyInteroperable<Self::i64xN, Self::u64xN, Lanes = Self::Native64Width, Element = f64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedFloatRegister<Bits = Self::u64xN, SignedBits = Self::i64xN> + IndexableRegister<Self::u64xN> + SwizzleRegister;
    type i64xN: FullyInteroperable<Self::f64xN, Self::u64xN, Lanes = Self::Native64Width, Element = i64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedSignedIntegerRegister + IndexableRegister<Self::u64xN> + SwizzleRegister;
    type u64xN: FullyInteroperable<Self::f64xN, Self::i64xN, Lanes = Self::Native64Width, Element = u64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedUnsignedIntegerRegister + IndexableRegister<Self::u64xN> + SwizzleRegister;
}

/// Helper traits to guarantee certain relationships between registers, like which registers can be used as
/// indices for which other registers, or which registers can be concatenated together.
pub mod helpers {
    use super::*;

    /// Helper trait to group together unsigned integer registers that can be used as indices in scatter/gather ops for a given register.
    pub trait IndexedBy<
        USIZE: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U32: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U64: UnsignedIntegerRegister<Lanes = Self::Lanes>,
    >: Register + IndexableRegister<USIZE> + IndexableRegister<U32> + IndexableRegister<U64>
    {
    }

    impl<
        USIZE: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U32: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U64: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        R,
    > IndexedBy<USIZE, U32, U64> for R
    where
        R: Register + IndexableRegister<USIZE> + IndexableRegister<U32> + IndexableRegister<U64>,
    {
    }

    pub trait FullExtendRegister<FROM: Register>:
        Register<Mask: ExtendRegister<FROM::Mask>> + ExtendRegister<FROM>
    {
    }
    pub trait FullConcatRegister<HALF: Register>:
        Register<Mask: ConcatRegister<HALF::Mask>> + FullExtendRegister<HALF> + ConcatRegister<HALF>
    {
    }

    impl<R: Register, F: Register> FullExtendRegister<F> for R where
        R: Register<Mask: ExtendRegister<F::Mask>> + ExtendRegister<F>
    {
    }
    impl<R: Register, H: Register> FullConcatRegister<H> for R where
        R: Register<Mask: ConcatRegister<H::Mask>> + FullExtendRegister<H> + ConcatRegister<H>
    {
    }

    pub trait VectorIndexedBy<
        USIZE: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U32: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U64: UnsignedIntegerVector<Lanes = Self::Lanes>,
    >: GenericVector + IndexableVector<USIZE> + IndexableVector<U32> + IndexableVector<U64>
    {
    }

    impl<
        USIZE: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U32: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U64: UnsignedIntegerVector<Lanes = Self::Lanes>,
        R,
    > VectorIndexedBy<USIZE, U32, U64> for R
    where
        R: GenericVector + IndexableVector<USIZE> + IndexableVector<U32> + IndexableVector<U64>,
    {
    }
}

use self::helpers::*;

/// Fixed-size SIMD types of various lane counts and element sizes.
#[rustfmt::skip]
pub trait Simd: NativeSimd {
    type usizex2: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U2> + SwizzleRegister
        + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2> + FullConcatRegister<USize>;
    type usizex4: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U4> + SwizzleRegister
        + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4> + FullConcatRegister<Self::usizex2>;
    type usizex8: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U8> + SwizzleRegister
        + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8> + FullConcatRegister<Self::usizex4>;
    type usizex16: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U16> + SwizzleRegister
        + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16> + FullConcatRegister<Self::usizex8>;

    // 64/32-bit SIMD types, almost always composite of scalar types
    type f32x2: WellFormedFloatRegister<Bits = Self::u32x2, SignedBits = Self::i32x2> + SwizzleRegister
        + FullyInteroperable<Self::i32x2, Self::u32x2, Lanes = U2, Element = f32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::f64x2> + FullConcatRegister<f32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type i32x2: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x2, Self::u32x2, Lanes = U2, Element = i32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::i64x2> + FullConcatRegister<i32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type u32x2: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x2, Self::i32x2, Lanes = U2, Element = u32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::u64x2> + FullConcatRegister<u32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;

    // 128/32-bit SIMD types
    type f32x4: WellFormedFloatRegister<Bits = Self::u32x4, SignedBits = Self::i32x4> + LinAlg4Register + SwizzleRegister
        + FullyInteroperable<Self::i32x4, Self::u32x4, Lanes = U4, Element = f32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::f64x4> + FullConcatRegister<Self::f32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type i32x4: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x4, Self::u32x4, Lanes = U4, Element = i32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::i64x4> + FullConcatRegister<Self::i32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type u32x4: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x4, Self::i32x4, Lanes = U4, Element = u32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::u64x4> + FullConcatRegister<Self::u32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;

    // 256/32-bit SIMD types
    type f32x8: WellFormedFloatRegister<Bits = Self::u32x8, SignedBits = Self::i32x8> + SwizzleRegister
        + FullyInteroperable<Self::i32x8, Self::u32x8, Lanes = U8, Element = f32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::f64x8> + FullConcatRegister<Self::f32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type i32x8: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x8, Self::u32x8, Lanes = U8, Element = i32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::i64x8> + FullConcatRegister<Self::i32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type u32x8: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x8, Self::i32x8, Lanes = U8, Element = u32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::u64x8> + FullConcatRegister<Self::u32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;

    // 128/64-bit SIMD types
    type f64x2: WellFormedFloatRegister<Bits = Self::u64x2, SignedBits = Self::i64x2> + SwizzleRegister
        + FullyInteroperable<Self::i64x2, Self::u64x2, Lanes = U2, Element = f64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::f32x2> + FullConcatRegister<f64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type i64x2: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x2, Self::u64x2, Lanes = U2, Element = i64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::i32x2> + FullConcatRegister<i64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type u64x2: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x2, Self::i64x2, Lanes = U2, Element = u64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::u32x2> + FullConcatRegister<u64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;

    // 256/64-bit SIMD types
    type f64x4: WellFormedFloatRegister<Bits = Self::u64x4, SignedBits = Self::i64x4> + LinAlg4Register + SwizzleRegister
        + FullyInteroperable<Self::i64x4, Self::u64x4, Lanes = U4, Element = f64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::f32x4> + FullConcatRegister<Self::f64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type i64x4: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x4, Self::u64x4, Lanes = U4, Element = i64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::i32x4> + FullConcatRegister<Self::i64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type u64x4: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x4, Self::i64x4, Lanes = U4, Element = u64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::u32x4> + FullConcatRegister<Self::u64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;

    // 512/64-bit SIMD types
    type f64x8: WellFormedFloatRegister<Bits = Self::u64x8, SignedBits = Self::i64x8> + SwizzleRegister
        + FullyInteroperable<Self::i64x8, Self::u64x8, Lanes = U8, Element = f64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::f32x8> + FullConcatRegister<Self::f64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type i64x8: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x8, Self::u64x8, Lanes = U8, Element = i64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::i32x8> + FullConcatRegister<Self::i64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type u64x8: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x8, Self::i64x8, Lanes = U8, Element = u64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::u32x8> + FullConcatRegister<Self::u64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;

    // 512/32-bit SIMD types
    type f32x16: WellFormedFloatRegister<Bits = Self::u32x16, SignedBits = Self::i32x16> + SwizzleRegister
        + FullyInteroperable<Self::i32x16, Self::u32x16, Lanes = U16, Element = f32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::f64x16> + FullConcatRegister<Self::f32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type i32x16: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x16, Self::u32x16, Lanes = U16, Element = i32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::i64x16> + FullConcatRegister<Self::i32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type u32x16: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x16, Self::i32x16, Lanes = U16, Element = u32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::u64x16> + FullConcatRegister<Self::u32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;

    // 1024/64-bit SIMD types
    type f64x16: WellFormedFloatRegister<Bits = Self::u64x16, SignedBits = Self::i64x16> + SwizzleRegister
        + FullyInteroperable<Self::i64x16, Self::u64x16, Lanes = U16, Element = f64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::f32x16> + FullConcatRegister<Self::f64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type i64x16: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x16, Self::u64x16, Lanes = U16, Element = i64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::i32x16> + FullConcatRegister<Self::i64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type u64x16: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x16, Self::i64x16, Lanes = U16, Element = u64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::u32x16> + FullConcatRegister<Self::u64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
}

/// Reduced registers to 3 lanes, for 3D math operations. They are backed by 4-lane registers internally
/// using [`ReducedRegister`], so they have the same alignment and nearly identical performance
/// characteristics as 4-lane registers, but only use 3 lanes for data.
#[rustfmt::skip]
pub trait Simd3A: Simd<
    usizex4: FullExtendRegister<Self::usizex3A>,
    f32x4: FullExtendRegister<Self::f32x3A>,
    i32x4: FullExtendRegister<Self::i32x3A>,
    u32x4: FullExtendRegister<Self::u32x3A>,
    f64x4: FullExtendRegister<Self::f64x3A>,
    i64x4: FullExtendRegister<Self::i64x3A>,
    u64x4: FullExtendRegister<Self::u64x3A>,
> {
    type usizex3A: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U3>
        + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A> + SwizzleRegister;

    type f32x3A: WellFormedFloatRegister<Bits = Self::u32x3A, SignedBits = Self::i32x3A> + LinAlg3Register + SwizzleRegister
        + FullyInteroperable<Self::i32x3A, Self::u32x3A, Lanes = U3, Element = f32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::f64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;
    type i32x3A: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x3A, Self::u32x3A, Lanes = U3, Element = i32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::i64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;
    type u32x3A: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f32x3A, Self::i32x3A, Lanes = U3, Element = u32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::u64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;

    type f64x3A: WellFormedFloatRegister<Bits = Self::u64x3A, SignedBits = Self::i64x3A> + LinAlg3Register + SwizzleRegister
        + FullyInteroperable<Self::i64x3A, Self::u64x3A, Lanes = U3, Element = f64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::f32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;
    type i64x3A: WellFormedSignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x3A, Self::u64x3A, Lanes = U3, Element = i64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::i32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;
    type u64x3A: WellFormedUnsignedIntegerRegister + SwizzleRegister
        + FullyInteroperable<Self::f64x3A, Self::i64x3A, Lanes = U3, Element = u64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::u32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;
}

impl<S: Simd> Simd3A for S {
    type usizex3A = ReducedRegister<S::usizex4, U1>;

    type f32x3A = ReducedRegister<S::f32x4, U1>;
    type i32x3A = ReducedRegister<S::i32x4, U1>;
    type u32x3A = ReducedRegister<S::u32x4, U1>;

    type f64x3A = ReducedRegister<S::f64x4, U1>;
    type i64x3A = ReducedRegister<S::i64x4, U1>;
    type u64x3A = ReducedRegister<S::u64x4, U1>;
}

/// Fixed-width SIMD registers, which may be native SIMD types or composite types of the appropriate size.
///
/// The width is specified by the `Width` type parameter, which must be a
/// type-level integer representing the number of lanes in the SIMD register.
pub trait FixedWidthSimd<Width: Lanes>: Simd {
    type usizexN: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = Width>;

    type f32xN: WellFormedFloatRegister<
            Bits = <Self as FixedWidthSimd<Width>>::u32xN,
            SignedBits = <Self as FixedWidthSimd<Width>>::i32xN,
        > + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::i32xN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            Lanes = Width,
            Element = f32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::f64xN>;
    type i32xN: WellFormedSignedIntegerRegister
        + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f32xN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            Lanes = Width,
            Element = i32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::i64xN>;
    type u32xN: WellFormedUnsignedIntegerRegister
        + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f32xN,
            <Self as FixedWidthSimd<Width>>::i32xN,
            Lanes = Width,
            Element = u32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::u64xN>;

    type f64xN: WellFormedFloatRegister<
            Bits = <Self as FixedWidthSimd<Width>>::u64xN,
            SignedBits = <Self as FixedWidthSimd<Width>>::i64xN,
        > + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::i64xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
            Lanes = Width,
            Element = f64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::f32xN>;
    type i64xN: WellFormedSignedIntegerRegister
        + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f64xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
            Lanes = Width,
            Element = i64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::i32xN>;
    type u64xN: WellFormedUnsignedIntegerRegister
        + SwizzleRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f64xN,
            <Self as FixedWidthSimd<Width>>::i64xN,
            Lanes = Width,
            Element = u64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::u32xN>;
}

macro_rules! impl_wide_simd {
    ($($width:literal),*) => {paste::paste! { $(
        impl<S: Simd> FixedWidthSimd<[<U $width>]> for S {
            type usizexN = S::[<usizex $width>];

            type f32xN = S::[<f32x $width>];
            type i32xN = S::[<i32x $width>];
            type u32xN = S::[<u32x $width>];

            type f64xN = S::[<f64x $width>];
            type i64xN = S::[<i64x $width>];
            type u64xN = S::[<u64x $width>];
        }
    )* }}
}

impl<S: Simd> FixedWidthSimd<U1> for S {
    type usizexN = crate::element::USize;

    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

impl_wide_simd!(2, 4, 8, 16);

/// SIMD types of the same element size but different lane counts, all based
/// on the given fully formed element types.
///
/// For example, `SizedSimd<f32, i32, u32>` provides all the fixed-size SIMD types
/// with 32-bit elements, and `SizedSimd<f64, i64, u64>` provides all the fixed-size SIMD types
/// with 64-bit elements.
///
/// You can use this generically to write code that works with either 32-bit or 64-bit
/// floating-point SIMD types and their associated integer types, like so:
/// ```ignore
/// fn my_func<S, T>(values: &[T]) -> T
/// where
///     T: WellFormedFloatElement,
///     S: SizedSimd<T, <T as FloatElementWithBits>::SignedBits, <T as FloatElementWithBits>::Bits>,
/// {
///     // do whatever you need with S::fxN, S::ixN, S::uxN, etc.
///     let (scalar_prefix, vectors, scalar_suffix) = Vector::<S::fxN>::from_slice(values);
/// }
/// ```
#[rustfmt::skip]
pub trait SizedSimd<
    F: WellFormedFloatElement + FloatElementWithBits<SignedBits = I, Bits = U>,
    I: WellFormedSignedIntegerElement,
    U: WellFormedUnsignedIntegerElement,
>: Simd {
    type NativeWidth: Lanes;

    // TODO: Figure out how to make these all WellFormed registers. There is currently some kind of mismatch
    // between the expected inner Element type and what is provided via the generic parameters. Specifying them
    // is weird/difficult.

    type fxN: FullyInteroperable<Self::ixN, Self::uxN, Lanes = Self::NativeWidth, Element = F, Unsigned = Self::uxN, Signed = Self::ixN> + SwizzleRegister
        + FloatRegister<Bits = Self::uxN, SignedBits = Self::ixN> + IndexableRegister<Self::uxN>;
    type ixN: FullyInteroperable<Self::fxN, Self::uxN, Lanes = Self::NativeWidth, Element = I, Unsigned = Self::uxN, Signed = Self::ixN> + SwizzleRegister
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + IndexableRegister<Self::uxN>;
    type uxN: FullyInteroperable<Self::fxN, Self::ixN, Lanes = Self::NativeWidth, Element = U, Unsigned = Self::uxN, Signed = Self::ixN> + SwizzleRegister
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + IndexableRegister<Self::uxN>;

    type fx2: FullyInteroperable<Self::ix2, Self::ux2, Lanes = U2, Element = F, Unsigned = Self::ux2, Signed = Self::ix2> + SwizzleRegister
        + FloatRegister<Bits = Self::ux2, SignedBits = Self::ix2> + FullConcatRegister<F> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type ix2: FullyInteroperable<Self::fx2, Self::ux2, Lanes = U2, Element = I, Unsigned = Self::ux2, Signed = Self::ix2> + SwizzleRegister
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<I> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type ux2: FullyInteroperable<Self::fx2, Self::ix2, Lanes = U2, Element = U, Unsigned = Self::ux2, Signed = Self::ix2> + SwizzleRegister
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<U> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;

    type fx4: FullyInteroperable<Self::ix4, Self::ux4, Lanes = U4, Element = F, Unsigned = Self::ux4, Signed = Self::ix4> + SwizzleRegister
        + FloatRegister<Bits = Self::ux4, SignedBits = Self::ix4> + LinAlg4Register
        + FullConcatRegister<Self::fx2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type ix4: FullyInteroperable<Self::fx4, Self::ux4, Lanes = U4, Element = I, Unsigned = Self::ux4, Signed = Self::ix4> + SwizzleRegister
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type ux4: FullyInteroperable<Self::fx4, Self::ix4, Lanes = U4, Element = U, Unsigned = Self::ux4, Signed = Self::ix4> + SwizzleRegister
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;

    type fx8: FullyInteroperable<Self::ix8, Self::ux8, Lanes = U8, Element = F, Unsigned = Self::ux8, Signed = Self::ix8> + SwizzleRegister
        + FloatRegister<Bits = Self::ux8, SignedBits = Self::ix8> + FullConcatRegister<Self::fx4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type ix8: FullyInteroperable<Self::fx8, Self::ux8, Lanes = U8, Element = I, Unsigned = Self::ux8, Signed = Self::ix8> + SwizzleRegister
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type ux8: FullyInteroperable<Self::fx8, Self::ix8, Lanes = U8, Element = U, Unsigned = Self::ux8, Signed = Self::ix8> + SwizzleRegister
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;

    type fx16: FullyInteroperable<Self::ix16, Self::ux16, Lanes = U16, Element = F, Unsigned = Self::ux16, Signed = Self::ix16> + SwizzleRegister
        + FloatRegister<Bits = Self::ux16, SignedBits = Self::ix16> + FullConcatRegister<Self::fx8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type ix16: FullyInteroperable<Self::fx16, Self::ux16, Lanes = U16, Element = I, Unsigned = Self::ux16, Signed = Self::ix16> + SwizzleRegister
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type ux16: FullyInteroperable<Self::fx16, Self::ix16, Lanes = U16, Element = U, Unsigned = Self::ux16, Signed = Self::ix16> + SwizzleRegister
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
}

/// SIMD types for floating-point elements and their associated signed and unsigned integer types.
pub trait FloatSimd<F: WellFormedFloatElement + FloatElementWithBits>:
    SizedSimd<F, <F as FloatElementWithBits>::SignedBits, <F as FloatElementWithBits>::Bits>
{
}

impl<S, F> FloatSimd<F> for S
where
    F: WellFormedFloatElement + FloatElementWithBits,
    S: SizedSimd<F, <F as FloatElementWithBits>::SignedBits, <F as FloatElementWithBits>::Bits>,
{
}

impl<S: Simd> SizedSimd<f32, i32, u32> for S {
    type NativeWidth = S::Native32Width;

    type fxN = S::f32xN;
    type ixN = S::i32xN;
    type uxN = S::u32xN;

    type fx2 = S::f32x2;
    type ix2 = S::i32x2;
    type ux2 = S::u32x2;

    type fx4 = S::f32x4;
    type ix4 = S::i32x4;
    type ux4 = S::u32x4;

    type fx8 = S::f32x8;
    type ix8 = S::i32x8;
    type ux8 = S::u32x8;

    type fx16 = S::f32x16;
    type ix16 = S::i32x16;
    type ux16 = S::u32x16;
}

impl<S: Simd> SizedSimd<f64, i64, u64> for S {
    type NativeWidth = S::Native64Width;

    type fxN = S::f64xN;
    type ixN = S::i64xN;
    type uxN = S::u64xN;

    type fx2 = S::f64x2;
    type ix2 = S::i64x2;
    type ux2 = S::u64x2;

    type fx4 = S::f64x4;
    type ix4 = S::i64x4;
    type ux4 = S::u64x4;

    type fx8 = S::f64x8;
    type ix8 = S::i64x8;
    type ux8 = S::u64x8;

    type fx16 = S::f64x16;
    type ix16 = S::i64x16;
    type ux16 = S::u64x16;
}

pub type f32xN<S> = Vector<<S as NativeSimd>::f32xN>;
pub type i32xN<S> = Vector<<S as NativeSimd>::i32xN>;
pub type u32xN<S> = Vector<<S as NativeSimd>::u32xN>;

pub type f64xN<S> = Vector<<S as NativeSimd>::f64xN>;
pub type i64xN<S> = Vector<<S as NativeSimd>::i64xN>;
pub type u64xN<S> = Vector<<S as NativeSimd>::u64xN>;

pub type f32x2<S> = Vector<<S as Simd>::f32x2>;
pub type i32x2<S> = Vector<<S as Simd>::i32x2>;
pub type u32x2<S> = Vector<<S as Simd>::u32x2>;

pub type f32x3A<S> = Vector<<S as Simd3A>::f32x3A>;
pub type i32x3A<S> = Vector<<S as Simd3A>::i32x3A>;
pub type u32x3A<S> = Vector<<S as Simd3A>::u32x3A>;

pub type f32x4<S> = Vector<<S as Simd>::f32x4>;
pub type i32x4<S> = Vector<<S as Simd>::i32x4>;
pub type u32x4<S> = Vector<<S as Simd>::u32x4>;

pub type f32x8<S> = Vector<<S as Simd>::f32x8>;
pub type i32x8<S> = Vector<<S as Simd>::i32x8>;
pub type u32x8<S> = Vector<<S as Simd>::u32x8>;

pub type f64x2<S> = Vector<<S as Simd>::f64x2>;
pub type i64x2<S> = Vector<<S as Simd>::i64x2>;
pub type u64x2<S> = Vector<<S as Simd>::u64x2>;

pub type f64x3A<S> = Vector<<S as Simd3A>::f64x3A>;
pub type i64x3A<S> = Vector<<S as Simd3A>::i64x3A>;
pub type u64x3A<S> = Vector<<S as Simd3A>::u64x3A>;

pub type f64x4<S> = Vector<<S as Simd>::f64x4>;
pub type i64x4<S> = Vector<<S as Simd>::i64x4>;
pub type u64x4<S> = Vector<<S as Simd>::u64x4>;

pub type f64x8<S> = Vector<<S as Simd>::f64x8>;
pub type i64x8<S> = Vector<<S as Simd>::i64x8>;
pub type u64x8<S> = Vector<<S as Simd>::u64x8>;

pub type f32x16<S> = Vector<<S as Simd>::f32x16>;
pub type i32x16<S> = Vector<<S as Simd>::i32x16>;
pub type u32x16<S> = Vector<<S as Simd>::u32x16>;

pub type f64x16<S> = Vector<<S as Simd>::f64x16>;
pub type i64x16<S> = Vector<<S as Simd>::i64x16>;
pub type u64x16<S> = Vector<<S as Simd>::u64x16>;

macro_rules! decl_aliases {
    ($simd:ty) => {
        #[allow(non_camel_case_types)]
        pub mod aliases {
            use super::*;

            pub type f32xN = crate::simd::f32xN<$simd>;
            pub type i32xN = crate::simd::i32xN<$simd>;
            pub type u32xN = crate::simd::u32xN<$simd>;

            pub type f64xN = crate::simd::f64xN<$simd>;
            pub type i64xN = crate::simd::i64xN<$simd>;
            pub type u64xN = crate::simd::u64xN<$simd>;

            pub type f32x2 = crate::simd::f32x2<$simd>;
            pub type i32x2 = crate::simd::i32x2<$simd>;
            pub type u32x2 = crate::simd::u32x2<$simd>;

            pub type f32x3A = crate::simd::f32x3A<$simd>;
            pub type i32x3A = crate::simd::i32x3A<$simd>;
            pub type u32x3A = crate::simd::u32x3A<$simd>;

            pub type f32x4 = crate::simd::f32x4<$simd>;
            pub type i32x4 = crate::simd::i32x4<$simd>;
            pub type u32x4 = crate::simd::u32x4<$simd>;

            pub type f32x8 = crate::simd::f32x8<$simd>;
            pub type i32x8 = crate::simd::i32x8<$simd>;
            pub type u32x8 = crate::simd::u32x8<$simd>;

            pub type f64x2 = crate::simd::f64x2<$simd>;
            pub type i64x2 = crate::simd::i64x2<$simd>;
            pub type u64x2 = crate::simd::u64x2<$simd>;

            pub type f64x3A = crate::simd::f64x3A<$simd>;
            pub type i64x3A = crate::simd::i64x3A<$simd>;
            pub type u64x3A = crate::simd::u64x3A<$simd>;

            pub type f64x4 = crate::simd::f64x4<$simd>;
            pub type i64x4 = crate::simd::i64x4<$simd>;
            pub type u64x4 = crate::simd::u64x4<$simd>;

            pub type f64x8 = crate::simd::f64x8<$simd>;
            pub type i64x8 = crate::simd::i64x8<$simd>;
            pub type u64x8 = crate::simd::u64x8<$simd>;

            pub type f32x16 = crate::simd::f32x16<$simd>;
            pub type i32x16 = crate::simd::i32x16<$simd>;
            pub type u32x16 = crate::simd::u32x16<$simd>;

            pub type f64x16 = crate::simd::f64x16<$simd>;
            pub type i64x16 = crate::simd::i64x16<$simd>;
            pub type u64x16 = crate::simd::u64x16<$simd>;
        }
    };
}

use crate::generic::{
    CastVector, ConcatVector, ExtendVector, FloatVector, FloatVectorWithBits, FloatVectorWithRegister,
    FullyInteroperable as FIV, GenericVector, IndexableVector, LinAlg4Vector, SignedIntegerVector,
    SignedIntegerVectorWithRegister, SwizzleVector, UnsignedIntegerVector, UnsignedIntegerVectorWithRegister,
};

pub trait NativeSimdVectors: NativeIsa {
    type f32xN: FloatVector<Lanes = <Self as NativeIsa>::Native32Width, Element = f32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>;
    type i32xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native32Width, Element = i32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>;
    type u32xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native32Width, Element = u32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>;

    type f64xN: FloatVector<Lanes = <Self as NativeIsa>::Native64Width, Element = f64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;
    type i64xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native64Width, Element = i64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;
    type u64xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native64Width, Element = u64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;
}

pub trait NativeSimdVectorsWithRegisters: NativeSimd + NativeSimdVectors<
    // 32xN
    f32xN: FloatVectorWithRegister<
        Register = <Self as NativeSimd>::f32xN,
        SignedBits = <Self as NativeSimdVectors>::i32xN,
        Bits = <Self as NativeSimdVectors>::u32xN,
    > + FIV<<Self as NativeSimdVectors>::i32xN, <Self as NativeSimdVectors>::u32xN>,
    i32xN: SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i32xN>
        + FIV<<Self as NativeSimdVectors>::f32xN, <Self as NativeSimdVectors>::u32xN>,
    u32xN: UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u32xN>
        + FIV<<Self as NativeSimdVectors>::f32xN, <Self as NativeSimdVectors>::i32xN>,

    // 64xN
    f64xN: FloatVectorWithRegister<
        Register = <Self as NativeSimd>::f64xN,
        SignedBits = <Self as NativeSimdVectors>::i64xN,
        Bits = <Self as NativeSimdVectors>::u64xN,
    > + FIV<<Self as NativeSimdVectors>::i64xN, <Self as NativeSimdVectors>::u64xN>,
    i64xN: SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i64xN>
        + FIV<<Self as NativeSimdVectors>::f64xN, <Self as NativeSimdVectors>::u64xN>,
    u64xN: UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u64xN>
        + FIV<<Self as NativeSimdVectors>::f64xN, <Self as NativeSimdVectors>::i64xN>,
>
{}

impl<S: NativeSimd> NativeSimdVectors for S {
    type f32xN = Vector<<Self as NativeSimd>::f32xN>;
    type i32xN = Vector<<Self as NativeSimd>::i32xN>;
    type u32xN = Vector<<Self as NativeSimd>::u32xN>;

    type f64xN = Vector<<Self as NativeSimd>::f64xN>;
    type i64xN = Vector<<Self as NativeSimd>::i64xN>;
    type u64xN = Vector<<Self as NativeSimd>::u64xN>;
}

impl<S: NativeSimd> NativeSimdVectorsWithRegisters for S {}

#[rustfmt::skip]
pub trait SimdVectors: NativeSimdVectors {
    type usizex2: UnsignedIntegerVector<Lanes = U2, Element = crate::element::USize>
        + ConcatVector<Vector<crate::element::USize>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type usizex4: UnsignedIntegerVector<Lanes = U4, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type usizex8: UnsignedIntegerVector<Lanes = U8, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type usizex16: UnsignedIntegerVector<Lanes = U16, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;

    type f32x2: FloatVector<Lanes = U2, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x2>
        + ConcatVector<Vector<f32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type i32x2: SignedIntegerVector<Lanes = U2, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x2>
        + ConcatVector<Vector<i32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type u32x2: UnsignedIntegerVector<Lanes = U2, Element = u32>
        + CastVector<<Self as SimdVectors>::u64x2>
        + ConcatVector<Vector<u32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;

    type f32x4: FloatVector<Lanes = U4, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x4>
        + ConcatVector<<Self as SimdVectors>::f32x2> + SwizzleVector
        + LinAlg4Vector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type i32x4: SignedIntegerVector<Lanes = U4, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x4>
        + ConcatVector<<Self as SimdVectors>::i32x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type u32x4: UnsignedIntegerVector<Lanes = U4, Element = u32>
        + CastVector<<Self as SimdVectors>::u64x4>
        + ConcatVector<<Self as SimdVectors>::u32x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;

    type f32x8: FloatVector<Lanes = U8, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x8>
        + ConcatVector<<Self as SimdVectors>::f32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type i32x8: SignedIntegerVector<Lanes = U8, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x8>
        + ConcatVector<<Self as SimdVectors>::i32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type u32x8: UnsignedIntegerVector<Lanes = U8, Element = u32>
        + CastVector<<Self as SimdVectors>::u64x8>
        + ConcatVector<<Self as SimdVectors>::u32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;

    type f64x2: FloatVector<Lanes = U2, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x2>
        + ConcatVector<Vector<f64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type i64x2: SignedIntegerVector<Lanes = U2, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x2>
        + ConcatVector<Vector<i64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type u64x2: UnsignedIntegerVector<Lanes = U2, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x2>
        + ConcatVector<Vector<u64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;

    type f64x4: FloatVector<Lanes = U4, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x4>
        + ConcatVector<<Self as SimdVectors>::f64x2> + SwizzleVector
        + LinAlg4Vector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type i64x4: SignedIntegerVector<Lanes = U4, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x4>
        + ConcatVector<<Self as SimdVectors>::i64x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type u64x4: UnsignedIntegerVector<Lanes = U4, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x4>
        + ConcatVector<<Self as SimdVectors>::u64x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;

    type f64x8: FloatVector<Lanes = U8, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x8>
        + ConcatVector<<Self as SimdVectors>::f64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type i64x8: SignedIntegerVector<Lanes = U8, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x8>
        + ConcatVector<<Self as SimdVectors>::i64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type u64x8: UnsignedIntegerVector<Lanes = U8, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x8>
        + ConcatVector<<Self as SimdVectors>::u64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;

    type f32x16: FloatVector<Lanes = U16, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x16>
        + ConcatVector<<Self as SimdVectors>::f32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;
    type i32x16: SignedIntegerVector<Lanes = U16, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x16>
        + ConcatVector<<Self as SimdVectors>::i32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;
    type u32x16: UnsignedIntegerVector<Lanes = U16, Element = u32>
        + CastVector<<Self as SimdVectors>::u64x16>
        + ConcatVector<<Self as SimdVectors>::u32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;

    type f64x16: FloatVector<Lanes = U16, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x16>
        + ConcatVector<<Self as SimdVectors>::f64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;
    type i64x16: SignedIntegerVector<Lanes = U16, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x16>
        + ConcatVector<<Self as SimdVectors>::i64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;
    type u64x16: UnsignedIntegerVector<Lanes = U16, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x16>
        + ConcatVector<<Self as SimdVectors>::u64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;
}

pub trait SimdVectorsWithRegisters: Simd + NativeSimdVectorsWithRegisters + SimdVectors<
    // usizes
    usizex2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex2>,
    usizex4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex4>,
    usizex8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex8>,
    usizex16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex16>,

    // 32x2
    f32x2: FloatVectorWithRegister<Register = <Self as Simd>::f32x2, SignedBits = <Self as SimdVectors>::i32x2, Bits = <Self as SimdVectors>::u32x2>
        + FIV<<Self as SimdVectors>::i32x2, <Self as SimdVectors>::u32x2>,
    i32x2: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x2>
        + FIV<<Self as SimdVectors>::f32x2, <Self as SimdVectors>::u32x2>,
    u32x2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x2>
        + FIV<<Self as SimdVectors>::f32x2, <Self as SimdVectors>::i32x2>,

    // 32x4
    f32x4: FloatVectorWithRegister<Register = <Self as Simd>::f32x4, SignedBits = <Self as SimdVectors>::i32x4, Bits = <Self as SimdVectors>::u32x4>
        + FIV<<Self as SimdVectors>::i32x4, <Self as SimdVectors>::u32x4>,
    i32x4: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x4>
        + FIV<<Self as SimdVectors>::f32x4, <Self as SimdVectors>::u32x4>,
    u32x4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x4>
        + FIV<<Self as SimdVectors>::f32x4, <Self as SimdVectors>::i32x4>,

    // 32x8
    f32x8: FloatVectorWithRegister<Register = <Self as Simd>::f32x8, SignedBits = <Self as SimdVectors>::i32x8, Bits = <Self as SimdVectors>::u32x8>
        + FIV<<Self as SimdVectors>::i32x8, <Self as SimdVectors>::u32x8>,
    i32x8: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x8>
        + FIV<<Self as SimdVectors>::f32x8, <Self as SimdVectors>::u32x8>,
    u32x8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x8>
        + FIV<<Self as SimdVectors>::f32x8, <Self as SimdVectors>::i32x8>,

    // 32x16
    f32x16: FloatVectorWithRegister<Register = <Self as Simd>::f32x16, SignedBits = <Self as SimdVectors>::i32x16, Bits = <Self as SimdVectors>::u32x16>
        + FIV<<Self as SimdVectors>::i32x16, <Self as SimdVectors>::u32x16>,
    i32x16: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x16>
        + FIV<<Self as SimdVectors>::f32x16, <Self as SimdVectors>::u32x16>,
    u32x16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x16>
        + FIV<<Self as SimdVectors>::f32x16, <Self as SimdVectors>::i32x16>,

    // 64x2
    f64x2: FloatVectorWithRegister<Register = <Self as Simd>::f64x2, SignedBits = <Self as SimdVectors>::i64x2, Bits = <Self as SimdVectors>::u64x2>,
    i64x2: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x2>
        + FIV<<Self as SimdVectors>::f64x2, <Self as SimdVectors>::u64x2>,
    u64x2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x2>
        + FIV<<Self as SimdVectors>::f64x2, <Self as SimdVectors>::i64x2>,

    // 64x4
    f64x4: FloatVectorWithRegister<Register = <Self as Simd>::f64x4, SignedBits = <Self as SimdVectors>::i64x4, Bits = <Self as SimdVectors>::u64x4>
        + FIV<<Self as SimdVectors>::i64x4, <Self as SimdVectors>::u64x4>,
    i64x4: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x4>
        + FIV<<Self as SimdVectors>::f64x4, <Self as SimdVectors>::u64x4>,
    u64x4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x4>
        + FIV<<Self as SimdVectors>::f64x4, <Self as SimdVectors>::i64x4>,

    // 64x8
    f64x8: FloatVectorWithRegister<Register = <Self as Simd>::f64x8, SignedBits = <Self as SimdVectors>::i64x8, Bits = <Self as SimdVectors>::u64x8>
        + FIV<<Self as SimdVectors>::i64x8, <Self as SimdVectors>::u64x8>,
    i64x8: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x8>
        + FIV<<Self as SimdVectors>::f64x8, <Self as SimdVectors>::u64x8>,
    u64x8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x8>
        + FIV<<Self as SimdVectors>::f64x8, <Self as SimdVectors>::i64x8>,

    // 64x16
    f64x16: FloatVectorWithRegister<Register = <Self as Simd>::f64x16, SignedBits = <Self as SimdVectors>::i64x16, Bits = <Self as SimdVectors>::u64x16>
        + FIV<<Self as SimdVectors>::i64x16, <Self as SimdVectors>::u64x16>,
    i64x16: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x16>
        + FIV<<Self as SimdVectors>::f64x16, <Self as SimdVectors>::u64x16>,
    u64x16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x16>
        + FIV<<Self as SimdVectors>::f64x16, <Self as SimdVectors>::i64x16>,
>{}

pub trait Simd3AVectors:
    SimdVectors<
        usizex4: ExtendVector<Self::usizex3A>,
        f32x4: ExtendVector<Self::f32x3A>,
        i32x4: ExtendVector<Self::i32x3A>,
        u32x4: ExtendVector<Self::u32x3A>,
        f64x4: ExtendVector<Self::f64x3A>,
        i64x4: ExtendVector<Self::i64x3A>,
        u64x4: ExtendVector<Self::u64x3A>,
    >
{
    type usizex3A: UnsignedIntegerVector<Lanes = U3, Element = crate::element::USize>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;

    type f32x3A: FloatVector<Lanes = U3, Element = f32>
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;
    type i32x3A: SignedIntegerVector<Lanes = U3, Element = i32>
        + CastVector<<Self as Simd3AVectors>::i64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;
    type u32x3A: UnsignedIntegerVector<Lanes = U3, Element = u32>
        + CastVector<<Self as Simd3AVectors>::u64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;

    type f64x3A: FloatVector<Lanes = U3, Element = f64>
        + CastVector<<Self as Simd3AVectors>::f32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;
    type i64x3A: SignedIntegerVector<Lanes = U3, Element = i64>
        + CastVector<<Self as Simd3AVectors>::i32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;
    type u64x3A: UnsignedIntegerVector<Lanes = U3, Element = u64>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;
}

pub trait Simd3AVectorsWithRegisters: SimdVectorsWithRegisters + Simd3A + Simd3AVectors<
    // usizex3A
    usizex3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::usizex3A>,

    // 32x3A
    f32x3A: FloatVectorWithRegister<Register = <Self as Simd3A>::f32x3A, SignedBits = <Self as Simd3AVectors>::i32x3A, Bits = <Self as Simd3AVectors>::u32x3A>
        + FIV<<Self as Simd3AVectors>::i32x3A, <Self as Simd3AVectors>::u32x3A>,
    i32x3A: SignedIntegerVectorWithRegister<Register = <Self as Simd3A>::i32x3A>
        + FIV<<Self as Simd3AVectors>::f32x3A, <Self as Simd3AVectors>::u32x3A>,
    u32x3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::u32x3A>
        + FIV<<Self as Simd3AVectors>::f32x3A, <Self as Simd3AVectors>::i32x3A>,

    // 64x3A
    f64x3A: FloatVectorWithRegister<Register = <Self as Simd3A>::f64x3A, SignedBits = <Self as Simd3AVectors>::i64x3A, Bits = <Self as Simd3AVectors>::u64x3A>
        + FIV<<Self as Simd3AVectors>::i64x3A, <Self as Simd3AVectors>::u64x3A>,
    i64x3A: SignedIntegerVectorWithRegister<Register = <Self as Simd3A>::i64x3A>
        + FIV<<Self as Simd3AVectors>::f64x3A, <Self as Simd3AVectors>::u64x3A>,
    u64x3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::u64x3A>
        + FIV<<Self as Simd3AVectors>::f64x3A, <Self as Simd3AVectors>::i64x3A>,
>{}

impl<S: Simd> SimdVectors for S {
    type usizex2 = Vector<<Self as Simd>::usizex2>;
    type usizex4 = Vector<<Self as Simd>::usizex4>;
    type usizex8 = Vector<<Self as Simd>::usizex8>;
    type usizex16 = Vector<<Self as Simd>::usizex16>;

    type f32x2 = Vector<<Self as Simd>::f32x2>;
    type i32x2 = Vector<<Self as Simd>::i32x2>;
    type u32x2 = Vector<<Self as Simd>::u32x2>;

    type f32x4 = Vector<<Self as Simd>::f32x4>;
    type i32x4 = Vector<<Self as Simd>::i32x4>;
    type u32x4 = Vector<<Self as Simd>::u32x4>;

    type f32x8 = Vector<<Self as Simd>::f32x8>;
    type i32x8 = Vector<<Self as Simd>::i32x8>;
    type u32x8 = Vector<<Self as Simd>::u32x8>;

    type f64x2 = Vector<<Self as Simd>::f64x2>;
    type i64x2 = Vector<<Self as Simd>::i64x2>;
    type u64x2 = Vector<<Self as Simd>::u64x2>;

    type f64x4 = Vector<<Self as Simd>::f64x4>;
    type i64x4 = Vector<<Self as Simd>::i64x4>;
    type u64x4 = Vector<<Self as Simd>::u64x4>;

    type f64x8 = Vector<<Self as Simd>::f64x8>;
    type i64x8 = Vector<<Self as Simd>::i64x8>;
    type u64x8 = Vector<<Self as Simd>::u64x8>;

    type f32x16 = Vector<<Self as Simd>::f32x16>;
    type i32x16 = Vector<<Self as Simd>::i32x16>;
    type u32x16 = Vector<<Self as Simd>::u32x16>;

    type f64x16 = Vector<<Self as Simd>::f64x16>;
    type i64x16 = Vector<<Self as Simd>::i64x16>;
    type u64x16 = Vector<<Self as Simd>::u64x16>;
}

impl<S: Simd> SimdVectorsWithRegisters for S {}

impl<S: Simd3A> Simd3AVectors for S {
    type usizex3A = Vector<<Self as Simd3A>::usizex3A>;

    type f32x3A = Vector<<Self as Simd3A>::f32x3A>;
    type i32x3A = Vector<<Self as Simd3A>::i32x3A>;
    type u32x3A = Vector<<Self as Simd3A>::u32x3A>;

    type f64x3A = Vector<<Self as Simd3A>::f64x3A>;
    type i64x3A = Vector<<Self as Simd3A>::i64x3A>;
    type u64x3A = Vector<<Self as Simd3A>::u64x3A>;
}

impl<S: Simd3A> Simd3AVectorsWithRegisters for S {}
