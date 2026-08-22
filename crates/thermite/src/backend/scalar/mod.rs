#![allow(clippy::useless_transmute, unnecessary_transmutes)]

use crate::{
    element::USize,
    register::{ExtendRegister, Storage, array::ArrayRegister},
    simd::{NativeIsa, NativeSimd, Simd, Simd3},
};

cfg_if::cfg_if! {
    if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
        #[path = "spirv/mod.rs"]
        mod registers;
    } else {
        #[path = "cpu/mod.rs"]
        mod registers;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar;

pub mod prelude {
    pub use super::Scalar;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}

#[thermite_macros::inline_always]
impl NativeIsa for Scalar {
    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;
    type Native16Width = generic_array::typenum::U1;
    type Native8Width = generic_array::typenum::U1;

    type NativeAlignment = ();

    // Prefetch is a memory hint, not a SIMD op: the scalar backend gets the host's
    // real instruction wherever one exists (x86, aarch64), not a no-op.
    const HAS_PREFETCH: bool = crate::backend::prefetch::HAS_PREFETCH;

    fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
        crate::backend::prefetch::prefetch::<LOCALITY, WRITE>(ptr);
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    unsafe fn disable_denormals() -> Result<bool, crate::simd::UnsupportedError> {
        unsafe { Ok(crate::backend::x86::sse2::disable_denormals()) }
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    #[allow(clippy::unit_arg)]
    unsafe fn enable_denormals() -> Result<(), crate::simd::UnsupportedError> {
        unsafe { Ok(crate::backend::x86::sse2::enable_denormals()) }
    }
}

impl NativeSimd for Scalar {
    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;

    type i16xN = i16;
    type u16xN = u16;

    type i8xN = i8;
    type u8xN = u8;
}

impl Simd for Scalar {
    type usizex2 = ArrayRegister<USize, 2>;
    type usizex4 = ArrayRegister<USize, 4>;
    type usizex8 = ArrayRegister<USize, 8>;
    type usizex16 = ArrayRegister<USize, 16>;

    type f32x2 = ArrayRegister<f32, 2>;
    type i32x2 = ArrayRegister<i32, 2>;
    type u32x2 = ArrayRegister<u32, 2>;

    type f32x4 = ArrayRegister<f32, 4>;
    type i32x4 = ArrayRegister<i32, 4>;
    type u32x4 = ArrayRegister<u32, 4>;

    type f32x8 = ArrayRegister<f32, 8>;
    type i32x8 = ArrayRegister<i32, 8>;
    type u32x8 = ArrayRegister<u32, 8>;

    type f64x2 = ArrayRegister<f64, 2>;
    type i64x2 = ArrayRegister<i64, 2>;
    type u64x2 = ArrayRegister<u64, 2>;

    type f64x4 = ArrayRegister<f64, 4>;
    type i64x4 = ArrayRegister<i64, 4>;
    type u64x4 = ArrayRegister<u64, 4>;

    type f64x8 = ArrayRegister<f64, 8>;
    type i64x8 = ArrayRegister<i64, 8>;
    type u64x8 = ArrayRegister<u64, 8>;

    type f32x16 = ArrayRegister<f32, 16>;
    type i32x16 = ArrayRegister<i32, 16>;
    type u32x16 = ArrayRegister<u32, 16>;

    type f64x16 = ArrayRegister<f64, 16>;
    type i64x16 = ArrayRegister<i64, 16>;
    type u64x16 = ArrayRegister<u64, 16>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;
    type i16x4 = ArrayRegister<i16, 4>;
    type u16x4 = ArrayRegister<u16, 4>;
    type i16x8 = ArrayRegister<i16, 8>;
    type u16x8 = ArrayRegister<u16, 8>;
    type i16x16 = ArrayRegister<i16, 16>;
    type u16x16 = ArrayRegister<u16, 16>;

    type i8x16 = ArrayRegister<i8, 16>;
    type u8x16 = ArrayRegister<u8, 16>;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = ArrayRegister<i8, 4>;
    type u8x4 = ArrayRegister<u8, 4>;
    type i8x8 = ArrayRegister<i8, 8>;
    type u8x8 = ArrayRegister<u8, 8>;
}

// Same-width, different-lane-count reinterprets of the 16-byte scalar byte register, so
// it can be viewed as wider accumulator lanes (the SAD family). These are distinct
// `ArrayRegister` shapes rather than one shared intrinsic, so they transmute by value;
// the element-wise array bitcast only relates arrays of equal lane count.
impl_bit_casts_transmute! {
    ArrayRegister<u8, 16> as ArrayRegister<u16, 8>,
    ArrayRegister<u8, 16> as ArrayRegister<u32, 4>,
    ArrayRegister<u8, 16> as ArrayRegister<u64, 2>,
    // ... and the same-width pairs one and two element sizes up (the 2:1 and 4:1
    // array ratios already come from the generic impls in `register/array.rs`).
    ArrayRegister<u16, 8> as ArrayRegister<u64, 2>,
    ArrayRegister<u16, 2> as u32,
    ArrayRegister<u16, 4> as u64,
    ArrayRegister<u32, 2> as u64,
}

impl_sad!(ArrayRegister<u8, 16> => (ArrayRegister<u16, 8>, ArrayRegister<u32, 4>, ArrayRegister<u64, 2>));

// Sub-native byte ladder: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    ArrayRegister<u8, 8> => (ArrayRegister<u16, 4>, ArrayRegister<u32, 2>, u64),
    ArrayRegister<u8, 4> => (ArrayRegister<u16, 2>, u32, u64),
}

// Wider-element SAD. The scalar backend is the differential oracle, so every rung takes
// the obvious lane-wise path rather than a reinterpret + SWAR cascade.
impl_sad_u16! {
    @scalar
    ArrayRegister<u16, 8> => (ArrayRegister<u32, 4>, ArrayRegister<u64, 2>),
    ArrayRegister<u16, 4> => (ArrayRegister<u32, 2>, u64),
}

impl_sad_u32! {
    @scalar
    ArrayRegister<u32, 4> => ArrayRegister<u64, 2>,
    ArrayRegister<u32, 2> => u64,
    ArrayRegister<u32, 8> => ArrayRegister<u64, 4>,
    ArrayRegister<u32, 16> => ArrayRegister<u64, 8>,
}

impl_sad_u16! {
    @scalar
    ArrayRegister<u16, 16> => (ArrayRegister<u32, 8>, ArrayRegister<u64, 4>),
}

// The scalar backend's `xN` slots are the bare 1-lane elements, so every grouping is
// partial there: one output lane summing the register's single value.
impl_sad_scalar!(u8 => (u16, u32, u64));

impl_sad_u16!(@scalar u16 => (u32, u64));
impl_sad_u32!(@scalar u32 => u64);

impl Simd3 for Scalar {
    type usizex3 = ArrayRegister<USize, 3>;

    type f32x3 = ArrayRegister<f32, 3>;
    type i32x3 = ArrayRegister<i32, 3>;
    type u32x3 = ArrayRegister<u32, 3>;

    type f64x3 = ArrayRegister<f64, 3>;
    type i64x3 = ArrayRegister<i64, 3>;
    type u64x3 = ArrayRegister<u64, 3>;
}

decl_aliases!(Scalar);
pub use self::aliases::*;

macro_rules! impl_extends {
    ($($ty:ty),* $(,)?) => {$( impl ExtendRegister<$ty> for $ty {
        #[inline(always)]
        fn extend(value: Storage<$ty>) -> Storage<Self> {
            value
        }

        #[inline(always)]
        fn narrow(value: Storage<Self>) -> Storage<$ty> {
            value
        }
    })*};
}

impl_extends!(f32, i32, u32, f64, i64, u64, i8, i16, u8, u16);

impl_has_isa!(Scalar: f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool);

impl_newregister!(f32, i32, u32, f64, i64, u64);

impl_newregister!(u8, u16, i8, i16); // extra integers
