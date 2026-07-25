use super::arch;

pub mod f32x4;
pub mod i32x4;
pub mod u32x4;

pub mod f64x2;
pub mod i64x2;
pub mod u64x2;

pub mod i16x8;
pub mod u16x8;

pub mod i8x16;
pub mod u8x16;

pub mod half;
pub mod half16;
pub mod half8; // sub-native 8-bit ReducedRegister ladder (i8x4/x8) + u8<->u32 casts
pub mod packed; // PackedFloatRegister (16-bit float) generic-default impls for native u16 regs

pub use f32x4::F32x4V1;
pub use i32x4::I32x4V1;
pub use u32x4::U32x4V1;

pub use f64x2::F64x2V1;
pub use i64x2::I64x2V1;
pub use u64x2::U64x2V1;

pub use i16x8::I16x8V1;
pub use u16x8::U16x8V1;

pub use i8x16::I8x16V1;
pub use u8x16::U8x16V1;

impl_newregister!(
    F32x4V1, I32x4V1, U32x4V1, F64x2V1, I64x2V1, U64x2V1, I16x8V1, U16x8V1, I8x16V1, U8x16V1
);

use crate::{
    backend::scalar::Scalar,
    element::FindUSize,
    isa::InstructionSet,
    register::{
        IndexableRegister, Storage,
        array::ArrayRegister,
        reduced::{HalfRegister2, ReducedRegister},
    },
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86V1;

impl HasIsa for X86V1 {
    const ISA: InstructionSet = InstructionSet::X86V1;
}

#[thermite_macros::inline_always]
impl NativeIsa for X86V1 {
    type Registers = generic_array::typenum::U8;

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;
    type Native16Width = generic_array::typenum::U8;
    type Native8Width = generic_array::typenum::U16;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes

    const HAS_PREFETCH: bool = arch::HAS_PREFETCH;

    fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
        arch::prefetch::<LOCALITY, WRITE>(ptr);
    }

    unsafe fn disable_denormals() -> Result<bool, crate::simd::UnsupportedError> {
        unsafe { Ok(arch::disable_denormals()) }
    }

    #[allow(clippy::unit_arg)]
    unsafe fn enable_denormals() -> Result<(), crate::simd::UnsupportedError> {
        unsafe { Ok(arch::enable_denormals()) }
    }
}

impl NativeSimd for X86V1 {
    type f32xN = F32x4V1;
    type i32xN = I32x4V1;
    type u32xN = U32x4V1;

    type f64xN = F64x2V1;
    type i64xN = I64x2V1;
    type u64xN = U64x2V1;

    type i16xN = I16x8V1;
    type u16xN = U16x8V1;

    type i8xN = I8x16V1;
    type u8xN = U8x16V1;
}

// Scatter/Gather is not available in x86v1, so we must use fallback impls
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

impl_indexable!(<X86V1 as Simd>::u32x2 => F64x2V1, I64x2V1, U64x2V1);
impl_indexable!(<X86V1 as Simd>::u64x2 => F64x2V1, I64x2V1, U64x2V1);
impl_indexable!(<X86V1 as Simd>::u32x4 => F32x4V1, I32x4V1, U32x4V1, <X86V1 as Simd>::u64x4, <X86V1 as Simd>::i64x4, <X86V1 as Simd>::f64x4);
impl_indexable!(<X86V1 as Simd>::u64x4 => F32x4V1, I32x4V1, U32x4V1);

impl Simd for X86V1 {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = half::F32x2V1;
    type i32x2 = half::I32x2V1;
    type u32x2 = half::U32x2V1;

    type f32x4 = F32x4V1;
    type i32x4 = I32x4V1;
    type u32x4 = U32x4V1;

    type f64x2 = F64x2V1;
    type i64x2 = I64x2V1;
    type u64x2 = U64x2V1;

    type f32x8 = ArrayRegister<F32x4V1, 2>;
    type i32x8 = ArrayRegister<I32x4V1, 2>;
    type u32x8 = ArrayRegister<U32x4V1, 2>;

    type f64x4 = ArrayRegister<F64x2V1, 2>;
    type i64x4 = ArrayRegister<I64x2V1, 2>;
    type u64x4 = ArrayRegister<U64x2V1, 2>;

    type f64x8 = ArrayRegister<F64x2V1, 4>;
    type i64x8 = ArrayRegister<I64x2V1, 4>;
    type u64x8 = ArrayRegister<U64x2V1, 4>;

    type f32x16 = ArrayRegister<F32x4V1, 4>;
    type i32x16 = ArrayRegister<I32x4V1, 4>;
    type u32x16 = ArrayRegister<U32x4V1, 4>;

    type f64x16 = ArrayRegister<F64x2V1, 8>;
    type i64x16 = ArrayRegister<I64x2V1, 8>;
    type u64x16 = ArrayRegister<U64x2V1, 8>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4V1;
    type u16x4 = half16::U16x4V1;

    type i16x8 = I16x8V1;
    type u16x8 = U16x8V1;

    type i16x16 = ArrayRegister<I16x8V1, 2>;
    type u16x16 = ArrayRegister<U16x8V1, 2>;

    type i8x16 = I8x16V1;
    type u8x16 = U8x16V1;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = half8::I8x4V1;
    type u8x4 = half8::U8x4V1;
    type i8x8 = half8::I8x8V1;
    type u8x8 = half8::U8x8V1;
}

impl Simd3 for X86V1 {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

// fp8 pack/unpack (generic branchless defaults) on the u8 ladder -> matching f32 widths.
impl_packed_fp8!(
    ArrayRegister<u8, 2> => half::F32x2V1,
    half8::U8x4V1 => F32x4V1,
    half8::U8x8V1 => ArrayRegister<F32x4V1, 2>,
    U8x16V1 => ArrayRegister<F32x4V1, 4>,
);

// Same-width, different-lane-count reinterprets of the 128-bit byte register (all
// `__m128i`, so identity), and the SAD family built on them.
impl_bit_casts_identity! {
    U8x16V1 as U16x8V1,
    U8x16V1 as U32x4V1,
    U8x16V1 as U64x2V1,
}

impl_sad_native_u64!(U8x16V1 => (U16x8V1, U32x4V1, U64x2V1) via _mm_sad_epu8);

// Sub-native byte ladder: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    half8::U8x8V1 => (half16::U16x4V1, half::U32x2V1, u64),
    half8::U8x4V1 => (ArrayRegister<u16, 2>, u32, u64),
}

// SAD on wider elements: `u16` pairs/quads -> `u32`/`u64`, `u32` pairs -> `u64`. Same
// same-width-reinterpret shape as the byte family (all `__m128i`, so identity casts).
impl_bit_casts_identity! {
    U16x8V1 as U32x4V1,
    U16x8V1 as U64x2V1,
    U32x4V1 as U64x2V1,
}

impl_sad_u16!(@swar U16x8V1 => (U32x4V1, U64x2V1));
impl_sad_u32!(@swar U32x4V1 => U64x2V1);

// Sub-native rungs: lane-wise.
impl_sad_u16!(@scalar half16::U16x4V1 => (half::U32x2V1, u64));
impl_sad_u32!(@scalar half::U32x2V1 => u64);

// 16-bit gather/scatter falls back to scalar (no hardware support on x86v1).
impl_indexable!(U16x8V1 => I16x8V1, U16x8V1);
impl_indexable!(<X86V1 as Simd>::u32x8 => I16x8V1, U16x8V1);
impl_indexable!(<X86V1 as Simd>::u64x8 => I16x8V1, U16x8V1);
impl_indexable!(<X86V1 as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<X86V1 as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<X86V1 as Simd>::u64x16 => ArrayRegister<I16x8V1, 2>, ArrayRegister<U16x8V1, 2>);

// Native 8-bit: same-width self-indexing plus the 16-lane usize/u32/u64 index trio (scalar-
// fallback markers, matching the sub-native i8x8/i8x4 ladder). usizex16 aliases u32x16/u64x16.
impl_indexable!(U8x16V1 => I8x16V1, U8x16V1);
impl_indexable!(<X86V1 as Simd>::u32x16 => I8x16V1, U8x16V1);
impl_indexable!(<X86V1 as Simd>::u64x16 => I8x16V1, U8x16V1);

impl_concat_bool_register2!(f32, half::F32x2V1);
impl_concat_bool_register2!(u32, half::U32x2V1);
impl_concat_bool_register2!(i32, half::I32x2V1);

impl_concat_bool_register2!(f64, F64x2V1);
impl_concat_bool_register2!(u64, U64x2V1);
impl_concat_bool_register2!(i64, I64x2V1);

impl_bit_casts! {
    F64x2V1 as I64x2V1 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V1 as U64x2V1 => _mm_castpd_si128, // f64x2 -> u64x2
    I64x2V1 as F64x2V1 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V1 as F64x2V1 => _mm_castsi128_pd, // u64x2 -> f64x2

    F32x4V1 as I32x4V1 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V1 as U32x4V1 => _mm_castps_si128, // f32x4 -> u32x4
    I32x4V1 as F32x4V1 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V1 as F32x4V1 => _mm_castsi128_ps, // u32x4 -> f32x4

    // integer casts use the same underlying storage, so identity casts
    U32x4V1 as I32x4V1 => identity, // u32x4 -> i32x4
    I32x4V1 as U32x4V1 => identity, // i32x4 -> u32x4
    U64x2V1 as I64x2V1 => identity, // u64x2 -> i64x2
    I64x2V1 as U64x2V1 => identity, // i64x2 -> u64x2

    // all the identity casts to self
    U32x4V1 as U32x4V1 => identity, // u32x4 -> u32x4
    I32x4V1 as I32x4V1 => identity, // i32x4 -> i32x4
    I64x2V1 as I64x2V1 => identity, // i64x2 -> i64x2
    F32x4V1 as F32x4V1 => identity, // f32x4 -> f32x4
    F64x2V1 as F64x2V1 => identity, // f64x2 -> f64x2
    U64x2V1 as U64x2V1 => identity, // u64x2 -> u64x2

    // 16-bit (same storage)
    U16x8V1 as I16x8V1 => identity, I16x8V1 as U16x8V1 => identity,
    I16x8V1 as I16x8V1 => identity, U16x8V1 as U16x8V1 => identity,

    // 8-bit (same storage)
    U8x16V1 as I8x16V1 => identity, I8x16V1 as U8x16V1 => identity,
    I8x16V1 as I8x16V1 => identity, U8x16V1 as U8x16V1 => identity,
}

impl_type_casts! {
    // self casts
    F32x4V1 as F32x4V1 => identity, // f32x4 -> f32x4
    F64x2V1 as F64x2V1 => identity, // f64x2 -> f64x2
    I32x4V1 as I32x4V1 => identity, // i32x4 -> i32x4
    I64x2V1 as I64x2V1 => identity, // i64x2 -> i64x2
    U32x4V1 as U32x4V1 => identity, // u32x4 -> u32x4
    U64x2V1 as U64x2V1 => identity, // u64x2 -> u64x2

    // f32x4 casts (truncate toward zero - `cast` is "like `as`")
    F32x4V1 as I32x4V1 => _mm_cvttps_epi32, // f32x4 -> i32x4
    F32x4V1 as U32x4V1 => _mm_cvtps_epu32x_v1, // f32x4 -> u32x4
    I32x4V1 as F32x4V1 => _mm_cvtepi32_ps, // i32x4 -> f32x4
    U32x4V1 as F32x4V1 => _mm_cvtepu32_psx_v1, // u32x4 -> f32x4

    // f64x2 casts
    F64x2V1 as I64x2V1 => _mm_cvtpd_epi64x_v1 | _mm_cvtpd_epi64x_limited_v1, // f64x2 -> i64x2
    F64x2V1 as U64x2V1 => _mm_cvtpd_epu64x_limited_v1, // f64x2 -> u64x2
    I64x2V1 as F64x2V1 => _mm_cvtepi64_pdx_v1 | _mm_cvtepi64_pdx_limited_v1, // i64x2 -> f64x2
    U64x2V1 as F64x2V1 => _mm_cvtepu64_pdx_v1 | _mm_cvtepu64_pdx_limited_v1, // u64x2 -> f64x2

    // for integer casts we don't do anything, basically bit casting, same as Rust
    I32x4V1 as U32x4V1 => identity, // i32x4 -> u32x4
    U32x4V1 as I32x4V1 => identity, // u32x4 -> i32x4
    I64x2V1 as U64x2V1 => identity, // i64x2 -> u64x2
    U64x2V1 as I64x2V1 => identity, // u64x2 -> i64x2

    // 16-bit self + sibling (i16<->u16). i16<->i32 widen/narrow live in-module.
    I16x8V1 as I16x8V1 => identity, U16x8V1 as U16x8V1 => identity,
    I16x8V1 as U16x8V1 => identity, U16x8V1 as I16x8V1 => identity,

    // 8-bit self + sibling (i8<->u8)
    I8x16V1 as I8x16V1 => identity, U8x16V1 as U8x16V1 => identity,
    I8x16V1 as U8x16V1 => identity, U8x16V1 as I8x16V1 => identity,
}

impl_mask_casts! {
    // self casts
    I32x4V1 as I32x4V1 => identity, // i32x4 -> i32x4
    U32x4V1 as U32x4V1 => identity, // u32x4 -> u32x4
    I64x2V1 as I64x2V1 => identity, // i64x2 -> i64x2
    U64x2V1 as U64x2V1 => identity, // u64x2 -> u64x2
    F32x4V1 as F32x4V1 => identity, // f32x4 -> f32x4
    F64x2V1 as F64x2V1 => identity, // f64x2 -> f64x2

    // same-size integer casts
    I32x4V1 as U32x4V1 => identity, // i32x4 -> u32x4
    U32x4V1 as I32x4V1 => identity, // u32x4 -> i32x4
    I64x2V1 as U64x2V1 => identity, // i64x2 -> u64x2
    U64x2V1 as I64x2V1 => identity, // u64x2 -> i64x2

    // same-size float/integer casts
    I32x4V1 as F32x4V1 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V1 as F32x4V1 => _mm_castsi128_ps, // u32x4 -> f32x4
    I64x2V1 as F64x2V1 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V1 as F64x2V1 => _mm_castsi128_pd, // u64x2 -> f64x2
    F32x4V1 as I32x4V1 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V1 as U32x4V1 => _mm_castps_si128, // f32x4 -> u32x4
    F64x2V1 as I64x2V1 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V1 as U64x2V1 => _mm_castpd_si128, // f64x2 -> u64x2

    // 16-bit self + sibling
    I16x8V1 as I16x8V1 => identity, U16x8V1 as U16x8V1 => identity,
    I16x8V1 as U16x8V1 => identity, U16x8V1 as I16x8V1 => identity,

    // 8-bit self + sibling
    I8x16V1 as I8x16V1 => identity, U8x16V1 as U8x16V1 => identity,
    I8x16V1 as U8x16V1 => identity, U8x16V1 as I8x16V1 => identity,
}

macro_rules! impl_extend_same {
    ($($ty:ty),* $(,)?) => {$( impl crate::register::ExtendRegister<$ty> for $ty {
        #[inline(always)]
        fn extend(value: Storage<$ty>) -> Storage<Self> {
            value
        }

        #[inline(always)]
        fn narrow(value: Storage<Self>) -> Storage<$ty> {
            value
        }
    } )*};
}

impl_extend_same!(
    F32x4V1, I32x4V1, U32x4V1, F64x2V1, I64x2V1, U64x2V1, I16x8V1, U16x8V1, I8x16V1, U8x16V1
);
