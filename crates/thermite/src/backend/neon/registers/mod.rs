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
pub mod half8;
pub mod packed;

pub use f32x4::F32x4Neon;
pub use i32x4::I32x4Neon;
pub use u32x4::U32x4Neon;

pub use f64x2::F64x2Neon;
pub use i64x2::I64x2Neon;
pub use u64x2::U64x2Neon;

pub use i16x8::I16x8Neon;
pub use u16x8::U16x8Neon;

pub use i8x16::I8x16Neon;
pub use u8x16::U8x16Neon;

pub use half::{F32x2Neon, I32x2Neon, U32x2Neon};

use crate::{isa::InstructionSet, register::Storage, simd::HasIsa};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Neon;

impl_newregister!(
    F32x4Neon, I32x4Neon, U32x4Neon, F64x2Neon, I64x2Neon, U64x2Neon, I16x8Neon, U16x8Neon, I8x16Neon, U8x16Neon
);

impl HasIsa for Neon {
    type Native = Self;

    const ISA: InstructionSet = arch::ISA;
}

// Cross-type identity bit/mask casts (`vreinterpretq`, zero-cost).
macro_rules! impl_reinterpret_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }

            #[thermite_macros::inline_always]
            impl $crate::register::CastMaskRegister<$from> for $to {
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

impl_reinterpret_casts! {
    // 32-bit
    F32x4Neon as U32x4Neon => vreinterpretq_u32_f32,
    F32x4Neon as I32x4Neon => vreinterpretq_s32_f32,
    U32x4Neon as F32x4Neon => vreinterpretq_f32_u32,
    U32x4Neon as I32x4Neon => vreinterpretq_s32_u32,
    I32x4Neon as F32x4Neon => vreinterpretq_f32_s32,
    I32x4Neon as U32x4Neon => vreinterpretq_u32_s32,

    // 64-bit
    F64x2Neon as U64x2Neon => vreinterpretq_u64_f64,
    F64x2Neon as I64x2Neon => vreinterpretq_s64_f64,
    U64x2Neon as F64x2Neon => vreinterpretq_f64_u64,
    U64x2Neon as I64x2Neon => vreinterpretq_s64_u64,
    I64x2Neon as F64x2Neon => vreinterpretq_f64_s64,
    I64x2Neon as U64x2Neon => vreinterpretq_u64_s64,

    // 16-bit sibling
    I16x8Neon as U16x8Neon => vreinterpretq_u16_s16,
    U16x8Neon as I16x8Neon => vreinterpretq_s16_u16,

    // 8-bit sibling
    I8x16Neon as U8x16Neon => vreinterpretq_u8_s8,
    U8x16Neon as I8x16Neon => vreinterpretq_s8_u8,

    // identity casts (BitCastRegister<Self>; CastMaskRegister<Self> comes from neon_mask_core!)
}

macro_rules! impl_self_bitcasts {
    ($($ty:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl $crate::register::BitCastRegister<$ty> for $ty {
            fn from_bits(value: Storage<Self>) -> Storage<Self> {
                value
            }
        }
    )*};
}

impl_self_bitcasts!(
    F32x4Neon, I32x4Neon, U32x4Neon, F64x2Neon, I64x2Neon, U64x2Neon, I16x8Neon, U16x8Neon, I8x16Neon, U8x16Neon
);

// Numeric casts (`CastRegister`, Rust `as` semantics). NEON's `vcvt` float->int
// conversions truncate toward zero *and* saturate on overflow, which is
// exactly the `as` contract - no fixup dance needed (unlike x86). Same-width
// int<->int casts are bit reinterprets.
impl_type_casts! {
    // identity
    F32x4Neon as F32x4Neon => identity,
    F64x2Neon as F64x2Neon => identity,
    I32x4Neon as I32x4Neon => identity,
    U32x4Neon as U32x4Neon => identity,
    I64x2Neon as I64x2Neon => identity,
    U64x2Neon as U64x2Neon => identity,
    I16x8Neon as I16x8Neon => identity,
    U16x8Neon as U16x8Neon => identity,
    I8x16Neon as I8x16Neon => identity,
    U8x16Neon as U8x16Neon => identity,

    // same-width integer casts (reinterpret)
    I32x4Neon as U32x4Neon => vreinterpretq_u32_s32,
    U32x4Neon as I32x4Neon => vreinterpretq_s32_u32,
    I64x2Neon as U64x2Neon => vreinterpretq_u64_s64,
    U64x2Neon as I64x2Neon => vreinterpretq_s64_u64,
    I16x8Neon as U16x8Neon => vreinterpretq_u16_s16,
    U16x8Neon as I16x8Neon => vreinterpretq_s16_u16,
    I8x16Neon as U8x16Neon => vreinterpretq_u8_s8,
    U8x16Neon as I8x16Neon => vreinterpretq_s8_u8,

    // float <-> int, 32-bit
    F32x4Neon as I32x4Neon => vcvtq_s32_f32,
    F32x4Neon as U32x4Neon => vcvtq_u32_f32,
    I32x4Neon as F32x4Neon => vcvtq_f32_s32,
    U32x4Neon as F32x4Neon => vcvtq_f32_u32,

    // float <-> int, 64-bit
    F64x2Neon as I64x2Neon => vcvtq_s64_f64,
    F64x2Neon as U64x2Neon => vcvtq_u64_f64,
    I64x2Neon as F64x2Neon => vcvtq_f64_s64,
    U64x2Neon as F64x2Neon => vcvtq_f64_u64,
}

// `u64x4 -> f32x4`, composed through f64 where both legs already exist.
impl_cast_via! {
    ArrayRegister<U64x2Neon, 2> as F32x4Neon => via ArrayRegister<F64x2Neon, 2>,
}

// Cross-width float -> int saturating casts, one row per Simd lane count
// (`[f32, f64, i32, u32, i64, u64, i16, u16, i8, u8]`), composed from the
// same-width saturating casts above and the integer-narrowing saturating matrix.
impl_float_cast_matrix! {
    [F32x2Neon, F64x2Neon, I32x2Neon, U32x2Neon, I64x2Neon, U64x2Neon,
        ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>],
    [F32x4Neon, ArrayRegister<F64x2Neon, 2>, I32x4Neon, U32x4Neon, ArrayRegister<I64x2Neon, 2>,
        ArrayRegister<U64x2Neon, 2>, half16::I16x4Neon, half16::U16x4Neon, half8::I8x4Neon, half8::U8x4Neon],
}

impl_cast_via! {
    // x8 (wide rows are partial: the ArrayRegister cast ladder bridges the
    // factor-of-two array<->array pairs from the x4 impls stamped above)
    ArrayRegister<F32x4Neon, 2> as I16x8Neon => via ArrayRegister<I32x4Neon, 2>,
    ArrayRegister<F32x4Neon, 2> as half8::I8x8Neon => via ArrayRegister<I32x4Neon, 2>,
    ArrayRegister<F32x4Neon, 2> as U16x8Neon => via ArrayRegister<U32x4Neon, 2>,
    ArrayRegister<F32x4Neon, 2> as half8::U8x8Neon => via ArrayRegister<U32x4Neon, 2>,
    ArrayRegister<F64x2Neon, 4> as I16x8Neon => via ArrayRegister<I64x2Neon, 4>,
    ArrayRegister<F64x2Neon, 4> as half8::I8x8Neon => via ArrayRegister<I64x2Neon, 4>,
    ArrayRegister<F64x2Neon, 4> as U16x8Neon => via ArrayRegister<U64x2Neon, 4>,
    ArrayRegister<F64x2Neon, 4> as half8::U8x8Neon => via ArrayRegister<U64x2Neon, 4>,
    // x16
    ArrayRegister<F32x4Neon, 4> as I8x16Neon => via ArrayRegister<I32x4Neon, 4>,
    ArrayRegister<F32x4Neon, 4> as U8x16Neon => via ArrayRegister<U32x4Neon, 4>,
    ArrayRegister<F64x2Neon, 8> as ArrayRegister<I16x8Neon, 2> => via ArrayRegister<I64x2Neon, 8>,
    ArrayRegister<F64x2Neon, 8> as ArrayRegister<U16x8Neon, 2> => via ArrayRegister<U64x2Neon, 8>,
    ArrayRegister<F64x2Neon, 8> as I8x16Neon => via ArrayRegister<I64x2Neon, 8>,
    ArrayRegister<F64x2Neon, 8> as U8x16Neon => via ArrayRegister<U64x2Neon, 8>,
}

use crate::{
    element::FindUSize,
    register::{IndexableRegister, array::ArrayRegister},
    simd::{NativeIsa, NativeSimd, Simd, Simd3, Simd3A},
};

#[thermite_macros::inline_always]
impl NativeIsa for Neon {
    type Registers = generic_array::typenum::U32; // 32 128-bit V registers on aarch64

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;
    type Native16Width = generic_array::typenum::U8;
    type Native8Width = generic_array::typenum::U16;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes

    const HAS_PREFETCH: bool = arch::HAS_PREFETCH; // `prfm` is baseline aarch64

    fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
        arch::prefetch::<LOCALITY, WRITE>(ptr);
    }
}

#[thermite_macros::inline_always]
impl NativeSimd for Neon {
    type f32xN = F32x4Neon;
    type i32xN = I32x4Neon;
    type u32xN = U32x4Neon;

    type f64xN = F64x2Neon;
    type i64xN = I64x2Neon;
    type u64xN = U64x2Neon;

    type i16xN = I16x8Neon;
    type u16xN = U16x8Neon;

    type i8xN = I8x16Neon;
    type u8xN = U8x16Neon;
}

// NEON has no gather/scatter; scalar-fallback marker impls, exactly like wasm.
// Only the concrete index types: usizex* resolves via type alias to u64x* on
// aarch64, so explicit usizex* impls would duplicate and conflict.
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

// x2: 64-bit native types need u32x2 and u64x2 index support
impl_indexable!(<Neon as Simd>::u32x2 => F64x2Neon, I64x2Neon, U64x2Neon);
impl_indexable!(<Neon as Simd>::u64x2 => F64x2Neon, I64x2Neon, U64x2Neon);

// x4: 32-bit and wider types need u32x4 and u64x4 index support
impl_indexable!(<Neon as Simd>::u32x4 => F32x4Neon, I32x4Neon, U32x4Neon,
    <Neon as Simd>::f64x4, <Neon as Simd>::i64x4, <Neon as Simd>::u64x4);
impl_indexable!(<Neon as Simd>::u64x4 => F32x4Neon, I32x4Neon, U32x4Neon);

// 16-bit: same-width self-indexing plus wider index types (scalar-fallback markers).
impl_indexable!(U16x8Neon => I16x8Neon, U16x8Neon);
impl_indexable!(<Neon as Simd>::u32x8 => I16x8Neon, U16x8Neon);
impl_indexable!(<Neon as Simd>::u64x8 => I16x8Neon, U16x8Neon);
impl_indexable!(<Neon as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<Neon as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<Neon as Simd>::u64x16 => ArrayRegister<I16x8Neon, 2>, ArrayRegister<U16x8Neon, 2>);

// 8-bit: same-width self-indexing plus the 16-lane index trio.
impl_indexable!(U8x16Neon => I8x16Neon, U8x16Neon);
impl_indexable!(<Neon as Simd>::u32x16 => I8x16Neon, U8x16Neon);
impl_indexable!(<Neon as Simd>::u64x16 => I8x16Neon, U8x16Neon);

impl Simd for Neon {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = F32x2Neon;
    type i32x2 = I32x2Neon;
    type u32x2 = U32x2Neon;

    type f32x4 = F32x4Neon;
    type i32x4 = I32x4Neon;
    type u32x4 = U32x4Neon;

    type f64x2 = F64x2Neon;
    type i64x2 = I64x2Neon;
    type u64x2 = U64x2Neon;

    type f32x8 = ArrayRegister<F32x4Neon, 2>;
    type i32x8 = ArrayRegister<I32x4Neon, 2>;
    type u32x8 = ArrayRegister<U32x4Neon, 2>;

    type f64x4 = ArrayRegister<F64x2Neon, 2>;
    type i64x4 = ArrayRegister<I64x2Neon, 2>;
    type u64x4 = ArrayRegister<U64x2Neon, 2>;

    type f32x16 = ArrayRegister<F32x4Neon, 4>;
    type i32x16 = ArrayRegister<I32x4Neon, 4>;
    type u32x16 = ArrayRegister<U32x4Neon, 4>;

    type f64x8 = ArrayRegister<F64x2Neon, 4>;
    type i64x8 = ArrayRegister<I64x2Neon, 4>;
    type u64x8 = ArrayRegister<U64x2Neon, 4>;

    type f64x16 = ArrayRegister<F64x2Neon, 8>;
    type i64x16 = ArrayRegister<I64x2Neon, 8>;
    type u64x16 = ArrayRegister<U64x2Neon, 8>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4Neon;
    type u16x4 = half16::U16x4Neon;

    type i16x8 = I16x8Neon;
    type u16x8 = U16x8Neon;

    type i16x16 = ArrayRegister<I16x8Neon, 2>;
    type u16x16 = ArrayRegister<U16x8Neon, 2>;

    type i8x16 = I8x16Neon;
    type u8x16 = U8x16Neon;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = half8::I8x4Neon;
    type u8x4 = half8::U8x4Neon;
    type i8x8 = half8::I8x8Neon;
    type u8x8 = half8::U8x8Neon;
}

// fp8 pack/unpack (generic branchless defaults) on the u8 ladder -> matching f32 widths.
impl_packed_fp8! {
    ArrayRegister<u8, 2> => F32x2Neon,
    half8::U8x4Neon => F32x4Neon,
    half8::U8x8Neon => ArrayRegister<F32x4Neon, 2>,
    U8x16Neon => ArrayRegister<F32x4Neon, 4>,
}

// Same-width, different-lane-count reinterprets of the byte register, so it can be viewed
// as wider accumulator lanes (the SAD family). Bit-casts only -- unlike the sibling
// reinterprets above these relate different lane counts, so there is no meaningful
// `CastMaskRegister` counterpart.
impl_bit_casts! {
    U8x16Neon as U16x8Neon => vreinterpretq_u16_u8,
    U8x16Neon as U32x4Neon => vreinterpretq_u32_u8,
    U8x16Neon as U64x2Neon => vreinterpretq_u64_u8,
    // ... and one/two element sizes up. NEON's own SAD uses `vpaddlq` (whose types
    // already line up), so these exist purely to satisfy the same-width reinterpret
    // contract the `Simd` slots advertise to downstream generic code.
    U16x8Neon as U32x4Neon => vreinterpretq_u32_u16,
    U16x8Neon as U64x2Neon => vreinterpretq_u64_u16,
    U32x4Neon as U64x2Neon => vreinterpretq_u64_u32,
}

// Sub-native byte ladder: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    half8::U8x8Neon => (half16::U16x4Neon, U32x2Neon, u64),
    half8::U8x4Neon => (ArrayRegister<u16, 2>, u32, u64),
}

// Wider-element SAD. `vabdq_u16`/`vabdq_u32` then widening pairwise adds - the register
// types line up exactly, so unlike the other backends NEON needs no reinterpret at all.
const _: () = {
    use crate::register::{Sad32Register, Sad64Register, UnsignedIntegerRegister};

    #[thermite_macros::inline_always]
    impl Sad32Register<U32x4Neon> for U16x8Neon {
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x4Neon> {
            unsafe { arch::vpaddlq_u16(Self::abs_diff(a, b)) }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x2Neon> for U16x8Neon {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x2Neon> {
            unsafe { arch::vpaddlq_u32(arch::vpaddlq_u16(Self::abs_diff(a, b))) }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x2Neon> for U32x4Neon {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x2Neon> {
            unsafe { arch::vpaddlq_u32(Self::abs_diff(a, b)) }
        }
    }
};

// Sub-native rungs: lane-wise.
impl_sad_u16!(@scalar half16::U16x4Neon => (U32x2Neon, u64));
impl_sad_u32!(@scalar U32x2Neon => u64);

// NEON does every SAD grouping natively: `vabdq_u8` for the absolute difference (already
// the `abs_diff` override), then a chain of widening pairwise adds. `vpaddlq_u8` sums
// adjacent `u8` lanes into `u16`, `vpaddlq_u16` into `u32`, `vpaddlq_u32` into `u64` -
// exactly the 2/4/8-byte groupings, and the register types line up with no reinterpret.
// Same cascade the popcount polyfill uses. 2/3/4 instructions vs the SWAR default's ~7/11/15.
const _: () = {
    use crate::register::{Sad16Register, Sad32Register, Sad64Register, UnsignedIntegerRegister};

    #[thermite_macros::inline_always]
    impl Sad16Register<U16x8Neon> for U8x16Neon {
        fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<U16x8Neon> {
            unsafe { arch::vpaddlq_u8(Self::abs_diff(a, b)) }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad32Register<U32x4Neon> for U8x16Neon {
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x4Neon> {
            unsafe { arch::vpaddlq_u16(arch::vpaddlq_u8(Self::abs_diff(a, b))) }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x2Neon> for U8x16Neon {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x2Neon> {
            unsafe { arch::vpaddlq_u32(arch::vpaddlq_u16(arch::vpaddlq_u8(Self::abs_diff(a, b)))) }
        }
    }
};

impl Simd3 for Neon {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

impl_concat_bool_register2!(f32, F32x2Neon);
impl_concat_bool_register2!(u32, U32x2Neon);
impl_concat_bool_register2!(i32, I32x2Neon);

impl_concat_bool_register2!(f64, F64x2Neon);
impl_concat_bool_register2!(u64, U64x2Neon);
impl_concat_bool_register2!(i64, I64x2Neon);

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
    F32x4Neon, I32x4Neon, U32x4Neon, F64x2Neon, I64x2Neon, U64x2Neon, I16x8Neon, U16x8Neon, I8x16Neon, U8x16Neon
);

// `WidenIndexRegister` for the table-path registers (`compress: table` in each
// register's stamping invocation).
impl_widen_indices_neon! {
    F32x4Neon => (x4, f32), I32x4Neon => (x4, s32), U32x4Neon => (x4, u32),
    F64x2Neon => (x2, f64), I64x2Neon => (x2, s64), U64x2Neon => (x2, u64),
    I16x8Neon => (x8, s16), U16x8Neon => (x8, u16),
}
