//! Native AVX2 registers.

use super::arch;

pub mod f32x4;
pub mod f32x8;
pub mod f64x2;
pub mod f64x4;
pub mod i32x4;
pub mod i32x8;
pub mod i64x2;
pub mod i64x4;
pub mod u32x4;
pub mod u32x8;
pub mod u64x2;
pub mod u64x4;

pub mod i16x16;
pub mod i16x8;
pub mod u16x16;
pub mod u16x8;

pub mod i8x16;
pub mod i8x32;
pub mod u8x16;
pub mod u8x32;

pub mod half16;
pub mod half8; // sub-native 8-bit ReducedRegister ladder (i8x4/x8) + u8<->u32 casts

// `PackedFloatRegister` for the native 16-bit registers: F16C hardware overrides for the binary16
// formats when `avx2-f16c` is on (which also enables `f16c` in dispatched codegen), generic
// branchless defaults otherwise and for bf16 always.
pub mod packed;

pub use i16x8::I16x8V3;
pub use i16x16::I16x16V3;
pub use u16x8::U16x8V3;
pub use u16x16::U16x16V3;

pub use i8x16::I8x16V3;
pub use i8x32::I8x32V3;
pub use u8x16::U8x16V3;
pub use u8x32::U8x32V3;

pub use f32x4::F32x4V3;
pub use f32x8::F32x8V3;
pub use f64x2::F64x2V3;
pub use f64x4::F64x4V3;
pub use i32x4::I32x4V3;
pub use i32x8::I32x8V3;
pub use i64x2::I64x2V3;
pub use i64x4::I64x4V3;
pub use u32x4::U32x4V3;
pub use u32x8::U32x8V3;
pub use u64x2::U64x2V3;
pub use u64x4::U64x4V3;

pub mod half;

pub use half::{F32x2V3, I32x2V3, U32x2V3};

impl_newregister!(
    F32x4V3, F32x8V3, F64x2V3, F64x4V3, I32x4V3, I32x8V3, I64x2V3, I64x4V3, U32x4V3, U32x8V3, U64x2V3, U64x4V3,
    I16x8V3, U16x8V3, I16x16V3, U16x16V3, I8x16V3, U8x16V3, I8x32V3, U8x32V3
);

impl_has_isa!(
    X86V3: F32x4V3, F32x8V3, F64x2V3, F64x4V3, I32x4V3, I32x8V3, I64x2V3, I64x4V3, U32x4V3, U32x8V3, U64x2V3,
    U64x4V3, I16x8V3, U16x8V3, I16x16V3, U16x16V3, I8x16V3, U8x16V3, I8x32V3, U8x32V3
);

use crate::{
    element::FindUSize,
    isa::InstructionSet,
    register::{ConcatRegister, IndexableRegister, Storage, array::ArrayRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86V3;

#[thermite_macros::inline_always]
impl HasIsa for X86V3 {
    type Native = Self;

    const ISA: InstructionSet = InstructionSet::X86V3;
}

#[thermite_macros::inline_always]
impl NativeIsa for X86V3 {
    type Registers = generic_array::typenum::U16;

    type Native32Width = generic_array::typenum::U8;
    type Native64Width = generic_array::typenum::U4;
    type Native16Width = generic_array::typenum::U16;
    type Native8Width = generic_array::typenum::U32;

    type NativeAlignment = crate::simd::Align32; // 256-bit vectors = 32 bytes

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

    unsafe fn zeroupper() -> bool {
        unsafe { arch::_mm256_zeroupper() };

        true
    }
}

impl NativeSimd for X86V3 {
    type f32xN = F32x8V3;
    type i32xN = I32x8V3;
    type u32xN = U32x8V3;

    type f64xN = F64x4V3;
    type i64xN = I64x4V3;
    type u64xN = U64x4V3;

    type i16xN = I16x16V3;
    type u16xN = U16x16V3;

    type i8xN = I8x32V3;
    type u8xN = U8x32V3;
}

impl Simd for X86V3 {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = F32x2V3;
    type i32x2 = I32x2V3;
    type u32x2 = U32x2V3;

    type f32x4 = F32x4V3;
    type i32x4 = I32x4V3;
    type u32x4 = U32x4V3;

    type f32x8 = F32x8V3;
    type i32x8 = I32x8V3;
    type u32x8 = U32x8V3;

    type f64x2 = F64x2V3;
    type i64x2 = I64x2V3;
    type u64x2 = U64x2V3;

    type f64x4 = F64x4V3;
    type i64x4 = I64x4V3;
    type u64x4 = U64x4V3;

    type f64x8 = ArrayRegister<F64x4V3, 2>;
    type i64x8 = ArrayRegister<I64x4V3, 2>;
    type u64x8 = ArrayRegister<U64x4V3, 2>;

    type f32x16 = ArrayRegister<F32x8V3, 2>;
    type i32x16 = ArrayRegister<I32x8V3, 2>;
    type u32x16 = ArrayRegister<U32x8V3, 2>;

    type f64x16 = ArrayRegister<F64x4V3, 4>;
    type i64x16 = ArrayRegister<I64x4V3, 4>;
    type u64x16 = ArrayRegister<U64x4V3, 4>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4V3;
    type u16x4 = half16::U16x4V3;

    type i16x8 = I16x8V3;
    type u16x8 = U16x8V3;

    type i16x16 = I16x16V3;
    type u16x16 = U16x16V3;

    type i8x16 = I8x16V3;
    type u8x16 = U8x16V3;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = half8::I8x4V3;
    type u8x4 = half8::U8x4V3;
    type i8x8 = half8::I8x8V3;
    type u8x8 = half8::U8x8V3;
}

impl Simd3 for X86V3 {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

// fp8 pack/unpack (generic branchless defaults) on the u8 ladder -> matching f32 widths
// (v3's f32x8 is native, f32x16 = ArrayRegister<F32x8V3, 2>).
impl_packed_fp8!(
    ArrayRegister<u8, 2> => F32x2V3,
    half8::U8x4V3 => F32x4V3,
    half8::U8x8V3 => F32x8V3,
    U8x16V3 => ArrayRegister<F32x8V3, 2>,
);

// Same-width, different-lane-count reinterprets of the 128-bit byte register, so it can
// be viewed as wider accumulator lanes (the SAD family). All `__m128i`, so identity.
impl_bit_casts_identity! {
    U8x16V3 as U16x8V3,
    U8x16V3 as U32x4V3,
    U8x16V3 as U64x2V3,
}

impl_sad_native_u64!(@ssse3 U8x16V3 => (U16x8V3, U32x4V3, U64x2V3) via _mm_sad_epu8);

// Sub-native byte ladder: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    half8::U8x8V3 => (half16::U16x4V3, U32x2V3, u64),
    half8::U8x4V3 => (ArrayRegister<u16, 2>, u32, u64),
}

// SAD on wider elements: `u16` pairs/quads -> `u32`/`u64`, `u32` pairs -> `u64`. Same
// same-width-reinterpret shape as the byte family (all `__m128i`, so identity casts).
impl_bit_casts_identity! {
    U16x8V3 as U32x4V3,
    U16x8V3 as U64x2V3,
    U32x4V3 as U64x2V3,
}

impl_sad_u16!(@swar U16x8V3 => (U32x4V3, U64x2V3));
impl_sad_u32!(@swar U32x4V3 => U64x2V3);

// Sub-native rungs: lane-wise.
impl_sad_u16!(@scalar half16::U16x4V3 => (U32x2V3, u64));
impl_sad_u32!(@scalar U32x2V3 => u64);

// Native 256-bit rungs (v3 only). All `__m256i`, so the reinterprets are identity.
impl_bit_casts_identity! {
    U16x16V3 as U32x8V3,
    U16x16V3 as U64x4V3,
    U32x8V3 as U64x4V3,
}

impl_sad_u16!(@swar U16x16V3 => (U32x8V3, U64x4V3));
impl_sad_u32!(@swar U32x8V3 => U64x4V3);

// The native 256-bit byte register (`u8xN` on AVX2). `_mm256_sad_epu8` is the full-width
// PSADBW, and `pmaddubsw`/`pmaddwd` have 256-bit forms too, so every grouping is native
// at twice the width of the 128-bit ladder.
impl_bit_casts_identity! {
    U8x32V3 as U16x16V3,
    U8x32V3 as U32x8V3,
    U8x32V3 as U64x4V3,
}

const _: () = {
    use crate::register::{Sad16Register, Sad32Register, Sad64Register, UnsignedIntegerRegister};

    #[thermite_macros::inline_always]
    impl Sad16Register<U16x16V3> for U8x32V3 {
        fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<U16x16V3> {
            unsafe { arch::_mm256_maddubs_epi16(Self::abs_diff(a, b), arch::_mm256_set1_epi8(1)) }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad32Register<U32x8V3> for U8x32V3 {
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x8V3> {
            unsafe {
                arch::_mm256_madd_epi16(
                    <Self as Sad16Register<U16x16V3>>::sad16(a, b),
                    arch::_mm256_set1_epi16(1),
                )
            }
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x4V3> for U8x32V3 {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x4V3> {
            unsafe { arch::_mm256_sad_epu8(a, b) }
        }
    }
};

// 16-bit gather/scatter on x86v3 has no hardware support; mark scalar-fallback impls.
macro_rules! impl_indexable16 {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

// native 8-lane indexed by same-width u16 and the 8-lane u32/u64 index types
impl_indexable16!(U16x8V3 => I16x8V3, U16x8V3);
impl_indexable16!(<X86V3 as Simd>::u32x8 => I16x8V3, U16x8V3);
impl_indexable16!(<X86V3 as Simd>::u64x8 => I16x8V3, U16x8V3);
// native 16-lane indexed by same-width u16 and the 16-lane u32/u64 index types
impl_indexable16!(U16x16V3 => I16x16V3, U16x16V3);
impl_indexable16!(<X86V3 as Simd>::u32x16 => I16x16V3, U16x16V3);
impl_indexable16!(<X86V3 as Simd>::u64x16 => I16x16V3, U16x16V3);
// 2-lane array indexed by the 2-lane u32/u64 index types
impl_indexable16!(<X86V3 as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable16!(<X86V3 as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
// native 32-lane 8-bit: same-width self-indexing only (no hardware 8-bit gather)
impl_indexable16!(U8x32V3 => I8x32V3, U8x32V3);
// fixed 16-lane 8-bit: same-width self-indexing plus the 16-lane usize/u32/u64 index trio
// (scalar-fallback markers, matching the sub-native i8x8/i8x4 ladder; usizex16 aliases u32x16/u64x16)
impl_indexable16!(U8x16V3 => I8x16V3, U8x16V3);
impl_indexable16!(<X86V3 as Simd>::u32x16 => I8x16V3, U8x16V3);
impl_indexable16!(<X86V3 as Simd>::u64x16 => I8x16V3, U8x16V3);

impl_concat_bool_register2!(f32, F32x2V3);
impl_concat_bool_register2!(u32, U32x2V3);
impl_concat_bool_register2!(i32, I32x2V3);

impl_concat_bool_register2!(f64, F64x2V3);
impl_concat_bool_register2!(u64, U64x2V3);
impl_concat_bool_register2!(i64, I64x2V3);

// --- extend-from-scalar: native (non-emulated) register widths ---
// The 64-bit x2 (F64x2V3/I64x2V3/U64x2V3), the 32-bit x2 (via half.rs), and the
// 128-bit 8/16-bit natives already carry explicit element-extend impls; the widths
// below are the ones that were only wired up pairwise.
impl_native_extend_from_scalar!(
    F32x4V3 => f32, F32x8V3 => f32,
    I32x4V3 => i32, I32x8V3 => i32,
    U32x4V3 => u32, U32x8V3 => u32,
    F64x4V3 => f64,
    I64x4V3 => i64,
    U64x4V3 => u64,
    I16x16V3 => i16,
);

const fn shuffle_to_m256i(bitmask: i32) -> arch::__m256i {
    let mut masks = [0i32; 8];

    let mut i = 0;

    while i < 8 {
        let k = i as i32;
        masks[i] = (bitmask >> (k * 3)) & 0b111;
        i += 1;
    }

    unsafe { generic_array::const_transmute(masks) }
}

impl_bit_casts! {
    F64x2V3 as I64x2V3 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_castpd_si128, // f64x2 -> u64x2
    I64x2V3 as F64x2V3 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_castsi128_pd, // u64x2 -> f64x2

    F64x4V3 as I64x4V3 => _mm256_castpd_si256, // f64x4 -> i64x4
    I64x4V3 as F64x4V3 => _mm256_castsi256_pd, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_castsi256_pd, // u64x4 -> f64x4
    F64x4V3 as U64x4V3 => _mm256_castpd_si256, // f64x4 -> u64x4

    F32x4V3 as I32x4V3 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_castps_si128, // f32x4 -> u32x4
    I32x4V3 as F32x4V3 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_castsi128_ps, // u32x4 -> f32x4

    F32x8V3 as I32x8V3 => _mm256_castps_si256, // f32x8 -> i32x8
    F32x8V3 as U32x8V3 => _mm256_castps_si256, // f32x8 -> u32x8
    I32x8V3 as F32x8V3 => _mm256_castsi256_ps, // i32x8 -> f32x8
    U32x8V3 as F32x8V3 => _mm256_castsi256_ps, // u32x8 -> f32x8

    // integer casts use the same underlying storage, so identity casts
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4

    // all the identity casts to self
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4

    // 16-bit (same storage)
    U16x8V3 as I16x8V3 => identity, U16x16V3 as I16x16V3 => identity,
    I16x8V3 as U16x8V3 => identity, I16x16V3 as U16x16V3 => identity,
    I16x8V3 as I16x8V3 => identity, I16x16V3 as I16x16V3 => identity,
    U16x8V3 as U16x8V3 => identity, U16x16V3 as U16x16V3 => identity,

    // 8-bit (same storage): native 256-bit + fixed 128-bit
    U8x32V3 as I8x32V3 => identity, I8x32V3 as U8x32V3 => identity,
    I8x32V3 as I8x32V3 => identity, U8x32V3 as U8x32V3 => identity,
    U8x16V3 as I8x16V3 => identity, I8x16V3 as U8x16V3 => identity,
    I8x16V3 as I8x16V3 => identity, U8x16V3 as U8x16V3 => identity,
}

impl_type_casts! {
    // self casts
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4

    // int -> float casts (float -> int live in `impl_float_to_int_casts!` below)
    I32x4V3 as F32x4V3 => _mm_cvtepi32_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_cvtepu32_psx_v2, // u32x4 -> f32x4
    I32x8V3 as F32x8V3 => _mm256_cvtepi32_ps, // i32x4 -> f32x4
    U32x8V3 as F32x8V3 => _mm256_cvtepu32_psx_v3, // i32x8 -> f32x8
    I64x2V3 as F64x2V3 => _mm_cvtepi64_pdx_v2 | _mm_cvtepi64_pdx_limited_v1, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_cvtepu64_pdx_v2 | _mm_cvtepu64_pdx_limited_v1, // u64x2 -> f64x2
    I64x4V3 as F64x4V3 => _mm256_cvtepi64_pdx_v3 | _mm256_cvtepi64_pdx_limited_v3, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_cvtepu64_pdx_v3 | _mm256_cvtepu64_pdx_limited_v3, // u64x4 -> f64x4

    // for integer casts we don't do anything, basically bit casting, same as Rust
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4

    // 32-bit int -> f64: `vcvtdq2pd` is native and signed-only, so the unsigned
    // form takes the magic-constant polyfill (exact, three instructions) rather
    // than routing through the full-range 64-bit conversion.
    I32x4V3 as F64x4V3 => _mm256_cvtepi32_pd, // i32x4 -> f64x4
    U32x4V3 as F64x4V3 => _mm256_cvtepu32_pdx_v3, // u32x4 -> f64x4

    // simple precision casts, others are implemented in-module
    F32x4V3 as F64x4V3 => _mm256_cvtps_pd, // f32x4 -> f64x4
    F64x4V3 as F32x4V3 => _mm256_cvtpd_ps, // f64x4 -> f32x4
    U32x4V3 as U64x4V3 => _mm256_cvtepu32_epi64, // u32x4 -> u64x4
    // U64x4V3 -> U32x4V3 lives in `half.rs` (needs both cast strengths).
    I32x4V3 as I64x4V3 => _mm256_cvtepi32_epi64, // i32x4 -> i64x4
    // I64x4V3 -> I32x4V3 lives in `half.rs` (needs both cast strengths).

    // 16-bit self + sibling (i16<->u16). i16<->i32 widen/narrow live in-module.
    I16x8V3 as I16x8V3 => identity, I16x16V3 as I16x16V3 => identity,
    U16x8V3 as U16x8V3 => identity, U16x16V3 as U16x16V3 => identity,
    I16x8V3 as U16x8V3 => identity, I16x16V3 as U16x16V3 => identity,
    U16x8V3 as I16x8V3 => identity, U16x16V3 as I16x16V3 => identity,

    // 8-bit self + sibling: native 256-bit + fixed 128-bit
    I8x32V3 as I8x32V3 => identity, U8x32V3 as U8x32V3 => identity,
    I8x32V3 as U8x32V3 => identity, U8x32V3 as I8x32V3 => identity,
    I8x16V3 as I8x16V3 => identity, U8x16V3 as U8x16V3 => identity,
    I8x16V3 as U8x16V3 => identity, U8x16V3 as I8x16V3 => identity,
}

impl_float_to_int_casts! {
    // NOTE: `cvtt` (truncate toward zero) everywhere - `cast` is "like `as`"
    // for in-range inputs; plain `cvtps_epi32` rounds in the current mode.
    // `sat` = the `as`-exact saturating variant (also `cast` under strict_ieee754).
    F32x4V3 as I32x4V3 => _mm_cvttps_epi32 sat _mm_cvtps_epi32_satx_v1, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_cvtps_epu32x_v2 sat _mm_cvtps_epu32_satx_v1, // f32x4 -> u32x4
    F32x8V3 as I32x8V3 => _mm256_cvttps_epi32 sat _mm256_cvtps_epi32_satx_v3, // f32x8 -> i32x8
    F32x8V3 as U32x8V3 => _mm256_cvtps_epu32x_v3 sat _mm256_cvtps_epu32_satx_v3, // f32x8 -> u32x8

    F64x2V3 as I64x2V3 => _mm_cvtpd_epi64x_v2 sat _mm_cvtpd_epi64_satx_v1 | _mm_cvtpd_epi64x_limited_v1, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_cvtpd_epu64x_v1 sat _mm_cvtpd_epu64_satx_v1 | _mm_cvtpd_epu64x_limited_v1, // f64x2 -> u64x2
    F64x4V3 as I64x4V3 => _mm256_cvtpd_epi64x_v3 sat _mm256_cvtpd_epi64_satx_v3 | _mm256_cvtpd_epi64x_limited_v3, // f64x4 -> i64x4
    F64x4V3 as U64x4V3 => _mm256_cvtpd_epu64x_v3 sat _mm256_cvtpd_epu64_satx_v3 | _mm256_cvtpd_epu64x_limited_v3, // f64x4 -> u64x4
}

// 64-bit int -> f32, the direction x86 has no instruction for at any level.
// Composed through f64, where both legs exist. The double rounding is harmless:
// f64 carries 53 mantissa bits against f32's 24, comfortably past the 2p+2
// threshold at which a round-to-f64-then-round-to-f32 is provably identical to
// rounding straight to f32.
//
// `cast_from` only - an int -> float conversion cannot leave the destination's
// range, only lose precision, so `saturating_cast_from` defaulting to it is
// already exact and a separate body would be dead weight.
impl_cast_from_via! {
    I64x2V3 as F32x2V3 => via F64x2V3,
    U64x2V3 as F32x2V3 => via F64x2V3,
    I64x4V3 as F32x4V3 => via F64x4V3,
    U64x4V3 as F32x4V3 => via F64x4V3,
    ArrayRegister<I64x4V3, 2> as F32x8V3 => via ArrayRegister<F64x4V3, 2>,
    ArrayRegister<U64x4V3, 2> as F32x8V3 => via ArrayRegister<F64x4V3, 2>,
}

// 32-bit int -> f64 at 8 lanes: two 128-bit halves, one convert each. Composing
// through i64 instead would drag in the full-range 64-bit magic-number
// conversion for what `vcvtdq2pd` does in a single instruction.
//
// The x16 rungs are not stated: the array cast ladder in `register/array.rs`
// derives `ArrayRegister<F64x4V3, 4>` from these, and stamping them would be a
// coherence conflict.
#[thermite_macros::inline_always]
impl crate::register::CastRegister<I32x8V3> for ArrayRegister<F64x4V3, 2> {
    fn cast_from(value: Storage<I32x8V3>) -> Storage<Self> {
        let (lo, hi) = I32x8V3::split(value);

        unsafe { ArrayRegister([arch::_mm256_cvtepi32_pd(lo), arch::_mm256_cvtepi32_pd(hi)]) }
    }
}

#[thermite_macros::inline_always]
impl crate::register::CastRegister<U32x8V3> for ArrayRegister<F64x4V3, 2> {
    fn cast_from(value: Storage<U32x8V3>) -> Storage<Self> {
        let (lo, hi) = U32x8V3::split(value);

        unsafe { ArrayRegister([arch::_mm256_cvtepu32_pdx_v3(lo), arch::_mm256_cvtepu32_pdx_v3(hi)]) }
    }
}

// Cross-width float -> int saturating casts, one row per Simd lane count
// (`[f32, f64, i32, u32, i64, u64, i16, u16, i8, u8]`), composed from the
// same-width saturating casts above and the integer-narrowing saturating matrix.
impl_float_cast_matrix! {
    [F32x2V3, F64x2V3, I32x2V3, U32x2V3, I64x2V3, U64x2V3,
        ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>],
    [F32x4V3, F64x4V3, I32x4V3, U32x4V3, I64x4V3, U64x4V3,
        half16::I16x4V3, half16::U16x4V3, half8::I8x4V3, half8::U8x4V3],
    [F32x8V3, ArrayRegister<F64x4V3, 2>, I32x8V3, U32x8V3, ArrayRegister<I64x4V3, 2>, ArrayRegister<U64x4V3, 2>,
        I16x8V3, U16x8V3, half8::I8x8V3, half8::U8x8V3],
}

// Sign-changing integer casts, one row per Simd lane count. Composed as a
// same-signedness width change followed by the free same-width reinterpret, so
// every pair is exactly the instruction sequence of its same-signedness twin.
// `cast_from` only - see `impl_cast_from_via!` for why saturating is left to
// default here.
impl_sign_cast_matrix! {
    [F32x2V3, F64x2V3, I32x2V3, U32x2V3, I64x2V3, U64x2V3,
        ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>],
    [F32x4V3, F64x4V3, I32x4V3, U32x4V3, I64x4V3, U64x4V3,
        half16::I16x4V3, half16::U16x4V3, half8::I8x4V3, half8::U8x4V3],
    [F32x8V3, ArrayRegister<F64x4V3, 2>, I32x8V3, U32x8V3, ArrayRegister<I64x4V3, 2>, ArrayRegister<U64x4V3, 2>,
        I16x8V3, U16x8V3, half8::I8x8V3, half8::U8x8V3],
}

// x16 is the row the matrix macro cannot take whole: both 32 <-> 64 slots are
// `ArrayRegister`s a factor of two apart, so the array cast ladder
// (`impl_casts!` in `register/array.rs`) already derives those pairs from the x8
// row above and stamping them here is a coherence conflict. Everything reaching
// the native 8/16-bit registers still needs stating.
impl_cast_from_via! {
    I8x16V3 as ArrayRegister<U32x8V3, 2> => via ArrayRegister<I32x8V3, 2>,
    I8x16V3 as ArrayRegister<U64x4V3, 4> => via ArrayRegister<I64x4V3, 4>,
    U8x16V3 as ArrayRegister<I32x8V3, 2> => via ArrayRegister<U32x8V3, 2>,
    U8x16V3 as ArrayRegister<I64x4V3, 4> => via ArrayRegister<U64x4V3, 4>,
    I16x16V3 as ArrayRegister<U32x8V3, 2> => via ArrayRegister<I32x8V3, 2>,
    I16x16V3 as ArrayRegister<U64x4V3, 4> => via ArrayRegister<I64x4V3, 4>,
    U16x16V3 as ArrayRegister<I32x8V3, 2> => via ArrayRegister<U32x8V3, 2>,
    U16x16V3 as ArrayRegister<I64x4V3, 4> => via ArrayRegister<U64x4V3, 4>,
    ArrayRegister<I32x8V3, 2> as U8x16V3 => via I8x16V3,
    ArrayRegister<I32x8V3, 2> as U16x16V3 => via I16x16V3,
    ArrayRegister<U32x8V3, 2> as I8x16V3 => via U8x16V3,
    ArrayRegister<U32x8V3, 2> as I16x16V3 => via U16x16V3,
    ArrayRegister<I64x4V3, 4> as U8x16V3 => via I8x16V3,
    ArrayRegister<I64x4V3, 4> as U16x16V3 => via I16x16V3,
    ArrayRegister<U64x4V3, 4> as I8x16V3 => via U8x16V3,
    ArrayRegister<U64x4V3, 4> as I16x16V3 => via U16x16V3,
}

// The 8 <-> 16 sign-changing pairs. Omitted for the 2-lane row, which already
// has them from the scalar `ArrayRegister` impls.
impl_sign_cast_matrix_8_16! {
    [half16::I16x4V3, half16::U16x4V3, half8::I8x4V3, half8::U8x4V3],
    [I16x8V3, U16x8V3, half8::I8x8V3, half8::U8x8V3],
    [I16x16V3, U16x16V3, I8x16V3, U8x16V3],
}

impl_cast_via! {
    // x16: pairs the ArrayRegister cast ladder cannot bridge (the ladder covers
    // the factor-of-two array<->array pairs from the impls stamped above)
    ArrayRegister<F32x8V3, 2> as I16x16V3 => via ArrayRegister<I32x8V3, 2>,
    ArrayRegister<F32x8V3, 2> as U16x16V3 => via ArrayRegister<U32x8V3, 2>,
    ArrayRegister<F32x8V3, 2> as I8x16V3 => via ArrayRegister<I32x8V3, 2>,
    ArrayRegister<F32x8V3, 2> as U8x16V3 => via ArrayRegister<U32x8V3, 2>,
    ArrayRegister<F64x4V3, 4> as I16x16V3 => via ArrayRegister<I64x4V3, 4>,
    ArrayRegister<F64x4V3, 4> as U16x16V3 => via ArrayRegister<U64x4V3, 4>,
    ArrayRegister<F64x4V3, 4> as I8x16V3 => via ArrayRegister<I64x4V3, 4>,
    ArrayRegister<F64x4V3, 4> as U8x16V3 => via ArrayRegister<U64x4V3, 4>,
}

impl_mask_casts! {
    // self casts
    I32x4V3 as I32x4V3 => identity, // i32x4 -> i32x4
    U32x4V3 as U32x4V3 => identity, // u32x4 -> u32x4
    I32x8V3 as I32x8V3 => identity, // i32x8 -> i32x8
    U32x8V3 as U32x8V3 => identity, // u32x8 -> u32x8
    I64x2V3 as I64x2V3 => identity, // i64x2 -> i64x2
    U64x2V3 as U64x2V3 => identity, // u64x2 -> u64x2
    I64x4V3 as I64x4V3 => identity, // i64x4 -> i64x4
    U64x4V3 as U64x4V3 => identity, // u64x4 -> u64x4
    F32x4V3 as F32x4V3 => identity, // f32x4 -> f32x4
    F32x8V3 as F32x8V3 => identity, // f32x8 -> f32x8
    F64x2V3 as F64x2V3 => identity, // f64x2 -> f64x2
    F64x4V3 as F64x4V3 => identity, // f64x4 -> f64x4

    // same-size integer casts
    I32x4V3 as U32x4V3 => identity, // i32x4 -> u32x4
    U32x4V3 as I32x4V3 => identity, // u32x4 -> i32x4
    I32x8V3 as U32x8V3 => identity, // i32x8 -> u32x8
    U32x8V3 as I32x8V3 => identity, // u32x8 -> i32x8
    I64x2V3 as U64x2V3 => identity, // i64x2 -> u64x2
    U64x2V3 as I64x2V3 => identity, // u64x2 -> i64x2
    I64x4V3 as U64x4V3 => identity, // i64x4 -> u64x4
    U64x4V3 as I64x4V3 => identity, // u64x4 -> i64x4

    // same-size float/integer casts
    I32x4V3 as F32x4V3 => _mm_castsi128_ps, // i32x4 -> f32x4
    U32x4V3 as F32x4V3 => _mm_castsi128_ps, // u32x4 -> f32x4
    I32x8V3 as F32x8V3 => _mm256_castsi256_ps, // i32x8 -> f32x8
    U32x8V3 as F32x8V3 => _mm256_castsi256_ps, // u32x8 -> f32x8
    I64x2V3 as F64x2V3 => _mm_castsi128_pd, // i64x2 -> f64x2
    U64x2V3 as F64x2V3 => _mm_castsi128_pd, // u64x2 -> f64x2
    I64x4V3 as F64x4V3 => _mm256_castsi256_pd, // i64x4 -> f64x4
    U64x4V3 as F64x4V3 => _mm256_castsi256_pd, // u64x4 -> f64x4
    F32x4V3 as I32x4V3 => _mm_castps_si128, // f32x4 -> i32x4
    F32x4V3 as U32x4V3 => _mm_castps_si128, // f32x4 -> u32x4
    F32x8V3 as I32x8V3 => _mm256_castps_si256, // f32x8 -> i32x8
    F32x8V3 as U32x8V3 => _mm256_castps_si256, // f32x8 -> u32x8
    F64x2V3 as I64x2V3 => _mm_castpd_si128, // f64x2 -> i64x2
    F64x2V3 as U64x2V3 => _mm_castpd_si128, // f64x2 -> u64x2
    F64x4V3 as I64x4V3 => _mm256_castpd_si256, // f64x4 -> i64x4
    F64x4V3 as U64x4V3 => _mm256_castpd_si256, // f64x4 -> u64x4

    // 16-bit self + sibling
    I16x8V3 as I16x8V3 => identity, I16x16V3 as I16x16V3 => identity,
    U16x8V3 as U16x8V3 => identity, U16x16V3 as U16x16V3 => identity,
    I16x8V3 as U16x8V3 => identity, I16x16V3 as U16x16V3 => identity,
    U16x8V3 as I16x8V3 => identity, U16x16V3 as I16x16V3 => identity,

    // 8-bit self + sibling: native 256-bit + fixed 128-bit
    I8x32V3 as I8x32V3 => identity, U8x32V3 as U8x32V3 => identity,
    I8x32V3 as U8x32V3 => identity, U8x32V3 as I8x32V3 => identity,
    I8x16V3 as I8x16V3 => identity, U8x16V3 as U8x16V3 => identity,
    I8x16V3 as U8x16V3 => identity, U8x16V3 as I8x16V3 => identity,
}

#[cfg(test)]
mod tests {
    use crate::register::{FloatRegister, IntegerRegister, SignedRegister, UnsignedIntegerRegister};

    use super::*;

    fn assert_is_float<R: FloatRegister>() {}
    fn assert_is_integer<R: IntegerRegister>() {}
    fn assert_is_unsigned<R: UnsignedIntegerRegister>() {}
    fn assert_is_signed<R: SignedRegister>() {}

    #[test]
    fn test_compiles() {
        assert_is_float::<F32x4V3>();
        assert_is_float::<F32x8V3>();
        assert_is_float::<F64x2V3>();
        assert_is_float::<F64x4V3>();

        assert_is_integer::<I32x4V3>();
        assert_is_integer::<I32x8V3>();
        assert_is_integer::<I64x2V3>();
        assert_is_integer::<I64x4V3>();
        assert_is_integer::<U32x4V3>();
        assert_is_integer::<U32x8V3>();
        assert_is_integer::<U64x2V3>();
        assert_is_integer::<U64x4V3>();

        assert_is_unsigned::<U32x4V3>();
        assert_is_unsigned::<U32x8V3>();
        assert_is_unsigned::<U64x2V3>();
        assert_is_unsigned::<U64x4V3>();

        assert_is_signed::<I32x4V3>();
        assert_is_signed::<I32x8V3>();
        assert_is_signed::<I64x2V3>();
        assert_is_signed::<I64x4V3>();

        assert_is_signed::<F32x4V3>();
        assert_is_signed::<F32x8V3>();
        assert_is_signed::<F64x2V3>();
        assert_is_signed::<F64x4V3>();
    }
}
