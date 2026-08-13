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

pub use f32x4::F32x4Wasm;
pub use i32x4::I32x4Wasm;
pub use u32x4::U32x4Wasm;

pub use f64x2::F64x2Wasm;
pub use i64x2::I64x2Wasm;
pub use u64x2::U64x2Wasm;

pub use i16x8::I16x8Wasm;
pub use u16x8::U16x8Wasm;

pub use i8x16::I8x16Wasm;
pub use u8x16::U8x16Wasm;

pub use half::{F32x2Wasm, I32x2Wasm, U32x2Wasm};

use crate::{
    element::FindUSize,
    isa::InstructionSet,
    register::{IndexableRegister, Storage, array::ArrayRegister},
    simd::{HasIsa, NativeIsa, NativeSimd, Simd, Simd3, Simd3A},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wasm;

impl_newregister!(
    F32x4Wasm, I32x4Wasm, U32x4Wasm, F64x2Wasm, I64x2Wasm, U64x2Wasm, I16x8Wasm, U16x8Wasm, I8x16Wasm, U8x16Wasm
);

// The other seven natives carry hand-written element-extend impls in their own
// register files; these three did not, and the x2 halves need them to reach the
// scalar rung. `single` is `f32x4(v, 0, 0, 0)` and `extract` is a native
// `*_extract_lane`, so the generic stamp is the same codegen either way.
impl_native_extend_from_scalar!(F32x4Wasm => f32, I32x4Wasm => i32, U32x4Wasm => u32);

impl HasIsa for Wasm {
    type Native = Self;

    const ISA: InstructionSet = arch::ISA;
}

impl NativeIsa for Wasm {
    type Registers = generic_array::typenum::U1; // WASM has no practical register count limit

    type Native32Width = generic_array::typenum::U4;
    type Native64Width = generic_array::typenum::U2;
    type Native16Width = generic_array::typenum::U8;
    type Native8Width = generic_array::typenum::U16;

    type NativeAlignment = crate::simd::Align16; // 128-bit vectors = 16 bytes

    // No `prefetch` override: WebAssembly has no software-prefetch hint (the engine's
    // own JIT/host does what it can), so the trait's no-op default is the honest
    // lowering and `HAS_PREFETCH` stays false.
}

#[thermite_macros::inline_always]
impl NativeSimd for Wasm {
    type f32xN = F32x4Wasm;
    type i32xN = I32x4Wasm;
    type u32xN = U32x4Wasm;

    type f64xN = F64x2Wasm;
    type i64xN = I64x2Wasm;
    type u64xN = U64x2Wasm;

    type i16xN = I16x8Wasm;
    type u16xN = U16x8Wasm;

    type i8xN = I8x16Wasm;
    type u8xN = U8x16Wasm;
}

// Scatter/Gather is not available on WASM, so we use fallback scalar impls.
// Only provide the concrete index types; usizex* resolves via type alias to u32x* on wasm32
// and to u64x* on wasm64, so explicit usizex* impls would duplicate and conflict.
macro_rules! impl_indexable {
    ($idx:ty => $($ty:ty),* $(,)?) => {$( impl IndexableRegister<$idx> for $ty {} )*};
}

// x2: 64-bit native types need u32x2 and u64x2 index support
impl_indexable!(<Wasm as Simd>::u32x2 => F64x2Wasm, I64x2Wasm, U64x2Wasm);
impl_indexable!(<Wasm as Simd>::u64x2 => F64x2Wasm, I64x2Wasm, U64x2Wasm);

// x4: 32-bit and wider types need u32x4 and u64x4 index support
impl_indexable!(<Wasm as Simd>::u32x4 => F32x4Wasm, I32x4Wasm, U32x4Wasm,
    <Wasm as Simd>::f64x4, <Wasm as Simd>::i64x4, <Wasm as Simd>::u64x4);
impl_indexable!(<Wasm as Simd>::u64x4 => F32x4Wasm, I32x4Wasm, U32x4Wasm);

// 16-bit gather/scatter has no hardware support on WASM; scalar-fallback marker impls.
impl_indexable!(U16x8Wasm => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u32x8 => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u64x8 => I16x8Wasm, U16x8Wasm);
impl_indexable!(<Wasm as Simd>::u32x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<Wasm as Simd>::u64x2 => ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>);
impl_indexable!(<Wasm as Simd>::u64x16 => ArrayRegister<I16x8Wasm, 2>, ArrayRegister<U16x8Wasm, 2>);

// 8-bit: same-width self-indexing plus the 16-lane usize/u32/u64 index trio (scalar-fallback
// markers, matching the sub-native i8x8/i8x4 ladder; usizex16 aliases u32x16/u64x16).
impl_indexable!(U8x16Wasm => I8x16Wasm, U8x16Wasm);
impl_indexable!(<Wasm as Simd>::u32x16 => I8x16Wasm, U8x16Wasm);
impl_indexable!(<Wasm as Simd>::u64x16 => I8x16Wasm, U8x16Wasm);

impl Simd for Wasm {
    type usizex2 = <() as FindUSize<(), Self::u32x2, Self::u64x2>>::Output;
    type usizex4 = <() as FindUSize<(), Self::u32x4, Self::u64x4>>::Output;
    type usizex8 = <() as FindUSize<(), Self::u32x8, Self::u64x8>>::Output;
    type usizex16 = <() as FindUSize<(), Self::u32x16, Self::u64x16>>::Output;

    type f32x2 = F32x2Wasm;
    type i32x2 = I32x2Wasm;
    type u32x2 = U32x2Wasm;

    type f32x4 = F32x4Wasm;
    type i32x4 = I32x4Wasm;
    type u32x4 = U32x4Wasm;

    type f64x2 = F64x2Wasm;
    type i64x2 = I64x2Wasm;
    type u64x2 = U64x2Wasm;

    type f32x8 = ArrayRegister<F32x4Wasm, 2>;
    type i32x8 = ArrayRegister<I32x4Wasm, 2>;
    type u32x8 = ArrayRegister<U32x4Wasm, 2>;

    type f64x4 = ArrayRegister<F64x2Wasm, 2>;
    type i64x4 = ArrayRegister<I64x2Wasm, 2>;
    type u64x4 = ArrayRegister<U64x2Wasm, 2>;

    type f32x16 = ArrayRegister<F32x4Wasm, 4>;
    type i32x16 = ArrayRegister<I32x4Wasm, 4>;
    type u32x16 = ArrayRegister<U32x4Wasm, 4>;

    type f64x8 = ArrayRegister<F64x2Wasm, 4>;
    type i64x8 = ArrayRegister<I64x2Wasm, 4>;
    type u64x8 = ArrayRegister<U64x2Wasm, 4>;

    type f64x16 = ArrayRegister<F64x2Wasm, 8>;
    type i64x16 = ArrayRegister<I64x2Wasm, 8>;
    type u64x16 = ArrayRegister<U64x2Wasm, 8>;

    type i16x2 = ArrayRegister<i16, 2>;
    type u16x2 = ArrayRegister<u16, 2>;

    type i16x4 = half16::I16x4Wasm;
    type u16x4 = half16::U16x4Wasm;

    type i16x8 = I16x8Wasm;
    type u16x8 = U16x8Wasm;

    type i16x16 = ArrayRegister<I16x8Wasm, 2>;
    type u16x16 = ArrayRegister<U16x8Wasm, 2>;

    type i8x16 = I8x16Wasm;
    type u8x16 = U8x16Wasm;

    type i8x2 = ArrayRegister<i8, 2>;
    type u8x2 = ArrayRegister<u8, 2>;
    type i8x4 = half8::I8x4Wasm;
    type u8x4 = half8::U8x4Wasm;
    type i8x8 = half8::I8x8Wasm;
    type u8x8 = half8::U8x8Wasm;
}

// fp8 pack/unpack (generic branchless defaults) on the u8 ladder -> matching f32 widths.
impl_packed_fp8! {
    ArrayRegister<u8, 2> => F32x2Wasm,
    half8::U8x4Wasm => F32x4Wasm,
    half8::U8x8Wasm => ArrayRegister<F32x4Wasm, 2>,
    U8x16Wasm => ArrayRegister<F32x4Wasm, 4>,
}

// Same-width, different-lane-count reinterprets of the byte register, so it can be viewed
// as wider accumulator lanes (the SAD family). Everything is `v128`, so identity.
impl_bit_casts_identity! {
    U8x16Wasm as U16x8Wasm,
    U8x16Wasm as U32x4Wasm,
    U8x16Wasm as U64x2Wasm,
}

// Sub-native byte ladder: lane-wise (see `impl_sad_scalar!`).
impl_sad_scalar! {
    half8::U8x8Wasm => (half16::U16x4Wasm, U32x2Wasm, u64),
    half8::U8x4Wasm => (ArrayRegister<u16, 2>, u32, u64),
}

// Wider-element SAD. `u32x4_extadd_pairwise_u16x8` is the native `u16` pair sum; the rest
// reinterpret and fold (everything is `v128`, so the casts are identity).
impl_bit_casts_identity! {
    U16x8Wasm as U32x4Wasm,
    U16x8Wasm as U64x2Wasm,
    U32x4Wasm as U64x2Wasm,
}

const _: () = {
    use crate::register::{Sad32Register, Sad64Register, UnsignedIntegerRegister};

    #[thermite_macros::inline_always]
    impl Sad32Register<U32x4Wasm> for U16x8Wasm {
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x4Wasm> {
            arch::u32x4_extadd_pairwise_u16x8(Self::abs_diff(a, b))
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x2Wasm> for U16x8Wasm {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x2Wasm> {
            let x = arch::u32x4_extadd_pairwise_u16x8(Self::abs_diff(a, b));
            arch::v128_and(
                arch::u64x2_add(x, arch::u64x2_shr(x, 32)),
                arch::u64x2_splat(0xffff_ffff),
            )
        }
    }
};

impl_sad_u32!(@swar U32x4Wasm => U64x2Wasm);

// Sub-native rungs: lane-wise.
impl_sad_u16!(@scalar half16::U16x4Wasm => (U32x2Wasm, u64));
impl_sad_u32!(@scalar U32x2Wasm => u64);

// SIMD128 has widening pairwise adds for the two narrow groupings
// (`extadd_pairwise`), so 2- and 4-byte SAD are one instruction past the absolute
// difference. There is no `i64x2` extadd, so the 8-byte grouping chains both and folds
// the final u32 pair into a u64 lane by hand - still well short of the full SWAR cascade.
const _: () = {
    use crate::register::{Sad16Register, Sad32Register, Sad64Register, UnsignedIntegerRegister};

    #[thermite_macros::inline_always]
    impl Sad16Register<U16x8Wasm> for U8x16Wasm {
        fn sad16(a: Storage<Self>, b: Storage<Self>) -> Storage<U16x8Wasm> {
            arch::u16x8_extadd_pairwise_u8x16(Self::abs_diff(a, b))
        }
    }

    #[thermite_macros::inline_always]
    impl Sad32Register<U32x4Wasm> for U8x16Wasm {
        fn sad32(a: Storage<Self>, b: Storage<Self>) -> Storage<U32x4Wasm> {
            arch::u32x4_extadd_pairwise_u16x8(arch::u16x8_extadd_pairwise_u8x16(Self::abs_diff(a, b)))
        }
    }

    #[thermite_macros::inline_always]
    impl Sad64Register<U64x2Wasm> for U8x16Wasm {
        fn sad64(a: Storage<Self>, b: Storage<Self>) -> Storage<U64x2Wasm> {
            let x = arch::u32x4_extadd_pairwise_u16x8(arch::u16x8_extadd_pairwise_u8x16(Self::abs_diff(a, b)));
            // Each u64 lane now holds two independent u32 sums; add them and drop the
            // high half. (`extend_low`/`extend_high` would pair lanes 0+2 / 1+3, which is
            // the wrong grouping.)
            arch::v128_and(
                arch::u64x2_add(x, arch::u64x2_shr(x, 32)),
                arch::u64x2_splat(0xffff_ffff),
            )
        }
    }
};

impl Simd3 for Wasm {
    type usizex3 = <Self as Simd3A>::usizex3A;

    type f32x3 = <Self as Simd3A>::f32x3A;
    type i32x3 = <Self as Simd3A>::i32x3A;
    type u32x3 = <Self as Simd3A>::u32x3A;

    type f64x3 = <Self as Simd3A>::f64x3A;
    type i64x3 = <Self as Simd3A>::i64x3A;
    type u64x3 = <Self as Simd3A>::u64x3A;
}

impl_concat_bool_register2!(f32, F32x2Wasm);
impl_concat_bool_register2!(u32, U32x2Wasm);
impl_concat_bool_register2!(i32, I32x2Wasm);

impl_concat_bool_register2!(f64, F64x2Wasm);
impl_concat_bool_register2!(u64, U64x2Wasm);
impl_concat_bool_register2!(i64, I64x2Wasm);

macro_rules! impl_identity_casts {
    ($($from:ty as $to:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    value // all bit casts are no-ops in Wasm
                }
            }

            #[thermite_macros::inline_always]
            impl $crate::register::CastMaskRegister<$from> for $to {
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    value // all mask casts are no-ops in Wasm
                }
            }
        )*};
    };
}

macro_rules! impl_type_casts {
    ($($from:ty as $to:ty => $conv:ident $(| $fast:ident)?),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    arch::$conv(value)
                }

                $(
                    fn fast_cast_from(value: Storage<$from>) -> Storage<Self> {
                        arch::$fast(value)
                    }
                )?
            }
        )*};
    };
}

impl_identity_casts! {
    // 32-bit
    F32x4Wasm as U32x4Wasm,
    F32x4Wasm as I32x4Wasm,
    U32x4Wasm as F32x4Wasm,
    U32x4Wasm as I32x4Wasm,
    I32x4Wasm as F32x4Wasm,
    I32x4Wasm as U32x4Wasm,

    // 64-bit
    F64x2Wasm as U64x2Wasm,
    F64x2Wasm as I64x2Wasm,
    U64x2Wasm as F64x2Wasm,
    U64x2Wasm as I64x2Wasm,
    I64x2Wasm as F64x2Wasm,
    I64x2Wasm as U64x2Wasm,

    // identity casts
    F32x4Wasm as F32x4Wasm,
    I32x4Wasm as I32x4Wasm,
    U32x4Wasm as U32x4Wasm,
    F64x2Wasm as F64x2Wasm,
    I64x2Wasm as I64x2Wasm,
    U64x2Wasm as U64x2Wasm,

    // 16-bit self + sibling (i16<->u16)
    I16x8Wasm as U16x8Wasm,
    U16x8Wasm as I16x8Wasm,
    I16x8Wasm as I16x8Wasm,
    U16x8Wasm as U16x8Wasm,

    // 8-bit self + sibling (i8<->u8)
    I8x16Wasm as U8x16Wasm,
    U8x16Wasm as I8x16Wasm,
    I8x16Wasm as I8x16Wasm,
    U8x16Wasm as U8x16Wasm,
}

impl_type_casts! {
    // same type casts (identity)
    F32x4Wasm as F32x4Wasm => identity,
    F64x2Wasm as F64x2Wasm => identity,
    I32x4Wasm as I32x4Wasm => identity,
    I64x2Wasm as I64x2Wasm => identity,
    U32x4Wasm as U32x4Wasm => identity,
    U64x2Wasm as U64x2Wasm => identity,

    // integer casts (also identity)
    I32x4Wasm as U32x4Wasm => identity,
    U32x4Wasm as I32x4Wasm => identity,
    I64x2Wasm as U64x2Wasm => identity,
    U64x2Wasm as I64x2Wasm => identity,

    F32x4Wasm as I32x4Wasm => i32x4_trunc_sat_f32x4 | i32x4_relaxed_trunc_f32x4,
    F32x4Wasm as U32x4Wasm => u32x4_trunc_sat_f32x4 | u32x4_relaxed_trunc_f32x4,
    I32x4Wasm as F32x4Wasm => f32x4_convert_i32x4,
    U32x4Wasm as F32x4Wasm => f32x4_convert_u32x4,

    // 16-bit self + sibling (i16<->i32 widen/narrow live in i16x8.rs / half16.rs)
    I16x8Wasm as I16x8Wasm => identity,
    U16x8Wasm as U16x8Wasm => identity,
    I16x8Wasm as U16x8Wasm => identity,
    U16x8Wasm as I16x8Wasm => identity,

    // 8-bit self + sibling
    I8x16Wasm as I8x16Wasm => identity,
    U8x16Wasm as U8x16Wasm => identity,
    I8x16Wasm as U8x16Wasm => identity,
    U8x16Wasm as I8x16Wasm => identity,
}

// `u64x4 -> f32x4`, composed through f64 where both legs already exist.
impl_cast_via! {
    ArrayRegister<U64x2Wasm, 2> as F32x4Wasm => via ArrayRegister<F64x2Wasm, 2>,
}

// Cross-width float -> int saturating casts, one row per Simd lane count
// (`[f32, f64, i32, u32, i64, u64, i16, u16, i8, u8]`), composed from the
// same-width saturating casts above and the integer-narrowing saturating matrix.
impl_float_cast_matrix! {
    [F32x2Wasm, F64x2Wasm, I32x2Wasm, U32x2Wasm, I64x2Wasm, U64x2Wasm,
        ArrayRegister<i16, 2>, ArrayRegister<u16, 2>, ArrayRegister<i8, 2>, ArrayRegister<u8, 2>],
    [F32x4Wasm, ArrayRegister<F64x2Wasm, 2>, I32x4Wasm, U32x4Wasm, ArrayRegister<I64x2Wasm, 2>,
        ArrayRegister<U64x2Wasm, 2>, half16::I16x4Wasm, half16::U16x4Wasm, half8::I8x4Wasm, half8::U8x4Wasm],
}

impl_cast_via! {
    // x8 (wide rows are partial: the ArrayRegister cast ladder bridges the
    // factor-of-two array<->array pairs from the x4 impls stamped above)
    ArrayRegister<F32x4Wasm, 2> as I16x8Wasm => via ArrayRegister<I32x4Wasm, 2>,
    ArrayRegister<F32x4Wasm, 2> as half8::I8x8Wasm => via ArrayRegister<I32x4Wasm, 2>,
    ArrayRegister<F32x4Wasm, 2> as U16x8Wasm => via ArrayRegister<U32x4Wasm, 2>,
    ArrayRegister<F32x4Wasm, 2> as half8::U8x8Wasm => via ArrayRegister<U32x4Wasm, 2>,
    ArrayRegister<F64x2Wasm, 4> as I16x8Wasm => via ArrayRegister<I64x2Wasm, 4>,
    ArrayRegister<F64x2Wasm, 4> as half8::I8x8Wasm => via ArrayRegister<I64x2Wasm, 4>,
    ArrayRegister<F64x2Wasm, 4> as U16x8Wasm => via ArrayRegister<U64x2Wasm, 4>,
    ArrayRegister<F64x2Wasm, 4> as half8::U8x8Wasm => via ArrayRegister<U64x2Wasm, 4>,
    // x16
    ArrayRegister<F32x4Wasm, 4> as I8x16Wasm => via ArrayRegister<I32x4Wasm, 4>,
    ArrayRegister<F32x4Wasm, 4> as U8x16Wasm => via ArrayRegister<U32x4Wasm, 4>,
    ArrayRegister<F64x2Wasm, 8> as ArrayRegister<I16x8Wasm, 2> => via ArrayRegister<I64x2Wasm, 8>,
    ArrayRegister<F64x2Wasm, 8> as ArrayRegister<U16x8Wasm, 2> => via ArrayRegister<U64x2Wasm, 8>,
    ArrayRegister<F64x2Wasm, 8> as I8x16Wasm => via ArrayRegister<I64x2Wasm, 8>,
    ArrayRegister<F64x2Wasm, 8> as U8x16Wasm => via ArrayRegister<U64x2Wasm, 8>,
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
    F32x4Wasm, I32x4Wasm, U32x4Wasm, F64x2Wasm, I64x2Wasm, U64x2Wasm, I16x8Wasm, U16x8Wasm, I8x16Wasm, U8x16Wasm
);

// `WidenIndexRegister` for the table-path registers (see the macro for why wasm
// needs a two-step extend ladder rather than a single widening load).
impl_widen_indices_wasm! {
    F32x4Wasm => x4, I32x4Wasm => x4, U32x4Wasm => x4,
    F64x2Wasm => x2, I64x2Wasm => x2, U64x2Wasm => x2,
    I16x8Wasm => x8, U16x8Wasm => x8,
}
