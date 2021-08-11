use crate::{
    arch::avx2::*,
    backends::{register::*, vector::Vector},
    widen::Widen,
    Simd, SimdInstructionSet,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AVX2;

pub mod polyfills;

pub mod vf32;

impl Simd for AVX2 {
    const INSTRSET: SimdInstructionSet = SimdInstructionSet::AVX2;

    type Vf32 = Self::Vf32x8;

    type Vf32x1 = (); // TODO: wrapped scalar float
    type Vf32x2 = (); // TODO: half a 128-bit register
    type Vf32x4 = Vector<AVX2F32Register<4>>;
    type Vf32x8 = Vector<AVX2F32Register<8>>;
    type Vf32x16 = Widen<Self, Self::Vf32x8, 2>; //2x wider
}

pub struct AVX2F32Register<const N: usize>([(); N]);
pub struct AVX2F64Register<const N: usize>([(); N]);
pub struct AVX2U32Register<const N: usize>([(); N]);
pub struct AVX2U64Register<const N: usize>([(); N]);
pub struct AVX2I32Register<const N: usize>([(); N]);
pub struct AVX2I64Register<const N: usize>([(); N]);
