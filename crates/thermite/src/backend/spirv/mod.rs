pub mod arch;
pub mod registers;

use crate::{
    isa::InstructionSet,
    simd::{HasIsa, NativeIsa, NativeSimd},
};

/// The SPIRV backend. Each shader invocation is a single lane (SIMT model),
/// so the "native" width is 1 for all element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SPIRV;

impl HasIsa for SPIRV {
    const ISA: InstructionSet = InstructionSet::SPIRV;
}

impl NativeIsa for SPIRV {
    // One register per invocation in the SIMT model
    type Registers = generic_array::typenum::U1;
    type Native32Width = generic_array::typenum::U1;
    type Native64Width = generic_array::typenum::U1;
    type NativeAlignment = (); // no alignment requirement for scalar SPIRV
}

impl NativeSimd for SPIRV {
    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

// TODO: impl Simd for SPIRV
