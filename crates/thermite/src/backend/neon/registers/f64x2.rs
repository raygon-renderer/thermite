use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastMaskRegister, CoreRegister, FloatRegister, InterleaveRegister, MaskElement, MaskRegister,
        NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedRegister, Storage, ZeroUpper, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2Neon;

neon_mask_core!(
    F64x2Neon, lanes: 2(typenum::U2), storage: float64x2_t, suffix: f64,
    truthy: f64::from_bits(!0), from_u: vreinterpretq_f64_u64
);

neon_register!(
    F64x2Neon, elem: f64, lanes: 2, suffix: f64,
    signed: super::I64x2Neon, unsigned: super::U64x2Neon,
    compress: table
);

neon_partial_ord!(F64x2Neon, suffix: f64, from_u: vreinterpretq_f64_u64);

neon_shuffle_permute!(F64x2Neon, suffix: f64, from_u: vreinterpretq_f64_u64; 2);

neon_float_register!(
    F64x2Neon, elem: f64, lanes: 2, suffix: f64, from_u: vreinterpretq_f64_u64,
    bits: super::U64x2Neon, signed_bits: super::I64x2Neon,
    extended: Self,
    exp_mask: 0x7FF0_0000_0000_0000
);

neon_concat_scalar2!(F64x2Neon, elem: f64, suffix: f64);
neon_extend_scalar!(F64x2Neon, elem: f64);
