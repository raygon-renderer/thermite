use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x2Neon;

neon_mask_core!(
    I64x2Neon, lanes: 2(typenum::U2), storage: int64x2_t, suffix: s64,
    truthy: -1, from_u: vreinterpretq_s64_u64
);

neon_register!(
    I64x2Neon, elem: i64, lanes: 2, suffix: s64, vec: int64x2,
    signed: super::I64x2Neon, unsigned: super::U64x2Neon,
    compress: table, bytes: (vreinterpretq_u8_s64, vreinterpretq_s64_u8)
);

neon_partial_ord!(I64x2Neon, suffix: s64, from_u: vreinterpretq_s64_u64);

neon_shuffle_permute!(I64x2Neon, suffix: s64, from_u: vreinterpretq_s64_u64; 2);

neon_int64_register!(
    I64x2Neon, elem: i64, suffix: s64, wide: i128,
    minmax: (neon_min_s64, neon_max_s64),
    to_u: vreinterpretq_u64_s64, from_u: vreinterpretq_s64_u64, div: epi
);

neon_bitshift!(
    I64x2Neon, suffix: s64, unsigned: u64, count: (i64, s64),
    to_u: vreinterpretq_u64_s64, from_u: vreinterpretq_s64_u64, to_c: vreinterpretq_s64_u64,
    bytes: (vreinterpretq_u8_s64, vreinterpretq_s64_u8)
);

neon_signed_int!(
    I64x2Neon, lanes: 2, suffix: s64, count: i64,
    from_u: vreinterpretq_s64_u64, to_c: vreinterpretq_s64_u64
);

neon_concat_scalar2!(I64x2Neon, elem: i64, suffix: s64);
neon_extend_scalar!(I64x2Neon, elem: i64);
