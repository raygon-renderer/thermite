use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        Storage, UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x4Neon;

neon_mask_core!(
    U32x4Neon, lanes: 4(typenum::U4), storage: uint32x4_t, suffix: u32,
    truthy: !0, from_u: identity
);

neon_register!(
    U32x4Neon, elem: u32, lanes: 4, suffix: u32, vec: uint32x4,
    signed: super::I32x4Neon, unsigned: super::U32x4Neon,
    compress: table, bytes: (vreinterpretq_u8_u32, vreinterpretq_u32_u8)
);

neon_partial_ord!(U32x4Neon, suffix: u32, from_u: identity);

neon_shuffle_permute!(U32x4Neon, suffix: u32, from_u: identity; 4);

neon_extend_scalar!(U32x4Neon, elem: u32);

neon_int_numeric!(U32x4Neon, elem: u32, lanes: 4, suffix: u32);

neon_bitshift!(
    U32x4Neon, suffix: u32, unsigned: u32, count: (i32, s32),
    to_u: identity, from_u: identity, to_c: vreinterpretq_s32_u32,
    bytes: (vreinterpretq_u8_u32, vreinterpretq_u32_u8)
);

neon_int_register!(U32x4Neon, suffix: u32, unsigned: u32, to_u: identity, from_u: identity, div: epu);

neon_unsigned_int!(
    U32x4Neon, suffix: u32, neg: (s32, vreinterpretq_s32_u32),
    extras: {
        /// 2D Morton via a `vqtbl1q_u8` nibble-LUT; every other `N` uses the cascade.
        fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
            if const { N == 2 } {
                arch::neon_morton2_u32(values[0], values[1])
            } else {
                crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
            }
        }

        /// 2D Morton decode via the `vqtbl1q_u8` compress; every other `N` uses the cascade.
        fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
            if const { N == 2 } {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::neon_morton2_compress_u32(code),
                    arch::neon_morton2_compress_u32(Self::shri::<1>(code)),
                )
            } else {
                crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
            }
        }
    }
);

neon_widen_casts!(U32x4Neon => [super::U64x2Neon; 2], suffixes: u32/u64);
