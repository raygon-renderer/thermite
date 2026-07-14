use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register, Storage,
        UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x8Neon;

neon_mask_core!(
    U16x8Neon, lanes: 8(typenum::U8), storage: uint16x8_t, suffix: u16,
    truthy: !0, from_u: identity
);

neon_register!(
    U16x8Neon, elem: u16, lanes: 8, suffix: u16, vec: uint16x8,
    signed: super::I16x8Neon, unsigned: super::U16x8Neon,
    compress: table, bytes: (vreinterpretq_u8_u16, vreinterpretq_u16_u8)
);

neon_partial_ord!(U16x8Neon, suffix: u16, from_u: identity);

neon_int_numeric!(U16x8Neon, elem: u16, lanes: 8, suffix: u16);

neon_bitshift!(
    U16x8Neon, suffix: u16, unsigned: u16, count: (i16, s16),
    to_u: identity, from_u: identity, to_c: vreinterpretq_s16_u16,
    bytes: (vreinterpretq_u8_u16, vreinterpretq_u16_u8)
);

neon_int_register!(U16x8Neon, suffix: u16, unsigned: u16, to_u: identity, from_u: identity, div: epu);

neon_unsigned_int!(
    U16x8Neon, suffix: u16, neg: (s16, vreinterpretq_s16_u16),
    extras: {
        /// 2D Morton via a `vqtbl1q_u8` nibble-LUT; every other `N` uses the cascade.
        fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
            if const { N == 2 } {
                arch::neon_morton2_u16(values[0], values[1])
            } else {
                crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
            }
        }

        /// 2D Morton decode via the `vqtbl1q_u8` compress; every other `N` uses the cascade.
        fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
            if const { N == 2 } {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::neon_morton2_compress_u16(code),
                    arch::neon_morton2_compress_u16(Self::shri::<1>(code)),
                )
            } else {
                crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
            }
        }
    }
);

neon_extend_scalar!(U16x8Neon, elem: u16);

neon_widen_casts!(U16x8Neon => [super::U32x4Neon; 2], suffixes: u16/u32);
