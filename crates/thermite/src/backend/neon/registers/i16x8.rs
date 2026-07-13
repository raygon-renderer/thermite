use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister,
        SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I16x8Neon;

neon_mask_core!(
    I16x8Neon, lanes: 8(typenum::U8), storage: int16x8_t, suffix: s16,
    truthy: -1, from_u: vreinterpretq_s16_u16
);

neon_register!(
    I16x8Neon, elem: i16, lanes: 8, suffix: s16,
    signed: super::I16x8Neon, unsigned: super::U16x8Neon,
    compress: table
);

neon_partial_ord!(I16x8Neon, suffix: s16, from_u: vreinterpretq_s16_u16);

neon_int_numeric!(I16x8Neon, elem: i16, lanes: 8, suffix: s16);

neon_bitshift!(
    I16x8Neon, suffix: s16, unsigned: u16, count: (i16, s16),
    to_u: vreinterpretq_u16_s16, from_u: vreinterpretq_s16_u16, to_c: vreinterpretq_s16_u16,
    bytes: (vreinterpretq_u8_s16, vreinterpretq_s16_u8)
);

neon_int_register!(
    I16x8Neon, suffix: s16, unsigned: u16,
    to_u: vreinterpretq_u16_s16, from_u: vreinterpretq_s16_u16, div: epi
);

neon_signed_int!(
    I16x8Neon, lanes: 8, suffix: s16, count: i16,
    from_u: vreinterpretq_s16_u16, to_c: vreinterpretq_s16_u16,
    signed_extras: {
        // Clamp to [-1, 1]: 2 instructions (SMAX/SMIN) vs the ~4-op
        // compare/select default. Zero stays zero.
        fn signum(value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vminq_s16(arch::vmaxq_s16(value, Self::NEG_ONE), Self::ONE) }
        }
    },
    extras: {
        // Single-instruction halving averages (SHADD/SRHADD) vs the 3-op
        // bit-trick defaults.
        fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vhaddq_s16(a, b) }
        }

        fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vrhaddq_s16(a, b) }
        }

        // Widening multiply + rounding-shift-narrow: 4 instructions
        // (SMULL/SMULL2/RSHRN/RSHRN2), no constants, and exactly PMULHRSW by
        // construction - `(a*b + 0x4000) >> 15` computed in i32 then a
        // *truncating* narrow, so the `MIN * MIN` corner wraps to `MIN` with
        // no fixup. (`vqrdmulhq_s16` is 1 instruction but *saturates* that
        // corner to `MAX` and needs a 3-op + 2-constant fixup; simde uses this
        // widening route too.)
        fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                let lo = arch::vmull_s16(arch::vget_low_s16(a), arch::vget_low_s16(b));
                let hi = arch::vmull_high_s16(a, b);
                arch::vrshrn_high_n_s32::<15>(arch::vrshrn_n_s32::<15>(lo), hi)
            }
        }
    }
);

neon_extend_scalar!(I16x8Neon, elem: i16);

neon_widen_casts!(I16x8Neon => [super::I32x4Neon; 2], suffixes: s16/s32);
