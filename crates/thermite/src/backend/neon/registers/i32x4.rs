use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I32x4Neon;

neon_mask_core!(
    I32x4Neon, lanes: 4(typenum::U4), storage: int32x4_t, suffix: s32,
    truthy: -1, from_u: vreinterpretq_s32_u32
);

neon_register!(
    I32x4Neon, elem: i32, lanes: 4, suffix: s32, vec: int32x4,
    signed: super::I32x4Neon, unsigned: super::U32x4Neon,
    compress: table, bytes: (vreinterpretq_u8_s32, vreinterpretq_s32_u8)
);

neon_partial_ord!(I32x4Neon, suffix: s32, from_u: vreinterpretq_s32_u32);

neon_shuffle_permute!(I32x4Neon, suffix: s32, from_u: vreinterpretq_s32_u32; 4);

neon_int_numeric!(I32x4Neon, elem: i32, lanes: 4, suffix: s32);

neon_bitshift!(
    I32x4Neon, suffix: s32, unsigned: u32, count: (i32, s32),
    to_u: vreinterpretq_u32_s32, from_u: vreinterpretq_s32_u32, to_c: vreinterpretq_s32_u32,
    bytes: (vreinterpretq_u8_s32, vreinterpretq_s32_u8)
);

neon_int_register!(
    I32x4Neon, suffix: s32, unsigned: u32,
    to_u: vreinterpretq_u32_s32, from_u: vreinterpretq_s32_u32, div: epi
);

neon_signed_int!(
    I32x4Neon, lanes: 4, suffix: s32, count: i32,
    from_u: vreinterpretq_s32_u32, to_c: vreinterpretq_s32_u32,
    signed_extras: {
        // Clamp to [-1, 1]: 2 instructions (SMAX/SMIN) vs the ~4-op
        // compare/select default. Zero stays zero.
        fn signum(value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vminq_s32(arch::vmaxq_s32(value, Self::NEG_ONE), Self::ONE) }
        }
    },
    extras: {
        // Single-instruction halving averages (SHADD/SRHADD) vs the 3-op
        // bit-trick defaults.
        fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vhaddq_s32(a, b) }
        }

        fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vrhaddq_s32(a, b) }
        }

        // Widening multiply + rounding-shift-narrow, exactly wrap-correct with
        // no fixup; see i16x8.rs for the rationale.
        fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                let lo = arch::vmull_s32(arch::vget_low_s32(a), arch::vget_low_s32(b));
                let hi = arch::vmull_high_s32(a, b);
                arch::vrshrn_high_n_s64::<31>(arch::vrshrn_n_s64::<31>(lo), hi)
            }
        }
    }
);

neon_widen_casts!(I32x4Neon => [super::I64x2Neon; 2], suffixes: s32/s64);
