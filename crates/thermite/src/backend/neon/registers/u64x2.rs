use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastMaskRegister, CoreRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        Storage, UnsignedIntegerRegister, ZeroUpper, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2Neon;

neon_mask_core!(
    U64x2Neon, lanes: 2(typenum::U2), storage: uint64x2_t, suffix: u64,
    truthy: !0, from_u: identity
);

neon_register!(
    U64x2Neon, elem: u64, lanes: 2, suffix: u64,
    signed: super::I64x2Neon, unsigned: super::U64x2Neon,
    compress: table
);

neon_partial_ord!(U64x2Neon, suffix: u64, from_u: identity);

neon_shuffle_permute!(U64x2Neon, suffix: u64, from_u: identity; 2);

neon_int64_register!(
    U64x2Neon, elem: u64, suffix: u64, wide: u128,
    minmax: (neon_min_u64, neon_max_u64),
    to_u: identity, from_u: identity, div: epu
);

neon_bitshift!(
    U64x2Neon, suffix: u64, unsigned: u64, count: (i64, s64),
    to_u: identity, from_u: identity, to_c: vreinterpretq_s64_u64,
    bytes: (vreinterpretq_u8_u64, vreinterpretq_u64_u8)
);

// No 64-bit `vrhaddq`/`vabdq`; the portable defaults are already optimal for
// those. `next_power_of_two_m1` still wins with `!0 >> clz(v)` even through the
// composed 64-bit clz (no `vclzq_u64` exists).
#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U64x2Neon {
    fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let clz = arch::neon_clz_u64(value);
            arch::vshlq_u64(Self::MAX, arch::vnegq_s64(arch::vreinterpretq_s64_u64(clz)))
        }
    }

    // 64 - clz(v) (see the 8/16/32-bit macro version for the rationale).
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::vsubq_u64(Self::splat(64), arch::neon_clz_u64(value)) }
    }
}

neon_concat_scalar2!(U64x2Neon, elem: u64, suffix: u64);
neon_extend_scalar!(U64x2Neon, elem: u64);
