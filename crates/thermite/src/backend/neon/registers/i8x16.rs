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
pub struct I8x16Neon;

neon_mask_core!(
    I8x16Neon, lanes: 16(typenum::U16), storage: int8x16_t, suffix: s8,
    truthy: -1, from_u: vreinterpretq_s8_u8
);

neon_register!(
    I8x16Neon, elem: i8, lanes: 16, suffix: s8, vec: int8x16,
    signed: super::I8x16Neon, unsigned: super::U8x16Neon,
    compress: wide, bytes: (vreinterpretq_u8_s8, vreinterpretq_s8_u8),
    extras: {
        // Whole-table TBL for byte tables of exactly 1/2/3/4 q-registers
        // (16/32/48/64 entries): one TBL(+table load) vs 16 scalar
        // bounds-checked loads. ~3c flat on Cortex-A53 for all table sizes;
        // slower (but still far ahead of scalar) on big cores, where A64
        // multi-register TBL is microcoded. Divergence note: for these exact
        // sizes an out-of-range index yields 0 (TBL semantics) where the
        // scalar path would panic - `lookup` is `unsafe fn` whose contract
        // already requires in-range indices.
        unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
            unsafe {
                match values.len() {
                    16 => arch::vreinterpretq_s8_u8(arch::vqtbl1q_u8(arch::vld1q_u8(values.as_ptr() as *const u8), indices)),
                    32 => arch::vreinterpretq_s8_u8(arch::vqtbl2q_u8(arch::vld1q_u8_x2(values.as_ptr() as *const u8), indices)),
                    48 => arch::vreinterpretq_s8_u8(arch::vqtbl3q_u8(arch::vld1q_u8_x3(values.as_ptr() as *const u8), indices)),
                    64 => arch::vreinterpretq_s8_u8(arch::vqtbl4q_u8(arch::vld1q_u8_x4(values.as_ptr() as *const u8), indices)),
                    _ => {
                        // scalar fallback, same as the trait default
                        let idxs = <Self::Unsigned as Register>::as_slice(&indices);
                        let mut res = Self::EMPTY;
                        let resa = Self::as_mut_slice(&mut res);
                        let mut i = 0;
                        while i < 16 {
                            resa[i] = values[idxs[i] as usize];
                            i += 1;
                        }
                        res
                    }
                }
            }
        }
    }
);

neon_partial_ord!(I8x16Neon, suffix: s8, from_u: vreinterpretq_s8_u8);

neon_int_numeric!(I8x16Neon, elem: i8, lanes: 16, suffix: s8);

neon_bitshift!(
    I8x16Neon, suffix: s8, unsigned: u8, count: (i8, s8),
    to_u: vreinterpretq_u8_s8, from_u: vreinterpretq_s8_u8, to_c: vreinterpretq_s8_u8,
    bytes: (vreinterpretq_u8_s8, vreinterpretq_s8_u8)
);

neon_int_register!(
    I8x16Neon, suffix: s8, unsigned: u8,
    to_u: vreinterpretq_u8_s8, from_u: vreinterpretq_s8_u8, div: epi
);

neon_signed_int!(
    I8x16Neon, lanes: 16, suffix: s8, count: i8,
    from_u: vreinterpretq_s8_u8, to_c: vreinterpretq_s8_u8,
    signed_extras: {
        // Clamp to [-1, 1]: 2 instructions (SMAX/SMIN) vs the ~4-op
        // compare/select default. Zero stays zero.
        fn signum(value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vminq_s8(arch::vmaxq_s8(value, Self::NEG_ONE), Self::ONE) }
        }
    },
    extras: {
        // Single-instruction halving averages (SHADD/SRHADD) vs the 3-op
        // bit-trick defaults. (No `vqrdmulhq_s8`, so `mulhrs` keeps its default.)
        fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vhaddq_s8(a, b) }
        }

        fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe { arch::vrhaddq_s8(a, b) }
        }
    }
);

neon_extend_scalar!(I8x16Neon, elem: i8);

neon_widen_casts!(I8x16Neon => [super::I16x8Neon; 2], suffixes: s8/s16);
