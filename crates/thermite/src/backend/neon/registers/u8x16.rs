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
pub struct U8x16Neon;

neon_mask_core!(
    U8x16Neon, lanes: 16(typenum::U16), storage: uint8x16_t, suffix: u8,
    truthy: !0, from_u: identity
);

neon_register!(
    U8x16Neon, elem: u8, lanes: 16, suffix: u8, vec: uint8x16,
    signed: super::I8x16Neon, unsigned: super::U8x16Neon,
    compress: wide, bytes: (identity, identity),
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
                    16 => arch::vqtbl1q_u8(arch::vld1q_u8(values.as_ptr()), indices),
                    32 => arch::vqtbl2q_u8(arch::vld1q_u8_x2(values.as_ptr()), indices),
                    48 => arch::vqtbl3q_u8(arch::vld1q_u8_x3(values.as_ptr()), indices),
                    64 => arch::vqtbl4q_u8(arch::vld1q_u8_x4(values.as_ptr()), indices),
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

neon_partial_ord!(U8x16Neon, suffix: u8, from_u: identity);

neon_int_numeric!(U8x16Neon, elem: u8, lanes: 16, suffix: u8);

neon_bitshift!(
    U8x16Neon, suffix: u8, unsigned: u8, count: (i8, s8),
    to_u: identity, from_u: identity, to_c: vreinterpretq_s8_u8,
    bytes: (identity, identity)
);

neon_int_register!(U8x16Neon, suffix: u8, unsigned: u8, to_u: identity, from_u: identity, div: epu);

neon_unsigned_int!(U8x16Neon, suffix: u8, neg: (s8, vreinterpretq_s8_u8));

neon_extend_scalar!(U8x16Neon, elem: u8);

neon_widen_casts!(U8x16Neon => [super::U16x8Neon; 2], suffixes: u8/u16);
