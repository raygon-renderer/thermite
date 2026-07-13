use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastMaskRegister, CastRegister, CoreRegister, FloatRegister, InterleaveRegister,
        LinAlg3Register, LinAlg4Register, MaskElement, MaskRegister, NativeCapability, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, ZeroUpper,
        array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4Neon;

neon_mask_core!(
    F32x4Neon, lanes: 4(typenum::U4), storage: float32x4_t, suffix: f32,
    truthy: f32::from_bits(!0), from_u: vreinterpretq_f32_u32
);

neon_register!(
    F32x4Neon, elem: f32, lanes: 4, suffix: f32,
    signed: super::I32x4Neon, unsigned: super::U32x4Neon,
    compress: table
);

neon_partial_ord!(F32x4Neon, suffix: f32, from_u: vreinterpretq_f32_u32);

neon_shuffle_permute!(F32x4Neon, suffix: f32, from_u: vreinterpretq_f32_u32; 4);

neon_float_register!(
    F32x4Neon, elem: f32, lanes: 4, suffix: f32, from_u: vreinterpretq_f32_u32,
    bits: super::U32x4Neon, signed_bits: super::I32x4Neon,
    extended: ArrayRegister<super::F64x2Neon, 2>,
    exp_mask: 0x7F80_0000, approx: yes
);

// f32x4 -> f64x4 (ExtendedPrecision): promote each half with `vcvt`.
#[thermite_macros::inline_always]
impl CastRegister<F32x4Neon> for ArrayRegister<super::F64x2Neon, 2> {
    fn cast_from(value: Storage<F32x4Neon>) -> Storage<Self> {
        unsafe {
            let lo = arch::vcvt_f64_f32(arch::vget_low_f32(value));
            let hi = arch::vcvt_high_f64_f32(value);
            ArrayRegister([lo, hi])
        }
    }
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F32x4Neon {
    fn min_element3(value: Storage<Self>) -> Self::Element {
        // Lane 3 replaced with the identity for min so the 4-lane reduction ignores it.
        Self::min_element(Self::insert::<3>(value, f32::INFINITY))
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        Self::max_element(Self::insert::<3>(value, f32::NEG_INFINITY))
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        Self::sum_elements(Self::insert::<3>(value, 0.0))
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        Self::prod_elements(Self::insert::<3>(value, 1.0))
    }
}

#[thermite_macros::inline_always]
impl LinAlg4Register for F32x4Neon {}

// f64x4 (ExtendedPrecision) -> f32x4: demote each half with `vcvt`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x2Neon, 2>> for F32x4Neon {
    fn cast_from(value: Storage<ArrayRegister<super::F64x2Neon, 2>>) -> Storage<Self> {
        unsafe { arch::vcvt_high_f32_f64(arch::vcvt_f32_f64(value.0[0]), value.0[1]) }
    }
}
