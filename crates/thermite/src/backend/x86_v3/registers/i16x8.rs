use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, Element, IntegerRegister, MaskElement, MaskRegister,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister,
        SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg, reg_splat,
    },
    simd::Simd,
};

use super::arch;
