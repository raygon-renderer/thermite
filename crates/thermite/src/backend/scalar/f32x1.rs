use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::register::{
    BlendRegister, FloatRegister, FloatRegisterExt, LinAlg3Register, MaskElement, MaskRegister, NumericRegister,
    PartialOrdRegister, PermuteRegister, Register, ShiftRegister, ShuffleRegister, SignedRegister, SwizzleRegister,
    dp::DoublePumpRegister,
};

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F32x1Scalar;

impl Register for F32x1Scalar {
    type Lanes = typenum::U1;

    type Element = f32;
    type Storage = f32;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const EMPTY: Self::Storage = 0.0;

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        value[0]
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        value
    }

    #[inline(always)]
    fn xor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bits(lhs.to_bits() ^ rhs.to_bits())
    }

    #[inline(always)]
    fn and(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bits(lhs.to_bits() & rhs.to_bits())
    }

    #[inline(always)]
    fn or(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bits(lhs.to_bits() | rhs.to_bits())
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        f32::from_bits(!value.to_bits())
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        f32::from_bits(value.to_bits() >> shift)
    }

    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        f32::from_bits(value.to_bits() << shift)
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        Self::shr(value, shifts[0])
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        Self::shl(value, shifts[0])
    }
}

impl MaskRegister for F32x1Scalar {
    const FALSY: Self::Storage = f32::from_bits(0);
    const TRUTHY: Self::Storage = f32::from_bits(!0);

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        value.to_bool()
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        value.to_bool()
    }
}

impl ShiftRegister for F32x1Scalar {
    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self::shl(value, IMM8 as u32)
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self::shr(value, IMM8 as u32)
    }
}

impl PartialOrdRegister for F32x1Scalar {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs < rhs)
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs <= rhs)
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs > rhs)
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs >= rhs)
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs == rhs)
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        f32::from_bool(lhs != rhs)
    }
}

impl NumericRegister for F32x1Scalar {
    const ZERO: Self::Storage = 0.0;
    const ONE: Self::Storage = 1.0;
    const TWO: Self::Storage = 2.0;

    const MIN: Self::Storage = f32::MIN;
    const MAX: Self::Storage = f32::MAX;

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs + rhs
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs - rhs
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs * rhs
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs / rhs
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs % rhs
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if lhs < rhs { lhs } else { rhs }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if lhs > rhs { lhs } else { rhs }
    }

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        value
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        value
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        value
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        value
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        1.0
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        0.0
    }
}

impl SignedRegister for F32x1Scalar {
    const NEG_ONE: Self::Storage = -1.0;

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        -value
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        value.abs()
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        value.signum()
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        lhs.copysign(rhs)
    }
}

impl BlendRegister for F32x1Scalar {
    fn blend<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if IMM8 == 0 {
            lhs
        } else if IMM8 == 1 {
            rhs
        } else {
            panic!("Invalid blend mask")
        }
    }
}

impl PermuteRegister for F32x1Scalar {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        value
    }
}

impl ShuffleRegister for F32x1Scalar {
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        if IMM8 == 0 {
            lhs
        } else if IMM8 == 1 {
            rhs
        } else {
            panic!("Invalid shuffle mask")
        }
    }
}
