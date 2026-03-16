use crate::register::{BitwiseRegister, CastMaskRegister, CoreRegister, InterleaveRegister, MaskRegister, Storage};

use generic_array::GenericArray;

impl CoreRegister for bool {
    type Lanes = generic_array::typenum::U1;
    type Mask = Self;
    type Storage = Self;

    const IS_EMULATED: bool = false;
    const ISA: crate::InstructionSet = crate::InstructionSet::Scalar;
    const EMPTY: Storage<Self> = false;

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, on_true, on_false)
    }

    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        mask & value
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        !mask & value
    }

    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 1 } { value } else { Self::EMPTY } // if N == 0 zero everything
    }
}

// TODO: implement masked variants using boolean logic?
#[rustfmt::skip]
impl BitwiseRegister for bool {
    #[inline(always)] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
    #[inline(always)] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }
    #[inline(always)] fn not(value: Storage<Self>) -> Storage<Self> { !value }
}

#[rustfmt::skip]
impl MaskRegister for bool {
    const TRUTHY: Storage<Self> = true;
    const FALSY: Storage<Self> = false;

    #[inline(always)] fn set(_mask: Storage<Self::Mask>, _lane: usize, value: bool) -> Storage<Self> { value }
    #[inline(always)] fn test(mask: Storage<Self::Mask>, _lane: usize) -> bool { mask }
    #[inline(always)] fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> { value[0] }

    #[inline(always)] fn all(value: Storage<Self>) -> bool { value }
    #[inline(always)] fn any(value: Storage<Self>) -> bool { value }

    #[inline(always)] fn native_bitmask(value: Storage<Self>) -> Option<u64> { Some(value as u64) }
    #[inline(always)] fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) { view.set(0, value); }
}

#[rustfmt::skip]
impl InterleaveRegister for bool {
    #[inline(always)] fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
    #[inline(always)] fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
}

#[rustfmt::skip]
impl CastMaskRegister<bool> for bool {
    #[inline(always)] fn mask_from(value: Storage<bool>) -> Storage<Self> { value }
}
