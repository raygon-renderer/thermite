use crate::register::{BitwiseRegister, CastMaskRegister, CoreRegister, InterleaveRegister, MaskRegister, Storage};

use generic_array::GenericArray;

#[thermite_macros::inline_always]
impl CoreRegister for bool {
    type Lanes = generic_array::typenum::U1;
    type Mask = Self;
    type Storage = Self;

    const IS_EMULATED: bool = false;
    const ISA: crate::InstructionSet = crate::InstructionSet::Scalar;
    const EMPTY: Storage<Self> = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;

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

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

// TODO: implement masked variants using boolean logic?
#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for bool {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }
    fn not(value: Storage<Self>) -> Storage<Self> { !value }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl MaskRegister for bool {
    const TRUTHY: Storage<Self> = true;
    const FALSY: Storage<Self> = false;

    fn set(_mask: Storage<Self::Mask>, _lane: usize, value: bool) -> Storage<Self> { value }
    fn test(mask: Storage<Self::Mask>, _lane: usize) -> bool { mask }
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> { value[0] }

    fn all(value: Storage<Self>) -> bool { value }
    fn any(value: Storage<Self>) -> bool { value }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> { Some(value as u64) }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) { view.set(0, value); }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl InterleaveRegister for bool {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl CastMaskRegister<bool> for bool {
    fn mask_from(value: Storage<bool>) -> Storage<Self> { value }
}
