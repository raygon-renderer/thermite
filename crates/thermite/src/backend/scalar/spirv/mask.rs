use crate::backend::spirv::arch::{self as arch, glsl};
use crate::{
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastMaskRegister, CoreRegister, Element, InterleaveRegister, MaskElement, MaskRegister,
        Storage, ZeroUpper,
    },
    simd::HasIsa,
};
use generic_array::{GenericArray, typenum};

#[thermite_macros::inline_always]
impl CoreRegister for bool {
    type NativeIsa = crate::backend::scalar::Scalar;
    type Lanes = typenum::U1;
    type Mask = Self;
    type Storage = Self;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: bool, on_false: bool, on_true: bool) -> bool {
        unsafe { arch::op_opselect::<bool, bool>(mask, on_true, on_false) }
    }

    fn zz(mask: bool, value: bool) -> bool {
        mask & value
    }
    fn nz(mask: bool, value: bool) -> bool {
        !mask & value
    }

    fn zeroupper_z<Z: ZeroUpper>(value: bool) -> bool {
        if const { Z::N >= 1 } { value } else { false }
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl BitwiseRegister for bool {
    fn bitxor(lhs: bool, rhs: bool) -> bool { lhs ^ rhs }
    fn bitand(lhs: bool, rhs: bool) -> bool { lhs & rhs }
    fn bitor (lhs: bool, rhs: bool) -> bool { lhs | rhs }
    fn not(value: bool) -> bool { !value }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for bool {
    fn interleave (a: bool, b: bool) -> (bool, bool) { (a, b) }
    fn deinterleave(a: bool, b: bool) -> (bool, bool) { (a, b) }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl MaskRegister for bool {
    const TRUTHY: bool = true;
    const FALSY:  bool = false;

    fn set(_mask: bool, _lane: usize, value: bool) -> bool { value }
    fn test(mask: bool, _lane: usize) -> bool { mask }
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> bool { value[0] }
    fn all(value: bool) -> bool { value }
    fn any(value: bool) -> bool { value }
    fn native_bitmask(value: bool) -> Option<u64> { Some(value as u64) }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: bool, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value);
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl CastMaskRegister<bool> for bool {
    fn mask_from(value: bool) -> bool { value }
}
