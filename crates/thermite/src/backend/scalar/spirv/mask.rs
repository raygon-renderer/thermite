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

impl CoreRegister for bool {
    type Lanes = typenum::U1;
    type Mask = Self;
    type Storage = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::SPIRV;
    const EMPTY: Storage<Self> = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)]
    fn blendv(mask: bool, on_false: bool, on_true: bool) -> bool {
        unsafe { arch::op_opselect::<bool, bool>(mask, on_true, on_false) }
    }

    #[inline(always)]
    fn z(mask: bool, value: bool) -> bool {
        mask & value
    }
    #[inline(always)]
    fn nz(mask: bool, value: bool) -> bool {
        !mask & value
    }

    #[inline(always)]
    fn zeroupper_z<Z: ZeroUpper>(value: bool) -> bool {
        if const { Z::N >= 1 } { value } else { false }
    }
}

#[rustfmt::skip]
impl BitwiseRegister for bool {
    #[inline(always)] fn bitxor(lhs: bool, rhs: bool) -> bool { lhs ^ rhs }
    #[inline(always)] fn bitand(lhs: bool, rhs: bool) -> bool { lhs & rhs }
    #[inline(always)] fn bitor (lhs: bool, rhs: bool) -> bool { lhs | rhs }
    #[inline(always)] fn not(value: bool) -> bool { !value }
}

#[rustfmt::skip]
impl InterleaveRegister for bool {
    #[inline(always)] fn interleave  (a: bool, b: bool) -> (bool, bool) { (a, b) }
    #[inline(always)] fn deinterleave(a: bool, b: bool) -> (bool, bool) { (a, b) }
}

#[rustfmt::skip]
impl MaskRegister for bool {
    const TRUTHY: bool = true;
    const FALSY:  bool = false;

    #[inline(always)] fn set(_mask: bool, _lane: usize, value: bool) -> bool { value }
    #[inline(always)] fn test(mask: bool, _lane: usize) -> bool { mask }
    #[inline(always)] fn new_mask(value: GenericArray<bool, Self::Lanes>) -> bool { value[0] }
    #[inline(always)] fn all(value: bool) -> bool { value }
    #[inline(always)] fn any(value: bool) -> bool { value }
    #[inline(always)] fn native_bitmask(value: bool) -> Option<u64> { Some(value as u64) }

    #[cfg(feature = "bitvec")]
    #[inline(always)]
    fn fill_bitmask(value: bool, view: &mut bitvec::slice::BitSlice<u32>) {
        view.set(0, value);
    }
}

#[rustfmt::skip]
impl CastMaskRegister<bool> for bool {
    #[inline(always)] fn mask_from(value: bool) -> bool { value }
}
