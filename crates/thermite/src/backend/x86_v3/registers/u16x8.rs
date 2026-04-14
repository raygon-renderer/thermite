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
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, Storage, SwizzleRegister,
        UnsignedIntegerRegister, dp::DoublePumpRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x8V3;

impl CoreRegister for U16x8V3 {
    type Lanes = typenum::U8;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::X86V3;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8(on_false, on_true, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }
}

impl MaskRegister for U16x8V3 {
    const FALSY: Storage<Self> = reg::<Self, 8>([0; 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([!0; 8]);

    #[inline(always)]
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    #[inline(always)]
    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        // unsafe { arch::_mm_cvtboolx4_to_epi32_mask_v2(value) }
        todo!()
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u64 })
    }

    #[inline(always)]
    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }

    #[inline(always)]
    fn interleave_mask(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        Self::interleave(a, b)
    }

    #[inline(always)]
    fn deinterleave_mask(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        Self::deinterleave(a, b)
    }
}

#[thermite_macros::bitand_z]
impl BitwiseRegister for U16x8V3 {
    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }
}

impl Register for U16x8V3 {
    type HalfRegister = <Scalar as Simd>::u16x4;
    type DoubleRegister = super::U16x16V3;

    type Element = u16;

    type Unsigned = super::U16x8V3;

    #[inline(always)]
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    #[inline(always)]
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    #[inline(always)]
    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }

    #[inline(always)]
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi32(value, 15) }
    }

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epu32x(value, 0, 0, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi16(value as i16) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        // unsafe {
        //     let mut arr = [0; 8];
        //     Self::store_unaligned(arr.as_mut_ptr(), value);
        //     (DoublePumpRegister(arr[0], arr[1]), DoublePumpRegister(arr[2], arr[3]))
        // }

        todo!()
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        // unsafe { arch::_mm_setr_epu32x(lo.0, lo.1, hi.0, hi.1) }

        todo!()
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        // unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }

        todo!()
    }

    const HAS_SIMPLE_INTERLEAVE: bool = super::I16x8V3::HAS_SIMPLE_INTERLEAVE;

    #[inline(always)]
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x8V3::interleave(a, b) // reuse signed implementation
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // unsafe { arch::_mm_bswap_epi32x_v2(value) }

        todo!()
    }
}
