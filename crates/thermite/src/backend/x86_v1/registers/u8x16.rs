//! Native 128-bit unsigned 8-bit register for x86-v1 (SSE2). See [`super::i8x16`] for the
//! general approach; unsigned min/max are native SSE2 here, unsigned compare uses a polyfill,
//! and the awkward ops fall back to scalar.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, Element, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SaturatingCastRegister, Storage, UnsignedIntegerRegister, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U8x16V1;

#[thermite_macros::inline_always]
impl CoreRegister for U8x16V1 {
    type Lanes = typenum::U16;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::X86V1;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8x_v1(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }

    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I8x16V1::zeroupper_z::<Z>(value) // same mask layout
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U8x16V1 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I8x16V1::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I8x16V1::deinterleave(a, b)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U8x16V1 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 16>([0; 16]);
    const TRUTHY: Storage<Self> = reg::<Self, 16>([!0; 16]);

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi8x_v1(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { (arch::_mm_movemask_epi8(value) as u32) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U8x16V1 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u8> for U8x16V1 {
    fn extend(value: Storage<u8>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi8(value as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<u8> {
        Self::as_slice(&value)[0]
    }
}

#[thermite_macros::inline_always]
impl Register for U8x16V1 {
    type Element = u8;

    type Signed = super::I8x16V1;
    type Unsigned = super::U8x16V1;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epi8(arch::_mm_setzero_si128(), value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    impl_native_extract!(@epi8x16_v1);

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi8(value as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi8(value as i8) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        let mut arr = [0u8; 16];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        arr.reverse();
        unsafe { arch::_mm_loadu_si128(arr.as_ptr() as *const _) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    const HAS_PERMUTEV: bool = false;

    impl_byteshift_align!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U8x16V1 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi8x_v1(value, shift) }
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi8x_v1(value, shift) }
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi8x_v1::<IMM8>(value) }
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi8x_v1::<IMM8>(value) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U8x16V1 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu8x_v1(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu8x_v1(rhs, lhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U8x16V1 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([u8::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([u8::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().min().unwrap()
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().max().unwrap()
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(0u8, u8::wrapping_add)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1u8, u8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u8))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi8(lhs, rhs) }
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi8(lhs, rhs) }
    }
    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi8(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi8(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v1(lhs, rhs) }
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epu8(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U8x16V1 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epu8x_v1(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v1(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epu8(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epu8(lhs, rhs) }
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi8x_v1(value) }
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(8), Self::ilog2p1(value))
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I8x16V1::trailing_zeros(value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U8x16V1 {}

// Saturating narrow u16x16 -> u8x16: clamp each half (`min_epu16x_v1`) then two-source `packuswb`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U16x8V1, 2>> for U8x16V1 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U16x8V1, 2>>) -> Storage<Self> {
        unsafe {
            let max = arch::_mm_set1_epi16(0xFF);
            let lo = arch::_mm_min_epu16x_v1(value.0[0], max);
            let hi = arch::_mm_min_epu16x_v1(value.0[1], max);
            arch::_mm_packus_epi16(lo, hi)
        }
    }
}

// SSE2 has no `packusdw`, so u32x16 -> u8x16 and u64x16 -> u8x16 clamp into range + truncating narrow.
macro_rules! sat_clamp_narrow {
    ($(($from:ty, $fe:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl SaturatingCastRegister<$from> for U8x16V1 {
            fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                let hi = <$from as Register>::splat(u8::MAX as $fe);
                let clamped = <$from as NumericRegister>::min(value, hi);
                <Self as CastRegister<$from>>::cast_from(clamped)
            }
        }
    )*};
}
sat_clamp_narrow! {
    (ArrayRegister<super::U32x4V1, 4>, u32),
    (ArrayRegister<super::U64x2V1, 8>, u64),
}
