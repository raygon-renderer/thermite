//! Native 128-bit unsigned 16-bit register for x86-v1 (SSE2). See `i16x8.rs` for the overall
//! approach (native SSE2 where possible, scalar fallbacks for the awkward ops).

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register,
        SaturatingCastRegister, Storage, UnsignedIntegerRegister, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x8V1;

#[thermite_macros::inline_always]
impl CoreRegister for U16x8V1 {
    type Lanes = typenum::U8;
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
        super::I16x8V1::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U16x8V1 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x8V1::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x8V1::deinterleave(a, b)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U16x8V1 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 8>([0; 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([!0; 8]);

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    fn count_set<const N: usize>(values: [Storage<Self>; N]) -> usize {
        unsafe { arch::_mm_count_mask_epi16x_v1(values) }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi16x_v1(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        let packed = unsafe { arch::_mm_packs_epi16(value, arch::_mm_setzero_si128()) };
        Some(unsafe { (arch::_mm_movemask_epi8(packed) as u64) & 0xFF })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U16x8V1 {
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
impl ExtendRegister<u16> for U16x8V1 {
    fn extend(value: Storage<u16>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi16(value as i16, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<u16> {
        unsafe { arch::_mm_extract_epi16::<0>(value) as u16 }
    }
}

#[thermite_macros::inline_always]
impl Register for U16x8V1 {
    type Element = u16;

    type Signed = super::I16x8V1;
    type Unsigned = super::U16x8V1;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi16(value, 15) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    impl_native_extract!(@epi16x8);

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi16(value as i16, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi16(value as i16) }
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
        super::I16x8V1::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi16x_v1(value) }
    }

    const HAS_PERMUTEV: bool = false;

    impl_byteshift_align!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U16x8V1 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi16(value, IMM8) }
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi16(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U16x8V1 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu16x_v1(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu16x_v1(rhs, lhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U16x8V1 {
    sort_via_network!(8);

    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([u16::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([u16::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().min().unwrap()
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().max().unwrap()
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(0u16, u16::wrapping_add)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1u16, u16::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u16))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi16(lhs, rhs) }
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi16(lhs, rhs) }
    }
    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi16(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi16(lhs, arch::_mm_and_si128(rhs, mask)) }
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi16(lhs, rhs) }
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epu16x_v1(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu16x_v1(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U16x8V1 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epu16(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi16(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epu16(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epu16(lhs, rhs) }
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
        unsafe { arch::_mm_popcnt_epi16x_v1(value) }
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(16), Self::ilog2p1(value))
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I16x8V1::trailing_zeros(value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U16x8V1 {}

// Widen u16x8 -> u32x8 (= ArrayRegister<U32x4V1, 2>): zero-extend via unpack.
#[thermite_macros::inline_always]
impl CastRegister<U16x8V1> for ArrayRegister<super::U32x4V1, 2> {
    fn cast_from(value: Storage<U16x8V1>) -> Storage<Self> {
        unsafe {
            let zero = arch::_mm_setzero_si128();
            let lo = arch::_mm_unpacklo_epi16(value, zero);
            let hi = arch::_mm_unpackhi_epi16(value, zero);
            ArrayRegister([lo, hi])
        }
    }
}

// Narrow u32x8 -> u16x8: truncate each lane (wrapping, like `as`). Scalar.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4V1, 2>> for U16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::U32x4V1, 2>>) -> Storage<Self> {
        let mut lanes = [0u32; 8];
        unsafe {
            arch::_mm_storeu_si128(lanes.as_mut_ptr() as *mut _, value.0[0]);
            arch::_mm_storeu_si128(lanes.as_mut_ptr().add(4) as *mut _, value.0[1]);
        }
        let words: [u16; 8] = core::array::from_fn(|i| lanes[i] as u16);
        unsafe { arch::_mm_loadu_si128(words.as_ptr() as *const _) }
    }
}

// SSE2 has no `packusdw`, so u32x8 -> u16x8 and u64x8 -> u16x8 clamp into range (the register
// `min` uses the v1 unsigned-min polyfills) then reuse the truncating narrow.
macro_rules! sat_clamp_narrow {
    ($(($from:ty, $fe:ty, $ie:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl SaturatingCastRegister<$from> for U16x8V1 {
            fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
                let clamped = <$from as NumericRegister>::min(value, hi);
                <Self as CastRegister<$from>>::cast_from(clamped)
            }
        }
    )*};
}
sat_clamp_narrow! {
    (ArrayRegister<super::U32x4V1, 2>, u32, u16),
    (ArrayRegister<super::U64x2V1, 4>, u64, u16),
}
