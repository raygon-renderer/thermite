//! Native 128-bit signed 16-bit register for x86-v1 (SSE2). Most 16-bit ops are native SSE2;
//! the few that aren't (unsigned min/max, popcount, byteswap) use SSE2 polyfills, and the
//! genuinely awkward ones (reductions, signum, leading-zeros, narrow cast, reverse,
//! deinterleave) fall back to scalar - x86-v1 is provided for completeness, not speed.

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
        SaturatingCastRegister, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister,
        empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I16x8V1;

#[thermite_macros::inline_always]
impl CoreRegister for I16x8V1 {
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

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            let mut arr = [0i16; 8];
            unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 8 - Z::N };
            while i < 8 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::_mm_loadu_si128(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I16x8V1 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi16(a, b), arch::_mm_unpackhi_epi16(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // SSE2 has no `pshufb`; gather even/odd 16-bit lanes scalar-wise.
        let mut ax = [0i16; 8];
        let mut bx = [0i16; 8];
        unsafe {
            arch::_mm_storeu_si128(ax.as_mut_ptr() as *mut _, a);
            arch::_mm_storeu_si128(bx.as_mut_ptr() as *mut _, b);
        }
        let cat: [i16; 16] = core::array::from_fn(|i| if i < 8 { ax[i] } else { bx[i - 8] });
        let evens: [i16; 8] = core::array::from_fn(|i| cat[2 * i]);
        let odds: [i16; 8] = core::array::from_fn(|i| cat[2 * i + 1]);
        unsafe {
            (
                arch::_mm_loadu_si128(evens.as_ptr() as *const _),
                arch::_mm_loadu_si128(odds.as_ptr() as *const _),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I16x8V1 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 8>([0; 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([-1; 8]);

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
impl BitwiseRegister for I16x8V1 {
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
impl ExtendRegister<i16> for I16x8V1 {
    fn extend(value: Storage<i16>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi16(value, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i16> {
        unsafe { arch::_mm_extract_epi16::<0>(value) as i16 }
    }
}

#[thermite_macros::inline_always]
impl Register for I16x8V1 {
    type Element = i16;

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
        unsafe { arch::_mm_setr_epi16(value, 0, 0, 0, 0, 0, 0, 0) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi16(value) }
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
        // no `movntdqa` before SSE4.1; a plain aligned load is the best SSE2 can do
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        // no `pshufb` on SSE2; reverse scalar-wise.
        let mut arr = [0i16; 8];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        arr.reverse();
        unsafe { arch::_mm_loadu_si128(arr.as_ptr() as *const _) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi16x_v1(value) }
    }

    // no `pshufb` on SSE2, so variable permutes fall back to the scalar defaults
    const HAS_PERMUTEV: bool = false;

    impl_byteshift_align!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I16x8V1 {
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
impl PartialOrdRegister for I16x8V1 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epi16(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I16x8V1 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([i16::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([i16::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().min().unwrap()
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().max().unwrap()
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(0i16, i16::wrapping_add)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1i16, i16::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i16))
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
        unsafe { arch::_mm_min_epi16(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi16(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I16x8V1 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1; 8]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi16(arch::_mm_setzero_si128(), value) }
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi16(value, 15) }
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        // max(v, -v); abs(MIN) wraps to MIN, matching scalar `wrapping_abs`.
        unsafe { arch::_mm_max_epi16(value, arch::_mm_sub_epi16(arch::_mm_setzero_si128(), value)) }
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.signum())
    }
    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<15>(mask))
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I16x8V1 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epi16(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi16(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi16(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi16(lhs, rhs) }
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi16x_v1(value) }
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        super::U16x8V1::leading_zeros(value)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I16x8V1 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi16(value, IMM8) }
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }
}

// Widen i16x8 -> i32x8 (= ArrayRegister<I32x4V1, 2>): sign-extend via unpack (no SSE4.1 cvt).
#[thermite_macros::inline_always]
impl CastRegister<I16x8V1> for ArrayRegister<super::I32x4V1, 2> {
    fn cast_from(value: Storage<I16x8V1>) -> Storage<Self> {
        unsafe {
            let sign = arch::_mm_srai_epi16(value, 15); // high 16 bits = sign extension
            let lo = arch::_mm_unpacklo_epi16(value, sign);
            let hi = arch::_mm_unpackhi_epi16(value, sign);
            ArrayRegister([lo, hi])
        }
    }
}

// Narrow i32x8 -> i16x8: truncate each lane (wrapping, like `as`). Scalar - SSE2 packs saturate.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I32x4V1, 2>> for I16x8V1 {
    fn cast_from(value: Storage<ArrayRegister<super::I32x4V1, 2>>) -> Storage<Self> {
        let mut lanes = [0i32; 8];
        unsafe {
            arch::_mm_storeu_si128(lanes.as_mut_ptr() as *mut _, value.0[0]);
            arch::_mm_storeu_si128(lanes.as_mut_ptr().add(4) as *mut _, value.0[1]);
        }
        let words: [i16; 8] = core::array::from_fn(|i| lanes[i] as i16);
        unsafe { arch::_mm_loadu_si128(words.as_ptr() as *const _) }
    }
}

// Saturating narrow i32x8 -> i16x8 via a single two-source SSE2 `packssdw` (no lane crossing).
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I32x4V1, 2>> for I16x8V1 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I32x4V1, 2>>) -> Storage<Self> {
        unsafe { arch::_mm_packs_epi32(value.0[0], value.0[1]) }
    }
}

// Saturating narrow i64x8 -> i16x8: no SSE 64-bit pack, so clamp + truncating narrow.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::I64x2V1, 4>> for I16x8V1 {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x2V1, 4>>) -> Storage<Self> {
        type Src = ArrayRegister<super::I64x2V1, 4>;
        let lo = <Src as Register>::splat(i16::MIN as i64);
        let hi = <Src as Register>::splat(i16::MAX as i64);
        let clamped = <Src as NumericRegister>::min(<Src as NumericRegister>::max(value, lo), hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }
}
