//! Native 128-bit unsigned 16-bit register for the WASM SIMD128 backend. See [`super::i16x8`]
//! for the general approach; unsigned compares/min/max/saturation are native, the rest mirror
//! the signed register (shared bit patterns) or fall back to scalar.

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
        SaturatingCastRegister, Storage, UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister, empty_reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x8Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for U16x8Wasm {
    type Lanes = typenum::U8;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = arch::ISA;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = empty_reg::<Self>();

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            let mut arr = [0u16; 8];
            unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 8 - Z::N };
            while i < 8 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::v128_load(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U16x8Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x8Wasm::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I16x8Wasm::deinterleave(a, b)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U16x8Wasm {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = arch::u16x8(0, 0, 0, 0, 0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::u16x8(!0, !0, !0, !0, !0, !0, !0, !0);

    fn all(value: Storage<Self>) -> bool {
        arch::i16x8_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn none(value: Storage<Self>) -> bool {
        !arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i16x8x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i16x8_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i16x8_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U16x8Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs)
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u16> for U16x8Wasm {
    fn extend(value: Storage<u16>) -> Storage<Self> {
        arch::u16x8(value, 0, 0, 0, 0, 0, 0, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<u16> {
        arch::u16x8_extract_lane::<0>(value)
    }
}

#[thermite_macros::inline_always]
impl Register for U16x8Wasm {
    type Element = u16;
    type Signed = super::I16x8Wasm;
    type Unsigned = super::U16x8Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i16x8_shr(value, 15)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::v128_load(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::u16x8(value, 0, 0, 0, 0, 0, 0, 0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u16x8_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        super::I16x8Wasm::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        super::I16x8Wasm::swap_bytes(value)
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        super::I16x8Wasm::permutev(value, idxs)
    }

    compress_via_table!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U16x8Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i16x8_shl(value, shift)
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u16x8_shr(value, shift)
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i16x8_shl(value, IMM8 as u32)
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u16x8_shr(value, IMM8 as u32)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for U16x8Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u16x8_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u16x8_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u16x8_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u16x8_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i16x8_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for U16x8Wasm {
    sort_via_network!(8);

    const ZERO: Storage<Self> = arch::u16x8(0, 0, 0, 0, 0, 0, 0, 0);
    const ONE: Storage<Self> = arch::u16x8(1, 1, 1, 1, 1, 1, 1, 1);
    const TWO: Storage<Self> = arch::u16x8(2, 2, 2, 2, 2, 2, 2, 2);

    const MIN: Storage<Self> = arch::u16x8(
        u16::MIN,
        u16::MIN,
        u16::MIN,
        u16::MIN,
        u16::MIN,
        u16::MIN,
        u16::MIN,
        u16::MIN,
    );
    const MAX: Storage<Self> = arch::u16x8(
        u16::MAX,
        u16::MAX,
        u16::MAX,
        u16::MAX,
        u16::MAX,
        u16::MAX,
        u16::MAX,
        u16::MAX,
    );

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
        arch::u16x8_splat(<Self::Lanes as Unsigned>::USIZE as u16)
    }

    fn indexed() -> Storage<Self> {
        arch::u16x8(0, 1, 2, 3, 4, 5, 6, 7)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_add(lhs, rhs)
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_sub(lhs, rhs)
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_mul(lhs, rhs)
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u16x8_min(lhs, rhs)
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u16x8_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U16x8Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| ((a as u32 * b as u32) >> 16) as u16)
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i16x8_mul(lhs, rhs)
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u16x8_add_sat(lhs, rhs)
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u16x8_sub_sat(lhs, rhs)
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
        arch::i16x8_extadd_pairwise_u8x16(arch::i8x16_popcnt(value))
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_zeros() as u16)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_zeros() as u16)
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_ones() as u16)
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_ones() as u16)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U16x8Wasm {
    /// 2D Morton via an `i8x16.swizzle` nibble-LUT; every other `N` uses the cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            arch::wasm_morton2_epu16x(values[0], values[1])
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    /// 2D Morton decode via the swizzle compress; every other `N` uses the cascade.
    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                arch::wasm_morton2_compress_epu16x(code),
                arch::wasm_morton2_compress_epu16x(arch::u16x8_shr(code, 1)),
            )
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }
}

// Widen u16x8 -> u32x8 (= ArrayRegister<U32x4Wasm, 2>): zero-extend the low/high 4 lanes.
#[thermite_macros::inline_always]
impl CastRegister<U16x8Wasm> for ArrayRegister<super::U32x4Wasm, 2> {
    fn cast_from(value: Storage<U16x8Wasm>) -> Storage<Self> {
        let lo = arch::i32x4_extend_low_u16x8(value);
        let hi = arch::i32x4_extend_high_u16x8(value);
        ArrayRegister([lo, hi])
    }
}

// Narrow u32x8 -> u16x8: truncate the low 16 bits of each 32-bit lane (wrapping, like `as`).
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4Wasm, 2>> for U16x8Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 2>>) -> Storage<Self> {
        arch::i8x16_shuffle::<
            0, 1, 4, 5, 8, 9, 12, 13,
            16, 17, 20, 21, 24, 25, 28, 29,
        >(value.0[0], value.0[1])
    }
}

// Saturating narrow u32x8 -> u16x8: clamp each half (`u32x4.min`) then two-source `u16x8.narrow_i32x4_u`.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U32x4Wasm, 2>> for U16x8Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 2>>) -> Storage<Self> {
        let max = arch::u32x4_splat(0xFFFF);
        let lo = arch::u32x4_min(value.0[0], max);
        let hi = arch::u32x4_min(value.0[1], max);
        arch::u16x8_narrow_i32x4(lo, hi)
    }
}

// Saturating narrow u64x8 -> u16x8: clamp the high end + truncating narrow.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Wasm, 4>> for U16x8Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 4>>) -> Storage<Self> {
        type Src = ArrayRegister<super::U64x2Wasm, 4>;
        let hi = <Src as Register>::splat(u16::MAX as u64);
        let clamped = <Src as NumericRegister>::min(value, hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }
}
