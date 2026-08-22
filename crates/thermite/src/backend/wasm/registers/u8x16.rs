//! Native 128-bit unsigned 8-bit register for the WASM SIMD128 backend. See [`super::i8x16`]
//! for the general approach; unsigned compares/min/max/saturation are native.

use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register, Storage,
        UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister, empty_reg,
    },
};

use super::arch;
use super::half8::store_qwords;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U8x16Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for U8x16Wasm {
    type Lanes = typenum::U16;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = empty_reg::<Self>();

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 16 } {
            value
        } else {
            let mut arr = [0u8; 16];
            unsafe { arch::v128_store(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 16 - Z::N };
            while i < 16 {
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
impl InterleaveRegister for U8x16Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I8x16Wasm::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I8x16Wasm::deinterleave(a, b)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U8x16Wasm {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = arch::u8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::u8x16(!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0);

    fn all(value: Storage<Self>) -> bool {
        arch::i8x16_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn none(value: Storage<Self>) -> bool {
        !arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i8x16x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i8x16_bitmask(value) as u16 as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i8x16_bitmask(value) as u16 as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U8x16Wasm {
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

#[rustfmt::skip] #[thermite_macros::inline_always]
impl ExtendRegister<u8> for U8x16Wasm {
    fn extend(value: Storage<u8>) -> Storage<Self> {
        arch::u8x16(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<u8> {
        arch::u8x16_extract_lane::<0>(value)
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl Register for U8x16Wasm {
    type Element = u8;
    type Signed = super::I8x16Wasm;
    type Unsigned = super::U8x16Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i8x16_shr(value, 7)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::v128_load(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::u8x16(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u8x16_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        super::I8x16Wasm::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        super::I8x16Wasm::permutev(value, idxs)
    }

    compress_via_wide!();
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for U8x16Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i8x16_shl(value, shift)
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u8x16_shr(value, shift)
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::i8x16_shl(value, IMM8 as u32)
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_shr(value, IMM8 as u32)
    }
    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        arch::wasm_shlv_i8x16(value, shifts)
    }
    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        arch::wasm_shrv_u8x16(value, shifts) // logical
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for U8x16Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u8x16_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u8x16_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u8x16_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u8x16_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i8x16_eq(lhs, rhs) }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl NumericRegister for U8x16Wasm {
    const ZERO: Storage<Self> = arch::u8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const ONE: Storage<Self> = arch::u8x16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    const TWO: Storage<Self> = arch::u8x16(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);

    const MIN: Storage<Self> = arch::u8x16(u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN);
    const MAX: Storage<Self> = arch::u8x16(u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX);

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
        arch::u8x16_splat(<Self::Lanes as Unsigned>::USIZE as u8)
    }

    fn indexed() -> Storage<Self> {
        arch::u8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_add(lhs, rhs)
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i8x16_sub(lhs, rhs)
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, u8::wrapping_mul)
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_min(lhs, rhs)
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U8x16Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| ((a as u16 * b as u16) >> 8) as u8)
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, u8::wrapping_mul)
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_add_sat(lhs, rhs)
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_sub_sat(lhs, rhs)
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
        arch::i8x16_popcnt(value)
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_zeros() as u8)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_zeros() as u8)
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.leading_ones() as u8)
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::map(value, |x| x.trailing_ones() as u8)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U8x16Wasm {}

// Saturating narrow u16x16 -> u8x16: clamp each half (`u16x8.min`) then two-source `u8x16.narrow_i16x8_u`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U16x8Wasm, 2>> for U8x16Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U16x8Wasm, 2>>) -> Storage<Self> {
        // Clamped first, so the low half of each lane is already the saturated
        // result and the truncating narrow is exact. See `half16.rs`'s
        // `u32x4 -> u16x4` for why this avoids the saturating narrow.
        let max = arch::u16x8_splat(0xFF);

        <Self as CastRegister<ArrayRegister<super::U16x8Wasm, 2>>>::cast_from(ArrayRegister([
            arch::u16x8_min(value.0[0], max),
            arch::u16x8_min(value.0[1], max),
        ]))
    }

    fn cast_from(value: Storage<ArrayRegister<super::U16x8Wasm, 2>>) -> Storage<Self> {
        unsafe { arch::narrow_2xi16x8_to_bytes(value.0) }
    }
}

// Saturating narrow u32x16 -> u8x16: clamp + `u16x8.narrow_i32x4_u` to u16, then clamp + `u8x16.narrow_i16x8_u`.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U32x4Wasm, 4>> for U8x16Wasm {
    // Clamp straight into `[0, 0xFF]` in the 32-bit domain, then gather the low
    // byte of each lane. `cast_from` is not reusable here - its
    // `narrow_4xi32x4_to_bytes` is a scalar store-and-rebuild - so the byte
    // gather is spelled out. Two 8-byte halves, then a splice.
    //
    // The previous form stepped down through `u16x8.narrow_i32x4_u` and
    // `u8x16.narrow_i16x8_u`; see `half16.rs`'s `u32x4 -> u16x4` for why those
    // are avoided. Clamping once at the destination range rather than twice also
    // makes this two instructions shorter.
    #[rustfmt::skip]
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 4>>) -> Storage<Self> {
        let max = arch::u32x4_splat(0xFF);
        let a = arch::u32x4_min(value.0[0], max);
        let b = arch::u32x4_min(value.0[1], max);
        let c = arch::u32x4_min(value.0[2], max);
        let d = arch::u32x4_min(value.0[3], max);

        let lo = arch::i8x16_shuffle::<
            0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0,
        >(a, b);
        let hi = arch::i8x16_shuffle::<
            0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0,
        >(c, d);

        arch::i8x16_shuffle::<
            0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
        >(lo, hi)
    }

    fn cast_from(value: Storage<ArrayRegister<super::U32x4Wasm, 4>>) -> Storage<Self> {
        unsafe { arch::narrow_4xi32x4_to_bytes(value.0) }
    }
}

// Saturating narrow u64x16 -> u8x16: clamp the high end + truncating narrow.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 8>> for U8x16Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 8>>) -> Storage<Self> {
        type Src = ArrayRegister<super::U64x2Wasm, 8>;
        let hi = <Src as Register>::splat(u8::MAX as u64);
        let clamped = <Src as NumericRegister>::min(value, hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }

    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 8>>) -> Storage<Self> {
        let v = value.0;
        let a = store_qwords(v[0]);
        let b = store_qwords(v[1]);
        let c = store_qwords(v[2]);
        let d = store_qwords(v[3]);
        let e = store_qwords(v[4]);
        let f = store_qwords(v[5]);
        let g = store_qwords(v[6]);
        let h = store_qwords(v[7]);
        arch::i8x16(
            a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8, e[0] as i8,
            e[1] as i8, f[0] as i8, f[1] as i8, g[0] as i8, g[1] as i8, h[0] as i8, h[1] as i8,
        )
    }
}
