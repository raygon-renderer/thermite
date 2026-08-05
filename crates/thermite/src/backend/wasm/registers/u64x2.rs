use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, ConcatRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, Storage, UnsignedIntegerRegister, ZeroUpper,
    },
    swizzle::SwizzleIndices,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for U64x2Wasm {
    type NativeIsa = crate::backend::wasm::Wasm;
    type Lanes = typenum::U2;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::u64x2(0, 0);

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else if const { Z::N == 1 } {
            arch::v128_and(value, arch::u64x2(!0, 0))
        } else {
            Self::EMPTY
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U64x2Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: WASM andnot has operands reversed vs. the trait
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U64x2Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i64x2_shuffle::<0, 2>(a, b);
        let high = arch::i64x2_shuffle::<1, 3>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i64x2_shuffle::<0, 2>(a, b);
        let odds = arch::i64x2_shuffle::<1, 3>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U64x2Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shr(value, shift) // Non-arithmetic
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u64x2_shl(value, shift)
    }

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U64x2Wasm {
    const FALSY: Storage<Self> = arch::u64x2(0, 0);
    const TRUTHY: Storage<Self> = arch::u64x2(!0, !0);

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx2_to_i64x2x(value)
    }

    fn all(value: Storage<Self>) -> bool {
        arch::u64x2_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i64x2x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::u64x2_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::u64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for U64x2Wasm {
    type Element = u64;
    type Signed = super::I64x2Wasm;
    type Unsigned = super::U64x2Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // Arithmetic shift right by 63 propagates the sign/MSB to all bits
        arch::i64x2_shr(value, 63)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::u64x2(value[0], value[1])
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u64x2_splat(value)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::u64x2(value, 0)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(1, 0))
    }

    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 64-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            7, 6, 5, 4, 3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8,
        ))
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::u64x2_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::u64x2_replace_lane::<I>(value, element)
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::wasm_lane_table_dyn::<2>(unsafe { core::mem::transmute(idxs) }),
        )
    }

    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        super::I64x2Wasm::permutev_const::<I>(value)
    }

    #[rustfmt::skip]
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        super::I64x2Wasm::swizzle_const::<I>(a, b)
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U64x2Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for U64x2Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl PartialOrdRegister for U64x2Wasm {
 fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_ne(lhs, rhs) }
 fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_eq(lhs, rhs) }
 fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_ge(lhs, rhs) }
 fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_lt(lhs, rhs) }
 fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_le(lhs, rhs) }
 fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u64x2_gt(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for U64x2Wasm {
    sort_via_network!(2);

    const ZERO: Storage<Self> = arch::u64x2(0, 0);
    const ONE: Storage<Self> = arch::u64x2(1, 1);
    const TWO: Storage<Self> = arch::u64x2(2, 2);

    const MIN: Storage<Self> = arch::u64x2(u64::MIN, u64::MIN);
    const MAX: Storage<Self> = arch::u64x2(u64::MAX, u64::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_min u64x2_min)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_max u64x2_max)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_add u64x2_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_mul u64x2_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i64x2_shuffle::<0, 2>(lo, hi); // [a0, b0]
        let odd = arch::i64x2_shuffle::<1, 3>(lo, hi); // [a1, b1]
        arch::u64x2_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::u64x2_splat(<Self::Lanes as Unsigned>::USIZE as u64)
    }

    fn indexed() -> Storage<Self> {
        arch::u64x2(0, 1)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_min(lhs, rhs)
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U64x2Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let a0 = arch::u64x2_extract_lane::<0>(lhs);
        let a1 = arch::u64x2_extract_lane::<1>(lhs);
        let b0 = arch::u64x2_extract_lane::<0>(rhs);
        let b1 = arch::u64x2_extract_lane::<1>(rhs);

        arch::u64x2(
            (((a0 as u128) * (b0 as u128)) >> 64) as u64,
            (((a1 as u128) * (b1 as u128)) >> 64) as u64,
        )
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_mul(lhs, rhs)
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_saturating_add(lhs, rhs)
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u64x2_saturating_sub(lhs, rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_add u64x2_add)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(u value; u64x2_mul u64x2_mul)
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
        // 1. Get 32-bit counts: [c0, c1, c2, c3]
        let counts_32 = super::U32x4Wasm::count_ones(value);

        // 2. We need [c0+c1, c2+c3].
        // Since max bit count for 64-bits is 64, this fits comfortably in i32
        // without overflow, so we stay in i32 domain for the add.

        // Move odd lanes to even positions: [c1, c1, c3, c3]
        let shifted = arch::i32x4_shuffle::<1, 1, 3, 3>(counts_32, counts_32);

        // Add: [c0+c1, ..., c2+c3, ...]
        let sums = arch::u32x4_add(counts_32, shifted);

        // 3. Arrange for 64-bit extension: [SumLo, SumHi, SumLo, SumHi]
        let arranged = arch::i32x4_shuffle::<0, 2, 0, 2>(sums, sums);

        // Extend to u64 (Zero extend is fine as count is positive)
        arch::u64x2_extend_low_u32x4(arranged)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.leading_zeros() as u64)
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.trailing_zeros() as u64)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U64x2Wasm {}

impl ConcatRegister<u64> for U64x2Wasm {
    fn concat(lo: Storage<u64>, hi: Storage<u64>) -> Storage<Self> {
        arch::u64x2(lo, hi)
    }

    fn split(value: Storage<Self>) -> (Storage<u64>, Storage<u64>) {
        (
            arch::u64x2_extract_lane::<0>(value),
            arch::u64x2_extract_lane::<1>(value),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<u64> for U64x2Wasm {
    fn extend(value: Storage<u64>) -> Storage<Self> {
        arch::u64x2(value, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<u64> {
        arch::u64x2_extract_lane::<0>(value)
    }
}

// Note: CastRegister<ArrayRegister<U64x2Wasm, 2>> for U32x4Wasm is defined in u32x4.rs
