use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, ConcatRegister, CoreRegister, ExtendRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper,
    },
    swizzle::SwizzleIndices,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x2Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for I64x2Wasm {
    type Lanes = typenum::U2;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::i64x2(0, 0);

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
impl BitwiseRegister for I64x2Wasm {
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
impl InterleaveRegister for I64x2Wasm {
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
impl BitshiftRegister for I64x2Wasm {
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
impl MaskRegister for I64x2Wasm {
    const FALSY: Storage<Self> = arch::i64x2(0, 0);
    const TRUTHY: Storage<Self> = arch::i64x2(-1, -1);

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
        arch::i64x2_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i64x2x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i64x2_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for I64x2Wasm {
    type Element = i64;
    type Signed = super::I64x2Wasm;
    type Unsigned = super::U64x2Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // Arithmetic shift right by 63 propagates sign bit to all bits
        arch::i64x2_shr(value, 63)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::i64x2(value[0], value[1])
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::i64x2_splat(value)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::i64x2(value, 0)
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
        arch::i64x2_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::i64x2_replace_lane::<I>(value, element)
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
        // Mask the indices to the 2 in-register lanes so an out-of-range index wraps instead of
        // falling through to `unreachable!()` (which is UB in release).
        match (I::INDICES[0] & 1, I::INDICES[1] & 1) {
            (0, 0) => arch::i64x2_shuffle::<0, 0>(value, value),
            (0, 1) => arch::i64x2_shuffle::<0, 1>(value, value),
            (1, 0) => arch::i64x2_shuffle::<1, 0>(value, value),
            (1, 1) => arch::i64x2_shuffle::<1, 1>(value, value),
            _ => unreachable!(),
        }
    }

    #[rustfmt::skip]
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        // Mask to the 4 source lanes (2 from `a`, 2 from `b`) so out-of-range indices wrap.
        match (I::INDICES[0] & 3, I::INDICES[1] & 3) {
            (0, 0) => arch::i64x2_shuffle::<0, 0>(a, b),
            (0, 1) => arch::i64x2_shuffle::<0, 1>(a, b),
            (0, 2) => arch::i64x2_shuffle::<0, 2>(a, b),
            (0, 3) => arch::i64x2_shuffle::<0, 3>(a, b),
            (1, 0) => arch::i64x2_shuffle::<1, 0>(a, b),
            (1, 1) => arch::i64x2_shuffle::<1, 1>(a, b),
            (1, 2) => arch::i64x2_shuffle::<1, 2>(a, b),
            (1, 3) => arch::i64x2_shuffle::<1, 3>(a, b),
            (2, 0) => arch::i64x2_shuffle::<2, 0>(a, b),
            (2, 1) => arch::i64x2_shuffle::<2, 1>(a, b),
            (2, 2) => arch::i64x2_shuffle::<2, 2>(a, b),
            (2, 3) => arch::i64x2_shuffle::<2, 3>(a, b),
            (3, 0) => arch::i64x2_shuffle::<3, 0>(a, b),
            (3, 1) => arch::i64x2_shuffle::<3, 1>(a, b),
            (3, 2) => arch::i64x2_shuffle::<3, 2>(a, b),
            (3, 3) => arch::i64x2_shuffle::<3, 3>(a, b),
            _ => unreachable!(),
        }
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl ShuffleRegister for I64x2Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for I64x2Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for I64x2Wasm {
 fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_ge(lhs, rhs) }
 fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_lt(lhs, rhs) }
 fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_le(lhs, rhs) }
 fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_ne(lhs, rhs) }
 fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_gt(lhs, rhs) }
 fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::i64x2_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for I64x2Wasm {
    sort_via_network!(2);

    const ZERO: Storage<Self> = arch::i64x2(0, 0);
    const ONE: Storage<Self> = arch::i64x2(1, 1);
    const TWO: Storage<Self> = arch::i64x2(2, 2);

    const MIN: Storage<Self> = arch::i64x2(i64::MIN, i64::MIN);
    const MAX: Storage<Self> = arch::i64x2(i64::MAX, i64::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_min i64x2_min)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_max i64x2_max)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_add i64x2_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_mul i64x2_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i64x2_shuffle::<0, 2>(lo, hi); // [a0, b0]
        let odd = arch::i64x2_shuffle::<1, 3>(lo, hi); // [a1, b1]
        arch::i64x2_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::i64x2_splat(<Self::Lanes as Unsigned>::USIZE as i64)
    }

    fn indexed() -> Storage<Self> {
        arch::i64x2(0, 1)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_min(lhs, rhs)
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I64x2Wasm {
    const NEG_ONE: Storage<Self> = arch::i64x2(-1, -1);
    const MIN_POSITIVE: Storage<Self> = arch::i64x2(1, 1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::i64x2_neg(value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        // Arithmetic shift right by 63 to propagate the sign bit
        arch::i64x2_shr(value, 63)
    }

    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::i64x2_ge(value, Self::ZERO)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::i64x2_abs(value)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I64x2Wasm {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        let a0 = arch::i64x2_extract_lane::<0>(lhs);
        let a1 = arch::i64x2_extract_lane::<1>(lhs);
        let b0 = arch::i64x2_extract_lane::<0>(rhs);
        let b1 = arch::i64x2_extract_lane::<1>(rhs);

        arch::u64x2(
            (((a0 as i128) * (b0 as i128)) >> 64) as u64,
            (((a1 as i128) * (b1 as i128)) >> 64) as u64,
        )
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_mul(lhs, rhs)
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_saturating_add(lhs, rhs)
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i64x2_saturating_sub(lhs, rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_add i64x2_add)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(i value; i64x2_mul i64x2_mul)
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
        // use unsigned version's popcnt implementation
        super::U64x2Wasm::count_ones(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.leading_zeros() as i64)
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.trailing_zeros() as i64)
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I64x2Wasm {
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::i64x2_shr(value, shift) // Arithmetic shift right
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<i64> for I64x2Wasm {
    fn concat(lo: Storage<i64>, hi: Storage<i64>) -> Storage<Self> {
        arch::i64x2(lo, hi)
    }

    fn split(value: Storage<Self>) -> (Storage<i64>, Storage<i64>) {
        (
            arch::i64x2_extract_lane::<0>(value),
            arch::i64x2_extract_lane::<1>(value),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<i64> for I64x2Wasm {
    fn extend(value: Storage<i64>) -> Storage<Self> {
        arch::i64x2(value, 0)
    }

    fn narrow(value: Storage<Self>) -> Storage<i64> {
        arch::i64x2_extract_lane::<0>(value)
    }
}
