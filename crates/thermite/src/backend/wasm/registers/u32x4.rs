use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IntegerRegister,
        InterleaveRegister, MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        SaturatingCastRegister, ShuffleRegister, Storage, UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister,
        empty_reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x4Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for U32x4Wasm {
    type Lanes = typenum::U4;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = arch::ISA;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::u32x4(0, 0, 0, 0);

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 3 } {
            arch::v128_and(value, arch::u32x4(!0, !0, !0, 0))
        } else if const { Z::N == 2 } {
            arch::v128_and(value, arch::u32x4(!0, !0, 0, 0))
        } else if const { Z::N == 1 } {
            arch::v128_and(value, arch::u32x4(!0, 0, 0, 0))
        } else {
            Self::EMPTY
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U32x4Wasm {
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
impl InterleaveRegister for U32x4Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i32x4_shuffle::<0, 2, 4, 6>(a, b);
        let odds = arch::i32x4_shuffle::<1, 3, 5, 7>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U32x4Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shr(value, shift) // Non-arithmetic
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shl(value, shift)
    }

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshli::<IMM8>(value)
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::wasm_bshri::<IMM8>(value)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U32x4Wasm {
    const FALSY: Storage<Self> = arch::u32x4(0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::u32x4(!0, !0, !0, !0);

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    fn all(value: Storage<Self>) -> bool {
        arch::u32x4_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i32x4x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::u32x4_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::u32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for U32x4Wasm {
    type Element = u32;
    type Signed = super::I32x4Wasm;
    type Unsigned = super::U32x4Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // MSB of u32 is bit 31; arithmetic shift treats it as a sign bit
        arch::i32x4_shr(value, 31)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::u32x4(value[0], value[1], value[2], value[3])
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u32x4_splat(value)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::u32x4(value, 0, 0, 0)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 32-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            3, 2, 1, 0,
            7, 6, 5, 4,
            11, 10, 9, 8,
            15, 14, 13, 12,
        ))
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::u32x4_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::u32x4_replace_lane::<I>(value, element)
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::wasm_lane_table_dyn::<4>(unsafe { core::mem::transmute(idxs) }),
        )
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U32x4Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for U32x4Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for U32x4Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for U32x4Wasm {
    const ZERO: Storage<Self> = arch::u32x4(0, 0, 0, 0);
    const ONE: Storage<Self> = arch::u32x4(1, 1, 1, 1);
    const TWO: Storage<Self> = arch::u32x4(2, 2, 2, 2);

    const MIN: Storage<Self> = arch::u32x4(u32::MIN, u32::MIN, u32::MIN, u32::MIN);
    const MAX: Storage<Self> = arch::u32x4(u32::MAX, u32::MAX, u32::MAX, u32::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_min u32x4_min)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_max u32x4_max)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_add u32x4_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_mul u32x4_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i32x4_shuffle::<0, 2, 4, 6>(lo, hi); // [a0,a2,b0,b2]
        let odd = arch::i32x4_shuffle::<1, 3, 5, 7>(lo, hi); // [a1,a3,b1,b3]
        arch::u32x4_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::u32x4_splat(<Self::Lanes as Unsigned>::USIZE as u32)
    }

    fn indexed() -> Storage<Self> {
        arch::u32x4(0, 1, 2, 3)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_min(lhs, rhs)
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_max(lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U32x4Wasm {
    #[cfg(target_arch = "wasm64")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_shuffle::<1, 3, 5, 7>(
            arch::i64x2_extmul_low_i32x4(lhs, rhs),
            arch::i64x2_extmul_high_i32x4(lhs, rhs),
        )
    }

    #[cfg(target_arch = "wasm32")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Stable Workaround: Manual Zero Extension via Shuffle
        let zero = arch::i32x4_splat(0);

        // Extend low lanes: [a0, 0, a1, 0]
        let prod_lo = arch::i64x2_mul(
            arch::i32x4_shuffle::<0, 4, 1, 5>(lhs, zero),
            arch::i32x4_shuffle::<0, 4, 1, 5>(rhs, zero),
        );

        // Extend high lanes: [a2, 0, a3, 0]
        let prod_hi = arch::i64x2_mul(
            arch::i32x4_shuffle::<2, 6, 3, 7>(lhs, zero),
            arch::i32x4_shuffle::<2, 6, 3, 7>(rhs, zero),
        );

        // Extract high 32 bits
        arch::i32x4_shuffle::<1, 3, 5, 7>(prod_lo, prod_hi)
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_mul(lhs, rhs)
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_saturating_add(lhs, rhs)
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_saturating_sub(lhs, rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_add u32x4_add)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_mul u32x4_mul)
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
        let counts = arch::i8x16_popcnt(value);
        // Sum pairs of u8 into u16 (0+1, 2+3...)
        // Equivalent to _mm_maddubs_epi16(counts, _mm_set1_epi8(1))
        let pairs = arch::i16x8_extadd_pairwise_i8x16(counts);

        // Sum pairs of u16 into u32
        // Equivalent to _mm_madd_epi16(pairs, _mm_set1_epi16(1))
        arch::i32x4_extadd_pairwise_i16x8(pairs)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.leading_zeros())
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::zip(value, value, |a, _| a.trailing_zeros())
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U32x4Wasm {
    /// 2D Morton via an `i8x16.swizzle` nibble-LUT; every other `N` uses the cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            arch::wasm_morton2_epu32x(values[0], values[1])
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    /// 2D Morton decode via the swizzle compress; every other `N` uses the cascade.
    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                arch::wasm_morton2_compress_epu32x(code),
                arch::wasm_morton2_compress_epu32x(arch::u32x4_shr(code, 1)),
            )
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }
}

impl CastRegister<U32x4Wasm> for ArrayRegister<super::U64x2Wasm, 2> {
    fn cast_from(value: Storage<U32x4Wasm>) -> Storage<Self> {
        // zero-extend each pair of u32 to u64
        let lo = arch::i64x2_extend_low_u32x4(value);
        let hi = arch::i64x2_extend_high_u32x4(value);
        ArrayRegister([lo, hi])
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x2Wasm, 2>> for U32x4Wasm {
    #[rustfmt::skip]
    fn cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 2>>) -> Storage<Self> {
        // Selects bytes 0-3 (lane 0 low) and 8-11 (lane 1 low) from lo
        // Selects bytes 16-19 (lane 0 low) and 24-27 (lane 1 low) from hi
        arch::i8x16_shuffle::<
            0, 1, 2, 3,     // Lo Vec, Lane 0 (Low bits)
            8, 9, 10, 11,   // Lo Vec, Lane 1 (Low bits)
            16, 17, 18, 19, // Hi Vec, Lane 0 (Low bits)
            24, 25, 26, 27  // Hi Vec, Lane 1 (Low bits)
        >(value.0[0], value.0[1])
    }
}

// Saturating narrow u64x4 -> u32x4: clamp the high end (no 64-bit narrow on WASM) + truncating narrow.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<ArrayRegister<super::U64x2Wasm, 2>> for U32x4Wasm {
    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x2Wasm, 2>>) -> Storage<Self> {
        type Src = ArrayRegister<super::U64x2Wasm, 2>;
        let hi = <Src as Register>::splat(u32::MAX as u64);
        let clamped = <Src as NumericRegister>::min(value, hi);
        <Self as CastRegister<Src>>::cast_from(clamped)
    }
}
