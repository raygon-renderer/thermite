use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, CastRegister, IntegerRegister, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        UnsignedIntegerRegister, dp::DoublePumpRegister,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x4Wasm;

impl Register for U32x4Wasm {
    type Lanes = typenum::U4;

    type Element = u32;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = arch::ISA;

    type ISize = super::I32x4Wasm;
    type USize = super::U32x4Wasm;

    const EMPTY: Storage<Self> = arch::u32x4(0, 0, 0, 0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::u32x4(value[0], value[1], value[2], value[3])
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::u32x4_splat(value)
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::u32x4(value, 0, 0, 0)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::u32x4_extract_lane::<I>(value)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: arguments are reversed
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);

        (low, high)
    }

    #[inline(always)]
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
}

impl BitshiftRegister for U32x4Wasm {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shr(value, shift) // Non-arithmetic
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        arch::u32x4_shl(value, shift)
    }
}

impl MaskRegister for U32x4Wasm {
    const FALSY: Storage<Self> = arch::u32x4(0, 0, 0, 0);
    const TRUTHY: Storage<Self> = arch::u32x4(!0, !0, !0, !0);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::u32x4_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::u32x4_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::u32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for U32x4Wasm {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for U32x4Wasm {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for U32x4Wasm {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::x4indices(idxs[0] as u8, idxs[1] as u8, idxs[2] as u8, idxs[3] as u8),
        )
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for U32x4Wasm {
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_ge(lhs, rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_lt(lhs, rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_le(lhs, rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_ne(lhs, rhs) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_gt(lhs, rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::u32x4_eq(lhs, rhs) }
}

impl NumericRegister for U32x4Wasm {
    const ZERO: Storage<Self> = arch::u32x4(0, 0, 0, 0);
    const ONE: Storage<Self> = arch::u32x4(1, 1, 1, 1);
    const TWO: Storage<Self> = arch::u32x4(2, 2, 2, 2);

    const MIN: Storage<Self> = arch::u32x4(u32::MIN, u32::MIN, u32::MIN, u32::MIN);
    const MAX: Storage<Self> = arch::u32x4(u32::MAX, u32::MAX, u32::MAX, u32::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_min u32x4_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_max u32x4_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_add u32x4_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_mul u32x4_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::u32x4_splat(<Self::Lanes as Unsigned>::USIZE as u32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::u32x4(0, 1, 2, 3)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_mul(lhs, rhs)
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_max(lhs, rhs)
    }
}

impl IntegerRegister for U32x4Wasm {
    #[inline(always)]
    #[cfg(target_arch = "wasm64")]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::i32x4_shuffle::<1, 3, 5, 7>(
            arch::i64x2_extmul_low_i32x4(a, b), //
            arch::i64x2_extmul_high_i32x4(a, b),
        )
    }

    #[inline(always)]
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

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_mul(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_saturating_add(lhs, rhs)
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u32x4_saturating_sub(lhs, rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_add u32x4_add)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(u value; u32x4_mul u32x4_mul)
    }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[inline(always)]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        let counts = arch::i8x16_popcnt(value);
        // Sum pairs of u8 into u16 (0+1, 2+3...)
        // Equivalent to _mm_maddubs_epi16(counts, _mm_set1_epi8(1))
        let pairs = arch::i16x8_extadd_pairwise_i8x16(counts);

        // Sum pairs of u16 into u32
        // Equivalent to _mm_madd_epi16(pairs, _mm_set1_epi16(1))
        let quads = arch::i32x4_extadd_pairwise_i16x8(pairs);

        quads
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }
}

impl UnsignedIntegerRegister for U32x4Wasm {}

impl CastRegister<U32x4Wasm> for DoublePumpRegister<super::U64x2Wasm> {
    #[inline(always)]
    fn cast_from(value: Storage<U32x4Wasm>) -> Storage<Self> {
        // zero-extend each pair of u32 to u64
        let lo = arch::i64x2_extend_low_u32x4(value);
        let hi = arch::i64x2_extend_high_u32x4(value);

        DoublePumpRegister(lo, hi)
    }
}
