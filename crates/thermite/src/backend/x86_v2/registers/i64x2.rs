use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, FloatRegister, IntegerRegister, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage,
        SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct I64x2V2;

impl Register for I64x2V2 {
    type Lanes = typenum::U2;

    type Element = i64;
    type Storage = arch::__m128i;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::X86V2;

    type SCOUNT = super::I64x2V2;
    type UCOUNT = super::U64x2V2;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_epi64x(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_epi8(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE_R!(2, 3, 0, 1) }>(value) }
    }

    #[inline(always)]
    fn unpack(a: Self::Storage, b: Self::Storage) -> (Self::Storage, Self::Storage) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    #[inline(always)]
    fn reduce<F>(value: Self::Storage, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

impl BitshiftRegister for I64x2V2 {
    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_srlv_epi64x_v1(value, shifts) }
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_sllv_epi64x_v1(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_slli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_srli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi64x_v2(value, shifts) }
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi64x_v2(value, shifts) }
    }

    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rolv_epi64x_v2(value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_rorv_epi64x_v2(value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_reverse_bits_epi64x_v1(value) }
    }
}

impl ShuffleRegister for I64x2V2 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm_castpd_si128(arch::_mm_shuffle_pd(
                arch::_mm_castsi128_pd(lhs),
                arch::_mm_castsi128_pd(rhs),
                IMM8,
            ))
        }
    }
}

impl SwizzleRegister for I64x2V2 {
    const HAS_PERMUTEV: bool = false;
}

impl MaskRegister for I64x2V2 {
    const FALSY: Self::Storage = reg::<Self, 2>([0; 2]);
    const TRUTHY: Self::Storage = reg::<Self, 2>([-1; 2]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_cvtboolx2_to_epi64_mask_v2(value) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Self::Storage) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Self::Storage, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl PartialOrdRegister for I64x2V2 {
    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpgt_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpeq_epi64x_v1(lhs, rhs) }
    }
}

impl NumericRegister for I64x2V2 {
    const ZERO: Self::Storage = reg::<Self, 2>([0; 2]);
    const ONE: Self::Storage = reg::<Self, 2>([1; 2]);
    const TWO: Self::Storage = reg::<Self, 2>([2; 2]);

    const MIN: Self::Storage = reg::<Self, 2>([i64::MIN; 2]);
    const MAX: Self::Storage = reg::<Self, 2>([i64::MAX; 2]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.min(hi)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.max(hi)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I64)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as i64))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_min_epi64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_epi64x_v2(lhs, rhs) }
    }
}

impl SignedRegister for I64x2V2 {
    const NEG_ONE: Self::Storage = reg::<Self, 2>([-1; 2]);
    const MIN_POSITIVE: Self::Storage = reg_splat::<Self>(1);

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        unsafe {
            arch::_mm_add_epi64(
                arch::_mm_xor_si128(value, arch::_mm_set1_epi64x(-1)),
                arch::_mm_set1_epi64x(1),
            )
        }
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        unsafe {
            let sign = arch::_mm_signbits_epi64x_v1(value);
            arch::_mm_xor_si128(sign, arch::_mm_add_epi64(value, sign))
        }
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_copysign_epi64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_epi8(Self::NEG_ONE, Self::ONE, arch::_mm_cmpgt_epi64(value, Self::NEG_ONE)) }
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self::add(Self::bitxor(value, mask), Self::shri::<63>(mask))
    }
}

impl IntegerRegister for I64x2V2 {
    #[inline(always)]
    fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_adds_epi64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_subs_epi64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn wrapping_sum(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    #[inline(always)]
    fn div_branched(value: Self::Storage, divider: crate::divider::Divider<Self::Element>) -> Self::Storage {
        unsafe { arch::_mm_div_epi64x_v1(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn div_branchfree(
        value: Self::Storage,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Self::Storage {
        unsafe { arch::_mm_div_epi64x_bf_v1(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn divv_branchfree(value: Self::Storage, dividers: crate::divider::vector::VectorDivider<Self>) -> Self::Storage {
        unsafe { arch::_mm_divv_epi64x_bf_v1(value, dividers.multipliers.0, dividers.shifts.0) }
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_popcnt_epi64x_v2(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        use crate::register::UnsignedIntegerRegister;

        Self::sub(Self::splat(64), super::U64x2V2::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }

    #[inline(always)]
    fn leading_ones(value: Self::Storage) -> Self::Storage {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Self::Storage) -> Self::Storage {
        Self::trailing_zeros(Self::not(value))
    }
}

impl SignedIntegerRegister for I64x2V2 {
    #[inline(always)]
    fn srai<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_srai_epi64x_v1(value, IMM8) }
    }

    #[inline(always)]
    fn sra(value: Self::Storage, shift: u32) -> Self::Storage {
        unsafe { arch::_mm_srai_epi64x_v1(value, shift as i32) }
    }

    #[inline(always)]
    fn srav(value: Self::Storage, shifts: Storage<Self::UCOUNT>) -> Self::Storage {
        unsafe { arch::_mm_srav_epi64x_v1(value, shifts) }
    }
}

impl CastRegister<DoublePumpRegister<I64x2V2>> for super::I32x4V2 {
    #[inline(always)]
    fn cast_from(value: <DoublePumpRegister<I64x2V2> as Register>::Storage) -> Self::Storage {
        let (lo, hi) = DoublePumpRegister::<I64x2V2>::split(value);
        unsafe { arch::_mm_cvtepi64_epi32x_v2(lo, hi) }
    }
}
