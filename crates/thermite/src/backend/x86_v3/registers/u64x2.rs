use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister,
        Register, ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x2V3;

impl Register for U64x2V3 {
    type Lanes = typenum::U2;

    type Element = u64;
    type Storage = arch::__m128i;
    type HalfRegister = ();
    type DoubleRegister = super::U64x4V3;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type ISize = super::I64x2V3;
    type USize = super::U64x2V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value as i64, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi64x(value as i64) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(value, arch::_mm_set1_epi8(-1)) }
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE_R!(2, 3, 0, 1) }>(value) }
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi64x_v2(value) }
    }

    #[inline(always)]
    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

impl BitshiftRegister for U64x2V3 {
    const HAS_TRUE_SHIFTV: bool = true;

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi64(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi64(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi64(value, IMM8) }
    }
}

impl ShuffleRegister for U64x2V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castpd_si128(arch::_mm_shuffle_pd(
                arch::_mm_castsi128_pd(lhs),
                arch::_mm_castsi128_pd(rhs),
                IMM8,
            ))
        }
    }
}

// impl PermuteRegister for U64x2V3 {
//     fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
//         unsafe { arch::_mm_permute_epi64x(value, IMM8) }
//     }
// }

impl MaskRegister for U64x2V3 {
    const FALSY: Storage<Self> = reg::<Self, 2>([0; 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([!0; 2]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_cvtboolx2_to_epi64_mask_v2(value) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0xFFFF }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl PartialOrdRegister for U64x2V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu64x_v2(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi64x_v1(lhs, rhs) }
    }
}

impl NumericRegister for U64x2V3 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([u64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([u64::MAX; 2]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.min(hi)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.max(hi)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
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
        unsafe { arch::_mm_min_epu64x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu64x_v2(lhs, rhs) }
    }
}

impl IntegerRegister for U64x2V3 {
    #[inline(always)]
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

    #[inline(always)]
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    #[inline(always)]
    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [u64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::divider::Divider<Self::Element>) -> Storage<Self> {
        unsafe { arch::_mm_div_epu64x_v1(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn div_branchfree(
        value: Storage<Self>,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_div_epu64x_bf_v1(value, divider.multiplier(), divider.shift()) }
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        unsafe { arch::_mm_divv_epu64x_bf_v1(value, dividers.multipliers.0, dividers.shifts.0) }
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    #[inline(always)]
    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_popcnt_epi64x_v2(value) }
    }

    #[inline(always)]
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    #[inline(always)]
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I64x2V3::count_ones(value)
    }

    #[inline(always)]
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

impl UnsignedIntegerRegister for U64x2V3 {}
