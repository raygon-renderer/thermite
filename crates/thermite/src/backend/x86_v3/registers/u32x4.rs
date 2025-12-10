use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister,
        Register, ShuffleRegister, Storage, SwizzleRegister, UnsignedIntegerRegister, dp::DoublePumpRegister,
        empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x4V3;

impl Register for U32x4V3 {
    type Lanes = typenum::U4;

    type Element = u32;
    type Storage = arch::__m128i;
    type HalfRegister = <Scalar as Simd>::u32x2;
    type DoubleRegister = super::U32x8V3;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type ISize = super::I32x4V3;
    type USize = super::U32x4V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epu32x(value, 0, 0, 0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        unsafe {
            let mut arr = [0; 4];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (DoublePumpRegister(arr[0], arr[1]), DoublePumpRegister(arr[2], arr[3]))
        }
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm_setr_epu32x(lo.0, lo.1, hi.0, hi.1) }
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
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi32(a, b), arch::_mm_unpackhi_epi32(a, b)) }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi32x_v2(value) }
    }
}

impl ShuffleRegister for U32x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_shuffle_ps(
                arch::_mm_castsi128_ps(lhs),
                arch::_mm_castsi128_ps(rhs),
                IMM8,
            ))
        }

        // unsafe {
        //     arch::_mm_blend_epi32(
        //         arch::_mm_shuffle_epi32(lhs, IMM8),
        //         arch::_mm_shuffle_epi32(rhs, IMM8),
        //         0xF0,
        //     )
        // }
    }
}

impl PermuteRegister for U32x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32(value, IMM8) }
    }
}

impl SwizzleRegister for U32x4V3 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_permutevar_ps(
                arch::_mm_castsi128_ps(value),
                core::mem::transmute(idxs),
            ))
        }
    }

    // #[inline(always)]
    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     a: Storage<Self>,
    //     b: Storage<Self>,
    // ) -> Storage<Self> {
    //     unsafe {
    //         arch::_mm_blend_epi16(
    //             arch::_mm_shuffle_epi32(a, AIMM8),
    //             arch::_mm_shuffle_epi32(b, BIMM8),
    //             BLEND,
    //         )
    //     }
    // }
}

impl BitshiftRegister for U32x4V3 {
    const HAS_TRUE_SHIFTV: bool = true;

    #[inline(always)]
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    #[inline(always)]
    fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi32(value, shifts) }
    }

    #[inline(always)]
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi32(value, IMM8) }
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi32(value, IMM8) }
    }
}

impl MaskRegister for U32x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([!0; 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_cvtboolx4_to_epi32_mask_v2(value) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_epi8(value) as u32 == 0xFFFF }
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
        Some(unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl PartialOrdRegister for U32x4V3 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmplt_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmple_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpge_epu32x_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi32(lhs, rhs) }
    }
}

impl NumericRegister for U32x4V3 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([u32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([u32::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_min_epu32 _mm_min_epu32) as u32
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_max_epu32 _mm_max_epu32) as u32
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u32))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi32(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
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
        unsafe { arch::_mm_min_epu32(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu32(lhs, rhs) }
    }
}

impl IntegerRegister for U32x4V3 {
    #[inline(always)]
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullhi_epu32x_v1(lhs, rhs) }
    }

    #[inline(always)]
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        todo!("arch::_mm_mullo_epu32x_v1(lhs, rhs)")
    }

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
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    #[inline(always)]
    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
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
        unsafe { arch::_mm_popcnt_epi32x_v2(value) }
    }

    #[inline(always)]
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::ilog2p1(value))
    }

    #[inline(always)]
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        // treat as unsigned
        super::I32x4V3::trailing_zeros(value)
    }
}

impl UnsignedIntegerRegister for U32x4V3 {}
