use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, IndexableRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedIntegerRegister, SignedRegister, Storage, WideRegister, ZeroUpper, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I32x4V3;

#[thermite_macros::inline_always]
impl CoreRegister for I32x4V3 {
    type NativeIsa = crate::backend::x86_v3::X86V3;
    type Lanes = typenum::U4;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_epi8(on_false, on_true, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 2 } {
            unsafe { arch::_mm_move_epi64(value) }
        } else {
            unsafe { arch::_mm_and_si128(value, arch::_mm_zeroupper_mask_epi32::<Z>()) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I32x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([-1; 4]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_cvtboolx4_to_epi32_mask_v2(value) }
    }

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
        unsafe { arch::_mm_count_mask_epi32x_v1(values) }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi32x_v1(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_ps(arch::_mm_castsi128_ps(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I32x4V3 {
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
impl WideRegister for I32x4V3 {
    type Wide = super::I32x8V3;
}

#[thermite_macros::inline_always]
impl Register for I32x4V3 {
    type Element = i32;

    type Signed = super::I32x4V3;
    type Unsigned = super::U32x4V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_srai_epi32(value, 31) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi32(value, 0, 0, 0) }
    }

    impl_native_radix3!(arch::_mm_interleave3_epi32, arch::_mm_deinterleave3_epi32);

    impl_native_extract!(@epi32x4);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi32(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskload_epi32(ptr, mask) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_maskstore_epi32(ptr, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 8 {
            let mut padded = [0i32; 8];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let table = arch::_mm256_loadu_si256(padded.as_ptr() as *const _);
                let idx = arch::_mm256_castsi128_si256(indices); // upper 128 bits undefined, don't care
                let result = arch::_mm256_permutevar8x32_epi32(table, idx);
                arch::_mm256_castsi256_si128(result)
            }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi32x_v2(value) }
    }

    const HAS_PERMUTEV: bool = true;

    impl_byte_align_alignr!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_permutevar_ps(
                arch::_mm_castsi128_ps(value),
                core::mem::transmute(idxs),
            ))
        }
    }

    //
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

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I32x4V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi32(a, b), arch::_mm_unpackhi_epi32(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // Zero-cost cast to use the efficient float shuffle
            let a = arch::_mm_castsi128_ps(a);
            let b = arch::_mm_castsi128_ps(b);

            let res_a = arch::_mm_castps_si128(arch::_mm_shuffle_ps(a, b, 0x88));
            let res_b = arch::_mm_castps_si128(arch::_mm_shuffle_ps(a, b, 0xDD));

            (res_a, res_b)
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x4V3> for I32x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_epi32::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_i32gather_epi32::<4>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V3> for I32x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_epi32::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i64gather_epi32::<4>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I32x4V3 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi32(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi32(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi32(value, IMM8) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi32(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for I32x4V3 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_shuffle_ps(
                arch::_mm_castsi128_ps(lhs),
                arch::_mm_castsi128_ps(rhs),
                IMM8,
            ))
        }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for I32x4V3 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I32x4V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epi32(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi32(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I32x4V3 {
    sort_via_network!(4);

    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([i32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([i32::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_min_epi32 _mm_min_epi32)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_max_epi32 _mm_max_epi32)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_hadd_epi32(lo, hi) }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi32(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi32(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi32(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi32(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I32x4V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1; 4]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sign_epi32(value, Self::NEG_ONE) }
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_signbits_epi32x_v1(value) }
    }

    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi32(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // NOT psignd: that negates whenever rhs is negative regardless of
        // lhs's own sign, which is wrong for negative lhs. True copysign
        // negates exactly where the signs differ.
        unsafe { arch::_mm_copysign_epi32x_v1(lhs, rhs) }
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        // psignd: +1 where value > 0, -1 where value < 0, 0 where value == 0
        // (three-valued, matching Rust `i32::signum`).
        unsafe { arch::_mm_sign_epi32(arch::_mm_set1_epi32(1), value) }
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<31>(mask))
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I32x4V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullhi_epi32x_v2(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi32x_v2(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi32x_v2(lhs, rhs) }
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32) as i32
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32) as i32
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
        unsafe { arch::_mm_popcnt_epi32x_v2(value) }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        // treat as unsigned
        super::U32x4V3::leading_zeros(value)
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
impl SignedIntegerRegister for I32x4V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi32(value, IMM8) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srav_epi32(value, shifts) }
    }
}
