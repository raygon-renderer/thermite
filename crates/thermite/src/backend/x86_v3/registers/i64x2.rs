use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        FloatRegister, IndexableRegister, IntegerRegister, InterleaveRegister, MaskElement, MaskRegister,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister,
        SignedRegister, Storage, WideRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg, reg_splat,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x2V3;

#[thermite_macros::inline_always]
impl CoreRegister for I64x2V3 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128i;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

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
        if const { Z::N >= 2 } {
            value
        } else if const { Z::N == 1 } {
            unsafe { arch::_mm_move_epi64(value) }
        } else {
            Self::EMPTY // N == 0, so zero everything
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I64x2V3 {
    const FALSY: Storage<Self> = reg::<Self, 2>([0; 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([-1; 2]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_cvtboolx2_to_epi64_mask_v2(value) }
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

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(arch::_mm_castsi128_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I64x2V3 {
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
impl ConcatRegister<i64> for I64x2V3 {
    fn concat(lo: Storage<i64>, hi: Storage<i64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<i64>, Storage<i64>) {
        let mut arr = [0i64; 2];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value) };
        (arr[0], arr[1])
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<i64> for I64x2V3 {
    fn extend(value: Storage<i64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_epi64x(value, 0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i64> {
        unsafe { arch::_mm_cvtsi128_si64(value) }
    }
}

#[thermite_macros::inline_always]
impl WideRegister for I64x2V3 {
    type Wide = super::I64x4V3;
}

impl Register for I64x2V3 {
    type Element = i64;

    type Signed = super::I64x2V3;
    type Unsigned = super::U64x2V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::is_negative(value)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(0, value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi64x(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskload_epi64(ptr, mask) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
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

    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<{ MM_SHUFFLE_R!(2, 3, 0, 1) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_epi64x_v2(value) }
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_slice(&value);

        f(arr[0], arr[1])
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let idxs = arch::_mm_setr_epu32x(idxs[0], idxs[1], 0, 0);
            let idxs = arch::_mm_cvtepu32_epi64(idxs);
            arch::_mm_permutevarx_epi64x_v2(value, idxs)
        }
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I64x2V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V3> for I64x2V3 {
    unsafe fn gather(ptr: *const <I64x2V3 as Register>::Element, indices: Storage<super::U64x2V3>) -> Storage<I64x2V3> {
        unsafe { arch::_mm_i64gather_epi64::<8>(ptr as *const _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<I64x2V3>,
        mask: Storage<<I64x2V3 as CoreRegister>::Mask>,
        ptr: *const <I64x2V3 as Register>::Element,
        indices: Storage<super::U64x2V3>,
    ) -> Storage<I64x2V3> {
        unsafe { arch::_mm_mask_i64gather_epi64::<8>(src, ptr as *const _, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x2V3> for I64x2V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_epi64::<8>(ptr as *const _, indices.0) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x2V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_i32gather_epi64::<8>(src, ptr as *const _, indices.0, mask) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I64x2V3 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128(value, IMM8) }
    }

    fn bshri<const IMM8: i32>(mut value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128(value, IMM8) }
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi64(value, shifts) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi64(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi64(value, IMM8) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi64(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for I64x2V3 {
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

#[thermite_macros::inline_always]
impl PartialOrdRegister for I64x2V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_epi64(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_epi64x_v1(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I64x2V3 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([i64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([i64::MAX; 2]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.min(hi)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.max(hi)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        _mm_pairwise_sum_epi64_v1!(lo, hi)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi64x_v2(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi64x_v2(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I64x2V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1; 2]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_add_epi64(
                arch::_mm_xor_si128(value, arch::_mm_set1_epi64x(-1)),
                arch::_mm_set1_epi64x(1),
            )
        }
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_signbits_epi64x_v1(value) }
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let sign = arch::_mm_signbits_epi64x_v1(value);
            arch::_mm_xor_si128(sign, arch::_mm_add_epi64(value, sign))
        }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_copysign_epi64x_v2(lhs, rhs) }
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        // (value < 0 ? -1 : 0) - (value > 0 ? -1 : 0)  =>  -1 / 0 / +1
        // (three-valued, matching Rust `i64::signum`; there is no `psignq`).
        unsafe {
            let lt = arch::_mm_cmpgt_epi64(Self::ZERO, value); // -1 where value < 0
            let gt = arch::_mm_cmpgt_epi64(value, Self::ZERO); // -1 where value > 0
            arch::_mm_sub_epi64(lt, gt)
        }
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::add(Self::bitxor(value, mask), Self::shri::<63>(mask))
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I64x2V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullhi_epi64x_v1(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64x_v2(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi64x_v2(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi64x_v2(lhs, rhs) }
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_add(hi)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        let [lo, hi]: [i64; 2] = unsafe { core::mem::transmute(value) };
        lo.wrapping_mul(hi)
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
        unsafe { arch::_mm_popcnt_epi64x_v2(value) }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        use crate::register::UnsignedIntegerRegister;

        Self::sub(Self::splat(64), super::U64x2V3::ilog2p1(value))
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
impl SignedIntegerRegister for I64x2V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi64x_v1(value, IMM8) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi64x_v1(value, shift as i32) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srav_epi64x_v3(value, shifts) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<<Scalar as Simd>::i32x2> for I64x2V3 {
    fn cast_from(value: Storage<<Scalar as Simd>::i32x2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_epi64(arch::_mm_setr_epi32(value.0[0], value.0[1], 0, 0)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I64x2V3> for <Scalar as Simd>::i32x2 {
    fn cast_from(value: Storage<I64x2V3>) -> Storage<<Scalar as Simd>::i32x2> {
        unsafe {
            ArrayRegister([
                arch::_mm_cvtsi128_si32(value),      // lowest 32 bits
                arch::_mm_extract_epi32::<2>(value), // next 32 bits after 64 bits
            ])
        }
    }
}
