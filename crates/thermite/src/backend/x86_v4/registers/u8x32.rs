//! `u8x32` on AVX-512: the 256-bit unsigned-byte register (BW+VL). See
//! [`u8x64`](super::u8x64) for the unsigned rules and [`i8x32`](super::i8x32)
//! for the width notes. Not a `Simd` grid slot, ladder only.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, ConcatRegister, CoreRegister, ExtendRegister, IndexableRegister,
    IntegerRegister, InterleaveRegister, NumericRegister, PartialOrdRegister, Register, Storage,
    UnsignedIntegerRegister, WideRegister, ZeroUpper, empty_reg, reg,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::i8x32::I8x32V4;
use super::kmask::KMask32;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U8x32V4;

#[thermite_macros::inline_always]
impl CoreRegister for U8x32V4 {
    type Lanes = typenum::U32;
    type Storage = arch::__m256i;
    type Mask = KMask32;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_blend_epi8(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi8(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi8(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi8(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U8x32V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        bitxor(lhs: Storage<Self>, rhs: Storage<Self>);
        bitand(lhs: Storage<Self>, rhs: Storage<Self>);
        bitor(lhs: Storage<Self>, rhs: Storage<Self>);
        bitandnot(lhs: Storage<Self>, rhs: Storage<Self>);
        not(value: Storage<Self>);
        ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>);
        bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::U8x16V4> for U8x32V4 {
    fn concat(lo: Storage<super::U8x16V4>, hi: Storage<super::U8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U8x16V4>, Storage<super::U8x16V4>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::U8x16V4> for U8x32V4 {
    fn extend(value: Storage<super::U8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U8x16V4> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl WideRegister for U8x32V4 {
    type Wide = super::U8x64V4;
}

#[thermite_macros::inline_always]
impl Register for U8x32V4 {
    type Element = u8;

    type Signed = I8x32V4;
    type Unsigned = U8x32V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_test_epi8_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_movepi8_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(arch::_mm_cvtsi32_si128(value as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi8(value as i8) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi8(src, mask, ptr as *const i8) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi8(mask, ptr as *const i8) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm256_mask_storeu_epi8(ptr as *mut i8, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_stream_load_si256(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_si256(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 32 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm256_permutexvar_epi8x_v4(indices, Self::new(padded)) }
        } else if values.len() <= 64 {
            let mut padded = [0u8; 64];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm256_loadu_si256(padded.as_ptr() as *const _);
                let hi = arch::_mm256_loadu_si256(padded.as_ptr().add(32) as *const _);
                arch::_mm256_permutex2var_epi8x_v4(lo, indices, hi)
            }
        } else {
            let idx = Self::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            compress_v4!(u32, _mm256_maskz_compress_epi8, _mm256_mask_expand_epi8, mask, value)
        } else {
            crate::backend::generic::polyfills::compress_grouped::<Self>(value, mask)
        }
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            expand_v4!(
                u32,
                _mm256_maskz_compress_epi8,
                _mm256_mask_expand_epi8,
                _mm256_maskz_expand_epi8,
                mask,
                value
            )
        } else {
            crate::backend::generic::polyfills::expand_grouped::<Self>(value, mask)
        }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutexvar_epi8x_v4(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutex2var_epi8x_v4(a, idxs, b) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::broadcast::<I>(value)
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        I8x32V4::broadcastv(value, idx)
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi8(src, mask, value.as_ptr() as *const i8) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi8(mask, value.as_ptr() as *const i8) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_set1_epi8(src, mask, value as i8) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_set1_epi8(mask, value as i8) }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        broadcast<const I: usize>(value: Storage<Self>);
        broadcastv(value: Storage<Self>, idx: usize);
        reverse(value: Storage<Self>);
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi8x_v4(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutexvar_epi8x_v4(mask, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mov_epi8(src, mask, Self::swizzle(a, b, idxs)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutex2var_epi8x_v4(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U8x32V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I8x32V4::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I8x32V4::deinterleave(a, b)
    }
}

impl IndexableRegister<U8x32V4> for U8x32V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for U8x32V4 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi8x_v4(value, shift) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi8x_v4(value, shift) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi8x_v4(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi8x_v4(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi8x_v4::<IMM8>(value) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi8x_v4::<IMM8>(value) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_rol_epi8x_v4(value, shift) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_ror_epi8x_v4(value, shift) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_roli_epi8x_v4::<IMM8>(value) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rori_epi8x_v4::<IMM8>(value) }
    }

    fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_reverse_epi8x_v4(value) }
    }

    // --- masked variants -----------------------------------------------------
    // One masked vgf2p8affineqb under GFNI. Below it the BW polyfill plus a merge.

    masked_poly_v4! {
        shl(value: Storage<Self>, shift: u32) => _mm256_mask_sll_epi8x_v4, _mm256_maskz_sll_epi8x_v4;
        shr(value: Storage<Self>, shift: u32) => _mm256_mask_srl_epi8x_v4, _mm256_maskz_srl_epi8x_v4;
        shli<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_slli_epi8x_v4, _mm256_maskz_slli_epi8x_v4;
        shri<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_srli_epi8x_v4, _mm256_maskz_srli_epi8x_v4;
        rol(value: Storage<Self>, shift: u32) => _mm256_mask_rol_epi8x_v4, _mm256_maskz_rol_epi8x_v4;
        ror(value: Storage<Self>, shift: u32) => _mm256_mask_ror_epi8x_v4, _mm256_maskz_ror_epi8x_v4;
        roli<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_roli_epi8x_v4, _mm256_maskz_roli_epi8x_v4;
        rori<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_rori_epi8x_v4, _mm256_maskz_rori_epi8x_v4;
        reverse_bits(value: Storage<Self>) => _mm256_mask_reverse_epi8x_v4, _mm256_maskz_reverse_epi8x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U8x32V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpgt_epu8_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpge_epu8_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmplt_epu8_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmple_epu8_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpeq_epi8_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpneq_epi8_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U8x32V4 {
    const ZERO: Storage<Self> = reg::<Self, 32>([0; 32]);
    const ONE: Storage<Self> = reg::<Self, 32>([1; 32]);
    const TWO: Storage<Self> = reg::<Self, 32>([2; 32]);

    const MIN: Storage<Self> = reg::<Self, 32>([u8::MIN; 32]);
    const MAX: Storage<Self> = reg::<Self, 32>([u8::MAX; 32]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_min_epu8) as u8
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_max_epu8) as u8
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_add_epi8) as u8
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1u8, u8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u8))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi8(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi8(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi8x_v4(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu8(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu8(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm256_mask_add_epi8, _mm256_maskz_add_epi8;
        sub => _mm256_mask_sub_epi8, _mm256_maskz_sub_epi8;
        min => _mm256_mask_min_epu8, _mm256_maskz_min_epu8;
        max => _mm256_mask_max_epu8, _mm256_maskz_max_epu8;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        mul(lhs: Storage<Self>, rhs: Storage<Self>);
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
        square(lhs: Storage<Self>);
        scale(value: Storage<Self>, scalar: Self::Element);
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U8x32V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhi_epu8x_v4(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi8x_v4(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epu8(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epu8(lhs, rhs) }
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

    const HAS_HARDWARE_POPCNT: bool = <F as Avx512Features>::AVX512BITALG;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::count_ones(value)
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::count_zeros(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_lzcnt_epi8x_v4(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_zeros(value)
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_ones(value)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        saturating_add => _mm256_mask_adds_epu8, _mm256_maskz_adds_epu8;
        saturating_sub => _mm256_mask_subs_epu8, _mm256_maskz_subs_epu8;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        mullo(lhs: Storage<Self>, rhs: Storage<Self>);
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm256_mask_popcnt_epi8x_v4, _mm256_maskz_popcnt_epi8x_v4;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::count_zeros_c(mask, value)
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::count_zeros_m(src, mask, value)
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::count_zeros_z(mask, value)
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_zeros_c(mask, value)
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_zeros_m(src, mask, value)
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_zeros_z(mask, value)
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_ones_c(mask, value)
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_ones_m(src, mask, value)
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x32V4::trailing_ones_z(mask, value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U8x32V4 {
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(8), Self::leading_zeros(value))
    }

    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_avg_epu8(a, b) }
    }

    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        avg => _mm256_mask_avg_epu8, _mm256_maskz_avg_epu8;
    }

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi8(value, mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi8(src, mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi8(mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi8(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi8(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi8(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi8, movz = _mm256_maskz_mov_epi8;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}
