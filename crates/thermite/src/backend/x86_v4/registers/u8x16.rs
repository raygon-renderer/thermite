//! `u8x16` on AVX-512: the 128-bit unsigned-byte register (BW+VL). See
//! [`u8x64`](super::u8x64) for the unsigned rules and [`i8x16`](super::i8x16)
//! for the width notes. The `Simd` grid's `u8x16` slot lives here (u16x16 /
//! u32x16 / u64x16 casts). Its SAD ladder and fp8 transcode are stamped in
//! `registers/mod.rs`.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
    InterleaveRegister, NumericRegister, PartialOrdRegister, Register, Storage, UnsignedIntegerRegister, WideRegister,
    ZeroUpper, array::ArrayRegister, empty_reg, reg,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::i8x16::I8x16V4;
use super::kmask::KMask16;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U8x16V4;

#[thermite_macros::inline_always]
impl CoreRegister for U8x16V4 {
    type Lanes = typenum::U16;
    type Storage = arch::__m128i;
    type Mask = KMask16;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_blend_epi8(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi8(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi8(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi8(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U8x16V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
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
impl WideRegister for U8x16V4 {
    type Wide = super::U8x32V4;
}

#[thermite_macros::inline_always]
impl Register for U8x16V4 {
    type Element = u8;

    type Signed = I8x16V4;
    type Unsigned = U8x16V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_test_epi8_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_movepi8_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_cvtsi32_si128(value as i32) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi8(value as i8) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi8(src, mask, ptr as *const i8) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi8(mask, ptr as *const i8) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_epi8(ptr as *mut i8, mask, value) }
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
        if values.len() <= 16 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm_shuffle_epi8(Self::new(padded), indices) }
        } else if values.len() <= 32 {
            let mut padded = [0u8; 32];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm_loadu_si128(padded.as_ptr() as *const _);
                let hi = arch::_mm_loadu_si128(padded.as_ptr().add(16) as *const _);
                arch::_mm_permutex2var_epi8x_v4(lo, indices, hi)
            }
        } else {
            let idx = Self::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            compress_v4!(u16, _mm_maskz_compress_epi8, _mm_mask_expand_epi8, mask, value)
        } else {
            crate::backend::generic::polyfills::compress_grouped::<Self>(value, mask)
        }
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            expand_v4!(
                u16,
                _mm_maskz_compress_epi8,
                _mm_mask_expand_epi8,
                _mm_maskz_expand_epi8,
                mask,
                value
            )
        } else {
            crate::backend::generic::polyfills::expand_grouped::<Self>(value, mask)
        }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi8(value, idxs) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi8x_v4(a, idxs, b) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::broadcast::<I>(value)
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        I8x16V4::broadcastv(value, idx)
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi8(src, mask, value.as_ptr() as *const i8) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi8(mask, value.as_ptr() as *const i8) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_set1_epi8(src, mask, value as i8) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_set1_epi8(mask, value as i8) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::broadcast_c::<I>(mask, value)
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        I8x16V4::broadcast_m::<I>(src, mask, value)
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::broadcast_z::<I>(mask, value)
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I8x16V4::broadcastv_c(mask, value, idx)
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I8x16V4::broadcastv_m(src, mask, value, idx)
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I8x16V4::broadcastv_z(mask, value, idx)
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::reverse_c(mask, value)
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::reverse_m(src, mask, value)
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::reverse_z(mask, value)
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(src, mask, value, idxs) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi8(mask, value, idxs) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi8(src, mask, Self::swizzle(a, b, idxs)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi8x_v4(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U8x16V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I8x16V4::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I8x16V4::deinterleave(a, b)
    }
}

impl IndexableRegister<U8x16V4> for U8x16V4 {}
impl IndexableRegister<super::U32x16V4> for U8x16V4 {}
impl IndexableRegister<ArrayRegister<super::U64x8V4, 2>> for U8x16V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for U8x16V4 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128::<IMM8>(value) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128::<IMM8>(value) }
    }

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi8x_v4(value, shift) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi8x_v4(value, shift) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi8x_v4(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi8x_v4(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_slli_epi8x_v4::<IMM8>(value) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srli_epi8x_v4::<IMM8>(value) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_rol_epi8x_v4(value, shift) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_ror_epi8x_v4(value, shift) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roli_epi8x_v4::<IMM8>(value) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rori_epi8x_v4::<IMM8>(value) }
    }

    fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_reverse_epi8x_v4(value) }
    }

    // --- masked variants -----------------------------------------------------
    // One masked vgf2p8affineqb under GFNI. Below it the BW polyfill plus a merge.

    masked_poly_v4! {
        shl(value: Storage<Self>, shift: u32) => _mm_mask_sll_epi8x_v4, _mm_maskz_sll_epi8x_v4;
        shr(value: Storage<Self>, shift: u32) => _mm_mask_srl_epi8x_v4, _mm_maskz_srl_epi8x_v4;
        shli<const IMM8: i32>(value: Storage<Self>) => _mm_mask_slli_epi8x_v4, _mm_maskz_slli_epi8x_v4;
        shri<const IMM8: i32>(value: Storage<Self>) => _mm_mask_srli_epi8x_v4, _mm_maskz_srli_epi8x_v4;
        rol(value: Storage<Self>, shift: u32) => _mm_mask_rol_epi8x_v4, _mm_maskz_rol_epi8x_v4;
        ror(value: Storage<Self>, shift: u32) => _mm_mask_ror_epi8x_v4, _mm_maskz_ror_epi8x_v4;
        roli<const IMM8: i32>(value: Storage<Self>) => _mm_mask_roli_epi8x_v4, _mm_maskz_roli_epi8x_v4;
        rori<const IMM8: i32>(value: Storage<Self>) => _mm_mask_rori_epi8x_v4, _mm_maskz_rori_epi8x_v4;
        reverse_bits(value: Storage<Self>) => _mm_mask_reverse_epi8x_v4, _mm_maskz_reverse_epi8x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U8x16V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epu8_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpge_epu8_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmplt_epu8_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmple_epu8_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpeq_epi8_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpneq_epi8_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U8x16V4 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([u8::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([u8::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_min_epu8) as u8
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_max_epu8) as u8
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_add_epi8) as u8
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
        unsafe { arch::_mm_add_epi8(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi8(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v4(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epu8(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu8(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_epi8, _mm_maskz_add_epi8;
        sub => _mm_mask_sub_epi8, _mm_maskz_sub_epi8;
        min => _mm_mask_min_epu8, _mm_maskz_min_epu8;
        max => _mm_mask_max_epu8, _mm_maskz_max_epu8;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        mul(lhs: Storage<Self>, rhs: Storage<Self>);
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
        square(lhs: Storage<Self>);
        scale(value: Storage<Self>, scalar: Self::Element);
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U8x16V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epu8x_v4(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v4(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epu8(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epu8(lhs, rhs) }
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
        I8x16V4::count_ones(value)
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::count_zeros(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_lzcnt_epi8x_v4(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_zeros(value)
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_ones(value)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        saturating_add => _mm_mask_adds_epu8, _mm_maskz_adds_epu8;
        saturating_sub => _mm_mask_subs_epu8, _mm_maskz_subs_epu8;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        mullo(lhs: Storage<Self>, rhs: Storage<Self>);
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm_mask_popcnt_epi8x_v4, _mm_maskz_popcnt_epi8x_v4;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::count_zeros_c(mask, value)
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::count_zeros_m(src, mask, value)
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::count_zeros_z(mask, value)
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_zeros_c(mask, value)
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_zeros_m(src, mask, value)
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_zeros_z(mask, value)
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_ones_c(mask, value)
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_ones_m(src, mask, value)
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I8x16V4::trailing_ones_z(mask, value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U8x16V4 {
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(8), Self::leading_zeros(value))
    }

    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_avg_epu8(a, b) }
    }

    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        avg => _mm_mask_avg_epu8, _mm_maskz_avg_epu8;
    }

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(value, mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(src, mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi8(mask, Self::splat(8), Self::leading_zeros(value)) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi8(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}

// --- Simd-grid casts: u8x16 <-> u16x16 / u32x16 / u64x16. ---

#[thermite_macros::inline_always]
impl CastRegister<U8x16V4> for super::U16x16V4 {
    fn cast_from(value: Storage<U8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu8_epi16(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U16x16V4> for U8x16V4 {
    fn cast_from(value: Storage<super::U16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi8(value) }
    }

    fn saturating_cast_from(value: Storage<super::U16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtusepi16_epi8(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x16V4> for super::U32x16V4 {
    fn cast_from(value: Storage<U8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepu8_epi32(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x16V4> for U8x16V4 {
    fn cast_from(value: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi32_epi8(value) }
    }

    fn saturating_cast_from(value: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtusepi32_epi8(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U8x16V4> for ArrayRegister<super::U64x8V4, 2> {
    fn cast_from(value: Storage<U8x16V4>) -> Storage<Self> {
        unsafe {
            let hi = arch::_mm_unpackhi_epi64(value, value);
            ArrayRegister([arch::_mm512_cvtepu8_epi64(value), arch::_mm512_cvtepu8_epi64(hi)])
        }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x8V4, 2>> for U8x16V4 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm_unpacklo_epi64(arch::_mm512_cvtepi64_epi8(lo), arch::_mm512_cvtepi64_epi8(hi)) }
    }

    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm_unpacklo_epi64(arch::_mm512_cvtusepi64_epi8(lo), arch::_mm512_cvtusepi64_epi8(hi)) }
    }
}
