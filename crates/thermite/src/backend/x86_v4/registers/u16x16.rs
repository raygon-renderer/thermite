//! `u16x16` on AVX-512: the 256-bit unsigned-word register (BW+VL). See
//! [`u16x32`](super::u16x32) for the unsigned rules and [`i16x16`](super::i16x16)
//! for the width notes. The `Simd` grid's `u16x16` slot lives here (ladder +
//! u32x16 / u64x16 casts).

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, ExtendRegister, IndexableRegister,
    IntegerRegister, InterleaveRegister, NumericRegister, PartialOrdRegister, Register, Storage,
    UnsignedIntegerRegister, WideRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::i16x16::I16x16V4;
use super::kmask::KMask16;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x16V4;

#[thermite_macros::inline_always]
impl CoreRegister for U16x16V4 {
    type Lanes = typenum::U16;
    type Storage = arch::__m256i;
    type Mask = KMask16;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_blend_epi16(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi16(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi16(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi16(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U16x16V4 {
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
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
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
impl ConcatRegister<super::U16x8V4> for U16x16V4 {
    fn concat(lo: Storage<super::U16x8V4>, hi: Storage<super::U16x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U16x8V4>, Storage<super::U16x8V4>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::U16x8V4> for U16x16V4 {
    fn extend(value: Storage<super::U16x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U16x8V4> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl WideRegister for U16x16V4 {
    type Wide = super::U16x32V4;
}

#[thermite_macros::inline_always]
impl Register for U16x16V4 {
    type Element = u16;

    type Signed = I16x16V4;
    type Unsigned = U16x16V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_test_epi16_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_movepi16_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(arch::_mm_cvtsi32_si128(value as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi16(value as i16) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi16(src, mask, ptr as *const i16) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi16(mask, ptr as *const i16) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm256_mask_storeu_epi16(ptr as *mut i16, mask, value) }
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
        if values.len() <= 16 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm256_permutexvar_epi16(indices, Self::new(padded)) }
        } else if values.len() <= 32 {
            let mut padded = [0u16; 32];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm256_loadu_si256(padded.as_ptr() as *const _);
                let hi = arch::_mm256_loadu_si256(padded.as_ptr().add(16) as *const _);
                arch::_mm256_permutex2var_epi16(lo, indices, hi)
            }
        } else {
            let idx = Self::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::swap_bytes(value)
    }

    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            compress_v4!(u16, _mm256_maskz_compress_epi16, _mm256_mask_expand_epi16, mask, value)
        } else {
            crate::backend::generic::polyfills::compress_grouped::<Self>(value, mask)
        }
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            expand_v4!(
                u16,
                _mm256_maskz_compress_epi16,
                _mm256_mask_expand_epi16,
                _mm256_maskz_expand_epi16,
                mask,
                value
            )
        } else {
            crate::backend::generic::polyfills::expand_grouped::<Self>(value, mask)
        }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutexvar_epi16(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutex2var_epi16(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi16(src, mask, value.as_ptr() as *const i16) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi16(mask, value.as_ptr() as *const i16) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_set1_epi16(src, mask, value as i16) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_set1_epi16(mask, value as i16) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::broadcast::<I>(value)
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::broadcast_c::<I>(mask, value)
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        I16x16V4::broadcast_m::<I>(src, mask, value)
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::broadcast_z::<I>(mask, value)
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x16V4::broadcastv(value, idx)
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x16V4::broadcastv_c(mask, value, idx)
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x16V4::broadcastv_m(src, mask, value, idx)
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x16V4::broadcastv_z(mask, value, idx)
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::reverse_c(mask, value)
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::reverse_m(src, mask, value)
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::reverse_z(mask, value)
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi16(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutexvar_epi16(mask, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mov_epi16(src, mask, arch::_mm256_permutex2var_epi16(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutex2var_epi16(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U16x16V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I16x16V4::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I16x16V4::deinterleave(a, b)
    }
}

impl IndexableRegister<U16x16V4> for U16x16V4 {}
impl IndexableRegister<super::U32x16V4> for U16x16V4 {}
impl IndexableRegister<ArrayRegister<super::U64x8V4, 2>> for U16x16V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for U16x16V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi16(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi16(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi16(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi16(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm256_mask_sll_epi16, _mm256_maskz_sll_epi16;
        shr => _mm256_mask_srl_epi16, _mm256_maskz_srl_epi16;
    }

    masked_shifti_v4! {
        shli => _mm256_mask_sll_epi16, _mm256_maskz_sll_epi16;
        shri => _mm256_mask_srl_epi16, _mm256_maskz_srl_epi16;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm256_mask_sllv_epi16, _mm256_maskz_sllv_epi16;
        shrv(Storage<Self::Unsigned>) => _mm256_mask_srlv_epi16, _mm256_maskz_srlv_epi16;
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_rol_epi16x_v4(value, shift) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_ror_epi16x_v4(value, shift) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_roli_epi16x_v4::<IMM8>(value) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rori_epi16x_v4::<IMM8>(value) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_rolv_epi16x_v4(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_rorv_epi16x_v4(value, shifts) }
    }

    // Rotates: one (masked) vpshldvw/vpshrdvw under VBMI2, the shift-or pair
    // below it. Byte shifts and the bit-reverse cascade end without a
    // maskable op.
    masked_poly_v4! {
        rol(value: Storage<Self>, shift: u32) => _mm256_mask_rol_epi16x_v4, _mm256_maskz_rol_epi16x_v4, _mm256_maskc_rol_epi16x_v4;
        ror(value: Storage<Self>, shift: u32) => _mm256_mask_ror_epi16x_v4, _mm256_maskz_ror_epi16x_v4, _mm256_maskc_ror_epi16x_v4;
        roli<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_roli_epi16x_v4, _mm256_maskz_roli_epi16x_v4, _mm256_maskc_roli_epi16x_v4;
        rori<const IMM8: i32>(value: Storage<Self>) => _mm256_mask_rori_epi16x_v4, _mm256_maskz_rori_epi16x_v4, _mm256_maskc_rori_epi16x_v4;
        rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) => _mm256_mask_rolv_epi16x_v4, _mm256_maskz_rolv_epi16x_v4, _mm256_maskc_rolv_epi16x_v4;
        rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) => _mm256_mask_rorv_epi16x_v4, _mm256_maskz_rorv_epi16x_v4, _mm256_maskc_rorv_epi16x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U16x16V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpgt_epu16_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpge_epu16_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmplt_epu16_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmple_epu16_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpeq_epi16_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpneq_epi16_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U16x16V4 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([u16::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([u16::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_min_epu16) as u16
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_max_epu16) as u16
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_add_epi16) as u16
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi16_v3!(value; _mm_mullo_epi16) as u16
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u16))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi16(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi16(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi16(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu16(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu16(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm256_mask_add_epi16, _mm256_maskz_add_epi16;
        sub => _mm256_mask_sub_epi16, _mm256_maskz_sub_epi16;
        mul => _mm256_mask_mullo_epi16, _mm256_maskz_mullo_epi16;
        min => _mm256_mask_min_epu16, _mm256_maskz_min_epu16;
        max => _mm256_mask_max_epu16, _mm256_maskz_max_epu16;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi16(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi16(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mullo_epi16(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi16(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi16(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mullo_epi16(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U16x16V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhi_epu16(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi16(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epu16(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epu16(lhs, rhs) }
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
        I16x16V4::count_ones(value)
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::count_zeros(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_lzcnt_epi16x_v4(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_zeros(value)
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_ones(value)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        mullo => _mm256_mask_mullo_epi16, _mm256_maskz_mullo_epi16;
        mulhi => _mm256_mask_mulhi_epu16, _mm256_maskz_mulhi_epu16;
        saturating_add => _mm256_mask_adds_epu16, _mm256_maskz_adds_epu16;
        saturating_sub => _mm256_mask_subs_epu16, _mm256_maskz_subs_epu16;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm256_mask_popcnt_epi16x_v4, _mm256_maskz_popcnt_epi16x_v4;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::count_zeros_c(mask, value)
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::count_zeros_m(src, mask, value)
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::count_zeros_z(mask, value)
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_zeros_c(mask, value)
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_zeros_m(src, mask, value)
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_zeros_z(mask, value)
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_ones_c(mask, value)
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_ones_m(src, mask, value)
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x16V4::trailing_ones_z(mask, value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U16x16V4 {
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(16), Self::leading_zeros(value))
    }

    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_avg_epu16(a, b) }
    }

    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    /// 2D Morton via the inherited 256-bit PSHUFB nibble-LUT. Other `N` use
    /// the generic cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm256_morton2_epu16x_v3(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            unsafe {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::_mm256_morton2_compress_epu16x_v3(code),
                    arch::_mm256_morton2_compress_epu16x_v3(Self::shri::<1>(code)),
                )
            }
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        avg => _mm256_mask_avg_epu16, _mm256_maskz_avg_epu16;
    }

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi16(value, mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi16(src, mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi16(mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi16(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi16(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi16(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi16, movz = _mm256_maskz_mov_epi16;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}

// --- Simd-grid casts: u16x16 <-> u32x16 / u64x16. ---

#[thermite_macros::inline_always]
impl CastRegister<U16x16V4> for super::U32x16V4 {
    fn cast_from(value: Storage<U16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepu16_epi32(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U32x16V4> for U16x16V4 {
    fn cast_from(value: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi32_epi16(value) }
    }

    // vpmovusdw saturates in hardware.
    fn saturating_cast_from(value: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtusepi32_epi16(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U16x16V4> for ArrayRegister<super::U64x8V4, 2> {
    fn cast_from(value: Storage<U16x16V4>) -> Storage<Self> {
        let (lo, hi) = U16x16V4::split(value);
        unsafe { ArrayRegister([arch::_mm512_cvtepu16_epi64(lo), arch::_mm512_cvtepu16_epi64(hi)]) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::U64x8V4, 2>> for U16x16V4 {
    fn cast_from(value: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm256_setr_m128i(arch::_mm512_cvtepi64_epi16(lo), arch::_mm512_cvtepi64_epi16(hi)) }
    }

    fn saturating_cast_from(value: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm256_setr_m128i(arch::_mm512_cvtusepi64_epi16(lo), arch::_mm512_cvtusepi64_epi16(hi)) }
    }
}
