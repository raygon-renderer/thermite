//! `u16x32` on AVX-512: the native 512-bit unsigned-word register. Same
//! storage and house rules as [`i16x32`](super::i16x32) (read its module
//! docs). Unsigned-specific notes:
//!
//! - Compares are the `epu16` opmask forms, min/max/mulhi/saturating ops the
//!   `epu16` instructions, and `avg` is the native rounding `vpavgw`.
//! - `ilog2p1 = 16 - lzcnt` (the widened `vplzcntd` polyfill), `abs_diff =
//!   max - min`, both ending in a maskable subtract.
//! - Everything that only depends on the bit pattern (`trailing_*`,
//!   `leading_ones`, the shifts' byte-granular cousins) delegates to the
//!   signed twin, which shares `__m512i` storage and the `KMask32` mask.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, CoreRegister, IndexableRegister, IntegerRegister, InterleaveRegister,
    NumericRegister, PartialOrdRegister, Register, Storage, UnsignedIntegerRegister, ZeroUpper, empty_reg, reg,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::i16x32::I16x32V4;
use super::kmask::KMask32;

/// Fold the two ymm halves with `$op256`, then the v3 ymm-to-scalar ladder.
macro_rules! _mm512_reduce_epi16_v4 {
    ($value:expr; $op256:ident, $op128:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let zmm = $value;
            let lo = arch::_mm512_castsi512_si256(zmm);
            let hi = arch::_mm512_extracti64x4_epi64::<1>(zmm);
            _mm256_reduce_epi16_v3!(arch::$op256(lo, hi); $op128)
        }
    }};
}

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U16x32V4;

#[thermite_macros::inline_always]
impl CoreRegister for U16x32V4 {
    type Lanes = typenum::U32;
    type Storage = arch::__m512i;
    type Mask = KMask32;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_epi16(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi16(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi16(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm512_movm_epi16(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U16x32V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_xor_si512(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_and_si512(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_andnot_si512(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_or_si512(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        bitxor(lhs: Storage<Self>, rhs: Storage<Self>);
        bitand(lhs: Storage<Self>, rhs: Storage<Self>);
        bitor(lhs: Storage<Self>, rhs: Storage<Self>);
        bitandnot(lhs: Storage<Self>, rhs: Storage<Self>);
        not(value: Storage<Self>);
        ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>);
        bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>);
    }
}

// Width ladder: zmm = two ymm. `vinserti64x4` into a zero-extended low half.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::U16x16V4> for U16x32V4 {
    fn concat(lo: Storage<super::U16x16V4>, hi: Storage<super::U16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti64x4::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U16x16V4>, Storage<super::U16x16V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti64x4_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::U16x16V4> for U16x32V4 {
    fn extend(value: Storage<super::U16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U16x16V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for U16x32V4 {
    type Element = u16;

    type Signed = I16x32V4;
    type Unsigned = U16x32V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_test_epi16_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_movepi16_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_cvtsi32_si128(value as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi16(value as i16) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi16(src, mask, ptr as *const i16) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi16(mask, ptr as *const i16) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi16(ptr as *mut i16, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_storeu_si512(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_stream_load_si512(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_stream_si512(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 32 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_epi16(indices, Self::new(padded)) }
        } else if values.len() <= 64 {
            let mut padded = [0u16; 64];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm512_loadu_si512(padded.as_ptr() as *const _);
                let hi = arch::_mm512_loadu_si512(padded.as_ptr().add(32) as *const _);
                arch::_mm512_permutex2var_epi16(lo, indices, hi)
            }
        } else {
            let idx = Self::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::reverse(value)
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::swap_bytes(value)
    }

    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            compress_v4!(u32, _mm512_maskz_compress_epi16, _mm512_mask_expand_epi16, mask, value)
        } else {
            crate::backend::generic::polyfills::compress_grouped::<Self>(value, mask)
        }
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            expand_v4!(
                u32,
                _mm512_maskz_compress_epi16,
                _mm512_mask_expand_epi16,
                _mm512_maskz_expand_epi16,
                mask,
                value
            )
        } else {
            crate::backend::generic::polyfills::expand_grouped::<Self>(value, mask)
        }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi16(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutex2var_epi16(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi16(src, mask, value.as_ptr() as *const i16) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi16(mask, value.as_ptr() as *const i16) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi16(src, mask, value as i16) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi16(mask, value as i16) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::broadcast::<I>(value)
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::broadcast_c::<I>(mask, value)
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        I16x32V4::broadcast_m::<I>(src, mask, value)
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::broadcast_z::<I>(mask, value)
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x32V4::broadcastv(value, idx)
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x32V4::broadcastv_c(mask, value, idx)
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x32V4::broadcastv_m(src, mask, value, idx)
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        I16x32V4::broadcastv_z(mask, value, idx)
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::reverse_c(mask, value)
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::reverse_m(src, mask, value)
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::reverse_z(mask, value)
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi16(mask, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi16(src, mask, arch::_mm512_permutex2var_epi16(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_epi16(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U16x32V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I16x32V4::interleave(a, b)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        I16x32V4::deinterleave(a, b)
    }
}

// No word gather/scatter in any AVX-512 extension: lane-wise defaults.
impl IndexableRegister<U16x32V4> for U16x32V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for U16x32V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_sllv_epi16(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srlv_epi16(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi16(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi16(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm512_mask_sll_epi16, _mm512_maskz_sll_epi16;
        shr => _mm512_mask_srl_epi16, _mm512_maskz_srl_epi16;
    }

    masked_shifti_v4! {
        shli => _mm512_mask_sll_epi16, _mm512_maskz_sll_epi16;
        shri => _mm512_mask_srl_epi16, _mm512_maskz_srl_epi16;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm512_mask_sllv_epi16, _mm512_maskz_sllv_epi16;
        shrv(Storage<Self::Unsigned>) => _mm512_mask_srlv_epi16, _mm512_maskz_srlv_epi16;
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rol_epi16x_v4(value, shift) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_ror_epi16x_v4(value, shift) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roli_epi16x_v4::<IMM8>(value) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rori_epi16x_v4::<IMM8>(value) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi16x_v4(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi16x_v4(value, shifts) }
    }

    // Rotates: one (masked) vpshldvw/vpshrdvw under VBMI2, the shift-or pair
    // below it. Byte shifts and the bit-reverse cascade end without a
    // maskable op.
    masked_poly_v4! {
        rol(value: Storage<Self>, shift: u32) => _mm512_mask_rol_epi16x_v4, _mm512_maskz_rol_epi16x_v4, _mm512_maskc_rol_epi16x_v4;
        ror(value: Storage<Self>, shift: u32) => _mm512_mask_ror_epi16x_v4, _mm512_maskz_ror_epi16x_v4, _mm512_maskc_ror_epi16x_v4;
        roli<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_roli_epi16x_v4, _mm512_maskz_roli_epi16x_v4, _mm512_maskc_roli_epi16x_v4;
        rori<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_rori_epi16x_v4, _mm512_maskz_rori_epi16x_v4, _mm512_maskc_rori_epi16x_v4;
        rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) => _mm512_mask_rolv_epi16x_v4, _mm512_maskz_rolv_epi16x_v4, _mm512_maskc_rolv_epi16x_v4;
        rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) => _mm512_mask_rorv_epi16x_v4, _mm512_maskz_rorv_epi16x_v4, _mm512_maskc_rorv_epi16x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U16x32V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epu16_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epu16_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epu16_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epu16_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epi16_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epi16_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U16x32V4 {
    const ZERO: Storage<Self> = reg::<Self, 32>([0; 32]);
    const ONE: Storage<Self> = reg::<Self, 32>([1; 32]);
    const TWO: Storage<Self> = reg::<Self, 32>([2; 32]);

    const MIN: Storage<Self> = reg::<Self, 32>([u16::MIN; 32]);
    const MAX: Storage<Self> = reg::<Self, 32>([u16::MAX; 32]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_min_epu16, _mm_min_epu16) as u16
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_max_epu16, _mm_max_epu16) as u16
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_add_epi16, _mm_add_epi16) as u16
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_mullo_epi16, _mm_mullo_epi16) as u16
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u16))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_epi16(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_epi16(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi16(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epu16(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epu16(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm512_mask_add_epi16, _mm512_maskz_add_epi16;
        sub => _mm512_mask_sub_epi16, _mm512_maskz_sub_epi16;
        mul => _mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16;
        min => _mm512_mask_min_epu16, _mm512_maskz_min_epu16;
        max => _mm512_mask_max_epu16, _mm512_maskz_max_epu16;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi16(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi16(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi16(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi16(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi16(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi16(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U16x32V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mulhi_epu16(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi16(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_adds_epu16(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_subs_epu16(lhs, rhs) }
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
        I16x32V4::count_ones(value)
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::count_zeros(value)
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_lzcnt_epi16x_v4(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_zeros(value)
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_ones(value)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        mullo => _mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16;
        mulhi => _mm512_mask_mulhi_epu16, _mm512_maskz_mulhi_epu16;
        saturating_add => _mm512_mask_adds_epu16, _mm512_maskz_adds_epu16;
        saturating_sub => _mm512_mask_subs_epu16, _mm512_maskz_subs_epu16;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm512_mask_popcnt_epi16x_v4, _mm512_maskz_popcnt_epi16x_v4;
    }

    // Same storage and mask as the signed twin: the bit-pattern counts
    // delegate wholesale.
    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::count_zeros_c(mask, value)
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::count_zeros_m(src, mask, value)
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::count_zeros_z(mask, value)
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_zeros_c(mask, value)
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_zeros_m(src, mask, value)
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_zeros_z(mask, value)
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_ones_c(mask, value)
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_ones_m(src, mask, value)
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        I16x32V4::trailing_ones_z(mask, value)
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U16x32V4 {
    /// `16 - lzcnt` (the widened `vplzcntd` polyfill).
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(16), Self::leading_zeros(value))
    }

    /// Native rounding average (`vpavgw`).
    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_avg_epu16(a, b) }
    }

    /// `|a - b| = max(a, b) - min(a, b)`.
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        avg => _mm512_mask_avg_epu16, _mm512_maskz_avg_epu16;
    }

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(value, mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(src, mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi16(mask, Self::splat(16), Self::leading_zeros(value)) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi16(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}
