//! `i8x16` on AVX-512: the 128-bit signed-byte register under EVEX (BW+VL).
//! Same pattern as [`i8x64`](super::i8x64) (read its module docs for the
//! byte rules). Width differences: `KMask16`, the v2 `_mm_reduce_epi8_v2!`
//! fold, `pshufb`/`punpck` for `reverse`/`interleave`, and the xmm
//! byte-permute polyfills (`vpermb` under VBMI, widen-to-words floor arm).
//! The `Simd` grid's `i8x16` slot lives here: i16x16 / i32x16 / i64x16 casts
//! (all `vpmov*`). The `i8x8`/`i8x4` rungs are `half8.rs`.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
    InterleaveRegister, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage,
    WideRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg, reg_splat,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask16;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I8x16V4;

#[rustfmt::skip]
const REVERSE_IDX: arch::__m128i = reg::<I8x16V4, 16>([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
#[rustfmt::skip]
const DEINTERLEAVE_EVEN: arch::__m128i = reg::<I8x16V4, 16>([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
#[rustfmt::skip]
const DEINTERLEAVE_ODD: arch::__m128i = reg::<I8x16V4, 16>([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]);

#[thermite_macros::inline_always]
impl CoreRegister for I8x16V4 {
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
        if const { Z::N >= 16 } {
            value
        } else {
            unsafe { arch::_mm_maskz_mov_epi8(const { ((1u32 << Z::N) - 1) as u16 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi8(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I8x16V4 {
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
impl WideRegister for I8x16V4 {
    type Wide = super::I8x32V4;
}

#[thermite_macros::inline_always]
impl Register for I8x16V4 {
    type Element = i8;

    type Signed = I8x16V4;
    type Unsigned = super::U8x16V4;

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
        unsafe { arch::_mm_cvtsi32_si128(value as u8 as i32) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi8(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi8(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi8(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_epi8(ptr, mask, value) }
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

    // One `pshufb` for a 16-entry table (in-range indices have bit 7 clear),
    // the two-source polyfill up to 32, then the bounds-checked loop.
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 16 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm_shuffle_epi8(Self::new(padded), indices) }
        } else if values.len() <= 32 {
            let mut padded = [0i8; 32];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm_loadu_si128(padded.as_ptr() as *const _);
                let hi = arch::_mm_loadu_si128(padded.as_ptr().add(16) as *const _);
                arch::_mm_permutex2var_epi8x_v4(lo, indices, hi)
            }
        } else {
            let idx = <Self::Unsigned as Register>::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi8(value, REVERSE_IDX) }
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

    // `pshufb` is the 16-entry byte permute on every x86 level.
    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi8(value, idxs) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi8x_v4(a, idxs, b) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_set1_epi8(I as i8)) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi8(value, arch::_mm_set1_epi8(idx as i8)) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi8(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi8(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_set1_epi8(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_set1_epi8(mask, value) }
    }

    // vpshufb has EVEX masked forms (BW+VL): the pshufb-based ops take the
    // mask directly.
    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(value, mask, value, arch::_mm_set1_epi8(I as i8)) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(src, mask, value, arch::_mm_set1_epi8(I as i8)) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi8(mask, value, arch::_mm_set1_epi8(I as i8)) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(value, mask, value, arch::_mm_set1_epi8(idx as i8)) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(src, mask, value, arch::_mm_set1_epi8(idx as i8)) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi8(mask, value, arch::_mm_set1_epi8(idx as i8)) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(value, mask, value, REVERSE_IDX) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi8(src, mask, value, REVERSE_IDX) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi8(mask, value, REVERSE_IDX) }
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
impl InterleaveRegister for I8x16V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi8(a, b), arch::_mm_unpackhi_epi8(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm_permutex2var_epi8x_v4(a, DEINTERLEAVE_EVEN, b),
                arch::_mm_permutex2var_epi8x_v4(a, DEINTERLEAVE_ODD, b),
            )
        }
    }
}

// No byte gather/scatter: lane-wise defaults for every index width.
impl IndexableRegister<super::U8x16V4> for I8x16V4 {}
impl IndexableRegister<super::U32x16V4> for I8x16V4 {}
impl IndexableRegister<ArrayRegister<super::U64x8V4, 2>> for I8x16V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for I8x16V4 {
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
impl PartialOrdRegister for I8x16V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epi8_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpge_epi8_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmplt_epi8_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmple_epi8_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpeq_epi8_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpneq_epi8_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I8x16V4 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([i8::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([i8::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_min_epi8)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_max_epi8)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi8_v2!(value; _mm_add_epi8)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        Self::as_slice(&value).iter().copied().fold(1i8, i8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::I8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i8))
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
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi8(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi8(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_epi8, _mm_maskz_add_epi8;
        sub => _mm_mask_sub_epi8, _mm_maskz_sub_epi8;
        min => _mm_mask_min_epi8, _mm_maskz_min_epi8;
        max => _mm_mask_max_epi8, _mm_maskz_max_epi8;
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
impl SignedRegister for I8x16V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 16>([-1; 16]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi8(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi8(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm_mask_abs_epi8, _mm_maskz_abs_epi8;
    }

    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(mask & Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi8(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi8(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I8x16V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mulhi_epi8x_v4(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi8x_v4(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_adds_epi8(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_subs_epi8(lhs, rhs) }
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

    const HAS_HARDWARE_POPCNT: bool = <F as Avx512Features>::AVX512BITALG;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512BITALG } {
            unsafe { arch::_mm_popcnt_epi8(value) }
        } else {
            unsafe { arch::_mm_popcnt_epi8x_v2(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_lzcnt_epi8x_v4(value) }
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

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        saturating_add => _mm_mask_adds_epi8, _mm_maskz_adds_epi8;
        saturating_sub => _mm_mask_subs_epi8, _mm_maskz_subs_epi8;
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
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_popcnt_epi8x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi8x_v4(mask, low) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi8x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi8x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I8x16V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srai_epi8x_v4::<IMM8>(value) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi8x_v4(value, shift) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srav_epi8x_v4(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_poly_v4! {
        srai<const IMM8: i32>(value: Storage<Self>) => _mm_mask_srai_epi8x_v4, _mm_maskz_srai_epi8x_v4;
        sra(value: Storage<Self>, shift: u32) => _mm_mask_sra_epi8x_v4, _mm_maskz_sra_epi8x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi8, movz = _mm_maskz_mov_epi8;
        srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        mulhrs(a: Storage<Self>, b: Storage<Self>);
    }

    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi8(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi8(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_epi8(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi8(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi8(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }
}

// --- Simd-grid casts: i8x16 <-> i16x16 / i32x16 / i64x16, all vpmov forms. ---

#[thermite_macros::inline_always]
impl CastRegister<I8x16V4> for super::I16x16V4 {
    fn cast_from(value: Storage<I8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi8_epi16(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I16x16V4> for I8x16V4 {
    fn cast_from(value: Storage<super::I16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi16_epi8(value) }
    }

    fn saturating_cast_from(value: Storage<super::I16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtsepi16_epi8(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I8x16V4> for super::I32x16V4 {
    fn cast_from(value: Storage<I8x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi8_epi32(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I32x16V4> for I8x16V4 {
    fn cast_from(value: Storage<super::I32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi32_epi8(value) }
    }

    fn saturating_cast_from(value: Storage<super::I32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtsepi32_epi8(value) }
    }
}

// vpmovsxbq reads the low 8 bytes of its xmm source. The high 8 bytes move
// down with one punpckhqdq. The narrows land in the low 8 bytes of an xmm.
#[thermite_macros::inline_always]
impl CastRegister<I8x16V4> for ArrayRegister<super::I64x8V4, 2> {
    fn cast_from(value: Storage<I8x16V4>) -> Storage<Self> {
        unsafe {
            let hi = arch::_mm_unpackhi_epi64(value, value);
            ArrayRegister([arch::_mm512_cvtepi8_epi64(value), arch::_mm512_cvtepi8_epi64(hi)])
        }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::I64x8V4, 2>> for I8x16V4 {
    fn cast_from(value: Storage<ArrayRegister<super::I64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm_unpacklo_epi64(arch::_mm512_cvtepi64_epi8(lo), arch::_mm512_cvtepi64_epi8(hi)) }
    }

    fn saturating_cast_from(value: Storage<ArrayRegister<super::I64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;
        unsafe { arch::_mm_unpacklo_epi64(arch::_mm512_cvtsepi64_epi8(lo), arch::_mm512_cvtsepi64_epi8(hi)) }
    }
}
