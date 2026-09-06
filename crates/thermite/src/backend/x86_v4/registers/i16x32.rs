//! `i16x32` on AVX-512: the native 512-bit signed-word register (BW is
//! floor, so every word op below is a single EVEX instruction). Follows the
//! [`i32x16`](super::i32x16) pattern file. Word-specific notes:
//!
//! - **No 16-bit masked logic ops exist** (`vpandd`/`vpxord` mask per
//!   dword), so the `BitwiseRegister` masked variants are op + one
//!   `vmovdqu16` merge. Unmasked logic still uses the dword forms (bitwise
//!   ops do not care about lane width) and `vpternlog` for `not`/`ternlog`.
//! - No `vprolw`/`vplzcntw`: rotates keep the trait's shift-or default (masked
//!   forms merge after), `leading_zeros` widens to dwords for `vplzcntd`
//!   (`_mm512_lzcnt_epi16x_v4`).
//! - No 16-bit `_mm512_reduce_*` in `core::arch`: reductions fold the two
//!   ymm halves and finish with the v3 `_mm256_reduce_epi16_v3!` ladder.
//! - `vpcompressw`/`vpexpandw` and `vpopcntw` are tier-2 (VBMI2 / BITALG).
//!   Both fork on the feature const with the generic grouped compress and the
//!   full-width nibble-LUT popcount as the floor arms.
//! - `vpermw`/`vpermi2w` (BW) give native `permutev`/`swizzle`/`lookup` up to
//!   64 entries. There is no 16-bit gather, so `IndexableRegister<u16x32>` is
//!   the lane-wise default.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::register::{
    BitshiftRegister, BitwiseRegister, CoreRegister, IndexableRegister, IntegerRegister, InterleaveRegister,
    NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper,
    empty_reg, reg, reg_splat,
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
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
pub struct I16x32V4;

#[rustfmt::skip]
const REVERSE_IDX: arch::__m512i = reg::<I16x32V4, 32>([
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
]);

#[thermite_macros::inline_always]
impl CoreRegister for I16x32V4 {
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
        if const { Z::N >= 32 } {
            value
        } else {
            unsafe { arch::_mm512_maskz_mov_epi16(const { ((1u64 << Z::N) - 1) as u32 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm512_movm_epi16(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I16x32V4 {
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
    // Logic ops mask per dword. The word mask needs a `vmovdqu16` merge.

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
impl crate::register::ConcatRegister<super::I16x16V4> for I16x32V4 {
    fn concat(lo: Storage<super::I16x16V4>, hi: Storage<super::I16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti64x4::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::I16x16V4>, Storage<super::I16x16V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti64x4_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::I16x16V4> for I16x32V4 {
    fn extend(value: Storage<super::I16x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::I16x16V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for I16x32V4 {
    type Element = i16;

    type Signed = I16x32V4;
    type Unsigned = super::U16x32V4;

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
        // Zero-extend the word so the upper word of the dword is clear.
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_cvtsi32_si128(value as u16 as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi16(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi16(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi16(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi16(ptr, mask, value) }
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

    // vpermw for up to 32 entries, vpermi2w for up to 64. There is no word
    // gather, so longer tables take the bounds-checked lane loop.
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 32 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_epi16(indices, Self::new(padded)) }
        } else if values.len() <= 64 {
            let mut padded = [0i16; 64];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm512_loadu_si512(padded.as_ptr() as *const _);
                let hi = arch::_mm512_loadu_si512(padded.as_ptr().add(32) as *const _);
                arch::_mm512_permutex2var_epi16(lo, indices, hi)
            }
        } else {
            let idx = <Self::Unsigned as Register>::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi16(REVERSE_IDX, value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm512_broadcast_i32x4(arch::_mm_setr_epi8(
                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            ));
            arch::_mm512_shuffle_epi8(value, pattern)
        }
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
        unsafe { arch::_mm512_mask_loadu_epi16(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi16(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi16(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi16(mask, value) }
    }

    // Lane broadcast: one vpermw with a constant index. Masked forms ride it.
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi16(arch::_mm512_set1_epi16(I as i16), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(value, mask, arch::_mm512_set1_epi16(I as i16), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(src, mask, arch::_mm512_set1_epi16(I as i16), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi16(mask, arch::_mm512_set1_epi16(I as i16), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi16(arch::_mm512_set1_epi16(idx as i16), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(value, mask, arch::_mm512_set1_epi16(idx as i16), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(src, mask, arch::_mm512_set1_epi16(idx as i16), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi16(mask, arch::_mm512_set1_epi16(idx as i16), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(value, mask, REVERSE_IDX, value) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi16(src, mask, REVERSE_IDX, value) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi16(mask, REVERSE_IDX, value) }
    }

    // vpshufb masks per byte. Shuffle, then one word merge.
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

    // vpermt2w merges only into `a`: permute, then one merge move. Zeroing
    // form is native.
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

#[rustfmt::skip]
const INTERLEAVE_LO: arch::__m512i = reg::<I16x32V4, 32>([
    0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
    8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
]);

#[rustfmt::skip]
const INTERLEAVE_HI: arch::__m512i = reg::<I16x32V4, 32>([
    16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
    24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63,
]);

#[rustfmt::skip]
const DEINTERLEAVE_EVEN: arch::__m512i = reg::<I16x32V4, 32>([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62,
]);

#[rustfmt::skip]
const DEINTERLEAVE_ODD: arch::__m512i = reg::<I16x32V4, 32>([
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
]);

// Two-source word permutes: one vpermi2w per output register.
#[thermite_macros::inline_always]
impl InterleaveRegister for I16x32V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm512_permutex2var_epi16(a, INTERLEAVE_LO, b),
                arch::_mm512_permutex2var_epi16(a, INTERLEAVE_HI, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm512_permutex2var_epi16(a, DEINTERLEAVE_EVEN, b),
                arch::_mm512_permutex2var_epi16(a, DEINTERLEAVE_ODD, b),
            )
        }
    }
}

// No word gather/scatter in any AVX-512 extension: lane-wise defaults.
impl IndexableRegister<super::U16x32V4> for I16x32V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for I16x32V4 {
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
impl PartialOrdRegister for I16x32V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epi16_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epi16_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epi16_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epi16_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epi16_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epi16_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I16x32V4 {
    const ZERO: Storage<Self> = reg::<Self, 32>([0; 32]);
    const ONE: Storage<Self> = reg::<Self, 32>([1; 32]);
    const TWO: Storage<Self> = reg::<Self, 32>([2; 32]);

    const MIN: Storage<Self> = reg::<Self, 32>([i16::MIN; 32]);
    const MAX: Storage<Self> = reg::<Self, 32>([i16::MAX; 32]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_min_epi16, _mm_min_epi16)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_max_epi16, _mm_max_epi16)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_add_epi16, _mm_add_epi16)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi16_v4!(value; _mm256_mullo_epi16, _mm_mullo_epi16)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::I16)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i16))
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
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epi16(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epi16(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm512_mask_add_epi16, _mm512_maskz_add_epi16;
        sub => _mm512_mask_sub_epi16, _mm512_maskz_sub_epi16;
        mul => _mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16;
        min => _mm512_mask_min_epi16, _mm512_maskz_min_epi16;
        max => _mm512_mask_max_epi16, _mm512_maskz_max_epi16;
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
impl SignedRegister for I16x32V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 32>([-1; 32]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_abs_epi16(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi16(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm512_mask_abs_epi16, _mm512_maskz_abs_epi16;
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
        unsafe { arch::_mm512_mask_mov_epi16(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi16(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I16x32V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mulhi_epi16(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi16(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_adds_epi16(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_subs_epi16(lhs, rhs) }
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
            unsafe { arch::_mm512_popcnt_epi16(value) }
        } else {
            unsafe { arch::_mm512_popcnt_epi16x_v4(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_lzcnt_epi16x_v4(value) }
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
        mullo => _mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16;
        mulhi => _mm512_mask_mulhi_epi16, _mm512_maskz_mulhi_epi16;
        saturating_add => _mm512_mask_adds_epi16, _mm512_maskz_adds_epi16;
        saturating_sub => _mm512_mask_subs_epi16, _mm512_maskz_subs_epi16;
    }

    // The divider polyfills and the widened lzcnt have no spare mask slot.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi16, movz = _mm512_maskz_mov_epi16;
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    // The popcount family ends in `count_ones`, whose masked polyfill forks
    // on BITALG exactly like the unmasked one (`bits.rs`).
    masked_unary_v4! {
        count_ones => _mm512_mask_popcnt_epi16x_v4, _mm512_maskz_popcnt_epi16x_v4;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_popcnt_epi16x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi16x_v4(mask, low) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi16x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi16x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I16x32V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sra_epi16(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sra_epi16(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srav_epi16(value, shifts) }
    }

    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mulhrs_epi16(a, b) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        sra => _mm512_mask_sra_epi16, _mm512_maskz_sra_epi16;
    }

    masked_shifti_v4! {
        srai => _mm512_mask_sra_epi16, _mm512_maskz_sra_epi16;
    }

    masked_binary_v4! {
        srav(Storage<Self::Unsigned>) => _mm512_mask_srav_epi16, _mm512_maskz_srav_epi16;
        mulhrs => _mm512_mask_mulhrs_epi16, _mm512_maskz_mulhrs_epi16;
    }

    // avg_floor = (a & b) + ((a ^ b) >>a 1), avg_ceil = (a | b) - ((a ^ b) >>a 1):
    // the final add/sub takes the mask.
    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi16(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi16(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_epi16(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi16(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi16(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }
}
