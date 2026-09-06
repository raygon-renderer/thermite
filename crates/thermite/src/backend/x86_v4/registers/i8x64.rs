//! `i8x64` on AVX-512: the native 512-bit signed-byte register (BW floor).
//! Follows [`i16x32`](super::i16x32). Byte-specific notes:
//!
//! - x86 has no byte shift or multiply at any level: `_mm512_*_epi8x_v4`
//!   polyfills (word ops + masks, `bits.rs`) do them, so every shift/multiply
//!   masked variant is op + one `vmovdqu8` merge.
//! - `vpermb`/`vpermi2b` are VBMI (tier 2): `permutev`/`swizzle`/`lookup`/
//!   `reverse`/`broadcast` go through `_mm512_permutex{,2}var_epi8x_v4`, which
//!   forks on the feature const with a widen-to-words `vpermi2w` floor arm.
//!   The masked forms merge after the polyfill on both arms.
//! - `vpcompressb` is VBMI2, `vpopcntb` is BITALG: same fork shape as the
//!   word register.
//! - `swap_bytes` is the identity, `leading_zeros` is a nibble LUT.

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
use super::kmask::KMask64;

/// Fold the two ymm halves with `$op256`, then the v3 ymm-to-scalar ladder.
macro_rules! _mm512_reduce_epi8_v4 {
    ($value:expr; $op256:ident, $op128:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let zmm = $value;
            let lo = arch::_mm512_castsi512_si256(zmm);
            let hi = arch::_mm512_extracti64x4_epi64::<1>(zmm);
            _mm256_reduce_epi8_v3!(arch::$op256(lo, hi); $op128)
        }
    }};
}

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I8x64V4;

const fn reverse_idx() -> [i8; 64] {
    let mut a = [0i8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = (63 - i) as i8;
        i += 1;
    }
    a
}

/// `[off, 64+off, off+1, 65+off, ..]`: the two-source interleave index for
/// output half `off / 32`.
const fn interleave_idx(off: usize) -> [i8; 64] {
    let mut a = [0i8; 64];
    let mut i = 0;
    while i < 32 {
        a[2 * i] = (off + i) as i8;
        a[2 * i + 1] = (64 + off + i) as i8;
        i += 1;
    }
    a
}

/// `[p, p+2, p+4, ..]` over both sources: even (`p = 0`) or odd (`p = 1`) lanes.
const fn deinterleave_idx(p: usize) -> [i8; 64] {
    let mut a = [0i8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = (2 * i + p) as i8;
        i += 1;
    }
    a
}

pub(super) const REVERSE_IDX: arch::__m512i = reg::<I8x64V4, 64>(reverse_idx());
const INTERLEAVE_LO: arch::__m512i = reg::<I8x64V4, 64>(interleave_idx(0));
const INTERLEAVE_HI: arch::__m512i = reg::<I8x64V4, 64>(interleave_idx(32));
const DEINTERLEAVE_EVEN: arch::__m512i = reg::<I8x64V4, 64>(deinterleave_idx(0));
const DEINTERLEAVE_ODD: arch::__m512i = reg::<I8x64V4, 64>(deinterleave_idx(1));

#[thermite_macros::inline_always]
impl CoreRegister for I8x64V4 {
    type Lanes = typenum::U64;
    type Storage = arch::__m512i;
    type Mask = KMask64;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_epi8(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi8(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi8(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 64 } {
            value
        } else {
            unsafe { arch::_mm512_maskz_mov_epi8(const { (1u64 << Z::N) - 1 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm512_movm_epi8(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I8x64V4 {
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
    // Logic ops mask per dword. The byte mask needs a `vmovdqu8` merge.

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
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
impl crate::register::ConcatRegister<super::I8x32V4> for I8x64V4 {
    fn concat(lo: Storage<super::I8x32V4>, hi: Storage<super::I8x32V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti64x4::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::I8x32V4>, Storage<super::I8x32V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti64x4_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::I8x32V4> for I8x64V4 {
    fn extend(value: Storage<super::I8x32V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::I8x32V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for I8x64V4 {
    type Element = i8;

    type Signed = I8x64V4;
    type Unsigned = super::U8x64V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_test_epi8_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_movepi8_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_cvtsi32_si128(value as u8 as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi8(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi8(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi8(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi8(ptr, mask, value) }
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

    // One byte permute for up to 64 entries, the two-source form up to 128.
    // Longer tables take the bounds-checked lane loop.
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 64 {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_epi8x_v4(indices, Self::new(padded)) }
        } else if values.len() <= 128 {
            let mut padded = [0i8; 128];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let lo = arch::_mm512_loadu_si512(padded.as_ptr() as *const _);
                let hi = arch::_mm512_loadu_si512(padded.as_ptr().add(64) as *const _);
                arch::_mm512_permutex2var_epi8x_v4(lo, indices, hi)
            }
        } else {
            let idx = <Self::Unsigned as Register>::as_slice(&indices);
            Self::new(GenericArray::generate(|i| values[idx[i] as usize]))
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi8x_v4(REVERSE_IDX, value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        value
    }

    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            compress_v4!(u64, _mm512_maskz_compress_epi8, _mm512_mask_expand_epi8, mask, value)
        } else {
            crate::backend::generic::polyfills::compress_grouped::<Self>(value, mask)
        }
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VBMI2 } {
            expand_v4!(
                u64,
                _mm512_maskz_compress_epi8,
                _mm512_mask_expand_epi8,
                _mm512_maskz_expand_epi8,
                mask,
                value
            )
        } else {
            crate::backend::generic::polyfills::expand_grouped::<Self>(value, mask)
        }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi8x_v4(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutex2var_epi8x_v4(a, idxs, b) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi8x_v4(arch::_mm512_set1_epi8(I as i8), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi8x_v4(arch::_mm512_set1_epi8(idx as i8), value) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi8(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi8(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi8(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi8(mask, value) }
    }

    // The byte permutes are feature-forked polyfills with no masked form on
    // the floor arm: permute, then one byte merge.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
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
        unsafe { arch::_mm512_mask_permutexvar_epi8x_v4(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi8x_v4(mask, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi8(src, mask, Self::swizzle(a, b, idxs)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_epi8x_v4(mask, a, idxs, b) }
    }
}

// Two-source byte permutes: one (forked) vpermi2b per output register.
#[thermite_macros::inline_always]
impl InterleaveRegister for I8x64V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm512_permutex2var_epi8x_v4(a, INTERLEAVE_LO, b),
                arch::_mm512_permutex2var_epi8x_v4(a, INTERLEAVE_HI, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm512_permutex2var_epi8x_v4(a, DEINTERLEAVE_EVEN, b),
                arch::_mm512_permutex2var_epi8x_v4(a, DEINTERLEAVE_ODD, b),
            )
        }
    }
}

// No byte gather/scatter: lane-wise defaults.
impl IndexableRegister<super::U8x64V4> for I8x64V4 {}

#[thermite_macros::inline_always]
impl BitshiftRegister for I8x64V4 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi8x_v4(value, shift) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi8x_v4(value, shift) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_sllv_epi8x_v4(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srlv_epi8x_v4(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_slli_epi8x_v4::<IMM8>(value) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_srli_epi8x_v4::<IMM8>(value) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rol_epi8x_v4(value, shift) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_ror_epi8x_v4(value, shift) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roli_epi8x_v4::<IMM8>(value) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rori_epi8x_v4::<IMM8>(value) }
    }

    fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_reverse_epi8x_v4(value) }
    }

    // --- masked variants -----------------------------------------------------
    // One masked vgf2p8affineqb under GFNI. Below it the BW polyfill plus a merge.

    masked_poly_v4! {
        shl(value: Storage<Self>, shift: u32) => _mm512_mask_sll_epi8x_v4, _mm512_maskz_sll_epi8x_v4;
        shr(value: Storage<Self>, shift: u32) => _mm512_mask_srl_epi8x_v4, _mm512_maskz_srl_epi8x_v4;
        shli<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_slli_epi8x_v4, _mm512_maskz_slli_epi8x_v4;
        shri<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_srli_epi8x_v4, _mm512_maskz_srli_epi8x_v4;
        rol(value: Storage<Self>, shift: u32) => _mm512_mask_rol_epi8x_v4, _mm512_maskz_rol_epi8x_v4;
        ror(value: Storage<Self>, shift: u32) => _mm512_mask_ror_epi8x_v4, _mm512_maskz_ror_epi8x_v4;
        roli<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_roli_epi8x_v4, _mm512_maskz_roli_epi8x_v4;
        rori<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_rori_epi8x_v4, _mm512_maskz_rori_epi8x_v4;
        reverse_bits(value: Storage<Self>) => _mm512_mask_reverse_epi8x_v4, _mm512_maskz_reverse_epi8x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
        shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I8x64V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epi8_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epi8_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epi8_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epi8_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epi8_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epi8_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I8x64V4 {
    const ZERO: Storage<Self> = reg::<Self, 64>([0; 64]);
    const ONE: Storage<Self> = reg::<Self, 64>([1; 64]);
    const TWO: Storage<Self> = reg::<Self, 64>([2; 64]);

    const MIN: Storage<Self> = reg::<Self, 64>([i8::MIN; 64]);
    const MAX: Storage<Self> = reg::<Self, 64>([i8::MAX; 64]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi8_v4!(value; _mm256_min_epi8, _mm_min_epi8)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi8_v4!(value; _mm256_max_epi8, _mm_max_epi8)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm512_reduce_epi8_v4!(value; _mm256_add_epi8, _mm_add_epi8)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        // No byte multiply, so a scalar fold (rarely used at byte width).
        Self::as_slice(&value).iter().copied().fold(1i8, i8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::I8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i8))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_epi8(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_epi8(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi8x_v4(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epi8(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epi8(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm512_mask_add_epi8, _mm512_maskz_add_epi8;
        sub => _mm512_mask_sub_epi8, _mm512_maskz_sub_epi8;
        min => _mm512_mask_min_epi8, _mm512_maskz_min_epi8;
        max => _mm512_mask_max_epi8, _mm512_maskz_max_epi8;
    }

    // Multiply is a polyfill, division lane-wise: one merge each.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
        mul(lhs: Storage<Self>, rhs: Storage<Self>);
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
        square(lhs: Storage<Self>);
        scale(value: Storage<Self>, scalar: Self::Element);
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I8x64V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 64>([-1; 64]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_abs_epi8(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi8(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi8(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi8(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm512_mask_abs_epi8, _mm512_maskz_abs_epi8;
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
        unsafe { arch::_mm512_mask_mov_epi8(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi8(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I8x64V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mulhi_epi8x_v4(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi8x_v4(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_adds_epi8(lhs, rhs) }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_subs_epi8(lhs, rhs) }
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
            unsafe { arch::_mm512_popcnt_epi8(value) }
        } else {
            unsafe { arch::_mm512_popcnt_epi8x_v4(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_lzcnt_epi8x_v4(value) }
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
        saturating_add => _mm512_mask_adds_epi8, _mm512_maskz_adds_epi8;
        saturating_sub => _mm512_mask_subs_epi8, _mm512_maskz_subs_epi8;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
        mullo(lhs: Storage<Self>, rhs: Storage<Self>);
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
        leading_zeros(value: Storage<Self>);
        leading_ones(value: Storage<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm512_mask_popcnt_epi8x_v4, _mm512_maskz_popcnt_epi8x_v4;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_popcnt_epi8x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi8x_v4(mask, low) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi8x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi8x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I8x64V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_srai_epi8x_v4::<IMM8>(value) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sra_epi8x_v4(value, shift) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srav_epi8x_v4(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_poly_v4! {
        srai<const IMM8: i32>(value: Storage<Self>) => _mm512_mask_srai_epi8x_v4, _mm512_maskz_srai_epi8x_v4;
        sra(value: Storage<Self>, shift: u32) => _mm512_mask_sra_epi8x_v4, _mm512_maskz_sra_epi8x_v4;
    }

    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi8, movz = _mm512_maskz_mov_epi8;
        srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>);
        mulhrs(a: Storage<Self>, b: Storage<Self>);
    }

    // avg_floor = (a & b) + ((a ^ b) >>a 1), avg_ceil = (a | b) - ((a ^ b) >>a 1):
    // the final add/sub takes the mask.
    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi8(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi8(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_epi8(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi8(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi8(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi8(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }
}
