//! 512-bit bit-twiddling polyfills: the software popcount family for tier-1
//! parts (no VPOPCNTDQ/BITALG), ported from the v3 bodies at full width.
//! `vpshufb`, `vpsadbw`, `vpmaddubsw` and `vpmaddwd` all have 512-bit forms
//! under BW (floor), so these are true ports, not half-splits.

use crate::register::{BitwiseRegister, Storage};

use crate::backend::x86::avx512f::avx512bitalg::*;
use crate::backend::x86::avx512f::avx512bw::*;
use crate::backend::x86::avx512f::avx512cd::*;
use crate::backend::x86::avx512f::avx512vbmi::*;
use crate::backend::x86::avx512f::avx512vpopcntdq::*;
use crate::backend::x86::avx512f::*;
use crate::backend::x86_v2::polyfills::{
    _mm_popcnt_epi8x_v2, _mm_popcnt_epi16x_v2, _mm_popcnt_epi32x_v2, _mm_popcnt_epi64x_v2,
};
use crate::backend::x86_v3::polyfills::{
    _mm256_popcnt_epi8x_v3, _mm256_popcnt_epi16x_v3, _mm256_popcnt_epi32x_v3, _mm256_popcnt_epi64x_v3,
};

use crate::backend::x86_v4::{Avx512Features, DefaultAvx512 as F};

use super::gfni::*;

// --- masked popcounts -------------------------------------------------------
//
// The register files fork `count_ones` on `F::AVX512VPOPCNTDQ`. The masked
// variants need the same fork, so it lives once here: `vpopcnt{d,q}` takes
// the mask itself, the tier-1 LUT port gets one merge/zero move after it.

/// Merge-masked per-dword popcount.
#[inline(always)]
pub unsafe fn _mm512_mask_popcnt_epi32x_v4(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
        unsafe { _mm512_mask_popcnt_epi32(src, k, a) }
    } else {
        unsafe { _mm512_mask_mov_epi32(src, k, _mm512_popcnt_epi32x_v4(a)) }
    }
}

/// Zero-masked per-dword popcount.
#[inline(always)]
pub unsafe fn _mm512_maskz_popcnt_epi32x_v4(k: __mmask16, a: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
        unsafe { _mm512_maskz_popcnt_epi32(k, a) }
    } else {
        unsafe { _mm512_maskz_mov_epi32(k, _mm512_popcnt_epi32x_v4(a)) }
    }
}

/// Merge-masked per-qword popcount.
#[inline(always)]
pub unsafe fn _mm512_mask_popcnt_epi64x_v4(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
        unsafe { _mm512_mask_popcnt_epi64(src, k, a) }
    } else {
        unsafe { _mm512_mask_mov_epi64(src, k, _mm512_popcnt_epi64x_v4(a)) }
    }
}

/// Zero-masked per-qword popcount.
#[inline(always)]
pub unsafe fn _mm512_maskz_popcnt_epi64x_v4(k: __mmask8, a: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
        unsafe { _mm512_maskz_popcnt_epi64(k, a) }
    } else {
        unsafe { _mm512_maskz_mov_epi64(k, _mm512_popcnt_epi64x_v4(a)) }
    }
}

/// Per-byte popcount via the classic nibble LUT (Mula et al.,
/// arXiv:1611.07612), at 512-bit: the 16-byte LUT broadcast to every block.
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm512_popcnt_epi8x_v4(v: __m512i) -> __m512i {
    unsafe {
        let lookup = _mm512_broadcast_i32x4(_mm_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        ));
        let low_mask = _mm512_set1_epi8(0x0f);
        let lo = _mm512_and_si512(v, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi32::<4>(v), low_mask);
        _mm512_add_epi8(_mm512_shuffle_epi8(lookup, lo), _mm512_shuffle_epi8(lookup, hi))
    }
}

/// Per-word popcount: byte counts summed pairwise by `vpmaddubsw`.
#[inline(always)]
pub unsafe fn _mm512_popcnt_epi16x_v4(v: __m512i) -> __m512i {
    unsafe { _mm512_maddubs_epi16(_mm512_popcnt_epi8x_v4(v), _mm512_set1_epi8(1)) }
}

/// Per-dword popcount: byte counts summed by `vpmaddubsw` + `vpmaddwd`.
#[inline(always)]
pub unsafe fn _mm512_popcnt_epi32x_v4(v: __m512i) -> __m512i {
    unsafe {
        _mm512_madd_epi16(
            _mm512_maddubs_epi16(_mm512_popcnt_epi8x_v4(v), _mm512_set1_epi8(1)),
            _mm512_set1_epi16(1),
        )
    }
}

/// Per-qword popcount: `vpsadbw` sums the byte counts straight into qwords.
#[inline(always)]
pub unsafe fn _mm512_popcnt_epi64x_v4(v: __m512i) -> __m512i {
    unsafe { _mm512_sad_epu8(_mm512_popcnt_epi8x_v4(v), _mm512_setzero_si512()) }
}

/// Spread the low 16 bits of each dword lane to even bit positions, the
/// nibble-LUT Morton spread (`x86_v3/polyfills/bits.rs`) at full width.
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm512_morton2_spread_epu32x_v4(v: __m512i) -> __m512i {
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
            0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
            0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
        ));
        let c = _mm512_and_si512(v, _mm512_set1_epi32(0x0000_FFFF));
        let c = _mm512_and_si512(_mm512_or_si512(c, _mm512_slli_epi32::<8>(c)), _mm512_set1_epi32(0x00FF_00FF));
        let n = _mm512_and_si512(_mm512_or_si512(c, _mm512_slli_epi32::<4>(c)), _mm512_set1_epi32(0x0F0F_0F0F));
        _mm512_shuffle_epi8(lut, n)
    }
}

/// Per-dword-lane 2D Morton encode: low 16 bits of `x` (even positions)
/// interleaved with low 16 bits of `y` (odd).
#[inline(always)]
pub unsafe fn _mm512_morton2_epu32x_v4(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        _mm512_or_si512(
            _mm512_morton2_spread_epu32x_v4(x),
            _mm512_slli_epi32::<1>(_mm512_morton2_spread_epu32x_v4(y)),
        )
    }
}

/// Compress the even bits of each dword lane back into a contiguous low 16
/// bits, the inverse of [`_mm512_morton2_spread_epu32x_v4`].
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm512_morton2_compress_epu32x_v4(v: __m512i) -> __m512i {
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
            0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
        ));
        let e = _mm512_and_si512(v, _mm512_set1_epi32(0x5555_5555));
        let lo = _mm512_shuffle_epi8(lut, e);
        let hi = _mm512_shuffle_epi8(lut, _mm512_srli_epi32::<4>(e));
        let n = _mm512_and_si512(_mm512_or_si512(lo, _mm512_slli_epi32::<2>(hi)), _mm512_set1_epi32(0x0F0F_0F0F));
        let c = _mm512_and_si512(_mm512_or_si512(n, _mm512_srli_epi32::<4>(n)), _mm512_set1_epi32(0x00FF_00FF));
        _mm512_and_si512(_mm512_or_si512(c, _mm512_srli_epi32::<8>(c)), _mm512_set1_epi32(0x0000_FFFF))
    }
}

// --- bilog as one vpternlog -------------------------------------------------
//
// `BitwiseRegister::bilog<IMM>` is a two-input truth table (`A = 0xC`,
// `B = 0xA`). Its default is a DNF over `not`/`and`/`or`. With a native
// `vpternlog` it is ONE instruction: widen the 4-bit table to the 8-bit
// three-input table that ignores operand `c` (each bit doubled, so
// `0xC -> 0xF0`, `0xA -> 0xCC`) and pass `b` twice. A const-generic argument
// cannot be computed from `IMM` on stable (`generic_const_exprs`), hence the
// 16-arm `if const` ladder, which folds to the single arm at monomorphization.

macro_rules! bilog_arms {
    ($imm:ident => $call:ident $args:tt) => {{
        const { assert!($imm >= 0 && $imm < 16, "bilog immediate is a 4-bit truth table") };

        bilog_arms!(@arm $imm; $call $args;
            0 => 0x00, 1 => 0x03, 2 => 0x0C, 3 => 0x0F, 4 => 0x30, 5 => 0x33, 6 => 0x3C, 7 => 0x3F,
            8 => 0xC0, 9 => 0xC3, 10 => 0xCC, 11 => 0xCF, 12 => 0xF0, 13 => 0xF3, 14 => 0xFC, 15 => 0xFF);

        unreachable!()
    }};
    (@arm $imm:ident; $call:ident $args:tt; $($i:literal => $t:literal),*) => {
        $( if const { $imm == $i } { return R::$call::<$t> $args; } )*
    };
}

/// `bilog<IMM>(a, b)` as a single `vpternlog`. Only for registers with
/// `HAS_NATIVE_TERNLOG`. Elsewhere the trait default is cheaper.
#[inline(always)]
pub fn bilog_ternlog<R: BitwiseRegister, const IMM: i32>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
    bilog_arms!(IMM => ternlog(a, b, b))
}

/// [`bilog_ternlog`] keeping `a` where the mask is false.
#[inline(always)]
pub fn bilog_ternlog_c<R: BitwiseRegister, const IMM: i32>(
    mask: Storage<R::Mask>,
    a: Storage<R>,
    b: Storage<R>,
) -> Storage<R> {
    bilog_arms!(IMM => ternlog_c(mask, a, b, b))
}

/// [`bilog_ternlog`] merged with `src` where the mask is false.
#[inline(always)]
pub fn bilog_ternlog_m<R: BitwiseRegister, const IMM: i32>(
    src: Storage<R>,
    mask: Storage<R::Mask>,
    a: Storage<R>,
    b: Storage<R>,
) -> Storage<R> {
    bilog_arms!(IMM => ternlog_m(src, mask, a, b, b))
}

/// [`bilog_ternlog`] zeroed where the mask is false.
#[inline(always)]
pub fn bilog_ternlog_z<R: BitwiseRegister, const IMM: i32>(
    mask: Storage<R::Mask>,
    a: Storage<R>,
    b: Storage<R>,
) -> Storage<R> {
    bilog_arms!(IMM => ternlog_z(mask, a, b, b))
}

// --- masked popcounts at ymm/xmm (VL) ---------------------------------------
//
// Same fork as the 512-bit pair above. The tier-1 fallback is the inherited
// same-width v3/v2 LUT popcount (not a half-split).

macro_rules! masked_popcnt_vl {
    ($($mask:ident, $maskz:ident: $v:ty, $k:ty => $native_mask:ident, $native_maskz:ident, $lut:ident, $mov:ident, $movz:ident;)*) => {$(
        #[inline(always)]
        pub unsafe fn $mask(src: $v, k: $k, a: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
                unsafe { $native_mask(src, k, a) }
            } else {
                unsafe { $mov(src, k, $lut(a)) }
            }
        }

        #[inline(always)]
        pub unsafe fn $maskz(k: $k, a: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
                unsafe { $native_maskz(k, a) }
            } else {
                unsafe { $movz(k, $lut(a)) }
            }
        }
    )*};
}

masked_popcnt_vl! {
    _mm256_mask_popcnt_epi32x_v4, _mm256_maskz_popcnt_epi32x_v4: __m256i, __mmask8
        => _mm256_mask_popcnt_epi32, _mm256_maskz_popcnt_epi32, _mm256_popcnt_epi32x_v3, _mm256_mask_mov_epi32, _mm256_maskz_mov_epi32;
    _mm256_mask_popcnt_epi64x_v4, _mm256_maskz_popcnt_epi64x_v4: __m256i, __mmask8
        => _mm256_mask_popcnt_epi64, _mm256_maskz_popcnt_epi64, _mm256_popcnt_epi64x_v3, _mm256_mask_mov_epi64, _mm256_maskz_mov_epi64;
    _mm_mask_popcnt_epi32x_v4, _mm_maskz_popcnt_epi32x_v4: __m128i, __mmask8
        => _mm_mask_popcnt_epi32, _mm_maskz_popcnt_epi32, _mm_popcnt_epi32x_v2, _mm_mask_mov_epi32, _mm_maskz_mov_epi32;
    _mm_mask_popcnt_epi64x_v4, _mm_maskz_popcnt_epi64x_v4: __m128i, __mmask8
        => _mm_mask_popcnt_epi64, _mm_maskz_popcnt_epi64, _mm_popcnt_epi64x_v2, _mm_mask_mov_epi64, _mm_maskz_mov_epi64;
}

// --- 8/16-bit masked popcounts (BITALG fork) --------------------------------
//
// `vpopcnt{b,w}` are BITALG (tier 2). The floor arm is the nibble-LUT port at
// the same width plus one merge/zero move.

macro_rules! masked_popcnt_bitalg {
    ($($mask:ident, $maskz:ident: $v:ty, $k:ty => $native_mask:ident, $native_maskz:ident, $lut:ident, $mov:ident, $movz:ident;)*) => {$(
        #[inline(always)]
        pub unsafe fn $mask(src: $v, k: $k, a: $v) -> $v {
            if const { <F as Avx512Features>::AVX512BITALG } {
                unsafe { $native_mask(src, k, a) }
            } else {
                unsafe { $mov(src, k, $lut(a)) }
            }
        }

        #[inline(always)]
        pub unsafe fn $maskz(k: $k, a: $v) -> $v {
            if const { <F as Avx512Features>::AVX512BITALG } {
                unsafe { $native_maskz(k, a) }
            } else {
                unsafe { $movz(k, $lut(a)) }
            }
        }
    )*};
}

masked_popcnt_bitalg! {
    _mm512_mask_popcnt_epi16x_v4, _mm512_maskz_popcnt_epi16x_v4: __m512i, __mmask32
        => _mm512_mask_popcnt_epi16, _mm512_maskz_popcnt_epi16, _mm512_popcnt_epi16x_v4, _mm512_mask_mov_epi16, _mm512_maskz_mov_epi16;
    _mm512_mask_popcnt_epi8x_v4, _mm512_maskz_popcnt_epi8x_v4: __m512i, __mmask64
        => _mm512_mask_popcnt_epi8, _mm512_maskz_popcnt_epi8, _mm512_popcnt_epi8x_v4, _mm512_mask_mov_epi8, _mm512_maskz_mov_epi8;
    _mm256_mask_popcnt_epi16x_v4, _mm256_maskz_popcnt_epi16x_v4: __m256i, __mmask16
        => _mm256_mask_popcnt_epi16, _mm256_maskz_popcnt_epi16, _mm256_popcnt_epi16x_v3, _mm256_mask_mov_epi16, _mm256_maskz_mov_epi16;
    _mm256_mask_popcnt_epi8x_v4, _mm256_maskz_popcnt_epi8x_v4: __m256i, __mmask32
        => _mm256_mask_popcnt_epi8, _mm256_maskz_popcnt_epi8, _mm256_popcnt_epi8x_v3, _mm256_mask_mov_epi8, _mm256_maskz_mov_epi8;
    _mm_mask_popcnt_epi16x_v4, _mm_maskz_popcnt_epi16x_v4: __m128i, __mmask8
        => _mm_mask_popcnt_epi16, _mm_maskz_popcnt_epi16, _mm_popcnt_epi16x_v2, _mm_mask_mov_epi16, _mm_maskz_mov_epi16;
    _mm_mask_popcnt_epi8x_v4, _mm_maskz_popcnt_epi8x_v4: __m128i, __mmask16
        => _mm_mask_popcnt_epi8, _mm_maskz_popcnt_epi8, _mm_popcnt_epi8x_v2, _mm_mask_mov_epi8, _mm_maskz_mov_epi8;
}

// --- leading zeros below dword width ----------------------------------------

/// Per-word leading zeros: no `vplzcntw`, so widen each ymm half to dwords for
/// `vplzcntd`, subtract the 16 phantom bits, and `vpmovdw` back.
#[inline(always)]
pub unsafe fn _mm512_lzcnt_epi16x_v4(v: __m512i) -> __m512i {
    unsafe {
        let sixteen = _mm512_set1_epi32(16);
        let lo = _mm512_lzcnt_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)));
        let hi = _mm512_lzcnt_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(v)));
        let lo = _mm512_cvtepi32_epi16(_mm512_sub_epi32(lo, sixteen));
        let hi = _mm512_cvtepi32_epi16(_mm512_sub_epi32(hi, sixteen));
        _mm512_inserti64x4::<1>(_mm512_zextsi256_si512(lo), hi)
    }
}

/// Per-byte leading zeros via a nibble LUT: `lz(hi) if hi != 0 else 4 + lz(lo)`.
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm512_lzcnt_epi8x_v4(v: __m512i) -> __m512i {
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
            4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ));
        let low_mask = _mm512_set1_epi8(0x0F);
        let lo = _mm512_and_si512(v, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(v), low_mask);
        let lz_hi = _mm512_shuffle_epi8(lut, hi);
        let lz_lo = _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_set1_epi8(4));
        let hi_zero = _mm512_cmpeq_epi8_mask(hi, _mm512_setzero_si512());
        _mm512_mask_blend_epi8(hi_zero, lz_hi, lz_lo)
    }
}

// --- 512-bit byte-lane shifts and multiplies --------------------------------
//
// No 8-bit shift or multiply exists at any AVX-512 level either. Same word-op
// + mask emulation as the v3 ymm forms. Immediate counts go through the
// count-in-xmm shift forms (`::<{ IMM8 as u32 }>` is not stable Rust).
// Counts are reduced mod 8 (only bits 0..3 examined), matching the scalar oracle.

/// POLYFILL: logical left shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm512_slli_epi8x_v4<const IMM8: i32>(v: __m512i) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_shl(IMM8 as u32)) }
    } else {
        let keep = 0xFFu8.wrapping_shl(IMM8 as u32) as i8;
        unsafe { _mm512_and_si512(_mm512_sll_epi16(v, _mm_cvtsi32_si128(IMM8)), _mm512_set1_epi8(keep)) }
    }
}

/// POLYFILL: logical right shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm512_srli_epi8x_v4<const IMM8: i32>(v: __m512i) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_shr(IMM8 as u32)) }
    } else {
        let keep = (0xFFu8 >> IMM8) as i8;
        unsafe { _mm512_and_si512(_mm512_srl_epi16(v, _mm_cvtsi32_si128(IMM8)), _mm512_set1_epi8(keep)) }
    }
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm512_srai_epi8x_v4<const IMM8: i32>(v: __m512i) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_sra(IMM8 as u32)) }
    } else {
        unsafe {
            let logical = _mm512_srli_epi8x_v4::<IMM8>(v);
            let m = _mm512_set1_epi8((0x80u8 >> IMM8) as i8);
            _mm512_sub_epi8(_mm512_xor_si512(logical, m), m)
        }
    }
}

/// POLYFILL: logical left shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm512_sll_epi8x_v4(v: __m512i, shift: u32) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_shl(shift)) }
    } else {
        let keep = (0xFFu32.wrapping_shl(shift) as u8) as i8;
        unsafe {
            _mm512_and_si512(
                _mm512_sll_epi16(v, _mm_cvtsi32_si128(shift as i32)),
                _mm512_set1_epi8(keep),
            )
        }
    }
}

/// POLYFILL: logical right shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm512_srl_epi8x_v4(v: __m512i, shift: u32) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_shr(shift)) }
    } else {
        let keep = ((0xFFu32 >> shift.min(31)) as u8) as i8;
        unsafe {
            _mm512_and_si512(
                _mm512_srl_epi16(v, _mm_cvtsi32_si128(shift as i32)),
                _mm512_set1_epi8(keep),
            )
        }
    }
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm512_sra_epi8x_v4(v: __m512i, shift: u32) -> __m512i {
    if const { <F as Avx512Features>::GFNI } {
        unsafe { _mm512_affine_v4(v, gf_sra(shift)) }
    } else {
        unsafe {
            let logical = _mm512_srl_epi8x_v4(v, shift);
            let m = _mm512_set1_epi8(((0x80u32 >> shift.min(31)) as u8) as i8);
            _mm512_sub_epi8(_mm512_xor_si512(logical, m), m)
        }
    }
}

/// POLYFILL: per-lane variable logical left shift of byte lanes. The even
/// and odd bytes of each word are shifted separately with `vpsllvw` (BW), so
/// a spill out of the even byte is masked off instead of landing in its
/// neighbour.
#[inline(always)]
pub unsafe fn _mm512_sllv_epi8x_v4(v: __m512i, shifts: __m512i) -> __m512i {
    unsafe {
        let lo_mask = _mm512_set1_epi16(0x00FF);
        let s = _mm512_and_si512(shifts, _mm512_set1_epi8(7));
        let even = _mm512_sllv_epi16(_mm512_and_si512(v, lo_mask), _mm512_and_si512(s, lo_mask));
        let odd = _mm512_sllv_epi16(_mm512_andnot_si512(lo_mask, v), _mm512_srli_epi16::<8>(s));
        _mm512_or_si512(_mm512_and_si512(even, lo_mask), _mm512_andnot_si512(lo_mask, odd))
    }
}

/// POLYFILL: per-lane variable logical right shift of byte lanes.
#[inline(always)]
pub unsafe fn _mm512_srlv_epi8x_v4(v: __m512i, shifts: __m512i) -> __m512i {
    unsafe {
        let lo_mask = _mm512_set1_epi16(0x00FF);
        let s = _mm512_and_si512(shifts, _mm512_set1_epi8(7));
        let even = _mm512_srlv_epi16(_mm512_and_si512(v, lo_mask), _mm512_and_si512(s, lo_mask));
        let odd = _mm512_srlv_epi16(_mm512_andnot_si512(lo_mask, v), _mm512_srli_epi16::<8>(s));
        _mm512_or_si512(even, _mm512_andnot_si512(lo_mask, odd))
    }
}

/// POLYFILL: per-lane variable arithmetic right shift of `i8` lanes. Each
/// byte is sign-extended into its own word (`vpsllw`/`vpsraw` by 8) before
/// the `vpsravw`.
#[inline(always)]
pub unsafe fn _mm512_srav_epi8x_v4(v: __m512i, shifts: __m512i) -> __m512i {
    unsafe {
        let lo_mask = _mm512_set1_epi16(0x00FF);
        let s = _mm512_and_si512(shifts, _mm512_set1_epi8(7));
        let even = _mm512_srav_epi16(
            _mm512_srai_epi16::<8>(_mm512_slli_epi16::<8>(v)),
            _mm512_and_si512(s, lo_mask),
        );
        let odd = _mm512_srav_epi16(_mm512_srai_epi16::<8>(v), _mm512_srli_epi16::<8>(s));
        _mm512_or_si512(_mm512_and_si512(even, lo_mask), _mm512_slli_epi16::<8>(odd))
    }
}

/// POLYFILL: low 8 bits of each byte product (`a[i].wrapping_mul(b[i])`).
#[inline(always)]
pub unsafe fn _mm512_mullo_epi8x_v4(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let lo_mask = _mm512_set1_epi16(0x00FF);
        let even = _mm512_mullo_epi16(_mm512_and_si512(a, lo_mask), _mm512_and_si512(b, lo_mask));
        let odd = _mm512_mullo_epi16(_mm512_srli_epi16::<8>(a), _mm512_srli_epi16::<8>(b));
        _mm512_or_si512(_mm512_and_si512(even, lo_mask), _mm512_slli_epi16::<8>(odd))
    }
}

/// POLYFILL: high 8 bits of each signed byte product. Unpack and pack are
/// both per-128-bit-lane, so the permutation cancels.
#[inline(always)]
pub unsafe fn _mm512_mulhi_epi8x_v4(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let zero = _mm512_setzero_si512();
        let a_lo = _mm512_srai_epi16::<8>(_mm512_unpacklo_epi8(zero, a));
        let b_lo = _mm512_srai_epi16::<8>(_mm512_unpacklo_epi8(zero, b));
        let a_hi = _mm512_srai_epi16::<8>(_mm512_unpackhi_epi8(zero, a));
        let b_hi = _mm512_srai_epi16::<8>(_mm512_unpackhi_epi8(zero, b));
        let p_lo = _mm512_srai_epi16::<8>(_mm512_mullo_epi16(a_lo, b_lo));
        let p_hi = _mm512_srai_epi16::<8>(_mm512_mullo_epi16(a_hi, b_hi));
        _mm512_packs_epi16(p_lo, p_hi)
    }
}

/// POLYFILL: high 8 bits of each unsigned byte product.
#[inline(always)]
pub unsafe fn _mm512_mulhi_epu8x_v4(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let zero = _mm512_setzero_si512();
        let a_lo = _mm512_unpacklo_epi8(a, zero);
        let b_lo = _mm512_unpacklo_epi8(b, zero);
        let a_hi = _mm512_unpackhi_epi8(a, zero);
        let b_hi = _mm512_unpackhi_epi8(b, zero);
        let p_lo = _mm512_srli_epi16::<8>(_mm512_mullo_epi16(a_lo, b_lo));
        let p_hi = _mm512_srli_epi16::<8>(_mm512_mullo_epi16(a_hi, b_hi));
        _mm512_packus_epi16(p_lo, p_hi)
    }
}

/// Per-word leading zeros at ymm: widen to one zmm of dwords.
#[inline(always)]
pub unsafe fn _mm256_lzcnt_epi16x_v4(v: __m256i) -> __m256i {
    unsafe {
        let lz = _mm512_lzcnt_epi32(_mm512_cvtepu16_epi32(v));
        _mm512_cvtepi32_epi16(_mm512_sub_epi32(lz, _mm512_set1_epi32(16)))
    }
}

/// Per-word leading zeros at xmm: widen to one ymm of dwords.
#[inline(always)]
pub unsafe fn _mm_lzcnt_epi16x_v4(v: __m128i) -> __m128i {
    unsafe {
        let lz = _mm256_lzcnt_epi32(_mm256_cvtepu16_epi32(v));
        _mm256_cvtepi32_epi16(_mm256_sub_epi32(lz, _mm256_set1_epi32(16)))
    }
}

/// Per-byte leading zeros at ymm (nibble LUT).
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_lzcnt_epi8x_v4(v: __m256i) -> __m256i {
    unsafe {
        let lut = _mm256_broadcast_i32x4(_mm_setr_epi8(
            4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ));
        let low_mask = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low_mask);
        let lz_hi = _mm256_shuffle_epi8(lut, hi);
        let lz_lo = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_set1_epi8(4));
        let hi_zero = _mm256_cmpeq_epi8_mask(hi, _mm256_setzero_si256());
        _mm256_mask_blend_epi8(hi_zero, lz_hi, lz_lo)
    }
}

/// Per-byte leading zeros at xmm (nibble LUT).
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_lzcnt_epi8x_v4(v: __m128i) -> __m128i {
    unsafe {
        let lut = _mm_setr_epi8(4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        let low_mask = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(v, low_mask);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), low_mask);
        let lz_hi = _mm_shuffle_epi8(lut, hi);
        let lz_lo = _mm_add_epi8(_mm_shuffle_epi8(lut, lo), _mm_set1_epi8(4));
        let hi_zero = _mm_cmpeq_epi8_mask(hi, _mm_setzero_si128());
        _mm_mask_blend_epi8(hi_zero, lz_hi, lz_lo)
    }
}

// --- ymm/xmm byte-lane shifts and multiplies (VL) ---------------------------
//
// The 512-bit bodies above, stamped at the two narrower widths: `vpsllvw` is
// BW+VL, so the variable forms are the even/odd-byte split rather than the v3
// three-step blend cascade.

macro_rules! byte_lanes_vl {
    ($($p:ident, $bits:literal: $v:ty;)*) => { paste::paste! { $(
        /// POLYFILL: logical left shift of each byte lane by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _slli_epi8x_v4>]<const IMM8: i32>(v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_shl(IMM8 as u32)) }
            } else {
                let keep = 0xFFu8.wrapping_shl(IMM8 as u32) as i8;
                unsafe { [<$p _and_si $bits>]([<$p _sll_epi16>](v, _mm_cvtsi32_si128(IMM8)), [<$p _set1_epi8>](keep)) }
            }
        }

        /// POLYFILL: logical right shift of each byte lane by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _srli_epi8x_v4>]<const IMM8: i32>(v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_shr(IMM8 as u32)) }
            } else {
                let keep = (0xFFu8 >> IMM8) as i8;
                unsafe { [<$p _and_si $bits>]([<$p _srl_epi16>](v, _mm_cvtsi32_si128(IMM8)), [<$p _set1_epi8>](keep)) }
            }
        }

        /// POLYFILL: arithmetic right shift of each `i8` lane by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _srai_epi8x_v4>]<const IMM8: i32>(v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_sra(IMM8 as u32)) }
            } else {
                unsafe {
                    let logical = [<$p _srli_epi8x_v4>]::<IMM8>(v);
                    let m = [<$p _set1_epi8>]((0x80u8 >> IMM8) as i8);
                    [<$p _sub_epi8>]([<$p _xor_si $bits>](logical, m), m)
                }
            }
        }

        /// POLYFILL: logical left shift of each byte lane by a runtime count.
        #[inline(always)]
        pub unsafe fn [<$p _sll_epi8x_v4>](v: $v, shift: u32) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_shl(shift)) }
            } else {
                let keep = (0xFFu32.wrapping_shl(shift) as u8) as i8;
                unsafe { [<$p _and_si $bits>]([<$p _sll_epi16>](v, _mm_cvtsi32_si128(shift as i32)), [<$p _set1_epi8>](keep)) }
            }
        }

        /// POLYFILL: logical right shift of each byte lane by a runtime count.
        #[inline(always)]
        pub unsafe fn [<$p _srl_epi8x_v4>](v: $v, shift: u32) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_shr(shift)) }
            } else {
                let keep = ((0xFFu32 >> shift.min(31)) as u8) as i8;
                unsafe { [<$p _and_si $bits>]([<$p _srl_epi16>](v, _mm_cvtsi32_si128(shift as i32)), [<$p _set1_epi8>](keep)) }
            }
        }

        /// POLYFILL: arithmetic right shift of each `i8` lane by a runtime count.
        #[inline(always)]
        pub unsafe fn [<$p _sra_epi8x_v4>](v: $v, shift: u32) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_sra(shift)) }
            } else {
                unsafe {
                    let logical = [<$p _srl_epi8x_v4>](v, shift);
                    let m = [<$p _set1_epi8>](((0x80u32 >> shift.min(31)) as u8) as i8);
                    [<$p _sub_epi8>]([<$p _xor_si $bits>](logical, m), m)
                }
            }
        }

        /// POLYFILL: per-lane variable logical left shift of byte lanes.
        #[inline(always)]
        pub unsafe fn [<$p _sllv_epi8x_v4>](v: $v, shifts: $v) -> $v {
            unsafe {
                let lo_mask = [<$p _set1_epi16>](0x00FF);
                let s = [<$p _and_si $bits>](shifts, [<$p _set1_epi8>](7));
                let even = [<$p _sllv_epi16>]([<$p _and_si $bits>](v, lo_mask), [<$p _and_si $bits>](s, lo_mask));
                let odd = [<$p _sllv_epi16>]([<$p _andnot_si $bits>](lo_mask, v), [<$p _srli_epi16>]::<8>(s));
                [<$p _or_si $bits>]([<$p _and_si $bits>](even, lo_mask), [<$p _andnot_si $bits>](lo_mask, odd))
            }
        }

        /// POLYFILL: per-lane variable logical right shift of byte lanes.
        #[inline(always)]
        pub unsafe fn [<$p _srlv_epi8x_v4>](v: $v, shifts: $v) -> $v {
            unsafe {
                let lo_mask = [<$p _set1_epi16>](0x00FF);
                let s = [<$p _and_si $bits>](shifts, [<$p _set1_epi8>](7));
                let even = [<$p _srlv_epi16>]([<$p _and_si $bits>](v, lo_mask), [<$p _and_si $bits>](s, lo_mask));
                let odd = [<$p _srlv_epi16>]([<$p _andnot_si $bits>](lo_mask, v), [<$p _srli_epi16>]::<8>(s));
                [<$p _or_si $bits>](even, [<$p _andnot_si $bits>](lo_mask, odd))
            }
        }

        /// POLYFILL: per-lane variable arithmetic right shift of `i8` lanes.
        #[inline(always)]
        pub unsafe fn [<$p _srav_epi8x_v4>](v: $v, shifts: $v) -> $v {
            unsafe {
                let lo_mask = [<$p _set1_epi16>](0x00FF);
                let s = [<$p _and_si $bits>](shifts, [<$p _set1_epi8>](7));
                let even = [<$p _srav_epi16>]([<$p _srai_epi16>]::<8>([<$p _slli_epi16>]::<8>(v)), [<$p _and_si $bits>](s, lo_mask));
                let odd = [<$p _srav_epi16>]([<$p _srai_epi16>]::<8>(v), [<$p _srli_epi16>]::<8>(s));
                [<$p _or_si $bits>]([<$p _and_si $bits>](even, lo_mask), [<$p _slli_epi16>]::<8>(odd))
            }
        }

        /// POLYFILL: low 8 bits of each byte product.
        #[inline(always)]
        pub unsafe fn [<$p _mullo_epi8x_v4>](a: $v, b: $v) -> $v {
            unsafe {
                let lo_mask = [<$p _set1_epi16>](0x00FF);
                let even = [<$p _mullo_epi16>]([<$p _and_si $bits>](a, lo_mask), [<$p _and_si $bits>](b, lo_mask));
                let odd = [<$p _mullo_epi16>]([<$p _srli_epi16>]::<8>(a), [<$p _srli_epi16>]::<8>(b));
                [<$p _or_si $bits>]([<$p _and_si $bits>](even, lo_mask), [<$p _slli_epi16>]::<8>(odd))
            }
        }

        /// POLYFILL: high 8 bits of each signed byte product.
        #[inline(always)]
        pub unsafe fn [<$p _mulhi_epi8x_v4>](a: $v, b: $v) -> $v {
            unsafe {
                let zero = [<$p _setzero_si $bits>]();
                let a_lo = [<$p _srai_epi16>]::<8>([<$p _unpacklo_epi8>](zero, a));
                let b_lo = [<$p _srai_epi16>]::<8>([<$p _unpacklo_epi8>](zero, b));
                let a_hi = [<$p _srai_epi16>]::<8>([<$p _unpackhi_epi8>](zero, a));
                let b_hi = [<$p _srai_epi16>]::<8>([<$p _unpackhi_epi8>](zero, b));
                let p_lo = [<$p _srai_epi16>]::<8>([<$p _mullo_epi16>](a_lo, b_lo));
                let p_hi = [<$p _srai_epi16>]::<8>([<$p _mullo_epi16>](a_hi, b_hi));
                [<$p _packs_epi16>](p_lo, p_hi)
            }
        }

        /// POLYFILL: high 8 bits of each unsigned byte product.
        #[inline(always)]
        pub unsafe fn [<$p _mulhi_epu8x_v4>](a: $v, b: $v) -> $v {
            unsafe {
                let zero = [<$p _setzero_si $bits>]();
                let a_lo = [<$p _unpacklo_epi8>](a, zero);
                let b_lo = [<$p _unpacklo_epi8>](b, zero);
                let a_hi = [<$p _unpackhi_epi8>](a, zero);
                let b_hi = [<$p _unpackhi_epi8>](b, zero);
                let p_lo = [<$p _srli_epi16>]::<8>([<$p _mullo_epi16>](a_lo, b_lo));
                let p_hi = [<$p _srli_epi16>]::<8>([<$p _mullo_epi16>](a_hi, b_hi));
                [<$p _packus_epi16>](p_lo, p_hi)
            }
        }
    )* } };
}

byte_lanes_vl! {
    _mm256, 256: __m256i;
    _mm, 128: __m128i;
}

/// `vpermb` at ymm with a floor arm: widen bytes and indices to one zmm of
/// words, `vpermw`, narrow. `result[i] = value[idx[i] & 31]`.
#[inline(always)]
pub unsafe fn _mm256_permutexvar_epi8x_v4(idx: __m256i, v: __m256i) -> __m256i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm256_permutexvar_epi8(idx, v) }
    } else {
        unsafe {
            _mm512_cvtepi16_epi8(_mm512_permutexvar_epi16(
                _mm512_cvtepu8_epi16(idx),
                _mm512_cvtepu8_epi16(v),
            ))
        }
    }
}

/// `vpermi2b` at ymm with a floor arm: both sources widened to words, one
/// 64-entry `vpermi2w`, narrow.
#[inline(always)]
pub unsafe fn _mm256_permutex2var_epi8x_v4(a: __m256i, idx: __m256i, b: __m256i) -> __m256i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm256_permutex2var_epi8(a, idx, b) }
    } else {
        unsafe {
            _mm512_cvtepi16_epi8(_mm512_permutex2var_epi16(
                _mm512_cvtepu8_epi16(a),
                _mm512_cvtepu8_epi16(idx),
                _mm512_cvtepu8_epi16(b),
            ))
        }
    }
}

/// `vpermb` at xmm with a floor arm: widen to one ymm of words, `vpermw`, narrow.
#[inline(always)]
pub unsafe fn _mm_permutexvar_epi8x_v4(idx: __m128i, v: __m128i) -> __m128i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm_permutexvar_epi8(idx, v) }
    } else {
        unsafe {
            _mm256_cvtepi16_epi8(_mm256_permutexvar_epi16(
                _mm256_cvtepu8_epi16(idx),
                _mm256_cvtepu8_epi16(v),
            ))
        }
    }
}

/// `vpermi2b` at xmm with a floor arm: one 32-entry `vpermi2w` over the
/// widened sources, narrow.
#[inline(always)]
pub unsafe fn _mm_permutex2var_epi8x_v4(a: __m128i, idx: __m128i, b: __m128i) -> __m128i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm_permutex2var_epi8(a, idx, b) }
    } else {
        unsafe {
            _mm256_cvtepi16_epi8(_mm256_permutex2var_epi16(
                _mm256_cvtepu8_epi16(a),
                _mm256_cvtepu8_epi16(idx),
                _mm256_cvtepu8_epi16(b),
            ))
        }
    }
}

// --- byte permutes below VBMI -----------------------------------------------

/// `vpermb` (VBMI, tier 2) with a floor arm: widen bytes and indices to words
/// and run the 64-entry `vpermi2w` (BW) over the two word halves, then narrow.
/// `result[i] = value[idx[i] & 63]`.
#[inline(always)]
pub unsafe fn _mm512_permutexvar_epi8x_v4(idx: __m512i, v: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm512_permutexvar_epi8(idx, v) }
    } else {
        unsafe {
            let v_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v));
            let v_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v));
            let i_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(idx));
            let i_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(idx));
            let r_lo = _mm512_cvtepi16_epi8(_mm512_permutex2var_epi16(v_lo, i_lo, v_hi));
            let r_hi = _mm512_cvtepi16_epi8(_mm512_permutex2var_epi16(v_lo, i_hi, v_hi));
            _mm512_inserti64x4::<1>(_mm512_zextsi256_si512(r_lo), r_hi)
        }
    }
}

/// `vpermi2b` (VBMI) with a floor arm: one [`_mm512_permutexvar_epi8x_v4`]
/// per source and a byte blend on index bit 6.
#[inline(always)]
pub unsafe fn _mm512_permutex2var_epi8x_v4(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    if const { <F as Avx512Features>::AVX512VBMI } {
        unsafe { _mm512_permutex2var_epi8(a, idx, b) }
    } else {
        unsafe {
            let from_a = _mm512_permutexvar_epi8x_v4(idx, a);
            let from_b = _mm512_permutexvar_epi8x_v4(idx, b);
            // Bit 6 -> MSB, which vpmovb2m reads.
            let use_b = _mm512_movepi8_mask(_mm512_add_epi8(idx, idx));
            _mm512_mask_blend_epi8(use_b, from_a, from_b)
        }
    }
}

// --- masked byte permutes -----------------------------------------------------
//
// `vpermb`/`vpermi2b` take the opmask (VBMI). Below VBMI the permute is the
// widen-`vpermw`-narrow sequence above plus one merge move.

macro_rules! masked_byte_permutes {
    ($($p:ident: $v:ty, $k:ty;)*) => { paste::paste! { $(
        #[inline(always)]
        pub unsafe fn [<$p _mask_permutexvar_epi8x_v4>](src: $v, k: $k, idx: $v, v: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI } {
                unsafe { [<$p _mask_permutexvar_epi8>](src, k, idx, v) }
            } else {
                unsafe { [<$p _mask_mov_epi8>](src, k, [<$p _permutexvar_epi8x_v4>](idx, v)) }
            }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_permutexvar_epi8x_v4>](k: $k, idx: $v, v: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI } {
                unsafe { [<$p _maskz_permutexvar_epi8>](k, idx, v) }
            } else {
                unsafe { [<$p _maskz_mov_epi8>](k, [<$p _permutexvar_epi8x_v4>](idx, v)) }
            }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_permutex2var_epi8x_v4>](k: $k, a: $v, idx: $v, b: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI } {
                unsafe { [<$p _maskz_permutex2var_epi8>](k, a, idx, b) }
            } else {
                unsafe { [<$p _maskz_mov_epi8>](k, [<$p _permutex2var_epi8x_v4>](a, idx, b)) }
            }
        }
    )* } };
}

masked_byte_permutes! {
    _mm512: __m512i, __mmask64;
    _mm256: __m256i, __mmask32;
    _mm: __m128i, __mmask16;
}
