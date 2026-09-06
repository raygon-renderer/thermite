//! GFNI (tier 2) byte-lane bit ops. `vgf2p8affineqb` multiplies every byte by
//! an 8x8 bit matrix over GF(2), so any per-byte bit permutation, shift or
//! sign fill is ONE instruction with a broadcast matrix, and it takes the
//! opmask itself, so the masked variants are one instruction too. Floor arms
//! (no GFNI) are the BW word-op polyfills in [`super::bits`] plus one merge
//! move.
//!
//! Matrix convention (Intel): output bit `i` of a byte is the parity of
//! `matrix.byte[7 - i] & input`, so the identity is `0x0102040810204080` and
//! a matrix is built by placing the input-bit mask for output bit `i` at byte
//! `7 - i`. Shifts and rotates are byte shifts/rotates of the identity.

use crate::backend::x86::avx512f::avx512bw::*;
use crate::backend::x86::avx512f::gfni::*;
use crate::backend::x86::avx512f::*;
use crate::backend::x86_v4::{Avx512Features, DefaultAvx512 as F};

use super::bits::*;

/// The identity matrix: output bit `i` reads input bit `i`.
pub const GF_IDENTITY: u64 = 0x0102_0408_1020_4080;
/// Bit reversal within each byte.
pub const GF_REVERSE: u64 = 0x8040_2010_0804_0201;
/// Every output bit reads the sign bit.
const GF_SIGN: u64 = 0x8080_8080_8080_8080;

/// Logical left shift by `n` (zero for `n >= 8`, like the register contract).
#[inline(always)]
pub const fn gf_shl(n: u32) -> u64 {
    if n >= 8 { 0 } else { GF_IDENTITY >> (8 * n) }
}

/// Logical right shift by `n` (zero for `n >= 8`).
#[inline(always)]
pub const fn gf_shr(n: u32) -> u64 {
    if n >= 8 { 0 } else { GF_IDENTITY << (8 * n) }
}

/// Arithmetic right shift by `n` (sign fill for `n >= 8`).
#[inline(always)]
pub const fn gf_sra(n: u32) -> u64 {
    if n == 0 {
        GF_IDENTITY
    } else if n >= 8 {
        GF_SIGN
    } else {
        (GF_IDENTITY << (8 * n)) | (GF_SIGN >> (8 * (8 - n)))
    }
}

/// Rotate left by `n mod 8`.
#[inline(always)]
pub const fn gf_rol(n: u32) -> u64 {
    GF_IDENTITY.rotate_right(8 * (n & 7))
}

/// Rotate right by `n mod 8`.
#[inline(always)]
pub const fn gf_ror(n: u32) -> u64 {
    GF_IDENTITY.rotate_left(8 * (n & 7))
}

/// Reversed nibble table for the floor bit-reverse: `REV4[x] = reverse4(x)`.
const REV4: [i8; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

#[inline(always)]
unsafe fn lut16(hi_nibble: bool) -> __m128i {
    let s = if hi_nibble { 4 } else { 0 };
    unsafe {
        _mm_setr_epi8(
            REV4[0] << s,
            REV4[1] << s,
            REV4[2] << s,
            REV4[3] << s,
            REV4[4] << s,
            REV4[5] << s,
            REV4[6] << s,
            REV4[7] << s,
            REV4[8] << s,
            REV4[9] << s,
            REV4[10] << s,
            REV4[11] << s,
            REV4[12] << s,
            REV4[13] << s,
            REV4[14] << s,
            REV4[15] << s,
        )
    }
}

#[inline(always)]
unsafe fn _mm_bcast16_v4(x: __m128i) -> __m128i {
    x
}

macro_rules! gfni_byte_ops {
    ($($p:ident, $bits:literal: $v:ty, $k:ty, $set1_64:ident, $bcast:ident;)*) => { paste::paste! { $(
        /// `vgf2p8affineqb` with a broadcast matrix, no XOR term.
        #[inline(always)]
        pub unsafe fn [<$p _affine_v4>](v: $v, m: u64) -> $v {
            unsafe { [<$p _gf2p8affine_epi64_epi8>]::<0>(v, [<$p $set1_64>](m as i64)) }
        }

        #[inline(always)]
        pub unsafe fn [<$p _mask_affine_v4>](src: $v, k: $k, v: $v, m: u64) -> $v {
            unsafe { [<$p _mask_gf2p8affine_epi64_epi8>]::<0>(src, k, v, [<$p $set1_64>](m as i64)) }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_affine_v4>](k: $k, v: $v, m: u64) -> $v {
            unsafe { [<$p _maskz_gf2p8affine_epi64_epi8>]::<0>(k, v, [<$p $set1_64>](m as i64)) }
        }

        // --- base ops without a BW polyfill of their own --------------------

        /// Rotate each byte left by a runtime count (reduced mod 8).
        #[inline(always)]
        pub unsafe fn [<$p _rol_epi8x_v4>](v: $v, n: u32) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_rol(n)) }
            } else {
                let s = n & 7;
                unsafe { [<$p _or_si $bits>]([<$p _sll_epi8x_v4>](v, s), [<$p _srl_epi8x_v4>](v, 8 - s)) }
            }
        }

        /// Rotate each byte right by a runtime count (reduced mod 8).
        #[inline(always)]
        pub unsafe fn [<$p _ror_epi8x_v4>](v: $v, n: u32) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, gf_ror(n)) }
            } else {
                let s = n & 7;
                unsafe { [<$p _or_si $bits>]([<$p _srl_epi8x_v4>](v, s), [<$p _sll_epi8x_v4>](v, 8 - s)) }
            }
        }

        /// Rotate each byte left by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _roli_epi8x_v4>]<const IMM8: i32>(v: $v) -> $v {
            unsafe { [<$p _rol_epi8x_v4>](v, IMM8 as u32) }
        }

        /// Rotate each byte right by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _rori_epi8x_v4>]<const IMM8: i32>(v: $v) -> $v {
            unsafe { [<$p _ror_epi8x_v4>](v, IMM8 as u32) }
        }

        /// Reverse the bits of each byte. Floor arm: two nibble LUT shuffles.
        #[inline(always)]
        pub unsafe fn [<$p _reverse_epi8x_v4>](v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _affine_v4>](v, GF_REVERSE) }
            } else {
                unsafe {
                    let nib = [<$p _set1_epi8>](0x0F);
                    let lo = [<$p _and_si $bits>](v, nib);
                    let hi = [<$p _and_si $bits>]([<$p _srli_epi16>]::<4>(v), nib);
                    [<$p _or_si $bits>](
                        [<$p _shuffle_epi8>]($bcast(lut16(true)), lo),
                        [<$p _shuffle_epi8>]($bcast(lut16(false)), hi),
                    )
                }
            }
        }

        // --- masked forms: one masked affine, or the base op plus a merge ---

        gfni_byte_masked! { $p, $v, $k;
            sll(n: u32) => gf_shl(n);
            srl(n: u32) => gf_shr(n);
            sra(n: u32) => gf_sra(n);
            rol(n: u32) => gf_rol(n);
            ror(n: u32) => gf_ror(n);
            reverse() => GF_REVERSE;
        }

        gfni_byte_masked! { $p, $v, $k; @imm
            slli => gf_shl(IMM8 as u32);
            srli => gf_shr(IMM8 as u32);
            srai => gf_sra(IMM8 as u32);
            roli => gf_rol(IMM8 as u32);
            rori => gf_ror(IMM8 as u32);
        }
    )* } };
}

macro_rules! gfni_byte_masked {
    ($p:ident, $v:ty, $k:ty; $($name:ident ($($arg:ident : $arg_ty:ty)?) => $m:expr;)*) => { paste::paste! { $(
        #[inline(always)]
        pub unsafe fn [<$p _mask_ $name _epi8x_v4>](src: $v, k: $k, v: $v $(, $arg: $arg_ty)?) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _mask_affine_v4>](src, k, v, $m) }
            } else {
                unsafe { [<$p _mask_mov_epi8>](src, k, [<$p _ $name _epi8x_v4>](v $(, $arg)?)) }
            }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_ $name _epi8x_v4>](k: $k, v: $v $(, $arg: $arg_ty)?) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _maskz_affine_v4>](k, v, $m) }
            } else {
                unsafe { [<$p _maskz_mov_epi8>](k, [<$p _ $name _epi8x_v4>](v $(, $arg)?)) }
            }
        }
    )* } };
    ($p:ident, $v:ty, $k:ty; @imm $($name:ident => $m:expr;)*) => { paste::paste! { $(
        #[inline(always)]
        pub unsafe fn [<$p _mask_ $name _epi8x_v4>]<const IMM8: i32>(src: $v, k: $k, v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _mask_affine_v4>](src, k, v, $m) }
            } else {
                unsafe { [<$p _mask_mov_epi8>](src, k, [<$p _ $name _epi8x_v4>]::<IMM8>(v)) }
            }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_ $name _epi8x_v4>]<const IMM8: i32>(k: $k, v: $v) -> $v {
            if const { <F as Avx512Features>::GFNI } {
                unsafe { [<$p _maskz_affine_v4>](k, v, $m) }
            } else {
                unsafe { [<$p _maskz_mov_epi8>](k, [<$p _ $name _epi8x_v4>]::<IMM8>(v)) }
            }
        }
    )* } };
}

gfni_byte_ops! {
    _mm512, 512: __m512i, __mmask64, _set1_epi64, _mm512_broadcast_i32x4;
    _mm256, 256: __m256i, __mmask32, _set1_epi64x, _mm256_broadcastsi128_si256;
    _mm, 128: __m128i, __mmask16, _set1_epi64x, _mm_bcast16_v4;
}
