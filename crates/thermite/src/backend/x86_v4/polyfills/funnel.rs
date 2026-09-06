//! VBMI2 (tier 2) funnel shifts as 16-bit rotates: `vpshldvw(v, v, n)` is
//! `rol`, `vpshrdvw(v, v, n)` is `ror`, both one instruction with the opmask
//! embedded, and the hardware reduces the count mod 16 like the register
//! contract. There is no `vprolw`, so the floor arm is the shift-or pair.
//!
//! Every masked form here is the `_c`-shaped merge (false lanes keep `v`):
//! `vpshldvw`'s merge operand is also its high data operand, so an arbitrary
//! `src` needs a separate merge move. The `mask_*` functions take `src` and
//! do that. The `maskc_*` functions are the one-instruction keep-`v` forms.

use crate::backend::x86::avx512f::avx512bw::*;
use crate::backend::x86::avx512f::avx512vbmi2::*;
use crate::backend::x86::avx512f::*;
use crate::backend::x86_v4::{Avx512Features, DefaultAvx512 as F};

macro_rules! funnel_word_rotates {
    ($($p:ident, $bits:literal: $v:ty, $k:ty;)*) => { paste::paste! { $(
        /// Rotate each word left by a runtime count (reduced mod 16).
        #[inline(always)]
        pub unsafe fn [<$p _rol_epi16x_v4>](v: $v, n: u32) -> $v {
            let s = n & 15;
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _shldv_epi16>](v, v, [<$p _set1_epi16>](s as i16)) }
            } else {
                unsafe {
                    [<$p _or_si $bits>](
                        [<$p _sll_epi16>](v, _mm_cvtsi32_si128(s as i32)),
                        [<$p _srl_epi16>](v, _mm_cvtsi32_si128((16 - s) as i32)),
                    )
                }
            }
        }

        /// Rotate each word right by a runtime count (reduced mod 16).
        #[inline(always)]
        pub unsafe fn [<$p _ror_epi16x_v4>](v: $v, n: u32) -> $v {
            let s = n & 15;
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _shrdv_epi16>](v, v, [<$p _set1_epi16>](s as i16)) }
            } else {
                unsafe {
                    [<$p _or_si $bits>](
                        [<$p _srl_epi16>](v, _mm_cvtsi32_si128(s as i32)),
                        [<$p _sll_epi16>](v, _mm_cvtsi32_si128((16 - s) as i32)),
                    )
                }
            }
        }

        /// Per-lane variable rotate left (counts reduced mod 16).
        #[inline(always)]
        pub unsafe fn [<$p _rolv_epi16x_v4>](v: $v, s: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _shldv_epi16>](v, v, s) }
            } else {
                unsafe {
                    let s = [<$p _and_si $bits>](s, [<$p _set1_epi16>](15));
                    [<$p _or_si $bits>](
                        [<$p _sllv_epi16>](v, s),
                        [<$p _srlv_epi16>](v, [<$p _sub_epi16>]([<$p _set1_epi16>](16), s)),
                    )
                }
            }
        }

        /// Per-lane variable rotate right (counts reduced mod 16).
        #[inline(always)]
        pub unsafe fn [<$p _rorv_epi16x_v4>](v: $v, s: $v) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _shrdv_epi16>](v, v, s) }
            } else {
                unsafe {
                    let s = [<$p _and_si $bits>](s, [<$p _set1_epi16>](15));
                    [<$p _or_si $bits>](
                        [<$p _srlv_epi16>](v, s),
                        [<$p _sllv_epi16>](v, [<$p _sub_epi16>]([<$p _set1_epi16>](16), s)),
                    )
                }
            }
        }

        /// Rotate each word left by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _roli_epi16x_v4>]<const IMM8: i32>(v: $v) -> $v {
            unsafe { [<$p _rol_epi16x_v4>](v, IMM8 as u32) }
        }

        /// Rotate each word right by a compile-time count.
        #[inline(always)]
        pub unsafe fn [<$p _rori_epi16x_v4>]<const IMM8: i32>(v: $v) -> $v {
            unsafe { [<$p _ror_epi16x_v4>](v, IMM8 as u32) }
        }

        funnel_masked_imm! { $p, $v, $k; roli => rol; rori => ror; }

        funnel_masked! { $p, $v, $k;
            rol(n: u32) => _shldv_epi16, [<$p _set1_epi16>]((n & 15) as i16);
            ror(n: u32) => _shrdv_epi16, [<$p _set1_epi16>]((n & 15) as i16);
            rolv(s: $v) => _shldv_epi16, s;
            rorv(s: $v) => _shrdv_epi16, s;
        }
    )* } };
}

macro_rules! funnel_masked {
    ($p:ident, $v:ty, $k:ty; $($name:ident ($arg:ident : $arg_ty:ty) => $fs:ident, $count:expr;)*) => { paste::paste! { $(
        /// Keep-`v` merge: one masked funnel shift.
        #[inline(always)]
        pub unsafe fn [<$p _maskc_ $name _epi16x_v4>](k: $k, v: $v, $arg: $arg_ty) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _mask $fs>](v, k, v, $count) }
            } else {
                unsafe { [<$p _mask_mov_epi16>](v, k, [<$p _ $name _epi16x_v4>](v, $arg)) }
            }
        }

        /// Arbitrary-`src` merge: the rotate plus one merge move.
        #[inline(always)]
        pub unsafe fn [<$p _mask_ $name _epi16x_v4>](src: $v, k: $k, v: $v, $arg: $arg_ty) -> $v {
            unsafe { [<$p _mask_mov_epi16>](src, k, [<$p _ $name _epi16x_v4>](v, $arg)) }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_ $name _epi16x_v4>](k: $k, v: $v, $arg: $arg_ty) -> $v {
            if const { <F as Avx512Features>::AVX512VBMI2 } {
                unsafe { [<$p _maskz $fs>](k, v, v, $count) }
            } else {
                unsafe { [<$p _maskz_mov_epi16>](k, [<$p _ $name _epi16x_v4>](v, $arg)) }
            }
        }
    )* } };
}

macro_rules! funnel_masked_imm {
    ($p:ident, $v:ty, $k:ty; $($name:ident => $rt:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        pub unsafe fn [<$p _maskc_ $name _epi16x_v4>]<const IMM8: i32>(k: $k, v: $v) -> $v {
            unsafe { [<$p _maskc_ $rt _epi16x_v4>](k, v, IMM8 as u32) }
        }

        #[inline(always)]
        pub unsafe fn [<$p _mask_ $name _epi16x_v4>]<const IMM8: i32>(src: $v, k: $k, v: $v) -> $v {
            unsafe { [<$p _mask_ $rt _epi16x_v4>](src, k, v, IMM8 as u32) }
        }

        #[inline(always)]
        pub unsafe fn [<$p _maskz_ $name _epi16x_v4>]<const IMM8: i32>(k: $k, v: $v) -> $v {
            unsafe { [<$p _maskz_ $rt _epi16x_v4>](k, v, IMM8 as u32) }
        }
    )* } };
}

funnel_word_rotates! {
    _mm512, 512: __m512i, __mmask32;
    _mm256, 256: __m256i, __mmask16;
    _mm, 128: __m128i, __mmask8;
}
