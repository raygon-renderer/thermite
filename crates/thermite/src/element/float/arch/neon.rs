//! AArch64 scalar float ops.
//!
//! AdvSIMD is mandatory in AArch64, so every rung here is unconditional. There are no
//! `s`/`d`-suffixed scalar intrinsics for these operations; the one-lane
//! (`float64x1_t`) and two-lane (`float32x2_t`) forms map to the generic
//! `llvm.sqrt`/`llvm.floor`/`llvm.fma` intrinsics, which LLVM narrows back to the
//! scalar `fsqrt`/`frintm`/`fmadd` and constant-folds.

use core::arch::aarch64::*;

macro_rules! impl_neon {
    ($t:ty, $sqrt:ident, $floor:ident, $ceil:ident, $trunc:ident, $round:ident, $fma:ident,
     [$vsqrt:ident, $vrndm:ident, $vrndp:ident, $vrnd:ident, $vrndn:ident, $vfma:ident, $dup:ident, $get:ident]) => {
        #[inline(always)]
        pub fn $sqrt(x: $t) -> $t {
            unsafe { $get::<0>($vsqrt($dup(x))) }
        }

        #[inline(always)]
        pub fn $floor(x: $t) -> $t {
            unsafe { $get::<0>($vrndm($dup(x))) } // frintm
        }

        #[inline(always)]
        pub fn $ceil(x: $t) -> $t {
            unsafe { $get::<0>($vrndp($dup(x))) } // frintp
        }

        #[inline(always)]
        pub fn $trunc(x: $t) -> $t {
            unsafe { $get::<0>($vrnd($dup(x))) } // frintz
        }

        // Ties to even, matching the vector backends (which use the `q` form of this).
        #[inline(always)]
        pub fn $round(x: $t) -> $t {
            unsafe { $get::<0>($vrndn($dup(x))) } // frintn
        }

        #[inline(always)]
        pub fn $fma(x: $t, y: $t, z: $t) -> $t {
            unsafe { $get::<0>($vfma($dup(z), $dup(x), $dup(y))) } // z + x * y
        }
    };
}

impl_neon!(f64, sqrt, floor, ceil, trunc, round, fma,
    [vsqrt_f64, vrndm_f64, vrndp_f64, vrnd_f64, vrndn_f64, vfma_f64, vdup_n_f64, vget_lane_f64]);

impl_neon!(f32, sqrtf, floorf, ceilf, truncf, roundf, fmaf,
    [vsqrt_f32, vrndm_f32, vrndp_f32, vrnd_f32, vrndn_f32, vfma_f32, vdup_n_f32, vget_lane_f32]);
