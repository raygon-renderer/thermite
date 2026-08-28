//! WASM scalar float ops.
//!
//! SIMD128's lane-0 forms, which cost only a splat and an extract that LLVM usually
//! folds away. The scalar numeric instructions (`f64.sqrt`, ...) would be marginally
//! tidier but their intrinsics are unstable, and a nightly build takes the
//! `core::intrinsics` rung above this one anyway, which lowers to the same thing.
//!
//! WASM has no FMA instruction, so `fma` is not overridden here.

use core::cfg_select;

use crate::backend::wasm::arch::*;

pub use super::soft::{fma, fmaf};

macro_rules! impl_wasm {
    ($t:ty, $sqrt:ident, $floor:ident, $ceil:ident, $trunc:ident, $round:ident,
     [$v_sqrt:ident, $v_floor:ident, $v_ceil:ident, $v_trunc:ident, $v_nearest:ident, $splat:ident, $lane:ident]) => {
        #[inline(always)]
        pub fn $sqrt(x: $t) -> $t {
            cfg_select! {
                target_feature = "simd128" => $lane::<0>($v_sqrt($splat(x))),
                _ => super::soft::$sqrt(x),
            }
        }

        #[inline(always)]
        pub fn $floor(x: $t) -> $t {
            cfg_select! {
                target_feature = "simd128" => $lane::<0>($v_floor($splat(x))),
                _ => super::soft::$floor(x),
            }
        }

        #[inline(always)]
        pub fn $ceil(x: $t) -> $t {
            cfg_select! {
                target_feature = "simd128" => $lane::<0>($v_ceil($splat(x))),
                _ => super::soft::$ceil(x),
            }
        }

        #[inline(always)]
        pub fn $trunc(x: $t) -> $t {
            cfg_select! {
                target_feature = "simd128" => $lane::<0>($v_trunc($splat(x))),
                _ => super::soft::$trunc(x),
            }
        }

        // Ties to even, matching the vector backends, which is exactly what `nearest` does.
        #[inline(always)]
        pub fn $round(x: $t) -> $t {
            cfg_select! {
                target_feature = "simd128" => $lane::<0>($v_nearest($splat(x))),
                _ => super::soft::$round(x),
            }
        }
    };
}

impl_wasm!(
    f64,
    sqrt,
    floor,
    ceil,
    trunc,
    round,
    [
        f64x2_sqrt,
        f64x2_floor,
        f64x2_ceil,
        f64x2_trunc,
        f64x2_nearest,
        f64x2_splat,
        f64x2_extract_lane
    ]
);

impl_wasm!(
    f32,
    sqrtf,
    floorf,
    ceilf,
    truncf,
    roundf,
    [
        f32x4_sqrt,
        f32x4_floor,
        f32x4_ceil,
        f32x4_trunc,
        f32x4_nearest,
        f32x4_splat,
        f32x4_extract_lane
    ]
);
