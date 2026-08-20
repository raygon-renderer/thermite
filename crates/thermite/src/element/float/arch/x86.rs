//! x86 / x86-64 scalar float ops.
//!
//! Three rungs, each gated on what the compilation unit was built for:
//! SSE2 (`sqrt`), SSE4.1 (`floor`/`ceil`/`trunc`/`round`) and FMA3 (`fma`).
//! On `x86_64` SSE2 is unconditional, so `sqrt` always takes the fast path.

use core::cfg_select;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `roundsd`/`roundss` immediate for truncation, suppressing the inexact exception.
#[cfg(target_feature = "sse4.1")]
const TRUNC_IMM: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;

/// Same, for round-to-nearest with ties to even.
#[cfg(target_feature = "sse4.1")]
const NEAREST_IMM: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

// --- SSE2: always available in this module -------------------------------------

#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    unsafe { _mm_cvtsd_f64(_mm_sqrt_sd(_mm_undefined_pd(), _mm_set_sd(x))) }
}

#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

// --- SSE4.1 rounding -----------------------------------------------------------

#[inline(always)]
pub fn floor(x: f64) -> f64 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtsd_f64(_mm_floor_sd(_mm_undefined_pd(), _mm_set_sd(x))) },
        _ => super::soft::floor(x),
    }
}

#[inline(always)]
pub fn floorf(x: f32) -> f32 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtss_f32(_mm_floor_ss(_mm_undefined_ps(), _mm_set_ss(x))) },
        _ => super::soft::floorf(x),
    }
}

#[inline(always)]
pub fn ceil(x: f64) -> f64 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtsd_f64(_mm_ceil_sd(_mm_undefined_pd(), _mm_set_sd(x))) },
        _ => super::soft::ceil(x),
    }
}

#[inline(always)]
pub fn ceilf(x: f32) -> f32 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtss_f32(_mm_ceil_ss(_mm_undefined_ps(), _mm_set_ss(x))) },
        _ => super::soft::ceilf(x),
    }
}

#[inline(always)]
pub fn trunc(x: f64) -> f64 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtsd_f64(_mm_round_sd::<TRUNC_IMM>(_mm_undefined_pd(), _mm_set_sd(x))) },
        _ => super::soft::trunc(x),
    }
}

#[inline(always)]
pub fn truncf(x: f32) -> f32 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtss_f32(_mm_round_ss::<TRUNC_IMM>(_mm_undefined_ps(), _mm_set_ss(x))) },
        _ => super::soft::truncf(x),
    }
}

// Ties to even, matching the vector backends. That is the mode `roundsd` actually has,
// so it is one instruction here.
#[inline(always)]
pub fn round(x: f64) -> f64 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtsd_f64(_mm_round_sd::<NEAREST_IMM>(_mm_undefined_pd(), _mm_set_sd(x))) },
        _ => super::soft::round(x),
    }
}

#[inline(always)]
pub fn roundf(x: f32) -> f32 {
    cfg_select! {
        target_feature = "sse4.1" => unsafe { _mm_cvtss_f32(_mm_round_ss::<NEAREST_IMM>(_mm_undefined_ps(), _mm_set_ss(x))) },
        _ => super::soft::roundf(x),
    }
}

// --- FMA3 ----------------------------------------------------------------------

#[inline(always)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    cfg_select! {
        target_feature = "fma" => unsafe { _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(x), _mm_set_sd(y), _mm_set_sd(z))) },
        _ => super::soft::fma(x, y, z),
    }
}

#[inline(always)]
pub fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    cfg_select! {
        target_feature = "fma" => unsafe { _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(x), _mm_set_ss(y), _mm_set_ss(z))) },
        _ => super::soft::fmaf(x, y, z),
    }
}
