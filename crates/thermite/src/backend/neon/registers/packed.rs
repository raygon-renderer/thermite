//! `PackedFloatRegister` for the native 16-bit registers on the NEON backend.
//!
//! Stable Rust has no aarch64 `float16` NEON intrinsics yet (and Cortex-A53
//! lacks FEAT_FP16 arithmetic anyway), so every 16-bit-float format uses the
//! generic branchless `pack`/`unpack` defaults, exactly like the wasm backend.
//! Revisit with a dedicated feature once `stdarch_neon_f16` stabilizes.

use crate::register::PackedFloatRegister;
use crate::register::array::ArrayRegister;

use super::half16::U16x4Neon;
use super::{F32x4Neon, U16x8Neon};

/// f32 partner per `u16` width - the backend's `Simd::f32xK`.
type F32x4 = F32x4Neon;
type F32x8 = ArrayRegister<F32x4Neon, 2>;
type F32x16 = ArrayRegister<F32x4Neon, 4>;

/// Attach `PackedFloatRegister` (generic defaults) for every 16-bit float
/// format on one `(u16 register, f32 register)` pair.
macro_rules! impl_packed_f16 {
    ($u16:ty, $f32:ty) => {
        impl PackedFloatRegister<crate::element::float::spec::Fp16, $f32> for $u16 {}
        impl PackedFloatRegister<crate::element::float::spec::Fp16Fast, $f32> for $u16 {}
        impl PackedFloatRegister<crate::element::float::spec::Bf16, $f32> for $u16 {}
    };
}

impl_packed_f16!(U16x4Neon, F32x4);
impl_packed_f16!(U16x8Neon, F32x8);
impl_packed_f16!(ArrayRegister<U16x8Neon, 2>, F32x16);
