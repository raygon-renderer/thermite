//! `PackedFloatRegister` for the native 16-bit registers on x86-v2.
//!
//! x86-v2 (SSE4.2) has no F16C, so every 16-bit-float format uses the generic branchless
//! `pack`/`unpack` defaults (see `register::{pack_packed, unpack_packed}`). These impls just
//! attach the trait - and hence those defaults - to the backend's native `u16` register types
//! (`u16x4`/`u16x8`/`u16x16`), for the IEEE half formats and their fast/unchecked variant. The
//! 8-bit (fp8) formats need a `u8` container and are handled elsewhere.

use crate::register::PackedFloatRegister;
use crate::register::array::ArrayRegister;

use super::half16::U16x4V2;
use super::{F32x4V2, U16x8V2};

/// f32 partner per `u16` width - the backend's `Simd::f32xK` (native f32 is 4-wide on v2, so the
/// wider ones are flat `ArrayRegister<f32x4, _>`). Each `F::Bits` is the matching `Simd::u32xK`,
/// which already implements the `u16 <-> u32` widen cast required by `SimdExperimental`.
type F32x4 = F32x4V2;
type F32x8 = ArrayRegister<F32x4V2, 2>;
type F32x16 = ArrayRegister<F32x4V2, 4>;

/// Attach `PackedFloatRegister` (generic defaults) for every 16-bit float format on one
/// `(u16 register, f32 register)` pair.
macro_rules! impl_packed_f16 {
    ($u16:ty, $f32:ty) => {
        impl PackedFloatRegister<crate::element::float::spec::Fp16, $f32> for $u16 {}
        impl PackedFloatRegister<crate::element::float::spec::Fp16Fast, $f32> for $u16 {}
        impl PackedFloatRegister<crate::element::float::spec::Bf16, $f32> for $u16 {}
    };
}

impl_packed_f16!(U16x4V2, F32x4);
impl_packed_f16!(U16x8V2, F32x8);
impl_packed_f16!(ArrayRegister<U16x8V2, 2>, F32x16);
