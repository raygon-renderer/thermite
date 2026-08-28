//! Reduced (2-lane) 32-bit registers, backed by the low half of a 128-bit
//! q-register via `ReducedRegister`. NEON has real 64-bit d-registers that
//! could host these natively; that is a planned follow-up specialization -
//! the masked-q-register form here is guaranteed-correct and shares all the
//! q-register impls.

use crate::register::{
    CastRegister, ConcatRegister, CoreRegister, IndexableRegister, Storage,
    reduced::{HalfRegister2, ReducedRegister},
};

use super::arch;

pub type F32x2Neon = HalfRegister2<super::F32x4Neon>;
pub type I32x2Neon = HalfRegister2<super::I32x4Neon>;
pub type U32x2Neon = HalfRegister2<super::U32x4Neon>;

// --- ConcatRegister / ExtendRegister: scalar <-> x2 <-> x4, per element type ---

macro_rules! impl_half_ladder {
    ($($e:ty: $half:ty, $full:ty, suffix: $s:ident, via_u64: ($to_u64:ident, $from_u64:ident)),* $(,)?) => {$(paste::paste! {
        impl ConcatRegister<$e> for $half {
            #[inline(always)]
            fn concat(lo: Storage<$e>, hi: Storage<$e>) -> Storage<Self> {
                unsafe {
                    ReducedRegister::new(arch::[<vsetq_lane_ $s>]::<1>(
                        hi,
                        arch::[<vsetq_lane_ $s>]::<0>(lo, <$full as CoreRegister>::EMPTY),
                    ))
                }
            }

            #[inline(always)]
            fn split(value: Storage<Self>) -> (Storage<$e>, Storage<$e>) {
                unsafe {
                    (
                        arch::[<vgetq_lane_ $s>]::<0>(value.0),
                        arch::[<vgetq_lane_ $s>]::<1>(value.0),
                    )
                }
            }
        }

        // `ExtendRegister<$e> for $half` comes from the generic extend-from-scalar
        // blanket in `register/reduced.rs`, which composes `$full`'s own
        // `neon_extend_scalar!` impl - same `vsetq_lane`/`vgetq_lane` codegen.

        impl ConcatRegister<$half> for $full {
            #[inline(always)]
            fn concat(lo: Storage<$half>, hi: Storage<$half>) -> Storage<Self> {
                // low 64-bit half of each source, zipped: [lo0, lo1, hi0, hi1]
                unsafe { arch::$from_u64(arch::vzip1q_u64(arch::$to_u64(lo.0), arch::$to_u64(hi.0))) }
            }

            #[inline(always)]
            fn split(value: Storage<Self>) -> (Storage<$half>, Storage<$half>) {
                unsafe {
                    (
                        ReducedRegister::new(value),
                        ReducedRegister::new(arch::$from_u64(arch::vextq_u64::<1>(
                            arch::$to_u64(value),
                            arch::$to_u64(value),
                        ))),
                    )
                }
            }
        }
    })*};
}

impl_half_ladder! {
    f32: F32x2Neon, super::F32x4Neon, suffix: f32, via_u64: (vreinterpretq_u64_f32, vreinterpretq_f32_u64),
    i32: I32x2Neon, super::I32x4Neon, suffix: s32, via_u64: (vreinterpretq_u64_s32, vreinterpretq_s32_u64),
    u32: U32x2Neon, super::U32x4Neon, suffix: u32, via_u64: (vreinterpretq_u64_u32, vreinterpretq_u32_u64),
}

// --- CastRegister between x2 types ---

// f32x2 <-> f64x2 (demote / promote)
#[thermite_macros::inline_always]
impl CastRegister<super::F64x2Neon> for F32x2Neon {
    fn cast_from(value: Storage<super::F64x2Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_f32(arch::vcvt_f32_f64(value), arch::vdup_n_f32(0.0))) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F32x2Neon> for super::F64x2Neon {
    fn cast_from(value: Storage<F32x2Neon>) -> Storage<Self> {
        unsafe { arch::vcvt_f64_f32(arch::vget_low_f32(value.0)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Neon> for I32x2Neon {
    // i32x2 <-> i64x2 (truncate / sign-extend)
    fn cast_from(value: Storage<super::I64x2Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s32(arch::vmovn_s64(value), arch::vdup_n_s32(0))) }
    }

    fn saturating_cast_from(value: Storage<super::I64x2Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_s32(arch::vqmovn_s64(value), arch::vdup_n_s32(0))) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2Neon> for super::I64x2Neon {
    fn cast_from(value: Storage<I32x2Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_s32(arch::vget_low_s32(value.0)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Neon> for U32x2Neon {
    // u32x2 <-> u64x2 (truncate / zero-extend)
    fn cast_from(value: Storage<super::U64x2Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u32(arch::vmovn_u64(value), arch::vdup_n_u32(0))) }
    }

    fn saturating_cast_from(value: Storage<super::U64x2Neon>) -> Storage<Self> {
        unsafe { ReducedRegister::new(arch::vcombine_u32(arch::vqmovn_u64(value), arch::vdup_n_u32(0))) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2Neon> for super::U64x2Neon {
    fn cast_from(value: Storage<U32x2Neon>) -> Storage<Self> {
        unsafe { arch::vmovl_u32(arch::vget_low_u32(value.0)) }
    }
}

// --- IndexableRegister marker impls ---
// IndexableRegister<U32x2Neon> for the half types is auto-derived by the
// ReducedRegister blanket impl; only the U64x2Neon index type needs markers.

impl IndexableRegister<super::U64x2Neon> for F32x2Neon {}
impl IndexableRegister<super::U64x2Neon> for I32x2Neon {}
impl IndexableRegister<super::U64x2Neon> for U32x2Neon {}
