
use super::arch;

use crate::register::{
        CastRegister, ConcatRegister, IndexableRegister, NumericRegister, Register,
        SaturatingCastRegister, Storage,
        reduced::{HalfRegister2, ReducedRegister},
    };

pub type F32x2V2 = HalfRegister2<super::F32x4V2>;
pub type I32x2V2 = HalfRegister2<super::I32x4V2>;
pub type U32x2V2 = HalfRegister2<super::U32x4V2>;

#[thermite_macros::inline_always]
impl ConcatRegister<f32> for F32x2V2 {
    fn concat(lo: Storage<f32>, hi: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_ps(lo, hi, 0.0, 0.0) })
    }

    fn split(value: Storage<Self>) -> (Storage<f32>, Storage<f32>) {
        let mut arr = [0.0f32; 4];
        unsafe { arch::_mm_storeu_ps(arr.as_mut_ptr(), value.0) };
        (arr[0], arr[1])
    }
}

// `ExtendRegister<f32> for F32x2V2` comes from the generic extend-from-scalar blanket
// in `register/reduced.rs` (F32x4V2::extend / ::narrow, same codegen).

#[thermite_macros::inline_always]
impl ConcatRegister<F32x2V2> for super::F32x4V2 {
    fn concat(lo: Storage<F32x2V2>, hi: Storage<F32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_movelh_ps(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<F32x2V2>, Storage<F32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_movehl_ps(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<i32> for I32x2V2 {
    fn concat(lo: Storage<i32>, hi: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo, hi, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<i32>, Storage<i32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0], arr[1])
    }
}

// See the F32x2V2 note above: supplied by the extend-from-scalar blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<I32x2V2> for super::I32x4V2 {
    fn concat(lo: Storage<I32x2V2>, hi: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I32x2V2>, Storage<I32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<u32> for U32x2V2 {
    fn concat(lo: Storage<u32>, hi: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo as i32, hi as i32, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<u32>, Storage<u32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0] as u32, arr[1] as u32)
    }
}

// See the F32x2V2 note above: supplied by the extend-from-scalar blanket.

// impl ConcatRegister<bool> for U32x2V2 {
//     #[inline(always)]
//     fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
//         ReducedRegister::new(unsafe { arch::_mm_setr_epu32x(u32::from_bool(lo), u32::from_bool(hi), 0, 0) })
//     }

//     #[inline(always)]
//     fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
//         let mut arr = [0i32; 4];
//         unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
//         (arr[0].to_bool(), arr[1].to_bool())
//     }
// }

// impl ExtendRegister<bool> for U32x2V2 {
//     #[inline(always)]
//     fn extend(value: Storage<bool>) -> Storage<Self> {
//         ReducedRegister::new(unsafe { arch::_mm_setr_epu32x(u32::from_bool(value), 0, 0, 0) })
//     }

//     #[inline(always)]
//     fn narrow(value: Storage<Self>) -> Storage<bool> {
//         unsafe { arch::_mm_cvtsi128_si32(value.0).to_bool() }
//     }
// }

#[thermite_macros::inline_always]
impl ConcatRegister<U32x2V2> for super::U32x4V2 {
    fn concat(lo: Storage<U32x2V2>, hi: Storage<U32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U32x2V2>, Storage<U32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F32x2V2> for super::F64x2V2 {
    fn cast_from(value: Storage<F32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for F32x2V2 {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_ps(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for I32x2V2 {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvttpd_epi32(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V2> for U32x2V2 {
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_epu32x_v2(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V2> for super::F64x2V2 {
    fn cast_from(value: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2V2> for super::U64x2V2 {
    fn cast_from(value: Storage<U32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V2> for U32x2V2 {
    fn cast_from(value: Storage<super::U64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V2> for super::I64x2V2 {
    fn cast_from(value: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V2> for I32x2V2 {
    fn cast_from(value: Storage<super::I64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

// Saturating narrow i64 -> i32: no SSE 64-bit saturating pack, so clamp into range with the
// (polyfilled) 64-bit min/max and reuse the truncating narrow above.
macro_rules! sat_clamp_narrow {
    ($(($from:ty, $fe:ty, $into:ty, $ie:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl SaturatingCastRegister<$from> for $into {
            fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                let lo = <$from as Register>::splat(<$ie>::MIN as $fe);
                let hi = <$from as Register>::splat(<$ie>::MAX as $fe);
                let clamped = <$from as NumericRegister>::min(<$from as NumericRegister>::max(value, lo), hi);
                <Self as CastRegister<$from>>::cast_from(clamped)
            }
        }
    )*};
}

sat_clamp_narrow! {
    (super::I64x2V2, i64, I32x2V2, i32),
    (super::U64x2V2, u64, U32x2V2, u32),
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V2> for F32x2V2 {}
impl IndexableRegister<super::U64x2V2> for U32x2V2 {}
impl IndexableRegister<super::U64x2V2> for I32x2V2 {}
