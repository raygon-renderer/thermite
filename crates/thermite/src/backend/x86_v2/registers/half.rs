use generic_array::typenum::U32;

use super::arch;

use crate::{
    element::MaskElement,
    register::{
        CastRegister, ConcatRegister, ExtendRegister, IndexableRegister, Storage,
        reduced::{HalfRegister2, ReducedRegister},
    },
};

pub type F32x2V2 = HalfRegister2<super::F32x4V2>;
pub type I32x2V2 = HalfRegister2<super::I32x4V2>;
pub type U32x2V2 = HalfRegister2<super::U32x4V2>;

impl ConcatRegister<f32> for F32x2V2 {
    #[inline(always)]
    fn concat(lo: Storage<f32>, hi: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_ps(lo, hi, 0.0, 0.0) })
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<f32>, Storage<f32>) {
        let mut arr = [0.0f32; 4];
        unsafe { arch::_mm_storeu_ps(arr.as_mut_ptr(), value.0) };
        (arr[0], arr[1])
    }
}

impl ExtendRegister<f32> for F32x2V2 {
    #[inline(always)]
    fn extend(value: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_ps(value, 0.0, 0.0, 0.0) })
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<f32> {
        unsafe { arch::_mm_cvtss_f32(value.0) }
    }
}

impl ConcatRegister<F32x2V2> for super::F32x4V2 {
    #[inline(always)]
    fn concat(lo: Storage<F32x2V2>, hi: Storage<F32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_movelh_ps(lo.0, hi.0) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<F32x2V2>, Storage<F32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_movehl_ps(value, value) }),
        )
    }
}

impl ConcatRegister<i32> for I32x2V2 {
    #[inline(always)]
    fn concat(lo: Storage<i32>, hi: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo, hi, 0, 0) })
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<i32>, Storage<i32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0], arr[1])
    }
}

impl ExtendRegister<i32> for I32x2V2 {
    #[inline(always)]
    fn extend(value: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(value, 0, 0, 0) })
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<i32> {
        unsafe { arch::_mm_cvtsi128_si32(value.0) }
    }
}

impl ConcatRegister<I32x2V2> for super::I32x4V2 {
    #[inline(always)]
    fn concat(lo: Storage<I32x2V2>, hi: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<I32x2V2>, Storage<I32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

impl ConcatRegister<u32> for U32x2V2 {
    #[inline(always)]
    fn concat(lo: Storage<u32>, hi: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo as i32, hi as i32, 0, 0) })
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<u32>, Storage<u32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0] as u32, arr[1] as u32)
    }
}

impl ExtendRegister<u32> for U32x2V2 {
    #[inline(always)]
    fn extend(value: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epu32x(value, 0, 0, 0) })
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<u32> {
        unsafe { arch::_mm_cvtsi128_si32(value.0) as u32 }
    }
}

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

impl ConcatRegister<U32x2V2> for super::U32x4V2 {
    #[inline(always)]
    fn concat(lo: Storage<U32x2V2>, hi: Storage<U32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<U32x2V2>, Storage<U32x2V2>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

impl CastRegister<F32x2V2> for super::F64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<F32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(value.0) }
    }
}

impl CastRegister<super::F64x2V2> for F32x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_ps(value) })
    }
}

impl CastRegister<super::F64x2V2> for I32x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvttpd_epi32(value) })
    }
}

impl CastRegister<super::F64x2V2> for U32x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<super::F64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_epu32x_v2(value) })
    }
}

impl CastRegister<I32x2V2> for super::F64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(value.0) }
    }
}

impl CastRegister<U32x2V2> for super::U64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<U32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu32_epi64(value.0) }
    }
}

impl CastRegister<super::U64x2V2> for U32x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<super::U64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

impl CastRegister<I32x2V2> for super::I64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<I32x2V2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_epi64(value.0) }
    }
}

impl CastRegister<super::I64x2V2> for I32x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<super::I64x2V2>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

impl IndexableRegister<super::U64x2V2> for F32x2V2 {}
impl IndexableRegister<super::U64x2V2> for U32x2V2 {}
impl IndexableRegister<super::U64x2V2> for I32x2V2 {}
