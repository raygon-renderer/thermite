use super::arch;

use crate::register::{
    CastRegister, ConcatRegister, IndexableRegister, NumericRegister, Register, SaturatingCastRegister, Storage,
    array::ArrayRegister,
    reduced::{HalfRegister2, ReducedRegister},
};

pub type F32x2V3 = HalfRegister2<super::F32x4V3>;
pub type I32x2V3 = HalfRegister2<super::I32x4V3>;
pub type U32x2V3 = HalfRegister2<super::U32x4V3>;

#[thermite_macros::inline_always]
impl ConcatRegister<f32> for F32x2V3 {
    fn concat(lo: Storage<f32>, hi: Storage<f32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_ps(lo, hi, 0.0, 0.0) })
    }

    fn split(value: Storage<Self>) -> (Storage<f32>, Storage<f32>) {
        let mut arr = [0.0f32; 4];
        unsafe { arch::_mm_storeu_ps(arr.as_mut_ptr(), value.0) };
        (arr[0], arr[1])
    }
}

// `ExtendRegister<f32> for F32x2V3` comes from the generic extend-from-scalar blanket
// in `register/reduced.rs` (F32x4V3::extend / ::narrow, same codegen).

#[thermite_macros::inline_always]
impl ConcatRegister<F32x2V3> for super::F32x4V3 {
    fn concat(lo: Storage<F32x2V3>, hi: Storage<F32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_movelh_ps(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<F32x2V3>, Storage<F32x2V3>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_movehl_ps(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<i32> for I32x2V3 {
    fn concat(lo: Storage<i32>, hi: Storage<i32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo, hi, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<i32>, Storage<i32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0], arr[1])
    }
}

// See the F32x2V3 note above: supplied by the extend-from-scalar blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<I32x2V3> for super::I32x4V3 {
    fn concat(lo: Storage<I32x2V3>, hi: Storage<I32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<I32x2V3>, Storage<I32x2V3>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<u32> for U32x2V3 {
    fn concat(lo: Storage<u32>, hi: Storage<u32>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_setr_epi32(lo as i32, hi as i32, 0, 0) })
    }

    fn split(value: Storage<Self>) -> (Storage<u32>, Storage<u32>) {
        let mut arr = [0i32; 4];
        unsafe { arch::_mm_storeu_si128(arr.as_mut_ptr() as *mut _, value.0) };
        (arr[0] as u32, arr[1] as u32)
    }
}

// See the F32x2V3 note above: supplied by the extend-from-scalar blanket.

#[thermite_macros::inline_always]
impl ConcatRegister<U32x2V3> for super::U32x4V3 {
    fn concat(lo: Storage<U32x2V3>, hi: Storage<U32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_unpacklo_epi64(lo.0, hi.0) }
    }

    fn split(value: Storage<Self>) -> (Storage<U32x2V3>, Storage<U32x2V3>) {
        (
            ReducedRegister::new(value),
            ReducedRegister::new(unsafe { arch::_mm_unpackhi_epi64(value, value) }),
        )
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F32x2V3> for super::F64x2V3 {
    fn cast_from(value: Storage<F32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V3> for F32x2V3 {
    fn cast_from(value: Storage<super::F64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_ps(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V3> for I32x2V3 {
    fn cast_from(value: Storage<super::F64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvttpd_epi32(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x2V3> for U32x2V3 {
    fn cast_from(value: Storage<super::F64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_cvtpd_epu32x_v2(value) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V3> for super::F64x2V3 {
    fn cast_from(value: Storage<I32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_pd(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<U32x2V3> for super::U64x2V3 {
    fn cast_from(value: Storage<U32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepu32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2V3> for U32x2V3 {
    fn cast_from(value: Storage<super::U64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

#[thermite_macros::inline_always]
impl CastRegister<I32x2V3> for super::I64x2V3 {
    fn cast_from(value: Storage<I32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_cvtepi32_epi64(value.0) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2V3> for I32x2V3 {
    fn cast_from(value: Storage<super::I64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_shuffle_epi32(value, 0b10_00_10_00) })
    }
}

// Saturating narrow i64 -> i32 / u64 -> u32. AVX2 has no 64-bit saturating pack (that
// arrives with AVX-512 `vpmovsqd`/`vpmovusqd`), so clamp into the destination range with
// the (polyfilled) 64-bit min/max, then reuse the truncating narrow above - the clamped
// value's low 32 bits are exactly the saturated result.
#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::I64x2V3> for I32x2V3 {
    fn saturating_cast_from(value: Storage<super::I64x2V3>) -> Storage<Self> {
        let lo = <super::I64x2V3 as Register>::splat(i32::MIN as i64);
        let hi = <super::I64x2V3 as Register>::splat(i32::MAX as i64);
        let clamped = super::I64x2V3::min(super::I64x2V3::max(value, lo), hi);
        <Self as CastRegister<super::I64x2V3>>::cast_from(clamped)
    }
}

#[thermite_macros::inline_always]
impl SaturatingCastRegister<super::U64x2V3> for U32x2V3 {
    fn saturating_cast_from(value: Storage<super::U64x2V3>) -> Storage<Self> {
        // Unsigned: only the high end can overflow (lanes are already >= 0).
        let hi = <super::U64x2V3 as Register>::splat(u32::MAX as u64);
        let clamped = super::U64x2V3::min(value, hi);
        <Self as CastRegister<super::U64x2V3>>::cast_from(clamped)
    }
}

// Remaining native/reduced-destination saturating narrows whose source is a wider native or
// emulated (`ArrayRegister`) AVX2 register. AVX2 has no 64-bit pack and these widths cross the
// native 256-bit boundary, so each clamps into range (register min/max) then reuses the existing
// truncating narrow. The `ArrayRegister`-destination cousins are generated by `impl_casts!` in
// `register/array.rs` once these native bases exist.
macro_rules! sat_gap {
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

// Only `i64 -> i32` genuinely needs a clamp here: AVX2 has no 64-bit saturating pack. The other
// wide narrows (`i32x16 -> i16x16/i8x16` and `i64 -> 16/8`) go through `vpack*` in their target
// register files (`i16x16.rs`/`i8x16.rs`/`i16x8.rs`/`half8.rs`), composing down from i32.
sat_gap! {
    (super::I64x4V3, i64, super::I32x4V3, i32),
    (ArrayRegister<super::I64x4V3, 2>, i64, super::I32x8V3, i32),
    (super::U64x4V3, u64, super::U32x4V3, u32),
    (ArrayRegister<super::U64x4V3, 2>, u64, super::U32x8V3, u32),
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V3> for F32x2V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_i64gather_ps::<4>(ptr as *const _, indices) })
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V3>,
    ) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_mask_i64gather_ps::<4>(src.0, ptr as *const _, indices, mask.0) })
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V3> for U32x2V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_i64gather_epi32::<4>(ptr as *const _, indices) })
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V3>,
    ) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_mask_i64gather_epi32::<4>(src.0, ptr as *const _, indices, mask.0) })
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V3> for I32x2V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V3>) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_i64gather_epi32::<4>(ptr as *const _, indices) })
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V3>,
    ) -> Storage<Self> {
        ReducedRegister::new(unsafe { arch::_mm_mask_i64gather_epi32::<4>(src.0, ptr as *const _, indices, mask.0) })
    }
}
