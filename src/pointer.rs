use std::marker::PhantomData;
use std::mem;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct VPtr<S: Simd, T> {
    ptr: S::Vusize,
    ty: PhantomData<T>,
}

impl<S: Simd, T> VPtr<S, T>
where
    T: SimdAssociatedVector<S>,
    S::Vusize: SimdPtrInternal<S, <T as SimdAssociatedVector<S>>::V>,
{
    #[inline(always)]
    pub fn splat(ptr: *mut T) -> Self {
        Self {
            ptr: S::Vusize::splat(ptr as _),
            ty: PhantomData,
        }
    }

    #[inline(always)]
    pub fn add(self, offset: S::Vusize) -> Self {
        Self {
            ptr: self.ptr + offset * S::Vusize::splat(mem::size_of::<T>() as _),
            ty: PhantomData,
        }
    }

    #[inline(always)]
    pub fn is_null(self) -> Mask<S, S::Vusize> {
        self.ptr.eq(S::Vusize::zero())
    }

    #[inline(always)]
    pub unsafe fn read(self) -> AssociatedVector<S, T> {
        self.ptr._mm_gather()
    }

    #[inline(always)]
    pub unsafe fn read_masked(
        self,
        mask: Mask<S, AssociatedVector<S, T>>,
        default: AssociatedVector<S, T>,
    ) -> AssociatedVector<S, T> {
        self.ptr._mm_gather_masked(mask, default)
    }

    #[inline(always)]
    pub unsafe fn write(self, value: AssociatedVector<S, T>) {
        self.ptr._mm_scatter(value)
    }

    #[inline(always)]
    pub unsafe fn write_masked(self, mask: Mask<S, AssociatedVector<S, T>>, value: AssociatedVector<S, T>) {
        self.ptr._mm_scatter_masked(mask, value)
    }
}

pub trait SimdAssociatedVector<S: Simd> {
    type V: SimdVector<S>;
}

/// Associated vector type for a scalar type
pub type AssociatedVector<S, T> = <T as SimdAssociatedVector<S>>::V;

macro_rules! impl_associated {
    ($($ty:ident),*) => {paste::paste!{$(
        impl<S: Simd> SimdAssociatedVector<S> for $ty {
            type V = <S as Simd>::[<V $ty>];
        }
    )*}};
}

impl_associated!(i32, u32, u64, f32, f64);

#[doc(hidden)]
pub trait SimdPtrInternal<S: Simd + ?Sized, V: SimdVector<S>>: SimdVector<S> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> V {
        self._mm_gather_masked(Mask::truthy(), V::default())
    }

    #[inline(always)]
    unsafe fn _mm_scatter(self, value: V) {
        self._mm_scatter_masked(Mask::truthy(), value)
    }

    unsafe fn _mm_gather_masked(self, mask: Mask<S, V>, default: V) -> V {
        unimplemented!()
        //let mut res = default;
        //for i in 0..Self::NUM_ELEMENTS {
        //    if mask.extract_unchecked(i) {
        //        res = res.replace_unchecked(i, *mem::transmute::<_, *const V::Element>(self.extract_unchecked(i)));
        //    }
        //}
        //res
    }

    unsafe fn _mm_scatter_masked(self, mask: Mask<S, V>, value: V) {
        unimplemented!()
        //for i in 0..Self::NUM_ELEMENTS {
        //    if mask.extract_unchecked(i) {
        //        *mem::transmute::<_, *mut V::Element>(self.extract_unchecked(i)) = value.extract_unchecked(i);
        //    }
        //}
    }
}
