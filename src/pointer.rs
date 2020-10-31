use std::marker::PhantomData;
use std::mem;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vptr<S: Simd, T> {
    ptr: S::Vusize,
    ty: PhantomData<T>,
}

impl<S: Simd, T> Vptr<S, T>
where
    Self: SimdPtr<S>,
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

    //#[inline(always)]
    //pub fn is_null(self) -> Vmsize<S> {
    //    self.ptr.eq(S::Vusize::zero())
    //}
}

#[doc(hidden)]
pub trait SimdPtr<S: Simd + ?Sized> {
    //unsafe fn read<V, U>(self, mask: MaskTy<S, V, U>) -> V
    //where
    //    V: SimdMasked<S, U>;
    //unsafe fn write<V, U>(self, mask: MaskTy<S, V, U>, value: V)
    //where
    //    V: SimdMasked<S, U>;
}
