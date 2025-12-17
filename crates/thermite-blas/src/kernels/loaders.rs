use thermite::vector::generic::FloatVector;

use super::MapKernel;

pub trait Loader<const N: usize> {
    fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N];
}

pub struct UnalignedLoader;
pub struct StreamingLoader;

impl<const N: usize> Loader<N> for UnalignedLoader {
    #[inline(always)]
    fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        ptrs.map(|ptr| unsafe { V::load_unaligned(ptr) })
    }
}

impl<const N: usize> Loader<N> for StreamingLoader {
    #[inline(always)]
    fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        ptrs.map(|ptr| unsafe { V::load_streaming(ptr) })
    }
}

#[derive(Clone, Copy)]
pub struct Producer<'a, T, const N: usize, I: Loader<N>> {
    ptrs: [*const T; N],
    offset: usize,
    len: usize,
    inner: I,
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T, const N: usize, I: Loader<N>> Producer<'a, T, N, I> {
    #[inline(always)]
    pub fn new(data: [&'a [T]; N]) -> Producer<'a, T, N, UnalignedLoader> {
        Producer {
            ptrs: data.map(|slice| slice.as_ptr()),
            offset: 0,
            len: data.iter().map(|slice| slice.len()).min().unwrap_or(0),
            inner: UnalignedLoader,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn new_streaming(data: [&'a [T]; N]) -> Producer<'a, T, N, StreamingLoader> {
        Producer {
            ptrs: data.map(|slice| slice.as_ptr()),
            offset: 0,
            len: data.iter().map(|slice| slice.len()).min().unwrap_or(0),
            inner: StreamingLoader,
            _marker: core::marker::PhantomData,
        }
    }
}

// ProducerKernel trait is automatically implemented for this
impl<T, const N: usize, I: Loader<N>> MapKernel<T, 0, N> for Producer<'_, T, N, I>
where
    T: Copy,
{
    const INTERMEDIATES: usize = 0;

    #[inline(always)]
    fn remaining<V: FloatVector<Element = T>>(&self) -> usize {
        // Calculate how many full vector loads remain
        self.len.saturating_sub(self.offset).div_ceil(V::LANES)
    }

    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, _inputs: [V; 0]) -> [V; N] {
        self.inner
            .load::<V>(self.ptrs.map(|ptr| unsafe { ptr.add(self.offset) }))
    }
}
