extern crate alloc;

use crate::*;

use alloc::alloc::{alloc, dealloc, Layout};
use core::{
    fmt, mem,
    ops::{Deref, DerefMut},
    ptr,
};

/// Aligned SIMD vector storage
#[repr(transparent)]
pub struct VectorBuffer<S: Simd, V: SimdVectorBase<S>> {
    buffer: *mut [V::Element],
}

impl<S: Simd, V: SimdVectorBase<S>> Deref for VectorBuffer<S, V> {
    type Target = [V::Element];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<S: Simd, V: SimdVectorBase<S>> DerefMut for VectorBuffer<S, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<S: Simd, V: SimdVectorBase<S>> fmt::Debug for VectorBuffer<S, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_vector_slice().fmt(f)
    }
}

impl<S: Simd, V: SimdVectorBase<S>> VectorBuffer<S, V> {
    /// Allocates a new SIMD-aligned element buffer and zeroes the elements.
    ///
    /// Due to the alignment, it will round up the number of elements to the nearest multiple of `V::NUM_ELEMENTS`,
    /// making the "wasted" space visible.
    pub fn alloc(count: usize) -> Self {
        unsafe {
            // round up to multiple of NUM_ELEMENTS
            // https://stackoverflow.com/a/9194117/2083075
            let count = (count + V::NUM_ELEMENTS - 1) & (-(V::NUM_ELEMENTS as isize) as usize);

            // allocate zeroed buffer. All SIMD types are valid when zeroed
            VectorBuffer {
                buffer: ptr::slice_from_raw_parts_mut(
                    alloc::alloc::alloc_zeroed(Self::layout(count)) as *mut V::Element,
                    count,
                ),
            }
        }
    }

    #[inline(always)]
    pub fn iter_vectors<'a>(&'a self) -> SimdSliceIter<'a, S, V> {
        SimdSliceIter::new(self.as_slice())
    }

    #[inline(always)]
    pub fn iter_vectors_mut<'a>(&'a mut self) -> AlignedMutIter<'a, S, V> {
        unsafe { AlignedMut::new_unchecked(self.as_mut_slice()).iter_mut() }
    }

    /// Gathers values from the buffer using more efficient instructions where possible
    #[inline(always)]
    pub fn gather(&self, indices: S::Vu32) -> V
    where
        V: SimdVector<S>,
    {
        V::gather(self.as_slice(), indices.cast())
    }

    /// Fills the buffer with vectors using aligned stores
    #[inline]
    pub fn fill(&mut self, value: V) {
        unsafe {
            let ptr = self.as_mut_slice().as_mut_ptr();
            let mut i = 0;
            while i < self.len() {
                value.store_aligned_unchecked(ptr.add(i));
                i += V::NUM_ELEMENTS;
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe { (*self.buffer).len() }
    }

    #[inline]
    pub fn len_vectors(&self) -> usize {
        self.len() / V::NUM_ELEMENTS
    }

    #[inline]
    pub fn as_slice(&self) -> &[V::Element] {
        unsafe { &*self.buffer }
    }

    #[inline]
    pub fn as_vector_slice(&self) -> &[V] {
        unsafe { &(*(self.buffer as *const [V]))[..self.len_vectors()] }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [V::Element] {
        unsafe { &mut *self.buffer }
    }

    #[inline]
    pub fn as_mut_vector_slice(&mut self) -> &mut [V] {
        unsafe { &mut (*(self.buffer as *mut [V]))[..self.len() / V::NUM_ELEMENTS] }
    }

    #[inline]
    pub fn load_vector(&self, vector_index: usize) -> V {
        let scalar_index = vector_index * V::NUM_ELEMENTS;
        let s = self.as_slice();
        assert!(scalar_index < s.len());

        unsafe { V::load_aligned_unchecked(s.as_ptr().add(vector_index)) }
    }

    #[inline]
    pub fn store_vector(&mut self, vector_index: usize, value: V) {
        let scalar_index = vector_index * V::NUM_ELEMENTS;
        let s = self.as_mut_slice();
        assert!(scalar_index < s.len());

        unsafe { value.store_aligned_unchecked(s.as_mut_ptr().add(vector_index)) }
    }

    #[inline(always)]
    fn layout(count: usize) -> Layout {
        // ensure the buffer has the proper size and alignment for SIMD values
        unsafe { Layout::from_size_align_unchecked(count * mem::size_of::<V::Element>(), V::ALIGNMENT) }
    }
}

unsafe impl<S: Simd, V: SimdVectorBase<S>> Send for VectorBuffer<S, V> {}
unsafe impl<S: Simd, V: SimdVectorBase<S>> Sync for VectorBuffer<S, V> {}

impl<S: Simd, V: SimdVectorBase<S>> Drop for VectorBuffer<S, V> {
    fn drop(&mut self) {
        unsafe { dealloc(self.buffer as *mut u8, Self::layout(self.len())) }
    }
}
