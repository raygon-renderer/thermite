use super::*;

pub struct AlignedMut<'a, S: Simd, V: SimdVectorBase<S>> {
    ptr: *mut V::Element,
    len: usize,
    _lt: PhantomData<&'a S>,
}

impl<'a, S: Simd, V: SimdVectorBase<S>> AlignedMut<'a, S, V> {
    #[inline]
    pub unsafe fn new_unchecked(slice: &'a mut [V::Element]) -> Self {
        AlignedMut {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _lt: PhantomData,
        }
    }

    #[inline]
    pub fn new(slice: &'a mut [V::Element]) -> Option<Self> {
        if slice.as_ptr().align_offset(V::ALIGNMENT) != 0 {
            None
        } else {
            Some(unsafe { AlignedMut::new_unchecked(slice) })
        }
    }

    #[inline]
    pub fn iter_mut(self) -> AlignedMutIter<'a, S, V> {
        AlignedMutIter(self)
    }
}

pub struct AlignedMutIter<'a, S: Simd, V: SimdVectorBase<S>>(AlignedMut<'a, S, V>);

impl<'a, S: Simd, V: SimdVectorBase<S>> AlignedMutIter<'a, S, V> {
    /// Returns the remainder of the slice that is being iterated over.
    ///
    /// If the iterator has been exhausted (`next()` returns `None`),
    /// this may still return elements that would not fill an SIMD vector.
    pub fn remainder(&mut self) -> &'a mut [V::Element] {
        unsafe { core::slice::from_raw_parts_mut(self.0.ptr, self.0.len) }
    }
}

impl<'a, S: Simd, V: SimdVectorBase<S>> Iterator for AlignedMutIter<'a, S, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<&'a mut V> {
        if self.0.len < V::NUM_ELEMENTS {
            None
        } else {
            unsafe {
                let ptr = self.0.ptr;
                self.0.ptr = self.0.ptr.add(V::NUM_ELEMENTS);
                self.0.len -= V::NUM_ELEMENTS;
                Some(&mut *(ptr as *mut V))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.0.len / V::NUM_ELEMENTS;
        (remaining, Some(remaining))
    }
}
