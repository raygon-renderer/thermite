//! Unaligned vector loads and stores, directly and via iterators.

use core::{
    iter::{DoubleEndedIterator, ExactSizeIterator, Iterator},
    ops::Deref,
};

use generic_array::typenum::Unsigned;

use super::GenericVector;

/// A view over a slice of elements that allows unaligned vector loads.
///
/// This can be constructed via [`Vector::from_slice_unaligned`].
#[repr(transparent)]
pub struct Unaligned<'a, V: GenericVector>(pub(crate) &'a [V::Element]);

impl<'a, V: GenericVector> Clone for Unaligned<'a, V> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: GenericVector> Copy for Unaligned<'a, V> {}

impl<'a, V: GenericVector> Unaligned<'a, V> {
    /// Read a vector at the given index. Returns `None` if the index is out of bounds.
    #[inline(always)]
    pub fn read(&self, idx: usize) -> Option<V> {
        let idx = idx * <V::Lanes as Unsigned>::USIZE;

        if crate::unlikely(idx >= self.0.len()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe { Some(V::load_unaligned(self.0.as_ptr().add(idx))) }
    }

    /// Get the underlying slice.
    #[inline(always)]
    pub const fn as_slice(&self) -> &'a [V::Element] {
        self.0
    }

    /// Attempt to get an aligned slice of vectors.
    ///
    /// Returns `None` if the underlying slice is not properly aligned.
    ///
    /// This is provided for cases where `Unaligned` was constructed from
    /// a slice that is actually aligned, allowing for more efficient processing.
    #[inline(always)]
    pub fn try_aligned(&self) -> Option<&'a [V]> {
        if self.0.as_ptr().align_offset(core::mem::align_of::<V>()) == 0 {
            let len = self.0.len() / <V::Lanes as Unsigned>::USIZE;
            // SAFETY: Alignment was checked.
            Some(unsafe { core::slice::from_raw_parts(self.0.as_ptr() as *const V, len) })
        } else {
            None
        }
    }
}

/// A mutable view over a slice of elements that allows unaligned vector loads and stores.
///
/// This can be constructed via [`Vector::from_slice_unaligned_mut`].
#[repr(transparent)]
pub struct UnalignedMut<'a, V: GenericVector>(pub(crate) &'a mut [V::Element]);

impl<'a, V: GenericVector> UnalignedMut<'a, V> {
    /// Write a vector at the given index. Returns `false` if the index is out of bounds.
    #[inline(always)]
    pub fn write(&mut self, idx: usize, value: V) -> bool {
        let idx = idx * <V::Lanes as Unsigned>::USIZE;

        if crate::unlikely(idx >= self.0.len()) {
            return false;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe { value.store_unaligned(self.0.as_mut_ptr().add(idx)) };

        true
    }

    /// Get the underlying mutable slice.
    #[inline(always)]
    pub const fn as_mut_slice<'b>(&'a mut self) -> &'b mut [V::Element]
    where
        'a: 'b,
    {
        self.0
    }

    /// Attempt to get an aligned mutable slice of vectors.
    ///
    /// Returns `None` if the underlying slice is not properly aligned.
    ///
    /// This is provided for cases where `UnalignedMut` was constructed from
    /// a slice that is actually aligned, allowing for more efficient processing.
    #[inline(always)]
    pub fn try_aligned_mut(mut self) -> Result<&'a mut [V], Self> {
        if self.0.as_ptr().align_offset(core::mem::align_of::<V>()) == 0 {
            let len = self.0.len() / <V::Lanes as Unsigned>::USIZE;
            // SAFETY: Alignment was checked.
            Ok(unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut V, len) })
        } else {
            Err(self)
        }
    }
}

impl<'a, V: GenericVector> Deref for UnalignedMut<'a, V> {
    type Target = Unaligned<'a, V>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: Unaligned and UnalignedMut have the same representation.
        unsafe { core::mem::transmute(self) }
    }
}

impl<'a, V: GenericVector> Iterator for Unaligned<'a, V> {
    type Item = V;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if crate::unlikely(self.0.is_empty()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe {
            let v = V::load_unaligned(self.0.as_ptr());

            self.0 = self.0.get_unchecked(<V::Lanes as Unsigned>::USIZE..); // offset slice

            Some(v)
        }
    }

    #[inline]
    fn fold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut i = 0;
        let chunk_size = <V::Lanes as Unsigned>::USIZE;

        while i + chunk_size <= self.0.len() {
            // SAFETY: Caller ensured sufficient length.
            let v = unsafe { V::load_unaligned(self.0.as_ptr().add(i)) };
            init = f(init, v);
            i += chunk_size;
        }

        init
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline(always)]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }

    #[inline(always)]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
}

impl<'a, V: GenericVector> DoubleEndedIterator for Unaligned<'a, V> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if crate::unlikely(self.0.is_empty()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe {
            let offset = self.0.len() - <V::Lanes as Unsigned>::USIZE;

            let v = V::load_unaligned(self.0.as_ptr().add(offset));

            self.0 = self.0.get_unchecked(..self.0.len() - <V::Lanes as Unsigned>::USIZE); // offset slice

            Some(v)
        }
    }

    fn rfold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut i = self.0.len();
        let chunk_size = <V::Lanes as Unsigned>::USIZE;

        while i >= chunk_size {
            i -= chunk_size;
            // SAFETY: Caller ensured sufficient length.
            let v = unsafe { V::load_unaligned(self.0.as_ptr().add(i)) };
            init = f(init, v);
        }

        init
    }
}

impl<'a, V: GenericVector> ExactSizeIterator for Unaligned<'a, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len() / <V::Lanes as Unsigned>::USIZE
    }
}
