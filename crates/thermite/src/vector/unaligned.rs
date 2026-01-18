//! Unaligned vector loads and stores, directly and via iterators.

use core::{
    iter::{DoubleEndedIterator, ExactSizeIterator, Iterator},
    ops::Deref,
};

use generic_array::typenum::Unsigned;

use crate::{Vector, register::Register};

/// A view over a slice of elements that allows unaligned vector loads.
///
/// This can be constructed via [`Vector::from_slice_unaligned`].
#[repr(transparent)]
pub struct Unaligned<'a, R: Register>(pub(crate) &'a [R::Element]);

impl<'a, R: Register> Clone for Unaligned<'a, R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: Register> Copy for Unaligned<'a, R> {}

impl<'a, R: Register> Unaligned<'a, R> {
    /// Read a vector at the given index. Returns `None` if the index is out of bounds.
    #[inline(always)]
    pub fn read(&self, idx: usize) -> Option<Vector<R>> {
        let idx = idx * <R::Lanes as Unsigned>::USIZE;

        if crate::unlikely(idx >= self.0.len()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe { Some(crate::Vector::load_unaligned(self.0.as_ptr().add(idx))) }
    }

    /// Get the underlying slice.
    #[inline(always)]
    pub const fn as_slice(&self) -> &'a [R::Element] {
        self.0
    }

    /// Attempt to get an aligned slice of vectors.
    ///
    /// Returns `None` if the underlying slice is not properly aligned.
    ///
    /// This is provided for cases where `Unaligned` was constructed from
    /// a slice that is actually aligned, allowing for more efficient processing.
    #[inline(always)]
    pub fn try_aligned(&self) -> Option<&'a [Vector<R>]> {
        if self.0.as_ptr().align_offset(core::mem::align_of::<Vector<R>>()) == 0 {
            let len = self.0.len() / <R::Lanes as Unsigned>::USIZE;
            // SAFETY: Alignment was checked.
            Some(unsafe { core::slice::from_raw_parts(self.0.as_ptr() as *const Vector<R>, len) })
        } else {
            None
        }
    }
}

/// A mutable view over a slice of elements that allows unaligned vector loads and stores.
///
/// This can be constructed via [`Vector::from_slice_unaligned_mut`].
#[repr(transparent)]
pub struct UnalignedMut<'a, R: Register>(pub(crate) &'a mut [R::Element]);

impl<'a, R: Register> UnalignedMut<'a, R> {
    /// Write a vector at the given index. Returns `false` if the index is out of bounds.
    #[inline(always)]
    pub fn write(&mut self, idx: usize, value: Vector<R>) -> bool {
        let idx = idx * <R::Lanes as Unsigned>::USIZE;

        if crate::unlikely(idx >= self.0.len()) {
            return false;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe { value.store_unaligned(self.0.as_mut_ptr().add(idx)) };

        true
    }

    /// Get the underlying mutable slice.
    #[inline(always)]
    pub const fn as_mut_slice<'b>(&'a mut self) -> &'b mut [R::Element]
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
    pub fn try_aligned_mut(mut self) -> Result<&'a mut [Vector<R>], Self> {
        if self.0.as_ptr().align_offset(core::mem::align_of::<Vector<R>>()) == 0 {
            let len = self.0.len() / <R::Lanes as Unsigned>::USIZE;
            // SAFETY: Alignment was checked.
            Ok(unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut Vector<R>, len) })
        } else {
            Err(self)
        }
    }
}

impl<'a, R: Register> Deref for UnalignedMut<'a, R> {
    type Target = Unaligned<'a, R>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: Unaligned and UnalignedMut have the same representation.
        unsafe { core::mem::transmute(self) }
    }
}

impl<'a, R: Register> Iterator for Unaligned<'a, R> {
    type Item = Vector<R>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if crate::unlikely(self.0.is_empty()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe {
            let v = crate::Vector::load_unaligned(self.0.as_ptr());

            self.0 = self.0.get_unchecked(<R::Lanes as Unsigned>::USIZE..); // offset slice

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
        let chunk_size = <R::Lanes as Unsigned>::USIZE;

        while i + chunk_size <= self.0.len() {
            // SAFETY: Caller ensured sufficient length.
            let v = unsafe { crate::Vector::load_unaligned(self.0.as_ptr().add(i)) };
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

impl<'a, R: Register> DoubleEndedIterator for Unaligned<'a, R> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if crate::unlikely(self.0.is_empty()) {
            return None;
        }

        // SAFETY: Length was assured to be a multiple of vector lanes at construction.
        unsafe {
            let offset = self.0.len() - <R::Lanes as Unsigned>::USIZE;

            let v = crate::Vector::load_unaligned(self.0.as_ptr().add(offset));

            self.0 = self.0.get_unchecked(..self.0.len() - <R::Lanes as Unsigned>::USIZE); // offset slice

            Some(v)
        }
    }

    fn rfold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut i = self.0.len();
        let chunk_size = <R::Lanes as Unsigned>::USIZE;

        while i >= chunk_size {
            i -= chunk_size;
            // SAFETY: Caller ensured sufficient length.
            let v = unsafe { crate::Vector::load_unaligned(self.0.as_ptr().add(i)) };
            init = f(init, v);
        }

        init
    }
}

impl<'a, R: Register> ExactSizeIterator for Unaligned<'a, R> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len() / <R::Lanes as Unsigned>::USIZE
    }
}
