#![warn(missing_docs, clippy::missing_safety_doc)]

//! Non-temporal (streaming) vector loads and stores.

use core::ops::Deref;

use super::GenericVector;

/// Wrapper around an immutable vector reference for non-temporal (streaming) loads.
#[repr(transparent)]
pub struct StreamingVector<'a, V: GenericVector>(pub(crate) &'a V);

impl<V: GenericVector> Clone for StreamingVector<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: GenericVector> Copy for StreamingVector<'_, V> {}

/// Wrapper around a mutable vector reference for non-temporal (streaming) loads and stores.
#[repr(transparent)]
pub struct StreamingVectorMut<'a, V: GenericVector>(pub(crate) &'a mut V);

impl<'a, V: GenericVector> Deref for StreamingVectorMut<'a, V> {
    type Target = StreamingVector<'a, V>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: StreamingVector and StreamingVectorMut have the same representation.
        unsafe { core::mem::transmute(self) }
    }
}

impl<V: GenericVector> StreamingVector<'_, V> {
    /// Load a vector using a non-temporal (streaming) load.
    ///
    /// This memory should not be accessed frequently by the CPU,
    /// as non-temporal loads are intended for data that will not be reused soon.
    #[inline(always)]
    pub fn load(&self) -> V {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { V::load_streaming(self.0 as *const _ as *const V::Element) }
    }

    /// Load a vector using a regular (cached) load.
    ///
    /// This is provided for cases where the user wants to load from a streaming source
    /// but still use a cached load. This will bring the data into the CPU cache.
    #[inline(always)]
    pub fn load_cached(&self) -> V {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { V::load(self.0 as *const _ as *const V::Element) }
    }
}

impl<V: GenericVector> StreamingVectorMut<'_, V> {
    /// Store a vector using a non-temporal (streaming) store.
    ///
    /// This memory should not be accessed frequently by the CPU,
    /// as non-temporal stores are intended for data that will not be reused soon.
    #[inline(always)]
    pub fn store(&mut self, vec: V) {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { vec.store_streaming(self.0 as *mut _ as *mut V::Element) }
    }

    /// Store a vector using a regular (cached) store.
    ///
    /// This is provided for cases where the user wants to store to a streaming destination
    /// but still use a cached store. This will bring the data into the CPU cache.
    #[inline(always)]
    pub fn store_cached(&mut self, vec: V) {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { vec.store(self.0 as *mut _ as *mut V::Element) }
    }
}
