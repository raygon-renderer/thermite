#![warn(missing_docs, clippy::missing_safety_doc)]

//! Non-temporal (streaming) vector loads and stores.

use core::ops::Deref;

use crate::{Vector, register::Register};

/// Wrapper around an immutable vector reference for non-temporal (streaming) loads.
#[repr(transparent)]
pub struct StreamingVector<'a, R: Register>(pub(crate) &'a R::Storage);

impl<R: Register> Clone for StreamingVector<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for StreamingVector<'_, R> {}

/// Wrapper around a mutable vector reference for non-temporal (streaming) loads and stores.
#[repr(transparent)]
pub struct StreamingVectorMut<'a, R: Register>(pub(crate) &'a mut R::Storage);

impl<'a, R: Register> Deref for StreamingVectorMut<'a, R> {
    type Target = StreamingVector<'a, R>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: StreamingVector and StreamingVectorMut have the same representation.
        unsafe { core::mem::transmute(self) }
    }
}

impl<R: Register> StreamingVector<'_, R> {
    /// Load a vector using a non-temporal (streaming) load.
    ///
    /// This memory should not be accessed frequently by the CPU,
    /// as non-temporal loads are intended for data that will not be reused soon.
    #[inline(always)]
    pub fn load(&self) -> Vector<R> {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { Vector::load_stream(self.0 as *const _ as *const R::Element) }
    }

    /// Load a vector using a regular (cached) load.
    ///
    /// This is provided for cases where the user wants to load from a streaming source
    /// but still use a cached load. This will bring the data into the CPU cache.
    #[inline(always)]
    pub fn load_cached(&self) -> Vector<R> {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { Vector::load(self.0 as *const _ as *const R::Element) }
    }
}

impl<R: Register> StreamingVectorMut<'_, R> {
    /// Store a vector using a non-temporal (streaming) store.
    ///
    /// This memory should not be accessed frequently by the CPU,
    /// as non-temporal stores are intended for data that will not be reused soon.
    #[inline(always)]
    pub fn store(&mut self, vec: Vector<R>) {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { vec.store_stream(self.0 as *mut _ as *mut R::Element) }
    }

    /// Store a vector using a regular (cached) store.
    ///
    /// This is provided for cases where the user wants to store to a streaming destination
    /// but still use a cached store. This will bring the data into the CPU cache.
    #[inline(always)]
    pub fn store_cached(&mut self, vec: Vector<R>) {
        // SAFETY: Ensured valid alignment and size by reference type.
        unsafe { vec.store(self.0 as *mut _ as *mut R::Element) }
    }
}
