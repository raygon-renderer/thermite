//! SIMD iteration over slices via the [`SimdSlice`] extension trait.

#![allow(clippy::into_iter_on_ref)]

use crate::vector::{
    GenericVector,
    streaming::{StreamingVector, StreamingVectorMut},
    unaligned::{Unaligned, UnalignedMut},
};

/// Extension methods on slices for SIMD iteration.
///
/// Implemented for `[T]`. All methods are parameterized over a vector type `V`
/// whose element type must match the slice's element type.
///
/// Three iteration strategies are provided, each with a shared and mutable variant:
///
/// - **Aligned**: the slice must be exactly aligned to the vector's alignment with no
///   leading or trailing scalar remainder. Panics if alignment does not hold. Use this
///   when you control the allocation (e.g., via [`NativeSimd`]-aligned containers).
///
/// - **Streaming**: like aligned, but yields [`StreamingVector`]/[`StreamingVectorMut`]
///   wrappers that expose non-temporal (NT) load/store instructions. Suitable for large
///   write-once data that should bypass the CPU cache.
///
/// - **Unaligned**: handles arbitrary slices. Returns `(Unaligned<V>, remainder)` where
///   `remainder` is the trailing scalar elements that did not fill a full vector. The
///   [`Unaligned`] handle performs unaligned loads/stores on each iteration step.
///   Note: the constructor trusts the caller to have truncated the slice to a multiple
///   of the lane count - no runtime check is performed in `next()`.
pub trait SimdSlice {
    type Element;

    /// Iterate over the slice as aligned SIMD vectors.
    ///
    /// # Panics
    ///
    /// Panics if the slice has a leading or trailing scalar remainder (i.e., its length
    /// is not a multiple of `V::Lanes` or its pointer is not aligned to `V`'s alignment).
    fn aligned_simd_iter<V>(&self) -> impl Iterator<Item = &'_ V>
    where
        V: GenericVector<Element = Self::Element>;

    /// Split the slice into a leading scalar remainder, an iterator of aligned SIMD vectors,
    /// and a trailing scalar remainder.
    ///
    /// Unlike [`aligned_simd_iter`](SimdSlice::aligned_simd_iter) this never panics - any
    /// leading bytes needed to reach alignment become the first `&Self`, and any
    /// trailing bytes that don't fill a complete vector become the last `&Self`.
    fn try_aligned_simd_iter<V>(&self) -> (&Self, impl Iterator<Item = &'_ V>, &Self)
    where
        V: GenericVector<Element = Self::Element>;

    /// Iterate over the slice as aligned SIMD vectors using non-temporal (streaming) loads.
    ///
    /// Each item is a [`StreamingVector`] - call `.load()` for a non-temporal load or
    /// `.load_cached()` to bring data into the CPU cache.
    ///
    /// Prefer this over [`aligned_simd_iter`](SimdSlice::aligned_simd_iter) for large
    /// sequential reads that will not be revisited, to avoid polluting the cache.
    ///
    /// # Panics
    ///
    /// Panics if the slice has a leading or trailing scalar remainder.
    fn streaming_simd_iter<V>(&self) -> impl Iterator<Item = StreamingVector<'_, V>>
    where
        V: GenericVector<Element = Self::Element>;

    /// Iterate over the slice as unaligned SIMD vectors.
    ///
    /// Returns `(iter, remainder)` where `remainder` is the trailing elements that did
    /// not fill a complete vector. The [`Unaligned`] iterator issues unaligned loads, so
    /// no alignment guarantee on the slice is required.
    fn unaligned_simd_iter<V>(&self) -> (Unaligned<'_, V>, &Self)
    where
        V: GenericVector<Element = Self::Element>;

    /// Iterate mutably over the slice as aligned SIMD vectors.
    ///
    /// # Panics
    ///
    /// Panics if the slice has a leading or trailing scalar remainder.
    fn aligned_simd_iter_mut<V>(&mut self) -> impl Iterator<Item = &'_ mut V>
    where
        V: GenericVector<Element = Self::Element>;

    /// Split the slice mutably into a leading scalar remainder, an iterator of aligned SIMD
    /// vectors, and a trailing scalar remainder.
    ///
    /// The mutable counterpart of [`try_aligned_simd_iter`](SimdSlice::try_aligned_simd_iter).
    /// Never panics.
    fn try_aligned_simd_iter_mut<V>(&mut self) -> (&mut Self, impl Iterator<Item = &'_ mut V>, &mut Self)
    where
        V: GenericVector<Element = Self::Element>;

    /// Iterate mutably over the slice as aligned SIMD vectors using non-temporal (streaming) stores.
    ///
    /// Each item is a [`StreamingVectorMut`] - call `.store(v)` for a non-temporal store
    /// that bypasses the cache, or dereference to load via a non-temporal read.
    ///
    /// # Panics
    ///
    /// Panics if the slice has a leading or trailing scalar remainder.
    fn streaming_simd_iter_mut<V>(&mut self) -> impl Iterator<Item = StreamingVectorMut<'_, V>>
    where
        V: GenericVector<Element = Self::Element>;

    /// Iterate mutably over the slice as unaligned SIMD vectors.
    ///
    /// Returns `(iter, remainder)` where `remainder` is the trailing elements that did
    /// not fill a complete vector. The [`UnalignedMut`] iterator issues unaligned
    /// loads and stores, so no alignment guarantee on the slice is required.
    fn unaligned_simd_iter_mut<V>(&mut self) -> (UnalignedMut<'_, V>, &mut Self)
    where
        V: GenericVector<Element = Self::Element>;
}

#[thermite_macros::inline_always]
impl<T> SimdSlice for [T] {
    type Element = T;

    fn aligned_simd_iter<V>(&self) -> impl Iterator<Item = &'_ V>
    where
        V: GenericVector<Element = Self::Element>,
    {
        let (&[], simd, &[]) = V::align_slice(self) else {
            panic!("Slice is not exactly aligned");
        };

        simd.into_iter()
    }

    fn try_aligned_simd_iter<V>(&self) -> (&Self, impl Iterator<Item = &'_ V>, &Self)
    where
        V: GenericVector<Element = Self::Element>,
    {
        let (head, simd, tail) = V::align_slice(self);
        (head, simd.into_iter(), tail)
    }

    fn streaming_simd_iter<V>(&self) -> impl Iterator<Item = StreamingVector<'_, V>>
    where
        V: GenericVector<Element = Self::Element>,
    {
        V::stream_aligned_slice(self)
    }

    fn unaligned_simd_iter<V>(&self) -> (Unaligned<'_, V>, &Self)
    where
        V: GenericVector<Element = Self::Element>,
    {
        V::iter_unaligned(self)
    }

    fn aligned_simd_iter_mut<V>(&mut self) -> impl Iterator<Item = &'_ mut V>
    where
        V: GenericVector<Element = Self::Element>,
    {
        let (&mut [], simd, &mut []) = V::align_slice_mut(self) else {
            panic!("Slice is not exactly aligned");
        };

        simd.into_iter()
    }

    fn try_aligned_simd_iter_mut<V>(&mut self) -> (&mut Self, impl Iterator<Item = &'_ mut V>, &mut Self)
    where
        V: GenericVector<Element = Self::Element>,
    {
        let (head, simd, tail) = V::align_slice_mut(self);
        (head, simd.into_iter(), tail)
    }

    fn streaming_simd_iter_mut<V>(&mut self) -> impl Iterator<Item = StreamingVectorMut<'_, V>>
    where
        V: GenericVector<Element = Self::Element>,
    {
        V::stream_aligned_slice_mut(self)
    }

    fn unaligned_simd_iter_mut<V>(&mut self) -> (UnalignedMut<'_, V>, &mut Self)
    where
        V: GenericVector<Element = Self::Element>,
    {
        V::iter_mut_unaligned(self)
    }
}
