use super::*;

pub struct SimdSliceIter<'a, S: Simd, V: SimdVectorBase<S>> {
    // TODO: Replace with pointer?
    slice: &'a [V::Element],
    _tys: PhantomData<&'a S>,
}

impl<S: Simd, V: SimdVectorBase<S>> Clone for SimdSliceIter<'_, S, V> {
    fn clone(&self) -> Self {
        SimdSliceIter {
            slice: self.slice.clone(),
            _tys: PhantomData,
        }
    }
}

impl<'a, S: Simd, T> IntoSimdIterator<S> for &'a [T]
where
    T: SimdAssociatedVector<S>,
    AssociatedVector<S, T>: SimdVectorBase<S, Element = T>,
{
    type Item = AssociatedVector<S, T>;
    type IntoIter = SimdSliceIter<'a, S, Self::Item>;

    fn into_simd_iter(self) -> SimdSliceIter<'a, S, Self::Item> {
        SimdSliceIter::new(self)
    }
}

impl<'a, S: Simd, V: SimdVectorBase<S>> SimdSliceIter<'a, S, V> {
    #[inline]
    pub fn new(slice: &'a [V::Element]) -> Self {
        SimdSliceIter {
            slice,
            _tys: PhantomData,
        }
    }

    /// Returns the remainder of the slice that is being iterated over.
    ///
    /// If the iterator has been exhausted (`next()` returns `None`),
    /// this may still return elements that would not fill an SIMD vector.
    #[inline]
    pub fn remainder(&self) -> &[V::Element] {
        self.slice
    }
}

impl<'a, S: Simd, V> Iterator for SimdSliceIter<'a, S, V>
where
    V: SimdVectorBase<S>,
{
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        if self.slice.len() < V::NUM_ELEMENTS {
            None
        } else {
            let vector = V::load_unaligned(self.slice);
            self.slice = &self.slice[V::NUM_ELEMENTS..];
            Some(vector)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() / V::NUM_ELEMENTS;
        (remaining, Some(remaining))
    }
}
