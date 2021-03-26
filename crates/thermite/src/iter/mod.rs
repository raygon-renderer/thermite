use crate::*;

mod aligned;
mod slice;

pub use self::aligned::*;
pub use self::slice::*;

pub trait SimdIteratorExt<S: Simd, V>: Iterator<Item = V>
where
    V: SimdVector<S>,
{
    fn store(self, dst: &mut [V::Element], write_zero: bool)
    where
        Self: Sized;

    #[inline]
    fn cast<U>(self) -> SimdCastIter<S, Self, V, U>
    where
        Self: Sized,
        U: SimdFromCast<S, V>,
    {
        SimdCastIter {
            src: self,
            _tys: PhantomData,
        }
    }
}

pub trait IntoSimdIterator<S: Simd> {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_simd_iter(self) -> Self::IntoIter;
}

pub struct SimdCastIter<S: Simd, I, V, U> {
    src: I,
    _tys: PhantomData<(S, V, U)>,
}

impl<S: Simd, I, V, U> Clone for SimdCastIter<S, I, V, U>
where
    I: Clone,
{
    fn clone(&self) -> Self {
        SimdCastIter {
            src: self.src.clone(),
            _tys: PhantomData,
        }
    }
}

impl<S: Simd, I, V, U> Iterator for SimdCastIter<S, I, V, U>
where
    I: Iterator<Item = V>,
    U: SimdFromCast<S, V>,
{
    type Item = U;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.src.next().map(|v| U::from_cast(v))
    }
}

impl<S: Simd, V, T> SimdIteratorExt<S, V> for T
where
    T: Iterator<Item = V>,
    V: SimdVector<S>,
{
    #[inline]
    fn store(mut self, dst: &mut [V::Element], write_zero: bool)
    where
        Self: Sized,
    {
        let mut chunks = dst.chunks_exact_mut(V::NUM_ELEMENTS);

        // normal writes
        (&mut self).zip(&mut chunks).for_each(|(src, dst)| unsafe {
            src.store_unaligned_unchecked(dst.as_mut_ptr());
        });

        if write_zero {
            // fill any remaining chunks with zero
            (&mut chunks).for_each(|dst| unsafe {
                V::zero().store_unaligned_unchecked(dst.as_mut_ptr());
            });
        }

        // if there is a remainder, check to fill it
        let rem = chunks.into_remainder();
        if thermite_unlikely!(!rem.is_empty()) {
            // if there are any values left, write what we can or zero it
            let value = match self.next() {
                Some(value) => value,
                None if write_zero => V::zero(),
                _ => return, // don't zero and nothing to write, so return
            };

            let indices = Vi32::<S>::indexed();
            let mask = Vi32::<S>::splat(rem.len() as i32).lt(indices);

            unsafe { value.scatter_masked_unchecked(rem.as_mut_ptr(), indices, mask.cast_to()) };
        }
    }
}
