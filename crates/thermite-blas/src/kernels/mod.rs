pub mod dot_product;
pub mod loaders;

use thermite::vector::generic::FloatVector;

const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// A kernel that produces N output vectors without any input vectors.
///
/// These can be constructed in various ways, such as generating
/// constant values, random values, or loading from memory.
///
/// This is a specialized form of a MapKernel with zero input vectors.
pub trait ProducerKernel<T, const N: usize>: MapKernel<T, 0, N> {
    #[inline(always)]
    fn produce<V: FloatVector<Element = T>>(&self) -> [V; N] {
        self.map::<V>([])
    }

    #[inline(always)]
    fn produce_opt<V: FloatVector<Element = T>>(&self) -> Option<[V; N]> {
        if self.remaining::<V>() > 0 {
            Some(self.produce::<V>())
        } else {
            None
        }
    }
}

impl<T, M, const N: usize> ProducerKernel<T, N> for M where M: MapKernel<T, 0, N> {}

/// Generic map kernel trait for mapping input vectors to output vectors.
pub trait MapKernel<T, const I: usize, const O: usize>: Sized {
    const INTERMEDIATES: usize = 1;

    /// Returns the number of remaining input vectors to process,
    /// at least for this type of FloatVector. Smaller vector sizes
    /// may have more remaining vectors.
    fn remaining<V: FloatVector<Element = T>>(&self) -> usize;

    fn map<V: FloatVector<Element = T>>(&self, inputs: [V; I]) -> [V; O];

    /// Compose this map kernel with another map kernel.
    #[inline(always)]
    fn map_with<const P: usize, M>(self, map: M) -> impl MapKernel<T, I, P>
    where
        M: MapKernel<T, O, P>,
    {
        CompositeMapKernel::<Self, M, O> {
            map1: self,
            map2: map,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn reduce_with<R>(self, reduce: R) -> impl MapReduceKernel<T, I, O>
    where
        R: ReduceKernel<T, O>,
    {
        ReduceMapWithKernel::<Self, R, I> {
            map: self,
            reduce,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Generic reduce kernel trait for reducing output vectors, combining
/// accumulated vectors with new input vectors.
pub trait ReduceKernel<T, const N: usize> {
    const INTERMEDIATES: usize = 1;

    fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; N], inputs: [V; N]) -> [V; N];

    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; N]) -> [T; N];

    /// Cleans output vectors before processing in some cases,
    /// such as applying Kahan summation correction.
    #[inline(always)]
    fn filter<V: FloatVector<Element = T>>(&self, v: [V; N]) -> [V; N] {
        v
    }
}

pub trait MapReduceKernel<T, const I: usize, const O: usize>: MapKernel<T, I, O> + ReduceKernel<T, O> {
    const INTERMEDIATES: usize = const_max(
        <Self as MapKernel<T, I, O>>::INTERMEDIATES,
        <Self as ReduceKernel<T, O>>::INTERMEDIATES,
    );

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], inputs: [V; I]) -> [V; O] {
        self.reduce(acc, self.map(inputs))
    }
}

struct CompositeMapKernel<M1, M2, const M: usize> {
    map1: M1,
    map2: M2,
    _marker: core::marker::PhantomData<[(); M]>,
}

struct ReduceMapWithKernel<M1, M2, const M: usize> {
    map: M1,
    reduce: M2,
    _marker: core::marker::PhantomData<[(); M]>,
}

impl<T, const I: usize, const M: usize, const O: usize, M1, M2> MapKernel<T, I, O> for CompositeMapKernel<M1, M2, M>
where
    M1: MapKernel<T, I, M>,
    M2: MapKernel<T, M, O>,
{
    const INTERMEDIATES: usize = const_max(M1::INTERMEDIATES, M2::INTERMEDIATES);

    #[inline(always)]
    fn remaining<V: FloatVector<Element = T>>(&self) -> usize {
        self.map1.remaining::<V>()
    }

    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, inputs: [V; I]) -> [V; O] {
        self.map2.map(self.map1.map(inputs))
    }
}

impl<T, const I: usize, const O: usize, M1, M2> MapReduceKernel<T, I, O> for ReduceMapWithKernel<M1, M2, I>
where
    M1: MapKernel<T, I, O>,
    M2: ReduceKernel<T, O>,
{
    const INTERMEDIATES: usize = const_max(M1::INTERMEDIATES, M2::INTERMEDIATES);
}

impl<T, const I: usize, const O: usize, M1, M2> MapKernel<T, I, O> for ReduceMapWithKernel<M1, M2, I>
where
    M1: MapKernel<T, I, O>,
    M2: ReduceKernel<T, O>,
{
    const INTERMEDIATES: usize = const_max(M1::INTERMEDIATES, M2::INTERMEDIATES);

    #[inline(always)]
    fn remaining<V: FloatVector<Element = T>>(&self) -> usize {
        self.map.remaining::<V>()
    }

    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, inputs: [V; I]) -> [V; O] {
        self.map.map(inputs)
    }
}

impl<T, const I: usize, const O: usize, M1, M2> ReduceKernel<T, O> for ReduceMapWithKernel<M1, M2, I>
where
    M1: MapKernel<T, I, O>,
    M2: ReduceKernel<T, O>,
{
    const INTERMEDIATES: usize = const_max(M1::INTERMEDIATES, M2::INTERMEDIATES);

    #[inline(always)]
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; O]) -> [T; O] {
        self.reduce.reduce_scalar(v)
    }

    #[inline(always)]
    fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], inputs: [V; O]) -> [V; O] {
        self.reduce.reduce(acc, inputs)
    }
}
