//! Algorithms for transforming data via kernels, such as map-reduce.

use crate::{
    prelude::*,
    register::{CoreRegister, well_formed::WellFormedFloatElement},
};

pub trait MapKernel<V> {
    fn map(&self, input: V) -> V;
}

#[inline(always)]
pub fn map_inplace<S, F, K, const UNROLL: usize>(mut data: &mut [F], kernel: &K)
where
    F: WellFormedFloatElement,
    S: FloatSimd<F>,
    K: MapKernel<Vector<S::fxN>> + MapKernel<Vector<S::fx4>> + MapKernel<Vector<S::fx2>> + MapKernel<Vector<F>>,
{
    let n = Vector::<S::fxN>::LANES;
    let len = data.len();

    // data is large enough to be worth using aligned loads of the native vector size,
    // so we load the ends using unaligned loads, map the main body using aligned loads,
    // and then store the ends using unaligned stores. There is some overlap usually,
    // but this is still faster than doing all unaligned loads and stores.
    if data.len() >= n * 2 && data.len() >= 32 {
        // LLVM should optimize this `align_slice` call to merge with the next,
        // but we need to check if there are remainders _before_ manipulating
        // other pointers and loading chunks. It's technically bad if we have the pointers loading
        // after consuming `data`
        let odd = len != n * Vector::<S::fxN>::align_slice(data).1.len();

        let (prefix_ptr, suffix_ptr) = unsafe { (data.as_mut_ptr(), data.as_mut_ptr().add(data.len() - n)) };

        let ends = if odd {
            [prefix_ptr, suffix_ptr].map(|ptr| unsafe { (ptr, Vector::<S::fxN>::load_unaligned(ptr)) })
        } else {
            // NOTE: These values will not be used. Consider this as the `None` case.
            [(core::ptr::null_mut(), Vector::EMPTY); 2]
        };

        let (_, chunks, _) = Vector::<S::fxN>::align_slice_mut(data);

        let rest = if const { UNROLL <= 1 } {
            chunks
        } else {
            // fast path for unrolled loops using a stack-allocated register array.
            let (chunks, rest) = chunks.as_chunks_mut::<UNROLL>();

            let mut registers = [Vector::<S::fxN>::ZERO; UNROLL];

            // load, map, store. Because these assign to unique "registers" in the array,
            // the instructions will be interleaved for instruction-level parallelism.
            for unroll_chunk in chunks {
                for (reg, chunk) in registers.iter_mut().zip(unroll_chunk.iter()) {
                    *reg = unsafe { Vector::<S::fxN>::load(chunk as *const _ as *const F::Element) };
                }

                for reg in registers.iter_mut() {
                    *reg = kernel.map(*reg);
                }

                for (reg, chunk) in registers.iter().zip(unroll_chunk.iter_mut()) {
                    unsafe { reg.store(chunk as *mut _ as *mut F::Element) };
                }
            }

            rest
        };

        // handle remainder
        for chunk in rest {
            let ptr = chunk as *mut _ as *mut F::Element;
            unsafe { kernel.map(Vector::<S::fxN>::load(ptr)).store(ptr) };
        }

        if odd {
            // process and store the remainder vectors
            // use a loop here to encourage less codegen
            for (ptr, vec) in ends {
                unsafe { kernel.map(vec).store_unaligned(ptr) };
            }
        }

        return;
    }

    if const { !<S::fx4 as CoreRegister>::IS_EMULATED } {
        let (chunks4, new_data) = data.as_chunks_mut::<4>();

        for chunk in chunks4 {
            kernel
                .map(<Vector<S::fx4> as GenericVector>::from_slice(chunk))
                .copy_to_slice(chunk);
        }

        data = new_data;
    };

    // only include the 2-lane loop if we don't have 4-lane vectors.
    if const { !<S::fx2 as CoreRegister>::IS_EMULATED && <S::fx4 as CoreRegister>::IS_EMULATED } {
        let (chunks2, new_data) = data.as_chunks_mut::<2>();

        for chunk in chunks2 {
            kernel
                .map(<Vector<S::fx2> as GenericVector>::from_slice(chunk))
                .copy_to_slice(chunk);
        }

        data = new_data;
    };

    // Scalar tail
    for elem in data {
        *elem = kernel.map(Vector::<F>(*elem)).extract::<0>();
    }
}
