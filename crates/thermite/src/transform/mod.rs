//! Algorithms for transforming data via kernels, such as map-reduce.

use crate::{
    prelude::*,
    register::{CoreRegister, well_formed::WellFormedFloatElement},
};

pub trait MapKernel<V> {
    fn map(&self, input: V) -> V;
}

#[cold]
fn map_scalar_inplace<F, K>(data: &mut [F], kernel: &K)
where
    F: WellFormedFloatElement,
    K: MapKernel<Vector<F>>,
{
    for elem in data {
        *elem = kernel.map(Vector::<F>(*elem)).extract::<0>();
    }
}

#[inline(always)]
pub fn map_inplace<S, F, K, const UNROLL: usize>(mut data: &mut [F], kernel: &K)
where
    F: WellFormedFloatElement,
    S: FloatSimd<F>,
    K: MapKernel<Vector<S::fxN>> + MapKernel<Vector<S::fx4>> + MapKernel<Vector<S::fx2>> + MapKernel<Vector<F>>,
{
    if const { matches!(S::ISA, crate::isa::InstructionSet::Scalar) } {
        map_scalar_inplace(data, kernel);

        return;
    }

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

    map_scalar_inplace(data, kernel);
}

pub trait MapKernel2<V, const I: usize, const O: usize> {
    fn map(&self, input: [V; I]) -> [V; O];
}

#[cold]
unsafe fn map_overlapping_scalar<F, K, const I: usize, const O: usize>(
    inputs: [*const F; I],
    outputs: [*mut F; O],
    len: usize, // Length of all input and output arrays.
    kernel: &K,
) where
    F: WellFormedFloatElement,
    K: MapKernel2<Vector<F>, I, O>,
{
    for i in 0..len {
        let res = kernel.map(inputs.map(|ptr| unsafe { Vector::<F>(*ptr.add(i)) }));

        for j in 0..O {
            unsafe { *outputs[j].add(i) = res[j].0 };
        }
    }
}

/// Applies a generic mapping kernel over I inputs and O outputs,
/// where the input and output vectors may overlap in memory. This can be
/// used for in-place transformations or for transformations where the output is stored
/// in a different location than the input.
///
/// # Safety
/// - The caller must ensure that the input and output pointers are valid for reads and writes of `F` respectively.
/// - The caller must ensure that the input and output slices do not overlap in a way that violates Rust's aliasing rules.
/// - The caller must ensure that the input and output slices are properly aligned for `F`.
/// - The caller must ensure that the length of the input and output slices is at least `I` and `O` respectively.
#[inline(always)]
pub unsafe fn map_overlapping<S, F, K, const I: usize, const O: usize>(
    len: usize, // Length of all input and output arrays.
    inputs: [*const F; I],
    outputs: [*mut F; O],
    kernel: &K,
) where
    F: WellFormedFloatElement,
    S: FloatSimd<F>,
    K: MapKernel2<Vector<S::fxN>, I, O>
        + MapKernel2<Vector<S::fx4>, I, O>
        + MapKernel2<Vector<S::fx2>, I, O>
        + MapKernel2<Vector<F>, I, O>,
{
    let n = Vector::<S::fxN>::LANES;

    if len < n || const { matches!(S::ISA, crate::isa::InstructionSet::Scalar) } {
        unsafe { map_overlapping_scalar(inputs, outputs, len, kernel) };

        return;
    }

    // 1. Calculate the offset for the final overlapping vector
    let suffix_offset = len - n;

    // 2. Pre-load the suffix for all inputs BEFORE any writes occur.
    // This perfectly preserves the original data, making exact-in-place safe.
    let mut suffix_inputs = [Vector::<S::fxN>::ZERO; I];
    for i in 0..I {
        suffix_inputs[i] = unsafe { Vector::<S::fxN>::load_unaligned(inputs[i].add(suffix_offset)) };
    }

    // 3. Main Loop
    // We use `< suffix_offset` rather than `<=`. If len is exactly divisible by n,
    // the suffix will exactly handle the final block, preventing unnecessary double-writes.
    let mut idx = 0;
    while idx < suffix_offset {
        let loop_inputs = inputs.map(|ptr| unsafe { Vector::<S::fxN>::load_unaligned(ptr.add(idx)) });
        let loop_outputs = kernel.map(loop_inputs); // --- KERNEL CALL SITE 1 ---

        for j in 0..O {
            unsafe { loop_outputs[j].store_unaligned(outputs[j].add(idx)) };
        }

        idx += n;
    }

    let suffix_outputs = kernel.map(suffix_inputs); // --- KERNEL CALL SITE 2 ---

    for j in 0..O {
        unsafe { suffix_outputs[j].store_unaligned(outputs[j].add(suffix_offset)) };
    }
}
