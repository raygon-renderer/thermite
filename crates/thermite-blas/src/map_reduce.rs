use thermite::{
    Vector,
    generic_array::{GenericArray, typenum::Unsigned},
    register::{FloatElement, Register},
    simd::{FullyFormedFloatElement, SizedSimd},
    vector::generic::FloatVector,
};

use crate::LoadKernel;

/// Generic map-reduce kernel trait for BLAS operations.
pub trait MapReduceKernel<T, const I: usize, const O: usize>: Sized {
    // Avoid calls to `map` when possible, e.g., when `map_reduce` can be implemented more efficiently.
    const AVOID_MAP: bool;

    /// Map input vectors to output vectors.
    fn map<V: FloatVector<Element = T>>(&self, v: [V; I]) -> [V; O];
    /// Reduce two sets of output vectors.
    fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], v: [V; O]) -> [V; O];
    /// Reduce output vectors to scalar values.
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; O]) -> [T; O];

    /// Map and reduce in a single step. Can take advantage of fused operations.
    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], v: [V; I]) -> [V; O] {
        self.reduce(acc, self.map(v))
    }

    #[inline(always)]
    fn run<S, L>(&self, loader: &L, values: [&[T]; I]) -> [T; O]
    where
        L: LoadKernel<I>,
        T: FullyFormedFloatElement,
        S: SizedSimd<T, <T as FloatElement>::Signed, <T as FloatElement>::Bits>,
    {
        map_reduce::<S, T, L, Self, 0, I, O>(loader, self, values)
    }

    #[inline(always)]
    fn run_n<S, L, const N: usize>(&self, loader: &L, values: [&[T; N]; I]) -> [T; O]
    where
        L: LoadKernel<I>,
        T: FullyFormedFloatElement,
        S: SizedSimd<T, <T as FloatElement>::Signed, <T as FloatElement>::Bits>,
    {
        map_reduce::<S, T, L, Self, N, I, O>(loader, self, values.map(|v| v.as_slice()))
    }
}

/// Generic map-reduce implementation using the provided kernel and loader.
#[allow(unused_assignments, unused_mut)]
#[inline(always)]
fn map_reduce<S, T, L: LoadKernel<I>, K, const N: usize, const I: usize, const O: usize>(
    loader: &L,
    kernel: &K,
    values: [&[T]; I],
) -> [T; O]
where
    // most of this is to assure the compiler that the types are compatible
    // and that the Element type itself can be used as a Scalar register
    T: FullyFormedFloatElement,
    // Contains SIMD types, like f32x4/f64x4 as fx4, etc., for the given precision T (f32 or f64)
    S: SizedSimd<T, <T as FloatElement>::Signed, <T as FloatElement>::Bits>,
    // Kernel for mapping and reducing
    K: MapReduceKernel<T, I, O>,
{
    let mut idx = 0;
    let len = values.iter().map(|v| v.len()).min().unwrap_or(0);

    if const { N != 0 } {
        unsafe {
            for v in &values {
                core::hint::assert_unchecked(v.len() == N);
            }

            core::hint::assert_unchecked(len == N);
        }
    }

    let mut sum = [Vector::ZERO; O];

    let ptrs = values.map(|v| v.as_ptr());

    macro_rules! descend {
        ($($width:ty),*) => {{
            $(
                if !<$width as Register>::IS_EMULATED {
                    let lane_width = <<$width as Register>::Lanes as Unsigned>::USIZE;

                    // accumulator registers, some of which may not be used depending on output size O,
                    // it's mostly just a compiler hint to accumulate in parallel
                    let mut registers: GenericArray<[Vector<$width>; O], S::Registers> = unsafe { core::mem::zeroed() };

                    // max number of registers we can use for accumulation, given the number of output vectors O
                    let max = registers.len() / O;

                    let mut register_is_used = false;

                    // for parallel accumulation, we need N-1 real registers, where ymm0 is reserved for temporary loads
                    // this ends up being optimal due to register renaming and avoiding stalls
                    if registers.len() > 1 && max > (registers.len() - 1) {
                        // elements processed per parallel iteration
                        let parallel_width = lane_width * (registers.len() - 1);

                        // first iteration using regular map to fill registers
                        if const { N > 0 && !K::AVOID_MAP } && (len - idx) >= parallel_width {
                            for (i, r) in registers[1..max].iter_mut().enumerate().skip(1) {
                                *r = kernel.map(unsafe {
                                    loader.load::<Vector<$width>>(ptrs.map(|p| p.add(idx + i * lane_width)))
                                });
                            }

                            idx += parallel_width;
                            register_is_used = true;
                        }

                        while (len - idx) >= parallel_width {
                            for (i, r) in registers[1..max].iter_mut().enumerate() {
                                *r = kernel.map_reduce(*r, unsafe {
                                    loader.load::<Vector<$width>>(ptrs.map(|p| p.add(idx + i * lane_width)))
                                });
                            }

                            idx += parallel_width;
                            register_is_used = true;
                        }
                    }

                    if { N > 0 && !K::AVOID_MAP } && !register_is_used && (len - idx) >= lane_width {
                        registers[0] = kernel.map(unsafe {
                            loader.load::<Vector<$width>>(ptrs.map(|p| p.add(idx)))
                        });
                        idx += lane_width;
                    }

                    // accumulate remaining full lanes in ymm0
                    while (len - idx) >= lane_width {
                        registers[0] = kernel.map_reduce(registers[0], unsafe {
                            loader.load::<Vector<$width>>(ptrs.map(|p| p.add(idx)))
                        });
                        idx += lane_width;
                    }

                    // Log2 Sum Reduction, minimizing latency by using a tree structure
                    // when unrolled this generates optimal instruction scheduling
                    let mut stride = 1;
                    while stride < registers.len() {
                        for i in (0..registers.len()).step_by(stride * 2) {
                            registers[i] = kernel.reduce(registers[i], registers[i + stride]);
                        }

                        stride <<= 1;
                    }

                    sum = kernel.reduce(sum, registers[0]); // accumulate to overall sum
                }

                // prep sum for next iteration by narrowing it down
                let mut sum = {
                    let mut lo = [Vector::ZERO; O];
                    let mut hi = [Vector::ZERO; O];

                    for i in 0..O {
                        let (l, h) = sum[i].split2();
                        lo[i] = l;
                        hi[i] = h;
                    }

                    kernel.reduce(lo, hi)
                };
            )*

            // Scalar tail
            let mut sum = kernel.reduce_scalar(sum).map(Vector);

            while idx < len {
                sum = kernel.map_reduce(sum, unsafe { loader.load::<Vector<T>>(ptrs.map(|p| p.add(idx))) });
                idx += 1;
            }

            kernel.reduce_scalar(sum)
        }};
    }

    descend!(S::fx16, S::fx8, S::fx4)
}
