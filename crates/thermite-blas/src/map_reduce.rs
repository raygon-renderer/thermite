use thermite::{
    Vector,
    element::FloatElementWithBits,
    generic_array::{GenericArray, typenum::Unsigned},
    prelude::*,
    register::{CoreRegister, well_formed::WellFormedFloatElement},
    simd::{FloatSimd, SizedSimd},
};

use crate::{LoadKernel, kernels::ProducerKernel};

/// Generic map-reduce kernel trait for BLAS operations.
pub trait MapReduceKernel<T, const I: usize, const O: usize>: Sized {
    /// Cleans output vectors before processing in some cases
    #[inline(always)]
    fn filter<V: FloatVector<Element = T>>(&self, v: [V; O]) -> [V; O] {
        v
    }

    /// Map input vectors to output vectors.
    fn map<V: FloatVector<Element = T>>(&self, v: [V; I]) -> [V; O];
    /// Reduce two sets of output vectors.
    fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], v: [V; O]) -> [V; O];
    /// Reduce output vectors to scalar values.
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; O]) -> [T; O];

    const INTERMEDIATES: usize;

    /// Map and reduce in a single step. Can take advantage of fused operations.
    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; O], v: [V; I]) -> [V; O] {
        self.reduce(acc, self.map(v))
    }

    #[inline(always)]
    fn run<S, L, const P: usize>(&self, loader: &L, values: [&[T]; P]) -> [T; O]
    where
        L: LoadKernel<P, I>,
        T: WellFormedFloatElement,
        S: SizedSimd<T, <T as FloatElementWithBits>::SignedBits, <T as FloatElementWithBits>::Bits>,
    {
        map_reduce::<S, T, L, Self, 0, P, I, O>(loader, self, values)
    }

    #[inline(always)]
    fn run_n<S, L, const N: usize, const P: usize>(&self, loader: &L, values: [&[T; N]; P]) -> [T; O]
    where
        L: LoadKernel<P, I>,
        T: WellFormedFloatElement,
        S: SizedSimd<T, <T as FloatElementWithBits>::SignedBits, <T as FloatElementWithBits>::Bits>,
    {
        map_reduce::<S, T, L, Self, N, P, I, O>(loader, self, values.map(|v| v.as_slice()))
    }

    /// Simple map-reduce implementation without using the generic SIMD selection.
    ///
    /// The "loader" is not a LoadKernel, but a simple function that loads I vectors from N input slices
    /// of an arbitrary type U.
    #[inline(always)]
    fn run_simple<U: Copy, V: FloatVector<Element = T>, L, const N: usize>(
        &self,
        loader: L,
        values: [&[U]; N],
    ) -> [T; O]
    where
        L: Fn([U; N]) -> [V; I],
    {
        let mut idx = 0;

        let len = values.iter().map(|v| v.len()).min().unwrap_or(0);

        let mut registers: GenericArray<[V; O], V::Lanes> = unsafe { core::mem::zeroed() };

        let reserved = (I.max(3) - 2) + Self::INTERMEDIATES;
        let min = reserved.min(registers.len());
        let max = ((registers.len() + 2).saturating_sub(min) / O).min(registers.len());

        let lane_width = V::LANES;

        if (min + 2) < max {
            let parallel_width = lane_width * (max - min);

            while (len - idx) >= parallel_width {
                for (i, r) in registers[min..max].iter_mut().enumerate() {
                    let idx = idx + i * lane_width;
                    *r = self.map_reduce(*r, loader(values.map(|p| p[idx])));
                }

                idx += parallel_width;
            }
        }

        while (len - idx) >= lane_width {
            registers[0] = self.map_reduce(registers[0], loader(values.map(|p| p[idx])));
            idx += lane_width;
        }

        let mut stride = 1;
        while stride < registers.len() {
            for i in (0..registers.len()).step_by(stride * 2) {
                registers[i] = self.reduce(registers[i], registers[i + stride]);
            }

            stride <<= 1;
        }

        let sum = self.filter(registers[0]);

        sum.map(|s| s.sum_elements())
    }
}

/// Generic map-reduce implementation using the provided kernel and loader.
///
/// The loader take P input pointers and loads I vectors from them,
/// but the loader is also allowed to load more data than it returns,
/// as indicated by the `READ_MULTIPLIER` associated constant.
///
/// The `MapReduceKernel` takes I input vectors and produces O output vectors.
///
/// The generic parameter N indicates a fixed size for the input slices,
/// allowing for additional compile-time optimizations when known.
///
/// SizedSimd allows selecting SIMD types based on the element type and precision, be it
/// f32 or f64, and their signed counterparts. `T` being a `WellFormedFloatElement` ensures
/// that `T` itself can be used as a Scalar register type within `Vector`.
#[allow(unused_assignments, unused_mut)]
#[inline(always)]
fn map_reduce<S, T, L, K, const N: usize, const P: usize, const I: usize, const O: usize>(
    loader: &L,
    kernel: &K,
    values: [&[T]; P],
) -> [T; O]
where
    // most of this is to assure the compiler that the types are compatible
    // and that the Element type itself can be used as a Scalar register
    T: WellFormedFloatElement,
    // Contains SIMD types, like f32x4/f64x4 as fx4, etc., for the given precision T (f32 or f64)
    S: SizedSimd<T, <T as FloatElementWithBits>::SignedBits, <T as FloatElementWithBits>::Bits>,
    // Kernel for mapping and reducing
    K: MapReduceKernel<T, I, O>,
    // Loader for loading input vectors from arbitrary memory
    L: LoadKernel<P, I>,
{
    let mut idx = 0;
    let len = values.iter().map(|v| v.len()).min().unwrap_or(0);

    if const { N != 0 } {
        unsafe {
            for v in values {
                core::hint::assert_unchecked(v.len() == N);
            }

            core::hint::assert_unchecked(len == N);
        }
    }

    // prepare input pointers
    let ptrs = values.map(|v| v.as_ptr());

    // start with zeroed sum of the largest vector type
    let mut sum = [Vector::ZERO; O];

    macro_rules! descend {
        ($($width:ty),*) => {{
            $(
                // we don't want to use emulated (double-pumped) registers, only real SIMD registers
                if !<$width as CoreRegister>::IS_EMULATED {
                    let lane_width = <<$width as CoreRegister>::Lanes as Unsigned>::USIZE * L::READ_MULTIPLIER;

                    // accumulator registers, some of which may not be used depending on output size O,
                    // it's mostly just a compiler hint to accumulate in parallel
                    let mut registers: GenericArray<[Vector<$width>; O], S::Registers> = unsafe { core::mem::zeroed() };

                    // reserve this many for loading inputs and intermediates,
                    // though the odd math here is because many ops load directly from memory addresses,
                    // bypassing registers, so we only need to reserve enough registers for intermediates and outputs
                    let reserved = (I.max(3) - 2) + K::INTERMEDIATES;

                    let min = reserved.min(registers.len());
                    // somewhat heuristic max to limit unrolling to reasonable levels
                    let max = ((registers.len() + 2).saturating_sub(min) / O).min(registers.len());

                    // only unroll if there are enough registers to make it worthwhile
                    if (min + 2) < max {
                        // elements processed per parallel iteration
                        let parallel_width = lane_width * (max - min);

                        while (len - idx) >= parallel_width {
                            for (i, r) in registers[min..max].iter_mut().enumerate() {
                                let idx = idx + i * lane_width;
                                *r = kernel.map_reduce(*r, unsafe { loader.load::<Vector<$width>>(ptrs.map(|p| p.add(idx))) });
                            }

                            idx += parallel_width;
                        }
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
                // e.g., f64x16 -> f64x8 -> f64x4
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

            // Scalar tail, filtering before reduction if needed
            let mut sum = kernel.reduce_scalar(kernel.filter(sum)).map(Vector);

            while idx < len {
                sum = kernel.map_reduce(sum, unsafe { loader.load::<Vector<T>>(ptrs.map(|p| p.add(idx))) });
                idx += L::READ_MULTIPLIER;
            }

            kernel.reduce_scalar(kernel.filter(sum))
        }};
    }

    descend!(S::fx16, S::fx8, S::fx4)
}

#[allow(unused_assignments, unused_mut)]
#[inline(always)]
fn map_reduce2<S, T, P, MR, const I: usize, const O: usize>(producer: &P, kernel: &MR) -> [T; O]
where
    // most of this is to assure the compiler that the types are compatible
    // and that the Element type itself can be used as a Scalar register
    T: WellFormedFloatElement,
    // Contains SIMD types, like f32x4/f64x4 as fx4, etc., for the given precision T (f32 or f64)
    S: FloatSimd<T>,
    // Kernel for mapping and reducing
    MR: MapReduceKernel<T, I, O>,
    // Producer for loading input vectors from arbitrary memory
    P: ProducerKernel<T, I>,
{
    let mut sum = [Vector::ZERO; O];

    macro_rules! descend {
        ($($width:ty),*) => {{
            $(
                if !<$width as CoreRegister>::IS_EMULATED {
                    // accumulator registers, some of which may not be used depending on output size O,
                    // it's mostly just a compiler hint to accumulate in parallel
                    let mut registers: GenericArray<[Vector<$width>; O], S::Registers> = unsafe { core::mem::zeroed() };

                    // reserve this many for loading inputs and intermediates,
                    // though the odd math here is because many ops load directly from memory addresses,
                    // bypassing registers, so we only need to reserve enough registers for intermediates and outputs
                    let reserved = (I.max(3) - 2) + MR::INTERMEDIATES;

                    let min = reserved.min(registers.len());
                    // somewhat heuristic max to limit unrolling to reasonable levels
                    let max = ((registers.len() + 2).saturating_sub(min) / O).min(registers.len());

                    // only unroll if there are enough registers to make it worthwhile
                    if (min + 2) < max {
                        // producer.remaining returns number of **full vectors** remaining
                        let parallel_width = (max - min);

                        while producer.remaining::<Vector<$width>>() >= parallel_width {
                            for r in registers[min..max].iter_mut() {
                                *r = kernel.map_reduce(*r, producer.produce::<Vector<$width>>());
                            }
                        }
                    }

                    while let Some(input) = producer.produce_opt::<Vector<$width>>() {
                        registers[0] = kernel.map_reduce(registers[0], input);
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
                // e.g., f64x16 -> f64x8 -> f64x4
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

            // Scalar tail, filtering before reduction if needed
            let mut sum: [Vector<T>; O] = kernel.reduce_scalar(kernel.filter(sum)).map(Vector);

            while let Some(input) = producer.produce_opt::<Vector<T>>() {
                sum = kernel.map_reduce(sum, input);
            }

            // final filter and reduction to scalars
            kernel.reduce_scalar(kernel.filter(sum))
        }};
    }

    descend!(S::fx16, S::fx8, S::fx4)
}
