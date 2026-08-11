use thermite::{prelude::*, register::well_formed::WellFormedFloatElement, vector::VectorWithRegister};

pub trait MapKernel2<V, const I: usize, const O: usize> {
    fn map(&self, input: [V; I]) -> [V; O];
}

/// Applies a generic mapping kernel over I inputs and O outputs,
/// where the input and output vectors may overlap in memory. This can be
/// used for in-place transformations or for transformations where the output is stored
/// in a different location than the input.
///
/// # One kernel call
///
/// There is exactly one `kernel.map` call in this function, and that is the whole
/// point of its shape. The kernel is `#[inline(always)]` into a
/// `#[target_feature]` body, so every call site is a full inlined copy of the math
/// it performs. Across ~250 exported functions, a second call site for the tail
/// costs a second copy of `exp`, `erf`, or whatever else, throughout the library.
///
/// The tail is therefore handled by filling a register's lanes directly rather
/// than by a second kernel: the final partial block writes its live elements into
/// a zeroed vector, runs through the same call, and writes the same count back
/// out. There is no masked load, because a masked load that reaches past the end
/// of the buffer is undefined behaviour in Rust regardless of what the hardware
/// does with the masked-off lanes.
///
/// # Blocks are disjoint
///
/// Block `k` reads and writes exactly `[k*n, min(k*n + n, len))`, so no element is
/// visited twice and an exact in-place transform (`outputs == inputs`) needs no
/// pre-loading. The kernel does not have to be pure, and nothing is computed
/// redundantly, which the earlier overlapping-suffix arrangement required on both
/// counts.
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
    K: MapKernel2<Vector<S::fxN>, I, O>,
{
    let n = Vector::<S::fxN>::lanes();

    let mut idx = 0;
    while idx < len {
        let remaining = len - idx;

        // `LANES > 1` is a constant, so the whole staging path folds away on the
        // scalar backend, where every block is exactly full.
        let partial = const { Vector::<S::fxN>::LANES > 1 } && remaining < n;

        let mut staged = [Vector::<S::fxN>::ZERO; I];
        let mut i = 0;
        while i < I {
            staged[i] = unsafe {
                let src = inputs[i].add(idx);

                if partial {
                    // Starts zeroed, so the padding lanes carry no denormals or
                    // NaNs into the kernel and their results are discarded.
                    //
                    // The trip count is the constant LANES with the live count as
                    // an inner predicate, not `remaining` directly: a loop bounded
                    // by a runtime length is exactly the shape LLVM's loop-idiom
                    // pass rewrites into a `memcpy` call, and this library links
                    // against no C runtime to resolve one.
                    let mut v = Vector::<S::fxN>::ZERO;
                    let lanes = v.as_mut_slice();
                    let mut e = 0;
                    while e < Vector::<S::fxN>::LANES {
                        if e < remaining {
                            lanes[e] = *src.add(e);
                        }
                        e += 1;
                    }
                    v
                } else {
                    Vector::<S::fxN>::load_unaligned(src)
                }
            };
            i += 1;
        }

        // --- THE ONLY KERNEL CALL SITE ---
        let results = kernel.map(staged);

        let mut j = 0;
        while j < O {
            unsafe {
                let dst = outputs[j].add(idx);

                if partial {
                    let lanes = results[j].as_slice();
                    let mut e = 0;
                    while e < Vector::<S::fxN>::LANES {
                        if e < remaining {
                            *dst.add(e) = lanes[e];
                        }
                        e += 1;
                    }
                } else {
                    results[j].store_unaligned(dst);
                }
            }
            j += 1;
        }

        idx += n;
    }
}
