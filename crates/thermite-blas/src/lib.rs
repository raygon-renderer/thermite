use thermite::{
    generic::FloatVector,
    register::well_formed::WellFormedFloatElement,
    simd::{FloatSimd, Simd},
};

pub mod argmx;
pub mod kernels;
pub mod map_reduce;

use map_reduce::MapReduceKernel;

impl<S: Simd> BLAS<S> for S {}

#[derive(Copy, Clone, Debug)]
pub struct Complex<V: FloatVector> {
    pub re: V,
    pub im: V,
}

macro_rules! decl_complex_dp {
    ($ty:ident, $conj:ident) => {paste::paste! {
        #[doc = concat!(decl_complex_dp!(TY_DOC $ty), " complex dot product ", decl_complex_dp!(CONJ_DOC $conj), " - planar layout.")]
        #[inline(always)]
        fn [<$ty dot $conj _planar>](
            ar: &[ decl_complex_dp!(TY_TY $ty) ],
            ai: &[ decl_complex_dp!(TY_TY $ty) ],
            br: &[ decl_complex_dp!(TY_TY $ty) ],
            bi: &[ decl_complex_dp!(TY_TY $ty) ],
        ) -> (
            decl_complex_dp!(TY_TY $ty),
            decl_complex_dp!(TY_TY $ty),
        ) {
            let [re, im] = kernels::dot_product::ComplexDotProductKernel::< { decl_complex_dp!(CONJ_VAL $conj) } >.run::<S, _, _>(&SimpleLoad, [ar, ai, br, bi]);
            (re, im)
        }

        #[doc = concat!(decl_complex_dp!(TY_DOC $ty), " complex dot product ", decl_complex_dp!(CONJ_DOC $conj), " - interleaved layout.")]
        ///
        /// # Panics
        ///
        /// Panics if the input arrays' lengths are not even, as interleaved complex numbers
        /// require pairs of real and imaginary parts.
        #[inline(always)]
        fn [<$ty dot $conj _interleaved>](
            a: &[ decl_complex_dp!(TY_TY $ty) ],
            b: &[ decl_complex_dp!(TY_TY $ty) ],
        ) -> (
            decl_complex_dp!(TY_TY $ty),
            decl_complex_dp!(TY_TY $ty),
        ) {
            assert!(a.len() % 2 == 0, "Input array 'a' length must be even for interleaved complex numbers.");
            assert!(b.len() % 2 == 0, "Input array 'b' length must be even for interleaved complex numbers.");

            let [re, im] = kernels::dot_product::ComplexDotProductKernel::< { decl_complex_dp!(CONJ_VAL $conj) } >.run::<S, _, _>(&DeInterleavedLoad(SimpleLoad), [a, b]);
            (re, im)
        }

        #[doc = concat!(decl_complex_dp!(TY_DOC $ty), " complex dot product ", decl_complex_dp!(CONJ_DOC $conj), " - interleaved layout.")]
        ///
        /// # Panics
        ///
        /// Panics if the input arrays' lengths are not even, as interleaved complex numbers
        /// require pairs of real and imaginary parts.
        #[inline(always)]
        fn [<$ty dot $conj _interleaved_kahan>](
            a: &[ decl_complex_dp!(TY_TY $ty) ],
            b: &[ decl_complex_dp!(TY_TY $ty) ],
        ) -> (
            decl_complex_dp!(TY_TY $ty),
            decl_complex_dp!(TY_TY $ty),
        ) {
            assert!(a.len() % 2 == 0, "Input array 'a' length must be even for interleaved complex numbers.");
            assert!(b.len() % 2 == 0, "Input array 'b' length must be even for interleaved complex numbers.");

            let [re, im, ..] = kernels::dot_product::ComplexDotProductKahanKernel::< { decl_complex_dp!(CONJ_VAL $conj) } >.run::<S, _, _>(&DeInterleavedLoad(SimpleLoad), [a, b]);
            (re, im)
        }

        #[doc = concat!(decl_complex_dp!(TY_DOC $ty), " complex dot product ", decl_complex_dp!(CONJ_DOC $conj), " - planar layout, fixed-size arrays.\n\n")]
        /// Using fixed size arrays allows for better optimization opportunities.
        #[inline(always)]
        fn [<$ty dot $conj _planar_n>]<const N: usize>(
            ar: &[ decl_complex_dp!(TY_TY $ty); N],
            ai: &[ decl_complex_dp!(TY_TY $ty); N],
            br: &[ decl_complex_dp!(TY_TY $ty); N],
            bi: &[ decl_complex_dp!(TY_TY $ty); N],
        ) -> (
            decl_complex_dp!(TY_TY $ty),
            decl_complex_dp!(TY_TY $ty),
        ) {
            let [re, im] = kernels::dot_product::ComplexDotProductKernel::< { decl_complex_dp!(CONJ_VAL $conj) } >.run_n::<S, _, N, _>(&SimpleLoad, [ar, ai, br, bi]);
            (re, im)
        }

        #[doc = concat!(decl_complex_dp!(TY_DOC $ty), " complex dot product ", decl_complex_dp!(CONJ_DOC $conj), " - interleaved layout, fixed-size arrays.\n\n")]
        /// Using fixed size arrays allows for better optimization opportunities.
        ///
        /// # Panics
        ///
        /// Panics if the input arrays' lengths are not even, as interleaved complex numbers
        /// require pairs of real and imaginary parts.
        #[inline(always)]
        fn [<$ty dot $conj _interleaved_n>]<const N: usize>(
            a: &[ decl_complex_dp!(TY_TY $ty); N],
            b: &[ decl_complex_dp!(TY_TY $ty); N],
        ) -> (
            decl_complex_dp!(TY_TY $ty),
            decl_complex_dp!(TY_TY $ty),
        ) {
            assert!(N % 2 == 0, "Input array length must be even for interleaved complex numbers.");
            let [re, im] = kernels::dot_product::ComplexDotProductKernel::< { decl_complex_dp!(CONJ_VAL $conj) } >.run_n::<S, _, N, _>(&DeInterleavedLoad(SimpleLoad), [a, b]);
            (re, im)
        }
    }};

    (TY_TY c) => { f32 };
    (TY_TY z) => { f64 };

    (TY_DOC c) => { "Single-precision" };
    (TY_DOC z) => { "Double-precision" };

    (CONJ_DOC c) => { "with conjugation" };
    (CONJ_DOC u) => { "without conjugation" };

    (CONJ_VAL c) => { true };
    (CONJ_VAL u) => { false };
}

/// WIP BLAS routines implemented using Thermite SIMD abstractions.
pub trait BLAS<S: Simd> {
    /// Find the indices of the minimum and maximum elements in a slice.
    ///
    /// Returns `None` if the input slice is empty.
    #[inline(always)]
    fn arg_minmax<T>(data: &[T]) -> Option<(usize, usize)>
    where
        T: WellFormedFloatElement,
        S: FloatSimd<T>,
    {
        argmx::arg_minmax::<S, T, _, 1>(&SimpleLoad, [data])
    }

    /// Single-precision dot product.
    #[inline(always)]
    fn sdot(a: &[f32], b: &[f32]) -> f32 {
        kernels::dot_product::ScalarDotProductKernel.run::<S, _, _>(&SimpleLoad, [a, b])[0]
    }

    /// Single-precision dot product for fixed-size arrays.
    #[inline(always)]
    fn sdot_n<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
        kernels::dot_product::ScalarDotProductKernel.run_n::<S, _, N, _>(&SimpleLoad, [a, b])[0]
    }

    /// Double-precision dot product.
    #[inline(always)]
    fn ddot(a: &[f64], b: &[f64]) -> f64 {
        // map_reduce::<S, f64, _, _, 0, _, _>(&SimpleLoad, &ScalarDotProductKernel, [a, b])[0]
        kernels::dot_product::ScalarDotProductKernel.run::<S, _, _>(&SimpleLoad, [a, b])[0]
    }

    /// Double-precision dot product for fixed-size arrays.
    #[inline(always)]
    fn ddot_n<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
        kernels::dot_product::ScalarDotProductKernel.run_n::<S, _, N, _>(&SimpleLoad, [a, b])[0]
    }

    decl_complex_dp!(c, u); // single-precision complex dot product without conjugation
    decl_complex_dp!(c, c); // single-precision complex dot product with conjugation
    decl_complex_dp!(z, u); // double-precision complex dot product without conjugation
    decl_complex_dp!(z, c); // double-precision complex dot product with conjugation

    /// Complex Dot Product using slices of `Complex<V>` structs.
    #[inline(always)]
    fn complex_dot_product<V: FloatVector, const CONJ: bool>(
        a: &[Complex<V>],
        b: &[Complex<V>],
    ) -> (V::Element, V::Element) {
        let loader = |[a, b]: [Complex<V>; 2]| -> [V; 4] { [a.re, a.im, b.re, b.im] };

        let res = kernels::dot_product::ComplexDotProductKernel::<CONJ>.run_simple(loader, [a, b]);

        (res[0], res[1])
    }
}

/// Trait for loading vectors from memory.
///
/// The generic parameters I and O represent the number of input pointers
/// and the number of output vectors, respectively.
pub trait LoadKernel<const I: usize, const O: usize> {
    /// The number of vectors wide to advance the pointer after each load.
    ///
    /// Some load kernels may load more data than they return.
    const READ_MULTIPLIER: usize;

    /// Load N vectors from the given pointers.
    ///
    /// # SAFETY
    /// The caller must ensure that all pointers are valid for reading.
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; I]) -> [V; O];
}

/// Simple load kernel that performs unaligned loads.
pub struct SimpleLoad;

/// Streaming load kernel that performs non-temporal loads to minimize cache pollution.
pub struct StreamLoad;

/// Loader that de-interleaves complex numbers from interleaved memory layout,
/// reading two vectors twice and unpacking them into 4 separate vectors.
pub struct DeInterleavedLoad<L: LoadKernel<2, 2>>(pub L);

impl<const N: usize> LoadKernel<N, N> for SimpleLoad {
    const READ_MULTIPLIER: usize = 1;

    #[inline(always)]
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        unsafe { ptrs.map(|ptr| V::load_unaligned(ptr)) }
    }
}

impl<const N: usize> LoadKernel<N, N> for StreamLoad {
    const READ_MULTIPLIER: usize = 1;

    #[inline(always)]
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        unsafe { ptrs.map(|ptr| V::load_streaming(ptr)) }
    }
}

impl<L: LoadKernel<2, 2>> LoadKernel<2, 4> for DeInterleavedLoad<L> {
    // we consume twice the advance for each load
    const READ_MULTIPLIER: usize = 2 * L::READ_MULTIPLIER;

    #[inline(always)]
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; 2]) -> [V; 4] {
        let [a0, b0] = unsafe { self.0.load::<V>(ptrs) };

        // advance pointers by one vector lane and load again
        let [a1, b1] = unsafe { self.0.load::<V>(ptrs.map(|ptr| ptr.add(V::LANES * L::READ_MULTIPLIER))) };

        // unpack interleaved complex numbers
        let ((ar, ai), (br, bi)) = (a0.unpack(a1), b0.unpack(b1));

        [ar, ai, br, bi]
    }
}
