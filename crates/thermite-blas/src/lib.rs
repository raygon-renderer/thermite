use thermite::{
    Vector,
    generic_array::{GenericArray, typenum::Unsigned},
    register::{FloatElement, FloatRegister, Register, SignedIntegerRegister, UnsignedIntegerRegister},
    simd::{Simd, SizedSimd},
    vector::generic::FloatVector,
};

pub mod kernels;
pub mod map_reduce;

use map_reduce::MapReduceKernel;

impl<S: Simd> BLAS1<S> for S {}

trait AsTuple {
    type Output;
    fn into_tuple(self) -> Self::Output;
}

impl<T: Copy> AsTuple for [T; 2] {
    type Output = (T, T);
    #[inline(always)]
    fn into_tuple(self) -> Self::Output {
        (self[0], self[1])
    }
}

/// WIP BLAS Level 1 routines implemented using Thermite SIMD abstractions.
pub trait BLAS1<S: Simd> {
    /// Single-precision dot product.
    #[inline(always)]
    fn sdot(a: &[f32], b: &[f32]) -> f32 {
        kernels::dot_product::ScalarDotProductKernel.run::<S, _>(&SimpleLoad, [a, b])[0]
    }

    /// Single-precision dot product for fixed-size arrays.
    #[inline(always)]
    fn sdot_n<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
        kernels::dot_product::ScalarDotProductKernel.run_n::<S, _, N>(&SimpleLoad, [a, b])[0]
    }

    /// Double-precision dot product.
    #[inline(always)]
    fn ddot(a: &[f64], b: &[f64]) -> f64 {
        // map_reduce::<S, f64, _, _, 0, _, _>(&SimpleLoad, &ScalarDotProductKernel, [a, b])[0]
        kernels::dot_product::ScalarDotProductKernel.run::<S, _>(&SimpleLoad, [a, b])[0]
    }

    /// Double-precision dot product for fixed-size arrays.
    #[inline(always)]
    fn ddot_n<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
        kernels::dot_product::ScalarDotProductKernel.run_n::<S, _, N>(&SimpleLoad, [a, b])[0]
    }

    /// Single-precision complex dot product without conjugation.
    #[inline(always)]
    fn cdotu(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32]) -> (f32, f32) {
        kernels::dot_product::ComplexDotProductKernel::<false>
            .run::<S, _>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Single-precision complex dot product without conjugation for fixed-size arrays.
    #[inline(always)]
    fn cdotu_n<const N: usize>(ar: &[f32; N], ai: &[f32; N], br: &[f32; N], bi: &[f32; N]) -> (f32, f32) {
        kernels::dot_product::ComplexDotProductKernel::<false>
            .run_n::<S, _, N>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Single-precision complex dot product with conjugation.
    #[inline(always)]
    fn cdotc(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32]) -> (f32, f32) {
        kernels::dot_product::ComplexDotProductKernel::<true>
            .run::<S, _>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Single-precision complex dot product with conjugation for fixed-size arrays.
    #[inline(always)]
    fn cdotc_n<const N: usize>(ar: &[f32; N], ai: &[f32; N], br: &[f32; N], bi: &[f32; N]) -> (f32, f32) {
        kernels::dot_product::ComplexDotProductKernel::<true>
            .run_n::<S, _, N>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Double-precision complex dot product without conjugation.
    #[inline(always)]
    fn zdotu(ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) -> (f64, f64) {
        kernels::dot_product::ComplexDotProductKernel::<false>
            .run::<S, _>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Double-precision complex dot product without conjugation for fixed-size arrays.
    #[inline(always)]
    fn zdotu_n<const N: usize>(ar: &[f64; N], ai: &[f64; N], br: &[f64; N], bi: &[f64; N]) -> (f64, f64) {
        kernels::dot_product::ComplexDotProductKernel::<false>
            .run_n::<S, _, N>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Double-precision complex dot product with conjugation.
    #[inline(always)]
    fn zdotc(ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) -> (f64, f64) {
        kernels::dot_product::ComplexDotProductKernel::<true>
            .run::<S, _>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }

    /// Double-precision complex dot product with conjugation for fixed-size arrays.
    #[inline(always)]
    fn zdotc_n<const N: usize>(ar: &[f64; N], ai: &[f64; N], br: &[f64; N], bi: &[f64; N]) -> (f64, f64) {
        kernels::dot_product::ComplexDotProductKernel::<true>
            .run_n::<S, _, N>(&SimpleLoad, [ar, ai, br, bi])
            .into_tuple()
    }
}

pub trait LoadKernel<const N: usize> {
    /// Load N vectors from the given pointers.
    ///
    /// # SAFETY
    /// The caller must ensure that all pointers are valid for reading.
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N];
}

pub struct SimpleLoad;
pub struct StreamLoad;

impl<const N: usize> LoadKernel<N> for SimpleLoad {
    #[inline(always)]
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        unsafe { ptrs.map(|ptr| V::load_unaligned(ptr)) }
    }
}

impl<const N: usize> LoadKernel<N> for StreamLoad {
    #[inline(always)]
    unsafe fn load<V: FloatVector>(&self, ptrs: [*const V::Element; N]) -> [V; N] {
        unsafe { ptrs.map(|ptr| V::load_streaming(ptr)) }
    }
}
