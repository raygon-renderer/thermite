use thermite::generic::{FloatVector, NumericVector};

use crate::map_reduce::MapReduceKernel;

pub struct ScalarDotProductKernel;
pub struct ScalarDotProductKahanKernel;
pub struct ComplexDotProductKernel<const CONJ: bool>;
pub struct ComplexDotProductKahanKernel<const CONJ: bool>;

impl<T> MapReduceKernel<T, 2, 1> for ScalarDotProductKernel {
    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [V; 1] {
        [v[0] * v[1]]
    }

    #[inline(always)]
    fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; 1], v: [V; 1]) -> [V; 1] {
        [acc[0] + v[0]]
    }

    #[inline(always)]
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 1]) -> [T; 1] {
        [v[0].sum_elements()]
    }

    const INTERMEDIATES: usize = 0;

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 1], v: [V; 2]) -> [V; 1] {
        [v[0].mul_adde(v[1], acc[0])]
    }
}

impl<T> MapReduceKernel<T, 2, 2> for ScalarDotProductKahanKernel {
    #[inline(always)]
    fn filter<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [V; 2] {
        // combine compensation into sum before further processing
        [v[0] + v[1], <V as NumericVector>::ZERO]
    }

    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [V; 2] {
        // Return product and zero compensation
        [v[0] * v[1], <V as NumericVector>::ZERO]
    }

    #[inline(always)]
    fn reduce<V: FloatVector<Element = T>>(&self, a: [V; 2], b: [V; 2]) -> [V; 2] {
        let sum = a[0] + b[0];
        // Recover rounding error: (Sum - A) - B
        let err = (sum - a[0]) - b[0];
        // Accumulate existing compensations plus the new error
        let comp = a[1] + b[1] + err;

        [sum, comp]
    }

    const INTERMEDIATES: usize = 2;

    #[inline(always)]
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [T; 2] {
        [v[0].sum_elements(), v[1].sum_elements()]
    }

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 2], v: [V; 2]) -> [V; 2] {
        let prod = v[0] * v[1];

        // Kahan summation
        // y = prod - c
        // t = sum + y
        // c = (t - sum) - y
        // sum = t
        let y = prod - acc[1];
        let t = acc[0] + y;
        let c = (t - acc[0]) - y;

        [t, c]
    }
}

impl<T, const CONJ: bool> MapReduceKernel<T, 4, 2> for ComplexDotProductKernel<CONJ> {
    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, v: [V; 4]) -> [V; 2] {
        if CONJ {
            // conjugate: (a_re + i*a_im) * (b_re - i*b_im) = a_re*b_re + a_im*b_im + i*(a_re*b_im - a_im*b_re)
            [v[0].mul_adde(v[2], v[1] * v[3]), v[0].mul_sube(v[3], v[1] * v[2])]
        } else {
            // no conjugate: (a_re + i*a_im) * (b_re + i*b_im) = a_re*b_re - a_im*b_im + i*(a_re*b_im + a_im*b_re)
            [v[0].mul_sube(v[2], v[1] * v[3]), v[0].mul_adde(v[3], v[1] * v[2])]
        }
    }

    #[inline(always)]
    fn reduce<V: FloatVector<Element = T>>(&self, a: [V; 2], b: [V; 2]) -> [V; 2] {
        [a[0] + b[0], a[1] + b[1]]
    }

    #[inline(always)]
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [T; 2] {
        [v[0].sum_elements(), v[1].sum_elements()]
    }

    const INTERMEDIATES: usize = 0;

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 2], v: [V; 4]) -> [V; 2] {
        let re = if CONJ {
            // conjugate: re += a_re * b_re + a_im * b_im
            v[0].mul_adde(v[2], v[1].mul_adde(v[3], acc[0]))
        } else {
            // no conjugate: re += a_re * b_re - a_im * b_im
            v[0].mul_adde(v[2], v[1].nmul_adde(v[3], acc[0]))
        };

        let im = if CONJ {
            // conjugate: im += a_re * b_im - a_im * b_re
            v[0].mul_adde(v[3], v[1].nmul_adde(v[2], acc[1]))
        } else {
            // no conjugate: im += a_re * b_im + a_im * b_re
            v[0].mul_adde(v[3], v[1].mul_adde(v[2], acc[1]))
        };

        [re, im]
    }
}

impl<T, const CONJ: bool> MapReduceKernel<T, 4, 4> for ComplexDotProductKahanKernel<CONJ> {
    #[inline(always)]
    fn filter<V: FloatVector<Element = T>>(&self, v: [V; 4]) -> [V; 4] {
        // combine compensation into sum before further processing
        let sum_re = v[0] + v[2];
        let sum_im = v[1] + v[3];

        [sum_re, sum_im, <V as NumericVector>::ZERO, <V as NumericVector>::ZERO]
    }

    #[inline(always)]
    fn map<V: FloatVector<Element = T>>(&self, v: [V; 4]) -> [V; 4] {
        let (prod_re, prod_im) = if CONJ {
            (v[0].mul_adde(v[2], v[1] * v[3]), v[0].mul_sube(v[3], v[1] * v[2]))
        } else {
            (v[0].mul_sube(v[2], v[1] * v[3]), v[0].mul_adde(v[3], v[1] * v[2]))
        };

        // Return product and zero-initialized compensation vectors
        [prod_re, prod_im, <V as NumericVector>::ZERO, <V as NumericVector>::ZERO]
    }

    #[inline(always)]
    fn reduce<V: FloatVector<Element = T>>(&self, a: [V; 4], b: [V; 4]) -> [V; 4] {
        // 1. Real Part
        let sum_re = a[0] + b[0];
        let err_re = (sum_re - a[0]) - b[0];
        let comp_re = a[2] + b[2] + err_re;

        // 2. Imaginary Part
        let sum_im = a[1] + b[1];
        let err_im = (sum_im - a[1]) - b[1];
        let comp_im = a[3] + b[3] + err_im;

        [sum_re, sum_im, comp_re, comp_im]
    }

    #[inline(always)]
    fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 4]) -> [T; 4] {
        // We sum the elements of the vectors.
        // The result is [TotalSumRe, TotalSumIm, TotalCompRe, TotalCompIm]
        // The caller is responsible for the final application of Comp to Sum if desired.
        [
            v[0].sum_elements(),
            v[1].sum_elements(),
            v[2].sum_elements(),
            v[3].sum_elements(),
        ]
    }

    const INTERMEDIATES: usize = 1;

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 4], v: [V; 4]) -> [V; 4] {
        // 1. Calculate Product
        let (prod_re, prod_im) = if CONJ {
            (v[0].mul_adde(v[2], v[1] * v[3]), v[0].mul_sube(v[3], v[1] * v[2]))
        } else {
            (v[0].mul_sube(v[2], v[1] * v[3]), v[0].mul_adde(v[3], v[1] * v[2]))
        };

        // 2. Kahan Summation for Real Part
        // y = input - c
        // t = sum + y
        // c = (t - sum) - y
        // sum = t
        let y_re = prod_re - acc[2];
        let t_re = acc[0] + y_re;
        let c_re = (t_re - acc[0]) - y_re;
        let sum_re = t_re;

        // 3. Kahan Summation for Imaginary Part
        let y_im = prod_im - acc[3];
        let t_im = acc[1] + y_im;
        let c_im = (t_im - acc[1]) - y_im;
        let sum_im = t_im;

        [sum_re, sum_im, c_re, c_im]
    }
}
