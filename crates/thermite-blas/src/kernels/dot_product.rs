use thermite::vector::generic::FloatVector;

use crate::map_reduce::MapReduceKernel;

pub struct ScalarDotProductKernel;
pub struct SoAComplexDotProductKernel<const CONJ: bool>;

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

    #[inline(always)]
    fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 1], v: [V; 2]) -> [V; 1] {
        [v[0].mul_adde(v[1], acc[0])]
    }
}

impl<T, const CONJ: bool> MapReduceKernel<T, 4, 2> for SoAComplexDotProductKernel<CONJ> {
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
