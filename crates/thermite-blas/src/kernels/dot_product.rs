use thermite::vector::generic::FloatVector;

use crate::map_reduce::MapReduceKernel;

pub struct ScalarDotProductKernel;
pub struct ComplexDotProductKernel<const CONJ: bool>;

#[rustfmt::skip]
impl<T> MapReduceKernel<T, 2, 1> for ScalarDotProductKernel {
    const AVOID_MAP: bool = false;
    #[inline(always)] fn map<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [V; 1] { [v[0] * v[1]] }
    #[inline(always)] fn reduce<V: FloatVector<Element = T>>(&self, acc: [V; 1], v: [V; 1]) -> [V; 1] { [acc[0] + v[0]] }
    #[inline(always)] fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 1]) -> [T; 1] { [v[0].sum_elements()] }
    #[inline(always)] fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 1], v: [V; 2]) -> [V; 1] { [v[0].mul_adde(v[1], acc[0])] }
}

// Complex dot product kernel, with optional conjugation.
#[rustfmt::skip]
impl<const CONJ: bool> ComplexDotProductKernel<CONJ> {
    #[inline(always)] fn fma_r<V: FloatVector<Element = T>, T>(&self, a: V, b: V, c: V) -> V {
        if CONJ { a.mul_adde(b, c) } else { a.mul_sube(b, c) }
    }

    #[inline(always)] fn fma_i<V: FloatVector<Element = T>, T>(&self, a: V, b: V, c: V) -> V {
        if CONJ { a.mul_sube(b, c) } else { a.mul_adde(b, c) }
    }
}

#[rustfmt::skip]
impl<T, const CONJ: bool> MapReduceKernel<T, 4, 2> for ComplexDotProductKernel<CONJ> {
    const AVOID_MAP: bool = true; // not worth the extra map pass

    #[inline(always)] fn map<V: FloatVector<Element = T>>(&self, v: [V; 4]) -> [V; 2] {
        [self.fma_r(v[0], v[2], v[1] * v[3]),
         self.fma_i(v[0], v[3], v[1] * v[2])]
    }

    #[inline(always)] fn reduce<V: FloatVector<Element = T>>(&self, a: [V; 2], b: [V; 2]) -> [V; 2] {
        [a[0] + b[0],
         a[1] + b[1]]
    }

    #[inline(always)] fn reduce_scalar<V: FloatVector<Element = T>>(&self, v: [V; 2]) -> [T; 2] {
        [v[0].sum_elements(),
         v[1].sum_elements()]
    }

    #[inline(always)] fn map_reduce<V: FloatVector<Element = T>>(&self, acc: [V; 2], v: [V; 4]) -> [V; 2] {
        [self.fma_r(v[0], v[2], v[1].mul_adde(v[3], acc[0])),
         self.fma_i(v[0], v[3], v[1].mul_adde(v[2], acc[1]))]
    }
}
