use core::ops::{Add, Div, Mul, Sub};

use thermite::generic::FloatVector;

use crate::Vector;

/// Column-major matrix with C columns and R rows
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<V: FloatVector, const C: usize, const R: usize>(pub [[V; C]; R]);

impl<V: FloatVector, const C: usize, const R: usize> Matrix<V, C, R> {
    #[inline(always)]
    pub const fn splat(value: V) -> Self {
        Self([[value; C]; R])
    }

    #[inline(always)]
    pub const fn new(elements: [[V; C]; R]) -> Self {
        Self(elements)
    }

    #[inline(always)]
    pub const fn as_slice(&self) -> &[V] {
        let res = self.0.as_flattened();

        unsafe { core::hint::assert_unchecked(res.len() == (C * R)) };

        res
    }

    #[inline(always)]
    pub const fn as_slice_mut(&mut self) -> &mut [V] {
        let res = self.0.as_flattened_mut();

        unsafe { core::hint::assert_unchecked(res.len() == (C * R)) };

        res
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Add<Self> for Matrix<V, C, R> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        for (dst, src) in self.as_slice_mut().iter_mut().zip(rhs.as_slice()) {
            *dst += *src;
        }

        self
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Sub<Self> for Matrix<V, C, R> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        for (dst, src) in self.as_slice_mut().iter_mut().zip(rhs.as_slice()) {
            *dst -= *src;
        }

        self
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Mul<V> for Matrix<V, C, R> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: V) -> Self::Output {
        for dst in self.as_slice_mut() {
            *dst *= rhs;
        }

        self
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Div<V> for Matrix<V, C, R> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, rhs: V) -> Self::Output {
        let rcp = V::ONE / rhs;

        for dst in self.as_slice_mut() {
            *dst *= rcp;
        }

        self
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Matrix<V, C, R> {
    #[inline(always)]
    pub const fn transpose(&self) -> Matrix<V, R, C> {
        let mut result = Matrix::<V, R, C>::splat(V::ZERO);

        let mut r = 0;

        while r < R {
            let mut c = 0;

            while c < C {
                result.0[c][r] = self.0[r][c];
                c += 1;
            }

            r += 1;
        }

        result
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Mul<Vector<V, C>> for Matrix<V, C, R> {
    type Output = Vector<V, R>;

    #[inline(always)]
    fn mul(self, rhs: Vector<V, C>) -> Self::Output {
        let mut result = Vector::<V, R>::splat(V::ZERO);

        for r in 0..R {
            let mut prod = rhs.0;

            // multiply, each component is independent so excellent ILP
            for (p, c) in prod.iter_mut().zip(&self.0[r]) {
                *p *= *c;

                // For Scalars, LLVM will attempt to merge these loops unless we
                // block autovectorization, and in this case the autovectorization
                // is shit. Weird blends abound. It's better for everything if
                // we just don't allow it.
                //
                // SAFETY: We want to prevent autovectorization here to improve ILP,
                // and this generally has no effect if V is not Scalar.
                unsafe { p.block_autovectorization() };
            }

            // reduce sum using log2(C) depth tree, good ILP
            thermite::math::algorithms::reduce_in_place(&mut prod, Add::add);

            result.0[r] = prod[0];
        }

        result
    }
}
