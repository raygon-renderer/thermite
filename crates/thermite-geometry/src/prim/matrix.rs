use core::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use thermite::vector::FloatVector;

use super::Vector;

/// Column-major matrix with C columns and R rows
///
/// The Index trait is implemented such that `matrix[c][r]`
/// accesses the element at column `c` and row `r`, zero-indexed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Matrix<V: FloatVector, const C: usize, const R: usize>(pub [[V; R]; C]);

impl<V: FloatVector, const C: usize, const R: usize> Matrix<V, C, R> {
    #[inline(always)]
    pub const fn splat(value: V) -> Self {
        Self([[value; R]; C])
    }

    #[inline(always)]
    pub const fn new(elements: [[V; R]; C]) -> Self {
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

impl<V: FloatVector, I, const C: usize, const R: usize> Index<I> for Matrix<V, C, R>
where
    [[V; R]; C]: Index<I>,
{
    type Output = <[[V; R]; C] as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<V: FloatVector, I, const C: usize, const R: usize> IndexMut<I> for Matrix<V, C, R>
where
    [[V; R]; C]: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
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

        let mut c = 0;

        while c < C {
            let mut r = 0;

            while r < R {
                result.0[r][c] = self.0[c][r];
                r += 1;
            }

            c += 1;
        }

        result
    }
}

impl<V: FloatVector, const C: usize, const R: usize, const K: usize> Mul<Matrix<V, K, C>> for Matrix<V, C, R> {
    type Output = Matrix<V, K, R>;

    #[inline(always)]
    fn mul(self, rhs: Matrix<V, K, C>) -> Self::Output {
        let mut result = Matrix::<V, K, R>::splat(V::ZERO);

        for c in 0..K {
            let col_rhs = &rhs[c];

            for r in 0..R {
                let mut sum = self[0][r] * col_rhs[0];

                for i in 1..C {
                    sum = sum.mul_adde(self[i][r], col_rhs[i]);
                }

                result[c][r] = sum;
            }

            // Carefully tuned to avoid weird shuffles
            unsafe { result[c][0].block_autovectorization() };
        }

        result
    }
}

impl<V: FloatVector, const C: usize, const R: usize> Mul<Vector<V, C>> for Matrix<V, C, R> {
    type Output = Vector<V, R>;

    #[inline(always)]
    fn mul(self, rhs: Vector<V, C>) -> Self::Output {
        let mut result = Vector::<V, R>::splat(V::ZERO);

        let scalar_0 = rhs[0];
        let col_0 = &self[0];

        for r in 0..R {
            result[r] = col_0[r] * scalar_0;
        }

        for c in 1..C {
            let scalar = rhs[c];
            let col = &self[c];

            for r in 0..R {
                result[r] = scalar.mul_adde(col[r], result[r]);
            }
        }

        result
    }
}

// treat this as if it's a vec4 with w=0
impl<V: FloatVector, const R: usize> Mul<Vector<V, 3>> for Matrix<V, 4, R> {
    type Output = Vector<V, 3>;

    fn mul(self, rhs: Vector<V, 3>) -> Self::Output {
        let mut result = Vector::<V, 3>::splat(V::ZERO);

        let scalar_0 = rhs[0];
        let col_0 = &self[0];

        for r in 0..3 {
            result[r] = col_0[r] * scalar_0;
        }

        for c in 1..3 {
            let scalar = rhs[c];
            let col = &self[c];

            for r in 0..3 {
                result[r] = scalar.mul_adde(col[r], result[r]);
            }
        }

        result
    }
}
