use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use thermite::{
    mask::GenericMask,
    math::{SpatialMathWithPolicy, TranscendentalMath},
    vector::FloatVector,
};

use super::{Bounds, Point, Vector};

#[doc(inline)]
pub use crate::gamma;

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
                    // sum += self[i][r] * col_rhs[i]
                    sum = self[i][r].mul_adde(col_rhs[i], sum);
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

/// `Matrix * Vector3` for the matrix shapes that have three rows to write into:
/// the vector is treated as a `vec4` with `w = 0`, so the translation column
/// does not apply and this is the operator form of
/// [`transform_vector`](Matrix::transform_vector).
///
/// Deliberately **not** generic over the row count. The body writes rows `0..3`,
/// so a `Matrix<V, 4, R>` with `R < 3` would index past the end of a column and
/// panic at runtime. Spelling out the two shapes that fit turns that into a
/// compile error instead. (No overlap with the general
/// `Mul<Vector<V, C>> for Matrix<V, C, R>` impl above: that one takes a
/// `Vector<V, 4>` when `C == 4`, this one a `Vector<V, 3>`.)
macro_rules! impl_mat4_mul_vector3 {
    ($($rows:literal),*) => {$(
        impl<V: FloatVector> Mul<Vector<V, 3>> for Matrix<V, 4, $rows> {
            type Output = Vector<V, 3>;

            #[inline(always)]
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
    )*};
}

// 4x4 (the usual affine/projective matrix) and 4x3 (a compact affine one).
impl_mat4_mul_vector3!(3, 4);

impl<V: FloatVector, const C: usize, const R: usize> Neg for Matrix<V, C, R> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self::Output {
        for dst in self.as_slice_mut() {
            *dst = -*dst;
        }

        self
    }
}

impl<V: FloatVector, const N: usize> Matrix<V, N, N> {
    /// The multiplicative identity: ones on the diagonal, zeroes elsewhere.
    pub const IDENTITY: Self = {
        let mut m = [[V::ZERO; N]; N];

        let mut i = 0;

        while i < N {
            m[i][i] = V::ONE;
            i += 1;
        }

        Self(m)
    };

    /// A diagonal matrix with `diagonal` on the diagonal.
    #[inline(always)]
    pub fn from_diagonal(diagonal: Vector<V, N>) -> Self {
        let mut m = Self::splat(V::ZERO);

        for i in 0..N {
            m.0[i][i] = diagonal[i];
        }

        m
    }

    /// The sum of the diagonal entries.
    #[inline(always)]
    pub fn trace(&self) -> V {
        let mut trace = self.0[0][0];

        for i in 1..N {
            trace += self.0[i][i];
        }

        trace
    }

    /// A mask set for the lanes whose matrix is the identity to within `tolerance`.
    #[inline(always)]
    pub fn is_identity(&self, tolerance: V) -> V::Mask {
        let identity = Self::IDENTITY;

        // Seeded from the truthy mask rather than from entry (0, 0), which the
        // loop below already covers. Seeding with it tested that entry twice.
        let mut mask = <V::Mask as GenericMask>::TRUTHY;

        for c in 0..N {
            for r in 0..N {
                mask &= (self.0[c][r] - identity.0[c][r]).abs().cmp_le(tolerance);
            }
        }

        mask
    }
}

/// The cross product of two 3-element columns, the building block of the 3x3
/// inverse (whose rows are the cross products of the input's columns).
#[inline(always)]
fn cross3<V: FloatVector>(a: [V; 3], b: [V; 3]) -> [V; 3] {
    [
        a[1].mul_sube(b[2], a[2] * b[1]),
        a[2].mul_sube(b[0], a[0] * b[2]),
        a[0].mul_sube(b[1], a[1] * b[0]),
    ]
}

/// `x*a - y*b + z*c` as one multiply and two FMAs.
///
/// The alternating signs ride in the FMA opcodes (`nmul_adde` is `c - a*b`), so
/// no separate negations are emitted. A serial FMA chain is deliberate here
/// rather than a balanced tree: for three terms both shapes are three
/// operations deep, but the chain is one instruction *fewer* (1 mul + 2 FMA
/// against 2 mul + 1 FMA + 1 add) and rounds three times instead of four, since
/// the products inside an FMA are exact. Balancing pays from four terms up -
/// see the determinant below.
#[inline(always)]
fn cofactor3<V: FloatVector>(x: V, a: V, y: V, b: V, z: V, c: V) -> V {
    x.mul_adde(a, y.nmul_adde(b, z * c))
}

/// `-(x*a) + y*b - z*c`, the sign-flipped [`cofactor3`], at the same cost.
#[inline(always)]
fn ncofactor3<V: FloatVector>(x: V, a: V, y: V, b: V, z: V, c: V) -> V {
    x.nmul_adde(a, z.nmul_adde(c, y * b))
}

/// The adjugate and the determinant of a 4x4, by Laplace expansion on 2x2
/// sub-determinants: 16 cofactors over 18 shared minors, entirely branch-free.
///
/// `m` is the column-major flattening of the input (`m[c * 4 + r]` is row `r` of
/// column `c`), but the returned adjugate is **row-major**: `adj[r * 4 + c]` is
/// the `(r, c)` entry, i.e. the transpose of the input's layout. That falls out
/// of the cofactor expansion (the adjugate *is* the transposed cofactor matrix)
/// and is why the determinant contracts `m`'s first column with `adj`'s first
/// row, and why `invert_det` reads `adj[r * 4 + c]` rather than `adj[c * 4 + r]`.
///
/// # Shape
///
/// Every cofactor is a 3x3 determinant, and expanding it along its first row
/// leaves `coeff * (2x2 minor)` summed three ways. The minors depend only on the
/// *row pair* they are drawn from, so the 18 named below cover all 16 entries,
/// each one reused by four cofactors.
///
/// Writing the same result flat (one triple product per term) costs 96
/// multiplies and leaves the optimizer to rediscover the common subexpressions,
/// which it only partly does. Measured on AVX2 `f32x8`, the arithmetic in
/// `invert()` went from **154 mul + 46 sub + 34 add + 3 FMA** to **52 mul +
/// 52 FMA + 1 add**, or 237 arithmetic instructions down to 105. Note the 3:
/// the flat form is not merely redundant, it hides the fusable shape, so almost
/// no FMA was emitted at all.
///
/// Each minor is a difference of products, built with `mul_sube` so it is one
/// FMA rather than two roundings and a cancelling subtraction.
///
/// Note that this is the ONE-SIDED form, not the compensated one
/// `cross3::<false>` uses: only the subtracted product rounds, so the two sides
/// no longer round alike and a minor of a rank-deficient matrix comes back as
/// the discarded rounding instead of exactly zero. That is a deliberate open
/// item, not a claim that one-sided is the accurate choice.
#[inline(always)]
#[rustfmt::skip]
fn adjugate_det4<V: FloatVector>(m: &[V]) -> ([V; 16], V) {
    // Name the entries by (row, column). Storage is column-major.
    let (m00, m10, m20, m30) = (m[ 0], m[ 1], m[ 2], m[ 3]);
    let (m01, m11, m21, m31) = (m[ 4], m[ 5], m[ 6], m[ 7]);
    let (m02, m12, m22, m32) = (m[ 8], m[ 9], m[10], m[11]);
    let (m03, m13, m23, m33) = (m[12], m[13], m[14], m[15]);

    // 2x2 minors, named `<rows><cols>`: `s` spans rows 2 and 3, `t` rows 1 and 3,
    // `u` rows 1 and 2. The trailing digits are the column pair.
    let s01 = m20.mul_sube(m31, m30 * m21);
    let s02 = m20.mul_sube(m32, m30 * m22);
    let s03 = m20.mul_sube(m33, m30 * m23);
    let s12 = m21.mul_sube(m32, m31 * m22);
    let s13 = m21.mul_sube(m33, m31 * m23);
    let s23 = m22.mul_sube(m33, m32 * m23);

    let t01 = m10.mul_sube(m31, m30 * m11);
    let t02 = m10.mul_sube(m32, m30 * m12);
    let t03 = m10.mul_sube(m33, m30 * m13);
    let t12 = m11.mul_sube(m32, m31 * m12);
    let t13 = m11.mul_sube(m33, m31 * m13);
    let t23 = m12.mul_sube(m33, m32 * m13);

    let u01 = m10.mul_sube(m21, m20 * m11);
    let u02 = m10.mul_sube(m22, m20 * m12);
    let u03 = m10.mul_sube(m23, m20 * m13);
    let u12 = m11.mul_sube(m22, m21 * m12);
    let u13 = m11.mul_sube(m23, m21 * m13);
    let u23 = m12.mul_sube(m23, m22 * m13);

    // Column `c` of the adjugate deletes column `c` of the input, so it draws on
    // the minors from the rows that survive: columns 0 and 1 use `s` (rows 2, 3),
    // column 2 uses `t` (rows 1, 3), column 3 uses `u` (rows 1, 2). The
    // coefficients are row 1 for column 0 and row 0 for the rest.
    let adj = [
        // column 0
         cofactor3(m11, s23, m12, s13, m13, s12),
        ncofactor3(m01, s23, m02, s13, m03, s12),
         cofactor3(m01, t23, m02, t13, m03, t12),
        ncofactor3(m01, u23, m02, u13, m03, u12),
        // column 1
        ncofactor3(m10, s23, m12, s03, m13, s02),
         cofactor3(m00, s23, m02, s03, m03, s02),
        ncofactor3(m00, t23, m02, t03, m03, t02),
         cofactor3(m00, u23, m02, u03, m03, u02),
        // column 2
         cofactor3(m10, s13, m11, s03, m13, s01),
        ncofactor3(m00, s13, m01, s03, m03, s01),
         cofactor3(m00, t13, m01, t03, m03, t01),
        ncofactor3(m00, u13, m01, u03, m03, u01),
        // column 3
        ncofactor3(m10, s12, m11, s02, m12, s01),
         cofactor3(m00, s12, m01, s02, m02, s01),
        ncofactor3(m00, t12, m01, t02, m02, t01),
         cofactor3(m00, u12, m01, u02, m02, u01),
    ];

    // Estrin: the two halves are independent, so this is three levels deep
    // instead of the four a serial FMA chain would need, and unlike the
    // three-term cofactors above, it costs no extra operation to say so.
    let det = m00.mul_adde(adj[0], m10 * adj[1]) + m20.mul_adde(adj[2], m30 * adj[3]);

    (adj, det)
}

impl<V: FloatVector> Matrix<V, 3, 3> {
    /// Widens a 3x3 into the linear block of a 4x4, filling in `[0, 0, 0, 1]`.
    ///
    /// The inverse of [`Matrix::linear`], and what recomposes the scale factor
    /// that [`Transform::decompose`](super::Transform::decompose) hands back.
    #[inline(always)]
    pub fn to_homogeneous(self) -> Matrix<V, 4, 4> {
        let mut out = Matrix::<V, 4, 4>::IDENTITY;

        for c in 0..3 {
            for r in 0..3 {
                out.0[c][r] = self.0[c][r];
            }
        }

        out
    }
}

impl<V: FloatVector, const N: usize> Matrix<V, N, N> {
    /// The determinant.
    ///
    /// Closed-form (a handful of multiplies, no division, no data-dependent
    /// control flow) for `N <= 4`; Gaussian elimination beyond that.
    #[inline(always)]
    pub fn determinant(&self) -> V {
        // `N` is a constant, so exactly one arm of this ladder survives
        // monomorphization, and the others fold away with their indices.
        if N == 1 {
            return self.0[0][0];
        }

        if N == 2 {
            return self.0[0][0].mul_sube(self.0[1][1], self.0[0][1] * self.0[1][0]);
        }

        if N == 3 {
            // det = c0 . (c1 x c2)
            let c0 = [self.0[0][0], self.0[0][1], self.0[0][2]];
            let c1 = [self.0[1][0], self.0[1][1], self.0[1][2]];
            let c2 = [self.0[2][0], self.0[2][1], self.0[2][2]];

            let k = cross3(c1, c2);

            return c0[0].mul_adde(k[0], c0[1].mul_adde(k[1], c0[2] * k[2]));
        }

        if N == 4 {
            return adjugate_det4(self.as_slice()).1;
        }

        self.eliminate().1
    }

    /// The inverse.
    ///
    /// Specializes to a closed form for the sizes that matter: a 2x2 is six
    /// operations, a 3x3 is three cross products (its rows are the cross products
    /// of the input's columns), and a 4x4 is the branch-free Laplace adjugate.
    /// Larger matrices fall back to Gauss-Jordan elimination with lane-wise
    /// partial pivoting.
    ///
    /// Singular lanes (zero determinant) come back as infinities or NaN; use
    /// [`try_invert`](Self::try_invert) when you need to know which.
    #[inline(always)]
    pub fn invert(&self) -> Self {
        self.invert_det().0
    }

    /// The inverse, plus a mask set for the lanes whose matrix was actually
    /// invertible (non-zero, finite determinant). Values in the other lanes are
    /// unspecified.
    #[inline(always)]
    pub fn try_invert(&self) -> (Self, V::Mask) {
        let (inverse, det) = self.invert_det();

        (inverse, det.cmp_ne(V::ZERO) & det.is_finite())
    }

    /// The inverse and the determinant, which every size computes together
    /// anyway (the determinant is the scale factor the adjugate is divided by).
    #[inline(always)]
    fn invert_det(&self) -> (Self, V) {
        let mut result = Self::splat(V::ZERO);

        if N == 1 {
            let det = self.0[0][0];

            result.0[0][0] = V::ONE / det;

            return (result, det);
        }

        if N == 2 {
            let (a, b) = (self.0[0][0], self.0[0][1]);
            let (c, d) = (self.0[1][0], self.0[1][1]);

            let det = a.mul_sube(d, b * c);
            let inv_det = V::ONE / det;

            result.0[0][0] = d * inv_det;
            result.0[0][1] = -b * inv_det;
            result.0[1][0] = -c * inv_det;
            result.0[1][1] = a * inv_det;

            return (result, det);
        }

        if N == 3 {
            let c0 = [self.0[0][0], self.0[0][1], self.0[0][2]];
            let c1 = [self.0[1][0], self.0[1][1], self.0[1][2]];
            let c2 = [self.0[2][0], self.0[2][1], self.0[2][2]];

            // The rows of the adjugate are the cross products of the columns.
            let r0 = cross3(c1, c2);
            let r1 = cross3(c2, c0);
            let r2 = cross3(c0, c1);

            let det = c0[0].mul_adde(r0[0], c0[1].mul_adde(r0[1], c0[2] * r0[2]));
            let inv_det = V::ONE / det;

            for c in 0..3 {
                result.0[c][0] = r0[c] * inv_det;
                result.0[c][1] = r1[c] * inv_det;
                result.0[c][2] = r2[c] * inv_det;
            }

            return (result, det);
        }

        if N == 4 {
            let (adj, det) = adjugate_det4(self.as_slice());

            let inv_det = V::ONE / det;

            for c in 0..4 {
                for r in 0..4 {
                    // `adj` is row-major (see `adjugate_det4`), so the transpose is
                    // in the index, not in an extra pass.
                    result.0[c][r] = adj[r * 4 + c] * inv_det;
                }

                unsafe { result.0[c][0].block_autovectorization() };
            }

            return (result, det);
        }

        self.eliminate()
    }

    /// Gauss-Jordan elimination with **lane-wise** partial pivoting: the general
    /// fallback for `N > 4`, returning the inverse and the determinant.
    ///
    /// Every lane holds a different matrix, so a pivot search cannot branch, since each
    /// lane may want a different pivot row. Instead the candidate rows are swapped
    /// *conditionally*, under a mask, which bubbles the largest-magnitude entry
    /// into the pivot position independently in each lane (a masked selection-sort
    /// pass). Every subsequent operation is unconditional, so all lanes stay in
    /// lockstep.
    #[inline(always)]
    fn eliminate(&self) -> (Self, V) {
        let mut a = *self;
        let mut inverse = Self::IDENTITY;

        // The determinant is the product of the pivots, negated once per row swap.
        let mut det = V::ONE;
        let mut sign = V::ONE;

        for k in 0..N {
            // Partial pivot: bubble the largest |a[k][r]| for r >= k into row k.
            for r in (k + 1)..N {
                let swap = a.0[k][r].abs().cmp_gt(a.0[k][k].abs());

                for c in 0..N {
                    // Copies, because the two rows live in the same column array and
                    // cannot both be borrowed mutably out of it.
                    let (mut top, mut bottom) = (a.0[c][k], a.0[c][r]);
                    let (mut inv_top, mut inv_bottom) = (inverse.0[c][k], inverse.0[c][r]);

                    swap.swap(&mut top, &mut bottom);
                    swap.swap(&mut inv_top, &mut inv_bottom);

                    a.0[c][k] = top;
                    a.0[c][r] = bottom;

                    inverse.0[c][k] = inv_top;
                    inverse.0[c][r] = inv_bottom;
                }

                sign = sign.neg_c(swap);
            }

            let pivot = a.0[k][k];

            det *= pivot;

            // Normalize the pivot row. A singular lane divides by zero here and
            // poisons that lane with infinities, which is exactly what the
            // `try_invert` mask reports on.
            let inv_pivot = V::ONE / pivot;

            for c in 0..N {
                a.0[c][k] *= inv_pivot;
                inverse.0[c][k] *= inv_pivot;
            }

            // Eliminate column k from every other row.
            for r in 0..N {
                if r == k {
                    continue;
                }

                let factor = a.0[k][r];

                for c in 0..N {
                    a.0[c][r] = a.0[c][k].nmul_adde(factor, a.0[c][r]);
                    inverse.0[c][r] = inverse.0[c][k].nmul_adde(factor, inverse.0[c][r]);
                }
            }
        }

        (inverse, det * sign)
    }
}

impl<V: FloatVector> Matrix<V, 4, 4> {
    /// The upper-left 3x3 block: the linear (rotation/scale/shear) part of an
    /// affine transform, with the translation column dropped.
    #[inline(always)]
    pub fn linear(&self) -> Matrix<V, 3, 3> {
        Matrix([
            [self.0[0][0], self.0[0][1], self.0[0][2]],
            [self.0[1][0], self.0[1][1], self.0[1][2]],
            [self.0[2][0], self.0[2][1], self.0[2][2]],
        ])
    }

    /// The translation column of an affine transform.
    #[inline(always)]
    pub fn translation(&self) -> Vector<V, 3> {
        Vector([self.0[3][0], self.0[3][1], self.0[3][2]])
    }

    /// Transform a **point** by an affine matrix: `M * [p, 1]`, dropping `w`.
    ///
    /// The translation column applies. Only correct when the last row is
    /// `[0, 0, 0, 1]` (no perspective). Use
    /// [`project_point`](Self::project_point) otherwise.
    #[inline(always)]
    pub fn transform_point(&self, p: Point<V, 3>) -> Point<V, 3> {
        let mut result = [V::ZERO; 3];

        for r in 0..3 {
            // The w = 1 of the point is what pulls in the translation column.
            result[r] = self.0[0][r].mul_adde(
                p[0],
                self.0[1][r].mul_adde(p[1], self.0[2][r].mul_adde(p[2], self.0[3][r])),
            );
        }

        Point(result)
    }

    /// Transform a **vector** by the matrix: `M * [v, 0]`.
    ///
    /// The `w = 0` suppresses the translation column, which is what distinguishes
    /// a direction from a position.
    #[inline(always)]
    pub fn transform_vector(&self, v: Vector<V, 3>) -> Vector<V, 3> {
        let mut result = [V::ZERO; 3];

        for r in 0..3 {
            result[r] = self.0[0][r].mul_adde(v[0], self.0[1][r].mul_adde(v[1], self.0[2][r] * v[2]));
        }

        Vector(result)
    }

    /// Transform a point by a **projective** matrix: `M * [p, 1]`, then divide
    /// `xyz` by the resulting `w`.
    ///
    /// Correct for any matrix, including perspective ones whose last row is not
    /// `[0, 0, 0, 1]`. Costs a reciprocal and three multiplies more than
    /// [`transform_point`](Self::transform_point), so prefer that one when the
    /// matrix is known to be affine.
    #[inline(always)]
    pub fn project_point(&self, p: Point<V, 3>) -> Point<V, 3> {
        let out = self.transform_point(p);

        let w = self.0[0][3].mul_adde(
            p[0],
            self.0[1][3].mul_adde(p[1], self.0[2][3].mul_adde(p[2], self.0[3][3])),
        );

        let inv_w = V::ONE / w;

        Point([out[0] * inv_w, out[1] * inv_w, out[2] * inv_w])
    }

    /// Transform a **normal** by the inverse-transpose of the 3x3 block.
    ///
    /// A normal is defined by the plane it is perpendicular to, and a
    /// non-uniform scale or shear tilts that plane the other way, so a normal
    /// does not transform like a direction. The result is **not** renormalized.
    ///
    /// This builds the cofactor matrix directly rather than inverting: the
    /// cofactor is the adjugate's transpose, so it points normals exactly where
    /// `$(M^{-1})^T$` would, differing only by the determinant, a positive
    /// scale that renormalizing removes anyway. It is cheaper and, unlike a true
    /// inverse, never singular. Prefer
    /// [`Transform::transform_normal`](super::Transform::transform_normal) when
    /// an inverse is already cached.
    #[inline(always)]
    pub fn transform_normal(&self, n: Vector<V, 3>) -> Vector<V, 3> {
        // Rows of the cofactor matrix are the cross products of the input's
        // columns, the same identity the 3x3 inverse uses.
        let c0 = [self.0[0][0], self.0[0][1], self.0[0][2]];
        let c1 = [self.0[1][0], self.0[1][1], self.0[1][2]];
        let c2 = [self.0[2][0], self.0[2][1], self.0[2][2]];

        let r0 = cross3(c1, c2);
        let r1 = cross3(c2, c0);
        let r2 = cross3(c0, c1);

        // Contract n against those rows: (cofactor * n) with cofactor row-major.
        Vector(core::array::from_fn(|i| {
            n[0].mul_adde(r0[i], n[1].mul_adde(r1[i], n[2] * r2[i]))
        }))
    }

    /// Transform an axis-aligned box, returning the AABB of the transformed box.
    ///
    /// A rotated box is no longer axis-aligned, so this is the union of all eight
    /// transformed corners, which is conservative and the tightest axis-aligned box that
    /// still contains the transformed geometry.
    #[inline(always)]
    pub fn transform_bounds(&self, bounds: &Bounds<V, 3>) -> Bounds<V, 3> {
        let mut result = Bounds::EMPTY;

        for corner in 0..8 {
            result |= self.transform_point(bounds.vertex(corner));
        }

        result
    }
}

impl<V: FloatVector> Matrix<V, 4, 4> {
    /// A pure translation.
    #[inline(always)]
    pub fn from_translation(delta: Vector<V, 3>) -> Self {
        let mut m = Self::IDENTITY;

        m.0[3][0] = delta[0];
        m.0[3][1] = delta[1];
        m.0[3][2] = delta[2];

        m
    }

    /// A pure (non-uniform) scale about the origin.
    #[inline(always)]
    pub fn from_scale(scale: Vector<V, 3>) -> Self {
        let mut m = Self::IDENTITY;

        m.0[0][0] = scale[0];
        m.0[1][1] = scale[1];
        m.0[2][2] = scale[2];

        m
    }

    /// Assemble an affine transform from its three basis columns and a translation.
    #[inline(always)]
    pub fn from_basis(x: Vector<V, 3>, y: Vector<V, 3>, z: Vector<V, 3>, translation: Vector<V, 3>) -> Self {
        Matrix([
            [x[0], x[1], x[2], V::ZERO],
            [y[0], y[1], y[2], V::ZERO],
            [z[0], z[1], z[2], V::ZERO],
            [translation[0], translation[1], translation[2], V::ONE],
        ])
    }
}

impl<V: FloatVector + TranscendentalMath> Matrix<V, 4, 4> {
    /// Rotation by `angle` radians about a **pre-normalized** axis (Rodrigues).
    #[inline(always)]
    pub fn from_axis_angle(axis: Vector<V, 3>, angle: V) -> Self {
        let (sin, cos) = angle.sin_cos();

        let (x, y, z) = (axis[0], axis[1], axis[2]);
        let omc = V::ONE - cos;

        // Each entry is  axis_i * axis_j * (1 - cos)  -/+ axis_k * sin.
        Matrix([
            [
                x.mul_adde(x * omc, cos),
                x.mul_adde(y * omc, z * sin),
                x.mul_adde(z * omc, -(y * sin)),
                V::ZERO,
            ],
            [
                x.mul_adde(y * omc, -(z * sin)),
                y.mul_adde(y * omc, cos),
                y.mul_adde(z * omc, x * sin),
                V::ZERO,
            ],
            [
                x.mul_adde(z * omc, y * sin),
                y.mul_adde(z * omc, -(x * sin)),
                z.mul_adde(z * omc, cos),
                V::ZERO,
            ],
            [V::ZERO, V::ZERO, V::ZERO, V::ONE],
        ])
    }

    /// Perspective projection with a vertical field of view (radians), mapping
    /// `z` from `[near, far]` onto `[0, 1]`.
    #[inline(always)]
    pub fn perspective(fov: V, near: V, far: V) -> Self {
        let inv_tan = V::ONE / (fov * V::HALF).tan();
        let range = far - near;

        Matrix([
            [inv_tan, V::ZERO, V::ZERO, V::ZERO],
            [V::ZERO, inv_tan, V::ZERO, V::ZERO],
            [V::ZERO, V::ZERO, far / range, V::ONE],
            [V::ZERO, V::ZERO, -(far * near) / range, V::ZERO],
        ])
    }

    /// Orthographic projection mapping `z` from `[near, far]` onto `[0, 1]`.
    #[inline(always)]
    pub fn orthographic(near: V, far: V) -> Self {
        let inv_range = V::ONE / (far - near);

        let mut m = Self::IDENTITY;

        m.0[2][2] = inv_range;
        m.0[3][2] = -near * inv_range;

        m
    }
}

impl<V: SpatialMathWithPolicy> Matrix<V, 4, 4> {
    /// Transform a vector, and bound the floating-point error of the result.
    ///
    /// Each row of the product is a dot product of 3 terms, so its error is bounded
    /// by `gamma(3) * sum_j |M_rj| * |v_j|`. See [`gamma`].
    #[inline(always)]
    pub fn transform_vector_with_error(&self, v: Vector<V, 3>) -> (Vector<V, 3>, Vector<V, 3>) {
        let mut error = [V::ZERO; 3];

        for r in 0..3 {
            error[r] = self.0[0][r]
                .abs()
                .mul_adde(v[0].abs(), self.0[1][r].abs().mul_adde(v[1].abs(), self.0[2][r].abs() * v[2].abs()));
        }

        (self.transform_vector(v), Vector(error) * gamma::<V>(3))
    }

    /// Transform a point, and bound the floating-point error of the result.
    ///
    /// The translation column contributes to the bound because the point's implicit
    /// `w = 1` multiplies it.
    #[inline(always)]
    pub fn transform_point_with_error(&self, p: Point<V, 3>) -> (Point<V, 3>, Vector<V, 3>) {
        let mut error = [V::ZERO; 3];

        for r in 0..3 {
            error[r] = self.0[0][r].abs().mul_adde(
                p[0].abs(),
                self.0[1][r].abs().mul_adde(
                    p[1].abs(),
                    self.0[2][r].abs().mul_adde(p[2].abs(), self.0[3][r].abs()),
                ),
            );
        }

        (self.transform_point(p), Vector(error) * gamma::<V>(3))
    }

    /// Transform a point that already carries an error interval, propagating it.
    ///
    /// `error' = gamma(3) * sum_j |M_rj| |p_j| + (1 + gamma(3)) * sum_j |M_rj| e_j`
    #[inline(always)]
    pub fn transform_point_propagate_error(
        &self,
        p: Point<V, 3>,
        error: Vector<V, 3>,
    ) -> (Point<V, 3>, Vector<V, 3>) {
        let (out, from_p) = self.transform_point_with_error(p);

        let mut from_e = [V::ZERO; 3];

        for r in 0..3 {
            from_e[r] = self.0[0][r].abs().mul_adde(
                error[0].abs(),
                self.0[1][r]
                    .abs()
                    .mul_adde(error[1].abs(), self.0[2][r].abs() * error[2].abs()),
            );
        }

        let g3 = gamma::<V>(3);
        let from_e = Vector(from_e);

        (out, from_p + from_e.mul_adde(g3, from_e))
    }
}

/// `Matrix4x4 * Point3`, the affine transform (the point's `w` is 1, so the
/// translation column applies).
impl<V: FloatVector> Mul<Point<V, 3>> for Matrix<V, 4, 4> {
    type Output = Point<V, 3>;

    #[inline(always)]
    fn mul(self, rhs: Point<V, 3>) -> Self::Output {
        self.transform_point(rhs)
    }
}

/// `Matrix4x4 * Bounds3`, the AABB of the transformed box.
impl<V: FloatVector> Mul<Bounds<V, 3>> for Matrix<V, 4, 4> {
    type Output = Bounds<V, 3>;

    #[inline(always)]
    fn mul(self, rhs: Bounds<V, 3>) -> Self::Output {
        self.transform_bounds(&rhs)
    }
}
