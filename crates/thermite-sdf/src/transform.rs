//! Rigid and similarity transforms of SDFs: [`Translate`], [`UniformScale`], and
//! [`Rotate`], after <https://iquilezles.org/articles/distfunctions> ("transforms").
//!
//! An SDF is transformed by mapping the query point through the *inverse* of the
//! transform and (for scale) correcting the returned distance:
//!
//! ```math
//! d_{T}(p) = s \, d\!\left(\tfrac{1}{s} R^{-1}(p - t)\right)
//! ```
//!
//! For a rotation `$R$` (orthonormal, so `$R^{-1} = R^{\top}$`) and a uniform scale
//! `$s > 0$`, this stays an exact metric SDF. Each transform is dimension-generic
//! and forwards [`GradientSdf`], [`BoundedSdf`] and [`FractalSdf`] when the operand
//! provides them - the gradient is carried back into world space (rotated for
//! [`Rotate`], unchanged otherwise), and the orbit data rides through untouched.
//!
//! Transforms compose by nesting, exactly like the boolean combinators:
//! `Translate { offset, shape: Rotate { rot, shape: UniformScale { factor, shape } } }`
//! applies scale, then rotation, then translation.
//!
//! For *domain*-repeating transforms (tiling space) see [`Repetition`] and friends
//! in [`crate::ops`]; this module is rigid/similarity transforms of a single copy.
//!
//! [`Repetition`]: crate::ops::Repetition

use thermite_geometry::soa::prim::{Bounds, Matrix, Vector, Vector3, vector::VectorOps as _};

use thermite::math::TranscendentalMath;

use crate::{BoundedSdf, FractalOrbit, FractalSdf, GradientSdf, SDF, SdfVector};

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------

/// Translates a shape by `offset`. A rigid motion: evaluate at `p - offset`; the
/// distance and gradient are unchanged, the bounding box is shifted.
#[derive(Debug, Clone, Copy)]
pub struct Translate<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub offset: Vector<V, N>,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Translate<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p - self.offset)
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Translate<V, S, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        self.shape.eval_grad(p - self.offset)
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Translate<V, S, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        let mut bb = self.shape.aabb();
        for k in 0..N {
            bb.0[k][0] += self.offset[k];
            bb.0[k][1] += self.offset[k];
        }
        bb
    }
}

impl<V: SdfVector, const N: usize, S: FractalSdf<V, N>> FractalSdf<V, N> for Translate<V, S, N> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector<V, N>) -> (V, FractalOrbit<V, N>) {
        self.shape.eval_orbit(p - self.offset)
    }
}

// ---------------------------------------------------------------------------
// Uniform scale
// ---------------------------------------------------------------------------

/// Uniformly scales a shape by `factor` (`> 0`). Evaluating at `p / factor` and
/// rescaling the distance by `factor` keeps it an exact SDF; the gradient is
/// already unit-length and so passes through unchanged.
///
/// Only *uniform* scale preserves the metric. A non-uniform (per-axis) scale turns
/// an SDF into a bound, not an exact distance - it is intentionally not offered
/// here; reach for [`Elongate`](crate::ops)-style operators if you need that.
#[derive(Debug, Clone, Copy)]
pub struct UniformScale<V: SdfVector, S> {
    pub shape: S,
    pub factor: V,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for UniformScale<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p / self.factor) * self.factor
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for UniformScale<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // d(p) = factor * f(p / factor); d/dp = factor * (1/factor) * f'(p/factor) = f'(.)
        let (d, g) = self.shape.eval_grad(p / self.factor);
        (d * self.factor, g)
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for UniformScale<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        let mut bb = self.shape.aabb();
        for k in 0..N {
            bb.0[k][0] *= self.factor;
            bb.0[k][1] *= self.factor;
        }
        bb
    }
}

impl<V: SdfVector, const N: usize, S: FractalSdf<V, N>> FractalSdf<V, N> for UniformScale<V, S> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector<V, N>) -> (V, FractalOrbit<V, N>) {
        let (d, o) = self.shape.eval_orbit(p / self.factor);
        (d * self.factor, o)
    }
}

// ---------------------------------------------------------------------------
// Rotate
// ---------------------------------------------------------------------------

/// Rotates a shape by the orthonormal matrix `rot` (mapping the shape's local
/// frame into world space). Because `rot` is a rotation, its inverse is its
/// transpose, so the query point is sent to local space with `rot^T * p` and the
/// returned gradient is carried back to world space with `rot * g` - both
/// length-preserving, so the result stays an exact SDF.
///
/// `rot` must be orthonormal (`det = 1`, `rot rot^T = I`); build one with
/// [`rotation_2d`] / [`rotation_3d`] or the [`Rotate::angle`] / [`Rotate::axis_angle`]
/// constructors. A general (non-orthonormal) linear map would need the
/// inverse-transpose for the gradient and would no longer be a metric SDF.
#[derive(Debug, Clone, Copy)]
pub struct Rotate<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub rot: Matrix<V, N, N>,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Rotate<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(self.rot.transpose() * p)
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Rotate<V, S, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (d, g) = self.shape.eval_grad(self.rot.transpose() * p);
        (d, self.rot * g)
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Rotate<V, S, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // AABB of the rotated AABB: new half-extent along world axis i is
        // sum_j |R[i][j]| * half[j], and the center maps through the rotation.
        let bb = self.shape.aabb();
        let mut center = Vector::<V, N>::ZERO;
        let mut half = Vector::<V, N>::ZERO;
        for k in 0..N {
            center[k] = (bb.0[k][0] + bb.0[k][1]) * V::HALF;
            half[k] = (bb.0[k][1] - bb.0[k][0]) * V::HALF;
        }
        let nc = self.rot * center;
        let mut out = Bounds::<V, N>([[V::ZERO; 2]; N]);
        for i in 0..N {
            let mut e = V::ZERO;
            // rot is column-major: rot.0[col][row] = R[row][col].
            for j in 0..N {
                e = self.rot.0[j][i].abs().mul_adde(half[j], e);
            }
            out.0[i][0] = nc[i] - e;
            out.0[i][1] = nc[i] + e;
        }
        out
    }
}

impl<V: SdfVector, const N: usize, S: FractalSdf<V, N>> FractalSdf<V, N> for Rotate<V, S, N> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector<V, N>) -> (V, FractalOrbit<V, N>) {
        // The orbit lives in the shape's local frame, so it forwards unchanged.
        self.shape.eval_orbit(self.rot.transpose() * p)
    }
}

impl<V: SdfVector, S, const N: usize> Rotate<V, S, N> {
    /// Wrap `shape` with an explicit orthonormal rotation matrix.
    #[inline(always)]
    pub fn new(rot: Matrix<V, N, N>, shape: S) -> Self {
        Self { shape, rot }
    }
}

impl<V: SdfVector + TranscendentalMath, S> Rotate<V, S, 2> {
    /// Rotate a 2D shape counter-clockwise by `angle` radians.
    #[inline(always)]
    pub fn angle(angle: V, shape: S) -> Self {
        Self {
            shape,
            rot: rotation_2d(angle),
        }
    }
}

impl<V: SdfVector + TranscendentalMath, S> Rotate<V, S, 3> {
    /// Rotate a 3D shape by `angle` radians about `axis` (normalized internally).
    #[inline(always)]
    pub fn axis_angle(axis: Vector3<V>, angle: V, shape: S) -> Self {
        Self {
            shape,
            rot: rotation_3d(axis, angle),
        }
    }
}

// ---------------------------------------------------------------------------
// Rotation-matrix builders
// ---------------------------------------------------------------------------

/// The 2x2 counter-clockwise rotation by `angle` radians.
#[inline]
pub fn rotation_2d<V: SdfVector + TranscendentalMath>(angle: V) -> Matrix<V, 2, 2> {
    let (s, c) = angle.sin_cos();
    // column-major: col0 = image of x-axis, col1 = image of y-axis
    Matrix::new([[c, s], [-s, c]])
}

/// The 3x3 rotation by `angle` radians about `axis` (Rodrigues' formula). `axis`
/// is normalized internally, so any nonzero length works.
#[inline]
pub fn rotation_3d<V: SdfVector + TranscendentalMath>(axis: Vector3<V>, angle: V) -> Matrix<V, 3, 3> {
    let a = axis.normalize();
    let (x, y, z) = (a[0], a[1], a[2]);
    let (s, c) = angle.sin_cos();
    let cc = V::ONE - c;

    // column-major: rot.0[col][row]; here each inner array is a column (image of a
    // basis vector). Rodrigues: R = cI + (1-c) a a^T + s [a]_x.
    Matrix::new([
        [c + x * x * cc, y * x * cc + z * s, z * x * cc - y * s],
        [x * y * cc - z * s, c + y * y * cc, z * y * cc + x * s],
        [x * z * cc + y * s, y * z * cc - x * s, c + z * z * cc],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use thermite::prelude::*;

    use crate::ops::FiniteDiff;
    use crate::{Box2D, Box3D, Circle2D, Sphere3D};
    use core::f32::consts::PI;
    use thermite_geometry::soa::prim::{Vector2, Vector3};

    type V = thermite::Vector<f32>;
    fn vs(x: f32) -> V {
        V::splat(x)
    }
    fn p2(x: f32, y: f32) -> Vector2<V> {
        Vector2::new([vs(x), vs(y)])
    }
    fn p3(x: f32, y: f32, z: f32) -> Vector3<V> {
        Vector3::new([vs(x), vs(y), vs(z)])
    }
    fn sc(x: V) -> f32 {
        x.extract::<0>()
    }

    #[test]
    fn translate_matches_offset_eval() {
        let c = Circle2D { radius: vs(1.0) };
        let t = Translate {
            shape: c,
            offset: p2(2.0, 0.0),
        };
        // distance at the new center is -radius
        assert!((sc(t.eval(p2(2.0, 0.0))) + 1.0).abs() < 1e-5);
        // matches evaluating the base shape at p - offset
        for q in [p2(3.5, 0.7), p2(0.1, -1.2), p2(2.0, 2.0)] {
            let manual = sc(c.eval(p2(sc(q[0]) - 2.0, sc(q[1]))));
            assert!((sc(t.eval(q)) - manual).abs() < 1e-5);
        }
        // bbox is shifted by the offset
        let bb = t.aabb();
        assert!((sc(bb.0[0][0]) - 1.0).abs() < 1e-5 && (sc(bb.0[0][1]) - 3.0).abs() < 1e-5);
        // gradient is the base gradient, unchanged direction
        let (_, g) = t.eval_grad(p2(4.0, 0.0));
        assert!((sc(g[0]) - 1.0).abs() < 1e-4 && sc(g[1]).abs() < 1e-4);
    }

    #[test]
    fn uniform_scale_distance_and_bounds() {
        let s = Sphere3D { radius: vs(1.0) };
        let big = UniformScale {
            shape: s,
            factor: vs(2.0),
        };
        // a radius-1 sphere scaled 2x has radius 2: surface at |p| = 2
        assert!(sc(big.eval(p3(2.0, 0.0, 0.0))).abs() < 1e-5);
        // distance at 3 along x is 1 (3 - 2)
        assert!((sc(big.eval(p3(3.0, 0.0, 0.0))) - 1.0).abs() < 1e-5);
        // gradient stays unit-length
        let (_, g) = big.eval_grad(p3(3.0, 0.0, 0.0));
        let len = (sc(g[0]).powi(2) + sc(g[1]).powi(2) + sc(g[2]).powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
        // bbox scaled (NSphere is N-generic, so the AABB's N needs annotating)
        let bb: Bounds<V, 3> = big.aabb();
        assert!((sc(bb.0[0][1]) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn rotate_2d_is_distance_invariant_and_rotates_gradient() {
        // square half-extent (1, 0.5), rotated 90 deg -> swaps the axes
        let b = Box2D { b: p2(1.0, 0.5) };
        let r = Rotate::angle(vs(PI / 2.0), b);

        // a point that was outside the +x face is now outside the +y face: distance
        // to the rotated box at (0, 1.5) equals the unrotated box distance at (1.5, 0)
        let got = sc(r.eval(p2(0.0, 1.5)));
        let want = sc(b.eval(p2(1.5, 0.0)));
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");

        // gradient cross-checks against a finite difference of the rotated field
        let fd = FiniteDiff::with_eps(r, vs(1e-3));
        for q in [p2(1.6, 0.2), p2(0.1, 1.4), p2(-1.5, -0.3)] {
            let (_, ga) = r.eval_grad(q);
            let (_, gf) = fd.eval_grad(q);
            for k in 0..2 {
                assert!(
                    (sc(ga[k]) - sc(gf[k])).abs() < 3e-2,
                    "axis {k}: {} vs {}",
                    sc(ga[k]),
                    sc(gf[k])
                );
            }
        }
    }

    #[test]
    fn rotate_3d_axis_angle_matches_finite_diff() {
        let b = Box3D {
            b: p3(1.0, 0.6, 0.4),
            r: vs(0.0),
        };
        let r = Rotate::axis_angle(p3(0.3, 1.0, 0.2), vs(0.7), b);
        let fd = FiniteDiff::with_eps(r, vs(1e-3));
        for q in [p3(1.5, 0.2, 0.1), p3(0.1, 1.3, 0.2), p3(-0.8, 0.3, 1.2)] {
            let (da, ga) = r.eval_grad(q);
            let (df, gf) = fd.eval_grad(q);
            assert!((sc(da) - sc(df)).abs() < 1e-4);
            for k in 0..3 {
                assert!((sc(ga[k]) - sc(gf[k])).abs() < 3e-2, "axis {k}");
            }
        }
    }

    #[test]
    fn rotate_aabb_contains_rotated_shape() {
        // 45-deg rotation of a unit square: the tight AABB grows to +/- sqrt(2)/2 * 2
        let b = Box2D { b: p2(1.0, 1.0) };
        let r = Rotate::angle(vs(PI / 4.0), b);
        let bb = r.aabb();
        let half = sc(bb.0[0][1]);
        assert!((half - core::f32::consts::SQRT_2).abs() < 1e-4, "half = {half}");

        // grid-sample: every interior point lies within the reported AABB
        for i in 0..=40 {
            for j in 0..=40 {
                let x = -2.0 + 4.0 * i as f32 / 40.0;
                let y = -2.0 + 4.0 * j as f32 / 40.0;
                if sc(r.eval(p2(x, y))) <= 0.0 {
                    assert!(
                        x >= sc(bb.0[0][0]) - 1e-3
                            && x <= sc(bb.0[0][1]) + 1e-3
                            && y >= sc(bb.0[1][0]) - 1e-3
                            && y <= sc(bb.0[1][1]) + 1e-3,
                        "interior ({x},{y}) escaped AABB"
                    );
                }
            }
        }
    }

    #[test]
    fn rotation_matrices_are_orthonormal() {
        // R R^T = I for both builders. Tolerance is loose because the default-policy
        // sin/cos are approximations (sin^2 + cos^2 deviates from 1 by ~1e-6), which
        // is negligible for an SDF but more than 1e-5.
        let r2 = rotation_2d(vs(0.9));
        let i2 = r2 * r2.transpose();
        assert!((sc(i2.0[0][0]) - 1.0).abs() < 1e-4 && sc(i2.0[1][0]).abs() < 1e-4);

        let r3 = rotation_3d(p3(0.2, 0.7, -0.4), vs(1.3));
        let i3 = r3 * r3.transpose();
        for c in 0..3 {
            for r in 0..3 {
                let expect = if c == r { 1.0 } else { 0.0 };
                assert!((sc(i3.0[c][r]) - expect).abs() < 1e-4, "({c},{r})");
            }
        }
    }
}
