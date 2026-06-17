//! SDF combinators: rounding, onioning, and the boolean / smooth-boolean
//! operators from <https://iquilezles.org/articles/distfunctions>.
//!
//! Each operator is generic over its operand SDFs and re-exposes [`SDF`],
//! [`GradientSdf`] and [`BoundedSdf`] when the corresponding closed form exists.
//! Note (as IQ does): only union and xor stay true SDFs in the interior;
//! subtraction and intersection are correct outside the surface and bounds
//! inside.

use core::marker::PhantomData;

use thermite::math::TranscendentalMathWithPolicy;
use thermite::math::policy::DefaultPolicy;
use thermite::prelude::*;

use thermite_geometry::prim::{Bounds, Vector, Vector2, vector::VectorOps as _};

use crate::consts::{cint, frac};
use crate::{BoundedSdf, GradientSdf, SDF, SdfVector, unit_or_zero};

// ---------------------------------------------------------------------------
// Rounding / onioning
// ---------------------------------------------------------------------------

/// Rounds (inflates) a shape by `radius`. A constant offset leaves the gradient
/// untouched.
#[derive(Debug, Clone, Copy)]
pub struct Round<V: SdfVector, S> {
    pub shape: S,
    pub radius: V,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Round<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p) - self.radius
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Round<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (d, g) = self.shape.eval_grad(p);
        (d - self.radius, g)
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Round<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.shape.aabb().expand(self.radius)
    }
}

/// Makes a shape annular by peeling a shell of half-thickness `radius` off the surface.
///
/// Computes `|d| - radius`, so the result is zero on the original surface displaced
/// inward and outward by `radius`. Total wall thickness is `2 * radius`.
#[derive(Debug, Clone, Copy)]
pub struct Onion<V: SdfVector, S> {
    pub shape: S,
    pub radius: V,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Onion<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p).abs() - self.radius
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Onion<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (d, g) = self.shape.eval_grad(p);
        (d.abs() - self.radius, g * d.signum())
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Onion<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // the annular shell extends one `radius` beyond the original surface
        self.shape.aabb().expand(self.radius)
    }
}

// ---------------------------------------------------------------------------
// Hard booleans
// ---------------------------------------------------------------------------

/// Union of two shapes, `$\min(a, b)$` (gradient follows the nearer one).
#[derive(Debug, Clone, Copy)]
pub struct Union<A, B> {
    pub a: A,
    pub b: B,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for Union<A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.a.eval(p).min(self.b.eval(p))
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N> for Union<A, B> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        let pick = da.cmp_lt(db);
        (pick.select(da, db), pick.select(ga, gb))
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for Union<A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.a.aabb() | self.b.aabb()
    }
}

/// Intersection of two shapes, `$\max(a, b)$` (gradient follows the binding one).
#[derive(Debug, Clone, Copy)]
pub struct Intersection<A, B> {
    pub a: A,
    pub b: B,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for Intersection<A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.a.eval(p).max(self.b.eval(p))
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N>
    for Intersection<A, B>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        let pick = da.cmp_gt(db);
        (pick.select(da, db), pick.select(ga, gb))
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for Intersection<A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.a.aabb().intersection(self.b.aabb())
    }
}

/// Subtraction `$\max(-a, b)$`: carves `a` out of `b`. Not commutative.
#[derive(Debug, Clone, Copy)]
pub struct Subtraction<A, B> {
    pub a: A,
    pub b: B,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for Subtraction<A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        (-self.a.eval(p)).max(self.b.eval(p))
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N> for Subtraction<A, B> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        let na = -da;
        let pick = na.cmp_gt(db);
        (pick.select(na, db), pick.select(ga * V::NEG_ONE, gb))
    }
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for Subtraction<A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // the result is contained in `b`
        self.b.aabb()
    }
}

/// Exclusive-or `$\max(\min(a, b),\, -\max(a, b))$`: the symmetric difference. Stays a
/// true SDF; gradient is not provided (the field is piecewise from four cases).
#[derive(Debug, Clone, Copy)]
pub struct Xor<A, B> {
    pub a: A,
    pub b: B,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for Xor<A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let da = self.a.eval(p);
        let db = self.b.eval(p);
        da.min(db).max(-da.max(db))
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for Xor<A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.a.aabb() | self.b.aabb()
    }
}

// ---------------------------------------------------------------------------
// Smooth booleans (quadratic polynomial smin/smax)
// ---------------------------------------------------------------------------

/// Smooth union with blend radius `k`. Only an approximate SDF near the seam.
///
/// Uses the polynomial smooth-min (Quilez): the blend is active where `$|d_a - d_b| < 4k$`;
/// outside that band the result equals the hard union.
#[derive(Debug, Clone, Copy)]
pub struct SmoothUnion<V: SdfVector, A, B> {
    pub a: A,
    pub b: B,
    /// Blend radius. Controls the width of the smooth transition zone; larger values give a
    /// wider, more gradual blend. With `$h = \max(4k - |d_a - d_b|,\, 0)$`, the result is
    /// `$\min(a, b) - \frac{h^2}{16k}$` (the factor-of-4 internal scaling appears twice).
    pub k: V,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for SmoothUnion<V, A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let da = self.a.eval(p);
        let db = self.b.eval(p);
        let k = self.k * cint::<V, 4>();
        let h = (k - (da - db).abs()).max(V::ZERO);
        (h * h).nmul_adde(frac::<V, 1, 4>() / k, da.min(db)) // min - 0.25/k*h^2
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N>
    for SmoothUnion<V, A, B>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);

        let k = self.k * cint::<V, 4>();
        let h = (k - (da - db).abs()).max(V::ZERO);
        let n = V::HALF * h / k;

        let t = da.cmp_lt(db).select(n, V::ONE - n);
        let dist = (h * h).nmul_adde(frac::<V, 1, 4>() / k, da.min(db));
        let grad = (gb - ga).mul_adde(t, ga); // mix(ga, gb, t)
        (dist, grad)
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for SmoothUnion<V, A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        (self.a.aabb() | self.b.aabb()).expand(self.k)
    }
}

/// Smooth intersection with blend radius `k`.
#[derive(Debug, Clone, Copy)]
pub struct SmoothIntersection<V: SdfVector, A, B> {
    pub a: A,
    pub b: B,
    /// Blend radius. Blend active where `$|d_a - d_b| < 4k$`; with `$h = \max(4k - |d_a - d_b|,\, 0)$`,
    /// the result is `$\max(a, b) + \frac{h^2}{16k}$`.
    pub k: V,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for SmoothIntersection<V, A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let da = self.a.eval(p);
        let db = self.b.eval(p);
        let k = self.k * cint::<V, 4>();
        let h = (k - (da - db).abs()).max(V::ZERO);
        (h * h).mul_adde(frac::<V, 1, 4>() / k, da.max(db)) // max + 0.25/k*h^2
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N>
    for SmoothIntersection<V, A, B>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);

        let k = self.k * cint::<V, 4>();
        let h = (k - (da - db).abs()).max(V::ZERO);
        let n = V::HALF * h / k;

        let t = da.cmp_gt(db).select(n, V::ONE - n);
        let dist = (h * h).mul_adde(frac::<V, 1, 4>() / k, da.max(db));
        let grad = (gb - ga).mul_adde(t, ga);
        (dist, grad)
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N>
    for SmoothIntersection<V, A, B>
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.a.aabb().intersection(self.b.aabb())
    }
}

/// Smooth subtraction with blend radius `k`: `-smoothUnion(a, -b, k)`, carving
/// `a` out of `b` with a rounded seam.
#[derive(Debug, Clone, Copy)]
pub struct SmoothSubtraction<V: SdfVector, A, B> {
    pub a: A,
    pub b: B,
    /// Blend radius. Blend active where `$|d_a + d_b| < 4k$`; with `$h = \max(4k - |d_a + d_b|,\, 0)$`,
    /// the result is `$\max(-a, b) + \frac{h^2}{16k}$`.
    pub k: V,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>> SDF<V, N> for SmoothSubtraction<V, A, B> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let da = self.a.eval(p);
        let db = self.b.eval(p);
        let k = self.k * cint::<V, 4>();
        let h = (k - (da + db).abs()).max(V::ZERO);
        (h * h).mul_adde(frac::<V, 1, 4>() / k, (-da).max(db)) // max(-a,b) + 0.25/k*h^2
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>> GradientSdf<V, N>
    for SmoothSubtraction<V, A, B>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // smooth-max of the operands (-a) and b: same blend as SmoothIntersection
        // with the first operand negated (value and gradient).
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);

        let k = self.k * cint::<V, 4>();
        let h = (k - (da + db).abs()).max(V::ZERO);
        let n = V::HALF * h / k;

        let t = (-da).cmp_gt(db).select(n, V::ONE - n);
        let dist = (h * h).mul_adde(frac::<V, 1, 4>() / k, (-da).max(db));
        // mix(-ga, gb, t) = -ga + (gb + ga)*t
        let grad = (gb + ga).mul_adde(t, ga * V::NEG_ONE);
        (dist, grad)
    }
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: BoundedSdf<V, N>> BoundedSdf<V, N> for SmoothSubtraction<V, A, B> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.b.aabb().expand(self.k)
    }
}

// ---------------------------------------------------------------------------
// Positioning / domain operators
// ---------------------------------------------------------------------------

/// Uniformly scales a shape by `scale`.
#[derive(Debug, Clone, Copy)]
pub struct Scale<V: SdfVector, S> {
    pub shape: S,
    pub scale: V,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Scale<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p / self.scale) * self.scale
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Scale<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (d, g) = self.shape.eval_grad(p / self.scale);
        (d * self.scale, g) // uniform scale preserves the normal direction
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Scale<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.shape.aabb() * Vector::splat(self.scale)
    }
}

/// Elongates a shape, splitting it along each axis by the half-lengths `h` and
/// connecting the halves. Exact for 1D elongations.
#[derive(Debug, Clone, Copy)]
pub struct Elongate<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub h: Vector<V, N>,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Elongate<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let q = p - p.clamp(self.h * V::NEG_ONE, self.h);
        self.shape.eval(q)
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Elongate<V, S, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // the domain map is a translation, so the gradient passes through
        self.shape.eval_grad(p - p.clamp(self.h * V::NEG_ONE, self.h))
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Elongate<V, S, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // the shape is split and pulled apart by +/- h on each axis
        let mut bb = self.shape.aabb();
        for i in 0..N {
            bb.0[i][0] -= self.h[i];
            bb.0[i][1] += self.h[i];
        }
        bb
    }
}

/// Mirrors the domain across the origin on the axes marked `true` in `axes`.
#[derive(Debug, Clone, Copy)]
pub struct Symmetry<S, const N: usize> {
    pub shape: S,
    pub axes: [bool; N],
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Symmetry<S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let mut q = p;
        for i in 0..N {
            if self.axes[i] {
                q.0[i] = q.0[i].abs();
            }
        }
        self.shape.eval(q)
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for Symmetry<S, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // a mirrored axis folds onto |x|, so its box becomes [-M, M] with
        // M = max(|min|, |max|); unmirrored axes pass through.
        let mut bb = self.shape.aabb();
        for i in 0..N {
            if self.axes[i] {
                let m = bb.0[i][0].abs().max(bb.0[i][1].abs());
                bb.0[i] = [-m, m];
            }
        }
        bb
    }
}

/// Infinite domain repetition with per-axis spacing `spacing`.
#[derive(Debug, Clone, Copy)]
pub struct Repetition<V: SdfVector, S, const N: usize> {
    pub shape: S,
    /// Per-axis tile period. The domain fold is
    /// `$q_i = p_i - s_i \operatorname{round}(p_i / s_i)$`, centering a copy at every integer
    /// multiple of `spacing`. The shape must fit within `[-spacing/2, spacing/2]` to avoid
    /// overlap (which would break the SDF metric).
    pub spacing: Vector<V, N>,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for Repetition<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let q = Vector::new(core::array::from_fn(|i| {
            let s = self.spacing.0[i];
            s.nmul_adde((p.0[i] / s).round(), p.0[i]) // p - s*round(p/s)
        }));
        self.shape.eval(q)
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for Repetition<V, S, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let q = Vector::new(core::array::from_fn(|i| {
            let s = self.spacing.0[i];
            s.nmul_adde((p.0[i] / s).round(), p.0[i])
        }));
        self.shape.eval_grad(q)
    }
}

// ---------------------------------------------------------------------------
// 2D -> 3D constructors
// ---------------------------------------------------------------------------

/// Extrudes a 2D shape along z to half-depth `h`, producing a 3D solid.
#[derive(Debug, Clone, Copy)]
pub struct Extrusion<V: SdfVector, S> {
    pub shape: S,
    pub h: V,
}

impl<V: SdfVector, S: SDF<V, 2>> SDF<V, 3> for Extrusion<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, 3>) -> V {
        let d = self.shape.eval(Vector2::new([p[0], p[1]]));
        let wz = p[2].abs() - self.h;
        d.max(wz).min(V::ZERO) + Vector2::new([d.max(V::ZERO), wz.max(V::ZERO)]).l2_norm()
    }
}

impl<V: SdfVector, S: BoundedSdf<V, 2>> BoundedSdf<V, 3> for Extrusion<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // 2D profile in xy, swept over z in [-h, h]
        let bb = self.shape.aabb().0;
        Bounds([bb[0], bb[1], [-self.h, self.h]])
    }
}

/// Revolves a 2D shape (in the xy-plane) around the y axis.
///
/// Maps 3D point `p` to 2D `$\left(\sqrt{p_x^2 + p_z^2} - \text{offset},\; p_y\right)$` before
/// evaluating the inner shape. With `offset = 0` this is a standard solid of revolution; with
/// `offset > 0` the profile is displaced radially, so a 2D disk of radius `r` becomes
/// a torus of major radius `offset` and minor radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct Revolution<V: SdfVector, S> {
    pub shape: S,
    /// Radial distance from the y axis to the origin of the 2D profile. Zero gives a
    /// standard solid of revolution; positive values create torus-family shapes.
    pub offset: V,
}

impl<V: SdfVector, S: SDF<V, 2>> SDF<V, 3> for Revolution<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, 3>) -> V {
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt() - self.offset;
        self.shape.eval(Vector2::new([qx, p[1]]))
    }
}

impl<V: SdfVector, S: BoundedSdf<V, 2>> BoundedSdf<V, 3> for Revolution<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // Profile's x maps to radius (qx + offset); the solid sweeps a disk of
        // outer radius R = profile.x_max + offset in the xz-plane; y passes through.
        let bb = self.shape.aabb().0;
        let r = (bb[0][1] + self.offset).max(V::ZERO);
        Bounds([[-r, r], bb[1], [-r, r]])
    }
}

// ---------------------------------------------------------------------------
// Distortions (not exact SDFs - they bound the true distance)
// ---------------------------------------------------------------------------

/// Twists a 3D shape around the y axis at rate `k` (radians per unit height).
///
/// This is a domain distortion, not an exact SDF — the result underestimates the true
/// distance. Reduce ray-march step size proportionally to `|k| * shape_radius`.
#[derive(Debug, Clone, Copy)]
pub struct Twist<V: SdfVector, S, P: Policy = DefaultPolicy> {
    pub shape: S,
    /// Twist rate in radians per unit of y. `$k = 2\pi$` rotates a full turn over 1 unit.
    pub k: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, S, P: Policy> Twist<V, S, P> {
    #[inline(always)]
    pub const fn new(shape: S, k: V) -> Self {
        Self {
            shape,
            k,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + TranscendentalMathWithPolicy, S: SDF<V, 3>, P: Policy> SDF<V, 3> for Twist<V, S, P> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, 3>) -> V {
        let (s, c) = (self.k * p[1]).sin_cos_p::<P>();
        let qx = c.mul_sube(p[0], s * p[2]); // c*x - s*z
        let qz = s.mul_adde(p[0], c * p[2]); // s*x + c*z
        self.shape.eval(Vector::new([qx, p[1], qz]))
    }
}

/// Bends a 3D shape in the xy-plane at rate `k` (radians per unit x).
///
/// Domain distortion — not an exact SDF. Keep `|k| * shape_x_extent` well below `$\pi/2$`
/// for a usable bound.
#[derive(Debug, Clone, Copy)]
pub struct Bend<V: SdfVector, S, P: Policy = DefaultPolicy> {
    pub shape: S,
    /// Bend rate in radians per unit of x. `$k = \pi/L$` curves a shape of x-extent `L`
    /// into a semicircle.
    pub k: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, S, P: Policy> Bend<V, S, P> {
    #[inline(always)]
    pub const fn new(shape: S, k: V) -> Self {
        Self {
            shape,
            k,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + TranscendentalMathWithPolicy, S: SDF<V, 3>, P: Policy> SDF<V, 3> for Bend<V, S, P> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, 3>) -> V {
        let (s, c) = (self.k * p[0]).sin_cos_p::<P>();
        let qx = c.mul_sube(p[0], s * p[1]); // c*x - s*y
        let qy = s.mul_adde(p[0], c * p[1]); // s*x + c*y
        self.shape.eval(Vector::new([qx, qy, p[2]]))
    }
}

/// Adds a user-supplied displacement field to a shape's distance:
/// `shape(p) + displacement(p)`.
///
/// This distorts the field into a *bound* on the true distance (no longer a
/// metric SDF), so only [`SDF`] is provided. Keep `displacement` small and
/// smooth - its gradient effectively rescales the marching step.
#[derive(Debug, Clone, Copy)]
pub struct Displace<S, F> {
    pub shape: S,
    pub displacement: F,
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>, F: Fn(Vector<V, N>) -> V> SDF<V, N> for Displace<S, F> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p) + (self.displacement)(p)
    }
}

// ---------------------------------------------------------------------------
// Automatic gradient
// ---------------------------------------------------------------------------

/// Supplies a [`GradientSdf`] for any [`SDF`] via finite differences.
///
/// The normal is sampled around `p` at radius `eps` and normalised. To stay
/// unbiased (no axis shift) while keeping the eval count low, this uses the
/// simplex schemes from IQ's "normals for an SDF": the 4-tap **tetrahedron** in
/// 3D and a 3-tap equilateral triangle in 2D, falling back to `2*N`-tap central
/// differences for `$N \ge 4$`. The returned distance is the *exact* `inner.eval(p)`
/// (one more eval); only the normal is approximate. This gives the distance-only
/// primitives - and arbitrary user shapes - a usable normal with no hand-derived
/// gradient. (When autodiff dual numbers land in Thermite this becomes exact.)
///
/// `eps` trades truncation error against the field's scale/smoothness; a real
/// raymarcher should scale it with the ray's distance to band-limit aliasing.
/// [`new`](Self::new) defaults it to `$1/4096$`. [`BoundedSdf`] is forwarded.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDiff<V: SdfVector, S> {
    pub shape: S,
    pub eps: V,
}

impl<V: SdfVector, S> FiniteDiff<V, S> {
    /// Wraps `shape` with the default step `$\text{eps} = 1/4096$`.
    #[inline(always)]
    pub fn new(shape: S) -> Self {
        Self {
            shape,
            eps: frac::<V, 1, 4096>(),
        }
    }

    /// Wraps `shape` with an explicit finite-difference step `eps`.
    #[inline(always)]
    pub fn with_eps(shape: S, eps: V) -> Self {
        Self { shape, eps }
    }
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        self.shape.eval(p)
    }
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> GradientSdf<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let h = self.eps;
        let dist = self.shape.eval(p); // distance stays exact
        let mut grad = Vector::ZERO;

        if const { N == 3 } {
            // Tetrahedron technique (Falcao/Iquilez): 4 taps with central-difference
            // quality and no axis bias - cheaper than the 6-tap central form. The
            // four offsets are a regular tetrahedron inscribed in the cube:
            // (+++), (+--), (-+-), (--+). Normalization absorbs the scale factor.
            let mut j = 0;
            while j < 4 {
                let mut e = Vector::ZERO;
                let mut k = 0;
                while k < N {
                    let plus = j == 0 || k + 1 == j; // single '+' at axis j-1, else all '+'
                    e[k] = if plus { h } else { -h };
                    k += 1;
                }
                grad = e.mul_adde(self.shape.eval(p + e), grad);
                j += 1;
            }
        } else if const { N == 2 } {
            // 2D analogue: an equilateral triangle of directions summing to zero
            // (3 taps, unbiased).
            let sx = h * V::SQRT_3 * V::HALF; // h*sqrt(3)/2
            let hy = h * V::HALF;
            let dirs = [[V::ZERO, h], [-sx, -hy], [sx, -hy]];
            let mut j = 0;
            while j < 3 {
                let mut e = Vector::ZERO;
                e[0] = dirs[j][0];
                e[1] = dirs[j][1];
                grad = e.mul_adde(self.shape.eval(p + e), grad);
                j += 1;
            }
        } else {
            // General N: central differences per axis (2N taps).
            let mut i = 0;
            while i < N {
                let mut hp = p;
                let mut hm = p;
                hp[i] = p[i] + h;
                hm[i] = p[i] - h;
                grad[i] = self.shape.eval(hp) - self.shape.eval(hm);
                i += 1;
            }
        }

        (dist, unit_or_zero(grad, grad.l2_norm()))
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.shape.aabb()
    }
}
