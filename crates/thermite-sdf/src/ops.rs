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

use crate::consts::{cint, frac, vint};
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
// Smooth-min kernel family (Quilez "smooth minimum", the DD family)
// ---------------------------------------------------------------------------

/// A smooth-minimum kernel `$g(x)$` of the "Direct-Difference" family
/// (`<https://iquilezles.org/articles/smin>`).
///
/// All members express the smooth-min through a single kernel `$g$`:
///
/// ```math
/// \operatorname{smin}(a, b) = b - k'\, g(x), \qquad
/// x = \frac{b - a}{k'}, \qquad k' = \frac{k}{g(0)}.
/// ```
///
/// The normalization `$g(0)$` makes the blend-band thickness equal `k` in
/// distance units across every kernel (so they are interchangeable). The kernel
/// behaves like a relaxed `$\max(x, 0)$`: `$g \to x$` as `$x \to +\infty$` and
/// `$g \to 0$` as `$x \to -\infty$`. Its derivative gives the analytic gradient
/// as a blend of the operand gradients,
///
/// ```math
/// \nabla\operatorname{smin} = g'(x)\,\nabla a + \bigl(1 - g'(x)\bigr)\,\nabla b,
/// \qquad g'(x) \in [0, 1],
/// ```
/// so `$g'(x)$` doubles as the material-mixing weight toward operand `a`.
///
/// The "Clamped-Difference" members (`Quadratic`, `Cubic`, `Quartic`,
/// `Circular`) satisfy `$g(\pm1)$`/`$g'(\pm1)$` boundary conditions that make the
/// blend strictly local to the `$|a-b| < k$` band and never overestimate
/// distance; `Root` is non-rigid (distorts everywhere) but cheap.
pub trait SmoothKernel: Copy {
    /// Normalization `$g(0)$`.
    fn g0<V: SdfVector>() -> V;
    /// Kernel `$g(x)$`.
    fn g<V: SdfVector>(x: V) -> V;
    /// Derivative `$g'(x) \in [0,1]$` (the blend weight toward operand `a`).
    fn gp<V: SdfVector>(x: V) -> V;
}

/// `(smin(a, b, k), weight_of_a)` for kernel `K`. `weight_of_a == g'(x)` is the
/// mix factor used by both the gradient and material blending.
#[inline(always)]
fn kernel_smin<K: SmoothKernel, V: SdfVector>(a: V, b: V, k: V) -> (V, V) {
    let kp = k / K::g0::<V>();
    let x = (b - a) / kp;
    let val = (-kp).mul_adde(K::g::<V>(x), b); // b - k'*g(x)
    (val, K::gp::<V>(x))
}

/// Scalar smooth-min `smin(a, b, k)` for kernel `K`, value only (skips the mix
/// weight that [`kernel_smin`] also returns).
#[inline(always)]
pub(crate) fn smin_k<K: SmoothKernel, V: SdfVector>(a: V, b: V, k: V) -> V {
    let kp = k / K::g0::<V>();
    let x = (b - a) / kp;
    (-kp).mul_adde(K::g::<V>(x), b) // b - k'*g(x)
}

/// Scalar smooth-max `smax(a, b, k) = -smin(-a, -b, k)` for kernel `K`.
#[inline(always)]
pub(crate) fn smax_k<K: SmoothKernel, V: SdfVector>(a: V, b: V, k: V) -> V {
    -smin_k::<K, V>(-a, -b, k)
}

/// Quadratic-polynomial kernel - fast, near-circular, conservative; the default
/// and most common choice.
///
/// ```math
/// g(x) = \begin{cases}
///   0 & x \le -1 \\[2pt]
///   \dfrac{x(2+x)+1}{4} & -1 \le x \le 1 \\[6pt]
///   x & x \ge 1
/// \end{cases}, \qquad g(0) = \tfrac14
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Quadratic;
impl SmoothKernel for Quadratic {
    #[inline(always)]
    fn g0<V: SdfVector>() -> V {
        frac::<V, 1, 4>()
    }
    #[inline(always)]
    fn g<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        let core = xc.mul_adde(xc + V::TWO, V::ONE) * frac::<V, 1, 4>(); // (x(2+x)+1)/4
        x.cmp_gt(V::ONE).select(x, core)
    }
    #[inline(always)]
    fn gp<V: SdfVector>(x: V) -> V {
        ((x + V::ONE) * V::HALF).clamp(V::ZERO, V::ONE) // clamp((x+1)/2, 0, 1)
    }
}

/// Cubic-polynomial kernel - slightly wider, smoother blend than [`Quadratic`].
/// Clamped to `$0$` / `$x$` outside `$[-1, 1]$`.
///
/// ```math
/// g(x) = \frac{1 + 3x(x+1) - |x|^3}{6} \ \ (-1 \le x \le 1), \qquad g(0) = \tfrac16
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Cubic;
impl SmoothKernel for Cubic {
    #[inline(always)]
    fn g0<V: SdfVector>() -> V {
        frac::<V, 1, 6>()
    }
    #[inline(always)]
    fn g<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // (1 + 3x(x+1) - |x|^3)/6
        let core = (xc * (xc + V::ONE)).mul_adde(cint::<V, 3>(), V::ONE - xc.abs() * (xc * xc))
            * frac::<V, 1, 6>();
        x.cmp_gt(V::ONE).select(x, core)
    }
    #[inline(always)]
    fn gp<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // (2x + 1 - x|x|)/2
        xc.nmul_adde(xc.abs(), xc.mul_adde(V::TWO, V::ONE)) * V::HALF
    }
}

/// Quartic-polynomial kernel - the smoothest of the polynomial CD members.
/// Clamped to `$0$` / `$x$` outside `$[-1, 1]$`.
///
/// ```math
/// g(x) = \frac{(x+1)^2\,(3 - x(x-2))}{16} \ \ (-1 \le x \le 1), \qquad g(0) = \tfrac{3}{16}
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Quartic;
impl SmoothKernel for Quartic {
    #[inline(always)]
    fn g0<V: SdfVector>() -> V {
        frac::<V, 3, 16>()
    }
    #[inline(always)]
    fn g<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // (x+1)^2 (3 - x(x-2)) / 16
        let xp = xc + V::ONE;
        let core = (xp * xp) * xc.nmul_adde(xc - V::TWO, cint::<V, 3>()) * frac::<V, 1, 16>();
        x.cmp_gt(V::ONE).select(x, core)
    }
    #[inline(always)]
    fn gp<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // (x+1)^2 (2 - x) / 4
        let xp = xc + V::ONE;
        (xp * xp) * (V::TWO - xc) * frac::<V, 1, 4>()
    }
}

/// Circular kernel - the only CD member with an exactly circular blend profile
/// between perpendicular surfaces (uses one sqrt). Clamped to `$0$` / `$x$`
/// outside `$[-1, 1]$`.
///
/// ```math
/// g(x) = 1 + \frac{x - \sqrt{2 - x^2}}{2} \ \ (-1 \le x \le 1), \qquad g(0) = 1 - \tfrac{1}{\sqrt2}
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Circular;
impl SmoothKernel for Circular {
    #[inline(always)]
    fn g0<V: SdfVector>() -> V {
        V::ONE - V::FRAC_1_SQRT_2
    }
    #[inline(always)]
    fn g<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // 1 + (x - sqrt(2 - x^2))/2
        let core = (xc - xc.nmul_adde(xc, V::TWO).sqrt()).mul_adde(V::HALF, V::ONE);
        x.cmp_gt(V::ONE).select(x, core)
    }
    #[inline(always)]
    fn gp<V: SdfVector>(x: V) -> V {
        let xc = x.clamp(V::NEG_ONE, V::ONE);
        // (1 + x/sqrt(2 - x^2))/2
        (xc / xc.nmul_adde(xc, V::TWO).sqrt()).mul_adde(V::HALF, V::HALF)
    }
}

/// Square-root kernel - smooth everywhere and associative, but non-rigid (it
/// distorts the operands at all distances, with no clamp).
///
/// ```math
/// g(x) = \frac{x + \sqrt{x^2 + 1}}{2}, \qquad g(0) = \tfrac12
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Root;
impl SmoothKernel for Root {
    #[inline(always)]
    fn g0<V: SdfVector>() -> V {
        V::HALF
    }
    #[inline(always)]
    fn g<V: SdfVector>(x: V) -> V {
        // (x + sqrt(x^2 + 1))/2
        (x + x.mul_adde(x, V::ONE).sqrt()) * V::HALF
    }
    #[inline(always)]
    fn gp<V: SdfVector>(x: V) -> V {
        // (1 + x/sqrt(x^2 + 1))/2
        (x / x.mul_adde(x, V::ONE).sqrt()).mul_adde(V::HALF, V::HALF)
    }
}

/// Smooth union with a selectable [`SmoothKernel`] and blend radius `k`.
///
/// Generalizes [`SmoothUnion`] (which is this with [`Quadratic`]) to the whole
/// DD family. Besides [`SDF`]/[`GradientSdf`], it exposes [`blend`](Self::blend)
/// returning the blend weight for mixing per-operand materials/colors.
#[derive(Debug, Clone, Copy)]
pub struct SmoothUnionK<V: SdfVector, A, B, K: SmoothKernel = Quadratic> {
    pub a: A,
    pub b: B,
    pub k: V,
    pub kernel: K,
}

impl<V: SdfVector, A, B, K: SmoothKernel> SmoothUnionK<V, A, B, K> {
    /// `(distance, weight)` where `weight in [0, 1]` is the blend fraction of
    /// operand `b` (0 = fully `a`, 1 = fully `b`) - use it to `mix` materials.
    #[inline(always)]
    pub fn blend<const N: usize>(&self, p: Vector<V, N>) -> (V, V)
    where
        A: SDF<V, N>,
        B: SDF<V, N>,
    {
        let (da, db) = (self.a.eval(p), self.b.eval(p));
        let (val, wa) = kernel_smin::<K, V>(da, db, self.k);
        (val, V::ONE - wa) // weight of b
    }
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>, K: SmoothKernel> SDF<V, N>
    for SmoothUnionK<V, A, B, K>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        kernel_smin::<K, V>(self.a.eval(p), self.b.eval(p), self.k).0
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>, K: SmoothKernel>
    GradientSdf<V, N> for SmoothUnionK<V, A, B, K>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        let (val, wa) = kernel_smin::<K, V>(da, db, self.k);
        // grad = wa*ga + (1-wa)*gb = mix(gb, ga, wa)
        let grad = (ga - gb).mul_adde(wa, gb);
        (val, grad)
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>, K: SmoothKernel>
    BoundedSdf<V, N> for SmoothUnionK<V, A, B, K>
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        (self.a.aabb() | self.b.aabb()).expand(self.k)
    }
}

/// Smooth intersection with a selectable [`SmoothKernel`]: `$\operatorname{smax}
/// (a,b) = -\operatorname{smin}(-a,-b)$`.
#[derive(Debug, Clone, Copy)]
pub struct SmoothIntersectionK<V: SdfVector, A, B, K: SmoothKernel = Quadratic> {
    pub a: A,
    pub b: B,
    pub k: V,
    pub kernel: K,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>, K: SmoothKernel> SDF<V, N>
    for SmoothIntersectionK<V, A, B, K>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        -kernel_smin::<K, V>(-self.a.eval(p), -self.b.eval(p), self.k).0
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>, K: SmoothKernel>
    GradientSdf<V, N> for SmoothIntersectionK<V, A, B, K>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        // wa = weight of (-a) in smin(-a,-b); grad(smax) = wa*ga + (1-wa)*gb
        let (val, wa) = kernel_smin::<K, V>(-da, -db, self.k);
        let grad = (ga - gb).mul_adde(wa, gb);
        (-val, grad)
    }
}

impl<V: SdfVector, const N: usize, A: BoundedSdf<V, N>, B: BoundedSdf<V, N>, K: SmoothKernel>
    BoundedSdf<V, N> for SmoothIntersectionK<V, A, B, K>
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.a.aabb().intersection(self.b.aabb())
    }
}

/// Smooth subtraction with a selectable [`SmoothKernel`]: carves `a` out of `b`
/// via `$\operatorname{smax}(-a, b) = -\operatorname{smin}(a, -b)$`.
#[derive(Debug, Clone, Copy)]
pub struct SmoothSubtractionK<V: SdfVector, A, B, K: SmoothKernel = Quadratic> {
    pub a: A,
    pub b: B,
    pub k: V,
    pub kernel: K,
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: SDF<V, N>, K: SmoothKernel> SDF<V, N>
    for SmoothSubtractionK<V, A, B, K>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        -kernel_smin::<K, V>(self.a.eval(p), -self.b.eval(p), self.k).0
    }
}

impl<V: SdfVector, const N: usize, A: GradientSdf<V, N>, B: GradientSdf<V, N>, K: SmoothKernel>
    GradientSdf<V, N> for SmoothSubtractionK<V, A, B, K>
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (da, ga) = self.a.eval_grad(p);
        let (db, gb) = self.b.eval_grad(p);
        // smax(-a, b) = -smin(a, -b); wa = weight of a in smin(a,-b).
        // grad = -wa*ga + (1-wa)*gb
        let (val, wa) = kernel_smin::<K, V>(da, -db, self.k);
        let grad = gb.mul_adde(V::ONE - wa, ga * -wa); // (1-wa)*gb + (-wa)*ga
        (-val, grad)
    }
}

impl<V: SdfVector, const N: usize, A: SDF<V, N>, B: BoundedSdf<V, N>, K: SmoothKernel> BoundedSdf<V, N>
    for SmoothSubtractionK<V, A, B, K>
{
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

/// Correct infinite domain repetition: scans the nearest `$2^N$` tiles so the
/// field stays a true SDF even when the inner shape is **not** symmetric about
/// the tile boundary (`<https://iquilezles.org/articles/sdfrepetition>`).
///
/// Plain [`Repetition`] only evaluates the tile containing `p`, which
/// underestimates distance whenever the closest copy lives in a neighbouring
/// tile (asymmetric or off-centre shapes). This variant additionally checks the
/// neighbour in the direction of `$\operatorname{sign}(p - s\,\text{id})$` on
/// each axis - 2 tiles in 1D, 4 in 2D, 8 in 3D - and takes the min. It assumes
/// each copy still fits within roughly one tile; larger shapes need a wider
/// scan. Costs `$2^N$` inner evals, so no analytic gradient is offered (wrap in
/// [`FiniteDiff`]).
#[derive(Debug, Clone, Copy)]
pub struct CorrectRepetition<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub spacing: Vector<V, N>,
}

#[inline(always)]
fn repetition_tile<V: SdfVector, const N: usize>(
    p: Vector<V, N>,
    spacing: Vector<V, N>,
) -> (Vector<V, N>, Vector<V, N>) {
    let mut id = Vector::ZERO;
    let mut o = Vector::ZERO;
    for i in 0..N {
        id[i] = (p[i] / spacing[i]).round();
        o[i] = spacing[i].nmul_adde(id[i], p[i]).signum(); // sign(p - s*id)
    }
    (id, o)
}

/// Min over the `$2^N$` candidate tiles whose id is `base + bit*o`, optionally
/// clamped to `[lo, hi]` (for the finite-grid variant). `clamp` is identity when
/// `lo`/`hi` are `None`.
#[inline(always)]
fn repetition_scan<V: SdfVector, const N: usize, S: SDF<V, N>>(
    shape: &S,
    p: Vector<V, N>,
    spacing: Vector<V, N>,
    base: Vector<V, N>,
    o: Vector<V, N>,
    limit: Option<(Vector<V, N>, Vector<V, N>)>,
) -> V {
    let mut d = V::INFINITY;
    let corners = 1usize << N;
    let mut mask = 0usize;
    while mask < corners {
        let mut r = Vector::ZERO;
        for i in 0..N {
            let mut rid = base[i] + if (mask >> i) & 1 == 1 { o[i] } else { V::ZERO };
            if let Some((lo, hi)) = limit {
                rid = rid.clamp(lo[i], hi[i]);
            }
            r[i] = spacing[i].nmul_adde(rid, p[i]); // p - s*rid
        }
        d = d.min(shape.eval(r));
        mask += 1;
    }
    d
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for CorrectRepetition<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (id, o) = repetition_tile(p, self.spacing);
        repetition_scan(&self.shape, p, self.spacing, id, o, None)
    }
}

/// Finite (limited) domain repetition: an `N`-D grid of copies whose tile ids
/// are clamped to `[lo, hi]` (`<https://iquilezles.org/articles/sdfrepetition>`).
///
/// Unlike intersecting an infinite [`Repetition`] with a container box - which
/// produces a broken field near the edges - clamping the *id* keeps the result a
/// correct SDF everywhere: outside the grid you measure distance to the nearest
/// edge copy. Like [`CorrectRepetition`] it scans `$2^N$` neighbours, so it also
/// works for asymmetric shapes.
#[derive(Debug, Clone, Copy)]
pub struct LimitedRepetition<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub spacing: Vector<V, N>,
    /// Inclusive minimum tile id on each axis.
    pub lo: Vector<V, N>,
    /// Inclusive maximum tile id on each axis.
    pub hi: Vector<V, N>,
}

impl<V: SdfVector, S, const N: usize> LimitedRepetition<V, S, N> {
    /// Grid of `counts[i]` copies per axis centred on the origin, spaced by
    /// `spacing`. Even counts straddle the origin (ids `..., -1, 0` etc.).
    #[inline(always)]
    pub fn centered(shape: S, spacing: Vector<V, N>, counts: [u32; N]) -> Self {
        let mut lo = Vector::ZERO;
        let mut hi = Vector::ZERO;
        for i in 0..N {
            // ids span `counts` consecutive integers centred on 0:
            // [-floor((n-1)/2), ceil((n-1)/2)] = [-(n/2), (n-1)/2] for our purposes.
            let n = counts[i] as thermite::LargeInt;
            lo[i] = vint::<V>(-(n / 2));
            hi[i] = vint::<V>((n - 1) / 2);
        }
        Self { shape, spacing, lo, hi }
    }
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for LimitedRepetition<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (id, o) = repetition_tile(p, self.spacing);
        repetition_scan(&self.shape, p, self.spacing, id, o, Some((self.lo, self.hi)))
    }
}

/// Fast infinite repetition that mirrors every other tile so the closest copy is
/// always in the current tile - a single inner eval, no neighbour scan
/// (`<https://iquilezles.org/articles/sdfrepetition>`, after Fizzer).
///
/// Reflecting odd tiles forces symmetry across every boundary, which is exactly
/// the condition under which naive single-tile repetition is already correct.
/// The trade-off is the mirrored layout (every other copy is flipped); when that
/// is acceptable it is the cheapest correct repetition. Because the domain map is
/// a (possibly reflected) translation, the analytic gradient passes through with
/// the mirrored axes negated.
#[derive(Debug, Clone, Copy)]
pub struct MirroredRepetition<V: SdfVector, S, const N: usize> {
    pub shape: S,
    pub spacing: Vector<V, N>,
}

#[inline(always)]
fn mirrored_fold<V: SdfVector, const N: usize>(
    p: Vector<V, N>,
    spacing: Vector<V, N>,
) -> (Vector<V, N>, Vector<V, N>) {
    // returns (folded point, per-axis sign +-1 for the gradient)
    let mut q = Vector::ZERO;
    let mut sign = Vector::ONE;
    for i in 0..N {
        let s = spacing[i];
        let id = (p[i] / s).round();
        let r = s.nmul_adde(id, p[i]); // p - s*id, in [-s/2, s/2]
        // odd tile <=> id - 2*trunc(id/2) != 0  (trunc avoids the .5 rounding trap)
        let parity = (id * V::HALF).trunc().nmul_adde(V::TWO, id); // id - 2*trunc(id/2)
        let odd = parity.cmp_ne(V::ZERO);
        sign[i] = odd.select(V::NEG_ONE, V::ONE);
        q[i] = odd.select(-r, r);
    }
    (q, sign)
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for MirroredRepetition<V, S, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (q, _) = mirrored_fold(p, self.spacing);
        self.shape.eval(q)
    }
}

impl<V: SdfVector, const N: usize, S: GradientSdf<V, N>> GradientSdf<V, N> for MirroredRepetition<V, S, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (q, sign) = mirrored_fold(p, self.spacing);
        let (d, mut g) = self.shape.eval_grad(q);
        for i in 0..N {
            g[i] = g[i] * sign[i]; // chain rule through the per-axis reflection
        }
        (d, g)
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

/// Shared finite-difference stencil used by [`FiniteDiff`] and
/// [`DistanceEstimate`]. Returns `(inner.eval(p), raw, magnitude)` where `raw`
/// points along `$\nabla f$` and `magnitude` is the true `$\lVert\nabla f\rVert$`.
///
/// The raw simplex sums are proportional to the gradient by a tap-dependent
/// constant (the tetrahedron and triangle taps integrate `$\sum u u^T$` to
/// `$4I$` / `$\tfrac32 I$` respectively, the central form to `$2I$`); the
/// per-branch `inv_scale` removes it so `magnitude` is in real units. The
/// direction `raw` is returned unscaled because [`FiniteDiff`] only needs it for
/// normalisation, where the constant cancels.
#[inline(always)]
fn finite_diff_gradient<V: SdfVector, const N: usize, S: SDF<V, N>>(
    shape: &S,
    p: Vector<V, N>,
    h: V,
) -> (V, Vector<V, N>, V) {
    let dist = shape.eval(p);
    let mut grad = Vector::ZERO;
    let inv_scale;

    if const { N == 3 } {
        // Tetrahedron technique (Falcao/Iquilez): 4 taps with central-difference
        // quality and no axis bias - cheaper than the 6-tap central form. The
        // four offsets are a regular tetrahedron inscribed in the cube:
        // (+++), (+--), (-+-), (--+). raw ~ 4 h^2 grad.
        let mut j = 0;
        while j < 4 {
            let mut e = Vector::ZERO;
            let mut k = 0;
            while k < N {
                let plus = j == 0 || k + 1 == j; // single '+' at axis j-1, else all '+'
                e[k] = if plus { h } else { -h };
                k += 1;
            }
            grad = e.mul_adde(shape.eval(p + e), grad);
            j += 1;
        }
        inv_scale = V::ONE / (cint::<V, 4>() * h * h);
    } else if const { N == 2 } {
        // 2D analogue: an equilateral triangle of directions summing to zero
        // (3 taps, unbiased). raw ~ (3/2) h^2 grad.
        let sx = h * V::SQRT_3 * V::HALF; // h*sqrt(3)/2
        let hy = h * V::HALF;
        let dirs = [[V::ZERO, h], [-sx, -hy], [sx, -hy]];
        let mut j = 0;
        while j < 3 {
            let mut e = Vector::ZERO;
            e[0] = dirs[j][0];
            e[1] = dirs[j][1];
            grad = e.mul_adde(shape.eval(p + e), grad);
            j += 1;
        }
        inv_scale = cint::<V, 2>() / (cint::<V, 3>() * h * h);
    } else {
        // General N: central differences per axis (2N taps). raw ~ 2 h grad.
        let mut i = 0;
        while i < N {
            let mut hp = p;
            let mut hm = p;
            hp[i] = p[i] + h;
            hm[i] = p[i] - h;
            grad[i] = shape.eval(hp) - shape.eval(hm);
            i += 1;
        }
        inv_scale = V::ONE / (V::TWO * h);
    }

    (dist, grad, grad.l2_norm() * inv_scale)
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> GradientSdf<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let (dist, grad, _) = finite_diff_gradient(&self.shape, p, self.eps);
        (dist, unit_or_zero(grad, grad.l2_norm()))
    }
}

impl<V: SdfVector, const N: usize, S: BoundedSdf<V, N>> BoundedSdf<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        self.shape.aabb()
    }
}

/// Adapts an arbitrary scalar field closure `Fn(p) -> value` into an [`SDF`].
///
/// Useful as the operand of [`DistanceEstimate`], whose whole purpose is to turn
/// a *non-metric* implicit field `$f$` (whose zero-set is the shape, but whose
/// gradient is not unit-length) into a usable approximate SDF.
#[derive(Debug, Clone, Copy)]
pub struct Field<F>(pub F);

impl<V: SdfVector, const N: usize, F: Fn(Vector<V, N>) -> V> SDF<V, N> for Field<F> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        (self.0)(p)
    }
}

/// Gradient-normalised distance estimate to the `$f = 0$` isosurface of an
/// arbitrary implicit field (`<https://iquilezles.org/articles/distance>`).
///
/// For a field `$f$` that is not a true distance function (e.g. a procedural
/// pattern, a fractal potential, or a `min`/`max` combination), the first-order
/// distance to its zero-set is
///
/// ```math
/// d(p) \approx \frac{f(p)}{\lVert \nabla f(p) \rVert}
/// ```
///
/// This rescales `$f$` so its gradient has length ~1 near the surface, which is
/// exactly what a raymarcher needs and what gives procedural outlines a constant
/// thickness instead of one that compresses where the field steepens. The
/// gradient is estimated with the same simplex stencil as [`FiniteDiff`] (4 taps
/// in 3D, 3 in 2D), so each `eval` costs one field sample plus the stencil. The
/// sign of `$f$` is preserved, so interior stays negative.
///
/// Unlike a real SDF this is only a *bound* (the Taylor estimate underestimates
/// where the field curves), so it is not marked [`BoundedSdf`]; wrap an
/// [`SDF`]-implementing field or a [`Field`] closure.
#[derive(Debug, Clone, Copy)]
pub struct DistanceEstimate<V: SdfVector, S> {
    pub shape: S,
    /// Finite-difference step for the gradient estimate (see [`FiniteDiff`]).
    pub eps: V,
}

impl<V: SdfVector, S> DistanceEstimate<V, S> {
    /// Wraps `shape` with the default step `$\text{eps} = 1/4096$`.
    #[inline(always)]
    pub fn new(shape: S) -> Self {
        Self {
            shape,
            eps: frac::<V, 1, 4096>(),
        }
    }

    /// Wraps `shape` with an explicit gradient step `eps`.
    #[inline(always)]
    pub fn with_eps(shape: S, eps: V) -> Self {
        Self { shape, eps }
    }
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> SDF<V, N> for DistanceEstimate<V, S> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (f, _, mag) = finite_diff_gradient(&self.shape, p, self.eps);
        // f / |grad f|, guarded so a vanishing gradient (an extremum) returns f
        // rather than +-inf/NaN.
        f / mag.cmp_gt(V::ZERO).select(mag, V::ONE)
    }
}

impl<V: SdfVector, const N: usize, S: SDF<V, N>> GradientSdf<V, N> for DistanceEstimate<V, S> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // The corrected field shares its surface normal with f to first order,
        // so reuse the raw stencil direction; the distance is the rescaled value.
        let (f, grad, mag) = finite_diff_gradient(&self.shape, p, self.eps);
        let d = f / mag.cmp_gt(V::ZERO).select(mag, V::ONE);
        (d, unit_or_zero(grad, grad.l2_norm()))
    }
}
