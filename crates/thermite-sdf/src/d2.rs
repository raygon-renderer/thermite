//! 2D signed-distance fields: distance, analytic gradient, and bounding box.
//!
//! Ported from <https://iquilezles.org/articles/distfunctions2d>,
//! <https://iquilezles.org/articles/distgradfunctions2d> and
//! <https://iquilezles.org/articles/bboxes2d>. Conditional branches in the
//! originals are lowered to branchless lane-wise `select`s.

use core::marker::PhantomData;

use thermite::math::policy::DefaultPolicy;
use thermite::math::{RealMathWithPolicy, TranscendentalMathWithPolicy};
use thermite::prelude::*;

use thermite_geometry::soa::algo::d2::point_on_ellipse;
use thermite_geometry::soa::prim::{Bounds, Point2, Vector, Vector2, vector::VectorOps as _};

use crate::consts::{cint, frac};
use crate::{BoundedSdf, GradientSdf, SDF, SdfConsts, SdfVector, unit_or_zero};

#[inline(always)]
fn closest_on_segment<V: SdfVector>(p: Vector2<V>, a: Vector2<V>, b: Vector2<V>) -> (V, Vector2<V>, V) {
    let e = b - a;
    let w = p - a;
    let h = (w.dot(&e) / e.dot(&e)).clamp(V::ZERO, V::ONE);
    let q = e.nmul_adde(h, w); // w - e*h
    (q.dot(&q), q, w.cross(e))
}

/// Circle of radius `radius` centered at the origin (the 2D [`NSphere`](crate::dn::NSphere)).
pub type Circle2D<V> = crate::dn::NSphere<V>;

/// Pie slice (circular sector) of radius `radius`, pointing up the +y axis.
///
/// `sc` is the `(sin, cos)` of the half-aperture angle, precomputed by the
/// caller (matching the reference `sdgPie`).
#[derive(Debug, Clone, Copy)]
pub struct Pie2D<V: SdfVector> {
    pub radius: V,
    pub sc: Vector2<V>,
}

impl<V: SdfVector> Pie2D<V> {
    /// Construct from a half-aperture angle (radians), computing `(sin, cos)`
    /// with the given precision policy `P`.
    #[inline(always)]
    pub fn from_aperture<P: Policy>(radius: V, aperture: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (sin, cos) = aperture.sin_cos_p::<P>();
        Self {
            radius,
            sc: Vector2::new([sin, cos]),
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for Pie2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        // distance only: the two gradient normalizes (p/l, q/m) and the sign
        // restoration are not needed, and the distance is simply max(n, m).
        let sc = self.sc;
        let (sx, cy) = (sc[0], sc[1]);
        let p = Vector2::new([p[0].abs(), p[1]]);

        let n = p.l2_norm() - self.radius;
        let q = sc.nmul_adde(p.dot(&sc).clamp(V::ZERO, self.radius), p);
        let m = q.l2_norm().mul_sign(cy.mul_sube(p[0], sx * p[1]));

        n.max(m)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Pie2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let sc = self.sc;
        let (sx, cy) = (sc[0], sc[1]);

        let s = p[0].signum();
        let p = Vector2::new([p[0].abs(), p[1]]);

        let l = p.l2_norm();
        let n = l - self.radius;

        let q = sc.nmul_adde(p.dot(&sc).clamp(V::ZERO, self.radius), p);
        let m = q.l2_norm().mul_sign(cy.mul_sube(p[0], sx * p[1]));

        let pick = n.cmp_gt(m);
        let dist = pick.select(n, m);
        let grad = pick.select(unit_or_zero(p, l), unit_or_zero(q, m));

        (dist, Vector2::new([s * grad[0], grad[1]]))
    }
}

/// Arc of radius `ra` and thickness `rb`, spanning a half-aperture whose
/// `(sin, cos)` is given by `sc` (precomputed by the caller).
#[derive(Debug, Clone, Copy)]
pub struct Arc2D<V: SdfVector> {
    pub sc: Vector2<V>,
    pub ra: V,
    pub rb: V,
}

impl<V: SdfVector> Arc2D<V> {
    /// Construct from a half-aperture angle (radians), computing `(sin, cos)`
    /// with the given precision policy `P`.
    #[inline(always)]
    pub fn from_aperture<P: Policy>(aperture: V, ra: V, rb: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (sin, cos) = aperture.sin_cos_p::<P>();
        Self {
            sc: Vector2::new([sin, cos]),
            ra,
            rb,
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for Arc2D<V> {
    #[inline(always)]
    fn eval(&self, q0: Vector2<V>) -> V {
        let sc = self.sc;
        let (sx, cy) = (sc[0], sc[1]);
        let p = Vector2::new([q0[0].abs(), q0[1]]);

        // both candidate distances without forming their gradients
        let dist_a = sc.nmul_adde(self.ra, p).l2_norm() - self.rb;
        let dist_b = (q0.l2_norm() - self.ra).abs() - self.rb;

        // cy*p.x > sx*p.y  <=>  cy*p.x - sx*p.y > 0
        cy.mul_sube(p[0], sx * p[1]).cmp_gt(V::ZERO).select(dist_a, dist_b)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Arc2D<V> {
    #[inline(always)]
    fn eval_grad(&self, q0: Vector2<V>) -> (V, Vector2<V>) {
        let sc = self.sc;
        let (sx, cy) = (sc[0], sc[1]);

        let s = q0[0].signum();
        let p = Vector2::new([q0[0].abs(), q0[1]]);

        // branch A: nearest to the arc's circular band endpoint
        let wa = sc.nmul_adde(self.ra, p); // p - sc*ra
        let da = wa.l2_norm();
        let dist_a = da - self.rb;
        let grad_a = unit_or_zero(Vector2::new([s * wa[0], wa[1]]), da);

        // branch B: nearest to the full ring
        let l = q0.l2_norm();
        let wb = l - self.ra;
        let dist_b = wb.abs() - self.rb;
        let grad_b = unit_or_zero(q0 * wb.signum(), l);

        let pick = cy.mul_sube(p[0], sx * p[1]).cmp_gt(V::ZERO);
        (pick.select(dist_a, dist_b), pick.select(grad_a, grad_b))
    }
}

/// Capsule / thick line segment from `a` to `b` with radius `r` (the 2D
/// [`NCapsule`](crate::dn::NCapsule)).
pub type Segment2D<V> = crate::dn::NCapsule<V, 2>;

/// Vesica (lens) formed by intersecting two circles of radius `r` whose centers
/// are offset by `d` either side of the origin.
///
/// Stored in the cheapest-to-evaluate form: `r`, `d`, and the precomputed
/// half-height `$b = \sqrt{r^2 - d^2}$` (so evaluation needs no square root).
/// Construct with [`from_circle`](Self::from_circle) or
/// [`from_size`](Self::from_size).
#[derive(Debug, Clone, Copy)]
pub struct Vesica2D<V: SdfVector> {
    r: V,
    d: V,
    b: V,
}

impl<V: SdfVector> Vesica2D<V> {
    /// From the intersecting-circles parameterization: circle radius `r` and
    /// center offset `d` (with `d < r`).
    #[inline(always)]
    pub fn from_circle(r: V, d: V) -> Self {
        Self {
            r,
            d,
            b: r.mul_sube(r, d * d).sqrt(),
        }
    }

    /// From the lens size: `half_width` (`w`, tip-to-center) and `half_height` (`h`,
    /// widest chord half-length). The two-circle parameterization follows from:
    ///
    /// ```math
    /// w = r - d, \quad h = \sqrt{r^2 - d^2}
    /// \implies d = \frac{h^2 - w^2}{2w}, \quad r = w + d
    /// ```
    #[inline(always)]
    pub fn from_size(half_width: V, half_height: V) -> Self {
        // half_width = r - d,  half_height = sqrt(r^2 - d^2)
        //   => d = (h^2 - w^2) / (2w),  r = w + d
        let d = half_height.mul_sube(half_height, half_width * half_width) / (half_width * V::TWO);
        Self {
            r: half_width + d,
            d,
            b: half_height,
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for Vesica2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let p = p.abs();
        let b = self.b;

        let la = Vector2::new([p[0], p[1] - b]).l2_norm().mul_sign(self.d);
        let lb = Vector2::new([p[0] + self.d, p[1]]).l2_norm();

        // (p.y-b)*d > p.x*b  <=>  (p.y-b)*d - p.x*b > 0
        (p[1] - b)
            .mul_sube(self.d, p[0] * b)
            .cmp_gt(V::ZERO)
            .select(la, lb - self.r)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Vesica2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let s = p.signum();
        let p = p.abs();
        let b = self.b;

        // branch A: along the chord
        let qa = Vector2::new([p[0], p[1] - b]);
        let la = qa.l2_norm().mul_sign(self.d);
        let grad_a = s * qa / la;

        // branch B: along the circular caps
        let qb = Vector2::new([p[0] + self.d, p[1]]);
        let lb = qb.l2_norm();
        let grad_b = s * qb / lb;

        let pick = (p[1] - b).mul_sube(self.d, p[0] * b).cmp_gt(V::ZERO);
        (pick.select(la, lb - self.r), pick.select(grad_a, grad_b))
    }
}

/// Axis-aligned box with half-extents `b` (the 2D [`NBox`](crate::dn::NBox)).
pub type Box2D<V> = crate::dn::NBox<V, 2>;

/// Plus/cross shape whose arms have half-extents `b = (length, width)`
/// (with `b.x >= b.y`).
#[derive(Debug, Clone, Copy)]
pub struct Cross2D<V: SdfVector> {
    pub b: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for Cross2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let b = self.b;
        let p = p.abs();

        let swap = p[1].cmp_gt(p[0]);
        let q = Vector2::new([swap.select(p[1], p[0]) - b[0], swap.select(p[0], p[1]) - b[1]]);

        let h = q[0].max(q[1]);
        let inner = h.cmp_lt(V::ZERO);
        let off = inner.select(Vector2::new([(b[1] - b[0]) - q[0], -q[1]]), q);
        let l = Vector2::new([off[0].max(V::ZERO), off[1].max(V::ZERO)]).l2_norm();

        // distance magnitude is -q.x at the concave corner, else the arm length;
        // the gradient direction (og, fold undo) is not needed here.
        let rx = (inner & (-q[0]).cmp_lt(l)).select(-q[0], l);
        rx.mul_sign(h)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Cross2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let b = self.b;
        let s = p.signum();
        let p = p.abs();

        // fold the larger component onto x: q = (p.y>p.x ? p.yx : p.xy) - b
        let swap = p[1].cmp_gt(p[0]);
        let q = Vector2::new([swap.select(p[1], p[0]) - b[0], swap.select(p[0], p[1]) - b[1]]);

        let h = q[0].max(q[1]);

        // inside the inner notch the reference offset is (b.y-b.x, 0) - q
        let inner = h.cmp_lt(V::ZERO);
        let off = inner.select(Vector2::new([(b[1] - b[0]) - q[0], -q[1]]), q);
        let o = Vector2::new([off[0].max(V::ZERO), off[1].max(V::ZERO)]);
        let l = o.l2_norm();

        // nearest feature is the concave corner along -x, or the rounded arm
        let corner = inner & (-q[0]).cmp_lt(l);
        let rx = corner.select(-q[0], l);
        let og = unit_or_zero(o, l);
        let ry = corner.select(V::ONE, og[0]);
        let rz = corner.select(V::ZERO, og[1]);

        let dist = rx.mul_sign(h);
        // undo the x-fold (and the original sign) on the gradient
        let g = Vector2::new([swap.select(rz, ry), swap.select(ry, rz)]);
        (dist, s * g)
    }
}

/// Regular hexagon with apothem `r` (flat-to-flat half-distance, also called the inradius).
///
/// The circumradius (center-to-vertex) is `$r \cdot \frac{2}{\sqrt{3}}$`.
#[derive(Debug, Clone, Copy)]
pub struct Hexagon2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for Hexagon2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let kx = -(V::SQRT_3 * V::HALF);
        let ky = V::HALF;
        let kz = V::FRAC_1_SQRT_3;

        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let m = kx.mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(m, px); // px - m*kx
        py = ky.nmul_adde(m, py); // py - m*ky

        px -= px.clamp(-(kz * self.r), kz * self.r);
        py -= self.r;

        // distance needs no gradient fold/normalize
        px.mul_adde(px, py * py).sqrt().mul_sign(py)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Hexagon2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        // k = (-sqrt(3)/2, 1/2, 1/sqrt(3))
        let kx = -(V::SQRT_3 * V::HALF);
        let ky = V::HALF;
        let kz = V::FRAC_1_SQRT_3;

        let s = p.signum();
        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let w = kx.mul_adde(px, ky * py);
        let m = w.min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(m, px); // px - m*kx
        py = ky.nmul_adde(m, py); // py - m*ky

        px -= px.clamp(-(kz * self.r), kz * self.r);
        py -= self.r;

        let d = px.mul_adde(px, py * py).sqrt().mul_sign(py);

        // gradient is reflected back through the folding plane when w < 0
        let folded = Vector2::new([ky.nmul_sube(px, kx * py), kx.nmul_adde(px, ky * py)]);
        let g = w.cmp_lt(V::ZERO).select(folded, Vector2::new([px, py]));

        (d, unit_or_zero(s * g, d))
    }
}

/// Isosceles triangle whose apex is at the origin and whose half-base/height is
/// given by `q = (half_width, height)`.
#[derive(Debug, Clone, Copy)]
pub struct IsoscelesTriangle2D<V: SdfVector> {
    pub q: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for IsoscelesTriangle2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let q = self.q;
        let p = Vector2::new([p[0].abs(), p[1]]);

        let a = q.nmul_adde((p.dot(&q) / q.dot(&q)).clamp(V::ZERO, V::ONE), p);
        let b = Vector2::new([q[0].nmul_adde((p[0] / q[0]).clamp(V::ZERO, V::ONE), p[0]), p[1] - q[1]]);

        let k = q[1].signum();
        let d = a.dot(&a).min(b.dot(&b)).sqrt(); // skips picking g and the normalize
        let s = (k * p[0].mul_sube(q[1], p[1] * q[0])).max(k * (p[1] - q[1]));
        d.mul_sign(s)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for IsoscelesTriangle2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let q = self.q;
        let w = p[0].signum();
        let p = Vector2::new([p[0].abs(), p[1]]);

        let a = q.nmul_adde((p.dot(&q) / q.dot(&q)).clamp(V::ZERO, V::ONE), p);
        let b = Vector2::new([q[0].nmul_adde((p[0] / q[0]).clamp(V::ZERO, V::ONE), p[0]), p[1] - q[1]]);

        let k = q[1].signum();
        let l1 = a.dot(&a);
        let l2 = b.dot(&b);

        let pick = l1.cmp_lt(l2);
        let d = pick.select(l1, l2).sqrt();
        let g = pick.select(a, b);

        let s = (k * p[0].mul_sube(q[1], p[1] * q[0])).max(k * (p[1] - q[1]));
        let sgn = s.signum();

        (d * sgn, unit_or_zero(Vector2::new([w * g[0], g[1]]), d) * sgn)
    }
}

/// Convex polygon distance shared by [`Triangle2D`] and [`Quad2D`].
#[inline(always)]
fn convex_polygon<V: SdfVector, const N: usize>(p: Vector2<V>, v: &[Vector2<V>; N]) -> (V, Vector2<V>) {
    // signed area orientation reference
    let gs = (v[0] - v[N - 1]).cross(v[1] - v[0]);

    let (mut dmin, mut qmin, c0) = closest_on_segment(p, v[0], v[1]);
    let mut smax = gs * c0;

    let mut i = 1;
    while i < N {
        let (d, q, c) = closest_on_segment(p, v[i], v[(i + 1) % N]);

        let closer = d.cmp_lt(dmin);
        qmin = closer.select(q, qmin);
        dmin = closer.select(d, dmin);

        let s = gs * c;
        smax = s.cmp_gt(smax).select(s, smax);

        i += 1;
    }

    let d = dmin.sqrt().mul_sign(smax);
    (d, unit_or_zero(qmin, d))
}

/// Distance-only variant of [`convex_polygon`] that skips the per-edge `qmin`
/// vector tracking and the final normalize.
#[inline(always)]
fn convex_polygon_dist<V: SdfVector, const N: usize>(p: Vector2<V>, v: &[Vector2<V>; N]) -> V {
    let gs = (v[0] - v[N - 1]).cross(v[1] - v[0]);

    let (mut dmin, _, c0) = closest_on_segment(p, v[0], v[1]);
    let mut smax = gs * c0;

    let mut i = 1;
    while i < N {
        let (d, _, c) = closest_on_segment(p, v[i], v[(i + 1) % N]);
        dmin = d.cmp_lt(dmin).select(d, dmin);
        let s = gs * c;
        smax = s.cmp_gt(smax).select(s, smax);
        i += 1;
    }

    dmin.sqrt().mul_sign(smax)
}

/// Arbitrary triangle given by its three vertices.
#[derive(Debug, Clone, Copy)]
pub struct Triangle2D<V: SdfVector> {
    pub v: [Vector2<V>; 3],
}

impl<V: SdfVector> SDF<V, 2> for Triangle2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        convex_polygon_dist(p, &self.v)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Triangle2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        convex_polygon(p, &self.v)
    }
}

/// Arbitrary convex quad given by its four vertices (in order).
#[derive(Debug, Clone, Copy)]
pub struct Quad2D<V: SdfVector> {
    pub v: [Vector2<V>; 4],
}

impl<V: SdfVector> SDF<V, 2> for Quad2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        convex_polygon_dist(p, &self.v)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Quad2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        convex_polygon(p, &self.v)
    }
}

/// Crescent moon: the difference of two circles of radius `ra` and `rb` whose
/// centers are `d` apart.
///
/// Stores the precomputed intersection point `(a, b)`. Build with
/// [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct Moon2D<V: SdfVector> {
    d: V,
    ra: V,
    rb: V,
    a: V,
    b: V,
}

impl<V: SdfVector> Moon2D<V> {
    /// Outer circle of radius `ra` at the origin; inner (cutting) circle of radius `rb`
    /// centered at `(d, 0)`. The crescent retains the region inside `ra` but outside `rb`.
    ///
    /// Precomputes the radical-axis coordinates:
    ///
    /// ```math
    /// a = \frac{r_a^2 - r_b^2 + d^2}{2d}, \quad b = \sqrt{\max(r_a^2 - a^2,\; 0)}
    /// ```
    #[inline(always)]
    pub fn new(d: V, ra: V, rb: V) -> Self {
        let a = d.mul_adde(d, ra.mul_sube(ra, rb * rb)) / (d * V::TWO); // (ra^2 - rb^2 + d^2)/(2d)
        let b = ra.mul_sube(ra, a * a).max(V::ZERO).sqrt(); // sqrt(max(ra^2 - a^2, 0))
        Self { d, ra, rb, a, b }
    }
}

impl<V: SdfVector> SDF<V, 2> for Moon2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (d, ra, rb, a, b) = (self.d, self.ra, self.rb, self.a, self.b);
        let p = Vector2::new([p[0], p[1].abs()]);

        let dist_a = (p - Vector2::new([a, b])).l2_norm();
        let d1 = p.l2_norm() - ra;
        let d2 = rb - (p - Vector2::new([d, V::ZERO])).l2_norm();
        let dist_b = d1.max(d2);

        let pick = (d * p[0].mul_sube(b, p[1] * a)).cmp_gt(d * d * (b - p[1]).max(V::ZERO));
        pick.select(dist_a, dist_b)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Moon2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let (d, ra, rb, a, b) = (self.d, self.ra, self.rb, self.a, self.b);

        let s = p[1].signum();
        let p = Vector2::new([p[0], p[1].abs()]);

        // branch A: distance to the intersection point (a, b)
        let wa = p - Vector2::new([a, b]);
        let dist_a = wa.l2_norm();
        let grad_a = unit_or_zero(Vector2::new([wa[0], wa[1] * s]), dist_a);

        // branch B: outer circle (ra) vs inner cut circle (rb)
        let w1 = p;
        let l1 = w1.l2_norm();
        let d1 = l1 - ra;
        let g1 = unit_or_zero(Vector2::new([w1[0], w1[1] * s]), l1);

        let w2 = p - Vector2::new([d, V::ZERO]);
        let l2 = w2.l2_norm();
        let d2 = rb - l2;
        let g2 = unit_or_zero(Vector2::new([w2[0], w2[1] * s]), l2) * V::NEG_ONE;

        let outer = d1.cmp_gt(d2);
        let dist_b = outer.select(d1, d2);
        let grad_b = outer.select(g1, g2);

        let pick = (d * p[0].mul_sube(b, p[1] * a)).cmp_gt(d * d * (b - p[1]).max(V::ZERO));
        (pick.select(dist_a, dist_b), pick.select(grad_a, grad_b))
    }
}

/// Symmetric trapezoid with bottom radius `ra`, top radius `rb` and half-height
/// `he`.
#[derive(Debug, Clone, Copy)]
pub struct Trapezoid2D<V: SdfVector> {
    pub ra: V,
    pub rb: V,
    pub he: V,
}

impl<V: SdfVector> SDF<V, 2> for Trapezoid2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (ra, rb, he) = (self.ra, self.rb, self.he);
        let sy = p[1].signum();
        let p = Vector2::new([p[0].abs(), p[1]]);

        let h1 = p[0].min(p[1].select_negative(ra, rb));
        let q1 = p - Vector2::new([h1, sy * he]);
        let d1 = q1.dot(&q1);
        let s1 = p[1].abs() - he;

        let k = Vector2::new([rb - ra, he * V::TWO]);
        let w = p - Vector2::new([ra, -he]);
        let h2 = (w.dot(&k) / k.dot(&k)).clamp(V::ZERO, V::ONE);
        let q2 = p - k.mul_adde(h2, Vector2::new([ra, -he])); // p - ((ra,-he) + k*h2)
        let d2 = q2.dot(&q2);
        let s2 = w[0].mul_sube(k[1], w[1] * k[0]); // cross(w, k)

        // skips picking qmin and the sx/normalize for the gradient
        d1.min(d2).sqrt().mul_sign(s1.max(s2))
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Trapezoid2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let (ra, rb, he) = (self.ra, self.rb, self.he);

        let sx = p[0].signum();
        let sy = p[1].signum();
        let p = Vector2::new([p[0].abs(), p[1]]);

        // candidate 1: nearest horizontal cap
        let h1 = p[0].min(p[1].select_negative(ra, rb));
        let c1 = Vector2::new([h1, sy * he]);
        let q1 = p - c1;
        let d1 = q1.dot(&q1);
        let s1 = p[1].abs() - he;

        // candidate 2: nearest slanted edge
        let k = Vector2::new([rb - ra, he * V::TWO]);
        let w = p - Vector2::new([ra, -he]);
        let h2 = (w.dot(&k) / k.dot(&k)).clamp(V::ZERO, V::ONE);
        let c2 = k.mul_adde(h2, Vector2::new([ra, -he]));
        let q2 = p - c2;
        let d2 = q2.dot(&q2);
        let s2 = w[0].mul_sube(k[1], w[1] * k[0]); // cross(w, k)

        let closer = d2.cmp_lt(d1);
        let dmin = closer.select(d2, d1);
        let qmin = closer.select(q2, q1);
        let smax = s2.cmp_gt(s1).select(s2, s1);

        let d = dmin.sqrt().mul_sign(smax);
        (d, Vector2::new([qmin[0] * sx, qmin[1]]) / d)
    }
}

/// Heart shape inscribed roughly in the unit square.
#[derive(Debug, Clone, Copy)]
pub struct Heart2D;

impl<V: SdfVector> SDF<V, 2> for Heart2D {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let p = Vector2::new([p[0].abs(), p[1]]);
        let quarter = frac::<V, 1, 4>();

        let r = V::SQRT_2 * quarter;
        let dist_a = (p - Vector2::new([quarter, V::ONE - quarter])).l2_norm() - r;

        let q1 = p - Vector2::new([V::ZERO, V::ONE]);
        let t = (p[0] + p[1]).max(V::ZERO) * V::HALF;
        let q2 = p - Vector2::new([t, t]);
        let db = q1.dot(&q1).min(q2.dot(&q2));
        let sgn = p[0].cmp_gt(p[1]).select(V::ONE, V::NEG_ONE);
        let dist_b = db.sqrt() * sgn;

        (p[1] + p[0]).cmp_gt(V::ONE).select(dist_a, dist_b)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Heart2D {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let sx = p[0].signum();
        let p = Vector2::new([p[0].abs(), p[1]]);

        let quarter = frac::<V, 1, 4>(); // 0.25

        // upper-lobe branch (p.x + p.y > 1)
        let r = V::SQRT_2 * quarter; // sqrt(2)/4
        let q0 = p - Vector2::new([quarter, V::ONE - quarter]); // (0.25, 0.75)
        let la = q0.l2_norm();
        let dist_a = la - r;
        let grad_a0 = q0 / la;
        let grad_a = Vector2::new([grad_a0[0] * sx, grad_a0[1]]);

        // lower-body branch
        let q1 = p - Vector2::new([V::ZERO, V::ONE]);
        let t = (p[0] + p[1]).max(V::ZERO) * V::HALF;
        let q2 = p - Vector2::new([t, t]);
        let dd1 = q1.dot(&q1);
        let dd2 = q2.dot(&q2);

        let near1 = dd1.cmp_lt(dd2);
        let db = near1.select(dd1, dd2);
        let qb = near1.select(q1, q2);
        let lb = db.sqrt();
        let sgn = p[0].cmp_gt(p[1]).select(V::ONE, V::NEG_ONE);
        let dist_b = lb * sgn;
        let grad_b0 = qb / lb * sgn;
        let grad_b = Vector2::new([grad_b0[0] * sx, grad_b0[1]]);

        let upper = (p[1] + p[0]).cmp_gt(V::ONE);
        (upper.select(dist_a, dist_b), upper.select(grad_a, grad_b))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Heart2D {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // Unit heart. The upper lobes are circles of radius sqrt(2)/4 centred at
        // (+/-1/4, 3/4): their rightmost/topmost points give x = +/-(1+sqrt2)/4 and
        // y_max = (3+sqrt2)/4. The bottom cusp sits at y = -sqrt(2)/2.
        let q = frac::<V, 1, 4>();
        let hx = (V::ONE + V::SQRT_2) * q; // (1+sqrt2)/4
        let y_max = (cint::<V, 3>() + V::SQRT_2) * q; // (3+sqrt2)/4
        let y_min = -(V::SQRT_2 * V::HALF); // -sqrt(2)/2
        Bounds::from_corners(Vector2::new([-hx, y_min]), Vector2::new([hx, y_max]))
    }
}

/// Ellipse with semi-axes `ab = (a, b)`.
///
/// Uses the iterative closest-point-on-ellipse solver from
/// [`thermite_geometry::soa::algo::d2`].
#[derive(Debug, Clone, Copy)]
pub struct Ellipse2D<V: SdfVector> {
    pub ab: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for Ellipse2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        // the iterative solver dominates and is unavoidable, but we still skip
        // the sign-restore and the final gradient normalize
        let pa = p.abs();
        let n = pa / self.ab;
        let outside = n.dot(&n).cmp_gt(V::ONE);

        let q = point_on_ellipse(Point2::new([pa[0], pa[1]]), self.ab);
        let d = (pa - Vector2::new([q[0], q[1]])).l2_norm();
        d.neg_c(!outside) // outside ? d : -d
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Ellipse2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let sp = p.signum();
        let pa = p.abs();

        // outside when (p / ab)^2 sums to > 1
        let n = pa / self.ab;
        let outside = n.dot(&n).cmp_gt(V::ONE);

        let q = point_on_ellipse(Point2::new([pa[0], pa[1]]), self.ab);
        let diff = pa - Vector2::new([q[0], q[1]]);
        let d = diff.l2_norm();

        let sgn = outside.select(V::ONE, V::NEG_ONE);
        (d * sgn, unit_or_zero(sp * diff, d) * sgn)
    }
}

/// Parabola `$y = k x^2$`, opening upward.
///
/// Finding the nearest point requires solving a depressed cubic, whose roots
/// need either a cube root (one-real-root case) or trigonometry (three-real-root
/// case), hence the [`RealMathWithPolicy`] bound and precision policy `P`.
#[derive(Debug, Clone, Copy)]
pub struct Parabola2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub k: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> Parabola2D<V, P> {
    #[inline(always)]
    pub const fn new(k: V) -> Self {
        Self {
            k,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> Parabola2D<V, P> {
    /// Solve for the nearest parameter `x` on the parabola. Returns the folded
    /// `|pos.x|` and the sign of the original `pos.x` so callers can build the
    /// offset vector and restore the gradient. This is the expensive part
    /// (cube root / trigonometry) shared by `eval` and `eval_grad`.
    #[inline(always)]
    fn solve(&self, pos: Vector2<V>) -> (V, V, V) {
        let third = V::ONE + V::TWO; // 3

        let s = pos[0].signum();
        let px = pos[0].abs();

        let ik = V::ONE / self.k;
        let p = ik * ik.nmul_adde(V::HALF, pos[1]) / third; // ik*(pos.y - 0.5*ik)/3
        let q = frac::<V, 1, 4>() * ik * ik * px; // 0.25 * ik^2 * px
        let h = q.mul_sube(q, p * p * p); // q^2 - p^3
        let r = h.abs().sqrt();

        // One-real-root (h > 0) Cardano form. Folding the original
        // `pow(|q-r|,1/3) * sign(r-q)` term into the sign-aware `cbrt` collapses
        // the whole thing to a single sum with no post-hoc branch:
        //     x = cbrt(q + r) + cbrt(q - r)
        let mut x = (q + r).cbrt_p::<P>() + (q - r).cbrt_p::<P>();

        // Three-real-root (h <= 0) case needs atan2/cos. These are by far the
        // most expensive ops here, so only evaluate them when a lane asks for it.
        let needs_trig = h.cmp_le(V::ZERO);
        if needs_trig.any() {
            let x_trig = (r.atan2_p::<P>(q) / third).cos_p::<P>() * V::TWO * p.sqrt();
            x = needs_trig.select(x_trig, x);
        }

        (px, x, s)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for Parabola2D<V, P> {
    #[inline(always)]
    fn eval(&self, pos: Vector2<V>) -> V {
        let (px, x, _) = self.solve(pos);
        let z = (px - x).signum();
        let w = Vector2::new([px - x, self.k.nmul_adde(x * x, pos[1])]); // (px-x, pos.y - k*x^2)
        w.l2_norm() * z // skips the gradient normalize
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> GradientSdf<V, 2> for Parabola2D<V, P> {
    #[inline(always)]
    fn eval_grad(&self, pos: Vector2<V>) -> (V, Vector2<V>) {
        let (px, x, s) = self.solve(pos);
        let z = (px - x).signum();
        let w = Vector2::new([px - x, self.k.nmul_adde(x * x, pos[1])]); // (px-x, pos.y - k*x^2)
        let l = w.l2_norm();

        // z * vec3(l, vec2(s*w.x, w.y)/l)
        (l * z, unit_or_zero(Vector2::new([s * w[0], w[1]]), l) * z)
    }
}

// --- bounding boxes ---

impl<V: SdfVector> BoundedSdf<V, 2> for Ellipse2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(self.ab)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Cross2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // arms reach +/- b.x along both axes
        Bounds::symmetric(Vector2::splat(self.b[0]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Vesica2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // half-width r-d, half-height b = sqrt(r^2-d^2) (precomputed)
        Bounds::symmetric(Vector2::new([self.r - self.d, self.b]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Trapezoid2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::new([self.ra.max(self.rb), self.he]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for IsoscelesTriangle2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // apex at origin, base corners at (+/- q.x, q.y)
        let qx = self.q[0].abs();
        let qy = self.q[1];
        Bounds::from_corners(
            Vector2::new([-qx, qy.min(V::ZERO)]),
            Vector2::new([qx, qy.max(V::ZERO)]),
        )
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Triangle2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let v = &self.v;
        Bounds::from_corners(v[0].min(v[1]).min(v[2]), v[0].max(v[1]).max(v[2]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Quad2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let v = &self.v;
        Bounds::from_corners(v[0].min(v[1]).min(v[2]).min(v[3]), v[0].max(v[1]).max(v[2]).max(v[3]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Pie2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // Tight box of the sector (opens along +y, half-aperture `a`, sc=(sin,cos)).
        // Matches IQ's boxPie with direction d=(0,1): x is symmetric with half-extent
        // r*sin a while the wedge stays in the upper half (cos a > 0), else the full r;
        // y spans [min(r*cos a, 0), r].
        let r = self.radius;
        let (s, c) = (self.sc[0], self.sc[1]);
        let hx = c.cmp_gt(V::ZERO).select(r * s, r);
        let y_min = (r * c).min(V::ZERO);
        Bounds::from_corners(Vector2::new([-hx, y_min]), Vector2::new([hx, r]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Arc2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // Centerline arc of radius ra spans angles about +y (half-aperture a),
        // thickened by rb in every direction. The rounded endpoint caps set the
        // extremes: x half-extent = ra*sin a + rb (or ra + rb past 90 deg), and
        // y spans [ra*cos a - rb, ra + rb].
        let (s, c) = (self.sc[0], self.sc[1]);
        let hx = c.cmp_gt(V::ZERO).select(self.ra * s, self.ra) + self.rb;
        let y_min = self.ra.mul_sube(c, self.rb); // ra*cos a - rb
        let y_max = self.ra + self.rb;
        Bounds::from_corners(Vector2::new([-hx, y_min]), Vector2::new([hx, y_max]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Hexagon2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // Flats top/bottom at y = +/-r (apothem); pointy vertices left/right on the
        // x-axis at x = +/-2r/sqrt(3) (circumradius). Tight, not the loose square.
        let hx = self.r * V::TWO * V::FRAC_1_SQRT_3;
        Bounds::symmetric(Vector2::new([hx, self.r]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Moon2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // conservative: the crescent is contained in the radius-ra disk
        Bounds::symmetric(Vector2::splat(self.ra))
    }
}

// ===========================================================================
// Additional exact primitives (distance-only) from distfunctions2d
// ===========================================================================

/// Box with per-corner rounding radii `r = [top-right, bottom-right, top-left,
/// bottom-left]` and half-extents `b`.
#[derive(Debug, Clone, Copy)]
pub struct RoundedBox2D<V: SdfVector> {
    pub b: Vector2<V>,
    pub r: [V; 4],
}

impl<V: SdfVector> SDF<V, 2> for RoundedBox2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let right = p[0].cmp_gt(V::ZERO);
        let rx = right.select(self.r[0], self.r[2]);
        let ry = right.select(self.r[1], self.r[3]);
        let rad = p[1].cmp_gt(V::ZERO).select(rx, ry);

        let qx = p[0].abs() - self.b[0] + rad;
        let qy = p[1].abs() - self.b[1] + rad;
        let outside = Vector2::new([qx.max(V::ZERO), qy.max(V::ZERO)]).l2_norm();
        qx.max(qy).min(V::ZERO) + outside - rad
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for RoundedBox2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        // `rad` is constant within each quadrant, so the gradient is the plain
        // box normal (Box2D): outside -> normalize(max(q,0)); inside -> dominant face.
        let right = p[0].cmp_gt(V::ZERO);
        let rx = right.select(self.r[0], self.r[2]);
        let ry = right.select(self.r[1], self.r[3]);
        let rad = p[1].cmp_gt(V::ZERO).select(rx, ry);

        let s = p.signum();
        let qx = p[0].abs() - self.b[0] + rad;
        let qy = p[1].abs() - self.b[1] + rad;
        let g = qx.max(qy);
        let mq = Vector2::new([qx.max(V::ZERO), qy.max(V::ZERO)]);
        let l = mq.l2_norm();
        let outside = g.cmp_gt(V::ZERO);
        let dist = g.min(V::ZERO) + l - rad;
        let face = qx
            .cmp_gt(qy)
            .select(Vector2::new([V::ONE, V::ZERO]), Vector2::new([V::ZERO, V::ONE]));
        (dist, s * outside.select(unit_or_zero(mq, l), face))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for RoundedBox2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(self.b)
    }
}

/// Box with 45-degree chamfered corners of size `chamfer` and half-extents `b`.
#[derive(Debug, Clone, Copy)]
pub struct ChamferBox2D<V: SdfVector> {
    pub b: Vector2<V>,
    pub chamfer: V,
}

impl<V: SdfVector> SDF<V, 2> for ChamferBox2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let ax = p[0].abs() - self.b[0];
        let ay = p[1].abs() - self.b[1];
        let px = ax.max(ay); // (p.y>p.x)? p.yx : p.xy  ->  x = max
        let py = ax.min(ay) + self.chamfer;

        let k = V::ONE - V::SQRT_2;
        let corner = py.cmp_lt(V::ZERO) & px.mul_adde(k, py).cmp_lt(V::ZERO);
        let diag = (px + py) * V::FRAC_1_SQRT_2; // sqrt(0.5)
        let len = px.mul_adde(px, py * py).sqrt();

        corner.select(px, px.cmp_lt(py).select(diag, len))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for ChamferBox2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(self.b)
    }
}

/// Box defined by the segment `a -> b` (its central axis) with thickness `th`.
#[derive(Debug, Clone, Copy)]
pub struct OrientedBox2D<V: SdfVector> {
    pub a: Vector2<V>,
    pub b: Vector2<V>,
    pub th: V,
}

impl<V: SdfVector> SDF<V, 2> for OrientedBox2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let ba = self.b - self.a;
        let l = ba.l2_norm();
        let d = ba / l;
        let qv = p - (self.a + self.b) * V::HALF;
        let qx = d[0].mul_adde(qv[0], d[1] * qv[1]); // d.x*qx + d.y*qy
        let qy = d[1].nmul_adde(qv[0], d[0] * qv[1]); // -d.y*qx + d.x*qy
        let q = Vector2::new([qx.abs() - l * V::HALF, qy.abs() - self.th * V::HALF]);
        Vector2::new([q[0].max(V::ZERO), q[1].max(V::ZERO)]).l2_norm() + q[0].max(q[1]).min(V::ZERO)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for OrientedBox2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let ba = self.b - self.a;
        let l = ba.l2_norm();
        let d = ba / l;
        let qv = p - (self.a + self.b) * V::HALF;
        let lx = d[0].mul_adde(qv[0], d[1] * qv[1]); // local x = R*qv
        let ly = d[1].nmul_adde(qv[0], d[0] * qv[1]); // local y

        // box normal in the rotated frame
        let s = Vector2::new([lx.signum(), ly.signum()]);
        let qx = lx.abs() - l * V::HALF;
        let qy = ly.abs() - self.th * V::HALF;
        let g = qx.max(qy);
        let mq = Vector2::new([qx.max(V::ZERO), qy.max(V::ZERO)]);
        let ll = mq.l2_norm();
        let face = qx
            .cmp_gt(qy)
            .select(Vector2::new([V::ONE, V::ZERO]), Vector2::new([V::ZERO, V::ONE]));
        let glocal = s * g.cmp_gt(V::ZERO).select(unit_or_zero(mq, ll), face);

        // rotate the local normal back to world: g_world = R^T * g_local
        let gw = Vector2::new([
            d[0].mul_sube(glocal[0], d[1] * glocal[1]),
            d[1].mul_adde(glocal[0], d[0] * glocal[1]),
        ]);
        (ll + g.min(V::ZERO), gw)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for OrientedBox2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let ba = self.b - self.a;
        let l = ba.l2_norm();
        // perpendicular spread of the half-thickness
        let v = Vector2::new([ba[1].abs(), ba[0].abs()]) * (self.th * V::HALF / l);
        Bounds::from_corners(self.a.min(self.b) - v, self.a.max(self.b) + v)
    }
}

/// Rhombus (diamond) with half-diagonals `b = (half_width, half_height)`.
#[derive(Debug, Clone, Copy)]
pub struct Rhombus2D<V: SdfVector> {
    pub b: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for Rhombus2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let bx = self.b[0];
        let by = -self.b[1];
        let px = p[0].abs();
        let py = p[1].abs();
        let bb = bx.mul_adde(bx, by * by);
        let h = (bx.mul_adde(px, by.mul_adde(py, by * by)) / bb).clamp(V::ZERO, V::ONE);
        let qx = bx.nmul_adde(h, px); // px - bx*h
        let qy = by.nmul_adde(h - V::ONE, py); // py - by*(h-1)
        qx.mul_adde(qx, qy * qy).sqrt().mul_sign(qx)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Rhombus2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::new([self.b[0].abs(), self.b[1].abs()]))
    }
}

/// Parallelogram of half-width `wi`, half-height `he` and shear `sk`.
#[derive(Debug, Clone, Copy)]
pub struct Parallelogram2D<V: SdfVector> {
    pub wi: V,
    pub he: V,
    pub sk: V,
}

impl<V: SdfVector> SDF<V, 2> for Parallelogram2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (ex, ey) = (self.sk, self.he);
        // p = (p.y<0)? -p : p
        let flip1 = p[1].cmp_lt(V::ZERO);
        let p1x = p[0].neg_c(flip1);
        let p1y = p[1].neg_c(flip1);

        let w0 = p1x - ex;
        let wx = w0 - w0.clamp(-self.wi, self.wi);
        let wy = p1y - ey;
        let d0x = wx.mul_adde(wx, wy * wy);
        let d0y = -wy;

        let s = p1x.mul_sube(ey, p1y * ex); // p.x*e.y - p.y*e.x
        let flip2 = s.cmp_lt(V::ZERO);
        let p2x = p1x.neg_c(flip2);
        let p2y = p1y.neg_c(flip2);

        let vx0 = p2x - self.wi;
        let vy0 = p2y;
        let t = (vx0.mul_adde(ex, vy0 * ey) / ex.mul_adde(ex, ey * ey)).clamp(-V::ONE, V::ONE);
        let vx = ex.nmul_adde(t, vx0);
        let vy = ey.nmul_adde(t, vy0);
        let d1x = vx.mul_adde(vx, vy * vy);
        let d1y = self.wi.mul_sube(self.he, s.abs()); // wi*he - |s|

        let dx = d0x.min(d1x);
        let dy = d0y.min(d1y);
        dx.sqrt().mul_sign(-dy)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Parallelogram2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::new([self.wi + self.sk.abs(), self.he]))
    }
}

/// Equilateral triangle of "radius" `r`, pointing down, centered at the origin.
#[derive(Debug, Clone, Copy)]
pub struct EquilateralTriangle2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for EquilateralTriangle2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let k = V::SQRT_3;
        let px0 = p[0].abs() - self.r;
        let py0 = p[1] + self.r / k;
        let fold = k.mul_adde(py0, px0).cmp_gt(V::ZERO); // p.x + k*p.y > 0
        let px1 = fold.select(k.nmul_adde(py0, px0) * V::HALF, px0); // (px0 - k*py0)/2
        let py1 = fold.select(k.nmul_adde(px0, -py0) * V::HALF, py0); // (-k*px - py)/2
        let px2 = px1 - px1.clamp(-(self.r * V::TWO), V::ZERO);
        -px2.mul_adde(px2, py1 * py1).sqrt().mul_sign(py1)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for EquilateralTriangle2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let k = V::SQRT_3;
        let sx = p[0].signum();
        let px0 = p[0].abs() - self.r;
        let py0 = p[1] + self.r / k;
        let fold = k.mul_adde(py0, px0).cmp_gt(V::ZERO); // px0 + k*py0 > 0
        let px1 = fold.select(k.nmul_adde(py0, px0) * V::HALF, px0); // (px0 - k*py0)/2
        let py1 = fold.select(k.nmul_adde(px0, -py0) * V::HALF, py0);
        let px2 = px1 - px1.clamp(-(self.r * V::TWO), V::ZERO);
        let l = px2.mul_adde(px2, py1 * py1).sqrt();
        let d = -l.mul_sign(py1);
        // px2 == 0 exactly when the clamp binds, so normalize(px2, py1) is the
        // gradient direction; eval negates and signs by py1.
        let g1 = unit_or_zero(Vector2::new([px2, py1]), l) * -py1.signum();
        // un-reflect through the fold plane (R is its own transpose)
        let half = V::HALF;
        let kh = k * half;
        let g0 = fold.select(
            Vector2::new([
                g1[0].mul_sube(half, g1[1] * kh),  // g1.x*half - g1.y*kh
                g1[0].nmul_sube(kh, g1[1] * half), // -g1.x*kh - g1.y*half
            ]),
            g1,
        );
        (d, Vector2::new([sx * g0[0], g0[1]]))
    }
}

/// Capsule between two circles of radii `r1` (at origin) and `r2` (at height `h`).
///
/// Stores the precomputed slope `$b = \frac{r_1 - r_2}{h}$` and
/// `$a = \sqrt{1 - b^2}$`. Build with [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct UnevenCapsule2D<V: SdfVector> {
    r1: V,
    r2: V,
    h: V,
    a: V,
    b: V,
}

impl<V: SdfVector> UnevenCapsule2D<V> {
    /// Capsule from bottom radius `r1` (at `y = 0`), top radius `r2` (at `y = h`), and
    /// center-to-center height `h`.
    ///
    /// Precomputes the lateral taper `$b = \frac{r_1 - r_2}{h}$` (sine of the flank
    /// half-angle) and `$a = \sqrt{1 - b^2}$` (cosine), which together give the flank's
    /// inward unit normal.
    #[inline(always)]
    pub fn new(r1: V, r2: V, h: V) -> Self {
        let b = (r1 - r2) / h;
        Self {
            r1,
            r2,
            h,
            a: b.nmul_adde(b, V::ONE).sqrt(),
            b,
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for UnevenCapsule2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let px = p[0].abs();
        let py = p[1];
        let (a, b) = (self.a, self.b);
        let k = b.nmul_adde(px, a * py); // dot(p, (-b, a))

        let d_low = px.mul_adde(px, py * py).sqrt() - self.r1;
        let ph = py - self.h;
        let d_high = px.mul_adde(px, ph * ph).sqrt() - self.r2;
        let d_mid = a.mul_adde(px, b * py) - self.r1; // dot(p, (a, b)) - r1

        k.select_negative(d_low, k.cmp_gt(a * self.h).select(d_high, d_mid))
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for UnevenCapsule2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let sx = p[0].signum();
        let px = p[0].abs();
        let py = p[1];
        let (a, b) = (self.a, self.b);
        let k = b.nmul_adde(px, a * py);

        // bottom cap: radial from origin
        let plow = Vector2::new([px, py]);
        let llow = plow.l2_norm();
        let d_low = llow - self.r1;
        // top cap: radial from (0, h)
        let phigh = Vector2::new([px, py - self.h]);
        let lhigh = phigh.l2_norm();
        let d_high = lhigh - self.r2;
        // conical flank: distance to the slanted line, gradient is the unit (a, b)
        let d_mid = a.mul_adde(px, b * py) - self.r1;

        let mid_or_high = k.cmp_gt(a * self.h);
        let dist = k.select_negative(d_low, mid_or_high.select(d_high, d_mid));
        let g = k.cmp_lt(V::ZERO).select(
            unit_or_zero(plow, llow),
            mid_or_high.select(unit_or_zero(phigh, lhigh), Vector2::new([a, b])),
        );
        (dist, Vector2::new([sx * g[0], g[1]]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for UnevenCapsule2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let w = self.r1.max(self.r2);
        Bounds::from_corners(Vector2::new([-w, -self.r1]), Vector2::new([w, self.h + self.r2]))
    }
}

/// Regular pentagon with apothem (flat-to-center distance) `r`.
#[derive(Debug, Clone, Copy)]
pub struct Pentagon2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for Pentagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let k = <V::Element as SdfConsts>::PENTAGON;
        let (kx, ky, kz) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]));
        let mut px = p[0].abs();
        let mut py = p[1];

        let d1 = (-kx).mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.mul_adde(d1, px); // px - d1*(-kx)
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let d2 = kx.mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d2, px); // px - d2*kx
        py = ky.nmul_adde(d2, py); // py - d2*ky

        px -= px.clamp(-(self.r * kz), self.r * kz);
        py -= self.r;
        px.mul_adde(px, py * py).sqrt().mul_sign(py)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Pentagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let k = <V::Element as SdfConsts>::PENTAGON;
        let (kx, ky, kz) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]));
        let sx = p[0].signum();
        let mut px = p[0].abs();
        let mut py = p[1];

        let dot1 = (-kx).mul_adde(px, ky * py); // dot((-kx, ky), p)
        let a1 = dot1.cmp_lt(V::ZERO);
        let d1 = dot1.min(V::ZERO) * V::TWO;
        px = kx.mul_adde(d1, px); // px - d1*(-kx)
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let dot2 = kx.mul_adde(px, ky * py); // dot((kx, ky), p)
        let a2 = dot2.cmp_lt(V::ZERO);
        let d2 = dot2.min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d2, px); // px - d2*kx
        py = ky.nmul_adde(d2, py); // py - d2*ky

        let px2 = px - px.clamp(-(self.r * kz), self.r * kz);
        let py2 = py - self.r;
        let l = px2.mul_adde(px2, py2 * py2).sqrt();
        let d = l.mul_sign(py2);

        let mut g = unit_or_zero(Vector2::new([px2, py2]), l) * py2.signum();
        // un-reflect in reverse order (each reflection is its own transpose)
        let t2 = kx.mul_adde(g[0], ky * g[1]) * V::TWO;
        g = a2.select(Vector2::new([kx.nmul_adde(t2, g[0]), ky.nmul_adde(t2, g[1])]), g);
        let t1 = (-kx).mul_adde(g[0], ky * g[1]) * V::TWO;
        g = a1.select(Vector2::new([kx.mul_adde(t1, g[0]), ky.nmul_adde(t1, g[1])]), g);

        (d, Vector2::new([sx * g[0], g[1]]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Pentagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // circumradius = apothem / cos(pi/5)
        let circ = self.r / V::splat(<V::Element as SdfConsts>::PENTAGON[0]);
        Bounds::symmetric(Vector2::splat(circ))
    }
}

/// Regular octagon with apothem `r`.
#[derive(Debug, Clone, Copy)]
pub struct Octagon2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for Octagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let k = <V::Element as SdfConsts>::OCTAGON;
        let (kx, ky, kz) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]));
        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let d1 = kx.mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d1, px); // px - d1*kx
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let d2 = (-kx).mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.mul_adde(d2, px); // px - d2*(-kx)
        py = ky.nmul_adde(d2, py); // py - d2*ky

        px -= px.clamp(-(kz * self.r), kz * self.r);
        py -= self.r;
        px.mul_adde(px, py * py).sqrt().mul_sign(py)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Octagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let k = <V::Element as SdfConsts>::OCTAGON;
        let (kx, ky, kz) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]));
        let sx = p[0].signum();
        let sy = p[1].signum();
        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let dot1 = kx.mul_adde(px, ky * py); // dot((kx, ky), p)
        let a1 = dot1.cmp_lt(V::ZERO);
        let d1 = dot1.min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d1, px); // px - d1*kx
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let dot2 = (-kx).mul_adde(px, ky * py); // dot((-kx, ky), p)
        let a2 = dot2.cmp_lt(V::ZERO);
        let d2 = dot2.min(V::ZERO) * V::TWO;
        px = kx.mul_adde(d2, px); // px - d2*(-kx)
        py = ky.nmul_adde(d2, py); // py - d2*ky

        let px2 = px - px.clamp(-(kz * self.r), kz * self.r);
        let py2 = py - self.r;
        let l = px2.mul_adde(px2, py2 * py2).sqrt();
        let d = l.mul_sign(py2);

        let mut g = unit_or_zero(Vector2::new([px2, py2]), l) * py2.signum();
        let t2 = (-kx).mul_adde(g[0], ky * g[1]) * V::TWO;
        g = a2.select(Vector2::new([kx.mul_adde(t2, g[0]), ky.nmul_adde(t2, g[1])]), g);
        let t1 = kx.mul_adde(g[0], ky * g[1]) * V::TWO;
        g = a1.select(Vector2::new([kx.nmul_adde(t1, g[0]), ky.nmul_adde(t1, g[1])]), g);

        (d, Vector2::new([sx * g[0], sy * g[1]]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Octagon2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // circumradius = apothem / cos(pi/8)
        let circ = self.r / V::splat(<V::Element as SdfConsts>::OCTAGON[0]).abs();
        Bounds::symmetric(Vector2::splat(circ))
    }
}

/// Hexagram (six-pointed star) with inner radius parameter `r`.
#[derive(Debug, Clone, Copy)]
pub struct Hexagram2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for Hexagram2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let k = <V::Element as SdfConsts>::HEXAGRAM;
        let (kx, ky, kz, kw) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]), V::splat(k[3]));
        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let d1 = kx.mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d1, px); // px - d1*kx
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let d2 = ky.mul_adde(px, kx * py).min(V::ZERO) * V::TWO; // k.yx = (ky, kx)
        px = ky.nmul_adde(d2, px); // px - d2*ky
        py = kx.nmul_adde(d2, py); // py - d2*kx

        px -= px.clamp(self.r * kz, self.r * kw);
        py -= self.r;
        px.mul_adde(px, py * py).sqrt().mul_sign(py)
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for Hexagram2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        let k = <V::Element as SdfConsts>::HEXAGRAM;
        let (kx, ky, kz, kw) = (V::splat(k[0]), V::splat(k[1]), V::splat(k[2]), V::splat(k[3]));
        let sx = p[0].signum();
        let sy = p[1].signum();
        let mut px = p[0].abs();
        let mut py = p[1].abs();

        let dot1 = kx.mul_adde(px, ky * py); // dot((kx, ky), p)
        let a1 = dot1.cmp_lt(V::ZERO);
        let d1 = dot1.min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(d1, px); // px - d1*kx
        py = ky.nmul_adde(d1, py); // py - d1*ky
        let dot2 = ky.mul_adde(px, kx * py); // dot((ky, kx), p)
        let a2 = dot2.cmp_lt(V::ZERO);
        let d2 = dot2.min(V::ZERO) * V::TWO;
        px = ky.nmul_adde(d2, px); // px - d2*ky
        py = kx.nmul_adde(d2, py); // py - d2*kx

        let px2 = px - px.clamp(self.r * kz, self.r * kw);
        let py2 = py - self.r;
        let l = px2.mul_adde(px2, py2 * py2).sqrt();
        let d = l.mul_sign(py2);

        let mut g = unit_or_zero(Vector2::new([px2, py2]), l) * py2.signum();
        let t2 = ky.mul_adde(g[0], kx * g[1]) * V::TWO;
        g = a2.select(Vector2::new([ky.nmul_adde(t2, g[0]), kx.nmul_adde(t2, g[1])]), g);
        let t1 = kx.mul_adde(g[0], ky * g[1]) * V::TWO;
        g = a1.select(Vector2::new([kx.nmul_adde(t1, g[0]), ky.nmul_adde(t1, g[1])]), g);

        (d, Vector2::new([sx * g[0], sy * g[1]]))
    }
}

/// Rounded X / cross of arm length `w` and rounding radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct RoundedX2D<V: SdfVector> {
    pub w: V,
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for RoundedX2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let px = p[0].abs();
        let py = p[1].abs();
        let m = (px + py).min(self.w) * V::HALF;
        let qx = px - m;
        let qy = py - m;
        qx.mul_adde(qx, qy * qy).sqrt() - self.r
    }
}

impl<V: SdfVector> GradientSdf<V, 2> for RoundedX2D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector2<V>) -> (V, Vector2<V>) {
        // In the clamped arm region qy = -qx, so normalize(q) is the gradient in
        // both the clamped and unclamped branches; only the abs-fold sign remains.
        let s = p.signum();
        let px = p[0].abs();
        let py = p[1].abs();
        let m = (px + py).min(self.w) * V::HALF;
        let q = Vector2::new([px - m, py - m]);
        let l = q.l2_norm();
        (l - self.r, unit_or_zero(s * q, l))
    }
}

/// Convex or concave polygon through `N` vertices `v` (in order).
#[derive(Debug, Clone, Copy)]
pub struct Polygon2D<V: SdfVector, const N: usize> {
    pub v: [Vector2<V>; N],
}

impl<V: SdfVector, const N: usize> SDF<V, 2> for Polygon2D<V, N> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let v = &self.v;
        let w0 = p - v[0];
        let mut d = w0.dot(&w0);
        let mut s = V::ONE;

        let mut j = N - 1;
        let mut i = 0;
        while i < N {
            let e = v[j] - v[i];
            let w = p - v[i];
            let b = e.nmul_adde((w.dot(&e) / e.dot(&e)).clamp(V::ZERO, V::ONE), w); // w - e*clamp(...)
            d = d.min(b.dot(&b));

            // winding: flip sign when all three or none of the conditions hold
            let c1 = p[1].cmp_ge(v[i][1]);
            let c2 = p[1].cmp_lt(v[j][1]);
            let c3 = (e[0] * w[1]).cmp_gt(e[1] * w[0]);
            let flip = (c1 & c2 & c3) | (!c1 & !c2 & !c3);
            s = s.neg_c(flip);

            j = i;
            i += 1;
        }

        s * d.sqrt()
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, 2> for Polygon2D<V, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let mut mn = self.v[0];
        let mut mx = self.v[0];
        let mut i = 1;
        while i < N {
            mn = mn.min(self.v[i]);
            mx = mx.max(self.v[i]);
            i += 1;
        }
        Bounds::from_corners(mn, mx)
    }
}

// ===========================================================================
// Batch 2a: more exact distance-only primitives (no trig / cubic)
// ===========================================================================

/// Five-pointed star polygon (pentagram) with outer radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct Pentagram2D<V: SdfVector> {
    pub r: V,
}

impl<V: SdfVector> SDF<V, 2> for Pentagram2D<V>
where
    V::Element: SdfConsts,
{
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let k = <V::Element as SdfConsts>::PENTAGRAM;
        let (k1x, k2x, k1y, k2y, k1z) = (
            V::splat(k[0]),
            V::splat(k[1]),
            V::splat(k[2]),
            V::splat(k[3]),
            V::splat(k[4]),
        );

        let mut px = p[0].abs();
        let mut py = p[1];

        // reflect across v1 = (k1x, -k1y)
        let m1 = k1x.mul_sube(px, k1y * py).max(V::ZERO) * V::TWO; // 2*max(dot(v1,p),0)
        px = k1x.nmul_adde(m1, px); // px - m1*k1x
        py = k1y.mul_adde(m1, py); // py - m1*(-k1y)
        // reflect across v2 = (-k1x, -k1y)
        let m2 = (-k1x).mul_sube(px, k1y * py).max(V::ZERO) * V::TWO;
        px = k1x.mul_adde(m2, px); // px - m2*(-k1x)
        py = k1y.mul_adde(m2, py); // py - m2*(-k1y)

        px = px.abs();
        py -= self.r;

        // q = p - v3 * clamp(dot(p, v3), 0, k1z*r),  v3 = (k2x, -k2y)
        let dpc = k2x.mul_sube(px, k2y * py).clamp(V::ZERO, k1z * self.r);
        let qx = k2x.nmul_adde(dpc, px);
        let qy = (-k2y).nmul_adde(dpc, py);
        let sgn = py.mul_adde(k2x, px * k2y).signum(); // p.y*v3.x - p.x*v3.y
        qx.mul_adde(qx, qy * qy).sqrt() * sgn
    }
}

/// Disk of radius `r` with a straight cut at height `h` (in `[-r, r]`).
///
/// Stores the precomputed half-chord `$w = \sqrt{r^2 - h^2}$`. Build with
/// [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct CutDisk2D<V: SdfVector> {
    r: V,
    h: V,
    w: V,
}

impl<V: SdfVector> CutDisk2D<V> {
    /// Disk of radius `r` cut by a horizontal chord at height `h`, retaining `y >= h`.
    /// `h` must be in `(-r, r)`; at the limits the shape degenerates. Precomputes the
    /// half-chord `$w = \sqrt{r^2 - h^2}$`.
    #[inline(always)]
    pub fn new(r: V, h: V) -> Self {
        Self {
            r,
            h,
            w: r.mul_sube(r, h * h).sqrt(),
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for CutDisk2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (r, h, w) = (self.r, self.h, self.w);
        let px = p[0].abs();
        let py = p[1];
        let s = (h - r)
            .mul_adde(px * px, w * w * (h + r - py * V::TWO))
            .max(h.mul_sube(px, w * py));

        let d_full = px.mul_adde(px, py * py).sqrt() - r;
        let dx = px - w;
        let dy = py - h;
        let d_corner = dx.mul_adde(dx, dy * dy).sqrt();
        let inner = px.cmp_lt(w).select(h - py, d_corner);
        s.select_negative(d_full, inner)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for CutDisk2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        let m = self.h.cmp_gt(V::ZERO).select(self.w, self.r);
        Bounds::from_corners(Vector2::new([-m, self.h]), Vector2::new([m, self.r]))
    }
}

/// Ring of radius `r` and thickness `th`, oriented by `n = (cos, sin)`, open on
/// the side opposite the orientation.
#[derive(Debug, Clone, Copy)]
pub struct Ring2D<V: SdfVector> {
    pub n: Vector2<V>,
    pub r: V,
    pub th: V,
}

impl<V: SdfVector> SDF<V, 2> for Ring2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (nx, ny) = (self.n[0], self.n[1]);
        let px0 = p[0].abs();
        let py0 = p[1];
        let pmx = nx.mul_sube(px0, ny * py0); // n.x*px - n.y*py
        let pmy = ny.mul_adde(px0, nx * py0); // n.y*px + n.x*py
        let l = pmx.mul_adde(pmx, pmy * pmy).sqrt();
        let half_th = self.th * V::HALF;
        let a = (l - self.r).abs() - half_th;
        let iy = ((self.r - pmy).abs() - half_th).max(V::ZERO);
        let b = pmx.mul_adde(pmx, iy * iy).sqrt().mul_sign(pmx);
        a.max(b)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Ring2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::splat(self.r + self.th * V::HALF))
    }
}

/// Horseshoe (a thick arc) opening by half-angle `c = (cos, sin)`, radius `r`,
/// with arm cross-section `w = (length, thickness)`.
#[derive(Debug, Clone, Copy)]
pub struct Horseshoe2D<V: SdfVector> {
    pub c: Vector2<V>,
    pub r: V,
    pub w: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for Horseshoe2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (cx, cy) = (self.c[0], self.c[1]);
        let px0 = p[0].abs();
        let py0 = p[1];
        let l = px0.mul_adde(px0, py0 * py0).sqrt();
        // mat2(-c.x, c.y, c.y, c.x) * p
        let mx = (-cx).mul_adde(px0, cy * py0);
        let my = cy.mul_adde(px0, cx * py0);
        let nx = (my.cmp_gt(V::ZERO) | mx.cmp_gt(V::ZERO)).select(mx, l.mul_sign(-cx));
        let ny = mx.cmp_gt(V::ZERO).select(my, l);
        let qx = nx - self.w[0];
        let qy = (ny - self.r).abs() - self.w[1];
        Vector2::new([qx.max(V::ZERO), qy.max(V::ZERO)]).l2_norm() + qx.max(qy).min(V::ZERO)
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for Horseshoe2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // With c = (cos, sin) unit, the m-transform preserves |p|, and the interior
        // is exactly {nx <= w.x, |ny - r| <= w.y}. Hence |p| <= sqrt(w.x^2 + (r+w.y)^2).
        let outer = self.r + self.w[1];
        let rad = self.w[0].mul_adde(self.w[0], outer * outer).sqrt();
        Bounds::symmetric(Vector2::splat(rad))
    }
}

/// Vesica defined by its two tip points `a`, `b` and width `w`.
#[derive(Debug, Clone, Copy)]
pub struct OrientedVesica2D<V: SdfVector> {
    pub a: Vector2<V>,
    pub b: Vector2<V>,
    pub w: V,
}

impl<V: SdfVector> SDF<V, 2> for OrientedVesica2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let ba = self.b - self.a;
        let r = ba.l2_norm() * V::HALF;
        let d = (r.mul_sube(r, self.w * self.w)) * V::HALF / self.w;
        let v = ba / r;
        let c = (self.a + self.b) * V::HALF;
        let dp = p - c;
        // 0.5 * abs(mat2(v.y, v.x, -v.x, v.y) * (p - c))
        let qx = (v[1].mul_sube(dp[0], v[0] * dp[1])).abs() * V::HALF;
        let qy = (v[0].mul_adde(dp[0], v[1] * dp[1])).abs() * V::HALF;
        let cond = (r * qx).cmp_lt(d * (qy - r));
        let hx = cond.select(V::ZERO, -d);
        let hy = cond.select(r, V::ZERO);
        let hz = cond.select(V::ZERO, d + self.w);
        let ex = qx - hx;
        let ey = qy - hy;
        ex.mul_adde(ex, ey * ey).sqrt() - hz
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for OrientedVesica2D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // The lens is inscribed in the circle through its tips: centre c=(a+b)/2,
        // radius r=|b-a|/2 (its flatter arcs bulge less than that circle). Bounding
        // disk - tight at the tips and waist, conservative elsewhere.
        let c = (self.a + self.b) * V::HALF;
        let r = (self.b - self.a).l2_norm() * V::HALF;
        Bounds::from_corners(c - Vector2::splat(r), c + Vector2::splat(r))
    }
}

/// Rounded plus / cross built from circle arcs, with proportion `h`.
#[derive(Debug, Clone, Copy)]
pub struct RoundedCross2D<V: SdfVector> {
    pub h: V,
}

impl<V: SdfVector> SDF<V, 2> for RoundedCross2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let h = self.h;
        let k = (h + V::ONE / h) * V::HALF;
        let px = p[0].abs();
        let py = p[1].abs();
        let cond = px.cmp_lt(V::ONE) & py.cmp_lt(px.mul_adde(k - h, h)); // p.x*(k-h)+h
        let ix = px - V::ONE;
        let iy = py - k;
        let inner = k - ix.mul_adde(ix, iy * iy).sqrt();
        let dy0 = py - h;
        let d0 = px.mul_adde(px, dy0 * dy0); // dot2(p-(0,h))
        let dx1 = px - V::ONE;
        let d1 = dx1.mul_adde(dx1, py * py); // dot2(p-(1,0))
        cond.select(inner, d0.min(d1).sqrt())
    }
}

/// Egg shape of height `he`, bottom radius `ra`, top radius `rb`, bulge `bu`.
///
/// All of `bu`'s derived geometry (`r`, and the tangent point `(x, y)`) is
/// precomputed - IQ notes this is per-shape. Build with [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct Egg2D<V: SdfVector> {
    he: V,
    ra: V,
    rb: V,
    r: V,
    x: V,
    y: V,
}

impl<V: SdfVector> Egg2D<V> {
    /// Egg from height `he`, bottom cap radius `ra`, top cap radius `rb`, and bulge `bu`.
    ///
    /// `bu` is a dimensionless curvature factor controlling the lateral arc that joins the
    /// two caps: its radius is `$r = \frac{h_e + r_a + r_b}{2\,b_u}$`, so smaller `bu` gives
    /// a larger, flatter arc and larger `bu` a tighter one. Also precomputes the tangency
    /// point `(x, y)` where the lateral arc meets the caps.
    #[inline(always)]
    pub fn new(he: V, ra: V, rb: V, bu: V) -> Self {
        let r = (he + ra + rb) * V::HALF / bu;
        let da = r - ra;
        let db = r - rb;
        let y = db.mul_sube(db, da.mul_adde(da, he * he)) / (he * V::TWO); // (db^2 - da^2 - he^2)/(2he)
        let x = da.mul_sube(da, y * y).sqrt();
        Self { he, ra, rb, r, x, y }
    }
}

impl<V: SdfVector> SDF<V, 2> for Egg2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (he, ra, rb, r, x, y) = (self.he, self.ra, self.rb, self.r, self.x, self.y);

        let px = p[0].abs();
        let py = p[1];
        let k = py.mul_sube(x, px * y);

        let ex = px + x;
        let ey = py + y;
        let d_top = ex.mul_adde(ex, ey * ey).sqrt() - r;
        let d_bot = px.mul_adde(px, py * py).sqrt() - ra;
        let pyh = py - he;
        let d_cap = px.mul_adde(px, pyh * pyh).sqrt() - rb;

        let cond = k.cmp_gt(V::ZERO) & k.cmp_lt(he * (px + x));
        cond.select(d_top, d_bot.min(d_cap))
    }
}

/// Tunnel / arch of half-width `wh.x` and height `wh.y`, opening downward.
#[derive(Debug, Clone, Copy)]
pub struct Tunnel2D<V: SdfVector> {
    pub wh: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 2> for Tunnel2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let px = p[0].abs();
        let py = -p[1];
        let qx0 = px - self.wh[0];
        let qy = py - self.wh[1];
        let mx = qx0.max(V::ZERO);
        let d1 = mx.mul_adde(mx, qy * qy);
        let qx = py
            .cmp_gt(V::ZERO)
            .select(qx0, px.mul_adde(px, py * py).sqrt() - self.wh[0]);
        let my = qy.max(V::ZERO);
        let d2 = qx.mul_adde(qx, my * my);
        let d = d1.min(d2).sqrt();
        qx.max(qy).select_negative(-d, d)
    }
}

/// Staircase of `n` steps, each `wh = (run, rise)`.
#[derive(Debug, Clone, Copy)]
pub struct Stairs2D<V: SdfVector> {
    pub wh: Vector2<V>,
    /// Number of steps as a float vector (use `V::splat(n as f32)`). Must be a positive
    /// integer value; non-integer values produce an undefined shape.
    pub n: V,
}

impl<V: SdfVector> SDF<V, 2> for Stairs2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (whx, why) = (self.wh[0], self.wh[1]);
        let bax = whx * self.n;
        let bay = why * self.n;

        let mut px = p[0];
        let mut py = p[1];

        let e0x = px - px.clamp(V::ZERO, bax);
        let d0 = e0x.mul_adde(e0x, py * py);
        let e1y = py - py.clamp(V::ZERO, bay);
        let exb = px - bax;
        let d1 = exb.mul_adde(exb, e1y * e1y);
        let mut d = d0.min(d1);

        let mut s = (-py).max(px - bax).signum();
        let dia = whx.mul_adde(whx, why * why).sqrt();

        // p = mat2(wh.x,-wh.y, wh.y,wh.x) * p / dia
        let r1x = whx.mul_adde(px, why * py) / dia;
        let r1y = whx.mul_sube(py, why * px) / dia; // -wh.y*px + wh.x*py
        px = r1x;
        py = r1y;

        let id = (px / dia).round().clamp(V::ZERO, self.n - V::ONE);
        px = dia.nmul_adde(id, px); // px - id*dia

        // p = mat2(wh.x, wh.y, -wh.y, wh.x) * p / dia
        let r2x = whx.mul_sube(px, why * py) / dia; // wh.x*px - wh.y*py
        let r2y = why.mul_adde(px, whx * py) / dia;
        px = r2x;
        py = r2y;

        let hh = why * V::HALF;
        py -= hh;
        s = py.cmp_gt(hh.mul_sign(px)).select(V::ONE, s);

        let flip = !(id.cmp_lt(V::HALF) | px.cmp_gt(V::ZERO));
        px = px.neg_c(flip);
        py = py.neg_c(flip);

        let cy = py.clamp(-hh, hh);
        d = d.min(px.mul_adde(px, (py - cy) * (py - cy)));
        let cx = px.clamp(V::ZERO, whx);
        let dyh = py - hh;
        let dxc = px - cx;
        d = d.min(dxc.mul_adde(dxc, dyh * dyh));

        d.sqrt() * s
    }
}

/// The "Cool S" glyph (fixed unit shape).
#[derive(Debug, Clone, Copy)]
pub struct CoolS2D;

impl<V: SdfVector> SDF<V, 2> for CoolS2D {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let six = p[1].select_negative(-p[0], p[0]);
        let px = p[0].abs();
        let py = p[1].abs() - frac::<V, 1, 5>(); // 0.2
        let rex = px - (px / frac::<V, 2, 5>()).round().min(frac::<V, 2, 5>()); // min(round(px/0.4),0.4)
        let aby = (py - frac::<V, 1, 5>()).abs() - frac::<V, 3, 5>(); // abs(py-0.2)-0.6

        let c1 = ((six - py) * V::HALF).clamp(V::ZERO, frac::<V, 1, 5>());
        let a1 = six - c1;
        let b1 = -py - c1;
        let mut d = a1.mul_adde(a1, b1 * b1);

        let c2 = ((px - aby) * V::HALF).clamp(V::ZERO, frac::<V, 2, 5>());
        let a2 = px - c2;
        let b2 = -aby - c2;
        d = d.min(a2.mul_adde(a2, b2 * b2));

        let c3 = py.clamp(V::ZERO, frac::<V, 2, 5>());
        let b3 = py - c3;
        d = d.min(rex.mul_adde(rex, b3 * b3));

        let s = px.mul_adde(V::TWO, aby) + (aby + frac::<V, 2, 5>()).abs() - frac::<V, 2, 5>();
        d.sqrt().mul_sign(s)
    }
}

// ===========================================================================
// Batch 2b: primitives needing per-evaluation trigonometry
// ===========================================================================

/// Regular `n`-pointed star with outer radius `r` and sharpness `m` (in
/// `[2, n]`). The angle and the precomputed cos/sin pairs are stored; evaluation
/// still needs `atan2`/`sin_cos`, hence the precision policy `P`.
#[derive(Debug, Clone, Copy)]
pub struct Star2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub r: V,
    /// half-sector angle `$\pi / n$`
    pub an: V,
    /// `$(\cos(\pi/n),\ \sin(\pi/n))$`
    pub acs: Vector2<V>,
    /// `$(\cos(\pi/m),\ \sin(\pi/m))$`
    pub ecs: Vector2<V>,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> Star2D<V, P> {
    /// Build from point count `n` and sharpness `m` (in `[2, n]`), precomputing angle
    /// constants with policy `P`.
    ///
    /// `m` controls tip sharpness: `m = n` gives a true star polygon with sharp points;
    /// `m = 2` gives wide, petal-like arms. The precomputed `$acs = (\cos(\pi/n),\ \sin(\pi/n))$`
    /// and `$ecs = (\cos(\pi/m),\ \sin(\pi/m))$` encode the sector and tip half-angles.
    #[inline(always)]
    pub fn from_params(r: V, n: u32, m: V) -> Self
    where
        V: RealMathWithPolicy,
    {
        let an = V::PI / V::splat(<V::Element as FloatElement>::from_int(n as i64));
        let en = V::PI / m;
        let (sa, ca) = an.sin_cos_p::<P>();
        let (se, ce) = en.sin_cos_p::<P>();
        Self {
            r,
            an,
            acs: Vector2::new([ca, sa]),
            ecs: Vector2::new([ce, se]),
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for Star2D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (acx, acy) = (self.acs[0], self.acs[1]);
        let (ecx, ecy) = (self.ecs[0], self.ecs[1]);

        let ang = p[0].atan2_p::<P>(p[1]); // atan(p.x, p.y)
        let two_an = self.an * V::TWO;
        // mod(ang, 2*an) - an  =  (ang - an) - 2*an*floor(ang / 2*an)
        let bn = two_an.nmul_adde((ang / two_an).floor(), ang - self.an);
        let (sb, cb) = bn.sin_cos_p::<P>();

        let l = p.l2_norm();
        let mut qx = l.mul_sube(cb, self.r * acx); // l*cos(bn) - r*acs.x
        let mut qy = l.mul_sube(sb.abs(), self.r * acy);

        let t = (-qx.mul_adde(ecx, qy * ecy)).clamp(V::ZERO, self.r * acy / ecy);
        qx = ecx.mul_adde(t, qx);
        qy = ecy.mul_adde(t, qy);
        qx.mul_adde(qx, qy * qy).sqrt().mul_sign(qx)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> BoundedSdf<V, 2> for Star2D<V, P> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::splat(self.r))
    }
}

/// Infinite horizontal wave of circular arcs (period from `tb`), radius `ra`.
///
/// The constructor precomputes the arc center `co`; evaluation itself is
/// trig-free, so the shape carries no policy.
#[derive(Debug, Clone, Copy)]
pub struct CircleWave2D<V: SdfVector> {
    /// arc center `$r_a \cdot (\sin\theta,\ \cos\theta)$`
    pub co: Vector2<V>,
    pub ra: V,
}

impl<V: SdfVector> CircleWave2D<V> {
    /// Build from the bend parameter `tb` and radius `ra`, with policy `P`.
    #[inline(always)]
    pub fn from_params<P: Policy>(tb: V, ra: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        // theta = pi * 5/6 * max(tb, 1e-4)
        let theta = V::PI
            * V::splat(<V::Element as FloatElement>::from_ratio(5, 6))
            * tb.max(V::splat(<V::Element as FloatElement>::from_ratio(1, 10_000)));
        let (s, c) = theta.sin_cos_p::<P>();
        Self {
            co: Vector2::new([ra * s, ra * c]),
            ra,
        }
    }
}

impl<V: SdfVector> SDF<V, 2> for CircleWave2D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let (cox, coy) = (self.co[0], self.co[1]);
        let four = cox * cint::<V, 4>();
        let two_co = cox * V::TWO;
        // p.x = abs(mod(p.x, 4*co.x) - 2*co.x)
        let pxm = four.nmul_adde((p[0] / four).floor(), p[0]); // p.x - 4co*floor
        let px = (pxm - two_co).abs();
        let py = p[1];

        let p2x = (px - two_co).abs();
        let p2y = -py + coy * V::TWO;

        let dist = |qx: V, qy: V| -> V {
            let cand_arc = {
                let dx = qx - cox;
                let dy = qy - coy;
                dx.mul_adde(dx, dy * dy).sqrt()
            };
            let cand_ring = (qx.mul_adde(qx, qy * qy).sqrt() - self.ra).abs();
            coy.mul_sube(qx, cox * qy).cmp_gt(V::ZERO).select(cand_arc, cand_ring)
        };

        dist(px, py).min(dist(p2x, p2y))
    }
}

// ===========================================================================
// Batch 2c: curves needing a per-evaluation cubic solve (policy `P`)
// ===========================================================================

/// Segment of the parabola clipped to `x in [-wi, wi]`, vertex at `(0, he)`.
///
/// The parabola passes through `(+-wi, 0)` with equation `$y = h_e \left(1 - x^2/w_i^2\right)$`.
/// Nearest-point computation reduces to a depressed cubic solved via Cardano / trig.
#[derive(Debug, Clone, Copy)]
pub struct ParabolaSegment2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub wi: V,
    pub he: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> ParabolaSegment2D<V, P> {
    #[inline(always)]
    pub const fn new(wi: V, he: V) -> Self {
        Self {
            wi,
            he,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for ParabolaSegment2D<V, P> {
    #[inline(always)]
    fn eval(&self, pos: Vector2<V>) -> V {
        let third = V::ONE + V::TWO;
        let px = pos[0].abs();
        let py = pos[1];
        let ik = self.wi * self.wi / self.he;
        let p = ik * ik.nmul_adde(V::HALF, self.he - py) / third; // ik*(he - py - 0.5*ik)/3
        let q = px * ik * ik * frac::<V, 1, 4>();
        let h = q.mul_sube(q, p * p * p);

        // h > 0: x = cbrt(q + sqrt(h)) + p / cbrt(...)
        let r = (q + h.max(V::ZERO).sqrt()).cbrt_p::<P>();
        let mut x = r + p / r;
        let needs_trig = h.cmp_le(V::ZERO);
        if needs_trig.any() {
            let r2 = p.sqrt();
            let xt = (q / (p * r2)).acos_p::<P>() / third;
            x = needs_trig.select(xt.cos_p::<P>() * V::TWO * r2, x);
        }
        x = x.min(self.wi);

        let wx = px - x;
        let wy = py - (self.he - x * x / ik);
        wx.mul_adde(wx, wy * wy)
            .sqrt()
            .mul_sign(ik.mul_adde(py - self.he, px * px))
    }
}

/// Quadratic Bezier curve through control points `p0`, `p1`, `p2` (unsigned).
#[derive(Debug, Clone, Copy)]
pub struct QuadraticBezier2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub p0: Vector2<V>,
    pub p1: Vector2<V>,
    pub p2: Vector2<V>,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> QuadraticBezier2D<V, P> {
    #[inline(always)]
    pub const fn new(p0: Vector2<V>, p1: Vector2<V>, p2: Vector2<V>) -> Self {
        Self {
            p0,
            p1,
            p2,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for QuadraticBezier2D<V, P> {
    #[inline(always)]
    fn eval(&self, pos: Vector2<V>) -> V {
        let third = V::ONE + V::TWO;
        let a = self.p1 - self.p0;
        let b = self.p0 - self.p1 * V::TWO + self.p2;
        let c = a * V::TWO;
        let d = self.p0 - pos;

        let kk = V::ONE / b.dot(&b);
        let kx = kk * a.dot(&b);
        let ky = kk * (V::TWO * a.dot(&a) + d.dot(&b)) / third;
        let kz = kk * d.dot(&a);

        let p = kx.nmul_adde(kx, ky); // ky - kx^2
        let p3 = p * p * p;
        let q = kx.mul_adde(kx.mul_sube(V::TWO * kx, third * ky), kz); // kx*(2kx^2-3ky)+kz
        let h = q.mul_adde(q, cint::<V, 4>() * p3); // q^2 + 4 p^3

        let bez = |t: V| -> V {
            let qx = b[0].mul_adde(t, c[0]).mul_adde(t, d[0]); // d + (c + b*t)*t
            let qy = b[1].mul_adde(t, c[1]).mul_adde(t, d[1]);
            qx.mul_adde(qx, qy * qy)
        };

        // h >= 0 branch
        let sh = h.max(V::ZERO).sqrt();
        let ux = ((sh - q) * V::HALF).cbrt_p::<P>();
        let uy = ((-sh - q) * V::HALF).cbrt_p::<P>();
        let t1 = (ux + uy - kx).clamp(V::ZERO, V::ONE);
        let mut res = bez(t1);

        let needs_trig = h.cmp_lt(V::ZERO);
        if needs_trig.any() {
            let z = (-p).max(V::ZERO).sqrt();
            let vv = (q / (p * z * V::TWO)).acos_p::<P>() / third;
            let m = vv.cos_p::<P>();
            let n = vv.sin_p::<P>() * V::SQRT_3;
            let tx = (m * V::TWO).mul_sube(z, kx).clamp(V::ZERO, V::ONE); // 2*m*z - kx
            let ty = (n + m).nmul_sube(z, kx).clamp(V::ZERO, V::ONE); // -(n+m)*z - kx
            res = needs_trig.select(bez(tx).min(bez(ty)), res);
        }
        res.sqrt()
    }
}

/// `clamp(num / den, 0, 1)`, returning `0` (not `NaN`) on the `0/0` lane.
///
/// Bezier extrema are roots of the derivative, found by dividing by a leading
/// coefficient that vanishes for degenerate (collinear) control polygons. A
/// vanishing denominator means the extremum is at infinity, outside `[0, 1]`,
/// so the endpoints already bound that axis; mapping it to `t = 0` keeps the
/// result finite without changing the box.
#[inline(always)]
fn bezier_root<V: SdfVector>(num: V, den: V) -> V {
    let zero = den.cmp_eq(V::ZERO);
    zero.select(V::ZERO, num / zero.select(V::ONE, den)).clamp(V::ZERO, V::ONE)
}

/// Exact axis-aligned bounding box of a quadratic Bezier segment with control
/// points `p0`, `p1`, `p2` (`<https://iquilezles.org/articles/bezierbbox>`).
///
/// The box always contains `p0` and `p2`; the interior extremum on each axis is
/// the derivative root `$t = -b/a$` with `$a = p_0 - 2p_1 + p_2$`,
/// `$b = p_1 - p_0$`, clamped to `[0, 1]`. Dimension-generic, so the same code
/// gives the 2D and 3D boxes.
#[inline(always)]
pub fn quadratic_bezier_aabb<V: SdfVector, const N: usize>(
    p0: Vector<V, N>,
    p1: Vector<V, N>,
    p2: Vector<V, N>,
) -> Bounds<V, N> {
    let a = p0 - p1 * V::TWO + p2;
    let b = p1 - p0;
    let mut q = p0;
    for i in 0..N {
        let t = bezier_root(-b[i], a[i]);
        // q_i = p0_i + t*(2 b_i + t a_i)
        q[i] = a[i].mul_adde(t, V::TWO * b[i]).mul_adde(t, p0[i]);
    }
    Bounds::from_corners(p0.min(p2).min(q), p0.max(p2).max(q))
}

/// Exact axis-aligned bounding box of a cubic Bezier segment with control
/// points `p0`..`p3` (`<https://iquilezles.org/articles/bezierbbox>`).
///
/// The derivative is a quadratic `$a t^2 + 2 b t + c$` with
/// `$a = -p_0 + 3p_1 - 3p_2 + p_3$`, `$b = p_0 - 2p_1 + p_2$`,
/// `$c = -p_0 + p_1$`; its two clamped roots, plus the endpoints `p0`/`p3`,
/// bound each axis. Dimension-generic.
#[inline(always)]
pub fn cubic_bezier_aabb<V: SdfVector, const N: usize>(
    p0: Vector<V, N>,
    p1: Vector<V, N>,
    p2: Vector<V, N>,
    p3: Vector<V, N>,
) -> Bounds<V, N> {
    let c = p1 - p0;
    let b = p0 - p1 * V::TWO + p2;
    let a = (p1 - p2) * (V::ONE + V::TWO) + (p3 - p0);
    let mut lo = p0.min(p3);
    let mut hi = p0.max(p3);
    for i in 0..N {
        let g = b[i].mul_sube(b[i], a[i] * c[i]).max(V::ZERO).sqrt(); // sqrt(max(b^2 - a c, 0))
        for t in [bezier_root(-b[i] - g, a[i]), bezier_root(-b[i] + g, a[i])] {
            // cubic at t: p0 + t*(3c + t*(3b + t*a))
            let three = V::ONE + V::TWO;
            let q = a[i]
                .mul_adde(t, three * b[i])
                .mul_adde(t, three * c[i])
                .mul_adde(t, p0[i]);
            lo[i] = lo[i].min(q);
            hi[i] = hi[i].max(q);
        }
    }
    Bounds::from_corners(lo, hi)
}

impl<V: SdfVector, P: Policy> BoundedSdf<V, 2> for QuadraticBezier2D<V, P>
where
    Self: SDF<V, 2>,
{
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        quadratic_bezier_aabb(self.p0, self.p1, self.p2)
    }
}

/// Blobby four-fold cross of size `he` (unit-ish).
#[derive(Debug, Clone, Copy)]
pub struct BlobbyCross2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub he: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> BlobbyCross2D<V, P> {
    #[inline(always)]
    pub const fn new(he: V) -> Self {
        Self {
            he,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for BlobbyCross2D<V, P> {
    #[inline(always)]
    fn eval(&self, pos: Vector2<V>) -> V {
        let third = V::ONE + V::TWO;
        let ax = pos[0].abs();
        let ay = pos[1].abs();
        let inv_sqrt2 = V::FRAC_1_SQRT_2;
        let qx = (ax - ay).abs() * inv_sqrt2;
        let qy = (V::ONE - ax - ay) * inv_sqrt2;

        let he = self.he;
        let p = (he - qy - frac::<V, 1, 4>() / he) / (he * cint::<V, 6>()); // /(6he)
        let q = qx / (he * he * cint::<V, 16>());
        let h = q.mul_sube(q, p * p * p);

        let r = h.max(V::ZERO).sqrt();
        let mut x = (q + r).cbrt_p::<P>() + (q - r).cbrt_p::<P>();
        let needs_trig = h.cmp_le(V::ZERO);
        if needs_trig.any() {
            let r2 = p.sqrt();
            let xt = (q / (p * r2)).acos_p::<P>() / third;
            x = needs_trig.select(xt.cos_p::<P>() * V::TWO * r2, x);
        }
        x = x.min(V::FRAC_1_SQRT_2);

        let zx = x - qx;
        let zy = he.mul_sube(x.mul_adde(x * -V::TWO, V::ONE), qy); // he*(1-2x^2) - qy
        zx.mul_adde(zx, zy * zy).sqrt().mul_sign(zy)
    }
}

/// Squircle-like "quadratic circle" (fixed unit shape).
#[derive(Debug, Clone, Copy)]
pub struct QuadraticCircle2D<V: SdfVector, P: Policy = DefaultPolicy> {
    _marker: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy> QuadraticCircle2D<V, P> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<V: SdfVector, P: Policy> Default for QuadraticCircle2D<V, P> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for QuadraticCircle2D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let third = V::ONE + V::TWO;
        let ax = p[0].abs();
        let ay = p[1].abs();
        let swap = ay.cmp_gt(ax);
        let px = swap.select(ay, ax);
        let py = swap.select(ax, ay);

        let a = px - py;
        let b = px + py;
        let c = (b * V::TWO - V::ONE) / third;
        let h = a.mul_adde(a, c * c * c);

        let sh = h.max(V::ZERO).sqrt();
        let mut t = (sh - a).cbrt_p::<P>() - (sh + a).cbrt_p::<P>();
        let needs_trig = h.cmp_lt(V::ZERO);
        if needs_trig.any() {
            let z = (-c).max(V::ZERO).sqrt();
            let vv = (a / (c * z)).acos_p::<P>() / third;
            let tt = -z * vv.sin_p::<P>().mul_adde(V::SQRT_3, vv.cos_p::<P>()); // cos + sin*sqrt3
            t = needs_trig.select(tt, t);
        }
        t *= V::HALF;

        let three_q = frac::<V, 3, 4>(); // 0.75
        let wx = t.nmul_adde(t, three_q - t - px); // (three_q - t - px) - t^2
        let wy = t.nmul_adde(t, three_q + t - py); // (three_q + t - py) - t^2
        let sgn = a.mul_adde(a * V::HALF, b - frac::<V, 3, 2>()).signum(); // a^2*0.5 + b - 1.5
        wx.mul_adde(wx, wy * wy).sqrt() * sgn
    }
}

/// Rectangular hyperbola `$xy = k$`, clipped to half-extent `he`.
#[derive(Debug, Clone, Copy)]
pub struct Hyperbola2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub k: V,
    pub he: V,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> Hyperbola2D<V, P> {
    #[inline(always)]
    pub const fn new(k: V, he: V) -> Self {
        Self {
            k,
            he,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for Hyperbola2D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let third = V::ONE + V::TWO;
        let k = self.k;
        let ax = p[0].abs();
        let ay = p[1].abs();
        let inv_sqrt2 = V::FRAC_1_SQRT_2;
        let px = (ax - ay) * inv_sqrt2;
        let py = (ax + ay) * inv_sqrt2;

        let x2 = px * px * frac::<V, 1, 16>();
        let y2 = py * py * frac::<V, 1, 16>();
        let r = k * cint::<V, 4>().mul_sube(k, px * py) / (V::TWO * cint::<V, 6>()); // k*(4k - px*py)/12
        let q = (x2 - y2) * k * k;
        let h = q.mul_adde(q, r * r * r);

        // h >= 0 branch (default): u = (m - r/m)/2, m = cbrt(sqrt(h) - q)
        let m = (h.max(V::ZERO).sqrt() - q).cbrt_p::<P>();
        let mut u = (m - r / m) * V::HALF;
        let needs_trig = h.cmp_lt(V::ZERO);
        if needs_trig.any() {
            let mm = (-r).max(V::ZERO).sqrt();
            let ut = mm * ((q / (r * mm)).acos_p::<P>() / third).cos_p::<P>();
            u = needs_trig.select(ut, u);
        }

        let w = (u + x2).sqrt();
        let b = k.mul_sube(py, x2 * px * V::TWO); // k*py - 2*x2*px
        let mut t = px / cint::<V, 4>() - w + (V::TWO * x2 - u + b / w / cint::<V, 4>()).sqrt();
        // sqrt(he^2/2 + k) - he/sqrt(2)
        let floor = inv_sqrt2.nmul_adde(self.he, (self.he * self.he).mul_adde(V::HALF, k).sqrt());
        t = t.max(floor);

        let dx = px - t;
        let dy = py - k / t;
        let d = dx.mul_adde(dx, dy * dy).sqrt();
        d.neg_c((px * py).cmp_ge(k)) // (px*py < k) ? d : -d
    }
}
