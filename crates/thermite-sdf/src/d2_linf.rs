//! Signed distance fields in the `$L^\infty$` (Chebyshev) metric, ported from
//! <https://iquilezles.org/articles/distfunctions2dlinf>.
//!
//! These measure distance as `$\max_i |\Delta x_i|$` instead of the Euclidean
//! `$\sqrt{\sum_i \Delta x_i^2}$`. They are still valid distance fields - usable
//! for raymarching, rasterization and collision - and have two practical perks:
//! they suit axis-aligned acceleration structures (grids, AABB trees), and some
//! shapes have *simpler* closed forms than in the Euclidean metric (the ellipse
//! and quadratic bezier need only a square root here, not a cubic solve).
//!
//! A key structural fact: an `$L^\infty$` field and its Euclidean counterpart
//! describe the **same shape** - they share the `$f = 0$` boundary and hence the
//! same `$f \le 0$` solid - and differ only in the off-surface magnitude (with
//! `$L^\infty \le L^2$` everywhere). So the bounding box is the same, and
//! [`BoundedLinfSdf`] just delegates to [`BoundedSdf`]. The
//! separate trait still earns its keep: an L∞-only primitive (one with no
//! Euclidean SDF at all - the article notes these exist) can implement its own
//! box without a Euclidean counterpart to borrow from.
//!
//! Rotation is the one weakness of the metric (a rotated `$L^\infty$` primitive
//! is awkward), so these are provided for the axis-aligned primitives only.

use thermite::math::TranscendentalMathWithPolicy;
use thermite::math::policy::Policy;
use thermite::prelude::*;

use thermite_geometry::prim::{Bounds, Vector, Vector2, vector::VectorOps as _};

use crate::consts::cint;
use crate::d2::{Ellipse2D, ParabolaSegment2D, Rhombus2D};
use crate::dn::{NBox, NSphere};
use crate::{BoundedSdf, SdfVector, unit_or_zero};

/// A signed distance field measured in the `$L^\infty$` (Chebyshev) norm.
///
/// The `$L^\infty$` analogue of [`SDF`](crate::SDF): same sign convention and
/// same `$f = 0$` surface, but the magnitude is the Chebyshev distance. A type
/// may implement both, sharing its [`BoundedSdf`].
pub trait LinfSdf<V: SdfVector, const N: usize> {
    /// Returns the signed `$L^\infty$` distance.
    fn eval_linf(&self, p: Vector<V, N>) -> V;
}

/// The `$L^\infty$` analogue of [`BoundedSdf`]: an
/// axis-aligned bounding box of the `$f \le 0$` solid.
///
/// For every primitive here the L∞ solid is identical to the Euclidean one, so
/// the impls forward to [`BoundedSdf::aabb`] - the box
/// is metric-independent. It is a distinct trait (rather than a blanket impl) on
/// purpose: a future L∞ primitive whose box genuinely differs, or one with no
/// Euclidean SDF to delegate to, can supply its own without a coherence clash.
pub trait BoundedLinfSdf<V: SdfVector, const N: usize>: LinfSdf<V, N> {
    /// Axis-aligned box enclosing the `$L^\infty$` solid.
    fn aabb_linf(&self) -> Bounds<V, N>;
}

/// The `$L^\infty$` analogue of [`GradientSdf`](crate::GradientSdf): distance plus
/// the gradient of the Chebyshev field.
///
/// **This is not the Euclidean surface normal.** A distance field in norm
/// `$\lVert\cdot\rVert$` satisfies the eikonal equation in the *dual* norm, and
/// the dual of `$L^\infty$` is `$L^1$`, so here `$\lVert\nabla f\rVert_1 = 1$`
/// (not `$\lVert\nabla f\rVert_2 = 1$`). The direction is axis-biased - for a
/// point source `$f = \max_i|p_i|$` the gradient is a single coordinate basis
/// vector `$\pm e_k$` - and there are kinks all along the `$L^\infty$` diagonals
/// where two coordinates tie. For lighting you almost always want the Euclidean
/// [`GradientSdf`](crate::GradientSdf) of the same shape instead; this is for
/// the field's own geometry (L∞ steepest descent, collision response).
pub trait GradientLinfSdf<V: SdfVector, const N: usize>: LinfSdf<V, N> {
    /// Returns the signed `$L^\infty$` distance and the (L2-normalised) gradient
    /// direction.
    fn eval_linf_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>);
}

impl<V: SdfVector, const N: usize> BoundedLinfSdf<V, N> for NBox<V, N> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, N> {
        self.aabb() // same solid as the Euclidean box
    }
}

impl<V: SdfVector> BoundedLinfSdf<V, 2> for NSphere<V> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, 2> {
        self.aabb()
    }
}

impl<V: SdfVector> BoundedLinfSdf<V, 2> for Ellipse2D<V> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, 2> {
        self.aabb()
    }
}

impl<V: SdfVector> BoundedLinfSdf<V, 2> for Rhombus2D<V> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, 2> {
        self.aabb()
    }
}

/// Axis-aligned box (and [`Box2D`](crate::Box2D)). In `$L^\infty$` the box SDF is
/// just `$\max_i (|p_i| - b_i)$` - dimension-generic and the cheapest field here.
impl<V: SdfVector, const N: usize> LinfSdf<V, N> for NBox<V, N> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector<V, N>) -> V {
        let mut d = p[0].abs() - self.b[0];
        for k in 1..N {
            d = d.max(p[k].abs() - self.b[k]);
        }
        d
    }
}

impl<V: SdfVector, const N: usize> GradientLinfSdf<V, N> for NBox<V, N> {
    #[inline(always)]
    fn eval_linf_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // f = max_k(|p_k| - b_k). Where smooth, the gradient is sign(p_k) e_k for
        // the single dominant axis k - the canonical axis-biased L-inf gradient.
        // On a tie (an L-inf diagonal) several lanes light up; L2-normalising
        // then splits the unit length between them.
        let mut d = p[0].abs() - self.b[0];
        for k in 1..N {
            d = d.max(p[k].abs() - self.b[k]);
        }
        let mut g = Vector::ZERO;
        for k in 0..N {
            let is_max = (p[k].abs() - self.b[k]).cmp_ge(d);
            g[k] = is_max.select(p[k].signum(), V::ZERO);
        }
        (d, unit_or_zero(g, g.l2_norm()))
    }
}

/// Circle (and [`Circle2D`](crate::Circle2D)) of radius `radius`.
///
/// Folds to the first octant, then solves the nearest-point along the `$L^\infty$`
/// diamond against the circle:
///
/// ```math
/// a - \sqrt{\tfrac{r^2}{2} - b^2}, \quad
/// a = \tfrac12(n_x + n_y), \quad b = \tfrac12(n_x - n_y),
/// ```
/// with `$n = \max(|p|,\ |p|_{yx} - r)$`.
impl<V: SdfVector> LinfSdf<V, 2> for NSphere<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let r = self.radius;
        let (ax, ay) = (p[0].abs(), p[1].abs());
        let nx = ax.max(ay - r);
        let ny = ay.max(ax - r);
        let a = (nx + ny) * V::HALF;
        let b = (nx - ny) * V::HALF;
        a - b.nmul_adde(b, r * r * V::HALF).max(V::ZERO).sqrt() // a - sqrt(r^2/2 - b^2)
    }
}

/// Ellipse with semi-axes `ab`. Exact, and unlike the Euclidean ellipse it needs
/// only a single square root (no cubic solve):
///
/// ```math
/// n_x - \bigl(r_y\sqrt{m - d^2} - r_x d\bigr)\frac{r_x}{m}, \quad
/// m = r_x^2 + r_y^2, \quad d = n_y - n_x.
/// ```
impl<V: SdfVector> LinfSdf<V, 2> for Ellipse2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let (rx, ry) = (self.ab[0], self.ab[1]);
        let (ax, ay) = (p[0].abs(), p[1].abs());
        let nx = ax.max(ay - ry);
        let ny = ay.max(ax - rx);
        let m = rx.mul_adde(rx, ry * ry);
        let d = ny - nx;
        let root = d.nmul_adde(d, m).max(V::ZERO).sqrt(); // sqrt(m - d^2)
        nx - ry.mul_adde(root, -(rx * d)) * rx / m
    }
}

/// Rhombus with axis half-extents `b = (w, h)` (vertices at `(±w, 0)`, `(0, ±h)`).
impl<V: SdfVector> LinfSdf<V, 2> for Rhombus2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let (w, h) = (self.b[0], self.b[1]);
        let px = p[0].abs() - w;
        let py = p[1].abs();
        let f = ((py - px) / (h + w)).clamp(V::ZERO, V::ONE);
        let qx = f.mul_adde(w, px).abs(); // |px - f*(-w)| = |px + f*w|
        let qy = f.nmul_adde(h, py).abs(); // |py - f*h|
        let d = qx.max(qy);
        // negate (flip to inside) where h*px + w*py <= 0
        d.neg_c(h.mul_adde(px, w * py).cmp_le(V::ZERO))
    }
}

/// Parabolic segment `$y = h_e (1 - x^2 / w_i^2)$` (matching
/// [`ParabolaSegment2D`]). Unsigned distance to the arc; three branches (off the
/// open end, past the apex, against the curve) lowered to lane-wise `select`s.
impl<V: SdfVector, P: Policy> LinfSdf<V, 2> for ParabolaSegment2D<V, P> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let (wi, he) = (self.wi, self.he);
        let px = p[0].abs();
        let py = p[1];
        let a = px - py - wi;
        let b = py - px - he;
        let r_a = (px - wi).abs().max(py.abs());
        let r_b = (py - he).abs().max(px);
        // wi^2 - 4*he*b, guarded (only the c-branch lanes use it; b <= 0 there)
        let disc = (cint::<V, 4>() * he).nmul_adde(b, wi * wi).max(V::ZERO);
        let r_c = (wi - disc.sqrt()).mul_adde(wi * V::HALF / he, px);
        // a > 0 ? r_a : (b > 0 ? r_b : r_c)   (a, b are mutually exclusive)
        let inner = b.cmp_gt(V::ZERO).select(r_b, r_c);
        a.cmp_gt(V::ZERO).select(r_a, inner)
    }
}

/// Rounded box: an axis-aligned box of half-extents `b` with a uniform corner
/// radius `r`, measured in `$L^\infty$`. This is a *new* struct (not the
/// Euclidean [`RoundedBox2D`](crate::RoundedBox2D), which carries four per-corner
/// radii) - the L∞ closed form assumes a single radius.
#[derive(Debug, Clone, Copy)]
pub struct LinfRoundBox2D<V: SdfVector> {
    pub b: Vector2<V>,
    pub r: V,
}

impl<V: SdfVector> LinfSdf<V, 2> for LinfRoundBox2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        // box-fold, then the same diamond-vs-circle solve as the L-inf circle
        let px0 = p[0].abs() - self.b[0];
        let py0 = p[1].abs() - self.b[1];
        let px = px0.max(py0 - self.r);
        let py = py0.max(px0 - self.r);
        let a = (px + py) * V::HALF;
        let b = (px - py) * V::HALF;
        a - b.nmul_adde(b, self.r * self.r * V::HALF).max(V::ZERO).sqrt()
    }
}

impl<V: SdfVector> BoundedLinfSdf<V, 2> for LinfRoundBox2D<V> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::new([self.b[0] + self.r, self.b[1] + self.r]))
    }
}

/// Oriented box: a rectangle of half-size `half`, rotated by an angle whose
/// `(cos, sin)` is `sc`, measured in `$L^\infty$`. Construct from an angle with
/// [`from_angle`](Self::from_angle).
#[derive(Debug, Clone, Copy)]
pub struct LinfOrientedBox2D<V: SdfVector> {
    pub half: Vector2<V>,
    /// `(cos angle, sin angle)`.
    pub sc: Vector2<V>,
}

impl<V: SdfVector> LinfOrientedBox2D<V> {
    /// Builds from a half-size and a rotation `angle` (radians).
    #[inline(always)]
    pub fn from_angle<P: Policy>(half: Vector2<V>, angle: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        Self {
            half,
            sc: Vector2::new([angle.cos_p::<P>(), angle.sin_p::<P>()]),
        }
    }

    /// `max(|s.x - s.z|, |s.x + s.z|)` style L∞ bbox half-extent on each axis.
    #[inline(always)]
    fn bbox_half(&self) -> Vector2<V> {
        let (c, s) = (self.sc[0], self.sc[1]);
        let (rx, ry) = (self.half[0], self.half[1]);
        let (sx, sy, sz, sw) = (c * rx, s * rx, s * ry, c * ry);
        Vector2::new([
            (sx - sz).abs().max((sx + sz).abs()),
            (sy + sw).abs().max((sy - sw).abs()),
        ])
    }
}

impl<V: SdfVector> LinfSdf<V, 2> for LinfOrientedBox2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let (c, s) = (self.sc[0], self.sc[1]);
        let (rx, ry) = (self.half[0], self.half[1]);
        // q = w.xyyx * p.xxyy = (c*px, s*px, s*py, c*py)
        let (qx, qy, qz, qw) = (c * p[0], s * p[0], s * p[1], c * p[1]);
        // rotated-rectangle term, divided by the L-inf correction factor
        let rot = ((qx + qz).abs() - rx).max((qw - qy).abs() - ry) / (c - s).abs().max((c + s).abs());
        // axis-aligned bbox term
        let bb = self.bbox_half();
        let bbox = (p[0].abs() - bb[0]).max(p[1].abs() - bb[1]);
        rot.max(bbox)
    }
}

impl<V: SdfVector> BoundedLinfSdf<V, 2> for LinfOrientedBox2D<V> {
    #[inline(always)]
    fn aabb_linf(&self) -> Bounds<V, 2> {
        Bounds::symmetric(self.bbox_half())
    }
}

/// Line segment from `a` to `b`, measured in `$L^\infty$` (unsigned distance to
/// the segment). Distinct from the Euclidean [`Segment2D`](crate::Segment2D),
/// which carries a thickness; this is the bare L∞ line.
#[derive(Debug, Clone, Copy)]
pub struct LinfSegment2D<V: SdfVector> {
    pub a: Vector2<V>,
    pub b: Vector2<V>,
}

impl<V: SdfVector> LinfSdf<V, 2> for LinfSegment2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let pa = p - self.a;
        let ba = self.b - self.a;
        // s picks the L-inf-aligned projection axis (kept as the exact compare,
        // not select_negative: an axis-aligned segment has ba.x*ba.y == 0 and the
        // sign there must match IQ's `> 0` convention).
        let sgn = (ba[0] * ba[1]).cmp_gt(V::ZERO).select(V::ONE, V::NEG_ONE);
        let num = sgn.mul_adde(pa[0], pa[1]); // pa.y + s*pa.x
        let den = sgn.mul_adde(ba[0], ba[1]); // ba.y + s*ba.x
        // h = clamp(num/den, 0, 1), guarded so den == 0 gives h = 0 (not NaN)
        let zero = den.cmp_eq(V::ZERO);
        let h = zero.select(V::ZERO, num / zero.select(V::ONE, den)).clamp(V::ZERO, V::ONE);
        let qx = ba[0].nmul_adde(h, pa[0]).abs(); // |pa.x - h*ba.x|
        let qy = ba[1].nmul_adde(h, pa[1]).abs();
        qx.max(qy)
    }
}

/// `$L^\infty$` distance between two points: `$\max(|p_x - c_x|, |p_y - c_y|)$`.
#[inline(always)]
fn linf_dist<V: SdfVector>(p: Vector2<V>, c: Vector2<V>) -> V {
    (p[0] - c[0]).abs().max((p[1] - c[1]).abs())
}

/// Capsule (a Euclidean stadium: segment `a`..`b` thickened by radius `r`),
/// measured in `$L^\infty$`. The signed distance is negative inside. Ported from
/// the article's `sdCapsule` - the body sides plus the two round caps, with the
/// per-cap up-to-3-root selection lowered to lane-wise masks.
#[derive(Debug, Clone, Copy)]
pub struct LinfCapsule2D<V: SdfVector> {
    pub a: Vector2<V>,
    pub b: Vector2<V>,
    pub r: V,
}

impl<V: SdfVector> LinfSdf<V, 2> for LinfCapsule2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let rb = self.r;
        let p = p - (self.a + self.b) * V::HALF; // recenter
        let ab = self.b - self.a;
        let len = ab.l2_norm();
        let u = ab / len;
        let v = Vector2::new([-u[1], u[0]]);
        let l = len * V::HALF;
        let a = u * l;
        let w = v * rb;
        let ss = (u[0] * u[1]).cmp_gt(V::ZERO).select(V::ONE, V::NEG_ONE);
        let de = V::ONE / ss.mul_adde(a[0], a[1]); // 1/(a.y + ss*a.x)

        // distance to the two straight body sides
        let side = |wn: Vector2<V>| -> V {
            let t = (ss.mul_adde(wn[0], wn[1]) * -de).clamp(V::ZERO, V::TWO);
            let q = (wn + a * t).abs();
            q[0].max(q[1])
        };
        let d_body = side(p - a - w).min(side(p - a + w));

        let pa = p - a;
        let pb = p + a;
        let (da, db) = (pa.dot(&pa), pb.dot(&pb));
        let rb2 = rb * rb;

        // Common path: body distance, refined by the round caps when past the body
        // bbox. qa = abs(R * p) - (l, rb),  R = mat2(u.x,-u.y, u.y,u.x).
        let rpx = u[0].mul_adde(p[0], u[1] * p[1]);
        let rpy = u[0].mul_adde(p[1], -(u[1] * p[0]));
        let di = (rpx.abs() - l).max(rpy.abs() - rb);
        let cap = |pabs: Vector2<V>| -> V {
            let near = (pabs[1] - pabs[0]).abs().cmp_lt(rb);
            let b = (pabs[0] + pabs[1]) * V::HALF;
            let cc = pabs.dot(&pabs) - rb2;
            let dc_near = b - b.mul_adde(b, -(cc * V::HALF)).max(V::ZERO).sqrt();
            near.select(dc_near, pabs[0].max(pabs[1]) - rb)
        };
        let d_far = d_body.min(cap(pa.abs())).min(cap(pb.abs()));
        let d_out = di.cmp_gt(V::ZERO).select(d_far, d_body);
        let mut result = d_out.mul_sign(di); // d_out * sign(di)

        // Rare path: a lane is inside one of the round caps (negative interior).
        // Only there do we need the cap's circle roots (two sqrts + four guards),
        // so skip the whole block unless some lane requires it.
        let inside = da.min(db).cmp_lt(rb2);
        if thermite::unlikely(inside.any()) {
            let swap = db.cmp_lt(da);
            let s = swap.select(V::NEG_ONE, V::ONE);
            let pc = swap.select(pb, pa);
            let b1 = (pc[0] + pc[1]) * V::HALF;
            let b2 = (pc[0] - pc[1]) * V::HALF;
            let c = pc.dot(&pc) - rb2;
            let r1 = b1.mul_adde(b1, -(c * V::HALF)).sqrt(); // sqrt(b1^2 - c/2), may be NaN
            let r2 = b2.mul_adde(b2, -(c * V::HALF)).sqrt();
            let (t1x, t1y) = (r1 - b1, r1 + b1);
            let (t2x, t2y) = (r2 - b2, r2 + b2);
            // guard g(dir, t) = s*dot(pc + dir*t, -u) < 0
            let guard = |d0: V, dirx: V, diry: V, t: V| -> V {
                let dot = u[0].mul_adde(dirx.mul_adde(t, pc[0]), u[1] * diry.mul_adde(t, pc[1]));
                (s * -dot).cmp_lt(V::ZERO).select(d0.min(t), d0)
            };
            let mut din = d_body;
            din = guard(din, V::ONE, V::ONE, t1x);
            din = guard(din, V::NEG_ONE, V::NEG_ONE, t1y);
            din = guard(din, V::ONE, V::NEG_ONE, t2x);
            din = guard(din, V::NEG_ONE, V::ONE, t2y);
            result = inside.select(-din, result);
        }
        result
    }
}

/// Circular arc of radius `rb` centred at the origin, measured in `$L^\infty$`
/// (unsigned distance to the arc curve). The arc is the set of circle points `p`
/// (`$|p| = r_b$`) whose projection onto the unit direction `sc` is below `w`,
/// i.e. `$\langle p, \mathrm{sc}\rangle < w$`: for `$w > 0$` this is the major
/// arc on the far side of `sc`, for `$w < 0$` the minor arc; the two endpoints
/// sit at offset `w` along `sc`. Ported from the article's `sdArc`.
#[derive(Debug, Clone, Copy)]
pub struct LinfArc2D<V: SdfVector> {
    /// Unit direction `(cos, sin)` the chord offset `w` is measured along.
    pub sc: Vector2<V>,
    pub rb: V,
    pub w: V,
}

impl<V: SdfVector> LinfSdf<V, 2> for LinfArc2D<V> {
    #[inline(always)]
    fn eval_linf(&self, p: Vector2<V>) -> V {
        let (rb, w) = (self.rb, self.w);
        let u = self.sc;
        let v = Vector2::new([-u[1], u[0]]);
        let hh = rb.mul_adde(rb, -(w * w)).max(V::ZERO).sqrt(); // sqrt(rb^2 - w^2)

        // bounding points: the two arc extremes ...
        let mut d = linf_dist(p, u * w + v * hh).min(linf_dist(p, u * w - v * hh));
        // ... and the cardinal points the arc reaches (axis crossings)
        let card = |d0: V, active: V, c: Vector2<V>| active.cmp_lt(w).select(d0.min(linf_dist(p, c)), d0);
        d = card(d, -rb * u[0], Vector2::new([-rb, V::ZERO]));
        d = card(d, rb * u[0], Vector2::new([rb, V::ZERO]));
        d = card(d, -rb * u[1], Vector2::new([V::ZERO, -rb]));
        d = card(d, rb * u[1], Vector2::new([V::ZERO, rb]));

        // circular section: L-inf rays hitting the circle, kept if within the arc
        let b1 = (p[0] + p[1]) * V::HALF;
        let b2 = (p[0] - p[1]) * V::HALF;
        let c = p.dot(&p) - rb * rb;
        let h1 = b1.mul_adde(b1, -(c * V::HALF));
        let h2 = b2.mul_adde(b2, -(c * V::HALF));
        let r1 = h1.max(V::ZERO).sqrt();
        let r2 = h2.max(V::ZERO).sqrt();
        // within-arc test: dot(p + dir*t, u) < w
        let sect = |d0: V, hh: V, dirx: V, diry: V, t: V| -> V {
            let dot = u[0].mul_adde(dirx.mul_adde(t, p[0]), u[1] * diry.mul_adde(t, p[1]));
            (hh.cmp_gt(V::ZERO) & dot.cmp_lt(w)).select(d0.min(t.abs()), d0)
        };
        d = sect(d, h1, V::ONE, V::ONE, r1 - b1);
        d = sect(d, h1, V::NEG_ONE, V::NEG_ONE, r1 + b1);
        d = sect(d, h2, V::ONE, V::NEG_ONE, r2 - b2);
        d = sect(d, h2, V::NEG_ONE, V::ONE, r2 + b2);
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Box2D, Circle2D, SDF};

    /// 1-lane scalar vector.
    type V = thermite::Vector<f32>;

    fn vv(x: f32) -> V {
        V::splat(x)
    }
    fn pp(x: f32, y: f32) -> Vector2<V> {
        Vector2::new([vv(x), vv(y)])
    }
    fn sc(x: V) -> f32 {
        x.extract::<0>()
    }

    #[test]
    fn on_surface_and_sign() {
        // L-inf shares the f=0 set with the Euclidean version.
        macro_rules! zero_at {
            ($sh:expr, $p:expr) => {
                assert!(
                    sc($sh.eval_linf($p)).abs() < 1e-4,
                    "{} not zero on surface",
                    stringify!($sh)
                );
            };
        }
        zero_at!(Box2D { b: pp(1.0, 1.0) }, pp(1.0, 0.3));
        zero_at!(Circle2D { radius: vv(1.0) }, pp(1.0, 0.0));
        zero_at!(Ellipse2D { ab: pp(2.0, 1.0) }, pp(2.0, 0.0));
        zero_at!(Rhombus2D { b: pp(1.0, 1.0) }, pp(1.0, 0.0));
        zero_at!(ParabolaSegment2D::<V>::new(vv(1.0), vv(1.0)), pp(0.0, 1.0));

        // interior negative, exterior positive (closed shapes)
        assert!(sc((Box2D { b: pp(1.0, 1.0) }).eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc((Circle2D { radius: vv(1.0) }).eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc((Ellipse2D { ab: pp(2.0, 1.0) }).eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc((Rhombus2D { b: pp(1.0, 1.0) }).eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc((Box2D { b: pp(1.0, 1.0) }).eval_linf(pp(5.0, 5.0))) > 0.0);
    }

    #[test]
    fn linf_at_most_euclidean_outside() {
        // For any shape, min_s L_inf(p,s) <= min_s L2(p,s), so the L-inf field
        // never exceeds the Euclidean one outside, and the signs agree.
        macro_rules! check {
            ($sh:expr, $pts:expr) => {{
                let sh = $sh;
                for q in $pts {
                    let le = sc(LinfSdf::eval_linf(&sh, q));
                    let eu = sc(SDF::eval(&sh, q));
                    assert_eq!(le < 0.0, eu < 0.0, "{} sign mismatch", stringify!($sh));
                    if eu > 0.0 {
                        assert!(le <= eu + 1e-4, "{}: linf {le} > euclid {eu}", stringify!($sh));
                        assert!(le > -1e-4, "{}: linf negative outside", stringify!($sh));
                    }
                }
            }};
        }
        check!(Box2D { b: pp(1.0, 0.6) }, [pp(2.0, 1.5), pp(3.0, 0.1), pp(0.2, 2.0)]);
        check!(Circle2D { radius: vv(1.0) }, [pp(3.0, 4.0), pp(2.0, 2.0), pp(0.0, 3.0)]);
        check!(Ellipse2D { ab: pp(1.5, 0.8) }, [pp(3.0, 2.0), pp(0.0, 2.5), pp(2.5, 0.0)]);
        check!(Rhombus2D { b: pp(1.3, 0.8) }, [pp(2.0, 1.5), pp(3.0, 0.2), pp(0.1, 2.0)]);
    }

    #[test]
    fn bounded_linf_matches_euclidean_and_contains_solid() {
        use crate::BoundedSdf;

        // The L-inf box equals the Euclidean box (shared solid) and actually
        // contains every interior (eval_linf <= 0) sample.
        macro_rules! check {
            ($sh:expr, $range:expr) => {{
                let sh = $sh;
                let bl = BoundedLinfSdf::aabb_linf(&sh);
                let be = BoundedSdf::<V, 2>::aabb(&sh);
                for k in 0..2 {
                    assert!((sc(bl.0[k][0]) - sc(be.0[k][0])).abs() < 1e-6);
                    assert!((sc(bl.0[k][1]) - sc(be.0[k][1])).abs() < 1e-6);
                }
                // containment of the L-inf solid
                let n = 50;
                let (xmin, xmax) = (sc(bl.0[0][0]), sc(bl.0[0][1]));
                let (ymin, ymax) = (sc(bl.0[1][0]), sc(bl.0[1][1]));
                for i in 0..=n {
                    for j in 0..=n {
                        let x = -$range + 2.0 * $range * (i as f32 / n as f32);
                        let y = -$range + 2.0 * $range * (j as f32 / n as f32);
                        if sc(sh.eval_linf(pp(x, y))) <= 0.0 {
                            assert!(
                                x >= xmin - 1e-2 && x <= xmax + 1e-2 && y >= ymin - 1e-2 && y <= ymax + 1e-2,
                                "L-inf interior ({x},{y}) outside aabb"
                            );
                        }
                    }
                }
            }};
        }
        check!(Box2D { b: pp(1.2, 0.7) }, 3.0);
        check!(Circle2D { radius: vv(1.3) }, 3.0);
        check!(Ellipse2D { ab: pp(1.5, 0.8) }, 3.0);
        check!(Rhombus2D { b: pp(1.3, 0.8) }, 3.0);
    }

    #[test]
    fn box_linf_gradient_matches_finite_diff() {
        use crate::FiniteDiff;
        // Analytic axis-pick gradient must agree with the FiniteDiff stencil over
        // eval_linf, away from the L-inf diagonals (kinks).
        let bx = Box2D { b: pp(1.0, 0.6) };
        let fd = FiniteDiff::with_eps(bx, vv(1e-3));
        for q in [pp(3.0, 0.1), pp(0.1, 2.0), pp(-2.5, 0.3), pp(0.2, -1.7)] {
            let (da, ga) = bx.eval_linf_grad(q);
            let (_, gf) = fd.eval_linf_grad(q);
            assert!((sc(da) - sc(bx.eval_linf(q))).abs() < 1e-6, "dist forwarded");
            assert!(
                (sc(ga[0]) - sc(gf[0])).abs() < 2e-2 && (sc(ga[1]) - sc(gf[1])).abs() < 2e-2,
                "box grad ({},{}) vs fd ({},{})",
                sc(ga[0]),
                sc(ga[1]),
                sc(gf[0]),
                sc(gf[1])
            );
            // it is a unit axis vector (one dominant component)
            assert!((sc(ga[0]).abs() - 1.0).abs() < 1e-3 || (sc(ga[1]).abs() - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn new_linf_primitives() {
        use thermite::math::policy::DefaultPolicy;

        // Rounded box: interior negative, surface ~0, exterior positive; box-fold
        // means a face point sits on the surface.
        let rb = LinfRoundBox2D { b: pp(1.0, 0.7), r: vv(0.2) };
        assert!(sc(rb.eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc(rb.eval_linf(pp(5.0, 5.0))) > 0.0);
        assert!(sc(rb.eval_linf(pp(1.2, 0.0))).abs() < 1e-4); // +x face (b.x + r)
        // bbox contains the solid
        let bb = rb.aabb_linf();
        assert!((sc(bb.0[0][1]) - 1.2).abs() < 1e-6 && (sc(bb.0[1][1]) - 0.9).abs() < 1e-6);

        // Oriented box: an axis-aligned one (angle 0) must match the plain L-inf box.
        let plain = Box2D { b: pp(1.0, 0.6) };
        let ob0 = LinfOrientedBox2D::from_angle::<DefaultPolicy>(pp(1.0, 0.6), vv(0.0));
        for q in [pp(2.0, 0.1), pp(0.3, 1.5), pp(0.0, 0.0), pp(-1.4, -0.9)] {
            assert!((sc(ob0.eval_linf(q)) - sc(plain.eval_linf(q))).abs() < 1e-4, "oriented@0 == box");
        }
        // rotated box: interior negative, far exterior positive, finite
        let ob = LinfOrientedBox2D::from_angle::<DefaultPolicy>(pp(1.2, 0.5), vv(0.6));
        assert!(sc(ob.eval_linf(pp(0.0, 0.0))) < 0.0);
        assert!(sc(ob.eval_linf(pp(6.0, 6.0))) > 0.0);
        // bbox contains the rotated solid (grid sample)
        let bb = ob.aabb_linf();
        let (xm, ym) = (sc(bb.0[0][1]), sc(bb.0[1][1]));
        for i in 0..=40 {
            for j in 0..=40 {
                let x = -4.0 + 8.0 * i as f32 / 40.0;
                let y = -4.0 + 8.0 * j as f32 / 40.0;
                if sc(ob.eval_linf(pp(x, y))) <= 0.0 {
                    assert!(x.abs() <= xm + 1e-2 && y.abs() <= ym + 1e-2, "oriented solid outside bbox");
                }
            }
        }

        // Segment: ~0 along the segment, positive off it, symmetric in endpoints.
        let seg = LinfSegment2D { a: pp(-1.0, -0.5), b: pp(1.0, 0.5) };
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let x = -1.0 + 2.0 * t;
            let y = -0.5 + 1.0 * t;
            assert!(sc(seg.eval_linf(pp(x, y))).abs() < 1e-4, "on segment t={t}");
        }
        assert!(sc(seg.eval_linf(pp(0.0, 2.0))) > 0.0);
        // axis-aligned (horizontal) segment: ba.x*ba.y == 0 path
        let h = LinfSegment2D { a: pp(-1.0, 0.0), b: pp(1.0, 0.0) };
        assert!(sc(h.eval_linf(pp(0.0, 0.0))).abs() < 1e-4);
        assert!((sc(h.eval_linf(pp(0.0, 0.8))) - 0.8).abs() < 1e-4); // L-inf dist = |y|
    }

    #[test]
    fn linf_capsule() {
        // Horizontal capsule, segment (-1,0)..(1,0), radius 0.5. Its f=0 set is the
        // Euclidean stadium boundary, independent of metric, so on-boundary points
        // must read ~0.
        let cap = LinfCapsule2D {
            a: pp(-1.0, 0.0),
            b: pp(1.0, 0.0),
            r: vv(0.5),
        };
        // straight sides: y = +-0.5 for x in [-1, 1]
        for x in [-0.9f32, -0.3, 0.0, 0.5, 0.95] {
            assert!(sc(cap.eval_linf(pp(x, 0.5))).abs() < 1e-4, "top side x={x}");
            assert!(sc(cap.eval_linf(pp(x, -0.5))).abs() < 1e-4, "bottom side x={x}");
        }
        // round caps: (+-1,0) + 0.5*(cos,sin) over the outward semicircle
        for deg in [-80.0f32, -40.0, 0.0, 40.0, 80.0] {
            let a = deg.to_radians();
            let (cx, cy) = (a.cos(), a.sin());
            assert!(sc(cap.eval_linf(pp(1.0 + 0.5 * cx, 0.5 * cy))).abs() < 1e-4, "+cap {deg}");
            assert!(sc(cap.eval_linf(pp(-1.0 - 0.5 * cx, 0.5 * cy))).abs() < 1e-4, "-cap {deg}");
        }
        // interior negative, far exterior positive
        assert!(sc(cap.eval_linf(pp(0.0, 0.0))) < -0.4); // deep inside
        assert!(sc(cap.eval_linf(pp(0.5, 0.2))) < 0.0);
        assert!(sc(cap.eval_linf(pp(5.0, 5.0))) > 0.0);
        // L-inf <= Euclidean-ish: a point straight above is at L-inf distance |y|-r
        assert!((sc(cap.eval_linf(pp(0.0, 2.0))) - 1.5).abs() < 1e-4);

        // a tilted capsule still reads ~0 on its boundary
        let cap2 = LinfCapsule2D {
            a: pp(-0.8, -0.6),
            b: pp(0.9, 0.7),
            r: vv(0.4),
        };
        let ab = (0.9f32 + 0.8, 0.7 + 0.6);
        let len = (ab.0 * ab.0 + ab.1 * ab.1).sqrt();
        let (ux, uy) = (ab.0 / len, ab.1 / len);
        let (vx, vy) = (-uy, ux); // perpendicular
        for t in [0.0f32, 0.5, 1.0] {
            let (mx, my) = (-0.8 + ab.0 * t, -0.6 + ab.1 * t);
            assert!(sc(cap2.eval_linf(pp(mx + 0.4 * vx, my + 0.4 * vy))).abs() < 1e-4, "tilt side t={t}");
        }
        assert!(sc(cap2.eval_linf(pp(0.05, 0.05))) < 0.0);
    }

    #[test]
    fn linf_arc() {
        // Radius 1, direction +x, w = 0.5: the arc is the circle points with
        // px < 0.5 (the major arc, angles ~60..300 deg).
        let arc = LinfArc2D {
            sc: pp(1.0, 0.0),
            rb: vv(1.0),
            w: vv(0.5),
        };
        // points ON the arc (px < 0.5) read ~0
        for deg in [90.0f32, 135.0, 180.0, 225.0, 270.0] {
            let a = deg.to_radians();
            assert!(sc(arc.eval_linf(pp(a.cos(), a.sin()))).abs() < 1e-4, "on arc {deg}");
        }
        // a circle point in the missing gap (angle 0, px = 1 > 0.5): nearest arc
        // point is an extreme (0.5, +-0.866), so L-inf distance ~ 0.866 > 0
        assert!(sc(arc.eval_linf(pp(1.0, 0.0))) > 0.3);
        // far exterior and centre are well off the curve
        assert!(sc(arc.eval_linf(pp(2.0, 0.0))) > 0.5);
        assert!(sc(arc.eval_linf(pp(0.0, 0.0))) > 0.5);

        // direction +y, w = 1.2/sqrt2: arc is the circle points with py < 0.849.
        let h = core::f32::consts::FRAC_1_SQRT_2;
        let arc2 = LinfArc2D {
            sc: pp(0.0, 1.0),
            rb: vv(1.2),
            w: vv(1.2 * h),
        };
        for deg in [0.0f32, 30.0, 180.0, 210.0, 300.0] {
            let a = deg.to_radians();
            assert!(sc(arc2.eval_linf(pp(1.2 * a.cos(), 1.2 * a.sin()))).abs() < 1e-3, "arc2 {deg}");
        }
    }

    #[test]
    fn finite_diff_forwards_and_differentiates_linf() {
        use crate::FiniteDiff;

        let bx = Box2D { b: pp(1.0, 1.0) };
        let fd = FiniteDiff::with_eps(bx, vv(1e-3));

        // eval_linf is forwarded unchanged through the wrapper
        for q in [pp(3.0, 0.1), pp(0.2, 2.0), pp(0.0, 0.0)] {
            assert!((sc(fd.eval_linf(q)) - sc(bx.eval_linf(q))).abs() < 1e-6);
        }

        // The L-inf gradient is axis-biased: at (3, 0.1) the x-overflow dominates
        // (|x|-1 = 2 vs |y|-1 = -0.9), so the field is locally x-1 and the
        // gradient points ~ along +x - unlike a Euclidean corner normal.
        let (_, g) = fd.eval_linf_grad(pp(3.0, 0.1));
        assert!(
            (sc(g[0]) - 1.0).abs() < 2e-2 && sc(g[1]).abs() < 2e-2,
            "linf grad ({}, {})",
            sc(g[0]),
            sc(g[1])
        );
        // returned direction is L2-normalised
        let l2 = (sc(g[0]) * sc(g[0]) + sc(g[1]) * sc(g[1])).sqrt();
        assert!((l2 - 1.0).abs() < 1e-2);

        // BoundedLinfSdf is forwarded too
        let bb = BoundedLinfSdf::aabb_linf(&fd);
        assert!((sc(bb.0[0][1]) - 1.0).abs() < 1e-6 && (sc(bb.0[1][0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_is_dimension_generic() {
        use thermite_geometry::prim::Vector as NV;
        let b = NBox {
            b: NV::<V, 3>::new([vv(1.0), vv(1.0), vv(1.0)]),
        };
        // (2,2,0): max(2-1, 2-1, 0-1) = 1
        assert!((sc(b.eval_linf(NV::new([vv(2.0), vv(2.0), vv(0.0)]))) - 1.0).abs() < 1e-5);
        assert!(sc(b.eval_linf(NV::new([vv(0.0), vv(0.0), vv(0.0)]))) < 0.0);
    }
}
