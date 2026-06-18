//! 3D signed-distance fields: distance, analytic gradient, and bounding box.
//!
//! Ported from <https://iquilezles.org/articles/distfunctions>,
//! <https://iquilezles.org/articles/distgradfunctions3d> and
//! <https://iquilezles.org/articles/bboxes3d>.

use thermite::math::TranscendentalMathWithPolicy;
use thermite::prelude::*;

use thermite_geometry::prim::{Bounds, Vector2, Vector3, vector::VectorOps as _};

use crate::consts::frac;
use crate::{BoundedSdf, GradientSdf, SDF, SdfVector, unit_or_zero};

/// Sphere of radius `radius` centered at the origin (the 3D [`NSphere`](crate::dn::NSphere)).
pub type Sphere3D<V> = crate::dn::NSphere<V>;

/// Rounded axis-aligned box with half-extents `b` and corner radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct Box3D<V: SdfVector> {
    pub b: Vector3<V>,
    pub r: V,
}

impl<V: SdfVector> SDF<V, 3> for Box3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let w = p.abs() - (self.b - Vector3::splat(self.r));
        let g = w[0].max(w[1].max(w[2]));
        let l = Vector3::new([w[0].max(V::ZERO), w[1].max(V::ZERO), w[2].max(V::ZERO)]).l2_norm();
        g.cmp_gt(V::ZERO).select(l, g) - self.r
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for Box3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let s = p.signum();
        let w = p.abs() - (self.b - Vector3::splat(self.r));
        let g = w[0].max(w[1].max(w[2]));
        let q = Vector3::new([w[0].max(V::ZERO), w[1].max(V::ZERO), w[2].max(V::ZERO)]);
        let l = q.l2_norm();

        let outside = g.cmp_gt(V::ZERO);
        // interior gradient snaps to whichever face(s) attain the max
        let face = Vector3::new([
            w[0].cmp_eq(g).select(V::ONE, V::ZERO),
            w[1].cmp_eq(g).select(V::ONE, V::ZERO),
            w[2].cmp_eq(g).select(V::ONE, V::ZERO),
        ]);
        let grad = outside.select(q / l, face) * s;

        (outside.select(l, g) - self.r, grad)
    }
}

/// Torus in the xz-plane with major radius `ra` and minor radius `rb`.
#[derive(Debug, Clone, Copy)]
pub struct Torus3D<V: SdfVector> {
    pub ra: V,
    pub rb: V,
}

impl<V: SdfVector> SDF<V, 3> for Torus3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let h = p[0].mul_adde(p[0], p[2] * p[2]).sqrt(); // length(p.xz)
        let hra = h - self.ra;
        hra.mul_adde(hra, p[1] * p[1]).sqrt() - self.rb
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for Torus3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let h = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let hra = h - self.ra;
        let dist = hra.mul_adde(hra, p[1] * p[1]).sqrt() - self.rb;
        let grad = Vector3::new([p[0] * hra, p[1] * h, p[2] * hra]).normalize();
        (dist, grad)
    }
}

/// Capsule / thick line segment from `a` to `b` with radius `r` (the 3D
/// [`NCapsule`](crate::dn::NCapsule)).
pub type Segment3D<V> = crate::dn::NCapsule<V, 3>;

/// Ellipsoid with semi-axes `r` - exact gradient, approximate (bounding)
/// distance (the 3D [`NEllipsoid`](crate::dn::NEllipsoid)).
pub type Ellipsoid3D<V> = crate::dn::NEllipsoid<V, 3>;

/// Link (a torus stretched into a chain link by `le`) with ring radius `r1` and
/// tube radius `r2`.
#[derive(Debug, Clone, Copy)]
pub struct Link3D<V: SdfVector> {
    pub le: V,
    pub r1: V,
    pub r2: V,
}

impl<V: SdfVector> SDF<V, 3> for Link3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qy = p[1] - p[1].clamp(-self.le, self.le);
        let w = p[0].mul_adde(p[0], qy * qy).sqrt(); // length(q.xy)
        let wr1 = w - self.r1;
        wr1.mul_adde(wr1, p[2] * p[2]).sqrt() - self.r2
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for Link3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let qy = p[1] - p[1].clamp(-self.le, self.le);
        let q = Vector3::new([p[0], qy, p[2]]);
        let w = p[0].mul_adde(p[0], qy * qy).sqrt();
        let wr1 = w - self.r1;
        let l = wr1.mul_adde(wr1, p[2] * p[2]).sqrt();

        // q - (r1*q.xy/w, 0)
        let sub = Vector3::new([self.r1 * q[0] / w, self.r1 * q[1] / w, V::ZERO]);
        (l - self.r2, (q - sub) / l)
    }
}

/// Rounded cone (a capsule with different end radii `r1` at `a` and `r2` at `b`).
#[derive(Debug, Clone, Copy)]
pub struct RoundCone3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub r1: V,
    pub r2: V,
}

impl<V: SdfVector> SDF<V, 3> for RoundCone3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        self.eval_grad(p).0
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for RoundCone3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let ba = self.b - self.a;
        let l2 = ba.dot(&ba);
        let rr = self.r1 - self.r2;
        let a2 = rr.nmul_adde(rr, l2);
        let il2 = V::ONE / l2;

        let pa = p - self.a;
        let pb = p - self.b;
        let y = pa.dot(&ba);
        let z = y - l2;
        let x2 = l2.mul_adde(pa.dot(&pa), -(y * y)); // l2*dot(pa,pa) - y^2
        let y2 = y * y;
        let z2 = z * z;
        let k = (rr * rr * x2).mul_sign(rr);

        // branch A: spherical cap at b (radius r2)
        let wa = (il2 * (x2 + z2)).sqrt();
        let dist_a = wa - self.r2;
        let grad_a = unit_or_zero(pb, wa);

        // branch B: spherical cap at a (radius r1)
        let wb = (il2 * (x2 + y2)).sqrt();
        let dist_b = wb - self.r1;
        let grad_b = unit_or_zero(pa, wb);

        // branch C: the conical side
        let wc = (x2 * a2).sqrt();
        let dist_c = y.mul_adde(rr, wc).mul_sube(il2, self.r1); // (wc + y*rr)*il2 - r1
        // il2 * (rr*ba + a2*(pa*l2 - y*ba)/wc)
        let grad_c = ba.nmul_adde(y, pa * l2).mul_adde(a2 / wc, ba * rr) * il2;

        let cond_a = ((a2 * z2).mul_sign(z)).cmp_gt(k);
        let cond_b = ((a2 * y2).mul_sign(y)).cmp_lt(k);

        let dist = cond_a.select(dist_a, cond_b.select(dist_b, dist_c));
        let grad = cond_a.select(grad_a, cond_b.select(grad_b, grad_c));
        (dist, grad)
    }
}

/// Vertical capped cone of height `$2 h_e$` (centered on the origin), base radius
/// `r1` (at `y < 0`) and top radius `r2`.
#[derive(Debug, Clone, Copy)]
pub struct CappedCone3D<V: SdfVector> {
    pub he: V,
    pub r1: V,
    pub r2: V,
}

impl<V: SdfVector> SDF<V, 3> for CappedCone3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        self.eval_grad(p).0
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for CappedCone3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let kx = self.r2 - self.r1;
        let ky = self.he * V::TWO;
        let m = kx.mul_adde(kx, ky * ky); // dot(k,k)

        let l = p[0].mul_adde(p[0], p[2] * p[2]).sqrt(); // length(p.xz)

        let qx = self.r2 - l;
        let qy = self.he - p[1];

        let ax = l - l.min(p[1].select_negative(self.r1, self.r2));
        let ay = p[1].abs() - self.he;

        // b = k*clamp(dot(q,k)/m, 0, 1) - q
        let t = (qx.mul_adde(kx, qy * ky) / m).clamp(V::ZERO, V::ONE);
        let bx = kx.mul_sube(t, qx); // kx*t - qx
        let by = ky.mul_sube(t, qy);

        let s = (bx.cmp_lt(V::ZERO) & ay.cmp_lt(V::ZERO)).select(V::NEG_ONE, V::ONE);
        let la = ax.mul_adde(ax, ay * ay);
        let lb = bx.mul_adde(bx, by * by);

        let pick = la.cmp_lt(lb);
        let dist = s * pick.select(la, lb).sqrt();

        // la-branch: straight up/down cap normal; lb-branch: the slanted side
        let grad_la = Vector3::new([V::ZERO, p[1].signum(), V::ZERO]);
        let inv_sqrt_m = V::ONE / m.sqrt();
        let ls = l.cmp_gt(V::ZERO).select(l, V::ONE); // guard the axis (l == 0)
        let grad_lb = Vector3::new([ky * p[0] / ls, -kx, ky * p[2] / ls]) * inv_sqrt_m;

        (dist, pick.select(grad_la, grad_lb))
    }
}

/// Vertical cylinder of radius `r`, centered on the origin.
///
/// Stored as the half-height (so evaluation avoids a per-call halving).
/// Construct with [`from_height`](Self::from_height) (total height) or
/// [`from_half_height`](Self::from_half_height).
#[derive(Debug, Clone, Copy)]
pub struct VerticalCylinder3D<V: SdfVector> {
    half_height: V,
    pub r: V,
}

impl<V: SdfVector> VerticalCylinder3D<V> {
    /// From the total height (tip to tip) and radius.
    #[inline(always)]
    pub fn from_height(height: V, r: V) -> Self {
        Self {
            half_height: height * V::HALF,
            r,
        }
    }

    /// From the half-height and radius.
    #[inline(always)]
    pub const fn from_half_height(half_height: V, r: V) -> Self {
        Self { half_height, r }
    }
}

impl<V: SdfVector> SDF<V, 3> for VerticalCylinder3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let l = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let ex = l - self.r;
        let ey = p[1].abs() - self.half_height;
        let hx = ex.max(V::ZERO);
        let hy = ey.max(V::ZERO);
        let f = hx.mul_adde(hx, hy * hy).sqrt();
        let g = ex.max(ey);
        g.cmp_le(V::ZERO).select(g, f)
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for VerticalCylinder3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let l = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let ex = l - self.r;
        let ey = p[1].abs() - self.half_height;
        let hx = ex.max(V::ZERO);
        let hy = ey.max(V::ZERO);
        let f = hx.mul_adde(hx, hy * hy).sqrt();
        let g = ex.max(ey);

        let ls = l.cmp_gt(V::ZERO).select(l, V::ONE); // guard the axis (l == 0)
        let du = Vector3::new([p[0] / ls, V::ZERO, p[2] / ls]);
        let dv = Vector3::new([V::ZERO, p[1].signum(), V::ZERO]);

        let inside = g.cmp_le(V::ZERO);
        let grad = inside.select(
            ex.cmp_gt(ey).select(du, dv),
            dv.mul_adde(hy, du * hx) / f, // (h.x*du + h.y*dv)/f
        );
        (inside.select(g, f), grad)
    }
}

/// Arbitrary cylinder spanning the segment `a` to `b` with radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct Cylinder3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub r: V,
}

impl<V: SdfVector> SDF<V, 3> for Cylinder3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        self.eval_grad(p).0
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for Cylinder3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let ba = (self.b - self.a) * V::HALF;
        let ce = (self.b + self.a) * V::HALF;
        let l = ba.l2_norm();
        let d = ba / l;

        let q = p - ce;
        let v = q.dot(&d);
        let u = d.nmul_adde(v, q); // q - v*d
        let k = u.l2_norm();

        let ex = k - self.r;
        let ey = v.abs() - l;
        let hx = ex.max(V::ZERO);
        let hy = ey.max(V::ZERO);
        let f = hx.mul_adde(hx, hy * hy).sqrt();
        let g = ex.max(ey);

        let du = unit_or_zero(u, k);
        let dv = d * v.signum(); // v<0 ? -d : d

        let inside = g.cmp_le(V::ZERO);
        let grad = inside.select(ex.cmp_gt(ey).select(du, dv), dv.mul_adde(hy, du * hx) / f);
        (inside.select(g, f), grad)
    }
}

// --- bounding boxes ---

impl<V: SdfVector> BoundedSdf<V, 3> for Box3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(self.b)
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Torus3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let outer = self.ra + self.rb;
        Bounds::symmetric(Vector3::new([outer, self.rb, outer]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Link3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let ring = self.r1 + self.r2;
        Bounds::symmetric(Vector3::new([ring, self.le + ring, self.r2]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for RoundCone3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // hull of the two end spheres: union of their boxes
        let r1 = Vector3::splat(self.r1);
        let r2 = Vector3::splat(self.r2);
        Bounds::from_corners((self.a - r1).min(self.b - r2), (self.a + r1).max(self.b + r2))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for CappedCone3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let w = self.r1.max(self.r2);
        Bounds::symmetric(Vector3::new([w, self.he, w]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for VerticalCylinder3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::new([self.r, self.half_height, self.r]))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Cylinder3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // disk-swept caps: e_i = r * sqrt(1 - axis_i^2 / |axis|^2)
        let axis = self.b - self.a;
        let inv = V::ONE / axis.dot(&axis);
        let cap = |a: V| self.r * (a * a).nmul_adde(inv, V::ONE).max(V::ZERO).sqrt();
        let e = Vector3::new([cap(axis[0]), cap(axis[1]), cap(axis[2])]);
        Bounds::from_corners(self.a.min(self.b) - e, self.a.max(self.b) + e)
    }
}

// ===========================================================================
// Additional 3D primitives from distfunctions
// ===========================================================================

/// Wireframe box: edges of half-extents `b` with thickness `e`.
#[derive(Debug, Clone, Copy)]
pub struct BoxFrame3D<V: SdfVector> {
    pub b: Vector3<V>,
    pub e: V,
}

impl<V: SdfVector> SDF<V, 3> for BoxFrame3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let px = p[0].abs() - self.b[0];
        let py = p[1].abs() - self.b[1];
        let pz = p[2].abs() - self.b[2];
        let qx = (px + self.e).abs() - self.e;
        let qy = (py + self.e).abs() - self.e;
        let qz = (pz + self.e).abs() - self.e;

        let edge = |a: V, b: V, c: V| -> V {
            let l = Vector3::new([a.max(V::ZERO), b.max(V::ZERO), c.max(V::ZERO)]).l2_norm();
            l + a.max(b.max(c)).min(V::ZERO)
        };
        edge(px, qy, qz).min(edge(qx, py, qz)).min(edge(qx, qy, pz))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for BoxFrame3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(self.b)
    }
}

/// Infinite cylinder parallel to the y axis.
///
/// `c` packs three values as `[center_x, center_z, radius]` to match Quilez's original
/// convention. The cylinder axis passes through `(c[0], *, c[1])` with radius `c[2]`.
#[derive(Debug, Clone, Copy)]
pub struct InfiniteCylinder3D<V: SdfVector> {
    pub c: Vector3<V>,
}

impl<V: SdfVector> SDF<V, 3> for InfiniteCylinder3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let dx = p[0] - self.c[0];
        let dz = p[2] - self.c[1];
        dx.mul_adde(dx, dz * dz).sqrt() - self.c[2]
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for InfiniteCylinder3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let dx = p[0] - self.c[0];
        let dz = p[2] - self.c[1];
        let l = dx.mul_adde(dx, dz * dz).sqrt();
        (l - self.c[2], unit_or_zero(Vector3::new([dx, V::ZERO, dz]), l))
    }
}

/// Plane with unit normal `n` and offset `h` (the 3D [`NPlane`](crate::dn::NPlane)).
pub type Plane3D<V> = crate::dn::NPlane<V, 3>;

/// Hexagonal prism, `h = (apothem, half_depth)`, axis along z.
#[derive(Debug, Clone, Copy)]
pub struct HexPrism3D<V: SdfVector> {
    pub h: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 3> for HexPrism3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let kx = -(V::SQRT_3 * V::HALF);
        let ky = V::HALF;
        let kz = V::FRAC_1_SQRT_3;

        let mut px = p[0].abs();
        let mut py = p[1].abs();
        let pz = p[2].abs();

        let m = kx.mul_adde(px, ky * py).min(V::ZERO) * V::TWO;
        px = kx.nmul_adde(m, px); // px - m*kx
        py = ky.nmul_adde(m, py); // py - m*ky

        let cx = px - px.clamp(-(kz * self.h[0]), kz * self.h[0]);
        let cy = py - self.h[0];
        let dx = cx.mul_adde(cx, cy * cy).sqrt().mul_sign(py - self.h[0]);
        let dy = pz - self.h[1];
        dx.max(dy).min(V::ZERO) + Vector2::new([dx.max(V::ZERO), dy.max(V::ZERO)]).l2_norm()
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for HexPrism3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let circ = self.h[0] * V::TWO * V::FRAC_1_SQRT_3;
        Bounds::symmetric(Vector3::new([circ, circ, self.h[1]]))
    }
}

/// Vertical capsule from the origin up to height `h`, radius `r`.
#[derive(Debug, Clone, Copy)]
pub struct VerticalCapsule3D<V: SdfVector> {
    pub h: V,
    pub r: V,
}

impl<V: SdfVector> SDF<V, 3> for VerticalCapsule3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let py = p[1] - p[1].clamp(V::ZERO, self.h);
        Vector3::new([p[0], py, p[2]]).l2_norm() - self.r
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for VerticalCapsule3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let py = p[1] - p[1].clamp(V::ZERO, self.h);
        let q = Vector3::new([p[0], py, p[2]]);
        let l = q.l2_norm();
        (l - self.r, unit_or_zero(q, l))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for VerticalCapsule3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::from_corners(
            Vector3::new([-self.r, -self.r, -self.r]),
            Vector3::new([self.r, self.h + self.r, self.r]),
        )
    }
}

/// Vertical cylinder of radius `ra` with rounded vertical edges `rb`, half-height `h`.
#[derive(Debug, Clone, Copy)]
pub struct RoundedCylinder3D<V: SdfVector> {
    pub ra: V,
    pub rb: V,
    pub h: V,
}

impl<V: SdfVector> SDF<V, 3> for RoundedCylinder3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let l = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let dx = l - self.ra + self.rb;
        let dy = p[1].abs() - self.h + self.rb;
        dx.max(dy).min(V::ZERO) + Vector2::new([dx.max(V::ZERO), dy.max(V::ZERO)]).l2_norm() - self.rb
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for RoundedCylinder3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        // a 2D box in (radial, |y|): solve the box normal there, then map the
        // radial axis back onto the xz unit direction and `|y|` onto sign(y).
        let l = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let dx = l - self.ra + self.rb;
        let dy = p[1].abs() - self.h + self.rb;
        let g = dx.max(dy);
        let mq = Vector2::new([dx.max(V::ZERO), dy.max(V::ZERO)]);
        let ll = mq.l2_norm();
        let n2 = g.cmp_gt(V::ZERO).select(
            unit_or_zero(mq, ll),
            dx.cmp_gt(dy)
                .select(Vector2::new([V::ONE, V::ZERO]), Vector2::new([V::ZERO, V::ONE])),
        );
        let rhat = unit_or_zero(Vector2::new([p[0], p[2]]), l); // radial direction in xz
        let world = Vector3::new([n2[0] * rhat[0], n2[1] * p[1].signum(), n2[0] * rhat[1]]);
        (g.min(V::ZERO) + ll - self.rb, world)
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for RoundedCylinder3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::new([self.ra, self.h, self.ra]))
    }
}

/// Sphere of radius `r` cut by a horizontal plane at height `h`.
///
/// Stores the precomputed half-chord `w = sqrt(r^2 - h^2)`. Build with
/// [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct CutSphere3D<V: SdfVector> {
    r: V,
    h: V,
    w: V,
}

impl<V: SdfVector> CutSphere3D<V> {
    /// Sphere of radius `r` cut by a horizontal plane at height `h`, retaining `y >= h`.
    /// `h` in `(-r, r)` for a proper cap. Precomputes `$w = \sqrt{r^2 - h^2}$`, the radius
    /// of the cut circle.
    #[inline(always)]
    pub fn new(r: V, h: V) -> Self {
        Self {
            r,
            h,
            w: r.mul_sube(r, h * h).sqrt(),
        }
    }
}

impl<V: SdfVector> SDF<V, 3> for CutSphere3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let (r, h, w) = (self.r, self.h, self.w);
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = p[1];
        let s = (h - r)
            .mul_adde(qx * qx, w * w * (h + r - qy * V::TWO))
            .max(h.mul_sube(qx, w * qy));
        let d_full = qx.mul_adde(qx, qy * qy).sqrt() - r;
        let cx = qx - w;
        let cy = qy - h;
        let d_corner = cx.mul_adde(cx, cy * cy).sqrt();
        let inner = qx.cmp_lt(w).select(h - qy, d_corner);
        s.select_negative(d_full, inner)
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for CutSphere3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let (r, h, w) = (self.r, self.h, self.w);
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = p[1];
        let s = (h - r)
            .mul_adde(qx * qx, w * w * (h + r - qy * V::TWO))
            .max(h.mul_sube(qx, w * qy));

        let lsphere = qx.mul_adde(qx, qy * qy).sqrt(); // == |p|
        let d_full = lsphere - r;
        let cx = qx - w;
        let cy = qy - h;
        let d_corner = cx.mul_adde(cx, cy * cy).sqrt();
        let inner = qx.cmp_lt(w).select(h - qy, d_corner);
        let dist = s.select_negative(d_full, inner);

        // sphere region: radial from origin. cap: the flat plane (0,-1,0).
        // rim: radial-in-(qx,qy), mapped back onto the xz unit direction.
        let g_sphere = unit_or_zero(p, lsphere);
        let rhat = unit_or_zero(Vector2::new([p[0], p[2]]), qx);
        let g_rim = unit_or_zero(Vector3::new([cx * rhat[0], cy, cx * rhat[1]]), d_corner);
        let g_cap = Vector3::new([V::ZERO, V::NEG_ONE, V::ZERO]);
        let inner_g = qx.cmp_lt(w).select(g_cap, g_rim);
        (dist, s.cmp_lt(V::ZERO).select(g_sphere, inner_g))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for CutSphere3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // conservative: the kept cap is contained in the full radius-r sphere
        Bounds::symmetric(Vector3::splat(self.r))
    }
}

/// Hollow spherical cap of radius `r`, cut at height `h`, with wall thickness `t`.
///
/// Stores the precomputed half-chord `w = sqrt(r^2 - h^2)`. Build with
/// [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct CutHollowSphere3D<V: SdfVector> {
    r: V,
    h: V,
    t: V,
    w: V,
}

impl<V: SdfVector> CutHollowSphere3D<V> {
    /// Hollow spherical cap: sphere of radius `r` cut at height `h` with shell thickness
    /// `t`. Precomputes `$w = \sqrt{r^2 - h^2}$`.
    #[inline(always)]
    pub fn new(r: V, h: V, t: V) -> Self {
        Self {
            r,
            h,
            t,
            w: r.mul_sube(r, h * h).sqrt(),
        }
    }
}

impl<V: SdfVector> SDF<V, 3> for CutHollowSphere3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let (r, h, w) = (self.r, self.h, self.w);
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = p[1];
        let cx = qx - w;
        let cy = qy - h;
        let cap = cx.mul_adde(cx, cy * cy).sqrt();
        let ring = (qx.mul_adde(qx, qy * qy).sqrt() - r).abs();
        (h * qx).cmp_lt(w * qy).select(cap, ring) - self.t
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for CutHollowSphere3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.r + self.t))
    }
}

/// Death star: sphere `ra` with a spherical bite of radius `rb` offset by `d`.
///
/// Stores the precomputed intersection point `(a, b)`. Build with
/// [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct DeathStar3D<V: SdfVector> {
    ra: V,
    rb: V,
    d: V,
    a: V,
    b: V,
}

impl<V: SdfVector> DeathStar3D<V> {
    /// Sphere of radius `ra` at the origin with a spherical bite of radius `rb` whose
    /// center is displaced by `d` along the x axis.
    ///
    /// Precomputes the intersection circle coordinates:
    ///
    /// ```math
    /// a = \frac{r_a^2 - r_b^2 + d^2}{2d}, \quad b = \sqrt{\max(r_a^2 - a^2,\; 0)}
    /// ```
    #[inline(always)]
    pub fn new(ra: V, rb: V, d: V) -> Self {
        let a = d.mul_adde(d, ra.mul_sube(ra, rb * rb)) / (d * V::TWO);
        let b = ra.mul_sube(ra, a * a).max(V::ZERO).sqrt();
        Self { ra, rb, d, a, b }
    }
}

impl<V: SdfVector> SDF<V, 3> for DeathStar3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let (ra, rb, d, a, b) = (self.ra, self.rb, self.d, self.a, self.b);
        let px = p[0];
        let py = p[1].mul_adde(p[1], p[2] * p[2]).sqrt(); // length(p2.yz)

        let wx = px - a;
        let wy = py - b;
        let d_tip = wx.mul_adde(wx, wy * wy).sqrt();
        let outer = px.mul_adde(px, py * py).sqrt() - ra;
        let cx = px - d;
        let inner = -(cx.mul_adde(cx, py * py).sqrt() - rb);
        let cond = px.mul_sube(b, py * a).cmp_gt(d * (b - py).max(V::ZERO));
        cond.select(d_tip, outer.max(inner))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for DeathStar3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.ra))
    }
}

/// Octahedron with "radius" `s` (exact).
#[derive(Debug, Clone, Copy)]
pub struct Octahedron3D<V: SdfVector> {
    pub s: V,
}

impl<V: SdfVector> SDF<V, 3> for Octahedron3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let s = self.s;
        let px = p[0].abs();
        let py = p[1].abs();
        let pz = p[2].abs();
        let m = px + py + pz - s;

        let cx = (px * (V::ONE + V::TWO)).cmp_lt(m); // 3*px < m
        let cy = (py * (V::ONE + V::TWO)).cmp_lt(m);
        let qx = cx.select(px, cy.select(py, pz));
        let qy = cx.select(py, cy.select(pz, px));
        let qz = cx.select(pz, cy.select(px, py));

        let k = (V::HALF * (qz - qy + s)).clamp(V::ZERO, s);
        let ex = qy - s + k;
        let ez = qz - k;
        let exact = qx.mul_adde(qx, ex.mul_adde(ex, ez * ez)).sqrt();
        let bound = m * V::FRAC_1_SQRT_3;
        (cx | cy | (pz * (V::ONE + V::TWO)).cmp_lt(m)).select(exact, bound)
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Octahedron3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.s))
    }
}

/// Octahedron lower-bound approximation (cheap, not exact).
///
/// Evaluates `$(|x| + |y| + |z| - s) / \sqrt{3}$`, which is the exact SDF of a regular
/// octahedron along its face normals but underestimates elsewhere. Valid for sphere
/// tracing with conservative step sizes; use [`Octahedron3D`] when an exact SDF is needed.
///
/// `s` is the L1 "radius": the surface satisfies `$|x| + |y| + |z| = s$`.
#[derive(Debug, Clone, Copy)]
pub struct OctahedronBound3D<V: SdfVector> {
    pub s: V,
}

impl<V: SdfVector> SDF<V, 3> for OctahedronBound3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        (p[0].abs() + p[1].abs() + p[2].abs() - self.s) * V::FRAC_1_SQRT_3
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for OctahedronBound3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        // grad of (|x|+|y|+|z|-s)/sqrt(3) is sign(p)/sqrt(3), already unit length
        let dist = (p[0].abs() + p[1].abs() + p[2].abs() - self.s) * V::FRAC_1_SQRT_3;
        (dist, p.signum() * V::FRAC_1_SQRT_3)
    }

    #[inline(always)]
    fn normal(&self, p: Vector3<V>) -> Vector3<V> {
        // sign(p)/sqrt(3) - skips the |p|_1 sum and the distance scaling
        p.signum() * V::FRAC_1_SQRT_3
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for OctahedronBound3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.s))
    }
}

/// Triangular prism (lower bound), `h = (height, half_depth)`.
#[derive(Debug, Clone, Copy)]
pub struct TriPrism3D<V: SdfVector> {
    pub h: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 3> for TriPrism3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qx = p[0].abs();
        let qz = p[2].abs();
        let kx = V::SQRT_3 * V::HALF; // 0.866025
        (qz - self.h[1]).max(qx.mul_adde(kx, p[1] * V::HALF).max(-p[1]) - self.h[0] * V::HALF)
    }
}

/// Solid angle (spherical sector) with aperture `c = (sin, cos)`, radius `ra`.
#[derive(Debug, Clone, Copy)]
pub struct SolidAngle3D<V: SdfVector> {
    pub c: Vector2<V>,
    pub ra: V,
}

impl<V: SdfVector> SDF<V, 3> for SolidAngle3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = p[1];
        let (cx, cy) = (self.c[0], self.c[1]);
        let l = qx.mul_adde(qx, qy * qy).sqrt() - self.ra;
        let t = qx.mul_adde(cx, qy * cy).clamp(V::ZERO, self.ra); // dot(q,c) clamped
        let mx = cx.nmul_adde(t, qx);
        let my = cy.nmul_adde(t, qy);
        let m = mx.mul_adde(mx, my * my).sqrt();
        l.max(m.mul_sign(cy.mul_sube(qx, cx * qy)))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for SolidAngle3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.ra))
    }
}

/// Capped torus: an arc of a torus with aperture `sc = (sin, cos)`, radii `ra`/`rb`.
#[derive(Debug, Clone, Copy)]
pub struct CappedTorus3D<V: SdfVector> {
    pub sc: Vector2<V>,
    pub ra: V,
    pub rb: V,
}

impl<V: SdfVector> SDF<V, 3> for CappedTorus3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let px = p[0].abs();
        let py = p[1];
        let (sx, sy) = (self.sc[0], self.sc[1]);
        let dotxy = px.mul_adde(sx, py * sy);
        let lenxy = px.mul_adde(px, py * py).sqrt();
        let k = sy.mul_sube(px, sx * py).cmp_gt(V::ZERO).select(dotxy, lenxy);
        let pp = p.dot(&p);
        self.ra.nmul_adde(k * V::TWO, self.ra.mul_adde(self.ra, pp)).sqrt() - self.rb
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for CappedTorus3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(self.ra + self.rb))
    }
}

/// Vertical capped cone, aperture `c = (sin, cos)` of the side angle, height `h`.
#[derive(Debug, Clone, Copy)]
pub struct Cone3DVert<V: SdfVector> {
    pub c: Vector2<V>,
    pub h: V,
}

impl<V: SdfVector> SDF<V, 3> for Cone3DVert<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qx = self.h * self.c[0] / self.c[1];
        let qy = -self.h;
        let wx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let wy = p[1];

        let t = (wx.mul_adde(qx, wy * qy) / qx.mul_adde(qx, qy * qy)).clamp(V::ZERO, V::ONE);
        let ax = qx.nmul_adde(t, wx);
        let ay = qy.nmul_adde(t, wy);
        let t2 = (wx / qx).clamp(V::ZERO, V::ONE);
        let bx = qx.nmul_adde(t2, wx);
        let by = wy - qy;
        let kk = qy.signum();
        let d = ax.mul_adde(ax, ay * ay).min(bx.mul_adde(bx, by * by));
        let s = (kk * wx.mul_sube(qy, wy * qx)).max(kk * (wy - qy));
        d.sqrt().mul_sign(s)
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Cone3DVert<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let rb = self.h * self.c[0] / self.c[1];
        Bounds::from_corners(Vector3::new([-rb, -self.h, -rb]), Vector3::new([rb, V::ZERO, rb]))
    }
}

/// Infinite cone with aperture `c = (sin, cos)`, apex at the origin opening down.
#[derive(Debug, Clone, Copy)]
pub struct InfiniteCone3D<V: SdfVector> {
    pub c: Vector2<V>,
}

impl<V: SdfVector> SDF<V, 3> for InfiniteCone3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = -p[1];
        let (cx, cy) = (self.c[0], self.c[1]);
        let t = qx.mul_adde(cx, qy * cy).max(V::ZERO);
        let mx = cx.nmul_adde(t, qx);
        let my = cy.nmul_adde(t, qy);
        let d = mx.mul_adde(mx, my * my).sqrt();
        // (q.x*c.y - q.y*c.x < 0)? -1 : 1
        let sgn = qx.mul_sube(cy, qy * cx).select_negative(V::NEG_ONE, V::ONE);
        d * sgn
    }
}

/// Vertical rounded cone from radius `r1` at the base to `r2` at height `h`.
///
/// Stores the precomputed slope `$b = \frac{r_1 - r_2}{h}$` and `$a = \sqrt{1 - b^2}$`. Build
/// with [`new`](Self::new).
#[derive(Debug, Clone, Copy)]
pub struct RoundCone3DVert<V: SdfVector> {
    r1: V,
    r2: V,
    h: V,
    a: V,
    b: V,
}

impl<V: SdfVector> RoundCone3DVert<V> {
    /// Vertical rounded cone from base radius `r1` (at `y = 0`) to apex radius `r2`
    /// (at `y = h`). Precomputes the lateral taper `$b = \frac{r_1 - r_2}{h}$` (radial
    /// slope) and `$a = \sqrt{1 - b^2}$` (axial component of the flank unit normal), reused
    /// to classify the nearest region in every `eval` call.
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

impl<V: SdfVector> SDF<V, 3> for RoundCone3DVert<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let qx = p[0].mul_adde(p[0], p[2] * p[2]).sqrt();
        let qy = p[1];
        let (a, b) = (self.a, self.b);
        let k = b.nmul_adde(qx, a * qy); // dot(q, (-b, a))

        let d_low = qx.mul_adde(qx, qy * qy).sqrt() - self.r1;
        let qyh = qy - self.h;
        let d_high = qx.mul_adde(qx, qyh * qyh).sqrt() - self.r2;
        let d_mid = a.mul_adde(qx, b * qy) - self.r1;
        k.select_negative(d_low, k.cmp_gt(a * self.h).select(d_high, d_mid))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for RoundCone3DVert<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        let w = self.r1.max(self.r2);
        Bounds::from_corners(Vector3::new([-w, -self.r1, -w]), Vector3::new([w, self.h + self.r2, w]))
    }
}

// ===========================================================================
// 3D primitives, batch 2 (pyramid, rhombus, vesica segment, cone, triangles)
// ===========================================================================

/// Square pyramid of height `h` over a unit base (`$[-0.5, 0.5]^2$`).
#[derive(Debug, Clone, Copy)]
pub struct Pyramid3D<V: SdfVector> {
    pub h: V,
}

impl<V: SdfVector> SDF<V, 3> for Pyramid3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let m2 = self.h.mul_adde(self.h, frac::<V, 1, 4>()); // h^2 + 0.25
        let ax = p[0].abs();
        let az = p[2].abs();
        // p.xz = (p.z > p.x)? p.zx : p.xz   then  -= 0.5
        let swap = az.cmp_gt(ax);
        let px = swap.select(az, ax) - V::HALF;
        let pz = swap.select(ax, az) - V::HALF;
        let py = p[1];

        let qx = pz;
        let qy = self.h.mul_sube(py, V::HALF * px); // h*p.y - 0.5*p.x
        let qz = self.h.mul_adde(px, V::HALF * py); // h*p.x + 0.5*p.y

        let s = (-qx).max(V::ZERO);
        let t = ((qy - V::HALF * pz) / (m2 + frac::<V, 1, 4>())).clamp(V::ZERO, V::ONE);
        let a = m2.mul_adde((qx + s) * (qx + s), qy * qy);
        let bym = m2.nmul_adde(t, qy);
        let b = m2.mul_adde((qx + V::HALF * t) * (qx + V::HALF * t), bym * bym);
        let inside = qy.min(qx.nmul_adde(m2, -(qy * V::HALF))).cmp_gt(V::ZERO); // min(q.y, -q.x*m2 - q.y*0.5) > 0
        let d2 = inside.select(V::ZERO, a.min(b));
        ((qz.mul_adde(qz, d2)) / m2).sqrt().mul_sign(qz.max(-py))
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Pyramid3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::from_corners(
            Vector3::new([-V::HALF, V::ZERO, -V::HALF]),
            Vector3::new([V::HALF, self.h, V::HALF]),
        )
    }
}

/// 3D rhombus (a flattened octahedron): half-diagonals `la`/`lb`, half-height
/// `h`, corner radius `ra`.
#[derive(Debug, Clone, Copy)]
pub struct Rhombus3D<V: SdfVector> {
    pub la: V,
    pub lb: V,
    pub h: V,
    pub ra: V,
}

impl<V: SdfVector> SDF<V, 3> for Rhombus3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let px = p[0].abs();
        let py = p[1].abs();
        let pz = p[2].abs();
        let (la, lb) = (self.la, self.lb);
        // (la*p.x - lb*p.z + lb*lb) / (la*la + lb*lb)
        let f = (la.mul_sube(px, lb.mul_sube(pz, lb * lb)) / la.mul_adde(la, lb * lb)).clamp(V::ZERO, V::ONE);
        // w = p.xz - (la, lb)*(f, 1-f)
        let wx = la.nmul_adde(f, px);
        let wz = lb.nmul_adde(V::ONE - f, pz);
        let qx = wx.mul_adde(wx, wz * wz).sqrt().mul_sign(wx) - self.ra;
        let qy = py - self.h;
        qx.max(qy).min(V::ZERO) + Vector2::new([qx.max(V::ZERO), qy.max(V::ZERO)]).l2_norm()
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for Rhombus3D<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::new([self.la + self.ra, self.h, self.lb + self.ra]))
    }
}

/// Vesica lens swept along the segment `a -> b`, with width `w`.
#[derive(Debug, Clone, Copy)]
pub struct VesicaSegment3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub w: V,
}

impl<V: SdfVector> SDF<V, 3> for VesicaSegment3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let c = (self.a + self.b) * V::HALF;
        let ba = self.b - self.a;
        let l = ba.l2_norm();
        let v = ba / l;
        let pc = p - c;
        let y = pc.dot(&v);
        let radial = v.nmul_adde(y, pc); // pc - v*y
        let qx = radial.l2_norm();
        let qy = y.abs();

        let r = l * V::HALF;
        let d = (r.mul_sube(r, self.w * self.w)) * V::HALF / self.w;
        let cond = (r * qx).cmp_lt(d * (qy - r));
        let hx = cond.select(V::ZERO, -d);
        let hy = cond.select(r, V::ZERO);
        let hz = cond.select(V::ZERO, d + self.w);
        let ex = qx - hx;
        let ey = qy - hy;
        ex.mul_adde(ex, ey * ey).sqrt() - hz
    }
}

/// Capped cone between `a` (radius `ra`) and `b` (radius `rb`).
#[derive(Debug, Clone, Copy)]
pub struct CappedConeAB3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub ra: V,
    pub rb: V,
}

impl<V: SdfVector> SDF<V, 3> for CappedConeAB3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let rba = self.rb - self.ra;
        let ba = self.b - self.a;
        let pa = p - self.a;
        let baba = ba.dot(&ba);
        let papa = pa.dot(&pa);
        let paba = pa.dot(&ba) / baba;
        let x = (paba * paba).nmul_adde(baba, papa).max(V::ZERO).sqrt();

        let cap_r = paba.cmp_lt(V::HALF).select(self.ra, self.rb);
        let cax = (x - cap_r).max(V::ZERO);
        let cay = (paba - V::HALF).abs() - V::HALF;
        let k = rba.mul_adde(rba, baba);
        let f = (rba.mul_adde(x - self.ra, paba * baba) / k).clamp(V::ZERO, V::ONE);
        let cbx = rba.nmul_adde(f, x - self.ra); // (x-ra) - f*rba
        let cby = paba - f;
        let s = (cbx.cmp_lt(V::ZERO) & cay.cmp_lt(V::ZERO)).select(V::NEG_ONE, V::ONE);
        let da = cax.mul_adde(cax, cay * cay * baba);
        let db = cbx.mul_adde(cbx, cby * cby * baba);
        s * da.min(db).sqrt()
    }
}

/// Unsigned distance to the (open) triangle `a`, `b`, `c`.
#[derive(Debug, Clone, Copy)]
pub struct UdTriangle3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub c: Vector3<V>,
}

#[inline(always)]
fn edge_dist2<V: SdfVector>(e: Vector3<V>, pv: Vector3<V>) -> V {
    let q = e * (e.dot(&pv) / e.dot(&e)).clamp(V::ZERO, V::ONE) - pv;
    q.dot(&q)
}

/// The vector from the closest point on edge `e` (rooted at `pv`'s origin) to the
/// query, plus its squared length. `w` is the gradient direction (unnormalized).
#[inline(always)]
fn edge_closest<V: SdfVector>(e: Vector3<V>, pv: Vector3<V>) -> (Vector3<V>, V) {
    let w = pv - e * (e.dot(&pv) / e.dot(&e)).clamp(V::ZERO, V::ONE);
    (w, w.dot(&w))
}

impl<V: SdfVector> SDF<V, 3> for UdTriangle3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let cb = self.c - self.b;
        let pb = p - self.b;
        let ac = self.a - self.c;
        let pc = p - self.c;
        let nor = ba.cross(ac);

        let inside =
            ba.cross(nor).dot(&pa).signum() + cb.cross(nor).dot(&pb).signum() + ac.cross(nor).dot(&pc).signum();
        let edges = edge_dist2(ba, pa).min(edge_dist2(cb, pb)).min(edge_dist2(ac, pc));
        let face = nor.dot(&pa) * nor.dot(&pa) / nor.dot(&nor);
        inside.cmp_lt(V::TWO).select(edges, face).sqrt()
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for UdTriangle3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let cb = self.c - self.b;
        let pb = p - self.b;
        let ac = self.a - self.c;
        let pc = p - self.c;
        let nor = ba.cross(ac);

        let inside =
            ba.cross(nor).dot(&pa).signum() + cb.cross(nor).dot(&pb).signum() + ac.cross(nor).dot(&pc).signum();

        // nearest of the three edges (gradient = unit vector to the query)
        let (w1, e1) = edge_closest(ba, pa);
        let (w2, e2) = edge_closest(cb, pb);
        let (w3, e3) = edge_closest(ac, pc);
        let lt12 = e1.cmp_lt(e2);
        let (we, ee) = (lt12.select(w1, w2), e1.min(e2));
        let lt = ee.cmp_lt(e3);
        let we = lt.select(we, w3);
        let edge_dist = ee.min(e3).sqrt();
        let edge_grad = unit_or_zero(we, edge_dist);

        // face: signed projection onto the plane normal
        let npa = nor.dot(&pa);
        let nlen = nor.l2_norm();
        let face_dist = npa.abs() / nlen;
        let face_grad = nor * (npa.signum() / nlen);

        let pick_edge = inside.cmp_lt(V::TWO);
        (
            pick_edge.select(edge_dist, face_dist),
            pick_edge.select(edge_grad, face_grad),
        )
    }
}

/// Unsigned distance to the (open) planar quad `a`, `b`, `c`, `d`.
#[derive(Debug, Clone, Copy)]
pub struct UdQuad3D<V: SdfVector> {
    pub a: Vector3<V>,
    pub b: Vector3<V>,
    pub c: Vector3<V>,
    pub d: Vector3<V>,
}

impl<V: SdfVector> SDF<V, 3> for UdQuad3D<V> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let cb = self.c - self.b;
        let pb = p - self.b;
        let dc = self.d - self.c;
        let pc = p - self.c;
        let ad = self.a - self.d;
        let pd = p - self.d;
        let nor = ba.cross(ad);

        let inside = ba.cross(nor).dot(&pa).signum()
            + cb.cross(nor).dot(&pb).signum()
            + dc.cross(nor).dot(&pc).signum()
            + ad.cross(nor).dot(&pd).signum();
        let edges = edge_dist2(ba, pa)
            .min(edge_dist2(cb, pb))
            .min(edge_dist2(dc, pc))
            .min(edge_dist2(ad, pd));
        let face = nor.dot(&pa) * nor.dot(&pa) / nor.dot(&nor);
        inside.cmp_lt(V::ONE + V::TWO).select(edges, face).sqrt()
    }
}

impl<V: SdfVector> GradientSdf<V, 3> for UdQuad3D<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector3<V>) -> (V, Vector3<V>) {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let cb = self.c - self.b;
        let pb = p - self.b;
        let dc = self.d - self.c;
        let pc = p - self.c;
        let ad = self.a - self.d;
        let pd = p - self.d;
        let nor = ba.cross(ad);

        let inside = ba.cross(nor).dot(&pa).signum()
            + cb.cross(nor).dot(&pb).signum()
            + dc.cross(nor).dot(&pc).signum()
            + ad.cross(nor).dot(&pd).signum();

        // nearest of the four edges
        let (w1, e1) = edge_closest(ba, pa);
        let (w2, e2) = edge_closest(cb, pb);
        let (w3, e3) = edge_closest(dc, pc);
        let (w4, e4) = edge_closest(ad, pd);
        let (wa, ea) = (e1.cmp_lt(e2).select(w1, w2), e1.min(e2));
        let (wb, eb) = (e3.cmp_lt(e4).select(w3, w4), e3.min(e4));
        let we = ea.cmp_lt(eb).select(wa, wb);
        let edge_dist = ea.min(eb).sqrt();
        let edge_grad = unit_or_zero(we, edge_dist);

        let npa = nor.dot(&pa);
        let nlen = nor.l2_norm();
        let face_dist = npa.abs() / nlen;
        let face_grad = nor * (npa.signum() / nlen);

        let pick_edge = inside.cmp_lt(V::ONE + V::TWO);
        (
            pick_edge.select(edge_dist, face_dist),
            pick_edge.select(edge_grad, face_grad),
        )
    }
}

// ---------------------------------------------------------------------------
// Angle-based constructors for the sin/cos-parameterized shapes
// ---------------------------------------------------------------------------

impl<V: SdfVector> CappedTorus3D<V> {
    /// Build from the half-aperture `angle` (radians), computing `(sin, cos)`
    /// with precision policy `P`.
    #[inline(always)]
    pub fn from_aperture<P: Policy>(angle: V, ra: V, rb: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (s, c) = angle.sin_cos_p::<P>();
        Self {
            sc: Vector2::new([s, c]),
            ra,
            rb,
        }
    }
}

impl<V: SdfVector> Cone3DVert<V> {
    /// Build from the side-angle `angle` (radians) and height `h`.
    #[inline(always)]
    pub fn from_angle<P: Policy>(angle: V, h: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (s, c) = angle.sin_cos_p::<P>();
        Self {
            c: Vector2::new([s, c]),
            h,
        }
    }
}

impl<V: SdfVector> InfiniteCone3D<V> {
    /// Build from the side-angle `angle` (radians).
    #[inline(always)]
    pub fn from_angle<P: Policy>(angle: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (s, c) = angle.sin_cos_p::<P>();
        Self {
            c: Vector2::new([s, c]),
        }
    }
}

impl<V: SdfVector> SolidAngle3D<V> {
    /// Build from the aperture `angle` (radians) and radius `ra`.
    #[inline(always)]
    pub fn from_angle<P: Policy>(angle: V, ra: V) -> Self
    where
        V: TranscendentalMathWithPolicy,
    {
        let (s, c) = angle.sin_cos_p::<P>();
        Self {
            c: Vector2::new([s, c]),
            ra,
        }
    }
}
