//! Fractal distance estimators (DE) for raymarching, after
//! <https://iquilezles.org/articles/distancefractals>,
//! <https://iquilezles.org/articles/juliasets3d> and
//! <https://iquilezles.org/articles/mandelbulb>.
//!
//! These are *unsigned* exterior distance estimators built from the
//! Hubbard-Douady potential `$G$`: tracking the orbit `$z_n$` and its
//! derivative magnitude `$|z'_n|$`, the distance to the set boundary is
//! approximated by
//!
//! ```math
//! d \approx \frac{|z_n|}{|z'_n|}\,\tfrac12 \log|z_n|^2 = \frac{|z_n|\,\log|z_n|}{|z'_n|}.
//! ```
//!
//! The estimate is `$\ge 0$` everywhere, reaching `$0$` on (and inside) the set.
//! Lanes escape the iteration at different counts, so the loop freezes escaped
//! lanes with a mask and stops early once none remain active.

use core::marker::PhantomData;

use thermite::math::RealMathWithPolicy;
use thermite::math::policy::{DefaultPolicy, Policy};
use thermite::prelude::*;

use thermite_geometry::prim::{Bounds, Vector, Vector2, Vector3, vector::VectorOps as _};

use crate::consts::{cint, frac};
use crate::ops::FiniteDiff;
use crate::{BoundedSdf, SDF, SdfVector, unit_or_zero};

/// Squared bail-out radius: once `$|z|^2$` exceeds this a lane has escaped.
#[inline(always)]
fn escape<V: SdfVector>() -> V {
    cint::<V, 256>()
}

/// `mod(x, 2) - 1` - folds a coordinate into a single `[-1, 1]` cell, for the
/// Menger/Sierpinski domain repetition.
#[inline(always)]
fn fold_cell<V: SdfVector>(x: V) -> V {
    (x * V::HALF).floor().nmul_adde(V::TWO, x) - V::ONE // x - 2*floor(x/2) - 1
}

/// Per-lane data captured while iterating a fractal, for shading. Fractals are
/// usually colored not by distance but by the *orbit* `$z_0, z_1, \dots$`: how
/// fast it escaped (the iteration count) and how close it passed to various
/// shapes (orbit traps). [`eval_orbit`](FractalSdf::eval_orbit) returns this
/// alongside the distance.
#[derive(Debug, Clone, Copy)]
pub struct FractalOrbit<V: SdfVector, const N: usize> {
    /// Number of iterations the lane survived before escaping (the max for
    /// interior lanes). A float for smooth shading; combine with
    /// [`final_m2`](Self::final_m2) for a smooth (fractional) iteration count.
    pub count: V,
    /// `true` where the lane escaped (exterior of the set), `false` where it
    /// stayed bounded (interior).
    pub escaped: V::Mask,
    /// `$|z_n|^2$` at the final iteration.
    pub final_m2: V,
    /// Point orbit trap: the minimum `$|z_i|$` over the orbit - distance to the
    /// origin. IQ uses this as an ambient-occlusion-like multiplier.
    pub trap_point: V,
    /// Plane orbit traps: per-axis minimum of `$|z_i.k|$` over the orbit -
    /// distance to each coordinate hyperplane. IQ mixes these into color channels.
    pub trap_planes: Vector<V, N>,
}

/// A fractal SDF that, besides the distance estimate, can report the [orbit
/// data](FractalOrbit) needed to shade it (escape-time count, orbit traps).
///
/// Call [`eval`](SDF::eval) on the fast raymarching path (distance only) and
/// [`eval_orbit`](Self::eval_orbit) once at the surface hit to get coloring data;
/// the trap accumulation is compiled out of the former.
pub trait FractalSdf<V: SdfVector, const N: usize>: SDF<V, N> {
    /// Distance estimate plus the orbit data for shading.
    fn eval_orbit(&self, p: Vector<V, N>) -> (V, FractalOrbit<V, N>);
}

/// A [`FractalSdf`] that also yields the surface normal - distance, gradient, and
/// orbit data in one go, for shading a fractal hit. 3D only.
///
/// The normal is numerical (fractal surfaces have no analytic gradient), so the
/// only implementor is [`FiniteDiff`]: its 4-tap tetrahedron differences the
/// distance, and a single central tap supplies the (exact) distance and the
/// orbit. The orbit is therefore evaluated once, not per gradient tap.
pub trait FractalGradientSdf<V: SdfVector>: FractalSdf<V, 3> {
    /// Distance, unit surface normal, and orbit data.
    fn eval_orbit_grad(&self, p: Vector3<V>) -> (V, Vector3<V>, FractalOrbit<V, 3>);

    /// Surface normal + orbit *without* the distance, for the march -> shade path
    /// where you already hold the distance from the surface hit.
    ///
    /// Defaults to dropping the distance from [`eval_orbit_grad`](Self::eval_orbit_grad);
    /// [`FiniteDiff`] overrides it to the 4-tap minimum (the orbit rides the first
    /// gradient tap instead of a separate central tap).
    #[inline(always)]
    fn normal_orbit(&self, p: Vector3<V>) -> (Vector3<V>, FractalOrbit<V, 3>) {
        let (_, n, o) = self.eval_orbit_grad(p);
        (n, o)
    }
}

/// Quaternionic Julia set `$q_{n+1} = q^2 + c$` (Hart/Crane), rendered as a 3D
/// slice (`w` fixes the 4th coordinate). Because quaternion multiplication is
/// norm-multiplicative, the derivative reduces to a scalar:
/// `$|z'_{n+1}|^2 = 4|z_n|^2 |z'_n|^2$`, tracked exactly with no Jacobian.
///
/// With `c = 0` the set is the unit sphere - a handy exact reference.
#[derive(Debug, Clone, Copy)]
pub struct QuaternionJulia3D<V: SdfVector, P: Policy = DefaultPolicy> {
    /// The quaternion constant `(real, i, j, k)`.
    pub c: [V; 4],
    /// 4th-coordinate of the 3D slice (the `xyz` come from the sample point).
    pub w: V,
    /// Iteration budget.
    pub iterations: u32,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> QuaternionJulia3D<V, P> {
    /// Builds with the slice `w = 0` and the given iteration budget.
    #[inline(always)]
    pub fn new(c: [V; 4], iterations: u32) -> Self {
        Self {
            c,
            w: V::ZERO,
            iterations,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> QuaternionJulia3D<V, P> {
    /// Shared iteration. `TRAP` gates orbit-trap accumulation (compiled out for
    /// the distance-only [`eval`](SDF::eval)).
    #[inline(always)]
    fn iterate<const TRAP: bool>(&self, p: Vector3<V>) -> (V, FractalOrbit<V, 3>) {
        let esc = escape::<V>();
        // quaternion z = (x, y, z, w), real part x (matching IQ's qsqr slicing)
        let (mut x, mut y, mut z, mut w) = (p[0], p[1], p[2], self.w);
        let mut m2 = x.mul_adde(x, y.mul_adde(y, z.mul_adde(z, w * w)));
        let mut dz2 = V::ONE; // |z'|^2
        let mut active = m2.cmp_lt(esc);
        let mut count = V::ZERO;
        // orbit traps, seeded from z0 (3D part)
        let mut tp = x.mul_adde(x, y.mul_adde(y, z * z));
        let (mut tx, mut ty, mut tz) = (x.abs(), y.abs(), z.abs());
        let four = cint::<V, 4>();
        let mut i = 0;
        while i < self.iterations {
            if !thermite::likely(active.any()) {
                break;
            }
            let x2 = V::TWO * x;
            let isq = y.mul_adde(y, z.mul_adde(z, w * w)); // |imag(z_n)|^2 (old)
            // Update each component with a masked FMA that keeps `self` (the old
            // value) where the lane has escaped - no separate selects. q^2 + c is
            // (x^2 - isq, 2xy, 2xz, 2xw) + c, written so the kept variable is self:
            //   |z'|^2 *= 4|z|^2 ;  y,z,w := comp*2x + c ;  x := x*x + (c0 - isq)
            dz2 = dz2.mul_adde_c(active, four * m2, V::ZERO);
            y = y.mul_adde_c(active, x2, self.c[1]);
            z = z.mul_adde_c(active, x2, self.c[2]);
            w = w.mul_adde_c(active, x2, self.c[3]);
            x = x.mul_adde_c(active, x, self.c[0] - isq);
            m2 = x.mul_adde(x, y.mul_adde(y, z.mul_adde(z, w * w))); // old where frozen
            count = count.add_c(active, V::ONE);
            if const { TRAP } {
                tp = tp.min(x.mul_adde(x, y.mul_adde(y, z * z)));
                tx = tx.min(x.abs());
                ty = ty.min(y.abs());
                tz = tz.min(z.abs());
            }
            active &= m2.cmp_lt(esc);
            i += 1;
        }
        // d = sqrt(m2/dz2) * 0.5*log(m2) * 0.5  (the trailing 0.5 keeps it an
        // upper bound, per the article's part 3). Lanes still active never
        // escaped (they are inside the set) -> distance 0.
        let d = (m2 / dz2).sqrt() * (m2.ln() * frac::<V, 1, 4>());
        let orbit = FractalOrbit {
            count,
            escaped: !active,
            final_m2: m2,
            trap_point: tp.sqrt(),
            trap_planes: Vector3::new([tx, ty, tz]),
        };
        (active.select(V::ZERO, d), orbit)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 3> for QuaternionJulia3D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        self.iterate::<false>(p).0
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> FractalSdf<V, 3> for QuaternionJulia3D<V, P> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector3<V>) -> (V, FractalOrbit<V, 3>) {
        self.iterate::<true>(p)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> BoundedSdf<V, 3> for QuaternionJulia3D<V, P> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // the filled set lives well within the escape radius (|q| <= 2)
        Bounds::symmetric(Vector3::splat(V::TWO))
    }
}

/// The classic power-8 Mandelbulb (`<https://iquilezles.org/articles/mandelbulb>`).
///
/// For each point `p` the orbit `$w_{n+1} = \mathrm{bulb}_8(w_n) + p$` is iterated
/// from `$w_0 = p$`, with the scalar derivative `$dr_{n+1} = 8\,r_n^{7}\,dr_n + 1$`
/// (`$r = |w|$`). The bulb cubes-and-rotates in spherical coordinates (polar form;
/// trig, but exact and GPU-friendly). The DE is `$0.5\,r\,\log r / dr$`.
///
/// This is a geometric (algebraically "incorrect") construction, so the DE is an
/// approximation, not the exact metric.
#[derive(Debug, Clone, Copy)]
pub struct Mandelbulb3D<V: SdfVector, P: Policy = DefaultPolicy> {
    /// Iteration budget (power-8 escapes fast; ~8-16 suffices).
    pub iterations: u32,
    _policy: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy> Mandelbulb3D<V, P> {
    /// Builds with the given iteration budget.
    #[inline(always)]
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> Mandelbulb3D<V, P> {
    /// Shared iteration. `TRAP` gates orbit-trap accumulation.
    #[inline(always)]
    fn iterate<const TRAP: bool>(&self, p: Vector3<V>) -> (V, FractalOrbit<V, 3>) {
        let esc = escape::<V>();
        let (mut wx, mut wy, mut wz) = (p[0], p[1], p[2]);
        let mut m2 = wx.mul_adde(wx, wy.mul_adde(wy, wz * wz));
        let mut dr = V::ONE;
        let mut active = m2.cmp_lt(esc);
        let mut count = V::ZERO;
        let mut tp = m2; // |w0|^2
        let (mut tx, mut ty, mut tz) = (wx.abs(), wy.abs(), wz.abs());
        let eight = cint::<V, 8>();
        let mut i = 0;
        while i < self.iterations {
            if !thermite::likely(active.any()) {
                break;
            }
            let r = m2.sqrt();
            let r2 = m2; // r^2
            let r4 = r2 * r2;
            let r7 = r4 * r2 * r;
            let r8 = r4 * r4;
            // polar: wo = acos(wy/r), wi = atan2(wx, wz); scale angles by 8, r by ^8.
            // Guard r == 0 (the origin) and clamp the acos argument against fp drift;
            // there r^8 == 0 anyway, so the exact angle is irrelevant.
            let rg = r.cmp_gt(V::ZERO).select(r, V::ONE);
            let wo = (wy / rg).clamp(V::NEG_ONE, V::ONE).acos_p::<P>();
            let wi = wx.atan2_p::<P>(wz);
            let (so, co) = (wo * eight).sin_cos_p::<P>();
            let (si, ci) = (wi * eight).sin_cos_p::<P>();
            let rs = r8 * so;
            // Masked FMAs that keep the old value (`src`) where the lane escaped.
            //   dr := 8 r^7 dr + 1 ;  w := r^8 (sin8o sin8i, cos8o, sin8o cos8i) + p
            dr = dr.mul_adde_c(active, eight * r7, V::ONE);
            wx = rs.mul_adde_m(wx, active, si, p[0]);
            wy = r8.mul_adde_m(wy, active, co, p[1]);
            wz = rs.mul_adde_m(wz, active, ci, p[2]);
            m2 = wx.mul_adde(wx, wy.mul_adde(wy, wz * wz)); // old where frozen
            count = count.add_c(active, V::ONE);
            if const { TRAP } {
                tp = tp.min(m2);
                tx = tx.min(wx.abs());
                ty = ty.min(wy.abs());
                tz = tz.min(wz.abs());
            }
            active &= m2.cmp_lt(esc);
            i += 1;
        }
        // d = 0.5 * r * log(r) / dr  (= 0.25 * sqrt(m2) * log(m2) / dr)
        let r = m2.sqrt();
        let d = r * m2.ln() * frac::<V, 1, 4>() / dr;
        let orbit = FractalOrbit {
            count,
            escaped: !active,
            final_m2: m2,
            trap_point: tp.sqrt(),
            trap_planes: Vector3::new([tx, ty, tz]),
        };
        (active.select(V::ZERO, d), orbit)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 3> for Mandelbulb3D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        self.iterate::<false>(p).0
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> FractalSdf<V, 3> for Mandelbulb3D<V, P> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector3<V>) -> (V, FractalOrbit<V, 3>) {
        self.iterate::<true>(p)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> BoundedSdf<V, 3> for Mandelbulb3D<V, P> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        // the power-8 bulb is contained in a sphere of radius ~1.2
        Bounds::symmetric(Vector3::splat(frac::<V, 5, 4>()))
    }
}

/// 2D complex Julia set `$z_{n+1} = z^2 + c$` (`$z, c \in \mathbb{C}$`). The
/// classic fractal; the returned distance is the (unsigned) exterior DE, `~0` on
/// the set and growing outside, for crisp anti-aliased rasterization. With
/// `c = 0` the set is the unit circle. Like the quaternion case the derivative is
/// a scalar (`$|z'|^2 \mathrel{*}= 4|z|^2$`).
#[derive(Debug, Clone, Copy)]
pub struct Julia2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub c: Vector2<V>,
    pub iterations: u32,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, P: Policy> Julia2D<V, P> {
    /// Builds with the given constant and iteration budget.
    #[inline(always)]
    pub fn new(c: Vector2<V>, iterations: u32) -> Self {
        Self {
            c,
            iterations,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> Julia2D<V, P> {
    #[inline(always)]
    fn iterate<const TRAP: bool>(&self, p: Vector2<V>) -> (V, FractalOrbit<V, 2>) {
        let esc = escape::<V>();
        let (mut x, mut y) = (p[0], p[1]);
        let mut m2 = x.mul_adde(x, y * y);
        let mut dz2 = V::ONE;
        let mut active = m2.cmp_lt(esc);
        let mut count = V::ZERO;
        let mut tp = m2;
        let (mut tx, mut ty) = (x.abs(), y.abs());
        let four = cint::<V, 4>();
        let mut i = 0;
        while i < self.iterations {
            if !thermite::likely(active.any()) {
                break;
            }
            let x2 = V::TWO * x;
            let ysq = y * y; // old
            dz2 = dz2.mul_adde_c(active, four * m2, V::ZERO);
            y = y.mul_adde_c(active, x2, self.c[1]); // 2xy + c.y, keeps old y
            x = x.mul_adde_c(active, x, self.c[0] - ysq); // x^2 + (c.x - y^2), keeps old x
            m2 = x.mul_adde(x, y * y);
            count = count.add_c(active, V::ONE);
            if const { TRAP } {
                tp = tp.min(m2);
                tx = tx.min(x.abs());
                ty = ty.min(y.abs());
            }
            active &= m2.cmp_lt(esc);
            i += 1;
        }
        let d = (m2 / dz2).sqrt() * (m2.ln() * V::HALF);
        let orbit = FractalOrbit {
            count,
            escaped: !active,
            final_m2: m2,
            trap_point: tp.sqrt(),
            trap_planes: Vector2::new([tx, ty]),
        };
        (active.select(V::ZERO, d), orbit)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for Julia2D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        self.iterate::<false>(p).0
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> FractalSdf<V, 2> for Julia2D<V, P> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector2<V>) -> (V, FractalOrbit<V, 2>) {
        self.iterate::<true>(p)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> BoundedSdf<V, 2> for Julia2D<V, P> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::splat(V::TWO))
    }
}

/// 2D complex Mandelbrot set: for each point `c = p`, iterate `$z_{n+1} = z^2 + c$`
/// from `$z_0 = 0$`. Unlike Julia, the Mandelbrot derivative carries the `+1`
/// term (`$z'_{n+1} = 2 z_n z'_n + 1$`), so the full complex `$z'$` is tracked.
#[derive(Debug, Clone, Copy)]
pub struct Mandelbrot2D<V: SdfVector, P: Policy = DefaultPolicy> {
    pub iterations: u32,
    _policy: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy> Mandelbrot2D<V, P> {
    /// Builds with the given iteration budget.
    #[inline(always)]
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> Mandelbrot2D<V, P> {
    #[inline(always)]
    fn iterate<const TRAP: bool>(&self, p: Vector2<V>) -> (V, FractalOrbit<V, 2>) {
        let esc = escape::<V>();
        let (cx, cy) = (p[0], p[1]);
        let (mut zx, mut zy) = (V::ZERO, V::ZERO);
        let (mut dx, mut dy) = (V::ZERO, V::ZERO); // z' (full complex)
        let mut m2 = V::ZERO;
        let mut active = m2.cmp_lt(esc);
        let mut count = V::ZERO;
        let mut tp = V::INFINITY;
        let (mut tx, mut ty) = (V::INFINITY, V::INFINITY);
        let mut i = 0;
        while i < self.iterations {
            if !thermite::likely(active.any()) {
                break;
            }
            let zysq = zy * zy;
            let t = zx.mul_sube(dx, zy * dy); // zx*dx - zy*dy
            let t2 = zx.mul_adde(dy, zy * dx); // zx*dy + zy*dx
            let z2x = V::TWO * zx;
            // z'_{n+1} = 2(z z') + 1 ; z_{n+1} = z^2 + c. Masked merges keep old.
            dx = t.mul_adde_m(dx, active, V::TWO, V::ONE); // 2t + 1
            dy = t2.mul_adde_m(dy, active, V::TWO, V::ZERO); // 2 t2
            zy = z2x.mul_adde_m(zy, active, zy, cy); // 2 zx zy + cy
            zx = zx.mul_adde_c(active, zx, cx - zysq); // zx^2 + (cx - zy^2)
            m2 = zx.mul_adde(zx, zy * zy);
            count = count.add_c(active, V::ONE);
            if const { TRAP } {
                tp = tp.min(m2);
                tx = tx.min(zx.abs());
                ty = ty.min(zy.abs());
            }
            active &= m2.cmp_lt(esc);
            i += 1;
        }
        // |z'|^2 = dx^2 + dy^2
        let dz2 = dx.mul_adde(dx, dy * dy);
        let d = (m2 / dz2).sqrt() * (m2.ln() * V::HALF);
        let orbit = FractalOrbit {
            count,
            escaped: !active,
            final_m2: m2,
            trap_point: tp.sqrt(),
            trap_planes: Vector2::new([tx, ty]),
        };
        (active.select(V::ZERO, d), orbit)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> SDF<V, 2> for Mandelbrot2D<V, P> {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        self.iterate::<false>(p).0
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> FractalSdf<V, 2> for Mandelbrot2D<V, P> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector2<V>) -> (V, FractalOrbit<V, 2>) {
        self.iterate::<true>(p)
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy> BoundedSdf<V, 2> for Mandelbrot2D<V, P> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        // the Mandelbrot set lies within [-2.5, 1] x [-1.25, 1.25]
        Bounds::from_corners(
            Vector2::new([frac::<V, -5, 2>(), -frac::<V, 5, 4>()]),
            Vector2::new([V::ONE, frac::<V, 5, 4>()]),
        )
    }
}

// ---------------------------------------------------------------------------
// IFS / carving fractals (exact SDFs, not escape-time DEs)
// ---------------------------------------------------------------------------

/// The Menger sponge (`<https://iquilezles.org/articles/menger>`): a unit cube
/// with an axis-aligned cross iteratively subtracted at thirds. Unlike the Julia
/// / Mandelbulb DEs this is an (essentially exact) SDF built from `box`/`cross`
/// CSG, so it raymarches crisply.
///
/// Each level folds the domain into a `[-1, 1]` cell (`mod(p s, 2) - 1`), forms
/// `$r = |1 - 3|a||$`, and carves the cross `$\min_i \max(r_j, r_k)$` scaled by
/// `$1/s$`.
#[derive(Debug, Clone, Copy)]
pub struct MengerSponge {
    /// Recursion depth (3-4 is plenty; cost is linear in this).
    pub iterations: u32,
}

impl<V: SdfVector> SDF<V, 3> for MengerSponge {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        // outer unit cube
        let w = Vector3::new([p[0].abs() - V::ONE, p[1].abs() - V::ONE, p[2].abs() - V::ONE]);
        let g = w[0].max(w[1]).max(w[2]);
        let q = Vector3::new([w[0].max(V::ZERO), w[1].max(V::ZERO), w[2].max(V::ZERO)]);
        let mut d = q.l2_norm() + g.min(V::ZERO);

        let three = cint::<V, 3>();
        let mut s = V::ONE;
        let mut m = 0;
        while m < self.iterations {
            let ax = fold_cell(p[0] * s).abs();
            let ay = fold_cell(p[1] * s).abs();
            let az = fold_cell(p[2] * s).abs();
            s = s * three;
            // r = |1 - 3|a||
            let rx = three.nmul_adde(ax, V::ONE).abs();
            let ry = three.nmul_adde(ay, V::ONE).abs();
            let rz = three.nmul_adde(az, V::ONE).abs();
            // cross = min(max(rx,ry), max(ry,rz), max(rz,rx)) - 1, scaled
            let cross = rx.max(ry).min(ry.max(rz)).min(rz.max(rx));
            d = d.max((cross - V::ONE) / s);
            m += 1;
        }
        d
    }
}

impl<V: SdfVector> BoundedSdf<V, 3> for MengerSponge {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 3> {
        Bounds::symmetric(Vector3::splat(V::ONE))
    }
}

/// The Sierpinski carpet - the 2D analogue of the [`MengerSponge`]: a unit square
/// with its centre cell iteratively removed. The carve is `$\min(r_x, r_y)$`
/// (both coordinates central) rather than the sponge's pairwise cross.
#[derive(Debug, Clone, Copy)]
pub struct SierpinskiCarpet {
    /// Recursion depth.
    pub iterations: u32,
}

impl<V: SdfVector> SDF<V, 2> for SierpinskiCarpet {
    #[inline(always)]
    fn eval(&self, p: Vector2<V>) -> V {
        let w = Vector2::new([p[0].abs() - V::ONE, p[1].abs() - V::ONE]);
        let g = w[0].max(w[1]);
        let q = Vector2::new([w[0].max(V::ZERO), w[1].max(V::ZERO)]);
        let mut d = q.l2_norm() + g.min(V::ZERO);

        let three = cint::<V, 3>();
        let mut s = V::ONE;
        let mut m = 0;
        while m < self.iterations {
            let ax = fold_cell(p[0] * s).abs();
            let ay = fold_cell(p[1] * s).abs();
            s = s * three;
            let rx = three.nmul_adde(ax, V::ONE).abs();
            let ry = three.nmul_adde(ay, V::ONE).abs();
            d = d.max((rx.min(ry) - V::ONE) / s);
            m += 1;
        }
        d
    }
}

impl<V: SdfVector> BoundedSdf<V, 2> for SierpinskiCarpet {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, 2> {
        Bounds::symmetric(Vector2::splat(V::ONE))
    }
}

/// Generalized Menger: a `base` solid with a `cell` pattern recursively
/// boolean-combined into ever-finer tiles. This is the construction behind the
/// [`MengerSponge`] abstracted over *any two* SDFs - dimension-generic.
///
/// Each level folds the domain into a unit cell (`mod(p s, 2) - 1`), evaluates
/// `cell` there, and `max`-combines it scaled by `$1/s$`:
///
/// ```math
/// d \leftarrow \max\!\Bigl(d,\ \tfrac1s\,\mathrm{cell}\bigl(\operatorname{fold}(p s)\bigr)\Bigr),
/// \qquad s \mathrel{*}= \text{lacunarity}.
/// ```
///
/// Since the combine is `max`, a `cell` that is **positive** in the region to
/// remove carves holes out of the (negative-interior) `base` - exactly the
/// "one positive, one negative" recursive subtraction. With the Menger cross as
/// `cell` and `lacunarity = 3` it reproduces [`MengerSponge`].
#[derive(Debug, Clone, Copy)]
pub struct RecursiveCarve<V: SdfVector, B, C> {
    /// The solid base (negative interior).
    pub base: B,
    /// The per-cell pattern; its positive region is carved from `base`.
    pub cell: C,
    pub iterations: u32,
    /// Scale factor per level (3 for Menger).
    pub lacunarity: V,
}

impl<V: SdfVector, const N: usize, B: SDF<V, N>, C: SDF<V, N>> SDF<V, N> for RecursiveCarve<V, B, C> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let mut d = self.base.eval(p);
        let mut s = V::ONE;
        let mut m = 0;
        while m < self.iterations {
            let mut a = Vector::ZERO;
            let mut i = 0;
            while i < N {
                a[i] = fold_cell(p[i] * s);
                i += 1;
            }
            s = s * self.lacunarity;
            d = d.max(self.cell.eval(a) / s);
            m += 1;
        }
        d
    }
}

impl<V: SdfVector, const N: usize, B: BoundedSdf<V, N>, C: SDF<V, N>> BoundedSdf<V, N> for RecursiveCarve<V, B, C> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // carving only shrinks the solid, so the base box still bounds it
        self.base.aabb()
    }
}

// Forward orbit queries through the FiniteDiff wrapper, and synthesize the
// fractal normal from its tetrahedron stencil while the central tap also carries
// the orbit (so the orbit is computed exactly once).
impl<V: SdfVector, const N: usize, S: FractalSdf<V, N>> FractalSdf<V, N> for FiniteDiff<V, S> {
    #[inline(always)]
    fn eval_orbit(&self, p: Vector<V, N>) -> (V, FractalOrbit<V, N>) {
        self.shape.eval_orbit(p)
    }
}

impl<V: SdfVector, S: FractalSdf<V, 3>> FractalGradientSdf<V> for FiniteDiff<V, S> {
    #[inline(always)]
    fn eval_orbit_grad(&self, p: Vector3<V>) -> (V, Vector3<V>, FractalOrbit<V, 3>) {
        let h = self.eps;
        // central tap: exact distance + the orbit (the one tap that needs it)
        let (dist, orbit) = self.shape.eval_orbit(p);
        // gradient from the 4-tap tetrahedron (distance-only), like FiniteDiff:
        // offsets (+++), (+--), (-+-), (--+); normalization absorbed by unit_or_zero.
        let mut grad = Vector3::ZERO;
        let mut j = 0;
        while j < 4 {
            let mut e = Vector3::ZERO;
            let mut k = 0;
            while k < 3 {
                let plus = j == 0 || k + 1 == j;
                e[k] = if plus { h } else { -h };
                k += 1;
            }
            grad = e.mul_adde(self.shape.eval(p + e), grad);
            j += 1;
        }
        (dist, unit_or_zero(grad, grad.l2_norm()), orbit)
    }

    /// The 4-tap minimum: skips the central tap and lets the orbit ride the first
    /// gradient tap (1 with traps, 3 distance-only). The orbit is sampled at
    /// `$p + e_0$` (one `eps` step off `p`) rather than exactly at `p`; for shading
    /// that shift is negligible.
    #[inline(always)]
    fn normal_orbit(&self, p: Vector3<V>) -> (Vector3<V>, FractalOrbit<V, 3>) {
        let h = self.eps;
        // tap 0 = (+++): also carries the orbit (the one tap that needs it)
        let e0 = Vector3::new([h, h, h]);
        let (d0, orbit) = self.shape.eval_orbit(p + e0);
        let mut grad = e0.mul_adde(d0, Vector3::ZERO);
        // taps 1..3 = (+--), (-+-), (--+): distance only
        let mut j = 1;
        while j < 4 {
            let mut e = Vector3::ZERO;
            let mut k = 0;
            while k < 3 {
                e[k] = if k + 1 == j { h } else { -h };
                k += 1;
            }
            grad = e.mul_adde(self.shape.eval(p + e), grad);
            j += 1;
        }
        (unit_or_zero(grad, grad.l2_norm()), orbit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V = thermite::Vector<f32>;

    fn vv(x: f32) -> V {
        V::splat(x)
    }
    fn p3(x: f32, y: f32, z: f32) -> Vector3<V> {
        Vector3::new([vv(x), vv(y), vv(z)])
    }
    fn sc(x: V) -> f32 {
        x.extract::<0>()
    }

    #[test]
    fn julia_c0_is_unit_sphere() {
        // With c = 0 the quaternion Julia set is exactly the unit sphere, so the
        // DE reads ~0 on |p| = 1, 0 inside (never escapes), and grows outside.
        let j = QuaternionJulia3D::<V>::new([vv(0.0); 4], 64);
        // on the surface (several directions) -> ~0
        for q in [p3(1.0, 0.0, 0.0), p3(0.0, 1.0, 0.0), p3(0.0, 0.0, 1.0), p3(-1.0, 0.0, 0.0)] {
            assert!(sc(j.eval(q)).abs() < 1e-3, "surface {:?}", (sc(q[0]), sc(q[1]), sc(q[2])));
        }
        // interior -> 0 (lane never escapes), exterior -> positive and increasing
        assert!(sc(j.eval(p3(0.5, 0.0, 0.0))).abs() < 1e-6);
        let d15 = sc(j.eval(p3(1.5, 0.0, 0.0)));
        let d30 = sc(j.eval(p3(3.0, 0.0, 0.0)));
        assert!(d15 > 0.0 && d30 > d15, "monotone outside: {d15} then {d30}");
        // everything finite
        for q in [p3(0.0, 0.0, 0.0), p3(1.1, 0.2, -0.3), p3(5.0, 5.0, 5.0)] {
            assert!(sc(j.eval(q)).is_finite());
        }
    }

    #[test]
    fn julia_fractal_and_bounds() {
        // A genuine fractal c: finite everywhere, ~0 near the set, bounded.
        let j = QuaternionJulia3D::<V>::new([vv(-0.45), vv(0.2), vv(0.0), vv(0.0)], 100);
        for q in [p3(0.0, 0.0, 0.0), p3(0.3, 0.1, 0.2), p3(2.0, 0.0, 0.0), p3(-1.5, 1.0, 0.5)] {
            assert!(sc(j.eval(q)).is_finite() && sc(j.eval(q)) >= -1e-4);
        }
        assert!(sc(j.eval(p3(3.0, 3.0, 3.0))) > 0.0);
        let bb = j.aabb();
        assert!((sc(bb.0[0][1]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn orbit_data() {
        let j = QuaternionJulia3D::<V>::new([vv(-0.45), vv(0.2), vv(0.0), vv(0.0)], 60);

        // eval must equal eval_orbit().0 exactly (same iteration, trap compiled out)
        for q in [p3(0.3, 0.1, 0.2), p3(2.0, 0.0, 0.0), p3(0.0, 0.0, 0.0)] {
            let (d, _) = j.eval_orbit(q);
            assert_eq!(sc(j.eval(q)), sc(d), "eval == eval_orbit().0");
        }

        // exterior point: escapes fast (low count), escaped mask set, traps finite.
        let (_, far) = j.eval_orbit(p3(3.0, 0.0, 0.0));
        assert!(far.escaped.all(), "far point escaped");
        assert!(sc(far.count) < 5.0, "escapes quickly: {}", sc(far.count));
        assert!(sc(far.trap_point) >= 0.0 && sc(far.trap_point).is_finite());

        // interior point: never escapes -> not escaped, count == max iterations.
        let (_, inside) = j.eval_orbit(p3(0.0, 0.0, 0.0));
        assert!(!inside.escaped.any(), "origin stays bounded");
        assert!((sc(inside.count) - 60.0).abs() < 0.5, "interior runs full budget");

        // traps are non-negative and the plane traps never exceed the point trap's
        // axis bound (each |z.k| <= |z|).
        let (_, o) = j.eval_orbit(p3(0.6, 0.3, 0.1));
        for k in 0..3 {
            assert!(sc(o.trap_planes[k]) >= 0.0 && sc(o.trap_planes[k]) <= sc(o.trap_point) + 1e-4);
        }

        // Mandelbulb orbit data is finite and the origin (interior) runs full budget.
        let m = Mandelbulb3D::<V>::new(10);
        let (_, mo) = m.eval_orbit(p3(0.0, 0.0, 0.0));
        assert!(!mo.escaped.any() && sc(mo.trap_point).is_finite());
        let (_, mo2) = m.eval_orbit(p3(3.0, 0.0, 0.0));
        assert!(mo2.escaped.all() && sc(mo2.count) < 5.0);
    }

    #[test]
    fn fractal_gradient_via_finite_diff() {
        use crate::FiniteDiff;

        // c=0 Julia is the unit sphere: the normal just outside (1.05,0,0) points
        // ~ +x, the distance/orbit match the inner shape's eval_orbit.
        let j = QuaternionJulia3D::<V>::new([vv(0.0); 4], 48);
        let fd = FiniteDiff::with_eps(j, vv(1e-3));
        let q = p3(1.05, 0.0, 0.0);
        let (d, n, o) = fd.eval_orbit_grad(q);

        // normal is unit length and ~ +x (sphere normal)
        let len = (sc(n[0]) * sc(n[0]) + sc(n[1]) * sc(n[1]) + sc(n[2]) * sc(n[2])).sqrt();
        assert!((len - 1.0).abs() < 1e-2, "unit normal, got {len}");
        assert!(sc(n[0]) > 0.9 && sc(n[1]).abs() < 0.1 && sc(n[2]).abs() < 0.1, "normal ~ +x");

        // distance and orbit come from the central tap == inner eval_orbit(q)
        let (dc, oc) = j.eval_orbit(q);
        assert_eq!(sc(d), sc(dc), "central distance forwarded");
        assert_eq!(sc(o.count), sc(oc.count));
        assert_eq!(sc(o.trap_point), sc(oc.trap_point));

        // a tilted point still gives a unit, roughly radial normal
        let q2 = p3(0.6, 0.6, 0.6); // |q2| ~ 1.04, just outside
        let (_, n2, _) = fd.eval_orbit_grad(q2);
        let l2 = (sc(n2[0]) * sc(n2[0]) + sc(n2[1]) * sc(n2[1]) + sc(n2[2]) * sc(n2[2])).sqrt();
        assert!((l2 - 1.0).abs() < 1e-2);
        // all components ~ equal (radial), positive
        assert!(sc(n2[0]) > 0.3 && sc(n2[1]) > 0.3 && sc(n2[2]) > 0.3);

        // works through FiniteDiff over a Mandelbulb too (finite, unit normal)
        let mb = FiniteDiff::with_eps(Mandelbulb3D::<V>::new(10), vv(1e-3));
        let (dm, nm, _) = mb.eval_orbit_grad(p3(1.15, 0.0, 0.0));
        assert!(sc(dm).is_finite());
        let lm = (sc(nm[0]) * sc(nm[0]) + sc(nm[1]) * sc(nm[1]) + sc(nm[2]) * sc(nm[2])).sqrt();
        assert!((lm - 1.0).abs() < 0.1);
    }

    #[test]
    fn minimum_tap_shade_paths() {
        use crate::{FiniteDiff, GradientSdf};

        // normal() must equal the gradient eval_grad() returns (same simplex taps),
        // just without the redundant central distance tap.
        let j = QuaternionJulia3D::<V>::new([vv(0.0); 4], 48);
        let fd = FiniteDiff::with_eps(j, vv(1e-3));
        for q in [p3(1.05, 0.0, 0.0), p3(0.6, 0.6, 0.6), p3(-1.1, 0.2, 0.3)] {
            let (_, g) = fd.eval_grad(q);
            let n = fd.normal(q);
            for k in 0..3 {
                assert_eq!(sc(g[k]), sc(n[k]), "normal == eval_grad gradient, axis {k}");
            }
        }

        // normal_orbit(): 4 taps, orbit on the first. The normal matches
        // eval_orbit_grad's to within the eps offset, and the orbit is the inner
        // shape's eval_orbit at the (slightly shifted) tap.
        let q = p3(1.05, 0.0, 0.0);
        let (_, ng, og) = fd.eval_orbit_grad(q);
        let (no, oo) = fd.normal_orbit(q);
        for k in 0..3 {
            assert!((sc(ng[k]) - sc(no[k])).abs() < 2e-2, "normal_orbit ~ eval_orbit_grad axis {k}");
        }
        // orbit comes from p + (eps,eps,eps); matches eval_orbit there
        let e0 = p3(1.05 + 1e-3, 1e-3, 1e-3);
        let (_, exact) = j.eval_orbit(e0);
        assert_eq!(sc(oo.count), sc(exact.count));
        assert_eq!(sc(oo.trap_point), sc(exact.trap_point));
        // and is close to the exact-centre orbit (eps is tiny)
        assert!((sc(oo.count) - sc(og.count)).abs() <= 1.0);
        let _ = no;
    }

    #[test]
    fn julia2d_c0_is_unit_circle() {
        use thermite_geometry::prim::Vector2 as V2;
        let p2 = |x: f32, y: f32| V2::<V>::new([vv(x), vv(y)]);
        // c = 0 -> the 2D Julia set is the unit circle.
        let j = Julia2D::<V>::new(p2(0.0, 0.0), 64);
        for q in [p2(1.0, 0.0), p2(0.0, 1.0), p2(-1.0, 0.0)] {
            assert!(sc(j.eval(q)).abs() < 1e-3, "on circle");
        }
        assert!(sc(j.eval(p2(0.5, 0.0))).abs() < 1e-6); // interior -> 0
        let d15 = sc(j.eval(p2(1.5, 0.0)));
        let d30 = sc(j.eval(p2(3.0, 0.0)));
        assert!(d15 > 0.0 && d30 > d15, "grows outside");
        // eval == eval_orbit().0, and orbit has 2 plane traps
        let (d, o) = j.eval_orbit(p2(1.4, 0.3));
        assert_eq!(sc(j.eval(p2(1.4, 0.3))), sc(d));
        assert_eq!(o.trap_planes.0.len(), 2);
        assert!(o.escaped.all());
    }

    #[test]
    fn mandelbrot2d_sanity() {
        use thermite_geometry::prim::Vector2 as V2;
        let p2 = |x: f32, y: f32| V2::<V>::new([vv(x), vv(y)]);
        let m = Mandelbrot2D::<V>::new(80);
        // origin and (-1,0) are in the set (never escape) -> 0
        assert!(sc(m.eval(p2(0.0, 0.0))).abs() < 1e-6);
        assert!(sc(m.eval(p2(-1.0, 0.0))).abs() < 1e-6);
        // clearly outside -> positive, finite, escapes fast
        assert!(sc(m.eval(p2(2.0, 0.0))) > 0.0);
        let (_, o) = m.eval_orbit(p2(2.0, 0.0));
        assert!(o.escaped.all() && sc(o.count) < 5.0 && sc(o.trap_point).is_finite());
        // interior point reports not-escaped
        let (_, oi) = m.eval_orbit(p2(-0.2, 0.0));
        assert!(!oi.escaped.any());
    }

    #[test]
    fn menger_and_carpet() {
        use thermite_geometry::prim::Vector2 as V2;
        let p2 = |x: f32, y: f32| V2::<V>::new([vv(x), vv(y)]);

        let m = MengerSponge { iterations: 3 };
        // The whole x=y=0 axis lies in both the x- and y-tubes, so it is a hole at
        // every level (robust). The centre too.
        assert!(sc(m.eval(p3(0.0, 0.0, 0.0))) > 0.0, "centre is a hole");
        assert!(sc(m.eval(p3(0.0, 0.0, 0.9))) > 0.0, "axis is a hole");
        // The corner subcube is always kept -> solid near a corner.
        assert!(sc(m.eval(p3(0.95, 0.95, 0.95))) < 0.0, "corner is solid");
        // far outside is positive ~ distance to the unit cube
        assert!((sc(m.eval(p3(3.0, 0.0, 0.0))) - 2.0).abs() < 1e-4);
        assert!((sc(m.aabb().0[0][1]) - 1.0).abs() < 1e-6);
        // deeper recursion only carves more (distance on a hole never shrinks)
        let m5 = MengerSponge { iterations: 5 };
        assert!(sc(m5.eval(p3(0.0, 0.0, 0.0))) >= sc(m.eval(p3(0.0, 0.0, 0.0))) - 1e-4);

        let c = SierpinskiCarpet { iterations: 3 };
        assert!(sc(c.eval(p2(0.0, 0.0))) > 0.0, "carpet centre is a hole");
        assert!(sc(c.eval(p2(0.95, 0.95))) < 0.0, "corner cell is solid");
        assert!((sc(c.eval(p2(3.0, 0.0))) - 2.0).abs() < 1e-4);

        // RecursiveCarve generalizes Menger: a unit-cube base + the cross cell
        // pattern (positive in the holes) must reproduce MengerSponge exactly.
        use crate::{Box3D, Field};
        let cross = Field(|a: Vector3<V>| {
            // r = |1 - 3|a|| ; cross = min(max(rx,ry), max(ry,rz), max(rz,rx)) - 1
            let three = vv(3.0);
            let rx = three.nmul_adde(a[0].abs(), vv(1.0)).abs();
            let ry = three.nmul_adde(a[1].abs(), vv(1.0)).abs();
            let rz = three.nmul_adde(a[2].abs(), vv(1.0)).abs();
            rx.max(ry).min(ry.max(rz)).min(rz.max(rx)) - vv(1.0)
        });
        let menger_rc = RecursiveCarve {
            base: Box3D { b: p3(1.0, 1.0, 1.0), r: vv(0.0) },
            cell: cross,
            iterations: 3,
            lacunarity: vv(3.0),
        };
        for q in [p3(0.0, 0.0, 0.0), p3(0.95, 0.95, 0.95), p3(0.4, 0.1, 0.7), p3(0.0, 0.0, 0.9)] {
            assert!((sc(menger_rc.eval(q)) - sc(m.eval(q))).abs() < 1e-5, "generic == MengerSponge");
        }
    }

    #[test]
    fn mandelbulb_sanity() {
        // No closed-form reference, but the DE must be finite, ~0/negative inside
        // the bulb, positive and growing outside, and the origin must not NaN.
        let m = Mandelbulb3D::<V>::new(12);
        assert!(sc(m.eval(p3(0.0, 0.0, 0.0))).is_finite()); // origin guard (r=0)
        assert!(sc(m.eval(p3(0.2, 0.1, 0.15))) <= 1e-4); // deep inside ~0
        let near = sc(m.eval(p3(1.3, 0.0, 0.0)));
        let far = sc(m.eval(p3(2.5, 0.0, 0.0)));
        assert!(near > 0.0 && far > near, "outside grows: {near} then {far}");
        for q in [p3(0.9, 0.9, 0.9), p3(-1.1, 0.3, -0.2), p3(5.0, 0.0, 0.0)] {
            assert!(sc(m.eval(q)).is_finite());
        }
        // bounded
        assert!((sc(m.aabb().0[1][1]) - 1.25).abs() < 1e-6);
    }
}
