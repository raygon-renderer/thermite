//! Surface-normal displacement and domain warping for any 3D SDF.
//!
//! These combinators add fine geometric detail - bumps, spikes, terrain, ridges -
//! to *any* [`SDF<V, 3>`], not just planes. They are 3D-only (the dimensionality
//! is hard-coded).
//!
//! # Why additive displacement is normal displacement
//!
//! Adding a scalar field `$h(p)$` to a signed distance `$d(p)$` moves the surface
//! `$d = 0$` to `$d = \pm h$`. Because the surface moves along the field's gradient
//! `$\nabla d$`, and near the surface `$\nabla d$` is exactly the unit outward
//! *normal*, the effect is to push each surface point **along its own normal** by
//! `$\approx h$`. This holds for every shape - a sphere gets radial bumps, a torus
//! gets bumps along its tube normals, a fractal gets crust - so a "bumpy" or
//! "spiky" sphere is just `Sphere3D` displaced by a bumpy/spiky `$h$`. `$h$` is a
//! *solid* field sampled at the query point `$p$` (the standard SDF technique; the
//! displacement is carved from 3D space rather than a surface parameterization,
//! which an SDF does not have). For a heightfield set the base to a plane.
//!
//! # Keeping it raymarchable (the `lipschitz` knob)
//!
//! Displacement breaks the metric. If `$h$` is `$K$`-Lipschitz (i.e. `$\lVert
//! \nabla h \rVert \le K$`), then `$d \pm h$` is only `$(1+K)$`-Lipschitz, so a
//! sphere tracer using it at full step overshoots and punches through bumps. The
//! fix (Hart 1989's distance-bound requirement; Heidrich & Seidel 1998 via a
//! user-supplied Lipschitz bound, since such bounds "cannot be automated") is to
//! divide by the bound:
//!
//! ```math
//! d_{\text{out}}(p) = \frac{d(p) - h(p)}{1 + K}
//! ```
//!
//! (positive `$h$` pushes the surface outward along its normal; negate to carve
//! inward) which is a *conservative* signed distance **bound** - exactly what sphere
//! tracing requires (Keinert et al. 2014, Eq. 1) - and is therefore safe to march
//! at full step. `K` must be an upper bound on the displacement's Lipschitz
//! constant: too large just means smaller (slower, still correct) steps; too small
//! risks overshoot. `K = 0` recovers the raw `$d \pm h$` (fast, but reduce your
//! march step yourself).
//!
//! This is an *estimate*, never an exact SDF - the exact distance to a displaced
//! surface has no closed form - so neither combinator implements [`BoundedSdf`] or
//! [`GradientSdf`]; wrap in [`FiniteDiff`](crate::ops::FiniteDiff) for normals.
//!
//! [`BoundedSdf`]: crate::BoundedSdf
//! [`GradientSdf`]: crate::GradientSdf

use thermite_geometry::soa::prim::Vector3;

use crate::{SDF, SdfVector};

/// Displaces the surface of any 3D shape `S` along its local normal by a scalar
/// height field `h: Fn(Vector3<V>) -> V`, Lipschitz-normalized so the result
/// stays a sphere-traceable distance bound. See the [module docs](self) for the
/// math and the `lipschitz` trade-off.
///
/// The sign of `h` *is* the direction: **positive `h` pushes the surface outward**
/// along its normal (`$d - h$`), so bumps and spikes stick out; negative `h`
/// carves pits inward. A bumpy sphere is `Displacement::new(Sphere3D { .. },
/// bumpy, k)`.
#[derive(Debug, Clone, Copy)]
pub struct Displacement<V: SdfVector, S, F> {
    pub shape: S,
    /// Per-point displacement height along the surface normal. Positive = outward,
    /// negative = inward (carving).
    pub height: F,
    /// Upper bound `K` on the height field's Lipschitz constant (`$\lVert \nabla h
    /// \rVert \le K$`). Result is divided by `1 + K`. `0` = no normalization.
    pub lipschitz: V,
}

impl<V: SdfVector, S, F> Displacement<V, S, F> {
    /// Builds a normal displacement: positive `height` pushes the surface out
    /// along its normal, negative carves in.
    #[inline(always)]
    pub fn new(shape: S, height: F, lipschitz: V) -> Self {
        Self { shape, height, lipschitz }
    }
}

impl<V: SdfVector, S: SDF<V, 3>, F: Fn(Vector3<V>) -> V> SDF<V, 3> for Displacement<V, S, F> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let d = self.shape.eval(p);
        let h = (self.height)(p);
        // positive h moves the surface out along +normal; negative carves in
        (d - h) / (V::ONE + self.lipschitz)
    }
}

/// Warps the *domain* of any 3D shape `S` by a vector offset field `warp: Fn(
/// Vector3<V>) -> Vector3<V>`, evaluating `$d(p - \text{warp}(p))$`,
/// Lipschitz-normalized.
///
/// Where [`Displacement`] moves the surface along its normal by a scalar, this
/// moves space itself - bending, swirling, or shifting features tangentially
/// (wood grain, flow, melt). If `warp` is `$K$`-Lipschitz (operator-norm bound on
/// its Jacobian) the composed map `$p \mapsto p - \text{warp}(p)$` is
/// `$(1+K)$`-Lipschitz, so the result is divided by `1 + K` to stay a
/// conservative, traceable bound (same contract as [`Displacement`]). `warp = 0`
/// is the identity.
#[derive(Debug, Clone, Copy)]
pub struct DomainWarp<V: SdfVector, S, F> {
    pub shape: S,
    /// Per-point domain offset, subtracted from `p` before evaluating `shape`.
    pub warp: F,
    /// Upper bound `K` on the warp's Lipschitz constant (operator norm of its
    /// Jacobian). Result is divided by `1 + K`. `0` = no normalization.
    pub lipschitz: V,
}

impl<V: SdfVector, S, F> DomainWarp<V, S, F> {
    /// Builds a domain warp; `warp(p)` is the offset subtracted from `p`.
    #[inline(always)]
    pub fn new(shape: S, warp: F, lipschitz: V) -> Self {
        Self { shape, warp, lipschitz }
    }
}

impl<V: SdfVector, S: SDF<V, 3>, F: Fn(Vector3<V>) -> Vector3<V>> SDF<V, 3> for DomainWarp<V, S, F> {
    #[inline(always)]
    fn eval(&self, p: Vector3<V>) -> V {
        let q = p - (self.warp)(p);
        self.shape.eval(q) / (V::ONE + self.lipschitz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Plane3D, Sphere3D};
    use thermite::prelude::*;
    use thermite_geometry::soa::prim::Vector3;

    type V = thermite::Vector<f32>;

    #[inline]
    fn v(x: f32) -> V {
        V::splat(x)
    }
    #[inline]
    fn p3(x: f32, y: f32, z: f32) -> Vector3<V> {
        Vector3::new([v(x), v(y), v(z)])
    }
    #[inline]
    fn s(x: V) -> f32 {
        x.extract::<0>()
    }

    const PTS3: [(f32, f32, f32); 5] = [
        (1.3, 0.2, -0.4),
        (-0.7, 1.1, 0.5),
        (0.3, -0.6, 1.4),
        (2.1, 0.0, 0.0),
        (-1.2, -0.9, 0.8),
    ];

    // Zero displacement is an exact passthrough of the base shape.
    #[test]
    fn zero_displacement_passthrough() {
        let base = Sphere3D { radius: v(1.0) };
        let d = Displacement::new(base, |_p: Vector3<V>| V::ZERO, v(0.0));
        for &(x, y, z) in &PTS3 {
            let p = p3(x, y, z);
            assert!((s(d.eval(p)) - s(base.eval(p))).abs() < 1e-6);
        }
    }

    // A constant displacement inflates/deflates any shape (here a sphere) like a
    // radius change: at K=0, sphere(1) pushed out 0.25 reads as sphere(1.25), and a
    // negative height carves inward.
    #[test]
    fn constant_displacement_changes_radius() {
        let base = Sphere3D { radius: v(1.0) };
        // positive height -> outward: at radius 3, base 3-1=2, displaced 2 - 0.25 = 1.75
        let out = Displacement::new(base, |_p: Vector3<V>| v(0.25), v(0.0));
        assert!((s(out.eval(p3(3.0, 0.0, 0.0))) - 1.75).abs() < 1e-5);
        // negative height -> inward: 2 - (-0.25) = 2.25
        let inn = Displacement::new(base, |_p: Vector3<V>| v(-0.25), v(0.0));
        assert!((s(inn.eval(p3(3.0, 0.0, 0.0))) - 2.25).abs() < 1e-5);
    }

    // The headline case: a *bumpy sphere*. A spatially varying height makes the
    // displaced radius differ by direction (bumps along the normal), and the
    // Lipschitz-normalized field stays ~1-Lipschitz (safe to sphere-trace).
    #[test]
    fn bumpy_sphere_is_lipschitz() {
        let base = Sphere3D { radius: v(1.0) };
        // h(p) = 0.15 * sin(4 x) ; |dh/dx| <= 0.6, so K = 0.6 bounds it
        let k = 0.6f32;
        let d = Displacement::new(base, |p: Vector3<V>| v(0.15) * (v(4.0) * p[0]).sin(), v(k));

        // inside stays negative, far outside positive (still a valid field)
        assert!(s(d.eval(p3(0.0, 0.0, 0.0))) < 0.0);
        assert!(s(d.eval(p3(3.0, 0.0, 0.0))) > 0.0);

        // bumps: the displaced surface crossing differs by direction. Probe the
        // field at radius 1 along +x vs +y; with a directional height they differ.
        let along_x = s(d.eval(p3(1.0, 0.0, 0.0)));
        let along_y = s(d.eval(p3(0.0, 1.0, 0.0)));
        assert!((along_x - along_y).abs() > 1e-3, "height should vary over the surface");

        // ~1-Lipschitz after normalization (base 1-Lipschitz + 0.6-Lipschitz h, /1.6)
        let eps = 0.02f32;
        for &(x, y, z) in &PTS3 {
            let d0 = s(d.eval(p3(x, y, z)));
            for (dx, dy, dz) in [(eps, 0.0, 0.0), (0.0, eps, 0.0), (0.0, 0.0, eps)] {
                let d1 = s(d.eval(p3(x + dx, y + dy, z + dz)));
                let step = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!((d1 - d0).abs() <= step + 1e-4, "displaced field not 1-Lipschitz");
            }
        }
    }

    // A heightfield is just a displaced plane (the special case of constant normal).
    #[test]
    fn heightfield_is_displaced_plane() {
        // plane y = 0 (eval = p.y); push up by a ramp h(p) = 0.3*x, K = 0.3
        let plane = Plane3D {
            n: p3(0.0, 1.0, 0.0),
            h: v(0.0),
        };
        let terrain = Displacement::new(plane, |p: Vector3<V>| v(0.3) * p[0], v(0.3));
        // on the displaced surface y = 0.3*x: at (1, 0.3, 0) the distance is ~0
        assert!(s(terrain.eval(p3(1.0, 0.3, 0.0))).abs() < 1e-5);
        // above it -> positive, below -> negative
        assert!(s(terrain.eval(p3(1.0, 1.0, 0.0))) > 0.0);
        assert!(s(terrain.eval(p3(1.0, -1.0, 0.0))) < 0.0);
    }

    // DomainWarp: identity warp is a passthrough; a bounded warp stays finite and
    // ~1-Lipschitz after normalization.
    #[test]
    fn domain_warp_passthrough_and_lipschitz() {
        let base = Sphere3D { radius: v(1.0) };
        let id = DomainWarp::new(base, |_p: Vector3<V>| p3(0.0, 0.0, 0.0), v(0.0));
        for &(x, y, z) in &PTS3 {
            let p = p3(x, y, z);
            assert!((s(id.eval(p)) - s(base.eval(p))).abs() < 1e-6);
        }

        // warp offset = (0.3*y, 0, 0): Jacobian has a single 0.3 entry, K = 0.3
        let k = 0.3f32;
        let w = DomainWarp::new(base, |p: Vector3<V>| Vector3::new([v(0.3) * p[1], V::ZERO, V::ZERO]), v(k));
        let eps = 0.02f32;
        for &(x, y, z) in &PTS3 {
            let d0 = s(w.eval(p3(x, y, z)));
            assert!(d0.is_finite());
            for (dx, dy, dz) in [(eps, 0.0, 0.0), (0.0, eps, 0.0), (0.0, 0.0, eps)] {
                let d1 = s(w.eval(p3(x + dx, y + dy, z + dz)));
                let step = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!((d1 - d0).abs() <= step + 1e-4, "warped field not 1-Lipschitz");
            }
        }
    }
}
