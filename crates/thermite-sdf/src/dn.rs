//! Dimension-generic signed-distance fields.
//!
//! These primitives have closed forms that do not reference the dimension, so a
//! single `impl<V, const N: usize>` covers every `N`. The common 2D/3D names
//! ([`Circle2D`](crate::d2::Circle2D), [`Sphere3D`](crate::d3::Sphere3D),
//! [`Box2D`](crate::d2::Box2D), [`Segment2D`](crate::d2::Segment2D),
//! [`Plane3D`](crate::d3::Plane3D), [`Ellipsoid3D`](crate::d3::Ellipsoid3D), ...)
//! are *type aliases* onto these - the specialized versions compiled to the same
//! code, so there is no reason to keep two.
//!
//! Reductions over the `N` components use hand-rolled `while` loops rather than
//! iterator/`array` combinators: under a fixed `const N` they unroll to the same
//! straight-line code, and they inline reliably inside `#[target_feature]`
//! (SIMD) bodies where `core::array::map` and friends silently fall back to a
//! scalar loop.

use thermite::prelude::*;

use thermite_geometry::prim::{Bounds, Vector, vector::VectorOps as _};

use crate::{BoundedSdf, GradientSdf, SDF, SdfVector, unit_or_zero};

// ===========================================================================
// N-ball
// ===========================================================================

/// N-dimensional ball (sphere) of radius `radius`, centered at the origin.
#[derive(Debug, Clone, Copy)]
pub struct NSphere<V: SdfVector> {
    pub radius: V,
}

impl<V: SdfVector, const N: usize> SDF<V, N> for NSphere<V> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        p.l2_norm() - self.radius
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for NSphere<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let l = p.l2_norm();
        // gradient points radially outward: p / |p|
        (l - self.radius, unit_or_zero(p, l))
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, N> for NSphere<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        Bounds::symmetric(Vector::splat(self.radius))
    }
}

// `Circle2D` / `Sphere3D` aliases live in `d2` / `d3`.

// ===========================================================================
// N-orthotope (box)
// ===========================================================================

/// N-dimensional axis-aligned box with half-extents `b`.
#[derive(Debug, Clone, Copy)]
pub struct NBox<V: SdfVector, const N: usize> {
    pub b: Vector<V, N>,
}

impl<V: SdfVector, const N: usize> SDF<V, N> for NBox<V, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let q = p.abs() - self.b;
        let mut g = q[0];
        let mut i = 1;
        while i < N {
            g = g.max(q[i]);
            i += 1;
        }
        q.max(Vector::ZERO).l2_norm() + g.min(V::ZERO)
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for NBox<V, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let s = p.signum();
        let q = p.abs() - self.b;

        // running maximum component, with a per-lane one-hot of its axis for the
        // interior (where the gradient snaps to the nearest face normal)
        let zero = Vector::ZERO;
        let mut g = q[0];
        let mut face = {
            let mut e = zero;
            e[0] = V::ONE;
            e
        };
        let mut i = 1;
        while i < N {
            let mut ei = zero;
            ei[i] = V::ONE;
            let take = q[i].cmp_gt(g);
            g = take.select(q[i], g);
            face = take.select(ei, face);
            i += 1;
        }

        let mq = q.max(zero);
        let l = mq.l2_norm();
        let outside = g.cmp_gt(V::ZERO);
        (l + g.min(V::ZERO), s * outside.select(unit_or_zero(mq, l), face))
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, N> for NBox<V, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        Bounds::symmetric(self.b)
    }
}

// `Box2D` alias lives in `d2`.

// ===========================================================================
// Half-space (plane)
// ===========================================================================

/// Half-space with unit normal `n` and offset `h`: `dot(p, n) + h`.
#[derive(Debug, Clone, Copy)]
pub struct NPlane<V: SdfVector, const N: usize> {
    pub n: Vector<V, N>,
    pub h: V,
}

impl<V: SdfVector, const N: usize> SDF<V, N> for NPlane<V, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        p.dot(&self.n) + self.h
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for NPlane<V, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        // the gradient is the (unit) plane normal everywhere
        (p.dot(&self.n) + self.h, self.n)
    }
}

// `Plane3D` alias lives in `d3`.

// ===========================================================================
// N-capsule (thick segment)
// ===========================================================================

/// Capsule: the segment `a -> b` thickened by radius `r`, in any dimension.
#[derive(Debug, Clone, Copy)]
pub struct NCapsule<V: SdfVector, const N: usize> {
    pub a: Vector<V, N>,
    pub b: Vector<V, N>,
    pub r: V,
}

impl<V: SdfVector, const N: usize> SDF<V, N> for NCapsule<V, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(V::ZERO, V::ONE);
        ba.nmul_adde(h, pa).l2_norm() - self.r // |pa - ba*h| - r
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for NCapsule<V, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let ba = self.b - self.a;
        let pa = p - self.a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(V::ZERO, V::ONE);
        let q = ba.nmul_adde(h, pa); // pa - ba*h
        let d = q.l2_norm();
        (d - self.r, unit_or_zero(q, d))
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, N> for NCapsule<V, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        Bounds::from_corners(self.a.min(self.b), self.a.max(self.b)).expand(self.r)
    }
}

// `Segment2D` / `Segment3D` aliases live in `d2` / `d3`.

// ===========================================================================
// N-ellipsoid (approximate)
// ===========================================================================

/// Axis-aligned ellipsoid with semi-axes `r`. Exact gradient direction, but -
/// as in the reference - only an approximate (bounding) distance.
#[derive(Debug, Clone, Copy)]
pub struct NEllipsoid<V: SdfVector, const N: usize> {
    pub r: Vector<V, N>,
}

impl<V: SdfVector, const N: usize> SDF<V, N> for NEllipsoid<V, N> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let p1 = p / self.r;
        let k0 = p1.dot(&p1).sqrt();
        let p2 = p1 / self.r;
        let dd = p2.dot(&p2);
        let k1 = dd.cmp_gt(V::ZERO).select(V::ONE / dd.sqrt(), V::ZERO); // guard origin
        k0.mul_sube(k0, k0) * k1
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for NEllipsoid<V, N> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let p1 = p / self.r;
        let k0 = p1.dot(&p1).sqrt();
        let p2 = p1 / self.r;
        let dd = p2.dot(&p2);
        let k1 = dd.cmp_gt(V::ZERO).select(V::ONE / dd.sqrt(), V::ZERO); // guard origin
        (k0.mul_sube(k0, k0) * k1, p2 * k1)
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, N> for NEllipsoid<V, N> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        Bounds::symmetric(self.r)
    }
}

// `Ellipsoid3D` alias lives in `d3`.

// ===========================================================================
// Cross-polytope (L1 ball) - lower bound, the N-D OctahedronBound
// ===========================================================================

/// N-dimensional cross-polytope (the L1 / orthoplex "ball") of radius `s`.
///
/// Returns `(|p|_1 - s) / sqrt(N)`, a 1-Lipschitz lower bound on the true
/// distance (exact along the face normals). The `1/sqrt(N)` factor is a runtime
/// reciprocal sqrt - unlike the baked `1/sqrt(3)` constant in the 3D-specific
/// `OctahedronBound3D`, which is why this one is its own type rather than an alias.
#[derive(Debug, Clone, Copy)]
pub struct CrossPolytope<V: SdfVector> {
    pub s: V,
}

#[inline(always)]
fn inv_sqrt_n<V: SdfVector, const N: usize>() -> V {
    // N as a vector, accumulated so it stays const-foldable under a fixed N
    let mut nf = V::ZERO;
    let mut i = 0;
    while i < N {
        nf += V::ONE;
        i += 1;
    }
    V::ONE / nf.sqrt()
}

impl<V: SdfVector, const N: usize> SDF<V, N> for CrossPolytope<V> {
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let mut acc = p[0].abs();
        let mut i = 1;
        while i < N {
            acc += p[i].abs();
            i += 1;
        }
        (acc - self.s) * inv_sqrt_n::<V, N>()
    }
}

impl<V: SdfVector, const N: usize> GradientSdf<V, N> for CrossPolytope<V> {
    #[inline(always)]
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>) {
        let scale = inv_sqrt_n::<V, N>();
        let mut acc = p[0].abs();
        let mut i = 1;
        while i < N {
            acc += p[i].abs();
            i += 1;
        }
        // grad of (|p|_1 - s)/sqrt(N) is sign(p)/sqrt(N), already unit length
        ((acc - self.s) * scale, p.signum() * scale)
    }
}

impl<V: SdfVector, const N: usize> BoundedSdf<V, N> for CrossPolytope<V> {
    #[inline(always)]
    fn aabb(&self) -> Bounds<V, N> {
        // vertices sit at +/- s on each axis
        Bounds::symmetric(Vector::splat(self.s))
    }
}
