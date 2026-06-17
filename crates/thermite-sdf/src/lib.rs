//! Generic, ISA-portable signed-distance fields (2D and 3D) built on Thermite.
//!
//! Shapes are organised by a three-trait hierarchy:
//! - [`SDF`] - the signed distance (every primitive implements this).
//! - [`GradientSdf`] - distance plus the analytic, unit-length gradient/normal.
//! - [`BoundedSdf`] - a closed-form axis-aligned bounding box.
//!
//! Primitives live in [`d2`], [`d3`], and [`dn`]; combinators (union, intersection,
//! rounding, domain repetition, ...) live in [`ops`]. All are generic over the
//! Thermite vector type, so the same code runs on any backend and lane width.
//!
//! [`FiniteDiff`] can be used to efficiently compute gradients/normals on SDFs that do
//! not have native analytic `GradientSdf` implementations.

#![no_std]

use thermite::mask::GenericMask as _;

pub use thermite::math::SpatialMath as SdfVector;

use thermite_geometry::prim::{Bounds, Vector};

/// `v / len`, returning the zero vector instead of `NaN` when `len == 0`.
///
/// Gradient normals are computed as `offset / distance`; the direction is
/// genuinely undefined where the distance is zero (the medial axis, or a
/// shape's center), which would otherwise produce `0/0 = NaN`. Those points are
/// measure-zero but are routinely sampled (e.g. the center of a filled disk), so
/// we return a finite zero vector there.
#[inline(always)]
pub(crate) fn unit_or_zero<V: SdfVector, const N: usize>(v: Vector<V, N>, len: V) -> Vector<V, N> {
    // len >= 0 always (it is a length); divide by 1 when it is exactly 0.
    v / len.cmp_gt(V::ZERO).select(len, V::ONE)
}

pub mod consts;
pub mod d2;
pub mod d3;
pub mod dn;
pub mod ops;

pub use consts::SdfConsts;
pub use d2::*;
pub use d3::*;
pub use dn::*;
pub use ops::*;

pub use ops::FiniteDiff;

pub trait SDF<V: SdfVector, const N: usize> {
    /// Returns the signed distance
    fn eval(&self, p: Vector<V, N>) -> V;
}

/// An [`SDF`] that also provides its analytic gradient (the unit-length surface
/// normal), with the distance and gradient sharing intermediate terms.
///
/// Not every SDF in the reference articles has a published closed-form gradient
/// (many are distance-only), so this is a separate trait that supersedes
/// [`SDF`]; shapes implement it only when an exact analytic gradient is known.
pub trait GradientSdf<V: SdfVector, const N: usize>: SDF<V, N> {
    /// Returns the signed distance + N-dimensional gradient.
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>);
}

/// An [`SDF`] that also has a closed-form axis-aligned bounding box of its solid
/// (the `distance <= 0` region).
///
/// This is a separate trait rather than a method on [`SDF`] because not every
/// SDF is bounded - e.g. [`Parabola2D`] is an infinite curve - and a few have no
/// convenient closed form. The returned box always *contains* the shape; for
/// most shapes it is a tight fit, for a handful (noted on the impl) it is a
/// simple conservative bound.
pub trait BoundedSdf<V: SdfVector, const N: usize>: SDF<V, N> {
    /// Axis-aligned bounding box enclosing the shape.
    fn aabb(&self) -> Bounds<V, N>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use thermite::prelude::*;
    use thermite_geometry::prim::{Vector2, Vector3};

    /// 1-lane scalar vector, the simplest concrete `FloatVector`.
    type V = thermite::Vector<f32>;

    #[inline]
    fn v(x: f32) -> V {
        V::splat(x)
    }

    #[inline]
    fn p(x: f32, y: f32) -> Vector2<V> {
        Vector2::new([v(x), v(y)])
    }

    #[inline]
    fn p3(x: f32, y: f32, z: f32) -> Vector3<V> {
        Vector3::new([v(x), v(y), v(z)])
    }

    #[inline]
    fn s(x: V) -> f32 {
        x.extract::<0>()
    }

    /// Asserts a 3D distance is close and that the gradient has unit length.
    fn check3(d: V, g: Vector3<V>, expected: f32) {
        assert!((s(d) - expected).abs() < 1e-4, "distance {} != {expected}", s(d));
        let len = (s(g[0]) * s(g[0]) + s(g[1]) * s(g[1]) + s(g[2]) * s(g[2])).sqrt();
        assert!((len - 1.0).abs() < 1e-3, "gradient length {len} != 1");
    }

    /// Asserts a distance is close and that the returned gradient has unit length.
    fn check(d: V, g: Vector2<V>, expected: f32) {
        assert!((s(d) - expected).abs() < 1e-4, "distance {} != {expected}", s(d));
        let len = (s(g[0]) * s(g[0]) + s(g[1]) * s(g[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3, "gradient length {len} != 1");
    }

    #[test]
    fn no_nans_at_degenerate_points() {
        // gradients/evals at centers and on surfaces must be finite (no 0/0 NaN)
        let fin2 = |(d, g): (V, Vector2<V>)| {
            assert!(s(d).is_finite() && s(g[0]).is_finite() && s(g[1]).is_finite());
        };
        fin2((Circle2D { radius: v(1.0) }).eval_grad(p(0.0, 0.0))); // center
        fin2((Circle2D { radius: v(1.0) }).eval_grad(p(1.0, 0.0))); // surface
        fin2(
            (Segment2D {
                a: p(-1.0, 0.0),
                b: p(1.0, 0.0),
                r: v(0.0),
            })
            .eval_grad(p(0.0, 0.0)),
        ); // on axis
        fin2((Moon2D::new(v(0.6), v(1.0), v(0.8))).eval_grad(p(0.0, 0.0)));
        fin2((Ellipse2D { ab: p(1.5, 0.8) }).eval_grad(p(0.0, 0.0)));

        let fin3 = |(d, g): (V, Vector3<V>)| {
            assert!(s(d).is_finite() && s(g[0]).is_finite() && s(g[1]).is_finite() && s(g[2]).is_finite());
        };
        fin3((Sphere3D { radius: v(1.0) }).eval_grad(p3(0.0, 0.0, 0.0)));
        fin3((VerticalCylinder3D::from_height(v(2.0), v(1.0))).eval_grad(p3(0.0, 0.0, 0.0))); // axis
        fin3(
            (Cylinder3D {
                a: p3(0.0, -1.0, 0.0),
                b: p3(0.0, 1.0, 0.0),
                r: v(1.0),
            })
            .eval_grad(p3(0.0, 0.5, 0.0)),
        );

        // Ellipsoid eval no longer NaN at the origin (was 1/sqrt(0))
        assert!(s((Ellipsoid3D { r: p3(1.0, 1.0, 1.0) }).eval(p3(0.0, 0.0, 0.0))).is_finite());
    }

    #[test]
    fn circle() {
        let c = Circle2D { radius: v(1.0) };
        let (d, g) = c.eval_grad(p(3.0, 4.0));
        check(d, g, 4.0);
        assert!((s(g[0]) - 0.6).abs() < 1e-4 && (s(g[1]) - 0.8).abs() < 1e-4);
        assert!((s(c.eval(p(3.0, 4.0))) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn segment() {
        // horizontal segment from (-1,0) to (1,0), radius 0; point above the middle
        let seg = Segment2D {
            a: p(-1.0, 0.0),
            b: p(1.0, 0.0),
            r: v(0.0),
        };
        let (d, g) = seg.eval_grad(p(0.0, 2.0));
        check(d, g, 2.0);
        // beyond the endpoint -> distance to the cap
        let (d2, g2) = seg.eval_grad(p(2.0, 0.0));
        check(d2, g2, 1.0);
    }

    #[test]
    fn box2d() {
        let b = Box2D { b: p(1.0, 1.0) };
        // outside on +x face
        let (d, g) = b.eval_grad(p(3.0, 0.0));
        check(d, g, 2.0);
        // inside
        let (di, _) = b.eval_grad(p(0.0, 0.0));
        assert!((s(di) - -1.0).abs() < 1e-4);
    }

    #[test]
    fn pie_and_arc() {
        // 45-degree half aperture pie of radius 1; caller supplies (sin, cos)
        let h = core::f32::consts::FRAC_1_SQRT_2; // sin(45) == cos(45)
        let pie = Pie2D {
            radius: v(1.0),
            sc: p(h, h),
        };
        let (_, g) = pie.eval_grad(p(0.3, 0.9));
        let len = (s(g[0]) * s(g[0]) + s(g[1]) * s(g[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3);

        let arc = Arc2D {
            sc: p(h, h),
            ra: v(1.0),
            rb: v(0.1),
        };
        let (_, ga) = arc.eval_grad(p(0.8, 0.7));
        let len = (s(ga[0]) * s(ga[0]) + s(ga[1]) * s(ga[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cross() {
        // arms 2 long, 0.5 wide
        let c = Cross2D { b: p(2.0, 0.5) };
        // outside off the +x arm tip
        let (d, g) = c.eval_grad(p(4.0, 0.0));
        check(d, g, 2.0);
        // inside the body -> negative distance
        let (di, _) = c.eval_grad(p(0.0, 0.0));
        assert!(s(di) < 0.0);
        // a point off the long side of the +x arm, unit gradient
        let (_, g2) = c.eval_grad(p(1.0, 1.5));
        let len = (s(g2[0]) * s(g2[0]) + s(g2[1]) * s(g2[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3);
    }

    #[test]
    fn parabola() {
        // y = x^2
        let par = Parabola2D::<V>::new(v(1.0));
        // below the focal point (y < 0.5) the vertex (0,0) is nearest -> trig branch
        let (d, g) = par.eval_grad(p(0.0, 0.3));
        check(d, g, 0.3);
        // a point well off to the side exercises the cbrt branch; gradient is unit length
        let (d1, g1) = par.eval_grad(p(3.0, 0.0));
        assert!(s(d1).is_finite() && s(d1) > 0.0);
        let len = (s(g1[0]) * s(g1[0]) + s(g1[1]) * s(g1[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3, "gradient length {len}");
        // symmetry: mirroring x flips only the gradient's x component
        let (d2, g2) = par.eval_grad(p(-3.0, 0.0));
        assert!((s(d1) - s(d2)).abs() < 1e-4);
        assert!((s(g1[0]) + s(g2[0])).abs() < 1e-4 && (s(g1[1]) - s(g2[1])).abs() < 1e-4);
    }

    #[test]
    fn triangle_inside_outside() {
        let tri = Triangle2D {
            v: [p(-1.0, -1.0), p(1.0, -1.0), p(0.0, 1.0)],
        };
        // centroid is inside -> negative
        let (d, _) = tri.eval_grad(p(0.0, -0.33));
        assert!(s(d) < 0.0);
        // far away -> positive, unit gradient
        let (d2, g2) = tri.eval_grad(p(5.0, 0.0));
        assert!(s(d2) > 0.0);
        let len = (s(g2[0]) * s(g2[0]) + s(g2[1]) * s(g2[1])).sqrt();
        assert!((len - 1.0).abs() < 1e-3);
    }

    /// The dedicated `eval` must agree with `eval_grad(..).0` for every shape.
    #[test]
    fn eval_matches_eval_grad() {
        let pts = [p(0.7, 1.3), p(-1.6, 0.4), p(2.2, -0.9), p(0.1, 2.5), p(-0.8, -1.1)];
        let h = core::f32::consts::FRAC_1_SQRT_2;

        macro_rules! agree {
            ($shape:expr) => {{
                let shape = $shape;
                for q in pts {
                    let a = s(shape.eval(q));
                    let b = s(shape.eval_grad(q).0);
                    // both NaN (degenerate point) is acceptable agreement
                    assert!(
                        (a - b).abs() < 1e-4 || (a.is_nan() && b.is_nan()),
                        "{} vs {}",
                        a,
                        b
                    );
                }
            }};
        }

        agree!(Circle2D { radius: v(1.0) });
        agree!(Pie2D {
            radius: v(1.0),
            sc: p(h, h)
        });
        agree!(Arc2D {
            sc: p(h, h),
            ra: v(1.0),
            rb: v(0.2)
        });
        agree!(Segment2D {
            a: p(-1.0, 0.0),
            b: p(1.0, 0.5),
            r: v(0.1)
        });
        agree!(Vesica2D::from_circle(v(1.2), v(0.5)));
        agree!(Box2D { b: p(1.0, 0.6) });
        agree!(Cross2D { b: p(1.5, 0.4) });
        agree!(Hexagon2D { r: v(1.0) });
        agree!(IsoscelesTriangle2D { q: p(0.8, 1.5) });
        agree!(Triangle2D {
            v: [p(-1.0, -1.0), p(1.0, -1.0), p(0.0, 1.0)]
        });
        agree!(Quad2D {
            v: [p(-1.0, -1.0), p(1.0, -1.0), p(1.2, 1.0), p(-1.0, 0.8)]
        });
        agree!(Moon2D::new(v(0.6), v(1.0), v(0.8)));
        agree!(Trapezoid2D {
            ra: v(1.0),
            rb: v(0.5),
            he: v(0.8)
        });
        agree!(Heart2D);
        agree!(Ellipse2D { ab: p(1.5, 0.8) });
        agree!(Parabola2D::<V>::new(v(1.0)));
    }

    #[test]
    fn finite_diff_normals() {
        // FiniteDiff wraps a plain SDF and reconstructs the normal via central
        // differences; it must match the analytic gradient and forward the exact
        // distance. Checked in 2D (circle) and 3D (sphere).
        let circle = Circle2D { radius: v(1.0) };
        let fd = FiniteDiff::new(circle);
        for q in [p(1.5, 0.0), p(0.7, 0.9), p(-1.2, 0.6), p(0.2, -1.4)] {
            let (df, gf) = fd.eval_grad(q);
            let (da, ga) = circle.eval_grad(q);
            assert!((s(df) - s(da)).abs() < 1e-4, "distance forwarded exactly");
            assert!((s(gf[0]) - s(ga[0])).abs() < 2e-2);
            assert!((s(gf[1]) - s(ga[1])).abs() < 2e-2);
        }

        let sphere = Sphere3D { radius: v(1.0) };
        let fd3 = FiniteDiff::with_eps(sphere, v(1e-3));
        for q in [p3(1.5, 0.0, 0.0), p3(0.6, 0.7, 0.8), p3(-1.0, 0.5, -0.4)] {
            let (_, gf) = fd3.eval_grad(q);
            let (_, ga) = sphere.eval_grad(q);
            for k in 0..3 {
                assert!((s(gf[k]) - s(ga[k])).abs() < 2e-2, "axis {k}");
            }
        }
    }

    // Every analytic gradient must agree with a central-difference of the same
    // field. This audits all hand-derived gradients - including the smooth
    // booleans - in one place. Points are chosen in smooth regions (away from
    // operand kinks) so the finite difference is well-defined.
    #[test]
    fn gradient_matches_finite_diff() {
        macro_rules! agrees {
            ($shape:expr, $pts:expr) => {{
                let shape = $shape;
                let fd = FiniteDiff::with_eps(shape, v(1e-3));
                for q in $pts {
                    let (_, ga) = shape.eval_grad(q);
                    let (_, gf) = fd.eval_grad(q);
                    for k in 0..ga.0.len() {
                        // 5e-2 absorbs finite-difference truncation in high-curvature
                        // blend regions; a real formula error is O(1), not O(5e-2).
                        assert!(
                            (s(ga[k]) - s(gf[k])).abs() < 5e-2,
                            "{} axis {k}: analytic {} vs fd {}",
                            stringify!($shape),
                            s(ga[k]),
                            s(gf[k])
                        );
                    }
                }
            }};
        }

        // primitives
        agrees!(Circle2D { radius: v(1.0) }, [p(1.4, 0.3), p(0.5, -0.7), p(-0.9, 0.55)]);
        agrees!(Box2D { b: p(1.0, 0.6) }, [p(1.5, 0.1), p(0.15, 1.1), p(-1.4, -0.2)]);
        agrees!(Ellipse2D { ab: p(1.5, 0.8) }, [p(2.0, 0.2), p(0.1, 1.3), p(-1.8, -0.3)]);
        agrees!(
            Segment2D {
                a: p(-1.0, 0.0),
                b: p(1.0, 0.5),
                r: v(0.3)
            },
            [p(0.0, 0.8), p(1.4, 0.6), p(-1.3, -0.4)]
        );
        // Tier A/B additions (interior + exterior; avoid abs-fold axes and kinks)
        agrees!(
            RoundedX2D { w: v(1.5), r: v(0.2) },
            [p(1.0, 0.2), p(0.9, 0.7), p(0.3, 0.15)]
        );
        agrees!(
            RoundedBox2D {
                b: p(1.0, 0.7),
                r: [v(0.3), v(0.1), v(0.4), v(0.2)]
            },
            [p(1.4, 0.2), p(0.2, 1.1), p(0.5, 0.3), p(-1.3, -0.2)]
        );
        agrees!(
            OrientedBox2D {
                a: p(-1.0, -0.4),
                b: p(1.0, 0.5),
                th: v(0.6)
            },
            [p(1.2, 0.9), p(-1.1, -0.7), p(0.1, 0.05)]
        );
        agrees!(
            UnevenCapsule2D::new(v(0.7), v(0.3), v(1.2)),
            [p(0.5, -0.3), p(0.4, 1.4), p(0.6, 0.6), p(0.2, 0.5)]
        );
        // fold-polygon family (avoid x=0 / y=0 abs axes and the fold seams)
        agrees!(
            EquilateralTriangle2D { r: v(1.0) },
            [p(0.4, -0.3), p(-0.5, 0.2), p(0.3, 0.5), p(0.55, -0.6)]
        );
        agrees!(
            Pentagon2D { r: v(1.0) },
            [p(0.5, 0.8), p(-0.6, 0.4), p(0.3, -0.7), p(0.25, 0.3)]
        );
        agrees!(
            Octagon2D { r: v(1.0) },
            [p(0.7, 0.4), p(-0.5, 0.6), p(0.4, -0.8), p(0.3, 0.25)]
        );
        agrees!(
            Hexagram2D { r: v(1.0) },
            [p(0.8, 0.3), p(-0.4, 0.7), p(0.5, -0.5), p(0.25, 0.3)]
        );

        // a domain/round op forwards the gradient unchanged
        agrees!(
            Round {
                shape: Box2D { b: p(0.8, 0.5) },
                radius: v(0.2)
            },
            [p(1.4, 0.1), p(0.1, 1.0), p(-1.2, -0.2)]
        );

        // 3D
        agrees!(Sphere3D { radius: v(1.0) }, [p3(1.4, 0.3, 0.2), p3(-0.7, 0.9, 0.5)]);
        agrees!(
            Torus3D { ra: v(1.0), rb: v(0.3) },
            [p3(1.6, 0.2, 0.1), p3(0.1, 0.5, 1.5)]
        );
        agrees!(
            Box3D {
                b: p3(1.0, 0.8, 0.6),
                r: v(0.0)
            },
            [p3(1.5, 0.1, 0.0), p3(0.1, 1.3, 0.1), p3(0.0, 0.0, 1.2)]
        );
        // Tier A 3D additions
        agrees!(
            Plane3D {
                n: p3(0.0, 1.0, 0.0),
                h: v(0.0)
            },
            [p3(0.5, 0.7, -0.3), p3(-0.4, -0.6, 0.2)]
        );
        agrees!(
            InfiniteCylinder3D { c: p3(0.0, 0.0, 1.0) },
            [p3(1.5, 0.3, 0.2), p3(0.4, 0.9, 0.5)]
        );
        agrees!(
            VerticalCapsule3D { h: v(1.0), r: v(0.4) },
            [p3(0.7, 0.5, 0.0), p3(0.0, 1.5, 0.3), p3(0.2, 0.5, 0.1)]
        );
        // octahedron-bound: keep components clearly nonzero (sign(p) is the gradient)
        agrees!(OctahedronBound3D { s: v(1.0) }, [p3(0.8, 0.3, 0.2), p3(0.3, 0.2, 0.15)]);
        agrees!(
            RoundedCylinder3D {
                ra: v(1.0),
                rb: v(0.2),
                h: v(0.6)
            },
            [
                p3(1.4, 0.2, 0.1),
                p3(0.3, 1.0, 0.2),
                p3(0.2, 0.1, 0.3),
                p3(0.8, 0.3, 0.0)
            ]
        );
        // Tier C: cut sphere (sphere / cap / rim regions) and unsigned tri/quad
        agrees!(
            CutSphere3D::new(v(1.0), v(0.3)),
            [
                p3(1.4, 0.3, 0.2),
                p3(0.2, 0.6, 0.15),
                p3(1.1, 0.5, 0.0),
                p3(0.0, -1.3, 0.1)
            ]
        );
        agrees!(
            UdTriangle3D {
                a: p3(-1.0, 0.0, 0.0),
                b: p3(1.0, 0.0, 0.0),
                c: p3(0.0, 1.2, 0.3)
            },
            [p3(0.0, 0.4, 0.8), p3(-0.8, 0.1, 0.5), p3(1.2, 0.2, -0.4)]
        );
        agrees!(
            UdQuad3D {
                a: p3(-1.0, -0.8, 0.0),
                b: p3(1.0, -0.8, 0.1),
                c: p3(1.1, 0.8, -0.1),
                d: p3(-0.9, 0.9, 0.0),
            },
            [p3(0.0, 0.0, 0.7), p3(0.5, 0.3, -0.6), p3(-1.3, 0.0, 0.4)]
        );
    }

    // The dimension-generic primitives, exercised at N=4 and N=5 (which no 2D/3D
    // shape reaches). Gradients are cross-checked against finite differences and
    // the generic BoundedSdf containment is sanity-checked.
    #[test]
    fn nd_primitives() {
        use thermite_geometry::prim::Vector as NV;
        let p4 = |a, b, c, d| NV::<V, 4>::new([v(a), v(b), v(c), v(d)]);
        let p5 = |a, b, c, d, e| NV::<V, 5>::new([v(a), v(b), v(c), v(d), v(e)]);

        macro_rules! grad_fd {
            ($shape:expr, [$($pt:expr),+ $(,)?]) => {{
                let shape = $shape;
                let fd = FiniteDiff::with_eps(shape, v(1e-3));
                for q in [$($pt),+] {
                    let (_, ga) = shape.eval_grad(q);
                    let (_, gf) = fd.eval_grad(q);
                    for k in 0..ga.0.len() {
                        assert!(
                            (s(ga[k]) - s(gf[k])).abs() < 5e-2,
                            "{} axis {k}: {} vs {}",
                            stringify!($shape),
                            s(ga[k]),
                            s(gf[k])
                        );
                    }
                }
            }};
        }

        // 4D
        grad_fd!(
            NSphere { radius: v(1.0) },
            [p4(1.2, 0.3, 0.2, 0.4), p4(-0.5, 0.6, 0.3, 0.2)]
        );
        grad_fd!(
            NBox {
                b: p4(1.0, 0.8, 0.6, 0.5)
            },
            [p4(1.4, 0.1, 0.1, 0.1), p4(0.2, 0.3, 0.1, 0.7)]
        );
        grad_fd!(
            NCapsule {
                a: p4(-1.0, 0.0, 0.0, 0.0),
                b: p4(1.0, 0.4, 0.2, 0.1),
                r: v(0.3)
            },
            [p4(0.0, 0.7, 0.3, 0.2), p4(1.3, 0.2, -0.3, 0.1)]
        );
        grad_fd!(
            CrossPolytope { s: v(1.0) },
            [p4(0.6, 0.3, 0.2, 0.15), p4(0.4, 0.25, 0.2, 0.1)]
        );
        // NEllipsoid is an approximate field with IQ's approximate normal, so it is
        // not finite-difference-checkable; just confirm it stays finite and signed.
        let e4 = NEllipsoid {
            r: p4(1.5, 0.8, 1.0, 0.6),
        };
        assert!(s(e4.eval(p4(0.1, 0.0, 0.0, 0.0))) < 0.0); // exact origin is guarded to 0
        assert!(s(e4.eval(p4(2.0, 0.0, 0.0, 0.0))) > 0.0);
        grad_fd!(
            NPlane {
                n: p4(0.5, 0.5, 0.5, 0.5),
                h: v(0.0)
            },
            [p4(0.3, 0.7, -0.2, 0.4)]
        );

        // 5D
        grad_fd!(
            NSphere { radius: v(1.0) },
            [p5(0.9, 0.4, 0.3, 0.2, 0.5), p5(-0.6, 0.5, 0.4, 0.3, 0.2)]
        );
        grad_fd!(
            NBox {
                b: p5(1.0, 0.8, 0.6, 0.5, 0.4)
            },
            [p5(1.3, 0.1, 0.1, 0.1, 0.1), p5(0.2, 0.3, 0.1, 0.1, 0.6)]
        );
        grad_fd!(CrossPolytope { s: v(1.0) }, [p5(0.5, 0.3, 0.2, 0.15, 0.1)]);

        // sign sanity + generic BoundedSdf containment (4D box)
        let b4 = NBox {
            b: p4(1.0, 1.0, 1.0, 1.0),
        };
        assert!(s(b4.eval(p4(0.0, 0.0, 0.0, 0.0))) < 0.0);
        assert!(s(b4.eval(p4(3.0, 3.0, 3.0, 3.0))) > 0.0);
        let bb: thermite_geometry::prim::Bounds<V, 4> = b4.aabb();
        assert!((s(bb.0[3][1]) - 1.0).abs() < 1e-6); // +w extent == half-extent
    }

    // The smooth booleans are validated exactly (no finite differences) by their
    // dominance property: where one operand strictly wins (|da -/+ db| > k, so the
    // blend `h` is zero), the result gradient must equal that operand's gradient
    // (negated for the subtracted operand). Runtime-filtered, so it can never land
    // on the kink that trips a finite difference.
    #[test]
    fn smooth_boolean_gradient_dominance() {
        let a = Circle2D { radius: v(0.7) };
        let b = Segment2D {
            a: p(0.6, -0.8),
            b: p(1.4, 0.7),
            r: v(0.25),
        };
        let k = 0.15f32;
        // The op scales its blend band to 4*k internally, so an operand strictly
        // dominates (h == 0) only past that.
        let band = 4.0 * k;
        let pts = [
            p(-0.95, 0.0),
            p(-0.6, -0.75),
            p(-0.2, -1.1),
            p(0.0, 1.2),
            p(-1.0, 0.5),
            p(1.5, 0.7),
            p(1.0, -0.1),
            p(0.9, 0.45),
        ];
        let close = |g: Vector2<V>, h: Vector2<V>| (s(g[0]) - s(h[0])).abs() < 1e-4 && (s(g[1]) - s(h[1])).abs() < 1e-4;

        let mut checked = 0;
        for q in pts {
            let (da, ga) = a.eval_grad(q);
            let (db, gb) = b.eval_grad(q);
            let (da, db) = (s(da), s(db));

            // union: min(a, b); dominant operand is the closer one
            if (da - db).abs() > band {
                checked += 1;
                let (_, g) = (SmoothUnion { a, b, k: v(k) }).eval_grad(q);
                assert!(close(g, if da < db { ga } else { gb }), "union dominance");
                let (_, g) = (SmoothIntersection { a, b, k: v(k) }).eval_grad(q);
                assert!(close(g, if da > db { ga } else { gb }), "intersection dominance");
            }
            // subtraction: max(-a, b); operands (-a) and b
            if (-da - db).abs() > band {
                checked += 1;
                let (_, g) = (SmoothSubtraction { a, b, k: v(k) }).eval_grad(q);
                let neg_ga = ga * V::NEG_ONE;
                assert!(close(g, if -da > db { neg_ga } else { gb }), "subtraction dominance");
            }
        }
        assert!(checked >= 4, "expected several dominant samples, got {checked}");
    }

    // Exact SDFs satisfy the eikonal equation |grad f| = 1 in smooth regions.
    #[test]
    fn eikonal_unit_gradient() {
        macro_rules! eikonal {
            ($shape:expr, $pts:expr) => {{
                let shape = $shape;
                for q in $pts {
                    let (_, g) = shape.eval_grad(q);
                    let mut sq = 0.0f32;
                    for k in 0..g.0.len() {
                        sq += s(g[k]) * s(g[k]);
                    }
                    assert!(
                        (sq.sqrt() - 1.0).abs() < 2e-2,
                        "{} |grad|={} at non-unit",
                        stringify!($shape),
                        sq.sqrt()
                    );
                }
            }};
        }
        eikonal!(Circle2D { radius: v(1.0) }, [p(1.5, 0.3), p(-0.9, 0.6), p(0.4, -1.4)]);
        eikonal!(Box2D { b: p(1.0, 0.6) }, [p(1.6, 0.1), p(0.1, 1.2), p(-1.5, -0.2)]);
        eikonal!(
            Segment2D {
                a: p(-1.0, 0.0),
                b: p(1.0, 0.5),
                r: v(0.3)
            },
            [p(0.0, 0.9), p(1.5, 0.7), p(-1.4, -0.5)]
        );
        eikonal!(Hexagon2D { r: v(1.0) }, [p(1.5, 0.2), p(-0.2, 1.4), p(0.9, -0.9)]);
        eikonal!(Sphere3D { radius: v(1.0) }, [p3(1.5, 0.2, 0.3), p3(-0.8, 0.9, 0.4)]);
        eikonal!(
            Torus3D { ra: v(1.0), rb: v(0.3) },
            [p3(1.7, 0.2, 0.1), p3(0.1, 0.6, 1.6)]
        );
    }

    #[test]
    fn sdf3d_distances_and_normals() {
        // Sphere: point (3,4,0), r=1 -> distance 4, gradient (0.6,0.8,0)
        let (d, g) = (Sphere3D { radius: v(1.0) }).eval_grad(p3(3.0, 4.0, 0.0));
        check3(d, g, 4.0);
        assert!((s(g[0]) - 0.6).abs() < 1e-4 && (s(g[1]) - 0.8).abs() < 1e-4);

        // Box: outside on +x face
        let (d, g) = (Box3D {
            b: p3(1.0, 1.0, 1.0),
            r: v(0.0),
        })
        .eval_grad(p3(3.0, 0.0, 0.0));
        check3(d, g, 2.0);
        // inside -> negative
        assert!(
            s((Box3D {
                b: p3(1.0, 1.0, 1.0),
                r: v(0.0)
            })
            .eval(p3(0.0, 0.0, 0.0)))
                < 0.0
        );

        // Torus (ra=2, rb=0.5): on the xz-plane at radius 3 -> dist = (3-2)-0.5 = 0.5
        let (d, g) = (Torus3D { ra: v(2.0), rb: v(0.5) }).eval_grad(p3(3.0, 0.0, 0.0));
        check3(d, g, 0.5);

        // Segment along x from (-1,0,0) to (1,0,0), r=0; query above the middle
        let seg = Segment3D {
            a: p3(-1.0, 0.0, 0.0),
            b: p3(1.0, 0.0, 0.0),
            r: v(0.0),
        };
        check3_grad_unit(seg.eval_grad(p3(0.0, 2.0, 0.0)));
        let (d, _) = seg.eval_grad(p3(0.0, 2.0, 0.0));
        assert!((s(d) - 2.0).abs() < 1e-4);

        // Sphere as ellipsoid sanity (r=(1,1,1)) -> on-axis distance ~ |p|-1
        let (d, g) = (Ellipsoid3D { r: p3(1.0, 1.0, 1.0) }).eval_grad(p3(2.0, 0.0, 0.0));
        check3(d, g, 1.0);

        // Vertical cylinder he=2, r=1: point outside the round side
        let cyl = VerticalCylinder3D::from_height(v(2.0), v(1.0));
        let (d, g) = cyl.eval_grad(p3(3.0, 0.0, 0.0));
        check3(d, g, 2.0);

        // Arbitrary cylinder equal to the vertical one (a=(0,-1,0), b=(0,1,0), r=1)
        let cyl2 = Cylinder3D {
            a: p3(0.0, -1.0, 0.0),
            b: p3(0.0, 1.0, 0.0),
            r: v(1.0),
        };
        let (d, g) = cyl2.eval_grad(p3(3.0, 0.0, 0.0));
        check3(d, g, 2.0);

        // Rounded cone and capped cone: just check finite distance + unit normal off-axis
        let rc = RoundCone3D {
            a: p3(0.0, -1.0, 0.0),
            b: p3(0.0, 1.0, 0.0),
            r1: v(0.8),
            r2: v(0.3),
        };
        check3_grad_unit(rc.eval_grad(p3(2.0, 0.5, 0.3)));
        let cc = CappedCone3D {
            he: v(1.0),
            r1: v(1.0),
            r2: v(0.4),
        };
        check3_grad_unit(cc.eval_grad(p3(2.0, 0.2, 0.5)));
        let lk = Link3D {
            le: v(0.5),
            r1: v(1.0),
            r2: v(0.3),
        };
        check3_grad_unit(lk.eval_grad(p3(1.8, 0.4, 0.6)));
    }

    fn check3_grad_unit((d, g): (V, Vector3<V>)) {
        assert!(s(d).is_finite());
        let len = (s(g[0]) * s(g[0]) + s(g[1]) * s(g[1]) + s(g[2]) * s(g[2])).sqrt();
        assert!((len - 1.0).abs() < 1e-3, "gradient length {len} != 1");
    }

    /// 3D `eval` must agree with `eval_grad(..).0` for every shape.
    #[test]
    fn eval3d_matches_eval_grad() {
        let pts = [
            p3(0.7, 1.3, -0.4),
            p3(-1.6, 0.4, 1.1),
            p3(2.2, -0.9, 0.3),
            p3(0.1, 2.5, -1.2),
        ];

        macro_rules! agree {
            ($shape:expr) => {{
                let shape = $shape;
                for q in pts {
                    let a = s(shape.eval(q));
                    let b = s(shape.eval_grad(q).0);
                    assert!(
                        (a - b).abs() < 1e-4 || (a.is_nan() && b.is_nan()),
                        "{} vs {}",
                        a,
                        b
                    );
                }
            }};
        }

        agree!(Sphere3D { radius: v(1.0) });
        agree!(Box3D {
            b: p3(1.0, 0.7, 1.2),
            r: v(0.2)
        });
        agree!(Torus3D { ra: v(1.5), rb: v(0.4) });
        agree!(Segment3D {
            a: p3(-1.0, 0.0, 0.0),
            b: p3(1.0, 0.5, 0.2),
            r: v(0.1)
        });
        agree!(Ellipsoid3D { r: p3(1.5, 0.8, 1.1) });
        agree!(Link3D {
            le: v(0.5),
            r1: v(1.0),
            r2: v(0.3)
        });
        agree!(RoundCone3D {
            a: p3(0.0, -1.0, 0.0),
            b: p3(0.3, 1.0, 0.1),
            r1: v(0.8),
            r2: v(0.3)
        });
        agree!(CappedCone3D {
            he: v(1.0),
            r1: v(1.0),
            r2: v(0.4)
        });
        agree!(VerticalCylinder3D::from_height(v(2.0), v(1.0)));
        agree!(Cylinder3D {
            a: p3(0.0, -1.0, 0.0),
            b: p3(0.0, 1.0, 0.0),
            r: v(1.0)
        });
    }

    /// Grid-sample a 2D shape over [-range, range]^2 and assert every interior
    /// point (eval <= 0) lies within the claimed AABB.
    fn aabb_contains_2d<S: BoundedSdf<V, 2>>(shape: &S, range: f32) {
        let bb = shape.aabb();
        let (xmin, xmax) = (s(bb.0[0][0]), s(bb.0[0][1]));
        let (ymin, ymax) = (s(bb.0[1][0]), s(bb.0[1][1]));
        let n = 60;
        let tol = 1e-2;
        for i in 0..=n {
            for j in 0..=n {
                let x = -range + 2.0 * range * (i as f32 / n as f32);
                let y = -range + 2.0 * range * (j as f32 / n as f32);
                if s(shape.eval(p(x, y))) <= 0.0 {
                    assert!(
                        x >= xmin - tol && x <= xmax + tol && y >= ymin - tol && y <= ymax + tol,
                        "interior ({x},{y}) outside aabb x[{xmin},{xmax}] y[{ymin},{ymax}]"
                    );
                }
            }
        }
    }

    fn aabb_contains_3d<S: BoundedSdf<V, 3>>(shape: &S, range: f32) {
        let bb = shape.aabb();
        let c = |k: usize| (s(bb.0[k][0]), s(bb.0[k][1]));
        let (bx, by, bz) = (c(0), c(1), c(2));
        let n = 32;
        let tol = 1e-2;
        for i in 0..=n {
            for j in 0..=n {
                for kk in 0..=n {
                    let g = |t: usize| -range + 2.0 * range * (t as f32 / n as f32);
                    let (x, y, z) = (g(i), g(j), g(kk));
                    if s(shape.eval(p3(x, y, z))) <= 0.0 {
                        assert!(
                            x >= bx.0 - tol
                                && x <= bx.1 + tol
                                && y >= by.0 - tol
                                && y <= by.1 + tol
                                && z >= bz.0 - tol
                                && z <= bz.1 + tol,
                            "interior ({x},{y},{z}) outside aabb"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn aabb_contains_2d_primitives() {
        aabb_contains_2d(&Circle2D { radius: v(1.3) }, 3.0);
        aabb_contains_2d(&Box2D { b: p(1.2, 0.7) }, 3.0);
        aabb_contains_2d(&Ellipse2D { ab: p(1.5, 0.8) }, 3.0);
        aabb_contains_2d(&Cross2D { b: p(1.5, 0.4) }, 3.0);
        aabb_contains_2d(
            &Segment2D {
                a: p(-1.0, -0.5),
                b: p(1.2, 0.8),
                r: v(0.3),
            },
            3.0,
        );
        aabb_contains_2d(&Vesica2D::from_circle(v(1.2), v(0.5)), 3.0);
        aabb_contains_2d(
            &Trapezoid2D {
                ra: v(1.0),
                rb: v(0.5),
                he: v(0.8),
            },
            3.0,
        );
        aabb_contains_2d(&IsoscelesTriangle2D { q: p(0.8, 1.4) }, 3.0);
        aabb_contains_2d(
            &Triangle2D {
                v: [p(-1.0, -0.8), p(1.1, -0.6), p(0.2, 1.3)],
            },
            3.0,
        );
        aabb_contains_2d(
            &Quad2D {
                v: [p(-1.0, -1.0), p(1.0, -1.0), p(1.2, 1.0), p(-0.8, 0.9)],
            },
            3.0,
        );
        aabb_contains_2d(
            &Pie2D {
                radius: v(1.3),
                sc: p(0.7, 0.7),
            },
            3.0,
        );
        aabb_contains_2d(
            &Arc2D {
                sc: p(0.7, 0.7),
                ra: v(1.0),
                rb: v(0.25),
            },
            3.0,
        );
        aabb_contains_2d(&Hexagon2D { r: v(1.1) }, 3.0);
        aabb_contains_2d(&Moon2D::new(v(0.6), v(1.2), v(0.9)), 3.0);
        // wide apertures (> 90 deg, cos < 0) exercise the other box branch
        aabb_contains_2d(
            &Pie2D {
                radius: v(1.3),
                sc: p(0.866, -0.5), // 120 deg half-aperture
            },
            3.0,
        );
        aabb_contains_2d(
            &Arc2D {
                sc: p(0.866, -0.5),
                ra: v(1.0),
                rb: v(0.25),
            },
            3.0,
        );
    }

    #[test]
    fn aabb_contains_new_2d_primitives() {
        aabb_contains_2d(
            &RoundedBox2D {
                b: p(1.2, 0.8),
                r: [v(0.3), v(0.1), v(0.4), v(0.2)],
            },
            3.0,
        );
        aabb_contains_2d(
            &ChamferBox2D {
                b: p(1.0, 0.7),
                chamfer: v(0.3),
            },
            3.0,
        );
        aabb_contains_2d(
            &OrientedBox2D {
                a: p(-1.0, -0.5),
                b: p(1.0, 0.6),
                th: v(0.6),
            },
            3.0,
        );
        aabb_contains_2d(&Rhombus2D { b: p(1.3, 0.8) }, 3.0);
        aabb_contains_2d(
            &Parallelogram2D {
                wi: v(1.0),
                he: v(0.7),
                sk: v(0.4),
            },
            3.0,
        );
        aabb_contains_2d(&UnevenCapsule2D::new(v(0.7), v(0.3), v(1.2)), 3.5);
        aabb_contains_2d(&Pentagon2D { r: v(1.1) }, 3.0);
        aabb_contains_2d(&Octagon2D { r: v(1.1) }, 3.0);
        aabb_contains_2d(
            &Polygon2D {
                v: [p(-1.0, -0.8), p(1.1, -0.6), p(0.7, 0.9), p(-0.5, 1.2)],
            },
            3.0,
        );
        aabb_contains_2d(&CutDisk2D::new(v(1.2), v(0.4)), 3.0);
        aabb_contains_2d(&CutDisk2D::new(v(1.2), v(-0.3)), 3.0);
        aabb_contains_2d(
            &Ring2D {
                n: p(1.0, 0.0),
                r: v(1.0),
                th: v(0.3),
            },
            3.0,
        );
        aabb_contains_2d(&Heart2D, 2.0);
        aabb_contains_2d(
            &OrientedVesica2D {
                a: p(-1.0, -0.3),
                b: p(1.0, 0.4),
                w: v(0.5),
            },
            3.0,
        );
        aabb_contains_2d(
            &Horseshoe2D {
                c: p(0.6, 0.8), // unit (cos, sin)
                r: v(1.0),
                w: p(0.3, 0.2),
            },
            3.0,
        );
    }

    #[test]
    fn batch2c_cubic() {
        // curve distances: finite, ~0 on the curve, correct sign where applicable
        let par = ParabolaSegment2D::<V>::new(v(1.0), v(1.0));
        assert!(s(par.eval(p(0.0, 1.0))).abs() < 5e-3); // vertex on the curve
        assert!(s(par.eval(p(0.5, 5.0))).is_finite());

        let bez = QuadraticBezier2D::<V>::new(p(-1.0, 0.0), p(0.0, 1.5), p(1.0, 0.0));
        assert!(s(bez.eval(p(0.0, 0.75))).abs() < 5e-3); // on the curve (B(0.5))
        assert!(s(bez.eval(p(2.0, 2.0))) > 0.0);

        let bc = BlobbyCross2D::<V>::new(v(1.0));
        assert!(s(bc.eval(p(0.3, 0.2))).is_finite());

        let qc = QuadraticCircle2D::<V>::new();
        assert!(s(qc.eval(p(0.1, 0.05))) < 0.0); // near center inside
        assert!(s(qc.eval(p(2.0, 2.0))) > 0.0);

        let hy = Hyperbola2D::<V>::new(v(0.5), v(2.0));
        assert!(s(hy.eval(p(1.0, 1.0))).is_finite());
    }

    #[test]
    fn batch2b_trig() {
        use thermite::math::policy::DefaultPolicy;
        let star = Star2D::<V>::from_params(v(1.0), 5, v(3.0));
        aabb_contains_2d(&star, 2.0);
        assert!(s(star.eval(p(0.0, 0.0))) < 0.0); // center is inside
        assert!(s(star.eval(p(5.0, 5.0))) > 0.0);

        let cw = CircleWave2D::from_params::<DefaultPolicy>(v(1.0), v(0.5));
        assert!(s(cw.eval(p(0.3, 0.2))).is_finite());
        assert!(s(cw.eval(p(10.0, 0.1))).is_finite()); // periodic in x
    }

    #[test]
    fn batch2a_sanity() {
        // finite everywhere + positive far away; closed shapes negative at center
        macro_rules! far_pos {
            ($shape:expr) => {{
                let sh = $shape;
                assert!(s(sh.eval(p(0.0, 0.0))).is_finite());
                assert!(s(sh.eval(p(6.0, 6.0))) > 0.0);
            }};
        }
        far_pos!(Pentagram2D { r: v(1.0) });
        far_pos!(Horseshoe2D {
            c: p(0.7, 0.7),
            r: v(1.0),
            w: p(0.3, 0.2)
        });
        far_pos!(OrientedVesica2D {
            a: p(-1.0, 0.0),
            b: p(1.0, 0.0),
            w: v(0.5)
        });
        far_pos!(RoundedCross2D { h: v(0.5) });
        far_pos!(Egg2D::new(v(1.0), v(0.7), v(0.4), v(1.0)));
        far_pos!(Tunnel2D { wh: p(0.8, 1.2) });
        far_pos!(Stairs2D {
            wh: p(0.5, 0.5),
            n: v(4.0)
        });
        far_pos!(CoolS2D);

        // interiors
        assert!(s((Pentagram2D { r: v(1.0) }).eval(p(0.0, 0.0))) < 0.0);
        assert!(s((Egg2D::new(v(1.0), v(0.7), v(0.4), v(1.0))).eval(p(0.0, 0.0))) < 0.0);
        assert!(
            s((OrientedVesica2D {
                a: p(-1.0, 0.0),
                b: p(1.0, 0.0),
                w: v(0.5)
            })
            .eval(p(0.0, 0.0)))
                < 0.0
        );
        assert!(s((RoundedCross2D { h: v(0.5) }).eval(p(0.0, 0.0))) < 0.0);
    }

    #[test]
    fn new_2d_primitives_sign() {
        // shapes without a bbox impl: interior negative, exterior positive
        let et = EquilateralTriangle2D { r: v(1.0) };
        assert!(s(et.eval(p(0.0, 0.0))) < 0.0);
        assert!(s(et.eval(p(3.0, 3.0))) > 0.0);

        let hx = Hexagram2D { r: v(1.0) };
        assert!(s(hx.eval(p(0.0, 0.0))) < 0.0);
        assert!(s(hx.eval(p(4.0, 4.0))) > 0.0);

        let rx = RoundedX2D { w: v(1.0), r: v(0.1) };
        assert!(s(rx.eval(p(0.0, 0.0))) < 0.0);
        assert!(s(rx.eval(p(3.0, 0.0))) > 0.0);

        // RoundedBox face point is on the surface
        let rb = RoundedBox2D {
            b: p(1.0, 1.0),
            r: [v(0.2), v(0.2), v(0.2), v(0.2)],
        };
        assert!(s(rb.eval(p(1.0, 0.0))).abs() < 1e-5);
    }

    #[test]
    fn aabb_contains_3d_primitives() {
        aabb_contains_3d(&Sphere3D { radius: v(1.2) }, 2.5);
        aabb_contains_3d(
            &Box3D {
                b: p3(1.0, 0.7, 1.2),
                r: v(0.2),
            },
            2.5,
        );
        aabb_contains_3d(&Ellipsoid3D { r: p3(1.4, 0.8, 1.1) }, 2.5);
        aabb_contains_3d(&Torus3D { ra: v(1.2), rb: v(0.4) }, 2.5);
        aabb_contains_3d(
            &Link3D {
                le: v(0.5),
                r1: v(0.9),
                r2: v(0.3),
            },
            2.5,
        );
        aabb_contains_3d(
            &Segment3D {
                a: p3(-0.8, -0.5, 0.2),
                b: p3(0.9, 0.7, -0.3),
                r: v(0.3),
            },
            2.5,
        );
        aabb_contains_3d(
            &RoundCone3D {
                a: p3(-0.5, -1.0, 0.0),
                b: p3(0.4, 1.0, 0.2),
                r1: v(0.7),
                r2: v(0.3),
            },
            2.5,
        );
        aabb_contains_3d(
            &CappedCone3D {
                he: v(1.0),
                r1: v(1.0),
                r2: v(0.4),
            },
            2.5,
        );
        aabb_contains_3d(&VerticalCylinder3D::from_height(v(2.0), v(0.9)), 2.5);
        aabb_contains_3d(
            &Cylinder3D {
                a: p3(-0.5, -0.8, 0.1),
                b: p3(0.6, 0.9, -0.2),
                r: v(0.5),
            },
            2.5,
        );
    }

    #[test]
    fn aabb_contains_new_3d_primitives() {
        aabb_contains_3d(
            &BoxFrame3D {
                b: p3(1.0, 0.8, 1.2),
                e: v(0.15),
            },
            3.0,
        );
        aabb_contains_3d(&HexPrism3D { h: p(1.0, 0.6) }, 3.0);
        aabb_contains_3d(&VerticalCapsule3D { h: v(1.0), r: v(0.5) }, 3.0);
        aabb_contains_3d(
            &RoundedCylinder3D {
                ra: v(1.0),
                rb: v(0.2),
                h: v(0.8),
            },
            3.0,
        );
        aabb_contains_3d(&CutSphere3D::new(v(1.2), v(0.4)), 3.0);
        aabb_contains_3d(&CutHollowSphere3D::new(v(1.0), v(0.2), v(0.1)), 3.0);
        aabb_contains_3d(&DeathStar3D::new(v(1.2), v(0.7), v(1.0)), 3.0);
        aabb_contains_3d(&Octahedron3D { s: v(1.2) }, 2.5);
        aabb_contains_3d(&OctahedronBound3D { s: v(1.2) }, 2.5);
        aabb_contains_3d(&RoundCone3DVert::new(v(0.7), v(0.3), v(1.0)), 3.0);
        aabb_contains_3d(&Pyramid3D { h: v(1.0) }, 2.0);
        aabb_contains_3d(
            &Rhombus3D {
                la: v(1.0),
                lb: v(0.6),
                h: v(0.3),
                ra: v(0.1),
            },
            2.5,
        );
        aabb_contains_3d(
            &SolidAngle3D {
                c: p(0.6, 0.8),
                ra: v(1.2),
            },
            2.5,
        );
        aabb_contains_3d(
            &CappedTorus3D {
                sc: p(0.8, 0.6),
                ra: v(1.0),
                rb: v(0.3),
            },
            3.0,
        );
        aabb_contains_3d(
            &Cone3DVert {
                c: p(0.6, 0.8),
                h: v(1.0),
            },
            2.5,
        );
    }

    #[test]
    fn batch3_sanity() {
        // unbounded / unsigned shapes: finite, and the documented behaviour
        assert!(
            s((Plane3D {
                n: p3(0.0, 1.0, 0.0),
                h: v(0.5)
            })
            .eval(p3(0.0, -0.5, 0.0)))
            .abs()
                < 1e-5
        );
        assert!(s((InfiniteCylinder3D { c: p3(0.0, 0.0, 1.0) }).eval(p3(1.0, 5.0, 0.0))).abs() < 1e-5);
        let ic = InfiniteCone3D { c: p(0.5, 0.866) };
        assert!(s(ic.eval(p3(1.0, -1.0, 0.0))).is_finite());
        assert!(s((TriPrism3D { h: p(1.0, 0.5) }).eval(p3(0.0, 0.0, 0.0))) < 0.0);

        let vs = VesicaSegment3D {
            a: p3(-1.0, 0.0, 0.0),
            b: p3(1.0, 0.0, 0.0),
            w: v(0.5),
        };
        assert!(s(vs.eval(p3(0.0, 0.0, 0.0))) < 0.0);
        let cc = CappedConeAB3D {
            a: p3(0.0, -1.0, 0.0),
            b: p3(0.0, 1.0, 0.0),
            ra: v(0.8),
            rb: v(0.4),
        };
        assert!(s(cc.eval(p3(0.0, 0.0, 0.0))) < 0.0);
        assert!(s(cc.eval(p3(3.0, 0.0, 0.0))) > 0.0);

        // unsigned triangle/quad: zero at a vertex, finite elsewhere
        let tri = UdTriangle3D {
            a: p3(-1.0, 0.0, 0.0),
            b: p3(1.0, 0.0, 0.0),
            c: p3(0.0, 1.0, 0.5),
        };
        assert!(s(tri.eval(p3(-1.0, 0.0, 0.0))).abs() < 1e-4);
        assert!(s(tri.eval(p3(0.0, 0.3, 3.0))) > 0.0);
        let quad = UdQuad3D {
            a: p3(-1.0, -1.0, 0.0),
            b: p3(1.0, -1.0, 0.0),
            c: p3(1.0, 1.0, 0.0),
            d: p3(-1.0, 1.0, 0.0),
        };
        assert!((s(quad.eval(p3(0.0, 0.0, 1.0))) - 1.0).abs() < 1e-4); // 1 unit off the plane
    }

    #[test]
    fn aabb_operators() {
        // Round expands the box by the radius
        aabb_contains_2d(
            &Round {
                shape: Box2D { b: p(1.0, 0.6) },
                radius: v(0.3),
            },
            3.0,
        );
        // union and intersection of two boxes
        aabb_contains_2d(
            &Union {
                a: Circle2D { radius: v(1.0) },
                b: Box2D { b: p(0.5, 1.5) },
            },
            3.0,
        );
        aabb_contains_2d(
            &Intersection {
                a: Box2D { b: p(2.0, 0.5) },
                b: Box2D { b: p(0.5, 2.0) },
            },
            3.0,
        );
        // smooth union bulges outward; box is expanded by k
        aabb_contains_2d(
            &SmoothUnion {
                a: Circle2D { radius: v(1.0) },
                b: Box2D { b: p(0.5, 1.0) },
                k: v(0.2),
            },
            3.0,
        );
        // and in 3D
        aabb_contains_3d(
            &Union {
                a: Sphere3D { radius: v(1.0) },
                b: Box3D {
                    b: p3(0.6, 1.4, 0.6),
                    r: v(0.1),
                },
            },
            2.5,
        );
    }

    #[test]
    fn alternate_constructors() {
        // Vesica: from_circle(r,d) == from_size(half_width=r-d, half_height=sqrt(r^2-d^2))
        let va = Vesica2D::from_circle(v(1.2), v(0.5));
        let vb = Vesica2D::from_size(v(0.7), v(1.090_871_2)); // sqrt(1.2^2 - 0.5^2)
        for q in [p(0.3, 0.8), p(-0.6, 0.2), p(0.1, 1.0)] {
            assert!((s(va.eval(q)) - s(vb.eval(q))).abs() < 1e-4);
        }

        // VerticalCylinder: from_height(2h) == from_half_height(h)
        let ca = VerticalCylinder3D::from_height(v(2.0), v(0.9));
        let cb = VerticalCylinder3D::from_half_height(v(1.0), v(0.9));
        for q in [p3(0.5, 0.3, 0.2), p3(1.2, 1.5, 0.0)] {
            assert!((s(ca.eval(q)) - s(cb.eval(q))).abs() < 1e-5);
        }

        // angle constructor matches a hand-built sin/cos for the cone family
        use thermite::math::policy::DefaultPolicy;
        let h = core::f32::consts::FRAC_1_SQRT_2; // sin(pi/4) == cos(pi/4)
        let sa = SolidAngle3D::from_angle::<DefaultPolicy>(v(core::f32::consts::FRAC_PI_4), v(1.0));
        let sb = SolidAngle3D { c: p(h, h), ra: v(1.0) };
        assert!((s(sa.eval(p3(0.4, 0.6, 0.2))) - s(sb.eval(p3(0.4, 0.6, 0.2)))).abs() < 1e-4);
    }

    #[test]
    fn domain_ops() {
        use thermite::math::policy::DefaultPolicy;
        // uniform scale: circle r=1 scaled by 2 == circle r=2
        let sc = Scale {
            shape: Circle2D { radius: v(1.0) },
            scale: v(2.0),
        };
        assert!((s(sc.eval(p(6.0, 8.0))) - 8.0).abs() < 1e-4); // |p|=10 -> 10-2
        aabb_contains_2d(&sc, 4.0);

        // elongate a circle into a horizontal capsule
        let el = Elongate {
            shape: Circle2D { radius: v(1.0) },
            h: p(1.0, 0.0),
        };
        assert!(s(el.eval(p(0.0, 0.0))) < 0.0);
        assert!((s(el.eval(p(0.0, 3.0))) - 2.0).abs() < 1e-4); // 1 above the capsule top

        // symmetry / repetition just need to run and stay finite
        let sym = Symmetry {
            shape: Circle2D { radius: v(1.0) },
            axes: [true, false],
        };
        assert!(s(sym.eval(p(2.0, 0.5))).is_finite());
        let rep = Repetition {
            shape: Circle2D { radius: v(0.3) },
            spacing: p(2.0, 2.0),
        };
        assert!(s(rep.eval(p(4.1, 0.0))).is_finite() && s(rep.eval(p(4.0, 0.0))) < 0.0);

        // extrude a disk -> cylinder; revolve a disk -> torus
        let ex = Extrusion {
            shape: Circle2D { radius: v(1.0) },
            h: v(0.5),
        };
        assert!(s(ex.eval(p3(0.0, 0.0, 0.0))) < 0.0);
        assert!((s(ex.eval(p3(2.0, 0.0, 0.0))) - 1.0).abs() < 1e-4);
        let rev = Revolution {
            shape: Circle2D { radius: v(0.3) },
            offset: v(1.0),
        };
        assert!((s(rev.eval(p3(1.0, 0.0, 0.0))) - -0.3).abs() < 1e-4); // tube center

        // displacement field added to the distance
        let dis = Displace {
            shape: Sphere3D { radius: v(1.0) },
            displacement: |q: Vector3<V>| q[0] * v(0.05),
        };
        assert!((s(dis.eval(p3(2.0, 0.0, 0.0))) - 1.1).abs() < 1e-4); // (|p|-1) + 2*0.05

        // distortions: finite
        let tw = Twist::<V, _>::new(
            Box3D {
                b: p3(0.5, 1.0, 0.5),
                r: v(0.0),
            },
            v(1.0),
        );
        assert!(s(tw.eval(p3(0.3, 0.2, 0.1))).is_finite());
        let bd = Bend::<V, _, DefaultPolicy>::new(
            Box3D {
                b: p3(1.0, 0.3, 0.3),
                r: v(0.0),
            },
            v(0.5),
        );
        assert!(s(bd.eval(p3(0.2, 0.1, 0.05))).is_finite());
    }

    #[test]
    fn operators() {
        let c = Circle2D { radius: v(1.0) };
        // rounding shrinks the field by the radius
        let r = Round {
            shape: Circle2D { radius: v(1.0) },
            radius: v(0.25),
        };
        let (d, _) = r.eval_grad(p(3.0, 4.0));
        assert!((s(d) - 3.75).abs() < 1e-4);

        // onion turns the disk into a ring
        let o = Onion {
            shape: Circle2D { radius: v(2.0) },
            radius: v(0.5),
        };
        let (d, _) = o.eval_grad(p(2.0, 0.0)); // exactly on the circle -> abs(0) - 0.5
        assert!((s(d) - -0.5).abs() < 1e-4);

        // union of two circles
        let u = Union {
            a: c,
            b: Circle2D { radius: v(1.0) },
        };
        let (d, _) = u.eval_grad(p(3.0, 0.0));
        assert!((s(d) - 2.0).abs() < 1e-4);

        // smooth-min dips below the hard min near the seam
        let sm = SmoothUnion {
            a: Circle2D { radius: v(1.0) },
            b: Circle2D { radius: v(1.0) },
            k: v(0.1),
        };
        let (d, _) = sm.eval_grad(p(0.5, 0.0));
        // hard min here would be -0.5; the smooth blend pushes it lower
        assert!(s(d) < -0.5);

        // intersection (max) of a wide box and a tall box -> a small plus-center square
        let inter = Intersection {
            a: Box2D { b: p(2.0, 0.5) },
            b: Box2D { b: p(0.5, 2.0) },
        };
        let (d, _) = inter.eval_grad(p(0.0, 0.0)); // inside both -> max(-0.5,-0.5)
        assert!((s(d) - -0.5).abs() < 1e-4);
        let (d, _) = inter.eval_grad(p(1.5, 0.0)); // outside the tall box -> +1.0 binds
        assert!((s(d) - 1.0).abs() < 1e-4);

        // smooth-max rises above the hard max near the seam
        let sx = SmoothIntersection {
            a: Circle2D { radius: v(1.0) },
            b: Circle2D { radius: v(1.0) },
            k: v(0.1),
        };
        let (d, _) = sx.eval_grad(p(0.5, 0.0));
        assert!(s(d) > -0.5);
    }
}
