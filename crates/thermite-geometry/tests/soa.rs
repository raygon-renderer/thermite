//! Behavioural tests for the SoA primitives.
//!
//! Everything runs on the 1-lane `thermite::Vector<f32>` so results can be read
//! back as plain scalars. The code under test is lane-count generic, so a passing
//! 1-lane run exercises the same arithmetic every wider backend runs.

use thermite::prelude::*;

use thermite_geometry::soa::prim::{
    Bounds, Matrix, Point, Vector as GVector,
    matrix::gamma,
    vector::VectorOps as _,
};

/// 1-lane scalar vector, the simplest concrete `FloatVector`.
type V = thermite::Vector<f32>;

#[inline]
fn v(x: f32) -> V {
    V::splat(x)
}

/// Read lane 0 back out as a plain `f32`.
#[inline]
fn s(x: V) -> f32 {
    x.extract::<0>()
}

fn vec3(x: f32, y: f32, z: f32) -> GVector<V, 3> {
    GVector::new([v(x), v(y), v(z)])
}

fn pt3(x: f32, y: f32, z: f32) -> Point<V, 3> {
    Point::new([v(x), v(y), v(z)])
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[track_caller]
fn assert_vec3(actual: GVector<V, 3>, expected: [f32; 3]) {
    for i in 0..3 {
        assert!(
            close(s(actual[i]), expected[i]),
            "component {i}: {} != {}",
            s(actual[i]),
            expected[i]
        );
    }
}

#[track_caller]
fn assert_point3(actual: Point<V, 3>, expected: [f32; 3]) {
    assert_vec3(GVector::from(actual), expected);
}

// ---------------------------------------------------------------- Vector ----

#[test]
fn vector_ops() {
    let a = vec3(1.0, 2.0, 3.0);
    let b = vec3(4.0, 5.0, 6.0);

    assert_vec3(-a, [-1.0, -2.0, -3.0]);
    assert_vec3(a + b, [5.0, 7.0, 9.0]);
    assert_vec3(b - a, [3.0, 3.0, 3.0]);
    assert_vec3(a * v(2.0), [2.0, 4.0, 6.0]);

    // The assign forms must agree with the by-value ones.
    let mut c = a;
    c += b;
    assert_vec3(c, [5.0, 7.0, 9.0]);
    c -= b;
    assert_vec3(c, [1.0, 2.0, 3.0]);
    c *= v(3.0);
    assert_vec3(c, [3.0, 6.0, 9.0]);
    c /= v(3.0);
    assert_vec3(c, [1.0, 2.0, 3.0]);

    assert_vec3(a.mix(b, v(0.5)), [2.5, 3.5, 4.5]);
    assert_vec3(a.cross(b), [-3.0, 6.0, -3.0]);

    assert!(close(s(a.dot(&b)), 32.0));
    assert!(close(s(a.norm_sqr()), 14.0));
    assert!(close(s(a.l2_norm()), 14.0f32.sqrt()));
    assert!(close(s(a.l1_norm()), 6.0));
    assert!(close(s(vec3(-1.0, 4.0, -2.0).linf_norm()), 4.0));

    assert!(a.is_finite().all());
    assert!(!vec3(1.0, f32::NAN, 3.0).is_finite().all());
    assert!(vec3(1.0, f32::NAN, 3.0).is_nan().any());
}

#[test]
fn vector_normalize() {
    let a = vec3(3.0, 4.0, 0.0);

    assert_vec3(a.normalize(), [0.6, 0.8, 0.0]);

    let (unit, norm) = a.normalize_norm();
    assert_vec3(unit, [0.6, 0.8, 0.0]);
    assert!(close(s(norm), 5.0));

    // A zero-length vector has no direction: try_normalize must give zero, not NaN.
    assert_vec3(GVector::<V, 3>::ZERO.try_normalize(), [0.0, 0.0, 0.0]);
    assert_vec3(a.try_normalize(), [0.6, 0.8, 0.0]);

    // ... whereas the plain normalize really does produce NaN there, which is why
    // try_normalize exists.
    assert!(GVector::<V, 3>::ZERO.normalize().is_nan().any());
}

#[test]
fn vector2_perp() {
    let a = GVector::<V, 2>::new([v(1.0), v(0.0)]);
    let p = a.perp();

    assert!(close(s(p[0]), 0.0) && close(s(p[1]), 1.0));
    // perp is a quarter turn, so it is orthogonal to the original.
    assert!(close(s(a.dot(&p)), 0.0));
}

// ------------------------------------------------------- shading vectors ----

#[test]
fn vector_reflect() {
    let n = vec3(0.0, 1.0, 0.0);

    // Incident points into the surface (GLSL convention), so the normal component
    // flips and the tangential component survives.
    assert_vec3(vec3(1.0, -1.0, 0.0).reflect(&n), [1.0, 1.0, 0.0]);

    // Straight down reflects straight back up.
    assert_vec3(vec3(0.0, -1.0, 0.0).reflect(&n), [0.0, 1.0, 0.0]);

    // Grazing along the surface is untouched: no normal component to flip.
    assert_vec3(vec3(1.0, 0.0, 0.0).reflect(&n), [1.0, 0.0, 0.0]);
}

#[test]
fn vector_reflect_preserves_length_and_angle() {
    let n = vec3(0.0, 0.0, 1.0);
    let i = vec3(0.3, -0.5, -0.8).normalize();

    let r = i.reflect(&n);

    // A reflection is an isometry.
    assert!(close(s(r.l2_norm()), 1.0));

    // Angle of incidence equals angle of reflection: the normal components are
    // negatives, the tangential components identical.
    assert!(close(s(i.dot(&n)), -s(r.dot(&n))));

    // Reflecting twice is the identity.
    assert_vec3(r.reflect(&n), [s(i[0]), s(i[1]), s(i[2])]);
}

#[test]
fn vector_refract_snell() {
    let n = vec3(0.0, 1.0, 0.0);

    // Straight-on incidence passes through undeviated regardless of eta.
    assert_vec3(vec3(0.0, -1.0, 0.0).refract(&n, v(0.75)), [0.0, -1.0, 0.0]);

    // 45 degrees into a denser medium (eta < 1) bends toward the normal, so the
    // transmitted angle is strictly smaller. Snell: sin(t) = eta * sin(i).
    let eta = 0.5;
    let i = vec3(1.0, -1.0, 0.0).normalize();
    let t = i.refract(&n, v(eta));

    assert!(close(s(t.l2_norm()), 1.0), "refracted ray must stay unit length");

    let sin_i = (1.0 - s(i.dot(&n)).powi(2)).sqrt();
    let sin_t = (1.0 - s(t.dot(&n)).powi(2)).sqrt();

    assert!(close(sin_t, eta * sin_i), "Snell's law: {sin_t} != {}", eta * sin_i);
    // Bent toward the normal, and still travelling into the surface.
    assert!(sin_t < sin_i);
    assert!(s(t[1]) < 0.0);
}

#[test]
fn vector_refract_total_internal_reflection() {
    let n = vec3(0.0, 1.0, 0.0);

    // Leaving a denser medium (eta > 1) past the critical angle: no solution.
    // Critical angle for eta = 2 is 30 degrees, so 60 degrees is well past it.
    let steep = vec3(3.0f32.sqrt(), -1.0, 0.0).normalize();

    let (t, ok) = steep.refract_mask(&n, v(2.0));

    assert!(ok.none(), "60 degrees at eta = 2 must totally internally reflect");
    assert_vec3(t, [0.0, 0.0, 0.0]);
    // The plain form agrees with the masked one.
    assert_vec3(steep.refract(&n, v(2.0)), [0.0, 0.0, 0.0]);

    // Just inside the critical angle still refracts, and stays finite.
    let shallow = vec3(0.2, -1.0, 0.0).normalize();
    let (t, ok) = shallow.refract_mask(&n, v(2.0));

    assert!(ok.all(), "shallow incidence at eta = 2 must refract");
    assert!(close(s(t.l2_norm()), 1.0));
}

#[test]
fn vector_faceforward() {
    let n = vec3(0.0, 1.0, 0.0);

    // Incident opposes the reference normal: already correct, kept as-is.
    let facing = vec3(0.0, -1.0, 0.0);
    assert_vec3(n.faceforward(&facing, &n), [0.0, 1.0, 0.0]);

    // Incident agrees with the reference (hitting the back face): flipped.
    let behind = vec3(0.0, 1.0, 0.0);
    assert_vec3(n.faceforward(&behind, &n), [0.0, -1.0, 0.0]);

    // The reference is what decides, so a shading normal can be flipped by the
    // geometric one rather than by itself.
    let shading = vec3(0.1, 1.0, 0.0);
    let flipped = shading.faceforward(&behind, &n);
    assert_vec3(flipped, [-0.1, -1.0, 0.0]);

    // Whatever the outcome, the result opposes the incident direction.
    assert!(s(flipped.dot(&behind)) < 0.0);
}

/// GLSL/GLM refract in f64, the same reference the core crate's
/// `diff_linalg_ext` suite checks `LinAlg3Vector::refract` against, so the SoA
/// implementation here is pinned to the identical semantics.
fn refract3_ref(i: [f64; 3], n: [f64; 3], eta: f64) -> ([f64; 3], f64) {
    let d = n[0] * i[0] + n[1] * i[1] + n[2] * i[2];
    let k = 1.0 - eta * eta * (1.0 - d * d);

    if k < 0.0 {
        return ([0.0; 3], k);
    }

    let c = eta * d + k.sqrt();

    (core::array::from_fn(|j| eta * i[j] - c * n[j]), k)
}

#[test]
fn vector_refract_matches_glsl_reference() {
    // Spread over both sides of the critical angle and both eta directions.
    let cases = [
        ([0.0f32, -1.0, 0.0], 0.5f32),
        ([0.6, -0.8, 0.0], 0.75),
        ([0.8, -0.6, 0.0], 1.5),
        ([0.9, -0.436_435, 0.0], 2.0),
        ([0.3, -0.6, 0.741_62], 1.0),
        ([-0.5, -0.5, core::f32::consts::FRAC_1_SQRT_2], 1.33),
    ];

    let n = vec3(0.0, 1.0, 0.0);

    for (raw, eta) in cases {
        let i = vec3(raw[0], raw[1], raw[2]).normalize();
        let iv = [s(i[0]) as f64, s(i[1]) as f64, s(i[2]) as f64];

        let (want, k) = refract3_ref(iv, [0.0, 1.0, 0.0], eta as f64);

        let (got, ok) = i.refract_mask(&n, v(eta));

        assert_eq!(
            ok.all(),
            k >= 0.0,
            "TIR disagreement for i = {raw:?}, eta = {eta}: k = {k}"
        );

        for j in 0..3 {
            assert!(
                close(s(got[j]), want[j] as f32),
                "component {j} for i = {raw:?}, eta = {eta}: {} != {}",
                s(got[j]),
                want[j]
            );
        }
    }
}

// ----------------------------------------------------------------- Point ----

#[test]
fn point_ops() {
    let a = pt3(1.0, 2.0, 3.0);
    let b = pt3(4.0, 6.0, 3.0);

    assert!(close(s(a.distance(b)), 5.0));
    assert!(close(s(a.distance_sqr(b)), 25.0));
    assert_point3(a.midpoint(b), [2.5, 4.0, 3.0]);
    assert_point3(a.min(b), [1.0, 2.0, 3.0]);
    assert_point3(a.max(b), [4.0, 6.0, 3.0]);

    // Affine algebra. A point minus a point is a vector, a point plus a vector is a point.
    assert_vec3(b - a, [3.0, 4.0, 0.0]);
    assert_point3(a + vec3(1.0, 1.0, 1.0), [2.0, 3.0, 4.0]);

    let mut c = a;
    c += vec3(1.0, 1.0, 1.0);
    assert_point3(c, [2.0, 3.0, 4.0]);
    c -= vec3(1.0, 1.0, 1.0);
    assert_point3(c, [1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------- Bounds ----

/// The regression this whole exercise started from: `EMPTY` is the *union*
/// identity, so growing it by a point must give exactly that point. The old
/// version had the sentinel inverted and stayed infinite forever.
#[test]
fn empty_is_the_union_identity() {
    let b = Bounds::<V, 3>::EMPTY | pt3(1.0, 2.0, 3.0);

    assert_point3(b.min_point(), [1.0, 2.0, 3.0]);
    assert_point3(b.max_point(), [1.0, 2.0, 3.0]);

    assert!(Bounds::<V, 3>::EMPTY.is_empty().all());
    assert!(!b.is_empty().all());

    // ... and UNIVERSE is the intersection identity.
    let c = Bounds::<V, 3>::UNIVERSE & b;
    assert_point3(c.min_point(), [1.0, 2.0, 3.0]);
    assert_point3(c.max_point(), [1.0, 2.0, 3.0]);
}

#[test]
fn bounds_from_iterator() {
    let pts = [pt3(1.0, -2.0, 0.5), pt3(-3.0, 4.0, 0.0), pt3(0.0, 0.0, 2.0)];

    let b: Bounds<V, 3> = pts.into_iter().collect();

    assert_point3(b.min_point(), [-3.0, -2.0, 0.0]);
    assert_point3(b.max_point(), [1.0, 4.0, 2.0]);
}

#[test]
fn bounds_queries() {
    let b = Bounds::<V, 3>::from_corners(vec3(-1.0, -2.0, -3.0), vec3(1.0, 2.0, 3.0));

    assert_vec3(b.diagonal(), [2.0, 4.0, 6.0]);
    assert_point3(b.centroid(), [0.0, 0.0, 0.0]);

    // volume = 2*4*6; surface area = 2*(2*4 + 2*6 + 4*6)
    assert!(close(s(b.volume()), 48.0));
    assert!(close(s(b.surface_area()), 88.0));

    // The longest axis is z (extent 6), which is what a BVH would split on.
    assert_eq!(b.max_extent_axis().extract::<0>(), 2);

    assert!(b.contains(pt3(0.0, 0.0, 0.0)).all());
    assert!(b.contains(pt3(1.0, 2.0, 3.0)).all()); // faces are inclusive
    assert!(!b.contains(pt3(1.1, 0.0, 0.0)).all());

    let inner = Bounds::<V, 3>::from_corners(vec3(-0.5, -0.5, -0.5), vec3(0.5, 0.5, 0.5));
    assert!(b.contains_bounds(&inner).all());
    assert!(!inner.contains_bounds(&b).all());
    assert!(b.overlaps(&inner).all());

    // Overlap needs EVERY axis to overlap: this box shares x and y with `b` but is
    // far away in z, so it must NOT count as overlapping.
    let disjoint_in_z = Bounds::<V, 3>::from_corners(vec3(-1.0, -1.0, 100.0), vec3(1.0, 1.0, 101.0));
    assert!(!b.overlaps(&disjoint_in_z).all());
    assert!((b & disjoint_in_z).is_empty().all());

    // offset maps min -> 0 and max -> 1
    assert_vec3(b.offset(b.min_point()), [0.0, 0.0, 0.0]);
    assert_vec3(b.offset(b.max_point()), [1.0, 1.0, 1.0]);
    assert_vec3(b.offset(b.centroid()), [0.5, 0.5, 0.5]);

    // A degenerate (zero-extent) axis must not produce 0/0 = NaN.
    let flat = Bounds::<V, 3>::from_corners(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0));
    assert!(!flat.offset(pt3(0.5, 0.5, 0.0)).is_nan().any());

    // closest_point clamps into the box, and inside points are returned unchanged.
    assert_point3(b.closest_point(pt3(5.0, 0.0, 0.0)), [1.0, 0.0, 0.0]);
    assert_point3(b.closest_point(pt3(0.5, 0.5, 0.5)), [0.5, 0.5, 0.5]);
    assert!(close(s(b.distance_sqr(pt3(4.0, 0.0, 0.0))), 9.0));
    assert!(close(s(b.distance_sqr(pt3(0.0, 0.0, 0.0))), 0.0));

    let (center, radius) = b.bounding_sphere();
    assert_point3(center, [0.0, 0.0, 0.0]);
    assert!(close(s(radius), 14.0f32.sqrt()));

    // vertex() walks the corners in bit order, where 0 is min and 2^N - 1 is max.
    assert_point3(b.vertex(0), [-1.0, -2.0, -3.0]);
    assert_point3(b.vertex(7), [1.0, 2.0, 3.0]);
    assert_point3(b.vertex(1), [1.0, -2.0, -3.0]);
}

// ---------------------------------------------------------------- Matrix ----

/// `m * m.invert()` must be the identity, at every size the ladder dispatches on:
/// 2x2 and 3x3 closed forms, the 4x4 adjugate, and the Gauss-Jordan fallback at 5.
#[test]
fn matrix_inverse_round_trips_at_every_size() {
    macro_rules! check {
        ($n:literal, $entries:expr) => {{
            let m = Matrix::<V, $n, $n>::new($entries.map(|col: [f32; $n]| col.map(v)));

            let product = m * m.invert();
            let identity = Matrix::<V, $n, $n>::IDENTITY;

            for c in 0..$n {
                for r in 0..$n {
                    assert!(
                        close(s(product[c][r]), s(identity[c][r])),
                        "{}x{}: entry ({r}, {c}) = {}",
                        $n,
                        $n,
                        s(product[c][r])
                    );
                }
            }

            // The determinant is non-zero, and try_invert agrees.
            assert!(s(m.determinant()).abs() > 1e-3);
            assert!(m.try_invert().1.all());
        }};
    }

    check!(2, [[4.0f32, 7.0], [2.0, 6.0]]);
    check!(3, [[2.0f32, 0.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]]);
    check!(
        4,
        [
            [1.0f32, 2.0, 0.0, 1.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 4.0],
            [1.0, 1.0, 0.0, 1.0]
        ]
    );
    // N = 5 takes the general Gauss-Jordan path with lane-wise partial pivoting.
    check!(
        5,
        [
            [2.0f32, 1.0, 0.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 5.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 6.0]
        ]
    );
}

#[test]
fn matrix_determinant() {
    let m2 = Matrix::<V, 2, 2>::new([[4.0, 2.0], [7.0, 6.0]].map(|c: [f32; 2]| c.map(v)));
    assert!(close(s(m2.determinant()), 4.0 * 6.0 - 2.0 * 7.0));

    // A singular matrix (two identical columns) must report det == 0 and fail
    // try_invert rather than silently returning garbage.
    let singular = Matrix::<V, 3, 3>::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 7.0]].map(|c: [f32; 3]| c.map(v)));

    assert!(close(s(singular.determinant()), 0.0));
    assert!(!singular.try_invert().1.all());

    assert!(close(s(Matrix::<V, 4, 4>::IDENTITY.determinant()), 1.0));
    assert!(Matrix::<V, 4, 4>::IDENTITY.is_identity(v(1e-6)).all());
    assert!(close(s(Matrix::<V, 4, 4>::IDENTITY.trace()), 4.0));
}

#[test]
fn matrix_transforms() {
    let m = Matrix::<V, 4, 4>::from_translation(vec3(10.0, 20.0, 30.0));

    // A point carries w = 1, so it picks up the translation ...
    assert_point3(m.transform_point(pt3(1.0, 2.0, 3.0)), [11.0, 22.0, 33.0]);
    // ... a vector carries w = 0, so it does not. This is the whole reason the two
    // types are distinct.
    assert_vec3(m.transform_vector(vec3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0]);

    let s_m = Matrix::<V, 4, 4>::from_scale(vec3(2.0, 3.0, 4.0));
    assert_point3(s_m.transform_point(pt3(1.0, 1.0, 1.0)), [2.0, 3.0, 4.0]);

    // Composition: scale then translate (column-vector convention, so the matrix
    // applied first is on the right).
    let composed = m * s_m;
    assert_point3(composed.transform_point(pt3(1.0, 1.0, 1.0)), [12.0, 23.0, 34.0]);

    // The affine inverse undoes it.
    assert_point3(composed.invert().transform_point(pt3(12.0, 23.0, 34.0)), [1.0, 1.0, 1.0]);

    // The Mul operators agree with the named methods.
    assert_point3(m * pt3(1.0, 2.0, 3.0), [11.0, 22.0, 33.0]);
}

#[test]
fn matrix_rotation() {
    // A quarter turn about +Z takes +X to +Y.
    let rot = Matrix::<V, 4, 4>::from_axis_angle(vec3(0.0, 0.0, 1.0), v(core::f32::consts::FRAC_PI_2));

    assert_vec3(rot.transform_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
    assert_vec3(rot.transform_vector(vec3(0.0, 1.0, 0.0)), [-1.0, 0.0, 0.0]);
    assert_vec3(rot.transform_vector(vec3(0.0, 0.0, 1.0)), [0.0, 0.0, 1.0]);

    // Rotations are orthogonal: the inverse is the transpose, and det == 1.
    assert!(close(s(rot.determinant()), 1.0));

    let should_be_identity = rot * rot.transpose();
    assert!(should_be_identity.is_identity(v(1e-5)).all());
}

#[test]
fn matrix_transform_bounds() {
    let b = Bounds::<V, 3>::from_corners(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));

    // Translation moves the box rigidly.
    let translated = Matrix::<V, 4, 4>::from_translation(vec3(5.0, 0.0, 0.0)).transform_bounds(&b);
    assert_point3(translated.min_point(), [4.0, -1.0, -1.0]);
    assert_point3(translated.max_point(), [6.0, 1.0, 1.0]);

    // A 45-degree turn about Z makes the box no longer axis-aligned, so its AABB
    // grows to sqrt(2) on x and y while z is untouched. This is the union of all
    // eight transformed corners, and it is why transform_bounds cannot just
    // transform min and max.
    let rot = Matrix::<V, 4, 4>::from_axis_angle(vec3(0.0, 0.0, 1.0), v(core::f32::consts::FRAC_PI_4));
    let rotated = rot.transform_bounds(&b);

    let sqrt2 = 2.0f32.sqrt();
    assert_point3(rotated.min_point(), [-sqrt2, -sqrt2, -1.0]);
    assert_point3(rotated.max_point(), [sqrt2, sqrt2, 1.0]);
}

#[test]
fn matrix_error_bounds() {
    // gamma(n) is tiny but strictly positive, and grows with n.
    let g3 = s(gamma::<V>(3));
    assert!(g3 > 0.0 && g3 < 1e-6);
    assert!(s(gamma::<V>(7)) > g3);

    let m = Matrix::<V, 4, 4>::from_translation(vec3(1.0, 2.0, 3.0));
    let (p, error) = m.transform_point_with_error(pt3(1.0, 1.0, 1.0));

    assert_point3(p, [2.0, 3.0, 4.0]);

    // The bound is conservative but must be finite, non-negative, and small.
    for i in 0..3 {
        assert!(s(error[i]) >= 0.0 && s(error[i]) < 1e-4);
    }
}

// ------------------------------------------------------- lane divergence ----

/// The masked paths above are invisible at one lane: a 1-lane batch can only be
/// all-TIR or all-refracting. These run a batch that is deliberately *mixed*, so
/// the select actually has to route per lane.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn refract_diverges_per_lane() {
    use thermite::backend::x86_v3::f32x8;

    type W = f32x8;

    let lanes = <W as GenericVector>::LANES;

    // Incidence angle sweeps from near-normal to near-grazing across the lanes,
    // against eta = 1.5 whose critical angle (~41.8 degrees) falls inside it.
    let angles: [f32; 8] = core::array::from_fn(|i| {
        let t = i as f32 / (8 - 1) as f32;
        t * 80.0f32.to_radians()
    });

    let eta = 1.5f32;

    let i = GVector::<W, 3>::new([
        W::from_slice(&angles.map(f32::sin)),
        W::from_slice(&angles.map(|a| -a.cos())),
        W::splat(0.0),
    ]);

    let n = GVector::<W, 3>::new([W::splat(0.0), W::splat(1.0), W::splat(0.0)]);

    let (got, ok) = i.refract_mask(&n, W::splat(eta));

    // Both outcomes must actually occur, or this test proves nothing.
    assert!(ok.any(), "expected some lanes to refract");
    assert!(!ok.all(), "expected some lanes to totally internally reflect");

    let flags = ok.native_bitmask().expect("x86 masks carry a native bitmask");

    for (lane, &angle) in angles.iter().enumerate().take(lanes) {
        let iv = [angle.sin() as f64, -(angle.cos() as f64), 0.0f64];

        let (want, k) = refract3_ref(iv, [0.0, 1.0, 0.0], eta as f64);

        let refracted = (flags >> lane) & 1 == 1;

        assert_eq!(refracted, k >= 0.0, "lane {lane}: TIR flag disagrees (k = {k})");

        for j in 0..3 {
            assert!(
                close(got[j].extractv(lane), want[j] as f32),
                "lane {lane} component {j}: {} != {}",
                got[j].extractv(lane),
                want[j]
            );
        }
    }
}

// ------------------------------------------------- axis-aligned ray regression ----

/// An axis-aligned ray must survive the slab test.
///
/// This is a regression test for a real bug: `inv_direction` used the
/// policy-based `approx_reciprocal()`, which on any backend with `HAS_APPROX_RCP`
/// computes `rcp` plus a Newton step. That step is `y * (2 - d*y)`, which at
/// `d = 0` evaluates to `inf * (2 - 0*inf)` = **NaN** rather than the infinity
/// the slab test needs, so every axis-aligned ray failed every comparison and
/// was reported as a miss.
///
/// It MUST run on a wide backend. The 1-lane `Vector<f32>` used elsewhere in
/// this file has `HAS_APPROX_RCP = false`, takes the exact path, and cannot
/// observe the bug at all, which is exactly why it went unnoticed.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn axis_aligned_rays_hit_on_a_wide_backend() {
    use thermite::backend::x86_v3::f32x8;
    use thermite_geometry::soa::prim::{Ray, ray::RayOps as _};

    type W = f32x8;

    let unit = |x: f32, y: f32, z: f32| GVector::<W, 3>::new([W::splat(x), W::splat(y), W::splat(z)]);
    let point = |x: f32, y: f32, z: f32| Point::<W, 3>::new([W::splat(x), W::splat(y), W::splat(z)]);

    let boxed = Bounds::<W, 3>::from_corners(unit(-1.0, -1.0, -1.0), unit(1.0, 1.0, 1.0));

    // One axis-aligned ray per cardinal direction, each aimed at the origin.
    let cases = [
        ((-5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((5.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, -5.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 5.0), (0.0, 0.0, -1.0)),
    ];

    for ((ox, oy, oz), (dx, dy, dz)) in cases {
        let ray = Ray::new(point(ox, oy, oz), unit(dx, dy, dz));

        let inv = ray.inv_direction();

        // The zero components must be infinities, never NaN.
        for c in 0..3 {
            let v = inv[c].extract::<0>();
            assert!(!v.is_nan(), "inv_direction component {c} is NaN for dir ({dx}, {dy}, {dz})");
        }

        let (t_min, t_max, hit) = ray.intersects_aabb_mask(&boxed, &inv);

        assert!(
            hit.all(),
            "axis-aligned ray from ({ox}, {oy}, {oz}) toward ({dx}, {dy}, {dz}) must hit the box"
        );

        // It enters at 4 and leaves at 6, having started 5 away from a unit box.
        assert!(close(s0(t_min), 4.0), "t_min = {}", s0(t_min));
        assert!(s0(t_max) >= 6.0, "t_max = {}", s0(t_max));
    }
}

/// Read lane 0 of a wide vector back as a scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn s0(v: thermite::backend::x86_v3::f32x8) -> f32 {
    v.extract::<0>()
}

// ------------------------------------------------------------ ray spawning ----

#[test]
fn offset_origin_escapes_the_surface() {
    use thermite_geometry::soa::prim::Ray;

    let p = pt3(1.0, 2.0, 3.0);
    let normal = vec3(0.0, 1.0, 0.0);
    let error = vec3(1e-5, 1e-5, 1e-5);

    // A ray leaving along the normal is pushed to the +y side...
    let out = Ray::offset_origin(p, error, normal, normal);
    assert!(s(out[1]) > s(p[1]), "offset must move off the surface: {} vs {}", s(out[1]), s(p[1]));

    // ... and one leaving into the surface, to the -y side. The side test is on
    // `direction`, so a transmitted ray needs nothing special at the call site.
    let into = Ray::offset_origin(p, error, normal, -normal);
    assert!(s(into[1]) < s(p[1]));

    // The offset tracks the error bound rather than a fixed epsilon: small
    // error, small offset.
    assert!((s(out[1]) - s(p[1])).abs() < 1e-3);

    // The tangential axes carry error too, so they move, but the normal axis is
    // the one that must clear the surface.
    assert!(s(out[1]) - s(p[1]) >= 1e-5);

    // Zero error still steps at least one ulp off the surface. The final nudge
    // exists because the addition above rounds, possibly back onto it.
    let tiny = Ray::offset_origin(p, GVector::<V, 3>::ZERO, normal, normal);
    assert!(s(tiny[1]) >= s(p[1]));
}

#[test]
fn offset_origin_scales_with_distance_from_the_origin() {
    use thermite_geometry::soa::prim::Ray;

    let normal = vec3(0.0, 1.0, 0.0);

    // The whole point of an error-bound offset over a fixed epsilon: at large
    // coordinates the representable gap is huge, and the offset must grow with
    // it. A hardcoded 1e-4 would vanish into the rounding here.
    let near = pt3(0.0, 1.0, 0.0);
    let far = pt3(0.0, 1.0e6, 0.0);

    // Error scaled the way gamma-based bounds actually scale: proportional to
    // the magnitude of the coordinates.
    let near_err = vec3(1e-7, 1e-7, 1e-7);
    let far_err = vec3(1e-1, 1e-1, 1e-1);

    let near_out = Ray::offset_origin(near, near_err, normal, normal);
    let far_out = Ray::offset_origin(far, far_err, normal, normal);

    let near_step = s(near_out[1]) - s(near[1]);
    let far_step = s(far_out[1]) - s(far[1]);

    assert!(near_step > 0.0, "near offset must be strictly positive");
    assert!(far_step > near_step, "the offset must grow with the error bound");

    // Both must actually be representable moves, not lost to rounding.
    assert_ne!(s(near_out[1]), s(near[1]));
    assert_ne!(s(far_out[1]), s(far[1]));
}

#[test]
fn offset_origin_diverges_per_lane() {
    // The side test is a mask, so one batch may carry reflected and transmitted
    // rays at once. A 1-lane run cannot show that.
    use thermite_geometry::soa::prim::{Point as SoaPoint, Ray, Vector as SoaVector};

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use thermite::backend::x86_v3::f32x4;

        type W = f32x4;

        let p = SoaPoint::<W, 3>::new([W::splat(0.0), W::splat(5.0), W::splat(0.0)]);
        let err = SoaVector::<W, 3>::new([W::splat(1e-4), W::splat(1e-4), W::splat(1e-4)]);
        let normal = SoaVector::<W, 3>::new([W::splat(0.0), W::splat(1.0), W::splat(0.0)]);

        // Lanes 0 and 2 reflect (leave along +n), and lanes 1 and 3 transmit.
        let dir = SoaVector::<W, 3>::new([
            W::splat(0.0),
            W::from_slice(&[1.0, -1.0, 1.0, -1.0]),
            W::splat(0.0),
        ]);

        let out = Ray::offset_origin(p, err, normal, dir);

        for lane in 0..4 {
            let y = out[1].extractv(lane);
            let leaving_front = lane % 2 == 0;

            if leaving_front {
                assert!(y > 5.0, "lane {lane} reflects, so it must move to +y: {y}");
            } else {
                assert!(y < 5.0, "lane {lane} transmits, so it must move to -y: {y}");
            }
        }
    }
}

// --------------------------------------------------------- regression: nits ----

#[test]
fn is_identity_detects_every_entry() {
    // Regression: `is_identity` used to seed its mask from entry (0, 0) and then
    // loop over every entry including (0, 0) again. Harmless, but the real
    // requirement is that a deviation in ANY entry is caught, including the
    // off-diagonal ones a seed-from-(0,0) reader might assume are skipped.
    let tol = v(1e-6);

    assert!(Matrix::<V, 4, 4>::IDENTITY.is_identity(tol).all());

    for c in 0..4 {
        for r in 0..4 {
            let mut m = Matrix::<V, 4, 4>::IDENTITY;
            m.0[c][r] += v(0.5);

            assert!(
                !m.is_identity(tol).all(),
                "a deviation at ({r}, {c}) must be detected"
            );
        }
    }
}

#[test]
fn matrix4_times_vector3_ignores_translation() {
    // `Matrix<V, 4, R> * Vector3` treats the vector as w = 0, so the translation
    // column must not apply, the operator form of `transform_vector`.
    //
    // Regression: this impl used to be generic over `R` while indexing rows
    // 0..3, so a `Matrix<V, 4, 2>` would panic at runtime. It is now spelled out
    // for the shapes that have three rows, making that a compile error.
    let m = Matrix::<V, 4, 4>::from_translation(vec3(10.0, 20.0, 30.0));

    let v3 = vec3(1.0, 2.0, 3.0);

    assert_vec3(m * v3, [1.0, 2.0, 3.0]);
    // ... and it agrees with the named method.
    assert_vec3(m * v3, [s(m.transform_vector(v3)[0]), s(m.transform_vector(v3)[1]), s(m.transform_vector(v3)[2])]);

    // A 4x3 (compact affine) matrix works too, and its first three rows are what
    // the product reads.
    let mut compact = Matrix::<V, 4, 3>::splat(V::ZERO);
    for i in 0..3 {
        compact.0[i][i] = v(2.0);
    }
    compact.0[3][0] = v(99.0); // translation column: must be ignored

    assert_vec3(compact * v3, [2.0, 4.0, 6.0]);
}

// ------------------------------------------------ adjugate differential ----

/// The original flat 4x4 adjugate, kept verbatim as a reference.
///
/// `adjugate_det4` was rewritten to share 2x2 minors and use the FMA family, and
/// this is the expression it replaced, so the test below pins the rewrite to
/// the exact algebra it came from. 16 entries of hand-transcribed cofactors is
/// precisely where a sign or index typo hides, and no round-trip test would
/// localize one.
#[rustfmt::skip]
fn reference_adjugate(m: &[f32; 16]) -> [f32; 16] {
    [
        m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10],
       -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10],
        m[4]*m[ 9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[ 9],
       -m[4]*m[ 9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[ 9],
       -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10],
        m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10],
       -m[0]*m[ 9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[ 9],
        m[0]*m[ 9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[ 9],
        m[1]*m[ 6]*m[15] - m[1]*m[ 7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[ 7] - m[13]*m[3]*m[ 6],
       -m[0]*m[ 6]*m[15] + m[0]*m[ 7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[ 7] + m[12]*m[3]*m[ 6],
        m[0]*m[ 5]*m[15] - m[0]*m[ 7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[ 7] - m[12]*m[3]*m[ 5],
       -m[0]*m[ 5]*m[14] + m[0]*m[ 6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[ 6] + m[12]*m[2]*m[ 5],
       -m[1]*m[ 6]*m[11] + m[1]*m[ 7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[ 9]*m[2]*m[ 7] + m[ 9]*m[3]*m[ 6],
        m[0]*m[ 6]*m[11] - m[0]*m[ 7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[ 8]*m[2]*m[ 7] - m[ 8]*m[3]*m[ 6],
       -m[0]*m[ 5]*m[11] + m[0]*m[ 7]*m[ 9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[ 9] - m[ 8]*m[1]*m[ 7] + m[ 8]*m[3]*m[ 5],
        m[0]*m[ 5]*m[10] - m[0]*m[ 6]*m[ 9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[ 9] + m[ 8]*m[1]*m[ 6] - m[ 8]*m[2]*m[ 5],
    ]
}

/// A tiny deterministic LCG, so the sweep is reproducible without a dep.
fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);

    // Roughly [-2, 2), avoiding a degenerate all-tiny distribution.
    ((*state >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
}

#[test]
fn adjugate_matches_the_flat_reference() {
    let mut state = 0x1234_5678u32;

    for case in 0..400 {
        let raw: [f32; 16] = core::array::from_fn(|_| lcg(&mut state));

        let want = reference_adjugate(&raw);

        // The crate exposes the adjugate through `invert`, which divides by the
        // determinant, so multiply it back out to recover the adjugate itself.
        let m = Matrix::<V, 4, 4>::new(core::array::from_fn(|c| {
            core::array::from_fn(|r| v(raw[c * 4 + r]))
        }));

        let det = s(m.determinant());

        // Skip the near-singular draws: recovering adj = inv * det through a
        // tiny determinant amplifies rounding without testing the algebra.
        if det.abs() < 1e-2 {
            continue;
        }

        let inv = m.invert();

        for c in 0..4 {
            for r in 0..4 {
                // `invert` writes adj[r * 4 + c] / det into result[c][r].
                let got = s(inv.0[c][r]) * det;
                let expected = want[r * 4 + c];

                let tol = 1e-3 * expected.abs().max(1.0);

                assert!(
                    (got - expected).abs() <= tol,
                    "case {case}: adjugate ({r}, {c}) = {got}, reference {expected}"
                );
            }
        }

        // The determinant itself must agree with contracting column 0 of the
        // input against row 0 of the reference adjugate.
        let want_det = raw[0] * want[0] + raw[1] * want[1] + raw[2] * want[2] + raw[3] * want[3];

        assert!(
            (det - want_det).abs() <= 1e-3 * want_det.abs().max(1.0),
            "case {case}: determinant {det} != reference {want_det}"
        );
    }
}

// -------------------------------------------- SoA decompose / kinematics ----

#[test]
fn quaternion_to_axis_angle_round_trips() {
    use thermite_geometry::soa::prim::Quaternion;

    let axis = vec3(1.0, 2.0, -0.5).normalize();
    let angle = 0.9f32;

    let q = Quaternion::<V>::from_axis_angle_raw(axis, v(angle));

    let (got_axis, got_angle) = q.to_axis_angle();

    assert!(close(s(got_angle), angle), "angle {} != {angle}", s(got_angle));
    assert_vec3(got_axis, [s(axis[0]), s(axis[1]), s(axis[2])]);

    // The identity has no axis, so it reports +X and a zero angle rather than a NaN.
    let (id_axis, id_angle) = Quaternion::<V>::IDENTITY.to_axis_angle();
    assert!(close(s(id_angle), 0.0));
    assert_vec3(id_axis, [1.0, 0.0, 0.0]);
}

#[test]
fn quaternion_look_at_aims_plus_z() {
    use thermite_geometry::soa::prim::Quaternion;

    // Regression: this used to build `axis = z_hat` cross the wrong operand and
    // aim +Y at the target, contradicting both its own doc and
    // `Transform::look_at`.
    let origin = pt3(0.0, 0.0, 0.0);

    let q = Quaternion::<V>::look_at(origin, pt3(1.0, 0.0, 0.0));
    assert_vec3(q.rotate_vector(vec3(0.0, 0.0, 1.0)), [1.0, 0.0, 0.0]);

    let q = Quaternion::<V>::look_at(origin, pt3(0.0, 1.0, 0.0));
    assert_vec3(q.rotate_vector(vec3(0.0, 0.0, 1.0)), [0.0, 1.0, 0.0]);

    // The poles: looking along +Z is the identity, along -Z a half turn.
    let q = Quaternion::<V>::look_at(origin, pt3(0.0, 0.0, 1.0));
    assert_vec3(q.rotate_vector(vec3(0.0, 0.0, 1.0)), [0.0, 0.0, 1.0]);

    let q = Quaternion::<V>::look_at(origin, pt3(0.0, 0.0, -1.0));
    assert_vec3(q.rotate_vector(vec3(0.0, 0.0, 1.0)), [0.0, 0.0, -1.0]);

    // A coincident target names no direction: identity, never NaN.
    let q = Quaternion::<V>::look_at(origin, origin);
    assert!(q.is_finite().all());
    assert_vec3(q.rotate_vector(vec3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0]);
}

#[test]
fn quaternion_exp_log_and_integrate() {
    use thermite_geometry::soa::prim::Quaternion;

    // exp/log invert each other.
    for raw in [[0.0f32, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, -0.7, 1.1]] {
        let rv = vec3(raw[0], raw[1], raw[2]);

        assert_vec3(Quaternion::<V>::exp(rv).log(), raw);
    }

    // log of the identity is zero, and exp of zero is the identity.
    assert_vec3(Quaternion::<V>::IDENTITY.log(), [0.0, 0.0, 0.0]);
    assert!(close(s(Quaternion::<V>::exp(GVector::ZERO).w()), 1.0));

    // A half turn about +Y in one big step: the exponential update is exact at
    // any step size, where a first-order one would badly undershoot.
    let q = Quaternion::<V>::IDENTITY.integrate(vec3(0.0, core::f32::consts::PI, 0.0), v(1.0));

    assert_vec3(q.rotate_vector(vec3(1.0, 0.0, 0.0)), [-1.0, 0.0, 0.0]);
    assert!(close(s(q.norm()), 1.0));

    // Many small steps converge on the same closed form, without drifting off
    // the unit sphere.
    let mut acc = Quaternion::<V>::IDENTITY;
    let steps = 1000;

    for _ in 0..steps {
        acc = acc.integrate(vec3(0.0, 0.0, 2.0), v(1.5 / steps as f32));
    }

    let want = Quaternion::<V>::from_axis_angle_raw(vec3(0.0, 0.0, 1.0), v(3.0));

    assert_vec3(
        acc.rotate_vector(vec3(1.0, 0.0, 0.0)),
        [
            s(want.rotate_vector(vec3(1.0, 0.0, 0.0))[0]),
            s(want.rotate_vector(vec3(1.0, 0.0, 0.0))[1]),
            s(want.rotate_vector(vec3(1.0, 0.0, 0.0))[2]),
        ],
    );
    assert!(close(s(acc.norm()), 1.0), "norm drifted to {}", s(acc.norm()));
}

#[test]
fn transform_decompose_round_trips() {
    use thermite_geometry::soa::prim::{Quaternion, Transform};

    let t = vec3(1.0, -2.0, 3.0);
    let axis = vec3(0.3, 1.0, -0.2).normalize();
    let angle = 0.7f32;
    let scale = vec3(2.0, 3.0, 4.0);

    let m = Matrix::<V, 4, 4>::from_translation(t)
        * Quaternion::<V>::from_axis_angle_raw(axis, v(angle)).to_matrix()
        * Matrix::<V, 4, 4>::from_scale(scale);

    let (got_t, got_r, got_s) = Transform::new(m).decompose();

    assert_vec3(got_t, [s(t[0]), s(t[1]), s(t[2])]);

    // The rotation is compared by its action: q and -q are the same rotation.
    let reference = Quaternion::<V>::from_axis_angle_raw(axis, v(angle));

    for probe in [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)] {
        let a = got_r.rotate_vector(probe);
        let b = reference.rotate_vector(probe);

        assert_vec3(a, [s(b[0]), s(b[1]), s(b[2])]);
    }

    // Recomposing must reproduce the original transform.
    let rebuilt = Matrix::<V, 4, 4>::from_translation(got_t) * got_r.to_matrix() * got_s.to_homogeneous();

    for p in [pt3(1.0, 0.0, 0.0), pt3(1.0, 2.0, 3.0), pt3(-2.0, 0.5, 1.0)] {
        let want = m.transform_point(p);
        let got = rebuilt.transform_point(p);

        for i in 0..3 {
            assert!(
                (s(got[i]) - s(want[i])).abs() < 1e-3,
                "recomposed differs at {i}: {} != {}",
                s(got[i]),
                s(want[i])
            );
        }
    }
}

#[test]
fn transform_decompose_moves_mirroring_into_the_scale() {
    use thermite_geometry::soa::prim::Transform;

    // A reflection has determinant -1. The polar factor would be improper, and
    // no quaternion can represent that, so decompose must flip it into a proper
    // rotation and push the mirror into the scale.
    let m = Matrix::<V, 4, 4>::from_scale(vec3(1.0, -1.0, 1.0));

    assert!(s(m.determinant()) < 0.0);

    let (_, r, scale) = Transform::new(m).decompose();

    // The rotation really is a rotation: it preserves length and handedness.
    let x = r.rotate_vector(vec3(1.0, 0.0, 0.0));
    let y = r.rotate_vector(vec3(0.0, 1.0, 0.0));
    let z = r.rotate_vector(vec3(0.0, 0.0, 1.0));

    assert!(close(s(x.norm_sqr()), 1.0) && close(s(y.norm_sqr()), 1.0) && close(s(z.norm_sqr()), 1.0));
    // x cross y == z for a proper rotation (a reflection would give -z).
    assert_vec3(x.cross(y), [s(z[0]), s(z[1]), s(z[2])]);

    // ... and the mirror survives in the scale.
    let diag = [s(scale.0[0][0]), s(scale.0[1][1]), s(scale.0[2][2])];

    assert!(
        diag.iter().any(|&d| d < 0.0),
        "the mirror must land in the scale factor: {diag:?}"
    );
}
