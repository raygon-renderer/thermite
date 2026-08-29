//! Behavioural tests for the AoS primitives, and cross-checks against the SoA
//! ones.
//!
//! The two layouts are independent implementations of the same geometry (SoA
//! open-codes the arithmetic component-by-component, AoS delegates to the
//! `LinAlg3Vector`/`LinAlg4Vector` backend primitives), so agreeing to float
//! tolerance is real evidence about both. Where they disagree, one of them is
//! wrong.

use thermite::{prelude::*, simd::Simd3Vectors};

use thermite_geometry::aos::{
    self, AosFloat, Bounds3, Matrix4, Quaternion, Ray3, RayError3, TangentFrame, Transform, Vector3Ext as _,
    matrix::gamma,
};

/// The backend under test. `X86V3` is AVX2 + FMA, the interesting one. The
/// module is generic, so this pins one instantiation.
type S = thermite::backend::x86_v3::X86V3;

/// 3-lane and 4-lane f32 registers for that backend.
type V3 = <f32 as AosFloat<S>>::V3;
type V4 = <f32 as AosFloat<S>>::V4;

type M4 = Matrix4<S, f32>;
type Quat = Quaternion<S, f32>;

fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3::from_slice(&[x, y, z])
}

fn arr(v: V3) -> [f32; 3] {
    [v.extract::<0>(), v.extract::<1>(), v.extract::<2>()]
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[track_caller]
fn assert_v3(actual: V3, expected: [f32; 3]) {
    let a = arr(actual);

    for i in 0..3 {
        assert!(close(a[i], expected[i]), "component {i}: {} != {}", a[i], expected[i]);
    }
}

#[track_caller]
fn assert_v3_eq(actual: V3, expected: V3) {
    assert_v3(actual, arr(expected));
}

// ---------------------------------------------------------------- vector ----

#[test]
fn vector_norms_and_normalize() {
    let a = v3(3.0, 4.0, 0.0);

    assert!(close(a.norm_sqr(), 25.0));
    assert!(close(a.norm(), 5.0));
    assert!(close(a.l1_norm3(), 7.0));
    assert!(close(a.linf_norm3(), 4.0));

    assert_v3(a.normalize(), [0.6, 0.8, 0.0]);

    let (unit, norm) = a.normalize_norm();
    assert_v3(unit, [0.6, 0.8, 0.0]);
    assert!(close(norm, 5.0));

    // A zero vector has no direction: try_normalize gives zero, not NaN.
    assert_v3(V3::ZERO.try_normalize(), [0.0, 0.0, 0.0]);
    assert!(arr(V3::ZERO.normalize()).iter().all(|c| c.is_nan()));
}

#[test]
fn vector_dot_and_cross() {
    let a = v3(1.0, 2.0, 3.0);
    let b = v3(4.0, 5.0, 6.0);

    assert!(close(a.dot3(b), 32.0));
    assert_v3(a.cross3::<false>(b), [-3.0, 6.0, -3.0]);

    // The cross product is perpendicular to both inputs.
    let c = a.cross3::<false>(b);
    assert!(close(c.dot3(a), 0.0));
    assert!(close(c.dot3(b), 0.0));

    // FAST and non-FAST forms agree to tolerance.
    assert_v3_eq(a.cross3::<true>(b), c);
}

#[test]
fn vector_shading_ops() {
    let n = v3(0.0, 1.0, 0.0);

    // reflect: the normal component flips, the tangential one survives.
    assert_v3(v3(1.0, -1.0, 0.0).reflect(n), [1.0, 1.0, 0.0]);

    // Reflecting twice is the identity.
    let i = v3(0.3, -0.5, -0.8).normalize();
    let nz = v3(0.0, 0.0, 1.0);
    assert_v3_eq(i.reflect(nz).reflect(nz), i);

    // refract: straight-on passes through undeviated.
    assert_v3(v3(0.0, -1.0, 0.0).refract(n, 0.75), [0.0, -1.0, 0.0]);

    // Total internal reflection returns zero and reports it.
    let steep = v3(3.0f32.sqrt(), -1.0, 0.0).normalize();
    let (t, ok) = steep.refract_checked(n, 2.0);
    assert!(!ok, "60 degrees at eta = 2 must totally internally reflect");
    assert_v3(t, [0.0, 0.0, 0.0]);

    // faceforward flips only when needed.
    assert_v3(n.faceforward(v3(0.0, -1.0, 0.0), n), [0.0, 1.0, 0.0]);
    assert_v3(n.faceforward(v3(0.0, 1.0, 0.0), n), [0.0, -1.0, 0.0]);
}

#[test]
fn vector_projection_and_angle() {
    let a = v3(2.0, 3.0, 0.0);
    let x = v3(1.0, 0.0, 0.0);

    assert_v3(a.project_onto(x), [2.0, 0.0, 0.0]);
    assert_v3(a.reject_from(x), [0.0, 3.0, 0.0]);

    // The two parts recompose.
    assert_v3_eq(a.project_onto(x) + a.reject_from(x), a);

    assert!(close(x.angle_between(v3(0.0, 1.0, 0.0)), core::f32::consts::FRAC_PI_2));
    assert!(close(x.angle_between(x), 0.0));
    assert!(close(x.angle_between(-x), core::f32::consts::PI));
}

// ---------------------------------------------------------------- matrix ----

#[test]
fn matrix_identity_is_a_real_constant() {
    let id = M4::IDENTITY;

    for c in 0..4 {
        for r in 0..4 {
            let want = if c == r { 1.0 } else { 0.0 };
            assert!(close(id.get(r, c), want), "identity[{r}][{c}]");
        }
    }

    assert!(id.is_identity(1e-6));
    assert!(close(id.determinant(), 1.0));
}

#[test]
fn matrix_transform_point_applies_translation() {
    let m = M4::from_translation(v3(1.0, 2.0, 3.0));

    // A point picks up the translation...
    assert_v3(m.transform_point(v3(1.0, 1.0, 1.0)), [2.0, 3.0, 4.0]);
    // ... a direction does not.
    assert_v3(m.transform_vector(v3(1.0, 1.0, 1.0)), [1.0, 1.0, 1.0]);
}

#[test]
fn matrix_compose_and_invert() {
    let t = M4::from_translation(v3(1.0, 2.0, 3.0));
    let s = M4::from_scale(v3(2.0, 4.0, 8.0));

    let m = t * s;

    // Composition applies the right-hand matrix first: scale, then translate.
    assert_v3(m.transform_point(v3(1.0, 1.0, 1.0)), [3.0, 6.0, 11.0]);

    let inv = m.invert().expect("scale-then-translate is invertible");

    // The inverse really is the inverse.
    assert_v3(
        inv.transform_point(m.transform_point(v3(5.0, -2.0, 0.5))),
        [5.0, -2.0, 0.5],
    );
    assert!((m * inv).is_identity(1e-5));

    // A singular matrix is reported rather than silently producing infinities.
    assert!(M4::from_scale(v3(1.0, 0.0, 1.0)).invert().is_none());
}

#[test]
fn matrix_projective_divide() {
    // A perspective matrix: the last row is not [0,0,0,1], so w != 1.
    let t = Transform::<S, f32>::perspective(core::f32::consts::FRAC_PI_2, 1.0, 100.0);

    let p = v3(1.0, 0.0, 4.0);

    let projected = t.forward.project_point(p);

    // tan(45 deg) = 1, so x/z survives as 1/4 after the divide.
    assert!(close(projected[0], 0.25), "x = {}", projected[0]);

    // The affine path would NOT divide, so it must differ here, which is exactly
    // why project_point exists.
    assert!(!close(t.forward.transform_point(p)[0], projected[0]));
}

#[test]
fn matrix_transform_normal_uses_the_cofactor() {
    // Non-uniform scale: a normal must NOT transform like a direction.
    let m = M4::from_scale(v3(2.0, 1.0, 1.0));

    // The plane x = y has normal (1, -1, 0)/sqrt2 and tangent (1, 1, 0).
    let n = v3(1.0, -1.0, 0.0);
    let tangent = v3(1.0, 1.0, 0.0);

    let n2 = m.transform_normal::<true>(n);
    let t2 = m.transform_vector(tangent);

    // Perpendicularity survives, the whole point of the inverse-transpose.
    assert!(
        close(n2.dot3(t2), 0.0),
        "normal must stay perpendicular: {}",
        n2.dot3(t2)
    );

    // Transforming it as a direction would NOT preserve that.
    assert!(!close(m.transform_vector(n).dot3(t2), 0.0));

    // The undivided cofactor form points the same way.
    assert_v3_eq(m.transform_normal::<false>(n).normalize(), n2.normalize());
}

#[test]
fn matrix_batch_transform_matches_the_scalar_one() {
    let m = M4::from_translation(v3(1.0, -2.0, 0.5)) * M4::from_scale(v3(2.0, 3.0, 4.0));

    let pts = [
        v3(1.0, 1.0, 1.0),
        v3(0.0, 0.0, 0.0),
        v3(-1.0, 2.0, -3.0),
        v3(5.0, 5.0, 5.0),
    ];

    let batched = m.transform_points(pts);

    for (i, p) in pts.iter().enumerate() {
        assert_v3_eq(batched[i], m.transform_point(*p));
    }

    let vecs = m.transform_vectors(pts);

    for (i, v) in pts.iter().enumerate() {
        assert_v3_eq(vecs[i], m.transform_vector(*v));
    }
}

// ------------------------------------------------------------ quaternion ----

#[test]
fn quaternion_rotation_basics() {
    let quarter = core::f32::consts::FRAC_PI_2;

    // A quarter turn about +Z takes +X to +Y.
    let q = Quat::from_axis_angle_raw(v3(0.0, 0.0, 1.0), quarter);

    assert_v3(q.rotate_vector::<false>(v3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);

    // The identity does nothing.
    assert_v3(
        Quat::IDENTITY.rotate_vector::<false>(v3(1.0, 2.0, 3.0)),
        [1.0, 2.0, 3.0],
    );

    // The inverse undoes it.
    let v = v3(0.3, -0.7, 1.1);
    assert_v3_eq(q.inverse().rotate_vector::<false>(q.rotate_vector::<false>(v)), v);

    // Composition applies the right-hand rotation first, so two quarter turns about
    // Z make a half turn.
    let half = q * q;
    assert_v3(half.rotate_vector::<false>(v3(1.0, 0.0, 0.0)), [-1.0, 0.0, 0.0]);
}

#[test]
fn quaternion_matrix_round_trip() {
    let q = Quat::from_euler(0.3, -0.7, 1.1).normalize();

    let m = q.to_matrix();
    let back = Quat::from_matrix(&m);

    // q and -q are the same rotation, so compare the action, not the components.
    for v in [
        v3(1.0, 0.0, 0.0),
        v3(0.0, 1.0, 0.0),
        v3(0.0, 0.0, 1.0),
        v3(1.0, 2.0, 3.0),
    ] {
        assert_v3_eq(back.rotate_vector::<false>(v), q.rotate_vector::<false>(v));
        // ... and the matrix agrees with the quaternion it came from.
        assert_v3_eq(m.transform_vector(v), q.rotate_vector::<false>(v));
    }
}

#[test]
fn quaternion_axis_angle_round_trip() {
    let axis = v3(1.0, 2.0, -0.5).normalize();
    let angle = 0.9f32;

    let q = Quat::from_axis_angle_raw(axis, angle);
    let (got_axis, got_angle) = q.to_axis_angle();

    assert!(close(got_angle, angle), "angle: {got_angle} != {angle}");
    assert_v3_eq(got_axis, axis);

    // The identity has no axis and a zero angle, rather than a NaN.
    let (_, zero) = Quat::IDENTITY.to_axis_angle();
    assert!(close(zero, 0.0));

    // The axis-angle *vector* form round-trips too.
    let q2 = Quat::from_axis_angle(axis * V3::splat(angle));
    assert_v3_eq(
        q2.rotate_vector::<false>(v3(1.0, 1.0, 1.0)),
        q.rotate_vector::<false>(v3(1.0, 1.0, 1.0)),
    );
}

#[test]
fn quaternion_slerp_endpoints_and_midpoint() {
    let a = Quat::IDENTITY;
    let b = Quat::from_axis_angle_raw(v3(0.0, 0.0, 1.0), core::f32::consts::FRAC_PI_2);

    let v = v3(1.0, 0.0, 0.0);

    // Endpoints are exact.
    assert_v3_eq(a.slerp(b, 0.0).rotate_vector::<false>(v), a.rotate_vector::<false>(v));
    assert_v3_eq(a.slerp(b, 1.0).rotate_vector::<false>(v), b.rotate_vector::<false>(v));

    // The midpoint of a quarter turn is an eighth turn.
    let mid = a.slerp(b, 0.5).rotate_vector::<false>(v);
    let eighth = core::f32::consts::FRAC_PI_4;
    assert_v3(mid, [eighth.cos(), eighth.sin(), 0.0]);

    // Slerp of a quaternion with itself is a no-op, not a division by zero.
    assert_v3_eq(b.slerp(b, 0.35).rotate_vector::<false>(v), b.rotate_vector::<false>(v));
}

#[test]
fn quaternion_rotation_arc() {
    let a = v3(1.0, 0.0, 0.0);
    let b = v3(0.0, 0.0, 1.0);

    assert_v3_eq(Quat::from_rotation_arc(a, b).rotate_vector::<false>(a), b);

    // Antiparallel: a half turn about *some* perpendicular axis. The formula
    // cannot name one, so this is the degenerate case worth pinning.
    let flipped = Quat::from_rotation_arc(a, -a).rotate_vector::<false>(a);
    assert_v3(flipped, [-1.0, 0.0, 0.0]);

    // Parallel: the identity.
    assert_v3_eq(Quat::from_rotation_arc(a, a).rotate_vector::<false>(a), a);
}

// ---------------------------------------------------------------- bounds ----

#[test]
fn bounds_algebra() {
    type B = Bounds3<S, f32>;

    // EMPTY is the identity for union.
    let b = B::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    assert_v3_eq((B::EMPTY | b).min, b.min);
    assert_v3_eq((B::EMPTY | b).max, b.max);
    assert!(B::EMPTY.is_empty());

    assert!(b.contains(v3(0.0, 0.0, 0.0)));
    assert!(!b.contains(v3(2.0, 0.0, 0.0)));
    assert!(b.contains(v3(1.0, 1.0, 1.0)), "faces are inclusive");

    assert!(close(b.volume(), 8.0));
    assert!(close(b.surface_area(), 24.0));
    assert_v3(b.centroid(), [0.0, 0.0, 0.0]);
    assert_v3(b.diagonal(), [2.0, 2.0, 2.0]);

    // Disjoint boxes intersect to something empty.
    let far = B::new(v3(5.0, 5.0, 5.0), v3(6.0, 6.0, 6.0));
    assert!((b & far).is_empty());
    assert!(!b.overlaps(&far));
    assert!(b.overlaps(&B::new(v3(0.5, 0.5, 0.5), v3(9.0, 9.0, 9.0))));

    // Collecting points builds their bounding box.
    let from_points: B = [v3(-1.0, 0.0, 2.0), v3(3.0, -4.0, 0.0)].into_iter().collect();
    assert_v3(from_points.min, [-1.0, -4.0, 0.0]);
    assert_v3(from_points.max, [3.0, 0.0, 2.0]);

    // ... and union_all does the same for boxes.
    let merged = B::union_all([b, far]);
    assert_v3(merged.min, [-1.0, -1.0, -1.0]);
    assert_v3(merged.max, [6.0, 6.0, 6.0]);
}

#[test]
fn bounds_corners_and_extent() {
    let b = Bounds3::<S, f32>::new(v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 4.0));

    // Corner 0 is the min corner, corner 7 the max.
    assert_v3(b.vertex(0), [0.0, 0.0, 0.0]);
    assert_v3(b.vertex(7), [1.0, 2.0, 4.0]);
    // Bit 0 selects x, bit 1 y, bit 2 z.
    assert_v3(b.vertex(1), [1.0, 0.0, 0.0]);
    assert_v3(b.vertex(2), [0.0, 2.0, 0.0]);
    assert_v3(b.vertex(4), [0.0, 0.0, 4.0]);

    // Every corner is inside the box, and they are the 8 distinct ones.
    for v in b.vertices() {
        assert!(b.contains(v));
    }

    // z is the longest axis.
    assert_eq!(b.max_extent_axis(), 2);

    // offset maps the box onto the unit cube.
    assert_v3(b.offset(b.max), [1.0, 1.0, 1.0]);
    assert_v3(b.offset(b.min), [0.0, 0.0, 0.0]);
    assert_v3(b.offset(v3(0.5, 1.0, 2.0)), [0.5, 0.5, 0.5]);
}

#[test]
fn bounds_closest_point_and_distance() {
    let b = Bounds3::<S, f32>::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));

    // Outside: clamped onto the surface.
    assert_v3(b.closest_point(v3(3.0, 0.5, -1.0)), [1.0, 0.5, 0.0]);
    assert!(close(b.distance_sqr(v3(3.0, 0.5, -1.0)), 4.0 + 1.0));

    // Inside: the point itself, at zero distance.
    let inside = v3(0.5, 0.5, 0.5);
    assert_v3_eq(b.closest_point(inside), inside);
    assert!(close(b.distance_sqr(inside), 0.0));
}

#[test]
fn bounds_ray_intersection() {
    let b = Bounds3::<S, f32>::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));

    // Straight through the middle along +X, from x = -5.
    let origin = v3(-5.0, 0.0, 0.0);
    let dir = v3(1.0, 0.0, 0.0);

    let (t0, t1) = b
        .intersect_ray(origin, dir.reciprocal_exact())
        .expect("a ray through the centre must hit");

    assert!(close(t0, 4.0), "entry t = {t0}");
    assert!(t1 >= 6.0 && t1 < 6.001, "exit t = {t1} (4-ulp widened)");

    // A ray pointing away still reports the hit behind it only if tmax >= 0;
    // fully-behind misses.
    assert!(b.intersect_ray(v3(5.0, 0.0, 0.0), dir.reciprocal_exact()).is_none());

    // A miss: parallel to the box but offset.
    assert!(b.intersect_ray(v3(-5.0, 3.0, 0.0), dir.reciprocal_exact()).is_none());

    // An axis-aligned direction gives infinities in inv_dir on the other axes,
    // which must NOT produce NaN and lose the hit.
    let inside = b.intersect_ray(v3(0.0, 0.0, 0.0), dir.reciprocal_exact());
    assert!(inside.is_some(), "a ray starting inside must hit");
}

// ------------------------------------------------------------- transform ----

#[test]
fn transform_pairs_forward_and_inverse() {
    let t = Transform::<S, f32>::translate(v3(1.0, 2.0, 3.0));

    let p = v3(0.5, -1.0, 2.0);

    assert_v3(t.transform_point(p), [1.5, 1.0, 5.0]);
    // The inverse is just the swapped pair, and it round-trips.
    assert_v3_eq(t.invert().transform_point(t.transform_point(p)), p);

    // Scale's inverse is the reciprocal, with no inversion performed.
    let s = Transform::<S, f32>::scale(v3(2.0, 4.0, 0.5));
    assert_v3_eq(s.invert().transform_point(s.transform_point(p)), p);

    // A zero axis collapses rather than producing infinities.
    let degenerate = Transform::<S, f32>::scale(v3(1.0, 0.0, 1.0));
    assert!(
        arr(degenerate.inverse.transform_point(v3(1.0, 1.0, 1.0)))
            .iter()
            .all(|c| c.is_finite())
    );

    // Composition applies rhs first, and the inverses compose in reverse.
    let composed = t * s;
    assert_v3_eq(composed.transform_point(p), t.transform_point(s.transform_point(p)));
    assert_v3_eq(composed.invert().transform_point(composed.transform_point(p)), p);
}

#[test]
fn transform_normal_matches_the_inverse_transpose() {
    let t = Transform::<S, f32>::scale(v3(2.0, 1.0, 1.0));

    let n = v3(1.0, -1.0, 0.0);
    let tangent = v3(1.0, 1.0, 0.0);

    let n2 = t.transform_normal(n);
    let t2 = t.transform_vector(tangent);

    assert!(close(n2.dot3(t2), 0.0), "normal stays perpendicular");

    // The reverse brings it back (up to scale).
    assert_v3_eq(t.reverse_transform_normal(n2).normalize(), n.normalize());
}

#[test]
fn transform_look_at_builds_an_orthonormal_view() {
    let eye = v3(0.0, 0.0, -5.0);
    let target = v3(0.0, 0.0, 0.0);

    let t = Transform::<S, f32>::look_at(eye, target, v3(0.0, 1.0, 0.0));

    // The eye maps to the camera-space origin.
    assert_v3(t.transform_point(eye), [0.0, 0.0, 0.0]);

    // The target lies straight down +Z at the eye-target distance.
    let seen = t.transform_point(target);
    assert!(close(seen[0], 0.0) && close(seen[1], 0.0), "target should be centred");
    assert!(close(seen[2], 5.0), "target depth = {}", seen[2]);

    // forward and inverse really are inverses.
    let p = v3(1.0, 2.0, 3.0);
    assert_v3_eq(t.invert().transform_point(t.transform_point(p)), p);

    // A degenerate direction yields the identity rather than NaN.
    let bad = Transform::<S, f32>::look_at(eye, eye, v3(0.0, 1.0, 0.0));
    assert!(bad.is_identity(1e-6));
}

#[test]
fn transform_rotate_round_trips() {
    let axis_angle = v3(0.3, 0.5, -0.2);

    let t = Transform::<S, f32>::rotate(axis_angle);

    let p = v3(1.0, 2.0, 3.0);

    // A rotation preserves length...
    assert!(close(t.transform_vector(p).norm(), p.norm()));
    // ... and its inverse is its transpose, so it round-trips.
    assert_v3_eq(t.invert().transform_point(t.transform_point(p)), p);

    // Zero rotation is the identity.
    assert!(Transform::<S, f32>::rotate(V3::ZERO).is_identity(1e-6));
}

// ---------------------------------------------------------------- frames ----

#[test]
fn tangent_frame_is_orthonormal_everywhere() {
    // Both poles included: nz near -1 is where the naive Frisvad formula dies.
    let normals = [
        v3(0.0, 0.0, 1.0),
        v3(0.0, 0.0, -1.0),
        v3(1.0, 0.0, 0.0),
        v3(0.0, 1.0, 0.0),
        v3(0.577, 0.577, 0.577),
        v3(0.001, 0.001, -0.999_999),
    ];

    for n in normals {
        let n = n.normalize();
        let f = TangentFrame::<S, f32>::new(n);

        assert!(close(f.tangent.norm(), 1.0), "tangent not unit for {:?}", arr(n));
        assert!(close(f.bitangent.norm(), 1.0), "bitangent not unit for {:?}", arr(n));

        assert!(close(f.tangent.dot3(f.bitangent), 0.0), "t.b != 0 for {:?}", arr(n));
        assert!(close(f.tangent.dot3(n), 0.0), "t.n != 0 for {:?}", arr(n));
        assert!(close(f.bitangent.dot3(n), 0.0), "b.n != 0 for {:?}", arr(n));

        // to_local and to_world are mutual inverses, and +Z is the normal.
        let w = v3(0.3, -0.6, 0.9);
        assert_v3_eq(f.to_world(f.to_local(w)), w);
        assert_v3(f.to_local(n), [0.0, 0.0, 1.0]);
    }
}

#[test]
fn tangent_frame_partial_falls_back_when_unusable() {
    let n = v3(0.0, 0.0, 1.0);

    // A usable mesh tangent is kept (normalized).
    let f = TangentFrame::<S, f32>::partial(n, v3(2.0, 0.0, 0.0));
    assert_v3(f.tangent, [1.0, 0.0, 0.0]);

    // A tangent parallel to the normal is unusable, so fall back and stay orthonormal.
    let degenerate = TangentFrame::<S, f32>::partial(n, n);
    assert!(close(degenerate.tangent.norm(), 1.0));
    assert!(close(degenerate.tangent.dot3(n), 0.0));

    // So is a zero tangent.
    let zero = TangentFrame::<S, f32>::partial(n, V3::ZERO);
    assert!(close(zero.tangent.norm(), 1.0));
    assert!(close(zero.tangent.dot3(n), 0.0));
}

// ------------------------------------------------------------------ rays ----

#[test]
fn ray_basics() {
    let r = Ray3::<S, f32>::new(v3(1.0, 0.0, 0.0), v3(0.0, 2.0, 0.0));

    assert_v3(r.at(0.0), [1.0, 0.0, 0.0]);
    assert_v3(r.at(2.0), [1.0, 4.0, 0.0]);

    assert_v3(r.move_to(1.5).origin, [1.0, 3.0, 0.0]);
    assert_v3((-r).direction, [0.0, -2.0, 0.0]);

    // An axis-aligned direction gives signed infinities, on purpose.
    let inv = r.inv_direction();
    assert!(inv[0].is_infinite() && close(inv[1], 0.5));
}

#[test]
fn ray_transform_preserves_the_parameterization() {
    let m = M4::from_translation(v3(1.0, 2.0, 3.0)) * M4::from_scale(v3(2.0, 2.0, 2.0));

    let r = Ray3::<S, f32>::new(v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0));

    let t = r.transform(&m);

    // The direction is NOT renormalized, so a t in object space is a t in world
    // space: r'(t) == M r(t).
    for probe in [0.0f32, 0.5, 2.0] {
        assert_v3_eq(t.at(probe), m.transform_point(r.at(probe)));
    }

    // transform_normalized instead rescales, and hands back the factor.
    let (n, len) = r.transform_normalized(&m);
    assert!(close(len, 2.0), "scale factor = {len}");
    assert!(close(n.direction.norm(), 1.0));
    // A t in the old parameterization maps to t * len in the new one.
    assert_v3_eq(n.at(0.5 * len), m.transform_point(r.at(0.5)));
}

#[test]
fn ray_error_bounds_are_conservative_and_finite() {
    // gamma(n) is tiny but strictly positive and grows with n.
    let g3 = gamma::<f32>(3);
    assert!(g3 > 0.0 && g3 < 1e-6);
    assert!(gamma::<f32>(7) > g3);

    let m = M4::from_translation(v3(1.0, 2.0, 3.0));

    let (p, e) = m.transform_point_with_error(v3(1.0, 1.0, 1.0));

    assert_v3(p, [2.0, 3.0, 4.0]);

    for c in arr(e) {
        assert!(c >= 0.0 && c < 1e-4, "error bound {c} must be small and non-negative");
    }
}

#[test]
fn ray_offset_origin_escapes_the_surface() {
    let p = v3(1.0, 2.0, 3.0);
    let normal = v3(0.0, 1.0, 0.0);
    let error = v3(1e-5, 1e-5, 1e-5);

    // A ray leaving along the normal is pushed to the +y side.
    let out = Ray3::<S, f32>::offset_origin(p, error, normal, normal);
    assert!(
        out[1] > p[1],
        "offset must move off the surface: {} vs {}",
        out[1],
        p[1]
    );

    // A ray leaving into the surface is pushed the other way.
    let into = Ray3::<S, f32>::offset_origin(p, error, normal, -normal);
    assert!(into[1] < p[1]);

    // The offset is small, scaling with the error rather than a fixed epsilon.
    assert!((out[1] - p[1]).abs() < 1e-3);

    // Zero error still moves off the surface by at least one ulp, which is the
    // whole point of the final nudge.
    let tiny = Ray3::<S, f32>::offset_origin(p, V3::ZERO, normal, normal);
    assert!(tiny[1] >= p[1]);
}

// ---------------------------------------------- AoS <-> SoA cross-checks ----

/// The two layouts are separate implementations of the same geometry, so they
/// must agree. These run the same inputs through both and compare.
mod cross_layout {
    use super::*;

    use thermite_geometry::soa::prim::{
        Bounds as SoaBounds, Matrix as SoaMatrix, Point as SoaPoint, Vector as SoaVector, vector::VectorOps as _,
    };

    /// 1-lane SoA vectors, so a lane reads back as a plain scalar.
    type L = thermite::Vector<f32>;

    fn soa_vec(x: f32, y: f32, z: f32) -> SoaVector<L, 3> {
        SoaVector::new([L::splat(x), L::splat(y), L::splat(z)])
    }

    fn soa_pt(x: f32, y: f32, z: f32) -> SoaPoint<L, 3> {
        SoaPoint::new([L::splat(x), L::splat(y), L::splat(z)])
    }

    fn soa_arr(v: SoaVector<L, 3>) -> [f32; 3] {
        [v[0].extract::<0>(), v[1].extract::<0>(), v[2].extract::<0>()]
    }

    fn soa_pt_arr(p: SoaPoint<L, 3>) -> [f32; 3] {
        soa_arr(SoaVector::from(p))
    }

    #[track_caller]
    fn agree(aos: [f32; 3], soa: [f32; 3], what: &str) {
        for i in 0..3 {
            assert!(
                close(aos[i], soa[i]),
                "{what} component {i}: aos {} != soa {}",
                aos[i],
                soa[i]
            );
        }
    }

    #[test]
    fn dot_cross_and_normalize_agree() {
        let (a, b) = ([1.0, 2.0, 3.0], [-4.0, 0.5, 6.0]);

        let av = v3(a[0], a[1], a[2]);
        let bv = v3(b[0], b[1], b[2]);

        let sa = soa_vec(a[0], a[1], a[2]);
        let sb = soa_vec(b[0], b[1], b[2]);

        assert!(close(av.dot3(bv), sa.dot(&sb).extract::<0>()), "dot");
        assert!(close(av.norm(), sa.l2_norm().extract::<0>()), "norm");
        assert!(close(av.l1_norm3(), sa.l1_norm().extract::<0>()), "l1");
        assert!(close(av.linf_norm3(), sa.linf_norm().extract::<0>()), "linf");

        agree(arr(av.cross3::<false>(bv)), soa_arr(sa.cross(sb)), "cross");
        agree(arr(av.normalize()), soa_arr(sa.normalize()), "normalize");
    }

    #[test]
    fn shading_ops_agree() {
        let n = [0.0, 1.0, 0.0];
        let i = [0.6, -0.8, 0.0];

        let (av, an) = (v3(i[0], i[1], i[2]), v3(n[0], n[1], n[2]));
        let (sv, sn) = (soa_vec(i[0], i[1], i[2]), soa_vec(n[0], n[1], n[2]));

        agree(arr(av.reflect(an)), soa_arr(sv.reflect(&sn)), "reflect");

        for eta in [0.5f32, 0.75, 1.5] {
            agree(
                arr(av.refract(an, eta)),
                soa_arr(sv.refract(&sn, L::splat(eta))),
                "refract",
            );
        }

        // ... including the total-internal-reflection case, where both give zero.
        let steep = [3.0f32.sqrt() / 2.0, -0.5, 0.0];
        let (asteep, ssteep) = (v3(steep[0], steep[1], steep[2]), soa_vec(steep[0], steep[1], steep[2]));

        agree(
            arr(asteep.refract(an, 2.0)),
            soa_arr(ssteep.refract(&sn, L::splat(2.0))),
            "refract TIR",
        );
    }

    #[test]
    fn point_transforms_agree() {
        // The same affine transform built in both layouts.
        let translation = [1.0f32, -2.0, 0.5];
        let scale = [2.0f32, 3.0, 4.0];

        let aos_m = M4::from_translation(v3(translation[0], translation[1], translation[2]))
            * M4::from_scale(v3(scale[0], scale[1], scale[2]));

        let soa_m = SoaMatrix::<L, 4, 4>::from_translation(soa_vec(translation[0], translation[1], translation[2]))
            * SoaMatrix::<L, 4, 4>::from_scale(soa_vec(scale[0], scale[1], scale[2]));

        for p in [[1.0f32, 1.0, 1.0], [0.0, 0.0, 0.0], [-3.0, 2.5, 7.0]] {
            agree(
                arr(aos_m.transform_point(v3(p[0], p[1], p[2]))),
                soa_pt_arr(soa_m.transform_point(soa_pt(p[0], p[1], p[2]))),
                "transform_point",
            );

            agree(
                arr(aos_m.transform_vector(v3(p[0], p[1], p[2]))),
                soa_arr(soa_m.transform_vector(soa_vec(p[0], p[1], p[2]))),
                "transform_vector",
            );
        }

        // The inverses agree too.
        let aos_inv = aos_m.invert().expect("invertible");
        let (soa_inv, ok) = soa_m.try_invert();
        assert!(ok.all());

        agree(
            arr(aos_inv.transform_point(v3(1.0, 2.0, 3.0))),
            soa_pt_arr(soa_inv.transform_point(soa_pt(1.0, 2.0, 3.0))),
            "inverse transform_point",
        );
    }

    #[test]
    fn determinants_agree() {
        let aos_m = M4::from_scale(v3(2.0, 3.0, 4.0));
        let soa_m = SoaMatrix::<L, 4, 4>::from_scale(soa_vec(2.0, 3.0, 4.0));

        assert!(
            close(aos_m.determinant(), soa_m.determinant().extract::<0>()),
            "determinant: {} != {}",
            aos_m.determinant(),
            soa_m.determinant().extract::<0>()
        );
    }

    #[test]
    fn bounds_queries_agree() {
        let (lo, hi) = ([-1.0f32, -2.0, -3.0], [4.0f32, 5.0, 6.0]);

        let aos_b = Bounds3::<S, f32>::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));

        let soa_b = SoaBounds::<L, 3>::from_corners(soa_vec(lo[0], lo[1], lo[2]), soa_vec(hi[0], hi[1], hi[2]));

        assert!(close(aos_b.volume(), soa_b.volume().extract::<0>()), "volume");
        assert!(
            close(aos_b.surface_area(), soa_b.surface_area().extract::<0>()),
            "surface_area: {} != {}",
            aos_b.surface_area(),
            soa_b.surface_area().extract::<0>()
        );

        agree(arr(aos_b.diagonal()), soa_arr(soa_b.diagonal()), "diagonal");
        agree(arr(aos_b.centroid()), soa_pt_arr(soa_b.centroid()), "centroid");

        // The split axis a BVH builder would pick must match.
        assert_eq!(
            aos_b.max_extent_axis() as u32,
            soa_b.max_extent_axis().extract::<0>(),
            "max_extent_axis"
        );

        // Corner order must match, or a transform_bounds would silently differ.
        for i in 0..8u8 {
            agree(arr(aos_b.vertex(i)), soa_pt_arr(soa_b.vertex(i as usize)), "vertex");
        }

        let p = [2.0f32, -7.0, 0.0];
        agree(
            arr(aos_b.closest_point(v3(p[0], p[1], p[2]))),
            soa_pt_arr(soa_b.closest_point(soa_pt(p[0], p[1], p[2]))),
            "closest_point",
        );
        assert!(
            close(
                aos_b.distance_sqr(v3(p[0], p[1], p[2])),
                soa_b.distance_sqr(soa_pt(p[0], p[1], p[2])).extract::<0>()
            ),
            "distance_sqr"
        );

        agree(
            arr(aos_b.offset(v3(p[0], p[1], p[2]))),
            soa_arr(soa_b.offset(soa_pt(p[0], p[1], p[2]))),
            "offset",
        );
    }

    #[test]
    fn ray_slab_test_agrees() {
        use thermite_geometry::soa::prim::{Ray as SoaRay, ray::RayOps as _};

        let aos_b = Bounds3::<S, f32>::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
        let soa_b = SoaBounds::<L, 3>::from_corners(soa_vec(-1.0, -1.0, -1.0), soa_vec(1.0, 1.0, 1.0));

        // A hit, a miss, and a diagonal, the three cases worth pinning.
        let cases = [
            ([-5.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0]),
            ([-5.0, 3.0, 0.0], [1.0, 0.0, 0.0]),
            ([-5.0, -5.0, -5.0], [1.0, 1.0, 1.0]),
        ];

        for (o, d) in cases {
            let ao = v3(o[0], o[1], o[2]);
            let ad = v3(d[0], d[1], d[2]);

            let aos_hit = aos_b.intersect_ray(ao, ad.reciprocal_exact());

            let sray = SoaRay::new(soa_pt(o[0], o[1], o[2]), soa_vec(d[0], d[1], d[2]));
            let (t0, t1, mask) = sray.intersects_aabb_mask(&soa_b, &sray.inv_direction());

            assert_eq!(
                aos_hit.is_some(),
                mask.all(),
                "hit/miss disagreement for origin {o:?} dir {d:?}"
            );

            if let Some((a0, a1)) = aos_hit {
                assert!(close(a0, t0.extract::<0>()), "t_min: {} != {}", a0, t0.extract::<0>());
                assert!(close(a1, t1.extract::<0>()), "t_max: {} != {}", a1, t1.extract::<0>());
            }
        }
    }

    #[test]
    fn tangent_frames_agree() {
        use thermite_geometry::soa::prim::TangentFrame as SoaFrame;

        for n in [
            [0.0f32, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.577, 0.577, 0.577],
            [0.001, 0.001, -0.999_999],
        ] {
            let an = v3(n[0], n[1], n[2]).normalize();
            let sn = soa_vec(n[0], n[1], n[2]).normalize();

            let af = TangentFrame::<S, f32>::new(an);
            let sf = SoaFrame::<L>::new(sn);

            agree(arr(af.tangent), soa_arr(sf.tangent), "tangent");
            agree(arr(af.bitangent), soa_arr(sf.bitangent), "bitangent");
        }
    }
}

// ------------------------------------------------ decompose / kinematics ----

#[test]
fn decompose_recovers_translation_rotation_scale() {
    let t = v3(1.0, -2.0, 3.0);
    let axis = v3(0.3, 1.0, -0.2).normalize();
    let angle = 0.7f32;
    let s = v3(2.0, 3.0, 4.0);

    // Build T * R * S, the order decompose assumes.
    let m = M4::from_translation(t) * M4::from_axis_angle(axis, angle) * M4::from_scale(s);

    let xf = Transform::<S, f32>::new(m).expect("invertible");

    let (got_t, got_r, got_s) = xf.decompose();

    assert_v3_eq(got_t, t);

    // The rotation is compared by its action, since q and -q are the same rotation.
    let reference = Quat::from_axis_angle_raw(axis, angle);

    for probe in [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)] {
        assert_v3_eq(
            got_r.rotate_vector::<false>(probe),
            reference.rotate_vector::<false>(probe),
        );
    }

    // The scale columns are the original scale on the diagonal, nothing off it.
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { arr(s)[i] } else { 0.0 };

            assert!(
                close(arr(got_s[i])[j], want),
                "scale ({j}, {i}) = {} != {want}",
                arr(got_s[i])[j]
            );
        }
    }
}

#[test]
fn decompose_round_trips_through_recomposition() {
    let cases = [
        // pure rotation
        (v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), 1.1f32, v3(1.0, 1.0, 1.0)),
        // translate + rotate
        (v3(5.0, 0.0, -3.0), v3(1.0, 1.0, 1.0), 0.4, v3(1.0, 1.0, 1.0)),
        // uniform scale
        (v3(1.0, 2.0, 3.0), v3(0.0, 0.0, 1.0), 2.0, v3(3.0, 3.0, 3.0)),
        // non-uniform scale
        (v3(-1.0, 0.5, 2.0), v3(0.2, -0.7, 0.5), 0.9, v3(1.0, 2.0, 0.5)),
    ];

    for (t, raw_axis, angle, s) in cases {
        let axis = raw_axis.normalize();

        let m = M4::from_translation(t) * M4::from_axis_angle(axis, angle) * M4::from_scale(s);

        let (dt, dr, ds) = Transform::<S, f32>::new(m).expect("invertible").decompose();

        // Recompose and check it transforms points the same way as the original.
        let mut scale4 = M4::IDENTITY;
        for c in 0..3 {
            for r in 0..3 {
                scale4.set(r, c, arr(ds[c])[r]);
            }
        }

        let rebuilt = M4::from_translation(dt) * dr.to_matrix() * scale4;

        for p in [
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
            v3(1.0, 2.0, 3.0),
            v3(-2.0, 0.5, 1.0),
        ] {
            let want = m.transform_point(p);
            let got = rebuilt.transform_point(p);

            for i in 0..3 {
                assert!(
                    (arr(got)[i] - arr(want)[i]).abs() < 1e-3,
                    "recomposed transform differs at component {i}: {} != {}",
                    arr(got)[i],
                    arr(want)[i]
                );
            }
        }
    }
}

#[test]
fn decompose_detects_mirroring() {
    // A negative scale on one axis mirrors: the determinant flips sign, which is
    // how a renderer knows winding order and normal handedness have reversed.
    let mirrored = M4::from_scale(v3(1.0, -1.0, 1.0));

    assert!(mirrored.determinant() < 0.0, "a mirror must have negative determinant");

    // The polar factor is still orthogonal (its own determinant is +/-1), and the
    // mirror lands in the scale factor where it can be inspected.
    let (_, _, s) = Transform::<S, f32>::new(mirrored).expect("invertible").decompose();

    let diag = [arr(s[0])[0], arr(s[1])[1], arr(s[2])[2]];

    assert!(
        diag.iter().any(|&d| d < 0.0),
        "the mirror must survive into the scale factor: {diag:?}"
    );
}

#[test]
fn quaternion_exp_log_round_trip() {
    for raw in [[0.0f32, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, -0.7, 1.1], [0.0, 2.0, 0.0]] {
        let rotation_vector = v3(raw[0], raw[1], raw[2]);

        // log(exp(v)) == v for any rotation vector shorter than a half turn.
        let back = Quat::exp(rotation_vector).log();

        assert_v3_eq(back, rotation_vector);
    }

    // exp of the zero vector is the identity, not a NaN.
    assert_v3(Quat::exp(V3::ZERO).log(), [0.0, 0.0, 0.0]);
    assert!(close(Quat::exp(V3::ZERO).w(), 1.0));

    // ... and log of the identity is the zero vector.
    assert_v3(Quat::IDENTITY.log(), [0.0, 0.0, 0.0]);
}

#[test]
fn quaternion_log_is_the_difference_between_orientations() {
    let a = Quat::from_axis_angle_raw(v3(0.0, 0.0, 1.0), 0.3);
    let b = Quat::from_axis_angle_raw(v3(0.0, 0.0, 1.0), 1.1);

    // The rotation carrying a onto b, as a plain vector.
    let delta = (b * a.inverse()).log();

    // Both are about +Z, so the difference is a pure +Z rotation of 0.8 rad.
    assert_v3(delta, [0.0, 0.0, 0.8]);

    // Applying it really does carry a onto b.
    let rebuilt = Quat::exp(delta) * a;

    for probe in [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)] {
        assert_v3_eq(rebuilt.rotate_vector::<false>(probe), b.rotate_vector::<false>(probe));
    }
}

#[test]
fn quaternion_slerp_equals_the_exp_log_form() {
    // slerp(q0, q1, t) == q0 * exp(t * log(q1 * q0^-1)), which is what makes
    // exp/log and slerp two views of one operation.
    let a = Quat::from_euler(0.2, -0.5, 0.9).normalize();
    let b = Quat::from_euler(-0.7, 0.3, 0.1).normalize();

    for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let via_slerp = a.slerp(b, t);

        let delta = (b * a.inverse()).log();
        let via_exp = Quat::exp(delta * V3::splat(t)) * a;

        for probe in [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)] {
            let want = via_slerp.rotate_vector::<false>(probe);
            let got = via_exp.rotate_vector::<false>(probe);

            for i in 0..3 {
                assert!(
                    (arr(got)[i] - arr(want)[i]).abs() < 1e-4,
                    "t = {t}: exp/log form differs from slerp at component {i}"
                );
            }
        }
    }
}

#[test]
fn quaternion_integrate_matches_a_closed_form_spin() {
    // A body spinning about +Z at 2 rad/s for 1.5 s has turned 3 rad.
    let omega = v3(0.0, 0.0, 2.0);

    let mut q = Quat::IDENTITY;

    // Many small steps must converge on the closed-form answer. This is the
    // property the exponential update buys and the first-order one does not.
    let steps = 1500;
    let dt = 1.5 / steps as f32;

    for _ in 0..steps {
        q = q.integrate(omega, dt);
    }

    let expected = Quat::from_axis_angle_raw(v3(0.0, 0.0, 1.0), 3.0);

    for probe in [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)] {
        let want = expected.rotate_vector::<false>(probe);
        let got = q.rotate_vector::<false>(probe);

        for i in 0..3 {
            assert!(
                (arr(got)[i] - arr(want)[i]).abs() < 1e-3,
                "integrated spin differs at component {i}: {} != {}",
                arr(got)[i],
                arr(want)[i]
            );
        }
    }

    // Integration must not drift off the unit sphere.
    assert!(close(q.norm(), 1.0), "norm drifted to {}", q.norm());
}

#[test]
fn quaternion_integrate_is_exact_in_one_big_step() {
    // The exponential update is a true rotation however large the step, so one
    // step of a half turn is exact, where a first-order update would badly undershoot.
    let omega = v3(0.0, core::f32::consts::PI, 0.0);

    let q = Quat::IDENTITY.integrate(omega, 1.0);

    // A half turn about +Y takes +X to -X.
    assert_v3(q.rotate_vector::<false>(v3(1.0, 0.0, 0.0)), [-1.0, 0.0, 0.0]);
    assert!(close(q.norm(), 1.0));

    // A zero angular velocity leaves the orientation alone.
    let still = Quat::from_euler(0.3, 0.4, 0.5).normalize();
    let after = still.integrate(V3::ZERO, 0.016);

    for probe in [v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0)] {
        assert_v3_eq(after.rotate_vector::<false>(probe), still.rotate_vector::<false>(probe));
    }
}

// ------------------------------------------------------- symmetry additions ----

#[test]
fn quaternion_look_at_aims_plus_z() {
    let origin = v3(0.0, 0.0, 0.0);

    // Looking toward +X should carry the camera's +Z onto +X.
    let q = Quat::look_at(origin, v3(1.0, 0.0, 0.0));
    assert_v3(q.rotate_vector::<false>(v3(0.0, 0.0, 1.0)), [1.0, 0.0, 0.0]);

    // Straight up and straight down are the pole cases the branch guards.
    let up = Quat::look_at(origin, v3(0.0, 1.0, 0.0));
    assert_v3(up.rotate_vector::<false>(v3(0.0, 0.0, 1.0)), [0.0, 1.0, 0.0]);

    let down = Quat::look_at(origin, v3(0.0, -1.0, 0.0));
    assert_v3(down.rotate_vector::<false>(v3(0.0, 0.0, 1.0)), [0.0, -1.0, 0.0]);

    // A target coincident with the origin names no direction: identity, no NaN.
    let degenerate = Quat::look_at(origin, origin);
    assert!(degenerate.is_finite());
    assert_v3(degenerate.rotate_vector::<false>(v3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0]);
}

#[test]
fn matrix_projection_constructors_match_the_transform_ones() {
    let fov = core::f32::consts::FRAC_PI_2;

    let from_matrix = M4::perspective(fov, 1.0, 100.0);
    let from_transform = Transform::<S, f32>::perspective(fov, 1.0, 100.0);

    for c in 0..4 {
        for r in 0..4 {
            assert!(
                close(from_matrix.get(r, c), from_transform.forward.get(r, c)),
                "perspective ({r}, {c})"
            );
        }
    }

    let ortho_m = M4::orthographic(1.0, 100.0);
    let ortho_t = Transform::<S, f32>::ortho(1.0, 100.0);

    for c in 0..4 {
        for r in 0..4 {
            assert!(close(ortho_m.get(r, c), ortho_t.forward.get(r, c)), "ortho ({r}, {c})");
        }
    }
}

#[test]
fn matrix_trace_and_transform_bounds() {
    assert!(close(M4::IDENTITY.trace(), 4.0));
    assert!(close(M4::from_scale(v3(2.0, 3.0, 4.0)).trace(), 10.0));

    // transform_bounds on the matrix agrees with Bounds3::transform.
    let b = Bounds3::<S, f32>::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
    let m = M4::from_translation(v3(5.0, 0.0, 0.0));

    let via_matrix = m.transform_bounds(&b);
    let via_bounds = b.transform(&m);

    assert_v3_eq(via_matrix.min, via_bounds.min);
    assert_v3_eq(via_matrix.max, via_bounds.max);
    assert_v3(via_matrix.min, [4.0, -1.0, -1.0]);
}

#[test]
fn bounds_symmetric_and_ray_error_constructors() {
    let b = Bounds3::<S, f32>::symmetric(v3(1.0, 2.0, 3.0));

    assert_v3(b.min, [-1.0, -2.0, -3.0]);
    assert_v3(b.max, [1.0, 2.0, 3.0]);

    let e = v3(1e-4, 2e-4, 3e-4);

    let pos_only = RayError3::<S, f32>::position(e);
    assert_v3_eq(pos_only.pos, e);
    assert_v3(pos_only.dir, [0.0, 0.0, 0.0]);

    let dir_only = RayError3::<S, f32>::direction(e);
    assert_v3(dir_only.pos, [0.0, 0.0, 0.0]);
    assert_v3_eq(dir_only.dir, e);
}

#[test]
fn error_propagation_grows_the_bound() {
    let m = M4::from_scale(v3(2.0, 2.0, 2.0));

    let p = v3(1.0, 1.0, 1.0);
    let incoming = v3(1e-3, 1e-3, 1e-3);

    let (_, fresh) = m.transform_point_with_error(p);
    let (_, propagated) = m.transform_point_propagate_error(p, incoming);

    // Carrying an incoming error in must widen the bound. That is the whole
    // point, and understating it is what makes a chained instance transform
    // spawn rays that re-hit their own surface.
    for i in 0..3 {
        assert!(
            arr(propagated)[i] > arr(fresh)[i],
            "component {i}: propagated {} must exceed fresh {}",
            arr(propagated)[i],
            arr(fresh)[i]
        );
        assert!(arr(propagated)[i].is_finite());
    }

    // A zero incoming error reduces to the fresh bound.
    let (_, none) = m.transform_point_propagate_error(p, V3::ZERO);

    for i in 0..3 {
        assert!(close(arr(none)[i], arr(fresh)[i]), "zero incoming error must match");
    }
}
