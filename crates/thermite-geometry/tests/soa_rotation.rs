//! Behavioural tests for the rotation/shading types: `Quaternion`, `Transform`,
//! `TangentFrame`, and `Ray`.
//!
//! All of these replaced a branch in the AoS original with a mask select, so the
//! degenerate cases (zero-length axis, poles, singular matrices, parallel
//! tangents) are tested as carefully as the happy paths, since a select that picks the
//! wrong side is silent, where a branch would have panicked.

use thermite::prelude::*;

use thermite_geometry::soa::prim::{
    Bounds, Matrix, Point, Quaternion, Ray, TangentFrame, Transform, Vector as GVector, ray::RayOps as _,
    vector::VectorOps as _,
};

type V = thermite::Vector<f32>;

#[inline]
fn v(x: f32) -> V {
    V::splat(x)
}

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
    (a - b).abs() < 1e-4
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

const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;

// ------------------------------------------------------------ Quaternion ----

#[test]
fn quaternion_basics() {
    let q = Quaternion::<V>::IDENTITY;

    assert!(close(s(q.norm()), 1.0));
    assert!(q.is_finite().all());
    assert_vec3(q.rotate_vector(vec3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0]);

    // A quarter turn about +Z takes +X to +Y.
    let rot = Quaternion::from_axis_angle_raw(vec3(0.0, 0.0, 1.0), v(FRAC_PI_2));

    assert_vec3(rot.rotate_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
    assert_vec3(rot.rotate_vector(vec3(0.0, 1.0, 0.0)), [-1.0, 0.0, 0.0]);

    // The conjugate of a unit quaternion is the inverse rotation.
    assert_vec3(rot.conjugate().rotate_vector(vec3(0.0, 1.0, 0.0)), [1.0, 0.0, 0.0]);
    assert_vec3(
        (rot * rot.conjugate()).rotate_vector(vec3(1.0, 2.0, 3.0)),
        [1.0, 2.0, 3.0],
    );

    // A degenerate quaternion has no direction, so try_normalize must give the
    // identity rather than NaN.
    let degenerate = Quaternion::<V>::new(V::ZERO, V::ZERO, V::ZERO, V::ZERO).try_normalize();
    assert_vec3(degenerate.rotate_vector(vec3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0]);
}

/// `rotate_vector` (the two-cross-product identity) and `to_matrix` must agree -
/// they are independent derivations of the same rotation.
#[test]
fn quaternion_matches_its_matrix() {
    let q = Quaternion::from_axis_angle(vec3(0.3, -0.5, 0.8)).try_normalize();
    let m = q.to_matrix();

    for p in [vec3(1.0, 0.0, 0.0), vec3(0.2, 1.3, -0.7), vec3(-2.0, 0.5, 0.1)] {
        let by_quat = q.rotate_vector(p);
        let by_matrix = m.transform_vector(p);

        for i in 0..3 {
            assert!(close(s(by_quat[i]), s(by_matrix[i])), "axis {i}");
        }
    }

    // A rotation matrix is orthogonal with unit determinant.
    assert!(close(s(m.determinant()), 1.0));
}

/// Shepperd's method picks one of four formulas by which component is largest.
/// In SoA that four-way branch became a chain of selects, so every arm needs a
/// rotation that actually lands on it.
#[test]
fn quaternion_matrix_round_trip_hits_every_shepperd_branch() {
    let cases = [
        vec3(0.0, 0.0, 0.0),  // identity: trace dominates (the w branch)
        vec3(3.1, 0.0, 0.0),  // ~pi about x: the x branch
        vec3(0.0, 3.1, 0.0),  // ~pi about y: the y branch
        vec3(0.0, 0.0, 3.1),  // ~pi about z: the z branch
        vec3(0.7, -1.2, 0.4), // a generic rotation
    ];

    for axis_angle in cases {
        let q = Quaternion::from_axis_angle(axis_angle).try_normalize();
        let back = Quaternion::from_matrix(&q.to_matrix());

        // q and -q are the same rotation, so compare by what they DO, not by
        // their components.
        for p in [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)] {
            let a = q.rotate_vector(p);
            let b = back.rotate_vector(p);

            for i in 0..3 {
                assert!(
                    close(s(a[i]), s(b[i])),
                    "round trip failed for axis_angle {:?}, axis {i}: {} != {}",
                    [s(axis_angle[0]), s(axis_angle[1]), s(axis_angle[2])],
                    s(a[i]),
                    s(b[i])
                );
            }
        }
    }
}

#[test]
fn quaternion_slerp() {
    let a = Quaternion::<V>::IDENTITY;
    let b = Quaternion::from_axis_angle_raw(vec3(0.0, 0.0, 1.0), v(FRAC_PI_2));

    // Endpoints are exact.
    assert_vec3(a.slerp(b, v(0.0)).rotate_vector(vec3(1.0, 0.0, 0.0)), [1.0, 0.0, 0.0]);
    assert_vec3(a.slerp(b, v(1.0)).rotate_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);

    // The midpoint of a quarter turn is an eighth turn: +X lands at 45 degrees.
    let half = a.slerp(b, v(0.5));
    let h = core::f32::consts::FRAC_1_SQRT_2;
    assert_vec3(half.rotate_vector(vec3(1.0, 0.0, 0.0)), [h, h, 0.0]);

    // Slerp of a quaternion with itself takes the near-identical fast path
    // (cos_theta ~ 1) and must not divide by a zero sin(omega).
    let same = b.slerp(b, v(0.5));
    assert!(same.is_finite().all());
    assert_vec3(same.rotate_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
}

#[test]
fn quaternion_rotation_arc() {
    let q = Quaternion::from_rotation_arc(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));

    // The shortest arc from +X to +Y must actually land on +Y.
    assert_vec3(q.rotate_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
}

// ------------------------------------------------------------- Transform ----

#[test]
fn transform_pairs_forward_and_inverse() {
    let t = Transform::<V>::translate(vec3(1.0, 2.0, 3.0));

    assert_point3(t.transform_point(pt3(0.0, 0.0, 0.0)), [1.0, 2.0, 3.0]);
    assert_point3(t.invert().transform_point(pt3(1.0, 2.0, 3.0)), [0.0, 0.0, 0.0]);

    // The cached inverse really is the inverse: forward * inverse == identity.
    let composed = t.forward * t.inverse;
    assert!(composed.is_identity(v(1e-5)).all());

    // Composition applies the right-hand transform first.
    let scale = Transform::<V>::scale(vec3(2.0, 2.0, 2.0));
    let both = t * scale;

    assert_point3(both.transform_point(pt3(1.0, 1.0, 1.0)), [3.0, 4.0, 5.0]);
    // ... and the composed inverse undoes it, which is the whole point of the pair.
    assert_point3(both.invert().transform_point(pt3(3.0, 4.0, 5.0)), [1.0, 1.0, 1.0]);

    assert!(Transform::<V>::IDENTITY.is_identity(v(1e-6)).all());
}

#[test]
fn transform_from_quaternion() {
    let q = Quaternion::from_axis_angle_raw(vec3(0.0, 0.0, 1.0), v(FRAC_PI_2));
    let t = Transform::from_quaternion(q);

    assert_vec3(t.transform_vector(vec3(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0]);
    // Rotations are orthogonal, so the stored inverse (a transpose) must undo it.
    assert_vec3(t.invert().transform_vector(vec3(0.0, 1.0, 0.0)), [1.0, 0.0, 0.0]);
}

/// The reason `transform_normal` exists: under a non-uniform scale a normal does
/// NOT transform like a direction. Squashing y by 1/2 tilts a 45-degree surface,
/// and only the inverse-transpose keeps the normal perpendicular to it.
#[test]
fn transform_normal_uses_the_inverse_transpose() {
    let t = Transform::<V>::scale(vec3(1.0, 0.5, 1.0));

    // A surface whose tangent is (1, 1, 0) has normal (1, -1, 0).
    let tangent = vec3(1.0, 1.0, 0.0);
    let normal = vec3(1.0, -1.0, 0.0);

    let new_tangent = t.transform_vector(tangent);
    let new_normal = t.transform_normal(normal);

    // Perpendicularity is preserved ...
    assert!(close(s(new_tangent.dot(&new_normal)), 0.0));

    // ... which naively transforming the normal as a direction would have broken.
    let wrong = t.transform_vector(normal);
    assert!(!close(s(new_tangent.dot(&wrong)), 0.0));
}

#[test]
fn transform_degenerate_lanes_do_not_panic() {
    // raygon asserts/panics here, but the SoA version must stay finite, because one
    // bad lane cannot be allowed to take down the seven good ones beside it.
    let zero_scale = Transform::<V>::scale(vec3(1.0, 0.0, 1.0));
    let p = zero_scale.transform_point(pt3(1.0, 5.0, 1.0));

    assert!(GVector::from(p).is_finite().all());
    // The collapsed axis stays collapsed rather than becoming an infinity.
    assert!(close(s(p[1]), 0.0));
    assert!(GVector::from(zero_scale.invert().transform_point(p)).is_finite().all());

    // A zero-magnitude axis-angle is not a rotation, so it must fall back to identity.
    let no_rotation = Transform::<V>::rotate(GVector::ZERO);
    assert!(no_rotation.is_identity(v(1e-5)).all());

    // A singular matrix is reported, not panicked on.
    let singular = Matrix::<V, 4, 4>::from_scale(vec3(1.0, 0.0, 1.0));
    assert!(!Transform::try_new(singular).1.all());
}

// ----------------------------------------------------------- TangentFrame ----

#[test]
fn tangent_frame_is_orthonormal_everywhere() {
    // Includes the poles: n_z near -1 is exactly where the naive Frisvad formula
    // divides by zero, and what the copysign sign-trick exists to survive.
    let normals = [
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.267, 0.535, 0.802),
        vec3(-0.577, 0.577, -0.577),
    ];

    for n in normals {
        let n = n.normalize();
        let frame = TangentFrame::new(n);

        let (t, b) = (frame.tangent, frame.bitangent);

        assert!(t.is_finite().all() && b.is_finite().all(), "non-finite frame");

        // Unit length ...
        assert!(close(s(t.norm_sqr()), 1.0), "tangent not unit: {}", s(t.norm_sqr()));
        assert!(close(s(b.norm_sqr()), 1.0), "bitangent not unit: {}", s(b.norm_sqr()));

        // ... and mutually perpendicular.
        assert!(close(s(t.dot(&b)), 0.0), "t . b = {}", s(t.dot(&b)));
        assert!(close(s(t.dot(&n)), 0.0), "t . n = {}", s(t.dot(&n)));
        assert!(close(s(b.dot(&n)), 0.0), "b . n = {}", s(b.dot(&n)));

        // to_local and to_world are inverses, and the normal maps to +Z in
        // shading space (which is what every BSDF assumes).
        assert_vec3(frame.to_local(n), [0.0, 0.0, 1.0]);

        let world = vec3(0.3, -0.6, 0.742);
        let round_trip = frame.to_world(frame.to_local(world));

        for i in 0..3 {
            assert!(close(s(round_trip[i]), s(world[i])), "round trip axis {i}");
        }
    }
}

#[test]
fn tangent_frame_partial_falls_back_when_the_tangent_is_unusable() {
    let n = vec3(0.0, 0.0, 1.0);

    // A usable mesh tangent is used as given.
    let frame = TangentFrame::partial(n, vec3(1.0, 0.0, 0.0));
    assert_vec3(frame.tangent, [1.0, 0.0, 0.0]);
    assert!(close(s(frame.tangent.dot(&frame.bitangent)), 0.0));

    // Zero, non-finite, and parallel-to-the-normal tangents are all unusable: the
    // cross product that defines the bitangent has no direction. Each must fall
    // back to a valid arbitrary frame instead of producing NaN.
    for bad in [GVector::ZERO, vec3(f32::NAN, 0.0, 0.0), vec3(0.0, 0.0, 1.0)] {
        let frame = TangentFrame::partial(n, bad);

        assert!(frame.tangent.is_finite().all(), "tangent went non-finite");
        assert!(frame.bitangent.is_finite().all(), "bitangent went non-finite");
        assert!(close(s(frame.tangent.norm_sqr()), 1.0));
        assert!(close(s(frame.tangent.dot(&frame.normal)), 0.0));
        assert!(close(s(frame.tangent.dot(&frame.bitangent)), 0.0));
    }
}

// ------------------------------------------------------------------- Ray ----

#[test]
fn ray_basics() {
    let ray = Ray::<V, 3>::new(pt3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));

    assert_point3(ray.at(v(2.5)), [0.0, 0.0, 2.5]);
    assert_point3(ray.move_to(v(2.0)).origin, [0.0, 0.0, 2.0]);
    assert_vec3((-ray).direction, [0.0, 0.0, -1.0]);
}

#[test]
fn ray_aabb_slab_test() {
    let unit = Bounds::<V, 3>::from_corners(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));

    // A ray down +Z from z = -5 enters at t = 4 and exits at t = 6.
    let hit = Ray::<V, 3>::new(pt3(0.0, 0.0, -5.0), vec3(0.0, 0.0, 1.0));
    let (t_min, t_max, mask) = hit.intersects_aabb_mask(&unit, &hit.inv_direction());

    assert!(mask.all());
    assert!(close(s(t_min), 4.0));
    // t_max carries the 4-ulp widening, so it is >= 6 but only barely.
    assert!(s(t_max) >= 6.0 && s(t_max) < 6.0 + 1e-3);

    // A ray that passes beside the box must miss.
    let miss = Ray::<V, 3>::new(pt3(5.0, 5.0, -5.0), vec3(0.0, 0.0, 1.0));
    assert!(!miss.intersects_aabb_mask(&unit, &miss.inv_direction()).2.all());

    // A ray starting inside has a negative entry and a positive exit, and still hits.
    let inside = Ray::<V, 3>::new(pt3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let (t_min, t_max, mask) = inside.intersects_aabb_mask(&unit, &inside.inv_direction());

    assert!(mask.all());
    assert!(s(t_min) < 0.0 && s(t_max) > 0.0);

    // An axis-parallel ray divides by a zero direction component, producing
    // infinities in the slab test. That is well-defined (the ray never leaves that
    // slab) and must NOT degrade into a NaN miss.
    let grazing = Ray::<V, 3>::new(pt3(0.5, 0.5, -5.0), vec3(0.0, 0.0, 1.0));
    assert!(grazing.intersects_aabb_mask(&unit, &grazing.inv_direction()).2.all());
}

#[test]
fn ray_transform() {
    let ray = Ray::<V, 3>::new(pt3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let m = Matrix::<V, 4, 4>::from_translation(vec3(10.0, 0.0, 0.0));

    let moved = ray.transform(&m);

    // The origin is a point (it translates) and the direction is a vector (it does not).
    assert_point3(moved.origin, [11.0, 0.0, 0.0]);
    assert_vec3(moved.direction, [0.0, 0.0, 1.0]);

    // The plain transform preserves the parameterization, so a hit at t survives.
    assert_point3(moved.at(v(3.0)), [11.0, 0.0, 3.0]);

    // Under a scale, the direction picks up the scale factor, and transform_normalized
    // hands back that factor so a caller can rescale its own t range.
    let scale = Matrix::<V, 4, 4>::from_scale(vec3(1.0, 1.0, 2.0));
    let (scaled, length) = ray.transform_normalized(&scale);

    assert!(close(s(length), 2.0));
    assert!(close(s(scaled.direction.norm_sqr()), 1.0));

    // Error bounds are finite, non-negative, and tiny.
    let (_, error) = ray.transform_with_error(&m);

    assert!(error.pos.is_finite().all() && error.dir.is_finite().all());

    for i in 0..3 {
        assert!(s(error.pos[i]) >= 0.0 && s(error.pos[i]) < 1e-4);
    }
}

#[test]
fn ray_transform_degenerate_does_not_panic() {
    // raygon panics ("invalid ray transform") when a matrix collapses the
    // direction. Per-lane, that is not an option: stay finite and let the caller
    // see a zero-length direction.
    let ray = Ray::<V, 3>::new(pt3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let collapse = Matrix::<V, 4, 4>::from_scale(vec3(1.0, 1.0, 0.0));

    let (degenerate, length) = ray.transform_normalized(&collapse);

    assert!(close(s(length), 0.0));
    assert!(degenerate.direction.is_finite().all(), "direction went non-finite");
    assert!(GVector::from(degenerate.origin).is_finite().all());
}
