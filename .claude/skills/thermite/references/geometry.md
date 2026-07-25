# thermite-geometry

Geometric primitives on the float-vector traits. Everything lives under the
`soa` module (structure-of-arrays): each component is a separate SIMD vector, so
`Vector3<f32x8>` holds x/y/z of **8 points** at once. (An `aos` counterpart --
one object per register, riding hardware `dot3`/`cross3`/`mat4_*` -- is planned,
not present.)

```rust
use thermite::prelude::*;
use thermite_geometry::soa::prim::{Vector3, Point3, Ray3, Bounds3, Matrix4x4, Quaternion, Transform, TangentFrame};
use thermite_geometry::soa::prim::vector::VectorOps as _;   // dot/normalize/... live on this trait
```

**One object per lane.** A `Bounds<f32x8, 3>` is eight independent boxes, not one
box of eight-wide coordinates. So every predicate returns a `V::Mask` (one answer
per lane), never a `bool`, and there is no branching on data anywhere in the
crate -- the AoS originals' `if`/`panic!` paths are all mask `select`s here.
Degenerate lanes stay finite rather than trapping: `try_normalize` gives zero,
`try_invert`/`try_new` hand back a validity mask, and a collapsed ray direction
reports length 0 instead of panicking.

## Core types (`src/soa/prim/`)

```rust
pub struct Vector<V: FloatVector, const N: usize>(pub [V; N]);   // direction/displacement
pub struct Point<V: FloatVector, const N: usize>(pub [V; N]);    // position (affine)
pub struct Ray<V: FloatVector, const N: usize>   { pub origin, pub direction }
pub struct Bounds<V: FloatVector, const N: usize>(pub [[V; 2]; N]);      // [min,max] per axis (AABB)
pub struct Matrix<V: FloatVector, const C, const R>(pub [[V; R]; C]);    // column-major: C columns of R rows
pub struct Quaternion<V: FloatVector>(pub [V; 4]);                       // [x, y, z, w], w scalar
pub struct Transform<V: FloatVector> { pub forward, pub inverse }        // paired 4x4s
pub struct TangentFrame<V: FloatVector> { pub normal, pub tangent, pub bitangent }
pub struct RayError<V: FloatVector, const N: usize> { pub pos, pub dir }  // conservative FP error bounds
```

Aliases: `Vector2/3/4`, `Point2/3/4`, `Ray2/3/4`, `Bounds2/3/4`, `Matrix2x2/3x3/4x4`.

## Operations

```rust
// Vector -- full arithmetic: + - * / (by Self or by scalar V), Neg, and every *Assign form
Vector3::splat(v)  Vector3::new([x,y,z])  Vector3::basis::<I>()  Vector3::ZERO/ONE
a.min(b)  a.max(b)  a.clamp(lo,hi)  a.abs()  a.signum()  a.mix(b, t)   // mix == lerp
a.cross(b)         // 2D: scalar perp-dot; 3D: Vector3.   2D also has a.perp()
a.mul_adde(scalar, acc)  a.nmul_adde(scalar, acc)
a.is_finite() -> V::Mask   a.is_nan() -> V::Mask
// VectorOps trait (import it): each also has a `_p::<Policy>` form
a.dot(&b)  a.norm_sqr()  a.l2_norm()  a.l1_norm()  a.linf_norm()
a.normalize()  a.normalize_norm() -> (Self, V)  a.try_normalize()  a.reciprocal()

// Point (affine algebra)
point + vector -> Point    point - point -> Vector    point - vector -> Point   (+ AddAssign/SubAssign)
p.min(q)  p.max(q)  p.mix(q, t)  p.midpoint(q)  p.distance(q)  p.distance_sqr(q)

// Bounds (AABB)
Bounds3::EMPTY      // min=+inf, max=-inf: the identity for UNION -- start folds here
Bounds3::UNIVERSE   // min=-inf, max=+inf: the identity for INTERSECTION
Bounds3::from_corners(min, max)  Bounds3::from_point(p)  Bounds3::symmetric(half)
b | other   b |= point   b.union(o)  b.union_point(p)      // union
b & other   b.intersection(o)                              // intersection (may invert -> is_empty)
points.into_iter().collect::<Bounds3<V>>()                 // FromIterator<Point> and <Bounds>
b.contains(p) / b.contains_bounds(&o) / b.overlaps(&o) / b.is_empty()   // -> V::Mask
b.diagonal()  b.centroid()  b.volume()  b.surface_area()   // surface_area = the SAH cost metric
b.max_extent_axis() -> V::Unsigned      // per-lane split axis for a BVH
b.offset(p)          // normalized [0,1] coords -- what a Morton/Hilbert code is built from
b.closest_point(p)  b.distance_sqr(p)  b.bounding_sphere() -> (Point, V)
b.expand(amount)  b.expand_axes(v)  b.min_point()  b.max_point()  b.vertex(i)

// Matrix -- column-major (matrix[c][r])
Matrix::<V,N,N>::IDENTITY   m.transpose()  m.trace()  m.from_diagonal(d)  m.is_identity(tol)
m.determinant()   m.invert()   m.try_invert() -> (Self, V::Mask)
    // ^ one generic method, specialized by a const-N ladder: closed forms at 2x2 (6 ops),
    //   3x3 (three cross products), 4x4 (branch-free Laplace adjugate); Gauss-Jordan with
    //   lane-wise partial pivoting beyond. Singular lanes -> inf/NaN; try_invert reports them.
// 4x4 only:
m.transform_point(p)     // w = 1: translation APPLIES        (also `m * point`)
m.transform_vector(v)    // w = 0: translation does NOT apply
m.project_point(p)       // full projective, with the w-divide
m.transform_bounds(&b)   // union of the 8 transformed corners (also `m * bounds`)
m.linear() -> Matrix3x3  m.translation() -> Vector3
Matrix4x4::from_translation/from_scale/from_basis/from_axis_angle/perspective/orthographic
// conservative float-error bounds (PBRT ch. 3.9), for watertight self-intersection offsets:
gamma::<V>(n)   m.transform_point_with_error(p) -> (Point, Vector)   m.transform_vector_with_error(v)
m.transform_point_propagate_error(p, e)

// Quaternion
Quaternion::IDENTITY  ::new(x,y,z,w)  ::from_axis_angle(axis_angle)  ::from_axis_angle_raw(axis, angle)
::from_euler(yaw,pitch,roll)  ::from_rotation_arc(a, b)  ::look_at(origin, target)  ::from_matrix(&m)
q.to_matrix()  q.rotate_vector(v)  q.transform_point(p)  q.slerp(other, t)   // slerp takes the shortest arc
q.dot/norm/norm_sqr/normalize/try_normalize/conjugate/inverse/is_finite;  q * q (Hamilton), + - Neg, * V

// Transform -- keeps forward AND inverse so the hot path never inverts
Transform::IDENTITY  ::from_raw(f, i)  ::new(f)  ::try_new(f) -> (Self, V::Mask)  ::new_inverse(i)
::translate(v)  ::scale(v)  ::rotate(axis_angle)  ::rotate_raw(axis, angle)  ::from_quaternion(q)
::look_at(origin, target, up)  ::look_at_dir(origin, dir, up)  ::ortho(near, far)  ::perspective(fov, n, f)
t.invert()   // free: just swaps the pair
t.transform_point/transform_vector/transform_bounds/transform_axis/is_identity(tol)
t.transform_normal(n)   // inverse-transpose -- a normal does NOT transform like a direction
t.reverse_transform_normal(n)
t1 * t2   // composition; the inverse composes in reverse order
// The constructors with a known closed-form inverse (translate/scale/rotate/ortho/perspective/
// look_at/from_quaternion) supply it directly and never call invert().

// TangentFrame -- branchless Frisvad/Duff (JCGT 2017) orthonormal basis
TangentFrame::new(unit_normal)          // arbitrary tangent; stable at both poles
TangentFrame::partial(normal, tangent)  // uses a mesh tangent, falls back where it is degenerate
f.to_world(local)  f.to_local(world)    // shading space has +Z = normal
f.to_transform()                        // orthonormal, so the inverse is the transpose

// Ray
Ray::new(origin, direction)   r.at(t)   r.move_to(t)   r.inv_direction()   -r
r.intersects_aabb(&aabb, &inv_dir) -> (t_min, t_max)                 // slab test
r.intersects_aabb_mask(&aabb, &inv_dir) -> (t_min, t_max, V::Mask)   // ... plus the hit mask
// t_max carries a 4-ulp widening (jcgt.org/published/0002/02/02) so thin boxes are not missed.
r.transform(&m)                       // plain map; PRESERVES the t parameterization
r.transform_normalized(&m) -> (Self, V)   // renormalizes, returns |M d| so you can rescale your own t
r.transform_with_error(&m) / r.transform_propagate_error(&m, e) -> (Self, RayError)
// AoS <-> SoA: Vector/Point/Ray have load_interleaved/store_interleaved (NEON LD3/ST3).
```

Algorithms live in `src/soa/algo/` (currently 2D only: `point_on_ellipse`).

For single-object 3D/4D matrix and quaternion math (one object per *register*),
the core `LinAlg3Vector`/`LinAlg4Vector` traits are often the better tool -- see
[vector-api.md](vector-api.md) section 11. They are also what a future `aos`
module would be built on.

`thermite-sdf` is built on these types ([sdf.md](sdf.md)).
