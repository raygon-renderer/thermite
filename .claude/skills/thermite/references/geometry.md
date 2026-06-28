# thermite-geometry

SoA (structure-of-arrays) geometric primitives built on the float-vector traits.
Each component of a point/vector is a separate SIMD vector, so a `Vector3<f32x8>`
holds the x/y/z of **8 points** at once.

```toml
thermite-geometry = { git = "https://github.com/raygon-renderer/thermite" }
```

```rust
use thermite::prelude::*;
use thermite_geometry::prim::{Vector3, Point3, Ray3, Bounds3};
```

## Core types (`src/prim/`)

```rust
pub struct Vector<V: FloatVector, const N: usize>(pub [V; N]);   // direction/displacement
pub struct Point<V: FloatVector, const N: usize>(pub [V; N]);    // position (affine)
pub struct Ray<V: FloatVector, const N: usize>   { pub origin, pub direction }
pub struct Bounds<V: FloatVector, const N: usize>(pub [[V; 2]; N]);          // [min,max] per axis (AABB)
pub struct Matrix<V: FloatVector, const C, const R>(pub [[V; R]; C]);   // column-major: C columns of R rows
```

Aliases: `Vector2/3/4`, `Point2/3/4`, `Ray3`, `Bounds3`, `Matrix2x2/3x3/4x4`.

## Operations

```rust
// Vector
Vector3::splat(v)   Vector3::new([x, y, z])   Vector3::basis::<I>()
a.min(b)  a.max(b)  a.clamp(lo, hi)  a.abs()  a.signum()
a.mul_adde(scalar, acc)  a.nmul_adde(scalar, acc)
a.cross(b)            // 2D: scalar perp-dot; 3D: Vector3 cross product

// Point (affine algebra)
Point3::splat(v)   Point3::new([x, y, z])
point + vector -> Point      point - point -> Vector      point - vector -> Point

// Bounds (AABB)
Bounds3::empty()   Bounds3::from_corners(min, max)   Bounds3::symmetric(half)
b.expand(amount)   b.intersection(other)   b.min_point()   b.max_point()   b.vertex(i)
b | other          b |= point             // union / grow to include
```

The `Matrix` type is column-major (`matrix[c][r]`); algorithms currently focus on 2D
(`src/algo/d2.rs`). For dense 3D/4D matrix and quaternion math, the core
`LinAlg3Vector`/`LinAlg4Vector` traits on a single SIMD vector (where the lanes are
the components) are often the better tool -- see [vector-api.md](vector-api.md)
section 9.

`thermite-sdf` is built on these types ([sdf.md](sdf.md)).
