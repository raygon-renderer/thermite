# thermite-sdf

Signed distance fields: distance primitives + combinators + domain transforms,
generic over a float vector (many points in parallel, any backend). Builds on
`thermite-geometry`.

Authoritative usage reference: the worked examples
`crates/thermite-sdf/examples/raymarch.rs` (sphere-tracing a scene to PNG) and
`crates/thermite-sdf/examples/voronoi.rs` -- read them for a real SDF pipeline.

## The traits

`src/lib.rs`. The vector bound is `SdfVector = thermite::math::SpatialMath`.

```rust
trait SDF<V: SdfVector, const N: usize> {
    fn eval(&self, p: Vector<V, N>) -> V;                 // signed distance at point p
}
trait GradientSdf<V, const N>: SDF<V, N> {
    fn eval_grad(&self, p: Vector<V, N>) -> (V, Vector<V, N>);  // distance + analytic unit gradient
    fn normal(&self, p: Vector<V, N>) -> Vector<V, N>;          // defaults to eval_grad().1
}
trait BoundedSdf<V, const N>: SDF<V, N> {
    fn aabb(&self) -> Bounds<V, N>;                       // axis-aligned bounding box
}
```

`Vector<V, N>` / `Bounds<V, N>` are the SoA geometry types from `thermite-geometry`.
`N` is the spatial dimension (2 or 3 for the named primitives; `dn` works at any N).

## What's in it

- **2D primitives** (`d2`): Circle, Box, RoundedBox, OrientedBox, Segment, Triangle,
  Pentagon/Hexagon/Octagon, Star, Heart, Moon, Vesica, Cross, Egg, Bezier, ... (~40).
- **3D primitives** (`d3`): Sphere, Box, BoxFrame, Torus, Cylinder, Capsule, Cone,
  Pyramid, Octahedron, Ellipsoid, Link, CappedTorus, CutSphere, ... (~35).
- **N-D primitives** (`dn`): `NSphere`, `NBox`, `NPlane`, `NCapsule`, `NEllipsoid`,
  `CrossPolytope` -- generic over the dimension.
- **Boolean ops** (`ops`): `Union`, `Intersection`, `Subtraction`, `Xor`, and smooth
  variants `SmoothUnion`/`SmoothIntersection`/`SmoothSubtraction` with selectable
  blend kernels (`Quadratic` default, `Cubic`, `Quartic`, `Circular`, `Root`).
- **Domain ops** (`ops`): `Round`, `Onion`, `Scale`, `Elongate`, `Symmetry`,
  `Repetition`/`CorrectRepetition`/`LimitedRepetition`/`MirroredRepetition`,
  `Extrusion`, `Revolution`, `Twist`, `Bend`.
- **Transforms** (`transform`): `Translate`, `UniformScale`, `Rotate`.
- **Estimation**: `FiniteDiff` (numeric gradient of any SDF), `DistanceEstimate`,
  `Field` (wrap an arbitrary function as an SDF proxy).
- **Fractals** (`fractal`): `Mandelbulb3D`, `QuaternionJulia3D`, `Julia2D`,
  `Mandelbrot2D`, `MengerSponge`, `SierpinskiCarpet`.
- **Procedural** (`fbm`, `voronoi`, `displace`): FBM noise, Voronoi cells/edges,
  `Displacement`, `DomainWarp`.
- **L-infinity (Chebyshev metric) variants** (`d2_linf`): `LinfSdf` family.

## Pattern

SDFs are plain structs implementing the traits; you compose them by nesting:

```rust
use thermite_sdf::{d3::Sphere3D, ops::{SmoothUnion, Round}, GradientSdf};

let s   = Sphere3D { radius: r };
let big = Round { shape: s, radius: edge };            // shell/onion
let scene = SmoothUnion { a: big, b: other, k: blend };
let (dist, normal) = scene.eval_grad(point);           // evaluate many points (SIMD lanes) at once
```

For exact constructor field names and a full scene, read
`crates/thermite-sdf/examples/raymarch.rs` -- it is the ground truth.
