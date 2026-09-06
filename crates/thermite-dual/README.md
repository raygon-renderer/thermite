thermite-dual
=============

Forward-mode automatic differentiation for
[Thermite](https://github.com/raygon-renderer/thermite), built on multidual
numbers.

`Dual<V, N>` carries a primal value plus `N` first-order derivative components.
Arithmetic propagates derivatives by the usual chain rule, so evaluating a
function on a `Dual` returns both the value and its gradient in a single pass.

```text
Dual<V, 0>  =>  a value, no derivatives tracked
Dual<V, 1>  =>  value + one derivative direction (a classic dual number)
Dual<V, N>  =>  value + N partials (the gradient of an N-variable function)
```

The inner type `V` is any Thermite `FloatVector` (each lane is then an
independent dual number, SIMD-parallel), or an `f32`/`f64` at the element level.
The derivative components live in a separate `[V; N]`, so the layout is
struct-of-arrays.

This is a _first-order multidual_. It tracks gradients, not Hessians. Some
libraries call this a hyperdual, though that name properly denotes the
second-order algebra.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_dual::AutoDiff;

// Written once against trait bounds, with no ISA, lane count, or element type
// named. `#[dispatch]` is mandatory: without it the intrinsics never inline.
#[thermite::dispatch(V)]
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

let r = gaussian.ad([V::splat(0.5)]);
let value = r.re.extract::<0>();      //  0.7788007830714049
let dydx = r.dual[0].extract::<0>();  // -0.7788007830714049  (= -2x e^{-x^2})
```

`Dual<V, N>` implements the same `GenericVector -> FloatVector` stack as
`Vector<R>` itself, which is why an unmodified generic function differentiates:
the type you instantiate decides whether you get a plain value or a value and a
gradient.

Features
--------

| Feature | Default | Effect |
|---|---|---|
| `special` | on | Special functions (`thermite-special`) differentiated by the chain rule: `erf`, `tgamma`, `lgamma`, `digamma`, `beta`, `lambert_w`, and everything that composes out of dual arithmetic. |
| `std` | off | Forwards to `thermite/std`. The crate is `no_std` otherwise. |

Relationship to the other crates
--------------------------------

- **thermite** is the base. `Dual` delegates every vector-trait method to its
  inner `V`, so it works on every backend and at every lane count.
- **thermite-special**, via the `special` feature. `Dual` implements
  `SpecializedSpecialMath`/`SpecializedRealSpecialMath` on top of it.
- **thermite-complex**, whose `dual` feature makes `Complex<Dual<V, N>>` valid:
  complex arithmetic that also carries derivatives.
- **thermite-compensated** composes the other way: `Dual<Compensated<V>, N>`
  differentiates in double-double precision.

Status
------

Pre-release. Core autodiff is complete and tested, and the special functions in
`thermite-special` differentiate too, the iterative ones through closed-form
derivative rules rather than through their loops.

License
-------

MIT OR Apache-2.0.
