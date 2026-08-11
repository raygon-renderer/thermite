thermite-compensated
====================

Double-double compensated arithmetic for
[Thermite](https://github.com/raygon-renderer/thermite).

`Compensated<V>` stores a value and an error term, together representing a
number to roughly twice the precision of `V` alone. Every operation is built
from error-free transformations (`two_sum`, `two_diff`, `two_prod`, Veltkamp
splitting) that recover the rounding error an ordinary float would discard, and
feed it back into the next operation.

```text
Compensated<f64>        => ~106-bit significand (a "double-double" scalar)
Compensated<Vector<R>>  => LANES independent double-doubles, SIMD-parallel
```

The inner type `V` is any Thermite `FloatVector`, or an `f32`/`f64` at the
element level.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_compensated::Compensated;

// Written once against trait bounds, so the same function that runs on a plain
// `Vector<R>` also runs at double-double precision. `#[dispatch]` is mandatory:
// without it the intrinsics never inline.
#[thermite::dispatch(V)]
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

let c = gaussian(Compensated::<V>::new(V::splat(0.5)));
let hi = c.value().extract::<0>();       // the rounded double result
let lo = c.error().extract::<0>();       // the bits below it
```

`Compensated<V>` implements the same `GenericVector -> FloatVector` stack as
`Vector<R>` itself, so generic code gains the extra precision without being
rewritten. `value()` folds the pair back into a single `V`. `uncompensated()`
and `error()` give the two halves separately.

Features
--------

| Feature | Default | Effect |
|---|---|---|
| `special` | on | Special functions (`thermite-special`) carried in compensated arithmetic: `erf`, `erfc`, `erfinv`, `probit`, `lambert_w`, and the activation family. |
| `std` | off | Forwards to `thermite/std`. The crate is `no_std` otherwise. |

### Incompatible with `thermite/algebraic-scalar`

Enforced by a `const` assertion at compile time, not left to the reader. Every
error-free transformation here depends on the compiler evaluating an expression
exactly as written. `algebraic-scalar` makes scalar-backend arithmetic
reassociable, at which point LLVM may fold `(a - (s - v)) + (b - v)` to zero and
every error term silently vanishes. Results stay plausible and lose all of the
extra precision this crate exists to provide, so the combination is refused.

Relationship to the other crates
--------------------------------

- **thermite** is the base. `Compensated` delegates every vector-trait method to
  its inner `V`, so it works on every backend and at every lane count.
- **thermite-special**, via the `special` feature. Provides the seeds that the
  compensated versions refine (for example `lambert_w` is seeded from the
  standard-precision result and refined with one compensated Halley step).
- **thermite-complex**, whose `compensated` feature makes
  `Complex<Compensated<V>>` valid: complex arithmetic in double-double.
- **thermite-dual** composes the other way: `Dual<Compensated<V>, N>`
  differentiates in double-double precision.

Status
------

Pre-release. Core arithmetic, the vector-trait surface, and the transcendental
library are complete. The Gamma family
(`tgamma`, `lgamma`, `lgamma_r`, `digamma`, `trigamma`, `beta`) is still
`todo!()` in `src/special.rs` and will panic if called. Those need genuine
double-double algorithms (a Lanczos or Stirling evaluation carried in
compensated arithmetic), not delegation to the inner `V`.

License
-------

MIT OR Apache-2.0.
