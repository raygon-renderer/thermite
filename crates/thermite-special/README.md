thermite-special
================

Special functions for [Thermite](https://github.com/raygon-renderer/thermite),
written once against the vector traits and compiled for every backend, lane
count and float type the core library supports.

Thermite's own math library covers the transcendentals you reach for constantly,
`exp`, `log`, `sin_cos`, `powf` and so on. This crate is the layer above that:
the error function, the gamma family, orthogonal polynomials, elliptic
integrals, and the activation functions, all vectorized rather than looped over
lanes.

## What's in it

* **Error function family.** `erf`, `erfc`, and the inverses `erfinv` and
  `probit`.
* **Gamma family.** `tgamma`, `lgamma`, `lgamma_r` (log-gamma with the sign of
  the gamma), `digamma`, and `beta`.
* **Orthogonal polynomials.** Legendre (including the associated form), Jacobi,
  Hermite at a const or a per-lane runtime order, and a Chebyshev series
  evaluated by Clenshaw recurrence for all four kinds.
* **Elliptic integrals.** Every Legendre form, complete and incomplete, over all
  five Carlson symmetric primitives (`R_F`, `R_C`, `R_D`, `R_J`, `R_G`). The
  form is selected by a request struct, so `EllintPiInc { n, phi, k }` and
  `EllintK { k }` carry exactly their own arguments and the wrong shape is a
  compile error rather than a silently ignored parameter.
* **Lambert W.** Both real branches, `W_0` and `W_{-1}`, from a single call. The
  two Halley iterations interleave, so the second branch is close to free on a
  wide machine.
* **The exponential integral** `E_n(x)` at integer order.
* **Activations,** each with a matching `_d` variant returning the value and its
  derivative together: `gelu`, `swish`, `softplus`, `logistic_sigmoid`, and the
  exp-free `algebraic_sigmoid` and `algebraic_swish`.

## Using it

```rust
use thermite::prelude::*;
use thermite_special::SpecialMath;

// Vector<f64> is the 1-lane scalar seed. Swap it for f64xN inside a
// `#[thermite::dispatch]` function and the same call is a wide kernel.
let x = Vector::<f64>::splat(0.5);

let e = x.erf();
let g = x.tgamma();
```

The traits are auto-implemented for every float vector, so there is nothing to
wire up. Bring `SpecialMath` (or `RealSpecialMath`, `RealPrimalMath`) into scope
and the methods appear. For a bare `f32` or `f64`, `ScalarSpecialMath` provides
the same set under `scalar_`-prefixed names.

## Precision is a type parameter

Every function has a `_p` form that takes a leading `P: Policy`, which is how
Thermite exposes the accuracy against speed tradeoff without a second set of
function names.

```rust,ignore
use thermite::math::policy::policies::{Precision, UltraPerformance};

let fast = x.erf_p::<UltraPerformance>();
let good = x.erf_p::<Precision>();
```

The presets run `UltraPerformance`, `HighPerformance`, `Performance` (the
default), `Precision`, `Size` and `Reference`. Policies compose, so a kernel can
run its initial iterations loose and its final one tight, which is exactly what
the Lambert W and elliptic implementations do internally.

`erf` is worth calling out. Its approximation doesn't lean on the accuracy of
`exp`, so on f32 it stays reasonable even at the low presets, and the cheap
policies are genuinely cheap.

## It composes

The functions are written against `FloatVector`, not a concrete type, so they
also run on Thermite's composite float types with no extra code. `Dual` gives
the derivative alongside the value, `Compensated` evaluates in double-double
precision, and `Complex` extends the ones that are holomorphic to the complex
plane.

## License

MIT or Apache-2.0, at your option.
