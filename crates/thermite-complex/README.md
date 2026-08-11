thermite-complex
================

SIMD complex numbers for [Thermite](../thermite).

`Complex<V>` stores a real and an imaginary part, each an inner value `V`. With
`V` a Thermite `FloatVector`, each lane is an independent complex number
(struct-of-arrays); with `V` an `f32`/`f64` it is a complex scalar, which is the
`Element` of the vector form.

```text
Complex<f32>        => a complex scalar
Complex<Vector<R>>  => LANES complex numbers, SIMD-parallel
```

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_complex::Complex;

// Written once against trait bounds, then evaluated over C.
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

// e^(-i^2) = e^1 = e
let z = gaussian(Complex::<V>::I);
assert!((z.re.extract::<0>() - core::f64::consts::E).abs() < 1e-12);
```

`Complex<V>` implements the `GenericVector -> FloatVector` stack, so `CoreMath`,
`TranscendentalMath` and `SpatialMath` (with their `_p::<P>()` policy forms) come
from the same blanket impls that serve `Vector<R>`. Operations whose result or
argument is *real* - `norm`, `arg`, polar form, real powers and bases - have no
place in those families and live on `ComplexMath`/`ComplexVector` instead.

### Ordering, sign and rounding

C is neither ordered nor signed, but the vector traits require both. The
resolutions are documented in full in the crate docs; in brief:

- Ordering (`cmp_lt`, `min`/`max`, `arg_minmax`, the derived `PartialOrd`) is
  lexicographic by `(re, im)`. It is a tiebreak rule, not a claim about
  magnitudes.
- `abs`/`signum` are modulus-based, preserving `abs(z) * signum(z) == z`.
- Sign-bit ops and rounding are componentwise.
- `RealMath` is deliberately *not* implemented: `atan2`, `wrap_angle`, `step`
  and friends are defined over an ordered field. For the argument of `z`, use
  `ComplexMath::arg`, which returns the real vector it is.

Features
--------

No features are on by default; every one of them is additive.

| Feature | Effect |
|---|---|
| `special` | Special functions over the complex plane (`thermite-special`): `erf`/`erfc`, the Faddeeva function `w(z)`, and the polynomial families. |
| `dual` | Lets a `Dual` be the storage: `Complex<Dual<V, N>>`, complex arithmetic that also carries derivatives. |
| `compensated` | Lets a `Compensated` be the storage: `Complex<Compensated<V>>`, complex arithmetic in double-double precision. |
| `std` | Forwards to `thermite/std`. The crate is `no_std` otherwise. |

Relationship to the other crates
--------------------------------

- **thermite** - the base. `Complex` delegates the vector traits to its inner
  `V`, so it works on every backend and at every lane count.
- **thermite-special** - the `special` feature; the real special functions the
  complex extensions are built from.
- **thermite-dual** - the `dual` feature. `Complex<Dual<V, N>>` composes the two
  in that order, so a complex-valued function returns complex derivatives.
- **thermite-compensated** - the `compensated` feature, for
  `Complex<Compensated<V>>`.

Status
------

Pre-release (`publish = false`). The complex vector surface, the transcendental
library and the Faddeeva implementation are complete and tested. The
`Complex<Compensated<..>>` path is complete for the element-agnostic functions
but still `todo!()`s the Gamma family, `lambert_w` and Faddeeva - those wait on
the corresponding real double-double implementations in `thermite-compensated`.

License
-------

MIT OR Apache-2.0.
