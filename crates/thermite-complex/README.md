thermite-complex
================

SIMD complex numbers for
[Thermite](https://github.com/raygon-renderer/thermite).

`Complex<V>` stores a real and an imaginary part, each an inner value `V`. With
`V` a Thermite `FloatVector`, each lane is an independent complex number
(struct-of-arrays). With `V` an `f32`/`f64` it is a complex scalar, which is the
`Element` of the vector form.

```text
Complex<f32>        => a complex scalar
Complex<Vector<R>>  => LANES complex numbers, SIMD-parallel
```

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_complex::Complex;

// Written once against trait bounds, then evaluated over C. `#[dispatch]` is
// mandatory: without it the intrinsics never inline.
#[thermite::dispatch(V)]
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

// e^(-i^2) = e^1 = e
let z = gaussian(Complex::<V>::I);
assert!((z.re.extract::<0>() - core::f64::consts::E).abs() < 1e-12);
assert!(z.im.extract::<0>().abs() < 1e-12);
```

`Complex<V>` implements the `GenericVector -> FloatVector` stack and the
`Specialized*Math` traits, so `CoreMath`, `TranscendentalMath` and `SpatialMath`
(with their `_p::<P>()` policy forms) come from the same blanket impls that serve
`Vector<R>`. Operations whose result or argument is _real_ (`norm`, `arg`, polar
form, real powers and bases) have no place in those families and get their own,
on `ComplexMath` and `ComplexVector`.

The inner `V` need not be a plain vector. Anything implementing `RealValue` will
do, including the other composites:

```text
Complex<Dual<V, N>>      => complex arithmetic carrying N derivatives  (`dual` feature)
Complex<Compensated<V>>  => complex arithmetic in double-double        (`compensated`)
```

### Ordering, sign and rounding

C is neither ordered nor signed, but the vector traits require both, so each one
needs an answer:

- Ordering (`cmp_lt` and friends, `min`, `max`, `clamp`, `arg_minmax`, the
  derived `PartialOrd`) is lexicographic by `(re, im)`. It is a tiebreak rule,
  not a statement about magnitudes.
- `abs` and `signum` are modulus-based, `|z|` as a real complex and `z/|z|`,
  preserving `abs(z) * signum(z) == z`. The spatial norms (`l1_norm`,
  `l2_norm`, `hypot`) are likewise the real quantities.
- The sign-bit ops (`copysign`, `mul_sign`, `signed_zero`) are componentwise.
  `is_negative` and `is_positive` report the sign of `re`, a mask having only
  one bit per lane.
- Rounding (`floor`, `ceil`, `round`, `trunc`, `fract`) is componentwise, and
  `%` is `z - trunc(z/w)*w` with that truncation. These satisfy the traits.
  They are not complex-analytic operations.
- `RealMath` is _not_ implemented. `atan2`, `wrap_angle`, `step`, `smoothstep`
  and the rest of that family are defined over an ordered field, so a
  `V: RealMath` bound will not accept a complex vector. For the argument of `z`,
  use `ComplexMath::arg`, which returns the real vector it is.

Features
--------

No features are on by default, and all are additive.

| Feature | Effect |
|---|---|
| `special` | Special functions over the complex plane (`thermite-special`): `erf`/`erfc`, the Faddeeva function `w(z)`, and the polynomial families. |
| `dual` | Lets a `Dual` be the storage: `Complex<Dual<V, N>>`, complex arithmetic that also carries derivatives. |
| `compensated` | Lets a `Compensated` be the storage: `Complex<Compensated<V>>`, complex arithmetic in double-double precision. |
| `std` | Forwards to `thermite/std`. The crate is `no_std` otherwise. |

Relationship to the other crates
--------------------------------

- **thermite** is the base. `Complex` delegates the vector traits to its inner
  `V`, so it works on every backend and at every lane count.
- **thermite-special**, via the `special` feature. The real special functions
  the complex extensions are built from.
- **thermite-dual**, via the `dual` feature. `Complex<Dual<V, N>>` composes the
  two in that order, so a complex-valued function returns complex derivatives.
- **thermite-compensated**, via the `compensated` feature, for
  `Complex<Compensated<V>>`.

Status
------

Pre-release. The complex vector surface, the transcendental library, the special
functions (Gamma family, Bessel and Airy at real order, zeta, polylog, Faddeeva)
are complete and tested, and `Dual` or `Compensated` can stand in as the storage.

License
-------

MIT OR Apache-2.0.
