# Composite vector types: Dual and Compensated

A composite wraps an inner vector `V` and **implements the same `*Vector` + math
traits** by delegating to `V` and layering its own semantics. A function written
over `V: FloatVector + TranscendentalMath` runs unchanged on the composite,
producing derivatives (`Dual`) or extra precision (`Compensated`) for free.

## Why it works: the Element/Vector tower

The unit of specialization is the **element type**, not the vector or backend. A
composite is both a valid element and a valid vector:

```
Dual<f64, N>            is an Element
Dual<Vector<f64>, N>    is a Vector    (FloatVector with Element = Dual<f64, N>)

for V: DualFloatVector.         Dual<V, N>: FloatVector     with Element = Dual<V::Element, N>
for V: CompensatedFloatVector.  Compensated<V>: FloatVector with Element = Compensated<V::Element>
```

Associated types resolve straight through: `Dual<V,N>::LANES == V::LANES`,
`::Mask == V::Mask`, `::Unsigned == V::Unsigned`, etc. -- that's what lets
generic code typecheck.

---

## thermite-dual: forward-mode autodiff

`crates/thermite-dual/src/lib.rs`:

```rust
#[repr(C)]
pub struct Dual<V, const N: usize> {
    pub re: V,          // primal (value)
    pub dual: [V; N],   // N first-order partials (a multidual)
}
```

`N` = number of independent variables. Each lane of `V` carries its own
value+derivatives, SIMD-parallel.

```rust
use thermite_dual::{Dual, AutoDiff};
type V = Vector<f64>;
type D = Dual<V, 2>;

D::constant(V::splat(3.0))       // value, all partials zero
D::new(re, [d0, d1])             // explicit primal and partials
D::variable(V::splat(4.0), 0)    // seed: partial 0 = 1, rest 0 (this IS variable x0)
D::ZERO   D::ONE
d.value() -> V                   // primal
d.gradient() -> [V; N]           // partials
```

### `AutoDiff`: differentiate a generic function

`f.ad([inputs])` seeds each input as an independent variable, instantiates `f` at
`Dual<V, N>`, returns value + full gradient. No turbofish, no manual seeding:

```rust
fn gaussian<W: FloatVector + TranscendentalMath>(x: W) -> W { (-(x * x)).exp() }

let r = gaussian.ad([V::splat(0.5)]);
let value = r.re.extract::<0>();      // exp(-0.25)
let dfdx  = r.dual[0].extract::<0>(); // -2*0.5*exp(-0.25)

// Multi-argument / multivariate, closures too:
let f = |x: D, y: D| x * x + (x * y).sin() + y;
let r = f.ad([V::splat(4.0), V::splat(5.0)]);
let (dfdx, dfdy) = (r.dual[0], r.dual[1]);
```

### Mechanics: chain rule in the impls

Primal computed with `V`'s method; partials propagated by the local derivative
(`chain` multiplies each partial by `f'(re)`):

```rust
fn sqrt(self) -> Self { let s = self.re.sqrt(); self.chain(s, V::HALF / s) }   // 1/(2 sqrt x)
fn exp<P>(self)  -> Self { let v = self.re.exp_p::<P>(); self.chain(v, v) }
fn sin_cos<P>(self) -> (Self, Self) {
    let (s, c) = self.re.sin_cos_p::<P>();
    (self.chain(s, c), self.chain(c, s.neg()))
}
```

Covers the full transcendental surface, FMA variants, masked `_c`/`_m`/`_z` ops
(blend value AND derivative), `hypot`, `powf`, `nth_root`, inverse trig, and
`inverse_smoothstep` (Dual override uses the implicit-function-theorem
derivative, not differentiation through the internal Newton loop).

### `special` feature (default on)

Pulls in `thermite-special` so `Dual` also differentiates `erf`, `gelu`, the
gamma family (`tgamma`/`lgamma`/`lgamma_r`/`digamma`/`beta`), `expint`,
`lambert_w`, `erfinv` and `probit`. The gamma derivatives run on
`thermite-special`'s own `digamma`/`trigamma`.

One hole: `Dual::trigamma` is `todo!()` in `src/special.rs`, because its
derivative is the tetragamma `psi_2` and nothing provides that yet. Calling it on
a `Dual` panics; `trigamma` on a plain vector is fine.

---

## thermite-compensated: double-double precision

`crates/thermite-compensated/src/lib.rs`:

```rust
#[repr(C)]
pub struct Compensated<V> {
    pub value: V,   // high-order part
    pub error: V,   // low-order correction (rounding residual)
}
```

`(value + error)` represents the number to ~2x the working mantissa. Arithmetic
uses error-free transforms (two-sum, two-prod/Dekker).

```rust
use thermite_compensated::Compensated;
type V = Vector<f64>;

Compensated::<V>::new(v)          // error = 0
Compensated::<V>::splat_value(e)
c.value()         -> V            // value+error folded to one V
c.uncompensated() -> V            // high part only
c.error()         -> V            // low part only
c.normalize()                     // renormalize the pair

fn cube<W: NumericVector>(x: W) -> W { x * x * x }
let c = cube(Compensated::<V>::new(V::splat(2.0)));  // value() == 8.0, error tracked
```

The benefit shows across **chains** of ops, not a single op immediately collapsed
with `value()`. Classic demo: summing `[1e16, 1.0, -1e16]` left-to-right gives
`0.0` in naive f64 but `1.0` with `Compensated` (two-sum captured the cancelled
low bits).

### Status

Full vector-trait surface implemented (`NumericVector`/`SignedVector`/
`FloatVector` incl. every masked `_c`/`_m`/`_z` variant, `scale`,
`pairwise_sum`, `arg_minmax`, `mix`) -- covered by
`crates/thermite-compensated/tests/ops.rs` -- plus
`SpecializedTranscendentalMath` (high-precision compensated series, e.g. ~20-term
reduced-argument Taylor `sin_cos`). The special-function surface is complete too:
the gamma family (`lgamma_r`/`lgamma`/`tgamma`/`beta`), `digamma`, `trigamma`,
`erf`/`erfc`/`erfinv` and `probit` all carry genuine double-double algorithms,
covered by `tests/gamma.rs`, `tests/erfinv.rs` and `tests/erfc_tail.rs`. Nothing
in the crate panics.

Representation gotchas (intended semantics): `value()` folds `value + error`, so
it normalizes `-0.0 + 0.0` to `+0.0` -- read `uncompensated()` when the sign of
zero matters; `next_up`/`next_down` step the *error* term by one ulp, far below
what the folded `value()` resolves -- observe the step via a compensated
difference.

---

## Nesting composites

A composite is itself a valid inner vector, so they stack -- one kernel, and the
instantiating type decides plain SIMD, derivatives, extra precision, or both:

```rust
Dual<Compensated<Vector<f64>>, N>   // high-precision forward-mode autodiff
Compensated<Vector<f32>>            // double-single (~f64-ish precision from f32 lanes)
```

> `Complex<V>` also exists (`thermite-complex`), implementing the vector traits
> for complex inputs -- WIP, not covered in depth here.
