# Composite vector types: Dual and Compensated

This is the payoff of the trait-based design. A composite type wraps an inner
vector `V` and **implements the same `*Vector` + math traits** by delegating to `V`
and adding its own semantics on top. So a function written once over
`V: FloatVector + TranscendentalMath` runs unchanged on the composite, producing
derivatives (`Dual`) or extra precision (`Compensated`) for free.

Both crates are `publish = false` (pre-release) but functional. The complete
example in [SKILL.md](../SKILL.md) runs one generic function across both.

```toml
thermite-dual         = { git = "https://github.com/raygon-renderer/thermite" }
thermite-compensated  = { git = "https://github.com/raygon-renderer/thermite" }
```

## Why it works: the Element/Vector tower

The unit of specialization in Thermite is the **element type**, not the vector or
backend. A composite is both a valid element and a valid vector:

```
Dual<f64, N>            is an Element  (implements Element)
Dual<Vector<f64>, N>    is a Vector    (implements FloatVector, with Element = Dual<f64, N>)
```

So the invariant is:

```
for V: DualFloatVector.         Dual<V, N>: FloatVector   with Element = Dual<V::Element, N>
for V: CompensatedFloatVector.  Compensated<V>: FloatVector with Element = Compensated<V::Element>
```

Associated types resolve straight through: `Dual<V,N>::LANES == V::LANES`,
`::Mask == V::Mask`, `::Unsigned == V::Unsigned`, etc. That is what lets the
generic code typecheck and run.

---

## thermite-dual: forward-mode autodiff

`crates/thermite-dual/src/lib.rs`:

```rust
#[repr(C)]
pub struct Dual<V, const N: usize> {
    pub re: V,          // the primal (value)
    pub dual: [V; N],   // N first-order partials (a multidual)
}
```

`N` is the number of independent variables you are differentiating with respect to.
Each lane of the inner `V` carries its own value+derivatives, SIMD-parallel.

### Constructing

```rust
use thermite_dual::{Dual, AutoDiff};
type V = Vector<f64>;
type D = Dual<V, 2>;

D::constant(V::splat(3.0))       // value, all partials zero
D::new(re, [d0, d1])             // explicit primal and partials
D::variable(V::splat(4.0), 0)    // seed: partial 0 = 1, rest 0  (this IS variable x0)
D::ZERO   D::ONE
d.value() -> V                   // the primal
d.gradient() -> [V; N]           // the partials
```

### The `AutoDiff` trait: differentiate a generic function

`f.ad([inputs])` seeds each input as an independent variable, instantiates your
generic function at `W = Dual<V, N>`, and returns the resulting `Dual` carrying
value + full gradient. No turbofish, no manual seeding:

```rust
fn gaussian<W: FloatVector + TranscendentalMath>(x: W) -> W { (-(x * x)).exp() }

let r = gaussian.ad([V::splat(0.5)]);
let value = r.re.extract::<0>();      // exp(-0.25)
let dfdx  = r.dual[0].extract::<0>(); // -2*0.5*exp(-0.25)
```

Multi-argument and multivariate work the same way:

```rust
let f = |x: D, y: D| x * x + (x * y).sin() + y; // any generic/closure over the trait
let r = f.ad([V::splat(4.0), V::splat(5.0)]);
let (dfdx, dfdy) = (r.dual[0], r.dual[1]);
```

### How it differentiates: chain rule in the impls

`Dual<V,N>` implements `FloatVector` and `SpecializedTranscendentalMath` by computing
the primal with `V`'s method, then propagating partials with the local derivative
(the `chain` helper multiplies each partial by `f'(re)`):

```rust
fn sqrt(self) -> Self { let s = self.re.sqrt(); self.chain(s, V::HALF / s) }   // d/dx = 1/(2*sqrt x)
fn exp<P>(self)  -> Self { let v = self.re.exp_p::<P>(); self.chain(v, v) }    // d/dx e^x = e^x
fn sin_cos<P>(self) -> (Self, Self) {
    let (s, c) = self.re.sin_cos_p::<P>();
    (self.chain(s, c), self.chain(c, s.neg()))
}
```

This covers the full transcendental surface, FMA variants, masked `_c`/`_m`/`_z`
ops (they blend value AND derivative), `hypot`, `powf`, `nth_root`, inverse trig,
and even `inverse_smoothstep` (whose Dual override uses the implicit-function-theorem
derivative rather than differentiating through the internal Newton loop). With the
default `special` feature it also covers `thermite-special` functions.

### Optional `special` feature

`thermite-dual`'s `special` feature (on by default) pulls in `thermite-special` so
`Dual` also differentiates `erf`, `gelu`, etc.

---

## thermite-compensated: double-double precision

`crates/thermite-compensated/src/lib.rs`:

```rust
#[repr(C)]
pub struct Compensated<V> {
    pub value: V,   // high-order part
    pub error: V,   // low-order correction (the rounding residual)
}
```

`(value + error)` represents the number to roughly twice the working mantissa.
Arithmetic uses error-free transforms (two-sum, two-prod/Dekker) to track the bits
a single float would drop.

### Constructing and reading

```rust
use thermite_compensated::Compensated;
type V = Vector<f64>;

Compensated::<V>::new(v)          // error = 0
Compensated::<V>::splat_value(e)  // splat an element
c.value()         -> V            // normalized value+error folded to one V
c.uncompensated() -> V            // just the high part
c.error()         -> V            // just the low part
c.normalize()                     // renormalize the pair
```

### Generic code keeps the precision

```rust
fn cube<W: NumericVector>(x: W) -> W { x * x * x }
let c = cube(Compensated::<V>::new(V::splat(2.0)));  // value() == 8.0, error tracked
```

The benefit shows across **chains** of operations, not a single op you immediately
collapse with `value()`. Classic demo: summing `[1e16, 1.0, -1e16]`
left-to-right gives `0.0` in naive f64 (the `1.0` is lost) but `1.0` with
`Compensated`, because the two-sum captured the cancelled low bits.

### Status

Core arithmetic (`+ - *`, the operators), `value()`/`error()`, and
`SpecializedTranscendentalMath` (high-precision compensated series, e.g. a ~20-term
reduced-argument Taylor `sin_cos`) are implemented. But the trait impls are **not
fully filled in**: scattered `todo!()` remain across `NumericVector`
(`min`/`max`/`scale` and their masked forms, `pairwise_sum`, `relaxed_pairwise_sum`,
`arg_minmax`), `SignedVector` (`abs`/`copysign` masked forms), and `FloatVector`
(many `_c`/`_m`/`_z` variants) -- on the order of dozens. Treat as solid-but-WIP:
prefer plain arithmetic and the implemented transcendentals; expect a `todo!()`
panic if you hit an unfinished masked/reduction op.

---

## Nesting composites

Because a composite is itself a valid inner vector, they stack:

```rust
Dual<Compensated<Vector<f64>>, N>   // high-precision forward-mode autodiff
Compensated<Vector<f32>>            // double-single (~f64-ish precision from f32 lanes)
```

The same generic function runs on all of them. This is the strongest expression of
the "write once" thesis: one kernel, and the *type you instantiate it with* decides
whether you get plain SIMD, derivatives, extra precision, or both at once.

> A `Complex<V>` type also exists (`thermite-complex`), implementing the vector
> traits so generic `FloatVector` code runs on complex inputs -- but that crate is
> WIP and not covered in depth here.
