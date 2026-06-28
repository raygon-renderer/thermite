# Generic programming over the `*Vector` traits

This is the core idea of Thermite. Read it before anything else.

You write a function **once**, parameterized by a trait bound from the
`GenericVector` hierarchy. You never name a concrete backend, a lane count, or an
element type in the signature. The **caller** chooses the concrete type, and the
compiler monomorphizes optimal code for it. Because composite types (`Dual`,
`Compensated`, `Complex`, ...) also implement these traits, the *same* function
runs on them too -- giving you automatic differentiation, double-double
precision, or complex arithmetic for free, with no rewrite.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;

// BAD: locked to one backend and one width. Works on exactly one machine.
fn bad(a: f32x8, b: f32x8) -> f32x8 { a * b }

// GOOD: any backend, any lane count, any float element -- and Dual/Compensated.
fn good<V: FloatVector>(a: V, b: V) -> V { a * b }

// Add transcendental math by adding a bound:
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

// Any integer vector:
fn pack<V: IntegerVector>(a: V, b: V, shift: u32) -> V { a | (b << shift) }
```

## How to pick your bound

Use the *weakest* trait that supplies the operations you call. The hierarchy
(see [trait-hierarchy.md](trait-hierarchy.md)) is, abbreviated:

```
GenericVector            load/store, splat, lanes, gather, interleave, map/fold, cast
  |- BitwiseVector       & | ^ ! andnot ternlog
  |    |- BitshiftVector  << >> rotates
  |- PartialOrdVector    cmp_lt/le/gt/ge/eq/ne  -> Mask
       |- NumericVector  + - * / min max clamp sum product FMA  (+ constants ZERO/ONE/...)
            |- SignedVector       abs signum copysign neg
                 |- IntegerVector         saturating/wrapping, count_ones, dividers
                 |- FloatVector           sqrt rcp rsqrt floor/ceil/round mix  (+ HALF/NAN/INF/EPSILON)
                      |- FloatVectorWithBits   ldexp frexp, native transcendentals, bit views
```

Math traits (each requires at least `FloatVector`): `CoreMath`,
`TranscendentalMath`, `SpatialMath`, `RealMath`, `FloatMath`. They are declared in
`thermite::math`. See [math.md](math.md).

Recipes:

| You want to... | Bound |
|---|---|
| add/multiply, min/max, FMA | `V: NumericVector` |
| negate, abs, copysign | `V: SignedVector` |
| sqrt, floor, reciprocal, float constants | `V: FloatVector` |
| sin/cos/exp/ln/pow/atan2 | `V: FloatVector + TranscendentalMath` (or `+ RealMath`) |
| hypot, norms | `V: FloatVector + SpatialMath` |
| ldexp/frexp, bit-level float tricks | `V: FloatVectorWithBits` |
| bit ops / shifts | `V: BitwiseVector` / `V: BitshiftVector` |
| integer saturating/wrapping/divide | `V: IntegerVector` (or `Signed`/`Unsigned` sub-traits) |
| erf, gamma, gelu, elliptic, ... | `V: SpecialMath` (from `thermite-special`) |

`RealMath` aggregates `TranscendentalMath + SpatialMath` plus angle/interp helpers;
reach for it when you use several families.

## Associated types: name things without naming a backend

Every `V: GenericVector` exposes these. Use them instead of hard-coding types:

```rust
V::Element        // scalar element type (f32, i32, ...)
V::LANES          // lane count, a usize const
V::Lanes          // lane count as a typenum type
V::Unsigned       // unsigned int vector, same lane count & bit width
V::Signed         // signed int vector, same lane count & bit width
V::Mask           // the boolean mask type for this vector
```

```rust
fn sum_lanes<V: NumericVector>(v: V) -> V::Element { v.sum_elements() }
fn abs_bits<V: FloatVectorWithBits>(v: V) -> V::Bits { v.abs().into_bits() }
fn pick<V: GenericVector>(m: V::Mask, a: V, b: V) -> V { m.select(a, b) }
```

To attach a per-element constant generically, bound the element:
`fn f<V: FloatVector<Element: MyConsts>>()` and read
`<V::Element as MyConsts>::FOO` (pattern used throughout the math kernels --
see [performance.md](performance.md) section 5).

## Running the generic function on concrete types

A generic `fn f<V: FloatVector + TranscendentalMath>(x: V) -> V` can be called with:

### 1. Scalar (1-lane) backends -- no dispatch needed

`Vector<f32>` and `Vector<f64>` are real vector types with `LANES == 1`, backed by
the scalar register impls. They need no `target_feature` and are perfect for tests
and for seeding composite types.

```rust
let y = f(Vector::<f64>::splat(0.5));
let s = y.extract::<0>(); // pull lane 0 out as f64
```

### 2. Native-width SIMD -- via runtime dispatch

`dispatch_dyn!` detects the best ISA at runtime (cached), establishes the
`target_feature` context, and rewrites bare width names (`f32xN`, `f32x8`,
`i32x4`, `f64xN`, ...) inside the block to `Vector<S::...>` for the chosen backend.
`f32xN` is the widest native f32 width on that backend.

```rust
let s = thermite::dispatch_dyn!(for<S> || -> f32 {
    f(f32xN::splat(0.5)).extract::<0>()
});
```

To give a reusable library kernel correct per-ISA codegen, annotate it with the
`#[thermite::dispatch(Self)]` / `#[thermite::dispatch(S)]` **attribute** (it is a
proc-macro attribute, not a `dispatch!(...)` bang macro) and make it generic over `S: Simd`. Both tools, and how they compose, are
in [slices-and-dispatch.md](slices-and-dispatch.md).

### 3. `Dual<V, N>` -- the same function becomes autodiff

`thermite-dual` provides `Dual<V, N>` (a primal plus `N` first-order partials) that
implements `FloatVector + TranscendentalMath + ...` by delegating to `V` and
applying the chain rule. So `f` differentiates itself:

```rust
use thermite_dual::AutoDiff;
type V = Vector<f64>;

// `.ad([...])` seeds each input as an independent variable, instantiates f at
// W = Dual<V, N>, and returns value + gradient. No turbofish, no closure wrapper.
let r = f.ad([V::splat(0.5)]);
let value = r.re.extract::<0>();      // f(0.5)
let dfdx  = r.dual[0].extract::<0>(); // f'(0.5)
```

### 4. `Compensated<V>` -- the same function in double-double

`thermite-compensated` provides `Compensated<V>` (a hi/lo pair, ~2x mantissa bits)
that implements the vector traits via error-free transforms. Generic code keeps
the extra precision through chains of operations:

```rust
use thermite_compensated::Compensated;
let c = f(Compensated::<Vector<f64>>::new(Vector::splat(0.5)));
let y = c.value().extract::<0>(); // hi+lo folded back to one f64
```

All four of the above appear together in the complete example in
[SKILL.md](../SKILL.md). See [composite-types.md](composite-types.md) for how the
delegation works and how to nest them (`Dual<Compensated<V>, N>` = high-precision
autodiff).

## The prelude, and naming math traits in bounds

```rust
use thermite::prelude::*;
```

brings in `Vector`, `Mask`, the whole vector-trait hierarchy, `SimdSlice`,
operator/masked-op traits, `FloatConsts`, `Policy`, the `Simd`/`FloatSimd`
families, dividers, and `Element`/`FloatElement`.

**Caveat:** the prelude imports the *math* traits anonymously (`CoreMath as _`,
`TranscendentalMath as _`, ...). Their methods are callable, but the names are not
in scope, so a bound like `<V: TranscendentalMath>` will not compile until you add:

```rust
use thermite::math::{TranscendentalMath, RealMath, CoreMath, SpatialMath, FloatMath};
```

## Real-world idiom: bundle traits into a domain trait

When a function needs several Thermite traits and you call it in many places, don't
repeat a long `where` clause everywhere. Define one **domain trait** that bundles
the bounds (often with associated-type bounds), and a blanket impl. This is the
dominant pattern in production code built on Thermite:

```rust
// Bundle the bounds your spectral/NN code needs into one name:
pub trait NnVector: SwizzleVector + RealMath + SpecialMath<Element = f32> {}
impl<V> NnVector for V where V: SwizzleVector + RealMath + SpecialMath<Element = f32> {}

fn activation<V: NnVector>(x: V) -> V { /* uses RealMath + SpecialMath methods */ }

// Bundle at the *backend* (Simd) level too, with per-type associated bounds:
pub trait NnBackend: SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x16: NnVector> {}
impl<S> NnBackend for S where S: SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x16: NnVector> {}
```

You can pin a width as part of the bound to compile-assert it, e.g.
`fn f<V: NnVector<Lanes = thermite::generic_array::typenum::U8>>(...)`. And remember
the math-trait names must be imported to *name* them in these bounds (see above).

A couple more idioms that show up constantly:

- **Table lookup / interpolation = gather + mix**, not a scalar loop. Compute an
  index vector `i` and fraction `t`, gather both neighbors, blend:
  `let (a, b) = (V::gather(table, i), V::gather(table, i + 1)); t.mix(a, b)`.
- **Stay in-register with `swizzle!`** to avoid extract/insert in hot loops:
  `let p = thermite::swizzle!(f, [1, 0, 1, 1]);` then combine with FMAs.
- **`if const { V::ISA.is_simd() }`** (or any `HasIsa`/capability const) to fork the
  scalar vs SIMD path at compile time with no runtime branch.

## Pitfalls

- **Bare `f32`/`f64` are not `FloatVector`.** `fn f<V: FloatVector>(...)` cannot take
  a raw `f64`. Wrap it: `Vector::<f64>::splat(x)`. For one-off scalar math without a
  vector, use `ScalarMath`: `use thermite::math::ScalarMath; let s = x.scalar_sin();`
  (methods are `scalar_`-prefixed to avoid clashing with inherent `f64::sin`). See
  [math.md](math.md).
- **Don't over-constrain.** Bounding on `FloatVector` when you only `+`/`*` needlessly
  excludes integer and some composite instantiations. Use `NumericVector`.
- **Don't name a backend type** (`f32x8`, `__m256`, `S::f32x8`) in a reusable
  function signature -- that defeats the entire design. Stay on the trait.
- **`V::LANES` is backend-dependent.** Never assume 4 or 8. Use slice iterators
  ([slices-and-dispatch.md](slices-and-dispatch.md)) and `V::indexed()` rather than
  hardcoding lane indices, except via the compile-time `extract::<I>()`/`insert::<I>()`.
- **Element constants in generic code**: you cannot write `V::splat(3.0)` if `3.0`
  isn't obviously the element type. Use the provided constants (`V::ONE`, `V::TWO`,
  `V::HALF`, `FloatConsts`), or `V::splat(V::Element::...)`, or the `ConstRatio`
  pattern in [performance.md](performance.md).
