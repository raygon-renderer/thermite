# Generic programming over the `*Vector` traits

The core idea: write a function **once**, parameterized by a trait bound from the
`GenericVector` hierarchy. Never name a backend, lane count, or element type in
the signature -- the caller picks the concrete type and the compiler
monomorphizes optimal code. Composite types (`Dual`, `Compensated`, `Complex`)
implement the same traits, so the same function gains autodiff / double-double /
complex arithmetic with no rewrite.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;

// BAD: locked to one backend and one width.
fn bad(a: f32x8, b: f32x8) -> f32x8 { a * b }

// GOOD: any backend, lane count, float element -- and Dual/Compensated.
fn good<V: FloatVector>(a: V, b: V) -> V { a * b }

// Add transcendental math by adding a bound:
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

// Any integer vector:
fn pack<V: IntegerVector>(a: V, b: V, shift: u32) -> V { a | (b << shift) }
```

## Picking the bound

Use the *weakest* trait that supplies what you call
([trait-hierarchy.md](trait-hierarchy.md)); abbreviated:

```
GenericVector            load/store, splat, lanes, gather, interleave, map/fold, cast
  |- BitwiseVector       & | ^ ! andnot ternlog
  |    |- BitshiftVector  << >> rotates
  |- PartialOrdVector    cmp_lt/le/gt/ge/eq/ne  -> Mask
       |- NumericVector  + - * / min max clamp sum product FMA  (+ ZERO/ONE/...)
            |- SignedVector       abs signum copysign neg
                 |- IntegerVector         saturating/wrapping, count_ones, dividers
                 |- FloatVector           sqrt rcp rsqrt floor/ceil/round mix  (+ HALF/NAN/INF/EPSILON)
                      |- FloatVectorWithBits   ldexp frexp, native transcendentals, bit views
```

Math traits (each requires at least `FloatVector`), declared in `thermite::math`:
`CoreMath`, `TranscendentalMath`, `SpatialMath`, `RealMath`, `FloatMath`. See
[math.md](math.md). `RealMath` aggregates `TranscendentalMath + SpatialMath` plus
angle/interp helpers -- use it when you need several families.

| You want... | Bound |
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

## Associated types: name things without naming a backend

```rust
V::Element        // scalar element type (f32, i32, ...)
V::LANES          // lane count, usize const
V::lanes()        // lane count as a value; prefer in loop bounds / address math
V::Lanes          // lane count as a typenum type
V::Unsigned       // unsigned int vector, same lane count & bit width
V::Signed         // signed int vector, same lane count & bit width
V::Mask           // boolean mask type for this vector
```

```rust
fn sum_lanes<V: NumericVector>(v: V) -> V::Element { v.sum_elements() }
fn abs_bits<V: FloatVectorWithBits>(v: V) -> V::Bits { v.abs().into_bits() }
fn pick<V: GenericVector>(m: V::Mask, a: V, b: V) -> V { m.select(a, b) }
```

Per-element constants generically: bound the element,
`fn f<V: FloatVector<Element: MyConsts>>()`, read
`<V::Element as MyConsts>::FOO` (used throughout the math kernels; see
[performance.md](performance.md) sec 5).

## Running the generic fn on concrete types

For `fn f<V: FloatVector + TranscendentalMath>(x: V) -> V`:

**1. Scalar (1-lane) backend, no dispatch.** `Vector<f32>`/`Vector<f64>` are real
vector types with `LANES == 1` (scalar register impls). No `target_feature`
needed; ideal for tests and for seeding composites. (This is the main place
naming a concrete `Vector<R>` is legitimate -- scalar code has no alternative.)

```rust
let y = f(0.5_f64.as_vector());          // Element::as_vector -> Vector<f64>, 1 lane
let s = y.extract::<0>();

let z = f(Vector::<f64>::splat(0.5));    // longhand, identical
```

**2. Native-width SIMD via runtime dispatch.** `dispatch_dyn!` detects the best
ISA at runtime (cached), establishes the `target_feature` context, and rewrites
bare width names (`f32xN`, `f32x8`, `i32x4`, `f64xN`, ...) in the block to
`Vector<S::...>` for the chosen backend. `f32xN` = widest native f32 width.

```rust
let s = thermite::dispatch_dyn!(for<S> || -> f32 {
    f(f32xN::splat(0.5)).extract::<0>()
});
```

For a reusable library kernel with per-ISA codegen, annotate with the
`#[thermite::dispatch(Self)]` / `#[thermite::dispatch(S)]` proc-macro
**attribute** (not a bang macro) and make it generic over `S: Simd`. Both tools
and their composition: [slices-and-dispatch.md](slices-and-dispatch.md).

**3. `Dual<V, N>` -- autodiff.** `thermite-dual`'s `Dual<V, N>` (primal + `N`
first-order partials) implements the float/math traits by delegating to `V` with
the chain rule:

```rust
use thermite_dual::AutoDiff;
type V = Vector<f64>;
// .ad([...]) seeds each input as an independent variable, instantiates f at
// Dual<V, N>, returns value + gradient. No turbofish, no closure wrapper.
let r = f.ad([V::splat(0.5)]);
let value = r.re.extract::<0>();      // f(0.5)
let dfdx  = r.dual[0].extract::<0>(); // f'(0.5)
```

**4. `Compensated<V>` -- double-double.** hi/lo pair, ~2x mantissa bits, via
error-free transforms:

```rust
use thermite_compensated::Compensated;
let c = f(Compensated::<Vector<f64>>::new(Vector::splat(0.5)));
let y = c.value().extract::<0>(); // hi+lo folded to one f64
```

See [composite-types.md](composite-types.md) for the delegation mechanics and
nesting (`Dual<Compensated<V>, N>` = high-precision autodiff).

## The prelude, and naming math traits in bounds

`use thermite::prelude::*;` brings in `Vector`, `Mask`, the vector-trait
hierarchy, `SimdSlice`, operator/masked-op traits, `FloatConsts`, `Policy`, the
`Simd`/`FloatSimd` families, dividers, `Element`/`FloatElement`.

**Caveat:** the prelude imports the math traits anonymously (`CoreMath as _`,
...). Methods are callable, but `<V: TranscendentalMath>` will not compile until:

```rust
use thermite::math::{TranscendentalMath, RealMath, CoreMath, SpatialMath, FloatMath};
```

## Idiom: bundle traits into a domain trait

When a function needs several traits and is called in many places, define one
domain trait bundling the bounds + a blanket impl (the dominant production
pattern):

```rust
pub trait NnVector: SwizzleVector + RealMath + SpecialMath<Element = f32> {}
impl<V> NnVector for V where V: SwizzleVector + RealMath + SpecialMath<Element = f32> {}

fn activation<V: NnVector>(x: V) -> V { /* RealMath + SpecialMath methods */ }

// Bundle at the backend (Simd) level too, with per-type associated bounds:
pub trait NnBackend: SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x16: NnVector> {}
impl<S> NnBackend for S where S: SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x16: NnVector> {}
```

Pin a width to compile-assert it:
`fn f<V: NnVector<Lanes = thermite::generic_array::typenum::U8>>(...)`.
Math-trait names must be imported to *name* them here (see above).

More constant idioms:

- **Table lookup / interpolation = gather + mix**, not a scalar loop: index
  vector `i`, fraction `t`, then
  `let (a, b) = (V::gather(table, i), V::gather(table, i + 1)); t.mix(a, b)`.
- **Stay in-register with `swizzle!`** to avoid extract/insert in hot loops:
  `let p = thermite::swizzle!(f, [1, 0, 1, 1]);` then combine with FMAs.
- **`if const { V::ISA.is_simd() }`** (or any `HasIsa`/capability const) forks
  scalar vs SIMD paths at compile time, zero runtime branch.

## Pitfalls

- **Bare `f32`/`f64` are not `FloatVector`.** Wrap: `x.as_vector()` (the
  `Element` method, prelude-imported) or the longhand `Vector::<f64>::splat(x)`.
  One-off scalar math: `use thermite::math::ScalarMath; let s = x.scalar_sin();`
  (`scalar_`-prefixed to avoid clashing with inherent `f64::sin`). See [math.md](math.md).
- **Don't over-constrain.** `FloatVector` when you only `+`/`*` excludes integer
  and some composite instantiations; use `NumericVector`.
- **Don't name a backend type** (`f32x8`, `__m256`, `S::f32x8`) in a reusable
  signature -- defeats the design. Stay on the trait. Even a concrete
  `Vector<R>` is a smell outside the two legitimate uses: scalar 1-lane seeds
  (`Vector<f32>`/`Vector<f64>`, unavoidable for scalar code and tests) and the
  inside of a `dispatch_dyn!` boundary.
- **Never drop to the Register layer** (`R::method(...)`, `Storage<R>`) in user
  or generic code. It is the backend-implementation surface: no operators, no
  ergonomics, and it works in raw `Storage` types rather than `Vector`/`Mask`.
  If an op you need is missing from the `*Vector` traits, that's a Thermite
  change ([development.md](development.md)), not a reason to call registers.
- **`V::LANES` is backend-dependent.** Never assume 4 or 8. Use slice iterators
  ([slices-and-dispatch.md](slices-and-dispatch.md)) and `V::indexed()` rather
  than hardcoded lane indices, except compile-time `extract::<I>()`/`insert::<I>()`.
- **Element constants in generic code**: `V::splat(3.0)` fails when `3.0` isn't
  obviously the element type. Use provided constants (`V::ONE`, `V::TWO`,
  `V::HALF`, `FloatConsts`), `V::splat(V::Element::...)`, or the `ConstRatio`
  pattern in [performance.md](performance.md).
