# The math library

Transcendental and real-math functions, a compile-time **policy** system for
precision/performance trade-offs, scalar shortcuts, and ~45 float constants.
Defined in `crates/thermite/src/math/`.

## Math trait families

Each family comes as a pair: a policy-parameterized `*WithPolicy` trait (methods
suffixed `_p`, generic over `<P: Policy>`) and a default-policy `*Math` trait
(same methods, no suffix, using `DefaultPolicy`). All require at least
`FloatVector`. Declared by the `decl_math!` macro in `math/mod.rs`.

| Trait | Requires | Provides (selection) |
|---|---|---|
| `FloatMath` | `FloatVectorWithBits` | `ldexp`, `frexp`, `flush_denormals` |
| `CoreMath` | `FloatVector` | `poly`, `poly_rev`, `poly_rational`, `reciprocal`, `approx_div`, `inverse_sqrt`, `powi`, `powiv` |
| `TranscendentalMath` | `CoreMath` | `sin`,`cos`,`tan`,`sin_cos`,`sin_pi`,`cos_pi`,`tan_pi`,`sinc`,`asin`,`acos`,`atan`,`sinh`,`cosh`,`tanh`,`asinh`,`acosh`,`atanh`,`exp`,`exp2`,`exp10`,`exp_m1`,`ln`,`ln_1p`,`log2`,`log10`,`log_n`,`cbrt`,`nth_root`,`powf`,`compound`,`versin`,`haversin`,... |
| `SpatialMath` | `CoreMath` | `hypot`, `hypot_n`, `inv_hypot_n`, `l1_norm`, `l2_norm`, `l2_norm_squared` |
| `RealMath` | `Transcendental + Spatial` | `atan2`, `lerp`, `rescale`, `to_degrees`, `to_radians`, `wrap_angle`, `angle_diff`, `logaddexp`, `smoothstep`, `inverse_smoothstep`, `smoothstep_derivative`, `smooth_interpolator`, `step` |

To name any of these in a generic bound you must import it explicitly (the prelude
brings them in anonymously):

```rust
use thermite::math::{CoreMath, TranscendentalMath, SpatialMath, RealMath, FloatMath};

fn f<V: FloatVector + TranscendentalMath>(x: V) -> V { x.sin() + x.exp() }
```

## Policy variants `_p::<P>()`

Every math method has a policy-parameterized sibling. The plain method delegates
to the `_p` form with `DefaultPolicy`:

```rust
let a = x.sin();                         // DefaultPolicy
let b = x.sin_p::<HighPerformance>();    // explicit policy
let c = x.exp_p::<Precision>();
```

### The Policy system (`math/policy.rs`)

A `Policy` is a compile-time type carrying a `PolicyParameters` const:

```rust
pub struct PolicyParameters {
    pub check_overflow: bool,        // handle inf/NaN/domain edges
    pub unroll_loops: bool,          // use unrolled/Estrin variants
    pub precision: PrecisionPolicy,  // Worst < Medium < Average < Best < Reference
    pub avoid_branching: bool,       // branchless even at a precision cost (good for SIMD)
    pub max_iterations: usize,       // cap for convergent methods
    pub use_compensation: bool,      // Kahan/compensated summation
    pub denormal_behavior: DenormalBehavior, // Ignore | FlushToZero | Crush | Preserve
}
```

Preset policies (in `thermite::math::policy::policies`, also re-exported):

| Preset | precision | overflow | branchless | denormals | use when |
|---|---|---|---|---|---|
| `UltraPerformance` | Worst | no | yes | Crush | fastest, sloppy |
| `HighPerformance` | Medium | no | no | Crush | fast |
| `Performance` | Average | yes | no | FlushToZero | **CPU default** |
| `Precision` | Best | yes | no (compensated) | FlushToZero | accuracy |
| `Size` | Average | yes | no | Crush | **WASM default**, small code |
| `Reference` | Reference | yes | no (compensated) | FlushToZero | validation only (slow) |
| `GpuDefault` | Average | yes | yes | Crush | SPIR-V default |

`DefaultPolicy` is `Performance` on CPU, `Size` on WASM, `GpuDefault` on SPIR-V
(selected by `cfg`).

Composable modifiers wrap a base policy `P` and tweak one axis -- all implement
`Policy`:

```rust
ExtraPrecision<P>   LessPrecision<P>
WorstPrecision<P>   MediumPrecision<P>   AveragePrecision<P>   BestPrecision<P>   ReferencePrecision<P>
CheckOverflow<P, const ON: bool>   AvoidBranching<P, const ON: bool>
UnrollLoops<P, const ON: bool>     UseCompensation<P, const ON: bool>
MaxIterations<P, const N: usize>   CmpLessPrecision<A, B>   // min precision of two

// e.g. "HighPerformance but checked and extra-precise":
x.exp_p::<ExtraPrecision<CheckOverflow<HighPerformance, true>>>()
```

## FMA semantics (the #1 footgun)

Two families, four signs each:

| Estimating (PREFER) | Always-fused | Computes |
|---|---|---|
| `mul_adde(b, c)` | `mul_add(b, c)` | `a*b + c` |
| `mul_sube(b, c)` | `mul_sub(b, c)` | `a*b - c` |
| `nmul_adde(b, c)` | `nmul_add(b, c)` | `c - a*b` |
| `nmul_sube(b, c)` | `nmul_sub(b, c)` | `-a*b - c` |

- **Estimating (`*e`)**: real FMA if the hardware has it, else separate `mul`+`add`.
  Use these by default -- they are fast everywhere.
- **Always-fused (no `e`)**: real FMA if available, else `libm::fma` (exact single
  rounding, but *dozens of times slower* per lane). Only use inside a block already
  gated on `V::HAS_TRUE_FMA`, or when you genuinely need exact single-rounding.

Detail and ILP techniques in [performance.md](performance.md).

## Scalar shortcut: `ScalarMath` for bare `f32`/`f64`

A bare `f32`/`f64` does **not** implement `FloatVector`. For one-off scalar math
without constructing a vector, `ScalarMath`/`ScalarMathWithPolicy` are implemented
directly on `f32`/`f64`, with every method `scalar_`-prefixed (to avoid clashing
with inherent `f64::sin` etc.):

```rust
use thermite::math::ScalarMath;
let s = 0.5_f64.scalar_sin();
let e = 2.0_f32.scalar_exp_p::<HighPerformance>();
```

`rcp`/`sqrt`/`abs`-style names are unaffected; only the math-trait methods get the
prefix. For *generic* code that must also accept scalars, wrap them:
`Vector::<f64>::splat(x)`.

## FloatConsts: ~45 constants

`FloatConsts` is implemented for `f32`, `f64`, and every `Vector<R: FloatRegister>`.
Prefer these over recomputing (e.g. use `V::SQRT_EPSILON`, not `V::EPSILON.sqrt()`):

```
PI TAU E PHI EULER_GAMMA  FRAC_PI_2 FRAC_PI_3 FRAC_PI_4 FRAC_PI_6 FRAC_PI_8
FRAC_1_PI FRAC_2_PI  LN_2 LN_10 LN_PI  LOG2_E LOG2_10 LOG10_E LOG10_2
SQRT_2 SQRT_3 SQRT_E  FRAC_1_SQRT_2 FRAC_1_SQRT_3 FRAC_1_SQRT_PI FRAC_2_SQRT_PI
EPSILON SQRT_EPSILON FOURTH_ROOT_EPSILON  PI_SQUARED PI_CUBED PI_FOURTH
FRAC_1_3 FRAC_2_3 FRAC_1_4 FRAC_1_6  FRAC_PI_180 FRAC_180_PI  NEG_ZERO ...
```

## Algorithms module (`math/algorithms/`)

Generic numerical building blocks. The convergent ones are over `V: FloatVector` +
a `Policy`; the reductions are over any `V: Copy` (no policy):

```rust
newtons_method::<V, P>(x0, tol, bounds, |x| (f, df)) -> (root, converged_mask)  // hybrid Newton-bisection
sum_f::<V, P>(tol, start, end, |n| term)   -> Result<V, V>   // convergent series, Kahan if policy asks
prod_f::<V, P>(...)                          // convergent product
aitken_sum::<V, P>(...)                      // Aitken delta^2 acceleration
reduce_in_place(&mut [V], op)   reduce_array::<V, N>([V; N], op) -> V   // O(log n)-depth tree reduction
```

These power the inverse-smoothstep and special-function kernels.

## Known math gotchas

- `smooth_interpolator_inverse` returns NaN at the midpoint (`y = 0.5`).
- `min`/`max` default to fast SSE-style asymmetric NaN propagation (performance
  first), not IEEE `minNum`/`maxNum`. Enable the IEEE-correct behavior with the
  `strict_ieee754` crate feature when you need it (that feature also governs
  IEEE-correct denormals/NaNs elsewhere, at some performance cost).
- `ldexp_f32` for extreme exponents (e.g. `ldexp(f32::MAX, i32::MIN)`) only flushes
  correctly under the `strict_ieee754` feature.
