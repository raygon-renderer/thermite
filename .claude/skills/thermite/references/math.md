# The math library

Transcendental/real-math functions, a compile-time **policy** system for
precision/perf trade-offs, scalar shortcuts, ~45 float constants. Defined in
`crates/thermite/src/math/`.

## Math trait families

Each family is a pair: policy-parameterized `*WithPolicy` (methods suffixed `_p`,
generic over `<P: Policy>`) and default-policy `*Math` (same methods, no suffix,
`DefaultPolicy`). All require at least `FloatVector`. Declared by `decl_math!` in
`math/mod.rs`.

| Trait | Requires | Provides (selection) |
|---|---|---|
| `FloatMath` | `FloatVectorWithBits` | `ldexp`, `frexp`, `flush_denormals` |
| `CoreMath` | `FloatVector` | `poly`, `poly_rev`, `poly_rational`, `reciprocal`, `approx_div`, `inverse_sqrt`, `powi`, `powiv`, `harmonic_mean`, `inv_sum_inv` |
| `TranscendentalMath` | `CoreMath` | `sin`,`cos`,`tan`,`sin_cos`,`sin_pi`,`cos_pi`,`tan_pi`,`sinc`,`sinhc`,`cosh_m1`,`atanhc`,`asin`,`acos`,`atan`,`sinh`,`cosh`,`tanh`,`asinh`,`acosh`,`atanh`,`exp`,`exp2`,`exp10`,`exp_m1`,`ln`,`ln_1p`,`log1pmx`,`xlogy`,`xlog1py`,`log2`,`log10`,`log_n`,`cbrt`,`nth_root`,`powf`,`compound`,`versin`,`haversin`,... |
| `SpatialMath` | `CoreMath` | `hypot`, `hypot_n`, `inv_hypot_n`, `l1_norm`, `l2_norm`, `l2_norm_squared` |
| `RealMath` | `Transcendental + Spatial` | `atan2`, `lerp`, `rescale`, `to_degrees`, `to_radians`, `wrap_angle`, `angle_diff`, `logaddexp`, `smoothstep`, `inverse_smoothstep`, `smoothstep_derivative`, `smooth_interpolator`, `step` |

To *name* these in a bound, import explicitly (prelude imports them anonymously):

```rust
use thermite::math::{CoreMath, TranscendentalMath, SpatialMath, RealMath, FloatMath};
fn f<V: FloatVector + TranscendentalMath>(x: V) -> V { x.sin() + x.exp() }
```

## Policy variants `_p::<P>()`

Every math method has a policy sibling; the plain method delegates with
`DefaultPolicy`:

```rust
let a = x.sin();                         // DefaultPolicy
let b = x.sin_p::<HighPerformance>();
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

Presets (`thermite::math::policy::policies`, re-exported):

| Preset | precision | overflow | branchless | compensated | denormals | use when |
|---|---|---|---|---|---|---|
| `UltraPerformance` | Worst | no | yes | no | Crush | fastest, sloppy |
| `HighPerformance` | Medium | no | no | no | Crush | fast |
| `Performance` | Average | yes | no | no | FlushToZero | **CPU default** |
| `Precision` | Best | yes | no | yes | FlushToZero | accuracy |
| `Size` | Average | yes | no | no | Crush | **WASM default**, small code |
| `Reference` | Reference | yes | no | yes | FlushToZero | validation only (slow) |
| `GpuDefault` | Average | yes | yes | no | Crush | SPIR-V default |

`DefaultPolicy` = `Performance` on CPU, `Size` on WASM, `GpuDefault` on SPIR-V
(cfg-selected).

Composable modifiers wrap a base policy `P`, all implement `Policy`:

```rust
ExtraPrecision<P>   LessPrecision<P>
WorstPrecision<P>   MediumPrecision<P>   AveragePrecision<P>   BestPrecision<P>   ReferencePrecision<P>
CheckOverflow<P, const ON: bool>   AvoidBranching<P, const ON: bool>
UnrollLoops<P, const ON: bool>     UseCompensation<P, const ON: bool>
MaxIterations<P, const N: usize>   CmpLessPrecision<A, B>   // min precision of two

// "HighPerformance but checked and extra-precise":
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

- **Estimating (`*e`)**: real FMA if hardware has it, else separate `mul`+`add`.
  Default choice -- fast everywhere.
- **FMA-quality (no `e`)**: real FMA if available -- then genuinely single-rounded.
  Otherwise **by default** a *vectorized emulated FMA* (compensated split --
  slower than true FMA but still SIMD and far cheaper than `libm`, and **not**
  bit-identical to one: measured, about 1 in 173,000 differ for f64 and 1 in
  3,000,000 for f32, worst relative error 2.0e-15). Only `disable_fast_fma` (implied by
  `strict_ieee754`) makes the fallback the exact scalar `libm::fma`, *dozens of
  times slower*. So non-`e` forms are a valid **accuracy** choice even without
  hardware FMA; gate on `V::HAS_TRUE_FMA` to avoid the *emulation* cost, not
  merely to avoid `libm`.

Detail and ILP techniques: [performance.md](performance.md).

## Scalar shortcut: `ScalarMath` for bare `f32`/`f64`

Bare `f32`/`f64` do **not** implement `FloatVector`. For one-off scalar math,
`ScalarMath`/`ScalarMathWithPolicy` are implemented directly on `f32`/`f64`, all
methods `scalar_`-prefixed (avoids clashing with inherent `f64::sin`):

```rust
use thermite::math::ScalarMath;
let s = 0.5_f64.scalar_sin();
let e = 2.0_f32.scalar_exp_p::<HighPerformance>();
```

For *generic* code that must accept scalars, wrap the element: `x.as_vector()`
(or the longhand `Vector::<f64>::splat(x)`).

## FloatConsts: ~85 constants

Implemented for `f32`, `f64`, and every `Vector<R: FloatRegister>`. Prefer these
over recomputing (`V::SQRT_EPSILON`, not `V::EPSILON.sqrt()`). Coverage is a
superset of Boost.Math's constants table:

```
PI TAU E PHI EULER_GAMMA  FRAC_PI_2 FRAC_PI_3 FRAC_PI_4 FRAC_PI_6 FRAC_PI_8
FRAC_1_PI FRAC_2_PI  LN_2 LN_10 LN_PI  LOG2_E LOG2_10 LOG10_E LOG10_2
SQRT_2 SQRT_3 SQRT_E  FRAC_1_SQRT_2 FRAC_1_SQRT_3 FRAC_1_SQRT_PI FRAC_2_SQRT_PI
EPSILON SQRT_EPSILON FOURTH_ROOT_EPSILON  PI_SQUARED PI_CUBED PI_FOURTH
FRAC_1_3 FRAC_2_3 FRAC_1_4 FRAC_1_6  FRAC_PI_180 FRAC_180_PI  NEG_ZERO
SQRT_PI CBRT_PI FRAC_1_CBRT_PI  FRAC_2PI_3 FRAC_3PI_4 FRAC_4PI_3 FRAC_1_TAU
PI_MINUS_3 FOUR_MINUS_PI PI_POW_E E_POW_PI  SIN_1 COS_1 SINH_1 COSH_1
LN_PHI FRAC_1_LN_PHI  ZETA_2 ZETA_3 CATALAN GLAISHER KHINCHIN
FEIGENBAUM_DELTA PLASTIC_RATIO GAUSS DOTTIE PSI LAPLACE_LIMIT ...
```

**Constants are generated, never typed by hand.** `gen_consts.py` at the
workspace root is the single source of truth: its `CONSTS` list produces
thermite's `math/consts/mod.rs`, thermite-compensated's double-double splits,
and thermite-interval's enclosure pairs. Adding one is a single edit there plus
`python gen_consts.py` (`--check` fails if the tree is stale). No other crate
lists constant names - `thermite::for_each_float_const!(my_macro)` expands with
all of them, `for_each_math_const!` with all but the format-specific epsilons.
Verify with `cargo nextest run --release -p thermite-interval --test consts`,
which audits every constant for correct rounding by exact rational comparison.

## Algorithms module (`math/algorithms/`)

Generic numerical building blocks; convergent ones over `V: FloatVector` + a
`Policy`, reductions over any `V: Copy` (no policy):

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
- `min`/`max` default to fast SSE-style asymmetric NaN propagation, not IEEE
  `minNum`/`maxNum`. `strict_ieee754` enables IEEE-correct behavior (also governs
  denormals/NaNs elsewhere, at a perf cost).
- `ldexp_f32` at extreme exponents (`ldexp(f32::MAX, i32::MIN)`) only flushes
  correctly under `strict_ieee754`.
- **Denormals are handled unevenly, and mixing the two halves is silently wrong.**
  `hypot` flushes them -- at *every* policy, `Precision` included -- while `*`, `+`
  and `sqrt` beside it do not. A formula combining both gives an answer that is
  neither the flushed one nor the true one: complex `sqrt` of a subnormal came out a
  factor of `sqrt(2)` low, because `|z|` was flushed while the `|re|` added to it
  survived. If a kernel touches the bottom of the range, make the two agree.
- Low-precision `hypot` (`Worst`) squares directly, so it underflows to zero for
  inputs as large as `1e-300` -- well above the subnormal range.
- At policies with `check_overflow = false` (`UltraPerformance`, `HighPerformance`)
  `exp` **saturates to roughly MAX rather than to infinity**, deliberately: the
  recombination would otherwise hand back `0 * inf = NaN` for exact-integer inputs.
  Consumers that divide two saturated values (`sinh/cosh`) therefore still cancel to
  `1.0`; consumers expecting a literal `inf` should ask for a checked policy.

## Denormal configuration is a *behaviour* switch, not just a dial

`thermite::features` exposes const bools -- `PRESERVE_DENORMALS`, `IGNORE_DENORMALS`,
`STRICT_IEEE754`, `DISABLE_FAST_FMA`, `ALGEBRAIC_SCALAR` -- usable in `if const`.
Reach for them when a formula's *degenerate case moves* between configurations:

```rust
// Which value hits zero first depends on the build; neither implies the other.
let degenerate = if const { features::PRESERVE_DENORMALS || features::IGNORE_DENORMALS } {
    t.is_zero()   // modulus keeps the subnormal, but halving the smallest underflows
} else {
    m.is_zero()   // hypot flushed the modulus, while |re| beside it survived
};
```

Test both. A guard keyed on the right value for the default build was silently wrong
under `preserve_denormals` -- same function, same input, opposite failure. Nothing in
the default test suite catches that.
