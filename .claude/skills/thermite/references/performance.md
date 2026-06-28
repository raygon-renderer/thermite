# Performance: writing fast, accurate, portable kernels

Everything here is generic over the trait hierarchy, so it applies to every backend
at once -- and is drawn from real hot paths (the elliptic-integral and
transcendental kernels). The
theme: **the compiler will not reassociate FP math, will not pick the cheapest
hardware primitive, and will not hoist capability-specific paths -- you do that
once, in generic code, and every backend benefits.**

## 1. FMA: pick the right variant

| Variant | Meaning | Use when |
|---|---|---|
| `mul_adde` / `mul_sube` / `nmul_adde` / `nmul_sube` | **Estimating**: real FMA if HW has it, else `mul`+`add`. | **Default. Almost everything.** |
| `mul_add` / `mul_sub` / `nmul_add` / `nmul_sub` | **Always single-rounded**: real FMA if HW has it; else a *vectorized emulated FMA* (compensated split) by default, or exact scalar `libm::fma` under `disable_fast_fma`. | When you want FMA-quality single-rounding even on non-FMA hardware and can accept the emulation cost. |

Signs: `mul_adde(a,b,c)=a*b+c`, `mul_sube=a*b-c`, `nmul_adde=c-a*b`, `nmul_sube=-a*b-c`.

On a non-FMA backend (pre-Haswell x86, much of WASM) the `e` variants are just
`mul`+`add` (two roundings, fastest). The non-`e` variants do **not** fall straight to
`libm::fma`: by default they lower to a SIMD **emulated FMA** (a Dekker/Veltkamp
compensated split) that is single-rounding-accurate, slower than true FMA but far
cheaper than `libm`, and **not bit-identical** to true FMA. Only the `disable_fast_fma`
feature (implied by `strict_ieee754`) swaps in the exact scalar `libm::fma`, which *is*
dozens of times slower. So: **reach for the `e` variants by default for speed**, but
`mul_add` is a legitimate *accuracy* choice on non-FMA hardware when a modest slowdown is
fine -- you don't have to gate it behind `HAS_TRUE_FMA` just to dodge `libm`. Gate it only
to avoid the emulation cost. If you truly need extra precision, pulling in
`thermite-compensated` directly is often cleaner than relying on emulated FMA.

## 2. Fold negations into constants

`nmul_adde(c, x, acc)` negates a runtime product. If `c` is a compile-time constant,
negate the *constant* and use a plain `mul_adde` (one fewer op):

```rust
x.nmul_adde(c!(3 / 14), acc)   // acc - (3/14)x
x.mul_adde(c!(-3 / 14), acc)   // (-3/14)x + acc   <-- prefer; ConstRatio allows negative numerators
```

Reserve `nmul_*`/`mul_sub*` for negating genuine *variables*.

## 3. Gate hardware paths with `if const`

Capability constants resolve at compile time inside the dispatcher's
`target_feature` context. Branch with `if const { ... }` for zero-cost per-backend
code:

```rust
if const { V::HAS_TRUE_FMA } {
    let lq = lambda * quarter;
    an = an.mul_add(quarter, lq);   // always-fused is SAFE here: HW FMA is proven
} else {
    an = (an + lambda) * quarter;   // plain form; the extra mul would be wasted under FMA
}
```

`V::HAS_APPROX_RSQRT` is the other big one: **f32 has a hardware rsqrt, f64 does
not** (there `rsqrt = rcp(sqrt)` = sqrt+div). So `a / sqrt(b)` wins on f64 (one
`div`) but `a * rsqrt(b)` wins on f32. Gate it and recover roots by multiply on the
f32 path using `inverse_sqrt_p::<P>()`.

## 4. Shorten the critical path (ILP), not just op count

On a superscalar core the limiter is usually dependency-chain *latency*, not
instruction count. LLVM can't do these (FP is non-associative without fast-math):

- **Split a serial Horner chain into parallel sub-chains by degree**, sum at the end.
- **Re-parenthesize reductions into balanced subtrees**: `((x+y)+(z+z+z))` (depth 3)
  beats `(x+y+z+z+z)` (depth 4); `(x+y+z)-(lo+hi)` beats `(x+y+z)-lo-hi`.
- **Factor to collapse a dependency**: `rx.mul_adde(ry+rz, ry*rz)` (depth 2) beats the
  3-deep serial FMA chain for `rx*ry+rx*rz+ry*rz`.
- **Hoist work independent of the latest-arriving input** so the result is one FMA
  past it.
- **Hoist reciprocals**: compute `1/x` once, multiply -- division latency dwarfs
  multiply. Share one `reciprocal_p`/`inverse_sqrt_p` between code paths that divide
  by the same root.

Algebraic transforms via the `e` variants change rounding by <1 ulp -- fine for
convergent iterations and small corrections; verify with accuracy tests.

## 5. Precompute runtime work as compile-time constants

If a value is a fixed function of the element type, make it a `const` in a small
per-element trait and `V::splat` it (no runtime math):

```rust
pub trait EllipticConsts { const CARLSON_THRESH: Self; }
impl EllipticConsts for f32 { const CARLSON_THRESH: f32 = 0.156379178; }
impl EllipticConsts for f64 { const CARLSON_THRESH: f64 = 0.012674919; }
fn thresh<V: FloatVector<Element: EllipticConsts>>() -> V {
    V::splat(<V::Element as EllipticConsts>::CARLSON_THRESH)
}
```

Check `FloatConsts` first -- `V::SQRT_EPSILON`, `V::FRAC_1_PI`, etc. already exist;
prefer them to `V::EPSILON.sqrt()`. Guard hand-entered literals with a test that
recomputes and asserts bit-equality.

## 6. Polynomials

- Univariate: `t.poly_rev_p::<P, _>(&coeffs)` (leading-coeff-first, hybrid
  Estrin/Horner with FMA, from `fast_polynomial`).
- Multivariate (several distinct variables): hand-roll FMA chains + apply the ILP
  splitting from section 4.

## 7. Numerical accuracy

The enemy is **catastrophic cancellation**.

- `1 - x*x`: use `x.one_minus_sq()` (FMA `nmul_add(x,x,1)`, or `(1-x)(1+x)` -- both
  cancellation-free). Same for `sqrt(1 - k*k)`.
- Reconstruct a small cancelling quantity from its full-precision origin, not from
  the converged values.
- Switch to a small-argument series where a closed form cancels, with a branchless
  `select` crossover.
- Accuracy is **policy-gated**: `reciprocal_p`/`inverse_sqrt_p`/transcendentals are
  exact under `Precision`, approximate under perf policies. If a kernel needs one
  exact, say so and test it under the precision policy.

## 8. `scale` for scalar multiply (SPIR-V codegen)

`v.scale(elem)` is bit-identical to `v * splat(elem)` on CPU but lowers to a single
`OpVectorTimesScalar` on SPIR-V. Applies to pure `vector * scalar` only -- a
constant *inside* an FMA must stay a splatted vector.

## 9. Branchless selection and `0 * NaN`

Prefer `mask.select(a, b)` over data-dependent branches -- lanes diverge, and the
policy may forbid branching (`P::POLICY.avoid_branching`). Only branch behind a
runtime `.all()`/`.any()` the policy permits. Beware `0 * NaN = NaN` poisoning a
conditionally-invalid term -- use `select` to keep the good lane.

## 10. Where to put a new operation

- A **capability-gated algebraic identity** (no precision dimension, e.g.
  `one_minus_sq`) -> a *provided/default method on a vector trait*. It inlines and is
  inherited by wrappers like `Compensated<V>`/`Complex<V>` automatically. A new
  *required* method would break their impls.
- A **precision-tunable function** -> the `decl_math!` family (generates `_p::<P>()`,
  the scalar surface, and ISA dispatch). But `decl_math!` creates a dispatch boundary
  (a real call) -- don't put trivial one-liners there; they should inline.

## 11. `target_feature` codegen gotchas (these pass tests but tank the benchmark)

- `core::array::map` / `core::array::from_fn` **fail to inline** in SIMD/`target_feature`
  code and fall back to scalar -- hand-roll a `while` loop.
- Bare closures passed to `std` combinators (`map`, `from_fn`) often don't inline
  either. `#[inline(always)]` helpers are fine; bare closures are not.

## 12. Verify your own kernels

- Keep tight accuracy tests (e.g. `1e-13` for f64) so a "harmless" transform is
  *proven* harmless; test boundary cases.
- Add an f32 case for any `if const { HAS_APPROX_RSQRT }`-style branch -- f64-only
  tests never exercise it.
- When a kernel is hot, confirm the codegen is what you expect: emit assembly
  (e.g. `cargo rustc --release -- -C target-cpu=x86-64-v3 --emit asm`, or
  `cargo asm`) and check for real `vfmadd*`, the expected `vsqrtps`/`vdivps`
  count, broadcast constants, and no spills. Measure with a real harness
  (`cargo bench` / Criterion), not by eyeballing or invoking compiled artifacts.
