# Performance: writing fast, accurate, portable kernels

All generic over the trait hierarchy (applies to every backend at once), drawn
from real hot paths (elliptic-integral and transcendental kernels). Theme: **the
compiler will not reassociate FP math, pick the cheapest hardware primitive, or
hoist capability-specific paths -- you do that once, in generic code, and every
backend benefits.**

## 0. Rule zero: `#[thermite::dispatch]` on the entry, `#[inline(always)]` on the helpers

**This outranks every other item on this page**, and the failure mode is far
worse than "falls back to the baseline ISA". `core::arch` intrinsics are
`#[target_feature]`-gated `#[inline]` functions: **rustc refuses to inline a
`target_feature` function into a caller that does not enable those features.** In
a body without the features, every intrinsic Thermite uses internally stays
out-of-line -- a `call` per `_mm256_add_ps`, per shuffle, per load, with ABI
register shuffling around each and no scheduling, register allocation, or
constant folding across ops. An order-of-magnitude class of loss, not a
percentage -- and it compiles and is correct, so nothing in the type system or
test suite catches it. Check it *first* when a kernel underperforms.

Two attributes, and they do different jobs:

- **`#[thermite::dispatch(S)]`** (or `(Self)` / `(TypeName)`) on any fn, impl,
  trait or mod generic over the backend. It emits one
  `#[target_feature(enable = "avx2,fma,...")]` trampoline per backend and a
  `match <S as HasIsa>::ISA` that const-folds at monomorphization. This is the
  **only** way a function body gets per-ISA codegen -- i.e. the only way the
  intrinsics inside it are allowed to inline at all.
- **`#[inline(always)]`** on every small helper called from inside a dispatched
  body. Target features propagate into a callee only when it is inlined into the
  enabled context; a helper that does not inline is compiled *without* the
  features, so the intrinsics inside *it* in turn refuse to inline -- the
  call-per-instruction soup, one level down. `#[inline]` (or nothing) is a
  *hint* the optimizer routinely declines in exactly the big `target_feature`
  bodies where it matters most.

Why this pairing is cheap rather than a code-size disaster: **`#[dispatch]` is a
real function boundary, deliberately.** The trampoline carries the target
features itself, so the compiler is free to *not* inline the dispatched fn --
one out-of-line copy per backend, called normally -- and the body still gets
full AVX2/FMA/NEON codegen. Inline aggressively *inside* the kernel; let the
compiler size the boundary.

Practical shape:

```rust
#[inline(always)]                                  // interior: MUST inline to keep features
fn step<V: FloatVector>(v: V) -> V { v.mul_adde(v, v) }

#[thermite::dispatch(S)]                           // boundary: per-backend target_feature
pub fn kernel<S: FloatSimd<f32>>(data: &mut [f32]) {
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<Vector<S::fxN>>();
    for v in chunks { *v = step(*v); }
}

let _ = thermite::dispatch_dyn!(kernel(&mut data)); // runtime ISA selection
```

Rules of thumb:

- Every public SIMD entry point in a dependent crate gets `#[dispatch]`. If a
  generic-over-`S` fn has no `#[dispatch]` above it and no `#[dispatch]` ancestor
  it is inlined into, it is a bug.
- `dispatch_dyn!`'s call form only emits the runtime match -- it assumes the
  callee is `#[dispatch]`. A plain generic callee there is *correct but
  un-inlined*: the classic silent-slow case.
- Don't reach for `#[dispatch]` on tiny leaf helpers; a dispatch boundary on a
  one-liner just blocks inlining. Those are `#[inline(always)]` (that is what
  `#[skip_dispatch]` exists for inside `decl_math!`).
- Closures and `core::array::map`/`from_fn` inside a dispatched body often fail
  to inline, which puts their intrinsic bodies outside the feature context again
  -- hand-roll loops (sec 11).
- Verify, don't assume: dump asm (sec 13). The tell is unmistakable -- streams of
  `call` into one-instruction stubs (`_mm256_*` symbols surviving in the binary)
  instead of straight-line `vfmadd*`/`vmulps` on `ymm` registers.

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

There is one more reason to gate an *`e`-form* fold, and it is about op count, not
accuracy: if the plain spelling would **share** the product you are folding
(`t3 - a*s` and `t3 + a*s` both want `a*s`), the non-FMA lowering of two folds
recomputes it. Fold ungated when the product appears once; gate when it would be shared.
See [optimization-pass.md](optimization-pass.md) sec 1 - and sec 2 for why gating on a
`Complex<V>` or `Dual<V, N>`'s own `HAS_TRUE_FMA` (always `false`) is a trap.

## 2. Fold negations into constants

`nmul_adde(c, x, acc)` negates a runtime product. If `c` is a compile-time constant,
negate the *constant* and use a plain `mul_adde` (one fewer op):

```rust
x.nmul_adde(c!(3 / 14), acc)   // acc - (3/14)x
x.mul_adde(c!(-3 / 14), acc)   // (-3/14)x + acc   <-- prefer; ConstRatio allows negative numerators
```

Reserve `nmul_*`/`mul_sub*` for negating genuine *variables*.

An FMA against a *structural* constant is not free either: IEEE forbids folding
`a*0.0 + b` to `b` (`a` may be infinite, and the zero has a sign), so `z.mul_add(I, w)`
really does emit four inner FMAs over a table of zeros and ones. Spell the swap by
hand -- `iz = Complex::new(-z.im, z.re)` is two moves.

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
- **Two quotients, one divide**: `(x/p, y/q) = (x*q*r, y*p*r)` with `r = 1/(p*q)`.
  Any function returning a pair of ratios should pay one division, and a constant
  numerator (`2a/...`) rides in the numerator of that division for free.
  [optimization-pass.md](optimization-pass.md) sec 3-4 has the pattern and the larger
  win behind it: substituting out intermediates (a relative index, a `cos_t`) that exist
  only to be multiplied back.

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

For plain integer/rational literals prefer `const_splat!(int <V::Element>: N)` /
`const_splat!(ratio <E>: N, D)` over `V::splat(E::from_int(N))` -- a true const splat,
no runtime conversion. It will *not* accept a generic const parameter (`N as LargeInt`
where `N: const usize`): that is a const operation over a generic, which rustc rejects.
Fall back to `from_int` there. A `let` binding may need an explicit `: V`.

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

### 7a. Keep both forms, pick with `precision` (best-of-both)

Where the accurate form costs more, keep both and choose with
`if const { P::POLICY.precision.le(PrecisionPolicy::Average) }`. Measured cases:

- `2^z` through the *real* `exp2`, not `exp(z ln 2)` -- scaling the argument rounds
  it and `exp` then amplifies by the argument. `exp2(1000)` was ~300 ulp out.
  Same for `exp10`/`exp2_m1`/`exp10_m1`.
- `z^w` as one fused `exp(c ln r - d t)` vs `powf(r,c) * exp(-d t)`. The fused form
  is *cheaper* (`ln r` is needed for the angle anyway, and `powf` is `exp(c ln r)`
  underneath) and cannot overflow an intermediate; the split form measured ~1.7x
  more accurate. Cheap below `Average`, accurate above -- plus the fused form as a
  `check_overflow` fallback for lanes where the split one left the range.

### 7b. Cancellation: reach for the algebraic conjugate

When `w = a + b` cancels, `w' = a - b` does not, and often `w * w' = const`. Every
inverse trig/hyperbolic log has `w * w' = 1`, so `ln w = -ln w'`. Blend the
*argument* before the transcendental and it is still one `ln`:

```rust
let flip = w.norm_sqr().cmp_lt(V::ONE) & p.norm_sqr().cmp_gt(V::ONE);
let l = flip.select(companion, w).ln_p::<P>();   // one ln, then negate where flipped
```

The second half of that test is load-bearing: `|w| < 1` alone also fires for tiny
`w = 1 + p`, where nothing cancelled. At *that* end the fix is `ln_1p` on `w - 1`,
computed without a subtraction (`s - 1 = p^2/(s + 1)`) -- worth 8 digits.

### 7c. Range guards: key on the damage, not on a threshold

`tan`/`tanh` saturate but their `sinh/cosh` overflow to `inf/inf`. Testing
`denom.is_infinite()` fires on exactly the broken lanes and needs no per-element
constant; use `is_infinite`, not `!is_finite`, so a NaN argument still yields NaN.
Gate on `P::POLICY.check_overflow`.

When saturating an exponent by hand, stop one below the all-ones field. Recombination
is `z * n2 + n2`, and `z` is *exactly* zero whenever the reduced argument is (every
integer input to `exp2`), so an infinite `n2` yields `0 * inf = NaN` on the cleanest
inputs in the range.

### 7d. A cold branch is often free

If a mask is already computed to drive a blend, branch on it instead and move the
blend inside. The hot path loses an unconditional `select` from its dependency chain
and gains a test the predictor calls correctly. That is what makes an exact
subnormal/overflow fallback affordable in a policy-free method like `FloatVector::sqrt`.

### 7e. Do not rig your own reference

Compute expected values with an independent oracle (mpmath at 50 dps), not with the
same expression the code uses. Two "measurements" once agreed to 0.00e0 purely
because reference and implementation were both the fused form.

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

## 12. Bounds checks throttle SIMD kernels

`from_slice`/`copy_to_slice` bounds checks inside a hot loop cost more than a
compare: they break LLVM's block scheduling around the SIMD ops (in one real
kernel this was the *entire* performance gap vs RustFFT). Hoist one length
assert before the loop, or elide with `core::hint::assert_unchecked` on the
index bound, so the loop body is check-free and schedules as a single block.

## 13. Verify your own kernels

- Keep tight accuracy tests (e.g. `1e-13` for f64) so a "harmless" transform is
  *proven* harmless; test boundary cases.
- Add an f32 case for any `if const { HAS_APPROX_RSQRT }`-style branch -- f64-only
  tests never exercise it.
- When a kernel is hot, confirm the codegen is what you expect: emit assembly
  (e.g. `cargo rustc --release -- -C target-cpu=x86-64-v3 --emit asm`, or
  `cargo asm`) and check for real `vfmadd*`, the expected `vsqrtps`/`vdivps`
  count, broadcast constants, and no spills. Measure with a real harness
  (`cargo bench` / Criterion), not by eyeballing or invoking compiled artifacts.
