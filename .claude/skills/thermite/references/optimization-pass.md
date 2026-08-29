# The optimization pass: turning a correct kernel into a fast one

This is the checklist for the request "now do an optimization pass" on a kernel that is
already correct and tested. It is deliberately procedural. Every rule below was learned
by watching a pass go wrong; the failure it prevents is named next to it.

Read [performance.md](performance.md) first for the vocabulary (FMA variants, `if const`,
ILP). This file is about *deciding* which of those to apply, and *proving* you applied
them.

## 0. Before touching code

1. **Read the actual API, do not recall it.** `grep` for the sign conventions
   (`mul_sube = a*b - c`, `nmul_adde = c - a*b`, `nmul_sube = -a*b - c`), for where
   `HAS_NATIVE_FMA` is defined *for the type you are gating on*, and for which constants
   exist (`FRAC_1_2` yes, `FOUR` no). Two of the three "simple mistakes" in a typical
   pass are a mis-remembered sign or a constant that does not exist.
2. **Write the op-count table first.** For each kernel: multiplies, adds, FMAs, divides,
   square roots, transcendentals, before and after. Divides and roots are the columns
   that matter; a divide is roughly 10 multiplies of throughput and 3-4x the latency.
   If a rewrite does not move those columns it is probably not worth its complexity.
3. **One rewrite, one test run.** Never stack three algebraic transforms and then test.
   When the test fails you will not know which one broke it.

## 1. The shared-product rule: *when* to gate an FMA fold

The estimating forms (`mul_adde` and friends) lower to `mul` + `add` when the hardware
has no FMA. That means:

- **A fold whose product appears nowhere else is free everywhere.** Write
  `x.nmul_adde(x, V::ONE)` for `1 - x*x` and do not gate it - the non-FMA lowering *is*
  the plain spelling.
- **A fold whose product the plain form would share must be gated.** Folding `t4 = a*s`
  into both `t3 - t4` and `t3 + t4` as `a.nmul_adde(s, t3)` / `a.mul_adde(s, t3)` is two
  instructions with FMA and *four* without (the multiply is recomputed). The plain form is
  three (`t4`, sub, add). So:

```rust
let (num, den) = if const { has_fma::<V>() } {
    (a.nmul_adde(s, t3), a.mul_adde(s, t3))     // 2 fused ops
} else {
    let t4 = a * s;                              // share the product
    (t3 - t4, t3 + t4)
};
```

The gate is a **sharing** question, not an FMA-availability question. Ask "would the
non-FMA path materialize this product anyway?" - if yes, gate; if no, fold unconditionally.

Failure this prevents: folding everything into `_e` forms and silently paying a
duplicated multiply on SSE2/WASM; or, after being told to gate, gating *everything*
including folds that were already free.

## 2. Composite types report no FMA. Gate on the inner real type.

`Complex<V>`, `Dual<V, N>` and other composites set `HAS_NATIVE_FMA = tribool::False` at
their own layer, because a composite FMA rounds each component more than once. If your
kernel is generic over `T` and instantiated at `Complex<V>`, gating on `T`'s flag takes the
non-fused arm on AVX2 with a real FMA sitting right there.

Fix: write the component-wise pieces against the real `V` and gate on that:

```rust
const fn has_fma<V: RealFloatVector>() -> bool {
    matches!(<V as MulAddExt<V, V>>::HAS_NATIVE_FMA, tribool::True)
}
```

Corollary: a "shared generic body over `T`" that serves both real and complex
instantiations usually cannot be optimized for both. Give each its own body. The
performance mandate expects specialization; it is not over-engineering.

## 3. One division per call

Two quotients with independent denominators cost one division:

```
(x/p, y/q)  =  (x*q*r, y*p*r),   r = 1/(p*q)
```

Each numerator is scaled by the *other* denominator, then by the shared reciprocal. Two
extra roundings on the results; one fewer divide. Apply it whenever a function returns
a pair (s/p polarization, two roots, two coefficients).

Fold constants into the division's numerator - it is free:

```rust
let inv = V::TWO / (ds * dp);            // the "2" in 2a/(...) rides here
let inv = (a.norm_sqr() * four) / (es * ep);
```

`quotients(x, p, y, q)` is the helper shape; write it once per crate.

Complex quotients: never form them for a *power* result - `|x/y|^2 = |x|^2 / |y|^2`,
two real moduli and one real reciprocal. For an *amplitude* result, `x/y = x*conj(y) /
|y|^2`, and two of those share the reciprocal exactly as above. `Complex::div` costs a
real divide plus ~8 ops; do not call it twice when once will do.

## 4. Substitute out the intermediates that only get multiplied back

The biggest wins are not instruction-level; they come from noticing that a derived
quantity exists only to be undone. Ask, for each intermediate: *is this ever used except
as a factor in something I could compute directly?*

- **Relative index -> admittance.** Fresnel wants `n_t*cos_t`, never `cos_t`. And
  `(n_t cos_t)^2 = n_t^2 - n_i^2 sin^2 = (n_t^2 - n_i^2) + (n_i cos_i)^2` - no
  `eta = n_i/n_t` division, no `cos_t`, no multiply back. Terms that need `n_i cos_t`
  instead get scaled through by `n_t` (`n_t^2 cos_i -+ n_i b`); the factor cancels in
  every ratio.
- **Scale a whole formula by a power of the divisor.** The textbook conductor
  reflectance is written in `eta = n_t/n_i, eta_k = k_t/n_i` (two divides before the
  physics starts). Multiply every term through by `n_i^2`: `T0 = n_i^2 t0`,
  `A = n_i a`, and so on. Same expressions, zero divides.
- **`Re()` of a complex product is half a product.** If a formula wants only
  `Re(x*y)`, do not build the complex product. And `Re(x*conj(y))` differs from
  `Re(x*y)` by one sign on one partial product - both fall out of one helper for one
  extra fused op.

Failure this prevents: micro-optimizing the arithmetic of a formula whose *shape* has
three divides that a one-line algebraic identity removes.

## 5. Small spellings (each is one op cheaper, or one constant fewer)

| Instead of | Write | Why |
|---|---|---|
| `x * V::TWO` | `x + x` | add, no constant register; and on `Complex` a real add pair, not a complex product |
| `x / V::TWO` | `x * V::FRAC_1_2` | multiply, not divide |
| `V::ONE - x * x` | `x.nmul_adde(x, V::ONE)` | one fused op; ungated (product not shared) |
| `4 * p` | `p * (V::TWO + V::TWO)` | the sum folds to a constant; `(p+p)+(p+p)` is two dependent adds |
| `-a * b + c` and `-a * b - c` | `nmul_adde` / `nmul_sube` | check the sign table, do not guess |
| `sqrt(x).max(0)` clamps | keep only the ones a proof cannot remove | analytic non-negativity survives correctly-rounded ops (rounding is monotone); it does *not* survive reassociation. One `max` against a NaN pixel is usually worth it |
| `mask.select(a, -a)` | `a.neg_c(!mask)` | masked negation, no separate negate + blend |
| three compares + and/andnot/or | `absorbing.select(im, re).cmp_gt(0)` | blend the *deciding value*, then one compare |

## 6. Prove it. Three checks, in this order.

**(a) Reference tests against the naive spelling.** Write the textbook formula the slow
obvious way (plain `f64`, or `Complex`'s own `*` `/` `.sqrt()`), and compare *every*
public method to it. Choose inputs that make every term nonzero:

- complex `cos_i`, not just real - with real `cos_i` a conjugate is a no-op and a
  wrong sign in `Re(n conj(cos))` is invisible;
- `n_i != 1` - with `n_i = 1` a scaling by `n_i^2` is invisible;
- both sides of any tolerance / branch cut you touched.

**(b) Execute both arms of every `if const`.** Stamp the test suite once per backend
with a `macro_rules!` taking the vector type. Scalar `Vector<f64>` has no true FMA in the
default configuration (its flag needs thermite's `std` feature *and* an FMA target
feature); `thermite::backend::x86_v3::f64x4` has one. A suite that runs on one backend
only has tested half the kernel.

**(c) Count the instructions.** A tiny `bin/*_asm_probe` crate with
`#[unsafe(no_mangle)] #[target_feature(enable = "avx2,fma")]` wrappers over each kernel,
then `cargo rustc -p <probe> --release -- --emit asm -C llvm-args=-x86-asm-syntax=intel`
and histogram `vdivps` / `vsqrtps` / `vfmadd*` per function. The op-count table from
step 0 is the prediction; this is the measurement. If they disagree, the code is not what
you think it is.

MSVC-target asm has no `.Lfunc_end` markers - bound each function by the *line range*
between `^probe_...:` labels, or the histogram bleeds into the next function.

## 7. What "correct but slow" looks like in the asm

- Two `vdivps` in a function that returns a pair: the reciprocal was not shared.
- `vmulps` immediately followed by `vaddps` on the same registers, on an FMA target:
  an `_e` fold was missed - or the compiler could not prove the reassociation (it will
  not; FP is non-associative without fast-math).
- A `vmulps` whose result feeds two FMAs on non-FMA hardware: a gated fold was not
  gated.
- `call` inside the body: a helper without `#[inline(always)]`, or a cold path
  (`Complex::sqrt`'s rescale) - the latter is fine, the former is rule zero.
- Any `vdivps` whose numerator is a broadcast constant *and* whose result is only ever
  multiplied by another constant: fold them.
