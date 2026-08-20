---
name: thermite
description: Thermite, the generic ISA-portable Rust SIMD library, and its thermite-* companions (special, dual/autodiff, compensated/double-double, sort, sdf, geometry, ffi). Load when writing/reviewing/debugging code using thermite/thermite-* crates - GenericVector/NumericVector/FloatVector bounds, the policy math library, masks, dispatch, slice iteration, sorting, or composite vector types (Dual/Compensated). ALWAYS load before modifying Thermite's OWN source (register op, backend, math kernel, polyfill, macro, trait) and read references/development.md first. Triggers - "write a SIMD kernel", "generic over vector types", "use thermite", "FloatVector bound", "autodiff with Dual", "compensated arithmetic", "SDF", "SIMD sort"/"sort by key", thermite build/test errors, edits under crates/thermite*/src.
---

# Thermite

Pure-Rust SIMD abstraction. Write a function **once** over a trait hierarchy
(`GenericVector -> NumericVector -> FloatVector` + math traits); it compiles to
optimal code for every backend (SSE2, SSE4.2, AVX2, WASM SIMD128, NEON, scalar,
experimental AVX-512 + SPIR-V) at every lane count. The *same* function also runs
on **composite** types -- `Dual` (autodiff), `Compensated` (double-double) -- with
no changes, because they implement the same traits.

This skill is for crates that **depend on** Thermite. Sub-file paths like
`crates/thermite/src/...` point into the Thermite source (authoritative impl), not
your project. If prose ever disagrees with the code, trust the code and fix the skill.

## Step 1: `Read` the reference for your task

Loading this skill loaded **only this file**; the rules live in `references/` and nothing
loads them for you. Match the request below and make your next tool call a `Read` of that
file (both, if two match) before writing code. Every sub-file exists because a
SKILL.md-only attempt compiled, passed tests, and was wrong or slow anyway.

| Request involves | Read |
|---|---|
| any edit under `crates/thermite/src` (op, backend, math fn, polyfill, macro, trait) | [development.md](references/development.md) (routes onward) |
| a new fn/kernel over `*Vector` bounds | [generic-programming.md](references/generic-programming.md), then [performance.md](references/performance.md) sec 0 |
| "optimize" / FMA / divisions / op count / asm of an *existing* kernel | [optimization-pass.md](references/optimization-pass.md) |
| accuracy, ulp, policy tiers `_p::<P>()`, a new math fn | [math.md](references/math.md), [performance.md](references/performance.md) sec 7 |
| `Dual`, `Compensated`, `Complex`, or code generic over a type that might be one | [composite-types.md](references/composite-types.md) |
| masks, `_c`/`_m`/`_z`, select vs masked ops | [masks.md](references/masks.md), [vector-api.md](references/vector-api.md) |
| slices, alignment, `#[dispatch]`, `dispatch_dyn!` | [slices-and-dispatch.md](references/slices-and-dispatch.md) |
| a type error you do not understand | [architecture.md](references/architecture.md), [trait-hierarchy.md](references/trait-hierarchy.md) |
| the full method list for a trait | [vector-api.md](references/vector-api.md) |
| `thermite-special` / `-sort` / `-sdf` / `-geometry` / `-ffi` | [special.md](references/special.md) / [sort.md](references/sort.md) / [sdf.md](references/sdf.md) / [geometry.md](references/geometry.md) / [ffi.md](references/ffi.md) |

You may not know which to read at first. Read one the moment any of these happen:
you are about to guess at a method name or signature; a compile error names a
trait you did not write; you are reaching for `#[dispatch]`, a masked variant, or
a policy tier; you have written 30+ lines without opening one.

## Add to a project

On crates.io at `0.2.0`. Add `thermite = "0.2.0"` (its proc-macro crate
`thermite-macros` comes along automatically), plus only the companions you use -
`thermite-special`, `-dual`, `-complex`, `-compensated`, `-sort`, `-sdf`,
`-geometry`. See the sub-file list for what each provides. Every companion pulls
in `thermite` transitively, but list it anyway when you name its types (you
usually do), and keep all of them on the same version.

**Toolchain: stable Rust** (MSRV **1.95**, edition 2024). The `nightly` feature
unlocks nightly-only paths (smarter const splat, wasm64 SIMD, SPIR-V) and then
*requires* nightly (`compile_error!`s otherwise). Only FFI mandates nightly -- see
[references/ffi.md](references/ffi.md). Crate is `no_std` by default.

### Feature flags (`thermite`)

Defaults: `document_registers`, `bitvec`, `avx2-f16c`, `avx2-pclmul`.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enable std (fmt in panics, `num-traits/std`). |
| `bitvec` | **on** | `bitvec` mask integration. |
| `avx2-f16c` | **on** | Assume `f16c` w/ AVX2 (all AVX2 CPUs have it); enables f16 conv, no runtime check. x86. |
| `avx2-pclmul` | **on** | Assume `pclmulqdq` w/ AVX2; CLMUL 2D-Morton fast path on u64 lanes. x86. |
| `partial-ord` | off | `PartialOrd` for `Vector` (only when all lanes share order). |
| `strict_ieee754` | off | Spec-exact denormals/NaN/min-max; implies `preserve_denormals`+`disable_fast_fma`. Much slower. |
| `preserve_denormals` | off | Keep denormals (required for strict IEEE-754); slower on denormal-heavy data. |
| `ignore_denormals` | off | Flush denormals by default. |
| `disable_fast_fma` | off | Exact but very slow scalar `libm::fma` instead of the emulated FMA on non-FMA backends. The emulation is close, not bit-identical (~1 in 173k for f64). |
| `algebraic-scalar` | off | Scalar (1-lane) backend uses LLVM `algebraic_*` ops so loops written against it can autovectorize. Costs exact cancellation: `thermite-compensated` rejects it at compile time. `strict_ieee754` overrides it. Needs `nightly` until 1.98. |
| `disable_dispatch` | off | Replace runtime dispatch with `#[inline(always)]`. Bloats/slows unless all inlines. Advanced. |
| `nightly` | off | Nightly-only paths (requires nightly compiler). |
| `wasm` | off | wasm32/wasm64 SIMD128 backend. |
| `avx512-tier1..4` | off | **RESERVED, no-op.** The x86-v4 backend has no registers; these select nothing. AVX-512 CPUs run the x86-v3 (AVX2) backend, which dispatch maps `X86V4` onto. |
| `spirv` | off | **HARD COMPILE ERROR in released versions.** Incomplete: no `impl Simd` (so no vector type aliases, not a dispatch target), f32/i32/u32 only, nothing off `target_arch = "spirv"`. Needs a git dep plus `RUSTFLAGS='--cfg thermite_unstable_spirv'`. |

**No `neon` feature**: NEON/AdvSIMD is mandatory in AArch64, so the backend is
gated on `target_arch = "aarch64"` and is **always compiled** there -- nothing to
opt into, and `--features neon` is now an error. (Contrast `wasm`, which stays
opt-in because SIMD128 is a proposal the engine may not enable.)

Strict-numerics std app: `features = ["std"]`. Fully `no_std`:
`default-features = false` + re-enable what you need.

## The thesis, in one example

One generic fn run 4 ways -- scalar, native SIMD, `Dual`, `Compensated`. The fn is
never edited; the *type you instantiate* decides plain SIMD vs derivative vs precision.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath; // math trait names need explicit import to NAME in bounds
use thermite_dual::AutoDiff;
use thermite_compensated::Compensated;

// Written ONCE over trait bounds - no backend, lane count, or element named.
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

fn main() {
    let s = gaussian(0.5_f64.as_vector()).extract::<0>();      // 0.7788007830714049 (scalar)
                                                               // as_vector: bare f64 -> 1-lane V

    let simd = thermite::dispatch_dyn!(for<S> || -> f32 {      // runtime-dispatched to best ISA
        gaussian(f32xN::splat(0.5)).extract::<0>()
    });                                                        // ~0.77880079 (f32)

    let r = gaussian.ad([V::splat(0.5)]);                      // value AND gradient
    let value = r.re.extract::<0>();                           // 0.7788007830714049
    let dydx  = r.dual[0].extract::<0>();                      // -0.7788... (= -2x e^{-x^2})

    let hi = gaussian(Compensated::<V>::new(V::splat(0.5)))    // double-double
        .value().extract::<0>();                               // 0.7788007830714049
}
```

Constrain on **traits**, never a concrete backend/width -- that is the whole point.
Layer discipline: user code targets the `*Vector` traits ONLY. The `*Register`
layer (`R::op(storage)`) is backend-implementation machinery -- never call it
from user or generic code (its semantics can even differ from the vector layer,
e.g. `bitandnot` operand order). Concrete `Vector<R>` types are a last resort
too: legitimate mainly as the scalar 1-lane seeds (`Vector<f32>`/`Vector<f64>`)
and at `dispatch_dyn!` boundaries; anything reusable stays generic over bounds.
See [references/generic-programming.md](references/generic-programming.md) (read
first) and [references/composite-types.md](references/composite-types.md).

## Rule zero: `#[thermite::dispatch]` + `#[inline(always)]`

**The single highest-impact rule in Thermite code, and invisible to the type
system.** rustc **will not inline a `#[target_feature]` fn into a caller lacking
those features**, and every `core::arch` intrinsic is one. A generic-over-`S`
body with no `#[dispatch]` above it (and no `#[dispatch]` ancestor inlining it)
is compiled featureless: every intrinsic stays out-of-line, a `call` per single
instruction, no scheduling or regalloc across ops. Compiles, correct,
catastrophically slow. Two attributes fix it:

- **`#[thermite::dispatch(S)]`** (or `(Self)` / `(TypeName)`) on every SIMD entry
  point -- fn, `impl`, `trait` or `mod`. Emits a `#[target_feature]` trampoline
  per backend plus a const-folded ISA match; the only way the body gets per-ISA
  codegen at all.
- **`#[inline(always)]`** on every helper called from inside a dispatched body.
  Features propagate into a callee **only if it is inlined**; a non-inlined
  helper is featureless and hits the same soup one level down. Plain `#[inline]`
  is a hint the optimizer declines in exactly the big bodies that matter.

The pairing is cheap, not bloat: the trampoline itself carries the features, so
the dispatched fn need not inline (one out-of-line copy per backend) while the
interior inlines aggressively. Exception: don't `#[dispatch]` one-line leaf
helpers -- it only blocks inlining; keep them `#[inline(always)]`.

```rust
#[inline(always)]                                   // interior: must inline to keep features
fn step<V: FloatVector>(v: V) -> V { v.mul_adde(v, v) }

#[thermite::dispatch(S)]                            // boundary: per-backend target_feature
pub fn kernel<S: FloatSimd<f32>>(data: &mut [f32]) {
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<Vector<S::fxN>>();
    for v in chunks { *v = step(*v); }
}

let _ = thermite::dispatch_dyn!(kernel(&mut data)); // runtime ISA selection
```

Details: [references/performance.md](references/performance.md) sec 0,
[references/slices-and-dispatch.md](references/slices-and-dispatch.md).

## Quick reference

```rust
use thermite::prelude::*;                       // Vector, Mask, *Vector traits, SimdSlice
use thermite::math::{TranscendentalMath, RealMath, CoreMath, SpatialMath}; // to NAME math traits in bounds

a + b  a - b  a * b  a / b  -a                  // NumericVector / SignedVector
a & b  a | b  a ^ b  !a                         // BitwiseVector
a << n  a >> n                                  // BitshiftVector (>> is LOGICAL; use srai/sra/srav for arithmetic)
let m = a.cmp_lt(b); m.select(a, b)             // PartialOrdVector -> Mask, branchless select
v.sqrt()  v.exp()  v.sin()  a.mul_adde(b, c)    // FloatVector + math + estimating-FMA
v.sqrt_c(mask)  a.add_c(mask, b)                // masked variants: mask is the FIRST arg
v.compress_z(m) / v.expand_m(src, m)            // stream compaction and its exact inverse
v.prefix_sum()  v.count_conflicts()             // inclusive lane scan; duplicate-lane ranks
v.group_by_value(valid)                         // divergent packet -> uniform sub-packets
v.total_order()                                 // FloatVectorWithBits: NaN-safe integer sort keys

thermite_sort::sort::<i32x8>(&mut keys)         // thermite-sort: see references/sort.md
```

Build/test as part of your own crate -- normal `cargo build`/`cargo test`, no
separate library step.

## Gotchas (full list in sub-files)

- **Missing `#[thermite::dispatch]` / `#[inline(always)]` is the #1 silent perf bug** -- no compile error, no test failure, but the kernel degenerates to a `call` per intrinsic. See Rule zero above; check it first when a kernel underperforms.
- **Bare `f32`/`f64` don't impl `FloatVector`.** Wrap: `x.as_vector()` (`Element` method, in the prelude) / `Vector::<f64>::splat(x)` / `Vector(x)`, or use `ScalarMath` `scalar_`-prefixed methods (`x.scalar_sin()`).
- **Masked variants take the mask FIRST**: `a.add_c(mask, b)`, `v.sqrt_c(mask)`, `a.add_m(src, mask, b)` (merge: `src` then `mask`). Old `add_c(b, mask)` order is wrong.
- **Math trait names are `use`d anonymously by the prelude** (`as _`): methods work, but to write `<V: TranscendentalMath>` you must `use thermite::math::TranscendentalMath;`.
- **Prefer `mul_adde` (estimating FMA) over `mul_add` for speed.** On non-FMA backends `mul_add` lowers to vectorized emulated FMA (FMA-quality, still SIMD, but *not* bit-identical to a true FMA -- ~1 in 173k differ for f64, worst relative error 2.0e-15) by default -- only becomes slow scalar `libm::fma` under `disable_fast_fma`/`strict_ieee754`. So `mul_add` is a valid accuracy choice, just not an exact one.
- **`>>` is LOGICAL even on signed vectors.** Use `srai`/`sra`/`srav` for sign-filling shifts.
- **`bitandnot` differs by layer**: `a.bitandnot(b)` on `Vector`/`Mask` = `a & !b`, but the register layer `R::bitandnot(lhs, rhs)` = `!lhs & rhs` (x86 convention) -- the vector impls swap operands when delegating.
- **`V::load` is an ALIGNED load.** Loading a table from a plain `Box`/`Vec` faults nondeterministically; use `load_unaligned` or an aligned container.
- **Masked access does not license going out of bounds.** `store_masked` (and any masked-load idiom) is UB in Rust if the full-width access extends past the allocation, even on ISAs that guarantee the masked-off lanes never fault. A ragged tail therefore cannot be a masked access at `len - LANES + k`. Fill a zeroed vector's lanes directly instead, or use the slice iterators, which never form the out-of-bounds access.
- **`thermite-sort`: instantiate it at a NATIVE register width** (`i32x8` on AVX2, `i32x4` on SSE4.2/NEON), never an `ArrayRegister` composite - a two-chunk composite sorts ~30% slower than the width it is built from, because `compress` and the merge swizzles do not scale across sub-registers. See [references/sort.md](references/sort.md).
- **Two special functions still `todo!()`-panic, both on composites**: `Dual::trigamma` (its derivative needs the tetragamma `psi_2`), and six functions on the doubly-nested `Complex<Compensated<..>>` (`tgamma`/`lgamma`/`digamma`/`trigamma`/`lambert_w`/Faddeeva). Everything on `Vector<R>`, `Dual`, `Complex` and `Compensated` alone is implemented for both f32 and f64. Grep `todo!` before relying on a special function.
