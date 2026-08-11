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

**Modifying Thermite's own source** (adding an op / math fn / backend, fixing
internals)? Read [references/development.md](references/development.md) **first** --
it's the contributor map (crate layout, proc-macro toolbox, a full worked example
of adding a primitive across all backends, register-vs-math-fn checklists, the
build/test/verify loop). The user-facing references (architecture, trait-hierarchy,
math, performance) remain load-bearing when changing the code behind them.

## Add to a project

Not on crates.io (core `0.2.0-beta.0`; companions are `publish = false`). Git deps:

All git-only, same URL `https://github.com/raygon-renderer/thermite`. Add
`thermite` plus only the companions you use (see the sub-file list for what each
provides): `thermite-special`, `-dual`, `-compensated`, `-sort`, `-sdf`,
`-geometry`. `-dual`/`-compensated`/`-sdf`/`-sort` pull in `thermite`
transitively, but list it anyway when you name its types (you usually do).

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
| `disable_fast_fma` | off | Exact but very slow scalar `libm::fma` instead of accurate emulated FMA on non-FMA backends. |
| `disable_dispatch` | off | Replace runtime dispatch with `#[inline(always)]`. Bloats/slows unless all inlines. Advanced. |
| `nightly` | off | Nightly-only paths (requires nightly compiler). |
| `wasm` | off | wasm32/wasm64 SIMD128 backend. |
| `avx512-tier1..4` | off | AVX-512 tiers (tier4 cutting-edge; each implies lower). |
| `spirv` | off | Experimental SPIR-V GPU backend (implies `nightly`). |

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

## Sub-files: load what the task needs

Core usage:
- [generic-programming.md](references/generic-programming.md) -- **read first.** Functions over `*Vector` bounds; assoc types (`V::Element`/`V::Mask`/`V::LANES`); running on scalar/SIMD/composite; bound recipes; pitfalls.
- [trait-hierarchy.md](references/trait-hierarchy.md) -- full `GenericVector..FloatVector` tree, supertraits, assoc types, which methods live where.
- [vector-api.md](references/vector-api.md) -- method reference for the vector traits (construction, lanes, memory, gather/scatter, cast, interleave, compress/expand, prefix scans, conflict detection + `group_by_value`, reductions, FMA, packed fp16/bf16/fp8 storage via `PackedFloatVector`) + the `_c`/`_m`/`_z` masked system.
- [masks.md](references/masks.md) -- `Mask<R>`, `GenericMask` (`all`/`any`/`select`/`bitmask`), casting, `zz`/`nz`.
- [math.md](references/math.md) -- math trait families (`CoreMath`/`TranscendentalMath`/`SpatialMath`/`RealMath`/`FloatMath`), the `_p::<P>()` policy system + presets, `ScalarMath` for bare `f32`/`f64`, `FloatConsts`, FMA semantics, algorithms module.
- [slices-and-dispatch.md](references/slices-and-dispatch.md) -- `SimdSlice` iteration (aligned/try-aligned/unaligned/streaming), alignment, `#[dispatch]`/`dispatch_dyn!`.

Composite & companions:
- [composite-types.md](references/composite-types.md) -- **the compose story.** `Dual<V,N>` and `Compensated<V>` delegate the vector traits to inner `V` so generic code differentiates/error-tracks free. Nesting (`Dual<Compensated<V>>`).
- [special.md](references/special.md) -- `thermite-special`: erf, gamma, activations (gelu/swish), Lambert W, elliptic integrals.
- [sort.md](references/sort.md) -- `thermite-sort`: slice quicksort (`sort`/`sort_by`), key-value (`sort_kv_by`), cached-key object sort; partition/network building blocks.
- [sdf.md](references/sdf.md) -- `thermite-sdf`: primitives, boolean/smooth combinators, transforms, fractals; `SDF`/`GradientSdf`/`BoundedSdf`.
- [geometry.md](references/geometry.md) -- `thermite-geometry`: SoA `Vector`/`Point`/`Ray`/`Bounds`/`Matrix`.
- [ffi.md](references/ffi.md) -- `thermite-ffi`: C ABI, header gen, `release-ffi` profile.

Cross-cutting:
- [performance.md](references/performance.md) -- FMA-variant choice, `if const` capability gating, ILP/critical-path, cancellation-avoidance, `scale` for SPIR-V, `target_feature` codegen gotchas.
- [architecture.md](references/architecture.md) -- **worth reading as a user.** Element -> Register -> Vector layering, backends/ISA levels, the `Simd` type hierarchy, how composites slot in; demystifies type errors.
- [development.md](references/development.md) -- **contributor-only.** Modifying Thermite's source: crate map, proc-macro toolbox, worked example (Morton interleave across all backends), register-vs-math-fn checklists, build/test/verify loop.

## Gotchas (full list in sub-files)

- **Missing `#[thermite::dispatch]` / `#[inline(always)]` is the #1 silent perf bug** -- no compile error, no test failure, but the kernel degenerates to a `call` per intrinsic. See Rule zero above; check it first when a kernel underperforms.
- **Bare `f32`/`f64` don't impl `FloatVector`.** Wrap: `x.as_vector()` (`Element` method, in the prelude) / `Vector::<f64>::splat(x)` / `Vector(x)`, or use `ScalarMath` `scalar_`-prefixed methods (`x.scalar_sin()`).
- **Masked variants take the mask FIRST**: `a.add_c(mask, b)`, `v.sqrt_c(mask)`, `a.add_m(src, mask, b)` (merge: `src` then `mask`). Old `add_c(b, mask)` order is wrong.
- **Math trait names are `use`d anonymously by the prelude** (`as _`): methods work, but to write `<V: TranscendentalMath>` you must `use thermite::math::TranscendentalMath;`.
- **Prefer `mul_adde` (estimating FMA) over `mul_add` for speed.** On non-FMA backends `mul_add` lowers to vectorized emulated FMA (single-rounding-accurate, still SIMD) by default -- only becomes slow scalar `libm::fma` under `disable_fast_fma`/`strict_ieee754`. So `mul_add` is a valid accuracy choice.
- **`>>` is LOGICAL even on signed vectors.** Use `srai`/`sra`/`srav` for sign-filling shifts.
- **`bitandnot` differs by layer**: `a.bitandnot(b)` on `Vector`/`Mask` = `a & !b`, but the register layer `R::bitandnot(lhs, rhs)` = `!lhs & rhs` (x86 convention) -- the vector impls swap operands when delegating.
- **`V::load` is an ALIGNED load.** Loading a table from a plain `Box`/`Vec` faults nondeterministically; use `load_unaligned` or an aligned container.
- **`thermite-sort`: instantiate it at a NATIVE register width** (`i32x8` on AVX2, `i32x4` on SSE4.2/NEON), never an `ArrayRegister` composite - a two-chunk composite sorts ~30% slower than the width it is built from, because `compress` and the merge swizzles do not scale across sub-registers. See [references/sort.md](references/sort.md).
- **`-dual`/`-compensated`/`-special` are `publish = false` (pre-release).** Vector-trait surfaces are complete but some special fns `todo!()`-panic: `bessel_j` beyond f32 `J_0`; the gamma family (`tgamma`/`lgamma`/`digamma`/`beta`) + `bessel_j` on `Dual`/`Compensated`. Grep `todo!` before relying on a special function.
