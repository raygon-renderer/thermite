---
name: thermite
description: Use Thermite, the generic ISA-portable Rust SIMD library, and its companion crates (special functions, autodiff Dual, Compensated double-double, SDF, geometry, FFI). Load when writing, reviewing, or debugging code that uses thermite/thermite-* crates, GenericVector/FloatVector/NumericVector trait bounds, the policy-based math library, masks, dispatch, slice iteration, or composite vector types. Triggers: "write a SIMD kernel", "make this generic over vector types", "use thermite", "FloatVector bound", "autodiff with Dual", "compensated arithmetic", "SDF", thermite build/test errors.
---

# Thermite

Thermite is a pure-Rust SIMD abstraction library. You write a function **once**,
generic over a trait hierarchy (`GenericVector -> NumericVector -> FloatVector`,
plus math traits), and it compiles to optimal code for every backend (SSE2,
SSE4.2, AVX2, WASM SIMD128, scalar, and an experimental SPIR-V GPU path) at every
lane count. The same generic function also runs on **composite** vector types --
`Dual` (forward-mode autodiff), `Compensated` (double-double precision), and
others -- with zero changes, because those types implement the same traits.

This skill is for developers who **add Thermite as a dependency** to their own
crate. It documents the core `thermite` crate plus the companion crates you can
pull in alongside it. File paths in the sub-files (e.g. `crates/thermite/src/...`)
point into the Thermite source so you can read the authoritative implementation;
they are references, not files in your own project.

> This skill is the source of truth for *using* Thermite, second only to the code
> itself. Everything in it was verified against the current source. If any prose
> ever disagrees with the code, trust the code (and fix the skill).

## Add Thermite to your project

Thermite is **not on crates.io yet** (core is `0.2.0-beta.0`; the companion
crates are `publish = false`). Depend on it via git:

```toml
[dependencies]
# Core SIMD library. Default features are fine for most users; see the table below.
thermite = { git = "https://github.com/raygon-renderer/thermite" }

# Optional companion crates (each is a separate dependency; all currently git-only):
thermite-special    = { git = "https://github.com/raygon-renderer/thermite" } # erf, gamma, activations, Lambert W, ...
thermite-dual       = { git = "https://github.com/raygon-renderer/thermite" } # forward-mode autodiff (Dual)
thermite-compensated = { git = "https://github.com/raygon-renderer/thermite" } # double-double precision (Compensated)
thermite-sdf        = { git = "https://github.com/raygon-renderer/thermite" } # signed-distance fields
thermite-geometry   = { git = "https://github.com/raygon-renderer/thermite" } # SoA Vector/Point/Ray/Bounds/Matrix
```

Only add the companions you use. `thermite-dual`, `thermite-compensated`, and
`thermite-sdf` already depend on `thermite` transitively, so you don't need to
list `thermite` separately unless you name its types directly (you usually do).

**Toolchain: core Thermite works on stable Rust** (MSRV **1.95**, edition 2024).
You do *not* need nightly for normal use. The optional `nightly` feature unlocks
nightly-only paths (smarter const splat via `core_intrinsics`/`const_eval_select`,
wasm64 SIMD, the experimental SPIR-V backend) and, if enabled, *requires* a
nightly compiler (it `compile_error!`s otherwise). The FFI crate is the one place
that needs nightly -- see [references/ffi.md](references/ffi.md).

### Feature flags (`thermite`)

Default features are `document_registers`, `bitvec`, `avx2-f16c`. The crate is
`no_std` by default. User-relevant flags:

| Feature | Default | What it does |
|---|---|---|
| `std` | off | Enables `std` (e.g. fmt in panicking paths, `num-traits/std`). Turn on if you build for a std target and want those; leave off for `no_std`. |
| `bitvec` | **on** | Enables `bitvec` integration for masks (`dep:bitvec`). |
| `avx2-f16c` | **on** | Assume `f16c` whenever AVX2 is present (true for all AVX2 CPUs); enables half-precision conversion intrinsics in the AVX2 backend with no extra runtime check. x86 only. |
| `partial-ord` | off | Implements `PartialOrd` for `Vector` (only meaningful when all lanes share the order). |
| `strict_ieee754` | off | Follow IEEE-754 as closely as possible: implies `preserve_denormals` + `disable_fast_fma`. Significantly slower; only if you need spec-exact denormals/NaN/min-max behavior. |
| `preserve_denormals` | off | Keep denormals (required for strict IEEE-754); slower on denormal-heavy data. |
| `ignore_denormals` | off | Flush/ignore denormals by default; slight overhead to fix them, usually cheaper than operating on them. |
| `disable_fast_fma` | off | Use exact (but very slow) scalar `libm::fma` instead of the accurate emulated fast FMA on non-FMA backends. Only for bitwise-identical FMA results. |
| `disable_dispatch` | off | Replace runtime ISA dispatch with `#[inline(always)]`. Bloats code and can be very slow unless everything inlines -- advanced use only. |
| `nightly` | off | Unlocks nightly-only code paths (see toolchain note); requires a nightly compiler. |
| `wasm` | off | Use the wasm32/wasm64 SIMD128 backend. |
| `neon` | off | ARM NEON backend. |
| `avx512-tier1..4` | off | Opt into AVX-512 backend tiers (tier4 is cutting-edge; each tier implies the lower ones). |
| `spirv` | off | Experimental SPIR-V GPU backend (implies `nightly`). |

A common setup for a std application that wants strict numerics:
`thermite = { git = "...", features = ["std"] }`. To go fully `no_std`, add
`default-features = false` and re-enable just what you need.

## A complete example

One generic function, run on a scalar lane, native-width SIMD, `Dual` (autodiff),
and `Compensated` (double-double) -- the whole thesis on one screen. Drop this into
a binary that depends on `thermite`, `thermite-dual`, and `thermite-compensated`:

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_dual::AutoDiff;
use thermite_compensated::Compensated;

// Written ONCE, over trait bounds -- no backend, lane count, or element type named.
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V { (-(x * x)).exp() }

type V = Vector<f64>;

fn main() {
    // 1. Scalar (1-lane) backend -- no dispatch needed.
    let s = gaussian(V::splat(0.5)).extract::<0>();           // 0.7788007830714049

    // 2. Native-width SIMD -- runtime-dispatched to the best ISA for this CPU.
    let simd = thermite::dispatch_dyn!(for<S> || -> f32 {
        gaussian(f32xN::splat(0.5)).extract::<0>()
    });                                                        // ~0.77880079 (f32)

    // 3. Dual<V, N>: the SAME fn now returns value AND gradient.
    //    d/dx e^{-x^2} = -2x e^{-x^2}.
    let r = gaussian.ad([V::splat(0.5)]);
    let value = r.re.extract::<0>();                           // 0.7788007830714049
    let dydx  = r.dual[0].extract::<0>();                      // -0.7788007830714049

    // 4. Compensated<V>: the SAME fn carried in double-double precision.
    let hi = gaussian(Compensated::<V>::new(V::splat(0.5)))
        .value().extract::<0>();                               // 0.7788007830714049

    println!("{s}  {simd}  {value}  {dydx}  {hi}");
}
```

`gaussian` is never edited between cases -- the *type you instantiate it with*
decides whether you get plain SIMD, a derivative, or extra precision. (The values
in the comments were checked against this exact code.) See
[references/generic-programming.md](references/generic-programming.md) for the
mechanics and [references/composite-types.md](references/composite-types.md) for
how `Dual`/`Compensated` plug in.

## The one thing to internalize

Constrain on **traits**, never on a concrete backend or width:

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath; // math trait names need an explicit import

// Works on ANY backend, ANY lane count, ANY float element -- and on Dual,
// Compensated, etc. The caller picks the concrete type.
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V {
    (-(x * x)).exp()
}
```

`gaussian(Vector::<f64>::splat(0.5))` runs scalar; the *same* function under
`dispatch_dyn!` runs at AVX2 width; `gaussian.ad([V::splat(0.5)])` returns the
value **and** its derivative. See
[references/generic-programming.md](references/generic-programming.md) -- the
single most important file here.

## Quick reference

```rust
use thermite::prelude::*;                       // Vector, Mask, the *Vector traits, SimdSlice
use thermite::math::{TranscendentalMath, RealMath, CoreMath, SpatialMath}; // to NAME math traits in bounds

a + b   a - b   a * b   a / b   -a              // NumericVector / SignedVector ops
a & b   a | b   a ^ b   !a                      // BitwiseVector
a << n  a >> n                                  // BitshiftVector
let m = a.cmp_lt(b); m.select(a, b)             // PartialOrdVector -> Mask, branchless select
v.sqrt()  v.exp()  v.sin()  a.mul_adde(b, c)    // FloatVector + math + estimating-FMA
v.sqrt_c(mask)  a.add_c(mask, b)                // masked variants: mask is the FIRST arg
```

You compile and test all of this as part of your own crate -- a normal
`cargo build` / `cargo test` once Thermite is in your `[dependencies]`. There is
no separate build step for the library.

## Sub-files: load what the task needs

Core usage:

- [references/generic-programming.md](references/generic-programming.md) -- **read first.** Writing functions once over `*Vector` bounds; associated types (`V::Element`, `V::Mask`, `V::LANES`, ...); how the same code runs on scalar, SIMD, and composite types; common bound recipes; pitfalls.
- [references/trait-hierarchy.md](references/trait-hierarchy.md) -- the full `GenericVector ... FloatVector` tree, every trait, its supertraits and associated types, and which methods live where.
- [references/vector-api.md](references/vector-api.md) -- method-level reference for `GenericVector`/`NumericVector`/`FloatVector` (construction, lanes, memory, gather/scatter, cast, interleave, reductions, FMA), plus the `_c`/`_m`/`_z` masked-variant system.
- [references/masks.md](references/masks.md) -- `Mask<R>`, `GenericMask` (`all`/`any`/`select`/`bitmask`), mask casting, and `zz`/`nz` helpers.
- [references/math.md](references/math.md) -- the math trait families (`CoreMath`, `TranscendentalMath`, `SpatialMath`, `RealMath`, `FloatMath`), the `_p::<P>()` policy system and presets, `ScalarMath` for bare `f32`/`f64`, `FloatConsts`, FMA semantics, and the algorithms module.
- [references/slices-and-dispatch.md](references/slices-and-dispatch.md) -- slice iteration (`SimdSlice`: aligned / try-aligned / unaligned / streaming), alignment, and the `#[dispatch]` / `dispatch_dyn!` ISA macros.

Composite & companion crates:

- [references/composite-types.md](references/composite-types.md) -- **the compose story.** How `Dual<V, N>` (autodiff) and `Compensated<V>` (double-double) implement the vector traits by delegating to an inner `V`, so generic code differentiates / error-tracks for free. Nesting (`Dual<Compensated<V>>`) too.
- [references/special.md](references/special.md) -- `thermite-special`: `erf`, gamma, activations (`gelu`/`swish`), Lambert W, elliptic integrals, etc., all generic over any float vector.
- [references/sdf.md](references/sdf.md) -- `thermite-sdf`: signed-distance primitives, boolean/smooth combinators, transforms, fractals; the `SDF`/`GradientSdf`/`BoundedSdf` traits.
- [references/geometry.md](references/geometry.md) -- `thermite-geometry`: SoA `Vector`/`Point`/`Ray`/`Bounds`/`Matrix` built on the linalg vector traits.
- [references/ffi.md](references/ffi.md) -- `thermite-ffi`: the C ABI (`thermite_init`, batch math functions, the `Thermite` vtable), header generation, and the `release-ffi` profile.

Cross-cutting:

- [references/performance.md](references/performance.md) -- FMA-variant choice, `if const` capability gating, ILP/critical-path tricks, cancellation-avoidance, `scale` for SPIR-V, and `target_feature` codegen gotchas.
- [references/architecture.md](references/architecture.md) -- **worth reading even as a user**: the Element -> Register -> Vector layering, backends and ISA levels, the `Simd` type hierarchy, and how composites slot in. It explains *why* the API is shaped this way and demystifies the trickier type errors. Only the lowest level (per-ISA register impls) is contributor-only.

## Gotchas that bite (full list in the sub-files)

- **Bare `f32`/`f64` do not implement `FloatVector`.** Wrap a scalar in
  `Vector::<f64>::splat(x)` (or `Vector(x)`), or use `ScalarMath`'s `scalar_`-prefixed
  methods (`x.scalar_sin()`). See [references/math.md](references/math.md).
- **Masked variants take the mask FIRST**: `a.add_c(mask, b)`, `v.sqrt_c(mask)`,
  `a.add_m(src, mask, b)` (merge: `src` then `mask`). Some prose docs show the old
  `add_c(b, mask)` order -- that is wrong.
- **Math trait names are imported anonymously by the prelude** (`as _`): their
  methods work, but to write `<V: TranscendentalMath>` you must
  `use thermite::math::TranscendentalMath;` explicitly.
- **Prefer `mul_adde` (estimating FMA), not `mul_add`.** The non-`e` form drags in
  `libm::fma` (very slow) on non-FMA backends. See [references/performance.md](references/performance.md).
- **`thermite-dual` / `thermite-compensated` / `thermite-special` are `publish = false`**
  (pre-release). `Compensated`'s `FloatVector` masked variants are partly `todo!()`.
  Treat them as solid-but-WIP.
