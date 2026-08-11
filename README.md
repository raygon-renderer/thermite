Thermite SIMD: Melt Your CPU
============================

[![CI](https://github.com/raygon-renderer/thermite/actions/workflows/ci.yml/badge.svg)](https://github.com/raygon-renderer/thermite/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/novacrazy/7270c68e5927fa2a1ef2de9f4286010b/raw/thermite-coverage.json)](https://github.com/raygon-renderer/thermite/actions/workflows/ci.yml)
[![Thermite FFI Build](https://github.com/raygon-renderer/thermite/actions/workflows/ffi_artifacts.yaml/badge.svg)](https://github.com/raygon-renderer/thermite/actions/workflows/ffi_artifacts.yaml)
[![Docs](https://github.com/raygon-renderer/thermite/actions/workflows/rustdoc.yml/badge.svg)](https://github.com/raygon-renderer/thermite/actions/workflows/rustdoc.yml)

Thermite is a portable SIMD library for Rust with a singular goal: you write a
numeric kernel **once**, generic over a small trait hierarchy, and that one
function compiles to high-performance code for every instruction set at every vector
width. Unchanged, it also computes its own derivatives (`Dual`), runs in
double-double precision (`Compensated`), or evaluates over the complex plane
(`Complex`).

It ships a full, policy-tunable vectorized math library, from the ordinary
transcendentals through special functions, works on stable Rust, and is `no_std`
by default.

This repository is the Cargo workspace for Thermite and its companion crates.

## Example

The function below is written once, against traits. Nothing in it names an
instruction set, a lane count, or an element type. The _type you instantiate it
with_ decides whether you get plain SIMD, a gradient, or 106 bits of
significand.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;
use thermite_dual::AutoDiff;
use thermite_compensated::Compensated;

// `#[dispatch]` is what makes this fast. It isn't optional. See below.
#[thermite::dispatch(V)]
fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V {
    (-(x * x)).exp()
}

let xs: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001 - 2.0).collect();
let mut ys = vec![0.0f32; xs.len()];
let mut dydx = vec![0.0f32; xs.len()];

// `dispatch_dyn!` takes its parameters by value, so hand it slices.
let (xs, ys, dydx) = (xs.as_slice(), ys.as_mut_slice(), dydx.as_mut_slice());

thermite::dispatch_dyn!(|xs: &[f32], ys: &mut [f32], dydx: &mut [f32]| {
    // `f32xN` is the widest native f32 vector of whichever ISA this CPU turned
    // out to have, chosen once at runtime: 8 lanes on AVX2, 4 on SSE2 or NEON.
    let n = f32xN::lanes();

    for ((x, y), d) in xs.chunks_exact(n)
        .zip(ys.chunks_exact_mut(n))
        .zip(dydx.chunks_exact_mut(n))
    {
        let x = f32xN::from_slice(x);

        // Plain SIMD. `n` values through the kernel at once.
        gaussian(x).copy_to_slice(y);

        // The same function, differentiated. Value and gradient in one pass,
        // still `n` at a time.
        gaussian.ad([x]).dual[0].copy_to_slice(d);

        // The same function again, evaluated to a ~106-bit significand.
        let precise = gaussian(Compensated::new(x));
        let _ = (precise.value(), precise.error());
    }
});
```

Three meanings, one function body, no edits between them.

## History & Motivation

Thermite was originally conceived while working on the Raygon renderer, when it
was decided we needed a state of the art high-performance SIMD library focused
on SoA algorithms. Libraries at the time (and even now) were either too
low-level, lacking in features, or not optimized for modern hardware. The goal
was to let developers write efficient SIMD code without having to worry about
the underlying hardware details.

However, my first prototype was flawed. Too many leaky abstractions, and back in
2020 the Rust language and compiler were themselves too limited to support the
abstractions this needs. In 2025 I felt renewed interest in the project, and
with mature const generics, a better trait solver, edition 2024, and much wider
stable intrinsics coverage, I was able to redesign it from the ground up. Rust
is powerful enough to express all of this safely on stable, but it took me ten
years of writing it (and one failed prototype) to figure out how.

Melting your CPU is just a fun tagline, obviously. The real goal is to get as
much useful work out of one as it can physically do, and every design decision
follows from that. Estimating-FMA defaults, so no port waits on a scalar
fallback. Static dispatch, so no call boundary silently drops you to baseline
code. Precision policies, so you never pay for accuracy you didn't ask for.

Thermite is the library I always wanted for **single-machine HPC** and could
never find. The numeric computing world splits into cluster-scale frameworks on
one side and thin SIMD wrappers on the other, with a swamp of software bloat in
between. If you have one machine and a numerically heavy problem, you deserve
tooling that's simple to use, deep enough not to hit a wall in week two, and
wastes nothing. The receipts for that: `no_std` by default, a small dependency
tree, a stable toolchain, and no build scripts, no codegen step, and no FFI in
the core crate.

That comes down to three requirements, and I don't know of a single tool that
covers all of them together:

1. **Width- and ISA-generic programming on stable Rust.** Trait-generic kernels
   plus runtime dispatch that survives function-call boundaries, without
   picking a fixed width by hand or waiting on nightly.
2. **A real vectorized math library.** `exp`, `log`, and the trig family with
   per-call-site precision policies, plus erf, gamma, Lambert W, and elliptic
   integrals in `thermite-special`.
3. **Composability beyond hardware vectors.** The trait hierarchy is an
   interface, not a wrapper over one machine's types, so autodiff duals,
   compensated arithmetic, and complex numbers implement the same traits and
   reuse every kernel, math library included.

## Design

1. The `GenericVector -> NumericVector -> FloatVector` hierarchy defines a
   strong set of constraints and an extensive set of behaviors for any
   vector-like type. A kernel is bounded on the weakest trait it actually
   needs, and names nothing else.
2. The `Vector` type implements that hierarchy for every backend at every lane
   count, so `f32x8` on AVX2 and `f64x2` on NEON present the same interface and
   the same semantics.
3. The `*Math` traits, on top of `FloatVector`, provide a wide variety of math
   functions, each with adjustable behavior through a compile-time policy
   system.
4. Companion crates plug into the same hierarchy, either as extra behavior
   (`thermite-special`) or as new types that satisfy it (`thermite-dual`,
   `thermite-compensated`, `thermite-complex`).

### `#[thermite::dispatch]`

Rust has a problem with `#[target_feature(enable = "...")]`: if a function is
ever _not_ inlined, it "forgets" its target features. That's severe for a
program built at baseline SSE2 that wants to support modern hardware through
dynamic dispatch. Either everything inlines always, which is massive code bloat,
or the whole thing breaks down into deoptimization.

`#[thermite::dispatch]` rewrites functions to propagate target features
automatically and _statically_, which makes it effectively zero-cost after
dead-code elimination. It's used for all the `*Math` traits, so they stay
optimized and lightweight. It goes on a function, an `impl` block, a trait, or a
whole `mod`.

This is the highest-impact rule in Thermite code and it's invisible to the type
system. A generic SIMD body with no `#[dispatch]` above it compiles featureless:
every intrinsic becomes an out-of-line call, with nothing optimized across them.
It compiles, it's correct, it's catastrophically slow. Put
`#[thermite::dispatch]` on the outermost SIMD entry point and `#[inline(always)]`
on the helpers beneath it, and check that first when a kernel underperforms.

## Backends

| Backend | ISA | Status |
|---|---|---|
| `scalar` | none, 1 lane | Always available, and what the ragged ends of a loop run on |
| `x86_v1` | SSE2 | Complete |
| `x86_v2` | SSE4.2 | Complete, adds `pshufb` shuffles over v1 |
| `x86_v3` | AVX2 + FMA | Complete, the widest working backend and the primary optimization target |
| `x86_v4` | AVX-512 | Not implemented. The hardware still works, it runs the AVX2 backend |
| `neon` | AArch64 AdvSIMD | Complete. Mandatory on the architecture, so there's no feature to enable |
| `wasm` | SIMD128 | Complete, opt-in behind the `wasm` feature |
| `spirv` | SPIR-V | Incomplete and experimental, a compile error in released versions |

All three x86 backends are compiled unconditionally on `x86` and `x86_64`, so a
single binary carries all of them and `dispatch_dyn!` picks between them at
runtime. 32-bit ARM isn't supported, since its NEON intrinsics are still
unstable and ARMv7 NEON has no `f64` lanes. RISC-V V hasn't been started.

The `avx512-tier1` through `avx512-tier4` features are reserved names that
select nothing today. They exist so the tier names stay stable when the backend
lands.

## The crates

| Crate | What it is |
|---|---|
| [`thermite`](crates/thermite) | The core: vector traits, backends, dispatch, and the math library |
| [`thermite-macros`](crates/thermite-macros) | The proc macros behind `#[dispatch]`, re-exported by `thermite` |
| [`thermite-special`](crates/thermite-special) | Special functions: erf, gamma, elliptic integrals, Lambert W, activations |
| [`thermite-dual`](crates/thermite-dual) | Forward-mode automatic differentiation over multidual numbers |
| [`thermite-compensated`](crates/thermite-compensated) | Double-double compensated arithmetic |
| [`thermite-complex`](crates/thermite-complex) | SIMD complex numbers |

### `thermite`

The core. Generic code is written against `GenericVector` to `NumericVector` to
`FloatVector` plus the math traits, and nothing in a kernel names a vector width
or an instruction set.

Beyond arithmetic and the transcendentals, the vector API covers stream
compaction (`compress` and `expand`), lane prefix scans, duplicate-lane conflict
detection, gather and scatter, the full N-by-N cast matrix across element types,
interleaving at group granularity, Morton codes, branchfree dividers, saturating
and wrapping integer ops, and packed fp16, bf16, and fp8 storage formats that
hold half or a quarter of the bytes while the arithmetic runs at full precision.
`SimdSlice` handles aligned, unaligned, and streaming iteration over ordinary
slices, so you don't hand-write a prologue and epilogue per loop.

The math library is policy-configurable. Every function has a `_p` form taking a
`P: Policy`, so the same `exp` call gets tuned from `UltraPerformance` to
`Reference` at the call site rather than through a second set of function names.
Masks are first class, and every maskable operation has `_c`, `_m`, and `_z`
forms, so the branchless version is the one that's already written.

`no_std` by default, MSRV 1.95, edition 2024. The full tour is in
[the guide](crates/thermite/GUIDE.md), which also renders as the
`thermite::guide` module on docs.rs.

### `thermite-macros`

The procedural macros. **Don't depend on this crate directly.** Depend on
`thermite`, which re-exports the user-facing macros and pins this crate to an
exact version, so the generated code always matches its internals.

Three of them are user-facing. `#[dispatch]` propagates `#[target_feature]`
statically across call boundaries. `dispatch_dyn!` is the runtime boundary that
picks the best available ISA and monomorphizes a closure or call for it.
`#[derive(HasIsa)]` covers types generic over a `Simd` ISA parameter, forwarding
the `ISA` constant and the `Native` backend type from that parameter. The rest
are internal codegen helpers that build Thermite's own backends and trait
hierarchy.

### `thermite-special`

The layer above the core math library. `thermite` covers the transcendentals you
reach for constantly, and this covers the rest, all vectorized rather than
looped over lanes:

* **Error function family.** `erf`, `erfc`, and the inverses `erfinv` and
  `probit`.
* **Gamma family.** `tgamma`, `lgamma`, `lgamma_r`, `digamma`, and `beta`.
* **Orthogonal polynomials.** Legendre including the associated form, Jacobi,
  Hermite at a const or per-lane runtime order, and a Chebyshev series evaluated
  by Clenshaw recurrence for all four kinds.
* **Elliptic integrals.** Every Legendre form, complete and incomplete, over all
  five Carlson symmetric primitives (`R_F`, `R_C`, `R_D`, `R_J`, `R_G`). The
  form is chosen by a request struct, so `EllintPiInc { n, phi, k }` carries
  exactly its own arguments, and the wrong shape is a compile error rather than
  a silently ignored parameter.
* **Lambert W.** Both real branches, `W_0` and `W_{-1}`, from a single call. The
  two Halley iterations interleave, so the second branch is close to free on a
  wide machine.
* **The exponential integral** `E_n(x)` at integer order.
* **Activations,** each with a `_d` variant returning value and derivative
  together: `gelu`, `swish`, `softplus`, `logistic_sigmoid`, and the exp-free
  `algebraic_sigmoid` and `algebraic_swish`.

Same policy system as the core, so `x.erf_p::<UltraPerformance>()` and
`x.erf_p::<Precision>()` are both available at the call site. The traits are
auto-implemented for every float vector, and `ScalarSpecialMath` gives the same
set under `scalar_`-prefixed names for a bare `f32` or `f64`.

Not everything is finished. `bessel_j` is limited to f32 `J_0`, and the gamma
family and `bessel_j` are unimplemented on `Dual` and `Compensated`. Grep for
`todo!` before relying on a specific function.

### `thermite-dual`

Forward-mode automatic differentiation. `Dual<V, N>` carries a primal value plus
`N` first-order derivative components, propagated by the chain rule, so
evaluating a function on a `Dual` returns the value and its gradient in a single
pass.

`Dual<V, 0>` tracks no derivatives, `Dual<V, 1>` is a classic dual number, and
`Dual<V, N>` is the gradient of an N-variable function. The inner `V` is any
`FloatVector`, in which case each lane is an independent dual number, or an
`f32`/`f64` at the element level. Derivative components live in a separate
`[V; N]`, so the layout is struct-of-arrays.

This is a first-order multidual. It tracks gradients, not Hessians. `trigamma`
is deliberately unimplemented, because the Gamma-derivative family isn't closed
under differentiation (psi_1' is psi_2, whose derivative is psi_3, and so on),
so closing it properly needs a general `polygamma(n)`.

### `thermite-compensated`

`Compensated<V>` stores a value and an error term, together representing a
number to roughly twice the precision of `V` alone. Every operation is built
from error-free transformations (`two_sum`, `two_diff`, `two_prod`, Veltkamp
splitting) that recover the rounding error an ordinary float discards, and feed
it into the next operation.

`Compensated<f64>` is a double-double with a ~106-bit significand against
`f64`'s 53. `Compensated<Vector<..>>` is `LANES` independent double-doubles in
parallel. The precision isn't free: an addition is roughly eleven float
operations, and a multiplication is two with hardware FMA or around seventeen
without, where it falls back to Dekker splitting. On hardware with no `f64` at
all, `Compensated<f32>` recovers most of double precision out of
single-precision units.

It's incompatible with `thermite/algebraic-scalar`, and refuses to compile
alongside it via a `const` assertion rather than leaving that to the reader.
Reassociable float arithmetic is exactly what destroys error terms: LLVM may
fold `(a - (s - v)) + (b - v)` to zero, and then every error term silently
vanishes. Results stay plausible and lose all of the extra precision the crate
exists to provide, so the combination is refused.

The Gamma family (`tgamma`, `lgamma`, `lgamma_r`, `digamma`, `trigamma`, `beta`)
is still `todo!()` and will panic if called. Those need genuine double-double
algorithms, a Lanczos or Stirling evaluation carried in compensated arithmetic,
not delegation to the inner `V`.

### `thermite-complex`

`Complex<V>` stores a real and an imaginary part, each an inner `V`. With `V` a
`FloatVector` each lane is an independent complex number in struct-of-arrays
layout, and with `V` an `f32`/`f64` it's a complex scalar, which is the
`Element` of the vector form.

`CoreMath`, `TranscendentalMath`, and `SpatialMath` with their `_p::<P>()`
policy forms all come from the same blanket impls that serve the real types.
Operations whose result or argument is _real_ (`norm`, `arg`, polar form, real
powers and bases) have no place in those families and live on `ComplexMath` and
`ComplexVector` instead.

However, C is neither ordered nor signed, and the vector traits require both.
Ordering (`cmp_lt`, `min`/`max`, `arg_minmax`) is lexicographic by `(re, im)`,
which is a tiebreak rule and not a claim about magnitudes. `abs` and `signum`
are modulus-based, preserving `abs(z) * signum(z) == z`. Sign-bit ops and
rounding are componentwise. `RealMath` is deliberately not implemented, because
`atan2`, `wrap_angle`, and `step` are defined over an ordered field.

Optional features add complex special functions including the Faddeeva function
`w(z)`, and let a `Dual` be the storage, so `Complex<Dual<V, N>>` is valid.

## Status

Core `thermite` is `0.2.0` and the API is settling. The five other published
crates move in lockstep on the same version. The vector-trait surfaces are
complete across all of them, but a handful of special functions still `todo!()`
rather than compute, mostly in the gamma family and on the composite types, so
grep before depending on one.

MSRV is 1.95 on stable, edition 2024. Nightly is only needed for opt-in paths:
SPIR-V, wasm64, `algebraic-scalar`, and the const-splat fast path.

## License

MIT or Apache-2.0, at your option.
