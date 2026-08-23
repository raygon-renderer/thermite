Thermite: Melt your CPU
=======================

Thermite is a Rust library for portable SIMD programming, providing abstractions
over various SIMD instruction sets as well as generic implementations,
enabling high-performance vectorized computations across different hardware
architectures with ease.

A key aspect of Thermite's design is that it allows you
to write a function once (with generics) and evaluate it on any supported backend.

```toml
[dependencies]
thermite = "0.3"
```

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;

// The logistic sigmoid, written once against trait bounds: no ISA, lane count,
// or element type is named. It compiles for every backend, and the very same
// function also accepts `Dual` (autodiff) or `Compensated` (double-double).
#[thermite::dispatch(V)]
fn sigmoid<V: FloatVector + TranscendentalMath>(x: V) -> V {
    V::ONE / (V::ONE + (-x).exp())
}

let mut data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01 - 5.0).collect();

let data = data.as_mut_slice(); // dispatch_dyn! takes its parameters by value
thermite::dispatch_dyn!(for<S> |data: &mut [f32]| {
    // `f32xN` is the widest native f32 vector of the ISA selected at runtime,
    // so this is one AVX2 kernel on a modern x86 CPU and one SSE2 kernel on an
    // old one, from a single source.
    let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<f32xN>();

    for v in chunks {
        *v = sigmoid(*v);
    }

    // The ragged ends run through the same `sigmoid` on the 1-lane scalar backend.
    for x in head.iter_mut().chain(tail) {
        *x = sigmoid(Vector::<f32>::splat(*x)).extract::<0>();
    }
});

assert!((data[500] - 0.5).abs() < 1e-6); // sigmoid(0) == 0.5
```

## The Rules

Read these before writing anything. They are not style advice, and breaking
rule 1 in particular is worse than not using SIMD at all.

0. **Do not touch the `Register` layer** unless you know exactly what you're
   doing. Stick with the `*Vector` traits.
1. **`#[thermite::dispatch]` and `#[inline(always)]` are MANDATORY**, or else
   your code will be abysmally slow.
2. **Prefer trait bounds over concrete vector types.** Commit to a fixed lane
   count when the data structure is genuinely defined by it, not out of
   convenience.
3. **Thermite isn't magic.** You must consider what you're doing before you
   expect it to be fast.
4. **Look for any built-in methods before re-implementing things yourself.**
   All of Thermite's standard library is highly optimized.
5. **Avoid scalar work whenever possible.**
6. **Doing nothing is better than doing something clever.**

### Why rule 1 is not a suggestion

`rustc` will not inline a `#[target_feature]` function into a caller that lacks
those features, and every intrinsic is one. A generic SIMD body with no
`#[thermite::dispatch]` above it compiles featureless: every operation becomes an
out-of-line call, with no scheduling or register allocation across them. It still
compiles. It is still correct. However, it is catastrophically slow.

Two attributes avoid it. Put `#[thermite::dispatch]` on the outermost SIMD entry
point, and `#[inline(always)]` on the helpers called beneath it. The guide covers
the details, and this is the first thing to check when a kernel underperforms.

## What's in it

The trait ladder runs `GenericVector` to `NumericVector` to `FloatVector`, with
the math traits on top. A kernel is bounded on the weakest one it needs.

Beyond arithmetic and the usual transcendentals, the vector API covers stream
compaction (`compress` / `expand`), lane prefix scans, duplicate-lane conflict
detection, gather and scatter, the full N-by-N cast matrix across element types,
interleaving at group granularity, and packed fp16 / bf16 / fp8 storage.

The math library is **policy-configurable**. Every function has a `_p` form
taking a `P: Policy`, so the same `exp` call can be tuned from
`UltraPerformance` to `Reference` at the call site instead of through a second
set of function names.

## Backends

| Backend | ISA | Status |
|---|---|---|
| `scalar` | none, 1 lane | Always available, and what the ragged ends of a loop run on |
| `x86_v1` | SSE2 | Complete |
| `x86_v2` | SSE4.2 | Complete, adds `pshufb` shuffles over v1 |
| `x86_v3` | AVX2 + FMA | Complete, the widest working backend |
| `x86_v4` | AVX-512 | **Not implemented.** The hardware still works, it runs the AVX2 backend |
| `neon` | AArch64 AdvSIMD | Complete. Mandatory on the architecture, so there is no feature to enable |
| `wasm` | SIMD128 | Complete, opt-in behind the `wasm` feature |
| `spirv` | SPIR-V | Incomplete, and **a compile error in released versions** |
Backend selection is a runtime decision made once by `dispatch_dyn!`, not a
compile-time target flag, so a single binary runs the best kernel on whatever CPU
it lands on. On an AVX-512 machine `InstructionSet::get()` reports `X86V4` and
dispatch maps that rung onto `x86_v3`.

Which backends are compiled in the first place is covered under
[Reaching the other backends](#reaching-the-other-backends) below.

## Feature flags

Default: `document_registers`, `bitvec`, `avx2-f16c`, `avx2-pclmul`.

### Numerics, and what they change about results

These change the numbers coming out, not just the speed.

| Feature | Effect |
|---|---|
| `strict_ieee754` | Follow the spec exactly where SIMD instructions intentionally do not. Implies `preserve_denormals`, and turns off the approximate `rcp` / `rsqrt` estimates on backends that have them. Significantly slower |
| `preserve_denormals` | Every default math policy keeps denormal inputs instead of flushing them. Slow on denormal-heavy data, and required for strict IEEE-754 |
| `ignore_denormals` | The opposite assumption: the hardware already flushes, so skip the checks. Also drops the subnormal machinery from the emulated FMA on non-FMA backends (~20% faster `mul_add` there, and normal-range results stay bit-identical to hardware FMA while subnormal-scale ones become faithful). Ignored when `preserve_denormals` is also on |
| `algebraic-scalar` | The 1-lane scalar backend uses LLVM's `algebraic_*` float ops instead of strict `+ - * /`, which is what lets a loop written against the scalar backend autovectorize at all. Costs exact cancellation, so `thermite-compensated` is incompatible and rejects the combination at compile time. Needs `nightly` for now |

Combinations resolve rather than conflict, which matters because Cargo unifies
features across an entire dependency graph and an unrelated crate can switch one
on. `strict_ieee754` wins over `algebraic-scalar`, and `preserve_denormals` wins
over `ignore_denormals`. The **resolved** answer is readable at compile time from
the `thermite::features` module (`STRICT_IEEE754`, `ALGEBRAIC_SCALAR`, ...), so
downstream code can gate on it or reject it with a `const` assertion instead of
re-deriving the precedence itself.

### Codegen

| Feature | Effect |
|---|---|
| `avx2-f16c` (default) | Assume `f16c` whenever AVX2 is present, true of every AVX2 CPU. Adds it to the `target_feature` set the dispatch macros emit, so half-precision conversion needs no separate runtime check. No effect off x86 |
| `avx2-pclmul` (default) | Same bargain for `pclmulqdq`, which enables the CLMUL 2D-Morton fast path on `u64` lanes. No effect off x86 |
| `disable_dispatch` | Replace every static ISA dispatch with a plain `#[inline(always)]` signature. Only correct when the target ISA is pinned at compile time, and if a function then fails to inline it loses its target features and gets dramatically slower, which is the entire problem dispatch exists to solve |
| `nightly` | Unlocks nightly-only paths (smarter const splat, wasm64 SIMD, SPIR-V) and then requires a nightly compiler |

### API surface

`std` (off by default, this crate is `no_std`) enables formatted panic messages
and `num-traits/std`. `bitvec` (default) adds the `bitvec` mask integration.
`partial-ord` adds `PartialOrd` for `Vector`, which only holds when every lane
shares the order. `document_registers` (default) is documentation only.

## Reaching the other backends

The x86 backends need no feature at all. `x86_v1`, `x86_v2` and `x86_v3` are
compiled unconditionally on `x86` and `x86_64`, and `dispatch_dyn!` picks
between them at runtime, so one binary carries all three.

**AArch64 NEON** is the same story with no flag: AdvSIMD is mandatory in the
architecture, so the backend is gated on `target_arch = "aarch64"` and is always
compiled there. There is deliberately no `neon` feature, and passing one is an
error. 32-bit ARM is not supported, since its NEON intrinsics are still unstable
and ARMv7 NEON has no f64 lanes.

**WebAssembly** is opt-in, because SIMD128 is a proposal an engine may not have
enabled:

```toml
thermite = { version = "0.3", default-features = false, features = ["wasm"] }
```

`wasm32` works on stable. `wasm64` additionally needs `nightly`.

**AVX-512** has no registers yet. The `avx512-tier1` through `avx512-tier3`
features select which tier the in-progress x86-v4 backend compiles to, exactly
one per build, resolved to the highest requested. Tier 1 is the Skylake-SP set
(F+CD+BW+DQ+VL), the floor, and there is no Knights Landing tier. Today they
compile the module skeleton only, change no codegen, and AVX-512 hardware runs
the AVX2 backend in the meantime.

**SPIR-V** is unfinished, and enabling the `spirv` feature on a released version
is a hard compile error. Work on it continues and the code stays in the repository.
If you want to build it regardless, take a git dependency and set
`RUSTFLAGS='--cfg thermite_unstable_spirv'`.

## Companion crates

The same generic functions run unmodified on the composite types, which is the
point of writing them against traits in the first place.

* `thermite-special` for the error function, gamma, elliptic integrals and
  activations
* `thermite-dual` for forward-mode automatic differentiation
* `thermite-compensated` for double-double precision
* `thermite-complex` for complex numbers
* `thermite-sort`, `thermite-sdf`, `thermite-geometry` for sorting, signed
  distance fields, and SoA geometric primitives

## More

There is a full guide covering the trait hierarchy, slice iteration, masks, the
policy system and performance work. It ships with the crate and renders as the
`thermite::guide` module on docs.rs.

Stable Rust, MSRV 1.95, edition 2024. `no_std` by default.

### If you write a blanket impl over `S: Simd`

Proving `S: Simd3A` or `S: Simd3` for a *generic* `S` walks a deep enough chain
of `ReducedRegister` goals to exhaust the default `recursion_limit` of 128.
Thermite sets `#![recursion_limit = "256"]` for itself, but the limit is
per-crate and does not propagate, so add the same attribute to your crate root:

```rust,ignore
#![recursion_limit = "256"]
```

You only need it for a blanket `impl<S: Simd> YourTrait for S` that names the
3-lane associated types. Writing `S: Simd3A` as an ordinary bound, or using a
concrete backend, stays well under the limit and needs nothing.

## License

MIT or Apache-2.0, at your option.
