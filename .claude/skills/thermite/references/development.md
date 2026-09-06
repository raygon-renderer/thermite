# Developing Thermite itself (contributor guide)

This file inverts the rest of the skill: it is for *changing Thermite's own
source* -- adding an operation, math function, or backend path -- working at
every layer the user never touches. A user-facing method does not exist until
threaded through **all** of these, or it won't compile for some instantiation:

```
register trait  (register/mod.rs)            <- declare the primitive
   |  per-backend impls (backend/<isa>/...)  <- implement it on real hardware
   |  ArrayRegister / ReducedRegister        <- emulated widths (free via macro)
vector trait    (vector/mod.rs)              <- expose it on Vector<R>
   |  vector impl (vector/vector.rs)         <- delegate to the register (free via macro)
math layer      (math/...)                   <- ONLY if precision-tunable
composites      (thermite-dual/-compensated) <- ONLY if bespoke semantics needed
tests           (crates/thermite/tests/)     <- differential vs the scalar oracle
```

"Free via macro" = a proc-macro writes the boilerplate; you still write a stub.
Nothing is free *across backends*: a new primitive needs a real implementation
for **scalar, x86_v1 (SSE2), x86_v2 (SSE4.2), x86_v3 (AVX2), wasm, neon**
(optionally spirv) -- the dispatcher can select any of them. The neon backend
(macro-stamped registers, `backend/neon/macros.rs`) compiles on aarch64 and
**only** there, with no feature to enable -- NEON is mandatory in AArch64, so it
is gated on `target_arch` alone. Verify it via `just pi-build` + `just qemu-test`
(section 7).

Read [architecture.md](architecture.md) first (Element -> Register -> Vector).
Also load-bearing: [trait-hierarchy.md](trait-hierarchy.md) (what each trait
owns), [math.md](math.md) (policy surface), [performance.md](performance.md)
(kernel style), [masks.md](masks.md) (`_c`/`_m`/`_z` semantics you'll
*implement*). The code is the authority; line numbers drift -- grep the symbol.

---

## 0. The overriding goal: maximum performance

**Thermite exists to be the fastest portable SIMD library possible.**
Performance is the top priority and justifies effort that would be
over-engineering elsewhere:

- **Per-register, per-backend specialization is encouraged.** If one concrete
  register on one ISA can do an op faster with a bespoke intrinsic sequence,
  write it -- even if `u32x8`-on-AVX2, `u32x4`-on-SSE4.2, and `u32x4`-on-SSE2
  each get different hand-tuned bodies. Bending over backwards for a concrete
  type's fast path is the expected default, not a smell. The generic/portable
  path is the *correctness floor*; per-backend overrides are where wins live.
- **Any avenue to speed is on the table**: a new polyfill, a feature-gated CPU
  sub-extension (`avx2-pclmul`, `avx2-f16c`, an AVX-512 tier), a cheaper
  algebraic identity, a shorter dependency chain, a capability-gated `if const`
  fork, a tighter intrinsic, an extra width specialization.
- **Correctness and portability are constraints, not competitors.** Every
  specialization must pass the differential suite against the scalar oracle
  (section 7) and every backend must compile. Within those bounds, push as hard
  as the hardware allows.
- **Measure, don't assume**: confirm wins with `cargo bench` and check emitted
  asm. The `target_feature` inlining traps ([performance.md](performance.md)
  sec 11) routinely make "obviously faster" code slower.

The techniques themselves (FMA choice, ILP, capability gating, cancellation
avoidance) are in [performance.md](performance.md); this is the mandate to apply
them aggressively, down to individual registers.

---

## 1. The repository

Workspace root: `members = ["bin/*", "crates/*", "tests/*"]`, edition 2024,
MSRV 1.95, `resolver = "3"`, all crates share one version (`0.2.0`) via
`[workspace.package]`. Profiles: `release` = opt-level=3, lto=true,
codegen-units=1; `bench` same with lto="fat"; `release-ffi` adds strip,
panic="abort".

| Path | What |
|---|---|
| `crates/thermite` | Core crate (sections 2-6: `register/`, `vector/`, `math/`, `backend/`, `element/`). |
| `crates/thermite-macros` | All proc macros: `dispatch`, `dispatch_dyn`, `register_trait`, `vector_trait`, `vector_impl`, `array_impl`, `reduced_impl`, `inline_always`, `double_pump_impl`, `derive(HasIsa)`. |
| `crates/thermite-special` | erf/gamma/activations/elliptic; same `_p::<P>()` + `Specialized*` pattern as core math. |
| `crates/thermite-dual` | `Dual<V,N>` autodiff; implements `Specialized*Math` by chain rule. |
| `crates/thermite-compensated` | `Compensated<V>` double-double via error-free transforms. WIP (`todo!()` in places). |
| `crates/thermite-sort` | Vectorized quicksort + key-value/cached-key sorts on the core traits. See [sort.md](sort.md); `SORT_HANDOFF.md` at the repo root is its authoritative design/measurement record. Core owns the *primitives* it builds on (`NumericRegister::sort_by`, `bitonic_clean_by`, `thermite::sort`'s order markers and index math) - the dividing line is whether a backend could plausibly want to override it. |
| `crates/thermite-geometry`, `-sdf`, `-complex`, `-blas`, `-bignum`, `-rng` | Companions on the core traits; some mid-rewrite. |
| `crates/thermite-ffi` | C ABI `cdylib`, nightly-only. See [ffi.md](ffi.md). |
| `crates/testing` | Internal differential-test helpers. |
| `tests/wasm-runner` | wasmtime+WASI host running compiled libtest binaries (via `CARGO_TARGET_WASM32_WASIP1_RUNNER`). |
| `bin/docgen` | Register-coverage docs (incl. SVG). |
| `bin/remez` | Minimax (Remez) polynomial fitting -- source of transcendental coefficients. |
| `bin/spirv_testing` + `bin/spirv_builder` + `bin/spirv_runner` | GPU path: kernels, rust-gpu compile to `.spv`, wgpu execution vs CPU reference. |

**Toolchain.** Builds on **stable**. `rust-toolchain.toml` pins a rust-gpu
nightly only for the (inactive) spirv backend; the `justfile` overrides to
stable. Nightly needed only for wasm tests, miri, branch coverage, spirv, FFI.

### Where each layer lives (core crate)

| Path | Purpose |
|---|---|
| `register/mod.rs` | `*Register` hierarchy: `CoreRegister` (233) -> `BitwiseRegister` (291) -> `Register` (488) -> `NumericRegister` (1623) -> `FloatRegister` (2084); also `BitshiftRegister` (1325, `Element: IntegerElement`), `IntegerRegister`, `MaskRegister` (364), swizzle/concat/extend/blend traits. Methods are `fn(Storage<Self>, ...) -> Storage<Self>` -- no `&self`, no operators. Lane-wise defaults borrow via `Register::as_slice`/`as_mut_slice` (runtime-length slices; no array-typed borrow) and loop `0..Self::lanes()`. |
| `register/well_formed.rs`, `register/linalg.rs` | well-formedness bounds; linalg register ops. |
| `register/array.rs` | `ArrayRegister<R,N>` -- `[R; N]` emulates a wider width. |
| `register/reduced.rs` | `ReducedRegister<R,N>` -- masks upper lanes, emulates narrower. |
| `vector/mod.rs` | `*Vector` traits (`GenericVector` 512, `BitwiseVector` 1092, `BitshiftVector` 1182, ...), each `#[thermite_macros::vector_trait]`. |
| `vector/vector.rs` | `#[repr(transparent)] pub struct Vector<R>(pub Storage<R>)` + delegating trait impls + operators. |
| `vector/{splat,num,ops,streaming,unaligned}.rs` | `const_splat!`/`const_new!`, num-traits glue, masked-op traits, slice iterators. |
| `backend/<isa>/registers/<type>.rs` | One file per concrete register (`u32x8.rs`, `f32x4.rs`, ...) with its trait impls. |
| `backend/<isa>/polyfills/{bits,cmp,casts,divider,math}.rs` | Per-backend software fills. |
| `backend/generic/polyfills/` | Portable polyfills for all backends (`bits.rs`, `sort.rs`, `casts.rs`, `divider.rs`, `math.rs`). |
| `backend/prefetch.rs` | One per-arch software-prefetch impl (x86 `_mm_prefetch`, aarch64 `prfm`) behind `NativeIsa::prefetch`; re-exported into `arch::` for both. |
| `backend/<isa>/mod.rs` | Defines `pub mod arch { pub use super::polyfills::*; pub use crate::backend::x86::<tier>::*; }` -- the `arch::` namespace register files call into. |
| `backend/x86.rs` | x86 intrinsic re-export ladder shared across tiers. |
| `math/mod.rs` | `decl_math!` definition + invocations (public math surface). |
| `math/specialized/{ps,pd,generic}.rs` | Kernels: `ps.rs` = f32, `pd.rs` = f64, `generic.rs` = element-agnostic helpers. |
| `math/{policy,consts,scalar}.rs`, `math/algorithms/` | `Policy`, `FloatConsts`, scalar surface, generic numerics (`newtons_method`, `sum_f`, `reduce_in_place`). |
| `element/mod.rs`, `element/float/{mod,spec}.rs` | `Element`/`IntegerElement` (138)/`FloatElement` (49)/`FloatElementWithBits` (135), per-element bit constants. |

---

## 2. The macro toolbox

Never write masked variants, dispatch trampolines, or per-width boilerplate by
hand:

| Macro | Applied to | Generates |
|---|---|---|
| `#[thermite_macros::register_trait]` | `*Register` trait def | For each `#[conditional]`/`#[masked]` method, the `_c`/`_m`/`_z` siblings (mask-first) as provided methods; `#[inline(always)]` on any default-bodied method. |
| `#[conditional]` / `#[masked]` (markers) | method in the above | `_c` + `_m` + `_z` per the rule in 2a. Mask-ineligible return types are skipped. |
| `#[thermite_macros::vector_trait]` | `*Vector` trait def | Same masked-variant expansion at the vector layer. |
| `#[thermite_macros::vector_impl]` | `impl ... for Vector<R>` | Fills empty bodies with `Vector(R::method(self.0, ...))` delegation + masked siblings. |
| `#[thermite_macros::array_impl]` | `impl ... for ArrayRegister<R,N>` | Element-wise delegation to inner `R`. You write empty stubs. |
| `#[thermite_macros::reduced_impl]` | `impl ... for ReducedRegister<R,N>` | Delegates to the wider `R`, re-masks dead upper lanes. Empty stubs. |
| `#[thermite_macros::inline_always]` | any impl block | `#[inline(always)]` on every method (universal tag on backend impls). |
| `#[thermite_macros::double_pump_impl]` | `DoublePumpRegister<R>` impls | Legacy; mostly superseded by `ArrayRegister`. |
| `#[thermite::dispatch(S)]` / `(Self)` | fn / impl / mod over `S: HasIsa` | Per-backend `#[target_feature]` trampolines + `match <S as HasIsa>::ISA` that folds at monomorphization. `#[skip_dispatch]` opts a method out. |
| `thermite::dispatch_dyn!(for<S> ...)` | expression | Runtime `InstructionSet::get()` selection; rewrites bare `f32xN`/`f32x4`/... to `Vector<S::...>`. Signature must be ISA-agnostic. Call form: `dispatch_dyn!(func(args))` / `dispatch_dyn!(for<S> expr)` dispatches a `#[dispatch]` fn directly (match only, no trampolines). |
| `decl_math! { ... }` (`math/mod.rs`) | math signatures | `*MathWithPolicy` (`_p::<P>()`), default-policy `*Math`, `scalar_*` surface on f32/f64, blanket impl routing to `Specialized*Math<E>`. |
| `const_splat!` / `const_new!` (`vector/splat.rs`) | const expr | Compile-time splat / per-lane const vector. **Use for bitwise/coefficient constants** (`const_splat!(u32: 0x5555_5555)`), never a bare `const`. Generic-element rationals: `const_splat!(ratio <E>: -1 / 10)` / `const_splat!(int <E>: 8)`; the element itself (for `scale`, scalar helpers) is `const_element!(ratio <E>: 1 / 3)`. Never a per-file `macro_rules! c`. |
| `impl_bit_casts!` (`backend/macros.rs`) | `$from as $to => $conv` | `BitCastRegister` via an `arch::` cast intrinsic (float <-> int, NEON `vreinterpretq_*`). |
| `impl_bit_casts_identity!` | `$from as $to` | `BitCastRegister` where both registers already share a `Storage` type, so the reinterpret is the identity -- covers the whole x86 (`__m128i`) and wasm (`v128`) integer matrix, including SAME-WIDTH/DIFFERENT-LANE-COUNT pairs like `u8x16 <-> u64x2`. |
| `impl_bit_casts_transmute!` | `$from as $to` | Same, but for registers whose `Storage` types differ (the scalar backend's distinct `ArrayRegister` shapes) -- by-value transmute, so differing alignment is irrelevant. The element-wise array bitcast in `register/array.rs` only relates arrays of EQUAL lane count, which is why this exists. |
| `impl_sad!` | `u8reg => (u16, u32, u64 reg)` | Opts a `u8` register into the generic SWAR `Sad16/32/64Register` defaults. Output registers must be the same total width. |
| `impl_sad_native_u64!` (+ `@ssse3` arm) | ... `via $sad` | As above but overriding `sad64` with `psadbw`; the `@ssse3` arm also overrides `sad16`/`sad32` with `pmaddubsw`/`pmaddwd`. |

### 2a. What `register_trait` generates

Source: `crates/thermite-macros/src/internal.rs:101` (`register_trait_inner`).
Emits `_c`/`_m`/`_z` as *provided* methods; mask inserted as **first** arg, `_m`
inserts `src` *before* the mask (`src, mask, ...original`):

```rust
// You write, inside #[thermite_macros::register_trait] pub trait BitwiseRegister:
#[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

// Macro appends (paraphrased):
#[inline(always)] fn bitxor_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
    Self::blendv(mask, lhs, Self::bitxor(lhs, rhs))          // keep lhs where mask false
}
#[inline(always)] fn bitxor_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
    Self::blendv(mask, src, Self::bitxor(lhs, rhs))          // keep src where mask false
}
#[inline(always)] fn bitxor_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
    if const { <Self as CoreRegister>::HAS_EQUAL_SIZE_MASK } {
        Self::bitand(<Self as CoreRegister>::from_mask(mask), Self::bitxor(lhs, rhs))
    } else {
        Self::bitxor_m(Self::EMPTY, mask, lhs, rhs)
    }
}
```

Facts that bite:

- **The first existing argument is the "keep" value for `_c`** (`this` in the
  macro); unary ops keep self/input where mask false. Macro panics on
  zero-argument methods.
- A backend can **override** any variant with a faster encoding (AVX-512 maps
  to single masked instructions). The defaults above are the pre-AVX512
  lowering: `blendv`, or `bitand`-with-mask for `_z` under
  `HAS_EQUAL_SIZE_MASK`.
- A method not returning plain `Storage<Self>` (tuple, `Element`, foreign
  `Storage`) is *ineligible* -- no variants even if marked (why
  `morton_deinterleave`'s tuple return carries no marker).

`vector_trait`/`vector_impl` mirror this at the `Vector<R>` layer
(`v.op_c(mask, ...)` forwarding to `R::op_c(...)`).

### 2b. What `dispatch` / `dispatch_dyn!` generate

`#[dispatch(S)]` (`thermite-macros/src/dispatch.rs`): moves the body into an
`#[inline(always)]` inner copy, emits one `#[target_feature(enable = "...")]`
trampoline per backend, replaces the outer body with
`match <S as HasIsa>::ISA { ... }` (const, folds at monomorphization -- no
runtime branch). The backend set it iterates is the authoritative target list:

```
x86 build:   Scalar("")  X86V1("sse2")  X86V2("sse4.2,popcnt")
             X86V3("avx2,fma,popcnt" [+",f16c"] [+",pclmulqdq"] per avx2-f16c/avx2-pclmul)
neon build:  Scalar("")  NEON("neon")   -- aarch64-only, always on; NEON is baseline,
             so InstructionSet::get() is constant and the trampoline attr is a no-op
wasm build:  Scalar("")  WASM32("simd128")
```

(A `Backend` whose `simd_type` is `None` is skipped by `dispatch_dyn!` and its
hardware falls through to scalar -- the mechanism for backends without a
complete runtime-dispatchable `Simd` impl.)

`dispatch_dyn!(for<S> |...| { ... })`: runtime entry; calls cached
`InstructionSet::get()`, runs the body under the chosen backend. Bare width
names (`f32xN`, `f32x4`, `i32x8`, `usizex4`, ... -- every `Simd` assoc type)
rewrite to `Vector<S::...>`; explicit `S::f32x4` / multi-segment paths are left
alone. Signature must be ISA-agnostic.

The **call form** -- `dispatch_dyn!(dot(a, b))` (backend injected as the
callee's only generic arg) or `dispatch_dyn!(for<S> dot::<S, f32>(a, b))`
(token-level `S` substitution) -- expands to just the runtime match, no
trampolines/inner fn, because a `#[dispatch]` callee already carries its own
`#[target_feature]` codegen. Supported shape: a single dispatched call,
including method calls on a receiver (`for<S> kernel.run::<S>(&data)` against a
`#[dispatch(S)] impl`). The substitution technically accepts any expression but
that is deliberately undocumented (non-callee code inside compiles without
target features). Parsed in `DispatchDynInput`; codegen `dispatch_dyn_call` vs
`dispatch_dyn_closure`.

### 2c. What `decl_math!` generates

Source: `crates/thermite/src/math/mod.rs:54`. Unusual bracket syntax -- generics
go in `[ params ][ names ]`, not `<...>`:

```rust
decl_math! {
    trait Core<FloatElement>: FloatVector {
        fn approx_reciprocal[][](self: Self) -> Self;                 // no generics
        #[skip_dispatch] fn poly[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;
    }
}
```

expands (paraphrased) to:

```rust
pub trait CoreMathWithPolicy: FloatVector {
    fn approx_reciprocal_p<P: Policy>(self: Self) -> Self;
    fn poly_p<P: Policy, const N: usize>(self: Self, coeffs: &[Self::Element; N]) -> Self;
}
pub trait CoreMath: CoreMathWithPolicy {
    #[inline(always)] fn approx_reciprocal(self) -> Self { Self::approx_reciprocal_p::<DefaultPolicy>(self) }
    #[inline(always)] fn poly<const N: usize>(self, c: &[Self::Element; N]) -> Self { Self::poly_p::<DefaultPolicy, N>(self, c) }
}
impl<M> CoreMath for M where M: CoreMathWithPolicy {}

// blanket impl: any float vector whose element has a Specialized kernel:
impl<E: FloatElement, V: FloatVector<Element = E>> CoreMathWithPolicy for V
    where V: specialized::SpecializedCoreMath<E>
{
    #[inline(always)] fn approx_reciprocal_p<P: Policy>(self) -> Self {
        <V as specialized::SpecializedCoreMath<E>>::reciprocal::<P>(self)
    }
}
```

Plus `ScalarMathWithPolicy`/`ScalarMath` on bare f32/f64 (`scalar_`-prefixed).
Trait and impl are wrapped in
`#[thermite_macros::dispatch(Self, thermite = "crate")]`, so each `_p` gets
per-ISA codegen -- except `#[skip_dispatch]` items
(`poly`/`poly_rev`/`poly_rational`), which must inline rather than cross a
dispatch boundary.

**Declaring in `decl_math!` does not implement.** It creates the surface and
routes to `Specialized<Family>Math<E>::<name>`, which you implement per element
in `ps.rs`/`pd.rs` (section 6).

---

## 3. The backend and polyfill system

A register impl rarely calls raw `core::arch` directly; it calls its backend's
**`arch` namespace**, which fuses two separately-inherited layers:
(1) real intrinsics via the ISA ladder in `backend/x86.rs`, (2) polyfills under
each backend's `polyfills/`.

### 3a. The `arch` namespace per backend

```rust
// backend/x86_v3/mod.rs
pub mod arch {
    pub use super::polyfills::*;             // own polyfills (+ all inherited)
    pub use crate::backend::x86::avx2::*;    // real intrinsics at this ISA level
}
```

| Backend | `arch` real intrinsics | `arch` polyfills |
|---|---|---|
| `x86_v1` | `backend::x86::sse2::*` | own + generic |
| `x86_v2` | `backend::x86::sse42::*` | own + v1 + generic |
| `x86_v3` | `backend::x86::avx2::*` | own + v2 + v1 + generic |
| `wasm` | `core::arch::wasm32::*` (wasm64 on nightly) | own + generic |

A register file does `use super::arch::*`, so `arch::_mm256_xor_si256(...)` or
`arch::_mm256_blendv_epi32x_v3(...)` resolves to a real intrinsic *or* polyfill
transparently -- one flat namespace of "things callable at this ISA level".

### 3b. The real-intrinsic ISA ladder (`backend/x86.rs`)

Nested modules, each `pub use`-ing the tier below, so the set grows
monotonically:

```
sse -> sse2 -> sse3 -> ssse3 -> sse41 -> sse42
                                   \-> avx (= f16c + sse42) -> avx2 (= avx + fma)
                                                                  \-> avx512f -> tiers{1,2,3,4}
```

`sse42::*` contains everything from `sse` up; `avx2::*` all of SSE + AVX + FMA
(why `x86_v3` can still use `_mm_xor_si128` on a 128-bit half). Optional
sub-features layer in by `cfg`: `avx2-pclmul` adds `_mm_clmulepi64_si128`
(CLMUL 2D-Morton path), `avx2-f16c` adds f16c conversions -- both also add the
feature to the dispatched `#[target_feature]` set. AVX-512 splits into
`tiers::tier1..4` (CD; +BW/DQ; +VBMI/VBMI2/VNNI/BITALG/GFNI/...; +BF16),
matching the `avx512-tier1..4` crate features.

The `x86_v4` backend is generic over those tiers rather than being four
backends: `X86V4<F: Avx512Features>` (`backend/x86_v4/mod.rs`), where
`Avx512Features` is a const-per-extension trait implemented by the ZSTs
`Tier1..Tier4`. Register code forks on `if const { F::AVX512VBMI }` and folds at
monomorphization, exactly like `HAS_NATIVE_FMA`. Two rules: ask about a *feature*,
never `TIER`, so adding a rung never changes what an existing fork means; and a
const being `true` does not make the intrinsic callable -- the
`#[target_feature]` set in `thermite-macros/src/dispatch.rs` must enable the same
feature, and the two lists are maintained by hand.

### 3c. The polyfill inheritance chain

Each backend's `polyfills/mod.rs` re-exports the next lower backend's polyfills,
then declares its own (split by domain: `bits.rs`, `casts.rs`, `cmp.rs`,
`divider.rs`, `math.rs`; generic adds `compress.rs`, `sort.rs`):

```rust
// x86_v1: pub use crate::backend::generic::polyfills::*;
// x86_v2: pub use crate::backend::x86_v1::polyfills::*;
// x86_v3: pub use crate::backend::x86_v2::polyfills::*;
// wasm:   pub use crate::backend::generic::polyfills::*;
```

Chain: **generic ⊂ v1 ⊂ v2 ⊂ v3** (and **generic ⊂ wasm**). So a v3 256-bit op
processing two 128-bit halves reuses `_mm_blendv_epi8x_v1` (pure-SSE2 bitwise
select from v1) or `_mm_permutevarx_epi32x_v2` (SSSE3 `pshufb` permute from v2).
Write a helper once at the lowest tier that can express it; higher tiers inherit
free.

Naming: polyfills mimic the intrinsic they stand in for, with an `x_v<n>` suffix
("polyfill, introduced at tier n") -- `_mm_blendv_epi8x_v1`,
`_mm_popcnt_epi32x_v2`, `_mm256_srai_epi64x_v3`.

### 3d. Generic polyfills (`backend/generic/polyfills/`)

Bottom of the chain, different in kind: generic over `R: Register` (or a
sub-trait), written **entirely in register-trait ops** (`R::shl`, `R::bitand`,
...), never raw intrinsics -- so they compile for **every** backend. The
"better than scalar, portable" fallbacks: N-dimensional
`morton_cascade`/`reverse_morton_cascade` shift/mask bit-spread, `compress`
left-pack and its `expand` inverse, the `scan` prefix ladder, `conflict`
duplicate ranks, sorting networks (`sort.rs`), generic `casts`/`divider`. A backend
with a hardware shortcut overrides in its register impl; others delegate to the
cascade. (The CLMUL `N == 2` Morton fast path delegates every other dimension
count to `morton_cascade`.)

The compress/expand pair shares one 256-entry 8-lane table (`COMPRESS8` in
`compress.rs`, `EXPAND8` in `expand.rs` being its row-wise inverse); shared items
live on the compress side by convention. Rows store `u8` indices, so a register
must implement `WidenIndexRegister` to widen a row into the `u32` permute
control. That method has no default on purpose -- a portable widening loop does
not vectorize, and a defaulted one would cost ~8 instructions per compress with
every test still green. Byte-shuffle backends (NEON, wasm) override
`permutev_row` to feed the row in as bytes and skip the widen/narrow round trip.

### 3e. Where to put a new helper

- **Register-trait ops only, useful anywhere** -> `backend/generic/polyfills/`,
  generic over `R`. Backends with hardware paths override in register impls.
- **Needs ISA intrinsics** -> the polyfill file of the **lowest tier whose
  `arch` suffices**. `pshufb`-based helper goes in **v2** (SSE2 lacks pshufb);
  v3 inherits it for 128-bit-half work and adds its own `_mm256_*x_v3`. A pure
  shift/and/or helper can live in v1 and serve all three tiers.
- **Domain**: bit twiddling -> `bits.rs`, compares/masks -> `cmp.rs`, numeric
  casts -> `casts.rs`, division -> `divider.rs`, float math -> `math.rs`.

> **Tier discipline:** a polyfill at tier N may use ONLY intrinsics available at
> tier N (every backend >= N pulls it in). An SSE4.1-only helper in v1's
> polyfills breaks the v1 (SSE2) build. When unsure, push *up* to the first tier
> whose `arch::` has the instructions, or *down* to generic if expressible in
> register ops.

---

## 4. Worked example: accelerated Morton-code interleave

Morton (Z-order) encoding interleaves bits: bit `i` of `a` -> bit `2i`, bit `i`
of `b` -> bit `2i+1`. Per-lane integer bit op -- same shape as `reverse_bits` /
`count_ones`, which are the ground-truth templates to copy.

### Step 0 -- pick the layer and trait

Value per lane from integer lanes, no precision/policy dimension -> a
**register primitive**, not a math function. `BitshiftRegister`
(`Register<Element: IntegerElement>`, `register/mod.rs:1325`) already hosts the
bit-twiddling family -> put `morton_interleave` there. The inverse
`morton_deinterleave` returns a tuple -> mask-ineligible (2a) -> no marker, no
`_c`/`_m`/`_z`.

### Step 1 -- declare on the register trait (`register/mod.rs`)

```rust
#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait BitshiftRegister: Register<Element: IntegerElement> {
    // ... existing reverse_bits, shli/shri, rol/ror ...
    /// Interleave the low bits of `a` (even positions) and `b` (odd positions).
    #[conditional]
    fn morton_interleave(a: Storage<Self>, b: Storage<Self>) -> Storage<Self>;
    /// Inverse. Tuple return is not mask-eligible, so no marker.
    fn morton_deinterleave(value: Storage<Self>) -> (Storage<Self>, Storage<Self>);
}
```

No body -- abstract; backends supply it. A default body is allowed only when a
correct portable definition exists in other register ops (`bitandnot`,
`ternlog`, `shli` do that); Morton's fast paths are hardware-specific.

### Step 2 -- expose on the vector trait and impl

```rust
// vector/mod.rs, inside #[vector_trait] pub trait BitshiftVector
#[conditional] fn morton_interleave(self, other: Self) -> Self;
fn morton_deinterleave(self) -> (Self, Self);

// vector/vector.rs, inside the #[vector_impl] impl for Vector<R>
#[conditional] fn morton_interleave(self, other: Self) -> Self {}   // body filled by macro
fn morton_deinterleave(self) -> (Self, Self) {}
```

Empty braces are intentional: `vector_impl` rewrites to
`Vector(R::morton_interleave(self.0, other.0))` + masked siblings. Match the
register signature exactly or the expansion mis-types.

### Step 3 -- emulated widths come almost free

Same empty stubs in each `#[array_impl]`/`#[reduced_impl]` block for
`BitshiftRegister` (`register/array.rs`, `register/reduced.rs`):

```rust
#[conditional] fn morton_interleave(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
fn morton_deinterleave(v: Storage<Self>) -> (Storage<Self>, Storage<Self>) {}
```

`array_impl` writes the per-inner-register loop; `reduced_impl` delegates wide
and re-masks. This lights up `f32x8`-on-SSE (`ArrayRegister<_,2>`) and reduced
widths (`u32x3`) with no hand-written Morton.

### Step 4 -- implement per backend (the real work)

Real body in `backend/<isa>/registers/<inttype>.rs` for every integer register
width. Copy how `reverse_bits`/`count_ones` are done in those files.

- **Scalar** (`backend/scalar/cpu/unsigned.rs`, `signed.rs`): width macro
  delegating to std inherent methods where available; Morton has none, so write
  the classic magic-number bit-spread (`0x5555.../0x3333.../0x0f0f...` cascade,
  then `a_spread | (b_spread << 1)`). The scalar backend is the
  **differential-test oracle** -- obviously correct beats clever.
- **x86_v1 / SSE2** (`registers/u32x4.rs`, ...): no pshufb/blendv. Magic
  cascade with `_mm_slli_epi32`/`_mm_srli_epi32`/`_mm_and_si128`/`_mm_or_si128`
  via `arch::`. Shared logic -> `backend/x86_v1/polyfills/bits.rs`.
- **x86_v2 / SSE4.2**: `pshufb` (`_mm_shuffle_epi8`) available -- 4-bit-nibble
  LUT spread is the fast path. Add `_mm_morton_interleave_epi32_v2` to
  `backend/x86_v2/polyfills/bits.rs`, call from the register file (mirroring
  `count_ones` -> popcount polyfill).
- **x86_v3 / AVX2** (`registers/u32x8.rs`, ...): 256-bit analogue,
  `_mm256_shuffle_epi8` + polyfill in `backend/x86_v3/polyfills/bits.rs`. Blocks
  are `#[inline_always]`-tagged and call `unsafe { arch::_mm256_... }` (see how
  `bitxor` is `unsafe { arch::_mm256_xor_si256(lhs, rhs) }`).
- **wasm** (`backend/wasm/registers/*.rs`): shifts + `v128.and/or` magic
  cascade, or `i8x16.swizzle` LUT under `relaxed-simd`.
- **spirv** (`backend/scalar/spirv/` + `backend/spirv/registers/`): only if
  needed; one lane per invocation, scalar cascade applies.

> **PDEP/PEXT caveat.** BMI2 `_pdep_u32`/`_pext_u32` are the textbook Morton
> instructions but are **scalar GPRs, not SIMD** -- no packed PDEP before
> AVX-512 GFNI/`vpermb` tricks. SIMD Morton = magic-number shift/mask spread
> (any backend) or pshufb/swizzle nibble-LUT (v2/v3/wasm). Don't reach for
> `_pdep` in a register impl.

### Step 5 -- test across backends

Differential test alongside the existing bit-op tests in
`crates/thermite/tests/` (section 7): run on `Scalar`, `X86V2`, `X86V3`, assert
agreement with `Tol::Exact` (integer bit op). Then `just wasm-test`. Cheapest
strong check: round-trip
`deinterleave(interleave(a,b)) == (a & low_mask, b & low_mask)`.

---

## 5. Checklists for any new operation

Classify first -- the two kinds take different paths.

### A. Register primitive

(bit op, integer op, lane-shuffle, capability-gated identity.) The Morton path
is the template:

1. Declare on the right `*Register` trait (`register/mod.rs`);
   `#[conditional]`/`#[masked]` only if the return is a single `Storage<Self>`.
2. Declare on the matching `*Vector` trait + empty stub in `vector/vector.rs`.
3. Empty stubs in the `array_impl`/`reduced_impl` blocks.
4. Real impl per backend in `backend/<isa>/registers/<type>.rs`; shared logic
   into `backend/<isa>/polyfills/` or `backend/generic/polyfills/`.
5. Differential test.

**Exception:** a purely algebraic identity expressible in existing ops (like
`one_minus_sq`) can be a **provided default method on the vector trait** with no
per-backend work -- inherited by `Compensated`/`Complex` wrappers free. Prefer
that when it applies ([performance.md](performance.md) sec 10). A new
*required* register method *breaks* the composite wrappers until they implement
it.

### B. Precision-tunable math function

(transcendental, special function, norm.) Rides the math machinery, not the
backends:

1. Add the signature to the right `decl_math!` block in `math/mod.rs` (`Float`,
   `Core`, `Transcendental`, `Spatial`, `Real`); bracket-generics syntax (2c).
2. Implement the **f32 kernel** in `math/specialized/ps.rs` and **f64 kernel**
   in `pd.rs`, on the `Specialized*Math<E>` trait; element-agnostic helpers in
   `generic.rs`.
3. Thread `Policy` through the kernel (section 6).
4. Coefficients inline as `poly_p`/`poly_rev_p` arrays (derive with
   `bin/remez`); bit-pattern constants via `const_splat!`.
5. **Composites:** implement the same `Specialized*Math` entry in
   `thermite-dual` and `thermite-compensated` (6c) -- or generic code at those
   types fails to compile / hits `todo!()`.
6. Accuracy-test under multiple policies, incl. an f32 case for any
   `HAS_APPROX_*`/`HAS_NATIVE_FMA` branch.

---

## 6. The math-kernel layer in depth

### 6a. A real kernel, annotated -- `cbrt` for f64

`math/specialized/pd.rs`, `SpecializedTranscendentalMath::cbrt` (verbatim). The
canonical shape: bit-twiddled initial guess, polynomial refinement, Newton/
Halley steps, policy-gated paths and overflow check.

```rust
fn cbrt<P: Policy>(self) -> Self {
    let x = self.flush_denormals::<P>();                       // denormal handling per policy

    let b1 = crate::const_splat!(u64: 715094163);             // magic exponent biases ...
    let b2 = crate::const_splat!(u64: 696219795);             // ... as bit patterns, via const_splat!
    let m  = crate::const_splat!(u64: 0x7fffffff);

    let x1p54 = x * Self::splat(f64::from_bits(0x4350000000000000)); // 2^54 to rescale denormals
    let hx0   = (x.into_bits::<V::Bits>() >> 32) & m;
    let x_small = hx0.cmp_lt(V::Bits::splat(0x00100000));
    let xs = x_small.select(x1p54, x);
    let b  = x_small.select(b2, b1);

    let mut ui: V::Bits = xs.into_bits();
    let mut hx: V::Bits = (ui >> 32) & m;
    hx = hx / Divider::u64(3) + b;                            // exponent / 3 + bias = initial guess
    ui &= V::Bits::splat(1 << 63);
    ui |= hx << 32;
    let mut t = Self::from_bits(ui);

    let r = (t * t) * (t / x);                                // "encourage ILP" -- see perf doc
    t *= r.poly_p::<P, _>(&[                                  // minimax refinement polynomial
        1.87595182427177009643,
        -1.88497979543377169875,
        1.621429720105354466140,
        -0.758397934778766047437,
        0.145996192886612446982,
    ]);

    ui = t.into_bits();
    ui = (ui + V::Bits::splat(0x80000000)) & V::Bits::splat(0xffffffffc0000000);
    t  = Self::from_bits(ui);

    let r = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !matches!(Self::HAS_NATIVE_FMA, tribool::True) } {
        let xtt = x / (t * t);                                // exact form: 5 ops, 2 divisions
        (xtt - t) / ((t + t) + xtt)
    } else {
        let t3 = t * t * t;                                   // fast form: 1 division + 1 FMA
        (x - t3) / t3.mul_add(Self::TWO, x)
    };
    t = r.mul_adde(t, t);

    if const { !P::POLICY.check_overflow } {
        return x.cmp_eq(Self::ZERO).select(x, t);            // skip the inf/zero guard
    }
    (hx0.cmp_gt(V::Bits::splat(0x7f800000)) | hx0.cmp_eq(V::Bits::ZERO)).select(x, t)
}
```

What to copy:

- **`if const { P::POLICY.precision.ge(...) }`** forks codegen at compile time.
  The `Best` branch also fires when `HAS_NATIVE_FMA` is not `True` (fast form relies on fused
  `mul_add`). Note the deliberate mix: `mul_add` (always-fused) inside the
  FMA-gated branch, `mul_adde` (estimating) for the final unconditional step.
- **`P::POLICY.check_overflow`** gates NaN/inf/zero handling (perf policies drop
  it); `flush_denormals::<P>()` honors `P::POLICY.denormal_behavior`.
- **Coefficients are inline literals**; the f32 kernel in `ps.rs` uses
  *different* coefficients and often a different algorithm shape (more
  aggressive range reduction, `ExtendedPrecision` upcasts). f32 and f64 are
  separate hand-written kernels -- no shared generic body for transcendentals.
- **Bit constants always via `const_splat!(u64: ...)`**, never a bare `const`.

`Policy`/`PolicyParameters`/`PrecisionPolicy`: `math/policy.rs` (field list and
presets in [math.md](math.md)). `poly_p`/`poly_rev_p` (hybrid Estrin/Horner with
FMA) and convergent-series helpers (`newtons_method`, `sum_f`,
`reduce_in_place`): `math/algorithms/` and `Core`.

### 6b. The `Specialized*Math<E>` traits

`math/specialized/mod.rs` declares `SpecializedCoreMath<E>`,
`SpecializedTranscendentalMath<E>`, `SpecializedSpatialMath<E>`,
`SpecializedRealMath<E>` (and `Float`). Parameterized by the **element type**
`E`, not the vector -- the hinge that lets composites implement them. `ps.rs`
implements for f32 vectors, `pd.rs` for f64. The `decl_math!` blanket impl (2c)
connects "any `FloatVector<Element = E>` whose `E` has a `Specialized*Math`
impl" to the public `*Math` trait.

### 6c. Composites must implement the Specialized traits

A new math function is **not finished** until `Dual` and `Compensated` cover it
-- they implement `Specialized*Math` independently, not by inheritance.

`Dual` (`thermite-dual/src/math.rs`): primal via the inner `_p` method,
derivatives by chain rule:

```rust
impl<V: DualMathVector, const N: usize> SpecializedTranscendentalMath<Dual<V::Element, N>> for Dual<V, N> {
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();
        (self.chain(s, c), self.chain(c, s.neg()))           // d sin = cos, d cos = -sin
    }
    fn tan<P: Policy>(self) -> Self {
        let t = self.re.tan_p::<P>();
        self.chain(t, t.mul_adde(t, V::ONE))                 // d tan = 1 + tan^2
    }
}
impl<V: DualMathVector, const N: usize> SpecializedCoreMath<Dual<V::Element, N>> for Dual<V, N> {
    fn inverse_sqrt<P: Policy>(self) -> Self {
        let r = self.re.inverse_sqrt_p::<P>();
        self.chain(r, (V::HALF * r * r * r).neg())           // d x^-1/2 = -1/2 x^-3/2
    }
}
```

`chain(value, derivative)` sets the primal and multiplies every partial by the
local derivative. **Most functions need no touch**: `Dual` only overrides
primitives; `sin`, `cos`, `powi`, `lerp`, `smoothstep`, etc. fall out of trait
*defaults* composed over dual arithmetic. Add an override only when (a) a
dedicated inner primitive is cheaper/more accurate than the default
composition, or (b) the default would differentiate through an internal
iteration better bypassed with the analytic derivative (the
`inverse_smoothstep` implicit-function-theorem trick).

`Compensated` (`thermite-compensated/src/math.rs`) **re-derives** the algorithm
in double-double, e.g. `sin_cos` = compensated argument reduction
(`k = round(x * 2/pi)`, `r = x - k*(pi/2)` with the compensated `FRAC_PI_2`)
then a ~20-term series in `Compensated` arithmetic. Coverage is **partial** --
many masked/reduction ops still `todo!()`. If your function can't be expressed
from implemented compensated ops, a flagged `todo!()` is acceptable.

### 6d. Element constants kernels rely on

`element/float/mod.rs` (`FloatElementWithBits`, 135) and `element/float/spec.rs`:
`EXP_BITS`, `MANTISSA_BITS`, `EXP_BIAS`, `MAX_BIASED_EXP`, `SIGN_MASK`,
`MANTISSA_MASK`, `DENORMAL_TRICK`, plus `Bits`/`SignedBits` views. (Defined for
f16/bf16/f8 too, but active math kernels are f32/f64 only; the sub-f32 formats
are *storage* formats served by `PackedFloatRegister`/`PackedFloatVector` --
`FloatSpec` types in `element/float/spec.rs`, generic branchless
`pack_packed`/`unpack_packed` fallbacks at the bottom of `register/mod.rs`,
hardware overrides per backend. See [vector-api.md](vector-api.md) sec 9.) For a non-bit-layout
per-element constant, prefer `FloatConsts` (`math/consts.rs`) if it exists; else
a small per-element trait with `const` members + `V::splat`
([performance.md](performance.md) sec 5) -- runtime math for a compile-time
constant is wasted work.

---

## 7. Build / test / verify loop

The `justfile` is the task runner, scoped to the `thermite` crate (a
`--workspace` build fails: spirv/neon/wasm members don't build on an x86 host).
PowerShell on Windows.

| Recipe | Does |
|---|---|
| `just test` | `cargo +stable nextest run -p thermite --features std --release` + `cargo test --release --doc` (nextest can't run doctests). Per-test processes scheduled across all cores (~25% faster). Differential suites are slow in debug. **This is the gate.** Filter: `just test -E 'test(interleave)'` or `just test --test diff_ops`. |
| `just test-fast` | nextest only, no doctests. |
| `just wasm-test [filter]` | Builds `wasm-runner`, runs wasm-applicable tests on `wasm32-wasip1` under wasmtime, `--features "wasm,std"`, `--test-threads=1`, LTO off. `+simd128,+relaxed-simd` from `.cargo/config.toml`. Prereq: `rustup target add wasm32-wasip1`. Scope: `just wasm-test "--test diff_ops"`. |
| `just cov` / `cov-collect` / `cov-missing` / `cov-summary` / `cov-percent` | `cargo-llvm-cov`. Always `--ignore-run-fail` (one accepted failure: `frldexp` denormal-flush under off-by-default `preserve_denormals`). **Never `--all-features`** -- backend features are mutually exclusive on one host. `cov-missing` = authoritative uncovered-line list. |
| `just cov-branch` | Branch coverage (nightly). Reads low/misleading (LLVM doesn't credit diverging arms); informational only. |
| `just miri [filter]` | `cargo +nightly miri test`. Miri lacks x86 SIMD intrinsics (V2/V3 error) -- use for scalar + generic `unsafe` (slice iterators, gather/scatter bounds). Filter to Miri-safe tests. |
| `just pi-build [args]` | Cross-compiles the test binaries for `aarch64-unknown-linux-musl` with `--features "std"` (LTO off) and stages them in `target/pi-stage` for ARM hardware / qemu. The NEON backend needs no feature flag. |
| `just qemu-test` | Runs the staged aarch64 binaries under qemu via Podman (`--platform linux/arm64`, alpine). Prereq: `just pi-build`; binfmt install once per machine boot. |
| `just doc` / `doc-open` | rustdoc with the KaTeX header (`katex-header.html`). |
| `just sync-assets` | Re-propagate `LICENSE-*` and `katex-header.html`. |
| `just bundle-skill` | Zip this skill for distribution. |

### 7a. The differential test harness

`crates/thermite/tests/` `diff_*.rs` files (`diff_ops`, `diff_math`,
`diff_mask`, `diff_polyfill`, `diff_cast`, `diff_linalg`, `diff_swizzle`,
`diff_slice`, `diff_gather`, `diff_divider`, ...) with a shared `harness/`. The
**scalar backend is ground truth**; every SIMD backend (`X86V2`, `X86V3`, wasm
where applicable) runs the same inputs and is compared. Tolerance enum:
`Tol::Exact` (bit-exact; integer/bitwise ops), `Tol::Ulp(n)`, `Tol::Rel(eps)`,
`Tol::ExactOrNan` (`min`/`max` where NaN handling legitimately diverges).
Repo-root `TESTING.md` documents methodology + defect log.

For a new op: edge-case + random inputs, cross-backend agreement at the right
tolerance. Cover masked variants (`_c`/`_m`/`_z`) explicitly -- separate
codegen. Add f32 *and* f64 cases when behavior differs by element (approx
rcp/rsqrt, FMA). For a math kernel, validate against a high-precision oracle
(libm or the `Reference` policy) with tight tolerance (e.g. `1e-13` f64) so a
"harmless" algebraic transform is proven harmless.

### 7b. Cross-backend verification beyond the harness

- **wasm:** `just wasm-test "--test diff_ops"`. Runner is `tests/wasm-runner`.
- **spirv (GPU):** `cargo run -p spirv_builder` compiles `bin/spirv_testing`
  kernels to `.spv` (disassembled to `out.spv.txt`); `cargo run -p spirv_runner`
  executes via wgpu vs the CPU reference. Repo-root
  `before.spv.txt`/`after.spv.txt` are kept for diffing codegen changes. See
  `bin/spirv_testing/CLAUDE.md` for GPU setup.
- **assembly:** confirm a hot kernel lowered as intended -- `cargo asm` on a
  monomorphic fn, or
  `cargo rustc --release -- -C target-cpu=x86-64-v3 --emit asm`. Check for real
  `vfmadd*`/`vpshufb`, expected `vsqrtps`/`vdivps` counts, broadcast constants,
  no spills. Measure with `cargo bench` (Criterion), never by eyeballing or
  running artifacts directly ([performance.md](performance.md) sec 13).
- **force a backend:** an explicit `#[target_feature(enable = "sse4.2")]` (or
  `"avx2,fma"`) wrapper, or `Vector<thermite::backend::scalar::Scalar>` for
  scalar.

### 7c. Tooling crates

- **`bin/remez`**: minimax polynomial coefficients for `ps.rs`/`pd.rs`. Record
  the interval and achieved max error when adding/retuning a transcendental.
- **`bin/docgen`**: register-coverage docs (which ops each backend implements).

### 7d. CI

`.github/workflows/`: main workflow = stable test suite (`--features std`) +
wasm *build* check + coverage percent. `ffi_artifacts.yaml` builds the FFI
cdylib (+ cbindgen header) on nightly, `release-ffi`, on demand. A rustdoc
workflow deploys docs (KaTeX header) for the `rewrite` branch.

---

## 8. Development gotchas

- **Masked variants are mask-first; `src` precedes `mask` in `_m`.**
  `op_c(mask, ...)`, `op_m(src, mask, ...)`, `op_z(mask, ...)`. Any doc showing
  `add_c(b, mask)` is stale.
- **Marker eligibility.** Only a single `Storage<Self>` (register) / `Self`
  (vector) return gets variants; tuple/`Element`/foreign-`Storage` returns are
  skipped even if marked.
- **Watch each intrinsic's own operand convention.** `bitandnot(lhs, rhs)` is
  `lhs & !rhs` at EVERY layer (register, `Vector`, `Mask`) -- `vector/ops.rs`
  delegates straight through with no swap. NEON `vbic` and wasm `andnot` match
  that directly; x86 `andnot` and the AVX-512 `vandn`/`kandn` opmask forms
  negate their FIRST operand, so those backend impls swap the arguments
  internally. A self-cancelling wrapper bug here passes internal use and only
  diff tests catch it.
- **A default-bodied register/vector method is auto-`#[inline(always)]`.** Keep
  defaults expressible purely in other trait ops; hardware-specific bodies
  belong in backend impls.
- **A capability flag that drifts from its impl is invisible.** `HAS_NATIVE_ALIGN`
  says whether `Register::align` is a real cross-register align rather than the
  `swizzle_const` default. Both paths agree on results, so a register that
  silently keeps the `false` default changes nothing a functional test sees -- it
  just routes the whole prefix-scan family onto the sequential fallback. Set the
  flag in the same `impl_*_align*!` macro that emits the body, and if you hand-roll
  an `align`, set it by hand; `tests/align.rs::native_align_flag` asserts it for
  every full-width register, and that `GenericVector::HAS_NATIVE_ALIGN` (the
  vector-layer re-export, which composites forward from their inner vector) reports
  the same thing. Same discipline elsewhere: `compress_via_table!` /
  `compress_via_wide!` emit `compress` *and* `expand` together so a backend cannot
  take a fast one and a scalar other.
- **A new register gets `HasIsa` via `impl_has_isa!`, not `const ISA`.**
  `CoreRegister` requires `HasIsa` as a supertrait (this is what lets
  `#[thermite::dispatch(R)]` work over bare register types), and `HasIsa::ISA`
  defaults to `<Self::Native as HasIsa>::ISA`, so a register names the backend
  that owns it once. Hand-written backends list their registers in one
  `impl_has_isa!(X86V3: F32x4V3, ...)` invocation in `registers/mod.rs`
  (`backend/macros.rs`); macro-stamped families (neon, scalar via
  `impl_has_isa!` in `scalar/mod.rs`, spirv, the v4 kmasks) emit or invoke it
  next to their `CoreRegister` stamp. Backend types themselves are the
  exception: they implement `HasIsa` directly with `type Native = Self`, so
  they must still spell `const ISA` or the default would recurse. Emulated
  wrappers forward (`ArrayRegister`/`ReducedRegister` use `R::Native`), which is
  why `f32x16<X86V1>` correctly reports `X86V1` while the scalar-lane
  `ArrayRegister<i16, 2>` shared by every backend reports `Scalar`.
  `tests/native_isa.rs` asserts both the mapping and that `ISA` agrees with
  `Native::ISA`.
- **The scalar backend is mandatory and is the oracle.** Must compile and be
  correct for every primitive. Simple over fast.
- **x86_v1 is real and limited.** SSE2 has no `pshufb`, `blendv`, `round`,
  variable shift, or HW popcount -- polyfilled. Don't call a v2+ intrinsic from
  a v1 register file. (Backend set: `thermite-macros/src/dispatch.rs`, x86 =
  Scalar/X86V1/X86V2/X86V3.)
- **Constants via `const_splat!`/`const_new!`**, never bare `const`, for correct
  per-ISA splat codegen. Bit patterns: `const_splat!(u32: 0x5555_5555)`.
- **FMA in kernels -- know the non-`e` fallback.** Estimating `mul_adde` is the
  default (real FMA where present, else `mul`+`add`). Always-fused `mul_add`
  does **not** drop straight to scalar `libm::fma` on non-FMA backends: by
  default it lowers to a **vectorized, correctly rounded emulated FMA** --
  **bit-identical to a true hardware FMA for every input** -- via the
  Boldo-Melquiond 2008 round-to-odd polyfills (`fmadd_ro` for f64,
  `fmadd_widen_ro` for f32, in `backend/generic/polyfills/math.rs`; Coq-proved
  algorithm, verified against the hardware FMA instruction in
  `tests/fma_exact.rs`). The f64 path routes rare packets (subnormal-scale
  products via a pre-gate; overflow/inf/NaN via an `is_finite` post-check) to a
  fully VECTORIZED rescue -- no scalar loop, no libm, anywhere; the f32 path is
  branch-free. This is unconditional: the historical `disable_fast_fma`
  feature was REMOVED (there was nothing left to trade, and the libm f32 chain
  it selected carries a known subnormal bug). So `mul_add` is a full
  *accuracy* guarantee without hardware FMA; gate behind
  `if const { matches!(V::HAS_NATIVE_FMA, tribool::True) }` only to avoid the *emulation* cost. The `cbrt`
  kernel uses always-fused `mul_add` inside the FMA-gated branch and estimating
  `mul_adde` for the final unconditional step
  ([performance.md](performance.md) secs 1-2). For genuine double-double
  precision, `thermite-compensated` is often cleaner than leaning on emulated
  FMA. (The old Dekker/Veltkamp polyfills `_mm_fmadd_p[sd]x_v1` were deleted
  2026-08-22; their ~1-in-173k divergence is historical, bodies in git history.)
- **Rule zero: `#[dispatch]` on the boundary, `#[inline(always)]` on the
  interior.** A generic-over-`S` body with no `#[dispatch]` above it (and no
  `#[dispatch]` ancestor inlining it) is compiled without the target features,
  and rustc will not inline a `target_feature` intrinsic into it -- a `call` per
  single instruction. Correct, tests pass, catastrophically slow, invisible to
  the type system. Helpers need `#[inline(always)]` (features propagate only via
  inlining; `#[inline]` gets declined), but don't `#[dispatch]` one-liners --
  that only blocks inlining. Full detail: [performance.md](performance.md) sec 0.
- **`target_feature` codegen traps that pass tests but tank benches:**
  `core::array::map`/`from_fn` and bare closures fail to inline in
  `target_feature` code and fall back to scalar -- hand-roll `while` loops.
  `#[inline(always)]` helpers are fine ([performance.md](performance.md) sec 11).
- **Dead-code elimination can hide a monomorphization error** until a different
  call site exercises the path -- e.g. the F16C `cvtps_ph` rounding-immediate
  `static_assert_uimm_bits` only fires when the conversion is reached. Test
  every variant and element type, not one instantiation.
- **A new math function is incomplete until `Dual` and `Compensated` implement
  it.** They don't inherit kernels; generic code at those types won't compile
  (or hits `todo!()`). `Dual` usually needs a small chain-rule override or
  nothing; `Compensated` may need a re-derivation.
- **Don't assume an op exists on a companion because core has it.** The
  composites are much closer to complete than they used to be (`Compensated`'s
  masked `FloatVector` ops and its whole gamma family are done), but coverage is
  still per-crate: check the impl before writing generic code that needs it.
- **`#[skip_dispatch]`** opts a `decl_math!` method (or any `#[dispatch]` item)
  out of the per-ISA trampoline -- for things that must inline
  (`poly`/`poly_rev`/`poly_rational`) rather than cross a dispatch boundary.
