# Developing Thermite itself (contributor guide)

**This file inverts the rest of the skill.** Every other reference treats Thermite
as a dependency you *consume*: you bound on `FloatVector`, the caller picks the
backend, you never see a register. This file is for *changing Thermite's own
source* -- adding an operation, a math function, a backend path, fixing internals --
which means working at every layer the user never touches.

The mental model: a user-facing method does not exist until it has been threaded
through **all** of these, or it will not compile for some instantiation:

```
register trait  (register/mod.rs)            <- declare the primitive
   |  per-backend impls (backend/<isa>/...)  <- implement it on real hardware
   |  ArrayRegister / ReducedRegister        <- emulated widths (free via macro)
vector trait    (vector/mod.rs)              <- expose it on Vector<R>
   |  vector impl (vector/vector.rs)         <- delegate to the register (free via macro)
math layer      (math/...)                   <- ONLY if it is a precision-tunable function
composites      (thermite-dual/-compensated) <- ONLY if they need bespoke semantics
tests           (crates/thermite/tests/)     <- differential vs the scalar oracle
```

"Free via macro" means a proc-macro writes the boilerplate; you still write a
stub. Nothing is free *across backends* -- a new primitive needs a real
implementation for **scalar, x86_v1 (SSE2), x86_v2 (SSE4.2), x86_v3 (AVX2),
wasm**, and optionally spirv/neon, because the dispatcher can select any of them.

Read [architecture.md](architecture.md) first if you have not -- this guide
assumes the Element -> Register -> Vector layering. The user-facing references are
still load-bearing when you change the code behind them: [trait-hierarchy.md](trait-hierarchy.md)
for what each trait owns, [math.md](math.md) for the policy surface,
[performance.md](performance.md) for the kernel style, [masks.md](masks.md) for the
`_c`/`_m`/`_z` semantics you are about to *implement* rather than call.

The code is the authority. Every path below was checked against the current tree,
but line numbers drift -- grep for the symbol, not the line.

---

## 0. The overriding goal: maximum performance

**Thermite exists to be the fastest portable SIMD library possible.** Performance is
the top priority, and it justifies effort that would be over-engineering elsewhere.
When you work on the internals, internalize this:

- **Per-register, per-backend specialization is encouraged, not avoided.** If a
  single concrete register on a single ISA can do an op faster with a bespoke
  intrinsic sequence, write that specialization -- even if it means `u32x8` on AVX2
  and `u32x4` on SSE4.2 and `u32x4` on SSE2 each get a different hand-tuned body.
  "Bending over backwards" for a concrete type's fast path is the expected default,
  not a smell. The generic/portable path is the *correctness floor* every type falls
  back to; the per-backend overrides are where the wins live.
- **Any avenue to more speed is on the table:** a new polyfill, a feature-gated CPU
  sub-extension (`avx2-pclmul`, `avx2-f16c`, an AVX-512 tier), a cheaper algebraic
  identity, a shorter dependency chain, a capability-gated `if const` fork, a tighter
  intrinsic, an extra specialization for one width. If it is measurably faster and
  stays correct, it belongs.
- **Correctness and portability are constraints, not competitors.** Every
  specialization must still pass the differential suite against the scalar oracle
  (section 7) and every backend must still compile. Speed never licenses a wrong
  answer or a dropped backend -- but within those bounds, push as hard as the
  hardware allows.
- **Measure, don't assume** (section 7): confirm the win with `cargo bench` and check
  the emitted asm. A "faster-looking" change that the benchmark doesn't confirm is not
  a win. The `target_feature` inlining traps in [performance.md](performance.md)
  section 11 routinely make "obviously faster" code slower.

The performance techniques themselves -- FMA-variant choice, ILP/critical-path
shortening, capability gating, cancellation avoidance -- live in
[performance.md](performance.md); this section is the *mandate* to apply them
aggressively when developing Thermite, including down to individual registers.

---

## 1. The repository

Workspace root `Cargo.toml`: `members = ["bin/*", "crates/*", "tests/*"]`,
edition 2024, MSRV 1.95, `resolver = "3"`. Publishable crates share
`version = "0.2.0-beta.0"` via `[workspace.package]`. Profiles: `release` is
`opt-level=3, lto=true, codegen-units=1`; `bench` is the same with `lto="fat"`;
`release-ffi` adds `strip=true, panic="abort"`.

| Path | What it is |
|---|---|
| `crates/thermite` | The core crate. Everything in sections 2-6 (`register/`, `vector/`, `math/`, `backend/`, `element/`) lives here. |
| `crates/thermite-macros` | All proc macros: `dispatch`, `dispatch_dyn`, `register_trait`, `vector_trait`, `vector_impl`, `array_impl`, `reduced_impl`, `inline_always`, `double_pump_impl`, `derive(HasIsa)`. |
| `crates/thermite-special` | `erf`/gamma/activations/elliptic, same `_p::<P>()` + `Specialized*` pattern as core math. |
| `crates/thermite-dual` | `Dual<V,N>` autodiff. Implements the `Specialized*Math` traits by chain rule. |
| `crates/thermite-compensated` | `Compensated<V>` double-double. Implements them via error-free transforms. WIP (`todo!()` in places). |
| `crates/thermite-geometry`, `-sdf`, `-complex`, `-blas`, `-bignum`, `-rng` | Companion crates built on the core traits. Some are mid-rewrite. |
| `crates/thermite-ffi` | C ABI `cdylib`. Nightly-only. See [ffi.md](ffi.md). |
| `crates/testing` | Internal differential-test helpers. |
| `tests/wasm-runner` | Thin wasmtime+WASI host that runs compiled libtest binaries (wired via `CARGO_TARGET_WASM32_WASIP1_RUNNER`). |
| `bin/docgen` | Generates register-coverage docs (incl. SVG). |
| `bin/remez` | Minimax (Remez) polynomial fitting -- where transcendental coefficients come from. |
| `bin/spirv_testing` + `bin/spirv_builder` + `bin/spirv_runner` | GPU path: kernels, rust-gpu compile to `.spv`, wgpu execution vs CPU reference. |

**Toolchain.** The crate builds on **stable**. `rust-toolchain.toml` is pinned to a
rust-gpu nightly only for the (currently inactive) spirv backend; the `justfile`
deliberately overrides to `stable`. Nightly is needed only for wasm tests, miri,
branch coverage, spirv, and the FFI crate.

### Where each layer lives (core crate)

| Path | Purpose |
|---|---|
| `register/mod.rs` | The `*Register` trait hierarchy: `CoreRegister` (233) -> `BitwiseRegister` (291) -> `Register` (488) -> `NumericRegister` (1623) -> `FloatRegister` (2084); also `BitshiftRegister` (1325, `Element: IntegerElement`), `IntegerRegister`, `MaskRegister` (364), and the swizzle/concat/extend/blend traits. Methods are `fn(Storage<Self>, ...) -> Storage<Self>` -- no `&self`, no operators. Lane-wise defaults borrow storage via `Register::as_slice`/`as_mut_slice` (runtime-length slices; there is no array-typed borrow) and loop `0..Self::lanes()`. |
| `register/well_formed.rs`, `register/linalg.rs` | well-formedness bounds; linalg register ops. |
| `register/array.rs` | `ArrayRegister<R,N>` -- packs `[R; N]` to emulate a wider width. |
| `register/reduced.rs` | `ReducedRegister<R,N>` -- masks upper lanes to emulate a narrower width. |
| `vector/mod.rs` | The `*Vector` traits (`GenericVector` 512, `BitwiseVector` 1092, `BitshiftVector` 1182, `NumericVector`, `SignedVector`, `FloatVector`, ...), each tagged `#[thermite_macros::vector_trait]`. |
| `vector/vector.rs` | `Vector<R>` (`#[repr(transparent)] pub struct Vector<R>(pub Storage<R>)`) and its trait impls that delegate to the register, plus operator overloads. |
| `vector/{splat,num,ops,streaming,unaligned}.rs` | `const_splat!`/`const_new!`, num-traits glue, masked-op traits, slice iterators. |
| `backend/<isa>/registers/<type>.rs` | One file per concrete register (`u32x8.rs`, `f32x4.rs`, ...) holding its trait impls. |
| `backend/<isa>/polyfills/{bits,cmp,casts,divider,math}.rs` | Per-backend software fills for missing intrinsics. |
| `backend/generic/polyfills/` | Portable polyfills shared by all backends (`bits.rs`, `sort.rs`, `casts.rs`, `divider.rs`, `math.rs`). |
| `backend/<isa>/mod.rs` | Defines `pub mod arch { pub use super::polyfills::*; pub use crate::backend::x86::sse2::*; }` -- the `arch::` namespace each register file calls into. |
| `backend/x86.rs` | x86 intrinsic re-export layers (`sse2`, ...) shared across the x86 tiers. |
| `math/mod.rs` | `decl_math!` definition + invocations declaring the public math-trait surface. |
| `math/specialized/{ps,pd,generic}.rs` | The kernels: `ps.rs` = f32, `pd.rs` = f64, `generic.rs` = element-agnostic helpers. |
| `math/{policy,consts,scalar}.rs`, `math/algorithms/` | `Policy` system, `FloatConsts`, scalar surface, generic numerics (`newtons_method`, `sum_f`, `reduce_in_place`). |
| `element/mod.rs`, `element/float/{mod,spec}.rs` | `Element`/`IntegerElement` (138)/`FloatElement` (49)/`FloatElementWithBits` (135) and per-element bit constants. |

---

## 2. The macro toolbox

You will not write masked variants, dispatch trampolines, or per-width
boilerplate by hand. Know which macro owns which generation step.

| Macro | Applied to | Generates |
|---|---|---|
| `#[thermite_macros::register_trait]` | a `*Register` trait def | For each method marked `#[conditional]` or `#[masked]`, the `_c`/`_m`/`_z` siblings (mask-first arg order) as provided methods; also adds `#[inline(always)]` to any method that has a default body. |
| `#[conditional]` / `#[masked]` (markers) | a method in the above | `#[conditional]` -> `_c` + `_m` + `_z`... see the exact rule below. `#[masked]` -> the same generation but is the "always generate" marker. Methods whose return type is mask-ineligible are skipped. |
| `#[thermite_macros::vector_trait]` | a `*Vector` trait def | Same masked-variant expansion at the vector layer. |
| `#[thermite_macros::vector_impl]` | `impl ... for Vector<R>` | Fills empty method bodies with `Vector(R::method(self.0, ...))` delegation + the masked siblings. |
| `#[thermite_macros::array_impl]` | `impl ... for ArrayRegister<R,N>` | Element-wise delegation of each method to the inner `R`. You write empty stubs; it writes the loop. |
| `#[thermite_macros::reduced_impl]` | `impl ... for ReducedRegister<R,N>` | Delegates to the wider `R` then re-masks dead upper lanes. Empty stubs again. |
| `#[thermite_macros::inline_always]` | any impl block | `#[inline(always)]` on every method (the universal tag on backend impls). |
| `#[thermite_macros::double_pump_impl]` | `impl ... for DoublePumpRegister<R>` | Legacy double-pump delegation (the pattern is mostly superseded by `ArrayRegister`). |
| `#[thermite::dispatch(S)]` / `(Self)` | fn / impl / mod generic over `S: HasIsa` | Per-backend `#[target_feature]` trampolines + a `match <S as HasIsa>::ISA` that folds at monomorphization. `#[skip_dispatch]` opts a method out. |
| `thermite::dispatch_dyn!(for<S> ...)` | an expression | Runtime `InstructionSet::get()` selection; rewrites bare `f32xN`/`f32x4`/... to `Vector<S::...>` inside the body. Signature must be ISA-agnostic. Call form: `dispatch_dyn!(func(args))` / `dispatch_dyn!(for<S> expr)` dispatches a `#[dispatch]` fn directly (match only, no trampolines). |
| `decl_math! { ... }` (in `math/mod.rs`) | a list of math signatures | The `*MathWithPolicy` trait (`_p::<P>()`), the default-policy `*Math` trait, the `scalar_*` surface on `f32`/`f64`, and the blanket impl delegating to `Specialized*Math<E>`. |
| `const_splat!` / `const_new!` (`vector/splat.rs`) | a const expr | Compile-time splat / per-lane const vector. **Use these for bitwise/coefficient constants** (e.g. `const_splat!(u32: 0x5555_5555)`), never a bare `const`. |

### 2a. What `register_trait` actually generates

Source: `crates/thermite-macros/src/internal.rs:101` (`register_trait_inner`).
For a marked method it emits the `_c`/`_m`/`_z` siblings as *provided* methods on
the trait. The mask is inserted as the **first** argument; for `_m` a `src` is
inserted *before* the mask (so the final order is `src, mask, ...original`):

```rust
// You write, inside  #[thermite_macros::register_trait] pub trait BitwiseRegister:
#[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self>;

// The macro appends (paraphrasing the real expansion):
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
        Self::bitxor_m(Self::EMPTY, mask, lhs, rhs)          // fall back to _m with a zero src
    }
}
```

Key facts that bite if you forget them:

- **The first existing argument is the "keep" value for `_c`** (`this` in the
  macro). For a unary op like `sqrt`, `_c` keeps `self`/the input where the mask is
  false. The macro panics if the method has zero arguments.
- A backend can **override** any of these provided variants with a faster encoding
  (AVX-512 maps them to a single masked instruction). The defaults above are the
  pre-AVX512 lowering: `blendv`, or a `bitand`-with-mask for `_z` when the mask is
  full-width (`HAS_EQUAL_SIZE_MASK`).
- A method whose return type is **not** a plain `Storage<Self>` (a tuple, an
  `Element`, a differently-typed `Storage`) is *ineligible* and gets no variants --
  this is why `morton_deinterleave` (tuple return) below carries no marker.

`vector_trait` / `vector_impl` mirror this at the `Vector<R>` layer, generating the
public `v.op_c(mask, ...)` surface that forwards to `R::op_c(...)`.

### 2b. What `dispatch` / `dispatch_dyn!` generate

`#[dispatch(S)]` (`thermite-macros/src/dispatch.rs`) moves the body into an
`#[inline(always)]` inner copy, emits one `#[target_feature(enable = "...")]`
trampoline per backend, and replaces the outer body with
`match <S as HasIsa>::ISA { ... }`. Because `ISA` is a const, the match folds away
at monomorphization -- no runtime branch. The backend set it iterates is the
authoritative target list:

```
x86 build:   Scalar("")  X86V1("sse2")  X86V2("sse4.2")  X86V3("avx2,fma"[+",f16c"])
wasm build:  Scalar("")  WASM32("simd128")
```

`dispatch_dyn!(for<S> |...| { ... })` is the runtime entry: it calls
`InstructionSet::get()` (cached) and runs the body under the chosen backend. Inside
the body, bare width names (`f32xN`, `f32x4`, `i32x8`, `usizex4`, ... -- every
`Simd` associated type) are rewritten to `Vector<S::...>`. Explicit `S::f32x4` or
multi-segment paths are left alone. The signature must be ISA-agnostic (scalars,
slices, `Vec`); a SIMD type there has nowhere to come from. See
[slices-and-dispatch.md](slices-and-dispatch.md) for the user-facing contract.

`dispatch_dyn!` also has a **call form** for invoking a `#[dispatch]` function
directly: `dispatch_dyn!(dot(a, b))` (backend injected as the callee's only generic
argument) or `dispatch_dyn!(for<S> dot::<S, f32>(a, b))` (token-level substitution
of `S` in the call expression). It expands to just the runtime
`InstructionSet::get()` match -- no trampolines, no inner fn -- because a
`#[dispatch]` callee already carries its own `#[target_feature]` codegen. The
supported shape is a single dispatched call, including method calls on a receiver
(`for<S> kernel.run::<S>(&data)` against a `#[dispatch(S)] impl` block): the
substitution technically accepts any expression, but that is deliberately
undocumented (non-callee code inside the macro compiles without target
features). Both forms are parsed in
`DispatchDynInput`; codegen is `dispatch_dyn_call` vs `dispatch_dyn_closure` in
`thermite-macros/src/dispatch.rs`.

### 2c. What `decl_math!` generates

Source: `crates/thermite/src/math/mod.rs:54`. For each declared function it emits
four things. You declare with an unusual bracket syntax -- generics go in
`[ ... ][ names ]` rather than `<...>` (the macro re-splices them):

```rust
decl_math! {
    trait Core<FloatElement>: FloatVector {
        // no generics:        name [ generic params ][ names ]( args ) -> ret
        fn reciprocal[][](self: Self) -> Self;
        // with a const generic, listed once as params and once as names:
        #[skip_dispatch] fn poly[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;
    }
}
```

expands (paraphrased) to:

```rust
pub trait CoreMathWithPolicy: FloatVector {
    fn reciprocal_p<P: Policy>(self: Self) -> Self;
    fn poly_p<P: Policy, const N: usize>(self: Self, coeffs: &[Self::Element; N]) -> Self;
}
pub trait CoreMath: CoreMathWithPolicy {
    #[inline(always)] fn reciprocal(self) -> Self { Self::reciprocal_p::<DefaultPolicy>(self) }
    #[inline(always)] fn poly<const N: usize>(self, c: &[Self::Element; N]) -> Self { Self::poly_p::<DefaultPolicy, N>(self, c) }
}
impl<M> CoreMath for M where M: CoreMathWithPolicy {}

// blanket impl: any float vector whose element has a Specialized kernel gets it:
impl<E: FloatElement, V: FloatVector<Element = E>> CoreMathWithPolicy for V
    where V: specialized::SpecializedCoreMath<E>
{
    #[inline(always)] fn reciprocal_p<P: Policy>(self) -> Self {
        <V as specialized::SpecializedCoreMath<E>>::reciprocal::<P>(self)
    }
    // ... poly_p likewise ...
}
```

Plus a `ScalarMathWithPolicy`/`ScalarMath` aggregate implemented on bare `f32`/`f64`
with every method `scalar_`-prefixed. The whole trait and impl are themselves
wrapped in `#[thermite_macros::dispatch(Self, thermite = "crate")]`, so each `_p`
gets per-ISA codegen automatically (except where `#[skip_dispatch]` is set -- used
on `poly`/`poly_rev`/`poly_rational`, which must inline rather than cross a dispatch
boundary).

The upshot for a contributor: **declaring a function in `decl_math!` does not
implement it.** It creates the public surface and routes it to
`Specialized<Family>Math<E>::<name>`, which you implement per element type in
`ps.rs`/`pd.rs` (section 6).

---

## 3. The backend and polyfill system

A register impl rarely calls a raw `core::arch` intrinsic directly. It calls
through its backend's **`arch` namespace**, which fuses two separately-inherited
layers behind one `arch::` prefix:

1. **Real intrinsics**, organized as an ISA ladder in `backend/x86.rs`.
2. **Polyfills** -- software fills for ops a given ISA lacks -- organized as a
   parallel inheritance chain under each backend's `polyfills/` directory.

Understanding both chains, and that they *compose*, is the difference between
"where do I put this helper" being obvious vs guesswork.

### 3a. The `arch` namespace per backend

Each backend's `mod.rs` defines:

```rust
// backend/x86_v3/mod.rs
pub mod arch {
    pub use super::polyfills::*;             // this backend's polyfills (+ all inherited ones)
    pub use crate::backend::x86::avx2::*;    // the real intrinsics available at this ISA level
}
```

The three x86 tiers differ only in the second line -- the cumulative real-intrinsic
set they expose:

| Backend | `arch` real intrinsics | `arch` polyfills |
|---|---|---|
| `x86_v1` | `backend::x86::sse2::*` | own + generic |
| `x86_v2` | `backend::x86::sse42::*` | own + v1 + generic |
| `x86_v3` | `backend::x86::avx2::*` | own + v2 + v1 + generic |
| `wasm` | `core::arch::wasm32::*` (or `wasm64` on nightly) | own + generic |

A register file does `use super::arch::*` (or `super::super::arch::*`), so a single
`arch::_mm256_xor_si256(...)` or `arch::_mm256_blendv_epi32x_v3(...)` resolves to a
real intrinsic *or* a polyfill transparently -- the call site can't tell which, and
shouldn't care. That is the whole point: a backend's register code is written
against one flat namespace of "things I can call at this ISA level."

### 3b. The real-intrinsic ISA ladder (`backend/x86.rs`)

`backend/x86.rs` re-exports `core::arch` intrinsics in nested modules, each
`pub use`-ing the tier below it so the available set grows monotonically:

```
sse -> sse2 -> sse3 -> ssse3 -> sse41 -> sse42
                                   \-> avx (= f16c + sse42) -> avx2 (= avx + fma)
                                                                  \-> avx512f -> tiers{1,2,3,4}
```

So `sse42::*` already contains everything from `sse` up, and `avx2::*` contains all
of SSE + AVX + FMA. This is why `x86_v3`'s `arch` only names `avx2::*` yet can still
use `_mm_xor_si128` (an SSE2 instruction) on a 128-bit half. Optional CPU
sub-features layer in by `cfg`: `avx2-pclmul` adds `_mm_clmulepi64_si128` (powers
the CLMUL 2D-Morton path), `avx2-f16c` adds the `f16c` conversions -- both also add
the feature to the dispatched `#[target_feature]` set so the intrinsic is callable.
AVX-512 is split into `tiers::tier1..4` (CD; +BW/DQ; +VBMI/VBMI2/VNNI/BITALG/GFNI/...;
+BF16), matching the `avx512-tier1..4` crate features.

### 3c. The polyfill inheritance chain

Each backend's `polyfills/mod.rs` starts by re-exporting the **next lower backend's**
polyfills, then declares its own (split by domain: `bits.rs`, `casts.rs`, `cmp.rs`,
`divider.rs`, `math.rs`; the generic set adds `compress.rs`, `sort.rs`):

```rust
// backend/x86_v1/polyfills/mod.rs
pub use crate::backend::generic::polyfills::*;     // portable, any-backend fills
// backend/x86_v2/polyfills/mod.rs
pub use crate::backend::x86_v1::polyfills::*;      // v2 inherits v1 (-> generic)
// backend/x86_v3/polyfills/mod.rs
pub use crate::backend::x86_v2::polyfills::*;      // v3 inherits v2 (-> v1 -> generic)
// backend/wasm/polyfills/mod.rs
pub use crate::backend::generic::polyfills::*;     // wasm inherits generic only
```

So the chain is **generic ⊂ v1 ⊂ v2 ⊂ v3** (and **generic ⊂ wasm**). This is the
"some v1/v2 polyfills are still used in v3" fact: a v3 256-bit op that processes its
two 128-bit halves reuses `_mm_blendv_epi8x_v1` (a pure-SSE2 bitwise-select blend
defined in v1's polyfills) or `_mm_permutevarx_epi32x_v2` (an SSSE3 `pshufb`-based
permute defined in v2's), because v3's polyfill namespace re-exports them. You write
a helper once, at the lowest tier that can express it, and every higher tier inherits
it for free.

Naming convention: a polyfill mimics the intrinsic it stands in for, with an
`x_v<n>` suffix marking "polyfill, introduced at tier n" -- `_mm_blendv_epi8x_v1`,
`_mm_popcnt_epi32x_v2`, `_mm256_srai_epi64x_v3`. At a register call site they read
just like real intrinsics, which is the intent.

### 3d. Generic polyfills (`backend/generic/polyfills/`)

These sit at the **bottom** of the chain and are different in kind: they are generic
over `R: Register` (or a sub-trait like `R: UnsignedIntegerRegister`) and are written
**entirely in register-trait operations** (`R::shl`, `R::bitand`, `R::bitor`, ...),
never raw intrinsics. Because they bottom out in trait methods, they compile for
**every** backend -- x86 tiers, wasm, spirv, scalar -- which is why each backend's
polyfill chain ends at `generic`. They are the "better than scalar, portable"
fallbacks: the N-dimensional `morton_cascade`/`reverse_morton_cascade` shift/mask
bit-spread, the `compress` left-pack, sorting networks (`sort.rs`), generic
`casts`/`divider` paths. A backend that has a hardware shortcut overrides the op in
its register impl; a backend that doesn't delegates to the generic cascade. (The
generic Morton even lets a CLMUL `N == 2` fast path delegate every *other* dimension
count straight to `morton_cascade`.)

### 3e. Where to put a new helper

- **Expressible with only register-trait ops, useful on any backend** -> a
  `backend/generic/polyfills/` function generic over `R`. Every backend inherits it;
  backends with a hardware path override in their register impl.
- **Needs ISA-specific intrinsics** -> the polyfill file of the **lowest tier whose
  `arch` intrinsics suffice**. Put a `pshufb`-based helper in **v2** (SSSE3), not v1
  (SSE2 lacks `pshufb`); v3 inherits it automatically for 128-bit-half work and adds
  its own `_mm256_*x_v3` for the full 256-bit width. A pure shift/and/or helper can
  live in **v1** and serve all three tiers.
- **Domain**: bit twiddling -> `bits.rs`, comparisons/masks -> `cmp.rs`, numeric
  casts -> `casts.rs`, division -> `divider.rs`, float math -> `math.rs`.

> **The tier discipline that bites:** a polyfill at tier N may use **only**
> intrinsics available at tier N, because every backend at tier >= N pulls it in. Put
> an SSE4.1-only helper in v1's polyfills and the v1 (SSE2) build fails to compile.
> When unsure, push the helper *up* to the first tier whose `arch::` actually has the
> instructions, or *down* to generic if it can be expressed in register ops.

This is the system section 4's Morton step 4 walks through concretely, and section
8's "x86_v1 is real and limited" gotcha guards.

## 4. Worked example: accelerated Morton-code interleave

Morton (Z-order) encoding interleaves the bits of two integers: bit `i` of `a`
goes to bit `2i`, bit `i` of `b` goes to bit `2i+1`. It is a per-lane integer bit
operation -- the same shape as the existing `reverse_bits` and `count_ones`, which
makes those the ground-truth template to copy. Walk the full path; section 5 then
generalizes it.

### Step 0 -- pick the layer and the trait

It produces a value per lane from integer lanes, with no precision/policy
dimension, so it is a **register primitive**, not a math function. `BitshiftRegister`
is bounded `Register<Element: IntegerElement>` (`register/mod.rs:1325`) and already
hosts the bit-twiddling family (`reverse_bits`, `bshli`/`bshri`, rotates) -- put
`morton_interleave` there. The inverse `morton_deinterleave` returns *two* lanes
(`(Storage<Self>, Storage<Self>)`); a tuple return is mask-ineligible (section 2a),
so it gets no marker and no `_c`/`_m`/`_z`.

### Step 1 -- declare on the register trait (`register/mod.rs`)

```rust
#[rustfmt::skip] #[thermite_macros::register_trait]
pub trait BitshiftRegister: Register<Element: IntegerElement> {
    // ... existing reverse_bits, shli/shri, rol/ror ...

    /// Interleave the low bits of `a` (even positions) and `b` (odd positions).
    #[conditional]
    fn morton_interleave(a: Storage<Self>, b: Storage<Self>) -> Storage<Self>;

    /// Inverse of `morton_interleave`: split even/odd bits back out. Tuple return
    /// is not mask-eligible, so no marker.
    fn morton_deinterleave(value: Storage<Self>) -> (Storage<Self>, Storage<Self>);
}
```

`#[conditional]` makes `register_trait` emit `morton_interleave_c`/`_m`/`_z` (section
2a). **You provide no body here** -- the method is abstract; backends supply it. A
method *can* carry a default body (`bitandnot`, `ternlog`, `shli` do) but only when
a correct portable definition exists in terms of other register ops; Morton's fast
paths are hardware-specific, so leave it abstract and implement per backend.

### Step 2 -- expose on the vector trait and impl

```rust
// vector/mod.rs, inside  #[thermite_macros::vector_trait] pub trait BitshiftVector
#[conditional] fn morton_interleave(self, other: Self) -> Self;
fn morton_deinterleave(self) -> (Self, Self);
```

```rust
// vector/vector.rs, inside the  #[vector_impl] impl ... BitshiftVector for Vector<R>
#[conditional] fn morton_interleave(self, other: Self) -> Self {}   // body filled by macro
fn morton_deinterleave(self) -> (Self, Self) {}                     // -> delegates to R::...
```

The empty braces are intentional: `vector_impl` rewrites them to
`Vector(R::morton_interleave(self.0, other.0))` and generates the masked siblings.
Match the register signature exactly or the expansion mis-types.

### Step 3 -- emulated widths come almost free

In each `#[array_impl]` / `#[reduced_impl]` block for `BitshiftRegister`
(`register/array.rs`, `register/reduced.rs`), add the same empty stubs:

```rust
#[conditional] fn morton_interleave(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
fn morton_deinterleave(v: Storage<Self>) -> (Storage<Self>, Storage<Self>) {}
```

`array_impl` writes the per-inner-register loop; `reduced_impl` delegates to the
wide register and re-masks. This is how `f32x8`-on-SSE (= `ArrayRegister<_,2>`) and
a reduced width (`u32x3`) light up without a hand-written Morton.

### Step 4 -- implement per backend (the real work)

Each backend needs a real body in `backend/<isa>/registers/<inttype>.rs`, for every
integer register width. Copy how `reverse_bits`/`count_ones` are done in those same
files -- they are the exact same shape (per-lane integer transform).

**Scalar** (`backend/scalar/cpu/unsigned.rs`, and `signed.rs`). The integer impls
are generated by a width macro whose entries delegate to std inherent methods, e.g.
`fn reverse_bits(value) -> Storage { value.reverse_bits() }` and
`fn count_ones(value) -> Storage { value.count_ones() as _ }`. Morton has no std
method, so write the classic magic-number bit-spread in plain Rust (spread each
operand with the `0x5555.../0x3333.../0x0f0f...` mask cascade, then
`a_spread | (b_spread << 1)`). The scalar backend is the **differential-test
oracle**, so make it obviously correct, not clever.

**x86_v1 / SSE2** (`backend/x86_v1/registers/u32x4.rs`, etc.). SSE2 has no `pshufb`
and no `blendv`. Implement the magic cascade directly with `_mm_slli_epi32` /
`_mm_srli_epi32` / `_mm_and_si128` / `_mm_or_si128`, calling through the file's
`arch::` namespace (which re-exports `crate::backend::x86::sse2::*` plus this
backend's polyfills). If shared, factor the cascade into `backend/x86_v1/polyfills/bits.rs`.

**x86_v2 / SSE4.2** (`backend/x86_v2/registers/*.rs`). `pshufb`
(`_mm_shuffle_epi8`) is available: a 4-bit-nibble spread LUT applied with `pshufb`
is the fast path. Add `_mm_morton_interleave_epi32_v2` to
`backend/x86_v2/polyfills/bits.rs` and call it from the register file, mirroring how
`count_ones` calls its popcount polyfill there.

**x86_v3 / AVX2** (`backend/x86_v3/registers/u32x8.rs`, etc.). The 256-bit
analogue: `_mm256_shuffle_epi8` + a polyfill in `backend/x86_v3/polyfills/bits.rs`.
These register impl blocks are tagged `#[thermite_macros::inline_always]` and call
`unsafe { arch::_mm256_... }` -- follow that idiom exactly (look at how `bitxor` is
`unsafe { arch::_mm256_xor_si256(lhs, rhs) }`).

**wasm** (`backend/wasm/registers/*.rs`). SIMD128 has shifts and `v128.and/or`;
implement the magic cascade portably, or `i8x16.swizzle` for a LUT spread under
`relaxed-simd`.

**spirv** (`backend/scalar/spirv/` + `backend/spirv/registers/`) -- only if you need
the GPU path; each invocation is one lane, so the scalar cascade applies.

> **PDEP/PEXT caveat.** BMI2 `_pdep_u32`/`_pext_u32` are the textbook Morton
> instructions, but they are **scalar GPRs, not SIMD** -- there is no packed PDEP
> before AVX-512 GFNI/`vpermb` tricks. A SIMD Morton is the magic-number shift/mask
> spread (any backend) or a `pshufb`/`swizzle` nibble-LUT (v2/v3/wasm), *not* a
> single instruction. Don't reach for `_pdep` in a register impl.

### Step 5 -- test it across backends

Add a differential test alongside the existing bit-op tests in
`crates/thermite/tests/` (section 7). The harness runs your op on `Scalar`,
`X86V2`, `X86V3` and asserts they agree -- `Tol::Exact` for an integer bit op. Then
`just wasm-test` for the wasm path. A round-trip property
(`deinterleave(interleave(a,b)) == (a & low_mask, b & low_mask)`) is the cheapest
strong check.

---

## 5. Generalizing: checklists for any new operation

First classify -- the two kinds take different paths.

### A. A register primitive

(bit op, integer op, a new lane-shuffle, a capability-gated algebraic identity.) The
Morton path is the template:

1. Declare on the right `*Register` trait (`register/mod.rs`); marker
   `#[conditional]`/`#[masked]` only if the return is a single `Storage<Self>`.
2. Declare on the matching `*Vector` trait (`vector/mod.rs`) + empty stub in
   `vector/vector.rs`.
3. Empty stubs in the `array_impl`/`reduced_impl` blocks (`register/array.rs`,
   `register/reduced.rs`).
4. Real impl per backend in `backend/<isa>/registers/<type>.rs`, factoring shared
   logic into `backend/<isa>/polyfills/` or `backend/generic/polyfills/`.
5. Differential test.

   **Exception:** a purely algebraic identity expressible in existing ops (like
   `one_minus_sq`) can be a **provided default method** on the vector trait with no
   per-backend work -- and it is then inherited by `Compensated`/`Complex` wrappers
   for free. Prefer that when it applies ([performance.md](performance.md) section
   10). A new *required* register method does not get that for free, and adding one
   *breaks* the composite wrappers until they implement it.

### B. A precision-tunable math function

(a transcendental, a special function, a norm.) This rides the math machinery, not
the backends:

1. Add the signature to the appropriate `decl_math!` block in `math/mod.rs`
   (`Float`, `Core`, `Transcendental`, `Spatial`, or `Real`). Bracket-generics
   syntax (section 2c). The macro generates the public surface and routes it to
   `Specialized<Family>Math<E>`.
2. Implement the **f32 kernel** in `math/specialized/ps.rs` and the **f64 kernel**
   in `math/specialized/pd.rs`, on the `Specialized*Math<E>` trait. Element-agnostic
   helpers go in `generic.rs`.
3. Thread `Policy` through the kernel (section 6).
4. Put coefficients inline as `poly_p` / `poly_rev_p` arrays (derive with
   `bin/remez`); bit-pattern constants via `const_splat!`.
5. **Composites:** implement the same `Specialized*Math` entry in `thermite-dual`
   and `thermite-compensated` (section 6c) -- or generic code at those types fails
   to compile (or hits `todo!()`).
6. Accuracy-test under multiple policies, including an f32 case for any
   `HAS_APPROX_*`/`HAS_TRUE_FMA` branch.

---

## 6. The math-kernel layer in depth

### 5a. A real kernel, annotated -- `cbrt` for f64

`math/specialized/pd.rs`, the body of `SpecializedTranscendentalMath::cbrt`
(verbatim from source). This is the canonical shape: bit-twiddle an initial guess,
refine with a polynomial, then Newton/Halley steps, with policy-gated paths and a
policy-gated overflow check.

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

    let r = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !Self::HAS_TRUE_FMA } {
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

What to copy from this:

- **`if const { P::POLICY.precision.ge(...) }`** forks codegen at compile time. The
  `Best`-precision branch also fires when `!HAS_TRUE_FMA`, because the fast form
  relies on a fused `mul_add`. Note the deliberate mix: `mul_add` (always-fused) in
  the FMA-gated branch, `mul_adde` (estimating) for the final unconditional step.
- **`P::POLICY.check_overflow`** gates the NaN/inf/zero handling -- perf policies
  drop it. `flush_denormals::<P>()` at the top honors `P::POLICY.denormal_behavior`.
- **Coefficients are inline literals**; the f32 kernel in `ps.rs` uses *different*
  coefficients and often a different algorithm shape (more aggressive range
  reduction, `ExtendedPrecision` upcasts). f32 and f64 are separate hand-written
  kernels -- there is no shared generic body for transcendentals.
- **Bit constants always go through `const_splat!(u64: ...)`**, never a bare `const`.

The `Policy`/`PolicyParameters`/`PrecisionPolicy` types are in `math/policy.rs`
(see [math.md](math.md) for the field list and the preset table). `poly_p` /
`poly_rev_p` (hybrid Estrin/Horner with FMA) and the convergent-series helpers
(`newtons_method`, `sum_f`, `reduce_in_place`) are in `math/algorithms/` and
`Core`.

### 5b. The `Specialized*Math<E>` traits

`math/specialized/mod.rs` declares `SpecializedCoreMath<E>`,
`SpecializedTranscendentalMath<E>`, `SpecializedSpatialMath<E>`,
`SpecializedRealMath<E>` (and `Float`). They are **parameterized by the element
type** `E`, not the vector -- that is the hinge that lets composites implement them.
`ps.rs` implements them for f32 vectors, `pd.rs` for f64. The `decl_math!` blanket
impl (section 2c) is what connects "any `FloatVector<Element = E>` whose `E` has a
`Specialized*Math` impl" to the public `*Math` trait.

### 5c. Composites must implement the Specialized traits

A new math function is **not finished** until `Dual` and `Compensated` cover it,
because they implement the `Specialized*Math` traits independently rather than
inheriting a kernel.

`Dual` (`thermite-dual/src/math.rs`) computes the primal with the inner vector's
`_p` method and propagates derivatives by the chain rule. Real examples:

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
local derivative. **Most functions you do not need to touch**: `Dual` only overrides
primitives; `sin`, `cos`, `powi`, `lerp`, `smoothstep`, etc. fall out of the trait
*defaults* composed over dual arithmetic, and are differentiated automatically. Add
a `Dual` override only when (a) a dedicated inner primitive is cheaper/more accurate
than the default's composition, or (b) the default would differentiate through an
internal iteration you'd rather bypass with the analytic derivative (the
`inverse_smoothstep` implicit-function-theorem trick).

`Compensated` (`thermite-compensated/src/math.rs`) is different: it **re-derives** the
algorithm in double-double, e.g. `sin_cos` does compensated argument reduction
(`k = round(x * 2/pi)`, `r = x - k*(pi/2)` with the compensated `FRAC_PI_2`
constant) then a ~20-term series in `Compensated` arithmetic. Its trait coverage is
**partial** -- many masked/reduction ops are still `todo!()`. If your new function
can't be expressed from already-implemented compensated ops, you may add a `todo!()`
and note it, but flag it.

### 5d. Element constants the kernels rely on

`element/float/mod.rs` (`FloatElementWithBits`, 135) and `element/float/spec.rs`
expose the bit-layout constants kernels use: `EXP_BITS`, `MANTISSA_BITS`,
`EXP_BIAS`, `MAX_BIASED_EXP`, `SIGN_MASK`, `MANTISSA_MASK`, `DENORMAL_TRICK`, plus
the associated `Bits`/`SignedBits` integer views. (These are defined for f16/bf16
and the small f8 formats too, but the *active* math kernels are f32/f64 only.) When
a new function needs a per-element numeric constant that isn't a bit-layout fact,
prefer `FloatConsts` (`math/consts.rs`) if it exists; otherwise add a small
per-element trait with `const` members and `V::splat` it
([performance.md](performance.md) section 5) -- runtime math for a compile-time
constant is wasted work.

---

## 7. Build / test / verify loop

The `justfile` is the task runner (scoped to the `thermite` crate -- a
`--workspace` build fails because spirv/neon/wasm members don't build on an x86
host). It uses PowerShell on Windows.

| Recipe | Does |
|---|---|
| `just test` | `cargo +stable test -p thermite --features std --release`. Differential suites are slow in debug. **This is the gate.** |
| `just wasm-test [filter]` | Builds the `wasm-runner` host, then runs the wasm-applicable tests on `wasm32-wasip1` under wasmtime with `--features "wasm,std"`, `--test-threads=1`, LTO forced off. `+simd128,+relaxed-simd` come from `.cargo/config.toml`. Prereq: `rustup target add wasm32-wasip1`. Scope it: `just wasm-test "--test diff_ops"`. |
| `just cov` / `cov-collect` / `cov-missing` / `cov-summary` / `cov-percent` | `cargo-llvm-cov`. Always `--ignore-run-fail` (one accepted failure: `frldexp` denormal-flush under off-by-default `preserve_denormals`). **Never `--all-features`** -- backend features are mutually exclusive on one host. `cov-missing` is the authoritative uncovered-line list. |
| `just cov-branch` | Branch coverage (nightly). Reads low and misleading (LLVM doesn't credit diverging arms); informational only. |
| `just miri [filter]` | `cargo +nightly miri test`. Miri lacks x86 SIMD intrinsics, so V2/V3 error -- use it for scalar + generic `unsafe` (slice iterators, gather/scatter bounds). Filter to Miri-safe tests. |
| `just doc` / `doc-open` | rustdoc with the KaTeX header (`katex-header.html`). |
| `just sync-assets` | Re-propagate `LICENSE-*` to every crate and `katex-header.html` to the math crates. |
| `just bundle-skill` | Zip this skill for distribution. |

### 6a. The differential test harness

Tests live in `crates/thermite/tests/` as `diff_*.rs` files (`diff_ops`,
`diff_math`, `diff_mask`, `diff_polyfill`, `diff_cast`, `diff_linalg`,
`diff_swizzle`, `diff_slice`, `diff_gather`, `diff_divider`, ...), with a shared
`harness/`. The core idea: the **scalar backend is ground truth**, and every SIMD
backend (`X86V2`, `X86V3`, plus wasm where applicable) runs the same inputs and is
compared. Tolerance is an enum -- `Tol::Exact` (bit-exact; use for integer/bitwise
ops), `Tol::Ulp(n)`, `Tol::Rel(eps)`, `Tol::ExactOrNan` (for `min`/`max` where NaN
handling legitimately diverges by backend). The repo-root `TESTING.md` documents
the methodology and the defect log (the harness has caught dozens of
backend/polyfill divergences).

For a new op: add edge-case + random inputs and assert cross-backend agreement at
the right tolerance. Cover the masked variants (`_c`/`_m`/`_z`) explicitly -- they
are separate codegen. Add an f32 *and* f64 case when behavior differs by element
(approx rcp/rsqrt, FMA). For a math kernel, validate against a high-precision oracle
(libm, or the `Reference` policy) and keep the tolerance tight (e.g. `1e-13` for
f64) so a "harmless" algebraic transform is *proven* harmless.

### 6b. Cross-backend verification beyond the harness

- **wasm:** `just wasm-test "--test diff_ops"` to scope it. Runner is
  `tests/wasm-runner` (wasmtime+WASI), built fresh and wired by env var.
- **spirv (GPU):** `cargo run -p spirv_builder` compiles `bin/spirv_testing` kernels
  to `.spv` (disassembled to `out.spv.txt`); `cargo run -p spirv_runner` executes on
  the GPU via wgpu and compares to the CPU reference. The repo-root
  `before.spv.txt`/`after.spv.txt` are this disassembly, kept for diffing codegen
  changes. See `bin/spirv_testing/CLAUDE.md` for the full GPU setup (vector-type
  mapping, tier opcodes).
- **assembly:** confirm a hot kernel lowered as intended -- `cargo asm` on a
  monomorphic function, or `cargo rustc --release -- -C target-cpu=x86-64-v3 --emit asm`.
  Check for real `vfmadd*`/`vpshufb`, the expected `vsqrtps`/`vdivps` counts,
  broadcast constants, no spills. Measure with `cargo bench` (Criterion), never by
  eyeballing or running a compiled artifact directly ([performance.md](performance.md)
  section 12).
- **force a backend** for a focused bench: an explicit `#[target_feature(enable = "sse4.2")]`
  (or `"avx2,fma"`) wrapper, or instantiate `Vector<thermite::backend::scalar::Scalar>`
  for the scalar path.

### 6c. Tooling crates

- **`bin/remez`** -- generates the minimax polynomial coefficients you paste into
  `ps.rs`/`pd.rs`. When you add or retune a transcendental, this is where the
  coefficient arrays come from; record the interval and the achieved max error.
- **`bin/docgen`** -- generates register-coverage documentation (which ops each
  backend implements), including SVG.

### 6d. CI

`.github/workflows/` gates pushes: the main workflow runs the stable test suite
(`--features std`) and a wasm *build* check, and publishes the coverage percent.
`ffi_artifacts.yaml` builds the FFI `cdylib` (+ `cbindgen` header) on nightly with
the `release-ffi` profile, on demand. A rustdoc workflow deploys docs (with the
KaTeX header) for the `rewrite` branch.

---

## 8. Development gotchas

- **Masked variants are mask-first, and `src` precedes `mask` in `_m`.** The macros
  enforce `op_c(mask, ...)`, `op_m(src, mask, ...)`, `op_z(mask, ...)`. Any doc
  showing `add_c(b, mask)` is stale.
- **Marker eligibility.** Only a method returning a single `Storage<Self>` (register)
  / `Self` (vector) gets variants. Tuple/`Element`/foreign-`Storage` returns are
  skipped even if marked -- don't mark them and don't expect siblings.
- **A default-bodied register/vector method is auto-`#[inline(always)]`** (the macro
  adds it). Keep such defaults expressible purely in other trait ops; anything
  hardware-specific belongs in a backend impl.
- **The scalar backend is mandatory and is the oracle.** It must compile and be
  correct for every primitive; everything differential-tests against it. Keep it
  simple over fast.
- **x86_v1 is real and limited.** SSE2 has no `pshufb`, `blendv`, `round`, variable
  shift, or HW popcount -- those are polyfilled. Don't call a v2+ intrinsic from a v1
  register file. (Confirm the backend set in `thermite-macros/src/dispatch.rs`: x86 =
  Scalar/X86V1/X86V2/X86V3.)
- **Constants go through `const_splat!`/`const_new!`**, not bare `const`, so they
  splat with correct per-ISA codegen. Bit patterns: `const_splat!(u32: 0x5555_5555)`.
- **FMA in kernels -- know what the non-`e` form actually falls back to.** Estimating
  `mul_adde` is the default (real FMA where present, else `mul`+`add`). The always-fused
  `mul_add` family does **not** drop straight to scalar `libm::fma` on non-FMA backends:
  by default it lowers to a **vectorized emulated FMA** -- a Dekker/Veltkamp compensated
  split (see `_mm_fmadd_pdx_v1` in `backend/x86_v1/polyfills/math.rs`, `2^27+1` splitter).
  That emulation is slower than true FMA and **not bit-identical** to it, but it is still
  SIMD and far cheaper than `libm::fma`, and it gives single-rounding-quality accuracy.
  Only the `disable_fast_fma` feature (implied by `strict_ieee754`) replaces it with the
  exact-but-very-slow scalar `libm::fma`. So in a kernel, `mul_add` is a legitimate
  *accuracy* choice even without hardware FMA -- gate it behind `if const { V::HAS_TRUE_FMA }`
  only when you want to avoid the *emulation* cost, not out of fear of `libm`. The `cbrt`
  kernel above uses always-fused `mul_add` inside the FMA-gated branch and estimating
  `mul_adde` for the final unconditional step. ([performance.md](performance.md) sections 1-2.)
  When you actually need double-double precision, reaching for `thermite-compensated`
  directly is often cleaner than leaning on emulated FMA.
- **`target_feature` codegen traps that pass tests but tank benches:**
  `core::array::map`/`from_fn` and bare closures fail to inline in `target_feature`
  code and fall back to scalar -- hand-roll `while` loops. `#[inline(always)]` helpers
  are fine. ([performance.md](performance.md) section 11.)
- **Dead-code elimination can hide a monomorphization error** until a *different* call
  site exercises the path -- e.g. the F16C `cvtps_ph` rounding-immediate
  `static_assert_uimm_bits` only fires when the conversion is actually reached. Test
  every variant and every element type, not just one instantiation.
- **A new math function is incomplete until `Dual` and `Compensated` implement it.**
  They don't inherit kernels; generic code at those types won't compile (or hits
  `todo!()`) otherwise. (`Dual` usually needs only a small chain-rule override or
  nothing; `Compensated` may need a re-derivation.)
- **Companion crates are publish=false / WIP.** `Compensated`'s masked `FloatVector`
  ops are partly `todo!()`; some companions are mid-rewrite. Don't assume an op exists
  there because it exists in core.
- **`#[skip_dispatch]`** opts a `decl_math!` method (or any `#[dispatch]` item) out of
  the per-ISA trampoline -- used for things that must inline (`poly`/`poly_rev`/
  `poly_rational`) rather than cross a dispatch boundary.
