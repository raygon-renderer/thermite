# Slice iteration, alignment, and ISA dispatch

## Iterating slices as SIMD vectors: `SimdSlice`

Extension trait (`crates/thermite/src/slice.rs`, in the prelude) for walking
`&[E]` / `&mut [E]` as vectors, parameterized by vector type `V` (`Element` must
match). Three strategies, each with a `_mut` variant:

```rust
use thermite::prelude::*;   // SimdSlice

// 1. try-aligned: NEVER panics. (head scalars, aligned middle, tail scalars).
//    The standard safe pattern for arbitrary slices -- use 95% of the time.
let (head, chunks, tail) = data.try_aligned_simd_iter::<V>();
for e in head   { /* scalar prologue */ }
for v in chunks { /* v: &V */ }
for e in tail   { /* scalar epilogue */ }
let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<V>();
for v in chunks { *v = transform(*v); }

// 2. aligned: PANICS unless exactly aligned with no remainder.
//    Only when you control the allocation (aligned container).
for v in data.aligned_simd_iter::<V>() { /* &V */ }
data.aligned_simd_iter_mut::<V>();

// 3. unaligned: any slice via unaligned loads/stores -> (iter, remainder).
let (iter, remainder) = data.unaligned_simd_iter::<V>();
let (iter, remainder) = data.unaligned_simd_iter_mut::<V>();

// streaming (non-temporal, cache-bypassing) for write-once bulk data:
for sv in data.streaming_simd_iter::<V>()      { let v = sv.load(); /* or sv.load_cached() */ }
for sv in data.streaming_simd_iter_mut::<V>()  { sv.store(v);       /* NT store */ }
```

- The `Unaligned` iterator trusts its constructor to have truncated to whole
  lanes; `next()` does not re-check.
- Lowest level: `V::align_slice(&[E]) -> (&[E], &[V], &[E])` -- what
  `try_aligned_simd_iter` is built on.

## Alignment

Native register alignment differs per backend (16B SSE, 32B AVX2) and is part of
the register type. Aligned loads/stores require it; `try_aligned`/`unaligned`
iterators handle mismatch. For `aligned` iteration, allocate through a
`NativeSimd`-aligned container so head/tail are empty.

## ISA dispatch

The dispatcher detects the CPU once (cached `InstructionSet::get()`), then runs
code compiled with the right `target_feature`. **Two distinct tools**: a bang
macro for the entry point, an attribute for library code.

### `dispatch_dyn!` -- runtime entry point (bang form)

Call from scalar code to runtime-select the best ISA and run a SIMD block under
it. Inside the body, bare width identifiers are rewritten to `Vector<S::...>` for
the chosen backend `S`; `f32xN` = widest native f32. The rewrite list
(`thermite-macros/src/dispatch.rs`) covers every `Simd` associated type: the
32/64-bit and `usize` families, the `xN` natives, the `x3`/`x3A` forms, and the
8/16-bit families (`u8x16`, `i8x16`, `u16x8`, ...). Only bare, unqualified,
generic-argument-free occurrences are rewritten.

```rust
// for<S> names the backend type, in scope inside the body. Default bound Simd3
// (= Simd plus 3-lane vector support: f32x3/i32x3/... incl. padded x3A forms).
let total: f32 = thermite::dispatch_dyn!(for<S> |data: &[f32]| -> f32 {
    let v = f32xN::splat(1.0);
    v.sum_elements()
});

// In-place over a slice; no explicit binding (S still available):
thermite::dispatch_dyn!(for<S> |data: &mut [f32]| {
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<f32xN>();
    for v in chunks { *v = v.sin(); }
});

// Custom backend bound, extra generics, where-clause all supported:
thermite::dispatch_dyn!(for<S: Simd> |data: &[f32]| -> f32 { /* ... */ });
```

**Crucial constraint:** parameters and return type must be ISA-agnostic
(scalars, slices, `Vec`, `bool`, ...) -- they are the I/O contract with the
scalar world. **Never** put `f32xN`/`Vector<S::f32x4>`/`Mask<...>` in the
signature; the caller can't know which backend was chosen. All SIMD work happens
inside: load from slices, process, store back.

#### The call form -- dispatch a `#[dispatch]` fn without closure syntax

When the SIMD work is already a `#[dispatch]` function, the callee carries its
own per-backend `#[target_feature]` trampolines, so the macro only emits the
runtime `InstructionSet::get()` match:

```rust
#[thermite::dispatch(S)]
fn dot<S: Simd>(a: &[f32], b: &[f32]) -> f32 { /* ... */ }

// Bare form: backend injected as the callee's ONLY generic argument.
let r = thermite::dispatch_dyn!(dot(&a, &b));

// for<S> form: S marks where the backend goes -- required when the callee has
// extra generics (partial turbofish is not legal Rust):
let r = thermite::dispatch_dyn!(for<S> scale::<S, f32>(&a, factor));

// for<S> also dispatches METHOD calls when the method is generic over the
// backend (a `#[dispatch(S)] impl` block):
#[thermite::dispatch(S)]
impl Kernel {
    fn run<S: Simd>(&self, data: &[f32]) -> f32 { /* ... */ }
}
let r = thermite::dispatch_dyn!(for<S> kernel.run::<S>(&data));
```

Arguments are ordinary expressions evaluated in the selected arm -- no closure
capture/reborrow rules. Caveats: keep it to a single dispatched call (free fn or
method), everything else outside the macro (other code inside compiles WITHOUT
target features); the callee must be `#[dispatch]` (a plain generic fn runs
correctly but with scalar-quality codegen); no bare-`f32xN` rewriting in the
call form; bounds on the binder (`for<S: Bound>`) are rejected -- the callee's
own bounds apply.

### `#[thermite::dispatch(...)]` -- per-backend codegen for library code (attribute)

A `#[proc_macro_attribute]`, not a bang macro. Put it on a `fn`, `impl` block,
`trait`, or `mod` generic over `S: HasIsa`/`Simd`. For each backend it generates
a `#[target_feature]` trampoline and turns the body into a
`match <S as HasIsa>::ISA { ... }` that LLVM folds away at monomorphization (no
runtime branch). This is how a reusable kernel gets correct per-ISA codegen.

```rust
// Whole impl block -- `Self` resolves at the impl level:
#[thermite::dispatch(Self)]
impl Kernel {
    pub fn run<S: FloatSimd<f32>>(&self, data: &mut [f32]) {
        let (_, chunks, _) = data.try_aligned_simd_iter_mut::<Vector<S::fxN>>();
        for v in chunks { *v = v.sin(); }
    }
    #[skip_dispatch]                 // opt an individual method out
    fn helper(&self) { /* ... */ }
}

// Single method WITH a receiver: pass the concrete Self type explicitly
// (the macro can't see the surrounding impl):
impl Kernel { #[thermite::dispatch(Kernel)] fn process(&self) { /* ... */ } }

// Free function generic over the backend:
#[thermite::dispatch(S)]
fn kernel<S: FloatSimd<f32>>(data: &mut [f32]) { /* uses S::fxN, S::f32x8, ... */ }
```

The two compose: `dispatch_dyn!` is the runtime boundary that picks `S`; the
`#[dispatch]` functions it calls carry the per-backend `target_feature` codegen.
On AVX2 the boundary also calls `zeroupper` (avoids AVX<->SSE transition
penalties). `#[inline(always)]` helpers inside a dispatched body keep their
target-feature codegen.

### When you need `FloatSimd<F>` / `Simd` instead of a `*Vector` bound

Only when you need the native width *by name* or a float type plus its matching
integer type together; otherwise prefer plain `V: FloatVector`.

```rust
use thermite::simd::FloatSimd;
use thermite::element::WellFormedFloatElement;

fn process<S, F>(data: &mut [F])
where
    F: WellFormedFloatElement,
    S: FloatSimd<F>,
    Vector<S::fxN>: thermite::math::TranscendentalMath,
{
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<Vector<S::fxN>>();
    for v in chunks { *v = v.sin(); }
}
```

Hierarchy: `HasIsa -> NativeIsa -> NativeSimd -> Simd -> SizedSimd<F,I,U> -> FloatSimd<F>`.
`Simd` defines the fixed-width register aliases (`f32x2..f64x16`, `usizex2..16`,
8/16-bit `i8x16`/`u8x16`/`i16x8`/`u16x8`); `NativeSimd` defines `f32xN`/`f64xN`
(widest native).
