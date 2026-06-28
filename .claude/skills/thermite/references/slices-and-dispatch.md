# Slice iteration, alignment, and ISA dispatch

## Iterating slices as SIMD vectors: `SimdSlice`

The `SimdSlice` extension trait (`crates/thermite/src/slice.rs`, in the prelude) is
the ergonomic way to walk a `&[E]` / `&mut [E]` as vectors. It is parameterized by
the vector type `V` (whose `Element` must match the slice element).

Three strategies, each with a shared and a `_mut` variant:

```rust
use thermite::prelude::*;   // brings SimdSlice into scope

// 1. try-aligned: NEVER panics. Splits into (head scalars, aligned middle, tail scalars).
//    This is the standard, safe pattern for arbitrary slices.
let (head, chunks, tail) = data.try_aligned_simd_iter::<V>();
for e in head   { /* scalar prologue */ }
for v in chunks { /* v: &V */ }
for e in tail   { /* scalar epilogue */ }

// mutable:
let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<V>();
for v in chunks { *v = transform(*v); }

// 2. aligned: PANICS if the slice isn't exactly aligned with no remainder.
//    Use only when you control the allocation (aligned container).
for v in data.aligned_simd_iter::<V>() { /* &V */ }
data.aligned_simd_iter_mut::<V>();

// 3. unaligned: handles any slice via unaligned loads/stores. Returns (iter, remainder).
let (iter, remainder) = data.unaligned_simd_iter::<V>();
let (iter, remainder) = data.unaligned_simd_iter_mut::<V>();

// streaming (non-temporal, cache-bypassing) variants for write-once bulk data:
for sv in data.streaming_simd_iter::<V>()      { let v = sv.load(); /* or sv.load_cached() */ }
for sv in data.streaming_simd_iter_mut::<V>()  { sv.store(v);       /* NT store */ }
```

Notes:
- `try_aligned_*` is what you want 95% of the time -- it can't panic and handles any
  length and any starting alignment.
- The `Unaligned` iterator trusts its constructor to have truncated the slice to a
  whole number of lanes; it does not re-check in `next()`.
- For the lowest level, `V::align_slice(&[E]) -> (&[E], &[V], &[E])` is what
  `try_aligned_simd_iter` is built on.

## Alignment

Native register alignment differs per backend (16-byte for SSE, 32-byte for AVX2).
A vector type's alignment is part of its register. Aligned loads/stores require the
pointer to satisfy it; the `try_aligned`/`unaligned` iterators handle the mismatch
for you. When allocating buffers you intend to iterate `aligned`, allocate through
a `NativeSimd`-aligned container so the head/tail are empty.

## ISA dispatch

The dispatcher detects the CPU's instruction set once (cached via
`InstructionSet::get()`), then runs a version of your code compiled with the right
`target_feature`. There are **two distinct tools** -- a function-like macro for the
entry point, and an attribute macro for library code:

### `dispatch_dyn!` -- the runtime entry point (a `#[proc_macro]`, bang form)

This is what you call from ordinary scalar code to *runtime-select* the best ISA
and run a SIMD block under it. Inside the body, bare width identifiers (`f32xN`,
`f32x4`, `f32x8`, `i32x4`, `f64xN`, ...) are rewritten to `Vector<S::...>` for the
chosen backend `S`. `f32xN` is the widest native f32 width.

```rust
// for<S> names the backend type, in scope inside the body. Default bound is Simd3.
let total: f32 = thermite::dispatch_dyn!(for<S> |data: &[f32]| -> f32 {
    let v = f32xN::splat(1.0);
    v.sum_elements()
});

// In-place over a slice; no explicit binding (S still available):
thermite::dispatch_dyn!(for<S> |data: &mut [f32]| {
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<f32xN>();
    for v in chunks { *v = v.sin(); }
});

// Custom backend bound, extra generics, where-clause are all supported:
thermite::dispatch_dyn!(for<S: Simd> |data: &[f32]| -> f32 { /* ... */ });
```

**Crucial constraint:** the macro's parameters and return type must be
ISA-agnostic (scalars, slices, `Vec`, `bool`, ...). They are the I/O contract with
the scalar world. **Never** put `f32xN`, `Vector<S::f32x4>`, `Mask<...>` in the
signature -- the caller can't know which backend was chosen, so a SIMD-typed
parameter or return would have nowhere to come from. All SIMD work happens inside
the body: load from slices, process, store back.

### `#[thermite::dispatch(...)]` -- per-backend codegen for library code (an attribute)

`dispatch` is a `#[proc_macro_attribute]`, not a bang macro. Put it on a `fn`,
`impl` block, `trait`, or `mod` whose code is generic over `S: HasIsa`/`Simd`. For
each backend it generates a `#[target_feature]` trampoline and turns the body into
a `match <S as HasIsa>::ISA { ... }` that LLVM folds away at monomorphization (no
runtime branch). This is how you give a reusable kernel correct per-ISA codegen.

```rust
// On a whole impl block -- `Self` resolves at the impl level:
#[thermite::dispatch(Self)]
impl Kernel {
    pub fn run<S: FloatSimd<f32>>(&self, data: &mut [f32]) {
        let (_, chunks, _) = data.try_aligned_simd_iter_mut::<Vector<S::fxN>>();
        for v in chunks { *v = v.sin(); }
    }
    #[skip_dispatch]                 // opt an individual method out
    fn helper(&self) { /* ... */ }
}

// On a single method WITH a receiver, pass the concrete Self type explicitly
// (the macro can't see the surrounding impl):
impl Kernel { #[thermite::dispatch(Kernel)] fn process(&self) { /* ... */ } }

// On a free function generic over the backend:
#[thermite::dispatch(S)]
fn kernel<S: FloatSimd<f32>>(data: &mut [f32]) { /* uses S::fxN, S::f32x8, ... */ }
```

The two compose: a `dispatch_dyn!` block is the runtime boundary that picks `S`,
and the `#[dispatch]`-annotated functions it calls carry the per-backend
`target_feature` codegen. On AVX2 the boundary also calls `zeroupper` to avoid
AVX<->SSE transition penalties. `#[inline(always)]` helpers inside a dispatched
body keep their target-feature codegen.

### When you need `FloatSimd<F>` / `Simd` instead of a `*Vector` bound

Reach for the `Simd` family only when you need the native width *by name* or both a
float type and its matching integer type together. Prefer a plain
`V: FloatVector` bound otherwise.

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

The `Simd` hierarchy is `HasIsa -> NativeIsa -> NativeSimd -> Simd -> SizedSimd<F,I,U> -> FloatSimd<F>`;
`Simd` defines all the fixed-width register aliases (`f32x2..f64x16`, `usizex2..16`),
`NativeSimd` defines `f32xN`/`f64xN` (widest native).
