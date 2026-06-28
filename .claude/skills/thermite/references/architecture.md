# Architecture

**Understanding the architecture is worth your time even as a pure user.** The
Element -> Register -> Vector layering, the backend/ISA model, and the `Simd` type
hierarchy explain *why* the API is shaped the way it is: why you bound on traits
and let the caller pick the type, why composites like `Dual`/`Compensated` slot in
for free, why dispatch works, and what the more cryptic type-error messages and
associated-type names actually mean. Read this once and the rest of the library
stops looking like magic.

What you can safely skip unless you are *extending* Thermite (adding a backend or a
register implementation) is the lowest level -- the per-ISA register trait impls
and intrinsic plumbing. That is flagged as contributor territory below; the
conceptual layers above it are the part every user benefits from.

The code under `crates/thermite/src/` is the ultimate authority.

## Three layers

```
Element   ->   Register   ->   Vector
(scalar)       (hardware)      (user-facing)
```

- **Element** (`element/`): scalar types (`f32`, `f64`, `i32`, `u64`, ...) with trait
  bounds (`FloatElement`, `IntegerElement`, `FloatElementWithBits`). The element is
  the unit of math specialization -- which is why composites (`Dual`, `Compensated`)
  that are valid elements get the full math surface.
- **Register** (`register/`): the pure-functional backend layer. Every method is
  `fn(Storage<Self>, ...) -> Storage<Self>` -- no `&self`, no operators. This is what
  backends implement. Traits: `CoreRegister -> BitwiseRegister -> Register ->
  NumericRegister -> FloatRegister` (mirrors the vector hierarchy).
- **Vector** (`vector/`): the `#[repr(transparent)] struct Vector<R>(Storage<R>)`
  newtype users hold. Adds operator overloads, the ergonomic API, and the masked
  variant traits. The vector traits delegate to the register traits.

`Storage<R> = <R as CoreRegister>::Storage` is the actual data (an intrinsic like
`__m256`, a primitive, or an `ArrayRegister`). `Mask<R>` wraps `Storage<R::Mask>`.

## Masked variants and dispatch

`_c`/`_m`/`_z` variants are generated from a `#[conditional]` marker attribute on
register-trait methods (consumed by the `register_trait` attribute macro) and the
corresponding vector-trait macro layer. The `dispatch` macro
(`thermite_macros`) handles runtime ISA detection and `#[target_feature]`
propagation, creating a real function boundary while inner `#[inline(always)]` code
keeps its target-feature codegen.

## ISA backends (`backend/`)

| Backend | ISA | Native width | Notes |
|---|---|---|---|
| `scalar` | none | 1 | `Vector<f32>` etc.; `select_unpredictable` for blends |
| `x86_v1` | SSE2 | 128-bit | no blendv/round/pshufh/popcnt -- polyfilled |
| `x86_v2` | SSE4.2 + POPCNT | 128-bit | blendv, pshufb; no FMA/gather |
| `x86_v3` | AVX2 + FMA | 256-bit | `f32x8`/`f64x4` native, HW gather + FMA, `zeroupper` at boundaries |
| `wasm` | SIMD128 | 128-bit | `wasm` feature, `target_arch=wasm32/64` |
| `spirv` | SIMT (GPU) | 1 (per-invocation) | experimental; each shader invocation is one lane |
| *(planned)* | AVX-512 | 512-bit | k-register masks |

Detection is runtime via `InstructionSet::get()` (cached), selected by `dispatch`.
The x86 levels follow the x86-64 microarchitecture-level scheme (v1/v2/v3/v4).

## Emulating non-native widths

- **`ArrayRegister<R, N>`** (`register/array.rs`): packs `N` inner registers into
  `[R; N]` to emulate a wider vector (e.g. `f32x8` on SSE = `ArrayRegister<f32x4, 2>`).
  All register traits delegate element-wise. (The old `DoublePumpRegister` pattern was
  removed.)
- **`ReducedRegister`** (`register/reduced.rs`): wraps a wider register and masks the
  upper lanes to emulate a narrower type (e.g. `f32x3` in a 4-lane register). Better
  than scalar fallback. Cannot be nested inside itself (compile-time asserted).

## The `Simd` type hierarchy (`simd.rs`)

```
HasIsa -> NativeIsa -> NativeSimd -> Simd -> SizedSimd<F,I,U> -> FloatSimd<F>
```

`Simd` defines every fixed-width register alias (`f32x2..f64x16`, `usizex2..16`);
`NativeSimd` defines `f32xN`/`f64xN` (widest native). Backends populate these
associated types. Use them only when you need a width *by name* -- otherwise stay on
the `*Vector` traits.

## Where things live

| Path | Purpose |
|---|---|
| `simd.rs` | ISA traits, `Simd` type aliases, dispatch glue |
| `vector/` | `Vector<R>`, all vector traits, operators, `vector/ops.rs` masked-op traits |
| `register/` | register traits, `ArrayRegister`, `ReducedRegister`, linalg, well-formed |
| `mask.rs` | `Mask<R>`, `GenericMask`, `CastMask` |
| `math/` | policy system, `specialized/ps.rs` (f32), `pd.rs` (f64), `generic.rs`, algorithms |
| `element/` | element traits, bit-level constants |
| `backend/` | per-ISA register impls + `polyfills/` |
| `slice.rs`, `transform/` | slice iteration, bulk SIMD transforms |
| `divider/` | branchfree integer division |
| `isa.rs` | `InstructionSet` enum + runtime detection |
| `swizzle.rs` | swizzle/permute traits + `swizzle!` macro |

## Known rough edges (as a user, be aware)

- `is_power_of_two` reports 0 as a power of two.
- `mat4_inverse` only catches exactly-singular matrices.
- `copysign`'s default impl is suboptimal (backends may override with a bitwise form).

Adding a new backend, register, or math function is contributor territory and out
of scope for using Thermite; if you're curious, the `*Register` traits, the
`polyfills/` modules, and the `decl_math!` family in the source are the places to
look.
