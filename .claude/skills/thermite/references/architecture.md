# Architecture

The Element -> Register -> Vector layering, the backend/ISA model, and the `Simd`
hierarchy explain *why* you bound on traits and let the caller pick the type,
why composites slot in free, why dispatch works, and what cryptic type errors
mean. The lowest level (per-ISA register impls, intrinsic plumbing) is
contributor territory; the layers above matter to every user. The code under
`crates/thermite/src/` is the ultimate authority.

## Three layers

```
Element   ->   Register   ->   Vector
(scalar)       (hardware)      (user-facing)
```

- **Element** (`element/`): scalar types (`f32`, `f64`, `i32`, `u64`, ...) with
  trait bounds (`FloatElement`, `IntegerElement`, `FloatElementWithBits`). The
  element is the unit of math specialization -- which is why composites
  (`Dual`, `Compensated`) that are valid elements get the full math surface.
- **Register** (`register/`): pure-functional backend layer. Every method is
  `fn(Storage<Self>, ...) -> Storage<Self>` -- no `&self`, no operators. This is
  what backends implement -- and the ONLY thing it is for. **User and generic
  code must never call register methods directly**: the vector layer is the
  public API, and the register layer is raw `Storage` with no operators and no
  ergonomics. Traits: `CoreRegister ->
  BitwiseRegister -> Register -> NumericRegister -> FloatRegister` (mirrors the
  vector hierarchy).
- **Vector** (`vector/`): the `#[repr(transparent)] struct Vector<R>(Storage<R>)`
  newtype users hold; adds operators, the ergonomic API, masked-variant traits.
  Vector traits delegate to register traits.

`Storage<R> = <R as CoreRegister>::Storage` is the actual data (an intrinsic
like `__m256`, a primitive, or an `ArrayRegister`). `Mask<R>` wraps
`Storage<R::Mask>`.

## Masked variants and dispatch

`_c`/`_m`/`_z` variants are generated from a `#[conditional]` marker on
register-trait methods (consumed by the `register_trait` attribute macro) plus
the vector-trait macro layer. The `dispatch` macro (`thermite_macros`) handles
runtime ISA detection and `#[target_feature]` propagation: a real function
boundary, while inner `#[inline(always)]` code keeps target-feature codegen.

## ISA backends (`backend/`)

| Backend | ISA | Native width | Notes |
|---|---|---|---|
| `scalar` | none | 1 | `Vector<f32>` etc.; `select_unpredictable` for blends |
| `x86_v1` | SSE2 | 128-bit | no blendv/round/pshufh/popcnt -- polyfilled |
| `x86_v2` | SSE4.2 + POPCNT | 128-bit | blendv, pshufb; no FMA/gather |
| `x86_v3` | AVX2 + FMA | 256-bit | `f32x8`/`f64x4` native, HW gather + FMA, `zeroupper` at boundaries |
| `neon` | NEON/AdvSIMD | 128-bit | aarch64-only, **always on** (no feature to enable); NEON is baseline on aarch64 so dispatch is constant and the trampoline attr is a no-op |
| `wasm` | SIMD128 | 128-bit | `wasm` feature, `target_arch=wasm32/64` |
| `spirv` | SIMT (GPU) | 1 (per-invocation) | experimental; each shader invocation is one lane |
| *(planned)* | AVX-512 | 512-bit | k-register masks |

Runtime detection via `InstructionSet::get()` (cached), selected by `dispatch`.
x86 levels follow the x86-64 microarchitecture-level scheme (v1/v2/v3/v4).

## Emulating non-native widths

- **`ArrayRegister<R, N>`** (`register/array.rs`): `[R; N]` of inner registers
  emulates a wider vector (`f32x8` on SSE = `ArrayRegister<f32x4, 2>`); all
  register traits delegate element-wise. (Old `DoublePumpRegister` removed.)
- **`ReducedRegister`** (`register/reduced.rs`): wraps a wider register, masks
  upper lanes, to emulate a narrower type (`f32x3` in a 4-lane register). Better
  than scalar fallback. Cannot nest inside itself (compile-time asserted).

## The `Simd` type hierarchy (`simd.rs`)

```
HasIsa -> NativeIsa -> NativeSimd -> Simd -> SizedSimd<F,I,U> -> FloatSimd<F>
```

`Simd` defines every fixed-width register alias (`f32x2..f64x16`,
`usizex2..16`, plus 8/16-bit integer registers `i8x16`/`u8x16`/`i16x8`/`u16x8`);
`NativeSimd` defines `f32xN`/`f64xN` (widest native). Backends populate these.
Use only when you need a width *by name*; otherwise stay on the `*Vector`
traits.

`HasIsa` is the root and is implemented by far more than the backends -
`GenericVector: HasIsa`, so every vector and composite has one. Besides the
`ISA` constant it names `type Native: NativeIsa`, the backend type itself, so
a function bounded only on `V: FloatVector` can reach the per-ISA properties
below without threading a separate `S: Simd` parameter:

```rust
<V::Native as NativeIsa>::Registers          // architectural register count
<V::Native as NativeIsa>::Native32Width      // widest native 32-bit lane count
<V::Native as NativeIsa>::NativeAlignment    // the alignment marker type
V::Native::prefetch::<3, false>(ptr);        // and the CPU knobs below
```

**`Native` describes the value, not the host.** It answers "what executes
*this* vector". Sub-native slots (`i16x2`, `u8x2`) are `ArrayRegister`s of
scalar lanes on every backend, so they report `Scalar` even on an AVX2 host -
correct for that vector, wrong if you wanted the machine's register budget.
Emulated *wide* slots are fine (`f32x16<X86V1>` is four `F32x4V1`s, so it
reports `X86V1`). Tuning that is about the machine should still read the
dispatched `S`. Registers implement `HasIsa` directly (`CoreRegister` requires
it as a supertrait, so `#[dispatch(R)]` works over bare register types), and
`HasIsa::ISA` defaults to `<Self::Native as HasIsa>::ISA`, so a backend states
its ISA once. `Vector<R>` forwards its register's (`type Native = R::Native`).

`NativeIsa` also carries the whole-CPU knobs, each defaulting to
unsupported/no-op: `disable_denormals`/`enable_denormals` (+ the
`DisableDenormals` RAII guard), `zeroupper`, and **`prefetch`**:

```rust
S::prefetch::<LOCALITY, WRITE>(ptr);   // LOCALITY 0 (streaming) ..= 3 (keep in all levels)
```

`prefetcht0`-family on x86, `prfm` on aarch64, nothing on wasm/SPIR-V
(`S::HAS_PREFETCH` tells you which). **Safe for any pointer** -- a prefetch
never dereferences and cannot fault, so `base.wrapping_add(i)` needs no bounds
check, which is the point in a pointer-chasing loop.

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

## Known rough edges

- `is_power_of_two` reports 0 as a power of two.
- `mat4_inverse` only catches exactly-singular matrices.
- `copysign`'s default impl is suboptimal (backends may override bitwise).

Adding a backend/register/math function: see [development.md](development.md)
(the `*Register` traits, `polyfills/`, and the `decl_math!` family).
