# thermite-macros

Procedural macros for [Thermite](https://github.com/raygon-renderer/thermite),
the generic ISA-portable SIMD library.

**You should not depend on this crate directly.** Depend on `thermite` instead;
the user-facing macros are re-exported there, and `thermite` pins this crate to
an exact version so the generated code always matches its internals.

## What's in here

The macros users interact with (via `thermite`):

- **`#[dispatch]`** rewrites functions, impl blocks, or traits to propagate
  `#[target_feature]` statically across call boundaries, so runtime-dispatched
  SIMD code stays fully optimized without forcing everything to inline.
- **`dispatch_dyn!`** is the runtime dispatch boundary. It picks the best
  available ISA on the current CPU and monomorphizes the given closure for it.
  Also has
  a call form for invoking a `#[dispatch]` function directly:
  `dispatch_dyn!(my_kernel(a, b))`, or `dispatch_dyn!(for<S> my_kernel::<S, f32>(a, b))`
  when the callee has extra generics or is a method on a receiver
  (`for<S> kernel.run::<S>(&data)`).
- **`#[derive(HasIsa)]`** derives `thermite::simd::HasIsa` for types generic
  over a `Simd` ISA parameter, forwarding the `ISA` constant and the `Native`
  backend type from that parameter (first type parameter by default, override
  with `#[isa = S]`).

```rust
use thermite::prelude::*;

let result = thermite::dispatch_dyn!(for<S> || -> f32 {
    // Compiled per-ISA; runs at the widest width this CPU supports.
    (f32xN::splat(0.5) * f32xN::splat(2.0)).extract::<0>()
});
```

The remaining exports (`#[register_trait]`, `#[double_pump_impl]`,
`#[array_impl]`, `#[reduced_impl]`, `#[vector_trait]`, `#[vector_impl]`,
`#[inline_always]`) are internal codegen helpers used to build Thermite's
own backends and trait hierarchy.

See the [thermite documentation](https://docs.rs/thermite) for the full story
on dispatch and the vector trait system.

## License

MIT or Apache-2.0, at your option.
