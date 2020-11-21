Thermite SIMD: Melt your CPU
============================

Thermite is a WIP SIMD library focused on providing portable SIMD acceleratation of SoA (Structure of Arrays) algorithms, using consistent-length SIMD vectors for lockstep iteration and computation.

# Motivation and Goals

While working on Raygon renderer, I decided that I needed a state of the art high-performance SIMD vector library focused on faciliating SoA (Structure of Arrays) algorithms. Unfortunately, SIMDeez was not an option as it did not provide consistent-length vectors nor all the built-in functionality I've come to rely on. Alternatively, `packed_simd` was nightly-only and relied on the somewhat naive LLVM "platform intrinsics". I briefly explored the `Faster` library, as it did focus on SoA, but the iterator-based API was unsatisfactory and awkward to use.

Therefore, the only solution was to write my own, and thus Thermite was born.

TODO: Goals

# Features

* SSE2, SSE 4.1, AVX, AVX2 backends, with planned support for scalar, AVX512, WASM SIMD and ARM NEON backends.
* Extensive built-in vectorized math library.
* Compile-time monomorphisation with runtime selection
    * Aided by a `#[dispatch]` procedural macro to reduce bloat.
* Zero runtime overhead.
* Operator overloading on vector types.
* Abstracts over vector length, giving the same length to all vectors of an instruction set.
* Provides fast polyfills where necessary to provide the same API across all instruction sets.
* Highly optimized value cast routines between vector types where possible.
* Dedicated mask wrapper type with low-cost bitwise vector conversions built-in.

# Optimized Project Setup

For optimal performance, ensure you `Cargo.toml` profiles looks something like this:
```toml
[profile.dev]
opt-level = 2       # Required to use SIMD intrinsics internally

[profile.release]
opt-level = 3       # Should be at least 2; level 1 will not use SIMD intrinsics
lto = 'thin'        # 'fat' LTO may also improve things, but will increase compile time
codegen-units = 1   # Required for optimal inlining and optimizations

# optional release options depending on your project and preference
incremental = false # Release builds will take longer to compile, but inter-crate optimizations may work better
panic = 'abort'     # Very few functions in Thermite panic, but aborting will avoid the unwind mechanism overhead
```

# Misc. Usage Notes

* Vectors with 64-bit elements are approximately 2-4x slower than 32-bit vectors.
* Integer vectors are 2x slower on SSE2/AVX1, but nominal on SSE4.1 and AVX2. This compounds the first point.
* Casting floats to signed integers is faster than to unsigned integers.
* Equal-size Signed and and Unsigned integer vectors can be cast between each other at zero cost.
* Operations mixing float and integer types can incur a 1-cycle penalty on most modern CPUs.
* Integer division currently can only be done with a scalar fallback, so it's not recommended.
* Dividing integer vectors by constant uniform divisors should use `SimdIntVector::div_const`
* When reusing masks for `all`/`any`/`none` queries, consider using the bitmask directly to avoid recomputing.
* Avoid casting between differently-sized types in hot loops.
* Avoid extracting and replacing elements.
* LLVM will inline many math functions and const-eval as much as possible, but only if it was called in the same instruction-set context.

# Cargo `--features`

### `alloc` (enabled by default)

The `alloc` feature enables aligned allocation of buffers suitable to reading/writing to with SIMD.

### `nightly`

The `nightly` feature enables nightly-only optimizations such as accelerated half-precision encoding/decoding.

### `math` (enabled by default)

Enables the vectorized math modules

### `rng`

Enables the vectorized random number modules