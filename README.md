Thermite SIMD: Melt your CPU
============================

Thermite is a WIP SIMD library focused on providing portable SIMD acceleratation of SoA (Structure of Arrays) algorithms, using consistent-length SIMD vectors for lockstep iteration and computation.

Partially inspired by SIMDeez, Thermite supports static compilation to many instruction sets and runtime selection of the most performant instruction set, allowing you to write a single algorithm once and have it work on hardware ranging from early 2000s CPUs with only SSE2 all this way to modern cutting-edge CPUs with AVX2 and even AVX512.

However, unlike SIMDeez, all vectors within a supported instruction set have the same length, so you don't have to worry about how to handle **8** `u16` values alongside **4** `f32` values, if your data contains only **4** `u16` and **4** `f32`. In Thermite, there is only `Vu16` and `Vf32`, and they are the same length, allowing for easy loading and interop with other data types.

Furthermore, unlike `packed_simd`, Thermite compiles on Stable Rust, though there is a `nightly` feature flag for hardware-accelerated half-precision float conversions. (It's currently marked unstable for some reason).

Thermite also provides a set of high-performance vectorized special math functions that can take advantage of all instruction sets, with specialized versions for single and double precision floats.

# Motivation

TODO

# Optimization Setup

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

# Usage Notes

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

### `alloc`

The `alloc` feature enables aligned allocation of buffers suitable to reading/writing to with SIMD.

### `nightly`

The `nightly` feature enables nightly-only optimizations such as accelerated half-precision encoding/decoding.