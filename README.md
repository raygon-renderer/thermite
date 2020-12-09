Thermite SIMD: Melt your CPU
============================

Thermite is a WIP SIMD library focused on providing portable SIMD acceleration of SoA (Structure of Arrays) algorithms, using consistent-length<sup>1</sup> SIMD vectors for lockstep iteration and computation.

Thermite provides highly optimized **feature-rich backends** for SSE2, SSE4.2, AVX and AVX2, with planned support for AVX512, Aarch64, and WASM SIMD extensions.

In addition to that, Thermite includes a highly optimized **vectorized math library** with many special math functions and algorithms, specialized for both single and double precision.

<sub><small>
<sup>1</sup> All vectors in an instruction set are the same length, regardless of size.
</small></sub>

# Current Status

Refer to issue [#1](https://github.com/raygon-renderer/thermite/issues/1)

# Motivation and Goals

Thermite was conceived while working on Raygon renderer, when it was decided we needed a state of the art high-performance SIMD vector library focused on facilitating SoA algorithms. Using SIMD for AoS values was a nightmare, constantly shuffling vectors and performing unnecessary horizontal operations. We also weren't able to take advantage of AVX2 fully due to 3D vectors only using 3 or 4 lanes of a regular 128-bit register.

Using SIMDeez, `faster`, or redesigning `packed_simd` were all considered, but each has their flaws. SIMDeez is rather limited in functionality, and their handling of `target_feature` leaves much to be desired. `faster` fits well into the SoA paradigm, but the iterator-based API is rather unwieldy, and it is lacking many features. `packed_simd` isn't bad, but it's also missing many features and relies on the Nightly-only `"platform-intrinsic"`s, which can produce suboptimal code in some cases.

Therefore, the only solution was to write my own, and thus Thermite was born.

The primary goal of Thermite is to provide optimal codegen for every backend instruction set, and provide a consistent set of features on top of all of them, in such a way as to encourage using chunked SoA or AoSoA algorithms regardless of what data types you need. Furthermore, with the `#[dispatch]` macro, multiple instruction sets can be easily targetted within a single binary.

# Features

* SSE2, SSE4.2, AVX, AVX2 backends, with planned support for scalar, AVX512, WASM SIMD and ARM NEON backends.
* Extensive built-in vectorized math library.
* Compile-time policies to emphasize precision, performance or code size (useful for WASM)
* Compile-time monomorphisation with runtime selection
    * Aided by a `#[dispatch]` procedural macro to ensure optimal codegen.
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
opt-level = 2       # Required to inline SIMD intrinsics internally

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
* Equal-size Signed and Unsigned integer vectors can be cast between each other at zero cost.
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

### `emulate_fma`

Real fused multiply-add instructions are only enabled for AVX2 platforms. However, as FMA is used not only for performance but for its extended precision, falling back to a split multiply and addition will incur two rounding errors, and may be unacceptable for
some applications. Therefore, the `emulate_fma` Cargo feature will enable a slower but more accurate implementation on older platforms.

For single-precision floats, this is easiest done by simply casting it to double-precision, doing seperate multiply and additions, then casting back. For double-precision, it will use an infinite-precision implementation based on libm.

On SSE2 platforms, double-precision may fallback to scalar ops, as the effort needed to make it branchless will be more expensive than not. As of writing this, it has not been implemented, so benchmarks will reveal what is needed later.
