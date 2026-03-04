Thermite SIMD: Melt Your CPU
============================

[![Thermite FFI Build](https://github.com/raygon-renderer/thermite/actions/workflows/ffi_artifacts.yaml/badge.svg)](https://github.com/raygon-renderer/thermite/actions/workflows/ffi_artifacts.yaml)

Thermite SIMD is a library for high-performance portable SIMD (Single Instruction, Multiple Data) programming, primarily via SoA (Structure of Arrays) data structures and algorithms. It provides low-level and high-level abstractions for SIMD programming, allowing developers to write efficient code that can run on various hardware architectures without sacrificing performance.

Thermite provides highly optimized feature-rich backends for x86-v2 (SSE4.2), x86-v3 (AVX2+FMA), with planned support for x86-v1 (SSE2), x86-v4 (AVX-512), ARM (NEON), WASM (WebAssembly SIMD), and RISC-V (V extension). Thermite also includes a portable scalar fallback implementation for platforms without SIMD support.

In addition, Thermite provides a highly optimized vectorized math library with many special math functions and algorithms, specialized for SIMD execution. This allows developers to perform complex mathematical computations efficiently using SIMD instructions.

# Design

Thermite's design is split into a few layers:

1. The `Register` traits define the low-level SIMD register types and operations, providing a common interface for different hardware backends.
2. The `Vector` data type that wraps a `Register` and provides access to its behavior in a high-level way via the `GenericVector` traits
3. The `GenericVector` traits, that define a strong set of constraints and extensive set of behaviors for any vector-like type.
4. The `*Math` traits, for `FloatVector` vectors, defines a wide variety of math functions, each with adjustable behaviors via a compile-time policy system.
5. Extraneous traits included here, such as a `thermite-special` for even more special math functions, `thermite-complex` for vectorized complex numbers, and more!

# Features

## FFI Library

Thermite provides a C-ABI FFI library for batch-processing large chunks of data using SIMD-accelerated backends, determined at runtime.

## `#[thermite::dispatch]`

Rust has a problem with `#[target_feature(enable = "...")]` in that if a function is ever NOT inlined, it "forgets" about the target features. This is a severe problem for programs that intend to be built with basic a SSE2-level executable, but want to include support for more modern architecture via dynamic dispatch. Either everything needs to be inlined, always, leading to massive code bloat, or it totally breaks down with deoptimization.

Enter `#[thermite::dispatch]`, which rewrites functions to propagate target features, automatically, and _statically_, meaning it is effectively zero-cost after dead-code optimizations. The dispatch macro is used for all `*Math` traits, so they remain optimized and lightweight.


# History and Motivation

Thermite was originally conceived while working on Raygon renderer, when it was decided we needed a state of the art high-performance SIMD library focused on SoA algorithms. Libraries at the time (and even now) were either too low-level, lacking in features, or not optimized for modern hardware. The goal was to create a library that would allow developers to write efficient SIMD code without having to worry about the underlying hardware details.

However, my first prototype of Thermite was flawed, too many leaky abstractions, and back in 2020 the Rust language/compiler itself was too limited to support the necessary abstractions. In 2025, I felt renewed interest in the project, and with the advancements in Rust's type system and compiler optimizations, I was able to redesign Thermite from the ground up, resulting in a much more efficient and user-friendly library.