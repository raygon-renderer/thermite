Thermite SIMD
=============

Thermite is an SIMD library focused on providing portable SIMD acceleratation of SoA (Structure of Arrays) algorithms, using consistent-length SIMD vectors for lockstep iteration and computation.

Partially inspired by SIMDeez, Thermite supports static compilation to many instruction sets and runtime selection of the most performant instruction set, allowing you to write a single algorithm once and have it work on hardware ranging from late 2012 CPUs with only SSE2 all this way to modern cutting-edge CPUs with AVX2 and even AVX512.

However, unlike SIMDeez, all vectors within a supported instruction set have the same length, so you don't have to worry about how to handle **8** `u16` values alongside **4** `f32` values, if your data contains only **4** `u16` and **4** `f32`. In Thermite, there is only `Vu16` and `Vf32`, and they are the same length, allowing for easy loading and interop with other data types.

Furthermore, unlike `packed_simd`, Thermite compiles on Stable Rust, though there is a `nightly` feature flag for hardware-accelerated half-precision float conversions. (It's currently marked unstable for some reason).

Thermite also provides a set of high-performance vectorized special math functions that can take advantage of all instruction sets, with specialized versions for single and double precision floats.