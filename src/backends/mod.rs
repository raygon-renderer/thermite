#[macro_use]
mod macros;

pub mod polyfills;

//pub mod scalar;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod aarch64;
#[cfg(all(feature = "neon", target_arch = "arm"))]
pub mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx1;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse42;
#[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
pub mod wasm32;
