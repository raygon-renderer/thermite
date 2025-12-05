#[macro_use]
mod macros;

pub mod generic;

pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
pub mod x86_v1;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
pub mod x86_v2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
pub mod x86_v3;

#[cfg(all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")))]
pub mod wasm;

