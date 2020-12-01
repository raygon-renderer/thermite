#[macro_use]
mod macros;

pub mod polyfills;

pub mod avx1;
pub mod avx2;
pub mod sse2;
pub mod sse42;

pub use self::avx2::AVX2;
