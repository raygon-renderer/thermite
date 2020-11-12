#[macro_use]
mod macros;

pub mod avx1;
pub mod avx2;
//pub mod sse41;
//pub mod sse2;

pub use self::avx2::AVX2;
