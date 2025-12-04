#[macro_use]
pub mod polyfills;

#[macro_use]
pub mod macros;

pub mod arch {
    pub use super::polyfills::*;
    pub use core::arch::wasm32::*;
}

pub mod registers;
