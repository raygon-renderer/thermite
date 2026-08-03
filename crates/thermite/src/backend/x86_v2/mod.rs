#[macro_use]
mod macros;

pub mod arch {
    pub use super::polyfills::*;
    pub use crate::backend::x86::sse42::*;
}

pub mod polyfills;
pub mod registers;


pub use registers::X86V2;

decl_aliases!(X86V2);
pub use self::aliases::*;

pub mod prelude {
    pub use super::X86V2;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}
