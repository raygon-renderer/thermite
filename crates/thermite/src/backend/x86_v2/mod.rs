#[macro_use]
mod macros;

pub mod arch {
    pub use super::polyfills::*;
    pub use crate::backend::x86::sse42::*;
}

pub mod polyfills;
pub mod registers;

use crate::{register::DoublePump, vector::Vector};
