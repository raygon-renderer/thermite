#[macro_use]
pub mod polyfills;

#[macro_use]
pub mod macros;

pub mod arch {
    pub use super::polyfills::*;

    #[cfg(not(all(feature = "nightly", target_arch = "wasm64")))]
    pub use core::arch::wasm32::*;

    #[cfg(all(feature = "nightly", target_arch = "wasm64"))]
    pub use core::arch::wasm64::*;

    #[rustfmt::skip]
    pub const ISA: crate::isa::InstructionSet = {
        #[cfg(all(feature = "nightly", target_arch = "wasm64"))]
        { crate::isa::InstructionSet::WASM64 }

        #[cfg(not(all(feature = "nightly", target_arch = "wasm64")))]
        { crate::isa::InstructionSet::WASM32 }
    };
}

pub mod registers;
pub use registers::Wasm;

decl_aliases!(Wasm);
pub use self::aliases::*;
