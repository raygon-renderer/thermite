//! ARM NEON (AdvSIMD) backend, aarch64 only.
//!
//! Single-tier 128-bit backend, structurally modeled on the wasm backend (the
//! other single-tier sibling). Unlike wasm's untyped `v128`, NEON has a distinct
//! storage type per element (`float32x4_t`, `uint32x4_t`, ...), so the
//! `polyfills` layer provides a per-type "normalization" surface (`neon_and_f32`,
//! `neon_movemask_u32`, ...) that hides the `vreinterpretq` plumbing, and the
//! `macros` module stamps the highly regular trait impls per register.
//!
//! NEON/AdvSIMD is a mandatory part of AArch64, so there is no runtime feature
//! detection: on aarch64 with the `neon` crate feature, this backend is always
//! selectable.

#[macro_use]
pub mod polyfills;

#[macro_use]
pub mod macros;

pub mod arch {
    pub use super::polyfills::*;

    pub use core::arch::aarch64::*;

    pub const ISA: crate::isa::InstructionSet = crate::isa::InstructionSet::NEON;
}

pub mod registers;
pub use registers::Neon;

decl_aliases!(Neon);
pub use self::aliases::*;

pub mod prelude {
    pub use super::Neon;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}
