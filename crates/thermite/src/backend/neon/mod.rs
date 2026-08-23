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

/// The 64-bit-lane `permutev`/`swizzle` control builder (`neon_ctrl_x2`) is the
/// one width with no shared lowering. NEON has no 64-bit multiply, so it
/// replicates through `u32` lanes and a `TRN1` instead. `tests/diff_swizzle.rs`
/// only drives 8/16/32-bit lanes at runtime, so it is checked here.
#[cfg(test)]
mod ctrl_x2_tests {
    use crate::element::Element;
    use crate::register::{CoreRegister, NumericRegister, Register, Storage};

    fn check<R: Register + NumericRegister>()
    where
        R::Element: core::fmt::Debug + PartialEq,
    {
        let lanes = 2;
        let a = R::indexed();
        let b = R::add(a, R::splat(<R::Element as Element>::from_u16(16)));

        for i in 0..lanes {
            for j in 0..lanes {
                let idxs = index_reg::<R>(&[i as u64, j as u64]);
                assert_eq!(
                    R::as_slice(&R::permutev(a, idxs)),
                    R::as_slice(&R::scalar_permutev(a, idxs)),
                    "permutev [{i}, {j}]"
                );
            }
        }

        for i in 0..2 * lanes {
            for j in 0..2 * lanes {
                let idxs = index_reg::<R>(&[i as u64, j as u64]);
                assert_eq!(
                    R::as_slice(&R::swizzle(a, b, idxs)),
                    R::as_slice(&R::scalar_swizzle(a, b, idxs)),
                    "swizzle [{i}, {j}]"
                );
            }
        }
    }

    fn index_reg<R: Register>(idxs: &[u64; 2]) -> Storage<R::Unsigned> {
        let mut arr: generic_array::GenericArray<
            <R::Unsigned as Register>::Element,
            <R::Unsigned as CoreRegister>::Lanes,
        > = Default::default();
        for lane in 0..2 {
            arr[lane] = <<R::Unsigned as Register>::Element as Element>::from_u16(idxs[lane] as u16);
        }
        R::Unsigned::new(arr)
    }

    #[test]
    fn f64x2() {
        check::<super::registers::F64x2Neon>();
    }

    #[test]
    fn i64x2() {
        check::<super::registers::I64x2Neon>();
    }

    #[test]
    fn u64x2() {
        check::<super::registers::U64x2Neon>();
    }
}
