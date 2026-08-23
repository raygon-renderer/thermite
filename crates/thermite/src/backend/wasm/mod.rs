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

pub mod prelude {
    pub use super::Wasm;
    pub use super::aliases::*;
    pub use crate::prelude::*;
}

/// `tests/diff_swizzle.rs` only drives 8/16/32-bit lanes at runtime, so the
/// 64-bit-lane control builder (`wasm_ctrl_x2`) and the two-source `swizzle`
/// default riding on it are checked here.
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
        check::<super::registers::F64x2Wasm>();
    }

    #[test]
    fn i64x2() {
        check::<super::registers::I64x2Wasm>();
    }

    #[test]
    fn u64x2() {
        check::<super::registers::U64x2Wasm>();
    }
}
