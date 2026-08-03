//! Scalar float arithmetic that LLVM is allowed to rearrange, backing the
//! `algebraic-scalar` crate feature.
//!
//! Without `algebraic-scalar` these are plain IEEE-754 `+`/`-`/`*`/`/`/`%`. With
//! it, they lower to LLVM's *algebraic* float ops
//! (`core::intrinsics::fadd_algebraic` and friends, or the `f32::algebraic_add`
//! methods once those are stable), which carry every fast-math flag **except**
//! `nnan`/`ninf`:
//!
//! - `reassoc` - reassociation and redistribution across expressions
//! - `contract` - fusing `a * b + c` into an FMA
//! - `arcp`  - replacing division with a reciprocal multiply
//! - `nsz`   - the sign of zero is not observable
//! - `afn`   - approximate transcendental lowerings are allowed
//!
//! NaN and infinity still behave, so this is meaningfully weaker (and safer)
//! than C's `-ffast-math`. The point of it is `reassoc`: strict scalar ops stop
//! LLVM from reassociating a loop accumulator, which blocks vectorization of
//! anything written against the scalar backend.
//!
//! Only the scalar backend's [`NumericRegister`](crate::register::NumericRegister)
//! arithmetic routes through here; the SIMD backends issue intrinsics whose
//! semantics are fixed by the ISA, and the element layer stays strict so
//! `const` evaluation and bit-exact helpers are unaffected.

/// Scalar float ops whose strictness is controlled by the `algebraic-scalar` feature.
///
/// See the [module docs](self). Implemented for `f32` and `f64` only.
pub(crate) trait AlgebraicFloat: Copy {
    fn alg_add(self, rhs: Self) -> Self;
    fn alg_sub(self, rhs: Self) -> Self;
    fn alg_mul(self, rhs: Self) -> Self;
    fn alg_div(self, rhs: Self) -> Self;
    fn alg_rem(self, rhs: Self) -> Self;
}

// Feature off: strict IEEE-754, identical to what the backend did before the
// feature existed.
#[cfg(not(feature = "algebraic-scalar"))]
macro_rules! impl_algebraic_float {
    ($($t:ty),* $(,)?) => {$(
        impl AlgebraicFloat for $t {
            #[inline(always)] fn alg_add(self, rhs: Self) -> Self { self + rhs }
            #[inline(always)] fn alg_sub(self, rhs: Self) -> Self { self - rhs }
            #[inline(always)] fn alg_mul(self, rhs: Self) -> Self { self * rhs }
            #[inline(always)] fn alg_div(self, rhs: Self) -> Self { self / rhs }
            #[inline(always)] fn alg_rem(self, rhs: Self) -> Self { self % rhs }
        }
    )*};
}

// Feature on, nightly: the intrinsics. Available on every nightly, and the crate
// already turns on `core_intrinsics` for the const-splat paths, so this is the
// only route until the methods actually ship on stable.
#[cfg(all(feature = "algebraic-scalar", feature = "nightly"))]
macro_rules! impl_algebraic_float {
    ($($t:ty),* $(,)?) => {$(
        impl AlgebraicFloat for $t {
            #[inline(always)] fn alg_add(self, rhs: Self) -> Self { core::intrinsics::fadd_algebraic(self, rhs) }
            #[inline(always)] fn alg_sub(self, rhs: Self) -> Self { core::intrinsics::fsub_algebraic(self, rhs) }
            #[inline(always)] fn alg_mul(self, rhs: Self) -> Self { core::intrinsics::fmul_algebraic(self, rhs) }
            #[inline(always)] fn alg_div(self, rhs: Self) -> Self { core::intrinsics::fdiv_algebraic(self, rhs) }
            #[inline(always)] fn alg_rem(self, rhs: Self) -> Self { core::intrinsics::frem_algebraic(self, rhs) }
        }
    )*};
}

// Feature on, no `nightly`: the `algebraic_*` methods, which stabilize in 1.98.
// Until that ships this branch is unreachable in practice - anything older is
// rejected by `algebraic_scalar_version_check` below - but it means the feature
// starts working on stable the day 1.98 lands, with no code change here.
#[cfg(all(feature = "algebraic-scalar", not(feature = "nightly")))]
macro_rules! impl_algebraic_float {
    ($($t:ty),* $(,)?) => {$(
        impl AlgebraicFloat for $t {
            #[inline(always)] fn alg_add(self, rhs: Self) -> Self { self.algebraic_add(rhs) }
            #[inline(always)] fn alg_sub(self, rhs: Self) -> Self { self.algebraic_sub(rhs) }
            #[inline(always)] fn alg_mul(self, rhs: Self) -> Self { self.algebraic_mul(rhs) }
            #[inline(always)] fn alg_div(self, rhs: Self) -> Self { self.algebraic_div(rhs) }
            #[inline(always)] fn alg_rem(self, rhs: Self) -> Self { self.algebraic_rem(rhs) }
        }
    )*};
}

impl_algebraic_float!(f32, f64);

/// `f32::algebraic_add` and friends only stabilize in 1.98, above the crate MSRV,
/// so on anything older `algebraic-scalar` needs `nightly` for the intrinsics.
/// Without this the failure is a wall of E0658s about `float_algebraic`.
#[cfg(all(feature = "algebraic-scalar", not(feature = "nightly")))]
#[rustversion::before(1.98)]
fn algebraic_scalar_version_check() {
    compile_error!(
        "the `algebraic-scalar` feature needs Rust 1.98+ for the stable `algebraic_*` float \
         methods; until then, enable the `nightly` feature to use `core::intrinsics` instead."
    );
}
