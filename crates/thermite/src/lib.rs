#![no_std]
#![allow(clippy::missing_transmute_annotations, unused, clippy::let_and_return)]
// used for more intelligent const splat
#![cfg_attr(feature = "nightly", feature(core_intrinsics, const_eval_select))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

#[cfg(feature = "nightly")]
#[rustversion::not(nightly)]
fn nightly_check() {
    compile_error!("The `nightly` feature requires a nightly compiler.");
}

#[doc(hidden)]
pub extern crate generic_array;

/// Creates a shuffle mask for various instructions. Note
/// that the order of the arguments is reversed from the
/// normal order of the lanes, so `MM_SHUFFLE!(3, 2, 1, 0)`
/// would be the identity shuffle (unchanged).
#[macro_export]
macro_rules! MM_SHUFFLE {
    () => { 0 };
    ($v:expr) => { $v };

    ($($v:expr),* $(,)?) => {const {
        const LEN: usize = [$($v),*].len();
        assert!(LEN.is_power_of_two(), "MM_SHUFFLE! requires a power of two number of lanes");

        const SHIFT: u32 = LEN.ilog2();

        let mut mask = 0;

        $(
            mask <<= SHIFT;
            mask |= $v;
        )*

        mask
    }};
}

/// Like `MM_SHUFFLE!`, but the order of the arguments is
/// the same as the order of the lanes (reversed from
/// conventional order).
#[macro_export] #[rustfmt::skip]
macro_rules! MM_SHUFFLE_R {
    () => { 0 };
    ($v:expr) => { $v };

    ($($v:expr),* $(,)?) => {const {
        const LEN: usize = [$($v),*].len();
        assert!(LEN.is_power_of_two(), "MM_SHUFFLE_R! requires a power of two number of lanes");

        const SHIFT: u32 = LEN.ilog2();

        let mut mask = 0;
        let mut shift = 0;

        $(
            mask |= $v << shift;
            shift += SHIFT;
        )*

        mask
    }};
}

pub mod vector;

pub mod backend;
pub mod divider;
pub mod mask;
pub mod math;
pub mod register;
pub mod simd;

#[doc(hidden)]
pub mod swizzle;

pub use divider::{BranchfreeDivider, Divider};
pub use mask::Mask;
pub use register::DoublePump;
pub use swizzle::Swizzle;
pub use vector::Vector;

// borrows technique from https://github.com/rust-lang/hashbrown/pull/209
#[inline]
#[cold]
fn cold() {}

#[rustfmt::skip]
#[inline(always)]
pub fn likely(b: bool) -> bool {
    if !b { cold() } b
}

#[rustfmt::skip]
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b { cold() } b
}
