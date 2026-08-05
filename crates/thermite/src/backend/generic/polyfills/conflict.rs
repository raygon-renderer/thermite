//! Duplicate-value detection across lanes.
//!
//! [`count_conflicts_default`] answers, per lane, *how many earlier lanes hold
//! my value* - `out[i] == |{ j < i : v[j] == v[i] }|`, i.e. AVX-512CD's
//! `conflict(v).count_ones()`. Two things fall out of it:
//!
//! - `count_conflicts(v) == 0` is the **first-occurrence** mask.
//! - The count is the round number for a conflicting read-modify-write. A lane
//!   of rank `r` is safe to process in round `r`, since every earlier duplicate
//!   has a strictly smaller rank and goes first. That is what makes a vectorized
//!   histogram or SAH-bin increment correct where a plain scatter silently drops
//!   duplicate writes.
//!
//! # The ladder
//!
//! Step `j` compares each lane against the one `LANES - j` positions earlier -
//! [`align::<j>(v, v)`](crate::register::Register::align) is exactly that rotate - and
//! `suffix_mask(j)` is the set of lanes where the rotate did not wrap. Running
//! `j` over `1..LANES` visits every ordered pair `(i, j < i)` exactly once.
//!
//! Indexing by `j` rather than by the distance is what keeps every offset a
//! *literal*: `align`'s offset is a const-generic argument and stable Rust has
//! no const arithmetic in that position. So one `if const` chain serves every
//! width, including non-powers-of-two, with no per-width offset table - unlike
//! the forward prefix-scan ladder, which needs `LANES - j` and therefore a
//! match on the lane count.
//!
//! `LANES - 1` steps of ~4 vector ops: more than the single instruction
//! AVX-512CD needs, hence a *default* on [`IntegerRegister::count_conflicts`]
//! rather than the only implementation, but far under a scalar pass at
//! `LANES * (LANES - 1) / 2` compares over a spilled register.

use generic_array::typenum::Unsigned;

use crate::{
    element::Element,
    register::{BitwiseRegister, IntegerRegister, Storage},
};

/// Portable rotate-ladder body behind
/// [`crate::register::IntegerRegister::count_conflicts`].
///
/// Free-standing so blanket impls can reach it without `Self::count_conflicts`
/// recursion, matching [`compress_default`](super::compress::compress_default).
#[inline(always)]
pub fn count_conflicts_default<R: IntegerRegister>(value: Storage<R>) -> Storage<R> {
    let mut acc = R::ZERO;

    let n = <R::Lanes as Unsigned>::USIZE;
    let lanes = R::indexed();

    macro_rules! step {
        ($($j:literal),* $(,)?) => {$(
            if const { <R::Lanes as Unsigned>::USIZE > $j } {
                // Lanes where the rotate did not wrap: `i >= LANES - j`. (The
                // vector layer spells this `suffix_mask(j)`; the register layer
                // has no such helper, and both operands here are loop-invariant
                // constants that fold out.)
                let limit = R::splat(<R::Element as Element>::from_u8((n - $j) as u8));
                let unwrapped = R::ge(lanes, limit);

                let rotated = R::align::<$j>(value, value);
                let hit = <R::Mask as BitwiseRegister>::bitand(R::eq(value, rotated), unwrapped);

                // Increment the hit lanes. One maskable op, so the lowering is
                // the backend's to choose: `add` has a zero identity on its
                // right-hand side, so the generated `_c` masks the operand
                // (`pand` + `paddd`) rather than blending the result, and
                // AVX-512 can make it a single masked add.
                acc = R::add_c(hit, acc, R::ONE);
            }
        )*};
    }

    step!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63
    );

    acc
}
