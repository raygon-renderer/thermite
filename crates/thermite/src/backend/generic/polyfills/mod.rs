use crate::register::*;

pub mod bits;
pub mod casts;
pub mod divider;
pub mod math;
pub mod sort;

pub use bits::*;
pub use casts::*;
pub use divider::*;
pub use math::*;
pub use sort::*;

use generic_array::{ArrayLength, GenericArray, sequence::GenericSequence};

#[inline(always)]
pub fn zeroupper_mask<Z: ZeroUpper, N: ArrayLength>() -> GenericArray<bool, N> {
    GenericArray::generate(|i| i < Z::N)
}
