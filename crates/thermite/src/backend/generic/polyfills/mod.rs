use crate::register::*;

pub mod bits;
pub mod casts;
pub mod compress;
pub mod conflict;
pub mod divider;
pub mod expand;
pub mod interleave;
pub mod math;
pub mod scan;
pub mod sort;

pub use bits::*;
pub use casts::*;
pub use compress::*;
pub use conflict::*;
pub use divider::*;
pub use expand::*;
pub use interleave::*;
pub use math::*;
pub use scan::*;
pub use sort::*;

use generic_array::{ArrayLength, GenericArray, sequence::GenericSequence};

#[inline(always)]
pub fn zeroupper_mask<Z: ZeroUpper, N: ArrayLength>() -> GenericArray<bool, N> {
    GenericArray::generate(|i| i < Z::N)
}
