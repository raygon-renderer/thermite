//! x86-v4 polyfills. Inherits the whole x86 chain (generic < v1 < v2 < v3).
//! Helpers only expressible at v4 land here with the `*x_v4` naming scheme.

pub use crate::backend::x86_v3::polyfills::*;

pub mod bits;
pub mod casts;
mod funnel;
mod gfni;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use funnel::*;
pub use gfni::*;
pub use math::*;
