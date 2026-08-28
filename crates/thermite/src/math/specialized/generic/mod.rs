//! Kernel bodies shared by the `f32` and `f64` backends.
//!
//! Everything here is the *real-vector* lowering of a `Specialized*Math` method: `ps.rs`
//! and `pd.rs` both override the method and both delegate to the same function, so the
//! algorithm lives once and the two backends differ only in their element type. Composites
//! (`Complex`, `Dual`, `Interval`, `Compensated`) do not come through here at all. They
//! take the trait defaults in `specialized/mod.rs`, or their own overrides. One deliberate
//! exception: `hypot.rs` also holds the composite `*_recip_scaled` hypot pair (the bodies
//! behind the trait defaults), so every `hypot` lowering lives in one file.
//!
//! Split by topic from a single 1033-line `generic.rs`, mirroring
//! `thermite-special/src/specialized/generic/`. The submodules are an organizational
//! detail: every item is re-exported flat, so call sites stay `generic::<name>` and the
//! grouping can change without touching them.

mod cardinal;
mod hypot;
mod log;
mod poly;
mod reduce;
mod sqrt;

pub use cardinal::*;
pub use hypot::*;
pub use log::*;
pub use poly::*;
pub use reduce::*;
pub use sqrt::*;
