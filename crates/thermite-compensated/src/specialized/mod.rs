//! Compensated's own backend layer, one rung below `thermite-special`'s.
//!
//! `thermite-special` exposes its math through a two-rung ladder: the user-facing
//! `SpecialMath` / `SpecialMathWithPolicy` traits are blanket-implemented for anything
//! that implements the corresponding `Specialized*Math<E>`, so a type supplies one
//! backend and the whole public API materializes. That is how `Vector<R>`, `Dual`,
//! `Compensated` and `Complex` all present the same surface without any of them writing
//! it.
//!
//! For `Compensated` that ladder is one rung short. Its backend impl is a single blanket
//!
//! ```ignore
//! impl<V: CompensatedFloatVector> SpecializedSpecialMath<Compensated<V::Element>> for Compensated<V>
//! ```
//!
//! which covers every width at once - and the widths are not alike.
//! `Compensated<Vector<f32>>` is double-single at ~48 bits of mantissa;
//! `Compensated<Vector<f64>>` is double-double at ~106. Anything carrying fitted
//! coefficients needs a *different* table for each, and that impl has nowhere to put
//! two. That, not any missing algorithm, is why the gamma family sat unimplemented.
//!
//! So this module adds the missing rung. The traits here are the real backend for the
//! coefficient-bearing functions; `Compensated`'s `SpecializedSpecialMath` impl is
//! blanket-implemented over them exactly the way `SpecialMath` is blanket-implemented
//! over *it*. Per-width impls live in `ps` / `pd` submodules, named after
//! `thermite-special`'s own.
//!
//! Methods carry element-agnostic defaults - series, continued fractions, recurrences,
//! which are correct at any width - so a width that needs no special treatment
//! implements the trait with an empty block, and a per-element impl is only ever an
//! optimization or a precision fix, never a prerequisite.

pub mod special;
