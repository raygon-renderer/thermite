//! SIMD sorting algorithms built on [`thermite`]'s vector traits.
//!
//! The entry points are [`sort`] and [`sort_by`], a vectorized quicksort over
//! a slice. Underneath, and usable on their own:
//!
//! - [`partition`] - in-place branchless partition around a pivot.
//! - [`columns`] - columnar sorting networks, which sort *across* a group of
//!   vectors (lane `i` of each forms an independent column).
//! - [`merge`] - the block sort that turns column-sorted vectors into one
//!   sorted run.
//! - [`base`] - the recursion's base case, sorting a short slice entirely in
//!   registers.
//! - [`runs`] - SIMD scans for already-ordered input.
//!
//! ```
//! use thermite_sort::sort;
//! # #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
//! # {
//! use thermite::backend::x86_v3::i32x8;
//!
//! let mut keys = [5, 2, 9, 2, 7, 1, 8, 3];
//! sort::<i32x8>(&mut keys);
//! assert_eq!(keys, [1, 2, 2, 3, 5, 7, 8, 9]);
//! # }
//! ```
//!
//! # What lives here, and what stays in core
//!
//! Core `thermite` owns the *primitives*: `NumericRegister::sort_by` and
//! `bitonic_clean_by` are trait methods, so a backend can override either with
//! a bespoke instruction sequence, and the networks behind them sit with the
//! other polyfills. The order markers ([`Ascending`], [`Descending`]) are in
//! core too, because the register-trait signatures name them.
//!
//! Algorithms *over* those primitives live here - free functions generic over
//! `V: NumericVector`, with nothing per-backend to hang anywhere. The dividing
//! line: if a backend could plausibly want to override it, it is a trait
//! method and belongs in core.
//!
//! # Order genericity
//!
//! Everything here is generic over [`SortOrder`], so one body serves both
//! directions at identical cost - see [`thermite::sort`] for why the direction
//! is free rather than a post-hoc reverse. The unparameterized names
//! ([`sort_columns_4`] and friends) are ascending shorthands for the `_by`
//! forms.

#![no_std]

pub mod base;
pub mod columns;
pub mod merge;
pub mod partition;
mod quicksort;
pub mod runs;

pub use columns::{
    sort_columns_2, sort_columns_2_by, sort_columns_4, sort_columns_4_by, sort_columns_8, sort_columns_8_by,
    sort_columns_16, sort_columns_16_by,
};
pub use thermite::sort::{Ascending, Descending, SortOrder};

/// Sort a slice in `O` order, using vectors of type `V`.
///
/// A vectorized quicksort: [`partition`] splits, [`base`]'s network block sort
/// finishes short ranges, and a depth budget falls back to heapsort so
/// adversarial input stays `O(n log n)`. Already-ordered and reverse-ordered
/// input is detected at entry and costs one scan. Unstable, and allocation-free
/// - the scratch is a fixed stack buffer.
///
/// # Choosing `V`
///
/// Use the widest **native** vector for the target ISA: `i32x8` on AVX2,
/// `i32x4` on SSE4.2, NEON and wasm. Do not reach for a wider `ArrayRegister`
/// composite - the hot operations here are `compress` and the merge swizzles,
/// neither of which scales across sub-registers the way `min`/`max` does, so a
/// two-chunk composite sorts slower than the native width it is built from.
///
/// ISAs where the vector path cannot pay for itself are routed to `core`'s
/// `sort_unstable` instead, decided at compile time from `V::ISA`: `Scalar`
/// and `Unknown` have no real `compress`, and on x86-v1 (SSE2) the `compress`
/// polyfill has no byte shuffle to build on and lands well behind a scalar
/// sort.
///
/// # NaN
///
/// NaN keys are swept to the **back of the slice** - for both sort orders,
/// bit patterns preserved, in unspecified relative order - and the remaining
/// keys are sorted in `O` order in front of them. The output is always a
/// permutation of the input, on every code path.
///
/// This has to happen before the recursion: no padding sentinel can sort past
/// NaN, and `min`/`max` NaN semantics legitimately differ between backends.
/// The sweep is gated on `Element::HAS_UNORDERED`, so integer sorts compile
/// none of it, and float sorts pay one extra read-only scan only when the
/// entry run detection has not already proven the slice NaN-free.
#[thermite::dispatch(V)]
pub fn sort_by<V: merge::MergeVector, O: SortOrder>(keys: &mut [V::Element])
where
    V::Element: PartialOrd,
{
    if const { vector_sort_loses(V::ISA) } {
        // Scalar NaN sweep, same contract as the vector pre-pass: unordered
        // keys to the back, ordered prefix sorted.
        let mut ordered = keys.len();
        if const { <V::Element as thermite::element::Element>::HAS_UNORDERED } {
            let mut w = 0;
            for i in 0..keys.len() {
                if keys[i].partial_cmp(&keys[i]).is_some() {
                    keys.swap(i, w);
                    w += 1;
                }
            }
            ordered = w;
        }
        // The NaN-free prefix compares totally, so the `unwrap_or` is never
        // reached; it is there because `PartialOrd` cannot say so.
        let keys = &mut keys[..ordered];
        if const { O::IS_ASCENDING } {
            keys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            keys.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        }
        return;
    }
    quicksort::quicksort::<V, O>(keys);
}

/// ISAs where the vector path is slower than `core`'s scalar sort, and
/// [`sort_by`] defers to that instead. See its docs for the reasoning.
#[inline(always)]
const fn vector_sort_loses(isa: thermite::isa::InstructionSet) -> bool {
    use thermite::isa::InstructionSet;
    match isa {
        InstructionSet::Scalar | InstructionSet::Unknown => true,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        InstructionSet::X86V1 => true,
        _ => false,
    }
}

/// Sort a slice ascending, using vectors of type `V`. See [`sort_by`].
#[thermite::dispatch(V)]
pub fn sort<V: merge::MergeVector>(keys: &mut [V::Element])
where
    V::Element: PartialOrd,
{
    sort_by::<V, Ascending>(keys);
}
