//! Base case: sort a short slice entirely in registers.
//!
//! This is where the quicksort recursion bottoms out. It loads up to
//! `16 * LANES` keys, pads the unused lanes with a sentinel that sorts past
//! every real input, runs a sorting network over the block, and stores back
//! only the real keys.
//!
//! # Two block sorts, and which one runs
//!
//! [`crate::merge`] is the fast one - a columnar network followed by a hybrid
//! bitonic/odd-even merge - and covers registers up to 16 lanes.
//!
//! Wider registers fall back to the bitonic merge tree kept below
//! ([`bitonic_block_16`] and friends): sort each vector's lanes, reverse one
//! side, clean. That construction wastes no comparator - it sits exactly at
//! the bitonic floor - but the bitonic floor is not the achievable floor, and
//! [`crate::merge`] beats it on comparators, shuffles and blends alike.
//!
//! Core carries the same bitonic tree at the register layer
//! (`backend::generic::polyfills::sort`), where it implements
//! `ArrayRegister::sort_by` for one *logical register* built from N chunks.
//! This one sorts N *separate vectors* held by a slice-sorting caller. Same
//! algorithm, different clients, and they cannot share code: `NumericVector`
//! exposes no associated register type, so vector-layer code cannot reach the
//! register-layer networks.
//!
//! # Padding and NaN
//!
//! The sentinel is [`SortOrder::last_value`], which on floats is `+inf` rather
//! than `MAX` - a finite pad would sort *below* a real infinity and silently
//! misorder it. NaN is a separate problem no sentinel can solve, since nothing
//! sorts past NaN; [`crate::sort_by`] removes NaN before the recursion starts.

use thermite::element::Element;
use thermite::prelude::*;
use thermite::sort::SortOrder;

use crate::merge::MergeVector;

/// Chunks in the widest block; vqsort uses 16 rows as well.
///
/// This is an AVX2 ceiling rather than a universal one: 16 vectors is the
/// entire ymm register file, so going wider there spills without buying
/// anything. AVX-512 has 32 zmm registers and could take a 32-row block,
/// doubling the keys sorted per leaf and removing a partition level, which
/// needs a 32-input columnar network and a fifth row stage per merge.
pub const MAX_CHUNKS: usize = 16;

/// Largest slice [`base_case`] can sort, in elements.
#[inline(always)]
pub fn max_len<V: NumericVector>() -> usize {
    MAX_CHUNKS * V::LANES
}

/// Widest register this stages a ragged tail for, in lanes. Only one vector's
/// worth is ever needed, so this is 64 elements of stack rather than the
/// `MAX_CHUNKS * LANES` the old whole-block scratch cost.
const MAX_LANES: usize = 64;

/// Sort `keys` in `O` order. Requires `keys.len() <= max_len::<V>()`.
///
/// Rounds the length up to a power-of-two count of chunks, pads with the
/// sentinel, and runs the block sort for that width.
///
/// # Memory traffic
///
/// Only the ragged tail is staged through scratch. Whole vectors load and
/// store straight against `keys`, and chunks past the end of the data are
/// sentinel *splats* that never touch memory at all. Staging the whole block
/// instead costs a scratch fill plus two full copies per leaf, paid even for a
/// leaf of twenty keys, and that dominates the network it feeds.
#[thermite::dispatch(V)]
pub fn base_case<V: MergeVector, O: SortOrder>(keys: &mut [V::Element]) {
    let n = V::LANES;
    let len = keys.len();
    debug_assert!(len <= max_len::<V>(), "base case overflow: {len} > {}", max_len::<V>());
    debug_assert!(n <= MAX_LANES, "tail scratch too small for {n}-lane registers");

    if len < 2 {
        return;
    }

    let sentinel = O::last_value::<V>();
    let full = len / n; // whole vectors, readable in place
    let rem = len % n; // keys in the ragged final vector
    let chunks = len.div_ceil(n).next_power_of_two();
    let ptr = keys.as_mut_ptr();

    // The one vector that needs staging: real keys then sentinel.
    let tail_in = if rem != 0 {
        let mut tail = [core::mem::MaybeUninit::<V::Element>::uninit(); MAX_LANES];
        // SAFETY: `rem < n <= MAX_LANES`, and `full * n + rem == len`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ptr.add(full * n),
                tail.as_mut_ptr() as *mut V::Element,
                rem,
            );
            let pad = sentinel.extractv(0);
            for slot in tail.iter_mut().take(n).skip(rem) {
                slot.write(pad);
            }
            V::load_unaligned(tail.as_ptr() as *const V::Element)
        }
    } else {
        sentinel
    };

    macro_rules! run {
        ($count:literal, $f:ident) => {
            if chunks <= $count {
                // Chunks beyond the data are sentinel splats - no memory touched.
                let mut c = [sentinel; $count];
                let mut i = 0;
                while i < $count {
                    if i < full {
                        // SAFETY: `i < full` means the whole vector is inside `keys`.
                        c[i] = unsafe { V::load_unaligned(ptr.add(i * n)) };
                    } else if i == full {
                        c[i] = tail_in;
                    }
                    i += 1;
                }

                let c = $f::<V, O>(c);

                let mut i = 0;
                while i < $count {
                    if i < full {
                        // SAFETY: as above.
                        unsafe { c[i].store_unaligned(ptr.add(i * n)) };
                    } else if i == full && rem != 0 {
                        // Only the first `rem` lanes of this chunk are real; the
                        // rest are sentinels that must not be written back.
                        let mut tail = [core::mem::MaybeUninit::<V::Element>::uninit(); MAX_LANES];
                        // SAFETY: `n <= MAX_LANES`; `rem < n`; `full * n + rem == len`.
                        unsafe {
                            c[i].store_unaligned(tail.as_mut_ptr() as *mut V::Element);
                            core::ptr::copy_nonoverlapping(
                                tail.as_ptr() as *const V::Element,
                                ptr.add(full * n),
                                rem,
                            );
                        }
                    }
                    i += 1;
                }
                return;
            }
        };
    }

    run!(1, sort_block_1);
    run!(2, block_2);
    run!(4, block_4);
    run!(8, block_8);
    run!(16, block_16);

    unreachable!("chunks > MAX_CHUNKS despite the length check");
}

// ---------------------------------------------------------------------------
// Which block sort runs
//
// `crate::merge` is the fast one and covers registers up to 16 lanes; anything
// wider keeps the bitonic tree below. The test is on an associated const, so
// the arm a given `V` cannot reach folds away at monomorphization and only one
// network is ever emitted per instantiation.
// ---------------------------------------------------------------------------

macro_rules! block_dispatch {
    ($(#[$attr:meta])* $name:ident, $n:literal, $fast:ident, $slow:ident) => {
        $(#[$attr])*
        #[inline(always)]
        fn $name<V: MergeVector, O: SortOrder>(c: [V; $n]) -> [V; $n] {
            if const { V::LANES <= crate::merge::MAX_LANES } {
                crate::merge::$fast::<V, O>(c)
            } else {
                $slow::<V, O>(c)
            }
        }
    };
}

block_dispatch!(
    /// Sort `2 * LANES` keys held in 2 vectors into one row-major run.
    block_2, 2, sort_block_2, bitonic_block_2
);
block_dispatch!(
    /// Sort `4 * LANES` keys held in 4 vectors into one row-major run.
    block_4, 4, sort_block_4, bitonic_block_4
);
block_dispatch!(
    /// Sort `8 * LANES` keys held in 8 vectors into one row-major run.
    block_8, 8, sort_block_8, bitonic_block_8
);
block_dispatch!(
    /// Sort `16 * LANES` keys held in 16 vectors into one row-major run.
    block_16, 16, sort_block_16, bitonic_block_16
);

// ---------------------------------------------------------------------------
// The merge tree. `sort_block_N` sorts N vectors into one run of `N * LANES`;
// `clean_N` finishes an already-bitonic block.
//
// Cross-chunk comparators are whole-vector `first`/`last` - the columnar cost
// model, zero shuffles. Within-chunk work is confined to `sort_by` and
// `bitonic_clean_by`, so nothing here knows a lane count.
// ---------------------------------------------------------------------------

/// Cross-chunk compare-exchange: `c[i]` keeps per-lane firsts, `c[j]` the lasts.
macro_rules! ce {
    ($c:ident, $o:ty: $(($i:tt, $j:tt))+) => {$(
        {
            let (x, y) = ($c[$i], $c[$j]);
            $c[$i] = <$o>::vector_first(x, y);
            $c[$j] = <$o>::vector_last(x, y);
        }
    )+};
}

macro_rules! clean_chunks {
    ($c:ident, $o:ty: $($i:tt)+) => {$(
        $c[$i] = $c[$i].bitonic_clean_by::<$o>();
    )+};
}

#[inline(always)]
fn sort_block_1<V: NumericVector, O: SortOrder>(c: [V; 1]) -> [V; 1] {
    [c[0].sort_by::<O>()]
}

#[inline(always)]
fn bitonic_clean_2<V: NumericVector, O: SortOrder>(mut c: [V; 2]) -> [V; 2] {
    ce!(c, O: (0, 1));
    clean_chunks!(c, O: 0 1);
    c
}

#[inline(always)]
fn bitonic_clean_4<V: NumericVector, O: SortOrder>(mut c: [V; 4]) -> [V; 4] {
    ce!(c, O: (0, 2)(1, 3));
    ce!(c, O: (0, 1)(2, 3));
    clean_chunks!(c, O: 0 1 2 3);
    c
}

#[inline(always)]
fn bitonic_clean_8<V: NumericVector, O: SortOrder>(mut c: [V; 8]) -> [V; 8] {
    ce!(c, O: (0, 4)(1, 5)(2, 6)(3, 7));
    ce!(c, O: (0, 2)(1, 3)(4, 6)(5, 7));
    ce!(c, O: (0, 1)(2, 3)(4, 5)(6, 7));
    clean_chunks!(c, O: 0 1 2 3 4 5 6 7);
    c
}

#[inline(always)]
fn bitonic_clean_16<V: NumericVector, O: SortOrder>(mut c: [V; 16]) -> [V; 16] {
    ce!(c, O: (0, 8)(1, 9)(2, 10)(3, 11)(4, 12)(5, 13)(6, 14)(7, 15));
    ce!(c, O: (0, 4)(1, 5)(2, 6)(3, 7)(8, 12)(9, 13)(10, 14)(11, 15));
    ce!(c, O: (0, 2)(1, 3)(4, 6)(5, 7)(8, 10)(9, 11)(12, 14)(13, 15));
    ce!(c, O: (0, 1)(2, 3)(4, 5)(6, 7)(8, 9)(10, 11)(12, 13)(14, 15));
    clean_chunks!(c, O: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    c
}

/// Sort each half, reverse the second so the concatenation is bitonic, clean.
///
/// The reversal is order-independent: two descending runs with the second
/// reversed form a valley rather than a mountain, equally bitonic.
#[inline(always)]
fn bitonic_block_2<V: NumericVector, O: SortOrder>(c: [V; 2]) -> [V; 2] {
    bitonic_clean_2::<V, O>([c[0].sort_by::<O>(), c[1].sort_by::<O>().reverse()])
}

#[inline(always)]
fn bitonic_block_4<V: NumericVector, O: SortOrder>(c: [V; 4]) -> [V; 4] {
    let a = bitonic_block_2::<V, O>([c[0], c[1]]);
    let b = bitonic_block_2::<V, O>([c[2], c[3]]);
    bitonic_clean_4::<V, O>([a[0], a[1], b[1].reverse(), b[0].reverse()])
}

#[inline(always)]
fn bitonic_block_8<V: NumericVector, O: SortOrder>(c: [V; 8]) -> [V; 8] {
    let a = bitonic_block_4::<V, O>([c[0], c[1], c[2], c[3]]);
    let b = bitonic_block_4::<V, O>([c[4], c[5], c[6], c[7]]);
    bitonic_clean_8::<V, O>([
        a[0],
        a[1],
        a[2],
        a[3],
        b[3].reverse(),
        b[2].reverse(),
        b[1].reverse(),
        b[0].reverse(),
    ])
}

#[inline(always)]
fn bitonic_block_16<V: NumericVector, O: SortOrder>(c: [V; 16]) -> [V; 16] {
    let a = bitonic_block_8::<V, O>([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
    let b = bitonic_block_8::<V, O>([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]);
    bitonic_clean_16::<V, O>([
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        b[7].reverse(),
        b[6].reverse(),
        b[5].reverse(),
        b[4].reverse(),
        b[3].reverse(),
        b[2].reverse(),
        b[1].reverse(),
        b[0].reverse(),
    ])
}

/// `Element::ZERO` is not usable as an array initializer for a generic element
/// without this bound being visible; kept as a named helper so the intent is
/// obvious if the buffer type ever changes.
#[allow(dead_code)]
#[inline(always)]
fn zero<E: Element>() -> E {
    E::ZERO
}
