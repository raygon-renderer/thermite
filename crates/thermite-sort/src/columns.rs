//! Columnar sorting networks: sort *across* a group of vectors, not within one.
//!
//! Given `[v0, v1, .., v_{N-1}]`, lane `i` of every vector forms a column, and
//! each of the `LANES` columns is sorted independently and simultaneously. The
//! result is column-sorted, i.e. `out[j].extract(i) <= out[j+1].extract(i)` for
//! every lane `i` (ascending; reversed under
//! [`Descending`](thermite::sort::Descending)).
//!
//! # Why this is the cheap axis
//!
//! A compare-exchange between two whole vectors is a `min` and a `max` and
//! nothing else - no permute, no blend, no immediate encoding, no cross-lane
//! traffic. Contrast a lane sort *within* one vector, where every layer pays a
//! shuffle to bring lanes to face their partners.
//!
//! That inverts which sorting network is the best one, which is worth stating
//! plainly because the literature's "optimal" means the opposite thing here:
//!
//! | | unit of cost | minimize |
//! |---|---|---|
//! | lane sort (within one vector) | layers, each a shuffle + min + max + blend | **depth** - extra comparators inside a layer are free |
//! | columnar (this module) | comparators, each a min + max | **size** - a layer is not a unit of anything |
//!
//! So these use the size-minimal networks - 1, 5, 19 and 60 comparators for 2,
//! 4, 8 and 16 vectors - where a lane sort is right to spend 24 comparators
//! on 8 inputs rather than the minimal 19, to keep its shuffles in-lane.
//!
//! # What this is not
//!
//! Column-sorted is not sorted. Sorting `N * LANES` elements into one
//! ascending run needs these networks *plus* a merge phase, and that merge is
//! where the shuffles come back; see [`crate::merge`]. This module is the
//! cheap half only.
//!
//! # Widths
//!
//! Concrete array sizes rather than a const-generic `N`, deliberately. A
//! const-generic body behind an `if const { N == 16 }` ladder still emits the
//! dead arms' bounds checks and leans on LLVM to fold a large body, which is a
//! reliable way to lose const-folding. Fixed sizes have neither problem.

use thermite::sort::{Ascending, SortOrder};
use thermite::vector::NumericVector;

/// Define a columnar sort from a sorting network in the standard comparator-list
/// notation - one bracketed layer per line, exactly as the published tables give
/// it, so a network can be pasted in without hand-translation:
///
/// ```ignore
/// columnar_network! {
///     /// Sort 8 vectors columnwise.
///     pub fn sort_columns_8, inputs = 8;
///     [(0,2),(1,3),(4,6),(5,7)]
///     [(0,4),(1,5),(2,6),(3,7)]
///     [(0,1),(2,3),(4,5),(6,7)]
///     [(2,4),(3,5)]
///     [(1,4),(3,6)]
///     [(1,2),(3,4),(5,6)]
/// }
/// ```
///
/// Emits two functions: `<name>_by<V, O>` taking the order as a parameter, and
/// `<name><V>` as the ascending shorthand. They are the same code - `Ascending`
/// resolves `first`/`last` straight back to `min`/`max`.
///
/// Each `(a, b)` becomes one compare-exchange on whole vectors. The layer
/// brackets are documentation only - comparators run in written order, and
/// grouping them into layers just records which are independent, which is an ILP
/// hint to the reader and to the scheduler, not a semantic difference. That is
/// the opposite of the lane sort, where a layer is the literal unit of cost.
///
/// Every pair must be `(lo, hi)` with `lo < hi` - the convention the tables use,
/// and what puts the first-in-order element at the lower index. Both that and
/// the index range are checked at compile time.
///
/// Adding a width is a paste: grab the network for `n` inputs, and mind that the
/// useful ceiling is **register pressure**, not the table. 16 inputs already
/// means 16 live vectors, which is every ymm register on AVX2.
macro_rules! columnar_network {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident, $by:ident, inputs = $n:literal;
        $([$(($a:literal, $b:literal)),* $(,)?])*
    ) => {
        $(#[$attr])*
        ///
        /// Ordered by `O`; see [`SortOrder`]. The direction is free - a
        /// comparator is a `min` and a `max` either way, and only which result
        /// lands at which index changes.
        #[inline(always)]
        $vis fn $by<V: NumericVector, O: SortOrder>(mut v: [V; $n]) -> [V; $n] {
            $($(
                {
                    const _: () = assert!($a < $b, "comparator must be written (lo, hi)");
                    const _: () = assert!($b < $n, "comparator index out of range");

                    let (x, y) = (v[$a], v[$b]);
                    v[$a] = O::vector_first(x, y);
                    v[$b] = O::vector_last(x, y);
                }
            )*)*

            v
        }

        $(#[$attr])*
        ///
        #[doc = concat!("Ascending shorthand for [`", stringify!($by), "`].")]
        #[inline(always)]
        $vis fn $name<V: NumericVector>(v: [V; $n]) -> [V; $n] {
            $by::<V, Ascending>(v)
        }
    };
}

columnar_network! {
    /// Sort 2 vectors columnwise. 1 comparator, 1 layer.
    pub fn sort_columns_2, sort_columns_2_by, inputs = 2;
    [(0,1)]
}

columnar_network! {
    /// Sort 4 vectors columnwise. 5 comparators, 3 layers - the minimum size for
    /// 4 inputs.
    pub fn sort_columns_4, sort_columns_4_by, inputs = 4;
    [(0,2),(1,3)]
    [(0,1),(2,3)]
    [(1,2)]
}

columnar_network! {
    /// Sort 8 vectors columnwise. 19 comparators, 6 layers - the minimum size for
    /// 8 inputs.
    pub fn sort_columns_8, sort_columns_8_by, inputs = 8;
    [(0,2),(1,3),(4,6),(5,7)]
    [(0,4),(1,5),(2,6),(3,7)]
    [(0,1),(2,3),(4,5),(6,7)]
    [(2,4),(3,5)]
    [(1,4),(3,6)]
    [(1,2),(3,4),(5,6)]
}

columnar_network! {
    /// Sort 16 vectors columnwise. 60 comparators, 10 layers - the smallest
    /// network known for 16 inputs.
    ///
    /// This holds 16 live vectors, the entire ymm register file on AVX2, so
    /// expect spills.
    ///
    /// # FP-port-bound, not spill-bound
    ///
    /// Easy to assume otherwise at 16 live vectors on a 16-register ISA, and
    /// the stack frame does say it spills, but the spills ride load/store
    /// units that have slack while all four FP pipes sit saturated. The
    /// comparators are about 95% of the block; removing every spill would buy
    /// a few percent.
    ///
    /// Two consequences worth knowing before tuning this:
    ///
    /// - Do not trade comparators for register pressure. Comparators are the
    ///   saturated resource, so that trade runs backwards, and at 60 for 16
    ///   inputs this is already at the known floor.
    /// - The shallower published alternative (61 comparators, 9 layers) is a
    ///   wash. The block is nowhere near depth-bound - `min` and `max` are
    ///   1-cycle, so the whole 10-layer chain is a ~10-cycle critical path
    ///   against an order of magnitude more throughput-limited work - so one
    ///   fewer layer buys about a cycle while the extra comparator costs
    ///   roughly a third of one.
    ///
    /// The lever that does scale is elements-per-op: the same 60 comparators
    /// sort 128 elements at `i32x8` and 256 at `i32x16`, which is also where
    /// the spilling goes away.
    pub fn sort_columns_16, sort_columns_16_by, inputs = 16;
    [(0,13),(1,12),(2,15),(3,14),(4,8),(5,6),(7,11),(9,10)]
    [(0,5),(1,7),(2,9),(3,4),(6,13),(8,14),(10,15),(11,12)]
    [(0,1),(2,3),(4,5),(6,8),(7,9),(10,11),(12,13),(14,15)]
    [(0,2),(1,3),(4,10),(5,11),(6,7),(8,9),(12,14),(13,15)]
    [(1,2),(3,12),(4,6),(5,7),(8,10),(9,11),(13,14)]
    [(1,4),(2,6),(5,8),(7,10),(9,13),(11,14)]
    [(2,4),(3,6),(9,12),(11,13)]
    [(3,5),(6,8),(7,9),(10,12)]
    [(3,4),(5,6),(7,8),(9,10),(11,12)]
    [(6,7),(8,9)]
}
