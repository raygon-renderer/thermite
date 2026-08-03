//! The recursion driver: partition down to a block, then sort with networks.
//!
//! Structurally vqsort - [`crate::partition`] splits and [`crate::base`]
//! finishes - with a depth budget falling back to heapsort so adversarial
//! input degrades to `O(n log n)` rather than quadratic. Two things vqsort
//! does not have are taken from ipnsort: run detection at entry, and the
//! `ancestor_pivot` treatment of duplicate runs.
//!
//! On random input the recursion costs a little over `log2(n / 128)` passes
//! over the input, the floor set by the base-case width.

use thermite::element::Element;
use thermite::prelude::*;
use thermite::sort::SortOrder;

use crate::base::{base_case, max_len};
use crate::merge::MergeVector;
use crate::partition::{partition, partition_equal, partition_unordered_last};
use crate::runs::{has_unordered, is_ordered, is_reverse_ordered};

/// Sort `keys` in `O` order using vectors of type `V`.
///
/// Not public on its own - callers go through [`crate::sort_by`], which picks
/// `V` for the element type.
#[thermite::dispatch(V)]
pub(crate) fn quicksort<V: MergeVector, O: SortOrder>(keys: &mut [V::Element])
where
    V::Element: PartialOrd,
{
    // Run detection, at entry only.
    //
    // Per level it loses: after partitioning structured input the left region
    // is ordered for most of its length with a violation near the end, so the
    // scan reads nearly everything and then fails, wasting a pass at every
    // level. pdqsort can afford it there because its check *sorts* within a
    // bounded budget, so the work is never wasted; this one is pure detection
    // and cannot make progress. ipnsort places it at entry for the same
    // reason.
    //
    // At entry the trade is all upside: unstructured input bails after about
    // one vector, and fully ordered input finishes here.
    if is_ordered::<V, O>(keys) {
        return;
    }
    if is_reverse_ordered::<V, O>(keys) {
        keys.reverse();
        return;
    }

    // NaN pre-pass, after the run detectors on purpose: `is_ordered` returning
    // true already proves the slice NaN-free (a NaN fails both of its adjacent
    // comparisons), so sorted input never reaches this scan. The
    // `HAS_UNORDERED` gate folds the whole block away for integer elements.
    //
    // NaNs go to the back of the slice for both orders, bit patterns intact,
    // and the recursion below sees only the ordered prefix. That is what makes
    // the base case's sentinel padding sound: no value sorts past NaN, and
    // `min`/`max` NaN semantics legitimately differ between backends.
    let mut keys = keys;
    if const { <V::Element as Element>::HAS_UNORDERED } && has_unordered::<V>(keys) {
        let ordered = partition_unordered_last::<V>(keys);
        // SAFETY: a partition returns a split point within its range.
        unsafe { assume_split(ordered, keys.len()) };
        keys = &mut keys[..ordered];
        // The sweep can expose structure the entry check could not see -
        // sorted data with NaNs sprinkled through it - and the re-check is
        // cheap next to the sweep just paid for.
        if is_ordered::<V, O>(keys) {
            return;
        }
        if is_reverse_ordered::<V, O>(keys) {
            keys.reverse();
            return;
        }
    }

    // 2*log2(n) partitions is the standard introsort budget: a healthy pivot
    // halves the range, so exceeding it means the pivots are pathological.
    let depth = 2 * (usize::BITS - keys.len().leading_zeros()) as usize;
    recurse::<V, O>(keys, None, depth);
}

#[thermite::dispatch(V)]
fn recurse<V: MergeVector, O: SortOrder>(
    keys: &mut [V::Element],
    ancestor_pivot: Option<V::Element>,
    depth: usize,
) where
    V::Element: PartialOrd,
{
    let mut keys = keys;
    let mut depth = depth;
    let mut ancestor_pivot = ancestor_pivot;
    loop {
        if keys.len() <= max_len::<V>() {
            base_case::<V, O>(keys);
            return;
        }

        if depth == 0 {
            // Pivot quality has failed; fall back to something with a
            // guaranteed bound rather than risk quadratic behaviour.
            heapsort::<V, O>(keys);
            return;
        }
        depth -= 1;

        let pivot_val = choose_pivot::<V, O>(keys);
        let pivot = V::splat(pivot_val);

        // ipnsort's duplicate handling. If this pivot does not sort strictly
        // after the pivot the parent split on, every key here is at or after
        // that value, so this pivot is the minimum of the range. Partitioning
        // with equals sent *left* then consumes the whole run of duplicates in
        // one pass, making a k-distinct input `O(n * k)` instead of letting it
        // degrade.
        //
        // This also subsumes the extreme-pivot case below: a split of 0 means
        // nothing sorts before the pivot, which is the same situation reached
        // by a different route.
        let equal_run = match ancestor_pivot {
            Some(p) => !sorts_strictly_before::<V, O>(&p, &pivot_val),
            None => false,
        };

        if equal_run {
            let num_le = partition_equal::<V, O>(keys, pivot);
            // SAFETY: a partition returns a split point within the range it was
            // given. See `assume_split` for why this is stated rather than
            // relied on implicitly.
            unsafe { assume_split(num_le, keys.len()) };
            if num_le == keys.len() {
                return; // the whole range is one value
            }
            keys = &mut keys[num_le..];
            ancestor_pivot = None;
            continue;
        }

        let split = partition::<V, O>(keys, pivot);
        // SAFETY: as above.
        unsafe { assume_split(split, keys.len()) };

        if split == 0 {
            // The pivot is the minimum. Same remedy, and it guarantees progress:
            // at least the pivot's own copies are consumed.
            let num_le = partition_equal::<V, O>(keys, pivot);
            // SAFETY: as above.
            unsafe { assume_split(num_le, keys.len()) };
            if num_le == keys.len() {
                return;
            }
            keys = &mut keys[num_le..];
            ancestor_pivot = None;
            continue;
        }

        // Recurse into the smaller side, loop on the larger: bounds stack depth
        // at log2(n) regardless of how lopsided the splits are. Only the right
        // side inherits the pivot, since only there is it a lower bound.
        let (left, right) = keys.split_at_mut(split);
        if left.len() < right.len() {
            recurse::<V, O>(left, ancestor_pivot, depth);
            keys = right;
            ancestor_pivot = Some(pivot_val);
        } else {
            recurse::<V, O>(right, Some(pivot_val), depth);
            keys = left;
        }
    }
}

/// Tell the optimizer that a split point returned by a partition is within the
/// range that was partitioned.
///
/// The partitions are `#[thermite::dispatch]` boundaries, so they are separate
/// out-of-line functions and the optimizer cannot see that their return value
/// indexes the slice it was handed. Without this, the reslice and `split_at_mut`
/// below each keep a bounds check. ipnsort states the same fact the same way
/// (`intrinsics::assume(num_lt < v.len())` right after its partition).
///
/// Stating the invariant rather than switching to unchecked slicing keeps the
/// safe API at the use sites, so a future edit that breaks the invariant is a
/// panic in debug rather than silent unsoundness.
///
/// # Safety
///
/// `split <= len` must hold.
#[inline(always)]
unsafe fn assume_split(split: usize, len: usize) {
    debug_assert!(split <= len, "partition returned {split} for a range of {len}");
    // SAFETY: the caller guarantees the bound.
    unsafe { core::hint::assert_unchecked(split <= len) };
}

/// Strict "`a` sorts before `b`" under `O`.
#[inline(always)]
fn sorts_strictly_before<V: NumericVector, O: SortOrder>(a: &V::Element, b: &V::Element) -> bool
where
    V::Element: PartialOrd,
{
    if const { O::IS_ASCENDING } { a < b } else { a > b }
}

/// Choose a pivot: the recursive pseudomedian used by ipnsort.
///
/// Above [`PSEUDO_MEDIAN_REC_THRESHOLD`] each of the three samples is itself a
/// pseudomedian of three, so the sample grows with the input - `f(n) =
/// 3f(n/8)`, giving `O(n^0.528)` samples at logarithmic depth. A sample that
/// strong is what lets the recursion stay deterministic, with no
/// shuffle-on-a-bad-partition step.
///
/// **Median-of-three is not enough here, for a reason specific to this
/// partition.** [`crate::partition`] reads blocks alternately from both ends
/// and writes them in processing order, so structured input comes out
/// block-scrambled: the keys at the start, middle and end of a sub-range stop
/// being spread through its value range, and on sorted input they land near
/// the same extreme. A weak pivot rule therefore makes *structured* input the
/// worst case - splits of a few keys at a time, and enough of them to exhaust
/// the depth budget into heapsort - which is the opposite of what a caller
/// expects.
///
/// Sample offsets are 0, `4n/8` and `7n/8`, asymmetric so they do not line up
/// with periodic structure the way evenly spaced samples can.
#[inline(always)]
fn choose_pivot<V: NumericVector, O: SortOrder>(keys: &[V::Element]) -> V::Element
where
    V::Element: PartialOrd,
{
    let n = keys.len();
    if n < 8 {
        return keys[median3_idx::<V, O>(keys, 0, n / 2, n - 1)];
    }

    let d8 = n / 8;
    let i = if n < PSEUDO_MEDIAN_REC_THRESHOLD {
        median3_idx::<V, O>(keys, 0, d8 * 4, d8 * 7)
    } else {
        median3_rec::<V, O>(keys, 0, d8 * 4, d8 * 7, d8, MAX_SAMPLE_DEPTH)
    };
    debug_assert!(i < n);
    // SAFETY: every index the sampler produces is a section start, and each
    // section lies inside `keys` - see `median3_rec`.
    unsafe { *keys.get_unchecked(i) }
}

/// Below this, sample three points; above, sample each of the three recursively.
const PSEUDO_MEDIAN_REC_THRESHOLD: usize = 64;

/// Cap on [`median3_rec`]'s recursion depth, bounding the sample at
/// `3^(1 + MAX_SAMPLE_DEPTH)` points regardless of input size.
///
/// Uncapped by default, matching ipnsort. Capping it makes the sample a
/// constant size, as vqsort's does; that trades pivot quality on structured
/// input for sampling cost, and the sampling here is well under 1% of the
/// sort either way.
const MAX_SAMPLE_DEPTH: u32 = u32::MAX;

/// Recurse while each section is still large enough to be worth sampling.
///
/// Invariant, which is what licenses the unchecked loads in [`median3_idx`]:
/// each of `a`, `b`, `c` starts a section of length `n` lying wholly inside
/// `keys`. It holds at entry (`choose_pivot` passes `0`, `4*(len/8)`,
/// `7*(len/8)` with `n = len/8`, and `8*(len/8) <= len`), and each recursion
/// takes sub-sections at `+4*(n/8)` and `+7*(n/8)` of length `n/8`, all inside
/// `[a, a + 8*(n/8)) ⊆ [a, a + n)`.
fn median3_rec<V: NumericVector, O: SortOrder>(
    keys: &[V::Element],
    mut a: usize,
    mut b: usize,
    mut c: usize,
    n: usize,
    depth: u32,
) -> usize
where
    V::Element: PartialOrd,
{
    debug_assert!(a + n <= keys.len() && b + n <= keys.len() && c + n <= keys.len());

    if depth > 0 && n * 8 >= PSEUDO_MEDIAN_REC_THRESHOLD {
        let n8 = n / 8;
        a = median3_rec::<V, O>(keys, a, a + n8 * 4, a + n8 * 7, n8, depth - 1);
        b = median3_rec::<V, O>(keys, b, b + n8 * 4, b + n8 * 7, n8, depth - 1);
        c = median3_rec::<V, O>(keys, c, c + n8 * 4, c + n8 * 7, n8, depth - 1);
    }
    median3_idx::<V, O>(keys, a, b, c)
}

/// Index of the median of three positions. Two comparisons in the common case;
/// the XOR picks between `b` and `c` without a second branch (from ipnsort).
///
/// Unchecked loads: the sampler runs `O(n^0.528)` times per partition, which
/// sums to a substantial count across the recursion tree, and the bounds checks
/// were showing up in the emitted code. Callers guarantee the indices - see
/// [`median3_rec`] - and `debug_assert` re-checks them in the test builds.
#[inline(always)]
fn median3_idx<V: NumericVector, O: SortOrder>(keys: &[V::Element], a: usize, b: usize, c: usize) -> usize
where
    V::Element: PartialOrd,
{
    debug_assert!(a < keys.len() && b < keys.len() && c < keys.len());
    // SAFETY: asserted above; the release path relies on the caller invariant.
    let (ka, kb, kc) = unsafe { (keys.get_unchecked(a), keys.get_unchecked(b), keys.get_unchecked(c)) };

    let x = sorts_strictly_before::<V, O>(ka, kb);
    let y = sorts_strictly_before::<V, O>(ka, kc);
    if x == y {
        // Both on the same side of `a`, so the answer is whichever of b/c is
        // inward; XOR with `x` flips the sense for the two cases.
        let z = sorts_strictly_before::<V, O>(kb, kc);
        if z ^ x { c } else { b }
    } else {
        a
    }
}

/// `O(n log n)` worst-case fallback for a range that exhausts its depth
/// budget. Heapsort rather than a merge sort because it needs no allocation,
/// which keeps this crate `no_std`.
///
/// `inline(never)` on purpose: this sits in the hot loop of [`recurse`] but is
/// the unlikely branch - with the sampled pivot it does not fire on realistic
/// input at all. Inlining pulls its scalar indexing and bounds checks into
/// `recurse` for work that never runs.
#[inline(never)]
fn heapsort<V: NumericVector, O: SortOrder>(keys: &mut [V::Element])
where
    V::Element: PartialOrd,
{
    let after = |x: &V::Element, y: &V::Element| if const { O::IS_ASCENDING } { x > y } else { x < y };

    let n = keys.len();
    for start in (0..n / 2).rev() {
        sift_down::<V>(keys, start, n, &after);
    }
    for end in (1..n).rev() {
        keys.swap(0, end);
        sift_down::<V>(keys, 0, end, &after);
    }
}

#[inline(always)]
fn sift_down<V: NumericVector>(
    keys: &mut [V::Element],
    mut root: usize,
    end: usize,
    after: &impl Fn(&V::Element, &V::Element) -> bool,
) {
    loop {
        let mut child = 2 * root + 1;
        if child >= end {
            return;
        }
        if child + 1 < end && after(&keys[child + 1], &keys[child]) {
            child += 1;
        }
        if after(&keys[child], &keys[root]) {
            keys.swap(root, child);
            root = child;
        } else {
            return;
        }
    }
}
