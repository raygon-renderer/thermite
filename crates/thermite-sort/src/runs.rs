//! Structure detection: is this range already ordered?
//!
//! [`is_ordered`] compares whole vectors of adjacent pairs and returns at the
//! first violation, so on unstructured input it exits after about one vector -
//! the very first adjacent pair is out of order half the time. A full scan
//! costs a fraction of a cycle per element and finishes the range outright,
//! skipping every level of recursion beneath it.
//!
//! That asymmetry only pays at the top. [`crate::sort_by`] runs these once, at
//! entry, and not per level; see the note in `quicksort` for why.

use thermite::prelude::*;
use thermite::sort::SortOrder;

/// Is `keys` already in `O` order?
///
/// Returns at the first out-of-order pair, so failure costs O(1) in practice.
#[inline(always)]
pub fn is_ordered<V: NumericVector, O: SortOrder>(keys: &[V::Element]) -> bool
where
    V::Element: PartialOrd,
{
    let n = V::LANES;
    let len = keys.len();
    if len < 2 {
        return true;
    }

    let ptr = keys.as_ptr();
    let mut i = 0usize;

    // Each block checks the `n` adjacent pairs starting at `i` by comparing the
    // window at `i` against the window at `i + 1`, so advancing by `n` covers
    // every pair with no gaps.
    while i + n < len {
        // SAFETY: the guard keeps both windows inside `keys`.
        let (a, b) = unsafe { (V::load_unaligned(ptr.add(i)), V::load_unaligned(ptr.add(i + 1))) };
        let ok = if const { O::IS_ASCENDING } { a.cmp_le(b) } else { a.cmp_ge(b) };
        if !ok.all() {
            return false;
        }
        i += n;
    }

    // The scalar tail is the exact negation of the vector test, not the
    // flipped comparison. For a NaN pair, `a <= b` and `a > b` are both false,
    // so testing `a > b` as "out of order" would call a slice containing NaN
    // ordered while the vector path above rejects it.
    while i + 1 < len {
        let ok = if const { O::IS_ASCENDING } {
            keys[i] <= keys[i + 1]
        } else {
            keys[i] >= keys[i + 1]
        };
        if !ok {
            return false;
        }
        i += 1;
    }

    true
}

/// Does `keys` contain a value unordered under `PartialOrd` - float NaN?
///
/// A value is unordered iff it does not equal itself, so the test is
/// `v.cmp_eq(v)` per vector with no per-type code; on integer vectors it can
/// never fire (and the caller gates on `Element::HAS_UNORDERED`, so this is
/// not even emitted for them).
///
/// Runs *after* the run detectors on purpose: an adjacent NaN fails both
/// `cmp_le` and `cmp_gt`, so `is_ordered` returning `true` already proves the
/// slice NaN-free (any NaN makes both its adjacent pairs unordered) and
/// sorted input never pays this scan. Unrolled by 4 with the `any` test
/// per block: the NaN-free case wants full-pass throughput, and a hit still
/// exits within a block.
#[inline(always)]
pub fn has_unordered<V: NumericVector>(keys: &[V::Element]) -> bool {
    let n = V::LANES;
    let len = keys.len();
    let ptr = keys.as_ptr();
    let mut i = 0usize;

    while i + 4 * n <= len {
        // SAFETY: the guard keeps all four windows inside `keys`.
        let ordered = unsafe {
            let a = V::load_unaligned(ptr.add(i));
            let b = V::load_unaligned(ptr.add(i + n));
            let c = V::load_unaligned(ptr.add(i + 2 * n));
            let d = V::load_unaligned(ptr.add(i + 3 * n));
            a.cmp_eq(a) & b.cmp_eq(b) & c.cmp_eq(c) & d.cmp_eq(d)
        };
        if !ordered.all() {
            return true;
        }
        i += 4 * n;
    }
    while i + n <= len {
        // SAFETY: as above.
        let v = unsafe { V::load_unaligned(ptr.add(i)) };
        if !v.cmp_eq(v).all() {
            return true;
        }
        i += n;
    }
    while i < len {
        if keys[i].partial_cmp(&keys[i]).is_none() {
            return true;
        }
        i += 1;
    }

    false
}

/// Is `keys` in exactly the *opposite* of `O` order, strictly?
///
/// Strict on purpose: a run of equal keys is both ascending and descending, so
/// admitting ties here would let the caller "reverse" an all-equal range and
/// call it sorted, which is correct but pointless, and would misreport ranges
/// that are ordered apart from ties. [`is_ordered`] already claims those.
#[inline(always)]
pub fn is_reverse_ordered<V: NumericVector, O: SortOrder>(keys: &[V::Element]) -> bool
where
    V::Element: PartialOrd,
{
    let n = V::LANES;
    let len = keys.len();
    if len < 2 {
        return true;
    }

    let ptr = keys.as_ptr();
    let mut i = 0usize;

    while i + n < len {
        // SAFETY: the guard keeps both windows inside `keys`.
        let (a, b) = unsafe { (V::load_unaligned(ptr.add(i)), V::load_unaligned(ptr.add(i + 1))) };
        let ok = if const { O::IS_ASCENDING } { a.cmp_gt(b) } else { a.cmp_lt(b) };
        if !ok.all() {
            return false;
        }
        i += n;
    }

    // Exact negation of the vector test, for the same NaN reason as in
    // `is_ordered`: on a NaN pair both directions compare false, so a pair
    // containing NaN must fail this test rather than pass it.
    while i + 1 < len {
        let ok = if const { O::IS_ASCENDING } {
            keys[i] > keys[i + 1]
        } else {
            keys[i] < keys[i + 1]
        };
        if !ok {
            return false;
        }
        i += 1;
    }

    true
}
