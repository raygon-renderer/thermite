//! In-place branchless partition - the core of a quicksort.
//!
//! # The scheme
//!
//! Bidirectional and in place, as in BlockQuicksort and vqsort's `Partition`.
//! A prologue lifts one block from each end into registers to open a write
//! hole; the loop then reads from whichever end has more slack and writes both
//! ways into the hole.
//!
//! Two regions cannot use the overlapping-store trick and ride a small fixed
//! stack buffer instead. Both are important for correctness, not just for
//! order - getting either wrong duplicates and drops keys:
//!
//! - **The ragged tail.** The main loop consumes a whole block per iteration,
//!   so it only terminates when the region it walks is an exact multiple of
//!   one. The trailing `num % block` keys are lifted into the buffer up front,
//!   which also frees their slots for the loop to write into.
//! - **The last two vectors.** Each step needs at least two vectors of slack
//!   between the write cursors, or its two stores overwrite each other's
//!   useful halves. The final stores buffer their keys instead.
//!
//! At the end the buffered left keys fill the hole at the left cursor and the
//! buffered right keys sit directly above them, which is exactly where the two
//! cursors meet.
//!
//! # Why this needs no hardware compress
//!
//! Partitioning looks like it wants `vpcompressd`, which does not exist below
//! AVX-512. It turns out the table-based path is the better one per vector
//! anyway.
//!
//! Thermite's [`compress`](thermite::vector::GenericVector::compress) resolves
//! its permutation from a table whose rows **stably partition** the lanes:
//! selected lanes first in order, then unselected lanes, also in order. Both
//! sides survive. So one `compress` plus two full-vector stores partitions a
//! whole vector - the selected prefix lands at the left cursor, the unselected
//! suffix at the right cursor, and the overlap between them is slack that
//! later iterations overwrite.
//!
//! `vpcompressd` keeps only the selected side, so a native-compress target has
//! to issue *two* compress-stores per vector. vqsort carries both paths and
//! picks the table one wherever compress-is-partition holds, including on
//! AVX-512 for 64-bit lanes.
//!
//! That property is important and invisible to most tests, so
//! `tests/compress_is_partition.rs` pins it across every mask value on every
//! x86 tier: a `compress` that merely left-packed would pass every
//! stream-compaction test in the tree while silently destroying half of each
//! partition step.

use core::mem::MaybeUninit;

use thermite::element::Element;
use thermite::prelude::*;
use thermite::sort::SortOrder;

/// Vectors consumed per loop iteration. Two is enough to saturate memory
/// bandwidth, but four sorts measurably faster; vqsort uses four as well.
pub const UNROLL: usize = 4;

// Software prefetch, as vqsort does it (3 blocks ahead on whichever side was
// just read), was tried here and did not pay: no gain on a Zen 3, which
// already tracks both streams, and it costs an instruction per block plus
// prefetch-queue pressure. It may still be worth it on a machine with weaker
// hardware prefetchers - the loop waits on memory rather than on compute, and
// the right-hand cursor walks downwards, which prefetchers chase less eagerly.
// `thermite::backend::prefetch::prefetch::<3, false>` is the call, and it is
// safe for out-of-bounds addresses, which is what makes naming one three
// blocks past the end legal.

/// Stack scratch, in elements. Must hold the ragged tail (`< UNROLL * LANES`)
/// plus the final two vectors' keys, i.e. `(UNROLL + 2) * LANES`. Sized for
/// 32-lane registers with room to spare, and checked in debug rather than
/// trusted.
const BUF: usize = 256;

/// Partition `keys` around `pivot`, returning the index where the right side
/// begins: everything below it sorts before the pivot under `O`, everything
/// from it does not.
///
/// O(1) additional storage - the scratch is a fixed stack array, never
/// proportional to the input.
#[thermite::dispatch(V)]
pub fn partition<V: NumericVector, O: SortOrder>(keys: &mut [V::Element], pivot: V) -> usize
where
    V::Element: PartialOrd,
{
    partition_by::<V, PivotLeft<O, false>>(keys, pivot)
}

/// Partition with keys *equal* to the pivot sent left instead of right.
///
/// Used when the pivot is known to be the minimum of the range, which makes
/// this consume an entire run of duplicates in one pass, so a k-distinct input
/// costs `O(n * k)` rather than degrading. It is the same kernel and the same
/// machine code as [`partition`], with `cmp_le` in place of `cmp_lt`.
#[thermite::dispatch(V)]
pub fn partition_equal<V: NumericVector, O: SortOrder>(keys: &mut [V::Element], pivot: V) -> usize
where
    V::Element: PartialOrd,
{
    partition_by::<V, PivotLeft<O, true>>(keys, pivot)
}

/// Sweep *unordered* keys - float NaN - to the back of the slice, returning
/// the length of the ordered prefix.
///
/// Same kernel as [`partition`], with the predicate "is ordered" instead of a
/// pivot comparison: a value is unordered iff it does not equal itself, so
/// `v.cmp_eq(v)` masks the keepers with no per-type code - including for
/// composite vectors, whose comparisons delegate to their value part. Order
/// direction plays no role: NaNs go last for both.
///
/// Key bit patterns are preserved (everything moves through whole-lane
/// permutes and stores), so the result is a permutation of the input.
#[thermite::dispatch(V)]
pub fn partition_unordered_last<V: NumericVector>(keys: &mut [V::Element]) -> usize {
    partition_by::<V, OrderedLeft>(keys, V::ZERO)
}

/// How [`partition_by`] decides which side a key belongs on. The vector and
/// scalar answers must agree lane-for-lane; the scalar form serves the short
/// fallback path.
pub(crate) trait PartitionPredicate<V: NumericVector> {
    /// Mask of lanes that belong on the LEFT.
    fn left(v: V, pivot: V) -> V::Mask;
    /// Scalar twin of [`left`](Self::left).
    fn left_one(k: &V::Element, pivot: V) -> bool;
}

/// The quicksort predicate: lanes that sort before the pivot under `O` go
/// left. `INCLUSIVE` decides whether keys equal to the pivot do too. The
/// order picks the comparison, so descending needs no special case.
pub(crate) struct PivotLeft<O, const INCLUSIVE: bool>(core::marker::PhantomData<O>);

impl<V: NumericVector, O: SortOrder, const INCLUSIVE: bool> PartitionPredicate<V> for PivotLeft<O, INCLUSIVE>
where
    V::Element: PartialOrd,
{
    #[inline(always)]
    fn left(v: V, pivot: V) -> V::Mask {
        match (const { O::IS_ASCENDING }, const { INCLUSIVE }) {
            (true, false) => v.cmp_lt(pivot),
            (true, true) => v.cmp_le(pivot),
            (false, false) => v.cmp_gt(pivot),
            (false, true) => v.cmp_ge(pivot),
        }
    }

    #[inline(always)]
    fn left_one(k: &V::Element, pivot: V) -> bool {
        let p = pivot.extractv(0);
        match (const { O::IS_ASCENDING }, const { INCLUSIVE }) {
            (true, false) => *k < p,
            (true, true) => *k <= p,
            (false, false) => *k > p,
            (false, true) => *k >= p,
        }
    }
}

/// The NaN-sweep predicate: ordered lanes go left, unordered (NaN) right.
/// The pivot is ignored. On integer vectors `left` is always all-true.
pub(crate) struct OrderedLeft;

impl<V: NumericVector> PartitionPredicate<V> for OrderedLeft {
    #[inline(always)]
    fn left(v: V, _pivot: V) -> V::Mask {
        v.cmp_eq(v)
    }

    #[inline(always)]
    fn left_one(k: &V::Element, _pivot: V) -> bool {
        // `partial_cmp` with itself is `None` exactly for unordered values.
        k.partial_cmp(k).is_some()
    }
}

/// The shared kernel; `P` decides sides.
#[inline(always)]
fn partition_by<V: NumericVector, P: PartitionPredicate<V>>(
    keys: &mut [V::Element],
    pivot: V,
) -> usize
{
    let n = V::LANES;
    let block = UNROLL * n;
    let num = keys.len();

    // Below two blocks there is no room for the prologue, let alone slack.
    if num < 2 * block {
        return scalar_partition::<V, P>(keys, pivot);
    }

    debug_assert!(block + 2 * n <= BUF, "scratch too small for {n}-lane registers");

    // Buffered keys that cannot be placed until both cursors are final.
    //
    // Uninitialized on purpose: only `[..n_l]` and `[..n_r]` are ever written
    // or read, and a `[ZERO; BUF]` here is two unconditional 1 KiB `memset`
    // calls at every entry. The recursion calls this once per ~80 keys, so
    // per-call overhead is a fixed tax on the whole sort.
    let mut buf_l = [MaybeUninit::<V::Element>::uninit(); BUF];
    let mut buf_r = [MaybeUninit::<V::Element>::uninit(); BUF];
    let mut n_l = 0usize;
    let mut n_r = 0usize;

    let base = keys.as_mut_ptr();
    let mut write_l = 0usize;
    let mut remaining = num;

    // --- ragged tail -------------------------------------------------------
    // Lift `num % block` keys out so the main region is an exact multiple of a
    // block, and so their slots become writable. Vectorized rather than a
    // scalar walk: a per-key branch here is up to `block - 1` coin flips per
    // call on random input, which is the same per-call tax as the memsets.
    let main = (num / block) * block;
    {
        let mut i = main;
        while i + n <= num {
            // SAFETY: `[i, i + n)` is in bounds since `i + n <= num`; buffer
            // slack per the scratch check (see `place_buffered`).
            unsafe {
                let v = V::load_unaligned(base.add(i));
                let left = P::left(v, pivot);
                place_buffered(v, left, !left, &mut buf_l, &mut n_l, &mut buf_r, &mut n_r);
            }
            i += n;
        }
        if i < num {
            // The last `n` keys of the slice cover the remainder; the leading
            // lanes belong to vectors already processed (or to the main
            // region) and are masked out of both sides. In bounds because
            // `num >= 2 * block >= n`, and this runs before anything stores.
            unsafe {
                let v = V::load_unaligned(base.add(num - n));
                let lead = n - (num - i);
                let valid = V::indexed().cmp_ge(V::splat(<V::Element as Element>::from_u8(lead as u8)));
                let left = P::left(v, pivot);
                place_buffered(v, left & valid, valid.bitandnot(left), &mut buf_l, &mut n_l, &mut buf_r, &mut n_r);
            }
        }
    }

    // --- prologue ----------------------------------------------------------
    // Lift a block from each end into registers, opening the write hole.
    let mut head = [V::ZERO; UNROLL];
    let mut tail = [V::ZERO; UNROLL];
    unsafe {
        let mut i = 0;
        while i < UNROLL {
            head[i] = V::load_unaligned(base.add(i * n));
            tail[i] = V::load_unaligned(base.add(main - block + i * n));
            i += 1;
        }
    }

    let mut read_l = block;
    let mut read_r = main - block;

    // --- main loop ---------------------------------------------------------
    while read_l != read_r {
        let capacity_l = read_l - write_l;

        let mut blk = [V::ZERO; UNROLL];
        unsafe {
            if capacity_l > block {
                read_r -= block;
                let mut i = 0;
                while i < UNROLL {
                    blk[i] = V::load_unaligned(base.add(read_r + i * n));
                    i += 1;
                }
            } else {
                let mut i = 0;
                while i < UNROLL {
                    blk[i] = V::load_unaligned(base.add(read_l + i * n));
                    i += 1;
                }
                read_l += block;
            }
        }

        let mut i = 0;
        while i < UNROLL {
            // SAFETY: slack is guaranteed here - see the reservation argument
            // below the epilogue.
            unsafe { place_stores::<V, P>(blk[i], pivot, base, &mut write_l, &mut remaining) };
            i += 1;
        }
    }

    // --- epilogue: drain the register-held blocks --------------------------
    //
    // Slack shrinks by one vector per store, so only the very last stores can
    // run short of it. Reserving the final two for the buffered path keeps the
    // branch out of the loop entirely: at the third-from-last store the gap is
    // still `ragged + 3n >= 2n`, so every fast-path call is safe by
    // construction rather than by a runtime test.
    const RESERVED: usize = 2;

    let mut i = 0;
    while i < UNROLL {
        // SAFETY: at least `RESERVED + 1` stores remain after this one.
        unsafe { place_stores::<V, P>(head[i], pivot, base, &mut write_l, &mut remaining) };
        i += 1;
    }
    let mut i = 0;
    while i < UNROLL - RESERVED {
        // SAFETY: as above.
        unsafe { place_stores::<V, P>(tail[i], pivot, base, &mut write_l, &mut remaining) };
        i += 1;
    }
    let mut i = UNROLL - RESERVED;
    while i < UNROLL {
        let left = P::left(tail[i], pivot);
        // SAFETY: buffer slack per the scratch check (see `place_buffered`).
        unsafe { place_buffered(tail[i], left, !left, &mut buf_l, &mut n_l, &mut buf_r, &mut n_r) };
        i += 1;
    }

    // Everything not yet in the array is in the buffers, and the gap between the
    // cursors is exactly their size.
    debug_assert_eq!(remaining, n_l + n_r, "buffered keys must account for the gap");

    // SAFETY: every slot below `n_l` / `n_r` was written exactly once, by the
    // ragged tail or by `place_buffered`.
    let (init_l, init_r) = unsafe {
        (
            core::slice::from_raw_parts(buf_l.as_ptr().cast::<V::Element>(), n_l),
            core::slice::from_raw_parts(buf_r.as_ptr().cast::<V::Element>(), n_r),
        )
    };
    keys[write_l..write_l + n_l].copy_from_slice(init_l);
    keys[write_l + n_l..write_l + n_l + n_r].copy_from_slice(init_r);

    write_l + n_l
}

/// Compare under the predicate and left-pack. Returns the partitioned vector -
/// left keys in `[0, num_left)`, right keys in `[num_left, LANES)` - and the
/// split point.
#[inline(always)]
fn split_one<V: NumericVector, P: PartitionPredicate<V>>(v: V, pivot: V) -> (V, usize) {
    let goes_left = P::left(v, pivot);
    (v.compress(goes_left), goes_left.count_set())
}

/// The hot path: one shuffle and two overlapping full-vector stores.
///
/// No branch, on purpose. The caller guarantees at least two vectors of slack
/// between the cursors, which is what makes both stores safe. Testing that
/// here instead is far more expensive than the compare suggests: the other arm
/// (a stack spill, a per-lane scalar copy loop and its bounds checks) inlines
/// into the loop body at every unroll position, so a five-instruction fast
/// path ends up sharing a loop with several hundred instructions it never
/// executes.
#[inline(always)]
unsafe fn place_stores<V: NumericVector, P: PartitionPredicate<V>>(
    v: V,
    pivot: V,
    keys: *mut V::Element,
    write_l: &mut usize,
    remaining: &mut usize,
) {
    let n = V::LANES;
    debug_assert!(*remaining >= 2 * n, "caller must guarantee two vectors of slack");

    let (lr, num_left) = split_one::<V, P>(v, pivot);
    *remaining -= n;
    unsafe {
        // Left keys land at the left cursor; the tail of this store is slack.
        lr.store_unaligned(keys.add(*write_l));
        // The same vector placed so its right keys land just under the right
        // cursor; the head of this store is slack.
        lr.store_unaligned(keys.add(*remaining + *write_l));
    }
    *write_l += num_left;
}

/// The cold path, for the ragged tail and the final vectors where the cursors
/// are too close for the two overlapping stores to stay out of each other's
/// way.
///
/// Appends `v`'s keys under `left` to `buf_l` and those under `right` to
/// `buf_r`; lanes in neither mask are dropped, which is how the ragged tail's
/// partial vector masks off the lanes it does not own.
///
/// Both stores are full vectors. Everything past the counted prefix is junk
/// that the next append or the final copy-out overwrites, so this needs no
/// staging array, no `memcpy` and no scalar loop.
///
/// `remaining` deliberately does not move: it counts *unwritten slots*, and
/// these keys have not been placed. The ragged tail is accounted the same way,
/// so at the end `remaining == n_l + n_r`.
///
/// # Safety
///
/// Requires `*n_l + LANES <= BUF` and `*n_r + LANES <= BUF`, which the
/// partition's scratch check guarantees: at most `block - 1` tail keys plus
/// `RESERVED` whole vectors are ever buffered, and the last full store starts
/// no deeper than that total, so `block + 2 * LANES <= BUF` covers it.
#[inline(always)]
unsafe fn place_buffered<V: NumericVector>(
    v: V,
    left: V::Mask,
    right: V::Mask,
    buf_l: &mut [MaybeUninit<V::Element>; BUF],
    n_l: &mut usize,
    buf_r: &mut [MaybeUninit<V::Element>; BUF],
    n_r: &mut usize,
) {
    let n = V::LANES;
    debug_assert!(*n_l + n <= BUF && *n_r + n <= BUF);

    // Selected lanes pack to the front of a `compress`, in order.
    let l = v.compress(left);
    let r = v.compress(right);
    unsafe {
        l.store_unaligned(buf_l.as_mut_ptr().add(*n_l).cast::<V::Element>());
        r.store_unaligned(buf_r.as_mut_ptr().add(*n_r).cast::<V::Element>());
    }
    *n_l += left.count_set();
    *n_r += right.count_set();
}

/// Scalar Hoare partition, for runs too short for the vector path.
#[inline(always)]
fn scalar_partition<V: NumericVector, P: PartitionPredicate<V>>(
    keys: &mut [V::Element],
    pivot: V,
) -> usize {
    let mut w = 0;
    for i in 0..keys.len() {
        if P::left_one(&keys[i], pivot) {
            keys.swap(i, w);
            w += 1;
        }
    }
    w
}
