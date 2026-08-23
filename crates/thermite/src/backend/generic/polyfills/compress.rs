//! Generic stream-compaction (`compress` / left-pack) polyfills.
//!
//! "Better than scalar" generic implementations of
//! [`Register::compress`]/[`compress_z`](Register::compress_z) that a backend can
//! delegate to when its registers satisfy the extra trait bounds. The base trait
//! default is a scalar element-by-element compaction (correct for every
//! register, no bounds); these resolve the permutation from compile-time tables
//! instead.
//!
//! Pick by lane count and by whether the register is *native* (one hardware
//! permute covers all its lanes) or a chunked
//! [`ArrayRegister`](crate::register::array::ArrayRegister):
//!
//! - [`compress_permute`] handles up to 8 lanes ([`CompressTable`]): `movemask`
//!   -> table row -> one permute. Branchless and loop-free.
//! - [`compress_z_grouped`] is the *zeroing* left-pack for a native
//!   16/32/64-lane register: one permute compacts every 8-lane group in place,
//!   then `log2(LANES / 8)` permutes merge the groups pairwise. Branchless, no
//!   scalar scatter. This is the `compress_z` arm of `compress_via_wide!`.
//! - [`compress_grouped`] is the *non-zeroing* form at the same shapes, and the
//!   `compress` arm of `compress_via_wide!`. Two [`compress_z_grouped`] trees
//!   (one under the mask, one under its complement), shifted into place by a
//!   single `swizzle` against [`EMPTY`](CoreRegister::EMPTY) and combined with
//!   one `bitor`. Both trees share one `movemask`, and the complement tree needs
//!   no mask-register complement: it indexes the same tables with `!bm`.
//! - [`compress_z_merge2`] / [`compress_z_merge4`] / [`compress_z_merge8`] are
//!   the *zeroing* left-pack for an `ArrayRegister` of 2/4/8 native chunks:
//!   compact each chunk with its own `compress_z`, then merge the results
//!   pairwise up a binary tree. Each merge is [`merge_pair`]: one `swizzle` per
//!   output chunk plus one `bitor`, no control table and no cross-chunk
//!   `array_swizzle`. This is the `ArrayRegister::compress_z` path at every
//!   supported chunk shape.
//! - [`compress_permute_wide`] is the remaining fallback, for any multiple of 8
//!   lanes up to 64: per-8-lane table lookups assembled by a scalar per-lane
//!   scatter into one global gather index, resolved by one full-width permute.
//!   Non-zeroing. No native register routes here any more (they take
//!   [`compress_grouped`]). It survives as the `ArrayRegister` non-zeroing
//!   `compress`, which has no single-register permute to build a grouped kernel
//!   on. Branchless, but the index scatter is a long dependency chain, so prefer
//!   any of the above where the shape allows.
//!
//! The inverse direction lives in [`super::expand`], a mirror image of
//! this module. This module owns everything the two share: [`CompressRow`],
//! [`CompressTable`], and [`COMPRESS8`], which `EXPAND8` is the row-wise
//! inverse of. Keep it that way: compress-only code here, expand-only code
//! there, shared code here regardless of which side reads it more.

use core::mem::MaybeUninit;

use generic_array::{
    ArrayLength, GenericArray,
    sequence::GenericSequence,
    typenum::{U1, U2, U3, U4, U5, U6, U7, U8, U16, U32, U64, U256, Unsigned},
};

use super::*;

/// One table row: the stable-partition gather indices for an 8-lane mask plus
/// its population count (so callers never recompute it at runtime).
///
/// Indices are stored as `u8`, not the `u32` that
/// [`permutev`](Register::permutev) consumes. Every index is in `0..8`, so the
/// row shrinks from 36 padded bytes to 9 and the two tables together drop from
/// ~18 KB of `.rodata` to ~4.6 KB, worth it because the rows are indexed
/// randomly by mask and wavefront code hits both directions in one loop. The
/// widening back to the index element width is
/// [`Register::widen_index_bytes`], one `pmovzxbd`-shaped load on any backend
/// with SSE4.1 or better.
pub type CompressRow = (GenericArray<u8, U8>, u8);

/// The single 256-entry 8-lane left-pack table, shared by every lane count.
///
/// Row `m` (an 8-bit mask) holds the gather indices that *stably partition*
/// `[0, 8)` (selected lanes, those with a bit set in `m`, first and in order,
/// then the unselected lanes, also in order), plus the population count of `m`.
///
/// One table serves all lane counts. A register with `LANES <= 8` reads row `m`
/// (where `m` is its `LANES`-bit `movemask`) and takes the first `LANES`
/// indices through [`Register::permutev_row`]: the high padding lanes (`LANES..8`) are always unselected,
/// so they sort to the tail and are dropped, leaving exactly the `LANES`-lane
/// compaction. Wider registers index it per 8-lane chunk.
pub static COMPRESS8: GenericArray<CompressRow, U256> = build_table8();

/// Marker for lane counts (`1..=8`) small enough to take their compaction
/// indices straight out of a [`COMPRESS8`] row.
pub trait CompressTable: Lanes {}

impl CompressTable for U1 {}
impl CompressTable for U2 {}
impl CompressTable for U3 {}
impl CompressTable for U4 {}
impl CompressTable for U5 {}
impl CompressTable for U6 {}
impl CompressTable for U7 {}
impl CompressTable for U8 {}

/// Build the 256-entry 8-lane stable-partition table at compile time. Runs
/// entirely in const evaluation, so there is no runtime cost. Each row also carries its
/// population count.
const fn build_table8() -> GenericArray<CompressRow, U256> {
    let lanes = 8;
    let patterns = 256;

    // Zeroed is a valid initial state (u32 indices and the u8 count); every
    // entry is overwritten below.
    let mut table: GenericArray<CompressRow, U256> = unsafe { MaybeUninit::zeroed().assume_init() };
    let rows = table.as_mut_slice();

    let mut m = 0;
    while m < patterns {
        let row = rows[m].0.as_mut_slice();

        // Stable partition, both halves in order: selected indices first, then
        // the unselected ones. Two forward passes so the tail keeps the source
        // order (a non-zeroing compress; zeroing composes `zz` beforehand).
        let mut pos = 0;

        let mut i = 0;
        while i < lanes {
            if (m >> i) & 1 == 1 {
                row[pos] = i as u8;
                pos += 1;
            }
            i += 1;
        }

        // After the selected pass, `pos` is exactly the population count.
        let count = pos as u8;

        let mut i = 0;
        while i < lanes {
            if (m >> i) & 1 == 0 {
                row[pos] = i as u8;
                pos += 1;
            }
            i += 1;
        }

        rows[m].1 = count;

        m += 1;
    }

    table
}

/// Left-pack (`compress`, non-zeroing) via a single
/// [`permutev`](Register::permutev), with the permutation resolved from the
/// shared [`COMPRESS8`] table, with no runtime loop or branch.
///
/// Runtime cost is: read the lane bitmask
/// ([`native_bitmask`](MaskRegister::native_bitmask), one `movemask`),
/// `transmute_copy` the first `LANES` indices of row `bm`, and `permutev`. The
/// selected lanes are packed to the front in order. The unselected lanes keep
/// their values in the tail (also in order). The zero-filling variant composes a
/// [`zz`](CoreRegister::zz) beforehand (see
/// [`Register::compress_z`]).
///
/// On a backend with hardware permute this lowers to `movemask` + a table load +
/// `vpermps`/`pshufb`/`i8x16.swizzle`. For an emulated-wide
/// [`ArrayRegister`](crate::register::array::ArrayRegister) the index lookup is
/// still loop-free. Only the `permutev` itself does the cross-chunk routing.
///
/// Result for `value = [a, b, c, d]`, `mask = [T, F, T, F]` is `[a, c, b, d]`.
#[inline(always)]
pub fn compress_permute<R>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R>
where
    R: Register<Lanes: CompressTable>,
{
    // SAFETY: every lane count implementing `CompressTable` is <= 8.
    unsafe { compress_permute8_raw::<R>(value, mask) }
}

/// The bound-free body of [`compress_permute`], for callers that can prove
/// `LANES <= 8` but cannot name the [`CompressTable`] bound, specifically
/// blanket impls like `ArrayRegister`, which cannot add a per-method bound and
/// instead guard with `if const { Lanes::USIZE <= 8 && HAS_PERMUTEV }`.
///
/// # Safety
///
/// `R::Lanes` must be <= 8.
#[inline(always)]
pub unsafe fn compress_permute8_raw<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    // SAFETY: the caller guarantees `LANES <= 8`, and `native_bitmask` returns
    // `Some` for all registers with <= 64 lanes, so this is always `Some`.
    // Folds away on backends where it is a plain `movemask`, keeping the path
    // branchless. `native_bitmask` yields exactly `LANES` bits, so
    // `bm < 2^LANES <= 256` indexes the table directly.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() } as usize;

    // SAFETY: `bm < 256`, exactly the table length. Taking only the first
    // `LANES` indices is exactly the `LANES`-lane compaction: rows are
    // stable-partitioned and the padding lanes `LANES..8` are always unselected,
    // so they sort past the `LANES`-th slot and drop out.
    R::permutev_row(value, &unsafe { COMPRESS8.get_unchecked(bm) }.0)
}

/// Wide left-pack (`compress`, non-zeroing) for lane counts above 8, built by
/// applying the 8-lane [`CompressTable`] kernel once per 8-lane group and
/// merging the per-group results into a single global gather index that is
/// resolved with one full-width [`permutev`](Register::permutev).
///
/// Requires `LANES` to be a multiple of 8 and `<= 64` (so `native_bitmask` is
/// available), i.e. the `*x16`/`*x32`/`*x64` registers. The actual element
/// movement is a single permute. The only scalar work is the branchless
/// per-group index assembly, and the group count is a compile-time constant so
/// the outer loop unrolls. Versus the scalar default this replaces N
/// data-dependent element moves with one vector permute plus N branchless `u32`
/// index writes (8 indices resolved per table lookup).
///
/// The zero-filling variant composes a [`zz`](CoreRegister::zz) beforehand (see
/// [`Register::compress_z`]): zeroing the
/// unselected lanes first makes their values fall out as zeros in the tail.
///
/// Result for `value = [a, b, c, d, e, f, g, h, ...]` is the selected lanes in
/// order followed by the unselected lanes in order.
#[inline(always)]
pub fn compress_permute_wide<R>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R>
where
    R: Register,
{
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "compress_permute_wide requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE <= 64,
            "compress_permute_wide requires <= 64 lanes (native_bitmask bound)"
        );
    }

    // SAFETY: the const asserts above are exactly the contract.
    unsafe { compress_permute_wide_raw::<R>(value, mask) }
}

/// The assert-free body of [`compress_permute_wide`], for callers that can prove
/// the lane-count contract but cannot afford the `const { assert!(..) }` block,
/// meaning blanket impls like `ArrayRegister`, which select this path with an
/// `if const` guard. A `const` block inside an *untaken* `if const` arm is still
/// evaluated at monomorphization, so the asserts would fire for every
/// instantiation the guard exists to exclude. Same precedent as
/// [`compress_permute8_raw`]: the safety doc carries the contract, the safe
/// wrapper keeps the asserts.
///
/// # Safety
///
/// `R::Lanes` must be a nonzero multiple of 8 and `<= 64`.
#[inline(always)]
pub unsafe fn compress_permute_wide_raw<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let groups = n / 8;

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };

    // The fixed 8-lane kernel table, independent of the wide register type.
    let table8 = &COMPRESS8;

    // First pass: read each group's precomputed population count from the table
    // (no runtime `count_ones`) and total them. `groups <= 8` (64 / 8), so this
    // and the placement loop are compile-time-bounded and unroll.
    let mut counts = [0u8; 8];
    let mut total = 0usize;
    for group in 0..groups {
        let bmg = ((bm >> (group * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the 8-lane table's length minus one.
        let cnt = unsafe { table8.get_unchecked(bmg) }.1;
        counts[group] = cnt;
        total += cnt as usize;
    }

    // Second pass: assemble the global gather indices, selected lanes (stable
    // order) first, then the unselected lanes (stable order). Assembled
    // directly at the index register's element width (global indices are
    // < 64), so the register load below is the only conversion.
    let mut g: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> = GenericArray::default();
    let mut head = 0usize; // front cursor: where the next selected index goes
    let mut tail = total; // tail cursor: where the next unselected index goes

    for group in 0..groups {
        let base = group * 8;
        let bmg = ((bm >> base) & 0xFF) as usize;
        let cnt = counts[group] as usize;

        // SAFETY: `bmg <= 255`, exactly the 8-lane table's length minus one.
        let row = &unsafe { table8.get_unchecked(bmg) }.0;

        for j in 0..8 {
            let global = Element::from_u16((base + row[j] as usize) as u16);

            // `row[0..cnt]` are this group's selected local indices (-> front),
            // `row[cnt..8]` the unselected ones (-> tail). Branchless cursor pick.
            let pos = if j < cnt { head + j } else { tail + (j - cnt) };

            // SAFETY: across all groups `head` fills `[0, total)` and `tail`
            // fills `[total, n)`, each position exactly once, so `pos < n`.
            unsafe { *g.get_unchecked_mut(pos) = global };
        }

        head += cnt;
        tail += 8 - cnt;
    }

    R::permutev(value, R::Unsigned::new(g))
}

/// Chunked zeroing left-pack for any multiple-of-8 lane count up to 64, the
/// loop-free generalization that avoids the per-lane gather-index scatter.
///
/// For each 8-lane chunk it stores that chunk's selected-first table row (source
/// offset `+8c`) at a running *destination* cursor that advances by the chunk's
/// population count. The next chunk's 8-wide store overwrites the prior chunk's
/// unselected tail, so the selected indices end up compacted in `g[0..total]`
/// with `N/8` chunk-granular vector stores instead of `N` per-lane writes. The
/// unwritten tail keeps the sentinel index `LANES`, which the all-zero second
/// `swizzle` source resolves to zero (together with `zz` zeroing the unselected
/// lanes the written tail still points at), giving the zeroing partition with a
/// single permute.
#[inline(always)]
pub fn compress_z_wide<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "compress_z_wide requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE <= 64,
            "compress_z_wide requires <= 64 lanes (native_bitmask bound)"
        );
    }

    let n = <R::Lanes as Unsigned>::USIZE;
    let groups = n / 8;

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };
    let zeroed = R::zz(mask, value);

    // Gather indices, assembled directly at the index register's element
    // width. The sentinel `n` reads the all-zero second swizzle source.
    let mut g: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> =
        GenericArray::generate(|_| Element::from_u16(n as u16));

    let mut start = 0usize;
    for c in 0..groups {
        let base = (c * 8) as u16;
        let bmg = ((bm >> (c * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the table length minus one.
        let entry = unsafe { COMPRESS8.get_unchecked(bmg) };

        // The selected-first row, offset to source chunk `c`.
        let mut off = [<R::Unsigned as Register>::Element::default(); 8];
        let mut j = 0;
        while j < 8 {
            off[j] = Element::from_u16(entry.0[j] as u16 + base);
            j += 1;
        }

        // SAFETY: `start <= n - 8` for every chunk (the prefix sum of counts,
        // each <= 8, is at most `(groups - 1) * 8 = n - 8`), so the 8-wide store
        // stays in bounds.
        unsafe { core::ptr::copy_nonoverlapping(off.as_ptr(), g.as_mut_ptr().add(start), 8) };

        start += entry.1 as usize;
    }

    R::swizzle(zeroed, R::EMPTY, R::Unsigned::new(g))
}

/// Whether the merge tree ([`compress_z_merge2`]/`4`/`8`) supports this chunk
/// lane count / chunk count pair. The `ArrayRegister` routing guards on this so
/// an unsupported shape falls back to the scalar default instead.
///
/// The `chunk_lanes` list is inherited from the retired count-indexed control
/// tables. The pairwise merge ([`merge_pair`]) itself needs no table and only
/// wants `chunk_lanes * chunks <= 64` (the `native_bitmask` bound
/// [`compress_chunk_z`] relies on). Widening it is a separate change, since
/// every shape it would add is currently unreachable.
#[inline(always)]
pub const fn merge_ctrl_supported(chunk_lanes: usize, chunks: usize) -> bool {
    matches!(chunk_lanes, 4 | 8 | 16 | 32)
        && matches!(chunks, 2 | 4 | 8)
        && chunk_lanes * chunks <= 64
}

/// The largest `ext` scratch the pairwise merge needs: `3 * M` entries at the
/// tree's widest level, `M == 4` (see [`merge_pair`]). Entries past `3 * M` are
/// never read, so over-sizing the array costs nothing. The unused slots are
/// dead and eliminated.
const MERGE_EXT_MAX: usize = 12;

/// Lane iota `[0, 1, ..., LANES - 1]` at the index register's element width.
///
/// Every entry is a compile-time constant, so this folds to one `.rodata`
/// vector load. Hand-rolled `while` rather than `GenericArray::generate` for
/// the usual `target_feature` inlining reason.
///
/// Shared with [`super::expand`]'s two-tree composition (per the module header,
/// shared items live here), hence `pub(super)`.
#[inline(always)]
pub(super) fn lane_iota<B: Register + ?Sized>() -> Storage<B::Unsigned> {
    let mut v: GenericArray<<B::Unsigned as Register>::Element, B::Lanes> = GenericArray::default();

    let mut i = 0;
    while i < <B::Lanes as Unsigned>::USIZE {
        v[i] = Element::from_u16(i as u16);
        i += 1;
    }

    B::Unsigned::new(v)
}

/// An all-lanes-set / all-lanes-clear mask from a runtime `bit` (0 or 1), the
/// selector for one stage of the pairwise merge's chunk shift.
#[inline(always)]
fn splat_bit_mask<B: Register>(bit: usize) -> Storage<B::Mask> {
    let one = B::Unsigned::splat(Element::from_u16(1));
    let v = B::Unsigned::splat(Element::from_u16(bit as u16));

    <B::Mask as CastMaskRegister<<B::Unsigned as CoreRegister>::Mask>>::mask_from(B::Unsigned::eq(v, one))
}

/// Merge one aligned pair of `M`-chunk blocks (`TWO_M == 2 * M`) into a
/// `TWO_M`-chunk block, given the left block's selected-lane count.
///
/// # The identity
///
/// Both inputs satisfy the tree invariant `[selected..., 0...]`, `left` with
/// `c = count_left` live lanes and `right` with some `d`. The merged block is
///
/// ```text
/// merged[p] = left[p]          for p < c
///           = right[p - c]     for p >= c
/// ```
///
/// and because `left`'s lanes at and above `c` are *literally zero* (the level-0
/// groups come out of a [`zz`](CoreRegister::zz)ed input, and every merge below
/// preserves it), that is exactly `left | shift_up(right, c)`, one bitwise OR
/// with one operand all-zero bits in every lane. The merged tail past `c + d` is
/// zero from both sides, closing the induction.
///
/// # Why one swizzle per output chunk
///
/// Write `u = M*L - c` (`L` = chunk lanes) and split it as `u = a*L + off`.
/// Output chunk `k`, lane `t` wants `right[k*L + t - c] = right[(k - M + a)*L +
/// (off + t)]`, a *contiguous* window of the right block spanning at most two
/// adjacent right chunks, at the same local offset `off` for every `k`. So one
/// `B::swizzle(src_lo, src_hi, iota + off)` per output chunk resolves it, and
/// the index register is built once and shared by all `TWO_M` of them.
///
/// The split uses `a = (u - 1) / L` (and `a = 0` at `u == 0`), which puts `off`
/// in `1..=L` instead of `0..L` and shrinks `a`'s range from `M + 1` values to
/// `M`. That is what makes `M == 1` (every level-0 merge, and the whole of
/// [`compress_z_merge2`]) a *constant* `a == 0` with no chunk selection at all.
/// `off <= L` and `t < L` keep every index inside `swizzle`'s `0..2L` domain.
///
/// # Underflow and overflow, without a compare mask
///
/// Output chunk `k` reads `ext[k + a]` and `ext[k + a + 1]`, where `ext` holds
/// the right block at slots `M..2M` and [`EMPTY`](CoreRegister::EMPTY)
/// everywhere else. Lanes whose global position is below `c` index *before* the
/// right block (`k + a < M`) and therefore read `EMPTY`, contributing zero to
/// the OR, so `left`'s live prefix survives untouched. Lanes past the right
/// block (`k + a >= 2M`) read `EMPTY` too, which is the zero the tail needs.
/// No range compare and no masking are required: the padding *is* the mask.
///
/// # Chunk selection
///
/// `a` is data-dependent, so `e[i] = ext[i + a]` is resolved by `log2(M)` blend
/// stages over the constant-indexed scratch, branchless and a no-op at
/// `M == 1`. Blends whose two operands are both `EMPTY` fold away.
#[inline(always)]
fn merge_pair<B: Register, const M: usize, const TWO_M: usize>(
    left: [Storage<B>; M],
    right: [Storage<B>; M],
    count_left: usize,
) -> [Storage<B>; TWO_M] {
    let l = <B::Lanes as Unsigned>::USIZE;

    let u = M * l - count_left;
    let a = u.saturating_sub(1) / l;
    let off = u - a * l;

    // `ext[M..2M]` = the right block, everything else zero.
    let mut e = [B::EMPTY; MERGE_EXT_MAX];
    let mut i = 0;
    while i < M {
        e[M + i] = right[i];
        i += 1;
    }

    // Shift the scratch down by `a` chunks, one blend stage per bit of `a`.
    // `M` is a literal, so the stage count folds and the loops unroll.
    let mut step = 1;
    while step < M {
        let sel = splat_bit_mask::<B>((a / step) & 1);

        let mut i = 0;
        while i + step < MERGE_EXT_MAX {
            e[i] = B::blendv(sel, e[i], e[i + step]);
            i += 1;
        }

        step *= 2;
    }

    let idx = B::Unsigned::add(lane_iota::<B>(), B::Unsigned::splat(Element::from_u16(off as u16)));

    let mut out = [B::EMPTY; TWO_M];
    let mut k = 0;
    while k < TWO_M {
        let w = B::swizzle(e[k], e[k + 1], idx);
        // Only the left block's own chunks have anything to OR in.
        out[k] = if k < M { B::bitor(left[k], w) } else { w };
        k += 1;
    }

    out
}

/// Build a *single-register* merge-control table for two `H`-lane blocks
/// (`TwoH == 2*H`, `Entries == H+1`), with `u8` entries that are
/// **pair-relative**: index `k` addresses lane `k` of the `2H`-lane pair, so a
/// caller merging pair `p` at block half-size `H` adds the constant base
/// `p * 2H` bytewise.
///
/// Row `count_left` keeps the left block's first `count_left` lanes in place and
/// slides the right block (lanes `H..2H` of the *same* register) to begin at lane
/// `count_left`. Unlike [`build_merge_ctrl`], which merges two swizzle sources and
/// can point overflow at the all-zero second source (index `2H`), this control
/// feeds a one-source [`permutev`](Register::permutev): there is no second source,
/// so overflow must name a lane that is *already zero*. It names `H - 1`, the left
/// block's last lane.
///
/// # Sentinel proof
///
/// Output position `k` overflows (source `H + (k - count_left) >= 2H`) exactly when
/// `k >= H + count_left`. Such a `k` exists within `0..2H` only when
/// `count_left < H`. By induction every block entering a merge level is
/// `[selected..., 0...]` with exactly `count_left` selected lanes, so
/// `count_left < H` means the left block's lane `H - 1` is one of its zero pad
/// lanes. The sentinel is therefore zero whenever it is read, and rows with
/// `count_left == H` never read it. The merged `2H`-block is again
/// `[selected..., 0...]`, closing the induction (level 0 establishes the base case:
/// each 8-lane group is compacted after `zz`, so its unselected tail is zero).
const fn build_merge_ctrl_u8<TwoH: ArrayLength, Entries: ArrayLength>(
    h: usize,
) -> GenericArray<GenericArray<u8, TwoH>, Entries> {
    let two_h = <TwoH as Unsigned>::USIZE;
    let entries = <Entries as Unsigned>::USIZE;

    let mut t: GenericArray<GenericArray<u8, TwoH>, Entries> = unsafe { MaybeUninit::zeroed().assume_init() };
    let rows = t.as_mut_slice();

    let mut c = 0;
    while c < entries {
        let row = rows[c].as_mut_slice();
        let mut k = 0;
        while k < two_h {
            row[k] = if k < c {
                k as u8
            } else {
                let src = h + (k - c);
                // Overflow -> the left block's last lane, which is zero here.
                if src < two_h { src as u8 } else { (h - 1) as u8 }
            };
            k += 1;
        }
        c += 1;
    }

    t
}

// Single-register merge levels: 8+8->16, 16+16->32, 32+32->64. 144 + 544 + 2112
// bytes of `.rodata` total.
static MERGE_CTRL_U8_8: GenericArray<GenericArray<u8, U16>, generic_array::typenum::U9> = build_merge_ctrl_u8(8);
static MERGE_CTRL_U8_16: GenericArray<GenericArray<u8, U32>, generic_array::typenum::U17> = build_merge_ctrl_u8(16);
static MERGE_CTRL_U8_32: GenericArray<GenericArray<u8, U64>, generic_array::typenum::U33> = build_merge_ctrl_u8(32);

/// Pointer to the `2H`-byte single-register merge-control row for `count` selected
/// lanes in the left block. `H` is a literal at every call site, so the match
/// folds to one table base.
///
/// # Safety
///
/// `H` must be 8, 16 or 32, and `count <= H`.
#[inline(always)]
unsafe fn merge_ctrl_row_u8<const H: usize>(count: usize) -> *const u8 {
    unsafe {
        match H {
            8 => MERGE_CTRL_U8_8.get_unchecked(count).as_ptr(),
            16 => MERGE_CTRL_U8_16.get_unchecked(count).as_ptr(),
            _ => MERGE_CTRL_U8_32.get_unchecked(count).as_ptr(),
        }
    }
}

/// The per-lane *block base* vector for a level whose blocks are `UNIT` lanes
/// wide: lane `i` holds `i & !(UNIT - 1)`, the first lane of the block `i` lives
/// in (`UNIT` is always a power of two here).
///
/// Table rows are stored **block-relative**, so one `add` of this constant turns
/// them into global lane indices for every block at once. Every entry is a
/// compile-time constant, so the construction folds to a single `.rodata` vector
/// load, and to nothing at all on the last level, where `UNIT == LANES` makes it
/// the zero vector.
///
/// Shared with [`super::expand`]'s unmerge levels (per the module header, shared
/// items live here), hence `pub(super)`.
#[inline(always)]
pub(super) fn block_base<R: Register + ?Sized, const UNIT: usize>() -> Storage<R::Unsigned> {
    let mut b: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> = GenericArray::default();

    let mut i = 0;
    while i < <R::Lanes as Unsigned>::USIZE {
        b[i] = Element::from_u16((i & !(UNIT - 1)) as u16);
        i += 1;
    }

    R::Unsigned::new(b)
}

/// Copy `LEN` `u8` table entries out of `row` into `bytes[at..at + LEN]`.
///
/// Pure byte movement. LLVM lowers each row copy to one or a few plain
/// loads/stores (the rows are 8/16/32/64 bytes). The widening to the index
/// element width happens ONCE afterwards, via
/// [`Register::widen_index_bytes`], which backends override with a hardware
/// widening load. An earlier formulation widened per entry straight out of
/// the table and paid a `movzx` + insert chain per lane on 2-byte elements
/// (measured 117 vs ~40 instructions on `u16x16`).
///
/// Shared with [`super::expand`] (module header: shared items live here).
///
/// # Safety
///
/// `row` must be readable for `LEN` bytes and `at + LEN <= LANES`.
#[inline(always)]
pub(super) unsafe fn copy_row_into<R: Register + ?Sized, const LEN: usize>(
    bytes: &mut GenericArray<u8, R::Lanes>,
    at: usize,
    row: *const u8,
) {
    // SAFETY: per the contract above. The table row and the staging buffer
    // never overlap.
    unsafe { core::ptr::copy_nonoverlapping(row, bytes.as_mut_slice().as_mut_ptr().add(at), LEN) };
}

/// One single-register merge level at block half-size `H` (`TWO_H == 2 * H`):
/// build the index register that merges **every** block pair at this level
/// simultaneously, and fold the pair counts.
///
/// `counts[b]` is the number of selected lanes in block `b` on entry. On return
/// `counts[p]` holds the merged pair's count, ready for the next level.
#[inline(always)]
fn merge_indices<R: Register + ?Sized, const H: usize, const TWO_H: usize>(
    counts: &mut [u8; 8],
) -> Storage<R::Unsigned> {
    let pairs = <R::Lanes as Unsigned>::USIZE / TWO_H;

    let mut bytes: GenericArray<u8, R::Lanes> = GenericArray::default();

    let mut p = 0;
    while p < pairs {
        let left = counts[2 * p] as usize;
        // SAFETY: `H` is 8/16/32 at every instantiation and `left <= H`.
        let row = unsafe { merge_ctrl_row_u8::<H>(left) };
        // SAFETY: the row is `TWO_H` bytes and `pairs * TWO_H == LANES`.
        unsafe { copy_row_into::<R, TWO_H>(&mut bytes, p * TWO_H, row) };

        counts[p] = left as u8 + counts[2 * p + 1];
        p += 1;
    }

    R::Unsigned::add(R::widen_index_bytes(&bytes), block_base::<R, TWO_H>())
}

/// Zeroing left-pack for a native wide register: **grouped rows plus `log2` merge
/// passes**, entirely in one register and entirely branchless.
///
/// Replaces the per-lane gather-index scatter of [`compress_z_wide`] with
/// `1 + log2(LANES / 8)` [`permutev`](Register::permutev)s, each fed by an index
/// register assembled from `&'static` table rows plus one constant vector add:
///
/// 1. **Level 0**: [`zz`](CoreRegister::zz) the input, then compact every 8-lane
///    group *in place* with one permute. Group `g`'s indices are its
///    [`COMPRESS8`] row (entries `<= 7`, widened straight out of the table) plus
///    the block base `8g` supplied by [`block_base`]. Each group is now
///    `[selected..., 0...]`.
/// 2. **Merge levels**: `log2(LANES / 8)` passes at half-sizes 8, 16, 32. Each
///    pass reads the pair-relative `MERGE_CTRL_U8_*` rows for *all* pairs at that
///    level, adds the pair base, and resolves them with a single permute. See
///    [`build_merge_ctrl_u8`] for why a one-source control can sentinel overflow
///    at the left block's last lane.
///
/// Valid for `LANES % 8 == 0` and `16 <= LANES <= 64` (below 16 there is nothing
/// to merge, so use [`compress_permute`], and above 64 `native_bitmask` is
/// unavailable). This is the `compress_z` arm of `compress_via_wide!`. The
/// non-zeroing `compress` composes two of these trees, see [`compress_grouped`].
#[inline(always)]
pub fn compress_z_grouped<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "compress_z_grouped requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "compress_z_grouped requires 16..=64 lanes"
        );
    }

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };

    compress_z_grouped_bm::<R>(R::zz(mask, value), bm)
}

/// The body of [`compress_z_grouped`] with the `movemask` and the input zeroing
/// hoisted out, so a caller running the kernel **twice** (see
/// [`compress_grouped`]) pays for one
/// [`native_bitmask`](MaskRegister::native_bitmask) total.
///
/// `zeroed` must already be zero in every lane not selected by `bm` (the merge
/// tree's `[selected..., 0...]` invariant is what makes its one-source overflow
/// sentinel legal, see [`build_merge_ctrl_u8`]). `bm` supplies the per-group
/// table rows, and only its low `LANES` bits are read, so passing the bitwise
/// complement `!bm` runs the tree over the *unselected* lanes with no second
/// movemask and no mask-register complement. Each group reads row
/// `0xFF ^ bmg` directly.
#[inline(always)]
fn compress_z_grouped_bm<R: Register>(zeroed: Storage<R>, bm: u64) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let groups = n / 8;

    let mut cur = zeroed;

    let mut counts = [0u8; 8];

    // Level 0: every 8-lane group compacted in place by one permute.
    let mut bytes: GenericArray<u8, R::Lanes> = GenericArray::default();
    let mut g = 0;
    while g < groups {
        let bmg = ((bm >> (g * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the table length minus one.
        let entry = unsafe { COMPRESS8.get_unchecked(bmg) };
        // SAFETY: the row is 8 bytes and `8 * groups == n`.
        unsafe { copy_row_into::<R, 8>(&mut bytes, g * 8, entry.0.as_ptr()) };
        counts[g] = entry.1;
        g += 1;
    }

    cur = R::permutev(cur, R::Unsigned::add(R::widen_index_bytes(&bytes), block_base::<R, 8>()));

    // Merge levels, literal half-sizes so the control-table select folds.
    if const { <R::Lanes as Unsigned>::USIZE >= 16 } {
        cur = R::permutev(cur, merge_indices::<R, 8, 16>(&mut counts));
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 32 } {
        cur = R::permutev(cur, merge_indices::<R, 16, 32>(&mut counts));
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 64 } {
        cur = R::permutev(cur, merge_indices::<R, 32, 64>(&mut counts));
    }

    cur
}

/// Non-zeroing left-pack (`compress`) for a native wide register: **two
/// [`compress_z_grouped`] trees plus one shift and one OR**, entirely
/// branchless and with a single `movemask` for both trees.
///
/// A non-zeroing compress is the stable partition *selected lanes in order,
/// then unselected lanes in order*. Both halves are themselves zeroing
/// compressions (one under `mask`, one under its complement), so:
///
/// ```text
/// front = compress_z(v,  m)   // [sel_0 .. sel_{t-1}, 0 ...]
/// tail  = compress_z(v, !m)   // [uns_0 .. uns_{n-t-1}, 0 ...]
/// out   = front | shift_up(tail, t)
/// ```
///
/// where `t = popcount(m)`.
///
/// # The shift, and why the OR is exact
///
/// `shift_up(tail, t)` is [`merge_pair`]'s `M == 1` trick: one
/// [`swizzle`](Register::swizzle) whose **first** source is
/// [`EMPTY`](CoreRegister::EMPTY) and whose index register is the constant
/// [`lane_iota`] plus the broadcast `LANES - t`. Output lane `i` then reads
/// index `i + LANES - t`:
///
/// - `i < t`: index `< LANES`, inside the EMPTY window, so the lane is zero.
/// - `i >= t`: index `LANES + (i - t)`, i.e. `tail[i - t]`.
///
/// So each operand of the `bitor` is *all-zero bits* exactly where the other is
/// live: `front` is zero at and above lane `t` (that is what `compress_z`
/// guarantees), and the shifted `tail` is zero below lane `t` by the EMPTY
/// window. The OR is therefore a concatenation rather than a blend, with no
/// mask, no compare and no select. Indices stay in `swizzle`'s `0..2*LANES`
/// domain (`i + LANES - t <= 2*LANES - 1`), and the degenerate ends behave:
/// `t == 0` leaves `front` empty and reads all of `tail`, and `t == LANES`
/// makes the window entirely EMPTY and returns `front` unchanged.
///
/// The complement tree costs no extra `movemask`: only the low `LANES` bits of
/// the bitmask are ever read, per 8-lane group, so `!bm` is exactly the
/// per-group complement `0xFF ^ bmg`. The complement *zeroing* is
/// [`nz`](CoreRegister::nz), the purpose-built sibling of the `zz` the plain
/// tree uses.
///
/// Valid for the same shapes as [`compress_z_grouped`] (`LANES % 8 == 0`,
/// `16 <= LANES <= 64`). This is the `compress` arm of `compress_via_wide!`.
/// [`compress_permute_wide`] stays for the `ArrayRegister` routing, which has no
/// single-register permute to build on.
#[inline(always)]
pub fn compress_grouped<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "compress_grouped requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "compress_grouped requires 16..=64 lanes"
        );
    }

    let n = <R::Lanes as Unsigned>::USIZE;

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within
    // Thermite. Read ONCE and shared by both trees.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };
    let total = bm.count_ones() as usize;

    let front = compress_z_grouped_bm::<R>(R::zz(mask, value), bm);
    let tail = compress_z_grouped_bm::<R>(R::nz(mask, value), !bm);

    let idx = R::Unsigned::add(
        lane_iota::<R>(),
        R::Unsigned::splat(Element::from_u16((n - total) as u16)),
    );

    R::bitor(front, R::swizzle(R::EMPTY, tail, idx))
}

/// Compact one chunk (zeroing) via the chunk register's own `compress_z` and
/// return its population count: the chunk becomes `[selected..., 0...]` and
/// `count` is how many lanes are selected.
///
/// For an 8-lane-or-narrower native chunk `compress_z` is the table path
/// (movemask + row + one permute). Wider chunks bring whatever their register
/// implements, so the merge tree's quality composes recursively.
#[inline(always)]
fn compress_chunk_z<B: Register>(chunk: Storage<B>, mask: Storage<B::Mask>) -> (Storage<B>, usize) {
    let packed = B::compress_z(chunk, mask);
    // `native_bitmask` is `Some` for every <= 64-lane mask within Thermite.
    let bm = unsafe { <B::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };
    (packed, bm.count_ones() as usize)
}

/// Two-level zeroing left-pack for a 2-chunk register: compact each chunk
/// *natively* (its own `compress_z`, fully vectorized for table-path chunks)
/// and merge the two with [`merge_pair`], with no scalar gather-index scatter
/// and no cross-chunk `array_swizzle`.
///
/// Much faster than [`compress_permute_wide`] at 16 lanes (8-lane chunks): the
/// scalar scatter it replaces is a long data-dependent dependency chain, which
/// costs far more than the extra shuffles. Result is the zeroing
/// (`compress_z`) partition: selected lanes to the front, rest zeroed.
///
/// Callers must guard with [`merge_ctrl_supported`]`(B::LANES, 2)`.
#[inline(always)]
pub fn compress_z_merge2<B: Register>(chunks: [Storage<B>; 2], masks: [Storage<B::Mask>; 2]) -> [Storage<B>; 2] {
    let (lo, count_lo) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (hi, _) = compress_chunk_z::<B>(chunks[1], masks[1]);

    // Slide `hi` to begin at lane `count_lo` and OR it over `lo`'s zero tail.
    // `M == 1`, so the chunk shift is a compile-time constant: two swizzles
    // sharing one index register, plus one OR.
    merge_pair::<B, 1, 2>([lo], [hi], count_lo)
}

/// Four-chunk zeroing left-pack: a two-level merge tree. Compact each chunk
/// natively, merge `(0,1)` and `(2,3)`, then merge those two blocks. No
/// scalar gather-index scatter.
///
/// Callers must guard with [`merge_ctrl_supported`]`(B::LANES, 4)`.
#[inline(always)]
pub fn compress_z_merge4<B: Register>(chunks: [Storage<B>; 4], masks: [Storage<B::Mask>; 4]) -> [Storage<B>; 4] {
    let (c0, n0) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (c1, n1) = compress_chunk_z::<B>(chunks[1], masks[1]);
    let (c2, n2) = compress_chunk_z::<B>(chunks[2], masks[2]);
    let (c3, _) = compress_chunk_z::<B>(chunks[3], masks[3]);

    // Level 0: l+l -> 2l, a constant chunk shift each.
    let m01 = merge_pair::<B, 1, 2>([c0], [c1], n0);
    let m23 = merge_pair::<B, 1, 2>([c2], [c3], n2);

    // Level 1: 2l+2l -> 4l, placing m23 after m01's `n0 + n1` selected lanes.
    merge_pair::<B, 2, 4>(m01, m23, n0 + n1)
}

/// Eight-chunk zeroing left-pack: a three-level merge tree.
///
/// Callers must guard with [`merge_ctrl_supported`]`(B::LANES, 8)`.
#[inline(always)]
pub fn compress_z_merge8<B: Register>(chunks: [Storage<B>; 8], masks: [Storage<B::Mask>; 8]) -> [Storage<B>; 8] {
    let (c0, n0) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (c1, n1) = compress_chunk_z::<B>(chunks[1], masks[1]);
    let (c2, n2) = compress_chunk_z::<B>(chunks[2], masks[2]);
    let (c3, n3) = compress_chunk_z::<B>(chunks[3], masks[3]);
    let (c4, n4) = compress_chunk_z::<B>(chunks[4], masks[4]);
    let (c5, n5) = compress_chunk_z::<B>(chunks[5], masks[5]);
    let (c6, n6) = compress_chunk_z::<B>(chunks[6], masks[6]);
    let (c7, _) = compress_chunk_z::<B>(chunks[7], masks[7]);

    // Level 0: four l+l -> 2l merges (constant chunk shift).
    let m01 = merge_pair::<B, 1, 2>([c0], [c1], n0);
    let m23 = merge_pair::<B, 1, 2>([c2], [c3], n2);
    let m45 = merge_pair::<B, 1, 2>([c4], [c5], n4);
    let m67 = merge_pair::<B, 1, 2>([c6], [c7], n6);

    // Level 1: two 2l+2l -> 4l merges.
    let m0123 = merge_pair::<B, 2, 4>(m01, m23, n0 + n1);
    let m4567 = merge_pair::<B, 2, 4>(m45, m67, n4 + n5);

    // Level 2: 4l+4l -> 8l, placing m4567 after m0123's selected lanes.
    merge_pair::<B, 4, 8>(m0123, m4567, n0 + n1 + n2 + n3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::register::array::ArrayRegister;

    // Exhaustively check `compress_permute` against a scalar oracle over every
    // mask pattern. Uses `ArrayRegister<i32, N>` (a `Register` on every
    // host); `HAS_PERMUTEV` is false here so the data movement goes through
    // `scalar_permutev`, exercising the table lookup + tail-zero fusion.
    fn check<const N: usize>()
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32, Lanes: CompressTable>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 16];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());

        for bits in 0u32..(1 << N) {
            let mut sel = [0i32; 16];
            for lane in 0..N {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let mask = <R<N>>::into_mask(<R<N>>::new(GenericArray::from_slice(&sel[..N]).clone()));

            // Oracle: stable partition - selected lanes in order, then the
            // unselected lanes in order (non-zeroing).
            let mut expected = [0i32; 16];
            let mut pos = 0;
            for lane in 0..N {
                if (bits >> lane) & 1 == 1 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }
            for lane in 0..N {
                if (bits >> lane) & 1 == 0 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }

            let got = compress_permute::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn compress_permute_exhaustive() {
        check::<2>();
        check::<4>();
        check::<8>();
    }

    // Check `compress_permute_wide` against the same stable-partition oracle for
    // a set of mask patterns. `N` must be a multiple of 8.
    fn check_wide<const N: usize>(patterns: impl Iterator<Item = u64>)
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());

        for bits in patterns {
            let mut sel = [0i32; 64];
            for lane in 0..N {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let mask = <R<N>>::into_mask(<R<N>>::new(GenericArray::from_slice(&sel[..N]).clone()));

            // Stable-partition oracle: selected lanes in order, then unselected.
            let mut expected = [0i32; 64];
            let mut pos = 0;
            for lane in 0..N {
                if (bits >> lane) & 1 == 1 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }
            for lane in 0..N {
                if (bits >> lane) & 1 == 0 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }

            let got = compress_permute_wide::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn compress_permute_wide_correct() {
        // N=8 (single group) and N=16 exhaustively.
        check_wide::<8>(0..(1u64 << 8));
        check_wide::<16>(0..(1u64 << 16));

        // N=32 / N=64: exhaustive is infeasible, so sweep structured patterns
        // (none/all/alternating/halves/boundaries) plus a deterministic xorshift
        // sample - enough to exercise cross-group offset merging.
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0x00FF_00FF_00FF_00FF,
            0x0101_0101_0101_0101,
            0x8080_8080_8080_8080,
        ];

        check_wide::<32>(structured.into_iter().chain((0..2000).map(move |_| rng())));

        let mut s = 0xDEADBEEFCAFEBABEu64;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        check_wide::<64>(structured.into_iter().chain((0..2000).map(move |_| rng())));
    }

    // Check the chunked `compress_z_wide` against a zeroing oracle, on
    // `ArrayRegister<i32, N>` (N a multiple of 8).
    fn check_z_wide<const N: usize>(patterns: impl Iterator<Item = u64>)
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());

        for bits in patterns {
            let mut sel = [0i32; 64];
            for lane in 0..N {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let mask = <R<N>>::into_mask(<R<N>>::new(GenericArray::from_slice(&sel[..N]).clone()));
            let expected = zeroing_oracle::<N>(&data, bits);

            let got = compress_z_wide::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn compress_z_wide_correct() {
        check_z_wide::<8>(0..256);
        check_z_wide::<16>(0..(1 << 16));

        let mut s = 0x243F_6A88_85A3_08D3u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
        ];
        check_z_wide::<32>(structured.into_iter().chain((0..5000).map(|_| rng() & 0xFFFF_FFFF)));
        check_z_wide::<64>(structured.into_iter().chain((0..5000).map(|_| rng())));
    }

    // Check `compress_z_grouped` (grouped rows + log2 merges) against the
    // zeroing oracle on `ArrayRegister<i32, N>`, whose `permutev` is the scalar
    // default, so this exercises the index assembly and the merge sentinel,
    // not any backend permute.
    fn check_z_grouped<const N: usize>(patterns: impl Iterator<Item = u64>)
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());

        for bits in patterns {
            let mut sel = [0i32; 64];
            for lane in 0..N {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let mask = <R<N>>::into_mask(<R<N>>::new(GenericArray::from_slice(&sel[..N]).clone()));
            let expected = zeroing_oracle::<N>(&data, bits);

            let got = compress_z_grouped::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn compress_z_grouped_correct() {
        // 16 lanes exhaustively (one merge level), 32/64 sampled.
        check_z_grouped::<16>(0..(1 << 16));

        let mut s = 0x853C_49E6_748F_EA9Bu64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0x00FF_00FF_00FF_00FF,
        ];

        check_z_grouped::<32>(structured.into_iter().chain((0..5000).map(|_| rng() & 0xFFFF_FFFF)));
        check_z_grouped::<64>(structured.into_iter().chain((0..5000).map(|_| rng())));
    }

    // Check the non-zeroing two-tree composition against the stable-partition
    // oracle on `ArrayRegister<i32, N>` (scalar `permutev`/`swizzle`), so this
    // exercises the shift window and the OR-exactness argument rather than any
    // backend permute.
    fn check_grouped<const N: usize>(patterns: impl Iterator<Item = u64>)
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());

        for bits in patterns {
            let mut sel = [0i32; 64];
            for lane in 0..N {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let mask = <R<N>>::into_mask(<R<N>>::new(GenericArray::from_slice(&sel[..N]).clone()));

            // Stable partition: selected in order, then unselected in order.
            let mut expected = [0i32; 64];
            let mut pos = 0;
            for lane in 0..N {
                if (bits >> lane) & 1 == 1 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }
            for lane in 0..N {
                if (bits >> lane) & 1 == 0 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }

            let got = compress_grouped::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn compress_grouped_correct() {
        check_grouped::<16>(0..(1 << 16));

        let mut s = 0x6A09_E667_F3BC_C908u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0x00FF_00FF_00FF_00FF,
        ];

        check_grouped::<32>(structured.into_iter().chain((0..5000).map(|_| rng() & 0xFFFF_FFFF)));
        check_grouped::<64>(structured.into_iter().chain((0..5000).map(|_| rng())));
    }

    // Exhaustively check the two-level merge variant (16-lane zeroing compress)
    // against a zeroing oracle, using two `ArrayRegister<i32, 8>` chunks.
    #[test]
    fn compress_z_merge2_correct() {
        type B = ArrayRegister<i32, 8>;

        let mut data = [0i32; 16];
        for i in 0..16 {
            data[i] = ((i + 1) * 10) as i32;
        }
        let v0 = <B>::new(GenericArray::from_slice(&data[0..8]).clone());
        let v1 = <B>::new(GenericArray::from_slice(&data[8..16]).clone());

        for bits in 0u32..(1 << 16) {
            let mut sel = [0i32; 16];
            for lane in 0..16 {
                sel[lane] = ((bits >> lane) & 1) as i32;
            }
            let m0 = <B>::into_mask(<B>::new(GenericArray::from_slice(&sel[0..8]).clone()));
            let m1 = <B>::into_mask(<B>::new(GenericArray::from_slice(&sel[8..16]).clone()));

            // Zeroing oracle: selected lanes in order, then zeros.
            let mut expected = [0i32; 16];
            let mut pos = 0;
            for lane in 0..16 {
                if (bits >> lane) & 1 == 1 {
                    expected[pos] = data[lane];
                    pos += 1;
                }
            }

            let got = compress_z_merge2::<B>([v0, v1], [m0, m1]);
            let g0 = <B>::as_slice(&got[0]);
            let g1 = <B>::as_slice(&got[1]);
            for lane in 0..16 {
                let val = if lane < 8 { g0[lane] } else { g1[lane - 8] };
                assert_eq!(val, expected[lane], "bits={bits:b} lane={lane}");
            }
        }
    }

    // Zeroing oracle for a `LANES`-lane compress: selected lanes in order, rest 0.
    fn zeroing_oracle<const LANES: usize>(data: &[i32], bits: u64) -> [i32; 64] {
        let mut expected = [0i32; 64];
        let mut pos = 0;
        for lane in 0..LANES {
            if (bits >> lane) & 1 == 1 {
                expected[pos] = data[lane];
                pos += 1;
            }
        }
        expected
    }

    #[test]
    fn compress_z_merge4_correct() {
        type B = ArrayRegister<i32, 8>;

        let mut data = [0i32; 32];
        for i in 0..32 {
            data[i] = ((i + 1) * 10) as i32;
        }
        let v: [Storage<B>; 4] =
            core::array::from_fn(|c| <B>::new(GenericArray::from_slice(&data[c * 8..c * 8 + 8]).clone()));

        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [
            0u64,
            u32::MAX as u64,
            0x5555_5555,
            0xAAAA_AAAA,
            0x0000_FFFF,
            0xFFFF_0000,
            0x00FF_FF00,
        ];

        for bits in structured.into_iter().chain((0..5000).map(|_| rng() & 0xFFFF_FFFF)) {
            let m: [Storage<<B as CoreRegister>::Mask>; 4] = core::array::from_fn(|c| {
                let sel: [i32; 8] = core::array::from_fn(|l| ((bits >> (c * 8 + l)) & 1) as i32);
                <B>::into_mask(<B>::new(GenericArray::from_slice(&sel).clone()))
            });
            let expected = zeroing_oracle::<32>(&data, bits);

            let got = compress_z_merge4::<B>(v, m);
            for lane in 0..32 {
                assert_eq!(
                    <B>::as_slice(&got[lane / 8])[lane % 8],
                    expected[lane],
                    "bits={bits:b} lane={lane}"
                );
            }
        }
    }

    #[test]
    fn compress_z_merge8_correct() {
        type B = ArrayRegister<i32, 8>;

        let mut data = [0i32; 64];
        for i in 0..64 {
            data[i] = ((i + 1) * 10) as i32;
        }
        let v: [Storage<B>; 8] =
            core::array::from_fn(|c| <B>::new(GenericArray::from_slice(&data[c * 8..c * 8 + 8]).clone()));

        let mut s = 0x0F1E_2D3C_4B5A_6978u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // The last two rows pin the top merge's chunk-shift boundaries: its
        // `count_left` is the popcount of the low 32 bits, and the entries
        // below make that exactly 0/8/16/24/32, every value of the shift `a`
        // including both ends, with a nonempty right block to place.
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0x0F0F_0F0F_0000_0000,
            0x0F0F_0F0F_0000_00FF,
            0x0F0F_0F0F_0000_FFFF,
            0x0F0F_0F0F_00FF_FFFF,
            0x0F0F_0F0F_FFFF_FFFF,
        ];

        for bits in structured.into_iter().chain((0..5000).map(|_| rng())) {
            let m: [Storage<<B as CoreRegister>::Mask>; 8] = core::array::from_fn(|c| {
                let sel: [i32; 8] = core::array::from_fn(|l| ((bits >> (c * 8 + l)) & 1) as i32);
                <B>::into_mask(<B>::new(GenericArray::from_slice(&sel).clone()))
            });
            let expected = zeroing_oracle::<64>(&data, bits);

            let got = compress_z_merge8::<B>(v, m);
            for lane in 0..64 {
                assert_eq!(
                    <B>::as_slice(&got[lane / 8])[lane % 8],
                    expected[lane],
                    "bits={bits:b} lane={lane}"
                );
            }
        }
    }
}

/// The portable scalar stable-partition left-pack: the default body behind
/// [`Register::compress`]. Free-standing here so blanket impls (e.g.
/// `ArrayRegister`) can fall back to it from their own `compress` overrides,
/// which the trait system does not allow via `Self::compress` recursion or
/// per-method bounds.
pub fn compress_default<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;

    let src = R::as_slice(&value);
    let mut result = value;
    let dst = R::as_mut_slice(&mut result);

    let mut pos = 0;

    // Selected lanes first, in order.
    for i in 0..n {
        if <R::Mask as MaskRegister>::test(mask, i) {
            dst[pos] = src[i];
            pos += 1;
        }
    }

    // Unselected lanes after, in order.
    for i in 0..n {
        if !<R::Mask as MaskRegister>::test(mask, i) {
            dst[pos] = src[i];
            pos += 1;
        }
    }

    result
}

/// The single-pass zero-filling left-pack: the default body behind
/// [`Register::compress_z`]; see [`compress_default`] for why it is
/// free-standing.
pub fn compress_z_default<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let src = R::as_slice(&value);

    // `EMPTY` is zero, so the tail is already filled - only place selected.
    let mut result = R::EMPTY;
    let dst = R::as_mut_slice(&mut result);

    let mut pos = 0;
    for i in 0..n {
        if <R::Mask as MaskRegister>::test(mask, i) {
            dst[pos] = src[i];
            pos += 1;
        }
    }

    result
}

/// The portable scalar merge-masked left-pack: the fallback body behind
/// [`Register::compress_m`] for registers where the vectorized
/// prefix-mask-blend composition is unavailable (`LANES > 64`, past the
/// [`from_native_bitmask`](MaskRegister::from_native_bitmask) bound).
/// Free-standing for the same reason as [`compress_default`].
///
/// Result lane `i` is the `i`-th selected element for `i < popcount(mask)`, and
/// `src[i]` otherwise, matching AVX-512 merge-masked `vpcompress*`.
pub fn compress_m_default<R: Register>(src: Storage<R>, mask: Storage<R::Mask>, value: Storage<R>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let val = R::as_slice(&value);

    let mut result = src;
    let dst = R::as_mut_slice(&mut result);

    let mut pos = 0;
    for i in 0..n {
        if <R::Mask as MaskRegister>::test(mask, i) {
            dst[pos] = val[i];
            pos += 1;
        }
    }

    result
}
