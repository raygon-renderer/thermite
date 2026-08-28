//! Generic `expand` (right-scatter / inverse left-pack) polyfills.
//!
//! [`Register::expand`] is defined as the **exact inverse permutation** of
//! [`Register::compress`]: `compress` stably partitions the register (selected
//! lanes to the front in order, unselected lanes to the tail in order), and a
//! permutation has an exact inverse, so for every `value` and `mask`:
//!
//! ```text
//! expand(compress(v, m), m) == v
//! compress(expand(v, m), m) == v
//! ```
//!
//! Concretely, output lane `i` reads `value[rank of i among selected]` when
//! `mask[i]` is set, and `value[popcount + rank of i among unselected]` when it
//! is not. AVX-512 `vpexpand*` only defines the selected lanes (merge or zero
//! elsewhere); making the plain form a full permutation costs nothing - it is
//! the same single permute - and the merge/zero siblings fall out by composing
//! a blend or [`zz`](CoreRegister::zz) *after* (where `compress_z` composes its
//! `zz` *before*).
//!
//! This module holds expand-only code. Everything the two directions share -
//! the [`CompressRow`] row type, the [`CompressTable`] lane-count marker, and
//! [`COMPRESS8`] - lives in [`super::compress`]; see that header.
//!
//! The implementation mirrors `compress` ([`EXPAND8`] is [`COMPRESS8`] with each
//! row inverted, carrying the same population counts):
//!
//! - [`expand_permute`] - up to 8 lanes: `movemask` -> table row -> one permute.
//! - [`expand_z_grouped`] is the *zeroing* inverse left-pack for a native
//!   16/32/64-lane register: `log2(LANES / 8)` permutes *unmerge* the packed run
//!   back into per-8-lane-group runs, then one permute scatters each group and a
//!   `zz` zeroes the rest. Branchless, and the mirror of
//!   [`compress_z_grouped`](super::compress::compress_z_grouped), run backwards.
//!   This is the `expand_z` arm of `compress_via_wide!`.
//! - [`expand_grouped`] is the *non-zeroing* form at the same shapes, and the
//!   `expand` arm of `compress_via_wide!`. Two [`expand_z_grouped`] trees (one
//!   under the mask, one under its complement fed the input shifted down past
//!   the packed front by a single `swizzle`), combined with one `bitor`. Both
//!   trees share one `movemask`, and the complement tree indexes the same tables
//!   with `!bm`.
//! - [`expand_permute_wide`] is the remaining fallback, for any multiple of 8
//!   lanes up to 64: per-8-lane table lookups assembled by a scalar per-lane
//!   scatter into one global gather index + one full-width permute. No native
//!   register routes here any more (they take [`expand_grouped`]). It survives
//!   as the `ArrayRegister` non-zeroing `expand`, which has no single-register
//!   permute to build a grouped kernel on.
//!
//! The same plan/apply split as the compress side (see that module's header)
//! backs the `_n` family here: [`expand_permute_n`], [`expand_grouped_n`],
//! [`expand_z_grouped_n`] and [`expand_permute_wide_raw_n`] build the
//! mask-derived plan once and apply it to `N` values.
//!
//! There is no analogue of the `compress_z_merge*` chunk trees (the cross-chunk
//! `array_swizzle` forms). Expand *splits* a packed run across chunks rather
//! than merging per-chunk results, so that construction is not symmetric. The
//! single-register unmerge above is, because a permute inverts a permute.

use generic_array::{
    ArrayLength, GenericArray,
    typenum::{U16, U32, U64, U256, Unsigned},
};

use super::*;

// Named explicitly, though the glob above would supply them, to keep the
// dependency on the compress side visible at the top of the file.
use super::compress::{CompressRow, CompressTable, GROUPED_LEVELS, GroupedPlan, block_base, copy_row_into};

/// The single 256-entry 8-lane expand table: row `m` is the inverse of
/// [`COMPRESS8`]'s row `m` (same [`CompressRow`] shape, same population count).
///
/// Where the compress row answers "which source lane does packed position `p`
/// read?", the expand row answers "which packed position does lane `i` read?":
/// `EXPAND8[m].0[i]` is `rank of i among selected` if bit `i` of `m` is set,
/// else `popcount(m) + rank of i among unselected`. Equivalently
/// `EXPAND8[m].0[COMPRESS8[m].0[j]] == j` for all `j`.
///
/// The same one-table-serves-all-widths trick as [`COMPRESS8`] applies: for
/// `LANES < 8` the padding lanes `LANES..8` are unselected with the highest
/// unselected ranks, so their entries land in `popcount + ..` positions past
/// every real lane's, and truncating the row to `LANES` indices yields the
/// `LANES`-lane inverse permutation. Indices stay `< LANES` because ranks of
/// real lanes never count padding lanes, which sort after them.
pub static EXPAND8: GenericArray<CompressRow, U256> = build_expand_table8();

/// Build the expand table at compile time. `build_table8` writes
/// `row[pos] = i` (position `pos` reads lane `i`); this writes `row[i] = pos`
/// (lane `i` reads position `pos`) - the same double loop with the assignment
/// transposed, hence exact mutual inverses by construction.
const fn build_expand_table8() -> GenericArray<CompressRow, U256> {
    let lanes = 8;
    let patterns = 256;

    // Zeroed is a valid initial state (u8 indices and the u8 count); every
    // entry is overwritten below.
    let mut table: GenericArray<CompressRow, U256> = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
    let rows = table.as_mut_slice();

    let mut m = 0;
    while m < patterns {
        let row = rows[m].0.as_mut_slice();

        let mut pos = 0;

        let mut i = 0;
        while i < lanes {
            if (m >> i) & 1 == 1 {
                row[i] = pos as u8;
                pos += 1;
            }
            i += 1;
        }

        // After the selected pass, `pos` is exactly the population count.
        let count = pos as u8;

        let mut i = 0;
        while i < lanes {
            if (m >> i) & 1 == 0 {
                row[i] = pos as u8;
                pos += 1;
            }
            i += 1;
        }

        rows[m].1 = count;

        m += 1;
    }

    table
}

/// Inverse left-pack (`expand`, non-zeroing) via a single
/// [`permutev`](Register::permutev), with the permutation resolved from the
/// [`EXPAND8`] table - no runtime loop or branch. The exact mirror of
/// [`compress_permute`], and its exact inverse.
///
/// For `value = [a, c, b, d]` and `mask = [T, F, T, F]` the result is
/// `[a, b, c, d]`: lanes 0 and 2 read the packed front (`a`, `c`), lanes 1 and 3
/// read the tail (`b`, `d`).
#[inline(always)]
pub fn expand_permute<R>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R>
where
    R: Register<Lanes: CompressTable>,
{
    // SAFETY: every lane count implementing `CompressTable` is <= 8.
    unsafe { expand_permute8_raw::<R>(value, mask) }
}

/// The bound-free body of [`expand_permute`], for callers that can prove
/// `LANES <= 8` but cannot name the [`CompressTable`] bound (blanket impls like
/// `ArrayRegister`, which guard with `if const { Lanes::USIZE <= 8 && HAS_PERMUTEV }`).
///
/// # Safety
///
/// `R::Lanes` must be <= 8.
#[inline(always)]
pub unsafe fn expand_permute8_raw<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    // SAFETY: identical argument to `compress_permute8_raw`: <= 8 lanes makes
    // `native_bitmask` always `Some` and `bm < 256` indexes the table directly.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() } as usize;

    // SAFETY: `bm < 256`; truncating to the first `LANES` indices is sound per
    // the padding-lane argument in the `EXPAND8` docs (real lanes' indices never
    // reference padding positions).
    R::permutev_row(value, &unsafe { EXPAND8.get_unchecked(bm) }.0)
}

/// Same-mask multi-vector form of [`expand_permute`]: one table row fetch
/// shared by `N` [`permutev_row`](Register::permutev_row)s, the mirror of
/// [`compress_permute_n`](super::compress::compress_permute_n).
#[inline(always)]
pub fn expand_permute_n<R, const N: usize>(values: [Storage<R>; N], mask: Storage<R::Mask>) -> [Storage<R>; N]
where
    R: Register<Lanes: CompressTable>,
{
    // SAFETY: every lane count implementing `CompressTable` is <= 8.
    unsafe { expand_permute8_raw_n::<R, N>(values, mask) }
}

/// The bound-free body of [`expand_permute_n`].
///
/// # Safety
///
/// `R::Lanes` must be <= 8.
#[inline(always)]
pub unsafe fn expand_permute8_raw_n<R: Register, const N: usize>(
    values: [Storage<R>; N],
    mask: Storage<R::Mask>,
) -> [Storage<R>; N] {
    // SAFETY: as in `expand_permute8_raw`.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() } as usize;

    // SAFETY: `bm < 256`, exactly the table length.
    let row = &unsafe { EXPAND8.get_unchecked(bm) }.0;

    let mut out = [R::EMPTY; N];

    let mut i = 0;
    while i < N {
        out[i] = R::permutev_row(values[i], row);
        i += 1;
    }

    out
}

/// Wide inverse left-pack (`expand`, non-zeroing) for lane counts above 8: the
/// exact mirror of [`compress_permute_wide`], assembling one global gather
/// index from per-8-lane-group [`EXPAND8`] rows and resolving it with a single
/// full-width [`permutev`](Register::permutev).
///
/// The index math inverts the wide compress layout. Compress places group `g`'s
/// selected lanes at packed positions `[base_g, base_g + cnt_g)` (where `base_g`
/// is the sum of earlier groups' counts) and its unselected lanes at
/// `[total + ubase_g, ..)` (where `ubase_g = g*8 - base_g` counts earlier
/// groups' unselected lanes). So expand reads, for output lane `i` in group `g`
/// with local table entry `r = EXPAND8[bm_g].0[i%8]`:
///
/// - selected (`r < cnt_g`): source `base_g + r`
/// - unselected (`r >= cnt_g`): source `total + ubase_g + (r - cnt_g)`
///
/// Requires `LANES` to be a multiple of 8 and `<= 64`, like the compress
/// counterpart; group count is a compile-time constant so the loops unroll.
#[inline(always)]
pub fn expand_permute_wide<R>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R>
where
    R: Register,
{
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "expand_permute_wide requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE <= 64,
            "expand_permute_wide requires <= 64 lanes (native_bitmask bound)"
        );
    }

    // SAFETY: the const asserts above are exactly the contract.
    unsafe { expand_permute_wide_raw::<R>(value, mask) }
}

/// The assert-free body of [`expand_permute_wide`], the mirror of
/// [`compress_permute_wide_raw`](super::compress::compress_permute_wide_raw).
/// Blanket impls like `ArrayRegister` route here from an `if const` guard: a
/// `const` block inside an untaken `if const` arm still evaluates at
/// monomorphization, so the safe wrapper's asserts would fire for exactly the
/// instantiations the guard excludes.
///
/// # Safety
///
/// `R::Lanes` must be a nonzero multiple of 8 and `<= 64`.
#[inline(always)]
pub unsafe fn expand_permute_wide_raw<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    // SAFETY: the caller carries the lane-count contract.
    let idx = unsafe { expand_permute_wide_indices::<R>(mask) };

    R::permutev(value, idx)
}

/// Same-mask multi-vector form of [`expand_permute_wide_raw`]: the per-lane
/// scatter is entirely mask-derived, so it is assembled once and resolved with
/// one [`permutev`](Register::permutev) per value.
///
/// # Safety
///
/// `R::Lanes` must be a nonzero multiple of 8 and `<= 64`.
#[inline(always)]
pub unsafe fn expand_permute_wide_raw_n<R: Register, const N: usize>(
    values: [Storage<R>; N],
    mask: Storage<R::Mask>,
) -> [Storage<R>; N] {
    // SAFETY: the caller carries the lane-count contract.
    let idx = unsafe { expand_permute_wide_indices::<R>(mask) };

    let mut out = [R::EMPTY; N];

    let mut i = 0;
    while i < N {
        out[i] = R::permutev(values[i], idx);
        i += 1;
    }

    out
}

/// The mask-only half of [`expand_permute_wide_raw`]: the global gather index
/// register. Shared by the single- and multi-vector entry points.
///
/// # Safety
///
/// `R::Lanes` must be a nonzero multiple of 8 and `<= 64`.
#[inline(always)]
unsafe fn expand_permute_wide_indices<R: Register>(mask: Storage<R::Mask>) -> Storage<R::Unsigned> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let groups = n / 8;

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };

    // First pass: per-group population counts (straight from the table, no
    // runtime `count_ones`) and their total. `groups <= 8`, so both loops are
    // compile-time-bounded and unroll.
    let mut counts = [0u8; 8];
    let mut total = 0usize;
    for group in 0..groups {
        let bmg = ((bm >> (group * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the 8-lane table's length minus one.
        let cnt = unsafe { EXPAND8.get_unchecked(bmg) }.1;
        counts[group] = cnt;
        total += cnt as usize;
    }

    // Second pass: per-output-lane global source indices per the formula
    // above, assembled directly at the index register's element width.
    let mut g: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> = GenericArray::default();
    let mut base = 0usize; // selected packed prefix: sum of earlier counts

    for group in 0..groups {
        let out_base = group * 8;
        let bmg = ((bm >> out_base) & 0xFF) as usize;
        let cnt = counts[group] as usize;
        // Unselected tail start for this group: earlier groups' unselected lanes.
        let ubase = total + (out_base - base);

        // SAFETY: `bmg <= 255`, exactly the 8-lane table's length minus one.
        let row = &unsafe { EXPAND8.get_unchecked(bmg) }.0;

        for j in 0..8 {
            let r = row[j] as usize;
            // Branchless cursor pick, mirroring `compress_permute_wide`.
            let src = if r < cnt { base + r } else { ubase + (r - cnt) };

            // SAFETY: `base + r < total <= n` for selected lanes; for unselected
            // lanes `ubase + (r - cnt)` walks `[total, n)` exactly once each.
            unsafe { *g.get_unchecked_mut(out_base + j) = Element::from_u16(src as u16) };
        }

        base += cnt;
    }

    R::Unsigned::new(g)
}

/// Build a *single-register* **unmerge**-control table for splitting a merged
/// `2H`-lane block back into two packed `H`-lane blocks (`TwoH == 2*H`,
/// `Entries == H+1`), with `u8` entries that are **pair-relative**: index `k`
/// addresses lane `k` of the `2H`-lane pair, so a caller unmerging pair `p` at
/// block half-size `H` adds the constant base `p * 2H` bytewise.
///
/// This is the row-wise inverse of `build_merge_ctrl_u8`'s table (see
/// [`super::compress`]): the merge slid the right block's `d` live lanes up to
/// start at merged position `count_left`, and the unmerge slides them back down
/// to lane `H`. Row `c = count_left`, output position `k`:
///
/// - `k < c`: entry `k`. The left block's selected prefix never moved.
/// - `c <= k < H`: the left block's dead tail. Filler `2H - 1` (see below).
/// - `k >= H`: entry `c + (k - H)`. The right block's data begins at merged
///   position `c`, so its `j`-th lane is at `c + j` with `j = k - H`.
///
/// Every entry is in range: `c + j <= H + (H - 1) = 2H - 1`.
///
/// # Live-prefix invariant (why the dead entries need not be zero)
///
/// The compress-side merge table can prove its sentinel lane holds a *zero*,
/// because `compress_z` zeroes its input up front and each block entering a
/// merge is literally `[selected..., 0...]`. **The unmerge cannot**: `expand_z`
/// is handed a caller's register whose lanes past `popcount(mask)` are
/// unspecified, so no lane is guaranteed zero and there is nothing cheap to
/// point a zeroing sentinel at. (Zeroing the input tail up front would need a
/// count-derived prefix mask, strictly more work than the trailing
/// [`zz`](CoreRegister::zz) described below.)
///
/// What holds instead is weaker but sufficient. Call a block's first `count`
/// lanes its *live prefix*. Everything past it is unspecified. If the incoming
/// `2H`-block's live prefix (`c + d` lanes) is correct, then:
///
/// - left output `k < c` reads merged `k < c <= c + d`, which is live.
/// - right output `k >= H` with `j = k - H < d` reads merged `c + j < c + d`,
///   also live.
///
/// So both output halves have correct live prefixes (`c` and `d` lanes), and
/// unspecified data can only ever land in a dead position. The filler `2H - 1`
/// is therefore just "any in-range lane". The induction closes at level 0: each
/// 8-lane group's live prefix is correct, an [`EXPAND8`] row sends every
/// *selected* lane to a position `< count`, so every selected output lane is
/// correct, and the unselected lanes, which read dead data, are zeroed by the
/// `zz` that [`expand_z_grouped`] composes at the end.
const fn build_unmerge_ctrl_u8<TwoH: ArrayLength, Entries: ArrayLength>(
    h: usize,
) -> GenericArray<GenericArray<u8, TwoH>, Entries> {
    let two_h = <TwoH as Unsigned>::USIZE;
    let entries = <Entries as Unsigned>::USIZE;

    let mut t: GenericArray<GenericArray<u8, TwoH>, Entries> =
        unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
    let rows = t.as_mut_slice();

    let mut c = 0;
    while c < entries {
        let row = rows[c].as_mut_slice();
        let mut k = 0;
        while k < two_h {
            row[k] = if k < c {
                k as u8
            } else if k < h {
                // Left block's dead tail. Any in-range lane will do, see the
                // live-prefix invariant above.
                (two_h - 1) as u8
            } else {
                (c + (k - h)) as u8
            };
            k += 1;
        }
        c += 1;
    }

    t
}

// Single-register unmerge levels: 16->8+8, 32->16+16, 64->32+32. 144 + 544 +
// 2112 = 2800 bytes of `.rodata` total, mirroring `MERGE_CTRL_U8_*`.
static UNMERGE_CTRL_U8_8: GenericArray<GenericArray<u8, U16>, generic_array::typenum::U9> = build_unmerge_ctrl_u8(8);
static UNMERGE_CTRL_U8_16: GenericArray<GenericArray<u8, U32>, generic_array::typenum::U17> = build_unmerge_ctrl_u8(16);
static UNMERGE_CTRL_U8_32: GenericArray<GenericArray<u8, U64>, generic_array::typenum::U33> = build_unmerge_ctrl_u8(32);

/// Pointer to the `2H`-byte unmerge-control row for `count` selected lanes in
/// the left block. `H` is a literal at every call site, so the match folds to
/// one table base.
///
/// # Safety
///
/// `H` must be 8, 16 or 32, and `count <= H`.
#[inline(always)]
unsafe fn unmerge_ctrl_row_u8<const H: usize>(count: usize) -> *const u8 {
    unsafe {
        match H {
            8 => UNMERGE_CTRL_U8_8.get_unchecked(count).as_ptr(),
            16 => UNMERGE_CTRL_U8_16.get_unchecked(count).as_ptr(),
            _ => UNMERGE_CTRL_U8_32.get_unchecked(count).as_ptr(),
        }
    }
}

/// One single-register unmerge level at block half-size `H` (`TWO_H == 2 * H`):
/// build the index register that splits **every** merged block at this level
/// simultaneously.
///
/// The row for pair `p` is selected by that pair's `count_left` (the number of
/// selected lanes in its LEFT `H`-lane half), which is the sum of the 8-lane
/// group counts covered by that half. Unlike the compress-side
/// `merge_indices`, the counts cannot be folded level by level here: the levels
/// run coarsest-first, so each one re-sums the raw per-group counts (at most
/// `H / 8 <= 4` adds per pair, all compile-time bounded).
#[inline(always)]
fn unmerge_indices<R: Register + ?Sized, const H: usize, const TWO_H: usize>(
    group_counts: &[u8; 8],
) -> Storage<R::Unsigned> {
    let pairs = <R::Lanes as Unsigned>::USIZE / TWO_H;
    let groups_per_half = H / 8;

    let mut bytes: GenericArray<u8, R::Lanes> = GenericArray::default();

    let mut p = 0;
    while p < pairs {
        // Groups `[p * 2 * gph, p * 2 * gph + gph)` are this pair's left half.
        let first = p * 2 * groups_per_half;
        let mut left = 0usize;
        let mut i = 0;
        while i < groups_per_half {
            left += group_counts[first + i] as usize;
            i += 1;
        }

        // SAFETY: `H` is 8/16/32 at every instantiation and `left <= H` (the
        // half spans `groups_per_half * 8 == H` lanes).
        let row = unsafe { unmerge_ctrl_row_u8::<H>(left) };
        // SAFETY: the row is `TWO_H` bytes and `pairs * TWO_H == LANES`.
        unsafe { copy_row_into::<R, TWO_H>(&mut bytes, p * TWO_H, row) };

        p += 1;
    }

    R::Unsigned::add(R::widen_index_bytes(&bytes), block_base::<R, TWO_H>())
}

/// Zeroing inverse left-pack for a native wide register: **`log2` unmerge passes
/// plus grouped rows**, entirely in one register and entirely branchless. It is
/// the exact inverse of
/// [`compress_z_grouped`](super::compress::compress_z_grouped) on the selected
/// data, run backwards.
///
/// `compress_z_grouped` goes *group compress -> merge 8 -> merge 16 -> merge
/// 32*. This runs *unmerge 32 -> unmerge 16 -> unmerge 8 -> group expand*, one
/// [`permutev`](Register::permutev) per level, each fed by an index register
/// assembled from `&'static` table rows plus one constant vector add:
///
/// 1. **Unmerge levels**: `log2(LANES / 8)` passes at half-sizes 32, 16, 8,
///    **coarsest first**. Each pass reads the pair-relative
///    `UNMERGE_CTRL_U8_*` rows for all pairs at that level, adds the pair base,
///    and resolves them with a single permute. After the last pass every 8-lane
///    group's *live prefix* holds its own selected values in packed order.
/// 2. **Group expand**: one permute with [`EXPAND8`] rows plus the `8g` block
///    base scatters each group's packed prefix to its selected lanes.
/// 3. **`zz`**: unselected output lanes read their group's dead tail, so a
///    final [`zz`](CoreRegister::zz) zeroes them. This is the one asymmetry with
///    `compress_z_grouped`, which zeroes *first*: that direction may assume its
///    pad lanes are zero, whereas here the caller's lanes past
///    `popcount(mask)` are unspecified and there is no free zero to sentinel
///    at. See [`build_unmerge_ctrl_u8`] for the live-prefix invariant that
///    makes "dead lanes may hold anything" sound at every level.
///
/// Valid for `LANES % 8 == 0` and `16 <= LANES <= 64`, the same shapes as
/// `compress_z_grouped` (below 16 there is nothing to unmerge, so use
/// [`expand_permute`], and above 64 `native_bitmask` is unavailable). This is
/// the `expand_z` arm of `compress_via_wide!`. The non-zeroing `expand` composes
/// two of these trees, see [`expand_grouped`].
#[inline(always)]
pub fn expand_z_grouped<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "expand_z_grouped requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "expand_z_grouped requires 16..=64 lanes"
        );
    }

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };

    R::zz(mask, expand_z_grouped_bm::<R>(value, bm))
}

/// The permute chain of [`expand_z_grouped`] with the `movemask` hoisted out and
/// the trailing [`zz`](CoreRegister::zz) left to the caller, so a caller running
/// the kernel **twice** (see [`expand_grouped`]) pays for one
/// [`native_bitmask`](MaskRegister::native_bitmask) total and picks its own
/// zeroing polarity.
///
/// Only the low `LANES` bits of `bm` are read, one 8-lane group at a time, so
/// passing the bitwise complement `!bm` runs the whole unmerge/expand chain over
/// the *unselected* lanes (each group reading row `0xFF ^ bmg`) with no second
/// movemask and no mask-register complement.
///
/// The returned register is only correct in the lanes selected by `bm`. Every
/// other lane holds dead data (the live-prefix invariant of
/// [`build_unmerge_ctrl_u8`]), which is exactly what the caller's `zz` / `nz`
/// discards.
#[inline(always)]
fn expand_z_grouped_bm<R: Register>(value: Storage<R>, bm: u64) -> Storage<R> {
    expand_z_grouped_apply::<R>(&expand_z_grouped_plan::<R>(bm), value)
}

/// The **mask-only** half of [`expand_z_grouped_bm`]: every unmerge level's
/// index register (coarsest first) plus the level-0 row bytes.
///
/// Mirrors [`compress_z_grouped_plan`](super::compress). See [`GroupedPlan`]
/// for why the ladder is a fixed-size `EMPTY`-padded array.
#[inline(always)]
fn expand_z_grouped_plan<R: Register>(bm: u64) -> GroupedPlan<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let groups = n / 8;

    // One pass over the groups collects both things the kernel needs: the
    // per-group counts (which every unmerge level re-sums to pick its rows) and
    // the level-0 `EXPAND8` row bytes, staged now because the same table entry
    // carries both. The permute that consumes them runs last.
    let mut counts = [0u8; 8];
    let mut bytes: GenericArray<u8, R::Lanes> = GenericArray::default();

    let mut g = 0;
    while g < groups {
        let bmg = ((bm >> (g * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the table length minus one.
        let entry = unsafe { EXPAND8.get_unchecked(bmg) };
        // SAFETY: the row is 8 bytes and `8 * groups == n`.
        unsafe { copy_row_into::<R, 8>(&mut bytes, g * 8, entry.0.as_ptr()) };
        counts[g] = entry.1;
        g += 1;
    }

    let mut plan = [<R::Unsigned as CoreRegister>::EMPTY; GROUPED_LEVELS];

    // Unmerge levels, coarsest first. Literal half-sizes so the control-table
    // select folds. Slots are fixed per level (0 = the 64-lane level, 3 = level
    // 0), so a narrower shape simply leaves its high levels' slots at EMPTY and
    // the apply ladder's matching `if const` arms never read them.
    if const { <R::Lanes as Unsigned>::USIZE >= 64 } {
        plan[0] = unmerge_indices::<R, 32, 64>(&counts);
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 32 } {
        plan[1] = unmerge_indices::<R, 16, 32>(&counts);
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 16 } {
        plan[2] = unmerge_indices::<R, 8, 16>(&counts);
    }

    (plan, bytes)
}

/// The **data** half of [`expand_z_grouped_bm`]: the permute ladder, fed by
/// [`expand_z_grouped_plan`].
///
/// The returned register is only correct in the lanes selected by the plan's
/// bitmask, and the caller composes its own [`zz`](CoreRegister::zz) / `nz`.
#[inline(always)]
fn expand_z_grouped_apply<R: Register>(plan: &GroupedPlan<R>, value: Storage<R>) -> Storage<R> {
    let (idx, bytes) = plan;

    let mut cur = value;

    if const { <R::Lanes as Unsigned>::USIZE >= 64 } {
        cur = R::permutev(cur, idx[0]);
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 32 } {
        cur = R::permutev(cur, idx[1]);
    }
    if const { <R::Lanes as Unsigned>::USIZE >= 16 } {
        cur = R::permutev(cur, idx[2]);
    }

    // Level 0: scatter each group's packed prefix to its selected lanes. Every
    // selected lane is now correct. The unselected ones read dead data, which
    // the caller's `zz` discards.
    R::permutev(cur, R::Unsigned::add(R::widen_index_bytes(bytes), block_base::<R, 8>()))
}

/// Same-mask multi-vector [`expand_z_grouped`]: one plan, `N` applications plus
/// one [`zz`](CoreRegister::zz) each.
#[inline(always)]
pub fn expand_z_grouped_n<R: Register, const N: usize>(
    values: [Storage<R>; N],
    mask: Storage<R::Mask>,
) -> [Storage<R>; N] {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "expand_z_grouped_n requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "expand_z_grouped_n requires 16..=64 lanes"
        );
    }

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within Thermite.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };

    let plan = expand_z_grouped_plan::<R>(bm);

    let mut out = [R::EMPTY; N];

    let mut i = 0;
    while i < N {
        out[i] = R::zz(mask, expand_z_grouped_apply::<R>(&plan, values[i]));
        i += 1;
    }

    out
}

/// Non-zeroing inverse left-pack (`expand`) for a native wide register: **two
/// [`expand_z_grouped`] trees plus one shift and one OR**, entirely branchless
/// and with a single `movemask` for both trees.
///
/// A non-zeroing expand is the inverse stable partition: selected lane `i` reads
/// the packed front at its rank among the selected lanes, unselected lane `i`
/// reads position `t + (rank among the unselected)`, where `t = popcount(m)`.
/// Both halves are themselves zeroing expansions, the second one just needs its
/// source shifted down past the front:
///
/// ```text
/// sel   = expand_z(v, m)                    // correct at selected lanes, 0 elsewhere
/// tailv = shift_down(v, t)                  // tailv[j] = v[t + j]
/// out   = sel | expand_z(tailv, !m)         // correct at unselected lanes, 0 elsewhere
/// ```
///
/// # The shift, and why the OR is exact
///
/// `shift_down(v, t)` is one [`swizzle`](Register::swizzle) whose **second**
/// source is [`EMPTY`](CoreRegister::EMPTY) and whose index register is the
/// constant [`lane_iota`](super::compress::lane_iota) plus the broadcast `t`.
/// Output lane `j` reads index `j + t`: inside `0..LANES` that is `v[t + j]`,
/// and at `j + t >= LANES` it falls into the EMPTY window and reads zero. Only
/// `j < LANES - t` is ever consumed (there are exactly `LANES - t` unselected
/// lanes, so the largest rank is `LANES - t - 1`), so the overflow window is
/// pure padding.
///
/// Feeding that to the complement tree gives, at unselected lane `i`,
/// `tailv[rank_unsel(i)] = v[t + rank_unsel(i)]`, exactly the global source
/// index `expand_permute_wide` computes as `total + ubase_g + (r - cnt_g)`,
/// since `ubase_g + (r - cnt_g)` *is* the rank among the unselected lanes.
///
/// The OR is exact because the two operands never overlap: the first tree's
/// [`zz`](CoreRegister::zz) zeroes every unselected lane and the second tree's
/// [`nz`](CoreRegister::nz) zeroes every selected one, so each is all-zero bits
/// precisely where the other is live. As on the compress side, the complement
/// tree costs no extra `movemask`, since `!bm` supplies each group's
/// `0xFF ^ bmg` row.
///
/// Valid for the same shapes as [`expand_z_grouped`]. This is the `expand` arm
/// of `compress_via_wide!`. [`expand_permute_wide`] stays for the
/// `ArrayRegister` routing, which has no single-register permute to build on.
#[inline(always)]
pub fn expand_grouped<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "expand_grouped requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "expand_grouped requires 16..=64 lanes"
        );
    }

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within
    // Thermite. Read ONCE and shared by both trees.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };
    let total = bm.count_ones() as usize;

    let sel = R::zz(mask, expand_z_grouped_bm::<R>(value, bm));

    let idx = R::Unsigned::add(
        super::compress::lane_iota::<R>(),
        R::Unsigned::splat(Element::from_u16(total as u16)),
    );
    let tailv = R::swizzle(value, R::EMPTY, idx);

    R::bitor(sel, R::nz(mask, expand_z_grouped_bm::<R>(tailv, !bm)))
}

/// Same-mask multi-vector [`expand_grouped`]: **two** plans (selected and
/// complement) plus the shared shift index register built once, then per value
/// one selected ladder, one shift `swizzle`, one complement ladder and the
/// `bitor`.
#[inline(always)]
pub fn expand_grouped_n<R: Register, const N: usize>(
    values: [Storage<R>; N],
    mask: Storage<R::Mask>,
) -> [Storage<R>; N] {
    const {
        assert!(
            <R::Lanes as Unsigned>::USIZE % 8 == 0,
            "expand_grouped_n requires a lane count that is a multiple of 8"
        );
        assert!(
            <R::Lanes as Unsigned>::USIZE >= 16 && <R::Lanes as Unsigned>::USIZE <= 64,
            "expand_grouped_n requires 16..=64 lanes"
        );
    }

    // SAFETY: <= 64 lanes, so `native_bitmask` is always `Some` within
    // Thermite. Read ONCE and shared by both trees and every value.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() };
    let total = bm.count_ones() as usize;

    let plan_sel = expand_z_grouped_plan::<R>(bm);
    let plan_tail = expand_z_grouped_plan::<R>(!bm);

    let idx = R::Unsigned::add(
        super::compress::lane_iota::<R>(),
        R::Unsigned::splat(Element::from_u16(total as u16)),
    );

    let mut out = [R::EMPTY; N];

    let mut i = 0;
    while i < N {
        let sel = R::zz(mask, expand_z_grouped_apply::<R>(&plan_sel, values[i]));
        let tailv = R::swizzle(values[i], R::EMPTY, idx);

        out[i] = R::bitor(sel, R::nz(mask, expand_z_grouped_apply::<R>(&plan_tail, tailv)));
        i += 1;
    }

    out
}

/// The portable scalar inverse left-pack: the default body behind
/// [`Register::expand`]. Free-standing so blanket impls (e.g. `ArrayRegister`)
/// can fall back to it from their own `expand` overrides, like
/// [`compress_default`].
pub fn expand_default<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;

    let src = R::as_slice(&value);
    let mut result = value;
    let dst = R::as_mut_slice(&mut result);

    let mut pos = 0;

    // Selected lanes read the packed front, in order.
    for i in 0..n {
        if <R::Mask as MaskRegister>::test(mask, i) {
            dst[i] = src[pos];
            pos += 1;
        }
    }

    // Unselected lanes read the tail, in order.
    for i in 0..n {
        if !<R::Mask as MaskRegister>::test(mask, i) {
            dst[i] = src[pos];
            pos += 1;
        }
    }

    result
}

/// The single-pass zero-filling inverse left-pack: the default body behind
/// [`Register::expand_z`]. Selected lanes read the packed front in order,
/// everything else stays zero (matching AVX-512 zero-masking `vpexpand*`).
pub fn expand_z_default<R: Register>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    let n = <R::Lanes as Unsigned>::USIZE;
    let src = R::as_slice(&value);

    // `EMPTY` is zero, so unselected lanes are already filled.
    let mut result = R::EMPTY;
    let dst = R::as_mut_slice(&mut result);

    let mut pos = 0;
    for i in 0..n {
        if <R::Mask as MaskRegister>::test(mask, i) {
            dst[i] = src[pos];
            pos += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::register::array::ArrayRegister;

    // Scalar oracle for the plain (non-zeroing) expand: the inverse stable
    // partition. Kept independent of the tables so these tests do not validate
    // the implementation against itself.
    fn expand_oracle<const N: usize>(data: &[i32], bits: u64) -> [i32; 64] {
        let mut expected = [0i32; 64];

        let mut pos = 0;
        for lane in 0..N {
            if (bits >> lane) & 1 == 1 {
                expected[lane] = data[pos];
                pos += 1;
            }
        }
        for lane in 0..N {
            if (bits >> lane) & 1 == 0 {
                expected[lane] = data[pos];
                pos += 1;
            }
        }
        expected
    }

    fn make_mask<const N: usize>(bits: u64) -> Storage<<ArrayRegister<i32, N> as CoreRegister>::Mask>
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        let mut sel = [0i32; 64];
        for lane in 0..N {
            sel[lane] = ((bits >> lane) & 1) as i32;
        }
        <ArrayRegister<i32, N>>::into_mask(<ArrayRegister<i32, N>>::new(
            GenericArray::from_slice(&sel[..N]).clone(),
        ))
    }

    fn make_value<const N: usize>() -> Storage<ArrayRegister<i32, N>>
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        <ArrayRegister<i32, N>>::new(GenericArray::from_slice(&data[..N]).clone())
    }

    // Exhaustively check `expand_permute` against the oracle for every mask.
    fn check<const N: usize>()
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32, Lanes: CompressTable>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut data = [0i32; 64];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as i32;
        }
        let value = make_value::<N>();

        for bits in 0u64..(1 << N) {
            let mask = make_mask::<N>(bits);
            let expected = expand_oracle::<N>(&data, bits);

            let got = expand_permute::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn expand_permute_exhaustive() {
        check::<2>();
        check::<4>();
        check::<8>();
    }

    // Check `expand_permute_wide` against the oracle for a set of mask patterns.
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
        let value = make_value::<N>();

        for bits in patterns {
            let mask = make_mask::<N>(bits);
            let expected = expand_oracle::<N>(&data, bits);

            let got = expand_permute_wide::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn expand_permute_wide_correct() {
        check_wide::<8>(0..(1u64 << 8));
        check_wide::<16>(0..(1u64 << 16));

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

    // Zeroing oracle for a `LANES`-lane expand: selected lane `i` reads the
    // packed front in order, everything else is zero.
    fn expand_z_oracle<const N: usize>(data: &[i32], bits: u64) -> [i32; 64] {
        let mut expected = [0i32; 64];

        let mut pos = 0;
        for lane in 0..N {
            if (bits >> lane) & 1 == 1 {
                expected[lane] = data[pos];
                pos += 1;
            }
        }
        expected
    }

    // Check `expand_z_grouped` (log2 unmerges + grouped rows) against the
    // zeroing oracle on `ArrayRegister<i32, N>`, whose `permutev` is the scalar
    // default, so this exercises the index assembly and the unmerge sentinel,
    // not any backend permute. Mirrors `compress_z_grouped_correct`.
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
        let value = make_value::<N>();

        for bits in patterns {
            let mask = make_mask::<N>(bits);
            let expected = expand_z_oracle::<N>(&data, bits);

            let got = expand_z_grouped::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn expand_z_grouped_correct() {
        // 16 lanes exhaustively (one unmerge level), 32/64 sampled.
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

    // Check the non-zeroing two-tree composition against the inverse
    // stable-partition oracle on `ArrayRegister<i32, N>` (scalar
    // `permutev`/`swizzle`), so this exercises the shift-down window and the
    // OR-exactness argument rather than any backend permute.
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
        let value = make_value::<N>();

        for bits in patterns {
            let mask = make_mask::<N>(bits);
            let expected = expand_oracle::<N>(&data, bits);

            let got = expand_grouped::<R<N>>(value, mask);
            let got = <R<N>>::as_slice(&got);
            for lane in 0..N {
                assert_eq!(got[lane], expected[lane], "N={N} bits={bits:b} lane={lane}");
            }
        }
    }

    #[test]
    fn expand_grouped_correct() {
        check_grouped::<16>(0..(1 << 16));

        let mut s = 0xBB67_AE85_84CA_A73Bu64;
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

    // Exhaustive inverse laws for the non-zeroing pair: a permutation and its
    // exact inverse, so both round trips are the identity for every mask.
    #[test]
    fn grouped_round_trip_16() {
        type R = ArrayRegister<i32, 16>;

        let value = make_value::<16>();

        for bits in 0u64..(1 << 16) {
            let mask = make_mask::<16>(bits);

            let there = expand_grouped::<R>(compress_grouped::<R>(value, mask), mask);
            let back = compress_grouped::<R>(expand_grouped::<R>(value, mask), mask);

            assert_eq!(
                <R>::as_slice(&there),
                <R>::as_slice(&value),
                "expand(compress) bits={bits:b}"
            );
            assert_eq!(
                <R>::as_slice(&back),
                <R>::as_slice(&value),
                "compress(expand) bits={bits:b}"
            );
        }
    }

    // The inverse laws that justify wiring this as `expand_z`:
    //
    //   expand_z(compress_z(v, m), m) == zz(m, v)     (selected lanes restored)
    //   compress_z(expand_z(p, m), m) == p[0..count]  (packed prefix restored)
    //
    // The first is the strongest statement available: `compress_z` destroys the
    // unselected lanes, so the round trip can only recover the masked input.
    fn check_z_roundtrip<const N: usize>(patterns: impl Iterator<Item = u64>)
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength,
        ArrayRegister<i32, N>: Register<Element = i32>,
    {
        type R<const N: usize> = ArrayRegister<i32, N>;

        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for bits in patterns {
            // Fresh pseudo-random payload per mask, so a lucky value pattern
            // cannot hide a routing bug.
            let mut data = [0i32; 64];
            for i in 0..N {
                data[i] = (rng() as u32 as i32) | 1; // nonzero: zeros must come from masking
            }
            let value = <R<N>>::new(GenericArray::from_slice(&data[..N]).clone());
            let mask = make_mask::<N>(bits);
            let count = (bits & (u64::MAX >> (64 - N))).count_ones() as usize;

            // there: expand_z . compress_z == zz
            let there = expand_z_grouped::<R<N>>(compress_z_grouped::<R<N>>(value, mask), mask);
            let there = <R<N>>::as_slice(&there);
            for lane in 0..N {
                let want = if (bits >> lane) & 1 == 1 { data[lane] } else { 0 };
                assert_eq!(
                    there[lane], want,
                    "expand_z(compress_z) N={N} bits={bits:b} lane={lane}"
                );
            }

            // back: compress_z . expand_z restores the packed prefix (the tail
            // of `value` past `count` is discarded by `expand_z`, so only
            // `[0, count)` is recoverable).
            let back = compress_z_grouped::<R<N>>(expand_z_grouped::<R<N>>(value, mask), mask);
            let back = <R<N>>::as_slice(&back);
            for lane in 0..count {
                assert_eq!(
                    back[lane], data[lane],
                    "compress_z(expand_z) N={N} bits={bits:b} lane={lane}"
                );
            }
            for lane in count..N {
                assert_eq!(
                    back[lane], 0,
                    "compress_z(expand_z) tail N={N} bits={bits:b} lane={lane}"
                );
            }
        }
    }

    #[test]
    fn expand_z_grouped_roundtrip() {
        check_z_roundtrip::<16>(0..(1 << 16));

        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let structured = [0u64, 0xFFFF_FFFF, 0x5555_5555, 0xAAAA_AAAA, 0x0000_FFFF, 0xFFFF_0000];
        check_z_roundtrip::<32>(structured.into_iter().chain((0..5000).map(|_| rng() & 0xFFFF_FFFF)));
    }

    // The defining property, checked at the polyfill level: the tables are exact
    // mutual inverses, so both round trips are identities for every mask.
    #[test]
    fn round_trip_exhaustive() {
        type R = ArrayRegister<i32, 8>;

        let value = make_value::<8>();

        for bits in 0u64..256 {
            let mask = make_mask::<8>(bits);

            let there = expand_permute::<R>(compress_permute::<R>(value, mask), mask);
            let back = compress_permute::<R>(expand_permute::<R>(value, mask), mask);

            assert_eq!(
                <R>::as_slice(&there),
                <R>::as_slice(&value),
                "expand(compress(v)) bits={bits:b}"
            );
            assert_eq!(
                <R>::as_slice(&back),
                <R>::as_slice(&value),
                "compress(expand(v)) bits={bits:b}"
            );
        }
    }

    #[test]
    fn round_trip_wide_16() {
        type R = ArrayRegister<i32, 16>;

        let value = make_value::<16>();

        for bits in 0u64..(1 << 16) {
            let mask = make_mask::<16>(bits);

            let there = expand_permute_wide::<R>(compress_permute_wide::<R>(value, mask), mask);
            assert_eq!(
                <R>::as_slice(&there),
                <R>::as_slice(&value),
                "expand(compress(v)) bits={bits:b}"
            );
        }
    }
}
