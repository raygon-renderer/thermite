//! Generic stream-compaction (`compress` / left-pack) polyfills.
//!
//! "Better than scalar" generic implementations of
//! [`Register::compress`]/[`compress_z`](Register::compress_z) that a backend can
//! delegate to when its registers satisfy the extra trait bounds. The base trait
//! default is a scalar element-by-element compaction (correct for every
//! register, no bounds); these resolve the permutation from compile-time tables
//! instead.
//!
//! Pick by lane count:
//!
//! - [`compress_permute`] - up to 8 lanes ([`CompressTable`]): `movemask` ->
//!   table row -> one permute. Branchless and loop-free.
//! - [`compress_permute_wide`] - any multiple of 8 lanes up to 64: per-8-lane
//!   table lookups assembled into one global gather index + one full-width
//!   permute. Non-zeroing.
//! - [`compress_z_merge2`] / [`compress_z_merge4`] / [`compress_z_merge8`] -
//!   *zeroing* left-pack for 16/32/64-lane registers split into native 8-lane
//!   chunks: compact each chunk, then merge with a count-indexed permute tree.
//!   Benchmarks ~2.7x faster than `compress_permute_wide` at 16 lanes; the
//!   advantage shrinks with more chunks (the merge swizzle is O(chunks^2)), so
//!   it is the right choice at 16 lanes and `compress_permute_wide` wins beyond.

use core::mem::MaybeUninit;

use generic_array::{
    ArrayLength, GenericArray,
    sequence::GenericSequence,
    typenum::{U1, U2, U3, U4, U5, U6, U7, U8, U16, U32, U64, U128, U256, Unsigned},
};

use super::*;

/// One table row: the stable-partition gather indices for an 8-lane mask plus
/// its population count (so callers never recompute it at runtime).
pub type CompressRow = (GenericArray<u32, U8>, u8);

/// The single 256-entry 8-lane left-pack table, shared by every lane count.
///
/// Row `m` (an 8-bit mask) holds the gather indices that *stably partition*
/// `[0, 8)` - selected lanes (bit set in `m`) first, in order, then the
/// unselected lanes, also in order - plus the population count of `m`.
///
/// One table serves all lane counts. A register with `LANES <= 8` reads row `m`
/// (where `m` is its `LANES`-bit `movemask`) and [`transmute_copy`]s the first
/// `LANES` indices: the high padding lanes (`LANES..8`) are always unselected,
/// so they sort to the tail and are dropped, leaving exactly the `LANES`-lane
/// compaction. Wider registers index it per 8-lane chunk.
///
/// [`transmute_copy`]: core::mem::transmute_copy
pub static COMPRESS8: GenericArray<CompressRow, U256> = build_table8();

/// Marker for lane counts (`1..=8`) small enough to `transmute_copy` their
/// compaction indices straight out of [`COMPRESS8`].
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
/// entirely in const evaluation - no runtime cost. Each row also carries its
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
                row[pos] = i as u32;
                pos += 1;
            }
            i += 1;
        }

        // After the selected pass, `pos` is exactly the population count.
        let count = pos as u8;

        let mut i = 0;
        while i < lanes {
            if (m >> i) & 1 == 0 {
                row[pos] = i as u32;
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
/// shared [`COMPRESS8`] table - no runtime loop or branch.
///
/// Runtime cost is: read the lane bitmask
/// ([`native_bitmask`](MaskRegister::native_bitmask), one `movemask`),
/// `transmute_copy` the first `LANES` indices of row `bm`, and `permutev`. The
/// selected lanes are packed to the front in order; the unselected lanes keep
/// their values in the tail (also in order). The zero-filling variant composes a
/// [`zz`](CoreRegister::zz) beforehand (see
/// [`Register::compress_z`]).
///
/// On a backend with hardware permute this lowers to `movemask` + a table load +
/// `vpermps`/`pshufb`/`i8x16.swizzle`. For an emulated-wide
/// [`ArrayRegister`](crate::register::array::ArrayRegister) the index lookup is
/// still loop-free; only the `permutev` itself does the cross-chunk routing.
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
/// `LANES <= 8` but cannot name the [`CompressTable`] bound - specifically
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

    // SAFETY: `bm < 256`, exactly the table length. The row is `[u32; 8]`;
    // `GenericArray<u32, R::Lanes>` is `LANES * 4 <= 32` bytes, so `transmute_copy`
    // reads only the first `LANES` indices - exactly the `LANES`-lane compaction,
    // since rows are stable-partitioned and the padding lanes `LANES..8` are
    // always unselected (they sort past the `LANES`-th slot).
    let idxs: GenericArray<u32, R::Lanes> = unsafe { core::mem::transmute_copy(&COMPRESS8.get_unchecked(bm).0) };

    R::permutev(value, idxs)
}

/// Wide left-pack (`compress`, non-zeroing) for lane counts above 8, built by
/// applying the 8-lane [`CompressTable`] kernel once per 8-lane group and
/// merging the per-group results into a single global gather index that is
/// resolved with one full-width [`permutev`](Register::permutev).
///
/// Requires `LANES` to be a multiple of 8 and `<= 64` (so `native_bitmask` is
/// available) - i.e. the `*x16`/`*x32`/`*x64` registers. The actual element
/// movement is a single permute; the only scalar work is the branchless
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

    // Second pass: assemble the global gather indices - selected lanes (stable
    // order) first, then the unselected lanes (stable order).
    let mut g: GenericArray<u32, R::Lanes> = GenericArray::default();
    let mut head = 0usize; // front cursor: where the next selected index goes
    let mut tail = total; // tail cursor: where the next unselected index goes

    for group in 0..groups {
        let base = group * 8;
        let bmg = ((bm >> base) & 0xFF) as usize;
        let cnt = counts[group] as usize;

        // SAFETY: `bmg <= 255`, exactly the 8-lane table's length minus one.
        let row = &unsafe { table8.get_unchecked(bmg) }.0;

        for j in 0..8 {
            let global = (base + row[j] as usize) as u32;

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

    R::permutev(value, g)
}

/// Chunked zeroing left-pack for any multiple-of-8 lane count up to 64 - the
/// loop-free generalization that avoids the per-lane gather-index scatter.
///
/// For each 8-lane chunk it stores that chunk's selected-first table row (source
/// offset `+8c`) at a running *destination* cursor that advances by the chunk's
/// population count. The next chunk's 8-wide store overwrites the prior chunk's
/// unselected tail, so the selected indices end up compacted in `g[0..total]`
/// with `N/8` chunk-granular vector stores instead of `N` per-lane writes. The
/// unwritten tail keeps the sentinel index `LANES`, which the all-zero second
/// `swizzle` source resolves to zero (together with `zz` zeroing the unselected
/// lanes the written tail still points at) - giving the zeroing partition with a
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

    // Gather indices; the sentinel `n` reads the all-zero second swizzle source.
    let mut g: GenericArray<u32, R::Lanes> = GenericArray::generate(|_| n as u32);

    let mut start = 0usize;
    for c in 0..groups {
        let base = (c * 8) as u32;
        let bmg = ((bm >> (c * 8)) & 0xFF) as usize;
        // SAFETY: `bmg <= 255`, exactly the table length minus one.
        let entry = unsafe { COMPRESS8.get_unchecked(bmg) };

        // The selected-first row, offset to source chunk `c`.
        let mut off = [0u32; 8];
        let mut j = 0;
        while j < 8 {
            off[j] = entry.0[j] + base;
            j += 1;
        }

        // SAFETY: `start <= n - 8` for every chunk (the prefix sum of counts,
        // each <= 8, is at most `(groups - 1) * 8 = n - 8`), so the 8-wide store
        // stays in bounds.
        unsafe { core::ptr::copy_nonoverlapping(off.as_ptr(), g.as_mut_ptr().add(start), 8) };

        start += entry.1 as usize;
    }

    R::swizzle(zeroed, R::EMPTY, g)
}

/// Build a merge-control table for two `H`-lane halves (`TwoH == 2*H`,
/// `Entries == H+1`). For each `count_left` in `0..=H`, row `count_left` is the
/// `2H`-lane swizzle control that keeps the left half's first `count_left` lanes
/// in place and slides the right half (stored in the upper `H` lanes of the
/// first swizzle source) to begin at lane `count_left`. Lanes that overflow past
/// `2H-1` read index `2H` - the first lane of the all-zero second swizzle source
/// - so the tail zeros for free (each half already being zero-padded).
const fn build_merge_ctrl<TwoH: ArrayLength, Entries: ArrayLength>(
    h: usize,
) -> GenericArray<GenericArray<u32, TwoH>, Entries> {
    let two_h = <TwoH as Unsigned>::USIZE;
    let entries = <Entries as Unsigned>::USIZE;

    let mut t: GenericArray<GenericArray<u32, TwoH>, Entries> = unsafe { MaybeUninit::zeroed().assume_init() };
    let rows = t.as_mut_slice();

    let mut c = 0;
    while c < entries {
        let row = rows[c].as_mut_slice();
        let mut k = 0;
        while k < two_h {
            row[k] = if k < c {
                k as u32
            } else {
                let src = h + (k - c);
                if src < two_h { src as u32 } else { two_h as u32 }
            };
            k += 1;
        }
        c += 1;
    }

    t
}

// One table per merge level: 8+8->16, 16+16->32, 32+32->64.
const MERGE_CTRL_8: GenericArray<GenericArray<u32, U16>, generic_array::typenum::U9> = build_merge_ctrl(8);
const MERGE_CTRL_16: GenericArray<GenericArray<u32, U32>, generic_array::typenum::U17> = build_merge_ctrl(16);
const MERGE_CTRL_32: GenericArray<GenericArray<u32, U64>, generic_array::typenum::U33> = build_merge_ctrl(32);

/// Compact one 8-lane chunk (zeroing) and return its population count: the chunk
/// becomes `[selected..., 0...]` and `count` is how many lanes are selected.
#[inline(always)]
fn compress_chunk_z<B: Register<Lanes = U8>>(chunk: Storage<B>, mask: Storage<B::Mask>) -> (Storage<B>, usize) {
    let packed = compress_permute::<B>(B::zz(mask, chunk), mask);
    // `native_bitmask` of an 8-lane mask yields exactly 8 bits, so `bm <= 255`.
    let bm = unsafe { <B::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() } as usize;
    let count = unsafe { COMPRESS8.get_unchecked(bm) }.1 as usize;
    (packed, count)
}

/// Two-level zeroing left-pack for a 2-chunk (16-lane) register: compact each
/// 8-lane half *natively* (table -> permute, fully vectorized) and merge them
/// with one count-indexed permute - no scalar gather-index scatter.
///
/// Measured ~2.7x faster than [`compress_permute_wide`] at 16 lanes: the scalar
/// scatter it replaces is a long data-dependent dependency chain, which costs
/// far more than the two extra shuffles. Result is the zeroing (`compress_z`)
/// partition: selected lanes to the front, rest zeroed.
#[inline(always)]
pub fn compress_z_merge2<B>(chunks: [Storage<B>; 2], masks: [Storage<B::Mask>; 2]) -> [Storage<B>; 2]
where
    B: Register<Lanes = U8>,
    crate::register::array::ArrayRegister<B, 2>:
        Register<Lanes = U16, Storage = crate::register::array::ArrayRegister<B, 2>>,
{
    use crate::register::array::ArrayRegister;

    let (lo, count_lo) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (hi, _) = compress_chunk_z::<B>(chunks[1], masks[1]);

    // Slide `hi` to begin at lane `count_lo`; overflow + each half's zero pad
    // give the zero tail. The all-zero second source supplies index 16.
    let full = ArrayRegister::<B, 2>([lo, hi]);
    let zero = ArrayRegister::<B, 2>([B::EMPTY, B::EMPTY]);
    let ctrl = unsafe { MERGE_CTRL_8.get_unchecked(count_lo) }.clone();

    let merged: ArrayRegister<B, 2> = <ArrayRegister<B, 2>>::swizzle(full, zero, ctrl);
    merged.0
}

/// Four-chunk (32-lane) zeroing left-pack: a two-level merge tree. Compact each
/// 8-lane chunk natively, merge `(0,1)` and `(2,3)` into 16-lane blocks, then
/// merge those two into 32 lanes. No scalar gather-index scatter.
#[inline(always)]
pub fn compress_z_merge4<B>(chunks: [Storage<B>; 4], masks: [Storage<B::Mask>; 4]) -> [Storage<B>; 4]
where
    B: Register<Lanes = U8>,
    crate::register::array::ArrayRegister<B, 2>:
        Register<Lanes = U16, Storage = crate::register::array::ArrayRegister<B, 2>>,
    crate::register::array::ArrayRegister<B, 4>:
        Register<Lanes = U32, Storage = crate::register::array::ArrayRegister<B, 4>>,
{
    use crate::register::array::ArrayRegister;

    let (c0, n0) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (c1, n1) = compress_chunk_z::<B>(chunks[1], masks[1]);
    let (c2, n2) = compress_chunk_z::<B>(chunks[2], masks[2]);
    let (c3, _) = compress_chunk_z::<B>(chunks[3], masks[3]);

    // Level 0: 8+8 -> 16.
    let zero16 = ArrayRegister::<B, 2>([B::EMPTY; 2]);
    let m01: ArrayRegister<B, 2> = <ArrayRegister<B, 2>>::swizzle(
        ArrayRegister::<B, 2>([c0, c1]),
        zero16,
        unsafe { MERGE_CTRL_8.get_unchecked(n0) }.clone(),
    );
    let m23: ArrayRegister<B, 2> = <ArrayRegister<B, 2>>::swizzle(
        ArrayRegister::<B, 2>([c2, c3]),
        zero16,
        unsafe { MERGE_CTRL_8.get_unchecked(n2) }.clone(),
    );

    // Level 1: 16+16 -> 32, placing m23 after m01's `n0 + n1` selected lanes.
    let full = ArrayRegister::<B, 4>([m01.0[0], m01.0[1], m23.0[0], m23.0[1]]);
    let zero32 = ArrayRegister::<B, 4>([B::EMPTY; 4]);
    let result: ArrayRegister<B, 4> =
        <ArrayRegister<B, 4>>::swizzle(full, zero32, unsafe { MERGE_CTRL_16.get_unchecked(n0 + n1) }.clone());

    result.0
}

/// Eight-chunk (64-lane) zeroing left-pack: a three-level merge tree.
#[inline(always)]
pub fn compress_z_merge8<B>(chunks: [Storage<B>; 8], masks: [Storage<B::Mask>; 8]) -> [Storage<B>; 8]
where
    B: Register<Lanes = U8>,
    crate::register::array::ArrayRegister<B, 2>:
        Register<Lanes = U16, Storage = crate::register::array::ArrayRegister<B, 2>>,
    crate::register::array::ArrayRegister<B, 4>:
        Register<Lanes = U32, Storage = crate::register::array::ArrayRegister<B, 4>>,
    crate::register::array::ArrayRegister<B, 8>:
        Register<Lanes = U64, Storage = crate::register::array::ArrayRegister<B, 8>>,
{
    use crate::register::array::ArrayRegister;

    let (c0, n0) = compress_chunk_z::<B>(chunks[0], masks[0]);
    let (c1, n1) = compress_chunk_z::<B>(chunks[1], masks[1]);
    let (c2, n2) = compress_chunk_z::<B>(chunks[2], masks[2]);
    let (c3, n3) = compress_chunk_z::<B>(chunks[3], masks[3]);
    let (c4, n4) = compress_chunk_z::<B>(chunks[4], masks[4]);
    let (c5, n5) = compress_chunk_z::<B>(chunks[5], masks[5]);
    let (c6, n6) = compress_chunk_z::<B>(chunks[6], masks[6]);
    let (c7, _) = compress_chunk_z::<B>(chunks[7], masks[7]);

    // Level 0: four 8+8 -> 16 merges.
    let zero16 = ArrayRegister::<B, 2>([B::EMPTY; 2]);
    let merge16 = |a: Storage<B>, b: Storage<B>, na: usize| -> ArrayRegister<B, 2> {
        <ArrayRegister<B, 2>>::swizzle(
            ArrayRegister::<B, 2>([a, b]),
            zero16,
            unsafe { MERGE_CTRL_8.get_unchecked(na) }.clone(),
        )
    };
    let m01 = merge16(c0, c1, n0);
    let m23 = merge16(c2, c3, n2);
    let m45 = merge16(c4, c5, n4);
    let m67 = merge16(c6, c7, n6);

    // Level 1: two 16+16 -> 32 merges.
    let zero32 = ArrayRegister::<B, 4>([B::EMPTY; 4]);
    let merge32 = |lo: ArrayRegister<B, 2>, hi: ArrayRegister<B, 2>, nlo: usize| -> ArrayRegister<B, 4> {
        let full = ArrayRegister::<B, 4>([lo.0[0], lo.0[1], hi.0[0], hi.0[1]]);
        <ArrayRegister<B, 4>>::swizzle(full, zero32, unsafe { MERGE_CTRL_16.get_unchecked(nlo) }.clone())
    };
    let m0123 = merge32(m01, m23, n0 + n1);
    let m4567 = merge32(m45, m67, n4 + n5);

    // Level 2: 32+32 -> 64, placing m4567 after m0123's selected lanes.
    let full = ArrayRegister::<B, 8>([
        m0123.0[0], m0123.0[1], m0123.0[2], m0123.0[3], m4567.0[0], m4567.0[1], m4567.0[2], m4567.0[3],
    ]);
    let zero64 = ArrayRegister::<B, 8>([B::EMPTY; 8]);
    let result: ArrayRegister<B, 8> = <ArrayRegister<B, 8>>::swizzle(
        full,
        zero64,
        unsafe { MERGE_CTRL_32.get_unchecked(n0 + n1 + n2 + n3) }.clone(),
    );

    result.0
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
        let structured = [
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
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
