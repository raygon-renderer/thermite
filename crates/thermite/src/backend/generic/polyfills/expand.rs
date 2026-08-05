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
//! - [`expand_permute_wide`] - any multiple of 8 lanes up to 64: per-8-lane
//!   table lookups assembled into one global gather index + one full-width
//!   permute.
//!
//! There is no analogue of the `compress_z_merge*` chunk trees. Expand *splits*
//! a packed run across chunks rather than merging per-chunk results, so the
//! construction is not symmetric, and `expand_z` composes its `zz` after the
//! plain permute, which is already cheap. Revisit only with a benchmark in hand.

use generic_array::{
    GenericArray,
    typenum::{U256, Unsigned},
};

use super::*;

// Named explicitly, though the glob above would supply them, to keep the
// dependency on the compress side visible at the top of the file.
use super::compress::{CompressRow, CompressTable};

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
    R: WidenIndexRegister<Lanes: CompressTable>,
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
pub unsafe fn expand_permute8_raw<R: WidenIndexRegister>(value: Storage<R>, mask: Storage<R::Mask>) -> Storage<R> {
    // SAFETY: identical argument to `compress_permute8_raw`: <= 8 lanes makes
    // `native_bitmask` always `Some` and `bm < 256` indexes the table directly.
    let bm = unsafe { <R::Mask as MaskRegister>::native_bitmask(mask).unwrap_unchecked() } as usize;

    // SAFETY: `bm < 256`; truncating to the first `LANES` indices is sound per
    // the padding-lane argument in the `EXPAND8` docs (real lanes' indices never
    // reference padding positions).
    R::permutev_row(value, &unsafe { EXPAND8.get_unchecked(bm) }.0)
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

    // Second pass: per-output-lane global source indices per the formula above.
    let mut g: GenericArray<u32, R::Lanes> = GenericArray::default();
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
            unsafe { *g.get_unchecked_mut(out_base + j) = src as u32 };
        }

        base += cnt;
    }

    R::permutev(value, g)
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
        ArrayRegister<i32, N>: WidenIndexRegister<Element = i32, Lanes: CompressTable>,
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
