use generic_array::{ArrayLength, GenericArray, typenum, typenum::Unsigned};

use super::*;
use crate::sort::SortOrder;
use crate::swizzle::SwizzleIndices;

/// One layer of a sorting network: permute each lane to face its partner, take
/// the pair's first and last under `O`, then keep the first in the low lane of
/// every pair and the last in the high one.
///
/// `I` is the partner permutation (an involution - each lane's partner's partner
/// is itself, and lanes with no partner in this layer map to themselves).
/// `KEEP_MAX` is a lane bitmask: bit `i` set means lane `i` receives the *last*
/// of its pair (the max, ascending), clear means the *first*.
///
/// `O` is what makes the direction free: [`Ascending`](crate::sort::Ascending)
/// resolves `first`/`last` to `min`/`max` and
/// [`Descending`](crate::sort::Descending) to `max`/`min`, so the emitted layer
/// is the same permute + min + max + blend either way, with only the blend
/// operands swapped. See [`crate::sort`] for the measurements.
///
/// Built from [`permutev_const`](Register::permutev_const) and
/// [`blendv`](CoreRegister::blendv), both of which every register has.
/// The earlier version used `PermuteRegister::permute::<IMM8>` +
/// `BlendRegister::blend::<IMM8>`, which stranded the whole file: `BlendRegister`
/// is implemented for three registers, so `sort_2` and `sort_8` had no valid
/// instantiation at all. `permute::<IMM8>`'s bit packing is also per-backend and
/// per-width (1/2/3 bits per lane), which the old immediates got wrong at 8
/// lanes; explicit indices have one meaning everywhere.
#[inline(always)]
fn cmp_merge<R, O, I, const KEEP_MAX: u64>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister,
    O: SortOrder,
    I: SwizzleIndices<R::Lanes>,
{
    let v_shuf = R::permutev_const::<I>(v);
    let v_first = O::first::<R>(v, v_shuf);
    let v_last = O::last::<R>(v, v_shuf);

    // A constant bitmask, so this folds to a materialized mask constant.
    let keep_max = <R::Mask as MaskRegister>::from_native_bitmask(KEEP_MAX);

    R::blendv(keep_max, v_first, v_last)
}

/// Declare a `SwizzleIndices` type for one network layer.
macro_rules! layer_indices {
    ($name:ident, [$($i:expr),* $(,)?]) => {
        struct $name<N: ArrayLength>(core::marker::PhantomData<N>);

        impl<N: ArrayLength> SwizzleIndices<N> for $name<N> {
            const INDICES: GenericArray<u32, N> = const {
                let idxs = [$($i as u32),*];
                assert!(N::USIZE == idxs.len(), "layer width must equal the lane count");
                unsafe { generic_array::const_transmute::<_, GenericArray<u32, N>>(idxs) }
            };
        }
    };
}

// ---------------------------------------------------------------------------
// Bitonic cleanup
//
// A *bitonic* sequence rises then falls (or is a rotation of one). Any bitonic
// sequence of L lanes is sorted by `log2(L)` compare-exchange layers at halving
// strides L/2, L/4, .., 1 - no data-dependent work, the same layers every time.
//
// This is the back half of every bitonic sort, and it is also what a caller
// merging two already-sorted registers needs: reverse one, compare across, then
// clean each side. Factored out so `sort_8` and any cross-register merge share
// one copy of the layers rather than re-deriving the strides.
// ---------------------------------------------------------------------------

/// Sort a **bitonic** 2-lane register ascending. One layer.
///
/// Garbage in, garbage out: the input must already be bitonic. Use [`sort_2`]
/// for arbitrary input.
#[inline(always)]
pub fn bitonic_clean_2<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U2>,
{
    layer_indices!(S1, [1, 0]);

    cmp_merge::<R, O, S1<R::Lanes>, 0b10>(v)
}

/// Sort a **bitonic** 4-lane register ascending. Two layers (strides 2, 1).
///
/// The input must already be bitonic; see [`bitonic_clean_2`].
#[inline(always)]
pub fn bitonic_clean_4<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U4>,
{
    layer_indices!(S2, [2, 3, 0, 1]);
    layer_indices!(S1, [1, 0, 3, 2]);

    let v = cmp_merge::<R, O, S2<R::Lanes>, 0b1100>(v);
    cmp_merge::<R, O, S1<R::Lanes>, 0b1010>(v)
}

/// Sort a **bitonic** 8-lane register ascending. Three layers (strides 4, 2, 1).
///
/// The input must already be bitonic; see [`bitonic_clean_2`].
#[inline(always)]
pub fn bitonic_clean_8<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U8>,
{
    layer_indices!(S4, [4, 5, 6, 7, 0, 1, 2, 3]);
    layer_indices!(S2, [2, 3, 0, 1, 6, 7, 4, 5]);
    layer_indices!(S1, [1, 0, 3, 2, 5, 4, 7, 6]);

    let v = cmp_merge::<R, O, S4<R::Lanes>, 0b1111_0000>(v);
    let v = cmp_merge::<R, O, S2<R::Lanes>, 0b1100_1100>(v);
    cmp_merge::<R, O, S1<R::Lanes>, 0b1010_1010>(v)
}

/// Sort the 2 lanes of a register ascending. One comparator, depth 1.
#[inline(always)]
pub fn sort_2<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U2>,
{
    layer_indices!(L0, [1, 0]); // (0,1)

    cmp_merge::<R, O, L0<R::Lanes>, 0b10>(v)
}

/// Sort the 4 lanes of a register ascending. Three layers, which is the minimum
/// depth for 4 inputs.
///
/// Depth is the cost here, not comparator count: a layer is one permute + min +
/// max + blend whatever it compares, so the two extra comparators a wider layer
/// carries are free. See [`sort_8`] for where that distinction decides things.
#[inline(always)]
pub fn sort_4<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U4>,
{
    layer_indices!(L0, [2, 3, 0, 1]); // (0,2) (1,3)
    layer_indices!(L1, [1, 0, 3, 2]); // (0,1) (2,3)
    layer_indices!(L2, [0, 2, 1, 3]); // (1,2)

    let v = cmp_merge::<R, O, L0<R::Lanes>, 0b1100>(v);
    let v = cmp_merge::<R, O, L1<R::Lanes>, 0b1010>(v);
    cmp_merge::<R, O, L2<R::Lanes>, 0b0100>(v)
}

/// Sort the 8 lanes of a register ascending via a **bitonic** merge: sort each
/// 4-lane half, reverse the upper half so the whole register is bitonic, then
/// merge.
///
/// Six layers, the minimum depth for 8 inputs. It carries 24 comparators where
/// the size-minimal network for 8 inputs carries 19, and that costs nothing:
/// **depth is what a SIMD lane sort pays, size is free.** A layer is one permute
/// + min + max + blend regardless of how many pairs it compares, so both shapes
/// issue exactly 12 `min`/`max`.
///
/// What the extra comparators buy is *regularity*. Every shuffle here but one
/// stays within a 128-bit half, which x86 lowers to an immediate
/// `vshufps`/`vshufpd`/`vpermilps` at 1-cycle latency. Squeezing down to 19
/// comparators forces irregular partner sets - its last two layers pair lanes
/// across the half boundary - and AVX2 has no immediate form for those: two
/// `vpermps`, 3-4 cycles each, each fed by a `.rodata` load. Minimizing
/// comparator count is if anything anti-correlated with cheap machine code.
///
/// llvm-mca, znver3, f32x8 (`crates/sortasm` probe):
///
/// | | uOps | RThroughput | latency |
/// |---|---|---|---|
/// | bitonic (this) | 30 | 9.0 | **43 cyc** |
/// | 19-comparator network | 34 | 9.0 | 52 cyc |
///
/// Throughput ties because both pin the same FP pipes, so the extra uOps are
/// free; the 17% latency gap is the whole difference, and lane-sort callers
/// (a BVH child sort on the pop-to-descend critical path) are latency-bound.
///
/// # Superseded - nothing wires this any more
///
/// `sort_via_network!` has no `(8)` arm; 8-lane registers take the trait
/// default, [`sort_lanes`], which measured **34 cycles against this one's 43**
/// at identical instruction count, uOp count and RThroughput. Both are depth-6
/// bitonic. The difference is the reversal at the end of phase 1 below: here it
/// is a standalone shuffle sitting between the two phases, so the critical path
/// is 7 serial shuffles; `sort_lanes` folds the same reversal into a
/// comparator's partner permutation (`RevPairs<8>`) and needs 6.
///
/// Kept because the reasoning above is still the right reasoning - a
/// size-minimal network still loses, for the reason given - and because
/// `crates/sortasm` and `tests/sort.rs` both compare against it.
/// Re-measure before preferring the other shape on a backend with a cheap
/// cross-lane permute (AVX-512's `vpermt2ps`, SVE).
#[inline(always)]
pub fn sort_8<R, O>(v: Storage<R>) -> Storage<R>
where
    O: SortOrder,
    R: NumericRegister<Lanes = typenum::U8>,
{
    // Per-half patterns: the same 4-lane permutation applied in both halves.
    layer_indices!(H1, [1, 0, 3, 2, 5, 4, 7, 6]); // (0,1) (2,3) in each half
    layer_indices!(H2, [2, 3, 0, 1, 6, 7, 4, 5]); // (0,2) (1,3) in each half
    layer_indices!(H3, [0, 2, 1, 3, 4, 6, 5, 7]); // (1,2) in each half
    layer_indices!(RevHi, [0, 1, 2, 3, 7, 6, 5, 4]); // reverse the upper half

    // Phase 1: sort each half ascending (5 comparators per half).
    let v = cmp_merge::<R, O, H1<R::Lanes>, 0b1010_1010>(v);
    let v = cmp_merge::<R, O, H2<R::Lanes>, 0b1100_1100>(v);
    let v = cmp_merge::<R, O, H3<R::Lanes>, 0b0100_0100>(v);

    // Phase 2: reversing the upper half leaves an ascending run followed by a
    // descending one - a bitonic sequence, which the cleanup layers finish.
    let v = R::permutev_const::<RevHi<R::Lanes>>(v);

    bitonic_clean_8::<R, O>(v)
}

// ---------------------------------------------------------------------------
// Array sorts - one ascending run across the chunks of an ArrayRegister
//
// `[Storage<R>; N]` holds `N * <R::Lanes as Unsigned>::USIZE` elements; these produce a single fully
// sorted run, not N sorted chunks and not sorted columns. The construction is
// the bitonic sort over all `N * LANES` elements, split by comparator distance:
//
// - distance >= LANES: comparators land between *different* chunks, lane i
//   against lane i - a whole-register `R::min`/`R::max` pair. That is the
//   columnar cost model: comparators are what you pay, zero shuffles.
// - distance < LANES: comparators stay *within* one chunk - `R::sort` and
//   `R::bitonic_clean`, the lane cost model, where depth is what you pay.
//
// Nothing here names a lane count: `sort_8`-style networks require `Lanes` as a
// type equality that generic code cannot satisfy, so everything width-specific
// stays behind the two register methods. Chunk counts are literal (2/4/8/16)
// rather than const-generic on purpose - a `for i in 0..N` merge tree has a
// large per-stage body, which is the documented shape that defeats LLVM's
// unroller and forfeits const-folding.
//
// The recursion is plain merge sort: sort each half, then `merge_runs_*` -
// lane-reverse the second half (chunk order flips for free in the array
// literal; `R::reverse` flips lanes) so the concatenation is bitonic, then
// `bitonic_clean_array_*` finishes. The clean is public because it is also
// `ArrayRegister::bitonic_clean`: chunk-stride min/max layers are exactly the
// element-level halving strides down to LANES, after which each chunk is
// bitonic and inter-chunk ordered, and `R::bitonic_clean` finishes each.
// ---------------------------------------------------------------------------

/// Compare-exchange two whole chunks: the low chunk keeps per-lane minima, the
/// high chunk per-lane maxima. One columnar comparator - no shuffles.
macro_rules! ce_chunks {
    ($c:ident: $(($i:tt, $j:tt))+) => {$(
        {
            let lo = O::first::<R>($c[$i], $c[$j]);
            $c[$j] = O::last::<R>($c[$i], $c[$j]);
            $c[$i] = lo;
        }
    )+};
}

/// Finish each listed chunk with its register-level bitonic cleanup.
macro_rules! clean_chunks {
    ($c:ident: $($i:tt)+) => {$(
        $c[$i] = R::bitonic_clean_by::<O>($c[$i]);
    )+};
}

/// Sort a **bitonic** 2-chunk array ascending: one columnar compare-exchange,
/// then clean each chunk.
///
/// Garbage in, garbage out: all `2 * LANES` elements must already form one
/// bitonic sequence. Use [`sort_array_2`] for arbitrary input.
#[inline(always)]
pub fn bitonic_clean_array_2<R: NumericRegister, O: SortOrder>(mut c: [Storage<R>; 2]) -> [Storage<R>; 2] {
    ce_chunks!(c: (0, 1));
    clean_chunks!(c: 0 1);
    c
}

/// Sort a **bitonic** 4-chunk array ascending. Chunk strides 2, 1, then a
/// per-chunk clean. See [`bitonic_clean_array_2`].
#[inline(always)]
pub fn bitonic_clean_array_4<R: NumericRegister, O: SortOrder>(mut c: [Storage<R>; 4]) -> [Storage<R>; 4] {
    ce_chunks!(c: (0, 2)(1, 3));
    ce_chunks!(c: (0, 1)(2, 3));
    clean_chunks!(c: 0 1 2 3);
    c
}

/// Sort a **bitonic** 8-chunk array ascending. Chunk strides 4, 2, 1, then a
/// per-chunk clean. See [`bitonic_clean_array_2`].
#[inline(always)]
pub fn bitonic_clean_array_8<R: NumericRegister, O: SortOrder>(mut c: [Storage<R>; 8]) -> [Storage<R>; 8] {
    ce_chunks!(c: (0, 4)(1, 5)(2, 6)(3, 7));
    ce_chunks!(c: (0, 2)(1, 3)(4, 6)(5, 7));
    ce_chunks!(c: (0, 1)(2, 3)(4, 5)(6, 7));
    clean_chunks!(c: 0 1 2 3 4 5 6 7);
    c
}

/// Sort a **bitonic** 16-chunk array ascending. Chunk strides 8, 4, 2, 1, then
/// a per-chunk clean. See [`bitonic_clean_array_2`].
#[inline(always)]
pub fn bitonic_clean_array_16<R: NumericRegister, O: SortOrder>(mut c: [Storage<R>; 16]) -> [Storage<R>; 16] {
    ce_chunks!(c: (0, 8)(1, 9)(2, 10)(3, 11)(4, 12)(5, 13)(6, 14)(7, 15));
    ce_chunks!(c: (0, 4)(1, 5)(2, 6)(3, 7)(8, 12)(9, 13)(10, 14)(11, 15));
    ce_chunks!(c: (0, 2)(1, 3)(4, 6)(5, 7)(8, 10)(9, 11)(12, 14)(13, 15));
    ce_chunks!(c: (0, 1)(2, 3)(4, 5)(6, 7)(8, 9)(10, 11)(12, 13)(14, 15));
    clean_chunks!(c: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    c
}

/// Merge two sorted 1-chunk runs into one sorted 2-chunk run. Reversing the
/// second run makes the concatenation bitonic; the clean finishes it.
#[inline(always)]
fn merge_runs_1<R: NumericRegister, O: SortOrder>(a: Storage<R>, b: Storage<R>) -> [Storage<R>; 2] {
    bitonic_clean_array_2::<R, O>([a, R::reverse(b)])
}

/// Merge two sorted 2-chunk runs into one sorted 4-chunk run. The chunk-order
/// flip of the second run is free (array literal); only the lane reversals cost.
#[inline(always)]
fn merge_runs_2<R: NumericRegister, O: SortOrder>(a: [Storage<R>; 2], b: [Storage<R>; 2]) -> [Storage<R>; 4] {
    bitonic_clean_array_4::<R, O>([a[0], a[1], R::reverse(b[1]), R::reverse(b[0])])
}

/// Merge two sorted 4-chunk runs into one sorted 8-chunk run.
#[inline(always)]
fn merge_runs_4<R: NumericRegister, O: SortOrder>(a: [Storage<R>; 4], b: [Storage<R>; 4]) -> [Storage<R>; 8] {
    bitonic_clean_array_8::<R, O>([
        a[0],
        a[1],
        a[2],
        a[3],
        R::reverse(b[3]),
        R::reverse(b[2]),
        R::reverse(b[1]),
        R::reverse(b[0]),
    ])
}

/// Merge two sorted 8-chunk runs into one sorted 16-chunk run.
#[inline(always)]
fn merge_runs_8<R: NumericRegister, O: SortOrder>(a: [Storage<R>; 8], b: [Storage<R>; 8]) -> [Storage<R>; 16] {
    bitonic_clean_array_16::<R, O>([
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        R::reverse(b[7]),
        R::reverse(b[6]),
        R::reverse(b[5]),
        R::reverse(b[4]),
        R::reverse(b[3]),
        R::reverse(b[2]),
        R::reverse(b[1]),
        R::reverse(b[0]),
    ])
}

// ---------------------------------------------------------------------------
// The fast array sort: columnar network + Blacher's widening merge
//
// The bitonic tree above is at the *bitonic* floor - no comparator in it is
// wasted - and is still the slower construction, because the bitonic floor is
// not the achievable floor. This is Highway vqsort's decomposition
// (`hwy/contrib/sort/sorting_networks-inl.h`), which is not bitonic:
//
//   1. sort the chunks *columnwise* with a size-minimal network - whole-chunk
//      `first`/`last`, zero shuffles, the free axis
//   2. widen the sorted column groups 2 -> 4 -> 8 -> .. -> LANES, after which
//      the block is one sorted run read chunk-major
//
// The point is where the work happens. The bitonic tree runs a full LANES-lane
// network *inside* every chunk at every level, and an in-chunk layer costs a
// permute + first + last + blend for LANES/2 comparators. Here the only
// in-chunk work is `log2` tail layers per widening stage. Comparators migrate
// from the expensive axis to the free one.
//
// Measured, `crates/sortasm` probes + llvm-mca 22.1.8, znver3, `f32` chunks.
// `-iterations=1` Total Cycles for latency, 10000 for RThroughput:
//
// | chunks | construction | insns | min/max | shuf | blend | RThr | latency |
// |---|---|---|---|---|---|---|---|
// | 2 (`f32x16`) | bitonic | 89 | 38 | 23 | 17 | 27.5 | 68 |
// | 2 | **this** | 70 | 32 | 20 | 10 | **21.0** | **55** |
// | 4 (`f32x32`) | bitonic | 243 | 108 | 64 | 44 | 76.0 | 136 |
// | 4 | **this** | 166 | 82 | 50 | 20 | **51.0** | **87** |
// | 16 (`f32x128`) | bitonic | 1581 | 736 | 400 | 256 | 496.0 | 752 |
// | 16 | **this** | 1027 | 504 | 272 | 80 | **292.0** | **378** |
//
// **Latency was the number that had to be checked, not throughput.** Depth, not
// size, is what a lane sort pays (see `crate::sort` and `sort_8`), and the
// register-layer caller this exists for is a BVH child sort on a pop-to-descend
// dependency chain - a construction that won on op count and lost on depth
// would have been the wrong trade, exactly as the size-minimal 8-lane network
// was. It wins on both, and the latency margin *grows* with the chunk count
// (1.24x at N=2, 1.99x at N=16): the per-chunk tails belong to independent
// chunks and overlap, while the bitonic tree serializes a full LANES-lane
// network per chunk per level.
//
// Bottleneck attribution at N=16 says the two are limited by different things:
//
// | | resource pressure | data dependencies |
// |---|---|---|
// | bitonic | 63.8% (FP0 57.7 / FP1 57.4) | 45.8% |
// | this | **99.4%** (FP0 77.0 / FP1 93.8) | 35.7% |
//
// So the bitonic tree spent nearly half its time waiting on its own critical
// path, and this one is essentially pure issue-limited work on FP0/FP1 - which
// means further op reduction here converts to time roughly one for one, and
// FP1 (shared by `min`/`max` and the shuffles) is the pipe to attack.
//
// Covers `LANES <= MERGE_MAX_LANES`; `sort_array_*` keeps the bitonic tree for
// anything wider.
//
// NOTE this is a second transcription of the network `thermite_sort::merge`
// runs at the vector layer. They cannot share code - `NumericVector` exposes no
// associated register type - but they do share the index math, in `crate::sort`.
// ---------------------------------------------------------------------------

/// Widest chunk the widening stages here cover, in lanes.
const MERGE_MAX_LANES: usize = 16;

/// Run one in-chunk compare-exchange layer: permute each lane to face its
/// partner, take both sides of every pair, blend by a constant lane mask.
///
/// The same shape as [`cmp_merge`], but with the partner set and the blend mask
/// supplied together by a [`PairStage`](crate::sort::PairStage) type rather than
/// separately. That is what lets `RevPairs<K>` derive its mask from `K / 2`, a
/// const expression, where a const-generic expression would not be stable.
#[inline(always)]
fn pair_stage<R, O, S>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister,
    O: SortOrder,
    S: crate::sort::PairStage<R::Lanes>,
{
    let partner = R::permutev_const::<S::Indices>(v);
    let first = O::first::<R>(v, partner);
    let last = O::last::<R>(v, partner);

    // A constant bitmask, so this folds to a materialized mask constant.
    let keep_last = <R::Mask as MaskRegister>::from_native_bitmask(S::KEEP_LAST);

    R::blendv(keep_last, first, last)
}

/// Sort the chunks *columnwise*: lane `i` of every chunk is an independent
/// column.
///
/// A comparator here is one `first` + one `last` on whole chunks - no permute,
/// no blend, no cross-lane traffic - so the size-minimal networks from the
/// standard tables genuinely are optimal, which is the opposite of the lane-sort
/// case. Same networks as `thermite_sort::columns`.
macro_rules! columnar_chunks {
    ($name:ident, $n:literal; $([$(($a:literal, $b:literal)),* $(,)?])*) => {
        #[inline(always)]
        fn $name<R: NumericRegister, O: SortOrder>(mut c: [Storage<R>; $n]) -> [Storage<R>; $n] {
            $($(
                {
                    const _: () = assert!($a < $b, "comparator must be written (lo, hi)");
                    const _: () = assert!($b < $n, "comparator index out of range");

                    let lo = O::first::<R>(c[$a], c[$b]);
                    c[$b] = O::last::<R>(c[$a], c[$b]);
                    c[$a] = lo;
                }
            )*)*
            c
        }
    };
}

columnar_chunks! {
    columns_2, 2;
    [(0,1)]
}

columnar_chunks! {
    columns_4, 4;
    [(0,2),(1,3)]
    [(0,1),(2,3)]
    [(1,2)]
}

columnar_chunks! {
    columns_8, 8;
    [(0,2),(1,3),(4,6),(5,7)]
    [(0,4),(1,5),(2,6),(3,7)]
    [(0,1),(2,3),(4,5),(6,7)]
    [(2,4),(3,5)]
    [(1,4),(3,6)]
    [(1,2),(3,4),(5,6)]
}

columnar_chunks! {
    columns_16, 16;
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

/// Finish one chunk of a `C`-wide widening stage: `SortPairsReverse` at `C`,
/// then halving `SortPairsDistance` from `C/4` down to 1.
macro_rules! tail_fn {
    ($name:ident, $c:literal, [$($d:literal),*]) => {
        #[inline(always)]
        fn $name<R: NumericRegister, O: SortOrder>(v: Storage<R>) -> Storage<R> {
            let v = pair_stage::<R, O, crate::sort::RevPairs<$c>>(v);
            $( let v = pair_stage::<R, O, crate::sort::Distance<$d>>(v); )*
            v
        }
    };
}

// `tail_c2` is the reverse alone: reversing a group of 2 *is* the distance-1
// stage, so there is nothing left to halve.
tail_fn!(tail_c2, 2, []);
tail_fn!(tail_c4, 4, [1]);
tail_fn!(tail_c8, 8, [2, 1]);
tail_fn!(tail_c16, 16, [4, 2, 1]);

// ---------------------------------------------------------------------------
// Lane sorts from the tails alone
//
// The widening construction at ONE chunk degenerates into a complete lane sort:
// `log2(1) == 0` chunk stages, so only the tails run, and the chain
// `tail_c2 -> tail_c4 -> .. -> tail_cLANES` takes single elements to sorted
// groups of 2, then 4, then 8, then the whole register.
//
// That is Batcher's bitonic sort with the "reverse the second half" shuffle
// *fused into the partner permutation* rather than issued separately -
// `RevPairs<C>` pairs lane `i` with its mirror in its group of `C`, which is
// what comparing a run against its reversed neighbour amounts to. So it is
// depth-minimal at every power-of-two width (1+2+3+4 = 10 layers at 16 lanes)
// and spends exactly one shuffle per layer.
//
// This is what makes it a usable trait *default*: it needs no `Lanes` type
// equality, so unlike `sort_8` it can be called from code generic over
// `R: NumericRegister`, and a register with no `sort_via_network!` override
// gets a real network instead of the quadratic `sort_any`.
// ---------------------------------------------------------------------------

/// Sort the lanes of any register of at most `MERGE_MAX_LANES` lanes; wider
/// registers fall through to [`sort_any`].
///
/// The default body of [`NumericRegister::sort_by`]. A register whose lane count
/// has a hand-tuned network overrides it via `sort_via_network!`; the ones that
/// do not - every 16-lane register - used to get a scalar insertion sort with a
/// memory round-trip and now get 10 branchless layers.
#[inline(always)]
pub fn sort_lanes<R: NumericRegister, O: SortOrder>(v: Storage<R>) -> Storage<R> {
    if const { <R::Lanes as Unsigned>::USIZE > MERGE_MAX_LANES } {
        return sort_any::<R, O>(v);
    }

    let v = if const { <R::Lanes as Unsigned>::USIZE >= 2 } { tail_c2::<R, O>(v) } else { v };
    let v = if const { <R::Lanes as Unsigned>::USIZE >= 4 } { tail_c4::<R, O>(v) } else { v };
    let v = if const { <R::Lanes as Unsigned>::USIZE >= 8 } { tail_c8::<R, O>(v) } else { v };

    if const { <R::Lanes as Unsigned>::USIZE >= 16 } { tail_c16::<R, O>(v) } else { v }
}

/// Sort the lanes of a **bitonic** register of at most `MERGE_MAX_LANES`
/// lanes: the halving compare-exchange strides `LANES/2 .. 1`.
///
/// The default body of [`NumericRegister::bitonic_clean_by`]. Garbage in,
/// garbage out - see that method. Wider registers fall through to [`sort_any`],
/// which is a full sort and therefore happens to be correct for any input.
#[inline(always)]
pub fn bitonic_clean_lanes<R: NumericRegister, O: SortOrder>(v: Storage<R>) -> Storage<R> {
    if const { <R::Lanes as Unsigned>::USIZE > MERGE_MAX_LANES } {
        return sort_any::<R, O>(v);
    }

    let v = if const { <R::Lanes as Unsigned>::USIZE >= 16 } {
        pair_stage::<R, O, crate::sort::Distance<8>>(v)
    } else {
        v
    };
    let v = if const { <R::Lanes as Unsigned>::USIZE >= 8 } {
        pair_stage::<R, O, crate::sort::Distance<4>>(v)
    } else {
        v
    };
    let v = if const { <R::Lanes as Unsigned>::USIZE >= 4 } {
        pair_stage::<R, O, crate::sort::Distance<2>>(v)
    } else {
        v
    };

    if const { <R::Lanes as Unsigned>::USIZE >= 2 } {
        pair_stage::<R, O, crate::sort::Distance<1>>(v)
    } else {
        v
    }
}

/// Widen a block of `$n` chunks from `C/2`-wide to `C`-wide sorted column
/// groups: `log2(n)` chunk stages, then one tail per chunk.
///
/// A chunk stage lane-reverses the upper chunk of every pair within its groups
/// of `C`, then compare-exchanges the pair as whole chunks. The reversal is the
/// only shuffle in a stage; the compare-exchange is columnar.
macro_rules! merge_fn {
    (
        $name:ident, $n:literal, $c:literal, $tail:ident,
        [ $( [ $( ($lo:tt, $hi:tt) )+ ] )+ ],
        [ $($chunk:tt)+ ]
    ) => {
        #[inline(always)]
        fn $name<R: NumericRegister, O: SortOrder>(c: &mut [Storage<R>; $n]) {
            $(
                $( c[$hi] = R::permutev_const::<crate::sort::RevIdx<$c, R::Lanes>>(c[$hi]); )+
                $(
                    {
                        let lo = O::first::<R>(c[$lo], c[$hi]);
                        c[$hi] = O::last::<R>(c[$lo], c[$hi]);
                        c[$lo] = lo;
                    }
                )+
            )+

            $( c[$chunk] = $tail::<R, O>(c[$chunk]); )+
        }
    };
}

/// Emit the four column widths for one chunk count from a single stage list.
macro_rules! merge_group {
    (
        $n:literal, [ $($chunk:tt)+ ], $f2:ident, $f4:ident, $f8:ident, $f16:ident,
        $($stage:tt)+
    ) => {
        merge_fn!($f2,  $n, 2,  tail_c2,  [$($stage)+], [$($chunk)+]);
        merge_fn!($f4,  $n, 4,  tail_c4,  [$($stage)+], [$($chunk)+]);
        merge_fn!($f8,  $n, 8,  tail_c8,  [$($stage)+], [$($chunk)+]);
        merge_fn!($f16, $n, 16, tail_c16, [$($stage)+], [$($chunk)+]);
    };
}

merge_group! {
    2, [0 1], merge_2_c2, merge_2_c4, merge_2_c8, merge_2_c16,
    [(0,1)]
}

merge_group! {
    4, [0 1 2 3], merge_4_c2, merge_4_c4, merge_4_c8, merge_4_c16,
    [(0,3)(1,2)]
    [(0,1)(2,3)]
}

merge_group! {
    8, [0 1 2 3 4 5 6 7], merge_8_c2, merge_8_c4, merge_8_c8, merge_8_c16,
    [(0,7)(1,6)(2,5)(3,4)]
    [(0,3)(1,2)(4,7)(5,6)]
    [(0,1)(2,3)(4,5)(6,7)]
}

merge_group! {
    16, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15],
    merge_16_c2, merge_16_c4, merge_16_c8, merge_16_c16,
    [(0,15)(1,14)(2,13)(3,12)(4,11)(5,10)(6,9)(7,8)]
    [(0,7)(1,6)(2,5)(3,4)(8,15)(9,14)(10,13)(11,12)]
    [(0,3)(1,2)(4,7)(5,6)(8,11)(9,10)(12,15)(13,14)]
    [(0,1)(2,3)(4,5)(6,7)(8,9)(10,11)(12,13)(14,15)]
}

/// Column-sort the chunks, then widen the column groups until they span a
/// chunk. The ladder is on an associated const, so every arm a given `R` cannot
/// reach folds away at monomorphization.
macro_rules! merged_array_fn {
    ($name:ident, $n:literal, $cols:ident, $f2:ident, $f4:ident, $f8:ident, $f16:ident) => {
        #[inline(always)]
        fn $name<R: NumericRegister, O: SortOrder>(c: [Storage<R>; $n]) -> [Storage<R>; $n] {
            let mut c = $cols::<R, O>(c);

            if const { <R::Lanes as Unsigned>::USIZE >= 2 } {
                $f2::<R, O>(&mut c);

                if const { <R::Lanes as Unsigned>::USIZE >= 4 } {
                    $f4::<R, O>(&mut c);

                    if const { <R::Lanes as Unsigned>::USIZE >= 8 } {
                        $f8::<R, O>(&mut c);

                        if const { <R::Lanes as Unsigned>::USIZE >= 16 } {
                            $f16::<R, O>(&mut c);
                        }
                    }
                }
            }

            c
        }
    };
}

merged_array_fn!(merged_array_2, 2, columns_2, merge_2_c2, merge_2_c4, merge_2_c8, merge_2_c16);
merged_array_fn!(merged_array_4, 4, columns_4, merge_4_c2, merge_4_c4, merge_4_c8, merge_4_c16);
merged_array_fn!(merged_array_8, 8, columns_8, merge_8_c2, merge_8_c4, merge_8_c8, merge_8_c16);
merged_array_fn!(
    merged_array_16, 16, columns_16, merge_16_c2, merge_16_c4, merge_16_c8, merge_16_c16
);

/// The bitonic 2-chunk sort.
///
/// What [`sort_array_2`] runs for chunks wider than `MERGE_MAX_LANES`, and kept
/// public so an asm probe can measure it against the merged construction at a
/// width where it is *not* what runs.
#[inline(always)]
pub fn bitonic_array_2<R: NumericRegister, O: SortOrder>(c: [Storage<R>; 2]) -> [Storage<R>; 2] {
    merge_runs_1::<R, O>(R::sort_by::<O>(c[0]), R::sort_by::<O>(c[1]))
}

/// The bitonic 4-chunk sort. See [`sort_array_2`].
#[inline(always)]
pub fn bitonic_array_4<R: NumericRegister, O: SortOrder>(c: [Storage<R>; 4]) -> [Storage<R>; 4] {
    merge_runs_2::<R, O>(bitonic_array_2::<R, O>([c[0], c[1]]), bitonic_array_2::<R, O>([c[2], c[3]]))
}

/// The bitonic 8-chunk sort. See [`sort_array_2`].
#[inline(always)]
pub fn bitonic_array_8<R: NumericRegister, O: SortOrder>(c: [Storage<R>; 8]) -> [Storage<R>; 8] {
    merge_runs_4::<R, O>(
        bitonic_array_4::<R, O>([c[0], c[1], c[2], c[3]]),
        bitonic_array_4::<R, O>([c[4], c[5], c[6], c[7]]),
    )
}

/// The bitonic 16-chunk sort. See [`sort_array_2`].
#[inline(always)]
pub fn bitonic_array_16<R: NumericRegister, O: SortOrder>(c: [Storage<R>; 16]) -> [Storage<R>; 16] {
    merge_runs_8::<R, O>(
        bitonic_array_8::<R, O>([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
        bitonic_array_8::<R, O>([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]),
    )
}

// ---------------------------------------------------------------------------
// The array sorts, and which construction each one runs
//
// `merged_array_*` (below) covers chunks of at most `MERGE_MAX_LANES` lanes and
// is the faster one; wider chunks keep the bitonic tree. The test is on an
// associated const, so the arm a given `R` cannot reach folds away at
// monomorphization and only one network is emitted per instantiation.
// ---------------------------------------------------------------------------

macro_rules! array_sort_fn {
    ($(#[$attr:meta])* $name:ident, $n:literal, $fast:ident, $slow:ident) => {
        $(#[$attr])*
        #[inline(always)]
        pub fn $name<R: NumericRegister, O: SortOrder>(c: [Storage<R>; $n]) -> [Storage<R>; $n] {
            if const { <R::Lanes as Unsigned>::USIZE <= MERGE_MAX_LANES } {
                $fast::<R, O>(c)
            } else {
                $slow::<R, O>(c)
            }
        }
    };
}

array_sort_fn!(
    /// Sort all `2 * LANES` elements of a 2-chunk array into one ascending run.
    ///
    /// Two constructions live behind this: a columnar network plus Blacher's
    /// widening merge for chunks of at most 16 lanes, and the bitonic merge tree
    /// for anything wider. See the comment above `pair_stage` for why the first
    /// wins.
    sort_array_2, 2, merged_array_2, bitonic_array_2
);
array_sort_fn!(
    /// Sort all `4 * LANES` elements of a 4-chunk array into one ascending run.
    sort_array_4, 4, merged_array_4, bitonic_array_4
);
array_sort_fn!(
    /// Sort all `8 * LANES` elements of an 8-chunk array into one ascending run.
    sort_array_8, 8, merged_array_8, bitonic_array_8
);
array_sort_fn!(
    /// Sort all `16 * LANES` elements of a 16-chunk array into one ascending run.
    ///
    /// 16 live chunks is the whole AVX2 register file before temporaries; expect
    /// spills at 256-bit chunks. Still far ahead of the quadratic scalar
    /// fallback.
    sort_array_16, 16, merged_array_16, bitonic_array_16
);

/// The portable fallback: a scalar compare-and-swap walk over the element
/// slice, for registers with no network at their lane count.
///
/// Correct for every width and branchless, but quadratic and it round-trips
/// through memory. Any register that grows a network should be wired up in
/// `sort_via_network!` instead.
///
/// Sorts ascending and reverses when `O` is descending, rather than threading a
/// scalar comparator through [`SortOrder`] for the sake of the slow path - see
/// [`SortOrder::IS_ASCENDING`]. One permute against a quadratic body is noise.
#[inline(always)]
pub fn sort_any<R: NumericRegister, O: SortOrder>(mut value: Storage<R>) -> Storage<R> {
    let s = R::as_mut_slice(&mut value);

    /// Compare-and-Swap: The atomic primitive of sorting networks.
    /// LLVM optimizes this to `cmp` + `cmov` (Conditional Move), which is branchless.
    #[inline(always)]
    fn cas<T: PartialOrd>(s: &mut [T], i: usize, j: usize) {
        // Note: slice indexing checks bounds.
        // For maximal performance, you could use `get_unchecked` if unsafe is permitted,
        // but the optimizer often elides checks in fixed-size networks anyway.
        if s[i] > s[j] {
            s.swap(i, j);
        }
    }

    #[rustfmt::skip]
    let () = match s.len() {
        2 => cas(s, 0, 1),
        4 => {
            cas(s, 0, 1); cas(s, 2, 3); // Layer 1
            cas(s, 0, 2); cas(s, 1, 3); // Layer 2
            cas(s, 1, 2);               // Layer 3
        },
        // For N=8 or others, Insertion Sort is compact and very fast for N < 20
        _ => {
            for i in 1..s.len() {
                let mut j = i;
                // The compiler unrolls this loop well for small fixed bounds
                while j > 0 && s[j - 1] > s[j] {
                    s.swap(j - 1, j);
                    j -= 1;
                }
            }
        }
    };

    if const { !O::IS_ASCENDING } {
        return R::reverse(value);
    }

    value
}
