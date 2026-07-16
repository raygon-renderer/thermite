//! Semi-generic square AVX2 transposes, shared across the whole 256-bit register
//! family (`f32x8`, `f64x4`, `i32x8`, `u32x8`, `i64x4`, `i16x16`, `i8x32`, ...).
//!
//! A square `(de)interleave_radix_by::<N, GROUP>` (`N * GROUP == LANES`) on a 256-bit
//! register is an `N x N` transpose of `W = GROUP * sizeof(element)`-wide elements.
//! **It depends only on `W`, not on whether the lanes are float or int** - the shuffle
//! instructions act on bit patterns, and every 256-bit register has the same two-128-bit-
//! half sublane structure. So there is exactly one optimal sequence per width
//! `W in {4, 8, ...} bytes` (`N in {8, 4, ...}`), written once here and reused by every
//! register via a free `cast*` into the ps/pd domain the sequence is spelled in.
//!
//! This is why the per-register natives collapse: `f32x8` `(8,1)`, `i32x8`/`u32x8` `(8,1)`,
//! `i16x16` `(8,2)`, `i8x32` `(8,4)` are ALL the `W = 4` (8x8, 32-bit) transpose; `f32x8`
//! `(4,2)`, `f64x4`/`i64x4`/`u64x4` `(4,1)`, `i32x8` `(4,2)` are ALL the `W = 8` (4x4,
//! 64-bit) transpose. Each register's arm bit-casts into the shared body (the cast is a
//! no-op - `castps_si256`/`castsi256_ps`/... emit nothing).
//!
//! Each transpose is its own inverse, so `interleave_radix_by` reuses the same body. All
//! `#[inline(always)]` to lower inside the caller's `target_feature` context.

use super::*;

/// The `W = 4` bytes (32-bit element) square transpose: `8 x 8`. Serves any 256-bit
/// register whose effective element width is 32 bits (`f32x8`/`i32x8`/`u32x8` at
/// `GROUP=1`, `i16x16` at `GROUP=2`, `i8x32` at `GROUP=4`). Spelled in the ps domain
/// (`shuffle_ps` is a two-source within-128 shuffle; int callers `castsi256_ps` in,
/// `castps_si256` out - both free). 8 `unpck` + 8 `shuffle` (within-128) + 8
/// `permute2f128` (the only lane crossings). `out[j].lane(l) == in[l].lane(j)`.
#[inline(always)]
pub unsafe fn transpose256_w32(i: [__m256; 8]) -> [__m256; 8] {
    let t0 = _mm256_unpacklo_ps(i[0], i[1]);
    let t1 = _mm256_unpackhi_ps(i[0], i[1]);
    let t2 = _mm256_unpacklo_ps(i[2], i[3]);
    let t3 = _mm256_unpackhi_ps(i[2], i[3]);
    let t4 = _mm256_unpacklo_ps(i[4], i[5]);
    let t5 = _mm256_unpackhi_ps(i[4], i[5]);
    let t6 = _mm256_unpacklo_ps(i[6], i[7]);
    let t7 = _mm256_unpackhi_ps(i[6], i[7]);
    let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    [
        _mm256_permute2f128_ps::<0x20>(s0, s4),
        _mm256_permute2f128_ps::<0x20>(s1, s5),
        _mm256_permute2f128_ps::<0x20>(s2, s6),
        _mm256_permute2f128_ps::<0x20>(s3, s7),
        _mm256_permute2f128_ps::<0x31>(s0, s4),
        _mm256_permute2f128_ps::<0x31>(s1, s5),
        _mm256_permute2f128_ps::<0x31>(s2, s6),
        _mm256_permute2f128_ps::<0x31>(s3, s7),
    ]
}

/// The `W = 8` bytes (64-bit element) square transpose: `4 x 4`. Serves any 256-bit
/// register whose effective element width is 64 bits (`f64x4`/`i64x4`/`u64x4` at
/// `GROUP=1`, `f32x8`/`i32x8` at `GROUP=2`, `i16x16` at `GROUP=4`). Spelled in the pd
/// domain (`unpacklo/hi_pd` are within-128 64-bit unpacks); callers cast in/out free.
/// 4 `unpck_pd` + 4 `permute2f128` = the 8-op AVX2-optimal 4x4, its own inverse.
#[inline(always)]
pub unsafe fn transpose256_w64(i: [__m256d; 4]) -> [__m256d; 4] {
    let t0 = _mm256_unpacklo_pd(i[0], i[1]);
    let t1 = _mm256_unpackhi_pd(i[0], i[1]);
    let t2 = _mm256_unpacklo_pd(i[2], i[3]);
    let t3 = _mm256_unpackhi_pd(i[2], i[3]);
    [
        _mm256_permute2f128_pd(t0, t2, 0x20),
        _mm256_permute2f128_pd(t1, t3, 0x20),
        _mm256_permute2f128_pd(t0, t2, 0x31),
        _mm256_permute2f128_pd(t1, t3, 0x31),
    ]
}

/// `deinterleave_radix_by`/`interleave_radix_by` adapter for **any 256-bit integer
/// register** (`Storage == __m256i`) whose effective element width `GROUP * sizeof(elem)`
/// is 32 bits (so `N == 8`): the 8x8 [`transpose256_w32`] via free `castsi256_ps` /
/// `castps_si256`. Self-inverse, so both directions call this. The caller's `if const`
/// guard guarantees `N == 8`.
#[inline(always)]
pub unsafe fn radix_by_w32_si<const N: usize>(inputs: [__m256i; N]) -> [__m256i; N] {
    // SAFETY: caller's const guard ensures N == 8, so indices 0..8 are in bounds.
    let f = transpose256_w32([
        _mm256_castsi256_ps(*inputs.get_unchecked(0)),
        _mm256_castsi256_ps(*inputs.get_unchecked(1)),
        _mm256_castsi256_ps(*inputs.get_unchecked(2)),
        _mm256_castsi256_ps(*inputs.get_unchecked(3)),
        _mm256_castsi256_ps(*inputs.get_unchecked(4)),
        _mm256_castsi256_ps(*inputs.get_unchecked(5)),
        _mm256_castsi256_ps(*inputs.get_unchecked(6)),
        _mm256_castsi256_ps(*inputs.get_unchecked(7)),
    ]);
    let mut out = inputs;
    let mut k = 0;
    while k < 8 {
        *out.get_unchecked_mut(k) = _mm256_castps_si256(f[k]);
        k += 1;
    }
    out
}

/// As [`radix_by_w32_si`] for the `W = 8` bytes (64-bit effective, `N == 4`) case: the
/// 4x4 [`transpose256_w64`] via `castsi256_pd` / `castpd_si256`. Caller guarantees `N == 4`.
#[inline(always)]
pub unsafe fn radix_by_w64_si<const N: usize>(inputs: [__m256i; N]) -> [__m256i; N] {
    // SAFETY: caller's const guard ensures N == 4, so indices 0..4 are in bounds.
    let d = transpose256_w64([
        _mm256_castsi256_pd(*inputs.get_unchecked(0)),
        _mm256_castsi256_pd(*inputs.get_unchecked(1)),
        _mm256_castsi256_pd(*inputs.get_unchecked(2)),
        _mm256_castsi256_pd(*inputs.get_unchecked(3)),
    ]);
    let mut out = inputs;
    let mut k = 0;
    while k < 4 {
        *out.get_unchecked_mut(k) = _mm256_castpd_si256(d[k]);
        k += 1;
    }
    out
}

// ===========================================================================
// The certified ladder engine: near-optimal `(de)interleave_radix_by` for ANY
// power-of-two `(N, GROUP)` shape on the 256-bit family.
// ===========================================================================
//
// Every pow-2 stride permutation is a rotation of the flat index bits, and the
// optimal AVX2 realization is Eklundh-style: `min(log2 N, log2 M)` two-source
// rounds of `N` ops each (`M` = groups per register), where each round is a
// fixed-width block combine - `perm2f128` when the round crosses the 128-bit
// halves, bare `unpck`/`shufps` otherwise. The square natives above
// (`transpose256_w32`/`w64`) are the `N == M` special case of this scheme.
//
// The general rectangular case differs only in WHICH rounds run and where the
// outputs land (a register permutation - free, like `digit_reversal`). Hand-
// deriving that per shape is exactly the whack-a-mole this engine kills:
// instead, a compile-time SEARCH simulates a small family of candidate plans
// on group indices and emits the first plan whose result is exactly a register
// relabel of the target - a correctness certificate. No certificate, no
// ladder: the caller falls back to the portable engine. A certified plan is
// correct by construction, and every branch below folds because the plan comes
// from an inline `const` block at the call site.
//
// One ps-domain core serves the whole family: pd/int registers bit-cast in and
// out for free, declaring their group width in 32-bit slots (f64 GROUP=1 -> 2
// slots; i16 GROUP=2 -> 1 slot). Widths below 32 bits would need epi16/epi8
// rungs - not implemented; those shapes simply never certify.

/// Cap on `N` for the ladder engine (relabel tables and simulation grids are
/// sized by it). Shapes with more registers fall back.
pub const LADDER_MAX_N: usize = 32;

/// A compile-time ladder plan: rung + style per round, then a register relabel.
///
/// Rungs (all 1 op per output register): `0` = w32 interleave flavor
/// (`unpcklo/hi_ps`), `1` = w32 deinterleave flavor (`shufps 0x88/0xDD`),
/// `2` = w64 (`shufps 0x44/0xEE`), `3` = w128 (`perm2f128 0x20/0x31`).
/// Style `0` pairs adjacent registers `(2i, 2i+1)` and places outputs split
/// `(i, i + B/2)` within the block; style `1` pairs at distance `(i, i + B/2)`
/// and places outputs adjacent `(2i, 2i+1)`.
///
/// `relabel[j]` is the post-rounds buffer slot holding output register `j` of
/// the DEINTERLEAVE direction. The INTERLEAVE direction is the exact mirror:
/// scatter through the inverse relabel, then the rounds reversed with the
/// style flipped and the w32 flavor flipped (`unpck` and `shufps 0x88/0xDD`
/// are mutual inverses; the w64/w128 rungs are their own inverses).
#[derive(Debug, Clone, Copy)]
pub struct LadderPlan {
    pub ok: bool,
    pub rounds: usize,
    pub rung: [u8; 6],
    pub style: [u8; 6],
    pub relabel: [u8; LADDER_MAX_N],
}

const LADDER_NONE: LadderPlan = LadderPlan {
    ok: false,
    rounds: 0,
    rung: [0; 6],
    style: [0; 6],
    relabel: [0; LADDER_MAX_N],
};

/// Exact index-level simulation of one rung on a register pair. `x`/`y` hold
/// flat 32-bit-slot ids; returns the two output id rows. Mirrors the intrinsic
/// semantics EXACTLY (verify against the Intel pseudocode, not intuition).
const fn sim_rung(rung: u8, x: [u16; 8], y: [u16; 8]) -> ([u16; 8], [u16; 8]) {
    match rung {
        // unpcklo_ps / unpckhi_ps: per 128 half, interleave the low (resp.
        // high) two elements of each source.
        0 => (
            [x[0], y[0], x[1], y[1], x[4], y[4], x[5], y[5]],
            [x[2], y[2], x[3], y[3], x[6], y[6], x[7], y[7]],
        ),
        // shufps 0x88 / 0xDD: per 128 half, even (resp. odd) elements of x
        // then of y.
        1 => (
            [x[0], x[2], y[0], y[2], x[4], x[6], y[4], y[6]],
            [x[1], x[3], y[1], y[3], x[5], x[7], y[5], y[7]],
        ),
        // shufps 0x44 / 0xEE: per 128 half, the low (resp. high) 64-bit pair
        // of x then of y.
        2 => (
            [x[0], x[1], y[0], y[1], x[4], x[5], y[4], y[5]],
            [x[2], x[3], y[2], y[3], x[6], x[7], y[6], y[7]],
        ),
        // perm2f128 0x20 / 0x31: low halves (resp. high halves) of x then y.
        _ => (
            [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]],
            [x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7]],
        ),
    }
}

/// Simulate a full candidate plan (rounds only, no relabel) over `n` registers
/// and check whether the result is a pure register permutation of the
/// deinterleave target for group width `g` slots. On success writes the
/// relabel into `plan` and returns true.
const fn sim_plan(n: usize, g: usize, plan: &mut LadderPlan) -> bool {
    // State + scratch grids; row r = register r, entries are flat 32-bit ids.
    let mut buf = [[0u16; 8]; LADDER_MAX_N];
    let mut tmp = [[0u16; 8]; LADDER_MAX_N];

    let mut r = 0;
    while r < n {
        let mut s = 0;
        while s < 8 {
            buf[r][s] = (r * 8 + s) as u16;
            s += 1;
        }
        r += 1;
    }

    // Run the rounds. Block size starts at n and halves.
    let mut round = 0;
    let mut bsize = n;
    while round < plan.rounds {
        let half = bsize / 2;
        let mut base = 0;
        while base < n {
            let mut i = 0;
            while i < half {
                let (a, b, da, db) = if plan.style[round] == 0 {
                    // adjacent pairs -> split placement
                    (base + 2 * i, base + 2 * i + 1, base + i, base + half + i)
                } else {
                    // distance pairs -> adjacent placement
                    (base + i, base + half + i, base + 2 * i, base + 2 * i + 1)
                };
                let (lo, hi) = sim_rung(plan.rung[round], buf[a], buf[b]);
                tmp[da] = lo;
                tmp[db] = hi;
                i += 1;
            }
            base += bsize;
        }
        // buf = tmp
        let mut c = 0;
        while c < n {
            buf[c] = tmp[c];
            c += 1;
        }
        bsize = half;
        round += 1;
    }

    // The deinterleave target: out[r].slot[q*g + s] = 32-id of group (q*n + r),
    // slot s. Group j lives at input register j / m, group j % m (m groups/reg).
    let m = 8 / g;
    let mut out_reg = 0;
    while out_reg < n {
        // Find the state register that equals target row `out_reg` exactly.
        let mut found = usize::MAX;
        let mut cand = 0;
        while cand < n {
            let mut matches = true;
            let mut q = 0;
            while q < m && matches {
                let j = q * n + out_reg;
                let mut s = 0;
                while s < g && matches {
                    let want = ((j / m) * 8 + (j % m) * g + s) as u16;
                    if buf[cand][q * g + s] != want {
                        matches = false;
                    }
                    s += 1;
                }
                q += 1;
            }
            if matches {
                found = cand;
                break;
            }
            cand += 1;
        }
        if found == usize::MAX {
            return false;
        }
        plan.relabel[out_reg] = found as u8;
        out_reg += 1;
    }
    true
}

/// Number of trailing zeros (log2 for a power of two), const.
const fn ilog2(mut v: usize) -> usize {
    let mut l = 0;
    while v > 1 {
        v /= 2;
        l += 1;
    }
    l
}

/// Compile-time plan search for `deinterleave_radix_by` over `n` registers of
/// `8 / g` groups (`g` = group width in 32-bit slots). Tries a small curated
/// family of candidates - every injective rung sequence of the required length
/// (w32 in either flavor, w64, w128; distinct widths) crossed with both
/// pairing/placement styles per round - and returns the first that simulates
/// to an exact register relabel of the target. `ok == false` if none does or
/// the shape is out of scope.
pub const fn ladder_search(n: usize, g: usize) -> LadderPlan {
    // Scope: pow-2 everything, group fits the register, at least 2 registers,
    // relabel table bounds.
    if n < 2 || n > LADDER_MAX_N || !n.is_power_of_two() {
        return LADDER_NONE;
    }
    if g == 0 || g > 8 || !g.is_power_of_two() || 8 % g != 0 {
        return LADDER_NONE;
    }
    let m = 8 / g;
    if m < 2 {
        // One group per register: pure register relabeling, but then the
        // caller's shape is degenerate (identity for deinterleave) - skip.
        return LADDER_NONE;
    }

    let min_rounds = if ilog2(n) < ilog2(m) { ilog2(n) } else { ilog2(m) };
    if min_rounds == 0 || min_rounds > 4 {
        return LADDER_NONE;
    }

    // Up to three passes. First the min-round pass with distinct rung widths
    // (the information-theoretic optimum; finds the square shapes and most
    // N > M shapes quickly). The bare rungs carry positional side-effects (an
    // unpck rotates the remaining position bits while exchanging one), so some
    // shapes have no min-round plan in this family - for those, later passes
    // allow one then two extra rounds with repeated widths, still much cheaper
    // than the canonical-op engines (1 op per output per round vs 2; e.g.
    // (32, 1) needs min+2 = 5 rounds = 160 ops vs the staged engine's ~320).
    // The block-halving schedule needs `n >> round >= 2`, so extra rounds only
    // exist when `min_rounds < log2(n)` (i.e. N > M shapes).
    let mut pass = 0;
    while pass < 3 {
        let rounds = min_rounds + pass;
        if rounds > 6 || rounds > ilog2(n) {
            pass += 1;
            continue;
        }
        let distinct = pass == 0;

        if let Some(plan) = ladder_pass(n, g, rounds, distinct) {
            return plan;
        }
        pass += 1;
    }

    LADDER_NONE
}

/// One search pass at a fixed round count: enumerate rung sequences (optionally
/// distinct widths) x one plan-wide style bit, simulate, return the first
/// certified plan.
///
/// The space is pruned by two empirical structure facts (every plan ever
/// certified obeys them, and they keep the const evaluation fast enough for
/// rustc's long-running-const-eval budget): the style is uniform across rounds,
/// and the crossing rung (w128) appears only in the LAST round - except at
/// `g == 4`, where it is the only rung there is.
const fn ladder_pass(n: usize, g: usize, rounds: usize, distinct: bool) -> Option<LadderPlan> {
    // Rung candidates by width: w32 has two flavors (0 = unpck, 1 = shufps
    // 88/DD), w64 = 2, w128 = 3. A rung narrower than the group would tear
    // groups apart, so width >= g: g == 1 allows all, g == 2 allows w64/w128,
    // g == 4 allows w128 only.
    let allowed: [bool; 4] = [g <= 1, g <= 1, g <= 2, g <= 4];

    // Enumerate rung sequences (4^rounds) x the style bit.
    let mut code = 0usize;
    let total = {
        let mut t = 2; // x2 for the style bit
        let mut i = 0;
        while i < rounds {
            t *= 4;
            i += 1;
        }
        t
    };

    while code < total {
        let mut plan = LadderPlan {
            ok: false,
            rounds,
            rung: [0; 6],
            style: [0; 6],
            relabel: [0; LADDER_MAX_N],
        };

        // Decode `code` into a rung sequence + the plan-wide style.
        let mut c = code;
        let style = (c % 2) as u8;
        c /= 2;

        let mut valid = true;
        let mut used_width = [false; 3]; // widths: 0 = 32 (either flavor), 1 = 64, 2 = 128
        let mut i = 0;
        while i < rounds {
            let rung = (c % 4) as u8;
            c /= 4;

            if !allowed[rung as usize] {
                valid = false;
                break;
            }
            // Crossing rung only in the last round (unless nothing else exists).
            if rung == 3 && i + 1 != rounds && g < 4 {
                valid = false;
                break;
            }
            let w = if rung <= 1 { 0 } else { (rung - 1) as usize };
            if distinct {
                if used_width[w] {
                    valid = false;
                    break;
                }
                used_width[w] = true;
            }

            plan.rung[i] = rung;
            plan.style[i] = style;
            i += 1;
        }

        if valid && sim_plan(n, g, &mut plan) {
            plan.ok = true;
            return Some(plan);
        }
        code += 1;
    }

    None
}

/// [`ladder_search`] for a register of `elem_bytes`-wide elements: converts the
/// caller's `GROUP` into 32-bit slots (`g = group * elem_bytes / 4`). Group widths
/// that are not a whole number of 32-bit slots (16/8-bit elements at small GROUP)
/// would need epi16/epi8 rungs - not implemented, so they report not-ok and the
/// caller falls back.
pub const fn ladder_search_elem(n: usize, group: usize, elem_bytes: usize) -> LadderPlan {
    let bytes = group * elem_bytes;
    if !bytes.is_multiple_of(4) {
        return LADDER_NONE;
    }
    ladder_search(n, bytes / 4)
}

/// Whether the ladder should run for this shape - the register arms gate on this.
/// Currently every certified plan is profitable: measured on AVX2 (bench `radixby`,
/// arm vs the portable route): (8,2) 6.4 vs 9.8 ns, (16,4) 9.8 vs 13.6, (16,2) 10.2
/// vs 14.9, (16,1) 12.0 vs 24.9 (2.1x), (32,1) 56 vs 58. That held ONLY after the
/// plan moved to by-value passing + literal-unrolled rounds - with `&plan` the rung
/// dispatch survived as runtime branches and the big shapes were 3-5x SLOWER than
/// their fallbacks. If the engine grows shapes with much larger `N * rounds`,
/// re-measure before assuming this stays true.
pub const fn ladder_viable(n: usize, group: usize, elem_bytes: usize) -> bool {
    ladder_search_elem(n, group, elem_bytes).ok
}

/// One runtime rung application on a register pair, matching [`sim_rung`].
/// `inv` flips the w32 flavor (`unpck` <-> `shufps 0x88/0xDD`, mutual
/// inverses) for the interleave direction; w64/w128 are their own inverses.
/// Both arguments come from const-block-derived values at every call site, so
/// the match folds to a single intrinsic pair.
#[inline(always)]
unsafe fn rung_ps(rung: u8, inv: bool, x: __m256, y: __m256) -> (__m256, __m256) {
    let r = if inv && rung <= 1 { 1 - rung } else { rung };
    match r {
        0 => (_mm256_unpacklo_ps(x, y), _mm256_unpackhi_ps(x, y)),
        1 => (_mm256_shuffle_ps::<0x88>(x, y), _mm256_shuffle_ps::<0xDD>(x, y)),
        2 => (_mm256_shuffle_ps::<0x44>(x, y), _mm256_shuffle_ps::<0xEE>(x, y)),
        _ => (
            _mm256_permute2f128_ps::<0x20>(x, y),
            _mm256_permute2f128_ps::<0x31>(x, y),
        ),
    }
}

/// One ladder round over the whole buffer. `#[inline(always)]`; every argument
/// is a compile-time constant at the (unrolled) call sites, so the style branch
/// and all indices fold.
#[inline(always)]
unsafe fn ladder_round_ps<const N: usize>(buf: &mut [__m256; N], rung: u8, inv: bool, style: u8, bsize: usize) {
    let tmp = *buf;
    let half = bsize / 2;
    let mut base = 0;
    while base < N {
        let mut i = 0;
        while i < half {
            let (a, b, da, db) = if style == 0 {
                (base + 2 * i, base + 2 * i + 1, base + i, base + half + i)
            } else {
                (base + i, base + half + i, base + 2 * i, base + 2 * i + 1)
            };
            // SAFETY: all indices < base + bsize <= N.
            unsafe {
                let (lo, hi) = rung_ps(rung, inv, *tmp.get_unchecked(a), *tmp.get_unchecked(b));
                *buf.get_unchecked_mut(da) = lo;
                *buf.get_unchecked_mut(db) = hi;
            }
            i += 1;
        }
        base += bsize;
    }
}

/// The ladder engine core, ps domain. `plan` MUST come from an inline `const`
/// block at the call site and be passed BY VALUE - through a reference LLVM
/// keeps the plan in memory and the rung dispatch survives as runtime branches
/// (measured: the isolated (8, 2) probe kept 36 branches via `&plan`, and folds
/// to a straight line by value. SROA only explodes the struct when it owns it).
/// The round loop is literal-unrolled for the same reason. `DEINT` selects the
/// direction: `true` runs the plan as searched (rounds then relabel gather);
/// `false` runs the exact inverse (relabel scatter, then the rounds reversed
/// with styles and w32 flavors flipped).
#[inline(always)]
pub unsafe fn ladder_radix_by_ps<const N: usize, const DEINT: bool>(
    inputs: [__m256; N],
    plan: LadderPlan,
) -> [__m256; N] {
    debug_assert!(plan.ok && N <= LADDER_MAX_N);

    let mut buf = inputs;

    if !DEINT {
        // Interleave: scatter through the inverse relabel first.
        let tmp = buf;
        let mut j = 0;
        while j < N {
            // SAFETY: relabel entries are < N by construction (certificate).
            unsafe { *buf.get_unchecked_mut(plan.relabel[j] as usize) = *tmp.get_unchecked(j) };
            j += 1;
        }
    }

    // Rounds, literal-unrolled (<= 6 by construction). Deinterleave runs them
    // forward (block size N halving); interleave in reverse (growing back).
    // The inverse of (pair p -> place q) is (pair q -> place p) - the flipped
    // style - and the w32 rung flavor flips (`!DEINT`).
    macro_rules! round {
        ($step:literal) => {
            if $step < plan.rounds {
                let round = if DEINT { $step } else { plan.rounds - 1 - $step };
                let style = if DEINT {
                    plan.style[round]
                } else {
                    1 - plan.style[round]
                };
                // SAFETY: forwarded caller contract.
                unsafe { ladder_round_ps::<N>(&mut buf, plan.rung[round], !DEINT, style, N >> round) };
            }
        };
    }
    round!(0);
    round!(1);
    round!(2);
    round!(3);
    round!(4);
    round!(5);

    if DEINT {
        // Gather through the relabel: out[j] = buf[relabel[j]].
        let tmp = buf;
        let mut j = 0;
        while j < N {
            // SAFETY: relabel entries are < N by construction (certificate).
            unsafe { *buf.get_unchecked_mut(j) = *tmp.get_unchecked(plan.relabel[j] as usize) };
            j += 1;
        }
    }

    buf
}

/// [`ladder_radix_by_ps`] over `__m256i` inputs (free casts both ways).
#[inline(always)]
pub unsafe fn ladder_radix_by_si<const N: usize, const DEINT: bool>(
    inputs: [__m256i; N],
    plan: LadderPlan,
) -> [__m256i; N] {
    let mut ps = [_mm256_setzero_ps(); N];
    let mut k = 0;
    while k < N {
        ps[k] = _mm256_castsi256_ps(inputs[k]);
        k += 1;
    }
    let out = ladder_radix_by_ps::<N, DEINT>(ps, plan);
    let mut si = inputs;
    let mut k = 0;
    while k < N {
        si[k] = _mm256_castps_si256(out[k]);
        k += 1;
    }
    si
}

/// [`ladder_radix_by_ps`] over `__m256d` inputs (free casts both ways).
#[inline(always)]
pub unsafe fn ladder_radix_by_pd<const N: usize, const DEINT: bool>(
    inputs: [__m256d; N],
    plan: LadderPlan,
) -> [__m256d; N] {
    let mut ps = [_mm256_setzero_ps(); N];
    let mut k = 0;
    while k < N {
        ps[k] = _mm256_castpd_ps(inputs[k]);
        k += 1;
    }
    let out = ladder_radix_by_ps::<N, DEINT>(ps, plan);
    let mut pd = inputs;
    let mut k = 0;
    while k < N {
        pd[k] = _mm256_castps_pd(out[k]);
        k += 1;
    }
    pd
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Which `(n, g)` shapes the plan search certifies, and with what plans. Purely
    /// const-eval (no AVX2 execution), so it runs anywhere. The `ok` expectations pin
    /// the engine's coverage - if a search change loses a shape, this fails loudly
    /// instead of silently falling back to a slower path.
    #[test]
    fn ladder_certification_coverage() {
        // (n, g_slots, expect_ok)
        let cases: &[(usize, usize, bool)] = &[
            // Square shapes: the engine should rediscover the native transposes.
            (8, 1, true),
            (4, 2, true),
            (2, 4, true),
            // Non-square, N < M.
            (4, 1, false), // needs an internal position rotation no 2-round plan expresses
            (2, 2, false), // canonical deinterleave_by::<2> is not a single bare rung
            (2, 1, false),
            // Non-square, N > M: extra rounds are free relabels.
            (16, 1, true),
            (32, 1, true),
            (8, 2, true),
            (16, 2, true),
            // (32, 2): only two rung widths exist at g == 2, not enough freedom to
            // cancel the rotation debt of 5 register bits - falls back (legitimately).
            (32, 2, false),
            (8, 4, true),
            (16, 4, true),
            (32, 4, true),
            (4, 4, true),
            // Out of scope.
            (3, 2, false),
            (12, 1, false),
            (2, 8, false), // one group per register: degenerate
        ];

        for &(n, g, expect) in cases {
            let plan = ladder_search(n, g);
            assert_eq!(
                plan.ok, expect,
                "ladder_search({n}, {g}): expected ok={expect}, got {plan:?}"
            );
            if plan.ok {
                std::println!(
                    "ladder({n:2}, {g}): rounds={} rung={:?} style={:?} relabel={:?}",
                    plan.rounds,
                    &plan.rung[..plan.rounds],
                    &plan.style[..plan.rounds],
                    &plan.relabel[..n]
                );
            }
        }
    }
}
