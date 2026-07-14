//! Portable N-way interleave / de-interleave of a register array - the register
//! half of [`Register::load_deinterleaved`](crate::register::Register::load_deinterleaved)
//! and [`store_interleaved`](crate::register::Register::store_interleaved).
//!
//! The contract, over the flat span of `N * LANES` elements that `N` contiguous
//! registers cover (`flat[i * LANES + l]` is lane `l` of register `i`):
//!
//! ```text
//! deinterleave: out[j][lane] == flat[lane * N + j]     (AoS -> SoA)
//! interleave:   flat[lane * N + j] == values[j][lane]  (SoA -> AoS)
//! ```
//!
//! A 3-smooth `N` (`N = 3^b * 2^a`) is decomposed stage by stage like an FFT.
//! Each stage of radix `p` splits every block of `size` registers (an AoS of
//! `size` streams) into `p` sub-blocks by residue class mod `p` - groups of `p`
//! consecutive registers start at a flat position divisible by `p`, so one
//! `p`-way split per group sorts each element into its class. De-interleaving
//! runs the stages top-down:
//!
//! - **radix-3 stages** via [`Register::deinterleave3`] - a real register
//!   primitive, so a backend can give it a native sequence (NEON: three
//!   `TBL3`s).
//! - **radix-2 butterfly stages** over [`InterleaveRegister`]'s native 2-way
//!   ops - the same "treat the pair as one contiguous `2 * LANES` span" trick
//!   [`ArrayRegister`](crate::register::array::ArrayRegister) uses to chain
//!   chunks, lifted to any register count. `interleave`/`deinterleave` are
//!   single instructions nearly everywhere (`unpck` / `zip`+`uzp` /
//!   `i32x4_shuffle`), so these are the cheapest stages.
//!
//! Any other `N` - one carrying a prime factor >= 5 - takes a **single
//! full-width permute+blend gather** ([`deinterleave_any`]) instead. Only
//! `min(N, LANES)` sources can contribute lanes to a given output stream, and
//! non-contributors are skipped, so it costs `N * min(N, LANES)` permutes (one
//! fewer blend each), not `N^2`. It is deliberately NOT a stage: a stage's radix
//! would have to be the leftover factor, and `gather::<{ leftover(N) }>` is not
//! expressible as a const-generic argument on stable - and a *runtime* radix
//! makes every derived index unfoldable, which costs far more than the staging
//! saves. A staged version would only beat the full gather for mixed `N` like
//! 10 or 20, which are exotic.
//!
//! The stages leave the streams in mixed-radix *digit-reversed* register order
//! (the classic transpose artifact; pure bit-reversal when `N = 2^k`). The
//! final un-permutation costs nothing - it only decides which slot each
//! already-computed register lands in. Interleaving is the exact mirror:
//! digit-reversed scatter first, then the same stages bottom-up with the
//! inverse primitives.
//!
//! Stage order (3s, then 2s) is taste, not necessity - each radix's op count is
//! order-independent - but the digit-reversal permutation must be derived from
//! the same factor sequence the stages use, so both come from [`choose_radix`].
//!
//! **Everything here is written to const-fold.** Each radix gets its own loop
//! (a single loop with a `match` over the radices is too large a body for LLVM
//! to unroll, and then nothing folds), `N` and every radix are const generics
//! rather than slice lengths, and the digit-reversal is a const table rather
//! than a call to [`stream_pos`]. Skipping any of those turns a branch-free
//! straight-line sequence into a spilling loop nest: an `f32x8`
//! `load_deinterleaved::<4>` measured 668 instructions before, and 31 after.
//!
//! [`InterleaveRegister`]: crate::register::InterleaveRegister
//!
//! A backend with true structural loads (ARM `LD2`/`LD3`/`LD4`) overrides the
//! memory ops outright for the widths it supports and falls back here otherwise.

use generic_array::{GenericArray, sequence::GenericSequence};

use crate::register::{CoreRegister, MaskRegister, Register, Storage};

/// Radix of the top stage over a block of `size` registers (`size >= 2`):
/// the whole non-{2,3}-smooth leftover first, then 3s, then 2s.
#[inline(always)]
const fn choose_radix(size: usize) -> usize {
    let mut m = size;
    while m % 2 == 0 {
        m /= 2;
    }
    while m % 3 == 0 {
        m /= 3;
    }

    if m > 1 { m } else if size % 3 == 0 { 3 } else { 2 }
}

/// Buffer position stream `j` occupies after all de-interleave stages: the
/// mixed-radix digit reversal of `j` under the [`choose_radix`] factor
/// sequence (plain bit reversal when `n = 2^k`).
///
/// Stage 1 sends streams congruent to `r` mod `p` into sub-block `r`, where
/// the stream's index within its class is `j / p`; recursing gives the rest.
#[inline(always)]
const fn stream_pos(j: usize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }

    let p = choose_radix(n);
    let sub = n / p;

    (j % p) * sub + stream_pos(j / p, sub)
}

/// One de-interleaved stream of a `p`-register AoS group: lane `l` of stream
/// `j` is group-flat position `l * p + j`, i.e. lane `(l * p + j) % LANES` of
/// group register `(l * p + j) / lanes`.
///
/// Gathers with one permute per *contributing* source register - the sources
/// hit are a subrange of `[j / LANES, ((LANES-1) * p + j) / LANES]`, at most
/// `min(p, LANES)` of them - and folds them with blends. The first contributor
/// is taken whole (no blend into `EMPTY`): every lane it does not own is
/// overwritten by exactly one later blend, because lane sources are
/// non-decreasing in `l`.
/// `P` is a const generic, NOT `group.len()`. From a slice length the radix is a
/// runtime value, so `first`/`last`/the skip test never fold, the `k` loop stays
/// a real loop, and the whole gather keeps its divisions and branches - 309
/// instructions for an `f32x8` `load_deinterleaved::<5>`, versus 51 once `P` is
/// a constant. `group` is still a slice (the stage engine hands it a window),
/// but every index derived from it is now compile-time.
#[inline(always)]
fn gather_deinterleaved<R: Register, const P: usize>(group: &[Storage<R>], off: usize, j: usize) -> Storage<R> {
    let lanes = R::lanes();

    let local: GenericArray<u32, R::Lanes> = GenericArray::generate(|l| ((l * P + j) % lanes) as u32);

    let first = j / lanes;
    let last = ((lanes - 1) * P + j) / lanes;

    // SAFETY: the caller's window covers `off .. off + P`, and
    // `first <= last < P` (lane `LANES-1` reads the highest source).
    let mut acc = R::permutev(unsafe { *group.get_unchecked(off + first) }, local.clone());

    let mut k = first + 1;
    while k <= last {
        // Skip sources no lane reads from (possible only when `P > LANES`,
        // where consecutive lanes stride past whole registers): the lowest
        // lane at or beyond this register must land inside it.
        let l0 = (k * lanes - j).div_ceil(P);
        if l0 * P + j < (k + 1) * lanes {
            let selected: GenericArray<bool, <R::Mask as CoreRegister>::Lanes> =
                GenericArray::generate(|l| (l * P + j) / lanes == k);

            let mask = <R::Mask as MaskRegister>::new_mask(selected);
            acc = R::blendv(
                mask,
                acc,
                R::permutev(unsafe { *group.get_unchecked(off + k) }, local.clone()),
            );
        }
        k += 1;
    }

    acc
}

/// One register of a re-interleaved `p`-register group. The group's classes
/// live strided in `block`: class `r` is `block[r * sub + i]`. Lane `l` of
/// group register `t` is group-flat position `g = t * LANES + l`, i.e. lane
/// `g / p` of class `g % p`.
///
/// The `LANES` consecutive flat positions of register `t` cycle through
/// residues `(t * LANES) % p, +1, ...` mod `p`, so exactly `min(p, LANES)`
/// classes contribute; iterating cyclically from the first lets it be taken
/// whole, with one blend per remaining contributor.
/// `P` is a const generic for the same reason as in [`gather_deinterleaved`]:
/// a runtime radix stops every derived index from folding.
#[inline(always)]
fn gather_interleaved<R: Register, const P: usize>(
    block: &[Storage<R>],
    base: usize,
    sub: usize,
    i: usize,
    t: usize,
) -> Storage<R> {
    let lanes = R::lanes();

    let local: GenericArray<u32, R::Lanes> = GenericArray::generate(|l| ((t * lanes + l) / P) as u32);

    let r0 = (t * lanes) % P;

    // SAFETY: `r * sub + i < P * sub == size`, within the caller's block.
    let mut acc = R::permutev(unsafe { *block.get_unchecked(base + r0 * sub + i) }, local.clone());

    let mut dr = 1;
    while dr < P && dr < lanes {
        let r = (r0 + dr) % P;

        let selected: GenericArray<bool, <R::Mask as CoreRegister>::Lanes> =
            GenericArray::generate(|l| (t * lanes + l) % P == r);

        let mask = <R::Mask as MaskRegister>::new_mask(selected);
        acc = R::blendv(
            mask,
            acc,
            R::permutev(unsafe { *block.get_unchecked(base + r * sub + i) }, local.clone()),
        );

        dr += 1;
    }

    acc
}

/// The non-{2,3}-smooth leftover of `n` - the radix of the single gather stage
/// (1 when `n` is 3-smooth, i.e. no gather stage at all).
#[inline(always)]
const fn leftover(n: usize) -> usize {
    let mut m = n;
    while m % 2 == 0 {
        m /= 2;
    }
    while m % 3 == 0 {
        m /= 3;
    }
    m
}

/// Run the de-interleave stages over `buf` top-down, leaving the streams in
/// digit-reversed register order.
///
/// The stage sequence is fixed by [`choose_radix`]: the leftover gather (at most
/// once), then every 3, then every 2. Each radix therefore gets its OWN loop,
/// with only that radix's code in the body.
///
/// That split is load-bearing, not cosmetic. With one loop carrying a `match p`
/// over all three radices, the body is large enough that LLVM's unroller gives
/// up, `size` never const-folds, and even a pure power-of-two `N` drags the
/// whole gather path (permutes, blends, mask materialization) into the output
/// and spills: an `f32x8` `load_deinterleaved::<4>` measured **668
/// instructions** that way, versus **31** for a butterfly-only body. `N` is a
/// const generic here for the same reason - from a `&mut [Storage<R>]` the
/// length is not reliably a compile-time constant, and every derived radix stops
/// folding with it.
#[inline(always)]
fn stages_deinterleave<R: Register, const N: usize>(buf: &mut [Storage<R>; N], tmp: &mut [Storage<R>; N]) {
    let mut size = N;

    // --- radix-3 stages ---
    while size % 3 == 0 {
        let sub = size / 3;

        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < sub {
                let g = base + 3 * i;
                // SAFETY: `g + 2 < base + size <= N` and the three writes land in
                // the same block. Bounds checks here are not merely redundant -
                // their panic paths keep LLVM from promoting `buf`/`tmp` out of
                // memory, which turns the whole stage into stack traffic.
                unsafe {
                    let (p0, p1, p2) =
                        R::deinterleave3(*buf.get_unchecked(g), *buf.get_unchecked(g + 1), *buf.get_unchecked(g + 2));
                    *tmp.get_unchecked_mut(base + i) = p0;
                    *tmp.get_unchecked_mut(base + sub + i) = p1;
                    *tmp.get_unchecked_mut(base + 2 * sub + i) = p2;
                }
                i += 1;
            }
            base += size;
        }

        buf.copy_from_slice(tmp);
        size = sub;
    }

    // --- radix-2 butterfly stages ---
    while size > 1 {
        let sub = size / 2;

        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < sub {
                let g = base + 2 * i;
                // SAFETY: as above - `g + 1 < base + size <= N`.
                unsafe {
                    let (evens, odds) = R::deinterleave(*buf.get_unchecked(g), *buf.get_unchecked(g + 1));
                    *tmp.get_unchecked_mut(base + i) = evens;
                    *tmp.get_unchecked_mut(base + sub + i) = odds;
                }
                i += 1;
            }
            base += size;
        }

        buf.copy_from_slice(tmp);
        size = sub;
    }
}

/// Run the interleave stages bottom-up - the exact inverse of
/// [`stages_deinterleave`], expecting the streams already digit-reversed.
///
/// Same one-loop-per-radix split, replayed in reverse (2s, then 3s, then the
/// gather), for the same const-folding reason.
#[inline(always)]
fn stages_interleave<R: Register, const N: usize>(buf: &mut [Storage<R>; N], tmp: &mut [Storage<R>; N]) {
    // The 3-smooth part is `3^b * 2^a`; replay its stages smallest-block-first.
    let smooth = N / const { leftover(N) };
    let pow2 = 1usize << smooth.trailing_zeros(); // 2^a

    // --- radix-2 stages: block sizes 2, 4, ..., 2^a ---
    let mut size = 2;
    while size <= pow2 {
        let sub = size / 2;

        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < sub {
                let g = base + 2 * i;
                let (lo, hi) = R::interleave(buf[base + i], buf[base + sub + i]);
                tmp[g] = lo;
                tmp[g + 1] = hi;
                i += 1;
            }
            base += size;
        }

        buf.copy_from_slice(tmp);
        size *= 2;
    }

    // --- radix-3 stages: block sizes 3 * 2^a, 9 * 2^a, ..., 3^b * 2^a ---
    let mut size = pow2 * 3;
    while size <= smooth {
        let sub = size / 3;

        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < sub {
                let g = base + 3 * i;
                let (r0, r1, r2) = R::interleave3(buf[base + i], buf[base + sub + i], buf[base + 2 * sub + i]);
                tmp[g] = r0;
                tmp[g + 1] = r1;
                tmp[g + 2] = r2;
                i += 1;
            }
            base += size;
        }

        buf.copy_from_slice(tmp);
        size *= 3;
    }

}

/// Emit `$body` once per literal, each guarded by `if $lit < N` - a
/// force-unrolled `for $i in 0..N` for `N` up to the literal count.
///
/// A plain `while` loop here defeats the whole gather: one iteration per
/// output register is too large a body for LLVM's full unroller (index
/// arrays, mask materialization, the permute+blend chain), so the loop
/// survives, the induction variable stays runtime, and nothing derived from
/// it folds - the masks are rebuilt lane by lane with scalar compares on
/// every iteration (an `f32x8` `load_deinterleaved::<5>` measured 277
/// instructions and 10 branches that way, 72 and none this way). With a
/// literal index the guards are compile-time constants and every mask and
/// index array folds no matter what the unroller thinks of the body size.
///
/// `crunchy::unroll!` (already a dependency) is the same trick, but its range
/// end must be a literal too, so it cannot consume the const-generic `N`
/// directly and would still need the `if $lit < $n` guards plus a `limit_*`
/// feature to reach 32.
macro_rules! unroll {
    (for $i:ident < $n:ident $body:block) => {
        unroll!(@emit $i $n $body
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);

        // Runtime tail for an exotic `N` beyond the emitted literals.
        let mut $i = 32;
        while $i < $n {
            $body
            $i += 1;
        }
    };
    (@emit $i:ident $n:ident $body:block $($lit:literal)*) => {
        $(if $lit < $n {
            let $i = $lit;
            $body
        })*
    };
}

/// De-interleave `N` contiguous registers into `N` streams with a single
/// permute+blend gather round - no stage decomposition. This is the leftover
/// radix of [`deinterleave_n`] applied at full width, and the default body of
/// [`Register::deinterleave3`]; it needs only `GenericArray<_, LANES>` index
/// arrays, so `N` is genuinely unbounded.
#[inline(always)]
pub fn deinterleave_any<R: Register, const N: usize>(src: [Storage<R>; N]) -> [Storage<R>; N] {
    let mut out = [R::EMPTY; N];

    unroll!(for j < N {
        out[j] = gather_deinterleaved::<R, N>(&src, 0, j);
    });

    out
}

/// Interleave `N` streams into `N` contiguous registers with a single gather
/// round. The exact inverse of [`deinterleave_any`].
#[inline(always)]
pub fn interleave_any<R: Register, const N: usize>(values: [Storage<R>; N]) -> [Storage<R>; N] {
    let mut out = [R::EMPTY; N];

    unroll!(for t < N {
        out[t] = gather_interleaved::<R, N>(&values, 0, 1, 0, t);
    });

    out
}

/// The digit-reversal permutation, materialized at compile time:
/// `DIGIT_REVERSAL[j]` is the buffer slot stream `j` ends up in.
///
/// This MUST be a const table rather than a call to [`stream_pos`] in the loop.
/// `stream_pos` is recursive and full of `/` and `%`; with a runtime `j` it does
/// not fold, and the loop keeps its divisions, its branches, and a
/// `panic_bounds_check` (the compiler cannot prove `stream_pos(j, N) < N`). That
/// alone cost ~200 instructions in an `f32x8` `load_deinterleaved::<4>` - more
/// than the entire butterfly it was permuting.
#[inline(always)]
const fn digit_reversal<const N: usize>() -> [usize; N] {
    let mut t = [0usize; N];

    let mut j = 0;
    while j < N {
        t[j] = stream_pos(j, N);
        j += 1;
    }

    t
}

/// De-interleave `N` contiguous registers into `N` streams, for *any* `N >= 1`,
/// via the mixed-radix stage engine (see the module docs): a gather stage for
/// the non-{2,3}-smooth part, radix-3 rounds, a radix-2 butterfly, then the
/// free digit-reversal un-permutation.
#[inline(always)]
pub fn deinterleave_n<R: Register, const N: usize>(src: [Storage<R>; N]) -> [Storage<R>; N] {
    // A prime factor >= 5 cannot be a stage: its radix would have to be
    // `leftover(N)`, and `gather::<{ leftover(N) }>` is not expressible as a
    // const-generic argument on stable (`generic_const_exprs`). A runtime radix
    // is worse than useless here - nothing downstream of it folds, and an
    // `f32x8` N=5 measured 309 instructions that way. So such an `N` takes one
    // full-width gather instead, whose radix IS `N` and therefore folds (51
    // instructions for the same case). The staged decomposition would only beat
    // it for mixed `N` like 10 or 20, which are exotic.
    if const { leftover(N) > 1 } {
        return deinterleave_any::<R, N>(src);
    }

    let mut buf = src;
    let mut tmp = [R::EMPTY; N];
    stages_deinterleave::<R, N>(&mut buf, &mut tmp);

    // Undo the digit reversal the stages leave behind - a pure register
    // re-slotting, free once the table is a constant.
    let perm = const { digit_reversal::<N>() };

    let mut j = 0;
    while j < N {
        // SAFETY: `stream_pos(_, N) < N` by construction.
        unsafe { *tmp.get_unchecked_mut(j) = *buf.get_unchecked(perm[j]) };
        j += 1;
    }

    tmp
}

/// Interleave `N` streams into `N` contiguous registers, for *any* `N >= 1`.
/// The exact inverse of [`deinterleave_n`].
#[inline(always)]
pub fn interleave_n<R: Register, const N: usize>(values: [Storage<R>; N]) -> [Storage<R>; N] {
    // See `deinterleave_n`: a non-3-smooth `N` takes one full-width gather.
    if const { leftover(N) > 1 } {
        return interleave_any::<R, N>(values);
    }

    // Scatter into digit-reversed order, then replay the stages backwards.
    let perm = const { digit_reversal::<N>() };

    let mut buf = [R::EMPTY; N];
    let mut j = 0;
    while j < N {
        // SAFETY: `stream_pos(_, N) < N` by construction.
        unsafe { *buf.get_unchecked_mut(perm[j]) = *values.get_unchecked(j) };
        j += 1;
    }

    let mut tmp = [R::EMPTY; N];
    stages_interleave::<R, N>(&mut buf, &mut tmp);
    buf
}
