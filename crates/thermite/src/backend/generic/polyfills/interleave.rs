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
//! `N` is decomposed stage by stage like an FFT. Each stage of radix `p` splits
//! every block of `size` registers (an AoS of `size` streams) into `p`
//! sub-blocks by residue class mod `p` - groups of `p` consecutive registers
//! start at a flat position divisible by `p`, so one `p`-way split per group
//! sorts each element into its class. De-interleaving runs the stages top-down:
//!
//! - **the leftover gather stage**, at most once and first, for the
//!   non-{2,3}-smooth factor of `N` (radix [`leftover`]`(N)`): a permute+blend
//!   gather per output stream. Only `min(p, LANES)` sources can contribute lanes
//!   to a given stream, and non-contributors are skipped, so it costs
//!   `p * min(p, LANES)` permutes (one fewer blend each), not `p^2`.
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
//! For a prime `N >= 5` the leftover stage is the entire decomposition (one
//! group at radix `N`), so it degenerates exactly to [`deinterleave_any`] - the
//! full-width gather is not a special case, just the limit case.
//!
//! Staging that leftover factor rather than gathering at full width matters a
//! lot, because a full-width gather needs all `N` sources live at once and past
//! ~`LANES` registers it spills catastrophically. `f32x8` on AVX2, before -> after:
//! `N=10` 167 -> 155, `N=15` **875 -> 242**, `N=20` **1001 -> 375**. Nothing
//! 3-smooth changed by a single instruction.
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
//! ## Grouped streams: spelling a const-generic product on stable
//!
//! Composite element types are AoS records over a scalar: a dual number is
//! `1 + N` floats, a compensated float is `2`. De-interleaving `M` such streams
//! is *exactly* a radix-`M * C` de-interleave of the scalar (`C` components per
//! record) - but `V::load_deinterleaved::<{M * C}>` is not expressible on
//! stable, because a const-generic **argument** computed from other generic
//! parameters needs `generic_const_exprs`.
//!
//! The escape is that the engine never needed the product in a *type* - it
//! needs constant **trip counts**, and `M * (TAIL + 1)` as a *value* is a
//! perfectly good compile-time constant after monomorphization (a const generic
//! used in a function body is just an inlined constant; a product of two is no
//! different). The only places the count appears in a type are the register
//! buffers and the digit-reversal table, and [`StreamGroup`] fixes those with a
//! layout identity:
//!
//! ```text
//! [StreamGroup<T, TAIL>; M]  ==layout==  [T; M * (TAIL + 1)]
//! ```
//!
//! so [`deinterleave_grouped`] / [`interleave_grouped`] take group arrays,
//! flat-view them, and run the same engine with the count as a value. Stream
//! `j * C + c` of the flat problem lands at group `j`, component `c` - which is
//! the grouped output's own layout, so the re-typing is free.
//!
//! **Everything here is written to const-fold.** Each radix gets its own loop
//! (a single loop with a `match` over the radices is too large a body for LLVM
//! to unroll, and then nothing folds), and every count and radix is a
//! compile-time constant - a const generic or a product of them, threaded
//! through `#[inline(always)]` value parameters - never a runtime slice length.
//! The digit-reversal is a const table rather than a call to [`stream_pos`]
//! (recursive, `/`-and-`%`-heavy - with a runtime argument it does not fold and
//! alone cost ~200 instructions). Skipping any of those turns a branch-free
//! straight-line sequence into a spilling loop nest: an `f32x8`
//! `load_deinterleaved::<4>` measured 668 instructions before, and 31 after.
//!
//! [`InterleaveRegister`]: crate::register::InterleaveRegister
//!
//! A backend with true structural loads (ARM `LD2`/`LD3`/`LD4`) overrides the
//! memory ops outright for the widths it supports and falls back here otherwise.

use generic_array::{GenericArray, sequence::GenericSequence};

use crate::register::{CoreRegister, MaskRegister, Register, Storage};

/// One de-interleaved stream group: a `head` component plus `TAIL` trailing
/// components - `1 + TAIL` components in all.
///
/// `#[repr(C)]` with no padding possible (the array's alignment is `T`'s, and
/// `T`'s alignment divides its size), so `[StreamGroup<T, TAIL>; M]` is
/// layout-identical to `[T; M * (TAIL + 1)]`. That identity is the whole point
/// of the type: it spells a **product** of const generics in a type position on
/// stable Rust (see the module docs), which is what lets composite element
/// types - `Dual` (`TAIL = N`), `Compensated` (`TAIL = 1`) - route their
/// AoS <-> SoA memory ops through the tuned engine for any stream count `M`
/// with no dispatch ladder and no scalar fallback.
///
/// The field names are chosen to mirror the composite types it exists to serve:
/// `head`/`tail` map field-for-field onto `Dual { re, dual }` and
/// `Compensated { value, error: tail[0] }`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct StreamGroup<T, const TAIL: usize> {
    pub head: T,
    pub tail: [T; TAIL],
}

/// The flat view behind the [`StreamGroup`] layout identity:
/// `[StreamGroup<T, TAIL>; M]` as `[T; M * (TAIL + 1)]`.
#[inline(always)]
pub fn flat_groups<T, const TAIL: usize, const M: usize>(groups: &[StreamGroup<T, TAIL>; M]) -> &[T] {
    const {
        assert!(
            size_of::<StreamGroup<T, TAIL>>() == (TAIL + 1) * size_of::<T>(),
            "StreamGroup must have no padding for the flat view to be sound"
        );
    }

    // SAFETY: repr(C) with the size identity const-asserted above.
    unsafe { core::slice::from_raw_parts(groups.as_ptr() as *const T, M * (TAIL + 1)) }
}

/// Mutable [`flat_groups`].
#[inline(always)]
pub fn flat_groups_mut<T, const TAIL: usize, const M: usize>(groups: &mut [StreamGroup<T, TAIL>; M]) -> &mut [T] {
    const {
        assert!(
            size_of::<StreamGroup<T, TAIL>>() == (TAIL + 1) * size_of::<T>(),
            "StreamGroup must have no padding for the flat view to be sound"
        );
    }

    // SAFETY: as in `flat_groups`.
    unsafe { core::slice::from_raw_parts_mut(groups.as_mut_ptr() as *mut T, M * (TAIL + 1)) }
}

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

    if m > 1 {
        m
    } else if size % 3 == 0 {
        3
    } else {
        2
    }
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
///
/// `p` must be a **compile-time constant at every call site** - a const generic
/// or a product of const generics, never a slice length. From a runtime radix,
/// `first`/`last`/the skip test never fold, the `k` loop stays a real loop, and
/// the whole gather keeps its divisions and branches - 309 instructions for an
/// `f32x8` `load_deinterleaved::<5>`, versus 51 with a constant. (It was once a
/// const generic to enforce that; it is a value parameter now so the grouped
/// entry points can pass `M * (TAIL + 1)`, which cannot be spelled as a
/// const-generic argument on stable. Post-inlining the two are identical.)
#[inline(always)]
fn gather_deinterleaved<R: Register>(group: &[Storage<R>], off: usize, j: usize, p: usize) -> Storage<R> {
    let lanes = R::lanes();

    let local: GenericArray<u32, R::Lanes> = GenericArray::generate(|l| ((l * p + j) % lanes) as u32);

    let first = j / lanes;
    let last = ((lanes - 1) * p + j) / lanes;

    // SAFETY: the caller's window covers `off .. off + p`, and
    // `first <= last < p` (lane `LANES-1` reads the highest source).
    let mut acc = R::permutev(unsafe { *group.get_unchecked(off + first) }, local.clone());

    let mut k = first + 1;
    while k <= last {
        // Skip sources no lane reads from (possible only when `p > LANES`,
        // where consecutive lanes stride past whole registers): the lowest
        // lane at or beyond this register must land inside it.
        let l0 = (k * lanes - j).div_ceil(p);
        if l0 * p + j < (k + 1) * lanes {
            let selected: GenericArray<bool, <R::Mask as CoreRegister>::Lanes> =
                GenericArray::generate(|l| (l * p + j) / lanes == k);

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
///
/// `p` must be a compile-time constant at every call site, exactly as in
/// [`gather_deinterleaved`].
#[inline(always)]
fn gather_interleaved<R: Register>(
    block: &[Storage<R>],
    base: usize,
    sub: usize,
    i: usize,
    t: usize,
    p: usize,
) -> Storage<R> {
    let lanes = R::lanes();

    let local: GenericArray<u32, R::Lanes> = GenericArray::generate(|l| ((t * lanes + l) / p) as u32);

    let r0 = (t * lanes) % p;

    // SAFETY: `r * sub + i < p * sub == size`, within the caller's block.
    let mut acc = R::permutev(unsafe { *block.get_unchecked(base + r0 * sub + i) }, local.clone());

    let mut dr = 1;
    while dr < p && dr < lanes {
        let r = (r0 + dr) % p;

        let selected: GenericArray<bool, <R::Mask as CoreRegister>::Lanes> =
            GenericArray::generate(|l| (t * lanes + l) % p == r);

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

/// Emit `$body` once per literal, each guarded by `if $lit < N` - a
/// force-unrolled `for $i in 0..N` for `N` up to the literal count.
///
/// A plain `while` loop here defeats every gather: one iteration per output
/// register is too large a body for LLVM's full unroller (index arrays, mask
/// materialization, the permute+blend chain), so the loop survives, the
/// induction variable stays runtime, and nothing derived from it folds - the
/// masks are rebuilt lane by lane with scalar compares on every iteration (an
/// `f32x8` `load_deinterleaved::<5>` measured 277 instructions and 10 branches
/// that way, 72 and none this way). With a literal index the guards are
/// compile-time constants and every mask and index array folds no matter what
/// the unroller thinks of the body size.
///
/// This applies to the leftover STAGE as much as to the full-width gather: driving
/// it from a runtime `while` variable cost `ld5` 67 -> 184 and `ld10` 167 -> 342.
/// Hence the stage flattens its `(group, stream)` nest into one literal-indexed
/// loop over the `n` outputs, recovering both indices by `/` and `%` against
/// compile-time constants.
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

/// Run the de-interleave stages over `buf` top-down, leaving the streams in
/// digit-reversed register order. `n` (the register count, `== buf.len()`) must
/// be a compile-time constant; the wrappers guarantee it.
///
/// The stage sequence is fixed by [`choose_radix`]: the non-{2,3}-smooth
/// leftover (at most once, first), then every 3, then every 2. Each radix gets
/// its OWN loop, with only that radix's code in the body.
///
/// That split is load-bearing, not cosmetic. With one loop carrying a `match p`
/// over all radices, the body is large enough that LLVM's unroller gives up,
/// `size` never const-folds, and even a pure power-of-two `n` drags the whole
/// gather path (permutes, blends, mask materialization) into the output and
/// spills: an `f32x8` `load_deinterleaved::<4>` measured **668 instructions**
/// that way, versus **31** for a butterfly-only body.
///
/// The leftover stage is why the gathers take their radix as a *value* rather
/// than a const generic. Its radix is `leftover(n)`, and
/// `gather::<{ leftover(N) }>` cannot be spelled as a const-generic argument on
/// stable - which is why this stage did not exist before, and every `n` with a
/// prime factor >= 5 took a single full-width gather instead. That gather needs
/// all `n` sources live at once, so past ~8 registers it spills catastrophically
/// (`f32x8` `load_deinterleaved::<20>`: **1001 instructions**). Staged, the
/// gather only ever sees `leftover(n)` registers at a time and the rest is
/// butterflies (**205**). A value radix folds exactly like a const generic once
/// inlined, so nothing else pays for it.
#[inline(always)]
fn stages_deinterleave_flat<R: Register>(buf: &mut [Storage<R>], tmp: &mut [Storage<R>], n: usize, lo: usize) {
    let mut size = n;

    // --- the leftover gather stage, at most once, over the single top block ---
    //
    // The STREAM loop is force-unrolled (see `unroll!`) but the GROUP loop is
    // not, and the split is deliberate. Everything the gather derives from its
    // radix - the lane index array, the source range, the blend masks - depends
    // on the stream `r` and on `lo`, so a literal `r` folds all of it; the group
    // `i` only shifts the source offset. Driving BOTH from runtime variables
    // leaves the masks unfolded (`ld5` 67 -> 184, `ld10` 167 -> 342), while
    // unrolling both emits `n` whole gathers and blows up register pressure at
    // wide `n` (`ld20` 265 -> 318). Unrolling the inner one only gets both.
    //
    // `lo` is passed in from a `const { leftover(N) }` at the call site rather
    // than recomputed here: from a plain value argument the guard is a real
    // branch that LLVM keeps, and a 3-smooth `n` - which never enters this stage
    // at all - still pays for it (an 18-stream `Compensated` load measured 288
    // instructions that way versus 239).
    if lo > 1 {
        let sub = size / lo;

        let mut i = 0;
        while i < sub {
            unroll!(for r < lo {
                // SAFETY: the group spans `lo * i .. lo * i + lo <= n`, and
                // `r * sub + i < lo * sub == size == n`.
                unsafe { *tmp.get_unchecked_mut(r * sub + i) = gather_deinterleaved::<R>(buf, lo * i, r, lo) };
            });
            i += 1;
        }

        buf.copy_from_slice(tmp);
        size = sub;
    }

    // --- radix-3 stages ---
    while size % 3 == 0 {
        let sub = size / 3;

        let mut base = 0;
        while base < n {
            let mut i = 0;
            while i < sub {
                let g = base + 3 * i;
                // SAFETY: `g + 2 < base + size <= n` and the three writes land in
                // the same block. Bounds checks here are not merely redundant -
                // their panic paths keep LLVM from promoting `buf`/`tmp` out of
                // memory, which turns the whole stage into stack traffic.
                unsafe {
                    let (p0, p1, p2) = R::deinterleave3(
                        *buf.get_unchecked(g),
                        *buf.get_unchecked(g + 1),
                        *buf.get_unchecked(g + 2),
                    );
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
        while base < n {
            let mut i = 0;
            while i < sub {
                let g = base + 2 * i;
                // SAFETY: as above - `g + 1 < base + size <= n`.
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
/// [`stages_deinterleave_flat`], expecting the streams already digit-reversed.
/// `n` must be a compile-time constant, as there.
///
/// Same one-loop-per-radix split, replayed in reverse (2s, then 3s, then the
/// leftover gather), for the same const-folding reason.
#[inline(always)]
fn stages_interleave_flat<R: Register>(buf: &mut [Storage<R>], tmp: &mut [Storage<R>], n: usize, lo: usize) {
    // The de-interleave ran `leftover`, then 3s, then 2s over blocks that shrank
    // from `n` to 1. Replay it backwards: the 2s and 3s rebuild blocks up to the
    // 3-smooth part, and the leftover gather closes at full width. The 2/3 stages
    // are therefore bounded by `smooth`, NOT by `n` - with a leftover factor
    // those are different, and bounding by `n` would run a bogus extra stage.
    //
    // `lo` comes from a `const { leftover(N) }` at the call site; see
    // `stages_deinterleave_flat` for why it is not recomputed here.
    let smooth = n / lo; // 3^b * 2^a
    let pow2 = 1usize << smooth.trailing_zeros(); // 2^a

    // --- radix-2 stages: block sizes 2, 4, ..., 2^a ---
    let mut size = 2;
    while size <= pow2 {
        let sub = size / 2;

        let mut base = 0;
        while base < n {
            let mut i = 0;
            while i < sub {
                let g = base + 2 * i;
                // SAFETY: as in the de-interleave stages - `g + 1 < base + size <= n`.
                unsafe {
                    let (lo, hi) = R::interleave(*buf.get_unchecked(base + i), *buf.get_unchecked(base + sub + i));
                    *tmp.get_unchecked_mut(g) = lo;
                    *tmp.get_unchecked_mut(g + 1) = hi;
                }
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
        while base < n {
            let mut i = 0;
            while i < sub {
                let g = base + 3 * i;
                // SAFETY: as above - `g + 2 < base + size <= n`.
                unsafe {
                    let (r0, r1, r2) = R::interleave3(
                        *buf.get_unchecked(base + i),
                        *buf.get_unchecked(base + sub + i),
                        *buf.get_unchecked(base + 2 * sub + i),
                    );
                    *tmp.get_unchecked_mut(g) = r0;
                    *tmp.get_unchecked_mut(g + 1) = r1;
                    *tmp.get_unchecked_mut(g + 2) = r2;
                }
                i += 1;
            }
            base += size;
        }

        buf.copy_from_slice(tmp);
        size *= 3;
    }

    // --- the leftover gather stage, last, over the single full-width block ---
    //
    // Same split as the de-interleave side: unroll the register-within-group
    // loop (whose index the gather's masks derive from), not the group loop.
    if lo > 1 {
        let sub = smooth;

        let mut i = 0;
        while i < sub {
            unroll!(for t < lo {
                // SAFETY: `lo * i + t < lo * sub == n`, and the gather reads
                // `r * sub + i < lo * sub == n`.
                unsafe { *tmp.get_unchecked_mut(lo * i + t) = gather_interleaved::<R>(buf, 0, sub, i, t, lo) };
            });
            i += 1;
        }

        buf.copy_from_slice(tmp);
    }
}

/// [`deinterleave_any`] over flat slices: `p` streams from `p` contiguous
/// registers, one gather each. `p` must be a compile-time constant
/// (`== src.len() == out.len()`); the wrappers guarantee it.
#[inline(always)]
fn deinterleave_any_flat<R: Register>(src: &[Storage<R>], out: &mut [Storage<R>], p: usize) {
    unroll!(for j < p {
        // SAFETY: `j < p == out.len()`.
        unsafe { *out.get_unchecked_mut(j) = gather_deinterleaved::<R>(src, 0, j, p) };
    });
}

/// [`interleave_any`] over flat slices - the exact inverse of
/// [`deinterleave_any_flat`], same constant-`p` requirement.
#[inline(always)]
fn interleave_any_flat<R: Register>(values: &[Storage<R>], out: &mut [Storage<R>], p: usize) {
    unroll!(for t < p {
        // SAFETY: `t < p == out.len()`.
        unsafe { *out.get_unchecked_mut(t) = gather_interleaved::<R>(values, 0, 1, 0, t, p) };
    });
}

/// De-interleave `N` contiguous registers into `N` streams with a single
/// permute+blend gather round - no stage decomposition. This is the leftover
/// path of [`deinterleave_n`] applied at full width, and the default body of
/// [`Register::deinterleave3`]; it needs only `GenericArray<_, LANES>` index
/// arrays, so `N` is genuinely unbounded.
#[inline(always)]
pub fn deinterleave_any<R: Register, const N: usize>(src: [Storage<R>; N]) -> [Storage<R>; N] {
    let mut out = [R::EMPTY; N];
    deinterleave_any_flat::<R>(&src, &mut out, N);
    out
}

/// Interleave `N` streams into `N` contiguous registers with a single gather
/// round. The exact inverse of [`deinterleave_any`].
#[inline(always)]
pub fn interleave_any<R: Register, const N: usize>(values: [Storage<R>; N]) -> [Storage<R>; N] {
    let mut out = [R::EMPTY; N];
    interleave_any_flat::<R>(&values, &mut out, N);
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

/// [`digit_reversal`] for a grouped problem of `M * (TAIL + 1)` streams, stored
/// as a group array purely so the length is spellable; consumed through
/// [`flat_groups`].
#[inline(always)]
const fn digit_reversal_grouped<const M: usize, const TAIL: usize>() -> [StreamGroup<usize, TAIL>; M] {
    let mut t = [StreamGroup {
        head: 0usize,
        tail: [0usize; TAIL],
    }; M];

    let c = TAIL + 1;
    let n = M * c;

    let mut j = 0;
    while j < n {
        let pos = stream_pos(j, n);
        if j % c == 0 {
            t[j / c].head = pos;
        } else {
            t[j / c].tail[j % c - 1] = pos;
        }
        j += 1;
    }

    t
}

/// De-interleave `N` contiguous registers into `N` streams, for *any* `N >= 1`,
/// via the mixed-radix stage engine (see the module docs): a leftover gather
/// stage for the non-{2,3}-smooth factor, radix-3 rounds, a radix-2 butterfly,
/// then the free digit-reversal un-permutation.
///
/// For a prime `N >= 5` the leftover stage IS the whole decomposition (radix
/// `N`, one group), so this degenerates exactly to [`deinterleave_any`] - no
/// special case needed.
#[inline(always)]
pub fn deinterleave_n<R: Register, const N: usize>(src: [Storage<R>; N]) -> [Storage<R>; N] {
    let mut buf = src;
    let mut tmp = [R::EMPTY; N];
    stages_deinterleave_flat::<R>(&mut buf, &mut tmp, N, const { leftover(N) });

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
    stages_interleave_flat::<R>(&mut buf, &mut tmp, N, const { leftover(N) });
    buf
}

/// [`deinterleave_n`] for `M * (TAIL + 1)` streams - the grouped form that a
/// composite element type cannot spell as a plain const-generic count (see the
/// module docs). `src` is the flat AoS span typed as groups; the output's
/// group `j`, component `c` is stream `j * (TAIL + 1) + c`, i.e. group `j` IS
/// composite stream `j`, de-interleaved, with its components split out.
#[inline(always)]
pub fn deinterleave_grouped<R: Register, const M: usize, const TAIL: usize>(
    src: [StreamGroup<Storage<R>, TAIL>; M],
) -> [StreamGroup<Storage<R>, TAIL>; M] {
    const { assert!(M >= 1) };

    let n = M * (TAIL + 1);

    let empty = StreamGroup {
        head: R::EMPTY,
        tail: [R::EMPTY; TAIL],
    };

    let mut buf = src;
    let mut tmp = [empty; M];
    stages_deinterleave_flat::<R>(
        flat_groups_mut(&mut buf),
        flat_groups_mut(&mut tmp),
        n,
        const { leftover(M * (TAIL + 1)) },
    );

    let perm = const { digit_reversal_grouped::<M, TAIL>() };
    let perm = flat_groups(&perm);

    let src_flat = flat_groups(&buf);
    let out_flat = flat_groups_mut(&mut tmp);

    let mut j = 0;
    while j < n {
        // SAFETY: `stream_pos(_, n) < n` by construction, `j < n`.
        unsafe { *out_flat.get_unchecked_mut(j) = *src_flat.get_unchecked(*perm.get_unchecked(j)) };
        j += 1;
    }

    tmp
}

/// [`interleave_n`] for `M * (TAIL + 1)` streams. The exact inverse of
/// [`deinterleave_grouped`]: group `j`'s components become composite stream `j`
/// of the flat AoS output.
#[inline(always)]
pub fn interleave_grouped<R: Register, const M: usize, const TAIL: usize>(
    values: [StreamGroup<Storage<R>, TAIL>; M],
) -> [StreamGroup<Storage<R>, TAIL>; M] {
    const { assert!(M >= 1) };

    let n = M * (TAIL + 1);

    let empty = StreamGroup {
        head: R::EMPTY,
        tail: [R::EMPTY; TAIL],
    };

    let perm = const { digit_reversal_grouped::<M, TAIL>() };
    let perm = flat_groups(&perm);

    let mut buf = [empty; M];
    {
        let src_flat = flat_groups(&values);
        let buf_flat = flat_groups_mut(&mut buf);

        let mut j = 0;
        while j < n {
            // SAFETY: `stream_pos(_, n) < n` by construction, `j < n`.
            unsafe { *buf_flat.get_unchecked_mut(*perm.get_unchecked(j)) = *src_flat.get_unchecked(j) };
            j += 1;
        }
    }

    let mut tmp = [empty; M];
    stages_interleave_flat::<R>(
        flat_groups_mut(&mut buf),
        flat_groups_mut(&mut tmp),
        n,
        const { leftover(M * (TAIL + 1)) },
    );
    buf
}
