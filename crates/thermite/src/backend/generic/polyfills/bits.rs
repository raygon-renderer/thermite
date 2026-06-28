//! Generic bit-manipulation polyfills.
//!
//! "Better than scalar" generic implementations that a backend can delegate to.
//! Currently: the portable Morton-code (Z-order curve) bit-interleave cascade
//! used by [`UnsignedIntegerRegister::morton`](crate::register::UnsignedIntegerRegister::morton)
//! / [`reverse_morton`](crate::register::UnsignedIntegerRegister::reverse_morton)
//! and by backend overrides for the dimensions they do not specialize.

use super::*;

/// Generic `N`-dimensional Morton encode via the portable `O(log W)` shift/mask
/// bit-spread cascade (the generalized "magic number" method).
///
/// Interleaves the low `floor(W / N)` bits of each of the `N` coordinates,
/// placing bit `i` of `values[d]` at output position `i * N + d`. This is the
/// fallback used by
/// [`UnsignedIntegerRegister::morton`](crate::register::UnsignedIntegerRegister::morton)
/// and by backend overrides for the dimensions they do not specialize (e.g. a
/// CLMUL `N == 2` path delegates every other `N` here).
///
/// All loop bounds, shifts, and masks are compile-time constants at
/// monomorphization, so the cascade folds to a straight-line shift/and/or
/// sequence with immediate mask operands.
#[inline(always)]
pub fn morton_cascade<R: UnsignedIntegerRegister, const N: usize>(values: [Storage<R>; N]) -> Storage<R> {
    // N == 1 is the identity (every bit stays put); skip the no-op cascade.
    if const { N == 1 } {
        return values[0];
    }

    let width = (size_of::<R::Element>() * 8) as u32;
    let bits = morton_bits_per_lane(width, N); // usable bits per coordinate
    let passes = morton_passes(bits); // ceil(log2(bits)) shift/mask passes
    // mask of the low `bits` ones, isolating each coordinate's usable bits
    let low = R::shr(R::not(R::EMPTY), width - bits);

    let mut code = R::EMPTY;

    let mut d = 0;
    while d < N {
        // spread: send bit i of this coordinate to position i*N
        let mut x = R::bitand(values[d], low);

        let mut k = passes;
        while k > 0 {
            k -= 1;
            let shift = (N as u32 - 1) << k; // (N - 1) * 2^k
            let mask = morton_block_mask::<R>(k, N);
            x = R::bitand(R::bitor(x, R::shl(x, shift)), mask);
        }

        code = R::bitor(code, R::shl(x, d as u32));
        d += 1;
    }

    code
}

/// Widen the two decoded axes of a 2D Morton code into the `[Storage<R>; N]`
/// shape `reverse_morton` must return. A backend 2D-decode fast path produces
/// exactly two columns (`x`, `y`); this places them as `out[0]`/`out[1]` so the
/// `N == 2` override can return the generic array type. (`N != 2` never reaches
/// this - those dimensions use [`reverse_morton_cascade`].)
#[inline(always)]
pub fn morton_pack2<R: UnsignedIntegerRegister, const N: usize>(x: Storage<R>, y: Storage<R>) -> [Storage<R>; N] {
    let cols = [x, y];
    let mut out = [R::EMPTY; N];
    let mut d = 0;
    // runtime index (not a literal `out[1]`) so the fixed-size array bounds lint
    // stays quiet for the dead `N < 2` monomorphizations
    while d < 2 && d < N {
        out[d] = cols[d];
        d += 1;
    }
    out
}

/// Generic inverse of [`morton_cascade`]: de-interleave an `N`-dimensional
/// Morton code by running the spread cascade in reverse (right-shifts), so
/// `out[d]` gathers output bits `d, d + N, d + 2N, ...` back into the low
/// `floor(W / N)` bits.
#[inline(always)]
pub fn reverse_morton_cascade<R: UnsignedIntegerRegister, const N: usize>(code: Storage<R>) -> [Storage<R>; N] {
    if const { N == 1 } {
        return [code; N];
    }

    let width = (size_of::<R::Element>() * 8) as u32;
    let bits = morton_bits_per_lane(width, N);
    let passes = morton_passes(bits);
    // mask of the low `bits` ones (the final, fully-compacted layout)
    let low = R::shr(R::not(R::EMPTY), width - bits);
    // every N-th bit set: the spread (stride-N) layout to start compaction from
    let stride = morton_block_mask::<R>(0, N);

    let mut out = [R::EMPTY; N];

    let mut d = 0;
    while d < N {
        // compact: gather bits at positions d, d+N, d+2N, ... down to contiguous
        let mut x = R::bitand(R::shr(code, d as u32), stride);

        let mut k = 0;
        while k < passes {
            let shift = (N as u32 - 1) << k; // (N - 1) * 2^k
            // mask after this step: the next-coarser block layout, or the
            // final low-`bits` mask on the last pass
            let mask = if k + 1 >= passes { low } else { morton_block_mask::<R>(k + 1, N) };
            x = R::bitand(R::bitor(x, R::shr(x, shift)), mask);
            k += 1;
        }

        out[d] = x;
        d += 1;
    }

    out
}

/// Usable bits per coordinate for an `N`-dimensional Morton code over a
/// `width`-bit element: `floor(width / N)`, clamped to at least 1 so the
/// low-bits mask is always well-formed (`N > width` is degenerate but defined).
pub const fn morton_bits_per_lane(width: u32, dims: usize) -> u32 {
    let bits = width / dims as u32;
    if bits == 0 { 1 } else { bits }
}

/// Number of shift/mask passes in the generic Morton bit-spread: `ceil(log2(bits))`.
pub const fn morton_passes(bits: u32) -> u32 {
    if bits <= 1 { 0 } else { u32::BITS - (bits - 1).leading_zeros() }
}

/// Builds the pass-`block_log2` cascade mask for an `N`-dimensional Morton
/// spread: contiguous blocks of `2^block_log2` set bits repeating with period
/// `N * 2^block_log2`, tiled across the element width.
///
/// For `dims = 3, block_log2 = 0` this is the familiar `0x...09249249` (every
/// 3rd bit); larger `block_log2` give the coarser intermediate layouts. All
/// arguments are compile-time constants at monomorphization, so the whole
/// construction folds to immediate mask constants.
#[inline(always)]
pub fn morton_block_mask<R: UnsignedIntegerRegister>(block_log2: u32, dims: usize) -> Storage<R> {
    let width = (size_of::<R::Element>() * 8) as u32;
    let block = 1u32 << block_log2; // 2^block_log2 contiguous ones
    let period = dims as u32 * block; // stride between blocks

    // start with the low `block` ones, then tile with stride `period` by log-doubling
    let mut mask = R::shr(R::not(R::EMPTY), width - block);

    let mut s = period;
    while s < width {
        mask = R::bitor(mask, R::shl(mask, s));
        s <<= 1;
    }

    mask
}
