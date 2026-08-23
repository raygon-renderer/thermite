#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use super::arch::*;

pub use crate::backend::x86_v2::polyfills::*;

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod divider;
pub mod interleave;
pub mod math;
pub mod transpose256;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use divider::*;
pub use interleave::*;
pub use math::*;
pub use transpose256::*;

#[inline(always)]
pub unsafe fn _mm256_blendv_epi32x_v3(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(ymm0),
        _mm256_castsi256_ps(ymm1),
        _mm256_castsi256_ps(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm256_blendv_epi64x_v3(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(ymm0),
        _mm256_castsi256_pd(ymm1),
        _mm256_castsi256_pd(mask),
    ))
}

/// POLYFILL: Shift right and sign extend 64-bit integers
#[inline(always)]
pub unsafe fn _mm256_srai_epi64x_v3(v: __m256i, cnt: i32) -> __m256i {
    let m = _mm256_set1_epi64x(1i64 << (63 - cnt));
    _mm256_sub_epi64(_mm256_xor_si256(_mm256_srl_epi64(v, _mm_cvtsi32_si128(cnt)), m), m)
}

/// POLYFILL: Shift right and sign extend 64-bit integers (variable)
#[inline(always)]
pub unsafe fn _mm_srav_epi64x_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let m = _mm_srlv_epi64(_mm_set1_epu64x(1 << 63), shifts);
    _mm_sub_epi64(_mm_xor_si128(_mm_srlv_epi64(value, shifts), m), m)
}

/// POLYFILL: Shift right and sign extend 64-bit integers (variable)
#[inline(always)]
pub unsafe fn _mm256_srav_epi64x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let m = _mm256_srlv_epi64(_mm256_set1_epu64x(1 << 63), shifts);
    _mm256_sub_epi64(_mm256_xor_si256(_mm256_srlv_epi64(value, shifts), m), m)
}

/// Variable cross-lane permute of 4x64-bit lanes: `result[i] = value[idx[i] & 3]`.
///
/// AVX2 has no variable 64-bit cross-lane permute (`vpermq` takes an immediate), but
/// `vpermd` with each 64-bit index expanded to its dword pair `(2i, 2i+1)` is a single
/// shuffle. The expansion is three ops in-register: double the qword indices, duplicate
/// the low dword of each qword (`vpshufd` `[0,0,2,2]` per 128-bit lane), then add `0/1`.
///
/// Out-of-range indices are undefined per the `permutev` contract and are not masked.
#[inline(always)]
pub unsafe fn _mm256_permutevarx_epi64x_v3(value: __m256i, idxs: __m256i) -> __m256i {
    let doubled = _mm256_slli_epi64(idxs, 1); // [2i, ...] per qword
    let dup = _mm256_shuffle_epi32(doubled, 0b10_10_00_00); // low dword of each qword, twice
    let ctrl = _mm256_add_epi32(dup, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
    _mm256_permutevar8x32_epi32(value, ctrl)
}

/// [`_mm256_permutevarx_epi64x_v3`] through the `pd` reinterpretation (free casts).
#[inline(always)]
pub unsafe fn _mm256_permutevar4x64_pdx_v3(value: __m256d, idxs: __m256i) -> __m256d {
    _mm256_castsi256_pd(_mm256_permutevarx_epi64x_v3(_mm256_castpd_si256(value), idxs))
}

/// Variable within-register permute of 16x16-bit lanes: `result[i] = value[idx[i] & 15]`.
///
/// AVX2 has no 16-bit cross-lane permute (`vpermw` is AVX-512). `pshufb` only shuffles within
/// each 128-bit lane, so we run it twice: once against `value` and once against `value` with its
/// two 128-bit halves swapped, exposing the "other lane" as a source. A per-lane blend then
/// picks the half that actually holds the requested word (`bit3(idx) XOR bit3(position)`).
///
/// `widx` is the `u16x16` index register.
///
/// Out-of-range indices are undefined per the `permutev` contract and are not masked here
/// (only the functional `& 7` / `& 8` bit extractions remain); `pshufb`/`blendv` keep an OOB
/// lane memory-safe but unspecified.
#[inline(always)]
pub unsafe fn _mm256_permutev_epi16x_v3(value: __m256i, widx: __m256i) -> __m256i {
    // Byte-shuffle mask: bytes [2w, 2w+1] per lane. No `& 7` on the within-lane word index is
    // needed - `pshufb` masks each byte index to its 128-bit lane (`& 15`), so for w in 8..15
    // the byte `2w` (16..30) folds to `2(w & 7)`, exactly the intended within-lane word.
    let byte_mask = _mm256_add_epi16(
        _mm256_mullo_epi16(widx, _mm256_set1_epi16(0x0202u16 as i16)),
        _mm256_set1_epi16(0x0100),
    );

    let swapped = _mm256_permute2x128_si256(value, value, 0x01); // swap the two 128-bit halves
    let from_same = _mm256_shuffle_epi8(value, byte_mask);
    let from_other = _mm256_shuffle_epi8(swapped, byte_mask);

    // Pick `from_other` where the source half differs from the output half:
    // use_other[i] = bit3(idx[i]) XOR bit3(i).  pos_bit3 = [0;8, 8;8].  (bit 3 must be isolated.)
    let idx_bit3 = _mm256_and_si256(widx, _mm256_set1_epi16(8));

    let pos_bit3 = _mm256_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
    let blend = _mm256_cmpeq_epi16(_mm256_xor_si256(idx_bit3, pos_bit3), _mm256_set1_epi16(8));
    _mm256_blendv_epi8(from_same, from_other, blend)
}

/// Variable within-register permute of 32x8-bit lanes: `result[i] = value[idx[i] & 31]`.
///
/// AVX2 has no full 32-byte cross-lane shuffle (`vpermb` is AVX-512). `pshufb` only shuffles
/// within each 128-bit lane, so - exactly as in [`_mm256_permutev_epi16x_v3`] - we run it twice
/// (against `value` and against `value` with its 128-bit halves swapped) and blend per byte by
/// whether the requested source lives in the same or the other 128-bit half. For 32 byte lanes
/// the "which half" bit is bit 4 (`idx & 16`), not bit 3.
///
/// `ctrl` is the `u8x32` index register. For 8-bit lanes the index *is* the byte index.
///
/// Out-of-range indices are undefined per the `permutev` contract and are not masked.
#[inline(always)]
pub unsafe fn _mm256_permutev_epi8x_v3(value: __m256i, ctrl: __m256i) -> __m256i {
    let swapped = _mm256_permute2x128_si256(value, value, 0x01); // swap the two 128-bit halves
    let from_same = _mm256_shuffle_epi8(value, ctrl);
    let from_other = _mm256_shuffle_epi8(swapped, ctrl);

    // Pick `from_other` where the source half differs from the output half:
    // use_other[i] = bit4(idx[i]) XOR bit4(i).  pos_bit4 = [0;16, 16;16].
    let idx_bit4 = _mm256_and_si256(ctrl, _mm256_set1_epi8(16));
    let pos_bit4 = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // low 128-bit half
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // high half
    );
    let blend = _mm256_cmpeq_epi8(_mm256_xor_si256(idx_bit4, pos_bit4), _mm256_set1_epi8(16));
    _mm256_blendv_epi8(from_same, from_other, blend)
}

#[inline(always)]
pub unsafe fn _mm256_set1_epu32x(value: u32) -> __m256i {
    _mm256_set1_epi32(value as i32)
}

#[inline(always)]
pub unsafe fn _mm256_set1_epu64x(value: u64) -> __m256i {
    _mm256_set1_epi64x(value as i64)
}
