use super::*;

// ---------------------------------------------------------------------------
// Runtime detection: are the relaxed madd instructions true fused FMAs?
//
// `f32x4/f64x2_relaxed_madd`/`_nmadd` are allowed to be either a fused
// multiply-add or a separate multiply-then-add, chosen by the engine. The
// relaxed-simd spec resolves that nondeterminism at module instantiation (the
// `fpenv` model): a given relaxed instruction behaves as one fixed function
// for the lifetime of the instance, so a one-time canary check is sound.
//
// When the engine fuses (any host whose CPU has hardware FMA, so the common
// case), a fused relaxed madd is simply a hardware FMA: `mul_add` can be a single
// instruction, bit-identical to the round-to-odd emulation it replaces. Both
// branches produce the same bits, so this dispatch trades only speed, never
// results or reproducibility.
//
// The canaries force runtime evaluation with `black_box` (a constant-folded
// relaxed op could resolve to either semantic at compile time) and also
// require subnormal inputs/outputs to survive, so an engine that flushes
// denormals in its relaxed ops is conservatively treated as unfused.
// ---------------------------------------------------------------------------

const RELAXED_CHECKED: u8 = 1;
const RELAXED_F64_MADD_FUSED: u8 = 2;
const RELAXED_F64_NMADD_FUSED: u8 = 4;
const RELAXED_F32_MADD_FUSED: u8 = 8;
const RELAXED_F32_NMADD_FUSED: u8 = 16;

/// Written exactly once, by [`detect_relaxed_fma`] from the pre-`main` ctor
/// (single-threaded by construction: wasm threads can only be spawned after
/// `_start`, so every read happens-after the sole write, no data race). A
/// plain (non-atomic) static because LICM refuses to hoist even relaxed
/// atomic loads out of loops, while the plain load is loop-invariant-hoistable.
static mut RELAXED_FMA: u8 = 0;

/// Runs the canaries and caches the flags. Invoked from the `.init_array`
/// ctor in lib.rs (before `main`, via `__wasm_call_ctors`), not lazily from
/// the read path: a detect call on the read path would sit inside callers'
/// loops and block LICM from hoisting the flag load (its store to
/// `RELAXED_FMA` defeats the aliasing analysis, measured on a Horner loop).
/// Embedders that skip `__wasm_call_ctors` leave the flags at 0 and take the
/// (bit-identical, slower) emulation on every call.
pub(crate) fn detect_relaxed_fma() -> u8 {
    use core::hint::black_box;

    let mut flags = RELAXED_CHECKED;

    // f64: a * b = 1 - 2^-54 exactly, the midpoint between 1 - 2^-53 and 1.
    // Fused: (a*b) - 1 = -2^-54 exactly. Unfused: RN(a*b) = 1 (ties-even),
    // then 1 - 1 = 0.
    let a = black_box(f64x2_splat(1.0 + 2.0_f64.powi(-27)));
    let b = black_box(f64x2_splat(1.0 - 2.0_f64.powi(-27)));
    let fused_64 = f64x2_extract_lane::<0>(f64x2_relaxed_madd(a, b, f64x2_splat(-1.0)))
        == -(2.0_f64.powi(-54));
    let nfused_64 = f64x2_extract_lane::<0>(f64x2_relaxed_nmadd(a, b, f64x2_splat(1.0)))
        == 2.0_f64.powi(-54);

    // Subnormal guards: inputs must not be DAZed, subnormal results must not
    // be FTZed, or the "fused" answer is not a true FMA.
    let min_sub = black_box(f64x2_splat(f64::from_bits(1)));
    let tiny = black_box(f64x2_splat(2.0_f64.powi(-537)));
    let sub_in = f64x2_extract_lane::<0>(f64x2_relaxed_madd(min_sub, black_box(f64x2_splat(1.0)), f64x2_splat(0.0)))
        == f64::from_bits(1);
    let sub_out = f64x2_extract_lane::<0>(f64x2_relaxed_madd(tiny, tiny, f64x2_splat(0.0)))
        == 2.0_f64.powi(-1074);
    let nsub_in = f64x2_extract_lane::<0>(f64x2_relaxed_nmadd(min_sub, black_box(f64x2_splat(-1.0)), f64x2_splat(0.0)))
        == f64::from_bits(1);

    if fused_64 && sub_in && sub_out {
        flags |= RELAXED_F64_MADD_FUSED;
    }
    if nfused_64 && nsub_in && sub_out {
        flags |= RELAXED_F64_NMADD_FUSED;
    }

    // f32 analogue: a * b = 1 - 2^-24 exactly, the midpoint at 24 bits.
    let a = black_box(f32x4_splat(1.0 + 2.0_f32.powi(-12)));
    let b = black_box(f32x4_splat(1.0 - 2.0_f32.powi(-12)));
    let fused_32 = f32x4_extract_lane::<0>(f32x4_relaxed_madd(a, b, f32x4_splat(-1.0)))
        == -(2.0_f32.powi(-24));
    let nfused_32 = f32x4_extract_lane::<0>(f32x4_relaxed_nmadd(a, b, f32x4_splat(1.0)))
        == 2.0_f32.powi(-24);

    let min_sub = black_box(f32x4_splat(f32::from_bits(1)));
    let tiny_a = black_box(f32x4_splat(2.0_f32.powi(-75)));
    let tiny_b = black_box(f32x4_splat(2.0_f32.powi(-74)));
    let sub_in = f32x4_extract_lane::<0>(f32x4_relaxed_madd(min_sub, black_box(f32x4_splat(1.0)), f32x4_splat(0.0)))
        == f32::from_bits(1);
    let sub_out = f32x4_extract_lane::<0>(f32x4_relaxed_madd(tiny_a, tiny_b, f32x4_splat(0.0)))
        == 2.0_f32.powi(-149);
    let nsub_in = f32x4_extract_lane::<0>(f32x4_relaxed_nmadd(min_sub, black_box(f32x4_splat(-1.0)), f32x4_splat(0.0)))
        == f32::from_bits(1);

    if fused_32 && sub_in && sub_out {
        flags |= RELAXED_F32_MADD_FUSED;
    }
    if nfused_32 && nsub_in && sub_out {
        flags |= RELAXED_F32_NMADD_FUSED;
    }

    // SAFETY: sole write, pre-main, single-threaded (see RELAXED_FMA).
    unsafe { *(&raw mut RELAXED_FMA) = flags };
    flags
}

/// Cached relaxed-FMA capability flags, written once by the pre-main ctor.
/// Call-free and non-atomic by design so the load is loop-invariant and
/// hoistable (see `RELAXED_FMA`).
#[inline(always)]
fn relaxed_fma_flags() -> u8 {
    // SAFETY: read-only after the single pre-main write (see RELAXED_FMA).
    unsafe { *(&raw const RELAXED_FMA) }
}

// The emulation arms are outlined with `#[inline(never)]` (simd128 is baseline
// on this target, so no target_feature is lost and the rule-zero trap does not
// apply here): inlined, they bloated a Horner loop to ~1000 wasm lines around
// what should be a load + branch + relaxed_madd body, which hurts the JIT.

#[inline(never)]
fn f64x2_fmadd_emulated(x: v128, m: v128, a: v128) -> v128 {
    crate::backend::generic::polyfills::fmadd_ro::<crate::backend::wasm::registers::F64x2Wasm>(x, m, a)
}

#[inline(never)]
fn f32x4_fmadd_emulated(x: v128, m: v128, a: v128) -> v128 {
    crate::backend::generic::polyfills::fmadd_widen_ro::<crate::backend::wasm::registers::F32x4Wasm>(x, m, a)
}

/// `x*m + a` with a true single rounding: the engine's relaxed madd when it is
/// a genuine fused FMA (single instruction), else the round-to-odd emulation.
/// Both branches are bit-identical for every input, so which one runs only
/// affects speed.
#[inline(always)]
pub fn f64x2_fmadd_auto(x: v128, m: v128, a: v128) -> v128 {
    if relaxed_fma_flags() & RELAXED_F64_MADD_FUSED != 0 {
        f64x2_relaxed_madd(x, m, a)
    } else {
        f64x2_fmadd_emulated(x, m, a)
    }
}

/// `-(x*m) + a`, single-rounded, see [`f64x2_fmadd_auto`].
#[inline(always)]
pub fn f64x2_fnmadd_auto(x: v128, m: v128, a: v128) -> v128 {
    if relaxed_fma_flags() & RELAXED_F64_NMADD_FUSED != 0 {
        f64x2_relaxed_nmadd(x, m, a)
    } else {
        f64x2_fmadd_emulated(f64x2_neg(x), m, a)
    }
}

/// `x*m + a` with a true single rounding, see [`f64x2_fmadd_auto`].
#[inline(always)]
pub fn f32x4_fmadd_auto(x: v128, m: v128, a: v128) -> v128 {
    if relaxed_fma_flags() & RELAXED_F32_MADD_FUSED != 0 {
        f32x4_relaxed_madd(x, m, a)
    } else {
        f32x4_fmadd_emulated(x, m, a)
    }
}

/// `-(x*m) + a`, single-rounded, see [`f64x2_fmadd_auto`].
#[inline(always)]
pub fn f32x4_fnmadd_auto(x: v128, m: v128, a: v128) -> v128 {
    if relaxed_fma_flags() & RELAXED_F32_NMADD_FUSED != 0 {
        f32x4_relaxed_nmadd(x, m, a)
    } else {
        f32x4_fmadd_emulated(f32x4_neg(x), m, a)
    }
}

#[inline(always)]
pub fn f32x4_maddx(x: v128, m: v128, a: v128) -> v128 {
    // 1. Split 128-bit packed float (4 lanes) into two sets of doubles (2 lanes each)

    // Low 2 floats -> doubles (Lanes 0, 1)
    // Equivalent to: _mm_cvtps_pd(x)
    let x_lo = f64x2_promote_low_f32x4(x);
    let m_lo = f64x2_promote_low_f32x4(m);
    let a_lo = f64x2_promote_low_f32x4(a);

    // High 2 floats -> doubles
    // We shuffle lanes 2,3 to positions 0,1, then promote.
    // Equivalent to: _mm_cvtps_pd(_mm_movehl_ps(x, x))
    let x_hi_src = i32x4_shuffle::<2, 3, 2, 3>(x, x);
    let m_hi_src = i32x4_shuffle::<2, 3, 2, 3>(m, m);
    let a_hi_src = i32x4_shuffle::<2, 3, 2, 3>(a, a);

    let x_hi = f64x2_promote_low_f32x4(x_hi_src);
    let m_hi = f64x2_promote_low_f32x4(m_hi_src);
    let a_hi = f64x2_promote_low_f32x4(a_hi_src);

    // 2. Perform operation in f64
    let res_lo = f64x2_add(f64x2_mul(x_lo, m_lo), a_lo);
    let res_hi = f64x2_add(f64x2_mul(x_hi, m_hi), a_hi);

    // 3. Convert back to f32.
    // f32x4_demote_f64x2_zero converts [f64; 2] to [f32; 2] and fills the upper 64 bits with zero.
    // Result format: [A, B, 0, 0]
    let out_lo = f32x4_demote_f64x2_zero(res_lo);
    let out_hi = f32x4_demote_f64x2_zero(res_hi);

    // 4. Shuffle high results back into the upper lanes
    // We take the lower 2 lanes from out_lo (indices 0, 1)
    // And the lower 2 lanes from out_hi (indices 4, 5 referring to the second arg)
    // Equivalent to: _mm_movelh_ps(out_lo, out_hi)
    i32x4_shuffle::<0, 1, 4, 5>(out_lo, out_hi)
}

#[inline(always)]
pub fn f64x2_maddx(x: v128, m: v128, a: v128) -> v128 {
    // Constants for Veltkamp's splitting (2^27 + 1)
    let splitter = f64x2_splat(134217729.0);

    // 1. Veltkamp's Split for 'x'
    let c_x = f64x2_mul(x, splitter);
    let x_h = f64x2_sub(c_x, f64x2_sub(c_x, x));
    let x_l = f64x2_sub(x, x_h);

    // 2. Veltkamp's Split for 'm'
    let c_m = f64x2_mul(m, splitter);
    let m_h = f64x2_sub(c_m, f64x2_sub(c_m, m));
    let m_l = f64x2_sub(m, m_h);

    // 3. Dekker's Exact Product
    // p = x * m
    let p = f64x2_mul(x, m);

    // Calculate error term 'e'
    let t1 = f64x2_mul(x_h, m_h);
    let t2 = f64x2_sub(t1, p);
    let t3 = f64x2_mul(x_h, m_l);
    let t4 = f64x2_mul(x_l, m_h);
    let t5 = f64x2_mul(x_l, m_l);

    let e = f64x2_add(f64x2_add(f64x2_add(t2, t3), t4), t5);

    // 4. Knuth's TwoSum (Adding 'a' to the exact product)
    let sum = f64x2_add(p, a);

    // Recover rounding error: sum = p + a + err
    let v = f64x2_sub(sum, p);
    let z = f64x2_sub(sum, v); // Virtual p
    let err_a = f64x2_sub(a, v);
    let err_p = f64x2_sub(p, z);
    let err_add = f64x2_add(err_p, err_a);

    // 5. Final Combination
    let total_error = f64x2_add(e, err_add);

    f64x2_add(sum, total_error)
}

#[inline(always)]
pub fn i32x4_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping addition
    let sum = i32x4_add(a, b);

    // 2. Detect Overflow
    // Logic: ~(a ^ b) checks if signs are SAME
    //        (a ^ sum) checks if sign CHANGED
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i32x4_splat(i32::MIN); // 0x80000000
    let sign_check = v128_and(
        v128_not(v128_xor(a, b)), // Inputs have same sign
        v128_xor(a, sum),         // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create a mask of 0xFFFFFFFF where overflow occurred, 0 otherwise
    let mask = i32x4_shr(overflow_bits, 31);

    // 3. Calculate Saturation Target
    // If a is positive (sign bit 0): Target is INT_MAX
    // If a is negative (sign bit 1): Target is INT_MIN
    // Formula: INT_MAX ^ (a >> 31)
    //   pos: 0x7FFFFFFF ^ 0x00000000 = 0x7FFFFFFF (MAX)
    //   neg: 0x7FFFFFFF ^ 0xFFFFFFFF = 0x80000000 (MIN)
    let max = i32x4_splat(i32::MAX);
    let a_sign_extended = i32x4_shr(a, 31);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, sum, mask)
}

#[inline(always)]
pub fn i32x4_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping subtraction
    let diff = i32x4_sub(a, b);

    // 2. Detect Overflow
    // Logic: (a ^ b) checks if signs are DIFFERENT
    //        (a ^ diff) checks if sign CHANGED from a
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i32x4_splat(i32::MIN);
    let sign_check = v128_and(
        v128_xor(a, b),    // Inputs have different signs
        v128_xor(a, diff), // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create mask
    let mask = i32x4_shr(overflow_bits, 31);

    // 3. Calculate Saturation Target (Same as add)
    let max = i32x4_splat(i32::MAX);
    let a_sign_extended = i32x4_shr(a, 31);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, diff, mask)
}

#[inline(always)]
pub fn i64x2_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping addition
    let sum = i64x2_add(a, b);

    // 2. Detect Overflow
    // Logic: ~(a ^ b) checks if signs are SAME
    //        (a ^ sum) checks if sign CHANGED
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i64x2_splat(i64::MIN); // 0x80000000
    let sign_check = v128_and(
        v128_not(v128_xor(a, b)), // Inputs have same sign
        v128_xor(a, sum),         // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create a mask of all-ones where overflow occurred, 0 otherwise
    let mask = i64x2_shr(overflow_bits, 63);

    // 3. Calculate Saturation Target
    // If a is positive (sign bit 0): Target is INT_MAX
    // If a is negative (sign bit 1): Target is INT_MIN
    // Formula: INT_MAX ^ (a >> 63)
    //   pos: 0x7FFFFFFFFFFFFFFF ^ 0x0000000000000000 = INT_MAX
    //   neg: 0x7FFFFFFFFFFFFFFF ^ 0xFFFFFFFFFFFFFFFF = INT_MIN
    let max = i64x2_splat(i64::MAX);
    let a_sign_extended = i64x2_shr(a, 63);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, sum, mask)
}

#[inline(always)]
pub fn i64x2_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Standard wrapping subtraction
    let diff = i64x2_sub(a, b);

    // 2. Detect Overflow
    // Logic: (a ^ b) checks if signs are DIFFERENT
    //        (a ^ diff) checks if sign CHANGED from a
    //        Both true + Sign Bit set = Overflow
    let sign_bit = i64x2_splat(i64::MIN);
    let sign_check = v128_and(
        v128_xor(a, b),    // Inputs have different signs
        v128_xor(a, diff), // Result has different sign than a
    );
    let overflow_bits = v128_and(sign_check, sign_bit);

    // Create mask
    let mask = i64x2_shr(overflow_bits, 63);

    // 3. Calculate Saturation Target (Same as add)
    let max = i64x2_splat(i64::MAX);
    let a_sign_extended = i64x2_shr(a, 63);
    let sat_val = v128_xor(max, a_sign_extended);

    // 4. Select Result
    v128_bitselect(sat_val, diff, mask)
}

#[inline(always)]
pub fn u32x4_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Calculate remaining space until overflow
    // u32::MAX - a is equivalent to bitwise NOT (!a)
    let remaining = v128_not(a);

    // 2. Clamp b to the remaining space
    // If b > remaining, we add remaining (resulting in MAX)
    // If b <= remaining, we add b (normal behavior)
    let safe_b = u32x4_min(b, remaining);

    // 3. Standard wrapping addition
    u32x4_add(a, safe_b)
}

#[inline(always)]
pub fn u32x4_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Clamp b so it cannot exceed a
    // If b > a, we subtract a (resulting in 0)
    // If b <= a, we subtract b (normal behavior)
    let safe_b = u32x4_min(a, b);

    // 2. Standard wrapping subtraction
    u32x4_sub(a, safe_b)
}

#[inline(always)]
pub fn u64x2_saturating_add(a: v128, b: v128) -> v128 {
    // 1. Calculate remaining space until overflow
    // u64::MAX - a is equivalent to bitwise NOT (!a)
    let remaining = v128_not(a);

    // 2. Clamp b to the remaining space
    // If b > remaining, we add remaining (resulting in MAX)
    // If b <= remaining, we add b (normal behavior)
    let safe_b = u64x2_min(b, remaining);

    // 3. Standard wrapping addition
    u64x2_add(a, safe_b)
}

#[inline(always)]
pub fn u64x2_saturating_sub(a: v128, b: v128) -> v128 {
    // 1. Clamp b so it cannot exceed a
    // If b > a, we subtract a (resulting in 0)
    // If b <= a, we subtract b (normal behavior)
    let safe_b = u64x2_min(a, b);

    // 2. Standard wrapping subtraction
    u64x2_sub(a, safe_b)
}
