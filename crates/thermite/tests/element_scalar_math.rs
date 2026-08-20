//! `FloatElement` scalar ops must agree bit-for-bit with the `std` equivalents on every
//! rung of the `element::float::arch` ladder.
//!
//! Thermite is built here without `std`, so these exercise whichever arch rung the
//! compilation unit selected. Re-run under `-C target-cpu=native` to cover the SSE4.1
//! and FMA3 rungs, and under an aarch64/wasm target for those.

use thermite::element::FloatElement;
use thermite::prelude::*;
use thermite::vector::ops::MulAddExt;

/// Bit-equality, with all NaNs treated as equal (we never promise a payload).
fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn cases_f64() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        // The classic `trunc(x + 0.5)` trap: this is the f64 just below 0.5, so it must
        // round to zero, not one.
        0.499_999_999_999_999_94,
        -0.499_999_999_999_999_94,
        (1u64 << 52) as f64,
        (1u64 << 52) as f64 - 0.5,
        -((1u64 << 52) as f64 - 0.5),
        (1u64 << 53) as f64,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1), // smallest subnormal
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    // Deterministic spread across exponents and fractions.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = f64::from_bits(state);
        if x.is_finite() {
            v.push(x);
        }
        v.push((state as i64 as f64) / 65_536.0);
    }

    v
}

#[test]
fn float_element_f64_matches_std() {
    for &x in &cases_f64() {
        assert!(same_f64(FloatElement::sqrt(x), x.sqrt()), "sqrt({x:e})");
        assert!(same_f64(FloatElement::floor(x), x.floor()), "floor({x:e})");
        assert!(same_f64(FloatElement::ceil(x), x.ceil()), "ceil({x:e})");
        assert!(same_f64(FloatElement::trunc(x), x.trunc()), "trunc({x:e})");
        assert!(same_f64(FloatElement::round(x), x.round_ties_even()), "round({x:e})");
        assert!(same_f64(FloatElement::fract(x), x - x.trunc()), "fract({x:e})");
        assert!(same_f64(FloatElement::next_up(x), x.next_up()), "next_up({x:e})");
        assert!(same_f64(FloatElement::next_down(x), x.next_down()), "next_down({x:e})");
    }
}

#[test]
fn float_element_f32_matches_std() {
    for &x in &cases_f64() {
        let x = x as f32;
        assert!(same_f32(FloatElement::sqrt(x), x.sqrt()), "sqrtf({x:e})");
        assert!(same_f32(FloatElement::floor(x), x.floor()), "floorf({x:e})");
        assert!(same_f32(FloatElement::ceil(x), x.ceil()), "ceilf({x:e})");
        assert!(same_f32(FloatElement::trunc(x), x.trunc()), "truncf({x:e})");
        assert!(same_f32(FloatElement::round(x), x.round_ties_even()), "roundf({x:e})");
        assert!(same_f32(FloatElement::next_up(x), x.next_up()), "next_upf({x:e})");
        assert!(same_f32(FloatElement::next_down(x), x.next_down()), "next_downf({x:e})");
    }
}

/// Every f32 that rounding can distinguish, exhaustively over the low exponents where
/// fractions exist at all.
///
/// Separate from the rest so emulated targets (qemu, wasmtime, SDE) can filter it out.
///
/// Slow (~2 min) at a pre-SSE4.1 baseline, and the cost is on the REFERENCE side, not
/// thermite's: `f32::round_ties_even` has no x86 instruction below SSE4.1, so it becomes
/// a libcall into the C runtime's `rintf`. Under `-C target-cpu=native` both sides are
/// single instructions and this drops to a few seconds.
#[test]
fn float_element_f32_rounding_exhaustive() {
    for bits in 0u32..=0x4B80_0000 {
        let x = f32::from_bits(bits);
        assert!(same_f32(FloatElement::floor(x), x.floor()), "floorf({x:e})");
        assert!(same_f32(FloatElement::ceil(x), x.ceil()), "ceilf({x:e})");
        assert!(same_f32(FloatElement::trunc(x), x.trunc()), "truncf({x:e})");
        assert!(same_f32(FloatElement::round(x), x.round_ties_even()), "roundf({x:e})");

        let neg = f32::from_bits(bits | 0x8000_0000);
        assert!(
            same_f32(FloatElement::round(neg), neg.round_ties_even()),
            "roundf({neg:e})"
        );
    }
}

/// `mul_add` must stay single-rounding no matter which rung supplied it.
#[test]
fn mul_add_is_single_rounding() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..200_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let a = f64::from_bits(state) / 4.0;
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let b = f64::from_bits(state) / 4.0;
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let c = f64::from_bits(state) / 4.0;

        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            continue;
        }

        assert!(
            same_f64(MulAddExt::mul_add(a, b, c), a.mul_add(b, c)),
            "mul_add({a:e}, {b:e}, {c:e})"
        );
    }
}

/// The one-lane scalar backend must break ties the same way the SIMD backends do.
///
/// Regression: `libm::round` breaks ties away from zero (and sits out of line), while
/// every SIMD width uses its native nearest-integer instruction, which breaks them to
/// even. Routing the scalar seed through `libm` makes `0.5` round to `1` at one lane and
/// `0` at every other width.
#[test]
fn scalar_seed_agrees_with_simd_on_ties() {
    for x in [0.5f64, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5] {
        let scalar = Vector::<f64>::splat(x).round().extract::<0>();
        let simd = thermite::dispatch_dyn!(for<S> |x: f64| -> f64 { f64xN::splat(x).round().extract::<0>() });
        assert!(same_f64(scalar, simd), "round({x}): Vector<f64>={scalar}, f64xN={simd}");

        let x = x as f32;
        let scalar = Vector::<f32>::splat(x).round().extract::<0>();
        let simd = thermite::dispatch_dyn!(for<S> |x: f32| -> f32 { f32xN::splat(x).round().extract::<0>() });
        assert!(
            same_f32(scalar, simd),
            "roundf({x}): Vector<f32>={scalar}, f32xN={simd}"
        );
    }
}

/// f64 cannot be swept exhaustively, so walk every exponent across the range where the
/// masking algorithms change behaviour and hammer each with random significands.
///
/// The interesting boundaries are `e < 0` (|x| < 1, the constant-result paths), `e >= 0`
/// (significand masking), and `e >= 52` (no fractional part). Both signs, and the
/// already-exact case where the masked-off bits are zero.
#[test]
fn float_element_f64_rounding_exponent_sweep() {
    let mut state = 0x853C_49E6_748F_EA9Bu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for e in -4i64..=56 {
        let biased = ((e + 1023) as u64) << 52;

        for _ in 0..8_000 {
            let sig = next() & ((1u64 << 52) - 1);
            for &(sign, extra) in &[(0u64, sig), (1u64 << 63, sig), (0, 0), (1u64 << 63, 0)] {
                let x = f64::from_bits(sign | biased | extra);

                assert!(same_f64(FloatElement::floor(x), x.floor()), "floor({x:e})");
                assert!(same_f64(FloatElement::ceil(x), x.ceil()), "ceil({x:e})");
                assert!(same_f64(FloatElement::trunc(x), x.trunc()), "trunc({x:e})");
                assert!(same_f64(FloatElement::round(x), x.round_ties_even()), "round({x:e})");
            }
        }
    }
}
