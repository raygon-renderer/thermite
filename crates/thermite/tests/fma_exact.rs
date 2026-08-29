//! The emulated FMA on non-FMA backends must be bit-identical to hardware FMA.
//!
//! `mul_add` on x86_v1 (SSE2) and x86_v2 (SSE4.2) lowers to the round-to-odd
//! emulation (`fmadd_ro` / `fmadd_widen_ro`, Boldo-Melquiond 2008), which is
//! correctly rounded for every input, unlike the old Dekker/TwoSum path,
//! which was off by 1 ulp about once per 173k f64 triples.
//!
//! Ground truth is the hardware FMA instruction, runtime-detected. Tests that
//! need it skip (with a note) on machines without FMA3. The f32 tests also use
//! a portable scalar round-to-odd oracle, itself validated against hardware,
//! because `std`'s `fmaf` chain has a known subnormal-rounding bug
//! (rust-lang/compiler-builtins#1262, also in musl) and cannot serve as truth.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Callers gate on `have_fma()` before reaching these.
fn hw_fma64(a: f64, b: f64, c: f64) -> f64 {
    #[target_feature(enable = "fma")]
    fn inner(a: f64, b: f64, c: f64) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c))) }
    }
    unsafe { inner(a, b, c) }
}

fn hw_fma32(a: f32, b: f32, c: f32) -> f32 {
    #[target_feature(enable = "fma")]
    fn inner(a: f32, b: f32, c: f32) -> f32 {
        unsafe { _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
    }
    unsafe { inner(a, b, c) }
}

fn have_fma() -> bool {
    let ok = std::is_x86_feature_detected!("fma");
    if !ok {
        eprintln!("skipping: no hardware FMA on this machine to use as ground truth");
    }
    ok
}

/// Portable f32 FMA oracle: exact product in f64, round-to-odd sum, single
/// narrowing (BM 2008 Theorem 3). Scalar, independent of the vector backends.
fn oracle_fma32(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64; // exact: 24 + 24 <= 53
    let s = p + c as f64;
    let bb = s - p;
    let err = (p - (s - bb)) + (c as f64 - bb);

    let mut bits = s.to_bits();
    if err != 0.0 && s.is_finite() && bits & 1 == 0 {
        if (err > 0.0) ^ (s < 0.0) { bits += 1 } else { bits -= 1 }
    }
    f64::from_bits(bits) as f32
}

fn bit_eq64(got: f64, want: f64) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

fn bit_eq32(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Random finite f64 with unbiased exponent uniform in `[-emag, emag]`.
    fn f64_ranged(&mut self, emag: i32) -> f64 {
        let r = self.next();
        let sign = r & (1 << 63);
        let mantissa = self.next() & ((1u64 << 52) - 1);
        let exp = (self.next() % (2 * emag as u64 + 1)) as i64 - emag as i64;
        f64::from_bits(sign | (((exp + 1023) as u64) << 52) | mantissa)
    }

    /// Random finite f64 with unbiased exponent uniform in `[lo, hi]`.
    fn f64_ranged_low(&mut self, lo: i32, hi: i32) -> f64 {
        let r = self.next();
        let sign = r & (1 << 63);
        let mantissa = self.next() & ((1u64 << 52) - 1);
        let span = (hi - lo + 1) as u64;
        let exp = (self.next() % span) as i64 + lo as i64;
        f64::from_bits(sign | (((exp + 1023) as u64) << 52) | mantissa)
    }

    /// Random finite f32 with unbiased exponent uniform in `[-emag, emag]`.
    fn f32_ranged(&mut self, emag: i32) -> f32 {
        let r = self.next();
        let sign = (r as u32) & (1 << 31);
        let mantissa = (self.next() as u32) & ((1u32 << 23) - 1);
        let exp = (self.next() % (2 * emag as u64 + 1)) as i64 - emag as i64;
        f32::from_bits(sign | (((exp + 127) as u32) << 23) | mantissa)
    }
}

/// The f32 scalar oracle must agree with hardware everywhere, including the
/// subnormal-product zone where libm's `fmaf` is wrong.
#[test]
fn f32_oracle_matches_hardware() {
    if !have_fma() {
        return;
    }

    // The reproducer that exposed the libm/musl bug (fearless_simd #323).
    let bad_a = f32::from_bits(0x19ff_e002);
    let bad_b = f32::from_bits(0x1a00_1001);
    for c_bits in [0u32, 1, 0x8000_0001, 0x0080_0000, 0x3f80_0000, 0x8000_0000] {
        let c = f32::from_bits(c_bits);
        let want = hw_fma32(bad_a, bad_b, c);
        let got = oracle_fma32(bad_a, bad_b, c);
        assert!(
            bit_eq32(got, want),
            "oracle({bad_a:e}, {bad_b:e}, {c:e}) = {got:e}, hw = {want:e}"
        );
    }

    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for i in 0..2_000_000u32 {
        // Alternate between full-range draws and the subnormal-product zone.
        let emag = if i % 2 == 0 { 126 } else { 70 };
        let a = rng.f32_ranged(emag);
        let b = if i % 2 == 0 {
            rng.f32_ranged(emag)
        } else {
            rng.f32_ranged(126 - 60) * f32::from_bits(0x0080_0000)
        };
        let c = rng.f32_ranged(126);

        let want = hw_fma32(a, b, c);
        let got = oracle_fma32(a, b, c);
        assert!(
            bit_eq32(got, want),
            "case {i}: oracle({:#010x}, {:#010x}, {:#010x}) = {:#010x}, hw = {:#010x}",
            a.to_bits(),
            b.to_bits(),
            c.to_bits(),
            got.to_bits(),
            want.to_bits(),
        );
    }
}

macro_rules! check_backend {
    ($name:ident, $backend:path) => {
        mod $name {
            use super::*;
            use thermite::vector::ops::MulAddExt;
            use $backend::*;

            fn emul64(a: f64, b: f64, c: f64) -> f64 {
                f64x2::splat(a)
                    .mul_add(f64x2::splat(b), f64x2::splat(c))
                    .extract::<0>()
            }

            fn emul32(a: f32, b: f32, c: f32) -> f32 {
                f32x4::splat(a)
                    .mul_add(f32x4::splat(b), f32x4::splat(c))
                    .extract::<0>()
            }

            fn assert_case64(a: f64, b: f64, c: f64, tag: &str) {
                let want = hw_fma64(a, b, c);
                let got = emul64(a, b, c);
                assert!(
                    super::bit_eq64(got, want),
                    "{tag}: mul_add({:#018x}, {:#018x}, {:#018x}) = {:#018x}, hw fma = {:#018x}",
                    a.to_bits(),
                    b.to_bits(),
                    c.to_bits(),
                    got.to_bits(),
                    want.to_bits(),
                );
            }

            #[test]
            fn not_hardware_fma_backend() {
                assert!(
                    !matches!(<f64x2 as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True),
                    "this backend must lack hardware FMA, or these tests prove nothing"
                );
            }

            /// The case a naive double-rounded emulation gets wrong: the exact
            /// product lands on a midpoint, and a tiny `c` must tip the rounding.
            #[test]
            fn f64_midpoint_correction() {
                if !super::have_fma() {
                    return;
                }

                let a = 1.0 + 2.0_f64.powi(-27);
                let b = 1.0 - 2.0_f64.powi(-27);
                // a * b = 1 - 2^-54 exactly: the midpoint between 1 - 2^-53 and 1.
                let tiny = 2.0_f64.powi(-150);

                for c in [tiny, -tiny, 0.0, -0.0] {
                    assert_case64(a, b, c, "midpoint");
                    assert_case64(-a, b, -c, "midpoint, negated");
                }

                // The negative perturbation must differ from unfused mul-then-add,
                // or the case cannot catch an unfused lowering.
                assert_ne!(hw_fma64(a, b, -tiny).to_bits(), (a * b + -tiny).to_bits());
            }

            /// Exercise the integer-add Veltkamp split around its carry boundaries.
            #[test]
            fn f64_split_boundaries() {
                if !super::have_fma() {
                    return;
                }

                let low_bits = [
                    0u64,
                    1,
                    (1 << 26) - 1,
                    1 << 26,
                    (1 << 26) + 1,
                    (1 << 27) - 1,
                    0x0555_5555,
                ];
                let multipliers = [
                    f64::from_bits(0x3ff0_0000_0000_0001),
                    f64::from_bits(0x3fef_ffff_ffff_ffff),
                    f64::from_bits(0xbff0_0000_0400_0000),
                    f64::from_bits(0x4008_0000_0000_0001),
                ];
                let upper = ((1u64 << 52) - 1) & !((1u64 << 27) - 1);

                for exp_field in [623u64, 1023, 1423] {
                    for sign in [0u64, 1 << 63] {
                        for low in low_bits {
                            let a = f64::from_bits(sign | (exp_field << 52) | upper | low);
                            for b in multipliers {
                                let prod = a * b;
                                let neighbors = [
                                    0.0,
                                    -prod,
                                    f64::from_bits(prod.to_bits() ^ 1).copysign(prod) * -1.0,
                                ];
                                for c in neighbors {
                                    assert_case64(a, b, c, "split boundary");
                                }
                            }
                        }
                    }
                }
            }

            /// Subnormal and tiny `c` stay on the packed path (no gate on `c`),
            /// zero `a`/`b` lanes are rescued, and the product-underflow gate boundary
            /// itself is exact. (Not under `ignore_denormals`: the gate is
            /// compiled out and subnormal-scale exactness is waived.)
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_gate_and_zero_rescue() {
                if !super::have_fma() {
                    return;
                }

                let subn = f64::from_bits(1);
                let subn2 = f64::from_bits(0x000f_ffff_ffff_ffff);

                // Subnormal / tiny c against ordinary products.
                for c in [subn, -subn, subn2, -subn2, f64::MIN_POSITIVE, 0.0, -0.0] {
                    assert_case64(1.5, 3.0, c, "subnormal c");
                    assert_case64(
                        2.0_f64.powi(-500),
                        2.0_f64.powi(480),
                        c,
                        "tiny product, subnormal c",
                    );
                }

                // Zero a or b with every flavor of c.
                for z in [0.0, -0.0] {
                    for c in [1.0, -0.0, 0.0, subn, f64::MAX, f64::INFINITY, f64::NAN] {
                        assert_case64(z, 3.0, c, "zero a");
                        assert_case64(3.0, z, c, "zero b");
                    }
                    // 0 * inf = NaN must survive the zero rescue.
                    assert_case64(z, f64::INFINITY, 1.0, "zero times inf");
                }

                // Exact-zero results must carry the IEEE sign: -0 only when the
                // product and c are both negatively signed zeros. Exact
                // cancellation of nonzero values gives +0.
                assert_case64(-0.0, 3.0, -0.0, "signed zero, both negative");
                assert_case64(0.0, 3.0, -0.0, "signed zero, mixed");
                assert_case64(-0.0, -3.0, 0.0, "signed zero, positive product");
                assert_case64(2.0, 3.0, -6.0, "exact cancellation");
                assert_case64(-2.0, 3.0, 6.0, "exact cancellation, negative product");

                // Around the gate boundary Ea + Eb = BIAS + p = 1076: products with
                // representable and non-representable error terms.
                for (ea, eb) in [
                    (538i32, 538i32),
                    (537, 538),
                    (537, 537),
                    (100, 976),
                    (100, 975),
                    (-500, 1576),
                ] {
                    let a = f64::from_bits((((ea + 1023) as u64) << 52) | 0x000f_ffff_fc00_0001);
                    let b = f64::from_bits((((eb + 1023) as u64) << 52) | 0x0000_0000_0400_0001);
                    for c in [0.0, a * b * -1.0, 1.0, subn] {
                        assert_case64(a, b, c, "gate boundary");
                    }
                }
            }

            /// Non-finite inputs, genuine overflow, and the spurious-overflow shape
            /// (finite true result, overflowing intermediate).
            #[test]
            fn f64_specials_and_overflow() {
                if !super::have_fma() {
                    return;
                }

                let cases: &[(f64, f64, f64)] = &[
                    (f64::INFINITY, 2.0, -f64::MAX),
                    (f64::NEG_INFINITY, 2.0, f64::INFINITY),
                    (f64::INFINITY, 0.0, 1.0),
                    (f64::NAN, 1.0, 2.0),
                    (1.0, f64::NAN, 2.0),
                    (1.0, 2.0, f64::NAN),
                    (f64::MAX, 2.0, f64::NEG_INFINITY),
                    (f64::MAX, 2.0, -f64::MAX), // spurious: true result is MAX, t_h overflows
                    (f64::MAX, 1.5, -f64::MAX), // spurious: true result is MAX / 2
                    (f64::MAX, 2.0, 0.0),       // genuine overflow
                    (f64::MAX, f64::MAX, f64::NEG_INFINITY),
                    // Split-carry corruption: MAX has an all-ones significand, so the
                    // integer-add split rounds a_hi to infinity while u_h stays finite.
                    // The odd_round_add finite guard must keep the poisoned u_l from
                    // being stepped from inf-bits down into MAX (finite, wrong, and
                    // invisible to the post-check).
                    (f64::MAX, 1.0, 0.0),
                    (f64::MAX, 1.0, -f64::MAX),
                    (f64::MAX, 1.0, -f64::MAX / 2.0),
                    (-f64::MAX, 1.0, f64::MAX / 2.0),
                    (6.69692879491417e299, 3.0, 1.0), // old Veltkamp-overflow threshold
                    (f64::MAX, 0.5, 1.0),
                    (1.7e308, 1e-8, 2.0),
                ];

                for &(a, b, c) in cases {
                    assert_case64(a, b, c, "specials");
                }
            }

            /// One gate-failing lane must not corrupt its neighbor, and vice versa.
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_mixed_lanes() {
                if !super::have_fma() {
                    return;
                }

                let pairs: &[([f64; 2], [f64; 2], [f64; 2])] = &[
                    ([1.5, f64::from_bits(1)], [3.0, 0.5], [0.25, 1.0]),
                    ([f64::MAX, 3.0], [2.0, 7.0], [-f64::MAX, 2.0]),
                    (
                        [f64::NAN, 1.0 + 2.0_f64.powi(-27)],
                        [1.0, 1.0 - 2.0_f64.powi(-27)],
                        [1.0, -2.0_f64.powi(-150)],
                    ),
                ];

                for &(av, bv, cv) in pairs {
                    let got = f64x2::new(av).mul_add(f64x2::new(bv), f64x2::new(cv));
                    for i in 0..2 {
                        let want = hw_fma64(av[i], bv[i], cv[i]);
                        let g = got.extractv(i);
                        assert!(
                            super::bit_eq64(g, want),
                            "lane {i}: mul_add({:e}, {:e}, {:e}) = {:e}, hw = {:e}",
                            av[i],
                            bv[i],
                            cv[i],
                            g,
                            want,
                        );
                    }
                }
            }

            /// The vectorized rescue path's regimes, each pinned explicitly:
            /// destination scan across the subnormal boundary, half-min-subnormal
            /// ties, tiny-times-tiny subnormal inputs, the c-dominates and
            /// product-dominates threshold bands, and scaled midpoint corrections.
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_rescue_regimes() {
                if !super::have_fma() {
                    return;
                }

                // Destination scan: products landing at every exponent across the
                // subnormal boundary, with c = 0 and with c perturbing the tie.
                for e in -1080..=-1010i32 {
                    let ea = e / 2;
                    let eb = e - ea;
                    for mant in [
                        0u64,
                        1,
                        0x000f_ffff_ffff_ffff,
                        0x0008_0000_0000_0001,
                        0x0000_0000_5555_5555,
                    ] {
                        let a = f64::from_bits((((ea + 1023) as u64) << 52) | mant);
                        let b = f64::from_bits(((eb + 1023) as u64) << 52);
                        for c in [
                            0.0,
                            -0.0,
                            f64::from_bits(1),
                            -f64::from_bits(1),
                            2.0_f64.powi(-1074),
                            -(a * b),
                        ] {
                            assert_case64(a, b, c, "dest scan");
                            assert_case64(-a, b, -c, "dest scan, negated");
                        }
                    }
                }

                // Exact half-min-subnormal products: RN ties to zero (even), and
                // one ulp of product mantissa must tip it to the min subnormal.
                assert_case64(2.0_f64.powi(-537), 2.0_f64.powi(-538), 0.0, "half-min tie");
                assert_case64(
                    -(2.0_f64.powi(-537)),
                    2.0_f64.powi(-538),
                    0.0,
                    "half-min tie, negative",
                );
                let just_over = f64::from_bits((((-537 + 1023) as u64) << 52) | 1);
                assert_case64(just_over, 2.0_f64.powi(-538), 0.0, "just over half-min");
                // And a tiny same-sign c must break the tie upward.
                assert_case64(
                    2.0_f64.powi(-537),
                    2.0_f64.powi(-538),
                    2.0_f64.powi(-1074),
                    "half-min tie + c",
                );

                // Tiny-times-tiny: subnormal inputs normalize exactly.
                let sub_a = f64::from_bits(0x000f_ffff_ffff_ffff);
                let sub_b = f64::from_bits(0x0000_0000_0000_0003);
                for c in [0.0, f64::from_bits(1), 1.0, -1.0, f64::MAX] {
                    assert_case64(sub_a, sub_b, c, "subnormal times subnormal");
                    assert_case64(sub_a, 2.0_f64.powi(400), c, "subnormal times large");
                }

                // c-dominates threshold band: tiny product against c exponents
                // walking through the gap = ~220 cutoff (both sides must agree
                // with hardware regardless of which internal regime fires).
                let ta = 2.0_f64.powi(-500);
                let tb = 2.0_f64.powi(-600); // product 2^-1100, K ~ -1100
                for ec in [
                    -1074i32, -1022, -960, -900, -880, -870, -860, -700, -300, 0, 300, 1023,
                ] {
                    let c = if ec < -1022 {
                        f64::from_bits(1u64 << (ec + 1074))
                    } else {
                        f64::from_bits(((ec + 1023) as u64) << 52)
                    };
                    assert_case64(ta, tb, c, "c-dominates band");
                    assert_case64(ta, tb, -c, "c-dominates band, negative c");
                    assert_case64(-ta, tb, c, "c-dominates band, negative product");
                }

                // Product-dominates band: huge product, c walking down through the
                // sticky threshold, including the scaled midpoint correction where
                // c's only job is to tip the rounding of an exact-midpoint product.
                let ma = (1.0 + 2.0_f64.powi(-27)) * 2.0_f64.powi(900);
                let mb = (1.0 - 2.0_f64.powi(-27)) * 2.0_f64.powi(200);
                // ma * mb = (1 - 2^-54) * 2^1100 exactly: a midpoint at 53 bits.
                for ec in [1046i32, 990, 940, 900, 880, 860, 700, 300, 0, -300, -1000, -1074] {
                    let c = if ec < -1022 {
                        f64::from_bits(1u64 << (ec + 1074))
                    } else {
                        f64::from_bits(((ec + 1023) as u64) << 52)
                    };
                    assert_case64(ma, mb, c, "product-dominates band");
                    assert_case64(ma, mb, -c, "product-dominates band, negative c");
                }
                // True tie (c = 0) must go to even.
                assert_case64(ma, mb, 0.0, "scaled midpoint, tie to even");
                assert_case64(-ma, mb, -0.0, "scaled midpoint, tie to even, negative");

                // The min-normal boundary, the shape libm's generic fma dedicates a
                // special branch to ("min normal after rounding"): products landing
                // just below 2^-1022 that round up to exactly MIN_POSITIVE_NORMAL,
                // sit exactly on the top-subnormal/min-normal midpoint (tie rounds
                // UP: min-normal's mantissa is even), or stay top-subnormal.
                let min_sub = f64::from_bits(1);
                for b in [
                    1.0 - 2.0_f64.powi(-52),       // below halfway: top subnormal
                    1.0 - 3.0 * 2.0_f64.powi(-54), // between: rounds by position
                    1.0 - 2.0_f64.powi(-53),       // exact midpoint: tie to even = min-normal
                    1.0 - 2.0_f64.powi(-54),       // above halfway: min-normal
                ] {
                    for c in [0.0, min_sub, -min_sub] {
                        assert_case64(f64::MIN_POSITIVE, b, c, "min-normal boundary");
                        assert_case64(-f64::MIN_POSITIVE, b, -c, "min-normal boundary, negated");
                    }
                }

                // Overflow-side rescue: near-MAX operands whose split carries.
                for c in [0.0, 1.0, -f64::MAX, f64::MAX / 2.0, -(2.0_f64.powi(1000))] {
                    assert_case64(f64::MAX, 1.0, c, "split carry rescue");
                    assert_case64(f64::from_bits(0x7fe0_0000_0000_0001), 1.5, c, "near-max rescue");
                }
            }

            /// Edge cases harvested from Rust libm's fma test module (itself
            /// years of accumulated musl fixes), kept as a named corpus even
            /// though the sweeps cover the same shapes.
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_libm_regression_corpus() {
                if !super::have_fma() {
                    return;
                }

                // IEEE 754-2020: "When the exact result of (a * b) + c is non-zero
                // yet the result of fusedMultiplyAdd is zero because of rounding,
                // the zero result takes the sign of the exact result", even
                // against an opposite-signed zero c.
                let tiny = f64::from_bits(1);
                for z in [0.0, -0.0] {
                    assert_case64(tiny, tiny, z, "754-2020 zero sign ++");
                    assert_case64(tiny, -tiny, z, "754-2020 zero sign +-");
                    assert_case64(-tiny, tiny, z, "754-2020 zero sign -+");
                    assert_case64(-tiny, -tiny, z, "754-2020 zero sign --");
                }

                // libm `fma_sbb`: borrow-chain cancellation at full magnitude.
                assert_case64(-(1.0 - f64::EPSILON), f64::MIN, f64::MIN, "libm fma_sbb");

                // libm `fma_underflow`: subnormal c nearly cancelling a
                // subnormal-range product. The signed-zero outcome is decided by
                // the sub-subnormal residual.
                assert_case64(
                    1.1102230246251565e-16,
                    -9.812526705433188e-305,
                    1.0894e-320,
                    "libm fma_underflow",
                );

                // libm `fma_segfault` shapes (historical scalbn overflow crash).
                assert_case64(
                    -0.0000000000000002220446049250313,
                    -0.0000000000000002220446049250313,
                    -0.0000000000000002220446049250313,
                    "libm fma_segfault",
                );
                assert_case64(-0.992, -0.992, -0.992, "libm fma_segfault 2");

                // libm `expect_underflow` pair: results exactly at the subnormal
                // boundary from products of 2^-1070-scale operands.
                let p = 2.0_f64.powi(-1070);
                let z1 = f64::from_bits(0x000f_ffff_ffff_fff8); // 0x1.ffffffffffffp-1023
                assert_case64(p, p, z1, "libm expect_underflow 1");
                assert_case64(p, p, -(2.0_f64.powi(-1022)), "libm expect_underflow 2");
            }

            /// Random sweep forced into the rescue path: product exponents below
            /// the gate, mixed with every c magnitude class.
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_rescue_random_sweep() {
                if !super::have_fma() {
                    return;
                }

                let mut rng = super::XorShift(0x1319_8A2E_0370_7344);
                for i in 0..1_000_000u32 {
                    // Ea + Eb < 1076: draw both operands small, sometimes subnormal.
                    let a = if i % 7 == 0 {
                        f64::from_bits(rng.next() & 0x800f_ffff_ffff_ffff) // subnormal
                    } else {
                        rng.f64_ranged_low(-1022, -400)
                    };
                    let b = if i % 11 == 0 {
                        f64::from_bits(rng.next() & 0x800f_ffff_ffff_ffff)
                    } else {
                        rng.f64_ranged_low(-700, -100)
                    };
                    let c = match i % 8 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => f64::from_bits(rng.next() & 0x800f_ffff_ffff_ffff), // subnormal
                        3 => -(a * b),
                        4 => rng.f64_ranged(1000),
                        _ => rng.f64_ranged(200),
                    };

                    assert_case64(a, b, c, "rescue random");
                }
            }

            /// Random sweep across mixed exponent scales, bit-compared to hardware.
            #[test]
            #[cfg(not(feature = "ignore_denormals"))]
            fn f64_random_sweep() {
                if !super::have_fma() {
                    return;
                }

                let mut rng = super::XorShift(0x853C_49E6_748F_EA9B);
                for i in 0..1_000_000u32 {
                    let emag = [50, 300, 600, 1000][i as usize % 4];
                    let a = rng.f64_ranged(emag);
                    let b = rng.f64_ranged(emag);
                    let c = match i % 16 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => f64::from_bits(rng.next() & 0x000f_ffff_ffff_ffff), // subnormal
                        3 => -(a * b),                                           // force cancellation
                        _ => rng.f64_ranged(emag),
                    };

                    assert_case64(a, b, c, "random");
                }
            }

            /// The `ignore_denormals` contract: normal-range products and results
            /// remain bit-identical to hardware FMA. Only subnormal-scale
            /// exactness is waived. Pins the kept half of the guarantee.
            #[test]
            #[cfg(feature = "ignore_denormals")]
            fn f64_ignore_denormals_normal_contract() {
                if !super::have_fma() {
                    return;
                }

                // Midpoint correction still exact.
                let a = 1.0 + 2.0_f64.powi(-27);
                let b = 1.0 - 2.0_f64.powi(-27);
                for c in [2.0_f64.powi(-150), -(2.0_f64.powi(-150)), 0.0, -0.0] {
                    assert_case64(a, b, c, "ignore_denormals midpoint");
                }

                // Zero operands still exact without the gate's zero-rescue (the
                // split maps zeros to exact zeros).
                for z in [0.0, -0.0] {
                    for c in [1.0, -0.0, 0.0, f64::MAX, f64::NAN] {
                        assert_case64(z, 3.0, c, "ignore_denormals zero a");
                    }
                }

                // Overflow/specials still route through the rescue correctly.
                for &(x, y, z) in &[
                    (f64::MAX, 2.0, -f64::MAX),
                    (f64::MAX, 1.0, 0.0),
                    (f64::INFINITY, 2.0, -f64::MAX),
                    (f64::NAN, 1.0, 2.0),
                ] {
                    assert_case64(x, y, z, "ignore_denormals specials");
                }

                // Random sweep with products guaranteed above the (removed) gate
                // threshold: exponents in [-450, 450] keep Ea + Eb >= 1076 well
                // clear, so every result must still be bit-exact.
                let mut rng = super::XorShift(0x0D64_2C89_BDB8_57F1);
                for i in 0..1_000_000u32 {
                    let a = rng.f64_ranged(450);
                    let b = rng.f64_ranged(450);
                    let c = match i % 8 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => -(a * b),
                        _ => rng.f64_ranged(900),
                    };
                    assert_case64(a, b, c, "ignore_denormals normal sweep");
                }
            }

            /// The negated variants must compose exactly (negation is exact).
            #[test]
            fn f64_variant_composition() {
                if !super::have_fma() {
                    return;
                }

                let mut rng = super::XorShift(0xDA3E_39CB_94B9_5BDB);
                for _ in 0..100_000u32 {
                    let a = rng.f64_ranged(300);
                    let b = rng.f64_ranged(300);
                    let c = rng.f64_ranged(300);

                    let va = f64x2::splat(a);
                    let vb = f64x2::splat(b);
                    let vc = f64x2::splat(c);

                    for (got, want, tag) in [
                        (va.mul_sub(vb, vc).extract::<0>(), hw_fma64(a, b, -c), "mul_sub"),
                        (va.nmul_add(vb, vc).extract::<0>(), hw_fma64(-a, b, c), "nmul_add"),
                        (
                            va.nmul_sub(vb, vc).extract::<0>(),
                            hw_fma64(-a, b, -c),
                            "nmul_sub",
                        ),
                    ] {
                        assert!(
                            super::bit_eq64(got, want),
                            "{tag}({a:e}, {b:e}, {c:e}) = {got:e}, hw = {want:e}"
                        );
                    }
                }
            }

            /// f32 lanes against hardware, including the subnormal-product zone
            /// where the old widen-without-round-to-odd path double-rounded and
            /// libm's `fmaf` is outright wrong.
            #[test]
            fn f32_against_hardware() {
                if !super::have_fma() {
                    return;
                }

                // The libm-bug reproducer pair.
                let bad_a = f32::from_bits(0x19ff_e002);
                let bad_b = f32::from_bits(0x1a00_1001);
                for c_bits in [0u32, 1, 0x8000_0001, 0x0080_0000, 0x3f80_0000] {
                    let c = f32::from_bits(c_bits);
                    let want = hw_fma32(bad_a, bad_b, c);
                    let got = emul32(bad_a, bad_b, c);
                    assert!(
                        super::bit_eq32(got, want),
                        "libm-bug pair, c={c:e}: got {got:e}, hw {want:e}"
                    );
                }

                // f32 midpoint analogue of the f64 case.
                let a = 1.0_f32 + 2.0_f32.powi(-12);
                let b = 1.0_f32 - 2.0_f32.powi(-12);
                for c in [2.0_f32.powi(-120), -2.0_f32.powi(-120), 0.0, -0.0] {
                    let want = hw_fma32(a, b, c);
                    let got = emul32(a, b, c);
                    assert!(
                        super::bit_eq32(got, want),
                        "f32 midpoint, c={c:e}: got {got:e}, hw {want:e}"
                    );
                }

                let mut rng = super::XorShift(0xC0FF_EE12_3456_789B);
                for i in 0..1_000_000u32 {
                    let emag = [30, 90, 126][i as usize % 3];
                    let a = rng.f32_ranged(emag);
                    let b = if i % 3 == 2 {
                        // steer toward subnormal products
                        rng.f32_ranged(30) * f32::from_bits(0x0100_0000)
                    } else {
                        rng.f32_ranged(emag)
                    };
                    let c = match i % 8 {
                        0 => 0.0,
                        1 => f32::from_bits((rng.next() as u32) & 0x007f_ffff), // subnormal
                        2 => -(a * b),
                        _ => rng.f32_ranged(emag),
                    };

                    let want = hw_fma32(a, b, c);
                    let got = emul32(a, b, c);
                    assert!(
                        super::bit_eq32(got, want),
                        "case {i}: mul_add({:#010x}, {:#010x}, {:#010x}) = {:#010x}, hw = {:#010x}",
                        a.to_bits(),
                        b.to_bits(),
                        c.to_bits(),
                        got.to_bits(),
                        want.to_bits(),
                    );
                }
            }

            /// f32 specials: infinities and NaN must pass through the widen path.
            #[test]
            fn f32_specials() {
                if !super::have_fma() {
                    return;
                }

                let cases: &[(f32, f32, f32)] = &[
                    (f32::INFINITY, 2.0, -f32::MAX),
                    (f32::INFINITY, 0.0, 1.0),
                    (f32::NAN, 1.0, 2.0),
                    (1.0, 2.0, f32::NAN),
                    (f32::MAX, 2.0, f32::NEG_INFINITY),
                    (f32::MAX, 2.0, -f32::MAX),
                    (f32::MAX, 2.0, 0.0),
                    (0.0, f32::INFINITY, 5.0),
                    (-0.0, 3.0, -0.0),
                    (0.0, 3.0, -0.0),
                    (-0.0, -3.0, 0.0),
                    (2.0, 3.0, -6.0),
                    (-2.0, 3.0, 6.0),
                ];

                for &(a, b, c) in cases {
                    let want = hw_fma32(a, b, c);
                    let got = emul32(a, b, c);
                    assert!(
                        super::bit_eq32(got, want),
                        "f32 specials ({a:e}, {b:e}, {c:e}): got {got:e}, hw {want:e}"
                    );
                }
            }
        }
    };
}

check_backend!(sse2, thermite::backend::x86_v1::prelude);
check_backend!(sse42, thermite::backend::x86_v2::prelude);
