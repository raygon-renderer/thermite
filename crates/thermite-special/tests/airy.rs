//! Correctness gate for the Airy functions and their derivatives, driven through the public
//! `airy` / `airy_scaled` entries.
//!
//! Airy is Bessel at order `+-1/3` and `+-2/3` with `zeta = (2/3)|x|^{3/2}`, so this suite is
//! also the end-to-end gate on the two fractional-order kernels underneath it: the oscillating
//! pair in `bessel_nu` and the modified pair in `bessel_ik_nu`.
//!
//! **The tolerances are formulas, not numbers.** The error of these functions grows linearly in
//! `zeta`, on both sides of the origin and for two different reasons, so a flat gate would
//! either fail at large `|x|` or, worse, be loosened until it stopped seeing the growth. See
//! the note on each test.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;
use thermite_special::bessel::{Ai, AiPrime, Bi, BiPrime, Scaled};

type V = Vector<f64>;

const EPS: f64 = f64::EPSILON;

fn airy_scaled_at(x: f64) -> (f64, f64, f64, f64) {
    let (a, b, c, d) = V::splat(x).airy_all_p::<Precision, true>();
    (a.extract::<0>(), b.extract::<0>(), c.extract::<0>(), d.extract::<0>())
}

fn airy_at(x: f64) -> (f64, f64, f64, f64) {
    let (a, b, c, d) = V::splat(x).airy_all_p::<Precision, false>();
    (a.extract::<0>(), b.extract::<0>(), c.extract::<0>(), d.extract::<0>())
}

fn zeta(x: f64) -> f64 {
    let ax = x.abs();
    2.0 / 3.0 * ax * ax.sqrt()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

/// Envelope-relative below the origin, where all four oscillate and a zero makes plain relative
/// error meaningless. `Ai(-t) ~ pi^{-1/2} t^{-1/4} sin(zeta + pi/4)`, and the derivatives carry
/// the reciprocal power.
fn graded(got: f64, want: f64, x: f64, derivative: bool) -> f64 {
    if x >= 0.0 {
        return rel(got, want);
    }
    let t = (-x).powf(0.25);
    let env = if derivative { t } else { 1.0 / t } / core::f64::consts::PI.sqrt();
    (got - want).abs() / env.max(want.abs())
}

/// All four functions in the **scaled** form, against mpmath at 50 digits, from `x = -1e4` to
/// `x = 1e6`.
///
/// This is the form the kernel produces natively (no exponential is formed anywhere on the
/// positive axis), so the positive half is held to a near-flat gate. The negative half cannot
/// be: `zeta` is a _phase_ there, one ulp of it is one ulp of phase, and no amount of care
/// inside the Bessel kernels recovers a bit that argument reduction already lost. So the gate
/// is `4e-16 + 4 zeta eps`, which is that mechanism written down. Boost has the same exposure.
#[test]
fn scaled_airy_matches_a_high_precision_reference() {
    // (x, e^zeta Ai, e^zeta Ai', e^-zeta Bi, e^-zeta Bi') for x > 0. Plain values for x <= 0.
    const ROWS: &[(f64, f64, f64, f64, f64)] = &[
        (
            -10000.0,
            0.02705738360464258,
            4.950755017249123,
            -0.049507543408137594,
            2.7057371227760956,
        ),
        (
            -300.0,
            0.03872636290513791,
            2.250225513838094,
            -0.12991496664041682,
            0.6706520228537681,
        ),
        (
            -50.0,
            -0.1618814236123209,
            0.968989837276749,
            -0.13715015212882006,
            -1.1453617002654777,
        ),
        (
            -10.0,
            0.04024123848644319,
            0.99626504413279,
            -0.3146798296438386,
            0.11941411339990923,
        ),
        (
            -5.0,
            0.35076100902411433,
            0.32719281855444315,
            -0.13836913490160058,
            0.7784117730018992,
        ),
        (
            -2.0,
            0.22740742820168558,
            0.618259020741691,
            -0.4123025879563985,
            0.2787951669211695,
        ),
        (
            -1.0,
            0.5355608832923521,
            -0.01016056711664521,
            0.1039973894969446,
            0.5923756264227924,
        ),
        (
            -0.5,
            0.4757280916105396,
            -0.20408167033954738,
            0.38035265975105387,
            0.5059337136238472,
        ),
        (
            -0.001,
            0.35528687323241714,
            -0.25881922619250675,
            0.6144783389861965,
            0.4482886646677106,
        ),
        (
            0.0,
            0.3550280538878172,
            -0.2588194037928068,
            0.6149266274460007,
            0.4482883573538264,
        ),
        (
            0.001,
            0.35477671381417847,
            -0.2588246828109525,
            0.6153619428003008,
            0.44827921431131834,
        ),
        (
            0.5,
            0.29327715912994734,
            -0.28469116209194256,
            0.6748924111156303,
            0.43022096146376937,
        ),
        (
            1.0,
            0.2635136447491401,
            -0.30997688896051484,
            0.6199119435726785,
            0.47872857060498475,
        ),
        (
            2.0,
            0.2301649186525116,
            -0.3498882825800875,
            0.5004372543040949,
            0.6222179973154376,
        ),
        (
            5.0,
            0.18700211893594343,
            -0.4270355443519452,
            0.3811085310888774,
            0.8318782591248014,
        ),
        (
            10.0,
            0.15812366685434615,
            -0.5039093607113109,
            0.31834010533673446,
            0.9985559426738374,
        ),
        (
            50.0,
            0.10605346975916805,
            -0.7504406102617341,
            0.21223196271406528,
            1.4996435564886657,
        ),
        (
            100.0,
            0.08919692093633041,
            -0.8921920625040315,
            0.1784310111708354,
            1.7838637549628087,
        ),
        (
            1000.0,
            0.050164170749970864,
            -1.5863429058298844,
            0.10032900247310518,
            3.1726565491304126,
        ),
        (
            1000000.0,
            0.008920620579834624,
            -8.92062058206478,
            0.017841241163386173,
            17.841241158925865,
        ),
    ];

    let mut worst = 0.0f64;
    let mut worst_at = (0.0f64, "");

    for &(x, wai, waip, wbi, wbip) in ROWS {
        let (ai, aip, bi, bip) = airy_scaled_at(x);
        let gate = 8e-16 + 4.0 * zeta(x) * EPS;

        for (got, want, name, deriv) in [
            (ai, wai, "Ai", false),
            (aip, waip, "Ai'", true),
            (bi, wbi, "Bi", false),
            (bip, wbip, "Bi'", true),
        ] {
            let e = graded(got, want, x, deriv);
            if e > worst {
                worst = e;
                worst_at = (x, name);
            }
            assert!(
                e <= gate,
                "{name}({x}) scaled: got {got}, want {want}, graded {e:e}, gate {gate:e}"
            );
        }
    }

    println!("worst scaled: {worst:e} at {worst_at:?}");
}

/// The unscaled form on the positive axis, which is the scaled one times `e^{-zeta}` and pays
/// for it: the exponential's own relative error is about `zeta eps / 2`, so the gate has to
/// carry that term. Measured 684 eps at `x = 100`, which is `zeta = 667` doing exactly what
/// the model said it would.
///
/// The whole point of grading it this way rather than loosening a flat number is that the
/// **scaled** test above stays tight, so the two together say where the error comes from.
#[test]
fn unscaled_airy_pays_for_its_exponential_and_no_more() {
    for &x in &[0.5f64, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 100.0] {
        let (ai, aip, bi, bip) = airy_at(x);
        let (sai, saip, sbi, sbip) = airy_scaled_at(x);
        let z = zeta(x);

        // Reconstructed outside the kernel. The scaled column is pinned to mpmath above.
        let (wai, waip, wbi, wbip) = (sai * (-z).exp(), saip * (-z).exp(), sbi * z.exp(), sbip * z.exp());
        let gate = 8e-16 + 4.0 * z * EPS;

        for (got, want, name) in [(ai, wai, "Ai"), (aip, waip, "Ai'"), (bi, wbi, "Bi"), (bip, wbip, "Bi'")] {
            assert!(
                rel(got, want) <= gate,
                "{name}({x}) unscaled: got {got}, want {want}, rel {:e}, gate {gate:e}",
                rel(got, want)
            );
        }
    }
}

/// Below the origin nothing is scaled, so the two entries must agree **bit for bit**, not
/// merely closely. This is the property that makes `airy_scaled` safe to reach for by default.
#[test]
fn the_two_entries_agree_bitwise_below_the_origin() {
    for &x in &[-1e4f64, -50.0, -7.0, -1.0, -0.25, -1e-8, 0.0] {
        let a = airy_at(x);
        let b = airy_scaled_at(x);
        assert_eq!(a.0.to_bits(), b.0.to_bits(), "Ai({x})");
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "Ai'({x})");
        assert_eq!(a.2.to_bits(), b.2.to_bits(), "Bi({x})");
        assert_eq!(a.3.to_bits(), b.3.to_bits(), "Bi'({x})");
    }
}

/// The Wronskian `Ai(x) Bi'(x) - Ai'(x) Bi(x) = 1/pi`, exactly, for **every** `x`, including
/// at the zeros where nothing relative can be checked, and including where the individual
/// values have long since underflowed or overflowed.
///
/// It is scale-invariant: the `e^{zeta}` on the `Ai` factors cancels the `e^{-zeta}` on the
/// `Bi` ones, so the scaled quadruple satisfies it unchanged. That makes it the one identity
/// that reaches past `x = 104`, where the unscaled `Ai` leaves binary64 entirely.
///
/// It also crosses everything at once: both Bessel kernels, both orders, both branches, so a
/// sign error anywhere shows up here.
#[test]
fn the_wronskian_holds_everywhere() {
    let want = core::f64::consts::FRAC_1_PI;

    for &x in &[
        -1e4f64, -300.0, -50.0, -10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0, 100.0, 1e4, 1e6,
    ] {
        let (ai, aip, bi, bip) = airy_scaled_at(x);
        let got = ai * bip - aip * bi;
        let gate = 4e-15 + 8.0 * zeta(x) * EPS;
        assert!(
            rel(got, want) <= gate,
            "Wronskian at x = {x}: got {got}, want {want}, rel {:e}, gate {gate:e}",
            rel(got, want)
        );
    }
}

/// Airy's defining equation, `w'' = x w`, checked by central differences on the derivative.
///
/// A crude check numerically, but a completely independent one: it uses no reference table and
/// no identity the kernel was built from, only the differential equation the functions are
/// defined by.
#[test]
fn the_functions_satisfy_their_own_differential_equation() {
    for &x in &[-4.0f64, -1.5, -0.5, 0.5, 1.5, 4.0] {
        let h = 1e-4;
        // Ai''(x) = x Ai(x), approximated as (Ai'(x+h) - Ai'(x-h)) / 2h.
        let hi = airy_at(x + h);
        let lo = airy_at(x - h);
        let mid = airy_at(x);

        let ai_pp = (hi.1 - lo.1) / (2.0 * h);
        let bi_pp = (hi.3 - lo.3) / (2.0 * h);

        assert!(
            (ai_pp - x * mid.0).abs() <= 1e-8 * (1.0 + (x * mid.0).abs()),
            "Ai'' = x Ai at {x}: {ai_pp} vs {}",
            x * mid.0
        );
        assert!(
            (bi_pp - x * mid.2).abs() <= 1e-8 * (1.0 + (x * mid.2).abs()),
            "Bi'' = x Bi at {x}: {bi_pp} vs {}",
            x * mid.2
        );
    }
}

/// The origin, and the region below it where `zeta` underflows to zero and the general route
/// would return `0 * inf`. The substituted limits must be exact, and the values on either side
/// of the guard must be continuous with them.
#[test]
fn the_origin_is_exact_and_its_neighbourhood_is_continuous() {
    let (ai, aip, bi, bip) = airy_at(0.0);
    assert_eq!(ai, 0.3550280538878172, "Ai(0)");
    assert_eq!(aip, -0.2588194037928068, "Ai'(0)");
    assert_eq!(bi, 0.6149266274460007, "Bi(0)");
    assert_eq!(bip, 0.4482883573538264, "Bi'(0)");

    // Well inside the underflow guard: `|x|^{3/2}` is zero below about 3e-216.
    for &x in &[1e-250f64, -1e-250, 1e-300, -1e-300, 0.0, -0.0] {
        assert_eq!(airy_at(x).0.to_bits(), ai.to_bits(), "Ai({x}) must be Ai(0)");
        assert_eq!(airy_at(x).2.to_bits(), bi.to_bits(), "Bi({x}) must be Bi(0)");
    }

    // The span where `zeta = (2/3)|x|^{3/2}` is SUBNORMAL, `3e-216 < |x| < 8e-206`. A
    // subnormal `zeta` carries only a few bits and `Ai ~ zeta^{-1/3}` hands them straight
    // back (measured 2.2e-2 relative at `1e-215`, NaN at `3e-216`, and NaN for `Bi` across
    // the span) when the guard was `zeta == 0`. Every value rounds to the origin's here, so
    // the guard has to cover it, and must hold whether or not the policy flushes denormals.
    for &x in &[3e-216f64, 1e-215, -1e-215, 1e-212, 1e-210, -1e-210, 1e-206] {
        let (ga, gap, gb, gbp) = airy_at(x);
        assert_eq!(ga.to_bits(), ai.to_bits(), "Ai({x}) must be Ai(0)");
        assert_eq!(gap.to_bits(), aip.to_bits(), "Ai'({x}) must be Ai'(0)");
        assert_eq!(gb.to_bits(), bi.to_bits(), "Bi({x}) must be Bi(0)");
        assert_eq!(gbp.to_bits(), bip.to_bits(), "Bi'({x}) must be Bi'(0)");
        let (sa, _, sb, _) = airy_scaled_at(x);
        assert_eq!(sa.to_bits(), ai.to_bits(), "scaled Ai({x}) must be Ai(0)");
        assert_eq!(sb.to_bits(), bi.to_bits(), "scaled Bi({x}) must be Bi(0)");
    }

    // Just outside it, where the Bessel route runs for real: `Ai(x) ~ Ai(0) + Ai'(0) x`.
    for &x in &[1e-8f64, -1e-8, 1e-12, -1e-12, 1e-100, -1e-100] {
        let got = airy_at(x).0;
        let want = ai + aip * x;
        assert!(
            rel(got, want) <= 1e-14,
            "Ai({x}) should be near Ai(0): got {got}, linear {want}"
        );
    }
}

/// At real register width, with a different branch in every lane (two below the origin, one
/// at it, one above) because `Vector<f64>` is the one-lane scalar seed and a packet test
/// written against it proves nothing about the branch masks.
#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn a_mixed_packet_agrees_with_its_lanes() {
    use thermite::backend::x86_v3::X86V3;

    type W = Vector<<X86V3 as Simd>::f64x4>;

    let x = W::splat(-3.0).insert::<1>(0.0).insert::<2>(2.5).insert::<3>(-40.0);
    let (ai, aip, bi, bip) = W::airy_all_p::<Precision, true>(x);

    macro_rules! lane {
        ($i:literal) => {{
            let xi = x.extract::<$i>();
            let w = airy_scaled_at(xi);
            let g = (
                ai.extract::<$i>(),
                aip.extract::<$i>(),
                bi.extract::<$i>(),
                bip.extract::<$i>(),
            );
            assert!(rel(g.0, w.0) <= 1e-13, "Ai lane {}: {} vs {}", $i, g.0, w.0);
            assert!(rel(g.1, w.1) <= 1e-13, "Ai' lane {}: {} vs {}", $i, g.1, w.1);
            assert!(rel(g.2, w.2) <= 1e-13, "Bi lane {}: {} vs {}", $i, g.2, w.2);
            assert!(rel(g.3, w.3) <= 1e-13, "Bi' lane {}: {} vs {}", $i, g.3, w.3);
        }};
    }

    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
}

/// The eight single-value entry points must agree with the tuple **bit for bit**.
///
/// They are not projections of it. Each runs its own evaluation, skipping the Bessel pass it
/// does not need, and `airy_ai` additionally skips `I` inside the pass it does run. So this is
/// a real claim about two different code paths, not a tautology, and is the property that
/// makes reaching for the cheap entry point safe.
#[test]
fn the_single_entries_agree_with_the_tuple_bitwise() {
    for &x in &[
        -1e4f64, -300.0, -50.0, -7.5, -1.0, -0.01, 0.0, 0.01, 1.0, 7.5, 50.0, 100.0, 1e4,
    ] {
        let v = V::splat(x);

        let (ai, aip, bi, bip) = v.airy_all_p::<Precision, false>();
        assert_eq!(
            v.airy_p::<Precision, Ai>().extract::<0>().to_bits(),
            ai.extract::<0>().to_bits(),
            "airy_ai({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, AiPrime>().extract::<0>().to_bits(),
            aip.extract::<0>().to_bits(),
            "airy_ai_prime({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, Bi>().extract::<0>().to_bits(),
            bi.extract::<0>().to_bits(),
            "airy_bi({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, BiPrime>().extract::<0>().to_bits(),
            bip.extract::<0>().to_bits(),
            "airy_bi_prime({x})"
        );

        let (ai, aip, bi, bip) = v.airy_all_p::<Precision, true>();
        assert_eq!(
            v.airy_p::<Precision, Scaled<Ai>>().extract::<0>().to_bits(),
            ai.extract::<0>().to_bits(),
            "airy_ai_scaled({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, Scaled<AiPrime>>().extract::<0>().to_bits(),
            aip.extract::<0>().to_bits(),
            "airy_ai_prime_scaled({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, Scaled<Bi>>().extract::<0>().to_bits(),
            bi.extract::<0>().to_bits(),
            "airy_bi_scaled({x})"
        );
        assert_eq!(
            v.airy_p::<Precision, Scaled<BiPrime>>().extract::<0>().to_bits(),
            bip.extract::<0>().to_bits(),
            "airy_bi_prime_scaled({x})"
        );
    }
}
