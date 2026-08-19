//! `phi::<N>`, the exponential-integrator phi-functions.
//!
//! References are mpmath at 40 digits. The kernel is a two-arm split at `|z| = N` (series
//! below, recurrence from `expm1` above), so the probes straddle that line, and the test
//! that matters most is the one showing the naive recurrence really does lose the answer
//! where the series is used instead.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{BestPrecision, MediumPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

const E: f64 = core::f64::consts::E;
const LN_2: f64 = core::f64::consts::LN_2;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
}

fn phi1(x: f64) -> f64 {
    D::splat(x).phi::<1>().extract::<0>()
}

// --- phi_1(x) = (e^x - 1)/x, the removable singularity ---

#[test]
fn phi1_values_and_limits() {
    close("phi1(0)", phi1(0.0), 1.0, 0.0); // the removable singularity
    close("phi1(1)", phi1(1.0), E - 1.0, 1e-15);
    close("phi1(-1)", phi1(-1.0), 1.0 - 1.0 / E, 1e-15);
    close("phi1(ln2)", phi1(LN_2), 1.0 / LN_2, 1e-15);

    assert_eq!(phi1(f64::INFINITY), f64::INFINITY);
    close("phi1(-inf)", phi1(f64::NEG_INFINITY), 0.0, 0.0);
}

#[test]
fn phi1_beats_the_direct_form_near_zero() {
    // The in-scattering case: an optical depth small enough that exp(x) rounds to 1.
    let x = 1e-17_f64;
    let naive = (x.exp() - 1.0) / x;
    assert_eq!(naive, 0.0, "precondition: the direct form is expected to collapse here");

    close("phi1 tiny", phi1(x), 1.0, 1e-15);
}

#[test]
fn phi1_is_the_exponential_segment_integral() {
    // integral(0..t) e^(-sigma s) ds == t * phi1(-sigma * t), checked against the
    // closed form (1 - e^(-sigma t))/sigma at a sigma where that form is still fine.
    let (sigma, t) = (0.75_f64, 2.5_f64);
    let want = (1.0 - (-sigma * t).exp()) / sigma;
    close("segment integral", t * phi1(-sigma * t), want, 1e-14);

    // And the empty-medium case, which is the one that actually breaks naive code.
    close("sigma = 0", t * phi1(0.0), t, 0.0);
}

// --- phi_N for N >= 2 ---

/// (z, phi_2, phi_3, phi_4, phi_6) from mpmath.
const REFS: &[(f64, [f64; 4])] = &[
    (
        -30.0,
        [
            0.03222222222222232619581,
            0.01559259259259258912681,
            0.005035802469135802584662,
            0.0002370768175582990399089,
        ],
    ),
    (
        -7.5,
        [
            0.1155653881665804059304,
            0.05125794824445594587595,
            0.01538782912296142943876,
            0.000643931777000795782615,
        ],
    ),
    (
        -3.0,
        [
            0.2277541187075404381088,
            0.09074862709748652063039,
            0.02530601318972671534542,
            0.0009599273914511165198619,
        ],
    ),
    (
        -2.0,
        [
            0.2838338208091531729735,
            0.1080830895954234135133,
            0.02929178853562162657671,
            0.001072947133905406644177,
        ],
    ),
    (
        -1.0,
        [
            0.3678794411714423215955,
            0.1321205588285576784045,
            0.03454610783810898826219,
            0.001212774504775654928857,
        ],
    ),
    (
        -0.5,
        [
            0.4261226388505336944152,
            0.1477547222989326111696,
            0.03782388873546811099413,
            0.001295554941872443976504,
        ],
    ),
    (
        -0.01,
        [
            0.4983374916805357390598,
            0.1662508319464260940228,
            0.04158347202405726438467,
            0.001386907239310513369891,
        ],
    ),
    (
        0.01,
        [
            0.5016708416805754216546,
            0.1670841680575421654569,
            0.04175013908754987902362,
            0.001390875498790236193367,
        ],
    ),
    (
        0.5,
        [
            0.5948850828005125873946,
            0.1897701656010251747892,
            0.04620699786871701624508,
            0.001494658141534731646984,
        ],
    ),
    (
        1.0,
        [
            0.7182818284590452353603,
            0.2182818284590452353603,
            0.05161516179237856869362,
            0.001615161792378568693621,
        ],
    ),
    (
        1.9,
        [
            1.048724222238024768995,
            0.2888022222305393521027,
            0.06428187134940667654529,
            0.001878634722827334223072,
        ],
    ),
    (
        2.0,
        [
            1.097264024732662556808,
            0.2986320123663312784038,
            0.06598267284983230586857,
            0.001912334879124743133809,
        ],
    ),
    (
        2.1,
        [
            1.148791363394024959966,
            0.3089482682828690285552,
            0.06775314362676302947073,
            0.001947046929727066395479,
        ],
    ),
    (
        3.0,
        [
            1.787281880354185304548,
            0.4290939601180617681825,
            0.08747576448379836717196,
            0.002312121979681300056143,
        ],
    ),
    (
        7.5,
        [
            31.99186514588556812273,
            4.198915352784742416365,
            0.5376331581490767666264,
            0.007706070959687290665951,
        ],
    ),
    (
        30.0,
        [
            11873860646.10384682999,
            395795354.8534615609996,
            13193178.4895598298111,
            14659.08688654795904937,
        ],
    ),
];

fn all_orders_f64<P: thermite::math::policy::Policy>(z: f64) -> [f64; 4] {
    let v = D::splat(z);
    [
        v.phi_p::<P, 2>().extract::<0>(),
        v.phi_p::<P, 3>().extract::<0>(),
        v.phi_p::<P, 4>().extract::<0>(),
        v.phi_p::<P, 6>().extract::<0>(),
    ]
}

fn all_orders_f32<P: thermite::math::policy::Policy>(z: f32) -> [f32; 4] {
    let v = F::splat(z);
    [
        v.phi_p::<P, 2>().extract::<0>(),
        v.phi_p::<P, 3>().extract::<0>(),
        v.phi_p::<P, 4>().extract::<0>(),
        v.phi_p::<P, 6>().extract::<0>(),
    ]
}

#[test]
fn f64_matches_mpmath_at_best_and_default() {
    for &(z, want) in REFS {
        for (tier, got) in [
            ("best", all_orders_f64::<BestPrecision<DefaultPolicy>>(z)),
            ("default", all_orders_f64::<DefaultPolicy>(z)),
        ] {
            for (i, n) in [2, 3, 4, 6].into_iter().enumerate() {
                close(&format!("f64 {tier} phi_{n}({z})"), got[i], want[i], 2e-15);
            }
        }
    }
}

#[test]
fn f64_lower_tiers_stay_within_budget() {
    // Medium spends 10000 ulp and Worst 100000. The series truncation takes 1/32 of that
    // and the rest is `expm1`'s own error at the tier.
    for &(z, want) in REFS {
        let medium = all_orders_f64::<MediumPrecision<DefaultPolicy>>(z);
        let worst = all_orders_f64::<WorstPrecision<DefaultPolicy>>(z);
        for (i, n) in [2, 3, 4, 6].into_iter().enumerate() {
            close(&format!("f64 medium phi_{n}({z})"), medium[i], want[i], 2e-12);
            close(&format!("f64 worst phi_{n}({z})"), worst[i], want[i], 2e-11);
        }
    }
}

#[test]
fn f32_matches_mpmath() {
    for &(z, want) in REFS {
        let best = all_orders_f32::<BestPrecision<DefaultPolicy>>(z as f32);
        let default = all_orders_f32::<DefaultPolicy>(z as f32);
        let worst = all_orders_f32::<WorstPrecision<DefaultPolicy>>(z as f32);
        for (i, n) in [2, 3, 4, 6].into_iter().enumerate() {
            // The f32 argument is rounded, so allow for phi's slope: near +30 a one-ulp
            // change in z moves phi_2 by about an ulp of the result too.
            close(&format!("f32 best phi_{n}({z})"), best[i] as f64, want[i], 1e-6);
            close(&format!("f32 default phi_{n}({z})"), default[i] as f64, want[i], 1e-6);
            close(&format!("f32 worst phi_{n}({z})"), worst[i] as f64, want[i], 5e-4);
        }
    }
}

#[test]
fn low_orders_are_exp_and_expm1_over_x() {
    for &z in &[-3.0_f64, -0.25, 0.7, 4.0] {
        let v = D::splat(z);
        assert_eq!(v.phi::<0>().extract::<0>(), v.exp().extract::<0>(), "phi_0({z}) is exp");
        // Not bit-equal: the kernel's divide is the policy's `approx_div`.
        close(
            "phi_1 is expm1/x",
            v.phi::<1>().extract::<0>(),
            (v.exp_m1() / v).extract::<0>(),
            1e-15,
        );
    }
    // (phi_1's origin, its tiny-argument behaviour and its integral identity are covered
    // in removable_singularities.rs.)
}

#[test]
fn origin_and_infinities() {
    // phi_k(0) = 1/k!, exactly: the series arm's Horner collapses to its constant term.
    assert_eq!(D::splat(0.0).phi::<2>().extract::<0>(), 0.5);
    assert_eq!(D::splat(0.0).phi::<3>().extract::<0>(), 1.0 / 6.0);
    assert_eq!(D::splat(0.0).phi::<5>().extract::<0>(), 1.0 / 120.0);
    assert_eq!(F::splat(0.0).phi::<4>().extract::<0>(), 1.0 / 24.0);

    assert_eq!(D::splat(f64::INFINITY).phi::<2>().extract::<0>(), f64::INFINITY);
    assert_eq!(D::splat(f64::INFINITY).phi::<5>().extract::<0>(), f64::INFINITY);
    assert_eq!(D::splat(f64::NEG_INFINITY).phi::<2>().extract::<0>(), 0.0);
    assert_eq!(D::splat(f64::NEG_INFINITY).phi::<5>().extract::<0>(), 0.0);
    assert!(D::splat(f64::NAN).phi::<3>().extract::<0>().is_nan());

    // Large finite arguments: expm1 overflows and the recurrence keeps the inf.
    assert_eq!(D::splat(800.0).phi::<3>().extract::<0>(), f64::INFINITY);
    assert_eq!(F::splat(100.0).phi::<3>().extract::<0>(), f32::INFINITY);
}

/// The reason the series arm exists: run the recurrence where the kernel refuses to.
#[test]
fn recurrence_alone_loses_the_answer_below_the_split() {
    fn recurrence(n: usize, z: f64) -> f64 {
        let mut p = D::splat(z).phi::<1>().extract::<0>();
        let mut fact = 1.0;
        for k in 1..n {
            p = (p - 1.0 / fact) / z;
            fact *= (k + 1) as f64;
        }
        p
    }

    // At z = 1e-3, phi_3 by recurrence: two subtractions of nearly-equal numbers, each
    // costing about 10 bits. The kernel is unaffected because 1e-3 < 3 takes the series.
    let z = 1e-3;
    let want = 0.1667083416680557539930583; // mpmath
    let naive = ((recurrence(3, z) - want) / want).abs();
    let ours = ((D::splat(z).phi::<3>().extract::<0>() - want) / want).abs();
    assert!(
        naive > 1e-10,
        "precondition: recurrence expected to be visibly wrong, rel {naive:e}"
    );
    assert!(ours <= 2e-15, "kernel rel {ours:e}");

    // Even at z = 1, still inside the split for N = 4, the recurrence has lost ~5 bits.
    let want4 = 0.05161516179237856869362;
    let naive4 = ((recurrence(4, 1.0) - want4) / want4).abs();
    let ours4 = ((D::splat(1.0).phi::<4>().extract::<0>() - want4) / want4).abs();
    assert!(naive4 > 4.0 * f64::EPSILON, "precondition: recurrence rel {naive4:e}");
    assert!(ours4 <= 2e-15, "kernel rel {ours4:e}");
}

/// The recurrence step `phi_{k+1} = (phi_k - 1/k!)/z` holds across the split, so the
/// arms agree with each other and not just with the references.
#[test]
fn recurrence_identity_holds_across_the_split() {
    for &z in &[-4.5_f64, -2.5, -1.5, -0.3, 0.3, 1.5, 2.5, 4.5, 12.0] {
        let v = D::splat(z);
        let p2 = v.phi::<2>().extract::<0>();
        let p3 = v.phi::<3>().extract::<0>();
        let p4 = v.phi::<4>().extract::<0>();
        // Only checked where the subtraction is well conditioned.
        if z.abs() >= 1.5 {
            close("phi_3 from phi_2", (p2 - 0.5) / z, p3, 1e-13);
            close("phi_4 from phi_3", (p3 - 1.0 / 6.0) / z, p4, 1e-12);
        }
        // And the downward form, which is always well conditioned.
        close("phi_2 from phi_3", z * p3 + 0.5, p2, 1e-14);
        close("phi_3 from phi_4", z * p4 + 1.0 / 6.0, p3, 1e-14);
    }
}

// (The composite path, `Compensated` converging past f64, is tested in
// thermite-special/tests/cancellation_free_forms.rs, which has that crate as a
// dev-dependency.)

// --- a wide backend, to be sure nothing is scalar-only ---

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn wide_backend_agrees_with_scalar() {
    use thermite::simd::Simd;
    type W = Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>;

    // Mixed lanes on both sides of the split, so both arms run in one call.
    let zs = [-7.5, -0.5, 1.9, 30.0];
    let w = W::from_slice(&zs);
    let (p2, p4) = (w.phi::<2>().into_array(), w.phi::<4>().into_array());
    for (i, &z) in zs.iter().enumerate() {
        let d = D::splat(z);
        close("wide phi_2", p2.as_slice()[i], d.phi::<2>().extract::<0>(), 1e-15);
        close("wide phi_4", p4.as_slice()[i], d.phi::<4>().extract::<0>(), 1e-15);
    }
}
