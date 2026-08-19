//! The exponential integrals `$E_N(x)$`.
//!
//! This file exists because there was no coverage at all, and a dropped minus sign in one
//! denominator coefficient of the `x <= 1` rational fit went unnoticed as a result. It cost
//! **3.9e-7 relative** at `$x \to 1^-$` - roughly f32 accuracy in an f64 kernel - rising
//! smoothly across the whole small branch and invisible below `$x \approx 0.05$`, where the
//! `$-\ln x$` term dominates and suppresses it.
//!
//! Two properties made it hard to notice, and both are pinned below:
//!
//! - **No policy tier changed it.** `Reference` was exactly as wrong as `Performance`, so it
//!   read as the kernel's inherent accuracy rather than as a defect. Every accuracy assertion
//!   here therefore runs at `Reference` *and* checks that the tiers agree with each other to
//!   the precision each claims.
//! - **The error was zero at both ends of the branch.** A test sampling only small and large
//!   `x` passes cleanly. The grid below is deliberately dense either side of the `$x = 1$`
//!   seam, which is where the two rational fits meet and where any future retuning will show
//!   up first.
//!
//! References from mpmath at 50 digits.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::{Performance, Precision, Reference};
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
}

/// `(x, E_1, E_2, E_3, E_5, E_8)`.
type Row = (f64, f64, f64, f64, f64, f64);

#[rustfmt::skip]
const REF: [Row; 21] = [
    (0.001, 6.331539364136149,      0.9926689604692388,     0.4990039154364529,     0.24966691650035058,    0.1426905761488234),
    (0.01,  4.037929576538114,      0.9496705379837869,     0.4902765641846651,     0.24669150254720257,    0.14120043466228313),
    (0.05,  2.467898488509974,      0.8278345000752153,     0.45491884974847663,    0.23393867495012313,    0.13476868671487202),
    (0.1,   1.8229239584193906,     0.7225450221940205,     0.41629145790827876,    0.21901595224028048,    0.12715015804899488),
    (0.25,  1.0442826344437381,     0.5177301244604703,     0.32468412597814367,    0.18016624260999978,    0.10683991649859081),
    (0.5,   0.5597735947761608,     0.326643862324553,      0.22160436427517846,    0.13097731169586485,    0.08007073163363616),
    (0.75,  0.34034081291123003,    0.2171109430575922,     0.1547666727239103,     0.09582341642287355,    0.06011858424554789),
    (0.9,   0.26018393932599965,    0.17240411434719943,    0.1257029784140598,     0.07963464149755388,    0.050660924257730694),
    (0.99,  0.22309982579017723,    0.15070786348977022,    0.11118795308358659,    0.07132037181429458,    0.04572845875962816),
    (1.0,   0.21938393439552029,    0.14849550677592205,    0.10969196719776014,    0.0704542374617204,     0.04521148206188467),
    (1.01,  0.21574162379448997,    0.14631993953908845,    0.108217920318522,      0.06959907248868798,    0.04470045393993453),
    (1.25,  0.1464133725259102,     0.10348808120280238,    0.07857234767834356,    0.052012723979395374,   0.03405031104286336),
    (1.5,   0.10001958240663265,    0.07310078653848084,    0.056739490170354276,   0.03852992442549515,    0.02567825223208377),
    (2.0,   0.04890051070806112,    0.03753426182049045,    0.03013337979781589,    0.02132240020232302,    0.014654607600286288),
    (3.0,   0.013048381094197037,   0.01064192508527283,    0.008930646556022725,   0.006697984917017044,   0.004828781181282259),
    (5.0,   0.0011482955912753257,  0.000996469042708838,   0.0008778008927706383,  0.0007057606934245853,  0.0005424682061642422),
    (8.0,   3.76656228439249e-05,   3.413764515111262e-05,  3.1180733346805424e-05, 2.6521149556915645e-05, 2.1600730159975376e-05),
    (12.0,  4.751081824672494e-07,  4.4291416372121706e-07, 4.146211943368026e-07,  3.672950670454744e-07,  3.131158686896961e-07),
    (20.0,  9.835525290649882e-11,  9.404856430858148e-11,  9.009116813346401e-11,  8.307130599417691e-11,  7.43345116728989e-11),
    (35.0,  1.752705938994737e-17,  1.7064597366540917e-17, 1.6625384092883394e-17, 1.581015677598271e-17,  1.472446821721958e-17),
    (60.0,  1.4358675656812567e-28, 1.4130536860897962e-28, 1.3909432307887193e-28, 1.348708008531087e-28,  1.289897881330291e-28),
];

#[test]
fn it_matches_the_reference_at_every_order() {
    for &(x, e1, e2, e3, e5, e8) in REF.iter() {
        let v = D::splat(x);

        close(
            &format!("E_1({x})"),
            v.expint_p::<Reference, 1>().extract::<0>(),
            e1,
            64.0 * f64::EPSILON,
        );
        // Every order is held to the same margin. Below `recurrence_threshold` the forward
        // recurrence runs and its amplification is capped at AMP_CAP = 64 ulps by
        // construction, and above it the continued fraction is flat at ~1 ulp regardless of
        // order, so no regime exists where a higher N costs accuracy.
        close(
            &format!("E_2({x})"),
            v.expint_p::<Reference, 2>().extract::<0>(),
            e2,
            64.0 * f64::EPSILON,
        );
        close(
            &format!("E_3({x})"),
            v.expint_p::<Reference, 3>().extract::<0>(),
            e3,
            64.0 * f64::EPSILON,
        );
        let _ = x;
        close(
            &format!("E_5({x})"),
            v.expint_p::<Reference, 5>().extract::<0>(),
            e5,
            64.0 * f64::EPSILON,
        );
        close(
            &format!("E_8({x})"),
            v.expint_p::<Reference, 8>().extract::<0>(),
            e8,
            64.0 * f64::EPSILON,
        );
    }
}

#[test]
fn the_branch_seam_at_one_is_smooth_and_accurate() {
    // The regression. `x = 1` is where the two rational fits meet, and the dropped sign made
    // the small-x branch degrade monotonically toward it while staying clean on both far
    // sides, so this samples densely on the approach rather than at the endpoints.
    //
    // The reference here is the series E_1(x) = -gamma - ln x + sum_{k>=1} (-1)^{k+1} x^k/(k k!),
    // which converges fast on [0.5, 1.5] and shares nothing with the rational fit under test.
    #[allow(clippy::excessive_precision)]
    const GAMMA: f64 = 0.5772156649015328606065120900824;

    let series = |x: f64| {
        let (mut term, mut sum) = (1.0_f64, 0.0_f64);
        for k in 1..40 {
            term *= -x / (k as f64);
            sum -= term / (k as f64);
        }
        -GAMMA - x.ln() + sum
    };

    let mut worst = 0.0_f64;
    for i in 0..=200 {
        let x = 0.5 + 1.0 * (i as f64) / 200.0;
        let got = D::splat(x).expint_p::<Reference, 1>().extract::<0>();
        let want = series(x);
        let rel = ((got - want) / want).abs();

        assert!(
            rel < 1e-14,
            "E_1({x}) = {got:?} against the series {want:?} (rel {rel:e})"
        );
        worst = worst.max(rel);
    }

    // The bug measured 3.9e-7 here. Anything above ~1e-13 means the fit has moved.
    assert!(worst < 1e-14, "worst deviation across the seam was {worst:e}");
}

#[test]
fn the_policy_tiers_agree_with_each_other() {
    // The other half of why the bug hid: it was tier-independent, so no tier comparison would
    // have caught it either. This does not re-find that bug. It guards the opposite failure,
    // a future fix that lands on one tier and not the others.
    for &(x, ..) in REF.iter() {
        let v = D::splat(x);
        let r = v.expint_p::<Reference, 1>().extract::<0>();

        for (tag, got) in [
            ("Precision", v.expint_p::<Precision, 1>().extract::<0>()),
            ("Performance", v.expint_p::<Performance, 1>().extract::<0>()),
            ("default", v.expint::<1>().extract::<0>()),
        ] {
            close(&format!("{tag} vs Reference at x = {x}"), got, r, 1e-12);
        }
    }
}

#[test]
fn the_recurrence_relation_holds() {
    // E_{n+1}(x) = (e^-x - x E_n(x)) / n, independent of how either side was computed - so it
    // checks the orders against each other rather than against a table.
    for &(x, ..) in REF.iter().filter(|r| r.0 <= 20.0) {
        let v = D::splat(x);
        let e = (-x).exp();

        let e1 = v.expint_p::<Reference, 1>().extract::<0>();
        let e2 = v.expint_p::<Reference, 2>().extract::<0>();
        let e3 = v.expint_p::<Reference, 3>().extract::<0>();

        close(&format!("E_2 from E_1 at {x}"), e2, e - x * e1, 1e-11);
        close(&format!("E_3 from E_2 at {x}"), e3, (e - x * e2) / 2.0, 1e-11);
    }
}

#[test]
fn the_edges_are_the_limits() {
    // E_1(0) diverges; E_n(0) = 1/(n-1) above it.
    assert_eq!(D::ZERO.expint::<1>().extract::<0>(), f64::INFINITY);
    close("E_2(0)", D::ZERO.expint::<2>().extract::<0>(), 1.0, 0.0);
    close("E_3(0)", D::ZERO.expint::<3>().extract::<0>(), 0.5, 0.0);
    close(
        "E_8(0)",
        D::ZERO.expint::<8>().extract::<0>(),
        1.0 / 7.0,
        4.0 * f64::EPSILON,
    );

    // Negative argument is out of domain, and NaN propagates.
    assert!(D::splat(-1.0).expint::<1>().extract::<0>().is_nan());
    assert!(D::splat(-0.5).expint::<3>().extract::<0>().is_nan());
    assert!(D::splat(f64::NAN).expint::<2>().extract::<0>().is_nan());

    // E_n decreases in x and in n, everywhere.
    for &n_x in &[0.3_f64, 1.0, 1.7, 5.0] {
        let a = D::splat(n_x).expint_p::<Reference, 2>().extract::<0>();
        let b = D::splat(n_x + 0.1).expint_p::<Reference, 2>().extract::<0>();
        let c = D::splat(n_x).expint_p::<Reference, 3>().extract::<0>();

        assert!(b < a, "E_2 must decrease in x at {n_x}");
        assert!(c < a, "E_3 < E_2 at {n_x}");
    }
}

#[test]
fn f32_tracks_the_f64_kernel() {
    for &(x, ..) in REF.iter().filter(|r| r.0 <= 12.0) {
        let got = F::splat(x as f32).expint_p::<Reference, 1>().extract::<0>() as f64;
        let want = D::splat(x).expint_p::<Reference, 1>().extract::<0>();

        close(&format!("f32 E_1({x})"), got, want, 64.0 * f32::EPSILON as f64);
    }
}

#[test]
fn lanes_stay_independent_across_the_seam() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    // Two lanes either side of x = 1, so both rational branches run in one packet.
    let xs = [0.25, 0.99, 1.01, 8.0];
    let got = D4::new(xs).expint_p::<Reference, 3>().into_array();

    for (lane, &x) in xs.iter().enumerate() {
        assert_eq!(
            got.as_slice()[lane],
            D::splat(x).expint_p::<Reference, 3>().extract::<0>(),
            "lane {lane} (x = {x})"
        );
    }
}
