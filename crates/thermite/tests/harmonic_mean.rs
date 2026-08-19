//! `harmonic_mean` and `inv_sum_inv`: the two N-ary reciprocal-sum reductions.
//!
//! They differ by exactly a factor of `N`, which is the whole reason both exist as named
//! entry points: `1/sum(1/x)` is what resistors in parallel, series capacitors, spring
//! compliances and reduced mass all compute, and calling that "the harmonic mean" puts a
//! stray factor of `N` through a model. The identity that separates them is at the bottom
//! of `differ_by_exactly_n`: `N` copies of `x` give `x` and `x/N` respectively.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use thermite::Vector;
use thermite::math::CoreMath;
use thermite::prelude::*;

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

fn hm<const N: usize>(xs: [f64; N]) -> f64 {
    D::harmonic_mean(xs.map(D::splat)).extract::<0>()
}
fn isi<const N: usize>(xs: [f64; N]) -> f64 {
    D::inv_sum_inv(xs.map(D::splat)).extract::<0>()
}

#[test]
fn matches_the_reference() {
    // (harmonic_mean, inv_sum_inv) from mpmath at 60 digits.
    close("hm[1,2,4]", hm([1.0, 2.0, 4.0]), 1.7142857142857142, 4.0 * f64::EPSILON);
    close(
        "isi[1,2,4]",
        isi([1.0, 2.0, 4.0]),
        0.5714285714285714,
        4.0 * f64::EPSILON,
    );

    close("hm[0.5,0.25]", hm([0.5, 0.25]), 0.3333333333333333, 4.0 * f64::EPSILON);
    close(
        "isi[0.5,0.25]",
        isi([0.5, 0.25]),
        0.16666666666666666,
        4.0 * f64::EPSILON,
    );

    close(
        "hm[10,20,30,40]",
        hm([10.0, 20.0, 30.0, 40.0]),
        19.2,
        4.0 * f64::EPSILON,
    );
    close(
        "isi[10,20,30,40]",
        isi([10.0, 20.0, 30.0, 40.0]),
        4.8,
        4.0 * f64::EPSILON,
    );

    close("hm[2,8]", hm([2.0, 8.0]), 3.2, 4.0 * f64::EPSILON);
    close("isi[2,8]", isi([2.0, 8.0]), 1.6, 4.0 * f64::EPSILON);

    close(
        "hm tiny",
        hm([1e-300, 2e-300, 5e-301]),
        8.571428571428572e-301,
        4.0 * f64::EPSILON,
    );

    // f32 too, on a range it can hold.
    let got = F::harmonic_mean([1.0f32, 2.0, 4.0].map(F::splat)).extract::<0>() as f64;
    close("f32 hm", got, 1.7142857142857142, 8.0 * f32::EPSILON as f64);
}

#[test]
fn differ_by_exactly_n() {
    // The identity that tells them apart, and the one a caller reaching for the wrong name
    // will violate: N copies of x give x and x/N.
    for &x in &[0.5_f64, 3.0, 1e10] {
        close("hm of equal pair", hm([x, x]), x, 4.0 * f64::EPSILON);
        close("isi of equal pair", isi([x, x]), x / 2.0, 4.0 * f64::EPSILON);

        close("hm of equal quad", hm([x, x, x, x]), x, 4.0 * f64::EPSILON);
        close("isi of equal quad", isi([x, x, x, x]), x / 4.0, 4.0 * f64::EPSILON);
    }

    // And the general relation, on ordinary inputs.
    for xs in [[1.0_f64, 2.0, 4.0], [10.0, 20.0, 30.0], [0.25, 0.5, 8.0]] {
        close("hm = 3 * isi", hm(xs), 3.0 * isi(xs), 4.0 * f64::EPSILON);
    }

    // The two-element reduced-mass / parallel-resistance form, spelled out.
    let (a, b) = (3.0_f64, 6.0_f64);
    close("reduced mass", isi([a, b]), a * b / (a + b), 4.0 * f64::EPSILON);
    close(
        "harmonic mean of two",
        hm([a, b]),
        2.0 * a * b / (a + b),
        4.0 * f64::EPSILON,
    );
}

#[test]
fn the_scaled_sum_survives_inputs_the_direct_form_cannot() {
    // sum(1/x_i) overflows the moment any x_i is denormal, taking the answer to zero when
    // the true value is merely small. Scaling every reciprocal by the smallest element caps
    // the sum at N and removes the failure entirely.
    let xs = [1e-320_f64, 1.0, 2.0];

    let naive: f64 = 3.0 / xs.iter().map(|&x| 1.0 / x).sum::<f64>();
    assert_eq!(naive, 0.0, "precondition: the direct form is expected to collapse here");

    close("hm with a denormal", hm(xs), 3e-320, 1e-3);
    close("isi with a denormal", isi(xs), 1e-320, 1e-3);

    // The full representable spread, smallest denormal against a large normal.
    close("hm extreme spread", hm([5e-324, 1e300]), 1e-323, 0.5);
    close("isi extreme spread", isi([5e-324, 1e300]), 5e-324, 0.5);
}

#[test]
fn zeros_and_infinities_take_their_limits() {
    // A zero anywhere sends one reciprocal to infinity, so the mean is zero. This is the
    // limit, not a convention.
    assert_eq!(hm([0.0, 1.0, 2.0]), 0.0);
    assert_eq!(isi([0.0, 1.0, 2.0]), 0.0);
    assert_eq!(hm([1.0, 0.0]), 0.0);
    assert_eq!(hm([0.0, 0.0]), 0.0);

    // An infinite element contributes nothing to the sum of reciprocals, so it drops out and
    // the others decide the answer.
    close("hm with an inf", hm([f64::INFINITY, 2.0, 2.0]), 3.0, 4.0 * f64::EPSILON);
    close("isi with an inf", isi([f64::INFINITY, 2.0]), 2.0, 4.0 * f64::EPSILON);

    // All infinite is the other limit.
    assert_eq!(hm([f64::INFINITY, f64::INFINITY]), f64::INFINITY);
    assert_eq!(isi([f64::INFINITY, f64::INFINITY]), f64::INFINITY);

    assert!(hm([f64::NAN, 1.0]).is_nan());
}

#[test]
fn bounded_by_the_smallest_and_the_arithmetic_mean() {
    // min <= harmonic <= arithmetic, the defining inequality, and a decent check that
    // nothing is inverted or scaled wrong.
    for xs in [
        [1.0_f64, 2.0, 4.0],
        [0.25, 0.5, 8.0],
        [10.0, 20.0, 30.0],
        [1e-5, 1.0, 1e5],
    ] {
        let h = hm(xs);
        let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let arith = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!(
            min <= h && h <= arith,
            "harmonic {h} not in [{min}, {arith}] for {xs:?}"
        );
    }
}

#[test]
fn lanes_stay_independent() {
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    // One lane per branch: ordinary, a zero, a denormal, an infinity.
    let a = [1.0, 0.0, 1e-320, f64::INFINITY];
    let b = [2.0, 3.0, 1.0, 4.0];

    let got = D4::harmonic_mean([D4::new(a), D4::new(b)]);
    for lane in 0..4 {
        assert_eq!(got.as_slice()[lane], hm([a[lane], b[lane]]), "lane {lane}");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn wide_backend_agrees_with_scalar() {
    use thermite::simd::Simd;
    type W = Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>;

    let a = [1.0, 0.0, 1e-320, 6.0];
    let b = [2.0, 3.0, 1.0, 3.0];

    let h = W::harmonic_mean([W::from_slice(&a), W::from_slice(&b)]).into_array();
    let s = W::inv_sum_inv([W::from_slice(&a), W::from_slice(&b)]).into_array();

    for lane in 0..4 {
        assert_eq!(h.as_slice()[lane], hm([a[lane], b[lane]]), "wide hm lane {lane}");
        assert_eq!(s.as_slice()[lane], isi([a[lane], b[lane]]), "wide isi lane {lane}");
    }
}
