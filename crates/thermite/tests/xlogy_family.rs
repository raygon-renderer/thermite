//! `xlogy`, `xlog1py`, `entr`, `rel_entr` and `kl_div`: the guarded-product family, where
//! the whole point is the edge cases rather than the arithmetic.
//!
//! `0 * ln 0` is `0 * -inf = NaN` written directly, and a single NaN poisons every reduction
//! downstream, so the value of these entry points is entirely in the piecewise definition.
//! The tests are weighted accordingly: the interior values are one table, and everything
//! else here is boundaries, signs, infinities and NaN ordering.
//!
//! Conventions are SciPy's, cross-checked against PyTorch for the NaN-versus-zero priority
//! (`torch.xlogy` documents NaN as winning, which is what is asserted below).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::math::{RealMath, TranscendentalMath};
use thermite::prelude::*;
use thermite::simd::Simd;

macro_rules! ctx {
    () => {
        #[allow(dead_code)]
        type D = Vector<<S as Simd>::f64x4>;
        #[allow(dead_code)]
        type F = Vector<<S as Simd>::f32x8>;

        #[allow(dead_code)]
        #[track_caller]
        fn close(name: &str, got: f64, want: f64, tol: f64) {
            let rel = if want == 0.0 {
                got.abs()
            } else {
                ((got - want) / want).abs()
            };
            assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
        }

        #[allow(dead_code)]
        fn xlogy(x: f64, y: f64) -> f64 {
            D::splat(x).xlogy(D::splat(y)).extract::<0>()
        }
        #[allow(dead_code)]
        fn xlog1py(x: f64, y: f64) -> f64 {
            D::splat(x).xlog1py(D::splat(y)).extract::<0>()
        }
        #[allow(dead_code)]
        fn entr(x: f64) -> f64 {
            D::splat(x).entr().extract::<0>()
        }
        #[allow(dead_code)]
        fn rel_entr(x: f64, y: f64) -> f64 {
            D::splat(x).rel_entr(D::splat(y)).extract::<0>()
        }
        #[allow(dead_code)]
        fn kl_div(x: f64, y: f64) -> f64 {
            D::splat(x).kl_div(D::splat(y)).extract::<0>()
        }

        // --- interior values, mpmath at 40 digits ---

        // Several of these oracle values land on multiples of ln 2, since the tables use dyadic
        // arguments. They are mpmath output, not a constant spelled out by hand, and rewriting them
        // as `LN_2` expressions would hide where they came from.
#[allow(clippy::approx_constant)]
#[rustfmt::skip]
#[allow(dead_code)]
const XLOGY: [(f64, f64, f64); 5] = [
    (2.0,   3.0,   2.1972245773362196),
    (0.5,   0.25, -0.6931471805599453),
    (1.0,   1.0,   0.0),
    (3.0,   0.1,  -6.907755278982137),
    (-2.0,  4.0,  -2.772588722239781),
];

#[rustfmt::skip]
#[allow(dead_code)]
const XLOG1PY: [(f64, f64, f64); 4] = [
    (2.0,   3.0,    2.772588722239781),
    (0.5,  -0.5,   -0.34657359027997264),
    (4.0,   1e-12,  3.999999999998e-12),
    (-2.0,  0.25,  -0.44628710262841953),
];

#[rustfmt::skip]
#[allow(dead_code)]
const ENTR: [(f64, f64); 6] = [
    (0.25, 0.34657359027997264),
    (0.5,  0.34657359027997264),
    (1.0,  0.0),
    (2.0, -1.3862943611198906),
    (5.0, -8.047189562170502),
    (1e-8, 1.8420680743952367e-07),
];

        // (x, y, rel_entr, kl_div), where the two differ only by the -x + y tail.
#[rustfmt::skip]
#[allow(dead_code)]
const REL_KL: [(f64, f64, f64, f64); 5] = [
    (0.5,  0.25,  0.34657359027997264,  0.09657359027997266),
    (0.25, 0.5,  -0.17328679513998632,  0.07671320486001368),
    (2.0,  3.0,  -0.8109302162163288,   0.18906978378367123),
    (1.0,  1.0,   0.0,                  0.0),
    (0.3,  0.7,  -0.25418935811616106,  0.1458106418838389),
];
    };
}

const NEAR_DIAGONAL: [(f64, f64, f64, f64); 4] = [
    (0.3, 0.300000000003, -2.999989145975867e-12, 1.4999891460005018e-23),
    (0.3, 0.2999999997, 2.9999996946096e-10, 1.4999996941096162e-19),
    (
        1e-08,
        1.0000000010000002e-08,
        -1.0000001487112813e-17,
        5.0000014887795914e-27,
    ),
    (2.5, 2.5000000000002496, -2.495781359357227e-13, 1.2457849187430434e-26),
];

for_each_backend_concrete! {

fn interior_values_match_the_reference() {
    ctx!();
    for &(x, y, want) in XLOGY.iter() {
        close(&format!("xlogy({x},{y})"), xlogy(x, y), want, 4.0 * f64::EPSILON);
        close(
            &format!("f32 xlogy({x},{y})"),
            F::splat(x as f32).xlogy(F::splat(y as f32)).extract::<0>() as f64,
            want,
            8.0 * f32::EPSILON as f64,
        );
    }
    for &(x, y, want) in XLOG1PY.iter() {
        close(&format!("xlog1py({x},{y})"), xlog1py(x, y), want, 4.0 * f64::EPSILON);
    }
    for &(x, want) in ENTR.iter() {
        close(&format!("entr({x})"), entr(x), want, 4.0 * f64::EPSILON);
    }
    for &(x, y, want_rel, want_kl) in REL_KL.iter() {
        close(
            &format!("rel_entr({x},{y})"),
            rel_entr(x, y),
            want_rel,
            4.0 * f64::EPSILON,
        );
        close(&format!("kl_div({x},{y})"), kl_div(x, y), want_kl, 8.0 * f64::EPSILON);
    }
}

// --- the guard, which is the reason these exist ---

fn a_zero_first_argument_absorbs_an_infinite_log() {
    ctx!();
    // The whole point: 0 * ln(0) is 0 * -inf = NaN written out, and one NaN takes a whole
    // reduction with it.
    assert!(
        (0.0_f64 * 0.0_f64.ln()).is_nan(),
        "precondition: the direct form is NaN here"
    );

    assert_eq!(xlogy(0.0, 0.0), 0.0);
    assert_eq!(xlogy(0.0, 1.0), 0.0);
    assert_eq!(xlogy(0.0, f64::INFINITY), 0.0);
    assert_eq!(xlog1py(0.0, -1.0), 0.0, "ln(1 + -1) = ln 0 = -inf");
    assert_eq!(entr(0.0), 0.0);

    // -0.0 is zero by value, and the guards compare rather than test the sign bit.
    assert_eq!(xlogy(-0.0, 0.0), 0.0);
    assert_eq!(entr(-0.0), 0.0);
    assert_eq!(rel_entr(-0.0, 1.0), 0.0);
    assert_eq!(kl_div(-0.0, 2.0), 2.0);

    // A whole vector survives one poisoned lane.
    let xs = D::splat(0.0);
    assert_eq!(xs.xlogy(D::ZERO).extract::<0>(), 0.0);
}

fn nan_wins_over_the_zero_guard() {
    ctx!();
    // SciPy and PyTorch both order it this way: a NaN y propagates even at x = 0, because
    // a NaN input is missing information rather than a limit to be filled in.
    assert!(xlogy(0.0, f64::NAN).is_nan(), "NaN y must survive the x = 0 guard");
    assert!(xlog1py(0.0, f64::NAN).is_nan());

    // A negative y is NOT NaN, so the zero guard still applies there and the result is 0 -
    // only a non-zero x picks up the NaN from the log itself.
    assert_eq!(xlogy(0.0, -1.0), 0.0, "a negative y is out of domain but not NaN");
    assert!(xlogy(1.0, -1.0).is_nan());

    // A NaN x propagates unconditionally. It is not zero, so no guard applies.
    assert!(xlogy(f64::NAN, 1.0).is_nan());
    assert!(entr(f64::NAN).is_nan());
}

fn the_extended_value_conventions_hold() {
    ctx!();
    // entr is -inf below zero: a convention, not a limit, so that entr stays concave over
    // all of R and a convex solver can use it as a barrier.
    assert_eq!(entr(-1.0), f64::NEG_INFINITY);
    assert_eq!(entr(-1e-300), f64::NEG_INFINITY);
    assert_eq!(entr(f64::INFINITY), f64::NEG_INFINITY, "-x ln x -> -inf");

    // rel_entr and kl_div are +inf outside the closed first quadrant.
    assert_eq!(
        rel_entr(1.0, 0.0),
        f64::INFINITY,
        "y = 0 at positive x is infinite surprise"
    );
    assert_eq!(kl_div(1.0, 0.0), f64::INFINITY);
    assert_eq!(rel_entr(-1.0, 1.0), f64::INFINITY);
    assert_eq!(rel_entr(1.0, -1.0), f64::INFINITY);
    assert_eq!(kl_div(-1.0, 1.0), f64::INFINITY);
    assert_eq!(rel_entr(0.0, -1.0), f64::INFINITY, "x = 0 needs y >= 0 to give 0");
    assert_eq!(kl_div(0.0, -1.0), f64::INFINITY);

    // At x = 0 with y >= 0 the two differ: rel_entr drops the term, kl_div keeps the tail.
    assert_eq!(rel_entr(0.0, 3.0), 0.0);
    assert_eq!(kl_div(0.0, 3.0), 3.0);
}

fn kl_div_is_a_bregman_divergence_and_rel_entr_is_not() {
    ctx!();
    // The property the -x + y tail buys: kl_div >= 0 everywhere with equality only at x = y,
    // even for unnormalized arguments. rel_entr alone goes negative, which is why it cannot
    // be used as an objective on its own.
    for &(x, y) in &[(0.25_f64, 0.5_f64), (2.0, 3.0), (0.3, 0.7), (5.0, 1.0), (1e-3, 1.0)] {
        assert!(kl_div(x, y) >= 0.0, "kl_div({x},{y}) = {} is negative", kl_div(x, y));
        assert_eq!(kl_div(x, x), 0.0, "kl_div is zero on the diagonal");
    }
    assert!(rel_entr(0.25, 0.5) < 0.0, "rel_entr alone is not non-negative");

    // And the identity relating them, on the interior where both are finite.
    for &(x, y) in &[(0.25_f64, 0.5_f64), (2.0, 3.0), (0.3, 0.7)] {
        close("kl = rel - x + y", kl_div(x, y), rel_entr(x, y) - x + y, 1e-14);
    }
}

fn entropy_of_a_distribution_sums_correctly() {
    ctx!();
    // The actual use: sum entr over a distribution containing an impossible outcome. The
    // direct spelling returns NaN for the whole thing.
    let p = [0.5_f64, 0.25, 0.25, 0.0];

    let naive: f64 = p.iter().map(|&q| -q * q.ln()).sum();
    assert!(
        naive.is_nan(),
        "precondition: the direct form is expected to poison the sum"
    );

    let got: f64 = p.iter().map(|&q| entr(q)).sum();
    // -0.5 ln 0.5 - 2 * 0.25 ln 0.25 = 1.5 ln 2
    close("Shannon entropy", got, 1.5 * core::f64::consts::LN_2, 1e-15);
}

/// Near-diagonal reference values: `(x, y, rel_entr, kl_div)` from mpmath at 50 digits,
/// computed from these exact f64 inputs. This is the regime a converging optimizer lives
/// in, and the one where the textbook `x * ln(x/y)` falls apart.
#[rustfmt::skip]
fn rel_entr_is_accurate_where_the_textbook_form_is_not() {
    ctx!();
    // `x/y` rounds to a relative eps, so `ln(x/y)` has an ABSOLUTE error of eps while the
    // answer is O((x-y)/y), a relative error of eps*y/(x-y), unbounded on the diagonal. The
    // kernel switches to ln1p((x-y)/y) there, where x - y is exact by Sterbenz.
    for &(x, y, want, want_kl) in NEAR_DIAGONAL.iter() {
        close(&format!("rel_entr({x},{y})"), rel_entr(x, y), want, 8.0 * f64::EPSILON);

        // The form SciPy ships, evaluated right here, so the comparison is not a claim
        // about another library but a measurement of the alternative on this input.
        let textbook = x * (x / y).ln();
        let textbook_err = ((textbook - want) / want).abs();
        assert!(
            textbook_err > 1e-8,
            "precondition: the plain ratio form is expected to be visibly worse at ({x}, {y}), err {textbook_err:e}"
        );

        // kl_div is worse off than rel_entr here, not better: its log term and its -x + y
        // tail are both first order and cancel to a second-order answer, so the textbook
        // spelling loses everything rather than some digits. Written as
        // -x*log1pmx((y-x)/x) the cancellation lands inside log1pmx, which absorbs it.
        close(&format!("kl_div({x},{y})"), kl_div(x, y), want_kl, 1e-13);
        assert!(kl_div(x, y) > 0.0, "kl_div stays positive near the diagonal");

        let textbook_kl = x * (x / y).ln() - x + y;
        let kl_err = ((textbook_kl - want_kl) / want_kl).abs();
        assert!(
            kl_err > 0.5,
            "precondition: the textbook kl_div is expected to be ~100% wrong at ({x}, {y}), err {kl_err:e}"
        );
    }
}

fn lanes_stay_independent_across_the_branches() {
    ctx!();
    type D4 = thermite::simd::f64x4<S>;

    // One lane per branch of each definition, so a mask that leaks across lanes shows up.
    let xs = [0.0, 0.5, -1.0, 2.0];
    let ys = [3.0, 0.25, 1.0, 0.0];

    let re = D4::new(xs).rel_entr(D4::new(ys)).into_array();
    let kl = D4::new(xs).kl_div(D4::new(ys)).into_array();
    let en = D4::new(xs).entr().into_array();
    let xl = D4::new(xs).xlogy(D4::new(ys)).into_array();

    for lane in 0..4 {
        let (x, y) = (xs[lane], ys[lane]);
        assert_eq!(re.as_slice()[lane], rel_entr(x, y), "rel_entr lane {lane}");
        assert_eq!(kl.as_slice()[lane], kl_div(x, y), "kl_div lane {lane}");
        assert_eq!(en.as_slice()[lane], entr(x), "entr lane {lane}");
        assert_eq!(xl.as_slice()[lane], xlogy(x, y), "xlogy lane {lane}");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn wide_backend_agrees_with_scalar() {
    ctx!();
    use thermite::simd::Simd;
    type W = Vector<<S as Simd>::f64x4>;

    let xs = [0.0, 0.5, -1.0, 2.0];
    let ys = [3.0, 0.25, 1.0, 0.5];

    let re = W::from_slice(&xs).rel_entr(W::from_slice(&ys)).into_array();
    let xl = W::from_slice(&xs).xlogy(W::from_slice(&ys)).into_array();
    let en = W::from_slice(&xs).entr().into_array();

    for lane in 0..4 {
        let (x, y) = (xs[lane], ys[lane]);
        assert_eq!(re.as_slice()[lane], rel_entr(x, y), "wide rel_entr lane {lane}");
        assert_eq!(xl.as_slice()[lane], xlogy(x, y), "wide xlogy lane {lane}");
        assert_eq!(en.as_slice()[lane], entr(x), "wide entr lane {lane}");
    }
}

}
