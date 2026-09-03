//! Correctness gate for the **spherical** Bessel functions `j_n`, `y_n`, `i_n`, `k_n`.
//!
//! Both references ship these publicly (Boost as `sph_bessel` / `sph_neumann`, SciPy as
//! `spherical_jn` / `spherical_yn` / `spherical_in` / `spherical_kn`). This module is the
//! union of the two sets.
//!
//! They share the half-integer walk with `bessel_half`, but are seeded in the _spherical_
//! normalization rather than being that kernel's output rescaled, so this is a genuinely
//! separate code path and not a wrapper whose correctness follows from its callee's.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]
// `k_0(x) = pi/2x`, so several reference rows ARE multiples of pi. They are mpmath output,
// not an approximation of a constant anyone should substitute.
#![allow(clippy::approx_constant)]

use thermite::Vector;
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::bessel::{I, J, K, Scaled, Y};
use thermite_special::{BesselOrder, SpecialMathWithPolicy};

type V = Vector<f64>;
type S = <V as GenericVector>::Signed;

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

/// Envelope-relative for the oscillating pair, whose envelope is `1/x`. Plain relative for the
/// modified pair, which does not oscillate.
fn env_rel(got: f64, want: f64, x: f64) -> f64 {
    (got - want).abs() / (1.0 / x).max(want.abs())
}

macro_rules! at_order {
    ($n:expr, $x:expr, $f:ty) => {
        match $n {
            0 => V::splat($x).sph_bessel_n_p::<Precision, $f, 0>().extract::<0>(),
            1 => V::splat($x).sph_bessel_n_p::<Precision, $f, 1>().extract::<0>(),
            2 => V::splat($x).sph_bessel_n_p::<Precision, $f, 2>().extract::<0>(),
            5 => V::splat($x).sph_bessel_n_p::<Precision, $f, 5>().extract::<0>(),
            10 => V::splat($x).sph_bessel_n_p::<Precision, $f, 10>().extract::<0>(),
            _ => unreachable!(),
        }
    };
}

/// All four families against mpmath at 50 digits, over orders 0..10 and `x` from 0.05 to 200,
/// which crosses the forward/downward split for `j` (at `n < x`) in both directions.
///
/// The modified pair is graded in the scaled domain, the only one that still means anything at
/// `x = 200`.
#[test]
fn spherical_bessels_match_a_high_precision_reference() {
    // (n, x, j_n, y_n, e^{-x} i_n, e^{x} k_n)
    const ROWS: &[(usize, f64, f64, f64, f64, f64)] = &[
        (
            0,
            0.05,
            0.9995833854135666,
            -19.975005207899326,
            0.9516258196404043,
            31.41592653589793,
        ),
        (
            0,
            0.5,
            0.958851077208406,
            -1.7551651237807455,
            0.6321205588285577,
            3.141592653589793,
        ),
        (
            0,
            1.0,
            0.8414709848078965,
            -0.5403023058681398,
            0.43233235838169365,
            1.5707963267948966,
        ),
        (
            0,
            3.0,
            0.04704000268662241,
            0.3299974988668152,
            0.16625354130388895,
            0.5235987755982989,
        ),
        (
            0,
            9.0,
            0.04579094280463962,
            0.10123669576496411,
            0.05555555470944557,
            0.17453292519943295,
        ),
        (
            0,
            30.0,
            -0.03293438746976206,
            -0.005141714996252802,
            0.016666666666666666,
            0.05235987755982989,
        ),
        (
            0,
            200.0,
            -0.004366486486069973,
            -0.0024359383750350294,
            0.0025,
            0.007853981633974483,
        ),
        (
            1,
            0.05,
            0.016662500372006585,
            -400.49968754340006,
            0.015857787551510363,
            659.7344572538566,
        ),
        (
            1,
            0.5,
            0.16253703063606656,
            -4.469181324769897,
            0.10363832351432696,
            9.42477796076938,
        ),
        (
            1,
            1.0,
            0.3011686789397568,
            -1.3817732906760363,
            0.1353352832366127,
            3.141592653589793,
        ),
        (
            1,
            3.0,
            0.34567749976235596,
            0.06295916360231597,
            0.11166194492814809,
            0.6981317007977318,
        ),
        (
            1,
            9.0,
            0.10632457829881295,
            -0.03454242105297694,
            0.04938271698950492,
            0.1939254724438144,
        ),
        (
            1,
            30.0,
            -0.006239527911911537,
            0.032762996969886965,
            0.01611111111111111,
            0.054105206811824215,
        ),
        (
            1,
            200.0,
            -0.0024577708074653795,
            0.004354306794194798,
            0.0024875,
            0.007893251542144356,
        ),
        (
            2,
            0.05,
            0.0001666369068286254,
            -24010.006247396104,
            0.00015856654978239882,
            39615.48336176729,
        ),
        (
            2,
            0.5,
            0.016371106607993412,
            -25.059922824838637,
            0.01029061774259589,
            59.69026041820607,
        ),
        (
            2,
            1.0,
            0.06203505201137386,
            -3.605017566159969,
            0.02632650867185558,
            10.995574287564276,
        ),
        (
            2,
            3.0,
            0.29863749707573356,
            -0.26703833526449916,
            0.05459159637574086,
            1.2217304763960306,
        ),
        (
            2,
            9.0,
            -0.0103494167050353,
            -0.11275083611595642,
            0.03909464904627726,
            0.2391747493473711,
        ),
        (
            2,
            30.0,
            0.03231043467857091,
            0.008418014693241499,
            0.015055555555555556,
            0.05777039824101231,
        ),
        (
            2,
            200.0,
            0.004329619923957992,
            0.0025012529769479516,
            0.0024626875,
            0.007972380407106648,
        ),
        (
            5,
            0.05,
            3.0059639555079313e-11,
            -60488400750.06251,
            2.8599112935292302e-11,
            99858736375.05212,
        ),
        (
            5,
            0.5,
            2.9774668754574457e-06,
            -61327.56316698064,
            1.8409903951734993e-06,
            154475.25236966374,
        ),
        (
            5,
            1.0,
            9.256115861125816e-05,
            -999.4403433922364,
            3.677410272699715e-05,
            3818.6058704383936,
        ),
        (
            5,
            3.0,
            0.016397480955999102,
            -2.24702332846539,
            0.0016328186214172472,
            25.539984720850356,
        ),
        (
            5,
            9.0,
            0.03525480653749192,
            0.11899459885924012,
            0.010084848765902094,
            0.8201531194671113,
        ),
        (
            5,
            30.0,
            -0.020504008736827492,
            0.026639390496569996,
            0.010037314814814815,
            0.08552607844726946,
        ),
        (
            5,
            200.0,
            -0.0027568027343361752,
            0.004172459539443551,
            0.0023189327191796877,
            0.008464063954299227,
        ),
        (
            10,
            0.05,
            7.102242850295915e-24,
            -1.3409733649713302e+23,
            6.756596751929442e-24,
            2.2141019218284714e+23,
        ),
        (
            10,
            0.5,
            7.064123963661878e-14,
            -1349739281107.056,
            4.331433624321591e-14,
            3449868979240.394,
        ),
        (
            10,
            1.0,
            7.116552640047314e-11,
            -672215008.2562084,
            2.7343719371837067e-11,
            2723107545.8948145,
        ),
        (
            10,
            3.0,
            3.5260038931752564e-06,
            -4699.8591888113915,
            2.596368791487492e-07,
            92309.90678180744,
        ),
        (
            10,
            9.0,
            0.03742833632430661,
            -0.27829450961968527,
            0.00016368683374645812,
            38.53075273518477,
        ),
        (
            10,
            30.0,
            -0.0145296464038978,
            0.031219591064754935,
            0.0026336406797553727,
            0.31276588654725257,
        ),
        (
            10,
            200.0,
            0.003543172890314245,
            0.0035327568031017207,
            0.0018977399804731617,
            0.01033229566626682,
        ),
    ];

    let (mut wj, mut wy, mut wi, mut wk) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    for &(n, x, rj, ry, ri, rk) in ROWS {
        let gj = at_order!(n, x, J);
        let e = env_rel(gj, rj, x);
        wj = wj.max(e);
        assert!(e <= 1e-14, "j_{n}({x}): got {gj}, want {rj}, env-rel {e:e}");

        let gy = at_order!(n, x, Y);
        let e = env_rel(gy, ry, x);
        wy = wy.max(e);
        assert!(e <= 1e-14, "y_{n}({x}): got {gy}, want {ry}, env-rel {e:e}");

        let gi = at_order!(n, x, Scaled<I>);
        let e = rel(gi, ri);
        wi = wi.max(e);
        assert!(e <= 1e-14, "i_{n}({x}) scaled: got {gi}, want {ri}, rel {e:e}");

        let gk = at_order!(n, x, Scaled<K>);
        let e = rel(gk, rk);
        wk = wk.max(e);
        assert!(e <= 1e-14, "k_{n}({x}) scaled: got {gk}, want {rk}, rel {e:e}");
    }

    println!("worst: j {wj:e}  y {wy:e}  i {wi:e}  k {wk:e}");
}

/// The identity that defines them, `f_n(x) = sqrt(pi/2x) F_{n+1/2}(x)`, against the
/// half-integer cylindrical kernel.
///
/// This is the check that the _normalization_ is right, and is worth making even though the
/// reference table above already covers the values: the two paths share a walk but not a set of
/// seeds, so agreement here says the seed algebra is consistent, and the table says both are
/// consistent with the truth.
#[test]
fn the_defining_identity_against_the_cylindrical_kernel() {
    for &x in &[0.25f64, 1.0, 4.0, 12.0, 60.0] {
        let f = (core::f64::consts::PI / (2.0 * x)).sqrt();
        let v = V::splat(x);

        macro_rules! check {
            ($n:literal, $f:ty, $scaled:literal) => {{
                let got = v.sph_bessel_n_p::<Precision, $f, $n>().extract::<0>();
                let cyl = v
                    .bessel_p::<Precision, $f>(BesselOrder::HalfInteger(S::splat(2 * $n + 1)))
                    .extract::<0>();
                assert!(
                    rel(got, f * cyl) <= 1e-13,
                    "n = {}, x = {x}: spherical {got}, sqrt(pi/2x) * cylindrical {}",
                    $n,
                    f * cyl
                );
            }};
        }

        check!(0, J, false);
        check!(3, J, false);
        check!(7, J, false);
        check!(0, Y, false);
        check!(3, Y, false);
        check!(7, Y, false);
    }
}

/// The Wronskian `j_n(x) y_{n+1}(x) - j_{n+1}(x) y_n(x) = -1/x^2`, which holds _at_ the zeros
/// where nothing relative can be checked, and crosses both recurrence directions at once.
#[test]
fn the_spherical_wronskian_holds() {
    for &x in &[0.1f64, 1.0, 5.0, 25.0, 150.0] {
        let v = V::splat(x);
        let want = -1.0 / (x * x);

        macro_rules! pair {
            ($n:literal, $m:literal) => {{
                let got = v.sph_bessel_n_p::<Precision, J, $n>().extract::<0>()
                    * v.sph_bessel_n_p::<Precision, Y, $m>().extract::<0>()
                    - v.sph_bessel_n_p::<Precision, J, $m>().extract::<0>()
                        * v.sph_bessel_n_p::<Precision, Y, $n>().extract::<0>();
                assert!(
                    rel(got, want) <= 1e-12,
                    "Wronskian at n = {}, x = {x}: got {got}, want {want}",
                    $n
                );
            }};
        }

        pair!(0, 1);
        pair!(2, 3);
        pair!(6, 7);
    }
}

/// `j_0` is exactly `sinc`, which is the whole reason the seeds are spherical rather than
/// rescaled: the wrapper spelling is `0 * inf` at the origin. Bit-exact, and the origin is
/// included.
#[test]
fn j0_is_sinc_and_the_origin_is_exact() {
    use thermite::math::TranscendentalMathWithPolicy;

    for &x in &[0.0f64, 1e-300, 1e-8, 0.5, 3.0, 100.0] {
        let v = V::splat(x);
        assert_eq!(
            v.sph_bessel_n_p::<Precision, J, 0>().extract::<0>().to_bits(),
            v.sinc_p::<Precision>().extract::<0>().to_bits(),
            "j_0({x}) must be sinc({x})"
        );
    }

    // At the origin: j_0 = 1, every higher j is 0, i_0 = 1, y and k are infinite.
    let z = V::splat(0.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, J, 0>().extract::<0>(), 1.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, J, 1>().extract::<0>(), 0.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, J, 5>().extract::<0>(), 0.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, Scaled<I>, 0>().extract::<0>(), 1.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, Scaled<I>, 4>().extract::<0>(), 0.0);
    assert_eq!(z.sph_bessel_n_p::<Precision, Y, 0>().extract::<0>(), f64::NEG_INFINITY);
    assert_eq!(z.sph_bessel_n_p::<Precision, Y, 3>().extract::<0>(), f64::NEG_INFINITY);
    assert_eq!(
        z.sph_bessel_n_p::<Precision, Scaled<K>, 0>().extract::<0>(),
        f64::INFINITY
    );
}

/// Negative `x`: `j`, `y` and `i` are elementary on the whole line and fold by parity
/// (SciPy's convention), while `k` is `e^{-x}` against `e^{x}`, has no parity, and is NaN.
/// Asserted bit-exactly, since the kernels evaluate on `|x|` and apply a sign.
#[test]
fn negative_x_folds_by_parity_and_k_is_undefined() {
    macro_rules! check {
        ($n:literal, $x:expr) => {{
            let p = V::splat($x);
            let m = V::splat(-$x);
            let s: f64 = if $n % 2 == 1 { -1.0 } else { 1.0 };
            let one = |v: V| v.extract::<0>();
            assert_eq!(
                one(m.sph_bessel_n_p::<Precision, J, $n>()).to_bits(),
                (s * one(p.sph_bessel_n_p::<Precision, J, $n>())).to_bits(),
                "j_{}({})",
                $n,
                -$x
            );
            assert_eq!(
                one(m.sph_bessel_n_p::<Precision, Y, $n>()).to_bits(),
                (-s * one(p.sph_bessel_n_p::<Precision, Y, $n>())).to_bits(),
                "y_{}({})",
                $n,
                -$x
            );
            assert_eq!(
                one(m.sph_bessel_n_p::<Precision, Scaled<I>, $n>()).to_bits(),
                (s * one(p.sph_bessel_n_p::<Precision, Scaled<I>, $n>())).to_bits(),
                "e^-x i_{}({})",
                $n,
                -$x
            );
            assert_eq!(
                one(m.sph_bessel_n_p::<Precision, I, $n>()).to_bits(),
                (s * one(p.sph_bessel_n_p::<Precision, I, $n>())).to_bits(),
                "i_{}({})",
                $n,
                -$x
            );
            assert!(
                one(m.sph_bessel_n_p::<Precision, Scaled<K>, $n>()).is_nan(),
                "e^x k_{}({})",
                $n,
                -$x
            );
            assert!(
                one(m.sph_bessel_n_p::<Precision, K, $n>()).is_nan(),
                "k_{}({})",
                $n,
                -$x
            );
        }};
    }

    for &x in &[0.3f64, 2.0, 9.5, 100.0] {
        check!(0, x);
        check!(1, x);
        check!(2, x);
        check!(5, x);
        check!(8, x);
    }
}

/// The unscaled modified forms are the scaled ones times an exponential.
#[test]
fn the_unscaled_modified_spherical_forms_agree() {
    for &x in &[0.25f64, 1.0, 8.0, 60.0, 300.0] {
        let v = V::splat(x);

        macro_rules! check {
            ($n:literal) => {{
                let gi = v.sph_bessel_n_p::<Precision, I, $n>().extract::<0>();
                let wi = v.sph_bessel_n_p::<Precision, Scaled<I>, $n>().extract::<0>() * x.exp();
                assert!(rel(gi, wi) <= 1e-13, "i_{}({x}) unscaled: {gi} vs {wi}", $n);

                let gk = v.sph_bessel_n_p::<Precision, K, $n>().extract::<0>();
                let wk = v.sph_bessel_n_p::<Precision, Scaled<K>, $n>().extract::<0>() * (-x).exp();
                assert!(rel(gk, wk) <= 1e-13, "k_{}({x}) unscaled: {gk} vs {wk}", $n);
            }};
        }

        check!(0);
        check!(2);
        check!(6);
    }
}

/// At real register width, with a different recurrence arm in each lane: `Vector<f64>` is the
/// one-lane scalar seed, so a packet test against it proves nothing about the masks.
#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn a_mixed_packet_agrees_with_its_lanes() {
    use thermite::backend::x86_v3::X86V3;

    type W = Vector<<X86V3 as Simd>::f64x4>;

    // Order 4: lanes 0 and 1 take the downward arm (x < n), lanes 2 and 3 the forward one.
    let x = W::splat(0.5).insert::<1>(2.0).insert::<2>(20.0).insert::<3>(90.0);
    let gj = W::sph_bessel_n_p::<Precision, J, 4>(x);
    let gy = W::sph_bessel_n_p::<Precision, Y, 4>(x);

    macro_rules! lane {
        ($i:literal) => {{
            let xi = x.extract::<$i>();
            let wj = V::splat(xi).sph_bessel_n_p::<Precision, J, 4>().extract::<0>();
            let wy = V::splat(xi).sph_bessel_n_p::<Precision, Y, 4>().extract::<0>();
            assert!(rel(gj.extract::<$i>(), wj) <= 1e-13, "j lane {}", $i);
            assert!(rel(gy.extract::<$i>(), wy) <= 1e-13, "y lane {}", $i);
        }};
    }

    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
}
