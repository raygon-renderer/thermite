//! Correctness gate for **arbitrary real order** `I` and `K`, driven through the public
//! `bessel_iv` / `bessel_kv` entries.
//!
//! This is the arm Airy needs: Boost's Airy functions reach `Ai`/`Bi` for `x > 0` through
//! `cyl_bessel_k(1/3, p)` and `cyl_bessel_i(+-1/3, p)`, so the fractional-order `J`/`Y`
//! machinery does not reach it. Everything is graded in the **scaled** domain, which is the
//! form the algorithm natively produces and the only one that still means something at
//! `x = 1e5`.
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
use thermite_special::bessel::{I, J, K, Scaled, Y};
use thermite_special::{BesselOrder, SpecialMathWithPolicy};

type V = Vector<f64>;
type S = <V as GenericVector>::Signed;

fn real(v: f64) -> BesselOrder<V, S> {
    BesselOrder::Real(V::splat(v))
}

fn ie(v: f64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, Scaled<I>>(real(v)).extract::<0>()
}

fn ke(v: f64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, Scaled<K>>(real(v)).extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

/// Both kinds at arbitrary real order, in the scaled domain. mpmath, dps = 60.
///
/// The grid crosses every region boundary the kernel has: Temme below `x = 2`, the `CF2`
/// continued fraction above it, and (for `I`) the handover from the Wronskian route to the
/// asymptotic series at `max(40, nu^2/3)`, which the `x = 39` and `x = 60` columns straddle.
///
/// `I` at **negative** order is graded against the larger of the two terms its reflection
/// combines, for the same reason `bessel_half` does: `I_{-nu}` has genuine zeros where
/// `I_nu = -(2/pi) sin(nu pi) K_nu`.
#[test]
fn real_orders_match_a_high_precision_reference() {
    // (nu, x, e^{-x} I_nu(x), e^{x} K_nu(x))
    const ROWS: &[(f64, f64, f64, f64)] = &[
        (0.3333333333333333, 0.01, 0.18958924846366826, 7.56146247517869),
        (0.3333333333333333, 0.5, 0.4482098760764547, 1.630636569493972),
        (0.3333333333333333, 1.0, 0.3916560037091716, 1.1917780239239115),
        (0.3333333333333333, 2.0, 0.2921594520963623, 0.8611572570650664),
        (0.3333333333333333, 2.5, 0.26056440002871933, 0.7741065787900635),
        (0.3333333333333333, 5.0, 0.18123428698581887, 0.5534141329042368),
        (0.3333333333333333, 15.0, 0.1035016315040471, 0.32215624061671705),
        (0.3333333333333333, 39.0, 0.0639972428402738, 0.20033829925069443),
        (0.3333333333333333, 60.0, 0.051563377697468195, 0.16161652956463265),
        (0.3333333333333333, 745.0, 0.01461747766718456, 0.045913600065012904),
        (0.3333333333333333, 100000.0, 0.001261567137102446, 0.003963324545310105),
        (0.6666666666666666, 0.01, 0.03206847736848072, 23.330216878838684),
        (0.6666666666666666, 0.5, 0.2767497691964688, 1.988243208169708),
        (0.6666666666666666, 1.0, 0.2970704803864662, 1.3441225759440272),
        (0.6666666666666666, 2.0, 0.25834819376995755, 0.9224418385401156),
        (0.6666666666666666, 2.5, 0.2378918634206441, 0.8193375561809677),
        (0.6666666666666666, 5.0, 0.17452839586355412, 0.5705632049374416),
        (0.6666666666666666, 15.0, 0.10231728476034503, 0.3256423533462897),
        (0.6666666666666666, 39.0, 0.06372075038608373, 0.20118552472384155),
        (0.6666666666666666, 60.0, 0.051419134411961956, 0.16206239744062204),
        (0.6666666666666666, 745.0, 0.014614205706561176, 0.04592386582691784),
        (0.6666666666666666, 100000.0, 0.00126156503448179, 0.0039633311508234905),
        (-0.3333333333333333, 0.01, 4.275893187355048, 7.56146247517869),
        (-0.3333333333333333, 0.5, 0.7789397692322102, 1.630636569493972),
        (-0.3333333333333333, 1.0, 0.4805796297746208, 1.1917780239239115),
        (-0.3333333333333333, 2.0, 0.30085536723330236, 0.8611572570650664),
        (-0.3333333333333333, 2.5, 0.2634400704046062, 0.7741065787900635),
        (-0.3333333333333333, 5.0, 0.18124813910378673, 0.5534141329042368),
        (-0.3333333333333333, 15.0, 0.10350163150406373, 0.32215624061671705),
        (-0.3333333333333333, 39.0, 0.0639972428402738, 0.20033829925069443),
        (-0.3333333333333333, 60.0, 0.051563377697468195, 0.16161652956463265),
        (-0.3333333333333333, 745.0, 0.01461747766718456, 0.045913600065012904),
        (
            -0.3333333333333333,
            100000.0,
            0.001261567137102446,
            0.003963324545310105,
        ),
        (-0.6666666666666666, 0.01, 12.639994184879948, 23.330216878838684),
        (-0.6666666666666666, 0.5, 0.6800103584089905, 1.988243208169708),
        (-0.6666666666666666, 1.0, 0.39736118128044917, 1.3441225759440272),
        (-0.6666666666666666, 2.0, 0.2676629569922359, 0.9224418385401156),
        (-0.6666666666666666, 2.5, 0.24093555895652735, 0.8193375561809677),
        (-0.6666666666666666, 5.0, 0.17454267722782185, 0.5705632049374416),
        (-0.6666666666666666, 15.0, 0.10231728476036184, 0.3256423533462897),
        (-0.6666666666666666, 39.0, 0.06372075038608373, 0.20118552472384155),
        (-0.6666666666666666, 60.0, 0.051419134411961956, 0.16206239744062204),
        (-0.6666666666666666, 745.0, 0.014614205706561176, 0.04592386582691784),
        (
            -0.6666666666666666,
            100000.0,
            0.00126156503448179,
            0.0039633311508234905,
        ),
        (0.1, 0.01, 0.6126652553841286, 4.984260227658206),
        (0.1, 0.5, 0.5870242243518855, 1.533453444170723),
        (0.1, 1.0, 0.44780935055119725, 1.1486533294972618),
        (0.1, 2.0, 0.3058917345860589, 0.843314476515855),
        (0.1, 2.5, 0.26880737830903023, 0.7608484958574343),
        (0.1, 5.0, 0.18333005343608083, 0.5483099345775357),
        (0.1, 15.0, 0.10386365628187266, 0.32110603721457026),
        (0.1, 39.0, 0.06408135127526977, 0.20008200774046386),
        (0.1, 60.0, 0.0516072118879275, 0.16148152430761384),
        (0.1, 745.0, 0.01461847030677043, 0.04591048657067349),
        (0.1, 100000.0, 0.001261567774898071, 0.0039633225416398875),
        (2.25, 0.01, 2.5818397970309517e-06, 86070.2076143503),
        (2.25, 0.5, 0.010718578103833048, 20.148440701201928),
        (2.25, 1.0, 0.032740485857130436, 6.139518797400099),
        (2.25, 2.0, 0.07147171382572622, 2.300307090618907),
        (2.25, 2.5, 0.08397275310017453, 1.7571800123517136),
        (2.25, 5.0, 0.10523247221991969, 0.8668808907094938),
        (2.25, 15.0, 0.08726796679789102, 0.3779188425425007),
        (2.25, 39.0, 0.06001188878075518, 0.21329538096245623),
        (2.25, 60.0, 0.04946201901711514, 0.16836689525123733),
        (2.25, 745.0, 0.014568950740495704, 0.046066325702163805),
        (2.25, 100000.0, 0.0012615359047853686, 0.0039634226658396615),
        (-1.75, 0.01, -2178.337954374751, 4936.805696495882),
        (-1.75, 0.5, -1.286181224582209, 7.972500146780138),
        (-1.75, 1.0, -0.12506040796414722, 3.273906042852711),
        (-1.75, 2.0, 0.10624874303786494, 1.5613482734085404),
        (-1.75, 2.5, 0.1243946188709302, 1.2692657319090126),
        (-1.75, 5.0, 0.13056740348639786, 0.7239817275479057),
        (-1.75, 15.0, 0.09348540302335155, 0.35433950202382963),
        (-1.75, 39.0, 0.061590622780512416, 0.20796451983650263),
        (-1.75, 60.0, 0.05030024351113054, 0.16560712423938873),
        (-1.75, 745.0, 0.014588532646929182, 0.04600457464726732),
        (-1.75, 100000.0, 0.0012615485202705707, 0.003963383032009343),
        (7.4, 0.01, 8.14530374130211e-22, 8.295271345522513e+19),
        (7.4, 0.5, 1.8780614898379506e-09, 35893947.55034693),
        (7.4, 1.0, 1.9672497600992408e-07, 340314.0266607233),
        (7.4, 2.0, 1.3355531365177961e-05, 4881.3115821446745),
        (7.4, 2.5, 4.5109906714645705e-05, 1418.0104711945078),
        (7.4, 5.0, 0.0010655842961390185, 52.47785655786768),
        (7.4, 15.0, 0.016356205390241303, 1.827666638209827),
        (7.4, 39.0, 0.031539002542087975, 0.39939672736768733),
        (7.4, 60.0, 0.032594536105480625, 0.25375190306037365),
        (7.4, 745.0, 0.014090720478212868, 0.047627655803409774),
        (7.4, 100000.0, 0.0012612224662595735, 0.003964407644276332),
        (-7.4, 0.01, -4.9230149613045875e+19, 8.295271345522513e+19),
        (-7.4, 0.5, -7994902.844324457, 35893947.55034693),
        (-7.4, 1.0, -27885.42926814391, 340314.0266607233),
        (-7.4, 2.0, -54.130862125746226, 4881.3115821446745),
        (-7.4, 2.5, -5.784823200055453, 1418.0104711945078),
        (-7.4, 5.0, -0.00037692200059529743, 52.47785655786768),
        (-7.4, 15.0, 0.01635620539013775, 1.827666638209827),
        (-7.4, 39.0, 0.031539002542087975, 0.39939672736768733),
        (-7.4, 60.0, 0.032594536105480625, 0.25375190306037365),
        (-7.4, 745.0, 0.014090720478212868, 0.047627655803409774),
        (-7.4, 100000.0, 0.0012612224662595735, 0.003964407644276332),
    ];

    let mut worst_i = 0.0f64;
    let mut worst_k = 0.0f64;

    for &(v, x, wi, wk) in ROWS {
        let gk = ke(v, x);
        let ek = rel(gk, wk);
        worst_k = worst_k.max(ek);
        assert!(ek <= 5e-15, "K_{v}({x}) scaled: got {gk}, want {wk}, rel {ek:e}");

        let gi = ie(v, x);
        let scale = if v < 0.0 {
            let both = ie(-v, x).abs()
                + core::f64::consts::FRAC_2_PI * (v * core::f64::consts::PI).sin().abs() * (-2.0 * x).exp() * ke(-v, x);
            both.max(wi.abs())
        } else {
            wi.abs()
        };
        let ei = (gi - wi).abs() / scale;
        worst_i = worst_i.max(ei);
        assert!(
            ei <= 5e-15,
            "I_{v}({x}) scaled: got {gi}, want {wi}, rel-to-scale {ei:e}"
        );
    }

    println!("worst: I {worst_i:e}, K {worst_k:e}");
}

/// The modified Wronskian `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/x`, which crosses both `K` arms,
/// the upward walk and the `I` route in one identity, and is scale-invariant, since the two
/// scalings cancel in every product.
///
/// This is a genuinely independent check of the `I` values, because the kernel builds `I` from
/// _this_ identity at order `nu` only. The `nu + 1` evaluation is a separate call with its own
/// order reduction and its own walk length.
/// The origin and the negative axis at real order, for all four families. These lanes are
/// excluded from every convergence mask (a NaN term never converges, so before that a packet
/// with one such lane ran every series to `max_iterations`) and get their limits by select:
/// `J_a(0) = 0`, `Y_a(0) = -inf`, `I_a(0) = 0`, `K_a(0) = +inf` at `a > 0`, the rotation's
/// signed infinities at `-a`, and NaN for `x < 0` where the functions are complex.
#[test]
fn the_origin_and_the_negative_axis_at_real_order() {
    let jv = |v: f64, x: f64| V::splat(x).bessel_p::<Precision, J>(real(v)).extract::<0>();
    let yv = |v: f64, x: f64| V::splat(x).bessel_p::<Precision, Y>(real(v)).extract::<0>();

    for v in [1.0 / 3.0, 0.75, 2.25, 7.4] {
        assert_eq!(jv(v, 0.0).to_bits(), 0.0f64.to_bits(), "J_{v}(0)");
        assert_eq!(yv(v, 0.0), f64::NEG_INFINITY, "Y_{v}(0)");
        assert_eq!(ie(v, 0.0).to_bits(), 0.0f64.to_bits(), "I_{v}(0)");
        assert_eq!(ke(v, 0.0), f64::INFINITY, "K_{v}(0)");
        assert!(jv(-v, 0.0).is_infinite(), "J_-{v}(0) must be infinite");
        assert!(ie(-v, 0.0).is_infinite(), "I_-{v}(0) must be infinite");

        for x in [-1.0, -1e-3, -50.0] {
            assert!(jv(v, x).is_nan() && yv(v, x).is_nan(), "J/Y_{v}({x}) must be NaN");
            assert!(ie(v, x).is_nan() && ke(v, x).is_nan(), "I/K_{v}({x}) must be NaN");
        }
    }

    // `1/Gamma(1 - nu)` sets the sign of `J_{-nu}(0)`: positive on (0, 1), negative on (1, 2).
    assert_eq!(jv(-0.75, 0.0), f64::INFINITY);
    assert_eq!(jv(-1.25, 0.0), f64::NEG_INFINITY);
}

#[test]
fn the_modified_wronskian_holds_at_real_order() {
    for &x in &[0.1f64, 1.0, 2.0, 3.0, 10.0, 50.0, 300.0] {
        for &v in &[1.0 / 3.0, 2.0 / 3.0, 0.1, 1.25, 4.75] {
            let want = 1.0 / x;
            let got = ie(v, x) * ke(v + 1.0, x) + ie(v + 1.0, x) * ke(v, x);
            assert!(
                rel(got, want) <= 1e-13,
                "modified Wronskian at nu = {v}, x = {x}: got {got}, want {want}, rel {:e}",
                rel(got, want)
            );
        }
    }
}

/// `K` is even in order at every order, integer or not, and the kernel takes `|nu|` before
/// anything else, so this must hold **bit-exactly**.
#[test]
fn k_is_bit_exactly_even_in_order() {
    for &x in &[0.5f64, 2.5, 20.0] {
        for &v in &[1.0 / 3.0, 0.7, 3.25] {
            assert_eq!(ke(v, x).to_bits(), ke(-v, x).to_bits(), "K_{v}({x}) vs K_-{v}({x})");
        }
    }
}

/// A whole order handed in as `Real` must reach the integer kernels bit-exactly. That is what
/// `BesselOrder::simplify` is for, and is the property that makes `Real` safe to reach for.
#[test]
fn a_whole_order_given_as_real_reaches_the_integer_kernel() {
    for &x in &[0.5f64, 4.0, 30.0] {
        for n in 0i64..4 {
            let via_real = ie(n as f64, x);
            let via_int = V::splat(x)
                .bessel_p::<Precision, Scaled<I>>(BesselOrder::Integer(S::splat(n)))
                .extract::<0>();
            assert_eq!(via_real.to_bits(), via_int.to_bits(), "I_{n}({x}) via Real");

            let via_real = ke(n as f64, x);
            let via_int = V::splat(x)
                .bessel_p::<Precision, Scaled<K>>(BesselOrder::Integer(S::splat(n)))
                .extract::<0>();
            assert_eq!(via_real.to_bits(), via_int.to_bits(), "K_{n}({x}) via Real");
        }
    }
}

/// The unscaled forms are the scaled ones times an exponential, including the halved-exponent
/// path past `far_threshold` that keeps `I` representable a little further than a single
/// `exp` would.
#[test]
fn the_unscaled_real_order_forms_agree_with_the_scaled_ones() {
    for &x in &[0.25f64, 1.0, 7.0, 60.0, 300.0, 700.0] {
        for &v in &[1.0 / 3.0, -2.0 / 3.0, 2.25] {
            let gi = V::splat(x).bessel_p::<Precision, I>(real(v)).extract::<0>();
            let gk = V::splat(x).bessel_p::<Precision, K>(real(v)).extract::<0>();

            let wi = ie(v, x) * x.exp();
            let wk = ke(v, x) * (-x).exp();

            assert!(rel(gi, wi) <= 1e-13, "I_{v}({x}) unscaled: {gi} vs {wi}");
            assert!(rel(gk, wk) <= 1e-13, "K_{v}({x}) unscaled: {gk} vs {wk}");
        }
    }
}

/// At real register width, with a different order **and** a different region in every lane:
/// Temme, `CF2`, the `I` asymptotic handover, and a negative order that has to reflect.
/// `Vector<f64>` is the one-lane scalar seed, so a packet test written against it proves
/// nothing about the region masks.
#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn a_mixed_packet_agrees_with_its_lanes() {
    use thermite::backend::x86_v3::X86V3;

    type W = Vector<<X86V3 as Simd>::f64x4>;

    let x = W::splat(1.0).insert::<1>(5.0).insert::<2>(100.0).insert::<3>(0.5);
    let nus = W::splat(1.0 / 3.0)
        .insert::<1>(2.25)
        .insert::<2>(2.0 / 3.0)
        .insert::<3>(-1.75);

    let gi = W::bessel_p::<Precision, Scaled<I>>(x, BesselOrder::Real(nus));
    let gk = W::bessel_p::<Precision, Scaled<K>>(x, BesselOrder::Real(nus));

    macro_rules! lane {
        ($i:literal) => {{
            let xi = x.extract::<$i>();
            let vi = nus.extract::<$i>();
            assert!(
                rel(gi.extract::<$i>(), ie(vi, xi)) <= 1e-13,
                "I lane {}: packet {}, lane {}",
                $i,
                gi.extract::<$i>(),
                ie(vi, xi)
            );
            assert!(
                rel(gk.extract::<$i>(), ke(vi, xi)) <= 1e-13,
                "K lane {}: packet {}, lane {}",
                $i,
                gk.extract::<$i>(),
                ke(vi, xi)
            );
        }};
    }

    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
}
