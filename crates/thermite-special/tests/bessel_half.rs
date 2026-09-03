//! Correctness gate for **half-integer order** `J` and `Y`, driven through the public
//! `bessel_jv` / `bessel_yv` entries so the dispatch and the `BesselOrder` narrowing are
//! covered along with the kernel.
//!
//! Half-integer order is elementary (`J_{1/2} = sqrt(2/pi x) sin x` and a recurrence), so
//! this arm shares no code with either the integer-order minimax kernels or the general
//! real-order arms in `bessel_nu`. Two independent checks are therefore available and both
//! are used: closed forms at the lowest orders, where the reference has no error of its own,
//! and mpmath at 60 digits everywhere else.
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

include!("common/wide.rs");

type V = Vector<f64>;
type S = <V as GenericVector>::Signed;

fn half(k: i64) -> BesselOrder<V, S> {
    BesselOrder::HalfInteger(S::splat(k))
}

fn j(k: i64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, J>(half(k)).extract::<0>()
}

fn y(k: i64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, Y>(half(k)).extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

/// Envelope-relative, the only meaningful contract for a function with infinitely many zeros:
/// dividing by the true value alone is meaningless at one. The denominator is the **larger**
/// of the envelope and the value, so an unbounded `Y` at small `x` is still graded relatively
/// and only an oscillating value smaller than its own envelope gets the softer denominator.
fn env_rel(got: f64, want: f64, x: f64) -> f64 {
    let env = (2.0 / (core::f64::consts::PI * x)).sqrt();
    (got - want).abs() / env.max(want.abs())
}

/// `J_{\pm 1/2}` and `Y_{\pm 1/2}` against their closed forms, which is the strongest check
/// available anywhere in the Bessel family: the reference carries no error but the `sin`/`cos`
/// and the `sqrt`. This kernel evaluates exactly those.
#[test]
fn the_seed_orders_match_their_closed_forms() {
    for &x in &[0.125f64, 0.5, 1.0, 3.0, 9.0, 25.0, 100.0, 1000.0] {
        let amp = (2.0 / (core::f64::consts::PI * x)).sqrt();

        for (k, want) in [
            (1i64, amp * x.sin()), // J_{1/2}
            (-1, amp * x.cos()),   // J_{-1/2}
        ] {
            let got = j(k, x);
            assert!(
                env_rel(got, want, x) <= 4e-16,
                "J_{k}/2({x}): got {got}, want {want}, env-rel {:e}",
                env_rel(got, want, x)
            );
        }

        for (k, want) in [
            (1i64, -amp * x.cos()), // Y_{1/2}
            (-1, amp * x.sin()),    // Y_{-1/2}
        ] {
            let got = y(k, x);
            assert!(
                env_rel(got, want, x) <= 4e-16,
                "Y_{k}/2({x}): got {got}, want {want}, env-rel {:e}",
                env_rel(got, want, x)
            );
        }
    }
}

/// Both kinds, both signs of order, across all three regimes the kernel has: the forward
/// recurrence (`nu < x`), the downward ratio recurrence (`nu > x`, where `J` underflows hard),
/// and `Y`'s unconditional upward walk (where it overflows hard).
///
/// Graded on **plain relative** error, not the envelope-relative contract the integer-order
/// kernels use, because measured it earns it: the worst row over the whole table is 2.13e-15
/// for `J` and 1.75e-15 for `Y`, including `J_{41/2}(0.5) = 4e-32` thirty-one orders of
/// magnitude under its own envelope. The recurrences here carry no cancellation, so nothing
/// needs the softer denominator.
///
/// mpmath, dps = 60.
#[test]
fn half_integer_orders_match_a_high_precision_reference() {
    // (2*nu, x, J_nu(x), Y_nu(x))
    const ROWS: &[(i64, f64, f64, f64)] = &[
        (1, 0.5, 0.540973789934528, -0.9902458802434049),
        (1, 1.5, 0.6498380747537472, -0.04608316589309741),
        (1, 4.0, -0.30192051329163944, 0.2607660766771788),
        (1, 9.0, 0.10960765886528703, 0.24232558961268508),
        (1, 25.0, -0.021120283599650444, -0.15817308404205055),
        (1, 100.0, -0.04040213271625212, -0.06880309146872808),
        (-1, 0.5, 0.9902458802434049, 0.540973789934528),
        (-1, 1.5, 0.04608316589309741, 0.6498380747537472),
        (-1, 4.0, -0.2607660766771788, -0.30192051329163944),
        (-1, 9.0, -0.24232558961268508, 0.10960765886528703),
        (-1, 25.0, 0.15817308404205055, -0.021120283599650444),
        (-1, 100.0, 0.06880309146872808, -0.04040213271625212),
        (3, 0.5, 0.0917016996256513, -2.521465550421338),
        (3, 1.5, 0.38714221727606746, -0.6805601853491455),
        (3, 4.0, 0.18528594835426895, 0.36711203246093416),
        (3, 9.0, 0.25450421837549475, -0.08268259335276647),
        (3, 25.0, -0.15901789538603658, 0.014793360237968423),
        (3, 100.0, -0.0692071127958906, 0.039714101801564844),
        (-3, 0.5, -2.521465550421338, -0.0917016996256513),
        (-3, 1.5, -0.6805601853491455, -0.38714221727606746),
        (-3, 4.0, 0.36711203246093416, -0.18528594835426895),
        (-3, 9.0, -0.08268259335276647, -0.25450421837549475),
        (-3, 25.0, 0.014793360237968423, 0.15901789538603658),
        (-3, 100.0, 0.039714101801564844, 0.0692071127958906),
        (5, 0.5, 0.009236407819379724, -14.138547422284622),
        (5, 1.5, 0.1244463597983876, -1.3150372048051937),
        (5, 4.0, 0.44088497455734116, 0.0145679476685218),
        (5, 9.0, -0.024772919406788784, -0.26988645406360723),
        (5, 25.0, 0.0020381361533260553, 0.15994828727060678),
        (5, 100.0, 0.038325919332375405, 0.06999451452277503),
        (7, 0.5, 0.0006623785681459423, -138.8640086724249),
        (7, 1.5, 0.027678982051891236, -3.7028971640015),
        (7, 4.0, 0.3658202698424075, -0.3489020978752819),
        (7, 9.0, -0.26826695137926626, -0.06725432557145977),
        (7, 25.0, 0.1594255226167018, 0.01719629721615293),
        (7, 100.0, 0.07112340876250937, -0.03621437607542609),
        (-7, 0.5, -138.8640086724249, -0.0006623785681459423),
        (-7, 1.5, -3.7028971640015, -0.027678982051891236),
        (-7, 4.0, -0.3489020978752819, -0.3658202698424075),
        (-7, 9.0, -0.06725432557145977, 0.26826695137926626),
        (-7, 25.0, 0.01719629721615293, -0.1594255226167018),
        (-7, 100.0, -0.03621437607542609, -0.07112340876250937),
        (21, 0.5, 3.9855051571881206e-14, -761508842905.8378),
        (21, 1.5, 3.9024103138061135e-09, -7849622.952715437),
        (21, 4.0, 8.551705302457991e-05, -383.8391712314001),
        (21, 9.0, 0.08959047506910392, -0.6661406777452541),
        (21, 25.0, -0.14462968429758655, -0.0844095613028708),
        (21, 100.0, -0.0015611238546507794, 0.07999412976470988),
        (41, 0.5, 4.09127045948795e-32, -3.79636185189578e+29),
        (41, 1.5, 2.4140013349652383e-22, -6.449518647286145e+19),
        (41, 4.0, 1.1100150963572128e-13, -142632541459.18634),
        (41, 9.0, 8.474362125706283e-07, -20402.543039348544),
        (41, 25.0, 0.11369883509492514, 0.17566220775898003),
        (41, 100.0, 0.08064754863072786, 0.00044934699219910775),
        (-41, 0.5, 3.79636185189578e+29, 4.09127045948795e-32),
        (-41, 1.5, 6.449518647286145e+19, 2.4140013349652383e-22),
        (-41, 4.0, 142632541459.18634, 1.1100150963572128e-13),
        (-41, 9.0, 20402.543039348544, 8.474362125706283e-07),
        (-41, 25.0, -0.17566220775898003, 0.11369883509492514),
        (-41, 100.0, -0.00044934699219910775, 0.08064754863072786),
    ];

    let mut worst_j = 0.0f64;
    let mut worst_y = 0.0f64;

    for &(k, x, wj, wy) in ROWS {
        let gj = j(k, x);
        let ej = rel(gj, wj);
        worst_j = worst_j.max(ej);
        assert!(ej <= 1e-14, "J_{k}/2({x}): got {gj}, want {wj}, env-rel {ej:e}");

        let gy = y(k, x);
        let ey = rel(gy, wy);
        worst_y = worst_y.max(ey);
        assert!(ey <= 1e-14, "Y_{k}/2({x}): got {gy}, want {wy}, env-rel {ey:e}");
    }

    println!("worst relative: J {worst_j:e}, Y {worst_y:e}");
}

/// The Wronskian `J_{nu+1} Y_nu - J_nu Y_{nu+1} = 2/(pi x)`, which crosses both kinds and both
/// recurrence directions at once. Unlike a relative check, it holds _at_ the zeros.
#[test]
fn the_wronskian_holds_across_orders() {
    for &x in &[0.5f64, 2.0, 7.0, 20.0, 60.0] {
        for k in [-7i64, -3, -1, 1, 3, 9, 15] {
            let want = 2.0 / (core::f64::consts::PI * x);
            let got = j(k + 2, x) * y(k, x) - j(k, x) * y(k + 2, x);
            assert!(
                rel(got, want) <= 1e-13,
                "Wronskian at nu = {k}/2, x = {x}: got {got}, want {want}, rel {:e}",
                rel(got, want)
            );
        }
    }
}

/// The negative-order rule at half-integer order is an _exchange_, not the general rotation:
/// `J_{-(m+1/2)} = (-1)^{m+1} Y_{m+1/2}` and `Y_{-(m+1/2)} = (-1)^m J_{m+1/2}`. Asserted
/// bit-exactly, because the kernel really does reuse the same two values rather than
/// recomputing them through a `sincos_pi`.
#[test]
fn negative_order_is_a_bit_exact_exchange() {
    for &x in &[0.75f64, 3.5, 11.0, 40.0] {
        for m in 0i64..6 {
            let k = 2 * m + 1;
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };

            assert_eq!(
                j(-k, x).to_bits(),
                (-sign * y(k, x)).to_bits(),
                "J_-{k}/2({x}) must be the exchanged Y"
            );
            assert_eq!(
                y(-k, x).to_bits(),
                (sign * j(k, x)).to_bits(),
                "Y_-{k}/2({x}) must be the exchanged J"
            );
        }
    }
}

/// The origin is a limit the seeds cannot produce (`inf * 0`), so it is selected: `J_a(0) = 0`
/// and `Y_a(0) = -inf` at positive order, and the exchange gives the negative orders their
/// signed infinities. Off the positive axis the functions are complex, so NaN.
#[test]
fn the_origin_has_its_limits_and_the_negative_axis_is_nan() {
    for k in [1i64, 3, 9] {
        assert_eq!(j(k, 0.0).to_bits(), 0.0f64.to_bits(), "J_{k}/2(0)");
        assert_eq!(y(k, 0.0), f64::NEG_INFINITY, "Y_{k}/2(0)");
        assert!(
            j(-k, 0.0).is_infinite(),
            "J_-{k}/2(0) must be infinite, got {}",
            j(-k, 0.0)
        );
        assert!(
            j(k, -1.0).is_nan() && y(k, -1.0).is_nan(),
            "order {k}/2 at x = -1 must be NaN"
        );
    }
    // `J_{-1/2}(0) = +inf` and `J_{-3/2}(0) = -inf`: the sign alternates with `m`.
    assert_eq!(j(-1, 0.0), f64::INFINITY);
    assert_eq!(j(-3, 0.0), f64::NEG_INFINITY);
}

/// An even numerator is a whole order, and `simplify` must send it to the integer kernel
/// rather than to this one, so `HalfInteger(2n)` has to equal `Integer(n)` exactly.
#[test]
fn an_even_numerator_reaches_the_integer_kernel() {
    for &x in &[0.5f64, 4.0, 30.0] {
        for n in 0i64..5 {
            let via_half = V::splat(x).bessel_p::<Precision, J>(half(2 * n)).extract::<0>();
            let via_int = V::splat(x)
                .bessel_p::<Precision, J>(BesselOrder::Integer(S::splat(n)))
                .extract::<0>();
            assert_eq!(via_half.to_bits(), via_int.to_bits(), "J_{n}({x}) via HalfInteger(2n)");
        }
    }
}

/// At real register width, with a different order **and** a different recurrence arm in each
/// lane. `Vector<f64>` is the one-lane scalar seed, so a packet test written against it is a
/// single lane and proves nothing about the masks.
#[test]
fn a_mixed_packet_agrees_with_its_lanes() {
    type W = Vector<<Wide as Simd>::f64x4>;
    type WS = <W as GenericVector>::Signed;

    // Lane 0 and 1 take the forward arm (order under x), lane 2 and 3 the downward one.
    let x = W::splat(5.0).insert::<2>(0.5).insert::<3>(2.0);
    let orders = WS::splat(1).insert::<1>(-5).insert::<2>(15).insert::<3>(9);

    let gj = W::bessel_p::<Precision, J>(x, BesselOrder::HalfInteger(orders));
    let gy = W::bessel_p::<Precision, Y>(x, BesselOrder::HalfInteger(orders));

    macro_rules! lane {
        ($i:literal) => {{
            let xi = x.extract::<$i>();
            let ki = orders.extract::<$i>();
            assert!(
                rel(gj.extract::<$i>(), j(ki, xi)) <= 1e-14,
                "J lane {}: packet {}, lane {}",
                $i,
                gj.extract::<$i>(),
                j(ki, xi)
            );
            assert!(
                rel(gy.extract::<$i>(), y(ki, xi)) <= 1e-14,
                "Y lane {}: packet {}, lane {}",
                $i,
                gy.extract::<$i>(),
                y(ki, xi)
            );
        }};
    }

    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
}

// ---------------------------------------------------------------------------------------
// The modified pair, `I` and `K`, at half-integer order.
// ---------------------------------------------------------------------------------------

fn ie(k: i64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, Scaled<I>>(half(k)).extract::<0>()
}

fn ke(k: i64, x: f64) -> f64 {
    V::splat(x).bessel_p::<Precision, Scaled<K>>(half(k)).extract::<0>()
}

/// `I_{\pm 1/2}` and `K_{1/2}` against their closed forms, in the scaled domain so the check
/// still means something at `x = 700` where `sinh` has left binary64 entirely.
#[test]
fn the_modified_seed_orders_match_their_closed_forms() {
    for &x in &[1e-3f64, 0.25, 1.0, 5.0, 40.0, 700.0, 5000.0] {
        let amp = (2.0 / (core::f64::consts::PI * x)).sqrt();
        // e^{-x} sinh x and e^{-x} cosh x, written so neither cancels.
        let want_hi = amp * -((-2.0 * x).exp_m1()) * 0.5;
        let want_lo = amp * (1.0 + (-2.0 * x).exp()) * 0.5;
        let want_k = (core::f64::consts::PI / (2.0 * x)).sqrt();

        assert!(
            rel(ie(1, x), want_hi) <= 4e-16,
            "I_1/2({x}) scaled: {} vs {want_hi}",
            ie(1, x)
        );
        assert!(
            rel(ie(-1, x), want_lo) <= 4e-16,
            "I_-1/2({x}) scaled: {} vs {want_lo}",
            ie(-1, x)
        );
        assert!(
            rel(ke(1, x), want_k) <= 4e-16,
            "K_1/2({x}) scaled: {} vs {want_k}",
            ke(1, x)
        );
        assert!(ke(-1, x).to_bits() == ke(1, x).to_bits(), "K is even in order");
    }
}

/// Both kinds, both signs of order, in the scaled domain. mpmath, dps = 60.
///
/// `I` at **negative** order is graded against the larger of the two terms its reflection
/// combines rather than against its own value, because `I_{-(m+1/2)}` genuinely has zeros.
/// The row at `nu = -3/2, x = 1.1997` sits on one, where the true value is five orders of
/// magnitude below both terms that produce it. That is the same contract `J` gets at its
/// zeros, and Boost's `bessel_ik` has the identical exposure.
#[test]
fn modified_half_integer_orders_match_a_high_precision_reference() {
    // (2*nu, x, e^{-x} I_nu(x), e^{x} K_nu(x))
    const ROWS: &[(i64, f64, f64, f64)] = &[
        (1, 0.25, 0.3139431117645787, 2.5066282746310007),
        (1, 1.0, 0.3449513138882446, 1.2533141373155003),
        (1, 1.1997, 0.3311664607287076, 1.1442570821556068),
        (1, 4.0, 0.1994042250878339, 0.6266570686577502),
        (1, 15.0, 0.10300645387284092, 0.3236043187592832),
        (1, 80.0, 0.044603102903819275, 0.14012478040994822),
        (1, 700.0, 0.015078600877302686, 0.04737082174254673),
        (-1, 0.25, 1.281826009841152, 2.5066282746310007),
        (-1, 1.0, 0.4529332469146207, 1.2533141373155003),
        (-1, 1.1997, 0.3972902224431876, 1.1442570821556068),
        (-1, 4.0, 0.19953805531359878, 0.6266570686577502),
        (-1, 15.0, 0.1030064538728602, 0.3236043187592832),
        (-1, 80.0, 0.044603102903819275, 0.14012478040994822),
        (-1, 700.0, 0.015078600877302686, 0.04737082174254673),
        (3, 0.25, 0.02605356278283743, 12.533141373155003),
        (3, 1.0, 0.1079819330263761, 2.5066282746310007),
        (3, 1.1997, 0.12124916157071318, 2.0980430971223543),
        (3, 4.0, 0.14968699904164032, 0.7833213358221877),
        (3, 15.0, 0.09613935694800413, 0.3451779400099021),
        (3, 80.0, 0.044045564117521536, 0.14187634016507256),
        (3, 700.0, 0.01505706001890654, 0.04743849434503608),
        (-3, 0.25, -4.81336092760003, 12.533141373155003),
        (-3, 1.0, -0.1079819330263761, 2.5066282746310007),
        (-3, 1.1997, 8.485865668813028e-06, 2.0980430971223543),
        (-3, 4.0, 0.1495197112594342, 0.7833213358221877),
        (-3, 15.0, 0.09613935694798358, 0.3451779400099021),
        (-3, 80.0, 0.044045564117521536, 0.14187634016507256),
        (-3, 700.0, 0.01505706001890654, 0.04743849434503608),
        (7, 0.25, 4.6395372247585924e-05, 3070.6196364229754),
        (7, 1.0, 0.0029543589807945326, 46.372623080673506),
        (7, 1.1997, 0.004687699846848976, 28.73252017388489),
        (7, 4.0, 0.040763279283385724, 2.301006423977676),
        (7, 15.0, 0.0682131627869241, 0.4760579089303233),
        (7, 80.0, 0.041361101976933275, 0.15096666161295622),
        (7, 700.0, 0.014949816657334547, 0.047778309556289325),
        (-7, 0.25, -1185.6565037484302, 3070.6196364229754),
        (-7, 1.0, -3.9923771629951212, 46.372623080673506),
        (-7, 1.1997, -1.6556929498016784, 28.73252017388489),
        (-7, 4.0, 0.04027187142315529, 2.301006423977676),
        (-7, 15.0, 0.06821316278689574, 0.4760579089303233),
        (-7, 80.0, 0.041361101976933275, 0.15096666161295622),
        (-7, 700.0, 0.014949816657334547, 0.047778309556289325),
        (21, 0.25, 2.157971237671164e-17, 2206027206718349.2),
        (21, 1.0, 2.181713152171502e-11, 2172725468.275253),
        (21, 1.1997, 1.2203109880166935e-10, 387675981.42761016),
        (21, 4.0, 3.1415027556634833e-06, 14159.033401008608),
        (21, 15.0, 0.0027045272338924813, 10.095406390737033),
        (21, 80.0, 0.02235319570212109, 0.2772292535074653),
        (21, 700.0, 0.013938439053501126, 0.051239995373102094),
        (-21, 0.25, 851811984921714.1, 2206027206718349.2),
        (-21, 1.0, 187195762.82968208, 2172725468.275253),
        (-21, 1.1997, 22402827.667061318, 387675981.42761016),
        (-21, 4.0, 3.023836640625768, 14159.033401008608),
        (-21, 15.0, 0.0027045272344938896, 10.095406390737033),
        (-21, 80.0, 0.02235319570212109, 0.2772292535074653),
        (-21, 700.0, 0.013938439053501126, 0.051239995373102094),
    ];

    let mut worst_i = 0.0f64;
    let mut worst_k = 0.0f64;

    for &(k, x, wi, wk) in ROWS {
        let gk = ke(k, x);
        let ek = rel(gk, wk);
        worst_k = worst_k.max(ek);
        assert!(ek <= 4e-15, "K_{k}/2({x}) scaled: got {gk}, want {wk}, rel {ek:e}");

        let gi = ie(k, x);
        // At negative order the value is a difference of two terms. Grade against the larger.
        let scale = if k < 0 {
            let both = ie(-k, x).abs() + core::f64::consts::FRAC_2_PI * (-2.0 * x).exp() * ke(-k, x);
            both.max(wi.abs())
        } else {
            wi.abs()
        };
        let ei = (gi - wi).abs() / scale;
        worst_i = worst_i.max(ei);
        assert!(
            ei <= 4e-15,
            "I_{k}/2({x}) scaled: got {gi}, want {wi}, rel-to-scale {ei:e}"
        );
    }

    println!("worst: I {worst_i:e}, K {worst_k:e}");
}

/// The modified Wronskian `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/x`, which crosses the downward
/// ratio recurrence and the upward one in a single identity, and is scale-invariant since
/// the two scalings cancel in every product.
/// The modified origin: `I_a(0) = 0`, `K_a(0) = +inf`, and the reflection's `K` term gives
/// `I_{-a}(0)` a signed infinity, `+` at `m` even and `-` at `m` odd.
#[test]
fn the_modified_origin_has_its_limits() {
    for k in [1i64, 3, 9] {
        assert_eq!(ie(k, 0.0).to_bits(), 0.0f64.to_bits(), "I_{k}/2(0)");
        assert_eq!(ke(k, 0.0), f64::INFINITY, "K_{k}/2(0)");
        assert!(
            ie(k, -1.0).is_nan() && ke(k, -1.0).is_nan(),
            "order {k}/2 at x = -1 must be NaN"
        );
    }
    assert_eq!(ie(-1, 0.0), f64::INFINITY);
    assert_eq!(ie(-3, 0.0), f64::NEG_INFINITY);
    assert_eq!(ie(-5, 0.0), f64::INFINITY);
}

#[test]
fn the_modified_wronskian_holds_across_orders() {
    for &x in &[0.25f64, 1.0, 6.0, 30.0, 200.0] {
        for k in [1i64, 3, 5, 11, 25] {
            let want = 1.0 / x;
            let got = ie(k, x) * ke(k + 2, x) + ie(k + 2, x) * ke(k, x);
            assert!(
                rel(got, want) <= 1e-13,
                "modified Wronskian at nu = {k}/2, x = {x}: got {got}, want {want}, rel {:e}",
                rel(got, want)
            );
        }
    }
}

/// The unscaled forms are the scaled ones times an exponential, and must agree with them
/// wherever both are representable. The scaled column is itself pinned to mpmath above, so a
/// shared wrong value cannot pass this.
#[test]
fn the_unscaled_modified_forms_agree_with_the_scaled_ones() {
    for &x in &[0.25f64, 1.0, 7.0, 50.0, 300.0] {
        for k in [-7i64, -1, 1, 5, 15] {
            let gi = V::splat(x).bessel_p::<Precision, I>(half(k)).extract::<0>();
            let gk = V::splat(x).bessel_p::<Precision, K>(half(k)).extract::<0>();

            let wi = ie(k, x) * x.exp();
            let wk = ke(k, x) * (-x).exp();

            assert!(rel(gi, wi) <= 1e-13, "I_{k}/2({x}) unscaled: {gi} vs {wi}");
            assert!(rel(gk, wk) <= 1e-13, "K_{k}/2({x}) unscaled: {gk} vs {wk}");
        }
    }
}

/// An even numerator must reach the integer `I`/`K` kernels, not this one.
#[test]
fn an_even_numerator_reaches_the_integer_modified_kernel() {
    for &x in &[0.5f64, 4.0, 30.0] {
        for n in 0i64..5 {
            let via_half = V::splat(x).bessel_p::<Precision, Scaled<I>>(half(2 * n)).extract::<0>();
            let via_int = V::splat(x)
                .bessel_p::<Precision, Scaled<I>>(BesselOrder::Integer(S::splat(n)))
                .extract::<0>();
            assert_eq!(via_half.to_bits(), via_int.to_bits(), "I_{n}({x}) via HalfInteger(2n)");
        }
    }
}
