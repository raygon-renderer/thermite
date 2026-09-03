//! Correctness gate for the real-argument polylogarithm, driven through the public
//! `SpecialMath::polylog` entry (the dispatched path). References from
//! `scripts/polylog_ref.py` (mpmath, 40 digits). Only the real-argument rows are used here,
//! and the real kernel is graded on the real part.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
// The generated table carries values that happen to be ln 2 and the like.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::{PolylogOrder, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

include!("polylog_ref/table.rs");

fn order64(s: f64) -> PolylogOrder<f64, i64> {
    if s == s.round() {
        PolylogOrder::Integer(s as i64)
    } else {
        PolylogOrder::Real(s)
    }
}

fn order32(s: f32) -> PolylogOrder<f32, i32> {
    if s == s.round() {
        PolylogOrder::Integer(s as i32)
    } else {
        PolylogOrder::Real(s)
    }
}

fn li64<P: thermite::math::policy::Policy>(s: f64, z: f64) -> f64 {
    D::splat(z).polylog_p::<P>(order64(s)).extract::<0>()
}

fn li32(s: f32, z: f32) -> f32 {
    F::splat(z).polylog_p::<Precision>(order32(s)).extract::<0>()
}

/// Which arm a real `z` takes, for reporting.
fn region(z: f64) -> &'static str {
    let a = z.abs().ln();
    let b = if z < 0.0 { std::f64::consts::PI } else { 0.0 };
    let q = a * a + b * b;
    let tau = std::f64::consts::TAU;
    if (tau * z) * (tau * z) < q {
        "series"
    } else if q <= (tau * 0.512) * (tau * 0.512) {
        "unity"
    } else {
        "far"
    }
}

/// Relative error of the real part, judged **normwise** against the full complex value: on
/// the cut `z > 1` the real part can be a millionth of the imaginary part (`s = 1 + 1e-6`,
/// `z = 2` has `Re = 3.5e-6` against `Im = pi`), and the real kernel's contract there is the
/// real part to normwise precision, not componentwise.
fn rel(got: f64, want: f64, want_im: f64) -> f64 {
    let scale = want.abs().max(want_im.abs());
    if scale == 0.0 {
        return if got == 0.0 { 0.0 } else { got.abs() };
    }
    ((got - want) / scale).abs()
}

/// Per-kind gates, f64. Non-negative integer orders are table-driven and sit at a few ulp.
/// Negative integer orders pay an alternating defining series on the negative axis (the
/// terms `k^p z^k` peak at ~2500x the sum for `p = 6`, `z = -1/2`, and the far field
/// reflects onto exactly that point). Real order is Roughan's 1e-12 class. Its far field
/// is the m-th-root identity, whose terms are `m^{s-1}` times the answer.
fn gate64(s: f64, region: &str) -> f64 {
    if s == s.round() {
        if s >= 0.0 { 1e-13 } else { 4e-13 }
    } else if region == "far" {
        4e-12
    } else {
        1e-12
    }
}

/// f32 gates, same shape: the alternating series and the root cancellation scale with the
/// format's epsilon, so they are the same multiples of it.
fn gate32(s: f64, region: &str) -> f64 {
    if s == s.round() {
        if s >= 0.0 { 4e-6 } else { 2e-4 }
    } else if region == "far" {
        1e-3
    } else {
        2e-5
    }
}

struct Worst {
    err: f64,
    s: f64,
    z: f64,
    got: f64,
    want: f64,
}

/// Grade every real-argument row at one policy. Returns the count over the gate and prints
/// the worst per (kind, region).
fn sweep64<P: thermite::math::policy::Policy>() -> usize {
    let mut worst: std::collections::BTreeMap<(&str, &str), Worst> = Default::default();
    let mut bad = 0usize;
    let mut skipped = 0usize;
    for &(s, zr, zi, want, want_im) in POLYLOG.iter() {
        if zi != 0.0 {
            continue;
        }
        // Cancellation to (near) zero is judged on its own below. A relative gate is
        // unpassable there for any implementation.
        if want.abs().max(want_im.abs()) < 1e-12 {
            skipped += 1;
            continue;
        }
        let got = li64::<P>(s, zr);
        let e = rel(got, want, want_im);
        let kind = if s == s.round() { "int" } else { "real" };
        let gate = gate64(s, region(zr));
        if e.is_nan() || e > gate {
            bad += 1;
            if bad <= 40 {
                std::println!(
                    "  OVER {kind:4} {:6} s={s:<12} z={zr:<10e} got {got:.17e} want {want:.17e} rel {e:.2e}",
                    region(zr)
                );
            }
        }
        let w = worst.entry((kind, region(zr))).or_insert(Worst {
            err: -1.0,
            s,
            z: zr,
            got,
            want,
        });
        if e > w.err || e.is_nan() {
            *w = Worst {
                err: e,
                s,
                z: zr,
                got,
                want,
            };
        }
    }
    for ((kind, reg), w) in &worst {
        std::println!(
            "  worst {kind:4} {reg:6}: {:.2e} at s={} z={:e} (got {:.16e} want {:.16e})",
            w.err,
            w.s,
            w.z,
            w.got,
            w.want
        );
    }
    std::println!("  {bad} rows over the gate, {skipped} near-zero rows set aside");
    bad
}

#[test]
fn polylog_f64_precision_matches_mpmath() {
    std::println!("f64 Precision:");
    let bad = sweep64::<Precision>();
    assert_eq!(bad, 0);
}

#[test]
fn polylog_f64_default_matches_mpmath() {
    std::println!("f64 Performance (default):");
    let bad = sweep64::<Performance>();
    assert_eq!(bad, 0);
}

#[test]
fn polylog_f32_matches_mpmath() {
    let mut worst = 0.0f64;
    let mut bad = 0usize;
    let mut rows = 0usize;
    for &(s, zr, zi, want, want_im) in POLYLOG.iter() {
        if zi != 0.0 || want.abs().max(want_im.abs()) < 1e-6 {
            continue;
        }
        let sf = s as f32;
        let zf = zr as f32;
        // Only arguments and orders exact in f32, so there is no condition-number term.
        if sf as f64 != s || zf as f64 != zr || !(want as f32).is_finite() || want.abs() < 1e-30 {
            continue;
        }
        rows += 1;
        let got = li32(sf, zf) as f64;
        let e = rel(got, want, want_im);
        let kind = if s == s.round() { "int" } else { "real" };
        let gate = gate32(s, region(zr));
        if e.is_nan() || e > gate {
            bad += 1;
            if bad <= 40 {
                std::println!(
                    "  OVER f32 {kind:4} {:6} s={s:<8} z={zr:<10e} got {got:.9e} want {want:.9e} rel {e:.2e}",
                    region(zr)
                );
            }
        }
        if e > worst {
            worst = e;
        }
    }
    std::println!("f32: worst {worst:.2e} over {rows} rows, {bad} over the gate");
    assert_eq!(bad, 0);
}

/// Rows whose reference is (nearly) zero: judge absolutely against the size of the terms
/// that cancelled, `|z|/(1-z)` for the rational arm and 1 elsewhere.
#[test]
fn polylog_f64_near_zero_rows() {
    let mut worst = 0.0f64;
    for &(s, zr, zi, want, want_im) in POLYLOG.iter() {
        if zi != 0.0 || want.abs().max(want_im.abs()) >= 1e-12 {
            continue;
        }
        let got = li64::<Precision>(s, zr);
        let scale = if s <= 0.0 {
            (zr / (1.0 - zr)).abs().max(1.0)
        } else {
            1.0
        };
        let e = (got - want).abs() / scale;
        if e > worst {
            worst = e;
            std::println!("  near-zero s={s} z={zr:e}: got {got:e} want {want:e} abs/scale {e:.2e}");
        }
    }
    assert!(worst < 1e-13, "worst {worst:e}");
}

/// Lanes on different arms inside one packet.
///
/// `Vector<f64>` is the 1-lane seed, so this is stamped on real registers. A mixed packet
/// is allowed to differ from the uniform ones by rounding only: the unity sweep runs to
/// the root arm's length when any lane needs it, and the root count follows the widest
/// lane, so the arithmetic is not bit-identical, but every lane must land within a few
/// ulp of its uniform answer.
macro_rules! mixed_packet_test {
    ($modname:ident, $backend:ty, $f64reg:ident) => {
        mod $modname {
            use super::*;
            use thermite::simd::Simd;

            type W = Vector<<$backend as Simd>::$f64reg>;

            fn check(order: PolylogOrder<f64, i64>, zs: &[f64], tol: f64) {
                let mut mixed = W::splat(zs[0]);
                for (i, &z) in zs.iter().enumerate().take(W::LANES) {
                    mixed = mixed.insertv(i, z);
                }
                let got = mixed.polylog_p::<Precision>(order);
                for (i, &z) in zs.iter().enumerate().take(W::LANES) {
                    let want = W::splat(z).polylog_p::<Precision>(order).extract::<0>();
                    let v = got.extractv(i);
                    let scale = want.abs().max(1e-300);
                    assert!(
                        ((v - want) / scale).abs() <= tol,
                        "lane {i} (z = {z}) moved in a mixed packet: {v:e} vs uniform {want:e} at {order:?}"
                    );
                }
            }

            #[test]
            fn polylog_mixed_packet_agrees_with_uniform_ones() {
                // series / unity / far / negative-axis far, positive integer order.
                check(PolylogOrder::Integer(3), &[0.1, 0.9, 1e6, -1e6], 1e-14);
                // Negative integer order: the reflection is per lane.
                check(PolylogOrder::Integer(-3), &[0.1, 0.9, 30.0, -30.0], 1e-13);
                // Real order: the root count follows the widest lane.
                check(PolylogOrder::Real(2.5), &[0.1, -0.9, 1e6, -1e3], 1e-12);
                check(PolylogOrder::Real(-1.5), &[0.5, 0.999, 30.0, -30.0], 1e-12);
            }
        }
    };
}

core::cfg_select! {
    any(target_arch = "x86", target_arch = "x86_64") => {
        mod mixed_x86 {
            use super::*;
            use thermite::backend::x86_v2::X86V2;
            use thermite::backend::x86_v3::X86V3;
            mixed_packet_test!(v3, X86V3, f64x4);
            mixed_packet_test!(v2, X86V2, f64x2);
        }
    }
    target_arch = "aarch64" => {
        mod mixed_neon {
            use super::*;
            use thermite::backend::neon::Neon;
            mixed_packet_test!(neon, Neon, f64x2);
        }
    }
    all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")) => {
        mod mixed_wasm {
            use super::*;
            use thermite::backend::wasm::Wasm;
            mixed_packet_test!(wasm, Wasm, f64x2);
        }
    }
    _ => {}
}

/// The special values, exactly.
#[test]
fn polylog_special_values() {
    // Li_s(0) = 0
    assert_eq!(li64::<Precision>(2.0, 0.0), 0.0);
    assert_eq!(li64::<Precision>(0.5, 0.0), 0.0);
    // Li_s(1) = zeta(s) for s > 1, +inf below
    assert!((li64::<Precision>(2.0, 1.0) - std::f64::consts::PI * std::f64::consts::PI / 6.0).abs() < 1e-15);
    assert_eq!(li64::<Precision>(1.0, 1.0), f64::INFINITY);
    assert_eq!(li64::<Precision>(0.5, 1.0), f64::INFINITY);
    // Li_1(z) = -ln(1 - z)
    assert!((li64::<Precision>(1.0, 0.5) - std::f64::consts::LN_2).abs() < 1e-16);
    // Li_0(z) = z / (1 - z), Li_{-1}(z) = z / (1 - z)^2
    assert!((li64::<Precision>(0.0, 0.25) - 1.0 / 3.0).abs() < 1e-16);
    assert!((li64::<Precision>(-1.0, 0.5) - 2.0).abs() < 1e-15);
    // Li_2(-1) = -pi^2/12
    assert!((li64::<Precision>(2.0, -1.0) + std::f64::consts::PI * std::f64::consts::PI / 12.0).abs() < 1e-15);
    // Real(2.0) simplifies to Integer(2): bit-identical.
    let a = D::splat(0.7)
        .polylog_p::<Precision>(PolylogOrder::Real(2.0))
        .extract::<0>();
    let b = D::splat(0.7)
        .polylog_p::<Precision>(PolylogOrder::Integer(2))
        .extract::<0>();
    assert_eq!(a, b);
    // NaN in, NaN out. Infinities are outside the domain.
    assert!(li64::<Precision>(2.0, f64::NAN).is_nan());
    assert!(li64::<Precision>(2.0, f64::INFINITY).is_nan());
}
