//! Correctness gate for `langevin` / `inv_langevin` / `langevin_d`, against an mpmath
//! oracle (`scripts/langevin_oracle.py`, 120 dps, with the inverse from `findroot`, not
//! from any formula the implementation uses).
//!
//! What is checked, per the handoff: both precisions, every policy tier at its own
//! target, both round trips (the vMF convolution composes them), the crossover at
//! `x = 2` and the seed crossover at `y = 0.85` (dense oracle points on both sides), the
//! pole sweep `y = 1 - 2^-k`, and the composites (`Dual` derivatives, `Compensated`
//! precision).
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(clippy::excessive_precision)]

use thermite::backend::x86_v3::prelude::*;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, UltraPerformance};
use thermite_compensated::Compensated;
use thermite_dual::Dual;
use thermite_special::{RealPrimalMathWithPolicy as _, RealSpecialMath as _, RealSpecialMathWithPolicy as _};

include!("langevin_ref/table.rs");

/// Relative error in units of the target type's epsilon (so "ulp-ish").
fn rel_eps(got: f64, want: f64, eps: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if !got.is_finite() || !want.is_finite() {
        return f64::INFINITY;
    }
    ((got - want) / want).abs() / eps
}

macro_rules! sweep {
    // Evaluate `$f` lane-wise over `$table` (mapping each row to an input via `$input`
    // and expected via `$want`), return the max relative error in epsilons.
    ($V:ty, $E:ty, $table:expr, $input:expr, $want:expr, $f:expr) => {{
        const LANES: usize = <$V as GenericVector>::LANES;
        let mut worst = (0.0f64, 0.0f64);
        let rows: Vec<_> = $table.iter().copied().collect();
        let mut i = 0;
        while i < rows.len() {
            let mut buf = [<$E>::default(); LANES];
            for k in 0..LANES {
                let row = rows[(i + k).min(rows.len() - 1)];
                buf[k] = ($input)(row) as $E;
            }
            let got = ($f)(<$V>::new(buf)).into_array();
            for k in 0..LANES {
                if i + k >= rows.len() {
                    break;
                }
                let row = rows[i + k];
                let want = ($want)(row);
                let e = rel_eps(got[k] as f64, want, <$E>::EPSILON as f64);
                if e > worst.0 {
                    worst = (e, ($input)(row));
                }
            }
            i += LANES;
        }
        worst
    }};
}

// ---------------------------------------------------------------------------------------
// Forward

#[test]
fn langevin_f64_precision() {
    let (e, at) = sweep!(
        f64x4,
        f64,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.langevin_p::<Precision>()
    );
    assert!(e <= 3.0, "langevin f64 worst {e:.2} eps at x={at}");
}

#[test]
fn langevin_f32_precision() {
    // The oracle rounds to f32 on the way in, so the reference is L(fl32(x)) up to
    // the input rounding's own effect (L' <= 1/3, so under one f32 ulp of L).
    let (e, at) = sweep!(
        f32x8,
        f32,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| {
            // reference at the rounded input via first-order correction
            r.1 + r.2 * ((r.0 as f32) as f64 - r.0)
        },
        |v: f32x8| v.langevin_p::<Precision>()
    );
    assert!(e <= 3.0, "langevin f32 worst {e:.2} eps at x={at}");
}

#[test]
fn langevin_lower_policies_track_precision() {
    // Worst/Medium take the short forward tables (f64: 1.9e-12, f32: 5.1e-6), and f32's
    // Worst tier also uses the ~12-bit hardware reciprocal on the large branch. Both are
    // inside those tiers' tolerances (1e5 / 1e4 eps).
    let (e, at) = sweep!(
        f64x4,
        f64,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.langevin_p::<UltraPerformance>()
    );
    let rel = e * f64::EPSILON;
    assert!(
        rel <= 4e-12,
        "langevin f64 UltraPerformance worst rel {rel:.2e} at x={at}"
    );
    let (e, at) = sweep!(
        f32x8,
        f32,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| { r.1 + r.2 * ((r.0 as f32) as f64 - r.0) },
        |v: f32x8| v.langevin_p::<UltraPerformance>()
    );
    let rel = e * f32::EPSILON as f64;
    assert!(
        rel <= 5e-4,
        "langevin f32 UltraPerformance worst rel {rel:.2e} at x={at}"
    );
    let (e, at) = sweep!(
        f32x8,
        f32,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| { r.1 + r.2 * ((r.0 as f32) as f64 - r.0) },
        |v: f32x8| v.langevin_p::<HighPerformance>()
    );
    let rel = e * f32::EPSILON as f64;
    assert!(
        rel <= 1e-5,
        "langevin f32 HighPerformance worst rel {rel:.2e} at x={at}"
    );
}

#[test]
fn langevin_d_matches_oracle() {
    let (e, at) = sweep!(
        f64x4,
        f64,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.2,
        |v: f64x4| v.langevin_d_p::<Precision>().1
    );
    assert!(e <= 12.0, "langevin' f64 worst {e:.2} eps at x={at}");
    let (e, at) = sweep!(
        f32x8,
        f32,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.2,
        |v: f32x8| v.langevin_d_p::<Precision>().1
    );
    assert!(e <= 12.0, "langevin' f32 worst {e:.2} eps at x={at}");
    // and the value half is the value
    let (e, _) = sweep!(
        f64x4,
        f64,
        FORWARD,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.langevin_d_p::<Precision>().0
    );
    assert!(e <= 3.0);
}

#[test]
fn langevin_edges() {
    let x = f64x4::new([0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]);
    let l = x.langevin_p::<Precision>().into_array();
    assert_eq!(l[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(l[1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(l[2], 1.0);
    assert_eq!(l[3], -1.0);
    let (_, d) = x.langevin_d_p::<Precision>();
    let d = d.into_array();
    assert!(rel_eps(d[0], 1.0 / 3.0, f64::EPSILON) <= 1.0); // 1 - 2 fl(1/3)
    assert_eq!(d[2], 0.0);
    assert!(f64x4::splat(f64::NAN).langevin().is_nan().all());

    // odd
    let x = f64x4::new([0.3, 1.7, 2.5, 30.0]);
    assert_eq!(x.langevin().into_array(), (-(-x).langevin()).into_array());
    // saturation: beyond 1/eps the result rounds to exactly 1
    assert_eq!(f64x4::splat(1e17).langevin().extract::<0>(), 1.0);
}

// ---------------------------------------------------------------------------------------
// Inverse

#[test]
fn inv_langevin_f64_precision() {
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_p::<Precision>()
    );
    // Includes the pole sweep to y = 1 - 2^-52: the accurate residual keeps even those
    // lanes within a few eps of the true inverse of the exact input.
    assert!(e <= 4.0, "inv_langevin f64 worst {e:.2} eps at y={at}");
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_p::<Reference>()
    );
    assert!(e <= 4.0, "inv_langevin f64 Reference worst {e:.2} eps at y={at}");
}

#[test]
fn inv_langevin_f32_precision() {
    // f32 input rounding moves the answer by L^-1'(y) dy = dy / L'(x), so fold that
    // into the reference. Skip rows whose y is not representable to that first order
    // (the far pole sweep, where 1 - y is below f32 resolution).
    let rows: Vec<(f64, f64, f64)> = INVERSE.iter().copied().filter(|r| 1.0 - r.0 > 1e-6).collect();
    let (e, at) = sweep!(
        f32x8,
        f32,
        rows,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| {
            let dy = (r.0 as f32) as f64 - r.0;
            let x = r.1;
            let dl = r.2;
            x + dy / dl
        },
        |v: f32x8| v.inv_langevin_p::<Precision>()
    );
    assert!(e <= 6.0, "inv_langevin f32 worst {e:.2} eps at y={at}");
}

#[test]
fn inv_langevin_policy_ladder() {
    // Worst ships the seed (~2.2e-5 at the 0.85 tail crossover, 1.1e-6 below it). Every
    // other tier takes one Halley step, which cubes that to under u in both types.
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_p::<UltraPerformance>()
    );
    let rel = e * f64::EPSILON;
    assert!(rel <= 3e-5, "inv_langevin f64 Worst rel {rel:.2e} at y={at}");
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_p::<HighPerformance>()
    );
    assert!(e <= 4.0, "inv_langevin f64 Medium worst {e:.2} eps at y={at}");
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_p::<Performance>()
    );
    assert!(e <= 4.0, "inv_langevin f64 Average worst {e:.2} eps at y={at}");

    let rows: Vec<(f64, f64, f64)> = INVERSE.iter().copied().filter(|r| 1.0 - r.0 > 1e-6).collect();
    let (e, at) = sweep!(
        f32x8,
        f32,
        rows,
        |r: (f64, f64, f64)| r.0,
        |r: (f64, f64, f64)| r.1,
        |v: f32x8| v.inv_langevin_p::<UltraPerformance>()
    );
    let rel = e * f32::EPSILON as f64;
    assert!(rel <= 5e-4, "inv_langevin f32 Worst rel {rel:.2e} at y={at}");
}

#[test]
fn inv_langevin_edges() {
    let y = f64x4::new([0.0, -0.0, 1.0, -1.0]);
    let x = y.inv_langevin_p::<Precision>().into_array();
    assert_eq!(x[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(x[1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(x[2], f64::INFINITY);
    assert_eq!(x[3], f64::NEG_INFINITY);

    let y = f64x4::new([1.5, -2.0, f64::NAN, f64::INFINITY]);
    assert!(y.inv_langevin_p::<Precision>().is_nan().all()); // check_overflow
    assert!(
        f64x4::splat(f64::NAN)
            .inv_langevin_p::<UltraPerformance>()
            .is_nan()
            .all()
    );

    // odd
    let y = f64x4::new([0.1, 0.5, 0.9, 0.999]);
    assert_eq!(y.inv_langevin().into_array(), (-(-y).inv_langevin()).into_array());
}

#[test]
fn round_trips() {
    // L(L^-1(y)) = y: the direction the vMF convolution kappa' = L^-1(L(k1) L(k2))
    // ends on. Bounded by the forward's own error, since L is contracting.
    let n = 2000;
    let mut worst = 0.0f64;
    for i in 0..n {
        let y = (i as f64 + 0.5) / n as f64;
        let back = f64x4::splat(y)
            .inv_langevin_p::<Precision>()
            .langevin_p::<Precision>()
            .extract::<0>();
        worst = worst.max(rel_eps(back, y, f64::EPSILON));
    }
    assert!(worst <= 3.0, "L(L^-1(y)) worst {worst:.2} eps");

    // L^-1(L(x)) = x: amplified by the condition number x (= 1/(1-y)). The error of
    // the rounded y alone accounts for x/2 eps, so measure against that.
    let mut worst = 0.0f64;
    for i in 1..=200 {
        let x = i as f64 * 0.1;
        let back = f64x4::splat(x)
            .langevin_p::<Precision>()
            .inv_langevin_p::<Precision>()
            .extract::<0>();
        let e = rel_eps(back, x, f64::EPSILON) / x.max(1.0);
        worst = worst.max(e);
    }
    assert!(worst <= 4.0, "L^-1(L(x)) worst {worst:.2} eps per unit condition");
}

// ---------------------------------------------------------------------------------------
// Composites

#[test]
fn dual_derivatives() {
    type D = Dual<f64x4, 1>;
    for r in FORWARD.iter().copied() {
        let x = D::variable(f64x4::splat(r.0), 0);
        let l = x.langevin_p::<Precision>();
        assert!(
            rel_eps(l.re.extract::<0>(), r.1, f64::EPSILON) <= 3.0,
            "dual L at {}",
            r.0
        );
        assert!(
            rel_eps(l.dual[0].extract::<0>(), r.2, f64::EPSILON) <= 16.0,
            "dual L' at {}: got {} want {}",
            r.0,
            l.dual[0].extract::<0>(),
            r.2
        );
    }
    for r in INVERSE.iter().copied().filter(|r| r.0 < 0.9999) {
        let y = D::variable(f64x4::splat(r.0), 0);
        let x = y.inv_langevin_p::<Precision>();
        assert!(
            rel_eps(x.re.extract::<0>(), r.1, f64::EPSILON) <= 4.0,
            "dual L^-1 at {}",
            r.0
        );
        let want = 1.0 / r.2;
        assert!(
            rel_eps(x.dual[0].extract::<0>(), want, f64::EPSILON) <= 16.0,
            "dual (L^-1)' at {}: got {} want {}",
            r.0,
            x.dual[0].extract::<0>(),
            want
        );
    }
}

#[test]
fn compensated_double_double() {
    type C = Compensated<f64x4>;
    // The f64 oracle only reaches 1e-16, so check the double-double result agrees with
    // it to f64 and, more tellingly, that its own round trip is tight at ~1e-30.
    for r in FORWARD.iter().copied() {
        let l = C::new(f64x4::splat(r.0)).langevin_p::<Precision>();
        assert!(
            rel_eps(l.value().extract::<0>(), r.1, f64::EPSILON) <= 1.0,
            "dd L at {}",
            r.0
        );
    }
    let mut worst = 0.0f64;
    for i in 0..500 {
        let y = C::new(f64x4::splat((i as f64 + 0.5) / 500.0));
        let back = y.inv_langevin_p::<Precision>().langevin_p::<Precision>();
        let d = (back - y).value().extract::<0>().abs() / y.value().extract::<0>();
        worst = worst.max(d);
    }
    assert!(worst <= 1e-30, "dd round trip worst {worst:.2e}");
}

// ---------------------------------------------------------------------------------------
// Complement pair (handoff section 12)

#[test]
fn langevin_1m_matches_oracle_into_the_tail() {
    // Where 1 - L is O(1) it agrees with 1 - langevin to a few ulp. Where L -> 1 only
    // the oracle is a valid reference, and langevin_1m must stay at a few ulp there
    // while 1 - langevin has already lost everything (x = 1e18: 1 - L rounds to 0).
    let (e, at) = sweep!(
        f64x4,
        f64,
        FORWARD_1M,
        |r: (f64, f64)| r.0,
        |r: (f64, f64)| r.1,
        |v: f64x4| v.langevin_1m_p::<Precision>()
    );
    assert!(e <= 3.0, "langevin_1m f64 worst {e:.2} eps at x={at}");
    let (e, at) = sweep!(
        f32x8,
        f32,
        FORWARD_1M,
        |r: (f64, f64)| r.0,
        |r: (f64, f64)| r.1
            * (r.0 / ((r.0 as f32) as f64))
                .max(1.0 / (r.0 / ((r.0 as f32) as f64)))
                .min(1.0 + 1e-3),
        |v: f32x8| v.langevin_1m_p::<Precision>()
    );
    // (the input rounding matters at x > 2^24 where 1 - L ~ 1/x tracks it, but the row
    // set there is round-in-f32 anyway, so the reference tweak above is a no-op guard)
    assert!(e <= 3.0, "langevin_1m f32 worst {e:.2} eps at x={at}");

    // and it really is better than the subtraction where it counts
    let x = f64x4::splat(1e12);
    let direct = 1.0 - x.langevin_p::<Precision>().extract::<0>();
    let comp = x.langevin_1m_p::<Precision>().extract::<0>();
    assert!(rel_eps(comp, 1e-12, f64::EPSILON) <= 3.0);
    assert!(
        rel_eps(direct, 1e-12, f64::EPSILON) > 1000.0,
        "1 - langevin should be hopeless here"
    );

    // agrees with 1 - langevin on the O(1) range, both signs
    for x in [-3.0, -1.5, -0.2, 0.0, 0.2, 1.5, 3.0] {
        let v = f64x4::splat(x);
        let a = v.langevin_1m().extract::<0>();
        let b = 1.0 - v.langevin().extract::<0>();
        assert!(rel_eps(a, b, f64::EPSILON) <= 4.0, "at {x}: {a} vs {b}");
    }
    let l = f64x4::new([f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0])
        .langevin_1m()
        .into_array();
    assert_eq!(&l[..], &[0.0, 2.0, 1.0, 1.0]);
}

#[test]
fn inv_langevin_1m_matches_oracle_and_round_trips() {
    // Pole sweep to t = 2^-60 in f64: the answer is ~1/t, and must be a few ulp of it.
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE_1M,
        |r: (f64, f64)| r.0,
        |r: (f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_1m_p::<Precision>()
    );
    assert!(e <= 4.0, "inv_langevin_1m f64 worst {e:.2} eps at t={at}");
    let (e, at) = sweep!(
        f32x8,
        f32,
        INVERSE_1M,
        |r: (f64, f64)| r.0,
        |r: (f64, f64)| r.1,
        |v: f32x8| v.inv_langevin_1m_p::<Precision>()
    );
    assert!(e <= 6.0, "inv_langevin_1m f32 worst {e:.2} eps at t={at}");

    // Round trip deep into the tail, exactly where the non-complement pair dies.
    for k in 0..=15 {
        let x = 10f64.powi(k);
        let back = f64x4::splat(x)
            .langevin_1m_p::<Precision>()
            .inv_langevin_1m_p::<Precision>()
            .extract::<0>();
        assert!(
            rel_eps(back, x, f64::EPSILON) <= 6.0,
            "f64 round trip at x=1e{k}: {back}"
        );
    }
    for k in 0..=8 {
        let x = 10f32.powi(k);
        let back = f32x8::splat(x)
            .langevin_1m_p::<Precision>()
            .inv_langevin_1m_p::<Precision>()
            .extract::<0>();
        assert!(
            rel_eps(back as f64, x as f64, f32::EPSILON as f64) <= 8.0,
            "f32 round trip at x=1e{k}: {back}"
        );
    }

    // Seam at the seed crossover (t = 0.15) and the sign fall-through (t > 1).
    let (e, at) = sweep!(
        f64x4,
        f64,
        INVERSE_1M.iter().copied().filter(|r| r.0 > 0.1).collect::<Vec<_>>(),
        |r: (f64, f64)| r.0,
        |r: (f64, f64)| r.1,
        |v: f64x4| v.inv_langevin_1m_p::<Precision>()
    );
    assert!(e <= 4.0, "inv_langevin_1m f64 seam worst {e:.2} eps at t={at}");
    let x = f64x4::new([0.0, 1.0, 2.0, -0.5])
        .inv_langevin_1m_p::<Precision>()
        .into_array();
    assert_eq!(x[0], f64::INFINITY);
    assert_eq!(x[1], 0.0);
    assert_eq!(x[2], f64::NEG_INFINITY);
    assert!(x[3].is_nan());
}

#[test]
fn vmf_convolution_chain_f32() {
    // kappa' = L^-1(L(k1) L(k2)) for two equal sharp lobes, formed as
    // inv_langevin_1m(a + b - a b) with a = b = langevin_1m(k), the consumer's chain.
    // Reference is the same chain in f64 (which is itself good to ~1e-16 here).
    for k in [1e2f32, 1e4, 1e6, 1e8] {
        let a32 = f32x8::splat(k).langevin_1m_p::<Precision>();
        let t32 = a32 + a32 - a32 * a32;
        let got = t32.inv_langevin_1m_p::<Precision>().extract::<0>() as f64;

        let a64 = f64x4::splat(k as f64).langevin_1m_p::<Precision>();
        let t64 = a64 + a64 - a64 * a64;
        let want = t64.inv_langevin_1m_p::<Precision>().extract::<0>();

        let e = rel_eps(got, want, f32::EPSILON as f64);
        assert!(e <= 8.0, "vMF chain f32 at k={k:e}: got {got}, want {want}, {e:.1} eps");
    }
}

#[test]
fn complement_pair_on_composites() {
    // Dual: d/dx (1 - L) = -L', d/dt L^-1(1 - t) = -1/L'.
    type D = Dual<f64x4, 1>;
    for r in FORWARD_1M.iter().copied() {
        let v = D::variable(f64x4::splat(r.0), 0).langevin_1m_p::<Precision>();
        assert!(
            rel_eps(v.re.extract::<0>(), r.1, f64::EPSILON) <= 4.0,
            "dual 1-L at {}",
            r.0
        );
    }
    for r in FORWARD.iter().copied() {
        let v = D::variable(f64x4::splat(r.0), 0).langevin_1m_p::<Precision>();
        assert!(
            rel_eps(v.dual[0].extract::<0>(), -r.2, f64::EPSILON) <= 16.0,
            "dual (1-L)' at {}",
            r.0
        );
    }
    for r in INVERSE.iter().copied().filter(|r| r.0 > 0.5 && r.0 < 0.9999) {
        let t = 1.0 - r.0; // exact
        let v = D::variable(f64x4::splat(t), 0).inv_langevin_1m_p::<Precision>();
        assert!(
            rel_eps(v.re.extract::<0>(), r.1, f64::EPSILON) <= 4.0,
            "dual L^-1(1-t) at t={t}"
        );
        assert!(
            rel_eps(v.dual[0].extract::<0>(), -1.0 / r.2, f64::EPSILON) <= 16.0,
            "dual (L^-1(1-t))' at t={t}"
        );
    }

    // Compensated: the complement keeps double-double precision deep in the tail, and
    // the pair round-trips there.
    type C = Compensated<f64x4>;
    for r in FORWARD_1M.iter().copied() {
        let v = C::new(f64x4::splat(r.0)).langevin_1m_p::<Precision>();
        assert!(
            rel_eps(v.value().extract::<0>(), r.1, f64::EPSILON) <= 1.0,
            "dd 1-L at {}",
            r.0
        );
    }
    for k in [1.0f64, 1e3, 1e8, 1e15, 1e20] {
        let x = C::new(f64x4::splat(k));
        let back = x.langevin_1m_p::<Precision>().inv_langevin_1m_p::<Precision>();
        let d = ((back - x).value() / x.value()).extract::<0>().abs();
        assert!(d <= 1e-30, "dd complement round trip at {k:e}: {d:e}");
    }
}
