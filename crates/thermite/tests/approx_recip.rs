//! Property tests for ops that are *approximate* or *order-sensitive*, so a
//! bit-exact differential against the scalar backend is the wrong tool:
//!
//!   - `rcp(x)`   ≈ 1/x        (hardware reciprocal estimate)
//!   - `rsqrt(x)` ≈ 1/sqrt(x)  (hardware reciprocal-sqrt estimate)
//!   - `sum_elements`          (non-associative; tree vs. left-fold)
//!
//! These had no coverage at all. Inputs are restricted to the well-behaved
//! normal range (no denormals/inf/NaN) where the accuracy contract holds.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use thermite::Vector;
use thermite::prelude::*;

/// Relative-error bound for the refined hardware estimates. Thermite refines
/// the raw ~12-bit estimate, so this is comfortably loose but still catches a
/// broken estimate or a missing refinement step.
const RECIP_REL: f64 = 1.0e-3;

macro_rules! recip_props {
    ($modname:ident, $backend:ty, $f32reg:ident, $f64reg:ident) => {
        mod $modname {
            use super::*;

            #[test]
            fn rcp_rsqrt_f32() {
                type R = <$backend as Simd>::$f32reg;
                let mut rng = harness::rng();
                let lanes = <Vector<R> as GenericVector>::LANES;
                for raw in harness::corpus::<f32>(lanes, &mut rng) {
                    // Tame, strictly-positive, normal magnitudes.
                    let x: Vec<f32> = raw
                        .iter()
                        .map(|v| {
                            if v.is_finite() && *v != 0.0 {
                                v.abs().clamp(1e-12, 1e12)
                            } else {
                                1.0
                            }
                        })
                        .collect();
                    let v = Vector::<R>(harness::make_array::<R>(&x));
                    let rcp = v.rcp().to_array();
                    let rsqrt = v.rsqrt().to_array();
                    for (i, &xi) in x.iter().enumerate() {
                        let want_rcp = 1.0 / xi as f64;
                        assert!(
                            (rcp[i] as f64 - want_rcp).abs() <= RECIP_REL * want_rcp.abs(),
                            "rcp f32 @ {xi}: got {}, want ~{want_rcp}",
                            rcp[i]
                        );
                        let want_rs = 1.0 / (xi as f64).sqrt();
                        assert!(
                            (rsqrt[i] as f64 - want_rs).abs() <= RECIP_REL * want_rs.abs(),
                            "rsqrt f32 @ {xi}: got {}, want ~{want_rs}",
                            rsqrt[i]
                        );
                    }
                }
            }

            #[test]
            fn rcp_rsqrt_f64() {
                type R = <$backend as Simd>::$f64reg;
                let mut rng = harness::rng();
                let lanes = <Vector<R> as GenericVector>::LANES;
                for raw in harness::corpus::<f64>(lanes, &mut rng) {
                    let x: Vec<f64> = raw
                        .iter()
                        .map(|v| {
                            if v.is_finite() && *v != 0.0 {
                                v.abs().clamp(1e-100, 1e100)
                            } else {
                                1.0
                            }
                        })
                        .collect();
                    let v = Vector::<R>(harness::make_array::<R>(&x));
                    let rcp = v.rcp().to_array();
                    let rsqrt = v.rsqrt().to_array();
                    for (i, &xi) in x.iter().enumerate() {
                        let want_rcp = 1.0 / xi;
                        assert!(
                            (rcp[i] - want_rcp).abs() <= RECIP_REL * want_rcp.abs(),
                            "rcp f64 @ {xi}: got {}, want ~{want_rcp}",
                            rcp[i]
                        );
                        let want_rs = 1.0 / xi.sqrt();
                        assert!(
                            (rsqrt[i] - want_rs).abs() <= RECIP_REL * want_rs.abs(),
                            "rsqrt f64 @ {xi}: got {}, want ~{want_rs}",
                            rsqrt[i]
                        );
                    }
                }
            }

            #[test]
            fn sum_elements_f32() {
                type R = <$backend as Simd>::$f32reg;
                let mut rng = harness::rng();
                let lanes = <Vector<R> as GenericVector>::LANES;
                for raw in harness::corpus::<f32>(lanes, &mut rng) {
                    // Bounded magnitudes: no cancellation catastrophes, so any
                    // summation order agrees to a small relative error.
                    let x: Vec<f32> = raw
                        .iter()
                        .map(|v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 0.0 })
                        .collect();
                    let got = Vector::<R>(harness::make_array::<R>(&x)).sum_elements() as f64;
                    let want: f64 = x.iter().map(|&v| v as f64).sum();
                    // Float summation error is bounded by the condition number
                    // Σ|xᵢ| (catastrophic cancellation makes the result tiny
                    // relative to the partial sums), not by |want|.
                    let cond: f64 = x.iter().map(|&v| (v as f64).abs()).sum();
                    assert!(
                        (got - want).abs() <= 1e-5 * cond.max(1.0),
                        "sum_elements f32: got {got}, want ~{want} (cond {cond}) for {x:?}"
                    );
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    recip_props!(v3, X86V3, f32x8, f64x4);
    recip_props!(v2, X86V2, f32x4, f64x2);
    recip_props!(v1, X86V1, f32x4, f64x2);
}

// On wasm rcp/rsqrt are exact (HAS_APPROX_RCP/RSQRT = false), which still
// satisfies the loose accuracy bound. Native widths are f32x4/f64x2.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    recip_props!(wasm, Wasm, f32x4, f64x2);
}
