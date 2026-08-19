//! Differential + oracle tests for the lane-alternating `addsub` / `fmaddsub` /
//! `fmsubadd` float register ops (even lanes subtract, odd lanes add).
//!
//! - `addsub` is exact (a single IEEE add per lane), so it is differenced
//!   bit-for-bit against the `Scalar` backend, exactly like `add`/`sub`.
//! - `fmaddsub` / `fmsubadd` are a fused multiply then alternating add/sub. A
//!   native FMA backend (v3) rounds once while the emulated backends round twice, so
//!   they legitimately differ. They are therefore checked against a
//!   correctly-rounded `mul_add` oracle with a bound relative to the *operand*
//!   magnitude `|a*b| + |c|` (not the possibly-cancelled result), which both
//!   fused and unfused evaluations satisfy.
//! - The masked `_c`/`_m`/`_z` siblings of `addsub` are differenced against the
//!   scalar backend (blendv composition is exact).
//!
//! See `harness/mod.rs` for methodology and the register `addsub` docs for the
//! interleaved-complex-multiply motivation.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::{Diff, Tol};

use thermite::register::{CoreRegister, FloatRegister as _, Register};
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

/// Fused alternating multiply-add vs. a correctly-rounded `mul_add` oracle, with
/// an operand-magnitude-relative bound so the fused (v3) and unfused (emulated)
/// paths both pass without cancellation noise. `$even_subtracts` is `true` for
/// `fmaddsub` (even lane = `a*b - c`) and `false` for `fmsubadd`.
macro_rules! oracle_fused {
    ($label:expr, $ut:ty, $method:ident, $even_subtracts:expr) => {{
        type E = <$ut as Register>::Element;
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        let a_in = harness::corpus::<E>(lanes, &mut rng);
        let b_in = harness::corpus::<E>(lanes, &mut rng);
        let c_in = harness::corpus::<E>(lanes, &mut rng);
        // Floor the bound so subnormal/zero results don't demand impossible
        // absolute accuracy, and still far below any parity/sign error's magnitude.
        let floor = (<E>::MIN_POSITIVE as f64) * 8.0;
        for ((a, b), c) in a_in.iter().zip(b_in.iter()).zip(c_in.iter()) {
            // Fused-vs-unfused divergence on inf/NaN is not meaningful here.
            if a.iter().chain(b).chain(c).any(|v| !Diff::finite(*v)) {
                continue;
            }
            let got = harness::read::<$ut>(&<$ut>::$method(
                harness::make_array::<$ut>(a),
                harness::make_array::<$ut>(b),
                harness::make_array::<$ut>(c),
            ));
            for i in 0..lanes {
                let (ai, bi, ci) = (a[i], b[i], c[i]);
                // even lanes take `$even_subtracts`; odd lanes take the opposite.
                let subtract = if i % 2 == 0 { $even_subtracts } else { !$even_subtracts };
                let signed_c = if subtract { -ci } else { ci };
                let want = E::mul_add(ai, bi, signed_c); // fused, correctly rounded
                if !want.is_finite() || !got[i].is_finite() {
                    continue; // overflow behaviour may differ between fused/unfused
                }
                let mag = (ai as f64 * bi as f64).abs() + (ci as f64).abs();
                let bound = (4.0 * (<E>::EPSILON as f64) * mag).max(floor);
                let diff = (got[i] as f64 - want as f64).abs();
                assert!(
                    diff <= bound,
                    "{} [{}]: lane {} mismatch\n  a={:?} b={:?} c={:?}\n  got={:?} want={:?} diff={:e} bound={:e}",
                    $label,
                    stringify!($method),
                    i,
                    ai,
                    bi,
                    ci,
                    got[i],
                    want,
                    diff,
                    bound,
                );
            }
        }
    }};
}

/// Masked binary diff (`addsub_c`/`_m`/`_z`) vs. the scalar backend. `_m`/`_z`
/// are exact (blendv / bitand over the exact `addsub`), and `_c` is exact up to the
/// sign of a zero result. See the comment on the assertion below.
macro_rules! diff_addsub_masked {
    ($label:expr, $ut:ty, $rf:ty) => {{
        type E = <$ut as Register>::Element;
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        let xs = harness::corpus::<E>(lanes, &mut rng);
        let ys = harness::corpus::<E>(lanes, &mut rng);
        let masks = harness::mask_patterns(lanes, xs.len(), &mut rng);
        for ((x, y), bools) in xs.iter().zip(ys.iter()).zip(masks.iter()) {
            let (ax, ay) = (harness::make_array::<$ut>(x), harness::make_array::<$ut>(y));
            let (rx, ry) = (harness::make_array::<$rf>(x), harness::make_array::<$rf>(y));
            let src = harness::make_array::<$ut>(y); // arbitrary merge source
            let rsrc = harness::make_array::<$rf>(y);
            let m_ut = harness::build_mask::<$ut>(bools);
            let m_rf = harness::build_mask::<$rf>(bools);

            // `_c` is `addsub(a, b & mask)` on an equal-size-mask backend, so a
            // masked-off `-0.0` lane returns `+0.0` where the scalar oracle's
            // blendv keeps the sign. See `Tol::ExactOrZeroSign`.
            let got_c = harness::read::<$ut>(&<$ut>::addsub_c(m_ut, ax, ay));
            let want_c = harness::read::<$rf>(&<$rf>::addsub_c(m_rf, rx, ry));
            harness::assert_lanes_eq(
                concat!($label, " [addsub_c]"),
                &[x, y],
                &got_c,
                &want_c,
                Tol::ExactOrZeroSign,
            );

            let got_m = harness::read::<$ut>(&<$ut>::addsub_m(src, m_ut, ax, ay));
            let want_m = harness::read::<$rf>(&<$rf>::addsub_m(rsrc, m_rf, rx, ry));
            harness::assert_lanes_eq(concat!($label, " [addsub_m]"), &[x, y], &got_m, &want_m, Tol::Exact);

            let got_z = harness::read::<$ut>(&<$ut>::addsub_z(m_ut, ax, ay));
            let want_z = harness::read::<$rf>(&<$rf>::addsub_z(m_rf, rx, ry));
            harness::assert_lanes_eq(concat!($label, " [addsub_z]"), &[x, y], &got_z, &want_z, Tol::Exact);
        }
    }};
}

macro_rules! addsub_tests {
    ($modname:ident, $ut_backend:ty, $reg:ident, $label:expr) => {
        #[test]
        fn $modname() {
            type UT = <$ut_backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;

            // addsub is a single exact add per lane -> bit-exact vs scalar.
            diff_binary!($label, UT, RF, addsub, Tol::Exact);

            // fused variants: operand-magnitude-relative oracle (fused/unfused safe).
            oracle_fused!($label, UT, fmaddsub, true);
            oracle_fused!($label, UT, fmsubadd, false);

            // masked addsub siblings.
            diff_addsub_masked!($label, UT, RF);
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    // X86V3: native 256-bit and 128-bit addsub / fmaddsub / fmsubadd.
    mod v3 {
        use super::*;
        addsub_tests!(f32x4, X86V3, f32x4, "x86_v3 f32x4");
        addsub_tests!(f32x8, X86V3, f32x8, "x86_v3 f32x8");
        addsub_tests!(f32x16, X86V3, f32x16, "x86_v3 f32x16"); // emulated (ArrayRegister)
        addsub_tests!(f64x2, X86V3, f64x2, "x86_v3 f64x2");
        addsub_tests!(f64x4, X86V3, f64x4, "x86_v3 f64x4");
        addsub_tests!(f64x8, X86V3, f64x8, "x86_v3 f64x8"); // emulated (ArrayRegister)
    }

    // X86V2: native addsub (SSE3), fused variants = mul + native addsub.
    mod v2 {
        use super::*;
        addsub_tests!(f32x4, X86V2, f32x4, "x86_v2 f32x4");
        addsub_tests!(f32x8, X86V2, f32x8, "x86_v2 f32x8"); // emulated (ArrayRegister)
        addsub_tests!(f64x2, X86V2, f64x2, "x86_v2 f64x2");
        addsub_tests!(f64x4, X86V2, f64x4, "x86_v2 f64x4"); // emulated (ArrayRegister)
    }

    // X86V1: no SSE3 addsub, no FMA -> the materialized-constant xor emulation.
    mod v1 {
        use super::*;
        addsub_tests!(f32x4, X86V1, f32x4, "x86_v1 f32x4");
        addsub_tests!(f32x8, X86V1, f32x8, "x86_v1 f32x8"); // emulated (ArrayRegister)
        addsub_tests!(f64x2, X86V1, f64x2, "x86_v1 f64x2");
        addsub_tests!(f64x4, X86V1, f64x4, "x86_v1 f64x4"); // emulated (ArrayRegister)
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    addsub_tests!(f32x4, Wasm, f32x4, "wasm f32x4");
    addsub_tests!(f64x2, Wasm, f64x2, "wasm f64x2");
}
