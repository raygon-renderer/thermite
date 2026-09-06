//! Differential + oracle tests for the lane-alternating `addsub` / `fmaddsub` /
//! `fmsubadd` float register ops (even lanes subtract, odd lanes add).
//!
//! - `addsub` is exact (a single IEEE add per lane), so it is differenced
//!   bit-for-bit against the `Scalar` backend, exactly like `add`/`sub`.
//! - `fmaddsub` / `fmsubadd` are a fused multiply then alternating add/sub. A
//!   native FMA backend (v3/v4/NEON) rounds once while the emulated backends round
//!   twice, so they legitimately differ. They are therefore checked against a
//!   correctly-rounded `mul_add` oracle with a bound relative to the *operand*
//!   magnitude `|a*b| + |c|` (not the possibly-cancelled result), which both
//!   fused and unfused evaluations satisfy.
//! - The masked `_c`/`_m`/`_z` siblings of `addsub` are differenced against the
//!   scalar backend (blendv composition is exact).
//!
//! See `harness/mod.rs` for methodology and the register `addsub` docs for the
//! interleaved-complex-multiply motivation. Every backend runs every slot. The
//! wide ones are `ArrayRegister`-emulated where the backend has no such register.
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
/// an operand-magnitude-relative bound so the fused and unfused paths both pass
/// without cancellation noise. `$even_subtracts` is `true` for `fmaddsub` (even
/// lane = `a*b - c`) and `false` for `fmsubadd`.
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
/// are exact. `_c` tolerates the documented signed-zero divergence.
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

            let got_c = harness::read::<$ut>(&<$ut>::addsub_c(m_ut, ax, ay));
            let want_c = harness::read::<$rf>(&<$rf>::addsub_c(m_rf, rx, ry));
            harness::assert_lanes_eq(
                &format!("{} [addsub_c]", $label),
                &[x, y],
                &got_c,
                &want_c,
                Tol::ExactOrZeroSign,
            );

            let got_m = harness::read::<$ut>(&<$ut>::addsub_m(src, m_ut, ax, ay));
            let want_m = harness::read::<$rf>(&<$rf>::addsub_m(rsrc, m_rf, rx, ry));
            harness::assert_lanes_eq(&format!("{} [addsub_m]", $label), &[x, y], &got_m, &want_m, Tol::Exact);

            let got_z = harness::read::<$ut>(&<$ut>::addsub_z(m_ut, ax, ay));
            let want_z = harness::read::<$rf>(&<$rf>::addsub_z(m_rf, rx, ry));
            harness::assert_lanes_eq(&format!("{} [addsub_z]", $label), &[x, y], &got_z, &want_z, Tol::Exact);
        }
    }};
}

macro_rules! addsub {
    ($reg:ident) => {{
        type UT = <S as Simd>::$reg;
        type RF = <Scalar as Simd>::$reg;
        let label = harness::label::<S>(stringify!($reg));
        let label = label.as_str();

        diff_binary!(label, UT, RF, addsub, Tol::Exact);

        oracle_fused!(label, UT, fmaddsub, true);
        oracle_fused!(label, UT, fmsubadd, false);

        diff_addsub_masked!(label, UT, RF);
    }};
}

for_each_backend_concrete! {
    fn f32x4() { addsub!(f32x4) }
    fn f32x8() { addsub!(f32x8) }
    fn f32x16() { addsub!(f32x16) }
    fn f64x2() { addsub!(f64x2) }
    fn f64x4() { addsub!(f64x4) }
    fn f64x8() { addsub!(f64x8) }
}
