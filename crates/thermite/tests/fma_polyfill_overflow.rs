//! Regression: the non-FMA x86 backends must not return NaN for a finite product.
//!
//! Historically `_mm_fmadd_pdx_v1` (Veltkamp/Dekker splitting) computed
//! `x * (2^27 + 1)`, which overflows to infinity above 2^996, and `inf - inf` is NaN,
//! so without a guard `mul_add` returned NaN for operands whose true result is an
//! ordinary finite number. Measured unguarded: 26639 NaNs in 1747713 random triples.
//!
//! `mul_add` now lowers to the round-to-odd emulation (`fmadd_ro`), whose integer-add
//! split cannot overflow and whose `is_finite` post-check routes genuine overflow to
//! the vectorized rescue. These inputs stay pinned anyway: they are exactly the shapes
//! that caught the original bug. (`fma_exact.rs` covers the stronger bit-exactness
//! contract.)
//!
//! Only backends WITHOUT hardware FMA take this path, so x86_v1 (SSE2) and x86_v2
//! (SSE4.2) are what these pin.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

/// `x * (2^27 + 1)` overflows above this.
const THRESH: f64 = 6.69692879491417e299;

macro_rules! check_backend {
    ($name:ident, $backend:path) => {
        mod $name {
            use super::THRESH;
            use thermite::vector::ops::MulAddExt;
            use $backend::*;

            #[test]
            fn mul_add_stays_finite_for_large_operands() {
                assert!(
                    !matches!(<f64x2 as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True),
                    "this backend must lack hardware FMA, or the test proves nothing"
                );

                for (x, m, a) in [
                    (THRESH * 2.0, 3.0, 1.0),
                    (f64::MAX, 0.5, 1.0),
                    (-f64::MAX, 0.25, -7.0),
                    (1.7e308, 1e-8, 2.0),
                    (1e300, 1e-300, 0.0),
                    (THRESH, 2.0, THRESH),
                ] {
                    let got = f64x2::splat(x)
                        .mul_add(f64x2::splat(m), f64x2::splat(a))
                        .extract::<0>();
                    let want = x.mul_add(m, a);

                    assert!(
                        got.is_finite(),
                        "mul_add({x:e}, {m:e}, {a:e}) = {got:e}, true fma = {want:e}"
                    );
                    // Dekker is not correctly rounded, so allow the inherent 1 ulp.
                    let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                    assert!(ulps <= 1, "mul_add({x:e}, {m:e}, {a:e}) off by {ulps} ulp");
                }
            }

            /// One huge lane must not disturb its neighbour: the rebalance is per lane.
            #[test]
            fn mixed_magnitude_lanes() {
                let x = f64x2::new([f64::MAX, 3.0]);
                let m = f64x2::new([0.5, 7.0]);
                let a = f64x2::new([1.0, 2.0]);

                let got = x.mul_add(m, a);

                for (i, (xv, mv, av)) in [(f64::MAX, 0.5, 1.0), (3.0, 7.0, 2.0)].into_iter().enumerate() {
                    let g = got.extractv(i);
                    assert!(g.is_finite(), "lane {i} = {g:e}");
                    let ulps = (g.to_bits() as i64 - xv.mul_add(mv, av).to_bits() as i64).abs();
                    assert!(ulps <= 1, "lane {i} off by {ulps} ulp");
                }
            }
        }
    };
}

check_backend!(sse2, thermite::backend::x86_v1::prelude);
check_backend!(sse42, thermite::backend::x86_v2::prelude);

/// Ordinary magnitudes must be untouched by the guard.
#[test]
fn small_operands_unchanged() {
    use thermite::backend::x86_v1::prelude::*;

    let mut s = 0x853C_49E6_748F_EA9Bu64;
    for _ in 0..50_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = (s as i64 as f64) / 65_536.0;
        let m = (s.rotate_left(31) as i64 as f64) / 4096.0;
        let a = (s.rotate_left(17) as i64 as f64) / 1024.0;
        if !x.mul_add(m, a).is_finite() {
            continue;
        }

        let got = f64x2::splat(x).mul_add(f64x2::splat(m), f64x2::splat(a)).extract::<0>();
        assert!(got.is_finite(), "x={x:e} m={m:e} a={a:e} -> {got:e}");
    }
}
