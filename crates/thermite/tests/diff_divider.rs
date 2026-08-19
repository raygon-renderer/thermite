//! Differential coverage for the scalar integer dividers (`divider/mod.rs`),
//! the libdivide-derived fast-division machinery (`Divider`/`BranchfreeDivider`).
//!
//! Oracle is Rust's own `/`. For every integer type we sweep a mix of edge and
//! random `(divisor, x)` pairs through both the branching `Divider` and the
//! `BranchfreeDivider`, hitting the power-of-two (`multiplier == 0`) path, the
//! general `mullhi` + `ADD_MARKER` path, and (for signed) the `NEG_DIVISOR` path.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use rand::RngExt;

use thermite::divider::{Denominator, UnsupportedDivisor};
use thermite::prelude::*;
use thermite::simd::Simd;
use thermite::vector::ops::DivMasked;
use thermite::{BranchfreeDivider, Divider, Vector};

macro_rules! div_check {
    ($t:ty, $signed:literal) => {{
        let mut rng = harness::rng();

        // Divisors: small values (powers of two AND non-powers), extremes, randoms.
        let small: &[$t] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127];
        let mut ds: Vec<$t> = small.iter().copied().collect();
        ds.push(<$t>::MAX);
        ds.push(<$t>::MAX / 2);
        ds.push(<$t>::MAX / 3);
        if $signed {
            for &v in small {
                ds.push((v as i64).wrapping_neg() as $t);
            }
            ds.push(<$t>::MIN);
            ds.push(<$t>::MIN + 1);
            ds.push(<$t>::MIN / 2);
        }
        for _ in 0..16 {
            let d: $t = rng.random();
            if d != 0 {
                ds.push(d);
            }
        }

        // Dividends: edges + randoms.
        let mut xs: Vec<$t> = vec![0, 1, 2, 3, 100, <$t>::MAX, <$t>::MAX - 1, <$t>::MAX / 2];
        if $signed {
            xs.push(<$t>::MIN);
            xs.push(<$t>::MIN + 1);
            xs.push(<$t>::MIN / 2);
        }
        for _ in 0..48 {
            xs.push(rng.random());
        }

        for &d in &ds {
            if d == 0 {
                continue;
            }
            let div = d.to_divider();
            // Unsigned branchfree does not support a divisor of 1.
            let bf = if !$signed && d == 1 {
                assert!(
                    d.try_to_branchfree_divider().is_err(),
                    "unsigned branchfree must reject 1"
                );
                None
            } else {
                let b = d.to_branchfree_divider();
                assert!(d.try_to_branchfree_divider().is_ok());
                Some(b)
            };

            for &x in &xs {
                // i_::MIN / -1 overflows Rust's `/` (and is UB-ish for the HW); skip it.
                if $signed && x == <$t>::MIN && d == (-1i64 as $t) {
                    continue;
                }
                assert_eq!(div.divide(x), x / d, "Divider: {} / {}", x, d);
                if let Some(bf) = bf {
                    assert_eq!(bf.divide(x), x / d, "BranchfreeDivider: {} / {}", x, d);
                }
            }
        }
    }};
}

#[test]
fn u8_div() {
    div_check!(u8, false);
}
#[test]
fn u16_div() {
    div_check!(u16, false);
}
#[test]
fn u32_div() {
    div_check!(u32, false);
}
#[test]
fn u64_div() {
    div_check!(u64, false);
}
#[test]
fn i8_div() {
    div_check!(i8, true);
}
#[test]
fn i16_div() {
    div_check!(i16, true);
}
#[test]
fn i32_div() {
    div_check!(i32, true);
}
#[test]
fn i64_div() {
    div_check!(i64, true);
}

/// Vectorized division: `vec / Divider`, `vec / BranchfreeDivider`, per-lane
/// `vec / VectorDivider` (via `to_divider`), and masked `div_c/_m/_z`. These run
/// in **debug** (unlike `divide.rs`, which skips debug), so they cover
/// `divider/vector.rs` and the `Vector` `Div`/`DivMasked` impls under cov-collect.
macro_rules! vdiv_check {
    ($reg:ty, $e:ty, $signed:literal) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let mut rng = harness::rng();
        let rdv = |v: V| v.into_array().as_slice().to_vec();

        let mut consts: Vec<$e> = vec![2, 3, 4, 7, 8, 16, 100, <$e>::MAX, <$e>::MAX / 3];
        if $signed {
            // written via wrapping_neg so the literals also type-check for the
            // unsigned instantiations (where this branch is never taken).
            consts.extend_from_slice(&[
                (2 as $e).wrapping_neg(),
                (3 as $e).wrapping_neg(),
                (7 as $e).wrapping_neg(),
                (8 as $e).wrapping_neg(),
                <$e>::MIN / 2,
            ]);
        }

        // --- constant divisors: vec / Divider and vec / BranchfreeDivider ---
        for &d in &consts {
            let dv = d.to_divider();
            let bf = (if $signed || d != 1 {
                Some(d.to_branchfree_divider())
            } else {
                None
            });
            for _ in 0..24 {
                let a: Vec<$e> = (0..n).map(|_| rng.random()).collect();
                let vnum = V::from_slice(&a);
                let got = rdv(vnum / dv);
                for i in 0..n {
                    assert_eq!(got[i], a[i].wrapping_div(d), "vec/Divider {} / {}", a[i], d);
                }
                if let Some(bf) = bf {
                    let gotb = rdv(vnum / bf);
                    for i in 0..n {
                        assert_eq!(gotb[i], a[i].wrapping_div(d), "vec/BF {} / {}", a[i], d);
                    }
                }
            }
        }

        // --- per-lane VectorDivider (vden.to_divider()) + masked variants ---
        for _ in 0..64 {
            let a: Vec<$e> = (0..n).map(|_| rng.random()).collect();
            let den: Vec<$e> = (0..n)
                .map(|_| {
                    let mut d: $e = rng.random();
                    while d == 0 || (!$signed && d == 1) {
                        d = rng.random();
                    }
                    d
                })
                .collect();
            let vnum = V::from_slice(&a);
            let vden = V::from_slice(&den);
            let vdiv = vden.to_divider(); // VectorDivider (IntegerVector::to_divider)
            let plain: Vec<$e> = (0..n).map(|i| a[i].wrapping_div(den[i])).collect();
            assert_eq!(rdv(vnum / vdiv), plain, "vec/VectorDivider");

            // masked: div_c ? a/den : a ; div_m ? : src ; div_z ? : 0
            let src = V::from_slice(&(0..n).map(|_| rng.random::<$e>()).collect::<Vec<_>>());
            let mask = vnum.cmp_lt(vden);
            let mb: Vec<bool> = (0..n).map(|i| a[i] < den[i]).collect();
            let sel = |t: &[$e], f: &[$e]| -> Vec<$e> { (0..n).map(|i| if mb[i] { t[i] } else { f[i] }).collect() };
            assert_eq!(rdv(vnum.div_c(mask, vdiv)), sel(&plain, &a), "div_c");
            assert_eq!(rdv(vnum.div_m(src, mask, vdiv)), sel(&plain, &rdv(src)), "div_m");
            assert_eq!(rdv(vnum.div_z(mask, vdiv)), sel(&plain, &vec![0 as $e; n]), "div_z");
        }
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    #[test]
    fn vector_div_v3() {
        vdiv_check!(<X86V3 as Simd>::u32x8, u32, false);
        vdiv_check!(<X86V3 as Simd>::i32x8, i32, true);
        vdiv_check!(<X86V3 as Simd>::u64x4, u64, false);
        vdiv_check!(<X86V3 as Simd>::i64x4, i64, true);
    }

    #[test]
    fn vector_div_v2() {
        vdiv_check!(<X86V2 as Simd>::u32x4, u32, false);
        vdiv_check!(<X86V2 as Simd>::i32x4, i32, true);
        // 64-bit native VectorDivider exercises the SSE variable-shift polyfill
        // (the lane-swap bug fixed in _mm_s{ll,rl}v_epi64x_v1).
        vdiv_check!(<X86V2 as Simd>::u64x2, u64, false);
        vdiv_check!(<X86V2 as Simd>::i64x2, i64, true);
    }

    #[test]
    fn vector_div_v1() {
        vdiv_check!(<X86V1 as Simd>::u32x4, u32, false);
        vdiv_check!(<X86V1 as Simd>::i32x4, i32, true);
        vdiv_check!(<X86V1 as Simd>::u64x2, u64, false);
        vdiv_check!(<X86V1 as Simd>::i64x2, i64, true);
    }
}

// wasm: native 128-bit u32x4/i32x4/u64x2/i64x2 divider path (libdivide polyfills).
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    #[test]
    fn vector_div_wasm() {
        vdiv_check!(<Wasm as Simd>::u32x4, u32, false);
        vdiv_check!(<Wasm as Simd>::i32x4, i32, true);
        vdiv_check!(<Wasm as Simd>::u64x2, u64, false);
        vdiv_check!(<Wasm as Simd>::i64x2, i64, true);
    }
}

// neon: native 128-bit u32x4/i32x4/u64x2/i64x2 divider path (libdivide polyfills).
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    #[test]
    fn vector_div_neon() {
        vdiv_check!(<Neon as Simd>::u32x4, u32, false);
        vdiv_check!(<Neon as Simd>::i32x4, i32, true);
        vdiv_check!(<Neon as Simd>::u64x2, u64, false);
        vdiv_check!(<Neon as Simd>::i64x2, i64, true);
    }
}

#[test]
fn d_zero_constructs() {
    // d == 0 hits the degenerate `_internal` branch, and must not panic to construct.
    let _ = 0u32.to_divider();
    let _ = 0i32.to_divider();
}

#[test]
#[should_panic]
fn unsigned_branchfree_one_panics() {
    let _ = BranchfreeDivider::u32(1);
}

#[test]
fn signed_branchfree_one_ok() {
    // Signed branchfree DOES support 1 (unlike unsigned).
    let bf = BranchfreeDivider::i32(1);
    for x in [-7, -1, 0, 1, 123, i32::MAX, i32::MIN] {
        assert_eq!(bf.divide(x), x);
    }
}

#[test]
fn from_and_tryfrom() {
    // From<$t> for Divider (signed + unsigned)
    let du: Divider<u32> = Divider::from(7u32);
    assert_eq!(du.divide(100), 100 / 7);
    let di: Divider<i32> = Divider::from(-7i32);
    assert_eq!(di.divide(100), 100 / -7);
    // TryFrom for unsigned branchfree: rejects 1, accepts others.
    assert!(BranchfreeDivider::<u32>::try_from(1u32).is_err());
    assert_eq!(BranchfreeDivider::<u32>::try_from(7u32).unwrap().divide(100), 100 / 7);
    // From for signed branchfree
    let bi: BranchfreeDivider<i32> = BranchfreeDivider::from(-3i32);
    assert_eq!(bi.divide(100), 100 / -3);
}

#[test]
fn accessors_eq_clone_deref() {
    let a = Divider::u32(7);
    let b = 7u32.to_divider();
    assert_eq!(a, b); // PartialEq
    let c = a; // Copy
    assert_eq!(c.multiplier(), a.multiplier());
    assert_eq!(c.shift(), a.shift());
    // BranchfreeDivider derefs to Divider (so multiplier()/shift() are reachable).
    let bf = BranchfreeDivider::u32(7);
    let _ = bf.multiplier();
    let _ = bf.shift();
    let bf2 = bf; // Copy
    assert!(bf == bf2); // PartialEq on BranchfreeDivider

    // explicit Clone (Copy bypasses the manual `clone` impls) + Debug
    #[allow(clippy::clone_on_copy)]
    let a_cl = a.clone();
    assert_eq!(a_cl, a);
    #[allow(clippy::clone_on_copy)]
    let bf_cl = bf.clone();
    assert!(bf_cl == bf);
    let ds = format!("{a:?}");
    assert!(
        ds.contains("Divider") && ds.contains("multiplier") && ds.contains("shift"),
        "{ds}"
    );
    let bs = format!("{bf:?}");
    assert!(bs.contains("BranchfreeDivider"), "{bs}");
}

#[test]
fn unsupported_divisor_error() {
    let e = BranchfreeDivider::<u32>::try_from(1u32).unwrap_err();
    assert_eq!(format!("{e}"), "unsupported divisor");
    // exercise Debug + the std::error::Error impl
    let _ = format!("{e:?}");
    let _: &dyn core::error::Error = &e as &dyn core::error::Error;
    let _ = UnsupportedDivisor;
}
