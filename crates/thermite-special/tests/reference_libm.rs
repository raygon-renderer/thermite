//! `PrecisionPolicy::Reference` must be bit-identical to the `libm` crate.
//!
//! The thermite-special half of the contract that `thermite/tests/reference_libm.rs`
//! covers for the core transcendentals. Only the functions libm actually provides get a
//! reference arm; everything else here (Lambert W, digamma, the elliptic integrals, the
//! activations) has no counterpart to be identical to, so it is deliberately absent.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::Vector;
use thermite::math::policy::policies::Reference;
use thermite::prelude::*;
use thermite::simd::Simd;

use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

/// Spread over the interesting structure: negative reflection, the poles at the
/// non-positive integers, the lgamma zeros at 1 and 2, and ordinary values.
const PROBES: &[f64] = &[
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.25, 7.5, 12.0, 30.0, 0.1, 0.01, 1e-5, -0.5, -1.5, -2.5, -3.75, -8.25, 0.0, -0.0,
    f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
];

fn same<E: PartialEq + Copy>(a: E, b: E, nan_a: bool, nan_b: bool) -> bool {
    (nan_a && nan_b) || a == b
}

macro_rules! suite {
    ($mod:ident, $vty:ty, $e:ty, $lgr:expr, $( $name:ident = $vf:expr, $sf:expr ; )*) => {
        mod $mod {
            use super::*;

            type V = $vty;
            const L: usize = <V as GenericVector>::LANES;

            $(
                #[test]
                fn $name() {
                    let vf: fn(V) -> V = $vf;
                    let sf: fn($e) -> $e = $sf;

                    for chunk in PROBES.chunks(L) {
                        let mut buf = [0 as $e; L];
                        for (slot, &p) in buf.iter_mut().zip(chunk) {
                            *slot = p as $e;
                        }

                        let got = vf(V::from_slice(&buf)).into_array();

                        for (i, &x) in buf.iter().enumerate() {
                            let (g, w) = (got.as_slice()[i], sf(x));
                            assert!(
                                same(g.to_bits(), w.to_bits(), g.is_nan(), w.is_nan()),
                                "{}({:?}) lane {}: got {:?}, libm {:?}",
                                stringify!($name), x, i, g, w,
                            );
                        }
                    }
                }
            )*

            // `lgamma_r` returns the sign alongside the value, so it gets its own body.
            #[test]
            fn lgamma_r() {
                let lgamma_r_scalar: fn($e) -> ($e, $e) = $lgr;

                for chunk in PROBES.chunks(L) {
                    let mut buf = [0 as $e; L];
                    for (slot, &p) in buf.iter_mut().zip(chunk) {
                        *slot = p as $e;
                    }

                    let (gv, gs) = V::from_slice(&buf).lgamma_r_p::<Reference>();
                    let (gv, gs) = (gv.into_array(), gs.into_array());

                    for (i, &x) in buf.iter().enumerate() {
                        let (wv, ws) = lgamma_r_scalar(x);
                        let g = gv.as_slice()[i];
                        assert!(
                            same(g.to_bits(), wv.to_bits(), g.is_nan(), wv.is_nan()),
                            "lgamma_r({:?}) value lane {}: got {:?}, libm {:?}", x, i, g, wv,
                        );
                        assert_eq!(gs.as_slice()[i], ws, "lgamma_r({:?}) sign lane {}", x, i);
                    }
                }
            }
        }
    };
}

macro_rules! f32_suite {
    ($mod:ident, $vty:ty) => {
        suite! {
            $mod, $vty, f32,
            |x| { let (v, s) = libm::lgammaf_r(x); (v, s as f32) },
            erf    = |v| v.erf_p::<Reference>(),    libm::erff;
            erfc   = |v| v.erfc_p::<Reference>(),   libm::erfcf;
            tgamma = |v| v.tgamma_p::<Reference>(), libm::tgammaf;
            lgamma = |v| v.lgamma_p::<Reference>(), libm::lgammaf;
        }
    };
}

macro_rules! f64_suite {
    ($mod:ident, $vty:ty) => {
        suite! {
            $mod, $vty, f64,
            |x| { let (v, s) = libm::lgamma_r(x); (v, s as f64) },
            erf    = |v| v.erf_p::<Reference>(),    libm::erf;
            erfc   = |v| v.erfc_p::<Reference>(),   libm::erfc;
            tgamma = |v| v.tgamma_p::<Reference>(), libm::tgamma;
            lgamma = |v| v.lgamma_p::<Reference>(), libm::lgamma;
        }
    };
}

/// The f32 functions with no libm entry point of their own, composed in f64.
///
/// Hand-computed exact values, so a wrong composition fails on the definition rather
/// than being graded against itself. `gelu` and `beta` are the ones worth pinning: gelu
/// goes through `erfc(-y)` instead of `1 + erf(y)` (which cancels to nothing for
/// negative x), and beta through `lgamma_r` instead of the `tgamma` product (which
/// overflows f64 past ~171 while f32 arguments run to 1e38).
mod composed_f32 {
    use super::*;

    type V = Vector<f32>;

    #[track_caller]
    fn spot(name: &str, got: f32, want: f64) {
        let want = want as f32;
        let ok = if want == 0.0 {
            got == 0.0
        } else {
            ((got - want) / want).abs() <= 1e-6
        };
        assert!(ok, "{name}: got {got:?}, want {want:?}");
    }

    #[test]
    fn logistic_sigmoid_definition() {
        // 1 / (1 + e^-x): 0 -> 1/2, ln 3 -> 3/4.
        let s = |x: f32| V::splat(x).logistic_sigmoid_p::<Reference>().extract::<0>();
        spot("sigmoid(0)", s(0.0), 0.5);
        spot("sigmoid(ln3)", s(core::f32::consts::LN_2 * 1.5849625), 0.75);
        spot("sigmoid(-inf)", s(f32::NEG_INFINITY), 0.0);
        spot("sigmoid(+inf)", s(f32::INFINITY), 1.0);
    }

    #[test]
    fn gelu_definition() {
        // GELU(x, 1) = x * Phi(x), the standard normal CDF.
        let g = |x: f32| V::splat(x).gelu_p::<Reference>(V::ONE).extract::<0>();
        spot("gelu(0)", g(0.0), 0.0);
        spot("gelu(1)", g(1.0), 0.841_344_746_068_542_9);
        spot("gelu(2)", g(2.0), 2.0 * 0.977_249_868_051_820_8);
        // The tail that `1 + erf(y)` would destroy: Phi(-5) = 2.866516e-7.
        spot("gelu(-5)", g(-5.0), -5.0 * 2.866_515_718_791_939_e-7);
    }

    #[test]
    fn beta_definition() {
        // B(a,b) = G(a)G(b)/G(a+b): B(1,1)=1, B(2,3)=1/12, B(1/2,1/2)=pi.
        let b = |x: f32, y: f32| V::splat(x).beta_p::<Reference>(V::splat(y)).extract::<0>();
        spot("beta(1,1)", b(1.0, 1.0), 1.0);
        spot("beta(2,3)", b(2.0, 3.0), 1.0 / 12.0);
        spot("beta(0.5,0.5)", b(0.5, 0.5), core::f64::consts::PI);
        // B(a,1) = 1/a exactly. At a = 200, `tgamma(a)` overflows f64 (it tops out
        // near 171) while the result is an ordinary f32, so the product form would
        // return inf/inf here and the log form returns 0.005.
        spot("beta(200,1)", b(200.0, 1.0), 1.0 / 200.0);
        spot("beta(300,2)", b(300.0, 2.0), 1.0 / (300.0 * 301.0));
    }
}

f32_suite!(scalar_f32, Vector<f32>);
f64_suite!(scalar_f64, Vector<f64>);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
f32_suite!(wide_f32, Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f32x8>);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
f64_suite!(wide_f64, Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>);
