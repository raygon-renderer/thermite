//! `PrecisionPolicy::Reference` is defined as bit-identical to the `libm` crate.
//!
//! This is the test that gives that definition teeth. Every function with a reference
//! arm in `math/specialized/{ps,pd}.rs` is checked lane-for-lane against the scalar
//! `libm` entry point it claims to reproduce, with an exact bit comparison rather than
//! a tolerance. A one-ulp difference here is a bug, not noise.
//!
//! Both the 1-lane scalar backend and a native wide backend are covered, because the
//! contract is that the tier is backend- and lane-count-independent. If those two ever
//! disagree, the lane loop is at fault.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use rand::RngExt;

use thermite::Vector;
use thermite::math::policy::policies::Reference;
use thermite::prelude::*;
use thermite::simd::Simd;

const TRIALS: usize = 64;

/// Bit-equal, treating any two NaNs as equal (libm and the lane loop can carry
/// different payloads through without either being wrong).
fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Interesting inputs every function gets tested on regardless of its random range.
const EDGES_F32: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];
const EDGES_F64: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

macro_rules! suite {
    (
        $mod:ident, $vty:ty, $e:ty, $same:ident, $edges:ident,
        unary { $( $u:ident ( $ulo:expr, $uhi:expr ) = $uv:expr, $us:expr ; )* }
        binary { $( $b:ident ( $blo:expr, $bhi:expr ) = $bv:expr, $bs:expr ; )* }
        pairs { $( $p:ident ( $plo:expr, $phi:expr ) = $pv:expr, $ps:expr ; )* }
    ) => {
        mod $mod {
            use super::*;

            type V = $vty;
            const L: usize = <V as GenericVector>::LANES;

            fn inputs(rng: &mut rand::rngs::SmallRng, lo: f64, hi: f64) -> Vec<$e> {
                (0..L).map(|_| rng.random_range(lo..hi) as $e).collect()
            }

            $(
                #[test]
                fn $u() {
                    let mut rng = harness::rng();
                    let vf: fn(V) -> V = $uv;
                    let sf: fn($e) -> $e = $us;

                    let mut cases: Vec<Vec<$e>> = vec![$edges.to_vec()];
                    for _ in 0..TRIALS {
                        cases.push(inputs(&mut rng, $ulo, $uhi));
                    }

                    for case in cases {
                        for chunk in case.chunks(L) {
                            let mut buf = [0 as $e; L];
                            buf[..chunk.len()].copy_from_slice(chunk);
                            let got = vf(V::from_slice(&buf)).into_array();

                            for (i, &x) in buf.iter().enumerate() {
                                let want = sf(x);
                                assert!(
                                    $same(got.as_slice()[i], want),
                                    "{}({:?}) lane {}: got {:?}, libm {:?}",
                                    stringify!($u), x, i, got.as_slice()[i], want,
                                );
                            }
                        }
                    }
                }
            )*

            $(
                #[test]
                fn $b() {
                    let mut rng = harness::rng();
                    let vf: fn(V, V) -> V = $bv;
                    let sf: fn($e, $e) -> $e = $bs;

                    for _ in 0..TRIALS {
                        let a = inputs(&mut rng, $blo, $bhi);
                        let b = inputs(&mut rng, $blo, $bhi);
                        let got = vf(V::from_slice(&a), V::from_slice(&b)).into_array();

                        for i in 0..L {
                            let want = sf(a[i], b[i]);
                            assert!(
                                $same(got.as_slice()[i], want),
                                "{}({:?}, {:?}) lane {}: got {:?}, libm {:?}",
                                stringify!($b), a[i], b[i], i, got.as_slice()[i], want,
                            );
                        }
                    }
                }
            )*

            $(
                #[test]
                fn $p() {
                    let mut rng = harness::rng();
                    let vf: fn(V) -> (V, V) = $pv;
                    let sf: fn($e) -> ($e, $e) = $ps;

                    for _ in 0..TRIALS {
                        let a = inputs(&mut rng, $plo, $phi);
                        let (g0, g1) = vf(V::from_slice(&a));
                        let (g0, g1) = (g0.into_array(), g1.into_array());

                        for i in 0..L {
                            let (w0, w1) = sf(a[i]);
                            assert!(
                                $same(g0.as_slice()[i], w0) && $same(g1.as_slice()[i], w1),
                                "{}({:?}) lane {}: got {:?}, libm {:?}",
                                stringify!($p), a[i], i,
                                (g0.as_slice()[i], g1.as_slice()[i]), (w0, w1),
                            );
                        }
                    }
                }
            )*
        }
    };
}

macro_rules! f32_suite {
    ($mod:ident, $vty:ty) => {
        suite! {
            $mod, $vty, f32, same_f32, EDGES_F32,
            unary {
                sin(-100.0, 100.0)    = |v| v.sin_p::<Reference>(),    libm::sinf;
                cos(-100.0, 100.0)    = |v| v.cos_p::<Reference>(),    libm::cosf;
                tan(-100.0, 100.0)    = |v| v.tan_p::<Reference>(),    libm::tanf;
                asin(-1.0, 1.0)       = |v| v.asin_p::<Reference>(),   libm::asinf;
                acos(-1.0, 1.0)       = |v| v.acos_p::<Reference>(),   libm::acosf;
                atan(-50.0, 50.0)     = |v| v.atan_p::<Reference>(),   libm::atanf;
                sinh(-20.0, 20.0)     = |v| v.sinh_p::<Reference>(),   libm::sinhf;
                cosh(-20.0, 20.0)     = |v| v.cosh_p::<Reference>(),   libm::coshf;
                tanh(-20.0, 20.0)     = |v| v.tanh_p::<Reference>(),   libm::tanhf;
                asinh(-50.0, 50.0)    = |v| v.asinh_p::<Reference>(),  libm::asinhf;
                acosh(1.0, 100.0)     = |v| v.acosh_p::<Reference>(),  libm::acoshf;
                atanh(-1.0, 1.0)      = |v| v.atanh_p::<Reference>(),  libm::atanhf;
                exp(-50.0, 50.0)      = |v| v.exp_p::<Reference>(),    libm::expf;
                exp2(-50.0, 50.0)     = |v| v.exp2_p::<Reference>(),   libm::exp2f;
                exp10(-30.0, 30.0)    = |v| v.exp10_p::<Reference>(),  libm::exp10f;
                exp_m1(-20.0, 20.0)   = |v| v.exp_m1_p::<Reference>(), libm::expm1f;
                ln(0.0, 1000.0)       = |v| v.ln_p::<Reference>(),     libm::logf;
                ln_1p(-1.0, 1000.0)   = |v| v.ln_1p_p::<Reference>(),  libm::log1pf;
                log2(0.0, 1000.0)     = |v| v.log2_p::<Reference>(),   libm::log2f;
                log10(0.0, 1000.0)    = |v| v.log10_p::<Reference>(),  libm::log10f;
                cbrt(-1000.0, 1000.0) = |v| v.cbrt_p::<Reference>(),   libm::cbrtf;
            }
            binary {
                powf(0.0, 30.0)    = |a, b| a.powf_p::<Reference>(b),  libm::powf;
                atan2(-50.0, 50.0) = |a, b| a.atan2_p::<Reference>(b), libm::atan2f;
                hypot(-50.0, 50.0) = |a, b| a.hypot_p::<Reference>(b), libm::hypotf;
            }
            pairs {
                sin_cos(-100.0, 100.0) = |v| v.sin_cos_p::<Reference>(), libm::sincosf;
                sinh_cosh(-20.0, 20.0) = |v| v.sinh_cosh_p::<Reference>(),
                    (|x| (libm::sinhf(x), libm::coshf(x))) as fn(f32) -> (f32, f32);
            }
        }
    };
}

macro_rules! f64_suite {
    ($mod:ident, $vty:ty) => {
        suite! {
            $mod, $vty, f64, same_f64, EDGES_F64,
            unary {
                sin(-100.0, 100.0)    = |v| v.sin_p::<Reference>(),    libm::sin;
                cos(-100.0, 100.0)    = |v| v.cos_p::<Reference>(),    libm::cos;
                tan(-100.0, 100.0)    = |v| v.tan_p::<Reference>(),    libm::tan;
                asin(-1.0, 1.0)       = |v| v.asin_p::<Reference>(),   libm::asin;
                acos(-1.0, 1.0)       = |v| v.acos_p::<Reference>(),   libm::acos;
                atan(-50.0, 50.0)     = |v| v.atan_p::<Reference>(),   libm::atan;
                sinh(-20.0, 20.0)     = |v| v.sinh_p::<Reference>(),   libm::sinh;
                cosh(-20.0, 20.0)     = |v| v.cosh_p::<Reference>(),   libm::cosh;
                tanh(-20.0, 20.0)     = |v| v.tanh_p::<Reference>(),   libm::tanh;
                asinh(-50.0, 50.0)    = |v| v.asinh_p::<Reference>(),  libm::asinh;
                acosh(1.0, 100.0)     = |v| v.acosh_p::<Reference>(),  libm::acosh;
                atanh(-1.0, 1.0)      = |v| v.atanh_p::<Reference>(),  libm::atanh;
                exp(-50.0, 50.0)      = |v| v.exp_p::<Reference>(),    libm::exp;
                exp2(-50.0, 50.0)     = |v| v.exp2_p::<Reference>(),   libm::exp2;
                exp10(-30.0, 30.0)    = |v| v.exp10_p::<Reference>(),  libm::exp10;
                exp_m1(-20.0, 20.0)   = |v| v.exp_m1_p::<Reference>(), libm::expm1;
                ln(0.0, 1000.0)       = |v| v.ln_p::<Reference>(),     libm::log;
                ln_1p(-1.0, 1000.0)   = |v| v.ln_1p_p::<Reference>(),  libm::log1p;
                log2(0.0, 1000.0)     = |v| v.log2_p::<Reference>(),   libm::log2;
                log10(0.0, 1000.0)    = |v| v.log10_p::<Reference>(),  libm::log10;
                cbrt(-1000.0, 1000.0) = |v| v.cbrt_p::<Reference>(),   libm::cbrt;
            }
            binary {
                powf(0.0, 30.0)    = |a, b| a.powf_p::<Reference>(b),  libm::pow;
                atan2(-50.0, 50.0) = |a, b| a.atan2_p::<Reference>(b), libm::atan2;
                hypot(-50.0, 50.0) = |a, b| a.hypot_p::<Reference>(b), libm::hypot;
            }
            pairs {
                sin_cos(-100.0, 100.0) = |v| v.sin_cos_p::<Reference>(), libm::sincos;
                sinh_cosh(-20.0, 20.0) = |v| v.sinh_cosh_p::<Reference>(),
                    (|x| (libm::sinh(x), libm::cosh(x))) as fn(f64) -> (f64, f64);
            }
        }
    };
}

/// The f32 functions with no libm counterpart, composed in f64 and rounded once.
///
/// Bit-identity is not the contract here, as there is no external implementation to be
/// identical to. Two checks instead: hand-computed exact values pin each *definition*
/// independently of the implementation (a wrong composition, `exp(x/2)` for `exp(x)/2`,
/// fails loudly), and agreement with the `Best` tier confirms the two are computing the
/// same function rather than diverging quietly.
mod composed_f32 {
    use super::*;
    use thermite::math::policy::DefaultPolicy;
    use thermite::math::policy::policies::BestPrecision;

    type V = Vector<f32>;
    type Best = BestPrecision<DefaultPolicy>;

    #[track_caller]
    fn spot(name: &str, got: f32, want: f64) {
        let want = want as f32;
        let err = if want == 0.0 {
            got.abs() as f64
        } else {
            ((got - want) / want).abs() as f64
        };
        assert!(err <= 1e-6, "{name}: got {got:?}, want {want:?} (rel {err:e})");
    }

    #[track_caller]
    fn agrees(name: &str, reference: f32, best: f32) {
        if reference == best || (reference.is_nan() && best.is_nan()) {
            return;
        }
        let rel = ((reference - best) / reference).abs();
        assert!(
            rel <= 1e-4,
            "{name}: Reference {reference:?} vs Best {best:?} (rel {rel:e})"
        );
    }

    #[test]
    fn definitions_are_right() {
        let e = core::f64::consts::E;
        let ln2 = core::f64::consts::LN_2;

        // exph(x) = exp(x) / 2
        spot("exph(0)", V::splat(0.0).exph_p::<Reference>().extract::<0>(), 0.5);
        spot("exph(1)", V::splat(1.0).exph_p::<Reference>().extract::<0>(), e / 2.0);

        // exp2_m1(x) = 2^x - 1, exp10_m1(x) = 10^x - 1
        spot("exp2_m1(3)", V::splat(3.0).exp2_m1_p::<Reference>().extract::<0>(), 7.0);
        spot("exp2_m1(0)", V::splat(0.0).exp2_m1_p::<Reference>().extract::<0>(), 0.0);
        spot(
            "exp10_m1(2)",
            V::splat(2.0).exp10_m1_p::<Reference>().extract::<0>(),
            99.0,
        );

        // ln1m_expnx(x) = ln(1 - e^-x); at x = ln 2 that is ln(1/2) = -ln 2.
        spot(
            "ln1m_expnx(ln2)",
            V::splat(ln2 as f32).ln1m_expnx_p::<Reference>().extract::<0>(),
            -ln2,
        );

        // log_n::<N>(x) = log base N
        spot("log_3(9)", V::splat(9.0).log_n_p::<Reference, 3>().extract::<0>(), 2.0);
        spot("log_2(8)", V::splat(8.0).log_n_p::<Reference, 2>().extract::<0>(), 3.0);
        spot(
            "log_10(1000)",
            V::splat(1000.0).log_n_p::<Reference, 10>().extract::<0>(),
            3.0,
        );
    }

    #[test]
    fn reference_tracks_best() {
        // Ranges chosen to stay inside each function's finite, well-conditioned domain;
        // the point is catching a wrong formula, not probing overflow edges.
        for i in 0..=40 {
            let t = -4.0 + (i as f32) * 0.2;
            let v = V::splat(t);

            agrees(
                "exph",
                v.exph_p::<Reference>().extract::<0>(),
                v.exph_p::<Best>().extract::<0>(),
            );
            agrees(
                "exp2_m1",
                v.exp2_m1_p::<Reference>().extract::<0>(),
                v.exp2_m1_p::<Best>().extract::<0>(),
            );
            agrees(
                "exp10_m1",
                v.exp10_m1_p::<Reference>().extract::<0>(),
                v.exp10_m1_p::<Best>().extract::<0>(),
            );

            // ln1m_expnx needs x > 0.
            let p = V::splat(0.05 + (i as f32) * 0.25);
            agrees(
                "ln1m_expnx",
                p.ln1m_expnx_p::<Reference>().extract::<0>(),
                p.ln1m_expnx_p::<Best>().extract::<0>(),
            );

            // log_n needs x > 0.
            agrees(
                "log_n::<7>",
                p.log_n_p::<Reference, 7>().extract::<0>(),
                p.log_n_p::<Best, 7>().extract::<0>(),
            );
        }
    }
}

// The 1-lane scalar seeds: the tier's "no slower than calling libm yourself" case.
f32_suite!(scalar_f32, Vector<f32>);
f64_suite!(scalar_f64, Vector<f64>);

// A native wide backend: same contract, exercising the lane loop.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
f32_suite!(wide_f32, Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f32x8>);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
f64_suite!(wide_f64, Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>);
