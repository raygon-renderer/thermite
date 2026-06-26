//! Coverage for the algebraic `RealMath`/`SpatialMath`/`FloatVector` functions
//! that `diff_math` (transcendentals) doesn't touch: norms, angle conversion and
//! wrapping, linear/affine maps, and smoothstep. Implementations live mostly in
//! `math/specialized/mod.rs`.
//!
//! Most are exact-ish arithmetic checked against an `f64` oracle. The angle
//! wrappers are checked *by property* (result in range, and congruent mod 2π) to
//! avoid floating-point boundary ambiguity at ±π.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use rand::RngExt;

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

const TRIALS: usize = 400;
const PI: f64 = core::f64::consts::PI;

macro_rules! real_suite {
    ($mod:ident, $backend:ty, $reg:ident, $e:ty, $rel:expr) => {
        mod $mod {
            use super::*;
            type V = Vector<<$backend as Simd>::$reg>;
            const L: usize = <V as GenericVector>::LANES;

            fn rd(v: V) -> Vec<f64> {
                v.into_array().as_slice().iter().map(|&x| x as f64).collect()
            }
            fn mk(rng: &mut rand::rngs::SmallRng, lo: f64, hi: f64) -> (V, Vec<f64>) {
                let a: Vec<$e> = (0..L).map(|_| rng.random_range(lo..hi) as $e).collect();
                (V::from_slice(&a), a.iter().map(|&x| x as f64).collect())
            }
            #[track_caller]
            fn close(name: &str, got: &[f64], want: &[f64]) {
                for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (g - w).abs() <= ($rel) * w.abs().max(1.0),
                        "{} [{}] lane {}: got {} want {}",
                        stringify!($mod), name, i, g, w
                    );
                }
            }

            #[test]
            fn norms_and_scale() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (v, x) = mk(&mut rng, -50.0, 50.0);
                    close("l1_norm", &rd(v.l1_norm()), &x.iter().map(|&x| x.abs()).collect::<Vec<_>>());
                    close("l2_norm", &rd(v.l2_norm()), &x.iter().map(|&x| x.abs()).collect::<Vec<_>>());
                    close("l2_norm_squared", &rd(v.l2_norm_squared()), &x.iter().map(|&x| x * x).collect::<Vec<_>>());
                    close("to_degrees", &rd(v.to_degrees()), &x.iter().map(|&x| x * 180.0 / PI).collect::<Vec<_>>());
                    close("to_radians", &rd(v.to_radians()), &x.iter().map(|&x| x * PI / 180.0).collect::<Vec<_>>());
                }
            }

            #[test]
            fn angle_wrapping() {
                let mut rng = harness::rng();
                let tol = ($rel as f64).max(1e-5) * 8.0;
                for _ in 0..TRIALS {
                    let (v, x) = mk(&mut rng, -20.0, 20.0);
                    // wrap_angle: in [-π, π) and congruent to x mod 2π.
                    let w = rd(v.wrap_angle());
                    for (i, &g) in w.iter().enumerate() {
                        assert!(g >= -PI - tol && g < PI + tol, "{} wrap_angle lane {} out of range: {}", stringify!($mod), i, g);
                        let d = g - x[i];
                        let k = (d / (2.0 * PI)).round();
                        assert!((d - k * 2.0 * PI).abs() <= tol, "{} wrap_angle lane {} not congruent: {} vs {}", stringify!($mod), i, g, x[i]);
                    }
                    // angle_diff(a, b): in [-π, π) and congruent to (a-b) mod 2π.
                    let (vb, xb) = mk(&mut rng, -20.0, 20.0);
                    let dgot = rd(v.angle_diff(vb));
                    for (i, &g) in dgot.iter().enumerate() {
                        assert!(g >= -PI - tol && g < PI + tol, "{} angle_diff lane {} out of range: {}", stringify!($mod), i, g);
                        let d = g - (x[i] - xb[i]);
                        let k = (d / (2.0 * PI)).round();
                        assert!((d - k * 2.0 * PI).abs() <= tol, "{} angle_diff lane {} not congruent", stringify!($mod), i);
                    }
                }
            }

            #[test]
            fn interpolation() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (vt, t) = mk(&mut rng, 0.0, 1.0);
                    let (va, a) = mk(&mut rng, -10.0, 10.0);
                    let (vb, b) = mk(&mut rng, -10.0, 10.0);

                    // lerp(t, a, b) = a + (b - a) * t
                    let want: Vec<f64> = (0..L).map(|i| a[i] + (b[i] - a[i]) * t[i]).collect();
                    close("lerp", &rd(vt.lerp(va, vb)), &want);
                    // mix(t, a, b) = a*(1-t) + b*t  (== lerp)
                    let want: Vec<f64> = (0..L).map(|i| a[i] * (1.0 - t[i]) + b[i] * t[i]).collect();
                    close("mix", &rd(vt.mix(va, vb)), &want);

                    // rescale from [im, iM] to [om, oM]
                    let (im, iM, om, oM) = (-2.0_f64, 5.0_f64, 10.0_f64, 20.0_f64);
                    let (vx, x) = mk(&mut rng, -2.0, 5.0);
                    let want: Vec<f64> = (0..L).map(|i| om + (x[i] - im) / (iM - im) * (oM - om)).collect();
                    let got = vx.rescale(V::splat(im as $e), V::splat(iM as $e), V::splat(om as $e), V::splat(oM as $e));
                    close("rescale", &rd(got), &want);
                }
            }

            #[test]
            fn step_and_smoothstep() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    // step(x, edge): 1 if x >= edge else 0
                    let (vx, x) = mk(&mut rng, -5.0, 5.0);
                    let (ve, e) = mk(&mut rng, -5.0, 5.0);
                    let want: Vec<f64> = (0..L).map(|i| if x[i] >= e[i] { 1.0 } else { 0.0 }).collect();
                    close("step", &rd(vx.step(ve)), &want);

                    // smoothstep over the default [0, 1] edges, inputs spanning the clamp region.
                    let (vs, s) = mk(&mut rng, -0.5, 1.5);
                    let clamp01 = |v: f64| v.clamp(0.0, 1.0);
                    // N=1: linear (clamped)
                    close("smoothstep<1>", &rd(vs.smoothstep::<1>(None)), &s.iter().map(|&v| clamp01(v)).collect::<Vec<_>>());
                    // N=2: 3t^2 - 2t^3
                    let want: Vec<f64> = s.iter().map(|&v| { let t = clamp01(v); t * t * (3.0 - 2.0 * t) }).collect();
                    close("smoothstep<2>", &rd(vs.smoothstep::<2>(None)), &want);
                    // derivative of N=2: 6t - 6t^2 (0 in the clamped tails)
                    let want: Vec<f64> = s.iter().map(|&v| { let t = clamp01(v); 6.0 * t * (1.0 - t) }).collect();
                    close("smoothstep_derivative<2>", &rd(vs.smoothstep_derivative::<2>(None)), &want);

                    // N=3 "smootherstep": 6t^5 - 15t^4 + 10t^3
                    let want: Vec<f64> = s.iter().map(|&v| { let t = clamp01(v); t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }).collect();
                    close("smoothstep<3>", &rd(vs.smoothstep::<3>(None)), &want);

                    // inverse_smoothstep round-trips smoothstep on [0,1] for N=1,2,3
                    let (vx, _x) = mk(&mut rng, 0.02, 0.98);
                    // the N>=3 inverse is an iterative (Newton) solve, so under the default
                    // policy the round-trip is only good to ~1e-2; the closed forms are tighter.
                    let rt = |g: V, want: V, tag: &str| {
                        for (i, (&a, &b)) in rd(g).iter().zip(rd(want).iter()).enumerate() {
                            assert!((a - b).abs() <= 1.0e-2 + 80.0 * ($rel), "{} {} lane {}: got {} want {}", stringify!($mod), tag, i, a, b);
                        }
                    };
                    rt(vx.smoothstep::<1>(None).inverse_smoothstep::<1>(None), vx, "inv_smoothstep<1>");
                    rt(vx.smoothstep::<2>(None).inverse_smoothstep::<2>(None), vx, "inv_smoothstep<2>");
                    rt(vx.smoothstep::<3>(None).inverse_smoothstep::<3>(None), vx, "inv_smoothstep<3>"); // Newton path

                    // with explicit edges [a,b]: smoothstep maps [a,b]->[0,1], inverse maps back
                    let edges = Some((V::splat(-2.0 as $e), V::splat(5.0 as $e)));
                    let (vxe, _) = mk(&mut rng, -1.8, 4.8);
                    rt(vxe.smoothstep::<2>(edges).inverse_smoothstep::<2>(edges), vxe, "inv_smoothstep<2>+edges");
                    rt(vxe.smoothstep::<3>(edges).inverse_smoothstep::<3>(edges), vxe, "inv_smoothstep<3>+edges");

                    // N=0 is the (non-invertible) step function — just exercise both paths
                    let _ = vx.inverse_smoothstep::<0>(None);
                    let _ = vx.inverse_smoothstep::<0>(edges);
                }
            }

            #[test]
            fn power_variants() {
                let mut rng = harness::rng();
                let loose = 5e-3_f64.max(60.0 * ($rel));
                for _ in 0..TRIALS {
                    let (vx, x) = mk(&mut rng, 0.4, 3.0); // positive base, moderate magnitude
                    // powiv: per-lane integer exponent (indexed = [0, 1, 2, ...])
                    let e = <V as GenericVector>::Signed::indexed();
                    let want: Vec<f64> = (0..L).map(|i| x[i].powi(i as i32)).collect();
                    for (i, (&g, &w)) in rd(vx.powiv(e)).iter().zip(&want).enumerate() {
                        assert!((g - w).abs() <= loose * w.abs().max(1.0), "{} powiv lane {}: {} vs {}", stringify!($mod), i, g, w);
                    }
                    // nth_root::<N> == x^(1/N)
                    for (n, root) in [(2i32, 0.5f64), (3, 1.0 / 3.0), (4, 0.25), (5, 0.2)] {
                        let want: Vec<f64> = x.iter().map(|&v| v.powf(root)).collect();
                        let got = match n { 2 => rd(vx.nth_root::<2>()), 3 => rd(vx.nth_root::<3>()), 4 => rd(vx.nth_root::<4>()), _ => rd(vx.nth_root::<5>()) };
                        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                            assert!((g - w).abs() <= loose * w.abs().max(1.0), "{} nth_root<{}> lane {}: {} vs {}", stringify!($mod), n, i, g, w);
                        }
                    }
                }
            }

            #[test]
            fn smooth_interpolator_props() {
                let k = V::splat(0.5 as $e);
                // midpoint maps to 0.5 (e == 0 -> 1/(exp(0)+1))
                for &v in rd(V::splat(0.5 as $e).smooth_interpolator(None, k)).iter() {
                    assert!((v - 0.5).abs() <= 1e-3, "{} smooth_interpolator(0.5) = {}", stringify!($mod), v);
                }
                // clamps outside [0,1]
                for &v in rd(V::splat(-0.3 as $e).smooth_interpolator(None, k)).iter() {
                    assert!(v.abs() <= 1e-6, "{} smooth_interpolator(-0.3) = {}", stringify!($mod), v);
                }
                for &v in rd(V::splat(1.3 as $e).smooth_interpolator(None, k)).iter() {
                    assert!((v - 1.0).abs() <= 1e-6, "{} smooth_interpolator(1.3) = {}", stringify!($mod), v);
                }
                // in-range outputs stay within [0,1]; exercise the edges-rescale path too
                let mut rng = harness::rng();
                let edges = Some((V::splat(-2.0 as $e), V::splat(5.0 as $e)));
                for _ in 0..TRIALS {
                    let (vt, _) = mk(&mut rng, 0.05, 0.95);
                    for &v in rd(vt.smooth_interpolator(None, k)).iter() {
                        assert!((-1e-6..=1.0 + 1e-6).contains(&v), "{} smooth_interpolator out of [0,1]: {}", stringify!($mod), v);
                    }
                    let (vte, _) = mk(&mut rng, -1.8, 4.8);
                    for &v in rd(vte.smooth_interpolator(edges, k)).iter() {
                        assert!((-1e-6..=1.0 + 1e-6).contains(&v), "{} smooth_interpolator+edges out of [0,1]: {}", stringify!($mod), v);
                    }
                }
            }

            #[test]
            fn n_dimensional_hypot() {
                let mut rng = harness::rng();
                for _ in 0..TRIALS {
                    let (va, a) = mk(&mut rng, -10.0, 10.0);
                    let (vb, b) = mk(&mut rng, -10.0, 10.0);
                    let (vc, c) = mk(&mut rng, -10.0, 10.0);

                    close("hypot_n<1>", &rd(V::hypot_n([va])), &a.iter().map(|&x| x.abs()).collect::<Vec<_>>());
                    let h2: Vec<f64> = (0..L).map(|i| (a[i] * a[i] + b[i] * b[i]).sqrt()).collect();
                    close("hypot_n<2>", &rd(V::hypot_n([va, vb])), &h2);
                    let h3: Vec<f64> = (0..L).map(|i| (a[i] * a[i] + b[i] * b[i] + c[i] * c[i]).sqrt()).collect();
                    close("hypot_n<3>", &rd(V::hypot_n([va, vb, vc])), &h3);

                    // inv_hypot_n = 1/hypot_n; approximate rsqrt under Performance, so loose,
                    // and skip near-zero magnitudes where 1/h blows up.
                    let inv = rd(V::inv_hypot_n([va, vb]));
                    for i in 0..L {
                        if h2[i] > 0.5 {
                            let want = 1.0 / h2[i];
                            assert!((inv[i] - want).abs() <= 1e-2 * want.abs().max(1.0), "{} inv_hypot_n<2> lane {}: {} vs {}", stringify!($mod), i, inv[i], want);
                        }
                    }
                }
            }
        }
    };
}

// scalar is the always-available oracle (runs on every target).
real_suite!(scalar_f32, Scalar, f32x4, f32, 2.0e-4);
real_suite!(scalar_f64, Scalar, f64x4, f64, 1.0e-10);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    real_suite!(v3_f32, X86V3, f32x4, f32, 2.0e-4);
    real_suite!(v3_f64, X86V3, f64x4, f64, 1.0e-10);
    real_suite!(v2_f32, X86V2, f32x4, f32, 2.0e-4);
    real_suite!(v2_f64, X86V2, f64x4, f64, 1.0e-10);
    real_suite!(v1_f32, X86V1, f32x4, f32, 2.0e-4);
    real_suite!(v1_f64, X86V1, f64x4, f64, 1.0e-10);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    real_suite!(wasm_f32, Wasm, f32x4, f32, 2.0e-4);
    real_suite!(wasm_f64, Wasm, f64x4, f64, 1.0e-10);
}
