//! Zernike: the batch basis against the per-mode loop it replaces, and the gradient
//! basis against forward-mode autodiff.
//!
//! Both comparisons are claims the documentation makes, so they are worth a number rather
//! than an argument. Everything runs in one process against the same inputs, which is what
//! makes the ratios trustworthy even though the absolute figures drift between runs - the
//! machine's noise floor on a cross-run delta is several percent, well above what a
//! same-process A/B needs to resolve.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::zernike::ansi_to_nm;
use thermite_special::{RealPrimalMath, SpecialMath, ZERNIKE_ORTHONORMAL};

type V = Vector<<X86V3 as Simd>::f32x8>;
type VD = Dual<V, 2>;

/// Pupil samples as Cartesian coordinates, which is how a real fit holds them, plus the
/// polar pair the per-mode form needs. Converting is itself part of what the batch form
/// avoids, so the conversion is done up front and given to the per-mode case for free -
/// this measures the kernels, not the coordinate change.
fn samples(count: usize) -> Vec<(V, V, V, V)> {
    (0..count)
        .map(|i| {
            let t = i as f32 * 0.7391 + 0.13;
            let r = ((i % 97) as f32 / 97.0).sqrt();
            let (s, c) = t.sin_cos();
            (V::splat(r * c), V::splat(r * s), V::splat(r), V::splat(t))
        })
        .collect()
}

macro_rules! degree {
    ($c:expr, $pts:expr, $L:literal, $N:literal) => {{
        let mut group = $c.benchmark_group(concat!("zernike/L", $L));
        group.throughput(criterion::Throughput::Elements(($pts.len() * V::LANES) as u64));

        // The whole basis in one call: the entry point a wavefront fit should use.
        group.bench_function("basis", |b| {
            let mut out = [V::ZERO; $N];
            b.iter(|| {
                for &(x, y, _, _) in $pts {
                    V::zernike_basis::<$L, ZERNIKE_ORTHONORMAL, $N>(black_box(x), y, &mut out);
                    black_box(&out);
                }
            })
        });

        // The same basis assembled one mode at a time, with the polar coordinates handed
        // over already converted. This is what the docs claim is O(L^3) against the
        // batch form's O(L^2).
        group.bench_function("per_mode_loop", |b| {
            let modes: Vec<(u32, i32)> = (0..$N as u32).map(ansi_to_nm).collect();
            let mut out = [V::ZERO; $N];
            b.iter(|| {
                for &(_, _, rho, theta) in $pts {
                    for (j, &(n, m)) in modes.iter().enumerate() {
                        out[j] = black_box(rho).zernike::<ZERNIKE_ORTHONORMAL>(theta, n, m);
                    }
                    black_box(&out);
                }
            })
        });

        // Values and both gradients, sharing the radial recurrence.
        group.bench_function("basis_d", |b| {
            let mut out = [V::ZERO; $N];
            let (mut ddx, mut ddy) = ([V::ZERO; $N], [V::ZERO; $N]);
            b.iter(|| {
                for &(x, y, _, _) in $pts {
                    V::zernike_basis_d::<$L, ZERNIKE_ORTHONORMAL, $N>(black_box(x), y, &mut out, &mut ddx, &mut ddy);
                    black_box((&out, &ddx, &ddy));
                }
            })
        });

        // The alternative the `basis_d` docs tell you not to use: seed a two-component
        // dual and run the value kernel, carrying both derivatives through every
        // operation instead of differentiating the recurrence.
        group.bench_function("basis_via_dual", |b| {
            let mut out = [VD::ZERO; $N];
            b.iter(|| {
                for &(x, y, _, _) in $pts {
                    VD::zernike_basis::<$L, ZERNIKE_ORTHONORMAL, $N>(
                        VD::variable(black_box(x), 0),
                        VD::variable(y, 1),
                        &mut out,
                    );
                    black_box(&out);
                }
            })
        });

        group.finish();
    }};
}

fn bench(c: &mut Criterion) {
    let pts = samples(256);

    degree!(c, &pts, 4, 15);
    degree!(c, &pts, 6, 28);
    degree!(c, &pts, 10, 66);
}

criterion_group!(benches, bench);
criterion_main!(benches);
