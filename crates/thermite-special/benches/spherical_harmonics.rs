//! Spherical harmonics: real vectors, and the `Dual` seeding fast paths.
//!
//! The seeding cases cannot be told apart from the generated code. `Dual::constant`
//! and `Dual::variable` produce the same type and therefore one monomorphization, so
//! all three paths are compiled into the same function and only runtime data selects
//! between them. Which makes this the only way to put a number on them.
//!
//! Everything runs in one process against the same inputs, which is what makes the
//! ratios trustworthy even though the absolute figures drift between runs.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::{RealPrimalMath, RealSpecialMath, ShTable};

type V = Vector<<X86V3 as Simd>::f32x8>;

const L: usize = 4;
const N: usize = (L + 1) * (L + 1);

/// A spread of directions, so nothing folds and the branch predictor sees real work.
fn directions(count: usize) -> Vec<(V, V, V)> {
    (0..count)
        .map(|i| {
            let t = i as f32 * 0.7391 + 0.13;
            let u = i as f32 * 1.3247 + 0.29;
            let (st, ct) = t.sin_cos();
            let (su, cu) = u.sin_cos();
            (V::splat(st * cu), V::splat(st * su), V::splat(ct))
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let dirs = directions(256);

    let mut group = c.benchmark_group("sh/L4");
    group.throughput(criterion::Throughput::Elements((dirs.len() * V::LANES) as u64));

    // --- Real vectors: the floor everything else is measured against ---

    group.bench_function("real/value", |b| {
        let mut out = [V::ZERO; N];
        b.iter(|| {
            for &(x, y, z) in &dirs {
                V::spherical_harmonics::<L, N, false>(black_box(x), y, z, &mut out);
                black_box(&out);
            }
        })
    });

    group.bench_function("real/value+gradients", |b| {
        let mut out = [V::ZERO; N];
        let (mut dx, mut dy, mut dz) = ([V::ZERO; N], [V::ZERO; N], [V::ZERO; N]);
        b.iter(|| {
            for &(x, y, z) in &dirs {
                V::spherical_harmonics_d::<L, N, false>(black_box(x), y, z, &mut out, &mut dx, &mut dy, &mut dz);
                black_box(&out);
            }
        })
    });

    // The hoisted-table path, for callers sweeping many directions.
    group.bench_function("real/value-with-table", |b| {
        let mut table = ShTable::<V, N>::zeroed();
        V::spherical_harmonics_table::<L, N, false>(&mut table);
        let mut out = [V::ZERO; N];
        b.iter(|| {
            for &(x, y, z) in &dirs {
                V::spherical_harmonics_with::<L, N>(&table, black_box(x), y, z, &mut out);
                black_box(&out);
            }
        })
    });

    // --- Dual, one type, three seedings ---

    type D = Dual<V, 3>;

    group.bench_function("dual3/constant", |b| {
        let mut out = [D::constant(V::ZERO); N];
        b.iter(|| {
            for &(x, y, z) in &dirs {
                D::spherical_harmonics::<L, N, false>(
                    D::constant(black_box(x)),
                    D::constant(y),
                    D::constant(z),
                    &mut out,
                );
                black_box(&out);
            }
        })
    });

    group.bench_function("dual3/identity", |b| {
        let mut out = [D::constant(V::ZERO); N];
        b.iter(|| {
            for &(x, y, z) in &dirs {
                D::spherical_harmonics::<L, N, false>(
                    D::variable(black_box(x), 0),
                    D::variable(y, 1),
                    D::variable(z, 2),
                    &mut out,
                );
                black_box(&out);
            }
        })
    });

    group.bench_function("dual3/general", |b| {
        // A Jacobian that is scaled rather than unit, so the classifier declines and
        // the general dual recurrence runs. Same amount of real information as the
        // identity case, only the route differs.
        let two = V::splat(2.0);
        let col = |v: V, slot: usize| {
            let mut d = [V::ZERO; 3];
            d[slot] = two;
            D::new(v, d)
        };
        let mut out = [D::constant(V::ZERO); N];
        b.iter(|| {
            for &(x, y, z) in &dirs {
                D::spherical_harmonics::<L, N, false>(col(black_box(x), 0), col(y, 1), col(z, 2), &mut out);
                black_box(&out);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
