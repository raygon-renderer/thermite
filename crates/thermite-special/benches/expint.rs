//! `expint` across the regimes its two algorithms cover, so the cost of the
//! correctness fix is on the record rather than assumed.
//!
//! What changed: the forward recurrence used to run until it had amplified a seed ulp by the
//! whole mantissa - cheap, and wrong by 100% for any order above 5 in the middle of the range
//! - and then handed off to an asymptotic series that could not converge where it was being
//! asked to. Both are now bounded by `AMP_CAP = 64` ulps of amplification, past which a
//! continued fraction takes over.
//!
//! The three groups below separate what that costs:
//!
//! - **small**: `x` under every threshold. Byte-identical code to before (rational `E_1` plus
//!   a few FMA recurrence steps), and the case radiative transfer actually runs at.
//! - **crossover**: `x` straddling the threshold, so lanes disagree about which branch to
//!   take and both run. The worst case for a packet, and the honest one.
//! - **large**: `x` well above the threshold, all lanes in the fraction, where it converges
//!   in 6 to 10 iterations rather than 26.
//!
//! Same inputs, one process, so the ratios hold even though absolute numbers drift.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::math::policy::policies::{HighPerformance, Performance, UltraPerformance};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

type V64 = Vector<<X86V3 as Simd>::f64x4>;

const COUNT: usize = 256;

/// `count` vectors with distinct lanes drawn from `[lo, hi)`, low-discrepancy so the
/// crossover group really does land on both sides of the threshold rather than clustering.
fn inputs(count: usize, lo: f64, hi: f64) -> Vec<V64> {
    (0..count)
        .map(|i| {
            let mut lanes = [0.0f64; 4];
            for (k, lane) in lanes.iter_mut().enumerate() {
                let t = (((i * 4 + k) as f64) * 0.6180339887498949).fract();
                *lane = lo + (hi - lo) * t;
            }
            V64::new(lanes)
        })
        .collect()
}

macro_rules! sweep {
    ($c:expr, $name:literal, $lo:expr, $hi:expr) => {{
        let xs = inputs(COUNT, $lo, $hi);
        let mut g = $c.benchmark_group($name);

        g.bench_function("E2/perf", |b| {
            b.iter(|| {
                let mut acc = V64::ZERO;
                for &x in black_box(&xs) {
                    acc += x.expint_p::<Performance, 2>();
                }
                acc
            })
        });
        g.bench_function("E3/perf", |b| {
            b.iter(|| {
                let mut acc = V64::ZERO;
                for &x in black_box(&xs) {
                    acc += x.expint_p::<Performance, 3>();
                }
                acc
            })
        });
        g.bench_function("E8/perf", |b| {
            b.iter(|| {
                let mut acc = V64::ZERO;
                for &x in black_box(&xs) {
                    acc += x.expint_p::<Performance, 8>();
                }
                acc
            })
        });
        // The precision-tier ladder, on the order where the fraction costs most.
        g.bench_function("E8/high", |b| {
            b.iter(|| {
                let mut acc = V64::ZERO;
                for &x in black_box(&xs) {
                    acc += x.expint_p::<HighPerformance, 8>();
                }
                acc
            })
        });
        g.bench_function("E8/ultra", |b| {
            b.iter(|| {
                let mut acc = V64::ZERO;
                for &x in black_box(&xs) {
                    acc += x.expint_p::<UltraPerformance, 8>();
                }
                acc
            })
        });
        g.finish();
    }};
}

fn bench(c: &mut Criterion) {
    // Under every threshold (N=8 bottoms out near 6.1), so the recurrence path only.
    sweep!(c, "expint/small", 0.05, 5.0);

    // Straddling: E_3 hands over at 11.3, E_8 at 6.1, E_2 not until 64. Mixed lanes, mixed
    // branches, both evaluated.
    sweep!(c, "expint/crossover", 4.0, 20.0);

    // All lanes in the fraction for E_3 and E_8; E_2 is still on the recurrence below 64,
    // which is why it stays flat across all three groups.
    sweep!(c, "expint/large", 25.0, 90.0);
}

criterion_group!(benches, bench);
criterion_main!(benches);
