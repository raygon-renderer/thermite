//! Langevin function and inverse against the forms a path guider would otherwise
//! write by hand:
//!
//! - forward: `1/tanh(x) - 1/x` and `coth` via `exp` (both cancel below `x ~ 1`, see the
//!   kernel docs, so they are faster-or-slower _and_ wrong there. The point is the
//!   cost of doing it right).
//! - inverse: the Banerjee/Cohen Pade `y(3-y^2)/(1-y^2)` alone (5% error, what vMF
//!   fitting code usually ships), and Sra 2012 (Cohen + two Newton steps with the
//!   identity `L' = 1 - L^2 - 2L/x`, `L` from `1/tanh`).
//!
//! Same inputs, one process, so the ratios hold even though absolute numbers drift.
//! Inputs cover the whole range in the mix a vMF workload sees: `kappa` from 0.05 to
//! 200 (log-uniform) for the forward, `r` in `(0.05, 0.9995)` for the inverse.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::element::FloatElement;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, UltraPerformance};
use thermite::prelude::*;
use thermite_special::{RealPrimalMathWithPolicy, RealSpecialMathWithPolicy};

type V32 = Vector<<X86V3 as Simd>::f32x8>;
type V64 = Vector<<X86V3 as Simd>::f64x4>;

const COUNT: usize = 256;

/// `count` vectors whose lanes are all distinct draws of `g(t)`, `t` low-discrepancy in
/// `[0, 1)`. Distinct lanes matter: with splatted inputs whole vectors fall on one side
/// of the kernel's `x = 2` crossover and the `is_small.all()` fast path skips the exp,
/// which made the branch-free `Worst` tier look slower than `Average` (it computes both
/// branches unconditionally). Real workloads have mixed lanes, so this does too.
fn inputs<V: FloatVector>(count: usize, g: impl Fn(f64) -> V::Element) -> Vec<V>
where
    V::Element: Copy + Default,
{
    (0..count)
        .map(|i| {
            let mut lanes = vec![V::Element::default(); V::LANES];
            for (k, lane) in lanes.iter_mut().enumerate() {
                let t = (((i * V::LANES + k) as f64) * 0.618033988749895).fract();
                *lane = g(t);
            }
            V::from_slice(&lanes)
        })
        .collect()
}

/// kappa log-uniform in [0.05, 200].
fn kappas<V: FloatVector>(count: usize, f: impl Fn(f64) -> V::Element) -> Vec<V>
where
    V::Element: Copy + Default,
{
    inputs::<V>(count, |t| f(0.05 * (4000.0f64).powf(t)))
}

/// r in (0.05, 0.9995).
fn rs<V: FloatVector>(count: usize, f: impl Fn(f64) -> V::Element) -> Vec<V>
where
    V::Element: Copy + Default,
{
    inputs::<V>(count, |t| f(0.05 + 0.9495 * t))
}

/// One dispatched kernel per measured form: the naive arithmetic has to sit inside a
/// `#[dispatch]` body too, or it compiles featureless and the comparison is bogus.
macro_rules! kernel {
    ($name:ident, |$x:ident: $V:ident| $body:expr) => {
        #[thermite::dispatch($V)]
        fn $name<$V>(xs: &[$V]) -> $V
        where
            $V: FloatVector + thermite::math::TranscendentalMath + RealSpecialMathWithPolicy + RealPrimalMathWithPolicy,
        {
            let mut acc = $V::ZERO;
            for &$x in xs {
                acc = acc + $body;
            }
            acc
        }
    };
}

// --- naive forms, written the way a caller would ---

#[inline(always)]
fn naive_tanh<V: FloatVector + thermite::math::TranscendentalMath>(x: V) -> V {
    V::ONE / x.tanh() - V::ONE / x
}

#[inline(always)]
fn naive_exp<V: FloatVector + thermite::math::TranscendentalMath>(x: V) -> V {
    let e = (x + x).exp();
    (e + V::ONE) / (e - V::ONE) - V::ONE / x
}

#[inline(always)]
fn cohen<V: FloatVector>(y: V) -> V {
    let s = y * y;
    y * (V::splat(<V::Element as FloatElement>::ConstInt::<3>::VALUE) - s) / (V::ONE - s)
}

/// Sra 2012: Cohen seed + 2 Newton steps, `L` via `1/tanh`.
#[inline(always)]
fn sra<V: FloatVector + thermite::math::TranscendentalMath>(y: V) -> V {
    let mut x = cohen(y);
    for _ in 0..2 {
        let l = naive_tanh(x);
        let dl = V::ONE - l * l - (l + l) / x;
        x -= (l - y) / dl;
    }
    x
}

kernel!(k_naive_tanh, |x: V| naive_tanh(x));
kernel!(k_naive_exp, |x: V| naive_exp(x));
kernel!(k_worst, |x: V| x.langevin_p::<UltraPerformance>());
kernel!(k_medium, |x: V| x.langevin_p::<HighPerformance>());
kernel!(k_average, |x: V| x.langevin_p::<Performance>());
kernel!(k_best, |x: V| x.langevin_p::<Precision>());
kernel!(k_best_d, |x: V| {
    let (l, d) = x.langevin_d_p::<Precision>();
    l + d
});
kernel!(k_cohen, |y: V| cohen(y));
kernel!(k_sra, |y: V| sra(y));
kernel!(k_inv_worst, |y: V| y.inv_langevin_p::<UltraPerformance>());
kernel!(k_inv_medium, |y: V| y.inv_langevin_p::<HighPerformance>());
kernel!(k_inv_average, |y: V| y.inv_langevin_p::<Performance>());
kernel!(k_inv_best, |y: V| y.inv_langevin_p::<Precision>());
// The composition the vMF convolution runs, kappa' = L^-1(L(k1) L(k2)). `xs` holds
// k1 and the kernel pairs each with its mirror as k2.
kernel!(k_conv_naive, |x: V| cohen(naive_tanh(x) * naive_tanh(V::ONE + x)));
kernel!(k_conv_average, |x: V| (x.langevin_p::<Performance>()
    * (V::ONE + x).langevin_p::<Performance>())
.inv_langevin_p::<Performance>());
kernel!(k_conv_best, |x: V| (x.langevin_p::<Precision>()
    * (V::ONE + x).langevin_p::<Precision>())
.inv_langevin_p::<Precision>());
// The complement chain: t = a + b - ab with a, b = 1 - L, then L^-1(1 - t). Same work
// as above but exact at any sharpness.
kernel!(k_conv_1m, |x: V| {
    let a = x.langevin_1m_p::<Precision>();
    let b = (V::ONE + x).langevin_1m_p::<Precision>();
    a.nmul_adde(b, a + b).inv_langevin_1m_p::<Precision>()
});
kernel!(k_1m_best, |x: V| x.langevin_1m_p::<Precision>());
kernel!(k_inv_1m_best, |t: V| t.inv_langevin_1m_p::<Precision>());

macro_rules! run {
    ($group:expr, $name:literal, $inputs:expr, $k:ident) => {
        $group.bench_function($name, |b| b.iter(|| black_box($k(black_box(&$inputs)))));
    };
}

fn bench(c: &mut Criterion) {
    let k32 = kappas::<V32>(COUNT, |v| v as f32);
    let k64 = kappas::<V64>(COUNT, |v| v);
    let r32 = rs::<V32>(COUNT, |v| v as f32);
    let r64 = rs::<V64>(COUNT, |v| v);

    let mut g = c.benchmark_group("langevin/f32x8");
    g.throughput(criterion::Throughput::Elements((COUNT * V32::LANES) as u64));
    run!(g, "naive 1/tanh-1/x", k32, k_naive_tanh);
    run!(g, "naive coth(exp)-1/x", k32, k_naive_exp);
    run!(g, "langevin/Worst", k32, k_worst);
    run!(g, "langevin/Medium", k32, k_medium);
    run!(g, "langevin/Average", k32, k_average);
    run!(g, "langevin/Best", k32, k_best);
    run!(g, "langevin_d/Best", k32, k_best_d);
    run!(g, "langevin_1m/Best", k32, k_1m_best);
    g.finish();

    let mut g = c.benchmark_group("langevin/f64x4");
    g.throughput(criterion::Throughput::Elements((COUNT * V64::LANES) as u64));
    run!(g, "naive 1/tanh-1/x", k64, k_naive_tanh);
    run!(g, "naive coth(exp)-1/x", k64, k_naive_exp);
    run!(g, "langevin/Worst", k64, k_worst);
    run!(g, "langevin/Average", k64, k_average);
    run!(g, "langevin/Best", k64, k_best);
    g.finish();

    let mut g = c.benchmark_group("inv_langevin/f32x8");
    g.throughput(criterion::Throughput::Elements((COUNT * V32::LANES) as u64));
    run!(g, "cohen/banerjee (5%)", r32, k_cohen);
    run!(g, "sra 2012 (cohen+2 newton, tanh)", r32, k_sra);
    run!(g, "inv_langevin/Worst (seed only)", r32, k_inv_worst);
    run!(g, "inv_langevin/Medium", r32, k_inv_medium);
    run!(g, "inv_langevin/Average", r32, k_inv_average);
    run!(g, "inv_langevin/Best", r32, k_inv_best);
    run!(g, "inv_langevin_1m/Best", r32, k_inv_1m_best);
    g.finish();

    let mut g = c.benchmark_group("inv_langevin/f64x4");
    g.throughput(criterion::Throughput::Elements((COUNT * V64::LANES) as u64));
    run!(g, "cohen/banerjee (5%)", r64, k_cohen);
    run!(g, "sra 2012 (cohen+2 newton, tanh)", r64, k_sra);
    run!(g, "inv_langevin/Worst (seed only)", r64, k_inv_worst);
    run!(g, "inv_langevin/Medium", r64, k_inv_medium);
    run!(g, "inv_langevin/Best", r64, k_inv_best);
    g.finish();

    let mut g = c.benchmark_group("vmf_convolve/f32x8");
    g.throughput(criterion::Throughput::Elements((COUNT * V32::LANES) as u64));
    run!(g, "naive (tanh + cohen)", k32, k_conv_naive);
    run!(g, "thermite/Average", k32, k_conv_average);
    run!(g, "thermite/Best", k32, k_conv_best);
    run!(g, "thermite/Best complement chain", k32, k_conv_1m);
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
