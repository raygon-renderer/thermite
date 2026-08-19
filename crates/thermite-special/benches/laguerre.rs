//! Orthonormal Laguerre functions: what the seed costs at each kind of weight.
//!
//! `l_n^{(alpha)}` is a cheap `n`-step recurrence over a seed
//! `x^{alpha/2} e^{-x/2} / sqrt(Gamma(alpha+1))`, so at low degree the seed *is* the
//! cost. Three seeds are measured at the same degree: the float form (vector `ln`,
//! `lgamma`, two `exp`s), the integer form under the product cap (`powi`, a scalar
//! factorial, one `exp`) and the integer form above it (falls back to the log seed).
//! `alpha = 0` is the shortcut both forms take.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite_special::SpecialMath;

type V32 = Vector<<X86V3 as Simd>::f32x8>;
type V64 = Vector<<X86V3 as Simd>::f64x4>;

const COUNT: usize = 256;

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

macro_rules! kernel {
    ($name:ident, |$x:ident: $V:ident| $body:expr) => {
        #[thermite::dispatch($V)]
        fn $name<$V>(xs: &[$V]) -> $V
        where
            $V: FloatVector + SpecialMath + thermite::math::TranscendentalMath,
        {
            let mut acc = $V::ZERO;
            for &$x in xs {
                acc += $body;
            }
            acc
        }
    };
}

const N: usize = 4;

kernel!(k_float_a0, |x: V| x.laguerre_function::<N>(V::ZERO));
kernel!(k_float_a3, |x: V| x
    .laguerre_function::<N>(V::splat(V::Element::from_int(3))));
kernel!(k_int_a0, |x: V| x.laguerre_function_i::<N>(0));
kernel!(k_int_a3, |x: V| x.laguerre_function_i::<N>(3));
kernel!(k_int_a20, |x: V| x.laguerre_function_i::<N>(20));
kernel!(k_int_a200, |x: V| x.laguerre_function_i::<N>(200));
// Runtime weight the compiler cannot see through.
kernel!(k_int_rt, |x: V| x.laguerre_function_i::<N>(black_box(3)));
kernel!(k_int_rt0, |x: V| x.laguerre_function_i::<N>(black_box(0)));
kernel!(k_int_rt2, |x: V| x.laguerre_function_i::<N>(black_box(2)));
kernel!(k_int_rt200, |x: V| x.laguerre_function_i::<N>(black_box(200)));
kernel!(k_float_a200, |x: V| x
    .laguerre_function::<N>(V::splat(V::Element::from_int(200))));
kernel!(k_float_rt, |x: V| x
    .laguerre_function::<N>(V::splat(V::Element::from_int(black_box(3)))));
// The Poisson mass itself, against the form everyone writes by hand.
kernel!(k_pois, |x: V| V::splat(V::Element::from_int(20)).poisson_pmf(x));
kernel!(k_pois_naive, |x: V| {
    let k = V::splat(V::Element::from_int(20));
    (k * x.ln() - x - (k + V::ONE).lgamma()).exp()
});

macro_rules! run {
    ($group:expr, $name:literal, $inputs:expr, $k:ident) => {
        $group.bench_function($name, |b| b.iter(|| black_box($k(black_box(&$inputs)))));
    };
}

fn bench(c: &mut Criterion) {
    let x32 = inputs::<V32>(COUNT, |t| (0.5 + 40.0 * t) as f32);
    let x64 = inputs::<V64>(COUNT, |t| 0.5 + 40.0 * t);

    let mut g = c.benchmark_group("laguerre_function/f32x8");
    g.throughput(criterion::Throughput::Elements((COUNT * V32::LANES) as u64));
    run!(g, "float alpha=0", x32, k_float_a0);
    run!(g, "float alpha=3", x32, k_float_a3);
    run!(g, "int alpha=0", x32, k_int_a0);
    run!(g, "int alpha=3", x32, k_int_a3);
    run!(g, "int alpha=3 runtime", x32, k_int_rt);
    run!(g, "int alpha=20", x32, k_int_a20);
    run!(g, "int alpha=200 (log seed)", x32, k_int_a200);
    g.finish();

    let mut g = c.benchmark_group("laguerre_function/f64x4");
    g.throughput(criterion::Throughput::Elements((COUNT * V64::LANES) as u64));
    run!(g, "float alpha=0", x64, k_float_a0);
    run!(g, "float alpha=3", x64, k_float_a3);
    run!(g, "int alpha=0", x64, k_int_a0);
    run!(g, "int alpha=3", x64, k_int_a3);
    run!(g, "int alpha=3 runtime", x64, k_int_rt);
    run!(g, "int alpha=0 runtime", x64, k_int_rt0);
    run!(g, "int alpha=2 runtime", x64, k_int_rt2);
    run!(g, "int alpha=200 runtime", x64, k_int_rt200);
    run!(g, "int alpha=20", x64, k_int_a20);
    run!(g, "int alpha=200 (log seed)", x64, k_int_a200);
    run!(g, "float alpha=200", x64, k_float_a200);
    run!(g, "float alpha=3 runtime", x64, k_float_rt);
    // x in [100, 300]: around the peak of alpha = 200, where the seed's bd0 is a series.
    let peak64 = inputs::<V64>(COUNT, |t| 100.0 + 200.0 * t);
    run!(g, "float alpha=200 near peak", peak64, k_float_a200);
    g.finish();

    let l64 = inputs::<V64>(COUNT, |t| 5.0 + 30.0 * t);
    let mut g = c.benchmark_group("poisson_pmf/f64x4");
    g.throughput(criterion::Throughput::Elements((COUNT * V64::LANES) as u64));
    run!(g, "k=20, lambda in [5, 35]", l64, k_pois);
    run!(g, "naive exp(k ln l - l - lgamma)", l64, k_pois_naive);
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
