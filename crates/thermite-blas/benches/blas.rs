use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
use std::hint::black_box;

fn naive_zdotc(a: &[f64], b: &[f64]) -> (f64, f64) {
    let a_pairs = a.as_chunks::<2>();
    let b_pairs = b.as_chunks::<2>();

    let mut sum_r = 0.0f64;
    let mut sum_i = 0.0f64;

    for ([ar, ai], [br, bi]) in a_pairs.0.iter().zip(b_pairs.0) {
        // process complex number (ar + i*ai) and (br + i*bi)
        sum_r = ar.mul_add(*br, ai.mul_add(*bi, sum_r));
        sum_i = ar.mul_add(*bi, ai.mul_add(-*br, sum_i));
    }

    (sum_r, sum_i)
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let test_a = (0..(1 << 12)).map(|_| rng.random_range(-4.0..4.0)).collect::<Vec<_>>();
    let test_b = (0..(1 << 12)).map(|_| rng.random_range(-4.0..4.0)).collect::<Vec<_>>();

    let mut zdotc = c.benchmark_group("zdotc");

    zdotc.bench_with_input("naive", &[&test_a, &test_b], |f, &[a, b]| {
        f.iter(|| black_box(naive_zdotc(a, b)));
    });

    zdotc.bench_with_input("x86-v3", &[&test_a, &test_b], |f, &[a, b]| {
        use thermite::backend::x86_v3::X86V3;
        use thermite_blas::BLAS;

        #[target_feature(enable = "avx2,fma")]
        unsafe fn do_zdotc(a: &[f64], b: &[f64]) -> (f64, f64) {
            <X86V3 as BLAS<X86V3>>::zdotc_interleaved(a, b)
        }

        f.iter(|| black_box(unsafe { do_zdotc(a, b) }));
    });

    zdotc.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
