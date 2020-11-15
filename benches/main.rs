#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};

use thermite::*;

type Vf32 = <thermite::backends::AVX2 as Simd>::Vf32;
type Vf64 = <thermite::backends::AVX2 as Simd>::Vf64;
type Vi32 = <thermite::backends::AVX2 as Simd>::Vi32;
type Vu64 = <thermite::backends::AVX2 as Simd>::Vu64;
type Vu32 = <thermite::backends::AVX2 as Simd>::Vu32;
type Vi64 = <thermite::backends::AVX2 as Simd>::Vi64;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "exp",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                let x = black_box(Vf32::splat(*x) + Vf32::index());
                b.iter(|| x.exp())
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            let x = black_box(Vf64::splat(*x as f64) + Vf64::index());
            b.iter(|| x.exp())
        })
        .with_function("scalar-ps", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].exp();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].exp();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "ln",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                let x = black_box(Vf32::splat(*x) + Vf32::index());
                b.iter(|| x.ln())
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            let x = black_box(Vf64::splat(*x as f64) + Vf64::index());
            b.iter(|| x.ln())
        })
        .with_function("scalar-ps", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].ln();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].ln();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "cbrt",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                let x = black_box(Vf32::splat(*x) + Vf32::index());
                b.iter(|| x.cbrt())
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            let x = black_box(Vf64::splat(*x as f64) + Vf64::index());
            b.iter(|| x.cbrt())
        })
        .with_function("scalar-ps", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].cbrt();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].cbrt();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "sin_cos",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                let x = black_box(Vf32::splat(*x) + Vf32::index()0;
                b.iter(|| x.sin_cos())
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            let x = black_box(Vf64::splat(*x as f64) + Vf64::index()0;
            b.iter(|| x.sin_cos())
        })
        .with_function("scalar-ps", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: [f32; 8]) -> ([f32; 8], [f32; 8]) {
                let mut s = [0.0; 8];
                let mut c = [0.0; 8];
                for i in 0..8 {
                    let (ss, sc) = x[i].sin_cos();
                    s[i] = ss;
                    c[i] = sc;
                }
                (s, c)
            }
            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: [f64; 8]) -> ([f64; 8], [f64; 8]) {
                let mut s = [0.0; 8];
                let mut c = [0.0; 8];
                for i in 0..8 {
                    let (ss, sc) = x[i].sin_cos();
                    s[i] = ss;
                    c[i] = sc;
                }
                (s, c)
            }
            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "tgamma",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| b.iter(|| Vf32::splat(*x).tgamma()),
            vec![-25.43, -4.83, 0.53, 20.3, 4.0, 20.0],
        )
        .with_function("libm", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = libm::tgammaf(x[i]);
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x)).store_unaligned(&mut xs);
            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "large_poly_eval",
        ParameterizedBenchmark::new(
            "thermite",
            |b, (x, poly)| {
                b.iter(move || Vf32::splat(*x).poly(poly));
            },
            vec![(
                1.001,
                (0..10004) // +4 to trigger cleanup iteration
                    .into_iter()
                    .map(|x| (x as f32).sqrt().sin())
                    .collect::<Vec<f32>>(),
            )],
        )
        .with_function("horners", |b, (x, poly)| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: Vf32, poly: &[f32]) -> Vf32 {
                let mut res = Vf32::one();
                for coeff in poly.iter().rev() {
                    res = res.mul_add(x, Vf32::splat(*coeff));
                }
                res
            }
            b.iter(move || unsafe { do_algorithm(Vf32::splat(*x), poly) })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
