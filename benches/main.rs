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
                #[inline]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(x: Vf32) -> Vf32 {
                    x.exp()
                }

                let x = Vf32::splat(*x) + Vf32::index();
                b.iter(|| unsafe { do_algorithm(x) })
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: Vf64) -> Vf64 {
                x.exp()
            }

            let x = Vf64::splat(*x as f64) + Vf64::index();
            b.iter(|| unsafe { do_algorithm(x) })
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
            (Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

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
            (Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "ln",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(x: Vf32) -> Vf32 {
                    x.ln()
                }

                let x = Vf32::splat(*x) + Vf32::index();
                b.iter(|| unsafe { do_algorithm(x) })
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: Vf64) -> Vf64 {
                x.ln()
            }

            let x = Vf64::splat(*x as f64) + Vf64::index();
            b.iter(|| unsafe { do_algorithm(x) })
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
            (Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

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
            (Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "cbrt",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(x: Vf32) -> Vf32 {
                    x.cbrt()
                }

                let x = Vf32::splat(*x) + Vf32::index();
                b.iter(|| unsafe { do_algorithm(x) })
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: Vf64) -> Vf64 {
                x.cbrt()
            }

            let x = Vf64::splat(*x as f64) + Vf64::index();
            b.iter(|| unsafe { do_algorithm(x) })
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
            (Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

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
            (Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "sin_cos",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(x: Vf32) -> (Vf32, Vf32) {
                    x.sin_cos()
                }

                let x = Vf32::splat(*x) + Vf32::index();
                b.iter(|| unsafe { do_algorithm(x) })
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(x: Vf64) -> (Vf64, Vf64) {
                x.sin_cos()
            }

            let x = Vf64::splat(*x as f64) + Vf64::index();
            b.iter(|| unsafe { do_algorithm(x) })
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
            (Vf32::splat(*x) + Vf32::index()).store_unaligned(&mut xs);

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
            (Vf64::splat(*x as f64) + Vf64::index()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "large_poly_eval",
        ParameterizedBenchmark::new(
            "thermite",
            |b, (x, poly)| {
                #[inline]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(x: Vf32, poly: &[f32]) -> Vf32 {
                    x.poly(poly)
                }

                let x = Vf32::splat(*x);
                b.iter(move || unsafe { do_algorithm(x, poly) });
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

            let x = Vf32::splat(*x);
            b.iter(move || unsafe { do_algorithm(x, poly) })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
