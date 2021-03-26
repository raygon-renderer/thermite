#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};

use thermite::*;
use thermite_special::*;

use thermite::backends::avx2::AVX2;

type Vf32 = <AVX2 as Simd>::Vf32;
type Vf64 = <AVX2 as Simd>::Vf64;
type Vi32 = <AVX2 as Simd>::Vi32;
type Vu64 = <AVX2 as Simd>::Vu64;
type Vu32 = <AVX2 as Simd>::Vu32;
type Vi64 = <AVX2 as Simd>::Vi64;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "sinh",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.sinh()
                }
                let x = black_box(Vf32::splat(*x) + Vf32::indexed());
                b.iter(|| do_algorithm(x))
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.sinh()
            }
            let x = black_box(Vf64::splat(*x as f64) + Vf64::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("scalar-ps", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].sinh();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].sinh();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "exp",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.exp()
                }
                let x = black_box(Vf32::splat(*x) + Vf32::indexed());
                b.iter(|| do_algorithm(x))
            },
            vec![0.5],
        )
        .with_function("thermite-ps-ultra", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf32) -> Vf32 {
                x.exp_p::<policies::UltraPerformance>()
            }
            let x = black_box(Vf32::splat(*x) + Vf32::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.exp()
            }
            let x = black_box(Vf64::splat(*x as f64) + Vf64::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("scalar-ps", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].exp();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].exp();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "ln",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.ln()
                }
                let x = black_box(Vf32::splat(*x) + Vf32::indexed());
                b.iter(|| do_algorithm(x))
            },
            vec![0.5],
        )
        .with_function("thermite-ps-ultra", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf32) -> Vf32 {
                x.ln_p::<policies::UltraPerformance>()
            }
            let x = black_box(Vf32::splat(*x) + Vf32::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.ln()
            }
            let x = black_box(Vf64::splat(*x as f64) + Vf64::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("scalar-ps", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].ln();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].ln();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "cbrt",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.cbrt()
                }
                let x = black_box(Vf32::splat(*x) + Vf32::indexed());
                b.iter(|| do_algorithm(x))
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.cbrt()
            }
            let x = black_box(Vf64::splat(*x as f64) + Vf64::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("scalar-ps", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = x[i].cbrt();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x) + Vf32::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = x[i].cbrt();
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64) + Vf64::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "sin_cos",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> (Vf32, Vf32) {
                    x.sin_cos()
                }
                let x = black_box(Vf32::splat(*x) + Vf32::indexed());
                b.iter(|| do_algorithm(x))
            },
            vec![0.5],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> (Vf64, Vf64) {
                x.sin_cos()
            }
            let x = black_box(Vf64::splat(*x as f64) + Vf64::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("thermite-ps-ultra", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf32) -> (Vf32, Vf32) {
                x.sin_cos_p::<policies::UltraPerformance>()
            }
            let x = black_box(Vf32::splat(*x) + Vf32::indexed());
            b.iter(|| do_algorithm(x))
        })
        .with_function("scalar-ps", |b, x| {
            #[inline(never)]
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
            black_box(Vf32::splat(*x) + Vf32::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("scalar-pd", |b, x| {
            #[inline(never)]
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
            black_box(Vf64::splat(*x as f64) + Vf64::indexed()).store_unaligned(&mut xs);

            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "tgamma",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.tgamma()
                }
                b.iter(|| do_algorithm(Vf32::splat(*x)))
            },
            vec![-25.43, -4.83, 0.53, 20.3, 4.0, 20.0],
        )
        .with_function("thermite-ps-ultra", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf32) -> Vf32 {
                x.tgamma_p::<policies::UltraPerformance>()
            }
            b.iter(|| do_algorithm(Vf32::splat(*x)))
        })
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.tgamma()
            }
            b.iter(|| do_algorithm(Vf64::splat(*x as f64)));
        })
        .with_function("libm-ps", |b, x| {
            #[inline(never)]
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
        })
        .with_function("libm-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = libm::tgamma(x[i]);
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64)).store_unaligned(&mut xs);
            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "lgamma",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.lgamma()
                }
                b.iter(|| do_algorithm(Vf32::splat(*x)))
            },
            vec![-25.43, -4.83, 0.53, 20.3, 4.0, 20.0],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.lgamma()
            }
            b.iter(|| do_algorithm(Vf64::splat(*x as f64)));
        })
        .with_function("libm-ps", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f32; 8]) -> [f32; 8] {
                for i in 0..8 {
                    x[i] = libm::lgammaf(x[i]);
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf32::splat(*x)).store_unaligned(&mut xs);
            b.iter(|| unsafe { do_algorithm(xs) })
        })
        .with_function("libm-pd", |b, x| {
            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut x: [f64; 8]) -> [f64; 8] {
                for i in 0..8 {
                    x[i] = libm::lgamma(x[i]);
                }
                x
            }

            let mut xs = [0.0; 8];
            black_box(Vf64::splat(*x as f64)).store_unaligned(&mut xs);
            b.iter(|| unsafe { do_algorithm(xs) })
        }),
    );

    c.bench(
        "digamma",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32) -> Vf32 {
                    x.digamma()
                }
                b.iter(|| do_algorithm(Vf32::splat(*x)))
            },
            vec![-25.43, -4.83, 0.53, 20.3, 4.0, 20.0],
        )
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64) -> Vf64 {
                x.digamma()
            }
            b.iter(|| do_algorithm(Vf64::splat(*x as f64)));
        }),
    );

    c.bench(
        "beta",
        ParameterizedBenchmark::new(
            "thermite-ps",
            |b, x| {
                #[inline(never)]
                fn do_algorithm(x: Vf32, y: Vf32) -> Vf32 {
                    x.beta_p::<policies::UltraPerformance>(y)
                }
                b.iter(|| do_algorithm(Vf32::splat(x.0), Vf32::splat(x.1)))
            },
            vec![(5.0, 0.5)],
        )
        .with_function("thermite-ps-precision", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf32, y: Vf32) -> Vf32 {
                x.beta_p::<policies::Precision>(y)
            }
            b.iter(|| do_algorithm(Vf32::splat(x.0), Vf32::splat(x.1)))
        })
        .with_function("thermite-pd", |b, x| {
            #[inline(never)]
            fn do_algorithm(x: Vf64, y: Vf64) -> Vf64 {
                x.beta(y)
            }
            b.iter(|| do_algorithm(Vf64::splat(x.0 as f64), Vf64::splat(x.1 as f64)));
        }),
    );

    c.bench(
        "large_poly_eval",
        ParameterizedBenchmark::new(
            "thermite",
            |b, (x, poly)| {
                #[inline(never)]
                fn do_algorithm(x: Vf32, poly: &[f32]) -> Vf32 {
                    x.poly(poly)
                }

                b.iter(move || do_algorithm(Vf32::splat(*x), poly));
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
            #[inline(never)]
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

    c.bench(
        "rng_gen2",
        ParameterizedBenchmark::new(
            "thermite",
            |b, len| {
                use thermite::rng::{xoshiro::Xoshiro128Plus, SimdRng};

                #[inline(never)]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(mut rng: Xoshiro128Plus<AVX2>, dst: &mut Vf64, iterations: usize) {
                    for _ in 0..iterations {
                        *dst = (*dst + rng.next_f64()) * Vf64::splat(0.5);
                    }
                }

                let rng: Xoshiro128Plus<AVX2> = Xoshiro128Plus::<AVX2>::new(Vu64::indexed());

                b.iter_with_setup(
                    || rng.clone(),
                    |rng| unsafe {
                        let mut out = Vf64::zero();
                        do_algorithm(rng, &mut out, *len);
                        out
                    },
                );
            },
            vec![1 << 8, 1 << 10, 1 << 12, 1 << 14],
        )
        .with_function("rand", |b, len| {
            use rand::{Rng, SeedableRng};
            use rand_xoshiro::Xoshiro128Plus;

            let rngs = unsafe {
                let mut rngs = std::mem::MaybeUninit::<[Xoshiro128Plus; 8]>::uninit();

                for i in 0..8 {
                    (rngs.as_mut_ptr() as *mut Xoshiro128Plus)
                        .add(i)
                        .write(Xoshiro128Plus::seed_from_u64(i as u64));
                }

                rngs.assume_init()
            };

            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut rngs: [Xoshiro128Plus; 8], buf: &mut [f64; 8], iterations: usize) {
                for _ in 0..iterations {
                    for (dst, rng) in buf.iter_mut().zip(rngs.iter_mut()) {
                        *dst = (*dst + rng.gen::<f64>()) * 0.5;
                    }
                }
            }

            b.iter_with_setup(
                || rngs.clone(),
                |rngs| unsafe {
                    let mut out = [0.0; 8];
                    do_algorithm(rngs, &mut out, *len);
                    out
                },
            );
        }),
    );

    c.bench(
        "rng_gen",
        ParameterizedBenchmark::new(
            "thermite",
            |b, len| {
                use thermite::rng::{xoshiro::Xoshiro128Plus, SimdRng};

                #[inline(never)]
                #[target_feature(enable = "avx2,fma")]
                unsafe fn do_algorithm(mut rng: Xoshiro128Plus<AVX2>, buf: &mut [f64]) {
                    for chunk in buf.chunks_exact_mut(Vf64::NUM_ELEMENTS) {
                        rng.next_f64().store_unaligned_unchecked(chunk.as_mut_ptr());
                    }
                }

                let rng: Xoshiro128Plus<AVX2> = Xoshiro128Plus::<AVX2>::new(Vu64::indexed());

                b.iter_with_large_setup(
                    move || (vec![0.0; *len], rng.clone()),
                    |(mut buf, rng)| unsafe {
                        do_algorithm(rng, &mut buf);
                        return buf;
                    },
                )
            },
            vec![1usize << 8, 1 << 10, 1 << 12, 1 << 14],
        )
        .with_function("rand", |b, len| {
            use rand::{Rng, SeedableRng};
            use rand_xoshiro::Xoshiro128Plus;

            #[inline(never)]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn do_algorithm(mut rngs: [Xoshiro128Plus; 8], buf: &mut [f64]) {
                for chunk in buf.chunks_exact_mut(8) {
                    for (res, rng) in chunk.iter_mut().zip(rngs.iter_mut()) {
                        *res = rng.gen();
                    }
                }
            }

            let rngs = unsafe {
                let mut rngs = std::mem::MaybeUninit::<[Xoshiro128Plus; 8]>::uninit();

                for i in 0..8 {
                    (rngs.as_mut_ptr() as *mut Xoshiro128Plus)
                        .add(i)
                        .write(Xoshiro128Plus::seed_from_u64(i as u64));
                }

                rngs.assume_init()
            };

            b.iter_with_large_setup(
                || (vec![0.0; *len], rngs.clone()),
                |(mut buf, rngs)| unsafe {
                    do_algorithm(rngs, &mut buf);
                    return buf;
                },
            )
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
