//! Transcendental math throughput at native width per backend: scalar `std`
//! baseline vs x86-v2 (SSE4.2, 128-bit) vs x86-v3 (AVX2+FMA, 256-bit), with a
//! `Precision`-policy variant on v3 to show the accuracy/speed axis.
//!
//! Each kernel streams a 4096-element slice through the dispatched function
//! and accumulates the results, so the number reported is sustained
//! throughput including loads, not a single-value latency. x86-only; the host
//! must support the forced ISA (SSE4.2 everywhere; AVX2 for v3).
//!
//! ```text
//! cargo bench --bench math
//! ```
#![allow(non_camel_case_types)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
use thermite::math::policy::{DefaultPolicy, policies::Precision};
use thermite::math::{RealMathWithPolicy, TranscendentalMathWithPolicy};
use thermite::prelude::*;
use thermite::simd::Simd;

/// Elements processed per timed iteration.
const N: usize = 4096;

/// Deterministic pseudo-random values in `[lo, hi)`.
fn make<E: From<f32>>(lo: f32, hi: f32) -> Vec<E> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = (s >> 40) as f32 / (1u32 << 24) as f32;
            E::from(lo + unit * (hi - lo))
        })
        .collect()
}

/// One module of kernels per (backend, vector type, element, policy) config.
/// Plain `while` loops and unaligned loads; the math calls are the workload.
macro_rules! math_kernels {
    ($mod:ident, $tf:literal, $V:ty, $E:ty, $P:ty) => {
        mod $mod {
            use super::*;

            pub type V = $V;

            #[target_feature(enable = $tf)]
            pub unsafe fn exp(data: &[$E]) -> V {
                let mut acc = V::ZERO;
                let mut i = 0;
                while i + V::lanes() <= data.len() {
                    let v = unsafe { V::load_unaligned(data.as_ptr().add(i)) };
                    acc += v.exp_p::<$P>();
                    i += V::lanes();
                }
                acc
            }

            #[target_feature(enable = $tf)]
            pub unsafe fn ln(data: &[$E]) -> V {
                let mut acc = V::ZERO;
                let mut i = 0;
                while i + V::lanes() <= data.len() {
                    let v = unsafe { V::load_unaligned(data.as_ptr().add(i)) };
                    acc += v.ln_p::<$P>();
                    i += V::lanes();
                }
                acc
            }

            #[target_feature(enable = $tf)]
            pub unsafe fn tanh(data: &[$E]) -> V {
                let mut acc = V::ZERO;
                let mut i = 0;
                while i + V::lanes() <= data.len() {
                    let v = unsafe { V::load_unaligned(data.as_ptr().add(i)) };
                    acc += v.tanh_p::<$P>();
                    i += V::lanes();
                }
                acc
            }

            #[target_feature(enable = $tf)]
            pub unsafe fn sin_cos(data: &[$E]) -> (V, V) {
                let mut s_acc = V::ZERO;
                let mut c_acc = V::ZERO;
                let mut i = 0;
                while i + V::lanes() <= data.len() {
                    let v = unsafe { V::load_unaligned(data.as_ptr().add(i)) };
                    let (s, c) = v.sin_cos_p::<$P>();
                    s_acc += s;
                    c_acc += c;
                    i += V::lanes();
                }
                (s_acc, c_acc)
            }

            #[target_feature(enable = $tf)]
            pub unsafe fn atan2(y: &[$E], x: &[$E]) -> V {
                let mut acc = V::ZERO;
                let mut i = 0;
                while i + V::lanes() <= y.len() {
                    let vy = unsafe { V::load_unaligned(y.as_ptr().add(i)) };
                    let vx = unsafe { V::load_unaligned(x.as_ptr().add(i)) };
                    acc += vy.atan2_p::<$P>(vx);
                    i += V::lanes();
                }
                acc
            }
        }
    };
}

// Native width per backend: 128-bit on v2, 256-bit on v3.
math_kernels!(v2_f32, "sse4.2", Vector<<X86V2 as Simd>::f32x4>, f32, DefaultPolicy);
math_kernels!(v3_f32, "avx2,fma", Vector<<X86V3 as Simd>::f32x8>, f32, DefaultPolicy);
math_kernels!(v3_f32_precise, "avx2,fma", Vector<<X86V3 as Simd>::f32x8>, f32, Precision);
math_kernels!(v2_f64, "sse4.2", Vector<<X86V2 as Simd>::f64x2>, f64, DefaultPolicy);
math_kernels!(v3_f64, "avx2,fma", Vector<<X86V3 as Simd>::f64x4>, f64, DefaultPolicy);
math_kernels!(v3_f64_precise, "avx2,fma", Vector<<X86V3 as Simd>::f64x4>, f64, Precision);

/// Scalar `std` baseline (autovectorization is welcome to try).
macro_rules! scalar_kernels {
    ($mod:ident, $E:ty) => {
        mod $mod {
            pub fn exp(data: &[$E]) -> $E {
                data.iter().fold(0.0, |a, &x| a + x.exp())
            }
            pub fn ln(data: &[$E]) -> $E {
                data.iter().fold(0.0, |a, &x| a + x.ln())
            }
            pub fn tanh(data: &[$E]) -> $E {
                data.iter().fold(0.0, |a, &x| a + x.tanh())
            }
            pub fn sin_cos(data: &[$E]) -> ($E, $E) {
                data.iter().fold((0.0, 0.0), |(s, c), &x| {
                    let (xs, xc) = x.sin_cos();
                    (s + xs, c + xc)
                })
            }
            pub fn atan2(y: &[$E], x: &[$E]) -> $E {
                y.iter().zip(x).fold(0.0, |a, (&y, &x)| a + y.atan2(x))
            }
        }
    };
}

scalar_kernels!(scalar_f32, f32);
scalar_kernels!(scalar_f64, f64);

/// One criterion group comparing scalar / v2 / v3 / v3-precise for one method.
macro_rules! op_group {
    ($c:expr, $group:expr, $scalar:ident, $v2:ident, $v3:ident, $v3p:ident, $method:ident ( $($data:expr),+ )) => {{
        let mut g = $c.benchmark_group($group);
        g.throughput(Throughput::Elements(N as u64));
        g.bench_function("scalar", |b| {
            b.iter(|| black_box($scalar::$method($(black_box($data)),+)))
        });
        g.bench_function("v2", |b| {
            b.iter(|| unsafe { black_box($v2::$method($(black_box($data)),+)) })
        });
        g.bench_function("v3", |b| {
            b.iter(|| unsafe { black_box($v3::$method($(black_box($data)),+)) })
        });
        g.bench_function("v3-precise", |b| {
            b.iter(|| unsafe { black_box($v3p::$method($(black_box($data)),+)) })
        });
        g.finish();
    }};
}

/// All op groups for one element type.
macro_rules! element {
    ($c:expr, $E:ty, $tag:literal, $scalar:ident, $v2:ident, $v3:ident, $v3p:ident) => {{
        // Input ranges chosen to stay in each function's primary path:
        // sin_cos over one period, exp/tanh moderate, ln strictly positive.
        let trig: Vec<$E> = make(-3.14159265, 3.14159265);
        let mid: Vec<$E> = make(-4.0, 4.0);
        let pos: Vec<$E> = make(0.001, 8.0);

        op_group!($c, concat!("exp/", $tag), $scalar, $v2, $v3, $v3p, exp(&mid));
        op_group!($c, concat!("ln/", $tag), $scalar, $v2, $v3, $v3p, ln(&pos));
        op_group!($c, concat!("tanh/", $tag), $scalar, $v2, $v3, $v3p, tanh(&mid));
        op_group!($c, concat!("sin_cos/", $tag), $scalar, $v2, $v3, $v3p, sin_cos(&trig));
        op_group!($c, concat!("atan2/", $tag), $scalar, $v2, $v3, $v3p, atan2(&mid, &trig));
    }};
}

fn bench(c: &mut Criterion) {
    element!(c, f32, "f32", scalar_f32, v2_f32, v3_f32, v3_f32_precise);
    element!(c, f64, "f64", scalar_f64, v2_f64, v3_f64, v3_f64_precise);
}

criterion_group!(benches, bench);
criterion_main!(benches);
