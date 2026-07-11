//! Cross-backend `f32x16` zeroing `compress`: the per-lane scatter
//! (`compress_permute_wide`) vs the loop-free chunked-cursor
//! (`compress_z_wide`), forced onto x86-v1 (SSE2), v2 (SSE4.2), and v3
//! (AVX2+FMA), plus the two-level merge on v3 for reference.
//!
//! The register methods are `#[inline(always)]`, so each kernel's
//! `#[target_feature]` gives real per-backend codegen. x86-only; the host must
//! support the forced ISA (every modern x86-64 has SSE4.2; AVX2 for v3).
//!
//! ```text
//! cargo bench --bench compress
//! ```
#![allow(non_camel_case_types)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use generic_array::{GenericArray, sequence::GenericSequence};

use thermite::backend::generic::polyfills::{compress_permute_wide, compress_z_merge2, compress_z_wide};
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
use thermite::register::{BitwiseRegister, CoreRegister, Register, Storage};
use thermite::simd::Simd;

/// Inputs processed per timed iteration.
const N: usize = 1024;

/// Deterministic pseudo-random 16-lane values + random-density masks for any
/// backend's `f32x16`.
fn make<RW: Register<Element = f32>>() -> Vec<(Storage<RW>, Storage<<RW as CoreRegister>::Mask>)> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    (0..N)
        .map(|_| {
            let vals: GenericArray<f32, RW::Lanes> =
                GenericArray::generate(|_| (next() as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0);
            let v = RW::new(vals);
            let sel: GenericArray<f32, RW::Lanes> = GenericArray::generate(|_| (next() & 1) as f32);
            let m = RW::into_mask(RW::new(sel));
            (v, m)
        })
        .collect()
}

macro_rules! backend {
    ($mod:ident, $simd:ty, $tf:literal) => {
        mod $mod {
            use super::*;

            pub type RW = <$simd as Simd>::f32x16;
            pub type Inp = (Storage<RW>, Storage<<RW as CoreRegister>::Mask>);

            // Per-lane scatter index build, then one permute.
            #[target_feature(enable = $tf)]
            pub unsafe fn scatter(inputs: &[Inp]) -> Storage<RW> {
                let mut acc = RW::EMPTY;
                for &(v, m) in inputs {
                    acc = RW::bitxor(acc, compress_permute_wide::<RW>(RW::zz(m, v), m));
                }
                acc
            }

            // Chunked-cursor: per-chunk table store at a running count cursor, one swizzle.
            #[target_feature(enable = $tf)]
            pub unsafe fn chunked(inputs: &[Inp]) -> Storage<RW> {
                let mut acc = RW::EMPTY;
                for &(v, m) in inputs {
                    acc = RW::bitxor(acc, compress_z_wide::<RW>(v, m));
                }
                acc
            }
        }
    };
}

backend!(v1, X86V1, "sse2");
backend!(v2, X86V2, "sse4.2");
backend!(v3, X86V3, "avx2,fma");

/// v3 two-level merge (its `f32x16` is `ArrayRegister<F32x8V3, 2>`, so `.0` is
/// the two native 8-lane halves).
mod v3_merge {
    use super::*;

    type B = <X86V3 as Simd>::f32x8;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn merge(inputs: &[v3::Inp]) -> [Storage<B>; 2] {
        let mut acc = [B::EMPTY; 2];
        for &(v, m) in inputs {
            let r = compress_z_merge2::<B>(v.0, m.0);
            acc[0] = B::bitxor(acc[0], r[0]);
            acc[1] = B::bitxor(acc[1], r[1]);
        }
        acc
    }
}

fn bench(c: &mut Criterion) {
    macro_rules! group {
        ($mod:ident, $label:literal) => {{
            let inputs = make::<$mod::RW>();
            let mut g = c.benchmark_group($label);
            g.throughput(Throughput::Elements(N as u64));
            g.bench_function("scatter", |b| {
                b.iter(|| unsafe { black_box($mod::scatter(black_box(&inputs))) })
            });
            g.bench_function("chunked", |b| {
                b.iter(|| unsafe { black_box($mod::chunked(black_box(&inputs))) })
            });
            g.finish();
        }};
    }

    group!(v1, "f32x16/v1");
    group!(v2, "f32x16/v2");
    group!(v3, "f32x16/v3");

    let inputs = make::<v3::RW>();
    let mut g = c.benchmark_group("f32x16/v3");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("merge", |b| {
        b.iter(|| unsafe { black_box(v3_merge::merge(black_box(&inputs))) })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
