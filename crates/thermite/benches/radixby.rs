//! `(de)interleave_radix_by` shape sweep: the register arm's chosen path (native
//! square / certified ladder / fallback) against the portable generic route, per
//! `(N, GROUP)` shape, on the AVX2 f32x8 register.
//!
//! This is the measurement behind the certified-ladder routing (see
//! `backend/x86_v3/polyfills/transpose256.rs`): a shape only deserves its ladder
//! arm while it beats the portable engine here. Run scoped:
//!
//! ```text
//! cargo bench -p thermite --bench radixby -- "d_8_2/"
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite::register::CoreRegister;
use thermite::simd::Simd;

type R = <X86V3 as Simd>::f32x8;
type V = Vector<R>;

/// One shape's contenders on a given register type, at native codegen (`avx2,fma`).
macro_rules! shape_on {
    ($c:expr, $tag:literal, $reg:ty, $elem:ty, $N:literal, $G:literal) => {{
        type VV = Vector<$reg>;

        #[target_feature(enable = "avx2,fma")]
        fn run_arm(inputs: [VV; $N]) -> [VV; $N] {
            VV::deinterleave_radix_by::<$N, $G>(inputs)
        }
        #[target_feature(enable = "avx2,fma")]
        fn run_generic(inputs: [VV; $N]) -> [VV; $N] {
            // The portable route the arm would otherwise take (bypasses the register
            // override entirely).
            let mut raw = [<$reg as CoreRegister>::EMPTY; $N];
            let mut k = 0;
            while k < $N {
                raw[k] = inputs[k].0;
                k += 1;
            }
            let out = thermite::backend::generic::polyfills::deinterleave_radix_by_default::<$reg, $N, $G>(raw);
            let mut v = inputs;
            let mut k = 0;
            while k < $N {
                v[k] = Vector(out[k]);
                k += 1;
            }
            v
        }

        let lanes = <VV as GenericVector>::LANES;
        let mut inputs = [VV::default(); $N];
        for (i, inp) in inputs.iter_mut().enumerate() {
            *inp = VV::indexed() + VV::splat((i * lanes) as $elem);
        }

        let mut group = $c.benchmark_group(concat!($tag, "_", stringify!($N), "_", stringify!($G)));
        group.bench_function("arm", |b| {
            // SAFETY: bench host is AVX2.
            b.iter(|| unsafe { run_arm(black_box(inputs)) });
        });
        group.bench_function("generic", |b| {
            // SAFETY: as above.
            b.iter(|| unsafe { run_generic(black_box(inputs)) });
        });
        group.finish();
    }};
}

/// The native f32x8 shapes.
macro_rules! shape {
    ($c:expr, $N:literal, $G:literal) => {
        shape_on!($c, "d", R, f32, $N, $G)
    };
}

/// The emulated `ArrayRegister<F32x8V3, 2>` (f32x16) shapes - measures whether the inner
/// register's natives + certified ladder reach the emulated width through the chunk-chain.
macro_rules! shape16 {
    ($c:expr, $N:literal, $G:literal) => {
        shape_on!($c, "a16", <X86V3 as Simd>::f32x16, f32, $N, $G)
    };
}

fn bench_shapes(c: &mut Criterion) {
    // Certified ladder shapes.
    shape!(c, 8, 2);
    shape!(c, 16, 2);
    shape!(c, 8, 4);
    shape!(c, 16, 4);
    shape!(c, 16, 1);
    shape!(c, 32, 1);
    shape!(c, 4, 4);
    // Native squares (sanity: arm == hand-written optimum).
    shape!(c, 8, 1);
    shape!(c, 4, 2);
    // Uncertified fallbacks (arm == generic; both bars should match).
    shape!(c, 4, 1);
    shape!(c, 32, 2);

    // ArrayRegister<F32x8V3, 2> (f32x16): the chunk-chain should carry the inner
    // register's natives/ladder up. `arm` = chunk-chain -> inner F32x8V3 radix_by;
    // `generic` = the portable engine run at the full 16-lane array width.
    shape16!(c, 8, 2); // inner square (8x8 32-bit via the (4,2)-equivalent native)
    shape16!(c, 4, 2);
    shape16!(c, 8, 1);
    shape16!(c, 16, 1);
    shape16!(c, 4, 4);
    shape16!(c, 2, 8);
}

criterion_group!(benches, bench_shapes);
criterion_main!(benches);
