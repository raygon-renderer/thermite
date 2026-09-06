//! Positive control for thermite-audit. `bad` runs a real SIMD kernel with no
//! `#[thermite::dispatch]` ancestor; `good` is the same kernel dispatched. The
//! audit must flag `bad` and pass `good`.
//!
//!   cargo thermite-audit -p cargo-thermite-audit --example positive_control
//!
//! must FAIL with findings; that is the pass condition.

use thermite::prelude::*;

#[inline(always)]
fn step<V: FloatVector>(x: V) -> V {
    x.mul_adde(x, x)
}

#[inline(always)]
fn kernel<S: FloatSimd<f32>>(data: &mut [f32]) {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<V<S>>();
    for v in chunks {
        *v = step(*v);
    }
}

/// Deliberately wrong: featureless entry, so every intrinsic goes out of line.
#[inline(never)]
fn bad<S: FloatSimd<f32>>(data: &mut [f32]) {
    kernel::<S>(data)
}

#[thermite::dispatch(S)]
fn good<S: FloatSimd<f32>>(data: &mut [f32]) {
    kernel::<S>(data)
}

fn main() {
    let mut a = vec![1.0f32; 1024];
    let mut b = a.clone();
    // Concrete backend on purpose; this control only runs on AVX2 hosts.
    #[cfg(target_arch = "x86_64")]
    bad::<thermite::backend::x86_v3::X86V3>(&mut a);
    thermite::dispatch_dyn!(good(&mut b));
    println!("{} {}", a[0], b[0]);
}
