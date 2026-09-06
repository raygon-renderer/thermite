//! Compile-checks the lane-type spellings the thermite skill documents in
//! references/slices-and-dispatch.md, so the skill cannot drift from rustc.
//! Not an audit input; `cargo check -p cargo-thermite-audit --example bound_spellings`.

use thermite::element::FloatElementWithBits;
use thermite::prelude::*;
use thermite::register::well_formed::WellFormedFloatElement;

/// Concrete element: name the `SizedSimd` path (bare `S::fxN` is E0221).
#[thermite::dispatch(S)]
fn concrete<S: FloatSimd<f32>>(data: &mut [f32]) {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<V<S>>();
    for v in chunks {
        *v = v.mul_adde(*v, *v);
    }
}

/// Generic element: the same path with the element's bit types spelled out.
#[thermite::dispatch(S)]
fn generic<S, F>(data: &mut [F])
where
    F: WellFormedFloatElement + FloatElementWithBits,
    S: FloatSimd<F>,
{
    type V<S, F> =
        Vector<<S as SizedSimd<F, <F as FloatElementWithBits>::SignedBits, <F as FloatElementWithBits>::Bits>>::fxN>;
    let (_, chunks, _) = data.try_aligned_simd_iter_mut::<V<S, F>>();
    for v in chunks {
        *v = v.mul_adde(*v, *v);
    }
}

fn main() {
    let mut a = vec![1.0f32; 64];
    let mut b = vec![1.0f64; 64];
    thermite::dispatch_dyn!(concrete(&mut a));
    thermite::dispatch_dyn!(for<S> generic::<S, f64>(&mut b));
    println!("{} {}", a[0], b[0]);
}
