//! Sanity-check: results from `dispatch_dyn!` on actual SIMD workloads match
//! a straightforward scalar reference. Whichever backend is selected at runtime,
//! the numbers must come out right.
//!
//! Note: `dispatch_dyn!` arg/return types cannot mention the dispatch generic
//! `S`, because the per-backend trampolines aren't generic over it. Restrict
//! signatures to concrete scalar/slice types; do all SIMD work inside the body.

use thermite::dispatch_dyn;

#[test]
fn add_two_slices_native_width() {
    let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| (2 * i) as f32).collect();
    let mut out: Vec<f32> = vec![0.0; 256];

    dispatch_dyn!(for<S> |a: &[f32], b: &[f32], out: &mut [f32]| {
        // Driving from `out` keeps slice indices consistent across all three Vecs,
        // which may not share alignment.
        let n = out.len();
        let lanes = <f32xN as GenericVector>::LANES;
        let chunks = n / lanes;
        for c in 0..chunks {
            let base = c * lanes;
            let av = unsafe { <f32xN>::load_unaligned(a.as_ptr().add(base)) };
            let bv = unsafe { <f32xN>::load_unaligned(b.as_ptr().add(base)) };
            let sv = av + bv;
            unsafe { sv.store_unaligned(out.as_mut_ptr().add(base)) };
        }
        for i in (chunks * lanes)..n {
            out[i] = a[i] + b[i];
        }
    });

    for i in 0..256 {
        assert_eq!(out[i], a[i] + b[i], "mismatch at {i}");
    }
}

#[test]
fn sum_native_width_matches_scalar() {
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let scalar_sum: f32 = data.iter().copied().sum();

    let total = dispatch_dyn!(for<S> |data: &[f32]| -> f32 {
        let (head, mid, tail) = data.try_aligned_simd_iter::<f32xN>();
        let mut acc = f32xN::ZERO;
        for c in mid {
            acc += *c;
        }
        let mut s = acc.sum_elements();
        for x in head.iter().chain(tail.iter()) {
            s += x;
        }
        s
    });

    assert!((total - scalar_sum).abs() < 1e-1, "{total} vs {scalar_sum}");
}

#[test]
fn passes_owned_vec_through_to_simd_body() {
    // Combines the reborrow fix (caller has Vec<f32>, body takes &[f32])
    // with real SIMD work.
    let owned: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let expected: f32 = owned.iter().map(|x| x * x).sum();

    let r = dispatch_dyn!(for<S> |owned: &[f32]| -> f32 {
        let (head, mid, tail) = owned.try_aligned_simd_iter::<f32xN>();
        let mut acc = f32xN::ZERO;
        for c in mid {
            acc += *c * *c;
        }
        let mut s = acc.sum_elements();
        for x in head.iter().chain(tail.iter()) {
            s += x * x;
        }
        s
    });

    assert!((r - expected).abs() < 1.0, "{r} vs {expected}");
}
