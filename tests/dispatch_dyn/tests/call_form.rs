//! Tests for the call form of `dispatch_dyn!`.
//!
//! `dispatch_dyn!(func(args))` runtime-dispatches a call to a `#[dispatch]` function
//! by injecting the selected backend as the callee's sole generic argument.
//! `dispatch_dyn!(for<S> expr)` substitutes `S` in an arbitrary expression instead,
//! for callees with extra generics or non-trivial call shapes.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;
use thermite::{dispatch, dispatch_dyn};

/// A representative `#[dispatch]` kernel: sum of squares over a slice.
#[dispatch(S)]
fn sum_squares<S: Simd>(data: &[f32]) -> f32 {
    let (head, mid, tail) = data.try_aligned_simd_iter::<Vector<S::f32xN>>();
    let mut acc = Vector::<S::f32xN>::ZERO;
    for v in mid {
        acc = v.mul_adde(*v, acc);
    }
    let mut s = acc.sum_elements();
    for x in head.iter().chain(tail.iter()) {
        s += x * x;
    }
    s
}

/// A `#[dispatch]` kernel with an extra generic parameter after the SIMD one,
/// so the bare call form cannot be used (partial turbofish is not allowed).
#[dispatch(S)]
fn splat_first<S: Simd, T: Copy + Default>(data: &[T]) -> T {
    data.first().copied().unwrap_or_default()
}

mod kernels {
    use super::*;

    #[dispatch(S)]
    pub fn double_sum<S: Simd>(data: &[f32]) -> f32 {
        super::sum_squares::<S>(data) * 0.0 + data.iter().sum::<f32>() * 2.0
    }
}

#[test]
fn bare_call_matches_scalar_backend() {
    let data: Vec<f32> = (0..117).map(|i| i as f32 * 0.25).collect();
    let expected = sum_squares::<Scalar>(&data);

    let r = dispatch_dyn!(sum_squares(&data));
    assert!((r - expected).abs() < expected * 1e-5, "{r} vs {expected}");
}

#[test]
fn bare_call_with_multi_segment_path() {
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let expected: f32 = data.iter().sum::<f32>() * 2.0;

    let r = dispatch_dyn!(kernels::double_sum(&data));
    assert_eq!(r, expected);
}

#[test]
fn bare_call_args_are_arbitrary_expressions() {
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let expected = sum_squares::<Scalar>(&data[10..20]);

    // Argument expressions are used verbatim in the selected arm.
    let r = dispatch_dyn!(sum_squares(&data[10..20]));
    assert!((r - expected).abs() < 1e-3, "{r} vs {expected}");
}

#[test]
fn for_binder_with_explicit_turbofish() {
    let data: Vec<f32> = (0..50).map(|i| i as f32 * 0.5).collect();
    let expected = sum_squares::<Scalar>(&data);

    let r = dispatch_dyn!(for<S> sum_squares::<S>(&data));
    assert!((r - expected).abs() < expected * 1e-5, "{r} vs {expected}");
}

#[test]
fn for_binder_with_extra_generic() {
    let data = [7i64, 8, 9];
    let r = dispatch_dyn!(for<S> splat_first::<S, i64>(&data));
    assert_eq!(r, 7);
}

#[test]
fn for_binder_with_custom_ident() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let expected = sum_squares::<Scalar>(&data);

    let r = dispatch_dyn!(for<Backend> sum_squares::<Backend>(&data));
    assert!((r - expected).abs() < 1e-3, "{r} vs {expected}");
}

/// A `#[dispatch]` impl block whose methods are generic over the SIMD backend,
/// so the *method's* `S` (not `Self`) is what the call form dispatches on.
struct Weighted {
    scale: f32,
}

#[dispatch(S)]
impl Weighted {
    fn sum<S: Simd>(&self, data: &[f32]) -> f32 {
        let (head, mid, tail) = data.try_aligned_simd_iter::<Vector<S::f32xN>>();
        let mut acc = Vector::<S::f32xN>::ZERO;
        for v in mid {
            acc += *v;
        }
        let mut s = acc.sum_elements();
        for x in head.iter().chain(tail.iter()) {
            s += x;
        }
        s * self.scale
    }
}

#[test]
fn for_binder_with_method_receiver() {
    let k = Weighted { scale: 2.0 };
    let data: Vec<f32> = (0..77).map(|i| i as f32 * 0.5).collect();
    let expected = k.sum::<Scalar>(&data);

    let r = dispatch_dyn!(for<S> k.sum::<S>(&data));
    assert!((r - expected).abs() < expected * 1e-5, "{r} vs {expected}");
}
