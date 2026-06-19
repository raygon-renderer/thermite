//! Regression tests for the auto-reborrow behavior at outermost call sites.
//!
//! `dispatch_dyn!(|data: &[f32]| { ... })` should accept any caller value whose
//! deref chain reaches `&[f32]` — `Vec<f32>`, `Box<[f32]>`, `&Vec<f32>`, an
//! actual `&[f32]`, etc.  The macro emits `&*ident` at outermost call sites,
//! which both reborrows references and triggers `Deref` coercion for owners.

use thermite::dispatch_dyn;

fn sum_slice(s: &[f32]) -> f32 {
    s.iter().copied().sum()
}

#[test]
fn accepts_vec_for_slice_param() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let r = dispatch_dyn!(|data: &[f32]| -> f32 { sum_slice(data) });
    assert_eq!(r, 10.0);
}

#[test]
fn accepts_box_slice_for_slice_param() {
    let data: Box<[f32]> = vec![1.0f32, 2.0, 3.0].into_boxed_slice();
    let r = dispatch_dyn!(|data: &[f32]| -> f32 { sum_slice(data) });
    assert_eq!(r, 6.0);
}

#[test]
fn accepts_actual_slice_for_slice_param() {
    let owned = vec![5.0f32, 5.0];
    let data: &[f32] = &owned;
    let r = dispatch_dyn!(|data: &[f32]| -> f32 { sum_slice(data) });
    assert_eq!(r, 10.0);
}

#[test]
fn accepts_ref_to_vec_for_slice_param() {
    let owned: Vec<f32> = vec![7.0, 8.0];
    let data: &Vec<f32> = &owned;
    let r = dispatch_dyn!(|data: &[f32]| -> f32 { sum_slice(data) });
    assert_eq!(r, 15.0);
}

#[test]
fn accepts_vec_for_mut_slice_param() {
    let mut data: Vec<f32> = vec![1.0, 2.0, 3.0];
    dispatch_dyn!(|data: &mut [f32]| {
        for x in data.iter_mut() {
            *x *= 2.0;
        }
    });
    assert_eq!(data, vec![2.0, 4.0, 6.0]);
}

#[test]
fn accepts_box_slice_for_mut_slice_param() {
    let mut data: Box<[f32]> = vec![1.0f32, 2.0].into_boxed_slice();
    dispatch_dyn!(|data: &mut [f32]| {
        for x in data.iter_mut() {
            *x += 10.0;
        }
    });
    assert_eq!(&*data, &[11.0, 12.0]);
}

#[test]
fn does_not_consume_caller_value() {
    // After dispatch_dyn! returns, the caller's `Vec` must still be usable —
    // the macro must borrow, not move.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let r = dispatch_dyn!(|data: &[f32]| -> f32 { sum_slice(data) });
    assert_eq!(r, 6.0);
    // `data` is still ours:
    assert_eq!(data.len(), 3);
    drop(data);
}

// For a `&mut [T]` slice parameter shape, the caller can hold a `&mut Vec`
// even as a non-`mut`-bound fn parameter — the macro's `&mut *ident` invokes
// `DerefMut` on the Vec to reach the `[T]` target.
fn caller_passes_vec_to_slice_param(buf: &mut Vec<f64>) {
    dispatch_dyn!(|buf: &mut [f64]| {
        for x in buf.iter_mut() {
            *x *= 3.0;
        }
    });
}

#[test]
fn mut_ref_vec_passed_as_mut_slice_through_fn_param() {
    let mut owned: Vec<f64> = vec![1.0, 2.0, 3.0];
    caller_passes_vec_to_slice_param(&mut owned);
    assert_eq!(owned, vec![3.0, 6.0, 9.0]);
}

// For a sized `&mut Vec<T>` parameter, the macro emits plain `&mut ident`
// (no deref — that would overshoot to `&mut [T]`). The caller must hold an
// owned, mut-bound Vec OR a mut-bound reference. Function parameters need
// `mut` on the binding (`fn foo(mut buf: &mut Vec<f64>)`).
#[test]
fn owned_vec_to_mut_vec_ref_param() {
    let mut buf: Vec<f64> = Vec::new();
    dispatch_dyn!(|buf: &mut Vec<f64>| {
        buf.push(1.0);
        buf.push(2.0);
    });
    assert_eq!(buf, vec![1.0, 2.0]);
}

fn caller_with_mut_bound_vec_ref(mut buf: &mut Vec<f64>) {
    dispatch_dyn!(|buf: &mut Vec<f64>| {
        buf.push(7.0);
    });
    let _ = &mut buf;
}

#[test]
fn mut_bound_vec_ref_param_from_fn() {
    let mut owned: Vec<f64> = Vec::new();
    caller_with_mut_bound_vec_ref(&mut owned);
    assert_eq!(owned, vec![7.0]);
}

#[test]
fn does_not_consume_caller_value_mut() {
    let mut data: Vec<f32> = vec![1.0, 2.0];
    dispatch_dyn!(|data: &mut [f32]| {
        data[0] = 9.0;
    });
    // Still ours, still mutable.
    data.push(3.0);
    assert_eq!(data, vec![9.0, 2.0, 3.0]);
}

#[test]
fn owned_param_is_moved_not_reborrowed() {
    // Non-reference param types should still receive the value by move.
    fn consume(v: Vec<f32>) -> usize {
        v.len()
    }
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let n = dispatch_dyn!(|data: Vec<f32>| -> usize { consume(data) });
    assert_eq!(n, 4);
    // `data` is moved — referencing it here would be a compile error, which is
    // the correct behavior. We only assert the move took effect (n is 4).
}

#[test]
fn scalar_copy_param_passes_by_value() {
    let x: i32 = 21;
    let r = dispatch_dyn!(|x: i32| -> i32 { x * 2 });
    assert_eq!(r, 42);
    // Copy type — still usable.
    let _ = x;
}
