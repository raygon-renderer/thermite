//! Verifies the bare-SIMD-type-name rewriter inside `dispatch_dyn!` bodies.
//!
//! The rewriter fires only inside the body block. In param/return-type positions
//! the user must write `Vector<S::f32x4>` (or use the `thermite::simd::f32x4<S>`
//! alias). Inside the body, bare `f32x4` / `f32x4::splat(…)` / `f32x4::ZERO` are
//! rewritten to `Vector<S::f32x4>` / `<Vector<S::f32x4>>::splat(…)` / etc.

use thermite::dispatch_dyn;

#[test]
fn bare_type_in_let_binding() {
    let sum = dispatch_dyn!(for<S> || -> f32 {
        let v: f32x4 = f32x4::splat(2.5);
        v.sum_elements()
    });
    assert_eq!(sum, 10.0);
}

#[test]
fn bare_expression_path_constant() {
    let total = dispatch_dyn!(for<S> || -> f32 {
        let v = f32x4::ZERO;
        v.sum_elements()
    });
    assert_eq!(total, 0.0);
}

#[test]
fn bare_expression_path_assoc_fn_add() {
    let r = dispatch_dyn!(for<S> || -> f32 {
        let v = f32x4::splat(3.0) + f32x4::splat(4.0);
        v.sum_elements()
    });
    assert_eq!(r, 28.0); // 7.0 * 4 lanes
}

#[test]
fn native_width_alias_fxn() {
    // `f32xN` -> Vector<S::f32xN>. Lane count is backend-dependent, so just
    // check we got something positive (each lane is 1.0).
    let total = dispatch_dyn!(for<S> || -> f32 {
        let v = f32xN::splat(1.0);
        v.sum_elements()
    });
    assert!(total >= 1.0);
}

#[test]
fn integer_simd_type_bare() {
    let s = dispatch_dyn!(for<S> || -> i32 {
        let v: i32x4 = i32x4::splat(7);
        v.sum_elements()
    });
    assert_eq!(s, 28);
}

#[test]
fn body_can_mix_bare_and_qualified() {
    // Use bare in one place, qualified in another — both must produce the same type.
    let r = dispatch_dyn!(for<S> || -> f32 {
        use thermite::Vector;
        let a: f32x4 = f32x4::splat(1.0);
        let b: Vector<S::f32x4> = Vector::<S::f32x4>::splat(2.0);
        (a + b).sum_elements()
    });
    assert_eq!(r, 12.0);
}
