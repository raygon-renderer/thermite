//! Basic smoke tests for `dispatch_dyn!` — does it parse, monomorphize, and execute?

#![allow(unused_imports)]

use thermite::dispatch_dyn;
use thermite::prelude::Simd;

#[test]
fn returns_a_constant() {
    let r: i32 = dispatch_dyn!(|| -> i32 { 42 });
    assert_eq!(r, 42);
}

#[test]
fn writes_through_mut_slice() {
    // Reference-typed params reborrow via `&mut *ident`, which requires the caller
    // to hold either a reference or a `DerefMut`-able owner. `Vec<bool>` derefs to
    // `[bool]`, so this works; raw `let mut x = false; |x: &mut bool|` would not.
    let mut flags = vec![false];
    dispatch_dyn!(|flags: &mut [bool]| {
        flags[0] = true;
    });
    assert_eq!(flags, vec![true]);
}

#[test]
fn forwards_simple_scalar_args() {
    let a = 3i32;
    let b = 4i32;
    let r = dispatch_dyn!(|a: i32, b: i32| -> i32 { a + b });
    assert_eq!(r, 7);
}

#[test]
fn explicit_for_binding_is_accepted() {
    let x = 2.0f32;
    let r = dispatch_dyn!(for<S> |x: f32| -> f32 {
        // S is in scope as a Simd type; the body need not actually use it.
        let _ = core::marker::PhantomData::<S>;
        x * 2.0
    });
    assert_eq!(r, 4.0);
}

#[test]
fn explicit_for_binding_with_bound() {
    let x = 2.0f32;
    let r = dispatch_dyn!(for<S: Simd> |x: f32| -> f32 {
        let _ = core::marker::PhantomData::<S>;
        x + 1.0
    });
    assert_eq!(r, 3.0);
}
