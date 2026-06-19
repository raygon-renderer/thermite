//! Tests for extra generic parameters and where-clauses on `dispatch_dyn!`.

use thermite::dispatch_dyn;

#[test]
fn extra_type_generic() {
    fn run<T: Copy + std::ops::Add<Output = T>>(a: T, b: T) -> T {
        dispatch_dyn!(for<S> <T: Copy + std::ops::Add<Output = T>> |a: T, b: T| -> T {
            a + b
        })
    }
    assert_eq!(run::<i32>(3, 4), 7);
    assert_eq!(run::<f64>(1.5, 2.5), 4.0);
}

#[test]
fn const_generic_param() {
    fn make<const N: usize>() -> [u32; N] {
        dispatch_dyn!(for<S> <const N: usize> || -> [u32; N] {
            [0u32; N]
        })
    }
    let a: [u32; 4] = make::<4>();
    assert_eq!(a, [0; 4]);
}

#[test]
fn where_clause_threaded_through() {
    fn copies<T>(x: T) -> T
    where
        T: Copy + Default,
    {
        dispatch_dyn!(for<S> <T> |x: T| -> T where T: Copy + Default {
            let _ = T::default();
            x
        })
    }
    assert_eq!(copies::<i32>(123), 123);
}

#[test]
fn omitted_for_clause_defaults_to_s() {
    // The macro should still work without an explicit `for<…>`.
    let x = 0i32;
    let r = dispatch_dyn!(|x: i32| -> i32 { x + 1 });
    assert_eq!(r, 1);
}
