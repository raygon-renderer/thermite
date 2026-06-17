//! Ergonomic forward-mode AD entry point.
//!
//! [`AutoDiff`] seeds a function's arguments as independent dual variables and
//! evaluates it in one call, so you don't have to spell out
//! `Dual::variable(x, 0)`, `Dual::variable(y, 1)`, ... by hand.

use crate::{Dual, DualValue};

/// Evaluate a function of `N` arguments under forward-mode automatic differentiation.
///
/// Implemented for every function or closure taking `N` arguments of type
/// `Dual<V, N>` (for `N` in `1..=12`). [`ad`](AutoDiff::ad) takes the `N` raw inputs,
/// seeds input `i` as the `i`-th independent variable (`Dual::variable(inputs[i], i)`),
/// evaluates the function, and returns its result -- whose dual parts are the gradient
/// with respect to the inputs.
///
/// The inputs are an array `[V; N]`, which fixes both the value type `V` and the arity
/// `N` at the call site, so the matching impl (and the function's required signature) is
/// inferred with no turbofish.
///
/// # Example
///
/// ```
/// use thermite_dual::{AutoDiff, Dual};
///
/// fn f(x: Dual<f64, 3>, y: Dual<f64, 3>, z: Dual<f64, 3>) -> Dual<f64, 3> {
///     x * x + y * z
/// }
///
/// let r = f.ad([2.0, 3.0, 4.0]);
/// assert_eq!(r.re, 16.0); // 2^2 + 3*4
/// assert_eq!(r.dual, [4.0, 4.0, 3.0]); // [df/dx, df/dy, df/dz] = [2x, z, y]
/// ```
///
/// The result type is whatever the function returns: a single `Dual<V, N>` for a
/// scalar-valued function, or e.g. a tuple/array of duals for a vector-valued function
/// (each carrying one row of the Jacobian).
///
/// # Generic functions
///
/// Functions written generically against the Thermite trait bounds (the intended style)
/// work directly: the input array fixes `V` (hence the dual element `Dual<V, N>`), and the
/// function's type parameter is inferred to that -- no turbofish or closure wrapper.
///
/// ```
/// # use thermite::prelude::*;
/// use thermite::math::TranscendentalMath;
/// use thermite_dual::AutoDiff;
///
/// // Generic over any float vector with transcendental math, in two variables.
/// fn f<W: FloatVector + TranscendentalMath>(x: W, y: W) -> W {
///     (x * y).exp() + x
/// }
///
/// // Instantiated at W = Dual<Vector<f64>, 2> via AutoDiff; the dual parts are the gradient.
/// let r = f.ad([Vector::<f64>::splat(0.5), Vector::<f64>::splat(2.0)]);
///
/// let (x, y) = (0.5_f64, 2.0_f64);
/// let e = (x * y).exp();
/// assert!((r.re.extract::<0>()      - (e + x)).abs()       < 1e-12); // f(x, y) = e^{xy} + x
/// assert!((r.dual[0].extract::<0>() - (y * e + 1.0)).abs() < 1e-12); // df/dx = y e^{xy} + 1
/// assert!((r.dual[1].extract::<0>() - (x * e)).abs()       < 1e-12); // df/dy = x e^{xy}
/// ```
///
/// This requires the dual element to satisfy the function's bounds. For the math traits the
/// inner `V` must be a real [`FloatVector`](thermite::prelude::FloatVector) (e.g.
/// `Vector<f64>`), since `Dual<Vector<R>, N>` implements `FloatVector` + the math traits;
/// a scalar inner like `Dual<f64, N>` only supports arithmetic.
pub trait AutoDiff<V: DualValue, const N: usize>: Sized {
    /// The wrapped function's return type.
    type Output;

    /// Seed each input as an independent variable and evaluate the function.
    fn ad(self, inputs: [V; N]) -> Self::Output;
}

/// Expands to its trailing tokens, discarding the leading index token. Lets a
/// repetition over argument indices emit a fixed type fragment per argument.
macro_rules! ad_subst {
    ($idx:tt, $($keep:tt)*) => { $($keep)* };
}

macro_rules! impl_autodiff {
    ($n:literal; $($idx:tt),+) => {
        impl<V: DualValue, F, R> AutoDiff<V, $n> for F
        where
            F: FnOnce($( ad_subst!($idx, Dual<V, $n>) ),+) -> R,
        {
            type Output = R;

            #[inline]
            fn ad(self, inputs: [V; $n]) -> R {
                self($( Dual::<V, $n>::variable(inputs[$idx], $idx) ),+)
            }
        }
    };
}

impl_autodiff!(1; 0);
impl_autodiff!(2; 0, 1);
impl_autodiff!(3; 0, 1, 2);
impl_autodiff!(4; 0, 1, 2, 3);
impl_autodiff!(5; 0, 1, 2, 3, 4);
impl_autodiff!(6; 0, 1, 2, 3, 4, 5);
impl_autodiff!(7; 0, 1, 2, 3, 4, 5, 6);
impl_autodiff!(8; 0, 1, 2, 3, 4, 5, 6, 7);
impl_autodiff!(9; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_autodiff!(10; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_autodiff!(11; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_autodiff!(12; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
