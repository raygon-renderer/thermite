#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Multidual numbers for forward-mode automatic differentiation
//!
//! A [`Dual<V, N>`] carries a primal value plus `N` first-order derivative
//! components (the "dual" parts). Arithmetic propagates derivatives via the
//! usual rules of differentiation, so evaluating a function on a `Dual` yields
//! both the value and its gradient with respect to the `N` seeded directions in
//! a single pass.
//!
//! ```text
//! Dual<V, 0>  =>  just a value (no derivatives tracked)
//! Dual<V, 1>  =>  value + one derivative direction (a classic dual number)
//! Dual<V, N>  =>  value + N partials (a gradient of an N-variable function)
//! ```
//!
//! The inner type `V` is any Thermite [`FloatVector`] (so each lane is an
//! independent dual number, SIMD-parallel) -- or, at the element level, an
//! `f32`/`f64`. The derivative components are stored as a separate `[V; N]`
//! (struct-of-arrays).
//!
//! This is a *first-order multidual* ("hyperdual" in some libraries, though that
//! name properly denotes the second-order algebra). It tracks gradients, not
//! Hessians.
//!
//! [`FloatVector`]: thermite::prelude::FloatVector

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use thermite::vector::ops::{MulAddAssignExt, MulAddExt, Square};

pub mod ad;
pub mod math;
pub mod vector;

#[cfg(feature = "special")]
pub mod special;

pub use ad::AutoDiff;
pub use vector::DualFloatVector;

/// A value usable as the primal/derivative storage of a [`Dual`].
///
/// Implemented for the scalar float elements (`f32`, `f64`) and for every
/// Thermite float [`Vector`](thermite::prelude::Vector). This lets the core
/// arithmetic be written once and reused both at the element level
/// (`Dual<f32, N>`, the [`Element`](thermite::element::Element) of a dual
/// vector) and at the vector level (`Dual<Vector<R>, N>`).
///
/// The scalar element impls only provide arithmetic,
/// the transcendental math library requires a real
/// [`FloatVector`](thermite::prelude::FloatVector) inner type.
pub trait DualValue:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + MulAddExt<Self, Self, Output = Self>
{
    /// The additive identity in this value type.
    const VAL_ZERO: Self;
    /// The multiplicative identity in this value type.
    const VAL_ONE: Self;

    /// Truncate towards zero. Used to give [`Dual`] a (locally-correct) `Rem`.
    fn val_trunc(self) -> Self;
}

impl DualValue for f32 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl DualValue for f64 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl<R: thermite::register::FloatRegister> DualValue for thermite::prelude::Vector<R> {
    const VAL_ZERO: Self = <Self as thermite::prelude::NumericVector>::ZERO;
    const VAL_ONE: Self = <Self as thermite::prelude::NumericVector>::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::prelude::FloatVector::trunc(self)
    }
}

/// A multidual number: a primal value plus `N` first-order derivative parts.
///
/// See the [crate docs](crate) for the high-level idea. The derivative parts are
/// indexed `0..N` and correspond to the `N` independent directions that were
/// seeded into the computation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Dual<V, const N: usize> {
    /// The primal value (the "real" part).
    pub re: V,
    /// The `N` first-order derivative components.
    pub dual: [V; N],
}

impl<V: DualValue, const N: usize> Default for Dual<V, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<V: DualValue, const N: usize> Dual<V, N> {
    /// A dual whose value and all derivatives are zero.
    pub const ZERO: Self = Self {
        re: V::VAL_ZERO,
        dual: [V::VAL_ZERO; N],
    };

    /// A dual with value one and all-zero derivatives (a constant `1`).
    pub const ONE: Self = Self {
        re: V::VAL_ONE,
        dual: [V::VAL_ZERO; N],
    };

    /// Create a dual from a primal value with all derivatives zero.
    ///
    /// Use this for *constants* in a differentiated computation -- values that do
    /// not depend on any seeded variable.
    #[inline(always)]
    pub const fn constant(re: V) -> Self {
        Self {
            re,
            dual: [V::VAL_ZERO; N],
        }
    }

    /// Create a dual from a primal value and its full derivative vector.
    #[inline(always)]
    pub const fn new(re: V, dual: [V; N]) -> Self {
        Self { re, dual }
    }

    /// Seed an independent variable: value `re`, with the `i`th derivative set to
    /// one and the rest zero.
    ///
    /// This is how you introduce the `i`th input of an `N`-variable function so
    /// that the result's `i`th dual part is the partial derivative with respect
    /// to it.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline(always)]
    pub fn variable(re: V, i: usize) -> Self {
        assert!(i < N, "dual variable index {i} out of range for N = {N}");
        let mut dual = [V::VAL_ZERO; N];
        dual[i] = V::VAL_ONE;
        Self { re, dual }
    }

    /// The primal value.
    #[inline(always)]
    pub const fn value(self) -> V {
        self.re
    }

    /// The `N` derivative components.
    #[inline(always)]
    pub const fn gradient(self) -> [V; N] {
        self.dual
    }

    /// Apply the chain rule for a unary function `f`: given the new primal
    /// `f(re)` and the scalar derivative `factor = f'(re)`, propagate the
    /// derivative parts as `factor * dual[i]`.
    #[inline(always)]
    pub fn chain(self, new_re: V, factor: V) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = factor * dual[i];
            i += 1;
        }
        Self { re: new_re, dual }
    }
}

// --- Arithmetic: Dual op Dual ---

impl<V: DualValue, const N: usize> Neg for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = -dual[i];
            i += 1;
        }
        Self { re: -self.re, dual }
    }
}

impl<V: DualValue, const N: usize> Add for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = dual[i] + rhs.dual[i];
            i += 1;
        }
        Self {
            re: self.re + rhs.re,
            dual,
        }
    }
}

impl<V: DualValue, const N: usize> Sub for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = dual[i] - rhs.dual[i];
            i += 1;
        }
        Self {
            re: self.re - rhs.re,
            dual,
        }
    }
}

impl<V: DualValue, const N: usize> Mul for Dual<V, N> {
    type Output = Self;

    // product rule: (a*b)' = a'b + ab'
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            // a'b + ab'  =  fma(a, b', a'*b)
            dual[i] = self.re.mul_adde(rhs.dual[i], self.dual[i] * rhs.re);
            i += 1;
        }
        Self {
            re: self.re * rhs.re,
            dual,
        }
    }
}

impl<V: DualValue, const N: usize> Div for Dual<V, N> {
    type Output = Self;

    // quotient rule: (a/b)' = (a' - (a/b) b') / b
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let q = self.re / rhs.re;
        // reciprocal of the denominator computed once: N divisions -> 1 div + N muls
        let inv = V::VAL_ONE / rhs.re;
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            // (a' - q*b') / b  =  fnma(q, b', a') * (1/b)
            dual[i] = q.nmul_adde(rhs.dual[i], self.dual[i]) * inv;
            i += 1;
        }
        Self { re: q, dual }
    }
}

// --- Arithmetic: Dual op scalar value (constant, no derivative) ---

impl<V: DualValue, const N: usize> Add<V> for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: V) -> Self {
        Self {
            re: self.re + rhs,
            dual: self.dual,
        }
    }
}

impl<V: DualValue, const N: usize> Sub<V> for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: V) -> Self {
        Self {
            re: self.re - rhs,
            dual: self.dual,
        }
    }
}

impl<V: DualValue, const N: usize> Mul<V> for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: V) -> Self {
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = dual[i] * rhs;
            i += 1;
        }
        Self {
            re: self.re * rhs,
            dual,
        }
    }
}

impl<V: DualValue, const N: usize> Div<V> for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: V) -> Self {
        // single reciprocal, then multiply through
        let inv = V::VAL_ONE / rhs;
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = dual[i] * inv;
            i += 1;
        }
        Self {
            re: self.re / rhs,
            dual,
        }
    }
}

// --- Assignment variants ---

impl<V: DualValue, const N: usize, T> AddAssign<T> for Dual<V, N>
where
    Self: Add<T, Output = Self>,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<V: DualValue, const N: usize, T> SubAssign<T> for Dual<V, N>
where
    Self: Sub<T, Output = Self>,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<V: DualValue, const N: usize, T> MulAssign<T> for Dual<V, N>
where
    Self: Mul<T, Output = Self>,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<V: DualValue, const N: usize, T> DivAssign<T> for Dual<V, N>
where
    Self: Div<T, Output = Self>,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

// --- Remainder ---
//
// `x % y = x - trunc(x/y) * y`. Treating the integer quotient `k = trunc(x/y)`
// as locally constant gives the correct one-sided derivatives away from the
// jump points: d/dx (x % y) = 1, d/dy (x % y) = -k.

impl<V: DualValue, const N: usize> Rem for Dual<V, N> {
    type Output = Self;

    // x - k*y with k = trunc(x/y) constant. Single fused pass over the components
    // instead of `self - rhs * k` (a scalar-mul pass followed by a subtract pass).
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let k = (self.re / rhs.re).val_trunc();
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = k.nmul_adde(rhs.dual[i], self.dual[i]); // self.dual - k*rhs.dual
            i += 1;
        }
        Self {
            re: k.nmul_adde(rhs.re, self.re), // self.re - k*rhs.re
            dual,
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<V: DualValue, const N: usize> Rem<V> for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: V) -> Self {
        let k = (self.re / rhs).val_trunc();
        Self {
            re: k.nmul_adde(rhs, self.re), // self.re - k*rhs; dual unchanged (derivative 1)
            dual: self.dual,
        }
    }
}

impl<V: DualValue, const N: usize, T> RemAssign<T> for Dual<V, N>
where
    Self: Rem<T, Output = Self>,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

// --- Fused multiply-add ---
//
// Each variant computes `self*a (+/-) b` on the primal and every derivative part
// using the inner value type's fused multiply-add directly, rather than
// composing the dual `Mul`/`Add` (which would round twice per component). The
// derivative of `self*a` is `self.re*a' + self.dual*a.re` by the product rule,
// so each part folds into two nested FMAs.
//
// There is no "true" hardware FMA for a multidual (each derivative part rounds
// independently), so `HAS_TRUE_FMA` is false; the `_e` variants use the inner
// estimating FMA while the exact variants use the inner exact FMA.

macro_rules! dual_fma {
    ($($name:ident => $re_op:ident, $outer:ident, $inner:ident);* $(;)?) => {
        $(
            #[inline(always)]
            fn $name(self, a: Self, b: Self) -> Self {
                let re = self.re.$re_op(a.re, b.re);
                let mut dual = self.dual;
                let mut i = 0;
                while i < N {
                    dual[i] = self.re.$outer(a.dual[i], self.dual[i].$inner(a.re, b.dual[i]));
                    i += 1;
                }
                Self { re, dual }
            }
        )*
    };
}

#[rustfmt::skip]
impl<V: DualValue, const N: usize> MulAddExt<Self, Self> for Dual<V, N> {
    type Output = Self;

    const HAS_TRUE_FMA: bool = false;

    dual_fma! {
        mul_add   => mul_add,   mul_add,   mul_add;
        mul_sub   => mul_sub,   mul_add,   mul_sub;
        nmul_add  => nmul_add,  nmul_add,  nmul_add;
        nmul_sub  => nmul_sub,  nmul_sub,  mul_add;
        mul_adde  => mul_adde,  mul_adde,  mul_adde;
        mul_sube  => mul_sube,  mul_adde,  mul_sube;
        nmul_adde => nmul_adde, nmul_adde, nmul_adde;
        nmul_sube => nmul_sube, nmul_sube, mul_adde;
    }
}

#[rustfmt::skip]
impl<V: DualValue, const N: usize, A, B> MulAddAssignExt<A, B> for Dual<V, N>
where
    Self: MulAddExt<A, B, Output = Self>,
{
    #[inline(always)] fn mul_add_assign(&mut self, a: A, b: B) { *self = self.mul_add(a, b); }
    #[inline(always)] fn mul_sub_assign(&mut self, a: A, b: B) { *self = self.mul_sub(a, b); }
    #[inline(always)] fn nmul_add_assign(&mut self, a: A, b: B) { *self = self.nmul_add(a, b); }
    #[inline(always)] fn nmul_sub_assign(&mut self, a: A, b: B) { *self = self.nmul_sub(a, b); }
    #[inline(always)] fn mul_adde_assign(&mut self, a: A, b: B) { *self = self.mul_adde(a, b); }
    #[inline(always)] fn mul_sube_assign(&mut self, a: A, b: B) { *self = self.mul_sube(a, b); }
    #[inline(always)] fn nmul_adde_assign(&mut self, a: A, b: B) { *self = self.nmul_adde(a, b); }
    #[inline(always)] fn nmul_sube_assign(&mut self, a: A, b: B) { *self = self.nmul_sube(a, b); }
}

impl<V: DualValue, const N: usize> Square for Dual<V, N> {
    type Output = Self;

    // (x^2)' = 2 x x'. Cheaper than the general product rule `self * self`: one add + N muls
    // instead of N (mul, fma) pairs, and no dependence on a splatted intermediate.
    #[inline(always)]
    fn square(self) -> Self {
        let two_re = self.re + self.re;
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = two_re * dual[i];
            i += 1;
        }
        Self {
            re: self.re * self.re,
            dual,
        }
    }
}

// --- Nesting: a Dual can itself be the storage type of another Dual ---

impl<V: DualValue, const N: usize> DualValue for Dual<V, N> {
    const VAL_ZERO: Self = Self::ZERO;
    const VAL_ONE: Self = Self::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        Self {
            re: self.re.val_trunc(),
            dual: self.dual,
        }
    }
}
