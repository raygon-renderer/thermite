#![allow(clippy::excessive_precision)]

pub mod consts;
pub mod policy;

use crate::{
    Vector,
    register::{FloatRegister, Register},
};

pub mod internal;

use internal::MathInternal;
use policy::{DefaultPolicy, Policy};

pub trait MathWithPolicy<R: FloatRegister>: Sized {
    fn poly_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self;
    fn poly_rev_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self;
    fn poly_rational_p<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self;

    fn lerp_p<P: Policy>(self, a: Self, b: Self) -> Self;
    fn reciprocal_p<P: Policy>(self) -> Self;
    fn inverse_sqrt_p<P: Policy>(self) -> Self;
    fn powi_p<P: Policy>(self, e: i32) -> Self;
    fn powiv_p<P: Policy>(self, e: Vector<R::Signed>) -> Self;
    fn hypot_p<P: Policy>(self, other: Self) -> Self;
    fn sincos_p<P: Policy>(self) -> (Self, Self);
    fn sin_p<P: Policy>(self) -> Self;
    fn cos_p<P: Policy>(self) -> Self;
    fn tan_p<P: Policy>(self) -> Self;

    fn sinh_p<P: Policy>(self) -> Self;
    fn cosh_p<P: Policy>(self) -> Self;
    fn tanh_p<P: Policy>(self) -> Self;

    fn asin_p<P: Policy>(self) -> Self;
    fn acos_p<P: Policy>(self) -> Self;
    fn atan_p<P: Policy>(self) -> Self;
    fn atan2_p<P: Policy>(self, x: Self) -> Self;

    fn asinh_p<P: Policy>(self) -> Self;
    fn acosh_p<P: Policy>(self) -> Self;
    fn atanh_p<P: Policy>(self) -> Self;

    fn exp_p<P: Policy>(self) -> Self;
    fn exph_p<P: Policy>(self) -> Self;
    fn exp2_p<P: Policy>(self) -> Self;
    fn exp10_p<P: Policy>(self) -> Self;
    fn exp_m1_p<P: Policy>(self) -> Self;

    fn powf_p<P: Policy>(self, e: Self) -> Self;
    fn cbrt_p<P: Policy>(self) -> Self;

    fn ln_p<P: Policy>(self) -> Self;
    fn ln1p_p<P: Policy>(self) -> Self;
    fn log2_p<P: Policy>(self) -> Self;
    fn log10_p<P: Policy>(self) -> Self;

    fn erf_p<P: Policy>(self) -> Self;
    fn erfc_p<P: Policy>(self) -> Self;
    fn erfinv_p<P: Policy>(self) -> Self;
}

#[rustfmt::skip]
pub trait Math<R: FloatRegister>: MathWithPolicy<R> {
    #[inline(always)] fn poly<const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        self.poly_p::<DefaultPolicy, N>(coeffs)
    }
    #[inline(always)] fn poly_rev<const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        self.poly_rev_p::<DefaultPolicy, N>(coeffs)
    }
    #[inline(always)] fn poly_rational<const N: usize, const D: usize>(
        self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self {
        self.poly_rational_p::<DefaultPolicy, N, D>(numerator, denominator)
    }

    #[inline(always)] fn lerp(self, a: Self, b: Self) -> Self { self.lerp_p::<DefaultPolicy>(a, b) }
    #[inline(always)] fn reciprocal(self) -> Self { self.reciprocal_p::<DefaultPolicy>() }
    #[inline(always)] fn inverse_sqrt(self) -> Self { self.inverse_sqrt_p::<DefaultPolicy>() }
    #[inline(always)] fn powi(self, e: i32) -> Self { self.powi_p::<DefaultPolicy>(e) }
    #[inline(always)] fn powiv(self, e: Vector<R::Signed>) -> Self { self.powiv_p::<DefaultPolicy>(e) }
    #[inline(always)] fn hypot(self, other: Self) -> Self { self.hypot_p::<DefaultPolicy>(other) }
    #[inline(always)] fn sincos(self) -> (Self, Self) { self.sincos_p::<DefaultPolicy>() }
    #[inline(always)] fn sin(self) -> Self { self.sin_p::<DefaultPolicy>() }
    #[inline(always)] fn cos(self) -> Self { self.cos_p::<DefaultPolicy>() }
    #[inline(always)] fn tan(self) -> Self { self.tan_p::<DefaultPolicy>() }
    #[inline(always)] fn sinh(self) -> Self { self.sinh_p::<DefaultPolicy>() }
    #[inline(always)] fn cosh(self) -> Self { self.cosh_p::<DefaultPolicy>() }
    #[inline(always)] fn tanh(self) -> Self { self.tanh_p::<DefaultPolicy>() }
    #[inline(always)] fn asin(self) -> Self { self.asin_p::<DefaultPolicy>() }
    #[inline(always)] fn acos(self) -> Self { self.acos_p::<DefaultPolicy>() }
    #[inline(always)] fn atan(self) -> Self { self.atan_p::<DefaultPolicy>() }
    #[inline(always)] fn atan2(self, x: Self) -> Self { self.atan2_p::<DefaultPolicy>(x) }
    #[inline(always)] fn asinh(self) -> Self { self.asinh_p::<DefaultPolicy>() }
    #[inline(always)] fn acosh(self) -> Self { self.acosh_p::<DefaultPolicy>() }
    #[inline(always)] fn atanh(self) -> Self { self.atanh_p::<DefaultPolicy>() }
    #[inline(always)] fn exp(self) -> Self { self.exp_p::<DefaultPolicy>() }
    #[inline(always)] fn exph(self) -> Self { self.exph_p::<DefaultPolicy>() }
    #[inline(always)] fn exp2(self) -> Self { self.exp2_p::<DefaultPolicy>() }
    #[inline(always)] fn exp10(self) -> Self { self.exp10_p::<DefaultPolicy>() }
    #[inline(always)] fn exp_m1(self) -> Self { self.exp_m1_p::<DefaultPolicy>() }
    #[inline(always)] fn powf(self, e: Self) -> Self { self.powf_p::<DefaultPolicy>(e) }
    #[inline(always)] fn cbrt(self) -> Self { self.cbrt_p::<DefaultPolicy>() }
    #[inline(always)] fn ln(self) -> Self { self.ln_p::<DefaultPolicy>() }
    #[inline(always)] fn ln1p(self) -> Self { self.ln1p_p::<DefaultPolicy>() }
    #[inline(always)] fn log2(self) -> Self { self.log2_p::<DefaultPolicy>() }
    #[inline(always)] fn log10(self) -> Self { self.log10_p::<DefaultPolicy>() }
    #[inline(always)] fn erf(self) -> Self { self.erf_p::<DefaultPolicy>() }
    #[inline(always)] fn erfc(self) -> Self { self.erfc_p::<DefaultPolicy>() }
    #[inline(always)] fn erfinv(self) -> Self { self.erfinv_p::<DefaultPolicy>() }
}

impl<M, R: FloatRegister> Math<R> for M where M: MathWithPolicy<R> {}

impl<E, R: MathInternal<E>> num_traits::Inv for Vector<R>
where
    R: FloatRegister<Element = E>,
{
    type Output = Self;

    #[inline(always)]
    fn inv(self) -> Self {
        self.reciprocal_p::<DefaultPolicy>()
    }
}

#[rustfmt::skip]
impl<E, R: MathInternal<E>> MathWithPolicy<R> for Vector<R>
where
    R: FloatRegister<Element = E>, // redundant, but binds E
{
    #[inline(always)] fn poly_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        R::poly::<P, N>(self, coeffs)
    }
    #[inline(always)] fn poly_rev_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        R::poly_rev::<P, N>(self, coeffs)
    }
    #[inline(always)] fn poly_rational_p<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self {
        R::poly_rational::<P, N, D>(self, numerator, denominator)
    }

    #[inline(always)] fn lerp_p<P: Policy>(self, a: Self, b: Self) -> Self { R::lerp::<P>(self, a, b) }
    #[inline(always)] fn reciprocal_p<P: Policy>(self) -> Self { R::reciprocal::<P>(self) }
    #[inline(always)] fn inverse_sqrt_p<P: Policy>(self) -> Self { R::invsqrt::<P>(self) }
    #[inline(always)] fn powi_p<P: Policy>(self, e: i32) -> Self { R::powi::<P>(self, e) }
    #[inline(always)] fn powiv_p<P: Policy>(self, e: Vector<R::Signed>) -> Self { R::powiv::<P>(self, e) }
    #[inline(always)] fn hypot_p<P: Policy>(self, other: Self) -> Self { R::hypot::<P>(self, other) }
    #[inline(always)] fn sincos_p<P: Policy>(self) -> (Self, Self) { R::sincos::<P>(self) }
    #[inline(always)] fn sin_p<P: Policy>(self) -> Self { R::sin::<P>(self) }
    #[inline(always)] fn cos_p<P: Policy>(self) -> Self { R::cos::<P>(self) }
    #[inline(always)] fn tan_p<P: Policy>(self) -> Self { R::tan::<P>(self) }
    #[inline(always)] fn sinh_p<P: Policy>(self) -> Self { R::sinh::<P>(self) }
    #[inline(always)] fn cosh_p<P: Policy>(self) -> Self { R::cosh::<P>(self) }
    #[inline(always)] fn tanh_p<P: Policy>(self) -> Self { R::tanh::<P>(self) }
    #[inline(always)] fn asin_p<P: Policy>(self) -> Self { R::asin::<P>(self) }
    #[inline(always)] fn acos_p<P: Policy>(self) -> Self { R::acos::<P>(self) }
    #[inline(always)] fn atan_p<P: Policy>(self) -> Self { R::atan::<P>(self) }
    #[inline(always)] fn atan2_p<P: Policy>(self, x: Self) -> Self { R::atan2::<P>(self, x) }
    #[inline(always)] fn asinh_p<P: Policy>(self) -> Self { R::asinh::<P>(self) }
    #[inline(always)] fn acosh_p<P: Policy>(self) -> Self { R::acosh::<P>(self) }
    #[inline(always)] fn atanh_p<P: Policy>(self) -> Self { R::atanh::<P>(self) }
    #[inline(always)] fn exp_p<P: Policy>(self) -> Self { R::exp::<P>(self) }
    #[inline(always)] fn exph_p<P: Policy>(self) -> Self { R::exph::<P>(self) }
    #[inline(always)] fn exp2_p<P: Policy>(self) -> Self { R::exp2::<P>(self) }
    #[inline(always)] fn exp10_p<P: Policy>(self) -> Self { R::exp10::<P>(self) }
    #[inline(always)] fn exp_m1_p<P: Policy>(self) -> Self { R::exp_m1::<P>(self) }
    #[inline(always)] fn powf_p<P: Policy>(self, e: Self) -> Self { R::powf::<P>(self, e) }
    #[inline(always)] fn cbrt_p<P: Policy>(self) -> Self { R::cbrt::<P>(self) }
    #[inline(always)] fn ln_p<P: Policy>(self) -> Self { R::ln::<P>(self) }
    #[inline(always)] fn ln1p_p<P: Policy>(self) -> Self { R::ln1p::<P>(self) }
    #[inline(always)] fn log2_p<P: Policy>(self) -> Self { R::log2::<P>(self) }
    #[inline(always)] fn log10_p<P: Policy>(self) -> Self { R::log10::<P>(self) }
    #[inline(always)] fn erf_p<P: Policy>(self) -> Self { R::erf::<P>(self) }
    #[inline(always)] fn erfc_p<P: Policy>(self) -> Self { R::erfc::<P>(self) }
    #[inline(always)] fn erfinv_p<P: Policy>(self) -> Self { R::erfinv::<P>(self) }
}
