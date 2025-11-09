#![allow(unused, clippy::needless_arbitrary_self_type)]

use thermite::{
    Vector,
    math::{
        FloatConsts, MathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
    register::FloatRegister,
};

use crate::internal::SpecialMathInternal;

mod internal;

macro_rules! decl_math {
    ($(
        $(#[$meta:meta])*
        fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
            $(where [ $($where_clause:tt)* ])?;
    )*) => {paste::paste! {
        /// Special Math functions for floating-point vectors with customizable policies.
        ///
        /// This trait provides a set of mathematical operations that can be performed
        /// on floating-point vector types. Each function has a variant that accepts
        /// a policy parameter, allowing for fine-tuned control over precision and performance.
        ///
        /// For convenience, a default implementation is also provided in the [`SpecialMath`] trait,
        /// which uses the [`DefaultPolicy`]. All floating-point vector types that implement
        /// the necessary internal math operations will automatically implement this trait, and
        /// the [`SpecialMath`] trait as well for all types that implement this one.
        pub trait SpecialMathWithPolicy<R: FloatRegister>: MathWithPolicy<R> {$(
            $(#[$meta])*
            fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*}

        /// Special Math functions for floating-point vectors using the default policy.
        ///
        /// This trait provides the same set of mathematical operations as [`SpecialMathWithPolicy`],
        /// but uses the [`DefaultPolicy`] for all operations. This allows for easier usage
        /// when specific policy customization is not required.
        ///
        /// Implementors of [`SpecialMathWithPolicy`] will automatically implement this trait as well.
        ///
        /// All methods here have an associated method in [`SpecialMathWithPolicy`] with a `_p` suffix
        /// that accepts a policy parameter as the first generic argument.
        pub trait SpecialMath<R: FloatRegister>: SpecialMathWithPolicy<R> {$(
            $(#[$meta])*
            #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                SpecialMathWithPolicy::<R>::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*)
            }
        )*}

        impl<M, R: FloatRegister> SpecialMath<R> for M where M: SpecialMathWithPolicy<R> {}

        impl<E, R> SpecialMathWithPolicy<R> for Vector<R>
        where
            R: SpecialMathInternal<E, Element = E>,
            E: FloatConsts,
        {$(
            #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                R::$name::<P, $($generic_names),*>($($arg_name),*)
            }
        )*}
    }};
}

decl_math! {
    /// Computes the Gamma function (`Γ(z)`) for any real input, for each value in a vector.
    ///
    /// This implementation uses a few different behaviors to ensure the greatest precision where possible.
    ///
    /// * For non-integer positive inputs, it uses the Lanczos approximation.
    /// * For small non-integer negative inputs, it uses the recursive identity `Γ(z)=Γ(z+1)/z` until `z` is positive.
    /// * For large non-integer negative inputs, it uses the reflection formula `-π/(Γ(z)sin(πz)z)`.
    /// * For positive integers, it simply computes the factorial in a tight loop to ensure precision. Lookup tables could not be used with SIMD.
    /// * At zero, the result will be positive or negative infinity based on the input sign (signed zero is a thing).
    ///
    /// **NOTE**: The Gamma function is not defined for negative integers.
    fn tgamma[][](self: Self) -> Self;

    /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector.
    fn lgamma[][](self: Self) -> Self;

    /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector,
    /// and returns the sign of the Gamma function from before the absolute value was taken.
    fn lgamma_r[][](self: Self) -> (Self, Self);

    /// Computes the Beta function `Β(x, y)`
    fn beta[][](self: Self, y: Self) -> Self;

    /// Computes the m-th derivative of the n-th degree Jacobi polynomial
    ///
    /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial.
    ///
    /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
    fn jacobi[][](self: Self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

    /// Computes the N-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `N` is the polynomial degree.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermite[const N: usize][N](self: Self) -> Self;

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
    ///
    /// The polynomial is calculated independently per-lane with the given degree in `n`.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermitev[][](self: Self, n: internal::Vu<R>) -> Self;

    /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
    ///
    /// The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
    fn gaussian[][](self: Self, a: Self, c: Self) -> Self;

    /// Computes the definite integral of the Gaussian function from `x0` to `x1`, with amplitude `a` and standard deviation `c`.
    /// This is more efficient than evaluating the indefinite integral at both limits and subtracting.
    ///
    /// The position `b` is assumed to be zero, so offset the limits accordingly for a non-zero position.
    fn gaussian_integral[][](x0: Self, x1: Self, a: Self, c: Self) -> Self;

    /// Computes the m-th associated n-th degree Legendre polynomial,
    /// where m=0 signifies the regular n-th degree Legendre polynomial.
    ///
    /// If `m` is odd, the input is only valid between -1 and 1
    ///
    /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
    ///
    /// Internally, this is computed with [`jacobi`](SpecialMath::jacobi) when m > 0.
    fn legendre[][](self: Self, n: u32, m: u32) -> Self;
}

/*

pub trait SpecialMathWithPolicyOld<R: FloatRegister>: MathWithPolicy<R> {
    fn hermite_p<P: Policy>(self, n: u32) -> Self;
    fn hermitev_p<P: Policy>(self, n: internal::Vu<R>) -> Self;
    fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;
    fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self;

    fn tgamma_p<P: Policy>(self) -> Self;
    fn lgamma_p<P: Policy>(self) -> Self;
    fn lgamma_r_p<P: Policy>(self) -> (Self, Self);
    fn digamma_p<P: Policy>(self) -> Self;
    fn beta_p<P: Policy>(self, y: Self) -> Self;

    //fn hankel_p<P: Policy>(self, x: Self, sign: bool) -> Complex<S, Self, P>;

    fn bessel_j_p<P: Policy>(self, n: u32) -> Self;
    //fn bessel_jf_p<P: Policy>(self, n: Self::Element) -> Self;
    fn bessel_y_p<P: Policy>(self, n: u32) -> Self;
    //fn bessel_yf_p<P: Policy>(self, n: Self::Element) -> Self;
}

#[rustfmt::skip]
pub trait SpecialMathOld<R: FloatRegister>: SpecialMathWithPolicyOld<R> {

    #[inline(always)] fn tgamma(self) -> Self { self.tgamma_p::<DefaultPolicy>() }

    /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector.
    #[inline(always)] fn lgamma(self) -> Self { self.lgamma_p::<DefaultPolicy>() }

    /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector,
    /// and returns the sign of the Gamma function from before the absolute value was taken.
    #[inline(always)] fn lgamma_r(self) -> (Self, Self) { self.lgamma_r_p::<DefaultPolicy>() }

    /// Computes the Digamma function `ψ(x)`, the first derivative of `ln(Γ(x))`, or `ln(Γ(x)) d/dx`
    #[inline(always)] fn digamma(self) -> Self { self.digamma_p::<DefaultPolicy>() }

    /// Computes the Beta function `Β(x, y)`
    #[inline(always)] fn beta(self, y: Self) -> Self { self.beta_p::<DefaultPolicy>(y) }

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is an unsigned integer representing the polynomial degree.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    ///
    /// **NOTE**: Given a constant `n`, LLVM will happily unroll and optimize the inner loop where possible.
    #[inline(always)] fn hermite(self, n: u32) -> Self { self.hermite_p::<DefaultPolicy>(n) }

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
    ///
    /// The polynomial is calculated independently per-lane with the given degree in `n`.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    #[inline(always)] fn hermitev(self, n: internal::Vu<R>) -> Self { self.hermitev_p::<DefaultPolicy>(n) }

    /// Computes the m-th derivative of the n-th degree Jacobi polynomial
    ///
    /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial.
    ///
    /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
    #[inline(always)] fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self {
        self.jacobi_p::<DefaultPolicy>(alpha, beta, n, m)
    }

    /// Computes the m-th associated n-th degree Legendre polynomial,
    /// where m=0 signifies the regular n-th degree Legendre polynomial.
    ///
    /// If `m` is odd, the input is only valid between -1 and 1
    ///
    /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
    ///
    /// Internally, this is computed with [`jacobi`](#tymethod.jacobi)
    #[inline(always)] fn legendre(self, n: u32, m: u32) -> Self {
        self.legendre_p::<DefaultPolicy>(n, m)
    }

    /// Computes the Bessel function of the first kind `J_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    #[inline(always)] fn bessel_j(self, n: u32) -> Self { self.bessel_j_p::<DefaultPolicy>(n) }

    // /// Computes the Bessel function of the first kind `J_n(x)` with real order `n`.
    // fn bessel_jf(self, n: Self::Element) -> Self;

    /// Computes the Bessel function of the second kind `Y_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    #[inline(always)] fn bessel_y(self, n: u32) -> Self { self.bessel_y_p::<DefaultPolicy>(n) }

    // /// Computes the Bessel function of the second kind `Y_n(x)` with real order `n`.
    // fn bessel_yf(self, n: Self::Element) -> Self;
}

impl<E, R> SpecialMathWithPolicyOld<R> for Vector<R>
where
    Vector<R>: MathWithPolicy<R>,
    R: SpecialMathInternal<E, Element = E>,
    E: FloatConsts,
{
    #[inline(always)]
    fn tgamma_p<P: Policy>(self) -> Self {
        R::tgamma::<P>(self)
    }

    #[inline(always)]
    fn hermite_p<P: Policy>(self, n: u32) -> Self {
        todo!()
    }

    #[inline(always)]
    fn hermitev_p<P: Policy>(self, n: internal::Vu<R>) -> Self {
        todo!()
    }

    #[inline(always)]
    fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self {
        todo!()
    }

    #[inline(always)]
    fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self {
        todo!()
    }

    #[inline(always)]
    fn lgamma_p<P: Policy>(self) -> Self {
        R::lgamma_r::<P>(self).0
    }

    #[inline(always)]
    fn lgamma_r_p<P: Policy>(self) -> (Self, Self) {
        R::lgamma_r::<P>(self)
    }

    #[inline(always)]
    fn digamma_p<P: Policy>(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn beta_p<P: Policy>(self, y: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn bessel_j_p<P: Policy>(self, n: u32) -> Self {
        todo!()
    }

    #[inline(always)]
    fn bessel_y_p<P: Policy>(self, n: u32) -> Self {
        todo!()
    }
}
 */
