#![no_std]
use thermite::*;

pub trait SimdVectorizedSpecialFunctionsPolicied<S: Simd>: SimdFloatVector<S> {
    fn hermite_p<P: Policy>(self, n: u32) -> Self;
    fn hermitev_p<P: Policy>(self, n: S::Vu32) -> Self;
    fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;
    fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self;
    fn bessel_j_p<P: Policy>(self, n: u32) -> Self;
    fn bessel_jf_p<P: Policy>(self, n: Self::Element) -> Self;
    fn bessel_y_p<P: Policy>(self, n: u32) -> Self;
    fn bessel_yf_p<P: Policy>(self, n: Self::Element) -> Self;
    fn tgamma_p<P: Policy>(self) -> Self;
    fn lgamma_p<P: Policy>(self) -> Self;
    fn digamma_p<P: Policy>(self) -> Self;
    fn beta_p<P: Policy>(self, y: Self) -> Self;
}

pub trait SimdVectorizedSpecialFunctions<S: Simd>: SimdVectorizedSpecialFunctionsPolicied<S> {
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
    fn tgamma(self) -> Self;

    /// Computes the natural log of the Gamma function (`ln(Γ(x))`) for any real positive input, for each value in a vector.
    fn lgamma(self) -> Self;

    /// Computes the Digamma function `ψ(x)`, the first derivative of `ln(Γ(x))`, or `ln(Γ(x)) d/dx`
    fn digamma(self) -> Self;

    /// Computes the Beta function `Β(x, y)`
    fn beta(self, y: Self) -> Self;

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is an unsigned integer representing the polynomial degree.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    ///
    /// **NOTE**: Given a constant `n`, LLVM will happily unroll and optimize the inner loop where possible.
    fn hermite(self, n: u32) -> Self;

    /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
    /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
    ///
    /// The polynomial is calculated independenty per-lane with the given degree in `n`.
    ///
    /// This uses the recurrence relation to compute the polynomial iteratively.
    fn hermitev(self, n: S::Vu32) -> Self;

    /// Computes the m-th derivative of the n-th degree Jacobi polynomial
    ///
    /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
    /// Legendre polynomial.
    ///
    /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
    fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

    /// Computes the m-th associated n-th degree Legendre polynomial,
    /// where m=0 signifies the regular n-th degree Legendre polynomial.
    ///
    /// If `m` is odd, the input is only valid between -1 and 1
    ///
    /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
    ///
    /// Internally, this is computed with [`jacobi_d`](#tymethod.jacobi_d)
    fn legendre(self, n: u32, m: u32) -> Self;

    /// Computes the Bessel function of the first kind `J_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    fn bessel_j(self, n: u32) -> Self;

    /// Computes the Bessel function of the first kind `J_n(x)` with real order `n`.
    fn bessel_jf(self, n: Self::Element) -> Self;

    /// Computes the Bessel function of the second kind `Y_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    fn bessel_y(self, n: u32) -> Self;

    /// Computes the Bessel function of the second kind `Y_n(x)` with real order `n`.
    fn bessel_yf(self, n: Self::Element) -> Self;
}

trait SimdVectorizedSpecialFunctionsInternal<S: Simd>: SimdVectorizedMathInternal<S> {}
