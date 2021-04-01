#![allow(unused_imports)]
#![no_std]

use thermite::*;

use thermite_complex::Complex;

mod pd;
mod ps;

pub trait SimdVectorizedSpecialFunctionsPolicied<S: Simd>: SimdFloatVector<S> {
    fn hermite_p<P: Policy>(self, n: u32) -> Self;
    fn hermitev_p<P: Policy>(self, n: S::Vu32) -> Self;
    fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;
    fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self;

    fn tgamma_p<P: Policy>(self) -> Self;
    fn lgamma_p<P: Policy>(self) -> Self;
    fn digamma_p<P: Policy>(self) -> Self;
    fn beta_p<P: Policy>(self, y: Self) -> Self;

    fn gaussian_p<P: Policy>(self, a: Self, c: Self) -> Self;
    fn gaussian_integral_p<P: Policy>(self, x1: Self, a: Self, c: Self) -> Self;

    //fn hankel_p<P: Policy>(self, x: Self, sign: bool) -> Complex<S, Self, P>;

    fn bessel_j_p<P: Policy>(self, n: u32) -> Self;
    //fn bessel_jf_p<P: Policy>(self, n: Self::Element) -> Self;
    fn bessel_y_p<P: Policy>(self, n: u32) -> Self;
    //fn bessel_yf_p<P: Policy>(self, n: Self::Element) -> Self;
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
    /// Internally, this is computed with [`jacobi`](#tymethod.jacobi)
    fn legendre(self, n: u32, m: u32) -> Self;

    /// Computes the generic Gaussian function:
    ///
    /// ```
    /// f(x) = a * e^(-1/2 * x^2/c^2)
    /// ```
    fn gaussian(self, a: Self, c: Self) -> Self;

    /// Integrates the generic Gaussian function from `x0`(`self`) to `x1`
    ///
    /// **NOTE**: This uses the Gaussian form `f(x) = a * e^(-1/2 * x^2/c^2)`, so if you offset `x` by some amount,
    /// make sure to do that here as well with `x0`(`self`) and `x1`
    fn gaussian_integral(self, x1: Self, a: Self, c: Self) -> Self;

    /// Computes the Bessel function of the first kind `J_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    fn bessel_j(self, n: u32) -> Self;

    // /// Computes the Bessel function of the first kind `J_n(x)` with real order `n`.
    // fn bessel_jf(self, n: Self::Element) -> Self;

    /// Computes the Bessel function of the second kind `Y_n(x)` with whole integer order `n`.
    ///
    /// **NOTE**: For `n < 2`, this uses an efficient rational polynomial approximation.
    fn bessel_y(self, n: u32) -> Self;

    // /// Computes the Bessel function of the second kind `Y_n(x)` with real order `n`.
    // fn bessel_yf(self, n: Self::Element) -> Self;
}

#[rustfmt::skip]
#[dispatch(S)]
impl<S: Simd, T> SimdVectorizedSpecialFunctionsPolicied<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedSpecialFunctionsInternal<S, Vf = T>,
{
    #[inline] fn hermite_p<P: Policy>(self, n: u32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::hermite::<P>(self, n)
    }
    #[inline] fn hermitev_p<P: Policy>(self, n: S::Vu32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::hermitev::<P>(self, n)
    }
    #[inline] fn jacobi_p<P: Policy>(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::jacobi::<P>(self, alpha, beta, n, m)
    }
    #[inline] fn legendre_p<P: Policy>(self, n: u32, m: u32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::legendre::<P>(self, n, m)
    }

    #[inline] fn tgamma_p<P: Policy>(self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::tgamma::<P>(self)
    }
    #[inline] fn lgamma_p<P: Policy>(self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::lgamma::<P>(self)
    }
    #[inline] fn digamma_p<P: Policy>(self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::digamma::<P>(self)
    }
    #[inline] fn beta_p<P: Policy>(self, y: Self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::beta::<P>(self, y)
    }

    #[inline] fn gaussian_p<P: Policy>(self, a: Self, c: Self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::gaussian::<P>(self, a, c)
    }
    #[inline] fn gaussian_integral_p<P: Policy>(self, x1: Self, a: Self, c: Self) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::gaussian_integral::<P>(self, x1, a, c)
    }

    #[inline] fn bessel_j_p<P: Policy>(self, n: u32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::bessel_j::<P>(self, n)
    }
    //#[inline] fn bessel_jf_p<P: Policy>(self, n: Self::Element) -> Self {
    //    <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::bessel_jf::<P>(self, n)
    //}
    #[inline] fn bessel_y_p<P: Policy>(self, n: u32) -> Self {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::bessel_y::<P>(self, n)
    }
    //#[inline] fn bessel_yf_p<P: Policy>(self, n: Self::Element) -> Self {
    //    <<Self as SimdVectorBase<S>>::Element as SimdVectorizedSpecialFunctionsInternal<S>>::bessel_yf::<P>(self, n)
    //}
}

#[rustfmt::skip]
impl<S: Simd, T> SimdVectorizedSpecialFunctions<S> for T
where
    T: SimdVectorizedSpecialFunctionsPolicied<S>,
{
    #[inline(always)] fn tgamma(self)                                           -> Self { self.tgamma_p::<DefaultPolicy>() }
    #[inline(always)] fn lgamma(self)                                           -> Self { self.lgamma_p::<DefaultPolicy>() }
    #[inline(always)] fn digamma(self)                                          -> Self { self.digamma_p::<DefaultPolicy>() }
    #[inline(always)] fn beta(self, y: Self)                                    -> Self { self.beta_p::<DefaultPolicy>(y) }
    #[inline(always)] fn hermite(self, n: u32)                                  -> Self { self.hermite_p::<DefaultPolicy>(n) }
    #[inline(always)] fn hermitev(self, n: S::Vu32)                             -> Self { self.hermitev_p::<DefaultPolicy>(n) }
    #[inline(always)] fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32)  -> Self { self.jacobi_p::<DefaultPolicy>(alpha, beta, n, m) }
    #[inline(always)] fn legendre(self, n: u32, m: u32)                         -> Self { self.legendre_p::<DefaultPolicy>(n, m) }
    #[inline(always)] fn gaussian(self, a: Self, c: Self)                       -> Self { self.gaussian_p::<DefaultPolicy>(a, c) }
    #[inline(always)] fn gaussian_integral(self, a: Self, c: Self, x1: Self)    -> Self { self.gaussian_integral_p::<DefaultPolicy>(x1, a, c) }

    #[inline(always)] fn bessel_j(self, n: u32)                                 -> Self { self.bessel_j_p::<DefaultPolicy>(n) }
    //#[inline(always)] fn bessel_jf(self, n: Self::Element)                      -> Self { self.bessel_jf_p::<DefaultPolicy>(n) }
    #[inline(always)] fn bessel_y(self, n: u32)                                 -> Self { self.bessel_y_p::<DefaultPolicy>(n) }
    //#[inline(always)] fn bessel_yf(self, n: Self::Element)                      -> Self { self.bessel_yf_p::<DefaultPolicy>(n) }
}

#[doc(hidden)]
pub trait SimdVectorizedSpecialFunctionsInternal<S: Simd>: SimdVectorizedMathInternal<S> {
    #[inline(always)]
    fn hermite<P: Policy>(x: Self::Vf, n: u32) -> Self::Vf {
        let one = Self::Vf::one();
        let mut p0 = one;

        if thermite_unlikely!(n == 0) {
            return p0;
        }

        let mut p1 = x + x; // 2 * x

        let mut c = 1;
        let mut cf = one;

        while c < n {
            // swap p0, p1
            let tmp = p0;
            p0 = p1;
            p1 = tmp;

            let next0 = x.mul_sube(p0, cf * p1);

            p1 = next0 + next0; // 2 * next0

            c += 1;
            cf += one;
        }

        p1
    }

    #[inline(always)]
    fn hermitev<P: Policy>(x: Self::Vf, n: S::Vu32) -> Self::Vf {
        let one = Self::Vf::one();
        let i1 = Vu32::<S>::one();
        let n_is_zero = n.eq(Vu32::<S>::zero());

        let mut c = i1;

        // count `n = c.to_float()` separately to avoid expensive converting every iteration
        let mut cf = one;

        let mut p0 = one;
        let mut p1 = x + x; // 2 * x

        loop {
            let cont = c.lt(n);

            if cont.none() {
                break;
            }

            // swap p0, p1
            let tmp = p0;
            p0 = p1;
            p1 = tmp;

            let next0 = x.mul_sube(p0, cf * p1);
            let next = next0 + next0; // 2 * next0

            p1 = cont.select(next, p1);

            c += i1;
            cf += one;
        }

        n_is_zero.select(one, p1)
    }

    // TODO: Add some associated forms?
    /// This is split into its own function to avoid direct recursion within `legendre_p`,
    /// as that will prevent loop unrolling or inlining and optimization at all.
    #[inline(always)]
    fn hardcoded_legendre(x: Self::Vf, n: u32) -> Self::Vf {
        macro_rules! c {
            ($n:expr, $d:expr) => {
                Self::Vf::splat(Self::cast_from($n) / Self::cast_from($d))
            };
        }

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        // hand-tuned Estrin's scheme polynomials
        match n {
            2 => x2.mul_adde(c!(3, 2), c!(-1, 2)),
            3 => x * x2.mul_adde(c!(5, 2), c!(-3, 2)),
            4 => x4.mul_adde(c!(35, 8), x2.mul_adde(c!(-15, 4), c!(3, 8))),
            5 => x * x4.mul_adde(c!(63, 8), x2.mul_adde(c!(-35, 4), c!(15, 8))),
            6 => x4.mul_adde(
                x2.mul_adde(c!(231, 16), c!(-315, 16)),
                x2.mul_adde(c!(105, 16), c!(-5, 16)),
            ),
            7 => {
                x * x4.mul_adde(
                    x2.mul_adde(c!(429, 16), c!(-693, 16)),
                    x2.mul_adde(c!(315, 16), c!(-35, 16)),
                )
            }
            8 => x8.mul_adde(
                c!(6435, 128),
                x4.mul_adde(
                    x2.mul_adde(c!(-3003, 32), c!(3465, 64)),
                    x2.mul_adde(c!(-315, 32), c!(35, 128)),
                ),
            ),
            9 => {
                x * x8.mul_adde(
                    c!(12155, 128),
                    x4.mul_adde(
                        x2.mul_adde(c!(-6435, 32), c!(9009, 64)),
                        x2.mul_adde(c!(-1155, 32), c!(315, 128)),
                    ),
                )
            }
            10 => x8.mul_adde(
                x2.mul_adde(c!(46189, 256), c!(-109395, 256)),
                x4.mul_adde(
                    x2.mul_adde(c!(45045, 128), c!(-15015, 128)),
                    x2.mul_adde(c!(3465, 256), c!(-63, 256)),
                ),
            ),
            11 => {
                x * x8.mul_adde(
                    x2.mul_adde(c!(88179, 256), c!(-230945, 256)),
                    x4.mul_adde(
                        x2.mul_adde(c!(109395, 128), c!(-45045, 128)),
                        x2.mul_adde(c!(15015, 256), c!(-693, 256)),
                    ),
                )
            }
            12 => x8.mul_adde(
                x4.mul_adde(c!(676039, 1024), x2.mul_adde(c!(-969969, 512), c!(2078505, 1024))),
                x4.mul_adde(
                    x2.mul_adde(c!(-255255, 256), c!(225225, 1024)),
                    x2.mul_adde(c!(-9009, 512), c!(231, 1024)),
                ),
            ),
            13 => {
                x * x8.mul_adde(
                    x4.mul_adde(c!(1300075, 1024), x2.mul_adde(c!(-2028117, 512), c!(4849845, 1024))),
                    x4.mul_adde(
                        x2.mul_adde(c!(-692835, 256), c!(765765, 1024)),
                        x2.mul_adde(c!(-45045, 512), c!(3003, 1024)),
                    ),
                )
            }
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn legendre<P: Policy>(x: Self::Vf, n: u32, m: u32) -> Self::Vf {
        let zero = Self::Vf::zero();
        let one = Self::Vf::one();

        match (n, m) {
            (0, 0) => return one,
            (1, 0) => return x,
            (n, 0) if n <= 13 => return Self::hardcoded_legendre(x, n),
            (n, 0) => {
                let mut k = 14; // set to max degree hard-coded + 1

                // these should inline
                let mut p0 = Self::hardcoded_legendre(x, k - 2);
                let mut p1 = Self::hardcoded_legendre(x, k - 1);

                while k <= n {
                    let nf = Self::Vf::splat_as::<u32>(k);

                    let tmp = p1;
                    p1 = x.mul_sube((nf + nf).mul_sube(p1, p1), nf.mul_sube(p0, p0)) / nf;
                    p0 = tmp;

                    k += 1;
                }

                return p1;
            }
            _ => {}
        }

        let jacobi = Self::jacobi::<P>(x, zero, zero, n, m);

        let x12 = x.nmul_adde(x, one); // (1 - x^2)

        if m & 1 == 0 {
            jacobi * Self::powi::<P>(x12, (m >> 1) as i32)
        } else {
            // negate sign for odd powers (-1)^m
            -jacobi * Self::powi::<P>(x12, m as i32).sqrt()
        }
    }

    #[inline(always)]
    fn jacobi<P: Policy>(x: Self::Vf, mut alpha: Self::Vf, mut beta: Self::Vf, mut n: u32, m: u32) -> Self::Vf {
        /*
            This implementation is a little weird since I wanted to keep it generic, but all casts
            from integers should be const-folded
        */

        if thermite_unlikely!(m > n) {
            return Self::Vf::zero();
        }

        let one = Self::Vf::one();
        let two = Self::Vf::splat_any(2);
        let half = Self::Vf::splat_any(0.5);

        let mut scale = one;

        if m > 0 {
            let mut jf = one;
            let nf = Self::Vf::splat_as::<u32>(n);

            let t0 = half * (nf + alpha + beta);

            for _ in 0..m {
                scale *= half.mul_adde(jf, t0);
                jf += one;
            }

            let mf = Self::Vf::splat_as::<u32>(m);

            alpha += mf;
            beta += mf;
            n -= m;
        }

        if thermite_unlikely!(n == 0) {
            return scale; // scale * one
        }

        let mut y0 = one;

        let alpha_p_beta = alpha + beta;
        let alpha_sqr = alpha * alpha;
        let beta_sqr = beta * beta;
        let alpha1 = alpha - one;
        let beta1 = beta - one;
        let alpha2beta2 = alpha_sqr - beta_sqr;

        //let mut y1 = alpha + one + half * (alpha_p_beta + two) * (x - one);
        let mut y1 = half * (x.mul_adde(alpha, alpha) + x.mul_sube(beta, beta) + x + x);

        let mut yk = y1;
        let mut k = Self::cast_from(2u32);

        let k_max = Self::cast_from(n) * (Self::cast_from(1u32) + Self::__EPSILON);

        while k < k_max {
            let kf = Self::Vf::splat(k);
            let kf2 = two * kf;

            let k_alpha_p_beta = kf + alpha_p_beta;
            let k2_alpha_p_beta = kf2 + alpha_p_beta;

            let k2_alpha_p_beta_m2 = k2_alpha_p_beta - two;

            let denom = kf2 * k_alpha_p_beta * k2_alpha_p_beta_m2;
            let t0 = x.mul_adde(k2_alpha_p_beta * k2_alpha_p_beta_m2, alpha2beta2);
            let gamma1 = k2_alpha_p_beta.mul_sube(t0, t0);
            let gamma0 = two * (kf + alpha1) * (kf + beta1) * k2_alpha_p_beta;

            yk = gamma1.mul_sube(y1, gamma0 * y0) / denom;

            y0 = y1;
            y1 = yk;

            k = k + Self::cast_from(1u32);
        }

        scale * yk
    }

    fn tgamma<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn lgamma<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn digamma<P: Policy>(x: Self::Vf) -> Self::Vf;
    fn beta<P: Policy>(x: Self::Vf, y: Self::Vf) -> Self::Vf;

    #[inline(always)]
    fn gaussian<P: Policy>(x: Self::Vf, a: Self::Vf, c: Self::Vf) -> Self::Vf {
        let xc = match P::POLICY.precision {
            PrecisionPolicy::Worst => x * c.reciprocal_p::<P>(),
            _ => x / c,
        };

        a * (Self::Vf::splat_as(-0.5) * xc * xc).exp_p::<P>()
    }

    #[inline(always)]
    fn gaussian_integral<P: Policy>(x0: Self::Vf, x1: Self::Vf, a: Self::Vf, c: Self::Vf) -> Self::Vf {
        let common = Self::Vf::SQRT_FRAC_PI_2() * a * c;
        let denom = Self::Vf::SQRT_2() * c;

        let (a1, a0) = match P::POLICY.precision {
            PrecisionPolicy::Worst => {
                let denom = denom.reciprocal_p::<P>();
                (x1 * denom, x0 * denom)
            }
            _ => (x1 / denom, x0 / denom),
        };

        common * (a1.erf_p::<P>() - a0.erf_p::<P>())
    }

    #[inline(always)]
    fn bessel_j<P: Policy>(x: Self::Vf, n: u32) -> Self::Vf {
        bessel::bessel_j::<S, Self, P>(x, n)
    }

    //#[inline(always)]
    //fn bessel_jf<P: Policy>(_x: Self::Vf, _n: Self) -> Self::Vf {
    //    unimplemented!()
    //}

    #[inline(always)]
    fn bessel_y<P: Policy>(x: Self::Vf, n: u32) -> Self::Vf {
        bessel::bessel_y::<S, Self, P>(x, n)
    }

    //#[inline(always)]
    //fn bessel_yf<P: Policy>(_x: Self::Vf, _n: Self) -> Self::Vf {
    //    unimplemented!()
    //}
}

mod bessel;
