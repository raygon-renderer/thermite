#![allow(clippy::excessive_precision)]

use thermite::{
    mask::GenericMask,
    math::{
        CoreMathWithPolicy, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{
            Policy, PrecisionPolicy,
            policies::{ExtraPrecision, LessPrecision},
        },
        specialized::FlushDenormals,
    },
    register::{Element, FloatElement},
    vector::{NumericVector, PartialOrdVector},
};

use super::SpecialMathWithPolicy as _;

mod pd;
mod ps;

pub trait SpecializedSpecialMath<E>: thermite::math::specialized::SpecializedTranscendentalMath<E> {
    fn erf<P: Policy>(self) -> Self;

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        Self::ONE - self.erf_p::<P>()
    }

    fn erfinv<P: Policy>(self) -> Self;

    fn tgamma<P: Policy>(x: Self) -> Self;
    fn lgamma_r<P: Policy>(x: Self) -> (Self, Self);

    #[inline(always)]
    fn lgamma<P: Policy>(x: Self) -> Self {
        Self::lgamma_r::<P>(x).0
    }

    #[inline(always)]
    fn hermite<P: Policy, const N: usize>(mut x: Self) -> Self {
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        let one = Self::ONE;
        let mut p0 = one;

        if N == 0 {
            return p0;
        }

        let mut p1 = x + x; // 2 * x

        let mut c = 1;
        let mut cf = one;

        while c < N {
            (p0, p1) = (p1, p0); // swap p0, p1

            let next0 = x.mul_sube(p0, cf * p1);

            p1 = next0 + next0; // 2 * next0

            c += 1;
            cf += one;
        }

        p1
    }

    #[inline(always)]
    fn hermitev<P: Policy>(mut x: Self, n: Self::Unsigned) -> Self {
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        let i1 = Self::Unsigned::ONE;
        let n_is_zero = n.cmp_eq(Self::Unsigned::ZERO);

        let mut c = i1;

        // count `n = c.to_float()` separately to avoid expensive converting every iteration
        let mut cf = Self::ONE;

        let mut p0 = Self::ONE;
        let mut p1 = x + x; // 2 * x

        loop {
            let cont = c.cmp_lt(n);

            if cont.none() {
                break;
            }

            (p0, p1) = (p1, p0); // swap p0, p1

            let next0 = x.mul_sube(p0, cf * p1);
            let next = next0 + next0; // 2 * next0

            p1 = cont.select(next, p1);

            c += i1;
            cf += Self::ONE;
        }

        n_is_zero.select(Self::ONE, p1)
    }

    #[inline(always)]
    fn jacobi<P: Policy>(mut x: Self, mut alpha: Self, mut beta: Self, mut n: u32, m: u32) -> Self {
        if thermite::unlikely(m > n) {
            return Self::ZERO;
        }

        if let Some(new) = FlushDenormals::<P>::flush_denormals([x, alpha, beta]) {
            x = new[0];
            alpha = new[1];
            beta = new[2];
        }

        let mut scale = Self::ONE;

        if m > 0 {
            let mut jf = Self::ONE;
            let nf = Self::splat(E::from_i64(n as i64));

            let t0 = Self::HALF * (nf + alpha + beta);

            for _ in 0..m {
                scale *= Self::HALF.mul_adde(jf, t0);
                jf += Self::ONE;
            }

            let mf = Self::splat(E::from_i64(m as i64));

            alpha += mf;
            beta += mf;
            n -= m;
        }

        if thermite::unlikely(n == 0) {
            return scale; // scale * one
        }

        let mut y0 = Self::ONE;

        let alpha_p_beta = alpha + beta;
        let alpha_sqr = alpha * alpha;
        let beta_sqr = beta * beta;
        let alpha1 = alpha - Self::ONE;
        let beta1 = beta - Self::ONE;
        let alpha2beta2 = alpha_sqr - beta_sqr;

        //let mut y1 = alpha + 1 + 0.5 * (alpha_p_beta + 2) * (x - 1);
        let mut y1 = Self::HALF * (x.mul_adde(alpha, alpha) + x.mul_sube(beta, beta) + x + x);

        let mut yk = y1;
        let mut k = E::from_i64(2);

        let k_max = E::from_i64(n as i64) * (<E as Element>::ONE + E::EPSILON);

        while k < k_max {
            let kf = Self::splat(k);
            let kf2 = Self::TWO * kf;

            let k_alpha_p_beta = kf + alpha_p_beta;
            let k2_alpha_p_beta = kf2 + alpha_p_beta;

            let k2_alpha_p_beta_m2 = k2_alpha_p_beta - Self::TWO;

            let denom = kf2 * k_alpha_p_beta * k2_alpha_p_beta_m2;
            let t0 = x.mul_adde(k2_alpha_p_beta * k2_alpha_p_beta_m2, alpha2beta2);
            let gamma1 = k2_alpha_p_beta.mul_sube(t0, t0);
            let gamma0 = Self::TWO * (kf + alpha1) * (kf + beta1) * k2_alpha_p_beta;

            yk = gamma1.mul_sube(y1, gamma0 * y0) / denom;

            y0 = y1;
            y1 = yk;

            k = k + <E as Element>::ONE;
        }

        scale * yk
    }

    #[inline(always)]
    fn gaussian<P: Policy>(mut x: Self, a: Self, c: Self) -> Self {
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        let xc = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            x * c.reciprocal_p::<P>()
        } else {
            x / c
        };

        a * (-Self::HALF * xc * xc).exp_p::<P>()
    }

    fn beta<P: Policy>(a: Self, b: Self) -> Self;

    #[inline(always)]
    fn gaussian_integral<P: Policy>(x0: Self, x1: Self, a: Self, c: Self) -> Self {
        // https://www.wolframalpha.com/input?i=integrate%20a*e%5E(-1%2F2%20*%20x%5E2%2Fc%5E2)%20from%20x%3Dx_0%20to%20x%3Dx_1
        let common = Self::SQRT_FRAC_PI_2 * a * c;
        let denom = Self::SQRT_2 * c;

        let (a1, a0) = if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            let d = denom.reciprocal_p::<P>();
            (x1 * d, x0 * d)
        } else {
            (x1 / denom, x0 / denom)
        };

        common * (a1.erf_p::<P>() - a0.erf_p::<P>())
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn legendre0<P: Policy, const N: u32>(x: Self, n: u32) -> Self {
        macro_rules! c { ($n:literal / $d:literal) => { Self::splat(E::from_i64($n) / E::from_i64($d)) }; }

        let x2 = x.square();
        let x4 = x2.square();
        let x8 = x4.square();

        if const { N != 0 } {
            unsafe { core::hint::assert_unchecked(N == n); }
        }

        // hand-tuned Estrin's scheme polynomials
        match n {
            1 => x,
            2 => x2.mul_adde(c!(3 / 2), c!(-1 / 2)),
            3 => x * x2.mul_adde(c!(5 / 2), c!(-3 / 2)),
            4 => x4.mul_adde(c!(35 / 8), x2.mul_adde(c!(-15 / 4), c!(3 / 8))),
            5 => x * x4.mul_adde(c!(63 / 8), x2.mul_adde(c!(-35 / 4), c!(15 / 8))),
            6 => x4.mul_adde(
                x2.mul_adde(c!(231 / 16), c!(-315 / 16)),
                x2.mul_adde(c!(105 / 16), c!(-5 / 16)),
            ),
            7 => x * x4.mul_adde(
                x2.mul_adde(c!(429 / 16), c!(-693 / 16)),
                x2.mul_adde(c!(315 / 16), c!(-35 / 16)),
            ),
            8 => x8.mul_adde(c!(6435 / 128), x4.mul_adde(
                x2.mul_adde(c!(-3003 / 32), c!(3465 / 64)),
                x2.mul_adde(c!(-315 / 32), c!(35 / 128)),
            )),
            9 => x * x8.mul_adde(c!(12155 / 128), x4.mul_adde(
                x2.mul_adde(c!(-6435 / 32), c!(9009 / 64)),
                x2.mul_adde(c!(-1155 / 32), c!(315 / 128)),
            )),
            10 => x8.mul_adde(
                x2.mul_adde(c!(46189 / 256), c!(-109395 / 256)),
                x4.mul_adde(
                    x2.mul_adde(c!(45045 / 128), c!(-15015 / 128)),
                    x2.mul_adde(c!(3465 / 256), c!(-63 / 256)),
                ),
            ),
            11 => x * x8.mul_adde(
                x2.mul_adde(c!(88179 / 256), c!(-230945 / 256)),
                x4.mul_adde(
                    x2.mul_adde(c!(109395 / 128), c!(-45045 / 128)),
                    x2.mul_adde(c!(15015 / 256), c!(-693 / 256)),
                ),
            ),
            12 => x8.mul_adde(
                x4.mul_adde(c!(676039 / 1024), x2.mul_adde(c!(-969969 / 512), c!(2078505 / 1024))),
                x4.mul_adde(
                    x2.mul_adde(c!(-255255 / 256), c!(225225 / 1024)),
                    x2.mul_adde(c!(-9009 / 512), c!(231 / 1024)),
                ),
            ),
            13 => x * x8.mul_adde(
                x4.mul_adde(c!(1300075 / 1024), x2.mul_adde(c!(-2028117 / 512), c!(4849845 / 1024))),
                x4.mul_adde(
                    x2.mul_adde(c!(-692835 / 256), c!(765765 / 1024)),
                    x2.mul_adde(c!(-45045 / 512), c!(3003 / 1024)),
                ),
            ),
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn legendre<P: Policy>(mut x: Self, n: u32, m: u32) -> Self {
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        match (n, m) {
            (0, 0) => return Self::ONE,
            (n, 0) if n < 14 => return Self::legendre0::<P, 0>(x, n),
            (n, 0) => {
                let mut k = 14; // set to max degree hard-coded + 1

                // these should inline
                let mut p0 = Self::legendre0::<P, 12>(x, 12); // n = k - 2
                let mut p1 = Self::legendre0::<P, 13>(x, 13); // n = k - 1

                while k <= n {
                    let nf = Self::splat(E::from_i64(k as i64));

                    let tmp = p1;
                    p1 = x.mul_sube((nf + nf).mul_sube(p1, p1), nf.mul_sube(p0, p0)) / nf;
                    p0 = tmp;

                    k += 1;
                }

                return p1;
            }
            _ => {}
        }

        let jacobi = Self::jacobi::<P>(x, Self::ZERO, Self::ZERO, n, m);

        let x12 = x.nmul_adde(x, Self::ONE); // (1 - x^2)

        if m & 1 == 0 {
            jacobi * Self::powi::<P>(x12, (m >> 1) as i32)
        } else {
            // negate sign for odd powers (-1)^m
            -jacobi * Self::powi::<P>(x12, m as i32).sqrt()
        }
    }
}
