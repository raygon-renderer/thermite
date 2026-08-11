#![allow(clippy::excessive_precision)]

use thermite::{
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _,
        policy::{
            Policy, PrecisionPolicy,
            policies::{CheckOverflow, ExtraPrecision, LessPrecision},
        },
        specialized::FlushDenormals,
    },
    register::{Element, FloatElement},
    vector::{NumericVector, PartialOrdVector, SplatConst},
};

use super::SpecialMathWithPolicy as _;

pub(crate) mod generic;
mod pd;
mod ps;

/// The decisions the [`expint`](SpecializedSpecialMath::expint) kernel has to make
/// differently depending on the arithmetic it is running in.
///
/// These are choices *inside* one algorithm, not part of the math surface, so they live
/// here rather than on [`SpecializedSpecialMath`] itself. They exist because a single
/// series/continued-fraction body serves both the real line and the complex cut plane,
/// and "the unit disc", "out of domain" and "negligible but nonzero" are three different
/// comparisons in those two worlds.
///
/// Every method defaults to the real-line answer, so a real vector's implementation is
/// empty and its [`ExpIntDetails`](SpecializedSpecialMath::ExpIntDetails) is `Self`.
pub trait ExpIntDetails<E, V: thermite::vector::FloatVector<Element = E>> {
    /// Lanes that should take the power series rather than the continued fraction.
    ///
    /// On the real line this is `x < 1`. Over C it is `|z| < 1`, which is *not* what a
    /// complex `cmp_lt` means - that is a lexicographic sort order, and reading it as a
    /// magnitude silently routes far-off-axis points into the wrong regime.
    #[inline(always)]
    fn use_series(z: V) -> V::Mask {
        z.cmp_lt(V::ONE)
    }

    /// Lanes outside the domain, forced to NaN when the policy checks overflow.
    ///
    /// Real `E_N` is defined for `x >= 0` only. The complex principal branch covers the
    /// whole cut plane `|Arg z| < pi`, so there the negative reals are in-domain and the
    /// cut is carried entirely by the principal `ln` inside the series.
    #[inline(always)]
    fn invalid(z: V) -> V::Mask {
        z.cmp_lt(V::ZERO) | z.is_nan()
    }

    /// Lentz sentinel: the stand-in for a denominator that came out exactly zero, small
    /// enough to be negligible against any real term.
    ///
    /// The safe magnitude depends on the arithmetic, not just the format. Real division
    /// only needs this to be tiny and nonzero, so `MIN_POSITIVE` is ideal. A complex
    /// reciprocal divides by `|z|^2`, so both the sentinel and its reciprocal have to
    /// survive being *squared* - `MIN_POSITIVE` underflows to zero there, which takes
    /// the whole fraction to NaN.
    #[inline(always)]
    fn cf_tiny() -> V {
        V::MIN_POSITIVE
    }
}

pub trait SpecializedSpecialMath<E>: thermite::math::specialized::SpecializedTranscendentalMath<E> {
    /// Per-arithmetic details of the [`expint`](Self::expint) kernel. Almost always
    /// `Self`, with an empty [`ExpIntDetails`] impl taking every default.
    type ExpIntDetails: ExpIntDetails<E, Self>;

    fn erf<P: Policy>(self) -> Self;

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        Self::ONE - self.erf_p::<P>()
    }

    /// Computes the exponential integral `E_N(x)` for integer order `N`.
    #[inline(always)]
    fn expint<P: Policy, const N: usize>(self) -> Self {
        self.expint_primal::<P, N>().0
    }

    /// Computes `$E_N(x)$` together with the adjacent lower order `$E_{N-1}(x)$`.
    ///
    /// Differentiating the integral definition under the integral sign gives
    /// `$E_N'(x) = -E_{N-1}(x)$`, so the second element is the derivative up to sign.
    /// The order recurrence already walks `E_1 -> E_N`, which makes `E_{N-1}` simply
    /// the previous iterate: the pair costs no more than the value alone. `thermite-dual`
    /// uses this to take the (guarded) real path for both parts rather than running
    /// this entire routine in dual arithmetic.
    ///
    /// Uses the power series for x < 1 and the Stieltjes continued fraction for x >= 1,
    /// computed in parallel across SIMD lanes and blended at the end.
    /// For N > 1, applies the recurrence `$E_{n+1}(x) = (e^{-x} - x \cdot E_n(x)) / n$`.
    #[inline(always)]
    fn expint_primal<P: Policy, const N: usize>(self) -> (Self, Self) {
        let x = self;

        // The series/continued-fraction path below produces E_1, so the two orders
        // beneath it come from their closed forms instead:
        //   E_0(x)    = e^-x / x
        //   E_{-1}(x) = e^-x (1 + 1/x) / x
        let exp_neg_x = (-x).exp_p::<P>();
        let inv_x = x.reciprocal_p::<P>();
        let e0 = exp_neg_x * inv_x;

        if const { N == 0 } {
            let mut value = e0;
            let mut prev = e0 * (Self::ONE + inv_x);

            if const { P::POLICY.check_overflow } {
                // Both orders have a pole at the branch point x = 0.
                let x_is_zero = x.is_zero();
                value = x_is_zero.select(Self::INFINITY, value);
                prev = x_is_zero.select(Self::INFINITY, prev);

                let bad = <Self::ExpIntDetails as ExpIntDetails<E, Self>>::invalid(x);
                value = bad.select(Self::NAN, value);
                prev = bad.select(Self::NAN, prev);
            }

            return (value, prev);
        }

        // E_n(x) is only defined for x > 0 (and x >= 0 for n > 1).
        // Compute E_1(x) first, then apply recurrence for higher orders.

        // === Interleaved power series (x < 1) and continued fraction (x >= 1) ===
        //
        // Power series: E_1(x) = -γ - ln(x) - Σ_{k=1}^∞ (-x)^k / (k*k!)
        //   Recurrence on terms: A_{k+1} = A_k * (-x * k) / (k+1)^2
        //   Starting with A_1 = -x, sum = A_1.
        //
        // Continued fraction (Stieltjes): E_1(x)*e^x = 1/(x+1 - 1^2/(x+3 - 2^2/(x+5 - 3^2/(x+7 - ...))))
        //   In standard Lentz form: b_0=0, a_1=1, b_1=x+1; then a_j=-(j-1)^2, b_j=x+2j-1 for j≥2.
        //   Bootstrap j=1 outside the loop, iterate j≥2 inside.
        //   Result: E_1(x) = f * e^{-x}

        let use_series = <Self::ExpIntDetails as ExpIntDetails<E, Self>>::use_series(x);

        // --- Power series state ---
        let neg_x = -x;
        let mut s_term = neg_x; // A_1 = -x
        let mut s_sum = s_term; // running sum starts at A_1

        // --- Continued fraction state (modified Lentz's method) ---
        //
        // E_1(x)*e^x = 1/(x+1 - 1^2/(x+3 - 2^2/(x+5 - 3^2/(x+7 - ...))))
        //
        // In standard Lentz form b_0 + a_1/(b_1 + a_2/(b_2 + ...)):
        //   b_0 = 0
        //   j=1: a_1 = 1,       b_1 = x+1
        //   j≥2: a_j = -(j-1)^2, b_j = x + 2j - 1
        //
        let tiny = <Self::ExpIntDetails as ExpIntDetails<E, Self>>::cf_tiny();

        // b_0 = 0, so f_0 = tiny, C_0 = tiny, D_0 = 0
        let mut cf_f = tiny;
        let mut cf_c = tiny;
        let mut cf_d = Self::ZERO;

        // Bootstrap j=1 step: a_1 = 1, b_1 = x+1
        {
            let b1 = x + Self::ONE;
            // D_1 = 1/(b_1 + a_1*D_0) = 1/(x+1)
            cf_d = b1.reciprocal_p::<P>();
            // C_1 = b_1 + a_1/C_0 = (x+1) + 1/tiny ≈ 1/tiny
            cf_c = b1 + cf_c.reciprocal_p::<P>();
            let delta = cf_c * cf_d;
            cf_f *= delta; // tiny * (1/tiny)/(x+1) ≈ 1/(x+1)
        }

        // Convergence tolerance
        let eps = Self::splat(E::EPSILON);

        let mut series_done = !use_series; // lanes not using series are "done" immediately
        let mut cf_done = use_series; // lanes not using CF are "done" immediately

        let mut k = 1usize;
        while k < const { P::POLICY.max_iterations } {
            let kf = Self::splat(E::from_int(k as thermite::LargeInt));
            let kp1 = Self::splat(E::from_int(k as thermite::LargeInt + 1));

            // --- Power series step ---
            // A_{k+1} = A_k * (-x * k) / (k+1)^2
            if !series_done.all() {
                s_term *= (neg_x * kf) / (kp1 * kp1);
                s_sum = series_done.select(s_sum, s_sum + s_term);

                let term_small = s_term.abs().cmp_lt(s_sum.abs() * eps);

                // series_done | (use_series & term_small)
                series_done = GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(
                    series_done,
                    use_series,
                    term_small,
                );
            }

            // --- Continued fraction step (j = k+1, so j ≥ 2) ---
            // a_j = -(j-1)^2 = -k^2, b_j = x + 2j - 1 = x + 2k + 1
            if !cf_done.all() {
                let neg_a_k = kf * kf; // |a_j| = k^2
                let b_k = (x + kf) + (kf + Self::ONE); // x + 2k + 1

                // D = 1 / (b - |a|*D_prev)  [note: subtraction because a is negative]
                let d_denom = neg_a_k.nmul_adde(cf_d, b_k); // b - |a|*D
                let new_d = d_denom.cmp_eq(Self::ZERO).select(tiny, d_denom).reciprocal_p::<P>();

                // C = b - |a|/C_prev  [same sign flip]
                let new_c = b_k - neg_a_k / cf_c;
                let new_c = new_c.cmp_eq(Self::ZERO).select(tiny, new_c);

                let delta = new_c * new_d;

                cf_d = new_d;
                cf_c = new_c;
                cf_f = cf_done.select(cf_f, cf_f * delta);

                let cf_converged = (delta - Self::ONE).abs().cmp_lt(eps);

                cf_done =
                    GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (!B & C)) }>(cf_done, use_series, cf_converged);
            }

            if (series_done & cf_done).all() {
                break;
            }

            k += 1;
        }

        // --- Assemble E_1(x) from both methods ---

        // Series: E_1(x) = -γ - ln(x) - sum
        let mut series_result = Self::EMPTY;

        // CF: E_1(x) = cf_f * e^{-x}  (cf_f approximates E_1(x)*e^x)
        let mut cf_result = Self::EMPTY;

        if use_series.any() {
            series_result = (-Self::EULER_GAMMA - s_sum) - x.ln_p::<P>();
        }

        if !use_series.all() {
            cf_result = cf_f * exp_neg_x;
        }

        let mut e_n = use_series.select(series_result, cf_result);

        // Order beneath the current one. Before the recurrence runs, E_N is E_1, so the
        // order below it is E_0.
        let mut e_prev = e0;

        // --- Apply recurrence for N > 1 ---
        // E_{n+1}(x) = (e^{-x} - x * E_n(x)) / n
        if const { N > 1 } {
            let mut n = 1u32;
            while n < N as u32 {
                let nf = Self::splat(E::from_int(n as thermite::LargeInt));
                e_prev = e_n;
                e_n = x.nmul_adde(e_n, exp_neg_x) / nf;
                n += 1;
            }
        }

        // --- Edge cases ---
        if const { P::POLICY.check_overflow } {
            // E_1(0) = +inf, E_n(0) = 1/(n-1) for n > 1
            let x_is_zero = x.is_zero();
            if const { N == 1 } {
                e_n = x_is_zero.select(Self::INFINITY, e_n);
            } else if const { N > 1 } {
                e_n = x_is_zero.select(Self::splat(E::ONE / E::from_int(N as thermite::LargeInt - 1)), e_n);
            }

            // Same rule one order down: E_0 and E_1 both diverge at zero, E_n does not.
            if const { N <= 2 } {
                e_prev = x_is_zero.select(Self::INFINITY, e_prev);
            } else {
                e_prev = x_is_zero.select(Self::splat(E::ONE / E::from_int(N as thermite::LargeInt - 2)), e_prev);
            }

            // Negative x: NaN, and NaN in, NaN out.
            let bad = <Self::ExpIntDetails as ExpIntDetails<E, Self>>::invalid(x);
            e_n = bad.select(Self::NAN, e_n);
            e_prev = bad.select(Self::NAN, e_prev);
        }

        (e_n, e_prev)
    }

    #[inline(always)]
    fn logistic_sigmoid<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            let is_pos = self.is_positive();
            let x = self.neg_c(is_pos); // conditionally negate if positive
            let e = x.exp_p::<P>();

            let n = is_pos.select(Self::ONE, e);
            let d = Self::ONE + e;

            return n / d;
        }

        (Self::ONE + (-self).exp_p::<P>()).reciprocal_p::<ExtraPrecision<P>>()
    }

    #[inline(always)]
    fn softplus<P: Policy>(self, k: Self, rcp_k: Self) -> Self {
        // For low precision, we can get better performance by computing in base-2 instead of base-e,
        // at the cost of some accuracy.
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            // adjust to be in base-2
            let k = k.scale(FloatConsts::LOG2_E);
            let rcp_k = rcp_k.scale(FloatConsts::LN_2);

            let kx = self * k;

            // e needs overflow checks to outright incorrect results here
            let e = kx.abs().neg().exp2_p::<CheckOverflow<P, true>>();
            return (Self::ONE + e).log2_p::<P>().mul_adde(rcp_k, self.max(Self::ZERO));
        }

        let kx = self * k;

        let e = kx.abs().neg().exp_p::<P>();

        // max(0, x) + lnp1(e^(-|x|)) is more stable than ln(1 + e^x) for large |x|.
        e.ln_1p_p::<P>().mul_adde(rcp_k, self.max(Self::ZERO))
    }

    fn tgamma<P: Policy>(self) -> Self;
    fn lgamma<P: Policy>(self) -> Self;
    fn digamma<P: Policy>(self) -> Self;

    /// The trigamma function `psi_1(x) = d/dx psi(x)`, the second derivative of `ln Gamma`.
    ///
    /// Deliberately absent from the public `SpecialMath` trait, unlike every sibling
    /// here. It exists only so that `digamma` is differentiable - forward-mode AD over
    /// the Gamma family needs `psi_1` the way `ln Gamma` needs `psi` - and keeping it
    /// off the public trait is what stops that need from cascading: a public
    /// `trigamma` would oblige `Dual` to implement it, which requires `psi_2`, which
    /// requires `psi_3`, and so on, because the Gamma-derivative family is not closed
    /// under differentiation. Closing it for real means a general `polygamma(n)`,
    /// whose derivative is simply `polygamma(n + 1)`.
    ///
    /// Not defined at zero or the negative integers.
    fn trigamma<P: Policy>(self) -> Self;

    #[inline(always)]
    fn hermite<P: Policy, const N: usize>(mut x: Self) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        let mut p0 = Self::ONE;

        if const { N == 0 } {
            return p0;
        }

        let mut p1 = x + x; // 2 * x

        cfg_if::cfg_if! {
            if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
                use crunchy::unroll;

                macro_rules! unroll_poly {
                    ($($len:tt),*) => {
                        $( if const { N == $len } {
                            unroll! { for n in 0..$len {
                                (p0, p1) = (p1, p0); // swap p0, p1

                                const cf: thermite::LargeInt = (1 + n) as thermite::LargeInt;
                                let next0 = x.mul_sube(p0, p1.scale(E::ConstInt::<{cf}>::VALUE));
                                p1 = next0 + next0; // 2 * next0
                            }}
                        } else )* {
                            let mut c = 1;
                            let mut cf = E::ONE;

                            while c < N {
                                (p0, p1) = (p1, p0); // swap p0, p1

                                let next0 = x.mul_sube(p0, p1.scale(cf));
                                p1 = next0 + next0; // 2 * next0

                                c += 1;
                                cf = cf + E::ONE;
                            }
                        }
                    };
                }

                unroll_poly!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16); // up to N=16
            } else {
                let mut c = 1;
                let mut cf = Self::ONE;

                while c < N {
                    (p0, p1) = (p1, p0); // swap p0, p1

                    let next0 = x.mul_sube(p0, cf * p1);
                    p1 = next0 + next0; // 2 * next0

                    c += 1;
                    cf += Self::ONE;
                }
            }
        }

        p1
    }

    #[inline(always)]
    fn hermitev<P: Policy>(mut x: Self, n: Self::Unsigned) -> Self {
        #[cfg(not(target_arch = "spirv"))]
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
    fn chebyshev<P: Policy, const K: usize, const N: usize>(self, coeffs: &[Self::Element; N]) -> Self {
        const {
            assert!(K >= 1 && K <= 4, "chebyshev: K must be 1, 2, 3, or 4");
            assert!(N >= 1, "chebyshev: N must be at least 1");
        }

        // S = Σ c_k P_0 = c_0 when N = 1; skip the whole recurrence.
        if const { N == 1 } {
            return Self::splat(coeffs[0]);
        }

        let x = self;
        let x2 = x + x;

        // P_1: T_1 = x, U_1 = 2x, V_1 = 2x - 1, W_1 = 2x + 1.
        let p1 = if const { K == 1 } {
            x
        } else if const { K == 2 } {
            x2
        } else if const { K == 3 } {
            x2 - Self::ONE
        } else if const { K == 4 } {
            x2 + Self::ONE
        } else {
            unsafe { core::hint::unreachable_unchecked() }
        };

        let cn1 = Self::splat(coeffs[N - 1]);
        let cn2 = Self::splat(coeffs[N - 2]);

        // S = c_0 + c_1*P_1(x) when N = 2.
        if const { N == 2 } {
            return p1.mul_adde(cn1, cn2);
        }

        // Clenshaw's backward recurrence. All four kinds share the recurrence
        // P_{k+1} = 2x*P_k - P_{k-1} with P_0 = 1, so the b_k loop is identical for all of them
        // and only the final-step P_1(x) differs:
        //
        //     b_{N+1} = b_N = 0
        //     for k = N-1 down to 1:  b_k = 2x*b_{k+1} - b_{k+2} + c_k
        //     S = (c_0 - b_2) + b_1 * P_1(x)
        //
        // This is more numerically stable than the forward sum (especially when the
        // partial sums of Σ c_k P_k are much smaller than max|c_k P_k|) and uses only two
        // running scalars instead of three.
        //
        // Hoist the first two iterations to eliminate the b_2 = 0 subtraction in the loop:
        //     k = N-1:  b_{N-1} = 2x*0 + c_{N-1} - 0          = c_{N-1}
        //     k = N-2:  b_{N-2} = 2x*c_{N-1} + c_{N-2} - 0    = 2x*c_{N-1} + c_{N-2}
        let mut b1 = x2.mul_adde(cn1, cn2); // b_{k+1} = b_{N-2}
        let mut b2 = cn1; // b_{k+2} = b_{N-1}

        // Iterate k = N-3, N-4, ..., 1.
        let mut k = N - 2;
        while k > 1 {
            k -= 1;
            // b_k = (2x*b_{k+1} + c_k) - b_{k+2}
            let bk = x2.mul_adde(b1, Self::splat(coeffs[k]) - b2);
            b2 = b1;
            b1 = bk;
        }

        // S = b_1 * P_1(x) + (c_0 - b_2)
        b1.mul_adde(p1, Self::splat(coeffs[0]) - b2)
    }

    #[inline(always)]
    fn jacobi<P: Policy>(mut x: Self, mut alpha: Self, mut beta: Self, mut n: u32, m: u32) -> Self {
        if thermite::unlikely(m > n) {
            return Self::ZERO;
        }

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([x, alpha, beta]) {
            x = new[0];
            alpha = new[1];
            beta = new[2];
        }

        let mut scale = Self::ONE;

        if m > 0 {
            let mut jf = Self::ONE;
            let nf = Self::splat(E::from_int(n as thermite::LargeInt));

            let t0 = Self::HALF * (nf + alpha + beta);

            let mut _iter = 0;
            while _iter < m {
                _iter += 1;
                scale *= Self::HALF.mul_adde(jf, t0);
                jf += Self::ONE;
            }

            let mf = Self::splat(E::from_int(m as thermite::LargeInt));

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
        let mut k = E::ConstInt::<2>::VALUE;

        let k_max = E::from_int(n as thermite::LargeInt) * (<E as Element>::ONE + E::EPSILON);

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
        #[cfg(not(target_arch = "spirv"))]
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

    #[rustfmt::skip]
    #[inline(always)]
    fn legendre0<P: Policy, const N: u32>(x: Self, n: u32) -> Self {
        macro_rules! c { ($n:literal / $d:literal) => { Self::splat(E::from_int($n) / E::from_int($d)) }; }

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
        #[cfg(not(target_arch = "spirv"))]
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
                    let nf = Self::splat(E::from_int(k as thermite::LargeInt));

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

    fn lambert_w<P: Policy>(self) -> (Self, Self);

    // TEMP(bessel_j): disabled until orders beyond J_0 exist - see the note in lib.rs.
    //fn bessel_j<P: Policy, const N: usize>(self) -> Self;
}

// The Carlson / Legendre entry points are kind-dispatched (`SpecialMath::carlson` / `::ellint`),
// generated by decl_math!'s `@kinds` blocks - they call the request struct's `eval` directly, so
// they need no method here. The request structs and their traits are re-exported below.
pub use generic::elliptic::{
    CarlsonKind, CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE, EllintEInc,
    EllintF, EllintK, EllintPi, EllintPiInc, EllipticKind, WrapTo,
};

/// Specialized implementation trait for real-only special math functions.
///
/// Extends [`SpecializedSpecialMath`] with functions that have no meaningful
/// complex analogue (e.g. functions using the real absolute value, or functions
/// that are inverses of real-domain-only operations).
pub trait SpecializedRealSpecialMath<E>: SpecializedSpecialMath<E> {
    fn erfinv<P: Policy>(self) -> Self;
    fn probit<P: Policy>(self) -> Self;

    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> Self {
        let alpha_x = alpha * self;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        let erf = alpha_x.scale(FloatConsts::FRAC_1_SQRT_2).erf_p::<P>();

        if Self::HAS_TRUE_FMA {
            // if we have true FMA, we can maintain precision while avoiding extra work.
            let half_x = self.scale(E::ConstRatio::<{ 1 }, { 2 }>::VALUE);
            half_x.mul_add(erf, half_x) // 0.5 * x + 0.5 * x * erf
        } else {
            self.scale(E::ConstRatio::<{ 1 }, { 2 }>::VALUE) * (Self::ONE + erf)
        }
    }

    #[inline(always)]
    fn swish<P: Policy>(self, beta: Self) -> Self {
        let x = self;
        let beta_x = beta * x;

        // sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
        let e = (-beta_x).exp_p::<P>();
        let s = (Self::ONE + e).reciprocal_p::<P>();

        x * s
    }

    fn lgamma_r<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn algebraic_sigmoid<P: Policy, const N: usize>(self) -> Self {
        if const { N == 0 } {
            return self; // identity function
        }

        let pre_root = Self::ONE + self.abs().powi_p::<P>(N as i32); // = 1 + |x|^N

        let denom = match N {
            1 => pre_root,
            2 => pre_root.sqrt(),
            3 => pre_root.cbrt_p::<P>(),
            4 if const { P::POLICY.precision.le(PrecisionPolicy::Average) } => pre_root.sqrt().sqrt(),
            _ => {
                // copied from `nth_root`, but without negative handling since we know the input is always ≥ 1
                let x = pre_root;

                // initial guess using reduced precision
                let mut y = x.powf_p::<CheckOverflow<LessPrecision<P>, false>>(Self::splat(
                    E::ONE / E::from_int(N as thermite::LargeInt),
                ));

                // One iteration of Halley's method for nth root
                let y_n = y.powi_p::<P>(N as i32);

                let np1 = Self::splat(E::from_int((N + 1) as thermite::LargeInt));
                let nm1 = Self::splat(E::from_int((N - 1) as thermite::LargeInt));

                let n = y * (x - y_n); // half of numerator
                let d = y_n.mul_adde(np1, x * nm1);

                y += (n + n) / d;

                y
            }
        };

        // denom now equals (1 + |x|^N)^(1/N)

        let mut y = if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            // this is the same number of operations as the more precise version, but
            // with better accuracy on large pre_root when using approximate rpc.
            self * denom.reciprocal_p::<P>()
        } else {
            self / denom
        };

        if const { P::POLICY.check_overflow } {
            y = pre_root.is_infinite().select(self.signum(), y);
        }

        y
    }

    // f(x)  = x*(1/2 + x/(2 sqrt(1 + x^2)))
    // f'(x) = (x^3 + sqrt(1 + x^2) x^2 + sqrt(1 + x^2) + 2 x) / (2 (1 + x^2)^(3/2))
    //
    // With a = 1 + x^2, r = sqrt(a), q = x/r:
    //   f(x)  = (x/2)*(1 + q)
    //   f'(x) = (1 + q + q/a) / 2     (since q' = 1/(a*r), so f' = g + x*g' = (1+q)/2 + q/(2a))
    #[inline(always)]
    fn algebraic_swish<P: Policy>(self) -> Self {
        let x = self;

        if const { Self::HAS_TRUE_FMA } {
            // rsqrt is about 30% faster than sqrt+div, even with the extra
            // newton iteration merged in.
            if const { Self::HAS_APPROX_RSQRT } {
                let a = x.mul_add(x, Self::ONE);
                let y0 = a.rsqrt();
                let ay2 = a * y0 * y0;
                let ch = ay2.nmul_add(Self::HALF, Self::splat(<E as FloatElement>::ConstRatio::<3, 2>::VALUE));
                let r_inv = y0 * ch; // Newton-refined 1/sqrt(a)
                let q = x * r_inv;
                let xh = Self::HALF * x;
                q.mul_add(xh, xh)
            } else {
                let a = x.mul_add(x, Self::ONE);
                let q = x / a.sqrt();
                let xh = x * Self::HALF;
                q.mul_add(xh, xh)
            }
        } else if const { Self::HAS_APPROX_RCP } {
            let a = x * x + Self::ONE;
            let y0 = a.rsqrt();
            let ay2 = a * y0 * y0;
            let c = Self::splat(<E as FloatElement>::ConstInt::<3>::VALUE) - ay2;
            let r_inv_2 = y0 * c; // = 2 * (Newton-refined 1/sqrt(a))
            let hxy1 = Self::splat(<E as FloatElement>::ConstRatio::<1, 4>::VALUE) * (x * r_inv_2); // = q/2
            let w = Self::HALF + hxy1; // = (1 + q)/2
            x * w
        } else {
            let a = x * x + Self::ONE;
            let q = x / a.sqrt();
            let q1 = q + Self::ONE;
            x * Self::HALF * q1
        }
    }

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
}

/// Value-and-derivative (`_d`) forms of the activation functions, for single-value real numbers.
///
/// Every method is a provided default returning `(value, derivative)`; the `value` matches the
/// like-named value-only function in [`SpecializedSpecialMath`] / [`SpecializedRealSpecialMath`].
/// Implemented (as an empty impl) only for primal types -- *not* for derivative-carrying numbers
/// like `Dual`, which obtain the derivative from the value form via automatic differentiation.
pub trait SpecializedRealPrimalMath<E>: SpecializedRealSpecialMath<E> {
    #[inline(always)]
    fn softplus_d<P: Policy>(self, k: Self, rcp_k: Self) -> (Self, Self) {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            let k = k.scale(FloatConsts::LOG2_E);
            let rcp_k = rcp_k.scale(FloatConsts::LN_2);

            let kx = self * k;

            let e = kx.abs().neg().exp2_p::<CheckOverflow<P, true>>();
            let y = (Self::ONE + e).log2_p::<P>().mul_adde(rcp_k, self.max(Self::ZERO));

            let rcp = (Self::ONE + e).reciprocal_p::<P>();
            let dy = kx.select_negative(e * rcp, rcp);

            return (y, dy);
        }

        let kx = self * k;

        let e = kx.abs().neg().exp_p::<P>();

        // max(0, x) + lnp1(e^(-|x|)) is more stable than ln(1 + e^x) for large |x|.
        let y = e.ln_1p_p::<P>().mul_adde(rcp_k, self.max(Self::ZERO));

        // sigmoid from already-computed e = exp(-|kx|)
        let rcp = (e + Self::ONE).reciprocal_p::<P>();
        let dy = kx.select_negative(e * rcp, rcp);

        (y, dy)
    }

    #[inline(always)]
    fn gelu_d<P: Policy>(self, alpha: Self) -> (Self, Self) {
        let alpha_x = alpha * self;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        let erf = alpha_x.scale(FloatConsts::FRAC_1_SQRT_2).erf_p::<P>();

        let y = if Self::HAS_TRUE_FMA {
            let half_x = self.scale(E::ConstRatio::<{ 1 }, { 2 }>::VALUE);
            half_x.mul_add(erf, half_x) // 0.5 * x + 0.5 * x * erf
        } else {
            self.scale(E::ConstRatio::<{ 1 }, { 2 }>::VALUE) * (Self::ONE + erf)
        };

        let dy = (alpha_x * alpha_x)
            .scale(E::ConstRatio::<{ -1 }, { 2 }>::VALUE)
            .exp_p::<P>()
            .scale(FloatConsts::FRAC_1_SQRT_TAU);

        (y, dy.mul_adde(alpha_x, y))
    }

    #[inline(always)]
    fn swish_d<P: Policy>(self, beta: Self) -> (Self, Self) {
        let x = self;
        let beta_x = beta * x;

        let e = (-beta_x).exp_p::<P>();
        let s = (Self::ONE + e).reciprocal_p::<P>();

        let y = x * s;

        // dy/dx = s + beta * y * (1 - s); 1 - s = e * s (stable near s ~ 1)
        let dy = (beta * y).mul_adde(e * s, s);

        (y, dy)
    }

    #[inline(always)]
    fn algebraic_sigmoid_d<P: Policy, const N: usize>(self) -> (Self, Self) {
        if const { N == 0 } {
            return (self, Self::ONE); // identity function
        }

        let pre_root = Self::ONE + self.abs().powi_p::<P>(N as i32); // = 1 + |x|^N

        let denom = match N {
            1 => pre_root,
            2 => pre_root.sqrt(),
            3 => pre_root.cbrt_p::<P>(),
            4 if const { P::POLICY.precision.le(PrecisionPolicy::Average) } => pre_root.sqrt().sqrt(),
            _ => {
                let x = pre_root;

                let mut y = x.powf_p::<CheckOverflow<LessPrecision<P>, false>>(Self::splat(
                    E::ONE / E::from_int(N as thermite::LargeInt),
                ));

                let y_n = y.powi_p::<P>(N as i32);

                let np1 = Self::splat(E::from_int((N + 1) as thermite::LargeInt));
                let nm1 = Self::splat(E::from_int((N - 1) as thermite::LargeInt));

                let n = y * (x - y_n); // half of numerator
                let d = y_n.mul_adde(np1, x * nm1);

                y += (n + n) / d;

                y
            }
        };

        // denom = (1 + |x|^N)^(1/N); f'(x) = 1 / (pre_root * denom)
        let mut y;
        let mut dy;

        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            let inv_denom = denom.reciprocal_p::<P>();
            y = self * inv_denom;
            dy = inv_denom / pre_root;
        } else {
            y = self / denom;
            dy = (pre_root * denom).reciprocal_p::<P>();
        }

        if const { P::POLICY.check_overflow } {
            let is_infinite = pre_root.is_infinite();

            y = is_infinite.select(self.signum(), y);
            dy = dy.nz(is_infinite); // zero if is_infinite
        }

        (y, dy)
    }

    #[inline(always)]
    fn algebraic_swish_d<P: Policy>(self) -> (Self, Self) {
        let x = self;

        if const { Self::HAS_TRUE_FMA } {
            if const { Self::HAS_APPROX_RSQRT } {
                let a = x.mul_add(x, Self::ONE);
                let y0 = a.rsqrt();
                let ay2 = a * y0 * y0;
                let ch = ay2.nmul_add(Self::HALF, Self::splat(<E as FloatElement>::ConstRatio::<3, 2>::VALUE));
                let r_inv = y0 * ch; // Newton-refined 1/sqrt(a)
                let q = x * r_inv;
                let xh = Self::HALF * x;
                let y = q.mul_add(xh, xh);

                let inv_a = r_inv * r_inv;
                let qa = q.mul_add(inv_a, q); // q + q/a
                let dy = qa.mul_add(Self::HALF, Self::HALF); // (qa + 1)/2

                (y, dy)
            } else {
                let a = x.mul_add(x, Self::ONE);
                let q = x / a.sqrt();
                let xh = x * Self::HALF;
                let y = q.mul_add(xh, xh);

                let inv_a = a.reciprocal_p::<P>();
                let qa = q.mul_add(inv_a, q);
                let dy = qa.mul_add(Self::HALF, Self::HALF);

                (y, dy)
            }
        } else if const { Self::HAS_APPROX_RCP } {
            let a = x * x + Self::ONE;
            let y0 = a.rsqrt();
            let ay2 = a * y0 * y0;
            let c = Self::splat(<E as FloatElement>::ConstInt::<3>::VALUE) - ay2;
            let r_inv_2 = y0 * c; // = 2 * (Newton-refined 1/sqrt(a))
            let hxy1 = Self::splat(<E as FloatElement>::ConstRatio::<1, 4>::VALUE) * (x * r_inv_2); // = q/2
            let w = Self::HALF + hxy1; // = (1 + q)/2
            let y = x * w;

            let inv_a = Self::splat(<E as FloatElement>::ConstRatio::<1, 4>::VALUE) * (r_inv_2 * r_inv_2);
            let dy = w + hxy1 * inv_a;

            (y, dy)
        } else {
            let a = x * x + Self::ONE;
            let q = x / a.sqrt();
            let q1 = q + Self::ONE;
            let y = x * Self::HALF * q1;

            let dy = Self::HALF * (q1 + q / a);

            (y, dy)
        }
    }
}
