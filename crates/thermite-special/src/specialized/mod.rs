#![allow(clippy::excessive_precision)]

use thermite::{
    mask::GenericMask,
    math::{
        CoreMathWithPolicy as _, FloatConsts, PrimalProjection, TranscendentalMathWithPolicy as _,
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
    /// complex `cmp_lt` means. That is a lexicographic sort order, and reading it as a
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

/// `1/(k+1)`, the Laguerre recurrence's leading coefficient, as one scalar divide.
///
/// The divisor is a small loop-invariant integer, so this sits off the recurrence's
/// critical path (and folds to a literal outright when the degree is a const generic).
/// Dividing the vector instead would put a full divide latency straight into the
/// dependency chain (roughly 14 cycles per step against 4 for the multiply) to save
/// half an ulp on a step that already carries several.
#[inline(always)]
fn laguerre_rcp<E: FloatElement>(k: usize) -> E {
    E::from_ratio(1, (k + 1) as thermite::LargeInt)
}

pub trait SpecializedSpecialMath<E>: thermite::math::specialized::SpecializedTranscendentalMath<E> {
    /// Per-arithmetic details of the [`expint`](Self::expint) kernel. Almost always
    /// `Self`, with an empty [`ExpIntDetails`] impl taking every default.
    type ExpIntDetails: ExpIntDetails<E, Self>;

    /// Largest integer weight for which `laguerre_function_i` seeds by the direct product
    /// `x^{alpha/2} / sqrt(alpha!)` (a scalar factorial, `powi`, at most one `sqrt`) instead
    /// of the general `exp(alpha/2 ln x - lgamma(alpha+1)/2)`. `0` disables it.
    ///
    /// The bound is per arithmetic because it is set by the exponent range: `alpha!` must
    /// stay finite, and `x^{alpha/2}` must stay finite wherever `e^{-x/4}` is still
    /// non-zero (so `inf * 0` cannot arise). Those give 170 / 29 for binary64 / binary32;
    /// see `generic::laguerre::product_seed`. The default is the safe "never".
    ///
    /// It lives on the trait rather than as a const generic on the kernel so that the
    /// composites can inherit it: `Dual<V, N>` is one blanket impl with no binary32/64
    /// split to hang a literal on, and forwarding `V`'s value is the only way it keeps the
    /// product seed at all.
    const LAGUERRE_PRODUCT_SEED_CAP: i32 = 0;

    fn erf<P: Policy>(self) -> Self;

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        Self::ONE - self.erf_p::<P>()
    }

    /// `$e^{x^2}\operatorname{erfc}(x)$`, which does not underflow where `erfc` does.
    ///
    /// This default is the direct form, whose failure is what the function exists to
    /// fix: `$e^{x^2}$` overflows just where `erfc` underflows, so it
    /// is useful only for `$|x|$` under about 26.6 (binary64) or 9.3 (binary32). The
    /// real backends override it with the imaginary-axis Weideman evaluation, which has
    /// no such limit (see `generic::erfcx`). Element types without a Weideman table
    /// (`Compensated`) take this and inherit its range.
    #[inline(always)]
    fn erfcx<P: Policy>(self) -> Self {
        (self * self).exp_p::<P>() * self.erfc_p::<P>()
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

        // Bootstrap j=1 step: a_1 = 1, b_1 = x+1. D_0 is only ever read here, so
        // it stays a comment rather than an initializer the next line overwrites.
        let mut cf_d = {
            let b1 = x + Self::ONE;
            // D_1 = 1/(b_1 + a_1*D_0) = 1/(x+1), since D_0 = 0
            let d1 = b1.reciprocal_p::<P>();
            // C_1 = b_1 + a_1/C_0 = (x+1) + 1/tiny ≈ 1/tiny
            cf_c = b1 + cf_c.reciprocal_p::<P>();
            let delta = cf_c * d1;
            cf_f *= delta; // tiny * (1/tiny)/(x+1) ≈ 1/(x+1)
            d1
        };

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
    /// here. It exists only so that `digamma` is differentiable (forward-mode AD over
    /// the Gamma family needs `psi_1` the way `ln Gamma` needs `psi`), and keeping it
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

            // H_{k+1} = 2x H_k - 2k H_{k-1}
            let next0 = x.mul_sube(p1, cf * p0);
            let next = next0 + next0; // 2 * next0

            // Freeze BOTH halves of the pair on lanes that have reached their own
            // degree. `hermite`'s unconditional (p0, p1) swap cannot be reused here:
            // on a retired lane it moves H_{k-1} into p1, and a select that only
            // guards p1 then preserves that instead of the lane's answer.
            p0 = cont.select(p1, p0);
            p1 = cont.select(next, p1);

            c += i1;
            cf += Self::ONE;
        }

        n_is_zero.select(Self::ONE, p1)
    }

    #[inline(always)]
    fn hermite_function<P: Policy, const N: usize>(mut x: Self) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_x) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new_x[0];
        }

        generic::hermite::hermite_function::<P, _, _, N>(x)
    }

    #[inline(always)]
    fn hermite_function_series<P: Policy, const N: usize>(self, coeffs: &[Self::Element; N]) -> Self {
        generic::hermite::hermite_function_series::<P, _, _, N>(self, coeffs)
    }

    #[inline(always)]
    fn laguerre<P: Policy, const N: usize>(mut x: Self, mut alpha: Self) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([x, alpha]) {
            x = new[0];
            alpha = new[1];
        }

        if const { N == 0 } {
            return Self::ONE;
        }

        let mut p0 = Self::ONE; // L_0 = 1
        let mut p1 = (Self::ONE + alpha) - x; // L_1 = 1 + a - x

        let mut k = 1;
        let mut kf = Self::ONE; // k as a float, counted alongside to avoid a convert per step

        while k < N {
            // (k+1) L_{k+1} = (2k + a + 1 - x) L_k - (k + a) L_{k-1}
            let b = ((kf + kf) + Self::ONE + alpha) - x;
            let c = kf + alpha;

            let next = b.mul_sube(p1, c * p0) * Self::splat(laguerre_rcp::<E>(k));

            p0 = p1;
            p1 = next;

            k += 1;
            kf += Self::ONE;
        }

        p1
    }

    #[inline(always)]
    fn laguerrev<P: Policy>(mut x: Self, mut alpha: Self, n: Self::Unsigned) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([x, alpha]) {
            x = new[0];
            alpha = new[1];
        }

        let i1 = Self::Unsigned::ONE;
        let n_is_zero = n.cmp_eq(Self::Unsigned::ZERO);

        let mut c = i1;

        let mut k = 1;
        let mut kf = Self::ONE;

        let mut p0 = Self::ONE;
        let mut p1 = (Self::ONE + alpha) - x;

        loop {
            let cont = c.cmp_lt(n);

            if cont.none() {
                break;
            }

            let b = ((kf + kf) + Self::ONE + alpha) - x;
            let ck = kf + alpha;

            let next = b.mul_sube(p1, ck * p0) * Self::splat(laguerre_rcp::<E>(k));

            // Freeze BOTH halves of the pair on lanes that have reached their own degree.
            // Carrying `p0` forward unconditionally would leave a finished lane holding
            // `L_{k-1}` in `p1` on the next step instead of its answer.
            p0 = cont.select(p1, p0);
            p1 = cont.select(next, p1);

            c += i1;
            k += 1;
            kf += Self::ONE;
        }

        n_is_zero.select(Self::ONE, p1)
    }

    #[inline(always)]
    fn laguerre_function<P: Policy, const N: usize>(mut x: Self, mut alpha: Self) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([x, alpha]) {
            x = new[0];
            alpha = new[1];
        }

        generic::laguerre::laguerre_function::<P, _, _, N, false>(x, alpha, 0)
    }

    #[inline(always)]
    fn laguerre_function_i<P: Policy, const N: usize>(mut x: Self, alpha: i32) -> Self {
        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([x]) {
            x = new[0];
        }

        generic::laguerre::laguerre_function::<P, _, _, N, true>(x, Self::ZERO, alpha)
    }

    #[inline(always)]
    fn poisson_pmf<P: Policy>(self, lambda: Self) -> Self {
        generic::poisson::poisson_pmf::<P, _, _, false>(self, lambda)
    }

    #[inline(always)]
    fn poisson_log_pmf<P: Policy>(self, lambda: Self) -> Self {
        generic::poisson::poisson_pmf::<P, _, _, true>(self, lambda)
    }

    #[inline(always)]
    fn laguerre_function_series<P: Policy, const N: usize>(self, alpha: Self, coeffs: &[Self::Element; N]) -> Self {
        generic::laguerre::laguerre_function_series::<P, _, _, N, false>(self, alpha, 0, coeffs)
    }

    #[inline(always)]
    fn laguerre_function_series_i<P: Policy, const N: usize>(self, alpha: i32, coeffs: &[Self::Element; N]) -> Self {
        generic::laguerre::laguerre_function_series::<P, _, _, N, true>(self, Self::ZERO, alpha, coeffs)
    }

    #[inline(always)]
    fn chebyshev<P: Policy, const K: usize, const N: usize>(self, coeffs: &[Self::Element; N]) -> Self {
        // Plain Clenshaw. Real vectors override this in `ps.rs`/`pd.rs` to pass `true` for
        // the kernel's `REINSCH` parameter, which buys accuracy near `$x = \pm 1$` under a
        // `Best`-or-better policy; `Complex` and the composites take this default, since the
        // endpoint form needs a real `copysign` and a meaningful nearest endpoint.
        generic::chebyshev::chebyshev_series::<P, _, _, K, N, false>(self, coeffs)
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

    #[inline(always)]
    fn lbeta<P: Policy>(a: Self, b: Self) -> Self {
        // ln|B(a,b)| = ln|G(a)| + ln|G(b)| - ln|G(a+b)|. The log form is the only one with
        // the range to cover f32 arguments: the Gamma product overflows f64 past ~171
        // while B itself stays perfectly ordinary.
        Self::lgamma::<P>(a) + Self::lgamma::<P>(b) - Self::lgamma::<P>(a + b)
    }

    #[inline(always)]
    fn logit<P: Policy>(self) -> Self {
        // ln(p) - ln(1 - p), with ln_1p carrying the second term so small p stays accurate.
        // Nothing can be done for p near 1 from this argument alone. See `logit_1m`.
        Self::ln::<P>(self) - Self::ln_1p::<P>(-self)
    }

    #[inline(always)]
    fn logit_1m<P: Policy>(self) -> Self {
        // logit(1 - q) = ln(1 - q) - ln(q), in terms of the complement throughout. Here q is
        // the small quantity, so ln_1p is at its most accurate exactly where `logit` is worst.
        Self::ln_1p::<P>(-self) - Self::ln::<P>(self)
    }

    #[inline(always)]
    fn planck<P: Policy>(self) -> Self {
        // x^3/(e^x - 1) = x^2 / phi_1(x). phi_1 is 1 at the origin, so the 0/0 of the direct
        // quotient never forms and the x^2 limit falls out on its own.
        (self * self).approx_div_p::<P>(Self::phi_p::<P, 1>(self))
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn legendre0<P: Policy, const N: u32>(x: Self, n: u32) -> Self {
        macro_rules! c { ($n:literal / $d:literal) => { Self::splat(<E as FloatElement>::ConstRatio::<{ $n }, { $d }>::VALUE) }; }

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

    #[inline(always)]
    fn legendre_series<P: Policy, const N: usize>(self, coeffs: &[Self::Element; N]) -> Self {
        // Plain Clenshaw at every policy, as the kernel has no policy-dependent path.
        generic::legendre::legendre_series::<_, _, N>(self, coeffs)
    }

    #[inline(always)]
    fn zernike_r<P: Policy>(mut rho: Self, n: u32, m: u32) -> Self {
        // A mode that does not exist contributes nothing, rather than whatever a
        // recurrence run outside its range happens to produce.
        if thermite::unlikely(m > n || (n - m) & 1 == 1) {
            return Self::ZERO;
        }

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new) = FlushDenormals::<P>::flush_denormals([rho]) {
            rho = new[0];
        }

        // R_n^m(rho) = rho^m * Q_{(n-m)/2, m}(rho^2), the shifted Jacobi identity with the
        // change of variable folded into the recurrence coefficients. See the trait docs
        // for why this rather than the direct factorial sum, and where the usual (-1)^k
        // prefactor went; `reduced_radial_impl` for why not a general `jacobi` call.
        let radial = generic::zernike::reduced_radial_impl::<E, Self>(rho.square(), (n - m) >> 1, m);

        if m == 0 {
            radial
        } else {
            radial * Self::powi::<P>(rho, m as i32)
        }
    }

    #[inline(always)]
    fn zernike<P: Policy, const NORM: u8>(rho: Self, theta: Self, n: u32, m: i32) -> Self {
        const {
            assert!(
                NORM == crate::ZERNIKE_UNIT_PEAK || NORM == crate::ZERNIKE_ORTHONORMAL,
                "zernike: NORM must be ZERNIKE_UNIT_PEAK or ZERNIKE_ORTHONORMAL"
            );
        }

        let am = m.unsigned_abs();

        let radial = Self::zernike_r::<P>(rho, n, am);

        let z = if m == 0 {
            radial // cos(0) = 1
        } else {
            let (sin, cos) = Self::sin_cos::<P>(theta * Self::splat(E::from_int(am as thermite::LargeInt)));

            radial * if m > 0 { cos } else { sin }
        };

        if const { NORM == crate::ZERNIKE_UNIT_PEAK } {
            return z;
        }

        // N_n^m = sqrt(2(n+1) / (1 + delta_{m,0})). Both the radicand and the root are
        // exact in the element type for any n a pupil fit will reach, and n is
        // loop-invariant, so this is a splat of a constant rather than a vector sqrt.
        let radicand = if m == 0 { n + 1 } else { 2 * (n + 1) };

        z * Self::splat(FloatElement::sqrt(E::from_int(radicand as thermite::LargeInt)))
    }

    #[inline(always)]
    fn zernike_basis<P: Policy, const L: usize, const NORM: u8, const N: usize>(x: Self, y: Self, out: &mut [Self; N]) {
        generic::zernike::zernike_basis_impl::<P, E, Self, L, NORM, N>(x, y, out);
    }

    fn lambert_w<P: Policy>(self) -> (Self, Self);

    // TEMP(bessel_j): disabled until orders beyond J_0 exist. See the note in lib.rs.
    //fn bessel_j<P: Policy, const N: usize>(self) -> Self;

    #[inline(always)]
    fn phi<P: Policy, const N: usize>(self) -> Self {
        // Element-agnostic form: the series arm runs until it converges to
        // `Self::EPSILON`, capped by the policy's iteration budget. The f32/f64
        // backends override this with a compile-time term count.
        generic::phi::phi_internal::<Self, E, P, N, true>(self, P::POLICY.max_iterations)
    }
}

// The Carlson / Legendre entry points are kind-dispatched (`SpecialMath::carlson` / `::ellint`),
// generated by decl_math!'s `@kinds` blocks. They call the request struct's `eval` directly, so
// they need no method here. The request structs and their traits are re-exported below.
// `EllipticConsts` is re-exported because `EllipticKind` is bounded on it: any
// generic caller of `ellint`/`carlson` has to name it in a where-clause, so
// leaving it unreachable made those two functions uncallable from generic code.
pub use generic::elliptic::{
    CarlsonKind, CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE, EllintEInc,
    EllintF, EllintK, EllintPi, EllintPiInc, EllipticConsts, EllipticKind, WrapTo,
};

// The spherical-harmonic kernels, ahead of their `RealSpecialMath` wiring. Re-exported
// the same way as the elliptic internals: the tables trait must be nameable by generic
// callers, and the tests drive the kernels through this path.
pub use generic::sh::{
    MAX_DEGREE as MAX_SH_DEGREE, ShConsts, ShTable, sh_d_impl, sh_eval_d_impl, sh_eval_impl, sh_eval_lifted_impl,
    sh_eval_mixed_impl, sh_impl, sh_table_impl,
};

// The batch Zernike kernel's unrolled-degree cap, named in `zernike_basis`'s docs as the
// point past which it stops being straight-line code.
pub use generic::zernike::{MAX_DEGREE as MAX_ZERNIKE_DEGREE, zernike_basis_d_impl, zernike_basis_impl};

/// Specialized implementation trait for real-only special math functions.
///
/// Extends [`SpecializedSpecialMath`] with functions that have no meaningful
/// complex analogue (e.g. functions using the real absolute value, or functions
/// that are inverses of real-domain-only operations).
pub trait SpecializedRealSpecialMath<E>: SpecializedSpecialMath<E> {
    fn erfinv<P: Policy>(self) -> Self;
    fn probit<P: Policy>(self) -> Self;

    /// `(x^lambda - 1)/lambda`, `ln x` at `lambda = 0`.
    ///
    /// `powf_m1` builds `x^lambda - 1` without forming `x^lambda`, so the division by lambda
    /// is the *whole* algorithm: there is no cancellation left to protect against and hence
    /// no near-zero series, which the obvious `(pow(x, l) - 1)/l` spelling would need by
    /// `l = 1e-8`. Verified a few ulp from `lambda = 1e-300` outward.
    ///
    /// `lambda` is a fitted parameter, so it is uniform across a vector in every real use and
    /// the two uniform branches are what actually run. The blend is there for correctness on
    /// a mixed vector, not for speed.
    /// The domain edge `x = 0` needs no guard here. `powf_m1(0, lambda)` is `-1` for
    /// `lambda > 0` and `+inf` below, so the division delivers the conventional `-1/lambda` and
    /// `-inf` on its own, and more accurately than a `reciprocal` would. The Best-tier Dekker
    /// residual in `powf_m1` carries the edge itself, so nothing is patched up here.
    #[inline(always)]
    fn boxcox<P: Policy>(self, lambda: Self) -> Self {
        let at_zero = lambda.is_zero();

        if const { !P::POLICY.avoid_branching } && at_zero.all() {
            Self::ln::<P>(self)
        } else if const { !P::POLICY.avoid_branching } && at_zero.none() {
            Self::powf_m1::<P>(self, lambda) / lambda
        } else {
            at_zero.select(Self::ln::<P>(self), Self::powf_m1::<P>(self, lambda) / lambda)
        }
    }

    /// `((1 + x)^lambda - 1)/lambda`, `ln(1 + x)` at `lambda = 0`.
    ///
    /// Structurally identical to [`boxcox`](Self::boxcox), over `compound_m1` instead of
    /// `powf_m1` so that `x` near zero keeps its low bits, which is the only reason to have
    /// it, and the reason Yeo-Johnson is built on it.
    #[inline(always)]
    fn boxcox_1p<P: Policy>(self, lambda: Self) -> Self {
        let at_zero = lambda.is_zero();

        if const { !P::POLICY.avoid_branching } && at_zero.all() {
            Self::ln_1p::<P>(self)
        } else if const { !P::POLICY.avoid_branching } && at_zero.none() {
            Self::compound_m1::<P>(self, lambda) / lambda
        } else {
            at_zero.select(Self::ln_1p::<P>(self), Self::compound_m1::<P>(self, lambda) / lambda)
        }
    }

    /// `(lambda*y + 1)^(1/lambda)`, `e^y` at `lambda = 0`. The inverse of
    /// [`boxcox`](Self::boxcox).
    ///
    /// Evaluated as `exp(ln1p(lambda*y)/lambda)` rather than `powf`:
    /// `lambda*y` is small exactly where the forward transform's `lambda` is, so `1 + lambda*y`
    /// would round it away and the whole reason `boxcox` is accurate near `lambda = 0` would
    /// be undone on the way back.
    #[inline(always)]
    fn inv_boxcox<P: Policy>(self, lambda: Self) -> Self {
        let at_zero = lambda.is_zero();

        if const { !P::POLICY.avoid_branching } && at_zero.all() {
            Self::exp::<P>(self)
        } else if const { !P::POLICY.avoid_branching } && at_zero.none() {
            Self::exp::<P>(Self::ln_1p::<P>(lambda * self) / lambda)
        } else {
            at_zero.select(
                Self::exp::<P>(self),
                Self::exp::<P>(Self::ln_1p::<P>(lambda * self) / lambda),
            )
        }
    }

    /// `(lambda*y + 1)^(1/lambda) - 1`, `e^y - 1` at `lambda = 0`. The inverse of
    /// [`boxcox_1p`](Self::boxcox_1p).
    ///
    /// Same exponent as [`inv_boxcox`](Self::inv_boxcox) with `expm1` outside it, so the
    /// result keeps its relative accuracy where it is near zero, which, this being the
    /// inverse of a transform of data centered near zero, is the ordinary case.
    #[inline(always)]
    fn inv_boxcox_1p<P: Policy>(self, lambda: Self) -> Self {
        let at_zero = lambda.is_zero();

        if const { !P::POLICY.avoid_branching } && at_zero.all() {
            Self::exp_m1::<P>(self)
        } else if const { !P::POLICY.avoid_branching } && at_zero.none() {
            Self::exp_m1::<P>(Self::ln_1p::<P>(lambda * self) / lambda)
        } else {
            at_zero.select(
                Self::exp_m1::<P>(self),
                Self::exp_m1::<P>(Self::ln_1p::<P>(lambda * self) / lambda),
            )
        }
    }

    /// The Yeo-Johnson transform of `y = self` with parameter `lambda`.
    ///
    /// Four cases in the literature, one kernel here: the transform is odd about the origin
    /// in the sense that the `y < 0` branch is the `y >= 0` branch applied to `|y|` with
    /// `lambda` reflected to `2 - lambda` and the result negated. Folding the sign out first
    /// collapses both `ln` special cases (`lambda = 0` above zero, `lambda = 2` below) into
    /// the single `lambda = 0` seam that [`boxcox_1p`](Self::boxcox_1p) already handles.
    #[inline(always)]
    fn yeo_johnson<P: Policy>(self, lambda: Self) -> Self {
        let neg = self.cmp_lt(Self::ZERO);
        let reflected = neg.select(Self::TWO - lambda, lambda);
        let r = Self::boxcox_1p::<P>(self.abs(), reflected);

        neg.select(-r, r)
    }

    /// The inverse Yeo-Johnson transform. The same sign fold as
    /// [`yeo_johnson`](Self::yeo_johnson), over [`inv_boxcox_1p`](Self::inv_boxcox_1p).
    ///
    /// The transform is monotone increasing and fixes the origin, so the branch condition on
    /// the way back is the sign of the *transformed* value, which is the sign of `y`.
    #[inline(always)]
    fn inv_yeo_johnson<P: Policy>(self, lambda: Self) -> Self {
        let neg = self.cmp_lt(Self::ZERO);
        let reflected = neg.select(Self::TWO - lambda, lambda);
        let r = Self::inv_boxcox_1p::<P>(self.abs(), reflected);

        neg.select(-r, r)
    }

    fn langevin<P: Policy>(self) -> Self;
    fn inv_langevin<P: Policy>(self) -> Self;
    fn langevin_1m<P: Policy>(self) -> Self;
    fn inv_langevin_1m<P: Policy>(self) -> Self;

    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> Self {
        let alpha_x = alpha * self;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        let erf = alpha_x.scale(FloatConsts::FRAC_1_SQRT_2).erf_p::<P>();

        if Self::HAS_TRUE_FMA {
            // if we have true FMA, we can maintain precision while avoiding extra work.
            let half_x = self.scale(E::ConstRatio::<1, 2>::VALUE);
            half_x.mul_add(erf, half_x) // 0.5 * x + 0.5 * x * erf
        } else {
            self.scale(E::ConstRatio::<1, 2>::VALUE) * (Self::ONE + erf)
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

    /// Fills a runtime coefficient table for degree `L` and phase `CS`. See
    /// [`sh_impl`] for the conventions, layout, and algorithm.
    ///
    /// The direction-independent half of the work, split out so a caller sweeping many
    /// directions pays it once: pair it with [`spherical_harmonics_with`](Self::spherical_harmonics_with).
    /// The table records its own phase, which is why the evaluators take no `CS`.
    ///
    /// The default computes every coefficient from its closed form in `l` and `m`
    /// (two `sqrt` and two divisions apiece) using nothing but `FloatVector`
    /// arithmetic, so it works at any degree and on any element type. Real `f32`/`f64`
    /// vectors override it to splat the compile-time table instead whenever
    /// `L <= MAX_DEGREE`, which removes the arithmetic entirely.
    #[inline(always)]
    fn spherical_harmonics_table<P: Policy, const L: usize, const N: usize, const CS: bool>(
        table: &mut ShTable<Self::Primal, N>,
    ) {
        generic::sh::sh_table_impl::<Self::Primal, L, N, CS>(table);
    }

    /// Evaluates all harmonics through degree `L` from a table built by
    /// [`spherical_harmonics_table`](Self::spherical_harmonics_table).
    ///
    /// The default lifts each `Self::Primal` coefficient through `from_primal` as it
    /// is read. That is the identity for types that are their own primal, so they
    /// keep the fused single-type kernel. Composites with a cheaper mixed multiply
    /// (`Dual`) override this.
    #[inline(always)]
    fn spherical_harmonics_with<P: Policy, const L: usize, const N: usize>(
        table: &ShTable<Self::Primal, N>,
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
    ) {
        generic::sh::sh_eval_lifted_impl::<Self, L, N>(table, x, y, z, out);
    }

    /// The one-shot form: build a table and evaluate it.
    ///
    /// This default composes [`spherical_harmonics_table`](Self::spherical_harmonics_table)
    /// with [`spherical_harmonics_with`](Self::spherical_harmonics_with), so it needs no
    /// compile-time table and works on every element type and at any degree. Real
    /// `f32`/`f64` vectors override it with the fully-unrolled kernel for
    /// `L <= MAX_DEGREE`.
    ///
    /// A caller in a loop over directions should build the table once and call
    /// `spherical_harmonics_with` instead. This rebuilds it on every invocation, and
    /// the table is the expensive part.
    #[inline(always)]
    fn spherical_harmonics<P: Policy, const L: usize, const N: usize, const CS: bool>(
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
    ) {
        let mut table = ShTable::<Self::Primal, N>::zeroed();
        Self::spherical_harmonics_table::<P, L, N, CS>(&mut table);
        Self::spherical_harmonics_with::<P, L, N>(&table, x, y, z, out);
    }
}

/// Value-and-derivative (`_d`) forms of the activation functions, for single-value real numbers.
///
/// Every method is a provided default returning `(value, derivative)`; the `value` matches the
/// like-named value-only function in [`SpecializedSpecialMath`] / [`SpecializedRealSpecialMath`].
/// Implemented (as an empty impl) only for primal types -- *not* for derivative-carrying numbers
/// like `Dual`, which obtain the derivative from the value form via automatic differentiation.
pub trait SpecializedRealPrimalMath<E>: SpecializedRealSpecialMath<E> + PrimalProjection<Primal = Self> {
    /// [`spherical_harmonics_with`](SpecializedRealSpecialMath::spherical_harmonics_with)
    /// plus the ambient Cartesian gradients, from a prebuilt table.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn spherical_harmonics_d_with<P: Policy, const L: usize, const N: usize>(
        table: &ShTable<Self, N>,
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
        ddx: &mut [Self; N],
        ddy: &mut [Self; N],
        ddz: &mut [Self; N],
    ) {
        generic::sh::sh_eval_d_impl::<Self, L, N>(table, x, y, z, out, ddx, ddy, ddz);
    }

    /// [`spherical_harmonics`](Self::spherical_harmonics) plus the ambient Cartesian
    /// gradient of every harmonic. See [`sh_d_impl`] for the gradient semantics.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn spherical_harmonics_d<P: Policy, const L: usize, const N: usize, const CS: bool>(
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
        ddx: &mut [Self; N],
        ddy: &mut [Self; N],
        ddz: &mut [Self; N],
    ) {
        let mut table = ShTable::<Self, N>::zeroed();
        Self::spherical_harmonics_table::<P, L, N, CS>(&mut table);
        Self::spherical_harmonics_d_with::<P, L, N>(&table, x, y, z, out, ddx, ddy, ddz);
    }

    /// [`zernike_basis`](SpecializedSpecialMath::zernike_basis) plus the Cartesian
    /// gradient of every mode. See [`zernike_basis_d_impl`] for the algorithm.
    #[inline(always)]
    fn zernike_basis_d<P: Policy, const L: usize, const NORM: u8, const N: usize>(
        x: Self,
        y: Self,
        out: &mut [Self; N],
        ddx: &mut [Self; N],
        ddy: &mut [Self; N],
    ) {
        generic::zernike::zernike_basis_d_impl::<P, E, Self, L, NORM, N>(x, y, out, ddx, ddy);
    }

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
            let half_x = self.scale(E::ConstRatio::<1, 2>::VALUE);
            half_x.mul_add(erf, half_x) // 0.5 * x + 0.5 * x * erf
        } else {
            self.scale(E::ConstRatio::<1, 2>::VALUE) * (Self::ONE + erf)
        };

        let dy = (alpha_x * alpha_x)
            .scale(E::ConstRatio::<{ -1 }, 2>::VALUE)
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

    /// `L(x)` and `L'(x)`. The derivative falls out of the value's own intermediates
    /// on both branches (see `generic::langevin`), so there is no default here that
    /// would recompute it.
    fn langevin_d<P: Policy>(self) -> (Self, Self);

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
