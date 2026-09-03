//! The Lambert W function over C, principal and `-1` branches, by a series seed and Halley
//! iteration.

use thermite::math::policy::Policy;
use thermite::math::{FloatConsts, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;

use crate::Complex;
use crate::vector::RealFloatVector;

/// Shared complex Lambert W, returning `$(W_0, W_{-1})$`.
///
/// Both are ordinary branches of `$W$` over C (unlike the real case, where `$W_{-1}$`
/// exists only on `$[-1/e, 0)$`). A Halley iteration from a per-region initial guess.
/// Halley is cubic, so a handful of steps suffices once the guess is in the right
/// basin. Picking that basin is the whole difficulty.
///
/// * `c` is `[11/72, r^2]`: the third Puiseux coefficient, and the squared radius
///   around `$-1/e$` inside which that series is used.
/// * `iters` is the Halley count.
///
/// Accurate away from the branch cuts. Near the cut on `$(-\infty, -1/e)$` the two
/// branches exchange values and a guess can land in the wrong basin, so results there
/// follow whichever branch the iteration converged to, not the principal labelling.
#[inline(always)]
pub(crate) fn lambert_w_impl<P: Policy, V: RealFloatVector>(
    z: Complex<V>,
    c: &[V::Element; 2],
    iters: usize,
) -> (Complex<V>, Complex<V>) {
    let e = <V as FloatConsts>::E;

    // --- Branch-point series in p = sqrt(2(ez + 1)) ---
    // The two branches meet at z = -1/e, where p = 0 and both equal -1:
    //   W_0    = -1 + p - p^2/3 + 11 p^3/72
    //   W_{-1} = -1 - p - p^2/3 - 11 p^3/72
    // The even term keeps its sign, the odd ones flip.
    let ez1 = z.mul_adde(e, Complex::ONE);
    let p = (ez1 + ez1).sqrt();
    let p2 = p.square();
    let p3 = p2 * p;

    // Both coefficients are real, so these are the componentwise (single-rounding) FMA.
    let odd = p3.mul_adde(V::splat(c[0]), p);
    let even = p2.nmul_adde(<V as FloatConsts>::FRAC_1_3, Complex::NEG_ONE);

    let w0_branch = even + odd;
    let wm1_branch = even - odd;

    // --- Logarithmic guess, W_k(z) ~ L1 - L2 + L2/L1, L1 = ln z + 2*pi*i*k ---
    let lnz = z.ln_p::<P>();
    let l1_0 = lnz;
    let l1_m1 = Complex::new(lnz.re, lnz.im - <V as FloatConsts>::TAU);

    let l2_0 = l1_0.ln_p::<P>();
    let l2_m1 = l1_m1.ln_p::<P>();

    let w0_log = l1_0 - l2_0 + l2_0 / l1_0;
    let wm1_log = l1_m1 - l2_m1 + l2_m1 / l1_m1;

    // --- Middle region for W_0 ---
    // ez/(2 + ez) is exact at z = -1/e and at z = 0 and decent between. It covers the
    // gap where L1 = ln z passes through zero and the logarithmic guess degenerates.
    let ez = z * e;
    let w0_mid = ez / (ez + Complex::real(V::TWO));

    // --- Region selection ---
    let d = Complex::new(z.re + <V as FloatConsts>::FRAC_NEG_1_E.abs(), z.im);
    let near_branch = d.norm_sqr().cmp_lt(V::splat(c[1]));
    let far = z.norm_sqr().cmp_gt(e * e);

    let mut w0 = near_branch.select(w0_branch, far.select(w0_log, w0_mid));
    let mut wm1 = near_branch.select(wm1_branch, wm1_log);

    // --- Halley ---
    let mut n = 0;
    while n < iters {
        n += 1;
        w0 = halley::<P, V>(w0, z);
        wm1 = halley::<P, V>(wm1, z);
    }

    // --- Edge cases ---
    if const { P::POLICY.check_overflow } {
        let zero = z.re.is_zero() & z.im.is_zero();
        let nonzero = !zero;

        // W_0(0) = 0; W_{-1}(0) = -inf, the real convention for the limit.
        w0 = nonzero.select(w0, Complex::ZERO);
        wm1 = nonzero.select(wm1, Complex::new(V::NEG_INFINITY, V::ZERO));
    }

    (w0, wm1)
}

/// One Halley step for `w e^w = z`, written through `e^{-w}` so that neither tail
/// overflows:
///
/// ```text
/// g = w - z e^{-w}
/// d = (w^2 + 2w + 2) + (w + 2) z e^{-w}
/// w' = w - 2(w + 1) g / d
/// ```
#[inline(always)]
fn halley<P: Policy, V: RealFloatVector>(w: Complex<V>, z: Complex<V>) -> Complex<V> {
    let enw = (-w).exp_p::<P>();
    let wp1 = w + Complex::ONE;
    let zenw = z * enw;

    // w^2 + 2w + 2 = (w + 1)^2 + 1. Each product below folds its addend into the
    // complex FMA rather than rounding the product first.
    let q = wp1.mul_adde(wp1, Complex::ONE);

    let g = w - zenw;
    let d = (wp1 + Complex::ONE).mul_adde(zenw, q);

    (wp1 + wp1).nmul_adde(g / d, w)
}
