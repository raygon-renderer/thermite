//! Gauss-Legendre nodes and weights, one root per lane.
//!
//! The `n`-point rule integrates polynomials through degree `2n - 1` exactly on `[-1, 1]`:
//! `int f ~ sum_k w_k f(x_k)` with `x_k` the roots of `P_n` and `w_k = 2 / ((1 - x_k^2) P_n'(x_k)^2)`.
//!
//! # Shape
//!
//! The root index `k` is the lane's input (`0` is the largest root, `n - 1` the smallest,
//! `x_{n-1-k} = -x_k`), and `n` is uniform, so a packet of consecutive indices _is_ the rule:
//! a caller sweeps `k` in packets and stores nodes and weights as it goes. Every lane runs the
//! same `O(n)` recurrence per Newton step, so the packet costs one root, not `LANES`.
//!
//! # Method
//!
//! Tricomi's seed `cos(pi (k + 3/4) / (n + 1/2))` is within `O(1/n)` of the root and inside
//! its Newton basin (the roots of `P_n` are separated by about `pi/n` and the seed's error
//! is a fraction of that), then `newtons_method` on `P_n` with `P_n'` from the recurrence,
//! `P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1)`, two to four steps in practice, eight at most.
//! The recurrence `j P_j = (2j - 1) x P_{j-1} - (j - 1) P_{j-2}` runs with its two scalar
//! coefficients formed once per `j`, no vector division. The residual tolerance scales with
//! `n`, which is the recurrence's own rounding. Through `|P_n'|`, that is a few `eps`
//! absolute on interior nodes and far less at the ends, where `P_n'` is `O(n^2)`.

use thermite::{
    element::FloatElement,
    math::{
        TranscendentalMathWithPolicy as _,
        algorithms::newtons_method,
        policy::{
            Policy,
            policies::{LessPrecision, MaxIterations},
        },
    },
    prelude::*,
};

use super::ndtr::residual_tolerance;
use crate::specialized::SpecializedSpecialMath;

/// `(P_n(x), P_n'(x))` by the three-term recurrence.
#[inline(always)]
fn legendre_pair<E, V>(x: V, n: u32) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let mut p_prev = V::ONE;
    let mut p = x;
    let mut j = 2u32;
    while j <= n {
        // Scalar coefficients once per step: (2j - 1)/j and (j - 1)/j.
        let jf = E::from_int(j as i64);
        let a = V::splat(E::from_int((2 * j - 1) as i64) / jf);
        let b = V::splat(E::from_int((j - 1) as i64) / jf);
        let next = (a * x).mul_sube(p, b * p_prev);
        p_prev = p;
        p = next;
        j += 1;
    }
    // P_n' = n (x P_n - P_{n-1}) / (x^2 - 1). `x^2 - 1` as `-(1 - x)(1 + x)`: near the
    // end roots `1 - x` is exact (Sterbenz) where `1 - x*x` would lose `eps / (1 - x^2)`,
    // 150 ulp of the extreme weight at n = 33.
    let nf = V::splat(E::from_int(n as i64));
    let dp = -(nf * (x * p - p_prev)) / ((V::ONE - x) * (V::ONE + x));
    (p, dp)
}

/// The `k`-th node and weight of the `n`-point Gauss-Legendre rule, `k` per lane.
#[inline(always)]
pub fn gauss_legendre_impl<P, E, V>(k: V, n: u32) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let nf = V::splat(E::from_int(n as i64));
    let valid = valid_index(k, nf);

    if n == 1 {
        // P_1 = x: the one root is 0 with weight 2.
        return (valid.select(V::ZERO, V::NAN), valid.select(V::TWO, V::NAN));
    }

    // Tricomi's seed, one tier down.
    let theta = (k + V::splat(<E as FloatElement>::ConstRatio::<3, 4>::VALUE)) / (nf + V::HALF) * V::PI;
    let x0 = theta.cos_p::<LessPrecision<P>>();

    let tol = residual_tolerance::<P, E, V>(nf);
    let (x, _) = newtons_method::<V, MaxIterations<P, 8>, _>(x0, tol, valid, None, |x| legendre_pair::<E, V>(x, n));

    // The tolerance is `n` ulp of residual, but `|P_n'|` at the interior roots is only
    // about `sqrt(n)`, so Newton may stop `8 sqrt(n)` eps short there (56 eps at n = 64).
    // The pair is needed for the weight anyway, and one more step from it costs a division and
    // lands the node at the recurrence's own noise floor. The weight keeps this pair's
    // `P_n'`: its error is `P''/P'` times the step, which is negligible where the step is
    // large (interior, `P''/P' ~ 2x`) and the step is negligible where `P''/P'` is large.
    let (p, dp) = legendre_pair::<E, V>(x, n);
    let x = x - p / dp;

    let one_m_x2 = (V::ONE - x) * (V::ONE + x);
    let w = V::TWO / (one_m_x2 * dp * dp);

    (valid.select(x, V::NAN), valid.select(w, V::NAN))
}

/// `k` is a whole number in `[0, n)`.
#[inline(always)]
fn valid_index<E: FloatElement, V: FloatVector<Element = E>>(k: V, nf: V) -> V::Mask {
    k.cmp_ge(V::ZERO) & k.cmp_lt(nf) & k.cmp_eq(k.round())
}

/// Newton in `x`-space on a `(value, derivative)` pair, for the polynomial rules whose
/// magnitude varies too much across the interval for a function-space tolerance: a lane
/// stops when its own step is under `8 sqrt(n)` eps of `1 + |x|`, at most eight steps. No
/// polishing step: with an `x`-space stop the iterate is already at the recurrence's noise
/// floor, and a further step there is a random walk of about an ulp (measured on the
/// smallest root of `L_16^2`, where the floor itself is 7 ulp of 0.38 because the
/// intermediate `L_m` are about 150 against `L' = 129`).
#[inline(always)]
fn newton_x<E, V, F>(mut x: V, n: u32, active: V::Mask, mut f: F) -> (V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
    F: FnMut(V) -> (V, V),
{
    let tol = <V as FloatVector>::EPSILON * V::splat(E::from_int(8)) * V::splat(E::from_int(n as i64)).sqrt();
    // A lane freezes once its own step is small: extra steps at the noise floor walk it
    // off its answer, and a lane's count of steps must not depend on its packet-mates.
    let mut frozen = !active;
    let mut i = 0;
    while i < 8 {
        let (p, dp) = f(x);
        let dx = p / dp;
        x = frozen.select(x, x - dx);
        frozen |= dx.abs().cmp_le(tol * (V::ONE + x.abs()));
        if frozen.all() {
            break;
        }
        i += 1;
    }
    let (p, dp) = f(x);
    (x, p, dp)
}

/// The `k`-th node and weight of the `n`-point Gauss-Hermite rule (weight `e^{-x^2}` on
/// the line), `k` per lane, `k = 0` the largest root.
///
/// Seed: the WKB phase of the Hermite equation, `x = sqrt(2n+1) cos(phi)` with
/// `phi - sin(2 phi)/2 = 2 pi (k + 3/4)/(2n + 1)`, solved per lane by four Newton steps
/// from `phi = (3c/2)^{1/3}`. At the edge this reproduces the Airy constant
/// (`x_0 ~ sqrt(2n+1) - 1.856 (2n+1)^{-1/6}`) to three digits. The left half is the mirror
/// of the right. Newton then runs on `h_m = H_m / m!`, whose recurrence
/// `h_{m+1} = (2x h_m - 2 h_{m-1})/(m + 1)` has a scalar divisor and stays in range where
/// the raw `H_m` overflows at degree 48, with `h_n' = 2 h_{n-1}`. Weight
/// `w = sqrt(pi) 2^{n-1} / ((n-1)! n h_{n-1}(x_k)^2)`, the scalar factor a running product.
/// That factor underflows past `n = 170` in f64 and `n = 40` in f32, which bounds the rule.
#[inline(always)]
pub fn gauss_hermite_impl<P, E, V>(k: V, n: u32) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let nf = V::splat(E::from_int(n as i64));
    let valid = valid_index(k, nf);

    if n == 1 {
        return (valid.select(V::ZERO, V::NAN), valid.select(V::SQRT_PI, V::NAN));
    }

    // Right-half index and the sign to put back.
    let kk = k.min(nf - V::ONE - k);
    let neg = k.cmp_gt(kk);

    // phi from the WKB phase.
    let two_n_p1 = nf.mul_adde(V::TWO, V::ONE);
    let c = (kk + V::splat(<E as FloatElement>::ConstRatio::<3, 4>::VALUE)) * (V::TAU / two_n_p1);
    let mut phi = (c * V::splat(<E as FloatElement>::ConstRatio::<3, 2>::VALUE))
        .cbrt_p::<LessPrecision<P>>()
        .min(V::FRAC_PI_2);
    let mut i = 0;
    while i < 4 {
        let (s, co) = (phi + phi).sin_cos_p::<LessPrecision<P>>();
        phi -= (phi - s * V::HALF - c) / (V::ONE - co).max(<V as FloatVector>::EPSILON);
        i += 1;
    }
    let x0 = two_n_p1.sqrt() * phi.cos_p::<LessPrecision<P>>();

    let pair = |x: V| -> (V, V) {
        let mut h_prev = V::ONE;
        let mut h = x + x;
        let mut m = 1u32;
        while m < n {
            // (2x h - 2 h_prev)/(m+1) as 2/(m+1) * (x h - h_prev): one FMA and one multiply.
            let two_inv = V::splat(E::from_ratio(2, (m + 1) as i64));
            let next = x.mul_sube(h, h_prev) * two_inv;
            h_prev = h;
            h = next;
            m += 1;
        }
        (h, h_prev + h_prev)
    };

    let (x, _, dp) = newton_x::<E, V, _>(x0, n, valid, pair);
    // dp = 2 h_{n-1} at the (nearly converged) node.
    let h_nm1 = dp * V::HALF;

    // 2^{n-1} / (n-1)! as a scalar product.
    let mut factor = E::ONE;
    let mut m = 1u32;
    while m < n {
        factor = factor * E::from_int(2) / E::from_int(m as i64);
        m += 1;
    }
    let w = V::SQRT_PI * V::splat(factor) / (nf * h_nm1 * h_nm1);

    (valid.select(x.neg_c(neg), V::NAN), valid.select(w, V::NAN))
}

/// The `k`-th node and weight of the `n`-point Gauss-Laguerre rule (weight `x^alpha e^{-x}`
/// on `[0, inf)`), `k` per lane, `k = 0` the largest root, `alpha > -1` per lane.
///
/// Seed: the WKB phase of the Laguerre equation with `x = nu cos^2(psi/2)`,
/// `nu = 4n + 2 alpha + 2`, `psi - sin psi = 4 pi (k + 3/4)/nu`, five Newton steps from
/// `psi = (6c)^{1/3}`. The count of phase between the two turning points is
/// `n + alpha/2 + 1/2`, which is the Bessel-zero offset `alpha/2 - 1/4` on the left and the
/// Airy `3/4` on the right, so the seed is uniformly within a fraction of a root spacing.
/// Newton on the raw `L_m^alpha`, `(m+1) L_{m+1} = (2m + alpha + 1 - x) L_m - (m + alpha) L_{m-1}`,
/// scalar divisor, with `x L_n' = n L_n - (n + alpha) L_{n-1}`. Weight from Hildebrand's
/// `w = Gamma(n + alpha + 1) / (n! x L_n'(x_k)^2)`, the Gamma ratio as the running product
/// `Gamma(alpha + 1) prod (m + alpha)/m`. `L_{n-1}` at the largest root grows like
/// `e^{x/2}`, which bounds the rule near `n = 170` in f64 and `n = 20` in f32.
#[inline(always)]
pub fn gauss_laguerre_impl<P, E, V>(k: V, alpha: V, n: u32) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    let nf = V::splat(E::from_int(n as i64));
    let valid = valid_index(k, nf) & alpha.cmp_gt(-V::ONE);
    let gamma_a1 = <V as SpecializedSpecialMath<E>>::tgamma::<P>(alpha + V::ONE);

    if n == 1 {
        // L_1 = 1 + alpha - x. The weight is the whole mass Gamma(alpha + 1).
        return (valid.select(alpha + V::ONE, V::NAN), valid.select(gamma_a1, V::NAN));
    }

    let nu = nf.mul_adde(
        V::splat(<E as FloatElement>::ConstInt::<4>::VALUE),
        alpha.mul_adde(V::TWO, V::TWO),
    );
    let c = (k + V::splat(<E as FloatElement>::ConstRatio::<3, 4>::VALUE))
        * (V::splat(<E as FloatElement>::ConstInt::<4>::VALUE) * V::PI / nu);
    let mut psi = (c * V::splat(<E as FloatElement>::ConstInt::<6>::VALUE))
        .cbrt_p::<LessPrecision<P>>()
        .min(V::PI);
    let mut i = 0;
    while i < 5 {
        let (s, co) = psi.sin_cos_p::<LessPrecision<P>>();
        psi -= (psi - s - c) / (V::ONE - co).max(<V as FloatVector>::EPSILON);
        i += 1;
    }
    // x = nu (1 + cos psi)/2, with 1 + cos psi as 2 cos^2(psi/2) for the small roots.
    let half_cos = (psi * V::HALF).cos_p::<LessPrecision<P>>();
    let x0 = nu * half_cos * half_cos;

    let pair = |x: V| -> (V, V) {
        let mut l_prev = V::ONE;
        let mut l = alpha + V::ONE - x;
        // `alpha - x` is loop-invariant, so only the `2m + 1` and `m` move.
        let amx = alpha - x;
        let mut m = 1u32;
        while m < n {
            let mf = V::splat(E::from_int(m as i64));
            let c1 = V::splat(E::from_int((2 * m + 1) as i64));
            let inv = V::splat(E::ONE / E::from_int((m + 1) as i64));
            let next = (amx + c1).mul_sube(l, (alpha + mf) * l_prev) * inv;
            l_prev = l;
            l = next;
            m += 1;
        }
        // x L_n' = n L_n - (n + alpha) L_{n-1}
        let dl = (nf * l - (nf + alpha) * l_prev) / x;
        (l, dl)
    };

    let (x, _, dl) = newton_x::<E, V, _>(x0, n, valid, pair);

    // Gamma(n + alpha + 1)/n! = Gamma(alpha + 1) prod_{m=1}^{n} (m + alpha)/m.
    let mut ratio = gamma_a1;
    let mut m = 1u32;
    while m <= n {
        // `1/m` is a scalar, so `n` vector divisions become `n` vector multiplies.
        let mf = V::splat(E::from_int(m as i64));
        let inv_m = V::splat(E::ONE / E::from_int(m as i64));
        ratio *= (mf + alpha) * inv_m;
        m += 1;
    }
    let w = ratio / (x * dl * dl);

    (valid.select(x, V::NAN), valid.select(w, V::NAN))
}
