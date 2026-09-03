//! The polylogarithm `$\mathrm{Li}_s(z) = \sum_{k \ge 1} z^k / k^s$` of a real argument, at
//! a scalar real order.
//!
//! Two things live here: the **order plan** (every order-dependent coefficient, computed
//! once per call in the element type through the scalar math surface, and shared with
//! thermite-complex's kernel) and the **real kernel**, which returns the real part of the
//! principal value for a real vector. Nothing in the real kernel is a complex number.
//! Where the mathematics is complex (the negative axis, the cut `$z > 1$`, the far-field
//! roots) the real part is taken analytically: the polynomial parts by a Goertzel
//! recurrence, the transcendental parts in polar form.
//!
//! # Regions
//!
//! With `$\mu = \ln z$` (principal: `$\ln|z| + i\pi$` on the negative axis) and
//! `$t = |\mu| / 2\pi$`, per lane:
//!
//! 1. **Defining series** where `$2\pi|z| < |\mu|$` (Roughan's rule, which crosses the
//!    positive axis at `z = 0.2323` and the negative axis at `z = -0.5113`), at a fixed
//!    term count set by the worst ratio `0.5113`.
//! 2. **Unity series** where `$t \le 0.512$` (Wood 9.3, Crandall 1.4, Roughan Series 2):
//!    ```math
//!    \mathrm{Li}_s(z) = \Gamma(1-s)(-\mu)^{s-1} + \sum_{k \ge 0} \zeta(s-k)\,\frac{\mu^k}{k!}
//!    ```
//!    whose tail falls like `$(|\mu|/2\pi)^k$`. For `$s = n + \varepsilon$` with `$n \ge 1$`
//!    the `$k = n-1$` term and the `$\Gamma$` term each have a pole that the other cancels.
//!    They are fused **algebraically** into `$\mu^{n-1} Q_{n-1}(L, \varepsilon)/(n-1)!$`
//!    with `$L = \ln(-\mu)$` and
//!    ```math
//!    Q_m(L, \varepsilon) = \Big[\zeta(1+\varepsilon) - \tfrac{1}{\varepsilon}\Big]
//!      + \Big[(-1)^m m!\,\Gamma(-m-\varepsilon) + \tfrac{1}{\varepsilon}\Big] e^{\varepsilon L}
//!      - \frac{e^{\varepsilon L} - 1}{\varepsilon}.
//!    ```
//!    Both brackets are scalars, finite and smooth for every `$\varepsilon$` (the second is
//!    `$-\mathrm{expm1}(u)/\varepsilon$` with
//!    `$u = \ln\Gamma(1-\varepsilon) - \sum_{k \le m}\ln(1 + \varepsilon/k)$`), so there is
//!    no near-integer threshold and no Taylor arm. Roughan's Series 3 and its `1e-3`
//!    switch are what this replaces. At `$\varepsilon = 0$` it collapses to Wood 9.5's
//!    `$H_m - L$` exactly, and every coefficient of the integer case is a table read.
//! 3. **Far field**, otherwise. Integer order takes the inversion formula (Crandall 1.3),
//!    `$\mathrm{Li}_n(z) = -(-1)^n \mathrm{Li}_n(1/z) - \frac{(2\pi i)^n}{n!} B_n\!\big(\tfrac{\mu}{2\pi i}\big) - \sigma(z)\,\frac{2\pi i\,\mu^{n-1}}{(n-1)!}$`,
//!    with `$\mathrm{Li}_n(1/z)$` from the defining series. The step term is imaginary and
//!    drops out of a real part. Real order takes Wood's m-th-root identity
//!    `$\mathrm{Li}_s(z) = m^{s-1}\sum_{j} \mathrm{Li}_s(z^{1/m}\,e^{2\pi i j/m})$`
//!    (Roughan 2026 Section 4.6): `m` is chosen so every root's `$\mu_j$` satisfies
//!    `$|\mu_j| \le 1.2\pi$`, the coefficient sweep is shared across the roots, each root
//!    runs region 2 (the fused form included, so this arm serves near-integer orders too),
//!    and for a real argument the roots come in conjugate pairs with equal real parts, so
//!    only half of them are evaluated. Negative integer orders run regions 1 and 2 as they
//!    stand and reflect `$\mathrm{Li}_{-p}(z) = -(-1)^p \mathrm{Li}_{-p}(1/z)$` (Wood 10.3) in
//!    the far field only.
//! 4. `$\mathrm{Li}_1 = -\ln(1-z)$` and `$\mathrm{Li}_0 = z/(1-z)$` are closed forms.
//!
//! Every arm is fixed-length, so a packet pays the count once rather than its worst lane's
//! convergence. The counts scale with the policy's precision tier.
//!
//! # Real parts without complex numbers
//!
//! A real polynomial at `$x = a + ib$` divides by `$(t-x)(t-\bar x) = t^2 - 2a\,t + |x|^2$`.
//! The Goertzel recurrence `$B_k = c_k + 2a B_{k+1} - |x|^2 B_{k+2}$` leaves
//! `$\mathrm{Re}\,P(x) = B_0 - a B_1$` and `$\mathrm{Im}\,P(x) = b B_1$`, two real FMAs per
//! term. The lead terms use `$r = |\mu|$`, `$\theta = \arg(-\mu)$`, `$\varphi = \arg\mu$`:
//! `$\mathrm{Re}\,\Gamma(1-s)(-\mu)^{s-1} = \Gamma(1-s)\,r^{s-1}\cos((s-1)\theta)$`, and for
//! the fused term `$\mathrm{Re}[\mu^m Q] = r^m(\cos m\varphi\,\mathrm{Re}\,Q - \sin m\varphi\,\mathrm{Im}\,Q)$`
//! with `$e^{\varepsilon L} = r^\varepsilon e^{i\varepsilon\theta}$` and the difference
//! quotient spelled `$\ln r\,\varphi_1(\varepsilon \ln r)\cos\varepsilon\theta - \mathrm{versin}(\varepsilon\theta)/\varepsilon$`
//! so that `$\varepsilon \to 0$` is exact. Angles are carried as multiples of `$\pi$` and go
//! through the `_pi` trig, which is exact on the axes. On the cut near `$z = 1$` the lead
//! term's imaginary part is enormous and a radian phase error leaks it into the real part.
//! All verified against mpmath before being written (scratch `goertzel.py`, 2026-09-02).

use thermite::{
    LargeInt,
    element::{FloatElement, FloatElementWithBits, SignedElement, SignedIntegerElement},
    math::{
        RealMathWithPolicy, ScalarMathWithPolicy,
        policy::{Policy, PrecisionPolicy},
    },
    prelude::*,
};
use thermite::{const_element, const_splat};

use crate::polylog::PolylogOrder;
use crate::tables::bernoulli::BernoulliNumbers;
use crate::tables::polylog::PolylogConsts;

/// Longest series any arm runs: binary64 at the root arm's ratio `0.6` needs 74 terms.
/// Also caps `|n|` for the inversion polynomial.
pub const KMAX: usize = 80;

/// Root-count cap. `|z| = 1e308` asks for 341.
pub const MMAX: usize = 512;

/// Unity-series boundary on `|mu| / 2pi`. Roughan's 0.512 rather than 0.5: the extra sliver
/// is what closes the gap the series-1 rule leaves near the negative axis.
#[inline(always)]
pub fn t1<E: FloatElement>() -> E {
    const_element!(ratio <E>: 512 / 1000)
}

/// `-ln 0.5113`, the worst ratio the defining series meets (the negative-axis crossing).
#[inline(always)]
fn ln_inv_series_ratio<E: FloatElement>() -> E {
    const_element!(ratio <E>: 6708 / 10000)
}

/// `-ln 0.512` and `-ln 0.6`: the unity series' ratio in region 2 and in the root arm
/// (`alpha / 2` with Roughan's `alpha = 1.2`).
#[inline(always)]
fn ln_inv_unity_ratio<E: FloatElement>(root_lanes: bool) -> E {
    if root_lanes {
        const_element!(ratio <E>: 5108 / 10000)
    } else {
        const_element!(ratio <E>: 6694 / 10000)
    }
}

/// Mantissa bits the policy's precision tier asks the series to reach.
#[inline(always)]
fn effective_bits<E: FloatElementWithBits>(precision: PrecisionPolicy) -> u32 {
    let m = E::MANTISSA_BITS + 1;
    match precision {
        PrecisionPolicy::Worst => m / 2,
        PrecisionPolicy::Medium => m * 3 / 4,
        _ => m,
    }
}

#[inline(always)]
fn int<E: FloatElement>(k: usize) -> E {
    E::from_int(k as LargeInt)
}

/// Terms for a tail `r^k k^-s` (`ln_inv_ratio = -ln r`) to fall below `2^-bits`: the
/// smallest `k` with `k ln(1/r) + s ln k >= bits ln 2`, plus two. Non-negative orders drop
/// the helping `s ln k` term and take the plain geometric count, but negative orders grow
/// before they decay (`s = -3.7`, `z = -1/2` needs 77 terms, not 57) and keep it. The
/// search is a comparison loop in `E`, so no count is ever converted out of a float.
#[inline(always)]
fn terms<P: Policy, E>(bits: u32, ln_inv_ratio: E, s: E) -> usize
where
    E: FloatElement + ScalarMathWithPolicy,
{
    let target = E::from_int(bits as LargeInt) * E::LN_2;
    let mut k = 1usize;
    if s < E::ZERO {
        while k < KMAX && int::<E>(k) * ln_inv_ratio + s * int::<E>(k).scalar_ln_p::<P>() < target {
            k += 1;
        }
    } else {
        while k < KMAX && int::<E>(k) * ln_inv_ratio < target {
            k += 1;
        }
    }
    (k + 2).min(KMAX)
}

/// `zeta(n)` at an integer: the table for `n >= 2` (exactly 1 past it), `-1/2` at zero,
/// `-B_{j+1}/(j+1)` at `-j` (zero for even `j`).
///
/// Past the Bernoulli table it returns **zero**, not infinity: the table ends where
/// `B_{2n}` overflows the format, and every use here divides by a `k!` that overflowed
/// earlier still, so the true coefficient is far below the format's precision and the
/// honest value of `inf/inf` is 0, not NaN. Reached in binary32 from `k = 58`.
#[inline(always)]
pub fn zeta_int<E: FloatElement + PolylogConsts + BernoulliNumbers>(n: isize) -> E {
    if n >= 2 {
        return E::ZETA_INT.get((n - 2) as usize).copied().unwrap_or(E::ONE);
    }
    if n == 1 {
        return E::ORDER_MAX;
    }
    if n == 0 {
        return -const_element!(ratio <E>: 1 / 2);
    }
    let j = (-n) as usize;
    if j.is_multiple_of(2) {
        return E::ZERO;
    }
    // B_{j+1} with j+1 even: table entry i holds B_{2i+2}.
    match E::B2N.get(j.div_ceil(2) - 1) {
        Some(&b) => -b / int::<E>(j + 1),
        None => E::ZERO,
    }
}

/// Finite and not NaN (a NaN fails the comparison on its own).
#[inline(always)]
fn finite<E: FloatElement>(x: E) -> bool {
    SignedElement::abs(x) < E::ORDER_MAX
}

/// The number of m-th roots the far field needs when the widest lane's `Re ln z` is
/// `re_mu_max`: the smallest `m >= 2` with `Re mu / m <= sqrt(alpha^2 - 1) pi = 2.0839`
/// (`alpha = 1.2`), capped. The identity holds for any `m`, so the widest lane sets it for
/// the packet. A comparison loop, so no count is converted out of a float.
#[inline(always)]
pub fn root_count<E: FloatElement>(re_mu_max: E) -> usize {
    let step = const_element!(ratio <E>: 20839 / 10000);
    let mut m = 2usize;
    while m < MMAX && int::<E>(m) * step < re_mu_max {
        m += 1;
    }
    m
}

/// Everything about the order, computed once per call in the element type. Shared with
/// the complex kernel in thermite-complex. Not a stable surface.
#[doc(hidden)]
pub struct PolylogPlan<E> {
    pub s: E,
    /// `round(s)`. The fused slot is `m = n - 1` when `n >= 1`.
    pub n: isize,
    pub eps: E,
    pub integer: bool,
    /// Integer order whose inversion coefficients are representable. False past `n = 79`
    /// (binary64) / `n = 34` (binary32), where the far field answers NaN.
    pub inversion: bool,
    pub fused: bool,
    pub m: usize,
    /// Defining-series length, and the one used on `1/z` in the inversion arm.
    pub k1: usize,
    pub k_inv: usize,
    /// Unity-series length actually swept into `c`.
    pub k2: usize,
    /// `d[k-1] = k^-s`.
    pub d: [E; KMAX],
    /// `c[k] = zeta(s-k)/k!`, with `c[m] = 0` when fused.
    pub c: [E; KMAX],
    /// `ln m!`: the fused lead is `exp(m ln mu - ln m!) Q`, never `mu^m / m!`.
    pub ln_fact_m: E,
    /// The two scalar brackets of `Q_m`. For integer order `b1 = H_m` and `b2 = 0`.
    pub b1: E,
    pub b2: E,
    /// `Gamma(1-s)`, for the unfused lead term (`n <= 0`).
    pub gamma_1ms: E,
    /// `Li_s(1)`.
    pub at_one: E,
    /// Inversion arm: coefficient of `x^j` in `B_n(x)`, the `(2 pi i)^n / n!` factor as
    /// `(re, im)`, and `2 pi / (n-1)!` for the step term.
    pub bp: [E; KMAX],
    pub bp_scale: (E, E),
    /// Read only by the complex kernel, as a real part never sees the step term.
    #[allow(dead_code)]
    pub sigma_scale: E,
}

/// The element bounds the plan's scalar precompute needs.
pub trait PolylogElement:
    FloatElementWithBits + PolylogConsts + BernoulliNumbers + ScalarMathWithPolicy + crate::ScalarSpecialMathWithPolicy
{
}
impl<E> PolylogElement for E where
    E: FloatElementWithBits
        + PolylogConsts
        + BernoulliNumbers
        + ScalarMathWithPolicy
        + crate::ScalarSpecialMathWithPolicy
{
}

impl<E: PolylogElement> PolylogPlan<E> {
    /// `s = n + eps` with `n >= 1`: the brackets of `Q_{n-1}`.
    #[inline(always)]
    fn brackets<P: Policy>(m: usize, eps: E) -> (E, E) {
        let mut h_m = E::ZERO;
        for k in 1..=m {
            h_m = h_m + E::ONE / int::<E>(k);
        }
        if eps == E::ZERO {
            return (h_m, E::ZERO);
        }
        let small = SignedElement::abs(eps) <= const_element!(ratio <E>: 1 / 10);

        // b1 = zeta(1+eps) - 1/eps. The Laurent remainder is entire, so the Stieltjes series
        // converges everywhere, and is used where the direct difference would cancel.
        let b1 = if small {
            let mut acc = E::ZERO;
            let mut i = E::STIELTJES.len();
            while i > 0 {
                i -= 1;
                // sum (-1)^k gamma_k eps^k / k!, Horner in eps with the 1/k! folded in.
                acc = E::STIELTJES[i] - acc * eps / int::<E>(i + 1);
            }
            acc
        } else {
            (E::ONE + eps).scalar_zeta_p::<P>() - E::ONE / eps
        };

        // b2 = -expm1(u)/eps with u = lgamma(1-eps) - sum_{k<=m} log1p(eps/k). Small eps takes
        // u's own series, u = eps (gamma - H_m) + sum_{j>=2} eps^j [zeta(j) + (-1)^j H_{m,j}] / j,
        // because lgamma near 1 and log1p do not hold the relative accuracy u/eps needs.
        let u = if small {
            let mut acc = E::ZERO;
            let mut j = 18usize;
            while j >= 2 {
                // H_{m,j} = sum_{k<=m} k^-j
                let mut hmj = E::ZERO;
                for k in 1..=m {
                    let r = E::ONE / int::<E>(k);
                    let mut p = r;
                    for _ in 1..j {
                        p = p * r;
                    }
                    hmj = hmj + p;
                }
                let zj = zeta_int::<E>(j as isize);
                let coeff = if j.is_multiple_of(2) { zj + hmj } else { zj - hmj } / int::<E>(j);
                acc = coeff + acc * eps;
                j -= 1;
            }
            eps * (E::STIELTJES[0] - h_m + acc * eps)
        } else {
            let mut u = (E::ONE - eps).scalar_lgamma_p::<P>();
            for k in 1..=m {
                u = u - (eps / int::<E>(k)).scalar_ln_1p_p::<P>();
            }
            u
        };
        let b2 = -u.scalar_exp_m1_p::<P>() / eps;
        (b1, b2)
    }

    /// Build the plan from a simplified order. `root_lanes` says whether any lane will run
    /// the root arm, whose unity series is longer (ratio 0.6 against 0.512).
    #[inline(always)]
    pub fn build<P: Policy, S: SignedIntegerElement>(order: PolylogOrder<E, S>, root_lanes: bool) -> Self {
        let bits = effective_bits::<E>(P::POLICY.precision);
        let (integer, s, n): (bool, E, isize) = match order {
            PolylogOrder::Integer(k) => {
                // The signed lane element's own sanctioned narrowing. Far beyond the table
                // reach in either direction is the same "unsupported" as the cap below.
                let n: isize = k
                    .try_into()
                    .unwrap_or(if k < S::ZERO { isize::MIN / 2 } else { isize::MAX / 2 });
                (
                    true,
                    E::from_int(n.clamp(-(KMAX as isize) * 4, (KMAX as isize) * 4) as LargeInt),
                    n,
                )
            }
            PolylogOrder::Real(s) => {
                // n = round(s) as an integer, found by comparison rather than conversion. Only
                // |n| up to the table reach matters (beyond it the fused slot is outside the
                // sweep and the lead term underflows), so the search is bounded.
                let r = FloatElement::round(s);
                let mut n = 0isize;
                let bound = KMAX as isize * 4;
                if r > E::ZERO {
                    while n < bound && int::<E>(n as usize) < r {
                        n += 1;
                    }
                } else {
                    while n > -bound && -int::<E>((-n) as usize) > r {
                        n -= 1;
                    }
                }
                (false, s, n)
            }
        };
        let eps = if integer {
            E::ZERO
        } else {
            s - E::from_int(n as LargeInt)
        };
        let fused = n >= 1;
        let m = if fused { (n - 1) as usize } else { 0 };
        let m_capped = m.min(KMAX * 4);

        // Series lengths. The unity sweep is as long as the longest lane needs: 0.512 for a
        // packet that stays in region 2, the root arm's 0.6 otherwise. The inversion arm's
        // inner series runs at 1/z, which on the negative axis is as large as 1/2 (the far
        // field starts at z = -2 there), so it takes the full count.
        let k1 = terms::<P, E>(bits, ln_inv_series_ratio::<E>(), s);
        let k_inv = k1;
        let k2 = terms::<P, E>(bits, ln_inv_unity_ratio::<E>(root_lanes), s);

        let mut d = [E::ZERO; KMAX];
        for k in 1..=k1 {
            let kf = int::<E>(k);
            d[k - 1] = if integer {
                kf.scalar_powi_p::<P>(-(n as i32))
            } else {
                kf.scalar_powf_p::<P>(-s)
            };
        }

        let mut c = [E::ZERO; KMAX];
        let mut fact = E::ONE;
        for k in 0..k2 {
            if k > 0 {
                fact = fact * int::<E>(k);
            }
            let zk = if integer {
                zeta_int::<E>(n - k as isize)
            } else {
                (s - int::<E>(k)).scalar_zeta_p::<P>()
            };
            // A non-finite zeta (the reflection's Gamma overflowing the format, binary32 from
            // k ~ 34 at negative s) sits where k! has overflowed too: the coefficient is
            // negligible and the honest value is 0, not inf/inf.
            c[k] = if (fused && k == m) || !finite(zk) {
                E::ZERO
            } else {
                zk / fact
            };
        }

        // ln m!, so the fused lead is exp(m ln r - ln m!) rather than r^m / m!, whose
        // two factors overflow separately (binary32 at m = 35) while the ratio is fine.
        let mut ln_fact_m = E::ZERO;
        let mut fact_m = E::ONE;
        for k in 1..=m_capped {
            ln_fact_m = ln_fact_m + int::<E>(k).scalar_ln_p::<P>();
            fact_m = fact_m * int::<E>(k);
        }
        let (b1, b2) = if fused {
            Self::brackets::<P>(m_capped, eps)
        } else {
            (E::ZERO, E::ZERO)
        };
        let gamma_1ms = if fused {
            E::ZERO
        } else {
            (E::ONE - s).scalar_tgamma_p::<P>()
        };

        let at_one = if s > E::ONE {
            if integer {
                zeta_int::<E>(n)
            } else {
                s.scalar_zeta_p::<P>()
            }
        } else {
            E::ORDER_MAX
        };

        // Inversion arm (integer n >= 1, capped by the array): B_n(x) = sum_k C(n,k) B_k x^{n-k}
        // with B_1 = -1/2, so the coefficient of x^j is C(n, j) B_{n-j}.
        let mut bp = [E::ZERO; KMAX];
        let mut bp_scale = (E::ZERO, E::ZERO);
        let mut sigma_scale = E::ZERO;
        // The inversion formula needs n! and B_n in the format: binary64 to n = 79 (the
        // array), binary32 to n = 34 (n! overflows at 35). Past that the far field is NaN.
        let inversion = integer && n >= 1 && (n as usize) < KMAX && finite(fact_m * int::<E>(n as usize));
        if inversion {
            let nu = n as usize;
            let mut binom = E::ONE; // C(n, j), walked up in j
            for j in 0..=nu {
                if j > 0 {
                    binom = binom * int::<E>(nu - j + 1) / int::<E>(j);
                }
                let i = nu - j;
                let b = if i == 0 {
                    E::ONE
                } else if i == 1 {
                    -const_element!(ratio <E>: 1 / 2)
                } else if !i.is_multiple_of(2) {
                    E::ZERO
                } else {
                    E::B2N.get(i / 2 - 1).copied().unwrap_or(E::ORDER_MAX)
                };
                bp[j] = binom * b;
            }
            // (2 pi i)^n / n! = (2 pi)^n / n! * i^n
            let mag = E::TAU.scalar_powi_p::<P>(n as i32) / (fact_m * int::<E>(nu));
            bp_scale = match nu % 4 {
                0 => (mag, E::ZERO),
                1 => (E::ZERO, mag),
                2 => (-mag, E::ZERO),
                _ => (E::ZERO, -mag),
            };
            sigma_scale = E::TAU / fact_m;
        }

        Self {
            s,
            n,
            eps,
            integer,
            inversion,
            fused,
            m: m_capped,
            k1,
            k_inv,
            k2,
            d,
            c,
            ln_fact_m,
            b1,
            b2,
            gamma_1ms,
            at_one,
            bp,
            bp_scale,
            sigma_scale,
        }
    }
}

/// The defining series `sum_{k=1}^{terms} d_k z^k` at a real `z`, Horner.
#[inline(always)]
fn series<E, V>(z: V, d: &[E; KMAX], terms: usize) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let mut acc = V::splat(d[terms - 1]);
    let mut k = terms - 1;
    while k > 0 {
        V::_loop_hint();
        k -= 1;
        acc = acc.mul_adde(z, V::splat(d[k]));
    }
    acc * z
}

/// `(Re, Im)` of `sum_{k < terms} c_k x^k` at `x = a + ib`, by Goertzel: `q = |x|^2`.
#[inline(always)]
fn goertzel<E, V>(c: &[E; KMAX], terms: usize, a: V, b: V, q: V) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let p = a + a;
    let mut b1 = V::ZERO;
    let mut b2 = V::ZERO;
    let mut k = terms;
    while k > 0 {
        V::_loop_hint();
        k -= 1;
        let b0 = q.nmul_adde(b2, p.mul_adde(b1, V::splat(c[k])));
        b2 = b1;
        b1 = b0;
    }
    (a.nmul_adde(b2, b1), b * b2)
}

/// `(e^x - 1)/x`, exactly 1 at the origin.
#[inline(always)]
fn phi1<P, E, V>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + RealMathWithPolicy<Element = E>,
{
    x.cmp_eq(V::ZERO).select(V::ONE, x.exp_m1_p::<P>() / x)
}

/// The real part of the unity series at `mu = a + ib`, including the fused or plain lead
/// term. `q = |mu|^2`. See the module docs for the identities.
#[inline(always)]
pub fn unity_re<P, E, V>(a: V, b: V, q: V, plan: &PolylogPlan<E>) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + RealMathWithPolicy<Element = E>,
{
    let (poly, _) = goertzel::<E, V>(&plan.c, plan.k2, a, b, q);

    let ln_r = q.ln_p::<P>() * V::HALF;
    // Angles as multiples of pi: exact on the axes, and the `_pi` trig is exact at
    // half-integers, which the cut near z = 1 needs (see the module docs).
    let theta_pi = (-b).atan2_p::<P>(-a) * V::FRAC_1_PI; // arg(-mu)/pi

    if plan.fused {
        let m = V::splat(int::<E>(plan.m));
        let phi_pi = b.atan2_p::<P>(a) * V::FRAC_1_PI; // arg(mu)/pi
        // r^m / m! as one exponential of a difference: neither factor need be representable.
        let r_m = (ln_r * m - V::splat(plan.ln_fact_m)).exp_p::<P>();
        let (s_m, c_m) = (phi_pi * m).sincos_pi_p::<P>();

        let (q_re, q_im) = if plan.eps == E::ZERO {
            (V::splat(plan.b1) - ln_r, -(theta_pi * V::PI))
        } else {
            let eps = V::splat(plan.eps);
            let r_eps = (ln_r * eps).exp_p::<P>();
            let et_pi = theta_pi * eps;
            let (s_e, c_e) = et_pi.sincos_pi_p::<P>();
            // D = (e^{eps L} - 1)/eps, spelled to be exact as eps -> 0: versin(x) = 2 sin^2(x/2)
            // and sin(x)/x through sinc_pi.
            let half_s = (et_pi * V::HALF).sin_pi_p::<P>();
            let versin = (half_s * half_s) * V::TWO;
            let d_re = (ln_r * phi1::<P, E, V>(ln_r * eps)).mul_sube(c_e, versin / eps);
            let d_im = r_eps * (theta_pi * V::PI) * et_pi.sinc_pi_p::<P>();
            let b2e = r_eps * V::splat(plan.b2);
            (b2e.mul_adde(c_e, V::splat(plan.b1)) - d_re, b2e * s_e - d_im)
        };
        let lead = r_m * (c_m * q_re - s_m * q_im);
        poly + lead
    } else {
        // Gamma(1-s) r^{s-1} cos((s-1) theta)
        let sm1 = V::splat(plan.s - E::ONE);
        let lead = (ln_r * sm1).exp_p::<P>() * (theta_pi * sm1).cos_pi_p::<P>() * V::splat(plan.gamma_1ms);
        poly + lead
    }
}

/// The real part of `Li_s(z)` for real `z`, every real order. See the [module
/// documentation](self).
#[inline(always)]
pub fn polylog_impl<P, E, V>(z_in: V, order: PolylogOrder<E, E::Signed>) -> V
where
    P: Policy,
    E: PolylogElement,
    V: FloatVectorWithBits<Element = E> + RealMathWithPolicy<Element = E>,
    V::Signed: GenericVector<Element = E::Signed>,
{
    let order = order.simplify::<V>();
    let one = V::ONE;

    // Closed forms: Li_1 = -ln(1 - z) (on the cut the real part is -ln(z - 1)), Li_0 = z/(1-z).
    if let PolylogOrder::Integer(k) = order {
        if k == E::Signed::ONE {
            let below = z_in.cmp_lt(one);
            return below.select(-(-z_in).ln_1p_p::<P>(), -(z_in - one).ln_p::<P>());
        }
        if k == E::Signed::ZERO {
            return z_in / (one - z_in);
        }
    }

    // mu = ln z = a + ib with b = pi on the negative axis. Near |z| = 1 the log is taken as
    // ln_1p(|z| - 1), the subtraction being exact there: a plain `ln` carries an absolute
    // error of an ulp of 1 into `a`, and every arm amplifies the relative error of `a` by
    // the order (the lead term is a power of it).
    let az = z_in.abs();
    let near_one = az.cmp_ge(V::HALF) & az.cmp_le(V::TWO);
    let a_in = near_one.select((az - one).ln_1p_p::<P>(), az.ln_p::<P>());
    let b = z_in.is_negative().select(V::PI, V::ZERO);
    let q = a_in.mul_adde(a_in, b * b);

    // Region masks: series where 2 pi |z| < |mu|, unity where |mu| <= 2 pi T1, far otherwise.
    let tau_t1 = V::TAU * V::splat(t1::<E>());
    let use_series = ((z_in * z_in) * (V::TAU * V::TAU)).cmp_lt(q);
    let in_unity = q.cmp_le(tau_t1 * tau_t1);
    let far = !(use_series | in_unity);

    // Negative integer order: the series and unity arms serve as they stand (every
    // coefficient is a tabulated zeta value, the lead term is `(-n)! (-mu)^{n-1}`), and the
    // far field is the exact reflection `Li_{-p}(z) = -(-1)^p Li_{-p}(1/z)` (Wood 10.3),
    // which lands every far lane in the series region. Only far lanes reflect: next to
    // z = 1 the rounding of 1/z is amplified by (p+1)/|mu| (measured 2.5e-13 at 1.001), and
    // the unity series needs no help there. `ln(1/z) = -ln z` exactly, so `a` just flips.
    // The closed rational form in z/(1-z) was tried first and cancels ~2000x near |z| = 1
    // at p = 6, as Wood warns.
    let negative = matches!(order, PolylogOrder::Integer(k) if k < E::Signed::ZERO);
    let none = V::ZERO.cmp_ne(V::ZERO);
    let reflect = if negative { far } else { none };
    let z = reflect.select(one / z_in, z_in);
    let a = a_in.neg_c(reflect);
    let use_series = use_series | reflect;
    let use_far = if negative { none } else { far };
    let branchy = !P::POLICY.avoid_branching;

    let any_far = if branchy { use_far.any() } else { true };
    let plan = PolylogPlan::<E>::build::<P, E::Signed>(order, any_far);

    let mut result = series::<E, V>(z, &plan.d, plan.k1);

    if !branchy || !use_series.all() {
        let u = unity_re::<P, E, V>(a, b, q, &plan);
        // z = 1 exactly: mu = 0 makes the lead term 0 * inf.
        let u = q.cmp_eq(V::ZERO).select(V::splat(plan.at_one), u);
        result = use_series.select(result, u);
    }

    if any_far {
        let far_value = if plan.integer {
            // Inversion, real part. The step term is imaginary and drops out.
            let n = plan.n;
            if !plan.inversion {
                V::NAN
            } else {
                let inner = series::<E, V>(one / z, &plan.d, plan.k_inv);
                let inner = if n % 2 == 0 { -inner } else { inner };

                // x = mu / (2 pi i) = (b - i a) / 2 pi
                let inv_tau = V::FRAC_1_TAU;
                let xr = b * inv_tau;
                let xi = -(a * inv_tau);
                let (pr, pi) = goertzel::<E, V>(&plan.bp, n as usize + 1, xr, xi, q * inv_tau * inv_tau);
                let bterm = pr.mul_sube(V::splat(plan.bp_scale.0), pi * V::splat(plan.bp_scale.1));
                inner - bterm
            }
        } else {
            // m-th roots, conjugate pairs folded: for k = 0..=m/2 the root's imaginary part
            // is (b + 2 pi k)/m, weighted 2 for a proper pair, 1 for the self-conjugate roots
            // at 0 and pi, 0 past pi.
            let m_roots = root_count::<E>(a.max_element());
            let inv_m = V::splat(E::ONE / int::<E>(m_roots));
            let a_m = a * inv_m;
            let tol = V::PI * const_splat!(ratio <E>: 1 / 1000000000);
            let mut sum = V::ZERO;
            for k in 0..=(m_roots / 2) {
                let b_j = (b + V::TAU * V::splat(int::<E>(k))) * inv_m;
                let q_j = a_m.mul_adde(a_m, b_j * b_j);
                let over = b_j.cmp_gt(V::PI + tol);
                let single = b_j.cmp_eq(V::ZERO) | (b_j - V::PI).abs().cmp_le(tol);
                let weight = over.select(V::ZERO, single.select(one, V::TWO));
                sum = weight.mul_adde(unity_re::<P, E, V>(a_m, b_j, q_j, &plan), sum);
            }
            sum * V::splat(int::<E>(m_roots).scalar_powf_p::<P>(plan.s - E::ONE))
        };
        result = use_far.select(far_value, result);
    }

    if negative && plan.n % 2 == 0 {
        result = result.neg_c(reflect);
    }

    if const { P::POLICY.check_overflow } {
        result = z_in.is_finite().select(result, V::NAN);
    }

    result
}
