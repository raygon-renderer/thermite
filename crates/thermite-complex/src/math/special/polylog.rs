//! The polylogarithm `$\mathrm{Li}_s(z)$` over C, at a scalar real order.
//!
//! The same regions and the same per-call order plan as thermite-special's real kernel
//! (`thermite_special::specialized::generic::polylog`, which documents the algorithm), run
//! in complex arithmetic: the defining series and the unity series are complex Horner sums
//! with real coefficients, the lead terms are complex powers, the integer far field is the
//! inversion formula with its step term, and the real-order far field sums the unity series
//! over all `m` roots (no conjugate pairing off the real axis).
//!
//! On the cut `$z \in (1, \infty)$` the value follows the sign of `$\mathrm{Im}\,z$`'s
//! zero: `+0` is the limit from above, `-0` from below. Both the principal `$\ln(-\mu)$` and
//! Crandall's step function `$\sigma(z)$` read the signed zero, so nothing special-cases it.
//!
//! The order's `Real` payload is this vector's element, a complex number. Only a real
//! order is implemented, and an order with a non-zero imaginary part returns NaN (the same
//! treatment a complex-typed [`BesselOrder`](thermite_special::BesselOrder) gets).

use thermite::element::{FloatElement, SignedIntegerElement};
use thermite::math::TranscendentalMathWithPolicy as _;
use thermite::math::policy::Policy;
use thermite::prelude::*;
use thermite_special::PolylogOrder;
use thermite_special::specialized::{
    POLYLOG_KMAX as KMAX, PolylogElement, PolylogPlan, polylog_root_count, polylog_t1,
};

use crate::Complex;
use crate::vector::RealFloatVector;

#[inline(always)]
fn real<V: RealFloatVector>(x: V::Element) -> Complex<V> {
    Complex::real(V::splat(x))
}

#[inline(always)]
fn scale<V: RealFloatVector>(c: Complex<V>, r: V::Element) -> Complex<V> {
    scale_v::<V>(c, V::splat(r))
}

#[inline(always)]
fn scale_v<V: RealFloatVector>(c: Complex<V>, r: V) -> Complex<V> {
    Complex::new(c.re * r, c.im * r)
}

/// `w^k` by repeated multiplication. `k` is a small scalar.
#[inline(always)]
fn powi<V: RealFloatVector>(w: Complex<V>, k: usize) -> Complex<V> {
    let mut acc = Complex::real(V::ONE);
    for _ in 0..k {
        acc *= w;
    }
    acc
}

/// `sum_{k=1}^{terms} d_k z^k`, Horner.
#[inline(always)]
fn series<V: RealFloatVector>(z: Complex<V>, d: &[V::Element; KMAX], terms: usize) -> Complex<V> {
    let mut acc = real::<V>(d[terms - 1]);
    let mut k = terms - 1;
    while k > 0 {
        V::_loop_hint();
        k -= 1;
        acc = acc * z + real::<V>(d[k]);
    }
    acc * z
}

/// The unity series at `mu`, including the fused or plain lead term.
#[inline(always)]
fn unity<P: Policy, V: RealFloatVector>(mu: Complex<V>, plan: &PolylogPlan<V::Element>) -> Complex<V>
where
    V::Element: FloatElement,
{
    let k = plan.k2;
    let mut acc = real::<V>(plan.c[k - 1]);
    let mut j = k - 1;
    while j > 0 {
        V::_loop_hint();
        j -= 1;
        acc = acc * mu + real::<V>(plan.c[j]);
    }

    let l = (-mu).ln_p::<P>();
    if plan.fused {
        // mu^m / m! as one exponential of a difference: neither factor need be representable.
        let m = V::splat(<V::Element as FloatElement>::from_int(plan.m as thermite::LargeInt));
        let pw = (scale_v::<V>(mu.ln_p::<P>(), m) - Complex::real(V::splat(plan.ln_fact_m))).exp_p::<P>();
        let q = if plan.eps == V::Element::ZERO {
            real::<V>(plan.b1) - l
        } else {
            let w = scale::<V>(l, plan.eps);
            let ew = w.exp_p::<P>();
            // phi_1(w) = expm1(w)/w, 1 at the origin.
            let phi = w
                .norm_sqr()
                .cmp_eq(V::ZERO)
                .select(Complex::real(V::ONE), w.exp_m1_p::<P>() / w);
            real::<V>(plan.b1) + scale::<V>(ew, plan.b2) - l * phi
        };
        acc + pw * q
    } else {
        // Gamma(1-s) (-mu)^{s-1}
        acc + scale::<V>(scale::<V>(l, plan.s - V::Element::ONE).exp_p::<P>(), plan.gamma_1ms)
    }
}

/// `Li_s(z)` over the whole plane.
#[inline(always)]
pub(crate) fn polylog_impl<P, V, S>(z_in: Complex<V>, order: PolylogOrder<Complex<V::Element>, S>) -> Complex<V>
where
    P: Policy,
    V: RealFloatVector,
    V::Element: PolylogElement<Signed = S>,
    V::Signed: GenericVector<Element = S>,
    S: SignedIntegerElement,
{
    let one = Complex::real(V::ONE);

    // A real order in the vector's complex element. Anything off the real axis (a NaN
    // imaginary part included) is not implemented and answers NaN.
    let order: PolylogOrder<V::Element, S> = match order {
        PolylogOrder::Integer(n) => PolylogOrder::Integer(n),
        PolylogOrder::Real(s) => {
            if s.im != V::Element::ZERO {
                return Complex::real(V::NAN);
            }
            PolylogOrder::Real(s.re)
        }
    };
    let order = order.simplify::<V>();

    // Closed forms.
    if let PolylogOrder::Integer(k) = order {
        if k == S::ONE {
            // -ln(1 - z): ln_1p where it is the accurate spelling (small z), the plain log
            // elsewhere (ln_1p at z near 1 is not).
            let small = z_in
                .norm_sqr()
                .cmp_lt(V::splat(<V::Element as FloatElement>::ConstRatio::<1, 4>::VALUE));
            return small.select(-(-z_in).ln_1p_p::<P>(), -(one - z_in).ln_p::<P>());
        }
        if k == S::ZERO {
            return z_in / (one - z_in);
        }
    }

    // Near z = 1 the log is taken as ln_1p(z - 1): a plain `ln` carries an absolute error of an
    // ulp of 1 into `mu`, which every arm amplifies by the order (see the real kernel).
    let zm1 = z_in - one;
    let near_one = zm1
        .norm_sqr()
        .cmp_lt(V::splat(<V::Element as FloatElement>::ConstRatio::<1, 4>::VALUE));
    let mu_in = near_one.select(zm1.ln_1p_p::<P>(), z_in.ln_p::<P>());
    let q = mu_in.norm_sqr();
    let z2 = z_in.norm_sqr();

    let tau = V::TAU;
    let tau_t1 = tau * V::splat(polylog_t1::<V::Element>());
    let use_series = (z2 * tau * tau).cmp_lt(q);
    let in_unity = q.cmp_le(tau_t1 * tau_t1);
    let far = !(use_series | in_unity);

    // Negative integer order: series and unity arms as they stand, and the exact reflection
    // `Li_{-p}(z) = -(-1)^p Li_{-p}(1/z)` (Wood 10.3) for the far field only. See the real
    // kernel for why not nearer. `ln(1/z) = -ln z` exactly.
    let negative = matches!(order, PolylogOrder::Integer(k) if k < S::ZERO);
    let none = V::ZERO.cmp_ne(V::ZERO);
    let reflect = if negative { far } else { none };
    let z = reflect.select(one / z_in, z_in);
    let mu = mu_in.neg_c(reflect);
    let use_series = use_series | reflect;
    let use_far = if negative { none } else { far };
    let branchy = !P::POLICY.avoid_branching;

    let any_far = if branchy { use_far.any() } else { true };
    let plan = PolylogPlan::<V::Element>::build::<P, S>(order, any_far);

    let mut result = series::<V>(z, &plan.d, plan.k1);

    if !branchy || !use_series.all() {
        let u = unity::<P, V>(mu, &plan);
        let u = q.cmp_eq(V::ZERO).select(real::<V>(plan.at_one), u);
        result = use_series.select(result, u);
    }

    if any_far {
        let far_value = if plan.integer {
            let n = plan.n;
            if !plan.inversion {
                Complex::real(V::NAN)
            } else {
                let nu = n as usize;
                let inner = series::<V>(one / z, &plan.d, plan.k_inv);
                let inner = if n % 2 == 0 { -inner } else { inner };

                // x = mu / (2 pi i) = -i mu / 2 pi
                let x = Complex::new(mu.im / tau, -(mu.re / tau));
                let mut bpoly = real::<V>(plan.bp[nu]);
                let mut j = nu;
                while j > 0 {
                    V::_loop_hint();
                    j -= 1;
                    bpoly = bpoly * x + real::<V>(plan.bp[j]);
                }
                let bterm = bpoly * Complex::new(V::splat(plan.bp_scale.0), V::splat(plan.bp_scale.1));

                // Crandall's sigma(z): 1 iff Im z < 0 or z is on the cut read from below. In
                // signed-zero terms that is exactly "the sign bit of Im z is set": a `-0`
                // imaginary part makes the principal log take the `-pi` branch on the negative
                // axis too, and the step term is what makes the formula continuous across it.
                // The sign of a zero is read through `copysign`. A `< 0` test would miss it.
                let sigma = V::ONE.copysign(z.im).cmp_lt(V::ZERO);
                let step = Complex::new(V::ZERO, V::splat(plan.sigma_scale)) * powi::<V>(mu, plan.m);
                let step = sigma.select(step, Complex::real(V::ZERO));

                inner - bterm - step
            }
        } else {
            // m-th roots, all of them: mu_j = (mu + 2 pi i k)/m for k = ell..ell+m, ell chosen
            // so every root's argument stays in (-pi, pi].
            let m_roots = polylog_root_count::<V::Element>(mu.re.max_element());
            let m_e = <V::Element as FloatElement>::from_int(m_roots as thermite::LargeInt);
            let inv_m = V::splat(V::Element::ONE / m_e);
            let ell = (-(mu.im / tau) - V::splat(m_e) * V::HALF).ceil();
            let re_m = mu.re * inv_m;
            let mut sum = Complex::real(V::ZERO);
            for j in 0..m_roots {
                let k = ell + V::splat(<V::Element as FloatElement>::from_int(j as thermite::LargeInt));
                let im_j = mu.im + k * tau;
                // The root on the positive axis (k = 0 for a real z) sits on the cut, and
                // `-0 + 0` is `+0`: give a zero back the sign of `Im mu`, so the side of the cut
                // the caller chose survives into that root's `ln(-mu_j)`.
                let im_j = im_j.cmp_eq(V::ZERO).select(V::ZERO.copysign(mu.im), im_j);
                let mu_j = Complex::new(re_m, im_j * inv_m);
                sum += unity::<P, V>(mu_j, &plan);
            }
            scale::<V>(sum, m_e.scalar_powf_p::<P>(plan.s - V::Element::ONE))
        };
        result = use_far.select(far_value, result);
    }

    if negative && plan.n % 2 == 0 {
        result = result.neg_c(reflect);
    }

    if const { P::POLICY.check_overflow } {
        let bad = !(z_in.re.is_finite() & z_in.im.is_finite());
        result = bad.select(Complex::real(V::NAN), result);
    }

    result
}
