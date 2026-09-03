//! The Jacobi elliptic functions `sn`, `cn` and `dn`.
//!
//! # What they are
//!
//! All three are built from one quantity, the **amplitude** `$\varphi = \mathrm{am}(u, k)$`,
//! defined as the angle whose incomplete elliptic integral of the first kind is `u`
//! (`$F(\varphi, k) = u$`, see [`ellint_impl`](super::elliptic::ellint_impl)). Then
//!
//! ```math
//! \mathrm{sn}(u, k) = \sin\varphi, \qquad
//! \mathrm{cn}(u, k) = \cos\varphi, \qquad
//! \mathrm{dn}(u, k) = \sqrt{1 - k^2\sin^2\varphi}
//! ```
//!
//! Hence the names: sine amplitude, cosine amplitude, delta amplitude. At `k = 0` the
//! amplitude is `u` itself and they degenerate to `sin u`, `cos u` and `1`. At `k = 1` they
//! become `tanh u`, `sech u` and `sech u`.
//!
//! They are returned together because they are a closed system, not merely because it is
//! cheaper: differentiating any one of them produces a product of the other two
//! (`$\mathrm{sn}' = \mathrm{cn}\,\mathrm{dn}$`,
//! `$\mathrm{cn}' = -\mathrm{sn}\,\mathrm{dn}$`,
//! `$\mathrm{dn}' = -k^2\mathrm{sn}\,\mathrm{cn}$`), exactly the way `sin` and `cos` close
//! under differentiation. The other nine Jacobi functions in Glaisher's notation (`ns`,
//! `nc`, `nd`, `sc`, `sd`, `cs`, `cd`, `ds`, `dc`) are reciprocals and ratios of these
//! three, so a caller holding the triple holds all twelve.
//!
//! # Algorithm
//!
//! The descending Landen transformation, in the arithmetic-only form due to Bulirsch
//! (1965) rather than the textbook one. Both walk the same AGM ladder down from `k` to
//! modulus zero and then climb back up, but they differ in what the climb costs:
//!
//! - The textbook descent (A&S 16.4, and Boost's `jacobi_elliptic`) carries an _angle_
//!   back up, `$\varphi_{n-1} = \tfrac12(\varphi_n + \arcsin(\tfrac{c_n}{a_n}\sin\varphi_n))$`.
//!   That is one `sin` and one `asin` per level, on a ladder several levels deep, the
//!   worst possible shape for a vector unit, where every lane pays for both.
//! - Bulirsch carries the _tangent_ of the angle instead. The half-angle step becomes
//!   rational, so the entire climb is multiplies and divides, and the whole function needs
//!   exactly **one `sin_cos`**, at the bottom of the ladder where the modulus is zero and
//!   the amplitude is just the argument.
//!
//! Measured against mpmath at 40 digits over `k` in `[0, 1)` and `|u| <= 8`, worst absolute
//! error 8.3 eps for `sn`, 4.1 for `cn`, 3.8 for `dn`. Absolute is the honest metric here:
//! all three are bounded by 1 and all three have zeros, so relative error at a zero is
//! governed by how well the zero's location is known, exactly as for `sin`.
//!
//! Accuracy degrades with `|u|` the way `sin`'s does and for the same reason: the one
//! trig call takes `u` scaled by the AGM limit, so a large `|u|` is a large argument to
//! reduce. The error above was measured to `|u| = 8`, and grows slowly beyond that.
//!
//! # The ladder is bounded, and short
//!
//! `$k' = \sqrt{1 - k^2}$` is what the AGM starts from, and for any `k` strictly below 1 in
//! binary64 the cancellation-free `one_minus_sq` bottoms out at `$2^{-52}$`, so `k'` never
//! falls below about `1.5e-8` and the ladder is never deeper than 8 rungs (measured, 4 to 6
//! is typical). [`NMAX`] carries two rungs of margin on top of that.
//!
//! Lanes converge at different depths, so the loop runs until _every_ lane has converged
//! and the climb then runs the full depth for all of them. That is safe: past convergence
//! `$a_n = b_n$`, so the extra rungs are identity transformations, and running them for
//! every lane unconditionally was measured to give bit-identical results to stopping each
//! lane at its own depth.

use thermite::{
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy,
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

/// Maximum depth of the AGM ladder.
///
/// Eight rungs is the measured worst case over the whole domain, reached only as `k`
/// approaches 1. Two more are carried as margin. The forward pass stops as soon as every
/// lane has converged, so this is a bound and not a trip count.
pub const NMAX: usize = 10;

/// `(sn, cn, dn)` at argument `u` and modulus `k`.
///
/// Only `$k^2$` enters, so the sign of `k` is irrelevant and `|k| > 1` is out of domain:
/// `$1 - k^2$` goes negative, its square root is NaN, and the NaN propagates on its own
/// without a guard. `k = 1` is the one modulus the ladder cannot walk (it starts at
/// `$k' = 0$` and never converges), and is taken by the hyperbolic limit instead.
#[inline(always)]
pub fn jacobi_elliptic<P, E, V>(u: V, k: V) -> (V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let half = V::HALF;

    // The AGM ladder on (a, b) = (1, k'), recording both sequences: the climb needs every
    // rung, and `b` cannot be recovered from `a` alone (b_i = 2 a_{i+1} - a_i loses all of
    // b_0's digits when k' is small, which is exactly the case that needs the depth).
    let mut asq = [V::ZERO; NMAX];
    let mut bsq = [V::ZERO; NMAX];

    let mut a = V::ONE;
    let mut b = k.one_minus_sq(); // k'^2, cancellation-free as |k| -> 1
    let mut c = V::ONE;

    // Same threshold and same pre-update gap test as `agm_complete_ke`: the limit sits near
    // the midpoint of the pair, so the gap going _into_ a rung is what bounds the error
    // coming out of it. See the load-bearing note on that function.
    let thresh = V::SQRT_EPSILON;
    let mut depth = NMAX - 1;
    for i in 0..NMAX {
        V::_loop_hint();

        asq[i] = a;
        b = b.sqrt();
        bsq[i] = b;
        c = (a + b) * half;

        if (a - b).abs().cmp_le(a * thresh).all() {
            depth = i;
            break;
        }

        b *= a;
        a = c;
    }

    // Bottom of the ladder: modulus zero, where the amplitude is the argument itself. This
    // is the only transcendental in the function.
    let (sin_u, cos_u) = (u * c).sin_cos_p::<P>();

    // The climb carries t = cot(amplitude) rather than the amplitude, which is what keeps it
    // rational. A zero sine is a pole of the cotangent. Those lanes are recovered at the end,
    // so all this has to do is keep the division finite.
    let at_zero = sin_u.is_zero();
    let sin_safe = at_zero.select(V::ONE, sin_u);

    let mut t = cos_u.approx_div_p::<P>(sin_safe);
    let mut w = c * t;
    let mut dn = V::ONE;

    let mut i = depth;
    loop {
        V::_loop_hint();

        let ai = asq[i];
        t *= w;
        w *= dn;
        dn = (bsq[i] + t).approx_div_p::<P>(ai + t);
        t = w.approx_div_p::<P>(ai);

        if i == 0 {
            break;
        }
        i -= 1;
    }

    // sin and cos recovered from the cotangent: |sn| = 1/sqrt(w^2 + 1), and the sign is the
    // one the bottom-of-ladder sine already carried.
    let mag = w.mul_adde(w, V::ONE).inverse_sqrt_p::<P>();
    let mut sn = mag.copysign(sin_u);
    let mut cn = w * sn;

    // Where the amplitude's sine vanished, the triple is (0, +-1, 1) exactly: the cotangent
    // route cannot produce it, and cos_u is already the correct +-1.
    sn = at_zero.select(V::ZERO, sn);
    cn = at_zero.select(cos_u, cn);
    dn = at_zero.select(V::ONE, dn);

    if const { P::POLICY.check_overflow } {
        // k = 1 makes k' = 0: the ladder starts at its own fixed point and never converges,
        // so the limit is substituted whole. sn -> tanh, cn and dn -> sech, and the three
        // stop being periodic.
        let unit = k.abs().cmp_eq(V::ONE);
        if const { P::POLICY.avoid_branching } || thermite::unlikely(unit.any()) {
            let sech = V::ONE.approx_div_p::<P>(u.cosh_p::<P>());
            sn = unit.select(u.tanh_p::<P>(), sn);
            cn = unit.select(sech, cn);
            dn = unit.select(sech, dn);
        }
    }

    (sn, cn, dn)
}
