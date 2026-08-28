use thermite::{
    element::{FloatElement, FloatElementWithBits},
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
    },
    prelude::*,
};

use crate::specialized::SpecializedSpecialMath;

/// Shared Acklam normal-quantile (probit) core for all real element types.
///
/// `probit(p) = Phi^-1(p)`, via Peter John Acklam's rational approximation
/// (a central region plus a `q = sqrt(-2 ln p)` tail branch):
/// <https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/>
///
/// `REFINE` enables a single Halley step that polishes Acklam's ~1.15e-9 fit up to
/// full double precision (used by f64; f32 is already at its precision limit without
/// it). The step only runs when the policy precision is `Best` or higher.
#[inline(always)]
pub fn probit_acklam<P, E, V, const REFINE: bool>(p_in: V, a: &[E; 6], b: &[E; 6], c: &[E; 6], d: &[E; 5]) -> V
where
    P: Policy,
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpecialMath<E>,
{
    let p = p_in.min(V::ONE - p_in); // reflect to (0, 0.5]
    // lower tail if p < 0.02425 (= 97/4000), upper tail if p > 0.97575
    let is_tail = p.cmp_lt(V::splat(<E as FloatElement>::ConstRatio::<97, 4000>::VALUE));

    let q = p - V::HALF;
    let mut y = q * (q * q).poly_rational_n_p::<P, _, _>(a, b);

    if const { P::POLICY.avoid_branching } || is_tail.any() {
        let q = (-V::TWO * p.ln_p::<P>()).sqrt();
        let t = q.poly_rational_n_p::<P, _, _>(c, d);

        y = is_tail.select(t, y);
    }

    let mut x = y.copysign(p_in - V::HALF);

    // Acklam's rational fit is only good to ~1.15e-9, which is ~30 of 53 bits and
    // not what an Average-or-better tier should be handing back. One Halley step on
    // the DEFINING equation refines it to full precision, so above `Medium` the
    // returned value is a root of `Phi(x) = p` rather than a fit to one. With
    // f(x) = Phi(x) - p, u = f/f' = (Phi(x) - p)/phi(x):
    //   Phi(x) = 0.5*erfc(-x/sqrt2),  1/phi(x) = sqrt(2pi)*exp(x^2/2)
    //   x <- x - u / (1 + x*u/2)
    //
    // The closed form `sqrt(2) * erfinv(2p - 1)` is the other way to "use the real
    // formula" and it is measurably worse: `2p - 1` rounds to exactly -1 once
    // p < 2^-54, so it returns -inf below p = 1e-17 and is already 4.4e-07 relative
    // at p = 1e-12, where this path is machine-precision. Acklam's `sqrt(-2 ln p)`
    // tail branch is what makes the far tail work at all, so it stays as the seed.
    if const { REFINE && P::POLICY.precision.ge(PrecisionPolicy::Average) } {
        // Halley is cubic, so one step takes Acklam's ~1e-9 past full precision and
        // a second has nothing left to converge on. `Reference` runs three anyway,
        // being the tier where cost is not a consideration and the extra steps are
        // free insurance if a future seed change makes the first step insufficient.
        //
        // They do NOT buy accuracy here. The step converges to the root of the
        // COMPUTED Phi(x) - p, so once the residual is dominated by `erfc`'s own
        // error rather than by the seed, iterating cannot move it. At p ~ 0.45 and
        // one step, 44 ulp at `Precision` against 6 ulp at `Reference` on the same
        // iteration count, and the whole gap is `erfc`/`exp` being libm-backed at
        // `Reference`. The remedy for that region is a better erfc, not more Halley.
        let steps = const {
            if P::POLICY.precision.ge(PrecisionPolicy::Reference) {
                3
            } else {
                1
            }
        };

        let mut i = 0;
        while i < steps {
            let e = <V as SpecializedSpecialMath<E>>::erfc::<P>(x * -V::FRAC_1_SQRT_2).mul_sube(V::HALF, p_in);
            let u = e * V::SQRT_TAU * (x * x * V::HALF).exp_p::<P>();
            x -= u / x.mul_adde(u * V::HALF, V::ONE);
            i += 1;
        }
    }

    x
}
