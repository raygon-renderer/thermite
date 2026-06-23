use thermite::{
    element::{FloatElement, FloatElementWithBits},
    math::{
        CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _,
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
    let mut y = q * (q * q).poly_rational_p::<P, _, _>(a, b);

    if const { P::POLICY.avoid_branching } || is_tail.any() {
        let q = (-V::TWO * p.ln_p::<P>()).sqrt();
        let t = q.poly_rational_p::<P, _, _>(c, d);

        y = is_tail.select(t, y);
    }

    let mut x = y.copysign(p_in - V::HALF);

    // Acklam's rational fit is only good to ~1.15e-9; one Halley step refines it
    // to full precision. With f(x) = Phi(x) - p, u = f/f' = (Phi(x) - p)/phi(x):
    //   Phi(x) = 0.5*erfc(-x/sqrt2),  1/phi(x) = sqrt(2pi)*exp(x^2/2)
    //   x <- x - u / (1 + x*u/2)
    if const { REFINE && P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let e = <V as SpecializedSpecialMath<E>>::erfc::<P>(x * -V::FRAC_1_SQRT_2).mul_adde(V::HALF, -p_in);
        let u = e * V::SQRT_TAU * (x * x * V::HALF).exp_p::<P>();
        x -= u / x.mul_adde(u * V::HALF, V::ONE);
    }

    x
}
