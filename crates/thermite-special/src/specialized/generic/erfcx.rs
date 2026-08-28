//! `erfcx(x) = e^{x^2} erfc(x)`, the scaled complementary error function.
//!
//! # Motivation
//!
//! `erfc` underflows to zero at `x ~ 27` in binary64 and `x ~ 9` in binary32, where the
//! true value is `e^{-x^2}/(x sqrt(pi))`, nonzero and merely unrepresentable. Every
//! Gaussian tail, importance weight and log-likelihood past that point silently becomes
//! zero. `erfcx` removes the exponential and is `O(1/x)`, so it stays representable for
//! every finite argument and carries full relative accuracy the whole way.
//!
//! # Algorithm
//!
//! The Faddeeva function restricted to the imaginary axis: `w(ix) = erfcx(x)` exactly.
//! Weideman's approximation (see [`crate::tables::weideman`]) is
//!
//! ```text
//! Z = (L + iz)/(L - iz),   w(z) = 1/(sqrt(pi)(L - iz)) + 2 P(Z)/(L - iz)^2
//! ```
//!
//! with `P` real. Substituting `z = ix` for real `x` makes `L - iz = L + x` and
//! `Z = (L - x)/(L + x)`, **both real**: every complex operation in the method
//! disappears and what is left is one reciprocal and one real Horner. There are no
//! transcendentals at all on the non-negative side, which makes this cheaper than the
//! `erfc` it complements.
//!
//! The domain is well conditioned throughout. `L + x >= L > 0` for every finite `x >= 0`,
//! so the reciprocal needs no guard, and `Z` runs monotonically over `(-1, 1]`, so the
//! Horner stays inside the unit disc the coefficients were fitted on. Only the
//! infinities fall outside that, `Z` there being `(L - inf) * 0`, and they are named
//! explicitly under `check_overflow`.
//!
//! Measured against mpmath at 50 digits with the `N = 40` table, the worst relative
//! error over `x` from 0 to `1e15` is 1.22 ulp.
//!
//! # Negative arguments
//!
//! `erfcx(-x) = 2 e^{x^2} - erfcx(x)`, which genuinely overflows for `x` below about
//! -26.6 (binary64). `erfcx` grows like `e^{x^2}` to the left, so the infinity is the
//! correct answer rather than a failure. This is the only branch, and it is the only
//! place an `exp` appears.

use thermite::{
    element::FloatElement,
    math::{
        CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _, policy::Policy,
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use crate::tables::weideman::{Weideman, WeidemanTables, weideman_n};

/// `erfcx` by the `N`-term Weideman approximation on the imaginary axis.
///
/// `N` is a literal at every call site (the ladder in [`erfcx_internal`] instantiates it
/// as one of 8/16/24/32/40), which is what lets the trip count and the coefficient loads
/// fold.
#[inline(always)]
pub fn erfcx_with<V, E, P, const N: usize>(x: V, l: E, a: &[E; N]) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let l = V::splat(l);
    let ax = x.abs();

    // L + |x| >= L > 0: the one reciprocal, and it needs no guard.
    let r = (l + ax).approx_reciprocal_p::<P>();
    let z = (l - ax) * r;

    let p = z.poly_rev_n_p::<P, _>(a);

    // w = r/sqrt(pi) + 2 P r^2, grouped so the second `r` multiplies once.
    let mut y = r * (p + p).mul_adde(r, V::FRAC_1_SQRT_PI);

    // erfcx(-|x|) = 2 e^{x^2} - erfcx(|x|). Overflows to +inf below x ~ -26.6 in
    // binary64, which is the true behaviour of the function and not a guard failure.
    let neg = x.cmp_lt(V::ZERO);
    if const { P::POLICY.avoid_branching } || thermite::unlikely(neg.any()) {
        let refl = (ax * ax).exp_p::<P>();
        y = neg.select(refl + refl - y, y);
    }

    if const { P::POLICY.check_overflow } {
        // Both infinities need saying. At +inf the reciprocal is 0 but `Z` is
        // `(L - inf) * 0`, i.e. NaN, which the Horner then spreads, and the limit is 0. At
        // -inf the reflection is `inf - NaN` for the same reason, and the limit is +inf.
        y = x.cmp_eq(V::INFINITY).select(V::ZERO, y);
        y = x.cmp_eq(V::NEG_INFINITY).select(V::INFINITY, y);
    }

    y
}

/// [`erfcx_with`], with `N` and the table chosen by the precision policy.
#[inline(always)]
pub fn erfcx_internal<V, E, P>(x: V) -> V
where
    E: FloatElement + WeidemanTables,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    macro_rules! tier {
        ($n:literal) => {
            erfcx_with::<V, E, P, $n>(x, <E as Weideman<$n>>::L, &<E as Weideman<$n>>::A)
        };
    }

    // Spelled out at each arm rather than bound to a `let`: a `const` block cannot
    // capture a local, even one whose initializer is itself constant.
    macro_rules! is {
        ($n:literal) => {
            const { weideman_n(P::POLICY.precision, <E as WeidemanTables>::MAX_N) <= $n }
        };
    }

    if is!(8) {
        tier!(8)
    } else if is!(16) {
        tier!(16)
    } else if is!(24) {
        tier!(24)
    } else if is!(32) {
        tier!(32)
    } else {
        tier!(40)
    }
}
