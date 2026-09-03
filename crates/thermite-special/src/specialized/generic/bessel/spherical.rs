//! The **spherical** Bessel functions `$j_n$`, `$y_n$`, `$i_n$`, `$k_n$`.
//!
//! ```math
//! j_n(x) = \sqrt{\tfrac{\pi}{2x}}\,J_{n+1/2}(x), \qquad
//! y_n(x) = \sqrt{\tfrac{\pi}{2x}}\,Y_{n+1/2}(x)
//! ```
//!
//! and likewise for the modified pair. Both references ship these publicly: Boost as
//! `sph_bessel` / `sph_neumann`, SciPy as `spherical_jn` /
//! `spherical_yn` / `spherical_in` / `spherical_kn`. Boost has no modified spherical pair.
//! SciPy has no oscillating primes as separate names. This module is the union.
//!
//! # The `sqrt` never gets formed, because it would only be cancelled
//!
//! Boost implements `sph_bessel` as literally `sqrt(pi/(2x)) * cyl_bessel_j(n + 1/2, x)`.
//! That is two square roots that multiply to `$1/x$`: the cylindrical
//! kernel builds its answer on `$\sqrt{2/\pi x}$` and the wrapper immediately multiplies by
//! `$\sqrt{\pi/2x}$`.
//!
//! Here the recurrence is seeded in the **spherical** normalization directly:
//!
//! ```math
//! j_{-1} = \frac{\cos x}{x},\quad j_0 = \frac{\sin x}{x}, \qquad
//! y_{-1} = \frac{\sin x}{x},\quad y_0 = -\frac{\cos x}{x}
//! ```
//!
//! The scaling factor between the two conventions does not depend on the order, so the
//! recurrence is unchanged and [`walk_jy`](super::half::walk_jy) is reused verbatim
//! (seeds in one normalization, values out in the same one). Two `sqrt`s, two divisions and a
//! rounding disappear, and `$j_0$` becomes exactly [`sinc`](thermite::math::TranscendentalMath::sinc),
//! which is correct **at `$x = 0$`** where the cylindrical route is `$0 \cdot \infty$`.
//!
//! Boost needs a small-`$z$` series below `$x = 1$` for that reason. This
//! module needs none: the downward recurrence already covers small `$x$`, and is the arm
//! that runs there anyway, since `$n < x$` is what selects the forward one.
//!
//! # Order
//!
//! `$n \ge 0$`, matching both references (Boost takes `unsigned`, SciPy documents `n >= 0`),
//! and spelled `usize` so the constraint is the type rather than an assertion. That is the one
//! place this family deliberately diverges from the cylindrical entry points, whose `i32`
//! exists because negative orders there are meaningful. A caller who wants one here can use
//! `$j_{-n-1}(x) = (-1)^{n+1} y_n(x)$`.
//!
//! # Negative `x`
//!
//! Unlike their cylindrical parents at half-integer order, `$j_n$`, `$y_n$` and `$i_n$` are
//! elementary in `$\sin x$`, `$\cos x$`, `$\sinh x$`, `$\cosh x$` and powers of `$1/x$`, so
//! they are real on the whole line and have definite parity:
//!
//! ```math
//! j_n(-x) = (-1)^n j_n(x), \qquad y_n(-x) = (-1)^{n+1} y_n(x), \qquad i_n(-x) = (-1)^n i_n(x)
//! ```
//!
//! which is SciPy's convention for `spherical_jn` / `spherical_yn` / `spherical_in`. The
//! kernels evaluate on `$\lvert x\rvert$` and apply the sign at the end. They must, because the
//! downward walk's trip count is `$a + 24 + c\,x$` and a negative `$x$` would shorten it to
//! nothing. `$k_n$` has no parity (it is `$e^{-x}$` against `$e^{x}$`) and is NaN off the
//! positive axis, as the cylindrical `$K$` and SciPy's `spherical_kn` are.

use thermite::{
    math::{TranscendentalMathWithPolicy, policy::Policy},
    prelude::*,
};

use thermite::element::FloatElement;

use super::half::{walk_ik, walk_jy};
use super::ik::unscale_i_pair;

/// `$(j_{n-1},\; j_n,\; y_{n-1},\; y_n)$`, on the whole real line.
///
/// The neighbour below comes back too, because every derivative identity in this family reaches
/// down one order and the walk passes through it regardless:
/// `$f_n' = f_{n-1} - \frac{n+1}{x} f_n$`.
#[inline(always)]
pub fn sph_jy_impl_n<P, E, V, const N: usize>(x: V) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = V::splat(E::from_ratio(2 * N as i64 + 1, 2));

    // Evaluated on `|x|` and signed at the end. See the module docs.
    let ax = x.abs();

    // Only the cosine is taken from here: the sine appears solely as `sin x / x`, which is
    // `sinc`: identical away from the origin, and exactly 1 at it.
    let cos_x = ax.cos_p::<P>();
    let inv_x = V::ONE / ax;

    // `sinc` rather than `sin_x * inv_x`: identical away from the origin, and exactly 1 at it,
    // which is the value `j_0(0)` actually has.
    let j_lo = cos_x * inv_x; // j_{-1}
    let j_hi = ax.sinc_p::<P>(); // j_0
    let y_lo = j_hi; // y_{-1} =  j_0
    let y_hi = -j_lo; // y_0    = -j_{-1}

    // Order zero is the seeds, taken directly: the walk would route `n = 0` through its
    // downward arm for `x < 1/2` and rebuild `j_0` as `j_{-1} r_{1/2}`, 1 ulp off at
    // `x = 10^{-300}` where `j_0` should be `sinc` to the bit.
    let (j_prev, j_n, y_prev, y_n) = match const { N == 0 } {
        true => (j_lo, j_hi, y_lo, y_hi),
        false => walk_jy::<P, E, V>(ax, a, j_lo, j_hi, y_lo, y_hi),
    };

    // `x = 0`: `j_0` is 1 and every higher order is 0, while every `y_n` is `-inf`. The
    // recurrence cannot produce these (`1/x` is infinite and the downward normalization goes
    // `inf * 0`), so the origin is a select. It is one compare for a value callers do ask for.
    //
    // The neighbour gets the same treatment one order down, which at `n = 0` means `j_{-1}`
    // and `y_{-1}`: `cos(0)/0` is infinite and `sin(0)/0` is one.
    let at_zero = x.is_zero();

    let j_zero = if const { N == 0 } { V::ONE } else { V::ZERO };
    let j_prev_zero = if const { N == 0 } {
        V::INFINITY
    } else if const { N == 1 } {
        V::ONE
    } else {
        V::ZERO
    };
    let y_prev_zero = if const { N == 0 } { V::ONE } else { V::NEG_INFINITY };

    let j_prev = at_zero.select(j_prev_zero, j_prev);
    let j_n = at_zero.select(j_zero, j_n);
    let y_prev = at_zero.select(y_prev_zero, y_prev);
    let y_n = at_zero.select(V::NEG_INFINITY, y_n);

    // All four vanish at infinity, where the seeds are `NaN * 0`.
    let inf = ax.cmp_eq(V::INFINITY);
    let (j_prev, j_n) = (inf.select(V::ZERO, j_prev), inf.select(V::ZERO, j_n));
    let (y_prev, y_n) = (inf.select(V::ZERO, y_prev), inf.select(V::ZERO, y_n));

    // The parity fold. Orders `n` and `n - 1` have opposite parity, and `y` has the opposite
    // of `j` at each.
    let neg = x.is_negative();
    match const { N % 2 == 1 } {
        true => (j_prev, j_n.neg_c(neg), y_prev.neg_c(neg), y_n),
        false => (j_prev.neg_c(neg), j_n, y_prev, y_n.neg_c(neg)),
    }
}

/// `$(i_{n-1},\; i_n,\; k_{n-1},\; k_n)$`, scaled by `$(e^{-|x|}, e^{x})$` when `SCALED`.
/// `$i_n$` is folded by parity onto the whole line. `$k_n$` is NaN for `$x < 0$`.
///
/// The modified spherical pair, `$i_n(x) = \sqrt{\pi/2x}\,I_{n+1/2}(x)$` and likewise for
/// `$k$`. SciPy ships both as `spherical_in` / `spherical_kn`. Boost ships neither.
///
/// Seeds are `$i_{-1} = \cosh x / x$`, `$i_0 = \sinh x / x$` and
/// `$k_{-1} = k_0 = \tfrac{\pi}{2}e^{-x}/x$`, the last two equal because `$K$` is even in
/// order. As with the oscillating pair, the work is done in the spherical normalization so no
/// square root is formed only to be cancelled.
///
/// `far_threshold` is where the unscaled `$i$` halves its exponential. See
/// [`unscale_i_pair`].
#[inline(always)]
pub fn sph_ik_impl_n<P, E, V, const N: usize, const SCALED: bool>(x: V, far_threshold: E) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = V::splat(E::from_ratio(2 * N as i64 + 1, 2));

    // Evaluated on `|x|` and signed at the end. See the module docs.
    let ax = x.abs();
    let inv_x = V::ONE / ax;

    // Scaled seeds, from one `exp_m1`. `e^{-x} sinh x = -expm1(-2x)/2` and
    // `e^{-x} cosh x = (1 + e^{-2x})/2`, neither of which cancels at small `x`. The
    // algebraically equal `(1 - e^{-2x})/2` would.
    let e2m1 = (-(ax + ax)).exp_m1_p::<P>();
    let half_inv_x = inv_x * V::HALF;

    let i_lo = (V::TWO + e2m1) * half_inv_x; // e^{-x} i_{-1}
    let i_hi = -e2m1 * half_inv_x; //          e^{-x} i_0
    let k_seed = V::FRAC_PI_2 * inv_x; //      e^{ x} k_0 = e^{x} k_{-1}

    // Order zero is the seeds: cheaper and exacter, for the reason given in `sph_jy_impl_n`.
    let (i_prev, i_n, k_prev, k_n) = match const { N == 0 } {
        true => (i_lo, i_hi, k_seed, k_seed),
        // The large-`x` asymptotic arm inside the walk produces a CYLINDRICAL value and is
        // handed the factor that brings it back to this normalization.
        false => walk_ik::<P, E, V>(ax, a, i_lo, i_hi, k_seed, (V::FRAC_PI_2 * inv_x).sqrt()),
    };

    // The origin, as for the oscillating pair: `i_0(0) = 1`, higher orders zero, every `k_n`
    // infinite. `k` reaches that on its own through `1/x`, but `i`'s seeds are `0/0` there.
    let at_zero = x.is_zero();

    let i_zero = if const { N == 0 } { V::ONE } else { V::ZERO };
    let i_prev_zero = if const { N == 0 } {
        V::INFINITY
    } else if const { N == 1 } {
        V::ONE
    } else {
        V::ZERO
    };

    let (i_prev, i_n) = (at_zero.select(i_prev_zero, i_prev), at_zero.select(i_zero, i_n));

    // The parity fold for `i`, as for `j`. `k` has none and is undefined there.
    let neg = x.is_negative();
    let (i_prev, i_n) = match const { N % 2 == 1 } {
        true => (i_prev, i_n.neg_c(neg)),
        false => (i_prev.neg_c(neg), i_n),
    };
    let bad = x.cmp_lt(V::ZERO);
    let (k_prev, k_n) = (bad.select(V::NAN, k_prev), bad.select(V::NAN, k_n));

    match const { SCALED } {
        true => (i_prev, i_n, k_prev, k_n),
        false => {
            let (i_prev, i_n) = unscale_i_pair::<P, E, V>(i_prev, i_n, ax, far_threshold);
            let em = (-ax).exp_p::<P>();
            (i_prev, i_n, k_prev * em, k_n * em)
        }
    }
}

/// `$f_n'(x)$` from the pair the walk returns: `$f_n' = \pm f_{n-1} - \frac{n+1}{x} f_n$`.
///
/// `MINUS` selects the `$k$` case, whose neighbour enters negated, the same asymmetry the
/// cylindrical `$K$` has, and for the same reason: `$K$` is the decaying solution, so its
/// derivative is negative where the others' are not.
///
/// # Where this comes from
///
/// Not a separate identity: the cylindrical one plus the derivative of the
/// normalization. With `$f_n = \sqrt{\pi/2x}\,F_{n+1/2}$` and
/// `$F_\nu' = F_{\nu-1} - \frac{\nu}{x}F_\nu$`, the extra `$-\frac{1}{2x}$` from
/// differentiating `$\sqrt{\pi/2x}$` turns `$\frac{n+1/2}{x}$` into `$\frac{n+1}{x}$`. That is
/// the whole difference. It is why the coefficient is `$n+1$` rather than the `$n$` a
/// half-remembered version of this formula would use.
#[inline(always)]
pub fn sph_deriv_n<E, V, const N: usize, const MINUS: bool>(x: V, prev: V, cur: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let coeff = V::splat(E::from_int(N as i64 + 1)) / x;
    let p = match const { MINUS } {
        true => -prev,
        false => prev,
    };
    let d = coeff.nmul_adde(cur, p);

    // The origin, where `(n+1)/x` is infinite and the identity reads `inf * 0` or
    // `inf - inf`. `j_1'(0) = i_1'(0) = 1/3` (from `j_1 ~ x/3`) and every other finite
    // member has a zero derivative there. The singular members' derivatives are infinite with
    // the opposite sign to the value, since `y_n -> -inf` rises and `k_n -> +inf` falls.
    let finite_limit = if const { N == 1 } {
        V::splat(E::from_ratio(1, 3))
    } else {
        V::ZERO
    };
    let limit = cur.is_finite().select(finite_limit, -cur);
    x.is_zero().select(limit, d)
}

// ---- runtime-order twins ------------------------------------------------------------------
//
// The same three kernels with the order as a value. Every `const { N .. }` above is a plain
// branch or select on `n` here and the walk is handed the same `a`, so the two forms agree to
// the bit. That equality is what `tests/bessel_sph.rs` checks. The const forms stay because
// their origin selects and parity fold cost nothing at a literal order.

/// The runtime-order twin of [`sph_jy_impl_n`].
#[inline(always)]
pub fn sph_jy_impl<P, E, V>(x: V, n: u32) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = V::splat(E::from_ratio(2 * n as i64 + 1, 2));

    let ax = x.abs();
    let cos_x = ax.cos_p::<P>();
    let inv_x = V::ONE / ax;

    let j_lo = cos_x * inv_x;
    let j_hi = ax.sinc_p::<P>();
    let y_lo = j_hi;
    let y_hi = -j_lo;

    let (j_prev, j_n, y_prev, y_n) = match n == 0 {
        true => (j_lo, j_hi, y_lo, y_hi),
        false => walk_jy::<P, E, V>(ax, a, j_lo, j_hi, y_lo, y_hi),
    };

    let at_zero = x.is_zero();

    let j_zero = if n == 0 { V::ONE } else { V::ZERO };
    let j_prev_zero = match n {
        0 => V::INFINITY,
        1 => V::ONE,
        _ => V::ZERO,
    };
    let y_prev_zero = if n == 0 { V::ONE } else { V::NEG_INFINITY };

    let j_prev = at_zero.select(j_prev_zero, j_prev);
    let j_n = at_zero.select(j_zero, j_n);
    let y_prev = at_zero.select(y_prev_zero, y_prev);
    let y_n = at_zero.select(V::NEG_INFINITY, y_n);

    let inf = ax.cmp_eq(V::INFINITY);
    let (j_prev, j_n) = (inf.select(V::ZERO, j_prev), inf.select(V::ZERO, j_n));
    let (y_prev, y_n) = (inf.select(V::ZERO, y_prev), inf.select(V::ZERO, y_n));

    let neg = x.is_negative();
    match n % 2 == 1 {
        true => (j_prev, j_n.neg_c(neg), y_prev.neg_c(neg), y_n),
        false => (j_prev.neg_c(neg), j_n, y_prev, y_n.neg_c(neg)),
    }
}

/// The runtime-order twin of [`sph_ik_impl_n`].
#[inline(always)]
pub fn sph_ik_impl<P, E, V, const SCALED: bool>(x: V, n: u32, far_threshold: E) -> (V, V, V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let a = V::splat(E::from_ratio(2 * n as i64 + 1, 2));

    let ax = x.abs();
    let inv_x = V::ONE / ax;

    let e2m1 = (-(ax + ax)).exp_m1_p::<P>();
    let half_inv_x = inv_x * V::HALF;

    let i_lo = (V::TWO + e2m1) * half_inv_x;
    let i_hi = -e2m1 * half_inv_x;
    let k_seed = V::FRAC_PI_2 * inv_x;

    let (i_prev, i_n, k_prev, k_n) = match n == 0 {
        true => (i_lo, i_hi, k_seed, k_seed),
        false => walk_ik::<P, E, V>(ax, a, i_lo, i_hi, k_seed, (V::FRAC_PI_2 * inv_x).sqrt()),
    };

    let at_zero = x.is_zero();

    let i_zero = if n == 0 { V::ONE } else { V::ZERO };
    let i_prev_zero = match n {
        0 => V::INFINITY,
        1 => V::ONE,
        _ => V::ZERO,
    };

    let (i_prev, i_n) = (at_zero.select(i_prev_zero, i_prev), at_zero.select(i_zero, i_n));

    let neg = x.is_negative();
    let (i_prev, i_n) = match n % 2 == 1 {
        true => (i_prev, i_n.neg_c(neg)),
        false => (i_prev.neg_c(neg), i_n),
    };
    let bad = x.cmp_lt(V::ZERO);
    let (k_prev, k_n) = (bad.select(V::NAN, k_prev), bad.select(V::NAN, k_n));

    match const { SCALED } {
        true => (i_prev, i_n, k_prev, k_n),
        false => {
            let (i_prev, i_n) = unscale_i_pair::<P, E, V>(i_prev, i_n, ax, far_threshold);
            let em = (-ax).exp_p::<P>();
            (i_prev, i_n, k_prev * em, k_n * em)
        }
    }
}

/// The runtime-order twin of [`sph_deriv_n`].
#[inline(always)]
pub fn sph_deriv<E, V, const MINUS: bool>(x: V, n: u32, prev: V, cur: V) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let coeff = V::splat(E::from_int(n as i64 + 1)) / x;
    let p = match const { MINUS } {
        true => -prev,
        false => prev,
    };
    let d = coeff.nmul_adde(cur, p);

    let finite_limit = if n == 1 { V::splat(E::from_ratio(1, 3)) } else { V::ZERO };
    let limit = cur.is_finite().select(finite_limit, -cur);
    x.is_zero().select(limit, d)
}
