//! The Faddeeva function `$w(z) = e^{-z^2}\operatorname{erfc}(-iz)$`.
//!
//! # Algorithm
//!
//! Weideman's rational approximation (*Computation of the complex error function*,
//! SIAM J. Numer. Anal. **31**(5), 1497-1518, 1994). For `$\operatorname{Im} z \ge 0$`,
//! with `$L = 2^{-1/4}\sqrt{N}$`:
//!
//! ```math
//! Z = \frac{L + iz}{L - iz}, \qquad
//! w(z) \approx \frac{1}{\sqrt{\pi}\,(L - iz)} + \frac{2\,P(Z)}{(L - iz)^2}
//! ```
//!
//! where `$P$` has degree `$N-1$` and **real** coefficients. This is the only method in
//! the literature that is a single branch-free closed form over the whole upper
//! half-plane, and that is why it was chosen:
//!
//! - No region split. Every competing algorithm of comparable accuracy (Humlicek's w4,
//!   Zaghloul's Algorithm 916/985, Poppe & Wijers' Algorithm 680, Al Azah &
//!   Chandler-Wilde's modified trapezoidal rules) partitions the plane, which under
//!   SIMD costs the sum of every region rather than the one that applies.
//! - **One real reciprocal.** `$L - iz = (L + y) - ix$`, so `$|L - iz| \ge L + y \ge L$`
//!   and the inversion needs no guard. See [`weideman`] for the single-reciprocal form.
//! - No transcendentals at all in the upper half-plane: no `exp`, `ln`, or `sincos`.
//! - `$|Z| \le 1$` for `$\operatorname{Im} z \ge 0$`, so the polynomial argument is
//!   confined to the closed unit disk and the Horner recurrence is well conditioned.
//!   The only pole is at `$z = -iL$`, in the lower half-plane, and the reflection
//!   evaluates at `$-z$`, so it is never approached.
//!
//! # Accuracy
//!
//! Measured against a 50-digit oracle over a grid covering `$|z|$` from `1e-8` to `1e5`
//! and `$\operatorname{Im} z$` from `0` to `30`:
//!
//! | N | 8 | 16 | 24 | 32 | 40 |
//! |---|---|----|----|----|----|
//! | max relative error | 3.1e-4 | 4.3e-7 | 4.2e-10 | 3.1e-13 | 8.7e-16 |
//!
//! Convergence is geometric, about `$10^{-0.385 N}$`. `N = 40` reaches the binary64
//! roundoff floor; `N = 48` measures no better (6.1e-16 against 8.7e-16 - both are the
//! floor, not convergence), so 40 ends the ladder and [`weideman_n`] never exceeds it.
//! f32 floors at `N = 16` (~5 ulp) for the same reason.
//!
//! This error is **normwise**, relative to `$|w|$`, and uniform over the upper
//! half-plane: it does not degrade at `$\operatorname{Im} z = 0$`, and it *improves*
//! with large `$|\operatorname{Re} z|$` (2.6e-16 at `x = 1e4`, `N = 32`).
//!
//! The received wisdom that this method "fails near the real axis" is a statement about
//! `$\operatorname{Re} w$` alone, and it is a property of `$w$` rather than of the
//! approximation. On the real axis `$\operatorname{Re} w(x) = e^{-x^2}$` while
//! `$|w| \sim 1/(\sqrt{\pi}x)$`, so a uniform normwise error `$\epsilon$` lands as
//! about `$\epsilon\,x/y$` on the real part alone. Consumers of the complex value -
//! `erf`, `erfc`, `erfi`, Dawson - are unaffected. A Voigt profile, which *is*
//! `$\operatorname{Re} w$`, is not, so `real_axis_w` takes that strip over at
//! `Best` and above. It needs no separate Dawson kernel: on the axis
//! `$\operatorname{Im} w(x) = \frac{2}{\sqrt{\pi}}F(x)$` *is* Dawson's integral, and it
//! is the component this approximation delivers to full relative accuracy, so the seed
//! is already in hand and only `$\operatorname{Re} w = e^{-x^2}$` has to be restored.

use thermite::math::policy::{Policy, PrecisionPolicy};
use thermite::math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite::register::FloatElement;

use crate::Complex;
use crate::vector::RealFloatVector;

mod tables;

/// Weideman's coefficients for an `N`-term approximation.
///
/// `A` is stored **leading-term-first**, matching both MATLAB's `polyval` order (which
/// is what the reference `cef.m` produces after its `flipud`) and
/// [`poly_rev`](thermite::math::specialized::SpecializedCoreMath::poly_rev). Feeding it
/// to a constant-term-first evaluator silently produces a different polynomial that
/// happens to agree at `Z = 1`; that exact confusion has already caused one shipped bug
/// in this workspace's Lanczos denominator.
///
/// The values come from the recipe in `reference/cef.m` - sample `$e^{-t^2}(L^2 + t^2)$`
/// on the tangent grid `$t = L\tan(\theta/2)$` at `$4N$` points, take the real FFT, keep
/// and reverse entries `$1..N$`. They are tabulated rather than computed because the FFT
/// is not available at const-eval time.
pub trait Weideman<const N: usize>: Sized {
    /// The scaling parameter `$L = 2^{-1/4}\sqrt{N}$`.
    const L: Self;

    /// Polynomial coefficients, leading-term-first.
    const A: [Self; N];
}

/// The tiers of [`Weideman`] an element type provides, and where its ladder stops.
pub trait WeidemanTables: Weideman<8> + Weideman<16> + Weideman<24> + Weideman<32> + Weideman<40> {
    /// The largest `N` worth using: past it the element's own roundoff dominates and
    /// more terms measure no better.
    const MAX_N: usize;

    /// `$|z|$` past which `$(L + y)^2 + x^2$` would overflow the element even though
    /// `$w(z)$` itself is perfectly representable. Roughly `$\sqrt{\text{MAX}}$`.
    const HUGE: Self;

    /// `$\operatorname{Im} z$` below which the near-real-axis path takes over the real
    /// part. See the [module docs](self); the correction is gated at `Best` and above.
    const REAL_AXIS_Y: Self;

    /// `$|\operatorname{Re} z|$` above which it does not, the correction's own error
    /// growing like `$x^2$` until the direct evaluation is the better of the two.
    const REAL_AXIS_X: Self;
}

/// The term count for a precision tier, clamped to what `max_n` can deliver.
///
/// Each tier is the smallest `N` whose measured error (see the [module docs](self))
/// clears the next format-relevant threshold.
pub const fn weideman_n(precision: PrecisionPolicy, max_n: usize) -> usize {
    let n = if precision.le(PrecisionPolicy::Worst) {
        8 // 3.1e-4
    } else if precision.le(PrecisionPolicy::Medium) {
        16 // 4.3e-7, the f32 floor
    } else if precision.le(PrecisionPolicy::Average) {
        24 // 4.2e-10
    } else if precision.le(PrecisionPolicy::Best) {
        32 // 3.1e-13
    } else {
        40 // 8.7e-16, the f64 floor
    };

    if n > max_n { max_n } else { n }
}

/// Horner over **real** coefficients at a complex argument, leading-term-first.
///
/// [`poly_rev`](thermite::math::specialized::SpecializedCoreMath::poly_rev) cannot serve
/// here: its coefficients are `Self::Element`, which for `Complex<V>` is `Complex<E>`,
/// so it would carry `N` known-zero imaginary parts through the whole chain - a wasted
/// add per term and twice the table. The iteration order matches `poly_rev`'s.
///
/// `N` is a literal at every call site (the ladder in [`faddeeva_w`] instantiates it as
/// one of 8/16/24/32/40), which is what lets the trip count and the coefficient loads
/// fold.
#[inline(always)]
fn horner_real<E, V, const N: usize>(z: Complex<V>, a: &[E; N]) -> Complex<V>
where
    E: FloatElement,
    V: RealFloatVector<Element = E>,
{
    let (zr, zi) = (z.re, z.im);

    // p = p*z + c, with c real:
    //   re = p.re*zr - p.im*zi + c
    //   im = p.re*zi + p.im*zr
    let mut re = V::splat(a[0]);
    let mut im = V::ZERO;

    let mut i = 1usize;
    while i < N {
        unsafe { core::hint::assert_unchecked(i < N) };

        let c = V::splat(a[i]);
        let next_re = re.mul_adde(zr, im.nmul_adde(zi, c));
        let next_im = re.mul_adde(zi, im * zr);

        re = next_re;
        im = next_im;
        i += 1;
    }

    Complex::new(re, im)
}

/// `w(z)` by the `N`-term Weideman approximation. **Valid only for `Im z >= 0`.**
///
/// Written to need exactly one real reciprocal:
///
/// ```text
/// r = 1/(L - iz)              -- Complex::rcp, i.e. conj(d)/|d|^2
/// Z = (L + iz) * r
/// w = (2*P(Z)*r + 1/sqrt(pi)) * r
/// ```
///
/// The factor 2 is applied as `pr + pr` rather than folded into the table, so the
/// coefficients stay byte-identical to `cef.m`'s output for review. Doubling is exact,
/// and a complex add costs the same as a complex scale.
#[inline(always)]
pub fn weideman_with<P: Policy, E, V, const N: usize>(z: Complex<V>, l: E, a: &[E; N]) -> Complex<V>
where
    E: FloatElement,
    V: RealFloatVector<Element = E>,
{
    let l = V::splat(l);

    // L - iz = (L + y) - ix  and  L + iz = (L - y) + ix
    let r = Complex::new(l + z.im, -z.re).reciprocal_p::<P>();
    let zz = Complex::new(l - z.im, z.re) * r;

    let pr = horner_real::<E, V, N>(zz, a) * r;

    (pr + pr + V::FRAC_1_SQRT_PI) * r
}

/// [`weideman_with`], taking the table from the element's own [`Weideman`] impl.
#[inline(always)]
pub fn weideman<P: Policy, E, V, const N: usize>(z: Complex<V>) -> Complex<V>
where
    E: FloatElement + Weideman<N>,
    V: RealFloatVector<Element = E>,
{
    weideman_with::<P, E, V, N>(z, <E as Weideman<N>>::L, &<E as Weideman<N>>::A)
}

/// `w(z)` near the real axis, where the direct evaluation loses the real part.
///
/// `$\operatorname{Re} w$` is not recoverable from a normwise-accurate evaluation there:
/// at `$z = -8$` the true value is `$e^{-64} = 1.6 \times 10^{-28}$` while the noise floor
/// `$\epsilon|w|$` is `$1.6 \times 10^{-17}$`, eleven orders above it. Every intermediate
/// is a normal number - the smallest is 0.07 - so this is lost information, not underflow,
/// and no rearrangement of the same expression recovers it.
///
/// So the real part is taken from the one place it is exact, and the imaginary part from
/// the one place the direct evaluation is reliable:
///
/// ```text
/// a_0 = exp(-x^2) + i*Im w(x, 0)
/// ```
///
/// `$\operatorname{Im} w(x) = \frac{2}{\sqrt{\pi}}F(x)$` is Dawson's integral, and it is
/// *what `$|w|$` is made of* out on the axis - so Weideman delivers it to full relative
/// accuracy (measured 1e-13 at N=32, 6e-16 at N=40 against a 50-digit oracle). No separate
/// Dawson kernel is needed; the value falls out of the kernel already here.
///
/// Stepping off the axis uses `$w'(z) = -2zw + 2i/\sqrt{\pi}$` differentiated `n` times,
/// which gives `$a_{n+1} = -2(x a_n + a_{n-1})/(n+1)$`. Folding `$(iy)^n$` into the terms
/// as `$b_n = a_n (iy)^n$` turns that into a recursion with no complex powers and no
/// divisions:
///
/// ```text
/// b_{n+1} = (-2/(n+1)) * (x*(iy)*b_n - y^2*b_{n-1})
/// ```
///
/// Four terms, and deliberately not more. The series is asymptotic rather than
/// convergent once `$xy$` grows, so outside the gate extra terms make it *worse* - at
/// `$x = 10^6, y = 10^{-5}$`, four give 3e-3 and twelve give 9.0. Inside the gate,
/// terms beyond the fourth contribute nothing measurable.
///
/// # Cost
///
/// A second [`weideman`] evaluation, at `$(x, 0)$`, plus one `exp` and two recursion
/// steps. The second evaluation is not avoidable: seeding from the imaginary part of the
/// `$(x, y)$` evaluation already in hand drifts by about `$0.02y$` relative, which is fine
/// at `$y = 10^{-8}$` and useless by `$y = 10^{-2}$`.
#[inline(always)]
fn real_axis_w<P: Policy, E, V, const N: usize>(x: V, y: V, l: E, a: &[E; N]) -> Complex<V>
where
    E: FloatElement,
    V: RealFloatVector<Element = E>,
{
    let on_axis = weideman_with::<P, E, V, N>(Complex::new(x, V::ZERO), l, a);

    // b_0 = w(x, 0), with the real part replaced by the value it provably has
    let b0 = Complex::new((-(x * x)).exp_p::<P>(), on_axis.im);

    // b_1 = a_1 * (iy), a_1 = w'(x) = -2x*b_0 + 2i/sqrt(pi). The 2x is real, so the
    // scale-and-subtract is the componentwise FMA.
    let a1 = b0.nmul_adde(x + x, Complex::new(V::ZERO, V::FRAC_2_SQRT_PI));
    let b1 = Complex::new(-y * a1.im, y * a1.re);

    // x*(iy)*b - y^2*prev, the bracket of the recursion. (iy)*b swaps the components
    // and negates one, so this costs four FMAs and no multiply by i.
    let c = x * y;
    let y2 = y * y;
    let bracket = |b: Complex<V>, prev: Complex<V>| {
        Complex::new(prev.re.nmul_adde(y2, -(c * b.im)), prev.im.nmul_adde(y2, c * b.re))
    };

    // n = 1: factor -2/2 = -1.  n = 2: factor -2/3.
    let b2 = -bracket(b1, b0);
    let factor: V = thermite::const_splat!(ratio <E>: -2, 3);
    let b3 = bracket(b2, b1) * factor;

    (b0 + b1) + (b2 + b3)
}

/// `w(z)` over the whole complex plane, with `N` chosen by the precision policy.
///
/// The upper half-plane is [`weideman`] directly. The lower half-plane uses the
/// reflection `$w(z) = 2e^{-z^2} - w(-z)$`, which is where the only two hazards live:
///
/// - `$e^{-z^2}$` overflows for `$\operatorname{Im} z < -\sqrt{\ln(\text{MAX})}$`
///   (about -26.64 in binary64), but `$w$` itself overflows there too, so the infinity
///   is the correct answer and is not guarded.
/// - `$e^{-z^2}$` needs `$\sin/\cos(2xy)$`, and `$2xy$` grows without bound - about
///   `2e5` radians at `$z = 10^4 - 10i$`. This is the one place in the function where
///   argument-reduction error is the *entire* error, the output range being fixed while
///   the argument is not. Accuracy in the lower half-plane at large `$|xy|$` is bounded
///   by the underlying `sincos` reduction, not by `N`.
#[inline(always)]
pub fn faddeeva_w_with<P: Policy, E, V, const N: usize>(
    z: Complex<V>,
    l: E,
    a: &[E; N],
    huge: E,
    real_axis_y: E,
    real_axis_x: E,
) -> Complex<V>
where
    E: FloatElement,
    V: RealFloatVector<Element = E>,
{
    // Signed zero deliberately reads as the upper half-plane: the reflection agrees
    // there anyway, and this way the real axis never pays for an exp.
    let lower = z.im.cmp_lt(V::ZERO);

    // A conditional negation, not a blend: `neg_c` is a masked sign-bit flip, where
    // `select(-z, z)` also pays for two `blendv`s.
    let zu = Complex::new(z.re.neg_c(lower), z.im.neg_c(lower));
    let mut w = weideman_with::<P, E, V, N>(zu, l, a);

    // The real-axis correction applies to the reflected point, before the reflection
    // undoes it: a lane just below the axis is just as close to it as one just above.
    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let near = zu.im.cmp_lt(V::splat(real_axis_y)) & zu.re.abs().cmp_lt(V::splat(real_axis_x));

        if thermite::unlikely(near.any()) {
            w = near.select(real_axis_w::<P, E, V, N>(zu.re, zu.im, l, a), w);
        }
    }

    let mut res = w;

    if thermite::unlikely(lower.any()) {
        let refl = (-z.square()).exp_p::<P>();
        res = lower.select(refl + refl - w, w);
    }

    if const { P::POLICY.check_overflow } {
        // Past sqrt(MAX) the reciprocal's (L+y)^2 + x^2 overflows even though
        // w(z) ~ i/(sqrt(pi) z) is representable. One leading convergent covers it.
        let h = V::splat(huge);
        let big = z.re.abs().cmp_gt(h) | z.im.abs().cmp_gt(h);

        if thermite::unlikely(big.any()) {
            let asym = Complex::new(V::ZERO, V::FRAC_1_SQRT_PI) * z.reciprocal_p::<P>();
            res = big.select(asym, res);
        }
    }

    res
}

/// [`faddeeva_w_with`], with `N` and the table chosen by the precision policy.
#[inline(always)]
pub fn faddeeva_w<P: Policy, E, V>(z: Complex<V>) -> Complex<V>
where
    E: FloatElement + WeidemanTables,
    V: RealFloatVector<Element = E>,
{
    macro_rules! tier {
        ($n:literal) => {
            faddeeva_w_with::<P, E, V, $n>(
                z,
                <E as Weideman<$n>>::L,
                &<E as Weideman<$n>>::A,
                E::HUGE,
                E::REAL_AXIS_Y,
                E::REAL_AXIS_X,
            )
        };
    }

    // Spelled out at each arm rather than bound to a `let`: a `const` block cannot
    // capture a local, even one whose initializer is itself constant.
    macro_rules! is {
        ($n:literal) => {
            const { weideman_n(P::POLICY.precision, E::MAX_N) <= $n }
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
