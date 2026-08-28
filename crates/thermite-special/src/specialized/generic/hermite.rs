use thermite::{
    element::FloatElement,
    math::{
        TranscendentalMathWithPolicy as _,
        policy::{Policy, PrecisionPolicy},
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

/// The two halves of `$\psi_0(x) = \pi^{-1/4} e^{-x^2/2}$`, as `(f, g)` with
/// `f = e^{-x^2/4}` and `g = pi^{-1/4} f`, so `psi_0 = g * f`.
///
/// Splitting the Gaussian in half is what buys the Hermite functions their range. Run
/// naively from `psi_0`, the recurrence carries `psi_k` values that are `O(1)` at most,
/// but the *seed* underflows once `x^2/2` passes the exponent range (about 87 in
/// binary32, 708 in binary64), and past a turning point `$x \approx \sqrt{2n+1}$` the true
/// `psi_n(x)` there is `O(1)`, so degrees above roughly 87 / 708 return zero where they
/// should not. Seeding with `g = pi^{-1/4} f` instead carries `psi_k / f`, which grows only
/// like `e^{+x^2/4}`, and multiplying by `f` at the end restores `psi_k`. Underflow of `f`
/// and overflow of `psi_k / f` now both sit at `x^2/4`, twice as far out: full accuracy at
/// every degree for `|x|` under about 18.7 (binary32) or 53 (binary64), which covers every
/// degree up to about 175 / 1400 everywhere on the line.
///
/// `x^2` is the whole error budget for a Gaussian: an absolute error `d` in the exponent
/// is a relative error `d` in the value, and rounding `x*x` costs `x^2 eps`. Under a
/// `Best`-or-better policy on true-FMA hardware the residual of the square is recovered
/// exactly and applied to first order, taking the seed from `O(x^2 eps)` to `O(eps)`. As
/// in `compound`, no correction is attempted without a fused multiply-add: the residual is
/// only a residual if the product was single-rounded.
#[inline(always)]
fn seed<P, E, V>(x: V) -> (V, V)
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
{
    let neg_quarter = V::splat(<E as FloatElement>::ConstRatio::<{ -1 }, 4>::VALUE);

    let q = x * x;
    let mut f = (q * neg_quarter).exp_p::<P>();

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) && V::HAS_TRUE_FMA } {
        // e^{-(q + q_lo)/4} = f * (1 - q_lo/4) to first order, and q_lo/4 is at most an ulp of
        // q/4 so the second-order term is below working precision. Guarded on a finite
        // square: past overflow f is already the correct zero and the residual is NaN.
        let q_lo = x.mul_sube(x, q);
        f = q.is_finite().select((q_lo * neg_quarter).mul_adde(f, f), f);
    }

    (f, V::FRAC_1_SQRT_SQRT_PI * f)
}

/// The orthonormal Hermite function `$\psi_N(x) = (2^N N! \sqrt{\pi})^{-1/2} e^{-x^2/2} H_N(x)$`.
///
/// Three-term recurrence on the functions themselves,
///
/// ```text
/// psi_{k+1} = sqrt(2/(k+1)) x psi_k - sqrt(k/(k+1)) psi_{k-1}
/// ```
///
/// which keeps every intermediate `O(1)`: the polynomial's growth and the Gaussian's decay
/// cancel *inside* each step instead of being formed separately and multiplied. Both
/// square roots are literals under the unrolled loop, so the per-step cost is one FMA on
/// the critical path plus one multiply beside it. See [`seed`] for the range and the
/// precision of the Gaussian factor.
#[inline(always)]
pub fn hermite_function<P, E, V, const N: usize>(x: V) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
{
    let (f, g0) = seed::<P, E, V>(x);

    if const { N == 0 } {
        return g0 * f;
    }

    // psi_1 = sqrt(2) x psi_0
    let mut p0 = g0;
    let mut p1 = (V::SQRT_2 * x) * g0;

    let mut k = 1;
    while k < N {
        // psi_{k+1} = sqrt(2/(k+1)) x psi_k - sqrt(k/(k+1)) psi_{k-1}, the subtraction
        // carried in the constant. p0 is two steps back, so its multiply is off the chain.
        let ax = x * V::splat(a::<E>(k));
        let next = ax.mul_adde(p1, p0 * V::splat(b::<E>(k)));
        p0 = p1;
        p1 = next;
        k += 1;
    }

    p1 * f
}

/// Runtime-length form of [`hermite_function_series`].
///
/// A genuine port of the recurrence rather than a fold over the const kernel: a series
/// carries `k`-dependent state and does not partition the way the slice reductions in
/// `thermite` do. Both forms must be edited together.
///
/// Same pre-scaling by `f`, same seed, same final factor. Read [`hermite_function_series`]
/// for why the split is there. The runtime length costs the unrolling and turns `a_k`,
/// `b_{k+1}` into per-step square roots of a ratio rather than folded literals, which is
/// the expensive part here.
///
/// The empty series is `0`, where the const form refuses to compile.
#[inline(always)]
pub fn hermite_function_series_slice<P, E, V>(x: V, coeffs: &[E]) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
{
    let n = coeffs.len();

    if n == 0 {
        return V::ZERO;
    }

    let (f, g0) = seed::<P, E, V>(x);

    if n == 1 {
        return (f * V::splat(coeffs[0])) * g0;
    }

    let sqrt2_x = V::SQRT_2 * x;
    let fcn1 = f * V::splat(coeffs[n - 1]);

    if n == 2 {
        return sqrt2_x.mul_adde(fcn1, f * V::splat(coeffs[0])) * g0;
    }

    let mut y2 = fcn1;
    let mut y1 = (x * V::splat(a::<E>(n - 2))).mul_adde(fcn1, f * V::splat(coeffs[n - 2]));

    let mut k = n - 2;
    while k > 1 {
        k -= 1;
        let ax = x * V::splat(a::<E>(k));
        let yk = ax.mul_adde(y1, y2.mul_adde(V::splat(b::<E>(k + 1)), f * V::splat(coeffs[k])));
        y2 = y1;
        y1 = yk;
    }

    sqrt2_x.mul_adde(y1, y2.mul_adde(-V::FRAC_1_SQRT_2, f * V::splat(coeffs[0]))) * g0
}

/// Clenshaw summation of a Hermite-function series, `$\sum_{k=0}^{N-1} c_k \psi_k(x)$`.
///
/// Runs Clenshaw over `h_k = psi_k / psi_0`, whose recurrence is the same as `psi_k`'s, and
/// multiplies by `psi_0` once at the end. `h_k` grows like `e^{+x^2/2}` where `psi_k` is
/// `O(1)`, so to keep the same range as [`hermite_function`] the coefficients are pre-scaled
/// by `f = e^{-x^2/4}` (Clenshaw is linear in them) and the final factor is only
/// `pi^{-1/4} f`: the running values stay within `e^{+x^2/4}` and the outer factor within
/// `e^{-x^2/4}`, the same split as the single-function kernel.
///
/// ```text
/// y_k = f c_k + sqrt(2/(k+1)) x y_{k+1} - sqrt((k+1)/(k+2)) y_{k+2}     k = N-1 down to 1
/// S   = pi^{-1/4} f * (f c_0 + sqrt(2) x y_1 - sqrt(1/2) y_2)
/// ```
#[inline(always)]
pub fn hermite_function_series<P, E, V, const N: usize>(x: V, coeffs: &[E; N]) -> V
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + SpecializedTranscendentalMath<E>,
{
    const {
        assert!(N >= 1, "hermite_function_series: N must be at least 1");
    }

    let (f, g0) = seed::<P, E, V>(x);

    // S = c_0 psi_0
    if const { N == 1 } {
        return (f * V::splat(coeffs[0])) * g0;
    }

    let sqrt2_x = V::SQRT_2 * x;
    let fcn1 = f * V::splat(coeffs[N - 1]);

    // S = psi_0 (c_0 + c_1 sqrt(2) x)
    if const { N == 2 } {
        return sqrt2_x.mul_adde(fcn1, f * V::splat(coeffs[0])) * g0;
    }

    // Hoist the top two steps (zero seeds):
    //     k = N-1:  y = f c_{N-1}
    //     k = N-2:  y = f c_{N-2} + a_{N-2} x (f c_{N-1})
    let mut y2 = fcn1;
    let mut y1 = (x * V::splat(a::<E>(N - 2))).mul_adde(fcn1, f * V::splat(coeffs[N - 2]));

    // k = N-3 down to 1.
    let mut k = N - 2;
    while k > 1 {
        k -= 1;
        // y_k = a_k x y_{k+1} + (f c_k + b_{k+1} y_{k+2}), b negative.
        let ax = x * V::splat(a::<E>(k));
        let yk = ax.mul_adde(y1, y2.mul_adde(V::splat(b::<E>(k + 1)), f * V::splat(coeffs[k])));
        y2 = y1;
        y1 = yk;
    }

    // S = g0 * (sqrt(2) x y_1 + (f c_0 - sqrt(1/2) y_2)); b_1 = -sqrt(1/2) = -1/sqrt(2).
    sqrt2_x.mul_adde(y1, y2.mul_adde(-V::FRAC_1_SQRT_2, f * V::splat(coeffs[0]))) * g0
}

/// `sqrt(2/(k+1))`, the coefficient of `x psi_k` in the step to `psi_{k+1}`.
#[inline(always)]
fn a<E: FloatElement>(k: usize) -> E {
    FloatElement::sqrt(E::from_ratio(2, (k + 1) as thermite::LargeInt))
}

/// `-sqrt(k/(k+1))`, the coefficient of `psi_{k-1}` in the step to `psi_{k+1}`, negated.
#[inline(always)]
fn b<E: FloatElement>(k: usize) -> E {
    -FloatElement::sqrt(E::from_ratio(k as thermite::LargeInt, (k + 1) as thermite::LargeInt))
}
