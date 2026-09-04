//! Differential tests for `Complex<V>`: every math kernel against `num_complex`
//! as a scalar oracle, plus the vector-trait surface (memory, lanes, ordering).

use num_complex::Complex64;
use thermite::math::policy::policies::Precision;
use thermite::math::{CoreMath, SpatialMath, TranscendentalMath};
use thermite::prelude::*;

use thermite_complex::Complex;
use thermite_complex::prelude::{ComplexMath, ComplexMathWithPolicy, ComplexVector};

/// A 1-lane f64 vector: the scalar reference backend, so a `Complex<V>` lane is
/// directly comparable to a `Complex64`.
type V = Vector<f64>;
type C = Complex<V>;

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

fn oracle(z: C) -> Complex64 {
    let (re, im) = parts(z);
    Complex64::new(re, im)
}

#[track_caller]
fn assert_close(what: &str, got: C, want: Complex64, tol: f64) {
    let (re, im) = parts(got);

    let err = ((re - want.re).powi(2) + (im - want.im).powi(2)).sqrt();
    let scale = want.norm().max(1.0);

    assert!(
        err <= tol * scale,
        "{what}: got {re} + {im}i, want {} + {}i (abs err {err:e})",
        want.re,
        want.im
    );
}

/// A spread of inputs: all four quadrants, purely real, purely imaginary, zero,
/// and a couple of small/large magnitudes.
fn samples() -> Vec<C> {
    let vals = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7, 1.3),
        (-0.7, 1.3),
        (0.7, -1.3),
        (-0.7, -1.3),
        (3.0, 4.0),
        (-2.5, 0.25),
        (1e-4, 3e-5),
        (12.0, -9.0),
    ];

    vals.iter().map(|&(re, im)| c(re, im)).collect()
}

const TOL: f64 = 1e-12;

macro_rules! diff_unary {
    ($($name:ident => |$z:ident| $ours:expr, |$o:ident| $theirs:expr, $tol:expr);* $(;)?) => {$(
        #[test]
        fn $name() {
            for z in samples() {
                let $z = z;
                let got = $ours;

                let $o = oracle(z);
                let want = $theirs;

                if !want.re.is_finite() || !want.im.is_finite() {
                    continue; // singular point (e.g. ln(0)); not what these compare
                }

                assert_close(stringify!($name), got, want, $tol);
            }
        }
    )*};
}

diff_unary! {
    diff_exp      => |z| z.exp(),   |o| o.exp(),   TOL;
    diff_ln       => |z| z.ln(),    |o| o.ln(),    TOL;
    diff_sqrt     => |z| z.sqrt(),  |o| o.sqrt(),  TOL;
    diff_cbrt     => |z| z.cbrt(),  |o| o.cbrt(),  TOL;
    diff_sin      => |z| z.sin(),   |o| o.sin(),   TOL;
    diff_cos      => |z| z.cos(),   |o| o.cos(),   TOL;
    diff_tan      => |z| z.tan(),   |o| o.tan(),   TOL;
    diff_sinh     => |z| z.sinh(),  |o| o.sinh(),  TOL;
    diff_cosh     => |z| z.cosh(),  |o| o.cosh(),  TOL;
    diff_tanh     => |z| z.tanh(),  |o| o.tanh(),  TOL;
    diff_asin     => |z| z.asin(),  |o| o.asin(),  TOL;
    diff_acos     => |z| z.acos(),  |o| o.acos(),  TOL;
    diff_atan     => |z| z.atan(),  |o| o.atan(),  TOL;
    diff_asinh    => |z| z.asinh(), |o| o.asinh(), TOL;
    diff_acosh    => |z| z.acosh(), |o| o.acosh(), TOL;
    diff_atanh    => |z| z.atanh(), |o| o.atanh(), TOL;
    diff_exp2     => |z| z.exp2(),  |o| o.exp2(),  TOL;
    diff_log2     => |z| z.log2(),  |o| o.ln() * std::f64::consts::LOG2_E, TOL;
    diff_log10    => |z| z.log10(), |o| o.ln() * std::f64::consts::LOG10_E, TOL;
    diff_exp10    => |z| z.exp10(), |o| (o * std::f64::consts::LN_10).exp(), TOL;
    diff_expm1    => |z| z.exp_m1(), |o| o.exp() - Complex64::new(1.0, 0.0), TOL;
    diff_ln_1p    => |z| z.ln_1p(), |o| (o + Complex64::new(1.0, 0.0)).ln(), TOL;
    diff_inv_sqrt => |z| z.inverse_sqrt(), |o| o.sqrt().inv(), TOL;
    diff_rcp_full => |z| z.approx_reciprocal(), |o| o.inv(), TOL;
}

#[test]
fn diff_powf() {
    let exps = [c(2.0, 0.0), c(0.5, 0.0), c(-1.5, 0.0), c(1.0, 1.0), c(-0.25, 0.75)];

    for z in samples() {
        for &e in &exps {
            let got = z.powf(e);
            let want = oracle(z).powc(oracle(e));

            if !want.re.is_finite() || !want.im.is_finite() {
                continue;
            }

            assert_close(&format!("powf({:?} ^ {:?})", parts(z), parts(e)), got, want, 1e-11);
        }
    }
}

/// The degenerate points of the polar form, where `ln|z|` is `+-inf`.
///
/// Two distinct failures live here. A REAL exponent multiplies that infinity by a
/// zero imaginary part, so the angle is NaN where its limit is plainly 0. A COMPLEX
/// exponent gives a genuine `+-inf` angle (the spiral never settles), but its modulus
/// has already collapsed, and C99 takes `e^(-inf + iy)` to `+-0` for every non-finite
/// `y`. Both come back NaN without a guard.
#[test]
fn degenerate_polar_points() {
    // z^w at z = 0, against the same rules `num_complex` follows.
    for &(re, im) in &[(2.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.25, 3.0)] {
        let got = c(0.0, 0.0).powf(c(re, im));
        let want = Complex64::new(0.0, 0.0).powc(Complex64::new(re, im));

        assert_close(&format!("0^({re} + {im}i)"), got, want, 1e-14);
    }

    // 0^0 is 1, not NaN: the modulus never collapses, so only the angle needs saving.
    assert_close("0^0", c(0.0, 0.0).powf(c(0.0, 0.0)), Complex64::new(1.0, 0.0), 1e-14);

    // b^z at b = 0 has the identical shape through `expf`, for both exponent kinds.
    for &(re, im) in &[(2.0, 0.0), (1.0, 1.0)] {
        let got = c(re, im).expf(V::ZERO);
        let want = Complex64::new(0.0, 0.0).powc(Complex64::new(re, im));

        assert_close(&format!("0^({re} + {im}i) via expf"), got, want, 1e-14);
    }

    // `exp` inherits the C99 rule from `from_polar`. At the default policy the trig
    // clamp hides this, so pin it at the tier that actually propagates non-finite
    // angles: a modulus of zero must win over an angle that never resolved.
    for &im in &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let got = c(f64::NEG_INFINITY, im).exp_p::<Precision>();
        let (gr, gi) = parts(got);

        assert!(
            gr == 0.0 && gi == 0.0,
            "exp(-inf + {im}i) should be 0 + 0i, got {gr} + {gi}i"
        );
    }

    // A finite angle is still honoured, signed zeros included: `from_polar` must not
    // flatten a legitimately underflowed modulus.
    let (re, im) = parts(C::from_polar_p::<Precision>(V::ZERO, V::splat(std::f64::consts::PI)));
    assert!(
        re.is_sign_negative() && !im.is_sign_negative(),
        "from_polar(0, pi) should be (-0, +0), got ({re}, {im})"
    );
}

#[test]
fn diff_sinc() {
    for z in samples() {
        let got = z.sinc();

        let o = oracle(z);
        let want = if o.norm() == 0.0 {
            Complex64::new(1.0, 0.0) // the removable singularity
        } else {
            o.sin() / o
        };

        assert_close("sinc", got, want, TOL);
    }
}

/// `exp_m1`/`ln_1p` exist for the arguments where the naive form cancels, so a tiny
/// z is the case to pin.
#[test]
fn small_argument_accuracy() {
    let z = c(1e-9, 2e-9);

    // e^z - 1 ~= z + z^2/2 (exact to well past f64 precision at this magnitude)
    let zo = oracle(z);
    let want = zo + zo * zo * 0.5;

    assert_close("exp_m1 near 0", z.exp_m1(), want, 1e-15);

    // ln(1 + z) ~= z - z^2/2
    let want = zo - zo * zo * 0.5;
    assert_close("ln_1p near 0", z.ln_1p(), want, 1e-15);
}

// --- Vector-trait surface ---

/// The modulus-based `abs`/`signum` must satisfy `abs(z) * signum(z) == z`.
#[test]
fn abs_signum_identity() {
    for z in samples() {
        let rebuilt = z.abs() * z.signum();
        assert_close("abs * signum", rebuilt, oracle(z), 1e-15);

        let (re, im) = parts(z.abs());
        assert!((re - oracle(z).norm()).abs() <= 1e-15 * re.max(1.0));
        assert_eq!(im, 0.0, "abs() must be real");
    }
}

#[test]
fn spatial_norms() {
    let z = c(3.0, -4.0);

    assert_eq!(parts(z.l2_norm()), (5.0, 0.0));
    assert_eq!(parts(z.l2_norm_squared()), (25.0, 0.0));
    assert_eq!(parts(SpatialMath::l1_norm(z)), (7.0, 0.0));
    assert_eq!(z.norm_l1().extract::<0>(), 7.0); // the real-valued inherent form
    assert_eq!(parts(z.hypot(c(0.0, 12.0))), (13.0, 0.0)); // sqrt(|z|^2 + 12^2)
}

/// Ordering is lexicographic by `(re, im)`, and `cmp_eq` is exact equality of
/// both components.
#[test]
fn lexicographic_ordering() {
    let a = c(1.0, 5.0);
    let b = c(1.0, 6.0);
    let d = c(2.0, 0.0);

    assert!(a.cmp_lt(b).all(), "same re, smaller im");
    assert!(a.cmp_lt(d).all(), "smaller re wins regardless of im");
    assert!(a.cmp_eq(a).all());
    assert!(a.cmp_ne(b).all());

    assert_eq!(parts(a.min(b)), (1.0, 5.0));
    assert_eq!(parts(a.max(b)), (1.0, 6.0));
    assert_eq!(parts(b.clamp(a, a)), (1.0, 5.0));
}

/// The interleaved (re, im, re, im, ...) layout must round-trip through the
/// load/store engine, and the lane ops must hit the right lanes.
///
/// The rest of this file runs on the 1-lane scalar backend, but this one needs a
/// genuinely multi-lane register, so it has to name a backend. Which one does not
/// matter, only that `LANES > 1`, hence one per arch rather than x86 only.
#[test]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
fn lanes_and_memory() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    type CW = Complex<thermite::backend::x86_v2::f64x2>;
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    type CW = Complex<thermite::backend::wasm::f64x2>;
    #[cfg(target_arch = "aarch64")]
    type CW = Complex<thermite::backend::neon::f64x2>;

    let elems: Vec<Complex<f64>> = (0..CW::LANES)
        .map(|i| Complex::new(i as f64 + 1.0, -(i as f64) - 1.0))
        .collect();

    let z = unsafe { CW::load_unaligned(elems.as_ptr()) };

    for i in 0..CW::LANES {
        assert_eq!(z.extractv(i), elems[i], "lane {i}");
    }

    let mut out = vec![Complex::new(0.0f64, 0.0); CW::LANES];
    unsafe { z.store_unaligned(out.as_mut_ptr()) };
    assert_eq!(out, elems, "store must invert load");

    // sum_elements is componentwise, while prod_elements needs real complex multiplies.
    let sum = z.sum_elements();
    let want: Complex<f64> = elems.iter().copied().fold(Complex::new(0.0, 0.0), |a, b| a + b);
    assert_eq!(sum, want);

    let prod = z.prod_elements();
    let want = elems
        .iter()
        .map(|e| Complex64::new(e.re, e.im))
        .fold(Complex64::new(1.0, 0.0), |a, b| a * b);
    assert!((prod.re - want.re).abs() < 1e-9 && (prod.im - want.im).abs() < 1e-9);
}

/// A function written once over the vector traits, evaluated at a complex argument
/// with no changes.
#[test]
fn generic_function_over_complex() {
    fn gaussian<T: FloatVector + TranscendentalMath>(x: T) -> T {
        (-(x * x)).exp()
    }

    // e^(-i^2) = e^1 = e
    assert_close(
        "gaussian(i)",
        gaussian(C::I),
        Complex64::new(std::f64::consts::E, 0.0),
        1e-14,
    );

    // and it still agrees with the real function on the real axis
    let (re, im) = parts(gaussian(c(0.5, 0.0)));
    assert!((re - (-0.25f64).exp()).abs() < 1e-15);
    assert_eq!(im, 0.0);
}

/// `powi`/`square`/`Mul` must agree, and `nth_root` must invert the power.
#[test]
fn powers_compose() {
    let z = c(0.7, 1.3);

    assert_close("z^2 == z*z", z.powi(2), oracle(z) * oracle(z), 1e-15);
    assert_close("z^-3", z.powi(-3), oracle(z).powi(-3), 1e-13);

    let r = z.nth_root_n::<5>();
    assert_close("nth_root(5)^5 == z", r.powi(5), oracle(z), 1e-10);
}

// --- The ComplexMath family: the real-valued / real-argument operations ---

/// `norm`/`arg`/`to_polar` return the *real* vector, and `from_polar` inverts them.
#[test]
fn complex_math_polar() {
    for z in samples() {
        let o = oracle(z);

        let (r, theta) = z.to_polar();
        let (wr, wtheta) = o.to_polar();

        assert!((r.extract::<0>() - wr).abs() <= 1e-15 * wr.max(1.0), "norm");
        assert!((theta.extract::<0>() - wtheta).abs() <= 1e-14, "arg");

        // norm/arg agree with to_polar, and the policy form agrees with the default
        assert_eq!(z.norm().extract::<0>(), r.extract::<0>());
        assert_eq!(z.arg().extract::<0>(), theta.extract::<0>());
        assert_eq!(z.norm_p::<Precision>().extract::<0>(), r.extract::<0>());

        assert_close("from_polar(to_polar(z)) == z", C::from_polar(r, theta), o, 1e-14);
    }
}

/// `from_angle(theta)` is the `r = 1` case of `from_polar`, exactly, and lands on
/// the unit circle for every angle including the ones past a range reduction.
#[test]
fn complex_math_from_angle() {
    let angles = [
        0.0,
        0.25,
        core::f64::consts::FRAC_PI_2,
        core::f64::consts::PI,
        -core::f64::consts::PI,
        -2.75,
        7.0,
        -1e6,
        1e8,
    ];

    for theta in angles {
        let t = V::splat(theta);

        // The defining identity, at both policies: exactly `from_polar` with r = 1,
        // down to the bit, since the only difference is a multiply by one and a
        // zero-modulus guard that a unit modulus can never trip.
        assert_eq!(
            parts(C::from_angle(t)),
            parts(C::from_polar(V::ONE, t)),
            "from_angle({theta}) != from_polar(1, {theta})"
        );
        assert_eq!(
            parts(C::from_angle_p::<Precision>(t)),
            parts(C::from_polar_p::<Precision>(V::ONE, t)),
            "from_angle_p({theta}) != from_polar_p(1, {theta})"
        );

        // On the unit circle whatever the range reduction did with the angle.
        assert!(
            (C::from_angle(t).norm().extract::<0>() - 1.0).abs() <= 1e-12,
            "from_angle({theta}) off the unit circle"
        );

        // Against `e^(i theta)`. Only at `Precision`: the default policy's range
        // reduction is not expected to hold an absolute angle for `theta = 1e8`.
        assert_close(
            "from_angle == exp(i*theta)",
            C::from_angle_p::<Precision>(t),
            (num_complex::Complex64::i() * theta).exp(),
            1e-13,
        );
    }
}

/// The real-argument powers/logs, and the overflow-safe inverse/division.
#[test]
fn complex_math_real_arguments() {
    for z in samples() {
        let o = oracle(z);

        // z^y for a real y
        let got = z.powfr(V::splat(2.5));
        let want = o.powf(2.5);
        if want.re.is_finite() && want.im.is_finite() {
            assert_close("powfr", got, want, 1e-12);
        }

        // b^z for a real b
        let got = z.expf(V::splat(3.0));
        let want = Complex64::new(3.0, 0.0).powc(o);
        if want.re.is_finite() && want.im.is_finite() {
            assert_close("expf", got, want, 1e-12);
        }

        // log_b(z) for a real b
        let got = z.logr(V::splat(2.0));
        let want = o.ln() / std::f64::consts::LN_2;
        if want.re.is_finite() && want.im.is_finite() {
            assert_close("logr", got, want, 1e-12);
        }

        if o.norm() != 0.0 {
            assert_close("finv", z.finv(), o.inv(), 1e-14);
            assert_close("fdiv", c(1.0, 2.0).fdiv(z), Complex64::new(1.0, 2.0) / o, 1e-13);
        }
    }
}

/// `finv`/`fdiv` survive the magnitudes where `norm_sqr()` overflows and the plain
/// `inv`/`/` collapse.
#[test]
fn finv_survives_where_inv_overflows() {
    let z = c(1e200, 1e200); // |z|^2 = 2e400 -> +inf in f64

    assert!(z.norm_sqr().extract::<0>().is_infinite(), "premise: norm_sqr overflows");

    // The naive inverse divides by that infinity and collapses to zero. So does
    // num_complex's `inv()`, which is the same conj/norm_sqr formula, hence the
    // expected value here is the analytic one:
    //   1/z = conj(z)/|z|^2 = (1e200 - 1e200 i) / 2e400 = 5e-201 - 5e-201 i
    assert_eq!(parts(z.inv()), (0.0, -0.0));

    let (re, im) = parts(z.finv());
    assert!(re != 0.0 && im != 0.0, "finv must not collapse: got {re} + {im}i");
    assert!((re - 5e-201).abs() <= 1e-14 * 5e-201, "got re = {re}");
    assert!((im + 5e-201).abs() <= 1e-14 * 5e-201, "got im = {im}");
}

/// Bounding on `ComplexMath` works for any backend, lane count and inner float.
#[test]
fn generic_over_complex_math() {
    fn unit<T: ComplexMath>(z: T) -> T {
        T::from_angle(z.arg())
    }

    let z = unit(c(3.0, 4.0));
    assert_close("unit(3+4i)", z, oracle(c(0.6, 0.8)), 1e-15);
}

// --- Mixed complex/real arithmetic ---

/// The operators against `Self::Real` must be usable from generic code, not only on
/// the concrete `Complex<V>`.
#[test]
fn generic_mixed_real_arithmetic() {
    use thermite::vector::ops::MulAddExt;

    fn affine<T: ComplexVector + MulAddExt<T::Real, T, Output = T>>(z: T, scale: T::Real, offset: T::Real) -> T {
        // z * scale + offset, where both parameters are *real*
        z.mul_adde(scale, T::real(offset))
    }

    let z = c(3.0, -4.0);
    let got = affine(z, V::splat(2.0), V::splat(1.0));

    assert_eq!(parts(got), (7.0, -8.0)); // (3-4i)*2 + 1

    // Each operator, against a real RHS.
    let r = V::splat(2.0);

    assert_eq!(parts(z + r), (5.0, -4.0)); // offsets the real part only
    assert_eq!(parts(z - r), (1.0, -4.0));
    assert_eq!(parts(z * r), (6.0, -8.0)); // scales both components
    assert_eq!(parts(z / r), (1.5, -2.0));

    // ... and the assigning forms.
    let mut w = z;
    w *= r;
    w += r;
    assert_eq!(parts(w), (8.0, -8.0));
}

/// Scaling by a real must agree with widening it to a complex first, and must be a
/// true single-rounding FMA where the complex-by-complex form cannot be.
#[test]
fn real_scaling_agrees_with_the_widened_form() {
    use thermite::vector::ops::MulAddExt;

    let z = c(0.7, 1.3);
    let r = V::splat(3.0);

    // Multiplication is bit-identical: the complex product against a zero imaginary
    // part reduces to the two real products.
    assert_eq!(parts(z * r), parts(z * C::real(r)));

    // Division is not, and should not be: Div<Real> takes one reciprocal and
    // multiplies through, where the complex divide runs the conj/|w|^2 formula. Same
    // value, different rounding.
    let (a, b) = (parts(z / r), parts(z / C::real(r)));
    assert!(
        (a.0 - b.0).abs() <= 1e-16 && (a.1 - b.1).abs() <= 1e-16,
        "{a:?} vs {b:?}"
    );

    // Both FMA impls inherit the inner vector's answer. The flag reports whether the
    // FMA is FAST (whether it dodges the slow software emulation), not whether
    // it is a single rounding: the real multiplier is one inner FMA per component and
    // the complex multiplier is two, so both are cheap exactly when the inner one is.
    assert_eq!(
        <C as MulAddExt<V, C>>::HAS_NATIVE_FMA,
        <V as MulAddExt<V, V>>::HAS_NATIVE_FMA,
        "complex-by-real FMA is exactly the inner FMA, per component"
    );
    assert_eq!(
        <C as MulAddExt<C, C>>::HAS_NATIVE_FMA,
        <V as MulAddExt<V, V>>::HAS_NATIVE_FMA,
        "complex-by-complex FMA is two inner FMAs per component"
    );
}

/// The masked `_c`/`_m`/`_z` forms are promised against a real RHS too.
#[test]
fn masked_mixed_real_arithmetic() {
    use thermite::vector::ops::MulMasked;

    let z = c(2.0, 3.0);
    let r = V::splat(10.0);

    let on = z.re.cmp_gt(V::ZERO); // true in every lane here
    let off = z.re.cmp_lt(V::ZERO);

    assert_eq!(parts(z.mul_c(on, r)), (20.0, 30.0));
    assert_eq!(parts(z.mul_c(off, r)), (2.0, 3.0)); // unmasked lanes keep self
    assert_eq!(parts(z.mul_z(off, r)), (0.0, 0.0));
}

// --- overridden trait defaults ----------------------------------------------
//
// Both of these are inherited defaults that are wrong or inconsistent over C. The
// tests are written so they fail against the defaults, not just so they pass against
// the overrides.

/// `sinc_pi` must be *exactly* zero at every non-zero integer, the property that
/// makes it an interpolating kernel. The default (`sinc(z * pi)`) is accurate to about
/// an ulp there but not exact, returning ~1e-16, so this asserts equality with zero
/// rather than a tolerance.
#[test]
fn sinc_pi_is_exact_at_the_integers() {
    for k in [1.0, 2.0, 5.0, 17.0, 64.0, -33.0, 1024.0] {
        let (re, im) = parts(c(k, 0.0).sinc_pi());

        assert!(re == 0.0 && im == 0.0, "sinc_pi({k}) = ({re}, {im}), want exactly 0");
    }
}

#[test]
fn sinc_pi_matches_its_definition_off_axis() {
    for &(x, y) in &[(0.25, 0.5), (2.5, -1.0), (-3.75, 0.25)] {
        let z = Complex64::new(x, y);
        let pz = z * std::f64::consts::PI;
        let want = pz.sin() / pz;

        assert_close("sinc_pi", c(x, y).sinc_pi(), want, 1e-12);
    }

    // Removable singularity.
    assert_close("sinc_pi(0)", c(0.0, 0.0).sinc_pi(), Complex64::new(1.0, 0.0), 0.0);
}

/// `hypot` over C must mean `sqrt(sum |z_i|^2)` at *every* policy.
///
/// The generic default did not: its high-precision path opens with `abs()` (the
/// modulus here) and yields the norm, while the `Worst` path squares directly and
/// yields the analytic continuation `sqrt(z^2 + w^2)`. For these inputs the two
/// disagree in the first digit, so this pins the meaning rather than the accuracy.
#[test]
fn hypot_is_the_norm_at_every_policy() {
    use thermite::math::SpatialMathWithPolicy;
    use thermite::math::policy::policies::{Performance, UltraPerformance};

    for &(a, b, cc, d) in &[(3.0, 4.0, 5.0, 12.0), (1.0, -1.0, 0.5, 2.0), (-2.5, 0.25, 1.5, -3.0)] {
        let z = Complex64::new(a, b);
        let w = Complex64::new(cc, d);
        let want = (z.norm().powi(2) + w.norm().powi(2)).sqrt();

        let ultra = c(a, b).hypot_p::<UltraPerformance>(c(cc, d));
        let perf = c(a, b).hypot_p::<Performance>(c(cc, d));
        let prec = c(a, b).hypot_p::<Precision>(c(cc, d));

        for (name, got) in [("UltraPerformance", ultra), ("Performance", perf), ("Precision", prec)] {
            let (re, im) = parts(got);

            assert!(
                (re - want).abs() < 1e-6 * want.max(1.0),
                "hypot re @ {name} ({z}, {w}): got {re}, want {want}"
            );
            assert!(im.abs() < 1e-12, "hypot im @ {name}: got {im}, want 0 (a norm is real)");
        }
    }
}

#[test]
fn hypot_agrees_with_l2_norm() {
    // hypot of one argument is that argument's modulus.
    for &(x, y) in &[(3.0, 4.0), (-1.5, 0.5), (0.0, 0.0)] {
        let h = parts(c(x, y).hypot(c(0.0, 0.0)));
        let n = parts(c(x, y).l2_norm());

        assert!(
            (h.0 - n.0).abs() < 1e-12 && h.1.abs() < 1e-12 && n.1.abs() < 1e-12,
            "hypot vs l2_norm @ ({x}, {y}): {h:?} vs {n:?}"
        );
    }
}

#[test]
fn inv_hypot_is_the_reciprocal_norm() {
    use thermite::math::SpatialMathWithPolicy;

    let (a, b, cc, d) = (3.0, 4.0, 5.0, 12.0);
    let want = 1.0 / (25.0f64 + 169.0).sqrt();

    let got = parts(<C as SpatialMathWithPolicy>::inv_hypot_n_p::<Precision, 2>([
        c(a, b),
        c(cc, d),
    ]));

    assert!(
        (got.0 - want).abs() < 1e-12 && got.1.abs() < 1e-12,
        "inv_hypot_n: got {got:?}, want {want}"
    );
}

/// `poly_rational_n` must pick its evaluation form on `|z|`, not on the lexicographic
/// order. At `z = 10^150 i` the real part is 0, so the generic default judges `z` "not
/// greater than one" and evaluates the direct form, where `z^3` overflows to infinity
/// and the ratio comes back NaN. Through `1/z` it is the ratio of leading coefficients.
#[test]
fn poly_rational_n_inverts_on_modulus_not_lexicographic_order() {
    use thermite::math::CoreMathWithPolicy;

    // constant-term-first, equal degree: the limit as |z| -> inf is 4/8.
    let num = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    let den = [c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)];

    let numc = num.map(|z| Complex::new(z.re.extract::<0>(), z.im.extract::<0>()));
    let denc = den.map(|z| Complex::new(z.re.extract::<0>(), z.im.extract::<0>()));

    // Purely imaginary, so Re z = 0 < 1 while |z| is enormous.
    let got = c(0.0, 1.0e150).poly_rational_n_p::<Precision, 4, 4>(&numc, &denc);
    let (re, im) = parts(got);

    assert!(
        re.is_finite() && im.is_finite(),
        "poly_rational_n @ 1e150i: got ({re}, {im}), want finite"
    );
    assert!(
        (re - 0.5).abs() < 1e-12 && im.abs() < 1e-12,
        "poly_rational_n @ 1e150i: got ({re}, {im}), want (0.5, 0)"
    );

    // And it still agrees with the direct form well inside the unit disc.
    let small = c(0.25, -0.125).poly_rational_n_p::<Precision, 4, 4>(&numc, &denc);
    let z = Complex64::new(0.25, -0.125);
    let want = (Complex64::new(1.0, 0.0) + 2.0 * z + 3.0 * z * z + 4.0 * z * z * z)
        / (Complex64::new(5.0, 0.0) + 6.0 * z + 7.0 * z * z + 8.0 * z * z * z);

    assert_close("poly_rational_n small", small, want, 1e-13);
}

/// `harmonic_mean` / `inv_sum_inv` on `Complex`, which take the DIRECT reciprocal-sum form
/// rather than the min-scaled one real vectors get.
///
/// `Complex::min` is lexicographic by `(re, im)`, so it can return an element of large
/// magnitude, and scaling by it would protect nothing. These tests exist to pin that the
/// composite really is on the unscaled path and that the path is correct.
#[test]
fn harmonic_mean_and_inv_sum_inv() {
    // 2 / (1/z0 + 1/z1), against num_complex directly.
    let pairs = [
        ((1.0, 2.0), (3.0, -1.0)),
        ((0.5, 0.0), (0.25, 0.0)),
        ((-2.0, 3.0), (1.0, 1.0)),
        ((1e-8, 1e-8), (2.0, -3.0)),
    ];

    for &((a, b), (p, q)) in pairs.iter() {
        let (z0, z1) = (Complex64::new(a, b), Complex64::new(p, q));
        let want_isi = 1.0 / (1.0 / z0 + 1.0 / z1);
        let want_hm = want_isi * 2.0;

        let got_hm = C::harmonic_mean_n([c(a, b), c(p, q)]);
        let got_isi = C::inv_sum_inv_n([c(a, b), c(p, q)]);

        let (hr, hi) = (got_hm.re.extract::<0>(), got_hm.im.extract::<0>());
        let (sr, si) = (got_isi.re.extract::<0>(), got_isi.im.extract::<0>());

        assert!(
            (hr - want_hm.re).abs() < 1e-12 && (hi - want_hm.im).abs() < 1e-12,
            "harmonic_mean({z0}, {z1}): got {hr}+{hi}i want {want_hm}"
        );
        assert!(
            (sr - want_isi.re).abs() < 1e-12 && (si - want_isi.im).abs() < 1e-12,
            "inv_sum_inv({z0}, {z1}): got {sr}+{si}i want {want_isi}"
        );
    }

    // The defining factor of N still holds on Complex: N copies give z and z/N.
    let z = c(2.0, -5.0);
    let hm = C::harmonic_mean_n([z, z, z]);
    let si = C::inv_sum_inv_n([z, z, z]);
    assert!((hm.re.extract::<0>() - 2.0).abs() < 1e-14 && (hm.im.extract::<0>() + 5.0).abs() < 1e-14);
    assert!((si.re.extract::<0>() - 2.0 / 3.0).abs() < 1e-14 && (si.im.extract::<0>() + 5.0 / 3.0).abs() < 1e-14);

    // A zero input does NOT take the real form's limit, and that is consistent rather than a
    // defect. On a real vector 1/0 is +inf, the sum saturates and the mean is 0. Complex
    // division computes 1/(c^2 + d^2) first, so at zero it forms 0 * inf and yields NaN.
    // There is no complex infinity in this representation to sum toward. The mean inherits
    // exactly what the crate's own division does, which the second assertion pins.
    let zero = c(0.0, 0.0);
    let hz = C::harmonic_mean_n([zero, c(1.0, 1.0)]);
    assert!(
        hz.re.extract::<0>().is_nan(),
        "a zero element gives NaN on Complex, not 0"
    );

    let recip = C::ONE / zero;
    assert!(recip.re.extract::<0>().is_nan(), "because 1/0 is itself NaN here");
}
