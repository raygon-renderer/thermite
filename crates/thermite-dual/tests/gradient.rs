//! End-to-end checks ported from the original `hyperdual` crate's doctests:
//! forward-mode gradients of a multivariate function, evaluated SIMD-parallel
//! through a Thermite vector.

use thermite::math::{CoreMath, RealMath, SpatialMath, TranscendentalMath};
use thermite::prelude::*;
use thermite::vector::ops::{MulAddExt, Square};
use thermite_dual::{AutoDiff, Dual};

/// Inner vector: the 1-lane scalar `f64` backend (no target features needed).
type V = Vector<f64>;
/// Dual carrying a value + 2 partials (df/dx, df/dy).
type D = Dual<V, 2>;

fn close(a: f64, b: f64, eps: f64) -> bool {
    let d = a - b;
    let d = if d < 0.0 { -d } else { d };
    d < eps
}

// A function generic over any float/math vector -- the whole point of Thermite.
fn gaussian<W: FloatVector + TranscendentalMath>(v: W) -> W {
    (-(v * v)).exp()
}

#[test]
fn autodiff_generic_math_function() {
    // `gaussian` is generic over any FloatVector + TranscendentalMath. AutoDiff infers and
    // instantiates it at W = Dual<Vector<f64>, 1> directly from the input array -- no
    // turbofish or closure wrapper needed.
    let r = gaussian.ad([V::splat(0.5)]);

    let x = 0.5_f64;
    let g = (-(x * x)).exp();
    assert!(close(r.re.extract::<0>(), g, 1e-12));
    // d/dx exp(-x^2) = -2x exp(-x^2)
    assert!(close(r.dual[0].extract::<0>(), -2.0 * x * g, 1e-12));

    // A multi-arg generic also works: gaussian-like product over two seeded variables.
    fn prod<W: FloatVector + TranscendentalMath>(a: W, b: W) -> W {
        (a * b).exp()
    }
    let r2 = prod.ad([V::splat(1.5), V::splat(2.0)]);
    let (a, b) = (1.5_f64, 2.0_f64);
    let p = (a * b).exp();
    assert!(close(r2.re.extract::<0>(), p, 1e-9));
    // d/da exp(ab) = b exp(ab); d/db = a exp(ab)
    assert!(close(r2.dual[0].extract::<0>(), b * p, 1e-9));
    assert!(close(r2.dual[1].extract::<0>(), a * p, 1e-9));
}

#[test]
fn autodiff_trait_matches_manual_seeding() {
    type D3 = Dual<V, 3>;

    // A closure of three variables over the SIMD-capable inner type (V = Vector<f64>).
    let f = |x: D3, y: D3, z: D3| x * x + y * z;

    let inputs = [V::splat(2.0), V::splat(3.0), V::splat(4.0)];

    // Trait entry point: seeds each input as an independent variable and evaluates.
    let auto = f.ad(inputs);

    // Manual equivalent.
    let manual = f(
        D3::variable(inputs[0], 0),
        D3::variable(inputs[1], 1),
        D3::variable(inputs[2], 2),
    );

    assert!(close(auto.re.extract::<0>(), manual.re.extract::<0>(), 1e-15));
    for i in 0..3 {
        assert!(close(auto.dual[i].extract::<0>(), manual.dual[i].extract::<0>(), 1e-15));
    }

    // f = x^2 + y*z at (2,3,4): value 16, gradient [2x, z, y] = [4, 4, 3].
    assert!(close(auto.re.extract::<0>(), 16.0, 1e-12));
    assert!(close(auto.dual[0].extract::<0>(), 4.0, 1e-12));
    assert!(close(auto.dual[1].extract::<0>(), 4.0, 1e-12));
    assert!(close(auto.dual[2].extract::<0>(), 3.0, 1e-12));
}

#[test]
fn multivariate_gradient() {
    // f(x, y) = x*x + sin(x*y) + y^3, evaluated at (4, 5).
    // Seed x as variable 0 and y as variable 1.
    let x = D::variable(V::splat(4.0), 0);
    let y = D::variable(V::splat(5.0), 1);

    let f = x * x + (x * y).sin() + y.powi(3);

    let value = f.re.extract::<0>();
    let dfdx = f.dual[0].extract::<0>();
    let dfdy = f.dual[1].extract::<0>();

    // Reference values from the original hyperdual crate.
    assert!(close(value, 141.91294525072763, 1e-10), "f(4,5) = {value}");
    assert!(close(dfdx, 10.04041030906696, 1e-10), "df/dx = {dfdx}");
    assert!(close(dfdy, 76.63232824725357, 1e-10), "df/dy = {dfdy}");
}

#[test]
fn univariate_sqrt_derivative() {
    // d/dx sqrt(x) at x = 4 is 1/(2*sqrt(4)) = 0.25.
    let x = D::variable(V::splat(4.0), 0);
    let r = x.sqrt() + D::constant(V::ONE);

    assert!(close(r.re.extract::<0>(), 3.0, 1e-12)); // sqrt(4) + 1
    assert!(close(r.dual[0].extract::<0>(), 0.25, 1e-12));
    assert!(close(r.dual[1].extract::<0>(), 0.0, 1e-12)); // no dependence on var 1
}

#[test]
fn exp_ln_roundtrip_and_chain() {
    // d/dx exp(x) = exp(x); d/dx ln(x) = 1/x.
    let x = D::variable(V::splat(2.0), 0);

    let e = x.exp();
    assert!(close(e.re.extract::<0>(), 2.0_f64.exp(), 1e-9));
    assert!(close(e.dual[0].extract::<0>(), 2.0_f64.exp(), 1e-9));

    let l = x.ln();
    assert!(close(l.re.extract::<0>(), 2.0_f64.ln(), 1e-12));
    assert!(close(l.dual[0].extract::<0>(), 0.5, 1e-12));
}

#[test]
fn product_and_quotient_rules() {
    // h(x) = x / (x + 1) at x = 3; h'(x) = 1/(x+1)^2 = 1/16.
    let x = D::variable(V::splat(3.0), 0);
    let h = x / (x + D::constant(V::ONE));

    assert!(close(h.re.extract::<0>(), 0.75, 1e-12));
    assert!(close(h.dual[0].extract::<0>(), 1.0 / 16.0, 1e-12));
}

#[test]
fn powi_and_chain() {
    // f(x) = sin(x^2); f'(x) = 2x cos(x^2), at x = 1.3 (exercises powi + chain).
    let xi = 1.3_f64;
    let x = D::variable(V::splat(xi), 0);
    let f = x.powi(2).sin();

    assert!(close(f.re.extract::<0>(), (xi * xi).sin(), 1e-9));
    assert!(close(f.dual[0].extract::<0>(), 2.0 * xi * (xi * xi).cos(), 1e-9));
}

#[test]
fn lowered_square_scale_clamp_fract() {
    // square: (x^2)' = 2x, at x = 3 -> 9, 6
    let sq = D::variable(V::splat(3.0), 0).square();
    assert!(close(sq.re.extract::<0>(), 9.0, 1e-12));
    assert!(close(sq.dual[0].extract::<0>(), 6.0, 1e-12));

    // scale by a constant element (3): 3x -> value 15, derivative 3
    let s = D::variable(V::splat(5.0), 0).scale(Dual::<f64, 2>::constant(3.0));
    assert!(close(s.re.extract::<0>(), 15.0, 1e-12));
    assert!(close(s.dual[0].extract::<0>(), 3.0, 1e-12));

    // scale by a *dual* element exercises the product rule:
    // x=5 (dx/dv0=1), factor=3 (df/dv0=2) -> (xf)' wrt v0 = x*f' + x'*f = 5*2 + 1*3 = 13
    let sd = D::variable(V::splat(5.0), 0).scale(Dual::<f64, 2>::new(3.0, [2.0, 0.0]));
    assert!(close(sd.re.extract::<0>(), 15.0, 1e-12));
    assert!(close(sd.dual[0].extract::<0>(), 13.0, 1e-12));

    // clamp: in-range keeps value + derivative; out-of-range takes the (constant) bound
    let lo = D::constant(V::splat(0.0));
    let hi = D::constant(V::splat(3.0));
    let inside = D::variable(V::splat(2.0), 0).clamp(lo, hi);
    assert!(close(inside.re.extract::<0>(), 2.0, 1e-12));
    assert!(close(inside.dual[0].extract::<0>(), 1.0, 1e-12));
    let above = D::variable(V::splat(5.0), 0).clamp(lo, hi);
    assert!(close(above.re.extract::<0>(), 3.0, 1e-12));
    assert!(close(above.dual[0].extract::<0>(), 0.0, 1e-12)); // pinned to constant hi

    // fract: fract(2.7) = 0.7, derivative 1
    let f = D::variable(V::splat(2.7), 0).fract();
    assert!(close(f.re.extract::<0>(), 0.7, 1e-9));
    assert!(close(f.dual[0].extract::<0>(), 1.0, 1e-12));
}

#[test]
fn tan_dedicated_override() {
    // d/dx tan(x) = 1 + tan(x)^2 = sec^2(x), at x = 0.7.
    let xi = 0.7_f64;
    let x = D::variable(V::splat(xi), 0);
    let t = x.tan();

    let tv = xi.tan();
    assert!(close(t.re.extract::<0>(), tv, 1e-12));
    assert!(close(t.dual[0].extract::<0>(), 1.0 + tv * tv, 1e-12));
}

#[test]
fn fma_variants_value_and_derivative() {
    // Validate each fused variant's primal AND derivative against the naive form,
    // with a = x (var), b = x, c = const, at x = 3.
    let x = D::variable(V::splat(3.0), 0);
    let c = D::constant(V::splat(2.0));
    let xv = 3.0_f64;

    // mul_add: x*x + 2  => 11, d = 2x = 6
    let r = x.mul_add(x, c);
    assert!(close(r.re.extract::<0>(), xv * xv + 2.0, 1e-12));
    assert!(close(r.dual[0].extract::<0>(), 2.0 * xv, 1e-12));

    // mul_sub: x*x - 2  => 7, d = 2x = 6
    let r = x.mul_sub(x, c);
    assert!(close(r.re.extract::<0>(), xv * xv - 2.0, 1e-12));
    assert!(close(r.dual[0].extract::<0>(), 2.0 * xv, 1e-12));

    // nmul_add: -(x*x) + 2  => -7, d = -2x = -6
    let r = x.nmul_add(x, c);
    assert!(close(r.re.extract::<0>(), -(xv * xv) + 2.0, 1e-12));
    assert!(close(r.dual[0].extract::<0>(), -2.0 * xv, 1e-12));

    // nmul_sub: -(x*x) - 2  => -11, d = -2x = -6
    let r = x.nmul_sub(x, c);
    assert!(close(r.re.extract::<0>(), -(xv * xv) - 2.0, 1e-12));
    assert!(close(r.dual[0].extract::<0>(), -2.0 * xv, 1e-12));

    // estimating variants must agree with the exact ones here
    assert!(close(x.mul_adde(x, c).dual[0].extract::<0>(), 2.0 * xv, 1e-12));
    assert!(close(x.nmul_sube(x, c).dual[0].extract::<0>(), -2.0 * xv, 1e-12));
}

#[test]
fn hypot_gradient() {
    // h = hypot(x, y) at (3, 4) = 5; dh/dx = x/h = 0.6, dh/dy = y/h = 0.8.
    let x = D::variable(V::splat(3.0), 0);
    let y = D::variable(V::splat(4.0), 1);
    let h = x.hypot(y);

    assert!(close(h.re.extract::<0>(), 5.0, 1e-9));
    assert!(close(h.dual[0].extract::<0>(), 0.6, 1e-9));
    assert!(close(h.dual[1].extract::<0>(), 0.8, 1e-9));
}

#[test]
fn inverse_smoothstep_implicit_derivative() {
    // N=3 inverse_smoothstep runs a Newton loop internally; the Dual override must
    // round-trip (inverse(smoothstep(x)) == x) and give the implicit-function-theorem
    // derivative d/dy inverse(y) = 1 / smoothstep'(x), NOT a value from differentiating
    // through Newton.
    let x0 = 0.3_f64;
    let y0 = V::splat(x0).smoothstep::<3>(None).extract::<0>();

    let y = D::variable(V::splat(y0), 0);
    let t = y.inverse_smoothstep::<3>(None);

    assert!(close(t.re.extract::<0>(), x0, 1e-6)); // round-trip

    let sd = V::splat(x0).smoothstep_derivative::<3>(None).extract::<0>();
    assert!(close(t.dual[0].extract::<0>(), 1.0 / sd, 1e-6));
}

#[test]
fn nth_root_override() {
    // d/dx x^(1/3) = 1/(3 x^(2/3)), at x = 8: value 2, deriv 1/(3*4) = 1/12.
    let x = D::variable(V::splat(8.0), 0);
    let r = x.nth_root::<3>();

    assert!(close(r.re.extract::<0>(), 2.0, 1e-9));
    assert!(close(r.dual[0].extract::<0>(), 1.0 / 12.0, 1e-9));
}

#[test]
fn abs_derivative_sign() {
    // d/dx |x| = sign(x): +1 for x>0, -1 for x<0.
    let xp = D::variable(V::splat(2.5), 0).abs();
    let xn = D::variable(V::splat(-2.5), 0).abs();

    assert!(close(xp.re.extract::<0>(), 2.5, 1e-12));
    assert!(close(xp.dual[0].extract::<0>(), 1.0, 1e-12));
    assert!(close(xn.re.extract::<0>(), 2.5, 1e-12));
    assert!(close(xn.dual[0].extract::<0>(), -1.0, 1e-12));
}

#[test]
fn mix_and_lerp() {
    // mix(t) = a*(1-t) + b*t. Differentiate wrt t at t=0.25 with a=2, b=10:
    // value = 2*0.75 + 10*0.25 = 4; d/dt = b - a = 8.
    let t = D::variable(V::splat(0.25), 0);
    let a = D::constant(V::splat(2.0));
    let b = D::constant(V::splat(10.0));

    let m = t.mix(a, b);
    assert!(close(m.re.extract::<0>(), 4.0, 1e-12));
    assert!(close(m.dual[0].extract::<0>(), 8.0, 1e-12));

    // `lerp` is a trait default that lowers to `mix` -- it used to panic on Dual.
    let l = t.lerp(a, b);
    assert!(close(l.re.extract::<0>(), 4.0, 1e-12));
    assert!(close(l.dual[0].extract::<0>(), 8.0, 1e-12));
}

#[test]
fn powf_constant_exponent_negative_base() {
    // f(x) = x^2 written via powf with a *constant* exponent, at x = -2.
    // Value (-2)^2 = 4, derivative 2x = -4. ln(x) is NaN for x<0, but a constant
    // exponent must not let that NaN poison d/dx.
    let x = D::variable(V::splat(-2.0), 0);
    let r = x.powf(D::constant(V::splat(2.0)));

    assert!(close(r.re.extract::<0>(), 4.0, 1e-9));
    let d = r.dual[0].extract::<0>();
    assert!(!d.is_nan(), "powf d/dx was NaN-poisoned");
    assert!(close(d, -4.0, 1e-9), "d/dx = {d}");
}

#[test]
fn log2_log10_derivatives() {
    // d/dx log2(x) = 1/(x ln2); d/dx log10(x) = 1/(x ln10), at x = 8.
    let x = D::variable(V::splat(8.0), 0);

    let l2 = x.log2();
    assert!(close(l2.re.extract::<0>(), 3.0, 1e-12)); // log2(8)
    assert!(close(l2.dual[0].extract::<0>(), 1.0 / (8.0 * 2.0_f64.ln()), 1e-12));

    let l10 = x.log10();
    assert!(close(l10.re.extract::<0>(), 8.0_f64.log10(), 1e-12));
    assert!(close(l10.dual[0].extract::<0>(), 1.0 / (8.0 * 10.0_f64.ln()), 1e-12));
}

#[test]
fn masked_variants_select_and_propagate() {
    // The `_c`/`_m`/`_z` masked ops used to be `todo!()`; verify they blend value AND
    // derivative per the mask. Use a 1-lane mask that is all-true / all-false.
    let x = D::variable(V::splat(4.0), 0); // sqrt -> 2, d = 0.25
    let all = x.re.cmp_gt(V::splat(0.0)); // true
    let none = x.re.cmp_lt(V::splat(0.0)); // false

    // sqrt_c: where true -> sqrt(x); where false -> self.
    let on = x.sqrt_c(all);
    assert!(close(on.re.extract::<0>(), 2.0, 1e-12));
    assert!(close(on.dual[0].extract::<0>(), 0.25, 1e-12));
    let off = x.sqrt_c(none);
    assert!(close(off.re.extract::<0>(), 4.0, 1e-12)); // unchanged self
    assert!(close(off.dual[0].extract::<0>(), 1.0, 1e-12)); // self derivative

    // sqrt_z: where false -> zero (value and derivative).
    let z = x.sqrt_z(none);
    assert!(close(z.re.extract::<0>(), 0.0, 1e-12));
    assert!(close(z.dual[0].extract::<0>(), 0.0, 1e-12));

    // binary masked: min_c with a smaller other.
    let other = D::constant(V::splat(1.0));
    let mn = x.min_c(all, other); // 1 < 4 -> takes other (constant, deriv 0)
    assert!(close(mn.re.extract::<0>(), 1.0, 1e-12));
    assert!(close(mn.dual[0].extract::<0>(), 0.0, 1e-12));
}
