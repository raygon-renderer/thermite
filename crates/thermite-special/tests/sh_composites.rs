//! Spherical harmonics on the composite float types.
//!
//! These matter because `spherical_harmonics` reaches `Dual` and `Compensated` through
//! a _provided default_ (the generic runtime-coefficient path) rather than any
//! per-type code. That is a compile-time argument until something actually instantiates
//! it, which is what this file is for.
//!
//! The gradient form is deliberately absent here. It lives on `RealPrimalMath`, which
//! `Dual` does not implement: an autodiff type is supposed to get derivatives from the
//! value form, and for spherical harmonics specifically that is also the faster route
//! (one recurrence produces the whole basis, so recomputing under dual arithmetic beats
//! contracting three gradient arrays per harmonic).

use thermite::math::policy::DefaultPolicy;
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_dual::Dual;
use thermite_special::specialized::sh_impl;
use thermite_special::{NO_PHASE, RealSpecialMath};

type V64 = Vector<f64>;

const L: usize = 4;
const N: usize = (L + 1) * (L + 1);

/// Plain `f64` reference, straight through the unrolled kernel.
fn reference(x: f64, y: f64, z: f64) -> [f64; N] {
    let mut out = [V64::splat(0.0); N];
    sh_impl::<DefaultPolicy, f64, V64, L, N, NO_PHASE>(V64::splat(x), V64::splat(y), V64::splat(z), &mut out);

    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = out[i].extract::<0>();
    }
    r
}

const DIRS: &[(f64, f64, f64)] = &[
    (0.267261241912424, 0.534522483824849, 0.801783725737273),
    (-0.6, 0.0, 0.8),
    (0.0, 0.0, 1.0),
    (0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    (0.0, 1.0, 0.0),
];

/// `Compensated<V>` runs the generic path in double-double arithmetic. Its `value()`
/// must agree with plain `f64` to within plain-`f64` accuracy. It is strictly more
/// precise, not different.
#[test]
fn sh_compensated_matches_f64() {
    type C = Compensated<V64>;

    for &(x, y, z) in DIRS {
        let want = reference(x, y, z);

        let mut out = [C::new(V64::splat(0.0)); N];
        C::spherical_harmonics::<L, N, NO_PHASE>(
            C::new(V64::splat(x)),
            C::new(V64::splat(y)),
            C::new(V64::splat(z)),
            &mut out,
        );

        for i in 0..N {
            let got = out[i].value().extract::<0>();
            assert!(
                (got - want[i]).abs() < 1e-13 * (1.0 + want[i].abs()),
                "Compensated sh[{i}] at ({x},{y},{z}): got {got}, f64 {}",
                want[i]
            );
        }
    }
}

/// `Dual<V, 3>` seeded with an identity Jacobian differentiates with respect to `x`,
/// `y` and `z` themselves, so its gradient must reproduce `spherical_harmonics_d` -
/// two entirely different mechanisms (dual arithmetic through the recurrence vs. the
/// hand-derived norm-ratio identities) landing on the same numbers.
///
/// This is also the comparison that justifies keeping `_d` off `Dual`: it is the case
/// where a caller might reach for `Dual<V, 3>`, and `_d` computes it about twice as
/// cheaply.
#[test]
fn sh_dual_identity_jacobian_matches_analytic_gradients() {
    use thermite_special::RealPrimalMath;

    type D = Dual<V64, 3>;

    for &(x, y, z) in DIRS {
        // Analytic gradients from the primal form.
        let mut a_val = [V64::splat(0.0); N];
        let (mut a_dx, mut a_dy, mut a_dz) = ([V64::splat(0.0); N], [V64::splat(0.0); N], [V64::splat(0.0); N]);
        V64::spherical_harmonics_d::<L, N, NO_PHASE>(
            V64::splat(x),
            V64::splat(y),
            V64::splat(z),
            &mut a_val,
            &mut a_dx,
            &mut a_dy,
            &mut a_dz,
        );

        // The same thing by forward-mode AD over the value form.
        let mut out = [D::constant(V64::splat(0.0)); N];
        D::spherical_harmonics::<L, N, NO_PHASE>(
            D::variable(V64::splat(x), 0),
            D::variable(V64::splat(y), 1),
            D::variable(V64::splat(z), 2),
            &mut out,
        );

        for i in 0..N {
            let value = out[i].value().extract::<0>();
            let grad = out[i].gradient();

            let want_v = a_val[i].extract::<0>();
            assert!(
                (value - want_v).abs() < 1e-13 * (1.0 + want_v.abs()),
                "Dual value sh[{i}] at ({x},{y},{z}): got {value}, want {want_v}"
            );

            for (axis, (g, want)) in [
                ('x', (grad[0].extract::<0>(), a_dx[i].extract::<0>())),
                ('y', (grad[1].extract::<0>(), a_dy[i].extract::<0>())),
                ('z', (grad[2].extract::<0>(), a_dz[i].extract::<0>())),
            ] {
                assert!(
                    (g - want).abs() < 1e-11 * (1.0 + want.abs()),
                    "Dual d/d{axis} sh[{i}] at ({x},{y},{z}): AD {g}, analytic {want}"
                );
            }
        }
    }
}

/// The composition case `Dual` is actually for: the direction is itself a function of
/// an upstream parameter, here rotation about `z` by an angle `t`. The chain rule
/// through the whole evaluation is checked against a central difference in `t`, which
/// the analytic `_d` form could only reproduce by contracting its three gradients
/// against `d(x,y,z)/dt` by hand.
#[test]
fn sh_dual_chains_through_an_upstream_parameter() {
    type D = Dual<V64, 1>;

    let eval = |t: f64| -> [f64; N] {
        let (x, y, z) = (t.cos() * 0.8, t.sin() * 0.8, 0.6);
        reference(x, y, z)
    };

    for &t in &[0.0f64, 0.7, -1.3, 2.5] {
        // x = 0.8 cos t, y = 0.8 sin t, z = 0.6, built in dual arithmetic so the
        // derivative flows from `t` all the way through the recurrence.
        let td = D::variable(V64::splat(t), 0);
        let (st, ct) = (t.sin(), t.cos());
        let x = D::new(V64::splat(0.8 * ct), [V64::splat(-0.8 * st)]);
        let y = D::new(V64::splat(0.8 * st), [V64::splat(0.8 * ct)]);
        let z = D::constant(V64::splat(0.6));
        let _ = td;

        let mut out = [D::constant(V64::splat(0.0)); N];
        D::spherical_harmonics::<L, N, NO_PHASE>(x, y, z, &mut out);

        let h = 1e-6;
        let (hi, lo) = (eval(t + h), eval(t - h));

        for i in 0..N {
            let got = out[i].gradient()[0].extract::<0>();
            let want = (hi[i] - lo[i]) / (2.0 * h);
            assert!(
                (got - want).abs() < 1e-6 * (1.0 + want.abs()),
                "d sh[{i}]/dt at t={t}: dual {got}, finite difference {want}"
            );
        }
    }
}

/// `Dual` and `Compensated` reach the kernel through the generic default, which is only
/// exercised for real vectors above `MAX_SH_DEGREE`. Running a composite at a small
/// degree and a real vector at the same degree therefore checks the same code twice
/// from opposite directions.
#[test]
fn sh_composite_and_real_agree_at_low_degree() {
    type C = Compensated<V64>;
    const LS: usize = 2;
    const NS: usize = (LS + 1) * (LS + 1);

    let (x, y, z) = (0.267261241912424, 0.534522483824849, 0.801783725737273);

    let mut real = [V64::splat(0.0); NS];
    V64::spherical_harmonics::<LS, NS, NO_PHASE>(V64::splat(x), V64::splat(y), V64::splat(z), &mut real);

    let mut comp = [C::new(V64::splat(0.0)); NS];
    C::spherical_harmonics::<LS, NS, NO_PHASE>(
        C::new(V64::splat(x)),
        C::new(V64::splat(y)),
        C::new(V64::splat(z)),
        &mut comp,
    );

    for i in 0..NS {
        let (a, b) = (real[i].extract::<0>(), comp[i].value().extract::<0>());
        assert!(
            (a - b).abs() < 1e-14 * (1.0 + a.abs()),
            "sh[{i}]: real {a}, compensated {b}"
        );
    }
}

/// Constant-seeded duals take the path that drops dual arithmetic altogether: the
/// values must match plain `f64` _exactly_ (same kernel, same inputs) and every
/// derivative must be a hard zero rather than a rounded one.
#[test]
fn sh_dual_constant_seeding_is_flat_and_exact() {
    type D = Dual<V64, 3>;

    for &(x, y, z) in DIRS {
        let want = reference(x, y, z);

        let mut out = [D::constant(V64::splat(0.0)); N];
        D::spherical_harmonics::<L, N, NO_PHASE>(
            D::constant(V64::splat(x)),
            D::constant(V64::splat(y)),
            D::constant(V64::splat(z)),
            &mut out,
        );

        for i in 0..N {
            assert_eq!(
                out[i].value().extract::<0>(),
                want[i],
                "constant-seeded value sh[{i}] at ({x},{y},{z}) should be bit-identical to f64"
            );
            for (j, g) in out[i].gradient().iter().enumerate() {
                assert_eq!(
                    g.extract::<0>(),
                    0.0,
                    "constant-seeded d/d{j} sh[{i}] at ({x},{y},{z}) must be exactly zero"
                );
            }
        }
    }
}

/// The seeding-aware fast paths and the general dual recurrence must agree.
///
/// Scaling each Jacobian column by two is enough to fail the classifier's "exactly
/// one" test, so the identical mathematical question routes through the general path
/// instead, and by linearity its gradients must come back at exactly twice the
/// identity-seeded ones. Any divergence between the two implementations shows up here
/// and nowhere else.
#[test]
fn sh_dual_fast_and_general_paths_agree() {
    type D = Dual<V64, 3>;

    let unit = |v: f64, slot: usize| {
        let mut d = [V64::splat(0.0); 3];
        d[slot] = V64::splat(1.0);
        D::new(V64::splat(v), d)
    };
    let scaled = |v: f64, slot: usize| {
        let mut d = [V64::splat(0.0); 3];
        d[slot] = V64::splat(2.0);
        D::new(V64::splat(v), d)
    };

    for &(x, y, z) in DIRS {
        let mut fast = [D::constant(V64::splat(0.0)); N];
        D::spherical_harmonics::<L, N, NO_PHASE>(unit(x, 0), unit(y, 1), unit(z, 2), &mut fast);

        let mut general = [D::constant(V64::splat(0.0)); N];
        D::spherical_harmonics::<L, N, NO_PHASE>(scaled(x, 0), scaled(y, 1), scaled(z, 2), &mut general);

        for i in 0..N {
            let (a, b) = (fast[i].value().extract::<0>(), general[i].value().extract::<0>());
            assert!(
                (a - b).abs() < 1e-13 * (1.0 + a.abs()),
                "value sh[{i}] at ({x},{y},{z}): fast {a}, general {b}"
            );

            let (fg, gg) = (fast[i].gradient(), general[i].gradient());
            for j in 0..3 {
                let (f, g) = (fg[j].extract::<0>(), gg[j].extract::<0>() * 0.5);
                assert!(
                    (f - g).abs() < 1e-11 * (1.0 + f.abs()),
                    "d/d{j} sh[{i}] at ({x},{y},{z}): fast {f}, general (halved) {g}"
                );
            }
        }
    }
}

/// Nested duals must keep working. The fast paths are reachable without narrowing the
/// impl's bounds precisely so that `Dual<Dual<..>>` survives (it recurses into its own
/// classification rather than being locked out by a `RealPrimalMath` requirement), and
/// this is what would catch a regression that quietly reintroduced one.
#[test]
fn sh_nested_dual_still_works() {
    type Inner = Dual<V64, 1>;
    type Outer = Dual<Inner, 1>;

    let c = |v: f64| Outer::constant(Inner::constant(V64::splat(v)));
    let (x, y, z) = (0.267261241912424, 0.534522483824849, 0.801783725737273);
    let want = reference(x, y, z);

    let mut out = [c(0.0); N];
    Outer::spherical_harmonics::<L, N, NO_PHASE>(c(x), c(y), c(z), &mut out);

    for i in 0..N {
        let got = out[i].value().value().extract::<0>();
        assert!(
            (got - want[i]).abs() < 1e-13 * (1.0 + want[i].abs()),
            "nested-dual sh[{i}]: got {got}, want {}",
            want[i]
        );
    }
}
