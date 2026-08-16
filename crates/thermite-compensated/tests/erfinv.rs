//! `erfinv` / `probit` on `Compensated`.
//!
//! `erfinv` refines by Halley, seeded from the inner vector's own `erfinv`. Both branches
//! are exercised: `|y| <= MAX_ERFINV_SERIES` takes a Maclaurin series and never reaches
//! the iteration, above it takes the seed-and-refine path.
//!
//! References are mpmath 1.3.0 at 45 digits, as `(hi, lo)` pairs.

use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::RealSpecialMath;

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

/// Full double-double across both branches.
///
/// The Halley branch used to fall to 3.4e-27 as y approached 1, and the cause turned out
/// to be upstream: it refines against `erfc`, and `erfc` was weakest exactly where erfinv
/// leans on it hardest. In the series regime `erfc` came out as `1 - erf`, so at x = 2.751
/// (which is erfinv(0.9999)) it cancelled down to 89 bits. Moving `erf_internal_p`'s
/// regime split from 3 to 2 hands that band to the continued fraction, which computes
/// `erfc` directly - 101 bits - and erfinv's residual fell from 5.5e-30 to 1.8e-36.
///
/// Note this also explains why swapping the residual between `erf(x) - y` and
/// `(1 - y) - erfc(x)` was bit-for-bit identical: while erfc *is* `1 - erf`, the two carry
/// the same absolute error, so neither spelling can help. The fix had to be upstream.
const TOL: f64 = 1e-30;

fn dd_err(got: C, hi: f64, lo: f64) -> f64 {
    let value = got.value.extract::<0>();
    let error = got.error.extract::<0>();

    (((value - hi) + (error - lo)) / hi.abs().max(1.0)).abs()
}
const ERFINV: &[(f64, f64, f64)] = &[
    (0.1, 8.88559904942576861e-02, 5.83342499936309827e-18),
    (0.3, 2.72462714726754318e-01, 2.68960541024924764e-17),
    (0.5, 4.76936276204469878e-01, -4.42269683178365060e-18),
    (0.545, 5.28283225323181616e-01, 3.20283192855668704e-17),
    (0.6, 5.95116081449994838e-01, -1.61813903337274776e-17),
    (0.75, 8.13419847597618539e-01, 2.28710348214814837e-18),
    (0.9, 1.16308715367667426e+00, -9.40417080815607050e-17),
    (0.99, 1.82138636771844942e+00, 3.86760221464996528e-17),
    (0.999, 2.32675376551352464e+00, -1.47184501618361593e-16),
    (0.9999, 2.75106390571207982e+00, -1.23914895641999639e-16),
    (-0.7, -7.32869077959216741e-01, -4.34096569757189126e-17),
    (-0.95, -1.38590382434967774e+00, 6.09812889348374283e-17),
    (1e-06, 8.86226925452989969e-07, 1.82664998950488979e-23),
];

#[test]
fn erfinv_matches_mpmath() {
    for &(y, hi, lo) in ERFINV {
        let err = dd_err(c(y).erfinv(), hi, lo);
        assert!(err <= TOL, "erfinv({y}): rel err {err:e}");
    }
}

#[test]
fn erfinv_inverts_erf() {
    // erf(erfinv(y)) == y, which needs no oracle and pins both directions at once.
    use thermite_special::SpecialMath;

    for &(y, _, _) in ERFINV {
        let round_trip = c(y).erfinv().erf();
        let err = ((round_trip.value.extract::<0>() - y) / y).abs();

        assert!(err <= 1e-28, "erf(erfinv({y})) = {}", round_trip.value.extract::<0>());
    }
}

#[test]
fn erfinv_edges() {
    assert_eq!(c(0.0).erfinv().value.extract::<0>(), 0.0);
    assert!(c(1.0).erfinv().value.extract::<0>().is_infinite());
    assert!(c(-1.0).erfinv().value.extract::<0>().is_infinite());
}

#[test]
fn probit_matches_mpmath() {
    // Checked against an oracle rather than against `sqrt(2) * erfinv(2p - 1)` rebuilt in
    // the test: doing that needs the *compensated* SQRT_2 and an exactly-formed 2p - 1,
    // and reaching for `f64::consts::SQRT_2` instead silently measures that constant's
    // own 1e-17 error rather than probit.
    for &(p, hi, lo) in PROBIT {
        let err = dd_err(c(p).probit(), hi, lo);
        assert!(err <= TOL, "probit({p}): rel err {err:e}");
    }
}

const PROBIT: &[(f64, f64, f64)] = &[
    (0.025, -1.95996398454005427e+00, 5.96974766712090382e-17),
    (0.1, -1.28155156554460037e+00, -6.68941748811947383e-17),
    (0.5, 0.00000000000000000e+00, 0.00000000000000000e+00),
    (0.75, 6.74489750196081705e-01, 3.77555113550502866e-17),
    (0.975, 1.95996398454005383e+00, 2.82165796344530472e-17),
    (0.001, -3.09023230616781364e+00, 1.00863803761464288e-16),
];
