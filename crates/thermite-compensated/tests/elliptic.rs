//! The elliptic family on `Compensated`, through the `carlson` / `ellint` request structs,
//! against mpmath at 50 digits as `(hi, lo)` pairs.
//!
//! The kernels are generic over any float vector whose element carries the Carlson
//! convergence threshold. `Compensated<f64>` supplies its own, `(3 * 2^-104)^(1/8)`, so the
//! duplication runs a few more steps and the 7th-order tail lands below the second word.
//! Every input is exactly representable, so the references describe the argument actually
//! passed.

// `R_C(1, 2)` is `pi/4` exactly, and the generated table spells it out.
#![allow(clippy::approx_constant)]

use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::SpecialMath;
use thermite_special::elliptic::{
    CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE, EllintEInc, EllintF, EllintK,
    EllintPi, EllintPiInc,
};

include!("elliptic_ref/table.rs");

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

/// Error across both words, relative to the magnitude of the result.
fn dd_rel(got: C, want: (f64, f64)) -> f64 {
    let (hi, lo) = (got.value.extract::<0>(), got.error.extract::<0>());
    // (hi - want.0) is exact or nearly so at this scale. The residual carries the low words.
    let diff = ((hi - want.0) + (lo - want.1)).abs();
    diff / want.0.abs().max(f64::MIN_POSITIVE)
}

fn eval_c(kind: &str, a: &[f64]) -> C {
    match kind {
        "K" => C::ellint(EllintK { k: c(a[0]) }),
        "E" => C::ellint(EllintE { k: c(a[0]) }),
        "D" => C::ellint(EllintD { k: c(a[0]) }),
        "PiC" => C::ellint(EllintPi { n: c(a[0]), k: c(a[1]) }),
        "F" => C::ellint(EllintF { phi: c(a[0]), k: c(a[1]) }),
        "Einc" => C::ellint(EllintEInc { phi: c(a[0]), k: c(a[1]) }),
        "Dinc" => C::ellint(EllintDInc { phi: c(a[0]), k: c(a[1]) }),
        "Pi" => C::ellint(EllintPiInc { n: c(a[0]), phi: c(a[1]), k: c(a[2]) }),
        "Rf" => C::carlson(CarlsonRf { x: c(a[0]), y: c(a[1]), z: c(a[2]) }),
        "Rd" => C::carlson(CarlsonRd { x: c(a[0]), y: c(a[1]), z: c(a[2]) }),
        "Rg" => C::carlson(CarlsonRg { x: c(a[0]), y: c(a[1]), z: c(a[2]) }),
        "Rj" => C::carlson(CarlsonRj { x: c(a[0]), y: c(a[1]), z: c(a[2]), p: c(a[3]) }),
        "Rc" => C::carlson(CarlsonRc { x: c(a[0]), y: c(a[1]) }),
        other => unreachable!("{other}"),
    }
}

fn eval_v(kind: &str, a: &[f64]) -> f64 {
    let s = |x: f64| V::splat(x);
    let r = match kind {
        "K" => V::ellint(EllintK { k: s(a[0]) }),
        "E" => V::ellint(EllintE { k: s(a[0]) }),
        "D" => V::ellint(EllintD { k: s(a[0]) }),
        "PiC" => V::ellint(EllintPi { n: s(a[0]), k: s(a[1]) }),
        "F" => V::ellint(EllintF { phi: s(a[0]), k: s(a[1]) }),
        "Einc" => V::ellint(EllintEInc { phi: s(a[0]), k: s(a[1]) }),
        "Dinc" => V::ellint(EllintDInc { phi: s(a[0]), k: s(a[1]) }),
        "Pi" => V::ellint(EllintPiInc { n: s(a[0]), phi: s(a[1]), k: s(a[2]) }),
        "Rf" => V::carlson(CarlsonRf { x: s(a[0]), y: s(a[1]), z: s(a[2]) }),
        "Rd" => V::carlson(CarlsonRd { x: s(a[0]), y: s(a[1]), z: s(a[2]) }),
        "Rg" => V::carlson(CarlsonRg { x: s(a[0]), y: s(a[1]), z: s(a[2]) }),
        "Rj" => V::carlson(CarlsonRj { x: s(a[0]), y: s(a[1]), z: s(a[2]), p: s(a[3]) }),
        "Rc" => V::carlson(CarlsonRc { x: s(a[0]), y: s(a[1]) }),
        other => unreachable!("{other}"),
    };
    r.extract::<0>()
}

/// Per-family gate. Prints the worst per kind so a regression shows its number.
#[test]
fn elliptic_on_compensated_matches_mpmath_at_double_double() {
    let mut worst: std::collections::BTreeMap<&str, f64> = Default::default();
    let mut bad = Vec::new();
    for &(kind, args, want) in ELLIPTIC_REF {
        let got = eval_c(kind, args);
        let e = dd_rel(got, want);
        let plain = ((eval_v(kind, args) - want.0) - want.1).abs() / want.0.abs();
        let w = worst.entry(kind).or_insert(0.0);
        *w = w.max(e);
        // The whole point: the compensated result must beat plain f64 at the same point.
        if e > 1e-26 || e > plain {
            bad.push(format!("{kind}{args:?}: dd rel {e:e}, plain f64 rel {plain:e}"));
        }
    }
    for (k, w) in &worst {
        std::println!("{k}: worst dd relative {w:e}");
    }
    assert!(bad.is_empty(), "{} rows over the gate:\n{}", bad.len(), bad.join("\n"));
}
