//! Cross-checks the generated `BoundedFloatConsts` table against an
//! INDEPENDENT high-precision source: thermite-compensated's double-double
//! constants (a different generator, decimal-string seeded). For every name:
//!
//! 1. the pair brackets the double-double value exactly (`RD <= v + e <= RU`),
//! 2. it contains the correctly-rounded `FloatConsts` point value,
//! 3. it is at most one ulp wide (`RU == RD` or `RU == next_up(RD)`),
//! 4. exactly representable constants are degenerate.

use thermite::math::FloatConsts;
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_interval::BoundedFloatConsts;

type V1 = Vector<f64>;
type C = Compensated<V1>;

/// `lo <= (v + e)` exactly, for a double-double `(v, e)`.
fn le_dd(lo: f64, v: f64, e: f64) -> bool {
    lo < v || (lo == v && e >= 0.0)
}
fn ge_dd(hi: f64, v: f64, e: f64) -> bool {
    hi > v || (hi == v && e <= 0.0)
}

macro_rules! check_all {
    ($($name:ident),* $(,)?) => {
        $(
            #[allow(non_snake_case)]
            #[test]
            fn $name() {
                let (lo, hi) = <f64 as BoundedFloatConsts>::$name;
                let point = <f64 as FloatConsts>::$name;
                let dd = <C as FloatConsts>::$name;
                let (v, e) = (dd.value().extract::<0>(), dd.error().extract::<0>());

                assert!(lo <= hi, "{}: lo > hi", stringify!($name));
                assert!(
                    le_dd(lo, v, e) && ge_dd(hi, v, e),
                    "{}: [{lo:e}, {hi:e}] does not bracket the double-double {v:e} + {e:e}",
                    stringify!($name)
                );
                assert!(lo <= point && point <= hi, "{}: point value {point:e} outside [{lo:e}, {hi:e}]", stringify!($name));
                assert!(
                    hi == lo || hi == lo.next_up(),
                    "{}: wider than one ulp: [{lo:e}, {hi:e}]",
                    stringify!($name)
                );

                // The vector lift agrees with the element table.
                let (vlo, vhi) = <V1 as BoundedFloatConsts<V1>>::$name;
                assert_eq!(vlo.extract::<0>().to_bits(), lo.to_bits());
                assert_eq!(vhi.extract::<0>().to_bits(), hi.to_bits());

                // f32 has the same properties against its own point value.
                let (lo32, hi32) = <f32 as BoundedFloatConsts>::$name;
                let point32 = <f32 as FloatConsts>::$name;
                assert!(lo32 <= point32 && point32 <= hi32, "{} f32: point outside", stringify!($name));
                assert!(hi32 == lo32 || hi32 == lo32.next_up(), "{} f32: wider than one ulp", stringify!($name));
                // and the f32 pair must bracket the f64 double-double too.
                assert!(
                    (lo32 as f64) <= v + e && v + e <= (hi32 as f64),
                    "{} f32: [{lo32:e}, {hi32:e}] does not bracket the true value",
                    stringify!($name)
                );
            }
        )*
    };
}

// The EPSILON family is format-specific, so `epsilon_family` below covers it instead.
thermite::for_each_math_const!(check_all);

/// The EPSILON family is format-specific (Compensated's is its own
/// double-double epsilon, so it is not a reference here). Check exactness of
/// the representable ones and the one-ulp property of the rest.
#[test]
fn epsilon_family() {
    let (lo, hi) = <f64 as BoundedFloatConsts>::EPSILON;
    assert_eq!((lo, hi), (f64::EPSILON, f64::EPSILON), "f64 EPSILON is exact");
    let (lo, hi) = <f64 as BoundedFloatConsts>::SQRT_EPSILON;
    assert_eq!((lo, hi), (2f64.powi(-26), 2f64.powi(-26)), "sqrt(2^-52) = 2^-26 exact");
    let (lo, hi) = <f64 as BoundedFloatConsts>::FOURTH_ROOT_EPSILON;
    assert_eq!((lo, hi), (2f64.powi(-13), 2f64.powi(-13)), "2^-13 exact");

    let (lo, hi) = <f32 as BoundedFloatConsts>::EPSILON;
    assert_eq!((lo, hi), (f32::EPSILON, f32::EPSILON));
    // sqrt(2^-23) = 2^-11.5 is NOT representable in f32: one ulp wide, brackets it.
    let (lo, hi) = <f32 as BoundedFloatConsts>::SQRT_EPSILON;
    let truth = (2f64.powi(-23)).sqrt();
    assert!((lo as f64) < truth && truth < (hi as f64) && hi == lo.next_up(), "[{lo:e}, {hi:e}] vs {truth:e}");
}

/// Exactly representable constants are degenerate.
#[test]
fn representable_are_degenerate() {
    let (lo, hi) = <f64 as BoundedFloatConsts>::FRAC_1_4;
    assert_eq!((lo, hi), (0.25, 0.25));
    let (lo, hi) = <f64 as BoundedFloatConsts>::NEG_ZERO;
    assert_eq!(lo.to_bits(), (-0.0f64).to_bits());
    assert_eq!(hi.to_bits(), (-0.0f64).to_bits());
    // ...and irrational ones are not.
    let (lo, hi) = <f64 as BoundedFloatConsts>::PI;
    assert!(lo < hi);
}

/// AUDIT of thermite's own `FloatConsts`: every point constant must be the
/// correctly rounded (nearest, ties-to-even) value of its mathematical
/// definition, bit for bit, in both formats. The reference table is generated
/// by exact rational comparison against 60-digit mpmath values.
///
/// Reports every mismatch at once (a computed constant like `PI / 180.0` is
/// only _usually_ correctly rounded, and this is how you find the ones that
/// are not).
#[test]
fn thermite_float_consts_are_correctly_rounded() {
    use thermite_interval::consts_table::nearest;

    let mut bad: Vec<String> = Vec::new();

    macro_rules! audit {
        ($($name:ident),* $(,)?) => {$(
            let have = <f64 as FloatConsts>::$name;
            let want = nearest::f64::$name;
            if have.to_bits() != want.to_bits() {
                bad.push(format!(
                    "f64 {}: thermite {:e} (0x{:016X}) != correctly rounded {:e} (0x{:016X})",
                    stringify!($name), have, have.to_bits(), want, want.to_bits()
                ));
            }
            let have = <f32 as FloatConsts>::$name;
            let want = nearest::f32::$name;
            if have.to_bits() != want.to_bits() {
                bad.push(format!(
                    "f32 {}: thermite {:e} (0x{:08X}) != correctly rounded {:e} (0x{:08X})",
                    stringify!($name), have, have.to_bits(), want, want.to_bits()
                ));
            }
        )*};
    }

    thermite::for_each_float_const!(audit);

    assert!(bad.is_empty(), "{} thermite constants are not correctly rounded:\n  {}", bad.len(), bad.join("\n  "));
}
