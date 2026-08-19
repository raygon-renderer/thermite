//! The vector constant tables must equal the scalar ones, lane for lane.
//!
//! `consts.rs` builds every `Vector<R>` constant through the `SplatConst`/`const_splat`
//! carrier path: a carrier *type* exposes one `const VALUE`, and `const_splat` turns it
//! into an all-lanes-equal vector. For the log tables the carrier takes the entry index
//! as a `const I: usize` and the table is unrolled one carrier per entry.
//!
//! That unrolling is exactly what this file guards. A wrong index literal, a swapped
//! `value`/`error` limb, or a short index list would still compile and still produce
//! plausible-looking constants, just silently returning the wrong number. So
//! check each entry against the scalar `R::Element` constant it is supposed to splat.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::prelude::*;
use thermite_compensated::consts::{CompensatedLogTable, LOG_TABLE_SIZE, SplitFloatConsts};

/// Every lane of `v` must equal `expected` bit-for-bit. These are table constants, not
/// computed results, so anything short of exact equality is a bug.
fn assert_all_lanes<V: FloatVector>(v: V, expected: V::Element, what: &str)
where
    V::Element: PartialEq + core::fmt::Debug,
{
    for lane in 0..V::LANES {
        assert_eq!(v.extractv(lane), expected, "{what}: lane {lane}");
    }
}

/// Checks one named constant per listed ident, both limbs, for a (vector, element) pair.
///
/// Lives at file scope rather than inside `const_suite!` because a `macro_rules!` defined
/// inside another macro's expansion cannot use `$` metavariables of its own.
macro_rules! check_named_consts {
    ($v:ty, $e:ty; $($name:ident),* $(,)?) => {$({
        let v = <$v as SplitFloatConsts<$v>>::$name;
        let s = <$e as SplitFloatConsts<$e>>::$name;
        assert_all_lanes(v.value, s.value, concat!(stringify!($name), ".value"));
        assert_all_lanes(v.error, s.error, concat!(stringify!($name), ".error"));
    })*};
}

macro_rules! const_suite {
    ($mod:ident, $vec:ty, $elem:ty, $tol:expr) => {
        mod $mod {
            use super::*;

            type V = $vec;
            type E = $elem;

            /// The 30-entry unroll in `log_table!`: entry `i` of the vector table must be
            /// entry `i` of the scalar table, both limbs. Catches a permuted, duplicated,
            /// or off-by-one index list.
            #[test]
            fn log_table_matches_scalar_entrywise() {
                let vec_table = <V as CompensatedLogTable<V>>::LOG_TABLE;
                let scalar_table = <E as CompensatedLogTable<E>>::LOG_TABLE;

                for i in 0..LOG_TABLE_SIZE {
                    assert_all_lanes(vec_table[i].value, scalar_table[i].value, &format!("LOG_TABLE[{i}].value"));
                    assert_all_lanes(vec_table[i].error, scalar_table[i].error, &format!("LOG_TABLE[{i}].error"));
                }
            }

            /// The two limbs must not be swapped. `error` is the low limb, so it is
            /// strictly smaller in magnitude than `value` for every non-zero entry -
            /// an entrywise-correct but limb-swapped table would still pass a weaker check.
            #[test]
            fn log_table_limbs_are_not_swapped() {
                let scalar_table = <E as CompensatedLogTable<E>>::LOG_TABLE;

                for (i, c) in scalar_table.iter().enumerate() {
                    if c.value != 0.0 {
                        assert!(
                            c.error.abs() < c.value.abs(),
                            "LOG_TABLE[{i}]: error limb {:?} is not smaller than value limb {:?}",
                            c.error,
                            c.value
                        );
                    }
                }
            }

            #[test]
            fn ln_2_extended_matches_scalar() {
                let vec_table = <V as CompensatedLogTable<V>>::LN_2_EXTENDED;
                let scalar_table = <E as CompensatedLogTable<E>>::LN_2_EXTENDED;

                for i in 0..3 {
                    assert_all_lanes(vec_table[i], scalar_table[i], &format!("LN_2_EXTENDED[{i}]"));
                }
            }

            /// The per-name carriers generated inside `impl_consts!`. One `$const _Value` /
            /// `$const _Error` pair per constant, so a paste mix-up would cross two
            /// constants' limbs.
            #[test]
            fn named_consts_match_scalar() {
                thermite::for_each_float_const!(check_named_consts, V, E;);
            }

            /// Independent anchor: the carriers could be self-consistently wrong if the
            /// scalar table itself were misread. Pin a few double-double constants against
            /// their true values: `value + error` must round-trip to the exact quantity.
            ///
            /// `$tol` is element-dependent: a double-f64 carries ~106 bits, but a double-f32
            /// only ~48, so its `hi + lo` sits a few times 1e-15 from the true value.
            #[test]
            fn named_consts_are_numerically_right() {
                let pi = <V as SplitFloatConsts<V>>::PI;
                let hi = pi.value.extractv(0) as f64;
                let lo = pi.error.extractv(0) as f64;
                assert!(
                    (hi + lo - core::f64::consts::PI).abs() <= $tol,
                    "PI: hi+lo = {} vs {}",
                    hi + lo,
                    core::f64::consts::PI
                );

                let ln2 = <V as SplitFloatConsts<V>>::LN_2;
                let hi = ln2.value.extractv(0) as f64;
                let lo = ln2.error.extractv(0) as f64;
                assert!(
                    (hi + lo - core::f64::consts::LN_2).abs() <= $tol,
                    "LN_2: hi+lo = {} vs {}",
                    hi + lo,
                    core::f64::consts::LN_2
                );
            }
        }
    };
}

const_suite!(v3_f64x4, thermite::backend::x86_v3::f64x4, f64, 1e-15);
const_suite!(v3_f32x8, thermite::backend::x86_v3::f32x8, f32, 1e-14);
const_suite!(v1_f64x2, thermite::backend::x86_v1::f64x2, f64, 1e-15);
const_suite!(scalar_f64, thermite::Vector<f64>, f64, 1e-15);

/// The double-double constants must agree with thermite's own `FloatConsts`
/// values (the same mathematical constant, to f64 precision), for every name.
///
/// Regression guard: `SQRT_FRAC_PI_2` was `sqrt(2/pi)` (0.7978...) here while
/// thermite defines it as `sqrt(pi/2)` (1.2533...). Found by the interval
/// crate's cross-check, and it made `Compensated::gaussian_integral` off by a
/// factor of pi/2. This test would have caught it: the double-double `value`
/// limb of a correctly-split constant is exactly the correctly-rounded f64.
#[test]
fn split_consts_agree_with_thermite_float_consts() {
    use thermite::math::FloatConsts;

    macro_rules! check {
        ($($name:ident),* $(,)?) => {$(
            let dd = <f64 as SplitFloatConsts<f64>>::$name;
            let point = <f64 as FloatConsts>::$name;
            assert_eq!(
                dd.value, point,
                concat!(stringify!($name), ": double-double value limb {} != thermite's {}"),
                dd.value, point
            );
            let dd32 = <f32 as SplitFloatConsts<f32>>::$name;
            let point32 = <f32 as FloatConsts>::$name;
            assert_eq!(dd32.value, point32, concat!(stringify!($name), " (f32)"));
        )*};
    }

    // The EPSILON family is deliberately excluded: a double-double carries its own,
    // much smaller epsilon (2^-105, not f64::EPSILON), so those three never match.
    thermite::for_each_math_const!(check);
}
