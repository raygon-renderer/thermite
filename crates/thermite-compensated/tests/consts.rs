//! The vector constant tables must equal the scalar ones, lane for lane.
//!
//! `consts.rs` builds every `Vector<R>` constant through the `SplatConst`/`const_splat`
//! carrier path: a carrier *type* exposes one `const VALUE`, and `const_splat` turns it
//! into an all-lanes-equal vector. For the log tables the carrier takes the entry index
//! as a `const I: usize` and the table is unrolled one carrier per entry.
//!
//! That unrolling is exactly what this file guards. A wrong index literal, a swapped
//! `value`/`error` limb, or a short index list would still compile and still produce
//! plausible-looking constants - it would just silently return the wrong number. So
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
                check_named_consts!(
                    V, E;
                    NEG_ZERO, E, EULER_GAMMA, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_1_SQRT_3, FRAC_2_PI,
                    FRAC_1_SQRT_PI, FRAC_2_SQRT_PI, FRAC_SQRT_PI_2, FRAC_1_SQRT_TAU, FRAC_PI_2,
                    FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, FRAC_PI_180, FRAC_180_PI, LN_2,
                    LN_10, LN_PI, FRAC_LN_PI_2, LOG2_10, LOG2_E, LOG10_2, LOG10_E, PI, PI_SQUARED,
                    PI_CUBED, PI_FOURTH, SQRT_2, SQRT_3, SQRT_E, TAU, SQRT_FRAC_PI_2, SQRT_TAU, PHI,
                    FRAC_1_3, FRAC_2_3, FRAC_1_4, FRAC_1_6, FRAC_NEG_1_E, EPSILON, SQRT_EPSILON,
                    FOURTH_ROOT_EPSILON,
                );
            }

            /// Independent anchor: the carriers could be self-consistently wrong if the
            /// scalar table itself were misread. Pin a few double-double constants against
            /// their true values - `value + error` must round-trip to the exact quantity.
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
