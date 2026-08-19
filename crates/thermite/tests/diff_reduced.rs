//! `ReducedRegister` coverage via 3-lane vectors (`f32x3A` = `ReducedRegister
//! <f32x4, U1>`, etc.) on X86V2/X86V3. `register/reduced.rs` had 0% coverage.
//!
//! Tested at the **`Vector` layer** against per-lane Rust oracles (the reduced
//! register emulates a 3-lane vector inside a 4-lane register, and only the first
//! three lanes are meaningful, so only those are checked). Pure arithmetic /
//! bitwise / rounding ops are bit-exact (NaN-aware via the harness).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;
use rand::RngExt;
use thermite::Vector;
use thermite::divider::Denominator;
use thermite::prelude::*;
use thermite::simd::Simd3A;
use thermite::vector::ops::DivMasked;

use thermite::backend::scalar::Scalar;

/// Binary op: `vop` on vectors vs `sop` per lane in Rust.
macro_rules! bin {
    ($label:expr, $V:ty, $e:ty, $vop:expr, $sop:expr) => {{
        let mut rng = harness::rng();
        let vop: fn($V, $V) -> $V = $vop;
        let sop: fn($e, $e) -> $e = $sop;
        let xs = harness::corpus::<$e>(3, &mut rng);
        let ys = harness::corpus::<$e>(3, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let g = vop(<$V>::from_slice(x), <$V>::from_slice(y)).into_array();
            let got = g.as_slice()[..3].to_vec();
            let want: Vec<$e> = (0..3).map(|i| sop(x[i], y[i])).collect();
            harness::assert_lanes_eq($label, &[x.as_slice(), y.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

/// Unary op.
macro_rules! un {
    ($label:expr, $V:ty, $e:ty, $vop:expr, $sop:expr) => {{
        let mut rng = harness::rng();
        let vop: fn($V) -> $V = $vop;
        let sop: fn($e) -> $e = $sop;
        for x in harness::corpus::<$e>(3, &mut rng) {
            let g = vop(<$V>::from_slice(&x)).into_array();
            let got = g.as_slice()[..3].to_vec();
            let want: Vec<$e> = (0..3).map(|i| sop(x[i])).collect();
            harness::assert_lanes_eq($label, &[x.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

macro_rules! reduced_suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;

            #[test]
            fn float_f32() {
                float_ops!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
            }
            #[test]
            fn float_f64() {
                float_ops!(<$backend as Simd3A>::f64x3A, f64, concat!($bl, " f64x3A"));
            }
            #[test]
            fn int_i32() {
                int_ops!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                un!(
                    concat!($bl, " i32x3A [neg]"),
                    Vector<<$backend as Simd3A>::i32x3A>,
                    i32,
                    |a| -a,
                    |x: i32| x.wrapping_neg()
                );
            }
            #[test]
            fn int_u32() {
                int_ops!(<$backend as Simd3A>::u32x3A, u32, concat!($bl, " u32x3A"));
            }
            #[test]
            fn int_i64() {
                int_ops!(<$backend as Simd3A>::i64x3A, i64, concat!($bl, " i64x3A"));
                un!(
                    concat!($bl, " i64x3A [neg]"),
                    Vector<<$backend as Simd3A>::i64x3A>,
                    i64,
                    |a| -a,
                    |x: i64| x.wrapping_neg()
                );
            }
            #[test]
            fn int_u64() {
                int_ops!(<$backend as Simd3A>::u64x3A, u64, concat!($bl, " u64x3A"));
            }

            // --- ReducedRegister-specific logic (upper-lane masking) ---

            #[test]
            fn reductions() {
                // sum/prod/min/max must ignore the (zero-padded) 4th lane.
                reduce!(
                    Vector<<$backend as Simd3A>::f32x3A>,
                    f32,
                    concat!($bl, " f32x3A"),
                    Tol::Rel(2.0e-4)
                );
                reduce!(
                    Vector<<$backend as Simd3A>::f64x3A>,
                    f64,
                    concat!($bl, " f64x3A"),
                    Tol::Rel(1.0e-12)
                );
                reduce!(
                    Vector<<$backend as Simd3A>::i32x3A>,
                    i32,
                    concat!($bl, " i32x3A"),
                    Tol::Exact
                );
            }

            #[test]
            fn masks() {
                // all/any/none/bitmask must be masked to 3 lanes (BITMASK).
                maskt!(Vector<<$backend as Simd3A>::f32x3A>, f32, concat!($bl, " f32x3A"));
                maskt!(Vector<<$backend as Simd3A>::i32x3A>, i32, concat!($bl, " i32x3A"));
            }

            #[test]
            fn memory() {
                // load/store roundtrip + masked gather/scatter over exactly 3 lanes.
                memt!(Vector<<$backend as Simd3A>::f32x3A>, f32, concat!($bl, " f32x3A"));
                memt!(Vector<<$backend as Simd3A>::i32x3A>, i32, concat!($bl, " i32x3A"));
            }

            #[test]
            fn swizzle() {
                // reverse/broadcast/extract/insert with the reduced index adjustment.
                swiz!(Vector<<$backend as Simd3A>::f32x3A>, f32, concat!($bl, " f32x3A"));
                swiz!(Vector<<$backend as Simd3A>::i32x3A>, i32, concat!($bl, " i32x3A"));
            }

            #[test]
            fn byte_shifts() {
                // elem-size byte shift = shift by exactly one lane
                bshift!(<$backend as Simd3A>::u32x3A, u32, 4, concat!($bl, " u32x3A"));
                bshift!(<$backend as Simd3A>::i32x3A, i32, 4, concat!($bl, " i32x3A"));
                bshift!(<$backend as Simd3A>::u64x3A, u64, 8, concat!($bl, " u64x3A"));
            }

            #[test]
            fn interleave() {
                // 3-lane (4-lane storage reduced by 1), including the
                // ReducedRegister-over-ArrayRegister case (f64x3A on v1/v2).
                ileave!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
                ileave!(<$backend as Simd3A>::f64x3A, f64, concat!($bl, " f64x3A"));
                ileave!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                ileave!(<$backend as Simd3A>::u64x3A, u64, concat!($bl, " u64x3A"));

                // 2-lane half registers (4-lane storage reduced by 2).
                ileave!(
                    <$backend as thermite::simd::Simd>::f32x2,
                    f32,
                    concat!($bl, " f32x2")
                );
                ileave!(
                    <$backend as thermite::simd::Simd>::i32x2,
                    i32,
                    concat!($bl, " i32x2")
                );
            }

            // --- broader register-trait coverage (the bulk of reduced.rs) ---

            #[test]
            fn compares() {
                // PartialOrdRegister: all six predicates -> Mask.
                cmpt!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
                cmpt!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                cmpt!(<$backend as Simd3A>::u32x3A, u32, concat!($bl, " u32x3A"));
                cmpt!(<$backend as Simd3A>::i64x3A, i64, concat!($bl, " i64x3A"));
            }

            #[test]
            fn predicates() {
                // FloatRegister classification predicates -> Mask.
                predt!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
                predt!(<$backend as Simd3A>::f64x3A, f64, concat!($bl, " f64x3A"));
            }

            #[test]
            fn signed_ops() {
                // SignedRegister: signum + is_negative.
                signedt!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                signedt!(<$backend as Simd3A>::i64x3A, i64, concat!($bl, " i64x3A"));
            }

            #[test]
            fn float_extra() {
                // copysign / min / max / mix (FloatRegister + NumericRegister).
                fextra!(
                    <$backend as Simd3A>::f32x3A,
                    f32,
                    concat!($bl, " f32x3A"),
                    Tol::Rel(2.0e-4)
                );
                fextra!(
                    <$backend as Simd3A>::f64x3A,
                    f64,
                    concat!($bl, " f64x3A"),
                    Tol::Rel(1.0e-12)
                );
            }

            #[test]
            fn casts() {
                // CastRegister + BitCastRegister.
                castt!(
                    <$backend as Simd3A>::f32x3A,
                    <$backend as Simd3A>::i32x3A,
                    <$backend as Simd3A>::u32x3A,
                    concat!($bl, " f32x3A")
                );
            }

            #[test]
            fn linalg3() {
                // LinAlg3Register: dot3/cross3/zero4/one4/element3/mat3 ops.
                linalg3!(
                    <$backend as Simd3A>::f32x3A,
                    f32,
                    concat!($bl, " f32x3A"),
                    Tol::Rel(2.0e-3)
                );
                linalg3!(
                    <$backend as Simd3A>::f64x3A,
                    f64,
                    concat!($bl, " f64x3A"),
                    Tol::Rel(1.0e-11)
                );
            }

            #[test]
            fn division() {
                // Constant Divider (divv_branchfree): correct on every backend/type.
                rdiv_const!(<$backend as Simd3A>::i32x3A, i32, true, concat!($bl, " i32x3A"));
                rdiv_const!(<$backend as Simd3A>::u32x3A, u32, false, concat!($bl, " u32x3A"));
                rdiv_const!(<$backend as Simd3A>::i64x3A, i64, true, concat!($bl, " i64x3A"));
                rdiv_const!(<$backend as Simd3A>::u64x3A, u64, false, concat!($bl, " u64x3A"));
                // Per-lane VectorDivider + masked div_c/_m/_z, all widths/backends
                // (the 64-bit SSE variable-shift path is exercised here too).
                rdiv_vec!(<$backend as Simd3A>::i32x3A, i32, true, concat!($bl, " i32x3A"));
                rdiv_vec!(<$backend as Simd3A>::u32x3A, u32, false, concat!($bl, " u32x3A"));
                rdiv_vec!(<$backend as Simd3A>::i64x3A, i64, true, concat!($bl, " i64x3A"));
                rdiv_vec!(<$backend as Simd3A>::u64x3A, u64, false, concat!($bl, " u64x3A"));
            }

            #[test]
            fn pow2() {
                // UnsignedIntegerRegister::is_power_of_two (nonzero inputs; 0 is a
                // documented divergence).
                pow2t!(<$backend as Simd3A>::u32x3A, u32, concat!($bl, " u32x3A"));
                pow2t!(<$backend as Simd3A>::u64x3A, u64, concat!($bl, " u64x3A"));
            }

            #[test]
            fn masked() {
                // CoreRegister blendv/zz/nz via masked arithmetic + zz/nz.
                maskedi!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                maskedf!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
            }

            #[test]
            fn numeric_extra() {
                // indexed / is_all_zero / is_zero.
                numx!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
                numx!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
            }

            #[test]
            fn swizzle_ops() {
                // Register: runtime permute (vs Rust oracle) and the
                // two-input swizzle (vs the scalar reduced backend, so the exact
                // index contract need not be restated here).
                sw_permute!(<$backend as Simd3A>::f32x3A, f32, concat!($bl, " f32x3A"));
                sw_permute!(<$backend as Simd3A>::i32x3A, i32, concat!($bl, " i32x3A"));
                sw_swizzle!(
                    <$backend as Simd3A>::f32x3A,
                    <Scalar as Simd3A>::f32x3A,
                    f32,
                    concat!($bl, " f32x3A")
                );
                sw_swizzle!(
                    <$backend as Simd3A>::i32x3A,
                    <Scalar as Simd3A>::i32x3A,
                    i32,
                    concat!($bl, " i32x3A")
                );
            }
        }
    };
}

/// Byte shifts on a 3-lane register must behave as if the register were only
/// 3 lanes wide: `bshri` shifts zeros in from the top, NOT the padding lane's
/// contents. A freshly-built reduced register has zero padding (which would
/// hide the bug), so the input is taken from `interleave`'s low output, whose
/// padding lane holds live junk (`b1`).
macro_rules! bshift {
    ($rt:ty, $e:ty, $elem_bytes:literal, $l:expr) => {{
        use thermite::register::{BitshiftRegister, InterleaveRegister};

        let a_in: [$e; 3] = [10 as $e, 11 as $e, 12 as $e];
        let b_in: [$e; 3] = [20 as $e, 21 as $e, 22 as $e];

        let a = harness::make_array::<$rt>(&a_in);
        let b = harness::make_array::<$rt>(&b_in);

        // lo = [a0, b0, a1] with junk = b1 in the padding lane
        let (lo, _) = <$rt>::interleave(a, b);

        // shift down one lane: [b0, a1, 0] - the buggy full-width shift gave [b0, a1, b1]
        let got = harness::read::<$rt>(&<$rt>::bshri::<$elem_bytes>(lo));
        harness::assert_lanes_eq(
            concat!($l, " [bshri]"),
            &[&a_in, &b_in],
            &got,
            &[b_in[0], a_in[1], 0 as $e],
            Tol::Exact,
        );

        // shift up one lane: [0, a0, b0] (junk only moves further into padding)
        let got = harness::read::<$rt>(&<$rt>::bshli::<$elem_bytes>(lo));
        harness::assert_lanes_eq(
            concat!($l, " [bshli]"),
            &[&a_in, &b_in],
            &got,
            &[0 as $e, a_in[0], b_in[0]],
            Tol::Exact,
        );
    }};
}

/// Register-layer interleave/deinterleave against the stream definition:
/// `interleave(a, b)` yields the stream `[a0, b0, a1, b1, ...]` split into two
/// L-lane registers; `deinterleave` inverts it. The deinterleave inputs come
/// straight from interleave's outputs, so their junk padding lanes exercise
/// the upper-lane don't-care handling.
macro_rules! ileave {
    ($rt:ty, $e:ty, $l:expr) => {{
        use thermite::register::InterleaveRegister;

        let mut rng = harness::rng();
        let lanes = <<$rt as thermite::register::CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;

        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ys = harness::corpus::<$e>(lanes, &mut rng);

        for (a_in, b_in) in xs.iter().zip(ys.iter()) {
            let a = harness::make_array::<$rt>(a_in);
            let b = harness::make_array::<$rt>(b_in);

            let mut stream: Vec<$e> = Vec::with_capacity(2 * lanes);
            for i in 0..lanes {
                stream.push(a_in[i]);
                stream.push(b_in[i]);
            }

            let (lo, hi) = <$rt>::interleave(a, b);

            harness::assert_lanes_eq(
                concat!($l, " [interleave lo]"),
                &[a_in.as_slice(), b_in.as_slice()],
                &harness::read::<$rt>(&lo),
                &stream[..lanes],
                Tol::Exact,
            );
            harness::assert_lanes_eq(
                concat!($l, " [interleave hi]"),
                &[a_in.as_slice(), b_in.as_slice()],
                &harness::read::<$rt>(&hi),
                &stream[lanes..],
                Tol::Exact,
            );

            let (ra, rb) = <$rt>::deinterleave(lo, hi);

            harness::assert_lanes_eq(
                concat!($l, " [deinterleave a]"),
                &[a_in.as_slice(), b_in.as_slice()],
                &harness::read::<$rt>(&ra),
                a_in,
                Tol::Exact,
            );
            harness::assert_lanes_eq(
                concat!($l, " [deinterleave b]"),
                &[a_in.as_slice(), b_in.as_slice()],
                &harness::read::<$rt>(&rb),
                b_in,
                Tol::Exact,
            );
        }
    }};
}

macro_rules! float_ops {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        bin!(concat!($l, " [add]"), V, $e, |a, b| a + b, |x, y| x + y);
        bin!(concat!($l, " [sub]"), V, $e, |a, b| a - b, |x, y| x - y);
        bin!(concat!($l, " [mul]"), V, $e, |a, b| a * b, |x, y| x * y);
        bin!(concat!($l, " [div]"), V, $e, |a, b| a / b, |x, y| x / y);
        un!(concat!($l, " [neg]"), V, $e, |a| -a, |x| -x);
        un!(concat!($l, " [abs]"), V, $e, |a| a.abs(), |x: $e| x.abs());
        un!(concat!($l, " [sqrt]"), V, $e, |a| a.sqrt(), |x: $e| x.sqrt());
        un!(concat!($l, " [floor]"), V, $e, |a| a.floor(), |x: $e| x.floor());
        un!(concat!($l, " [ceil]"), V, $e, |a| a.ceil(), |x: $e| x.ceil());
        un!(concat!($l, " [trunc]"), V, $e, |a| a.trunc(), |x: $e| x.trunc());
    }};
}

macro_rules! int_ops {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        bin!(concat!($l, " [add]"), V, $e, |a, b| a + b, |x, y| x.wrapping_add(y));
        bin!(concat!($l, " [sub]"), V, $e, |a, b| a - b, |x, y| x.wrapping_sub(y));
        bin!(concat!($l, " [mul]"), V, $e, |a, b| a * b, |x, y| x.wrapping_mul(y));
        bin!(concat!($l, " [and]"), V, $e, |a, b| a & b, |x, y| x & y);
        bin!(concat!($l, " [or]"), V, $e, |a, b| a | b, |x, y| x | y);
        bin!(concat!($l, " [xor]"), V, $e, |a, b| a ^ b, |x, y| x ^ y);
        un!(concat!($l, " [not]"), V, $e, |a| !a, |x| !x);
    }};
}

/// Horizontal reductions: sum/prod/min/max over exactly 3 lanes. The cases
/// include all-positive (min must skip the 0-pad lane) and all-negative (max
/// must skip it) so the upper-lane masking is actually exercised.
macro_rules! reduce {
    ($V:ty, $e:ty, $l:expr, $tol:expr) => {{
        let cases: &[[$e; 3]] = &[
            [1 as $e, 2 as $e, 3 as $e],
            [3 as $e, 1 as $e, 2 as $e],
            [2 as $e, 2 as $e, 2 as $e],
            [5 as $e, 4 as $e, 9 as $e],    // all positive: min must ignore 0-pad
            [-1 as $e, -2 as $e, -3 as $e], // all negative: max must ignore 0-pad
            [-4 as $e, 3 as $e, -1 as $e],
            [7 as $e, -7 as $e, 1 as $e],
        ];
        for x in cases {
            let v = <$V>::from_slice(x);
            let osum = x[0] + x[1] + x[2];
            let oprod = x[0] * x[1] * x[2];
            let omin = x.iter().copied().reduce(|a, b| if b < a { b } else { a }).unwrap();
            let omax = x.iter().copied().reduce(|a, b| if b > a { b } else { a }).unwrap();
            harness::assert_lanes_eq(
                concat!($l, " [sum_elements]"),
                &[x.as_slice()],
                &[v.sum_elements()],
                &[osum],
                $tol,
            );
            harness::assert_lanes_eq(
                concat!($l, " [prod_elements]"),
                &[x.as_slice()],
                &[v.prod_elements()],
                &[oprod],
                $tol,
            );
            harness::assert_lanes_eq(
                concat!($l, " [min_element]"),
                &[x.as_slice()],
                &[v.min_element()],
                &[omin],
                Tol::Exact,
            );
            harness::assert_lanes_eq(
                concat!($l, " [max_element]"),
                &[x.as_slice()],
                &[v.max_element()],
                &[omax],
                Tol::Exact,
            );
        }
    }};
}

/// Mask reductions: all/any/none/bitmask must be restricted to the 3 real lanes.
macro_rules! maskt {
    ($V:ty, $e:ty, $l:expr) => {{
        let pairs: &[([$e; 3], [$e; 3])] = &[
            ([1 as $e, 2 as $e, 3 as $e], [3 as $e, 2 as $e, 1 as $e]),
            ([5 as $e, 5 as $e, 5 as $e], [1 as $e, 1 as $e, 1 as $e]),
            ([0 as $e, 0 as $e, 0 as $e], [1 as $e, 1 as $e, 1 as $e]), // all true => BITMASK gate for all()
            ([2 as $e, 9 as $e, 1 as $e], [2 as $e, 1 as $e, 9 as $e]),
        ];
        for (x, y) in pairs {
            let a = <$V>::from_slice(x);
            let b = <$V>::from_slice(y);
            let m = a.cmp_lt(b);
            let exp = [x[0] < y[0], x[1] < y[1], x[2] < y[2]];
            assert_eq!(m.all(), exp.iter().all(|&t| t), concat!($l, " mask.all"));
            assert_eq!(m.any(), exp.iter().any(|&t| t), concat!($l, " mask.any"));
            assert_eq!(m.none(), !exp.iter().any(|&t| t), concat!($l, " mask.none"));
            // all/any/none above already exercise the BITMASK-gated `native_bitmask`.
            let gs = m.select(a, b).into_array().as_slice()[..3].to_vec();
            let ws: Vec<$e> = (0..3).map(|i| if x[i] < y[i] { x[i] } else { y[i] }).collect();
            assert_eq!(gs, ws, concat!($l, " mask.select"));
        }
    }};
}

/// Memory: copy/into_array roundtrip + masked gather/scatter over 3 lanes.
macro_rules! memt {
    ($V:ty, $e:ty, $l:expr) => {{
        type U = <$V as GenericVector>::Unsigned;
        let x = [1 as $e, 2 as $e, 3 as $e];
        let v = <$V>::from_slice(&x);

        let mut out = [0 as $e; 3];
        v.copy_to_slice(&mut out);
        assert_eq!(out, x, concat!($l, " copy_to_slice"));
        assert_eq!(
            v.into_array().as_slice()[..3].to_vec(),
            x.to_vec(),
            concat!($l, " into_array")
        );

        let table = [10 as $e, 11 as $e, 12 as $e, 13 as $e, 14 as $e];
        let idx = U::from_slice(&[0u32, 2u32, 4u32]);
        let g = <$V>::gather(&table, idx);
        assert_eq!(
            g.into_array().as_slice()[..3].to_vec(),
            vec![10 as $e, 12 as $e, 14 as $e],
            concat!($l, " gather")
        );

        let mut dst = [0 as $e; 6];
        v.scatter(&mut dst, idx);
        assert_eq!(dst[0], 1 as $e, concat!($l, " scatter[0]"));
        assert_eq!(dst[2], 2 as $e, concat!($l, " scatter[2]"));
        assert_eq!(dst[4], 3 as $e, concat!($l, " scatter[4]"));
    }};
}

/// Lane routing: reverse/broadcast/extract/insert through the reduced index map.
macro_rules! swiz {
    ($V:ty, $e:ty, $l:expr) => {{
        let x = [1 as $e, 2 as $e, 3 as $e];
        let v = <$V>::from_slice(&x);
        assert_eq!(
            v.reverse().into_array().as_slice()[..3].to_vec(),
            vec![3 as $e, 2 as $e, 1 as $e],
            concat!($l, " reverse")
        );
        assert_eq!(
            v.broadcast::<1>().into_array().as_slice()[..3].to_vec(),
            vec![2 as $e; 3],
            concat!($l, " broadcast")
        );
        assert_eq!(v.extract::<2>(), 3 as $e, concat!($l, " extract"));
        assert_eq!(
            v.insert::<0>(9 as $e).into_array().as_slice()[..3].to_vec(),
            vec![9 as $e, 2 as $e, 3 as $e],
            concat!($l, " insert")
        );
    }};
}

/// Binary op against a Rust per-lane oracle, with a caller-chosen tolerance
/// (the `bin!` macro is hard-wired to `Tol::Exact`).
macro_rules! bint {
    ($label:expr, $V:ty, $e:ty, $vop:expr, $sop:expr, $tol:expr) => {{
        let mut rng = harness::rng();
        let vop: fn($V, $V) -> $V = $vop;
        let sop: fn($e, $e) -> $e = $sop;
        let xs = harness::corpus::<$e>(3, &mut rng);
        let ys = harness::corpus::<$e>(3, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let g = vop(<$V>::from_slice(x), <$V>::from_slice(y)).into_array();
            let got = g.as_slice()[..3].to_vec();
            let want: Vec<$e> = (0..3).map(|i| sop(x[i], y[i])).collect();
            harness::assert_lanes_eq($label, &[x.as_slice(), y.as_slice()], &got, &want, $tol);
        }
    }};
}

/// Read a mask's three real lanes back as bools via `select(1, 0)`.
macro_rules! mbools {
    ($V:ty, $e:ty, $m:expr) => {{
        let g = $m.select(<$V>::splat(1 as $e), <$V>::splat(0 as $e)).into_array();
        let g = g.as_slice();
        [g[0] != 0 as $e, g[1] != 0 as $e, g[2] != 0 as $e]
    }};
}

/// All six comparison predicates -> `Mask`, checked per lane against Rust.
macro_rules! cmpt {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let xs = harness::corpus::<$e>(3, &mut rng);
        let ys = harness::corpus::<$e>(3, &mut rng);
        let cases: [(fn(V, V) -> <V as GenericVector>::Mask, fn($e, $e) -> bool, &str); 6] = [
            (|a, b| a.cmp_lt(b), |x, y| x < y, "[cmp_lt]"),
            (|a, b| a.cmp_le(b), |x, y| x <= y, "[cmp_le]"),
            (|a, b| a.cmp_gt(b), |x, y| x > y, "[cmp_gt]"),
            (|a, b| a.cmp_ge(b), |x, y| x >= y, "[cmp_ge]"),
            (|a, b| a.cmp_eq(b), |x, y| x == y, "[cmp_eq]"),
            (|a, b| a.cmp_ne(b), |x, y| x != y, "[cmp_ne]"),
        ];
        for (vop, sop, nm) in cases {
            for (x, y) in xs.iter().zip(ys.iter()).take(800) {
                let got = mbools!(V, $e, vop(V::from_slice(x), V::from_slice(y)));
                for i in 0..3 {
                    assert_eq!(got[i], sop(x[i], y[i]), "{} {} lane {}", $l, nm, i);
                }
            }
        }
    }};
}

/// FloatRegister classification predicates -> `Mask`, checked against Rust.
macro_rules! predt {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let cases: [(fn(V) -> <V as GenericVector>::Mask, fn($e) -> bool, &str); 4] = [
            (|v| v.is_nan(), |x: $e| x.is_nan(), "is_nan"),
            (|v| v.is_finite(), |x: $e| x.is_finite(), "is_finite"),
            (|v| v.is_infinite(), |x: $e| x.is_infinite(), "is_infinite"),
            (|v| v.is_normal(), |x: $e| x.is_normal(), "is_normal"),
        ];
        for x in harness::corpus::<$e>(3, &mut rng).into_iter().take(400) {
            let v = V::from_slice(&x);
            for (vop, sop, nm) in cases {
                let got = mbools!(V, $e, vop(v));
                for i in 0..3 {
                    assert_eq!(got[i], sop(x[i]), "{} {} lane {}", $l, nm, i);
                }
            }
        }
    }};
}

/// SignedRegister: `signum` (exact) and `is_negative`.
macro_rules! signedt {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        for x in harness::corpus::<$e>(3, &mut rng).into_iter().take(400) {
            let v = V::from_slice(&x);
            let g = v.signum().into_array().as_slice()[..3].to_vec();
            let w: Vec<$e> = (0..3).map(|i| x[i].signum()).collect();
            assert_eq!(g, w, "{} signum", $l);
            let neg = mbools!(V, $e, v.is_negative());
            for i in 0..3 {
                assert_eq!(neg[i], x[i] < 0 as $e, "{} is_negative lane {}", $l, i);
            }
        }
    }};
}

/// copysign (exact, full corpus), min/max (`ExactOrNan`), and mix (`Rel`).
macro_rules! fextra {
    ($reg:ty, $e:ty, $l:expr, $tol:expr) => {{
        type V = Vector<$reg>;
        bin!(
            concat!($l, " [copysign]"),
            V,
            $e,
            |a, b| a.copysign(b),
            |x: $e, y: $e| x.copysign(y)
        );
        // Under `strict_ieee754` min/max define the tie semantics exactly:
        // min(-0, +0) = -0 and max(-0, +0) = +0 in either operand order, and
        // min/max(x, NaN) = x. The plain oracle leaves ties to operand order.
        bint!(
            concat!($l, " [min]"),
            V,
            $e,
            |a, b| a.min(b),
            |x: $e, y: $e| {
                #[cfg(feature = "strict_ieee754")]
                {
                    if y != y {
                        x
                    } else if x == y {
                        <$e>::from_bits(x.to_bits() | y.to_bits())
                    } else if x < y {
                        x
                    } else {
                        y
                    }
                }
                #[cfg(not(feature = "strict_ieee754"))]
                {
                    if x < y { x } else { y }
                }
            },
            Tol::ExactOrNan
        );
        bint!(
            concat!($l, " [max]"),
            V,
            $e,
            |a, b| a.max(b),
            |x: $e, y: $e| {
                #[cfg(feature = "strict_ieee754")]
                {
                    if y != y {
                        x
                    } else if x == y {
                        <$e>::from_bits(x.to_bits() & y.to_bits())
                    } else if x > y {
                        x
                    } else {
                        y
                    }
                }
                #[cfg(not(feature = "strict_ieee754"))]
                {
                    if x > y { x } else { y }
                }
            },
            Tol::ExactOrNan
        );

        // mix(t; a, b) = a*(1-t) + b*t, on finite inputs.
        let mut rng = harness::rng();
        for _ in 0..300 {
            let t: [$e; 3] = core::array::from_fn(|_| rng.random_range(0.0 as $e..1.0 as $e));
            let pa: [$e; 3] = core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e));
            let pb: [$e; 3] = core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e));
            let g = V::from_slice(&t)
                .mix(V::from_slice(&pa), V::from_slice(&pb))
                .into_array()
                .as_slice()[..3]
                .to_vec();
            let w: Vec<$e> = (0..3).map(|i| pa[i] * (1.0 as $e - t[i]) + pb[i] * t[i]).collect();
            harness::assert_lanes_eq(concat!($l, " [mix]"), &[&t], &g, &w, $tol);
        }
    }};
}

/// CastRegister (`f32<->i32`, in-range) + BitCastRegister (`into_bits`). `cast`
/// and `into_bits` are parametrised by the *target vector* type.
macro_rules! castt {
    ($freg:ty, $ireg:ty, $ureg:ty, $l:expr) => {{
        type VF = Vector<$freg>;
        type VI = Vector<$ireg>;
        type VU = Vector<$ureg>;
        let mut rng = harness::rng();
        // bit reinterpret f32 -> u32: exact over the full corpus.
        for x in harness::corpus::<f32>(3, &mut rng).into_iter().take(300) {
            let bits = VF::from_slice(&x).into_bits::<VU>().into_array().as_slice()[..3].to_vec();
            for i in 0..3 {
                assert_eq!(bits[i], x[i].to_bits(), "{} into_bits lane {}", $l, i);
            }
        }
        // numeric cast i32 -> f32 (exactly representable small ints).
        for _ in 0..300 {
            let xi: [i32; 3] = core::array::from_fn(|_| rng.random_range(-100_000..100_000));
            let f = VI::from_slice(&xi).cast::<VF>().into_array().as_slice()[..3].to_vec();
            for i in 0..3 {
                assert_eq!(f[i], xi[i] as f32, "{} cast i32->f32 lane {}", $l, i);
            }
        }
        // numeric cast f32 -> i32 (in-range, truncating toward zero).
        for _ in 0..300 {
            let xf: [f32; 3] = core::array::from_fn(|_| rng.random_range(-1.0e6_f32..1.0e6_f32));
            let g = VF::from_slice(&xf).cast::<VI>().into_array().as_slice()[..3].to_vec();
            for i in 0..3 {
                assert_eq!(g[i], xf[i] as i32, "{} cast f32->i32 lane {}", $l, i);
            }
        }
    }};
}

/// LinAlg3Register: dot3, cross3 (both DOP), zero4/one4, the four element3
/// reductions, mat3_transpose, and column-major mat3_vec3_product.
macro_rules! linalg3 {
    ($reg:ty, $e:ty, $l:expr, $tol:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let r3 = |v: V| v.into_array().as_slice()[..3].to_vec();

        for _ in 0..300 {
            let xa: [$e; 3] = core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e));
            let xb: [$e; 3] = core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e));
            let a = V::from_slice(&xa);
            let b = V::from_slice(&xb);

            let dot = xa[0] * xb[0] + xa[1] * xb[1] + xa[2] * xb[2];
            harness::assert_lanes_eq(concat!($l, " [dot3]"), &[&xa, &xb], &[a.dot3(b)], &[dot], $tol);

            let cross = [
                xa[1] * xb[2] - xa[2] * xb[1],
                xa[2] * xb[0] - xa[0] * xb[2],
                xa[0] * xb[1] - xa[1] * xb[0],
            ];
            harness::assert_lanes_eq(concat!($l, " [cross3 F]"), &[&xa, &xb], &r3(a.cross3::<false>(b)), &cross, $tol);
            harness::assert_lanes_eq(concat!($l, " [cross3 T]"), &[&xa, &xb], &r3(a.cross3::<true>(b)), &cross, $tol);

            // On a genuine 3-lane register zero4/one4 leave the visible lanes alone.
            harness::assert_lanes_eq(concat!($l, " [zero4]"), &[&xa], &r3(a.zero4()), &xa, $tol);
            harness::assert_lanes_eq(concat!($l, " [one4]"), &[&xa], &r3(a.one4()), &xa, $tol);

            let omin = xa[0].min(xa[1]).min(xa[2]);
            let omax = xa[0].max(xa[1]).max(xa[2]);
            harness::assert_lanes_eq(concat!($l, " [min_element3]"), &[&xa], &[a.min_element3()], &[omin], Tol::Exact);
            harness::assert_lanes_eq(concat!($l, " [max_element3]"), &[&xa], &[a.max_element3()], &[omax], Tol::Exact);
            harness::assert_lanes_eq(concat!($l, " [sum_elements3]"), &[&xa], &[a.sum_elements3()], &[xa[0] + xa[1] + xa[2]], $tol);
            harness::assert_lanes_eq(concat!($l, " [prod_elements3]"), &[&xa], &[a.prod_elements3()], &[xa[0] * xa[1] * xa[2]], $tol);
        }

        for _ in 0..200 {
            let cm: [[$e; 3]; 3] = core::array::from_fn(|_| core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e)));
            let cols = [V::from_slice(&cm[0]), V::from_slice(&cm[1]), V::from_slice(&cm[2])];

            // transpose: out column i, row j = cm[j][i].
            let t = V::mat3_transpose(&cols);
            for i in 0..3 {
                let want: [$e; 3] = core::array::from_fn(|j| cm[j][i]);
                harness::assert_lanes_eq(concat!($l, " [mat3_transpose]"), &[&cm[0], &cm[1], &cm[2]], &r3(t[i]), &want, $tol);
            }

            // column-major M*v = v.x*c0 + v.y*c1 + v.z*c2.
            let xv: [$e; 3] = core::array::from_fn(|_| rng.random_range(-8.0 as $e..8.0 as $e));
            let prod = V::from_slice(&xv).mat3_vec3_product::<true>(&cols);
            let want: [$e; 3] = core::array::from_fn(|row| xv[0] * cm[0][row] + xv[1] * cm[1][row] + xv[2] * cm[2][row]);
            harness::assert_lanes_eq(concat!($l, " [mat3_vec3_product]"), &[&xv], &r3(prod), &want, $tol);
        }
    }};
}

/// IntegerRegister division by a constant `Divider` (`divv_branchfree`). Correct
/// on every backend/type.
macro_rules! rdiv_const {
    ($reg:ty, $e:ty, $signed:literal, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let r3 = |v: V| v.into_array().as_slice()[..3].to_vec();

        let mut consts: Vec<$e> = vec![2, 3, 4, 7, 8, 16, 100];
        if $signed {
            consts.push((3 as $e).wrapping_neg());
            consts.push((8 as $e).wrapping_neg());
        }
        for &d in &consts {
            let dv = d.to_divider();
            for _ in 0..16 {
                let a: [$e; 3] = core::array::from_fn(|_| rng.random());
                let got = r3(V::from_slice(&a) / dv);
                for i in 0..3 {
                    assert_eq!(got[i], a[i].wrapping_div(d), "{} vec/Divider {} / {}", $l, a[i], d);
                }
            }
        }
    }};
}

/// IntegerRegister per-lane `VectorDivider` + masked `div_c`/`div_m`/`div_z`.
///
/// Exercises the per-lane variable-shift path (`shlv`/`srav`/`shrv`), which on
/// SSE has no native 64-bit form and is polyfilled, where a lane-swap bug corrupts
/// to corrupt 64-bit results on x86_v1/v2 (fixed in `_mm_s{ll,rl}v_epi64x_v1`).
macro_rules! rdiv_vec {
    ($reg:ty, $e:ty, $signed:literal, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let r3 = |v: V| v.into_array().as_slice()[..3].to_vec();

        for _ in 0..64 {
            let a: [$e; 3] = core::array::from_fn(|_| rng.random());
            let den: [$e; 3] = core::array::from_fn(|_| {
                let mut d: $e = rng.random();
                // Exclude 0, unsigned-1 (branchfree-unsupported), and signed -1
                // (the MIN/-1 quotient overflows Rust's `/`, an undefined corner).
                while d == 0 || (!$signed && d == 1) || ($signed && d == (1 as $e).wrapping_neg()) {
                    d = rng.random();
                }
                d
            });
            let vnum = V::from_slice(&a);
            let vdiv = V::from_slice(&den).to_divider();
            let plain: Vec<$e> = (0..3).map(|i| a[i].wrapping_div(den[i])).collect();
            assert_eq!(r3(vnum / vdiv), plain, "{} vec/VectorDivider", $l);

            let src = V::splat(123 as $e);
            let mask = vnum.cmp_lt(V::from_slice(&den));
            let mb: Vec<bool> = (0..3).map(|i| a[i] < den[i]).collect();
            let sel = |t: &[$e], f: &[$e]| -> Vec<$e> { (0..3).map(|i| if mb[i] { t[i] } else { f[i] }).collect() };
            assert_eq!(r3(vnum.div_c(mask, vdiv)), sel(&plain, &a), "{} div_c", $l);
            assert_eq!(r3(vnum.div_m(src, mask, vdiv)), sel(&plain, &r3(src)), "{} div_m", $l);
            assert_eq!(r3(vnum.div_z(mask, vdiv)), sel(&plain, &[0 as $e; 3]), "{} div_z", $l);
        }
    }};
}

/// UnsignedIntegerRegister::is_power_of_two on nonzero inputs.
macro_rules! pow2t {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        for _ in 0..400 {
            let x: [$e; 3] = core::array::from_fn(|_| {
                let v: $e = rng.random();
                if v == 0 { 1 } else { v }
            });
            let got = mbools!(V, $e, V::from_slice(&x).is_power_of_two());
            for i in 0..3 {
                assert_eq!(got[i], x[i].is_power_of_two(), "{} is_power_of_two lane {}", $l, i);
            }
        }
    }};
}

/// Integer masked arithmetic: mul_c/_m/_z, sub_z, plus the zz/nz blends.
macro_rules! maskedi {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let xs = harness::corpus::<$e>(3, &mut rng);
        let ys = harness::corpus::<$e>(3, &mut rng);
        let r3 = |v: V| v.into_array().as_slice()[..3].to_vec();
        for (x, y) in xs.iter().zip(ys.iter()).take(300) {
            let a = V::from_slice(x);
            let b = V::from_slice(y);
            let src = V::splat(99 as $e);
            let m = a.cmp_lt(b);
            let mb: Vec<bool> = (0..3).map(|i| x[i] < y[i]).collect();
            let mul = |i: usize| x[i].wrapping_mul(y[i]);
            let pick = |t: &dyn Fn(usize) -> $e, f: &dyn Fn(usize) -> $e| -> Vec<$e> {
                (0..3).map(|i| if mb[i] { t(i) } else { f(i) }).collect()
            };
            assert_eq!(r3(a.mul_c(m, b)), pick(&mul, &|i| x[i]), "{} mul_c", $l);
            assert_eq!(r3(a.mul_m(src, m, b)), pick(&mul, &|_| 99 as $e), "{} mul_m", $l);
            assert_eq!(r3(a.mul_z(m, b)), pick(&mul, &|_| 0 as $e), "{} mul_z", $l);
            assert_eq!(r3(a.sub_z(m, b)), pick(&|i| x[i].wrapping_sub(y[i]), &|_| 0 as $e), "{} sub_z", $l);
            assert_eq!(r3(a.zz(m)), pick(&|i| x[i], &|_| 0 as $e), "{} zz", $l);
            assert_eq!(r3(a.nz(m)), pick(&|_| 0 as $e, &|i| x[i]), "{} nz", $l);
        }
    }};
}

/// Float masked ops: sqrt_c/sqrt_z (conditional) plus zz/nz.
macro_rules! maskedf {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let xs = harness::corpus::<$e>(3, &mut rng);
        let ys = harness::corpus::<$e>(3, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()).take(300) {
            let a = V::from_slice(x);
            let m = a.cmp_lt(V::from_slice(y));
            let mb: Vec<bool> = (0..3).map(|i| x[i] < y[i]).collect();
            let read = |v: V| v.into_array().as_slice()[..3].to_vec();

            let g = read(a.sqrt_c(m));
            let w: Vec<$e> = (0..3).map(|i| if mb[i] { x[i].sqrt() } else { x[i] }).collect();
            harness::assert_lanes_eq(concat!($l, " [sqrt_c]"), &[x.as_slice()], &g, &w, Tol::Exact);

            let g = read(a.sqrt_z(m));
            let w: Vec<$e> = (0..3).map(|i| if mb[i] { x[i].sqrt() } else { 0 as $e }).collect();
            harness::assert_lanes_eq(concat!($l, " [sqrt_z]"), &[x.as_slice()], &g, &w, Tol::Exact);

            let g = read(a.zz(m));
            let w: Vec<$e> = (0..3).map(|i| if mb[i] { x[i] } else { 0 as $e }).collect();
            harness::assert_lanes_eq(concat!($l, " [zz]"), &[x.as_slice()], &g, &w, Tol::Exact);

            let g = read(a.nz(m));
            let w: Vec<$e> = (0..3).map(|i| if mb[i] { 0 as $e } else { x[i] }).collect();
            harness::assert_lanes_eq(concat!($l, " [nz]"), &[x.as_slice()], &g, &w, Tol::Exact);
        }
    }};
}

/// indexed / is_all_zero / is_zero.
macro_rules! numx {
    ($reg:ty, $e:ty, $l:expr) => {{
        type V = Vector<$reg>;
        let idx = V::indexed().into_array().as_slice()[..3].to_vec();
        assert_eq!(idx, vec![0 as $e, 1 as $e, 2 as $e], "{} indexed", $l);

        assert!(V::ZERO.is_all_zero(), "{} is_all_zero(ZERO)", $l);
        assert!(!V::splat(1 as $e).is_all_zero(), "{} is_all_zero(ones)", $l);

        let got = mbools!(V, $e, V::from_slice(&[0 as $e, 5 as $e, 0 as $e]).is_zero());
        assert_eq!(got, [true, false, true], "{} is_zero", $l);
    }};
}

/// Runtime `permute` (`Register::permutev`): result[i] = input[idx[i]],
/// checked against a Rust oracle over a few index patterns.
macro_rules! sw_permute {
    ($reg:ty, $e:ty, $l:expr) => {{
        use thermite::swizzle::Swizzle;
        type V = Vector<$reg>;
        let x = [10 as $e, 20 as $e, 30 as $e];
        let v = V::from_slice(&x);
        for p in [[2u32, 1, 0], [0, 0, 0], [1, 2, 0], [2, 0, 1]] {
            let idx = generic_array::GenericArray::<u32, generic_array::typenum::U3>::from(p);
            let got = v.permute(idx).into_array().as_slice()[..3].to_vec();
            let want: Vec<$e> = (0..3).map(|i| x[p[i] as usize]).collect();
            assert_eq!(got, want, "{} permute {:?}", $l, p);
        }
    }};
}

/// Two-input `swizzle` (`Register::swizzle`), differenced against the
/// scalar reduced register so the index contract is whatever the trait defines.
macro_rules! sw_swizzle {
    ($reg_ut:ty, $reg_sc:ty, $e:ty, $l:expr) => {{
        use thermite::swizzle::Swizzle;
        type VU = Vector<$reg_ut>;
        type VS = Vector<$reg_sc>;
        let a = [10 as $e, 20 as $e, 30 as $e];
        let b = [40 as $e, 50 as $e, 60 as $e];
        for p in [[0u32, 3, 1], [3, 4, 5], [2, 5, 0], [0, 1, 2]] {
            let iu = generic_array::GenericArray::<u32, generic_array::typenum::U3>::from(p);
            let is = generic_array::GenericArray::<u32, generic_array::typenum::U3>::from(p);
            let gu = VU::from_slice(&a).swizzle(VU::from_slice(&b), iu).into_array().as_slice()[..3].to_vec();
            let gs = VS::from_slice(&a).swizzle(VS::from_slice(&b), is).into_array().as_slice()[..3].to_vec();
            assert_eq!(gu, gs, "{} swizzle {:?}", $l, p);
        }
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    reduced_suite!(v3, X86V3, "x86_v3");
    reduced_suite!(v2, X86V2, "x86_v2");
    reduced_suite!(v1, X86V1, "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    reduced_suite!(wasm, Wasm, "wasm");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    reduced_suite!(neon, Neon, "neon");
}
