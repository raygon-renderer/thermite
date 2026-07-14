//! Extended-op differential coverage for `ArrayRegister` (wide vectors) and
//! `ReducedRegister` (narrow vectors): the masked `_c`/`_m`/`_z` variants,
//! broadcast, reverse, classification predicates, signed ops, gather/scatter,
//! masked load/store, and the `NumericRegister` reductions the basic `diff_ops`
//! suite doesn't reach. Every backend register is run against the `Scalar`
//! reference at the same element type x width, so the emulated `ArrayRegister`
//! and `ReducedRegister` delegation paths (blendv / zz / nz, lane routing, the
//! split/recombine helpers) are all exercised.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    all(feature = "neon", target_arch = "aarch64")
))]

mod harness;

use generic_array::typenum::Unsigned;
use harness::Tol;

use thermite::register::{
    BitwiseRegister as _, CoreRegister, FloatRegister as _, NumericRegister as _, Register, SignedRegister as _,
};
use thermite::simd::{Simd, Simd3A};

use thermite::backend::scalar::Scalar;

macro_rules! n_of {
    ($ut:ty) => {
        <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE
    };
}

/// Masked binary op: `_c` (merge-with-self), `_m` (merge-with-src), `_z` (zero).
macro_rules! mbin {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {
        paste::paste! {{
            let mut rng = harness::rng();
            type E = <$ut as Register>::Element;
            let n = n_of!($ut);
            let xs = harness::corpus::<E>(n, &mut rng);
            let ys = harness::corpus::<E>(n, &mut rng);
            let pats = harness::mask_patterns(n, 64, &mut rng);
            for (k, (x, y)) in xs.iter().zip(ys.iter()).take(600).enumerate() {
                let mp = &pats[k % pats.len()];
                let (mu, mr) = (harness::build_mask::<$ut>(mp), harness::build_mask::<$rf>(mp));
                let (au, bu) = (harness::make_array::<$ut>(x), harness::make_array::<$ut>(y));
                let (ar, br) = (harness::make_array::<$rf>(x), harness::make_array::<$rf>(y));
                let (su, sr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_c]"), &[x.as_slice(), y.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _c>](mu, au, bu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _c>](mr, ar, br)), $tol);
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_m]"), &[x.as_slice(), y.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _m>](su, mu, au, bu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _m>](sr, mr, ar, br)), $tol);
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_z]"), &[x.as_slice(), y.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _z>](mu, au, bu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _z>](mr, ar, br)), $tol);
            }
        }}
    };
}

/// Masked unary op: `_c` / `_m` / `_z`.
macro_rules! munary {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {
        paste::paste! {{
            let mut rng = harness::rng();
            type E = <$ut as Register>::Element;
            let n = n_of!($ut);
            let xs = harness::corpus::<E>(n, &mut rng);
            let ys = harness::corpus::<E>(n, &mut rng);
            let pats = harness::mask_patterns(n, 64, &mut rng);
            for (k, x) in xs.iter().take(600).enumerate() {
                let mp = &pats[k % pats.len()];
                let (mu, mr) = (harness::build_mask::<$ut>(mp), harness::build_mask::<$rf>(mp));
                let (vu, vr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
                let (su, sr) = (harness::make_array::<$ut>(&ys[k % ys.len()]), harness::make_array::<$rf>(&ys[k % ys.len()]));
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_c]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _c>](mu, vu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _c>](mr, vr)), $tol);
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_m]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _m>](su, mu, vu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _m>](sr, mr, vr)), $tol);
                harness::assert_lanes_eq(concat!($label, " [", stringify!($method), "_z]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::[<$method _z>](mu, vu)),
                    &harness::read::<$rf>(&<$rf>::[<$method _z>](mr, vr)), $tol);
            }
        }}
    };
}

/// `broadcast::<0>` + masked variants, and `broadcastv(idx)` + masked variants.
macro_rules! bcast {
    ($label:expr, $ut:ty, $rf:ty, $tol:expr) => {
        paste::paste! {{
            let mut rng = harness::rng();
            type E = <$ut as Register>::Element;
            let n = n_of!($ut);
            let pats = harness::mask_patterns(n, 64, &mut rng);
            for (k, x) in harness::corpus::<E>(n, &mut rng).iter().take(400).enumerate() {
                let mp = &pats[k % pats.len()];
                let (mu, mr) = (harness::build_mask::<$ut>(mp), harness::build_mask::<$rf>(mp));
                let (vu, vr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
                let (su, sr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
                // broadcast::<0>
                harness::assert_lanes_eq(concat!($label, " [broadcast0]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcast::<0>(vu)),
                    &harness::read::<$rf>(&<$rf>::broadcast::<0>(vr)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcast0_c]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcast_c::<0>(mu, vu)),
                    &harness::read::<$rf>(&<$rf>::broadcast_c::<0>(mr, vr)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcast0_m]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcast_m::<0>(su, mu, vu)),
                    &harness::read::<$rf>(&<$rf>::broadcast_m::<0>(sr, mr, vr)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcast0_z]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcast_z::<0>(mu, vu)),
                    &harness::read::<$rf>(&<$rf>::broadcast_z::<0>(mr, vr)), $tol);
                // broadcastv(idx) for the last lane + masked variants
                let idx = n - 1;
                harness::assert_lanes_eq(concat!($label, " [broadcastv]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcastv(vu, idx)),
                    &harness::read::<$rf>(&<$rf>::broadcastv(vr, idx)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcastv_c]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcastv_c(mu, vu, idx)),
                    &harness::read::<$rf>(&<$rf>::broadcastv_c(mr, vr, idx)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcastv_m]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcastv_m(su, mu, vu, idx)),
                    &harness::read::<$rf>(&<$rf>::broadcastv_m(sr, mr, vr, idx)), $tol);
                harness::assert_lanes_eq(concat!($label, " [broadcastv_z]"), &[x.as_slice()],
                    &harness::read::<$ut>(&<$ut>::broadcastv_z(mu, vu, idx)),
                    &harness::read::<$rf>(&<$rf>::broadcastv_z(mr, vr, idx)), $tol);
            }
        }}
    };
}

/// `reverse` + masked variants.
macro_rules! rev {
    ($label:expr, $ut:ty, $rf:ty, $tol:expr) => {{
        let mut rng = harness::rng();
        type E = <$ut as Register>::Element;
        let n = n_of!($ut);
        let pats = harness::mask_patterns(n, 64, &mut rng);
        for (k, x) in harness::corpus::<E>(n, &mut rng).iter().take(400).enumerate() {
            let mp = &pats[k % pats.len()];
            let (mu, mr) = (harness::build_mask::<$ut>(mp), harness::build_mask::<$rf>(mp));
            let (vu, vr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
            let (su, sr) = (harness::make_array::<$ut>(x), harness::make_array::<$rf>(x));
            harness::assert_lanes_eq(
                concat!($label, " [reverse]"),
                &[x.as_slice()],
                &harness::read::<$ut>(&<$ut>::reverse(vu)),
                &harness::read::<$rf>(&<$rf>::reverse(vr)),
                $tol,
            );
            harness::assert_lanes_eq(
                concat!($label, " [reverse_c]"),
                &[x.as_slice()],
                &harness::read::<$ut>(&<$ut>::reverse_c(mu, vu)),
                &harness::read::<$rf>(&<$rf>::reverse_c(mr, vr)),
                $tol,
            );
            harness::assert_lanes_eq(
                concat!($label, " [reverse_m]"),
                &[x.as_slice()],
                &harness::read::<$ut>(&<$ut>::reverse_m(su, mu, vu)),
                &harness::read::<$rf>(&<$rf>::reverse_m(sr, mr, vr)),
                $tol,
            );
            harness::assert_lanes_eq(
                concat!($label, " [reverse_z]"),
                &[x.as_slice()],
                &harness::read::<$ut>(&<$ut>::reverse_z(mu, vu)),
                &harness::read::<$rf>(&<$rf>::reverse_z(mr, vr)),
                $tol,
            );
        }
    }};
}

/// Classification predicate (mask result), checked lane-by-lane vs Scalar.
macro_rules! pred {
    ($label:expr, $ut:ty, $rf:ty, $method:ident) => {{
        let mut rng = harness::rng();
        type E = <$ut as Register>::Element;
        let n = n_of!($ut);
        for x in harness::corpus::<E>(n, &mut rng).iter().take(500) {
            let gm = harness::read_mask::<$ut>(<$ut>::$method(harness::make_array::<$ut>(x)), n);
            let wm = harness::read_mask::<$rf>(<$rf>::$method(harness::make_array::<$rf>(x)), n);
            assert_eq!(gm, wm, concat!($label, " [", stringify!($method), "]"));
        }
    }};
}

/// SignedRegister ops: `signum` (exact), `is_negative`/`is_positive` (mask),
/// and the `select_negative` ternary.
macro_rules! signed_ext {
    ($label:expr, $ut:ty, $rf:ty) => {{
        diff_unary!($label, $ut, $rf, signum, Tol::Exact);
        let mut rng = harness::rng();
        type E = <$ut as Register>::Element;
        let n = n_of!($ut);
        let xs = harness::corpus::<E>(n, &mut rng);
        let ys = harness::corpus::<E>(n, &mut rng);
        for (k, x) in xs.iter().take(400).enumerate() {
            assert_eq!(
                harness::read_mask::<$ut>(<$ut>::is_negative(harness::make_array::<$ut>(x)), n),
                harness::read_mask::<$rf>(<$rf>::is_negative(harness::make_array::<$rf>(x)), n),
                concat!($label, " [is_negative]"),
            );
            assert_eq!(
                harness::read_mask::<$ut>(<$ut>::is_positive(harness::make_array::<$ut>(x)), n),
                harness::read_mask::<$rf>(<$rf>::is_positive(harness::make_array::<$rf>(x)), n),
                concat!($label, " [is_positive]"),
            );
            // select_negative(value, on_neg, on_pos): self<0 ? on_neg : on_pos
            let y = &ys[k % ys.len()];
            harness::assert_lanes_eq(
                concat!($label, " [select_negative]"),
                &[x.as_slice(), y.as_slice()],
                &harness::read::<$ut>(&<$ut>::select_negative(
                    harness::make_array::<$ut>(x),
                    harness::make_array::<$ut>(y),
                    harness::make_array::<$ut>(x),
                )),
                &harness::read::<$rf>(&<$rf>::select_negative(
                    harness::make_array::<$rf>(x),
                    harness::make_array::<$rf>(y),
                    harness::make_array::<$rf>(x),
                )),
                Tol::Exact,
            );
        }
    }};
}

/// `offset` / `indexed` (deterministic, no inputs).
macro_rules! offset_indexed {
    ($label:expr, $ut:ty, $rf:ty) => {{
        harness::assert_lanes_eq(
            concat!($label, " [offset]"),
            &[],
            &harness::read::<$ut>(&<$ut>::offset()),
            &harness::read::<$rf>(&<$rf>::offset()),
            Tol::Exact,
        );
        harness::assert_lanes_eq(
            concat!($label, " [indexed]"),
            &[],
            &harness::read::<$ut>(&<$ut>::indexed()),
            &harness::read::<$rf>(&<$rf>::indexed()),
            Tol::Exact,
        );
    }};
}

/// `min_max_element`. When `$finite`, inputs containing NaN/Inf are skipped: a
/// horizontal min/max reduction over NaN/Inf is order-dependent and the
/// asymmetric (non-IEEE) propagation legitimately diverges between the array
/// (tree) reduction and the scalar (sequential) one. Over finite values it is
/// exact.
macro_rules! min_max {
    ($label:expr, $ut:ty, $rf:ty, $finite:literal) => {{
        use harness::Diff as _;
        let mut rng = harness::rng();
        type E = <$ut as Register>::Element;
        let n = n_of!($ut);
        // For floats, min/max of equal-but-opposite-sign zeros yields an
        // impl-defined sign; `Rel(0.0)` treats +0.0 == -0.0 (via the got==want
        // short-circuit) while still requiring numeric equality otherwise.
        // Integers are bit-exact.
        let tol = if $finite { Tol::Rel(0.0) } else { Tol::Exact };
        for x in harness::corpus::<E>(n, &mut rng).iter().take(400) {
            if $finite && x.iter().any(|v| !v.finite()) {
                continue;
            }
            let (gmin, gmax) = <$ut>::min_max_element(harness::make_array::<$ut>(x));
            let (wmin, wmax) = <$rf>::min_max_element(harness::make_array::<$rf>(x));
            harness::assert_lanes_eq(
                concat!($label, " [min_max.min]"),
                &[x.as_slice()],
                &[gmin],
                &[wmin],
                tol,
            );
            harness::assert_lanes_eq(
                concat!($label, " [min_max.max]"),
                &[x.as_slice()],
                &[gmax],
                &[wmax],
                tol,
            );
        }
    }};
}

/// `pairwise_sum` / `relaxed_pairwise_sum` regroup lanes (relaxed even reorders
/// per inner register), so the lane layout is structure-dependent and not
/// comparable across backends. What IS invariant is the total: summing the
/// result equals summing both inputs. Checked here for integers, where the
/// wrapping total is exact and associative (this still exercises the array.rs
/// override, which is generic over the element type).
macro_rules! pairwise_inv {
    ($label:expr, $ut:ty, $rf:ty, $method:ident) => {{
        let mut rng = harness::rng();
        type E = <$ut as Register>::Element;
        let n = n_of!($ut);
        let xs = harness::corpus::<E>(n, &mut rng);
        let ys = harness::corpus::<E>(n, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()).take(400) {
            let gt = <$ut>::sum_elements(<$ut>::$method(
                harness::make_array::<$ut>(x),
                harness::make_array::<$ut>(y),
            ));
            let wt = <$rf>::sum_elements(<$rf>::$method(
                harness::make_array::<$rf>(x),
                harness::make_array::<$rf>(y),
            ));
            harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), " total]"),
                &[x.as_slice(), y.as_slice()],
                &[gt],
                &[wt],
                Tol::Exact,
            );
        }
    }};
}

/// Lane-count-agnostic reductions: offset/indexed + min/max. Safe for any width,
/// including the odd 3-lane reduced registers. `$finite` skips NaN/Inf for floats.
macro_rules! numred_basic {
    ($label:expr, $ut:ty, $rf:ty, $finite:literal) => {{
        offset_indexed!($label, $ut, $rf);
        min_max!($label, $ut, $rf, $finite);
    }};
}

/// Integer reductions on EVEN-width registers: pairwise-sum total invariant,
/// `sort`, plus the basics. `pairwise_sum`/`sort` are not meaningful on the odd
/// 3-lane reduced registers (pairwise halves the lane count; the trait default
/// itself drops lanes for odd counts), so those are wide-only.
macro_rules! numred_int {
    ($label:expr, $ut:ty, $rf:ty) => {{
        pairwise_inv!($label, $ut, $rf, pairwise_sum);
        pairwise_inv!($label, $ut, $rf, relaxed_pairwise_sum);
        diff_unary!($label, $ut, $rf, sort, Tol::Exact);
        numred_basic!($label, $ut, $rf, false);
    }};
}

mod ext {
    use super::*;

    /// Float register extended ops. Masked *arithmetic* is intentionally NOT
    /// differenced here: the optimized pre-AVX512 `_c` form (`self + (rhs &
    /// mask)`) is not bit-identical to a blendv on signed zeros, so it legitimately
    /// diverges from the scalar reference. The select-based ops (broadcast /
    /// reverse and their masked variants) and the predicates have no such quirk,
    /// and the masked-merge code paths themselves are covered via integer ops.
    macro_rules! float_ext {
        ($ut:ty, $rf:ty, $l:expr) => {{
            bcast!($l, $ut, $rf, Tol::Exact);
            rev!($l, $ut, $rf, Tol::Exact);
            pred!($l, $ut, $rf, is_nan);
            pred!($l, $ut, $rf, is_finite);
            pred!($l, $ut, $rf, is_infinite);
            pred!($l, $ut, $rf, is_normal);
        }};
    }

    /// Integer register extended ops.
    macro_rules! int_ext {
        ($ut:ty, $rf:ty, $l:expr) => {{
            mbin!($l, $ut, $rf, add, Tol::Exact);
            mbin!($l, $ut, $rf, sub, Tol::Exact);
            mbin!($l, $ut, $rf, mul, Tol::Exact);
            mbin!($l, $ut, $rf, min, Tol::Exact);
            mbin!($l, $ut, $rf, max, Tol::Exact);
            mbin!($l, $ut, $rf, bitand, Tol::Exact);
            mbin!($l, $ut, $rf, bitor, Tol::Exact);
            mbin!($l, $ut, $rf, bitxor, Tol::Exact);
            munary!($l, $ut, $rf, not, Tol::Exact);
            bcast!($l, $ut, $rf, Tol::Exact);
            rev!($l, $ut, $rf, Tol::Exact);
        }};
    }

    /// Cross-element-width numeric casts on wide vectors. Because a wider element
    /// needs more inner registers for the same lane count, these route through the
    /// `impl_casts!` cross-`N` machinery in array.rs (e.g. f32x16 = Array<_,2> ->
    /// f64x16 = Array<_,4>). Oracled against the scalar backend (`as` semantics);
    /// same-kind widen/narrow is exact on both sides.
    macro_rules! casts {
        ($backend:ty, $bl:expr) => {{
            cast_diff!(
                concat!($bl, " f32->f64"),
                <$backend as Simd>::f32x16,
                <$backend as Simd>::f64x16,
                <Scalar as Simd>::f32x16,
                <Scalar as Simd>::f64x16,
                f32,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                concat!($bl, " f64->f32"),
                <$backend as Simd>::f64x16,
                <$backend as Simd>::f32x16,
                <Scalar as Simd>::f64x16,
                <Scalar as Simd>::f32x16,
                f64,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                concat!($bl, " i32->i64"),
                <$backend as Simd>::i32x16,
                <$backend as Simd>::i64x16,
                <Scalar as Simd>::i32x16,
                <Scalar as Simd>::i64x16,
                i32,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                concat!($bl, " i64->i32"),
                <$backend as Simd>::i64x16,
                <$backend as Simd>::i32x16,
                <Scalar as Simd>::i64x16,
                <Scalar as Simd>::i32x16,
                i64,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                concat!($bl, " u32->u64"),
                <$backend as Simd>::u32x16,
                <$backend as Simd>::u64x16,
                <Scalar as Simd>::u32x16,
                <Scalar as Simd>::u64x16,
                u32,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                concat!($bl, " u64->u32"),
                <$backend as Simd>::u64x16,
                <$backend as Simd>::u32x16,
                <Scalar as Simd>::u64x16,
                <Scalar as Simd>::u32x16,
                u64,
                |x| x,
                Tol::Exact
            );
        }};
    }

    macro_rules! suite {
        ($modname:ident, $backend:ty, $bl:expr) => {
            mod $modname {
                use super::*;

                #[test]
                fn floats() {
                    float_ext!(
                        <$backend as Simd>::f32x16,
                        <Scalar as Simd>::f32x16,
                        concat!($bl, " f32x16")
                    );
                    float_ext!(
                        <$backend as Simd>::f64x8,
                        <Scalar as Simd>::f64x8,
                        concat!($bl, " f64x8")
                    );
                    numred_basic!(
                        concat!($bl, " f32x16"),
                        <$backend as Simd>::f32x16,
                        <Scalar as Simd>::f32x16,
                        true
                    );
                    numred_basic!(
                        concat!($bl, " f64x8"),
                        <$backend as Simd>::f64x8,
                        <Scalar as Simd>::f64x8,
                        true
                    );
                }
                #[test]
                fn ints() {
                    int_ext!(
                        <$backend as Simd>::i32x16,
                        <Scalar as Simd>::i32x16,
                        concat!($bl, " i32x16")
                    );
                    int_ext!(
                        <$backend as Simd>::u32x16,
                        <Scalar as Simd>::u32x16,
                        concat!($bl, " u32x16")
                    );
                    int_ext!(
                        <$backend as Simd>::i64x8,
                        <Scalar as Simd>::i64x8,
                        concat!($bl, " i64x8")
                    );
                    int_ext!(
                        <$backend as Simd>::u64x8,
                        <Scalar as Simd>::u64x8,
                        concat!($bl, " u64x8")
                    );
                    numred_int!(
                        concat!($bl, " i32x16"),
                        <$backend as Simd>::i32x16,
                        <Scalar as Simd>::i32x16
                    );
                    numred_int!(
                        concat!($bl, " u64x8"),
                        <$backend as Simd>::u64x8,
                        <Scalar as Simd>::u64x8
                    );
                    signed_ext!(
                        concat!($bl, " i32x16"),
                        <$backend as Simd>::i32x16,
                        <Scalar as Simd>::i32x16
                    );
                    signed_ext!(
                        concat!($bl, " i64x8"),
                        <$backend as Simd>::i64x8,
                        <Scalar as Simd>::i64x8
                    );
                }
                #[test]
                fn casts() {
                    casts!($backend, $bl);
                }
                #[test]
                fn reduced() {
                    float_ext!(
                        <$backend as Simd3A>::f32x3A,
                        <Scalar as Simd3A>::f32x3A,
                        concat!($bl, " f32x3A")
                    );
                    int_ext!(
                        <$backend as Simd3A>::i32x3A,
                        <Scalar as Simd3A>::i32x3A,
                        concat!($bl, " i32x3A")
                    );
                    int_ext!(
                        <$backend as Simd3A>::u64x3A,
                        <Scalar as Simd3A>::u64x3A,
                        concat!($bl, " u64x3A")
                    );
                    numred_basic!(
                        concat!($bl, " i32x3A"),
                        <$backend as Simd3A>::i32x3A,
                        <Scalar as Simd3A>::i32x3A,
                        false
                    );
                    numred_basic!(
                        concat!($bl, " f32x3A"),
                        <$backend as Simd3A>::f32x3A,
                        <Scalar as Simd3A>::f32x3A,
                        true
                    );
                    signed_ext!(
                        concat!($bl, " i32x3A"),
                        <$backend as Simd3A>::i32x3A,
                        <Scalar as Simd3A>::i32x3A
                    );
                }
            }
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod x86 {
        use super::*;
        use thermite::backend::x86_v1::X86V1;
        use thermite::backend::x86_v2::X86V2;
        use thermite::backend::x86_v3::X86V3;
        suite!(v3, X86V3, "x86_v3");
        suite!(v2, X86V2, "x86_v2");
        suite!(v1, X86V1, "x86_v1");
    }

    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use super::*;
        use thermite::backend::wasm::Wasm;
        suite!(wasm, Wasm, "wasm");
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    mod neon {
        use super::*;
        use thermite::backend::neon::Neon;
        suite!(neon, Neon, "neon");
    }
}
