//! Mask, comparison, select, and masked-variant (`_c`/`_m`/`_z`) coverage.
//!
//! This is the foundational gap the other differential suites left open
//! (`diff_ops` tests the *unmasked* register ops; nothing tested the masks
//! those ops blend against). Everything here is checked against an
//! **independent pure-Rust oracle**, not the scalar backend, so a shared
//! blend/select bug can't hide:
//!
//!   - `eq`/`ne`/`lt`/`le`/`gt`/`ge`  vs Rust's comparison operators,
//!   - `blendv`/`zz`/`nz`             vs a per-lane Rust `select`,
//!   - every `op_c`/`op_m`/`op_z`     vs the backend's own *unmasked* `op`
//!     blended per a known mask (so this isolates the *masking* logic, which
//!     is exactly where the pre-AVX512 optimized `_c` forms and the byte-blend
//!     bugs from the polyfill audit live).
//!
//! The masks themselves are built from known boolean patterns via
//! `MaskRegister::new_mask` and read back with `MaskRegister::test`, so those
//! primitives are exercised too. Only built where the x86 SIMD backends exist.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use generic_array::typenum::Unsigned;

use harness::Tol;
use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, CoreRegister, FloatRegister as _, IntegerRegister as _,
    NumericRegister as _, PartialOrdRegister as _, Register, SignedIntegerRegister as _, SignedRegister as _,
    UnsignedIntegerRegister as _,
};
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

// ---------------------------------------------------------------------------
// Comparisons: backend mask vs Rust's comparison operators.
//
// NaN is handled correctly by the operators themselves (`a < NaN` is `false`,
// `a != NaN` is `true`), which matches the unordered SSE/AVX compare result -
// so Rust's operators are a faithful oracle even on NaN/Inf inputs.
// ---------------------------------------------------------------------------
macro_rules! cmp {
    ($label:expr, $ut:ty, $e:ty, $method:ident, $op:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let oracle: fn($e, $e) -> bool = $op;
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ys = harness::corpus::<$e>(lanes, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let m = <$ut>::$method(harness::make_array::<$ut>(x), harness::make_array::<$ut>(y));
            let got = harness::read_mask::<$ut>(m, lanes);
            for (lane, &g) in got.iter().enumerate() {
                let w = oracle(x[lane], y[lane]);
                assert!(
                    g == w,
                    concat!(
                        $label,
                        " [",
                        stringify!($method),
                        "]: lane {} mismatch\n  a = {:?}\n  b = {:?}\n  got = {}  want = {}"
                    ),
                    lane,
                    x[lane],
                    y[lane],
                    g,
                    w
                );
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// Select: blendv / zz / nz vs a per-lane Rust select.
// ---------------------------------------------------------------------------
macro_rules! select {
    ($label:expr, $ut:ty, $e:ty) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let fs = harness::corpus::<$e>(lanes, &mut rng);
        let ts = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, fs.len(), &mut rng);
        let zero = <$e as Default>::default();
        for ((f, t), bits) in fs.iter().zip(ts.iter()).zip(pats.iter()) {
            let mask = harness::build_mask::<$ut>(bits);
            // blendv(mask, on_false, on_true) == mask ? on_true : on_false
            let got = harness::read::<$ut>(&<$ut>::blendv(mask, harness::make_array::<$ut>(f), harness::make_array::<$ut>(t)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { t[i] } else { f[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [blendv]"), &[f.as_slice(), t.as_slice()], &got, &want, Tol::Exact);

            // zz(mask, value) == mask ? value : 0
            let got = harness::read::<$ut>(&<$ut>::zz(mask, harness::make_array::<$ut>(t)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { t[i] } else { zero }).collect();
            harness::assert_lanes_eq(concat!($label, " [zz]"), &[t.as_slice()], &got, &want, Tol::Exact);

            // nz(mask, value) == mask ? 0 : value
            let got = harness::read::<$ut>(&<$ut>::nz(mask, harness::make_array::<$ut>(t)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { zero } else { t[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [nz]"), &[t.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

// ---------------------------------------------------------------------------
// Masked binary op: op_c / op_m / op_z vs the backend's own unmasked `op`
// blended per a known mask. This isolates the masking, not the arithmetic.
//
//   op_c(mask, a, b) == mask ? op(a, b) : a        (else = first arg)
//   op_m(src, mask, a, b) == mask ? op(a, b) : src
//   op_z(mask, a, b) == mask ? op(a, b) : 0
// ---------------------------------------------------------------------------
macro_rules! masked_bin {
    ($label:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident, $tol:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ys = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        for (((x, y), s), bits) in xs.iter().zip(ys.iter()).zip(ss.iter()).zip(pats.iter()) {
            let mask = harness::build_mask::<$ut>(bits);
            let base = harness::read::<$ut>(&<$ut>::$base(harness::make_array::<$ut>(x), harness::make_array::<$ut>(y)));

            let got = harness::read::<$ut>(&<$ut>::$c(mask, harness::make_array::<$ut>(x), harness::make_array::<$ut>(y)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { x[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($c), "]"), &[x.as_slice(), y.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$m(harness::make_array::<$ut>(s), mask, harness::make_array::<$ut>(x), harness::make_array::<$ut>(y)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($m), "]"), &[x.as_slice(), y.as_slice(), s.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$z(mask, harness::make_array::<$ut>(x), harness::make_array::<$ut>(y)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($z), "]"), &[x.as_slice(), y.as_slice()], &got, &want, $tol);
        }
    }};
}

// Masked unary op: same idea, single operand. else for `_c` is the operand.
macro_rules! masked_un {
    ($label:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident, $tol:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let vs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, vs.len(), &mut rng);
        let zero = <$e as Default>::default();
        for ((v, s), bits) in vs.iter().zip(ss.iter()).zip(pats.iter()) {
            let mask = harness::build_mask::<$ut>(bits);
            let base = harness::read::<$ut>(&<$ut>::$base(harness::make_array::<$ut>(v)));

            let got = harness::read::<$ut>(&<$ut>::$c(mask, harness::make_array::<$ut>(v)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { v[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($c), "]"), &[v.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$m(harness::make_array::<$ut>(s), mask, harness::make_array::<$ut>(v)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($m), "]"), &[v.as_slice(), s.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$z(mask, harness::make_array::<$ut>(v)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($z), "]"), &[v.as_slice()], &got, &want, $tol);
        }
    }};
}

// Masked ternary op (FMA family): op_c/op_m/op_z vs the backend's own unmasked
// `op(a, b, c)` blended per the mask. else for `_c` is the first operand `a`.
macro_rules! masked_tern {
    ($label:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident, $tol:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ys = harness::corpus::<$e>(lanes, &mut rng);
        let zs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for ((((x, y), z), s), bits) in xs.iter().zip(ys.iter()).zip(zs.iter()).zip(ss.iter()).zip(pats.iter()) {
            let mask = harness::build_mask::<$ut>(bits);
            let base = harness::read::<$ut>(&<$ut>::$base(mk(x), mk(y), mk(z)));

            let got = harness::read::<$ut>(&<$ut>::$c(mask, mk(x), mk(y), mk(z)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { x[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($c), "]"), &[x.as_slice(), y.as_slice(), z.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$m(mk(s), mask, mk(x), mk(y), mk(z)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($m), "]"), &[x.as_slice(), y.as_slice(), z.as_slice()], &got, &want, $tol);

            let got = harness::read::<$ut>(&<$ut>::$z(mask, mk(x), mk(y), mk(z)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(concat!($label, " [", stringify!($z), "]"), &[x.as_slice(), y.as_slice(), z.as_slice()], &got, &want, $tol);
        }
    }};
}

// Masked shift/rotate op taking a scalar `u32` count: op_c/op_m/op_z vs the
// backend's own unmasked `op(value, count)` blended per the mask. A couple of
// representative counts (valid for every width) are enough to cover the variants.
macro_rules! masked_shift {
    ($label:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let vs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, vs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for &sh in &[1u32, 5u32] {
            for ((v, s), bits) in vs.iter().zip(ss.iter()).zip(pats.iter()) {
                let mask = harness::build_mask::<$ut>(bits);
                let base = harness::read::<$ut>(&<$ut>::$base(mk(v), sh));

                let got = harness::read::<$ut>(&<$ut>::$c(mask, mk(v), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { v[i] }).collect();
                harness::assert_lanes_eq(concat!($label, " [", stringify!($c), "]"), &[v.as_slice()], &got, &want, Tol::Exact);

                let got = harness::read::<$ut>(&<$ut>::$m(mk(s), mask, mk(v), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
                harness::assert_lanes_eq(concat!($label, " [", stringify!($m), "]"), &[v.as_slice(), s.as_slice()], &got, &want, Tol::Exact);

                let got = harness::read::<$ut>(&<$ut>::$z(mask, mk(v), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
                harness::assert_lanes_eq(concat!($label, " [", stringify!($z), "]"), &[v.as_slice()], &got, &want, Tol::Exact);
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// Shared blocks.
// ---------------------------------------------------------------------------
macro_rules! cmp_common {
    ($ut:ty, $e:ty, $label:expr) => {{
        cmp!($label, $ut, $e, eq, |a, b| a == b);
        cmp!($label, $ut, $e, ne, |a, b| a != b);
        cmp!($label, $ut, $e, lt, |a, b| a < b);
        cmp!($label, $ut, $e, le, |a, b| a <= b);
        cmp!($label, $ut, $e, gt, |a, b| a > b);
        cmp!($label, $ut, $e, ge, |a, b| a >= b);
    }};
}

// `$tol` is `Tol::Exact` for integers, `Tol::Rel(0.0)` for floats. The latter
// treats +0.0 and -0.0 as equal: the optimized `_c` form some backends use
// (`a + (b & mask)`) yields +0.0 in a masked-off lane where the operand was
// -0.0 (since `-0.0 + 0.0 == +0.0`). That signed-zero relaxation is the same
// one diff_ops applies to `min`/`max`; any larger divergence still fails.
macro_rules! numeric_common {
    ($ut:ty, $e:ty, $label:expr, $tol:expr) => {{
        cmp_common!($ut, $e, $label);
        select!($label, $ut, $e);
        masked_bin!($label, $ut, $e, add, add_c, add_m, add_z, $tol);
        masked_bin!($label, $ut, $e, sub, sub_c, sub_m, sub_z, $tol);
        masked_bin!($label, $ut, $e, mul, mul_c, mul_m, mul_z, $tol);
        masked_bin!($label, $ut, $e, min, min_c, min_m, min_z, $tol);
        masked_bin!($label, $ut, $e, max, max_c, max_m, max_z, $tol);
        masked_un!($label, $ut, $e, square, square_c, square_m, square_z, $tol);
    }};
}

// Shared integer-only masked ops (signed and unsigned).
macro_rules! int_extra_masked {
    ($ut:ty, $e:ty, $label:expr) => {{
        masked_bin!($label, $ut, $e, mulhi, mulhi_c, mulhi_m, mulhi_z, Tol::Exact);
        masked_bin!($label, $ut, $e, mullo, mullo_c, mullo_m, mullo_z, Tol::Exact);
        masked_bin!($label, $ut, $e, saturating_add, saturating_add_c, saturating_add_m, saturating_add_z, Tol::Exact);
        masked_bin!($label, $ut, $e, saturating_sub, saturating_sub_c, saturating_sub_m, saturating_sub_z, Tol::Exact);
        masked_un!($label, $ut, $e, count_ones, count_ones_c, count_ones_m, count_ones_z, Tol::Exact);
        masked_un!($label, $ut, $e, count_zeros, count_zeros_c, count_zeros_m, count_zeros_z, Tol::Exact);
        masked_un!($label, $ut, $e, leading_zeros, leading_zeros_c, leading_zeros_m, leading_zeros_z, Tol::Exact);
        masked_un!($label, $ut, $e, leading_ones, leading_ones_c, leading_ones_m, leading_ones_z, Tol::Exact);
        masked_un!($label, $ut, $e, trailing_zeros, trailing_zeros_c, trailing_zeros_m, trailing_zeros_z, Tol::Exact);
        masked_un!($label, $ut, $e, trailing_ones, trailing_ones_c, trailing_ones_m, trailing_ones_z, Tol::Exact);
        masked_un!($label, $ut, $e, reverse_bits, reverse_bits_c, reverse_bits_m, reverse_bits_z, Tol::Exact);
        masked_shift!($label, $ut, $e, shl, shl_c, shl_m, shl_z);
        masked_shift!($label, $ut, $e, shr, shr_c, shr_m, shr_z);
        masked_shift!($label, $ut, $e, rol, rol_c, rol_m, rol_z);
        masked_shift!($label, $ut, $e, ror, ror_c, ror_m, ror_z);
    }};
}

// ---------------------------------------------------------------------------
// Per-(backend, width) test functions.
// ---------------------------------------------------------------------------
macro_rules! float_mask_tests {
    ($name:ident, $backend:ty, $reg:ident, $label:expr) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            type E = <UT as Register>::Element;
            numeric_common!(UT, E, $label, Tol::Rel(0.0));
            // Signed / float-only masked ops. base op blended per the mask, so
            // approximate ops (rcp/rsqrt) are self-consistent (same value blended).
            masked_bin!($label, UT, E, div, div_c, div_m, div_z, Tol::Rel(0.0));
            masked_bin!($label, UT, E, rem, rem_c, rem_m, rem_z, Tol::Rel(0.0));
            masked_bin!(
                $label,
                UT,
                E,
                mul_sign,
                mul_sign_c,
                mul_sign_m,
                mul_sign_z,
                Tol::Rel(0.0)
            );
            masked_un!($label, UT, E, neg, neg_c, neg_m, neg_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, abs, abs_c, abs_m, abs_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, sqrt, sqrt_c, sqrt_m, sqrt_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, rcp, rcp_c, rcp_m, rcp_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, rsqrt, rsqrt_c, rsqrt_m, rsqrt_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, floor, floor_c, floor_m, floor_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, ceil, ceil_c, ceil_m, ceil_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, round, round_c, round_m, round_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, trunc, trunc_c, trunc_m, trunc_z, Tol::Rel(0.0));
            masked_un!($label, UT, E, fract, fract_c, fract_m, fract_z, Tol::Rel(0.0));
            masked_un!(
                $label,
                UT,
                E,
                signed_zero,
                signed_zero_c,
                signed_zero_m,
                signed_zero_z,
                Tol::Rel(0.0)
            );
            masked_un!(
                $label,
                UT,
                E,
                next_up,
                next_up_c,
                next_up_m,
                next_up_z,
                Tol::Rel(0.0)
            );
            masked_un!(
                $label,
                UT,
                E,
                next_down,
                next_down_c,
                next_down_m,
                next_down_z,
                Tol::Rel(0.0)
            );
            // FMA family (ternary): masked variants vs the backend's own unmasked op.
            masked_tern!(
                $label,
                UT,
                E,
                mul_adde,
                mul_adde_c,
                mul_adde_m,
                mul_adde_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                mul_sube,
                mul_sube_c,
                mul_sube_m,
                mul_sube_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                nmul_adde,
                nmul_adde_c,
                nmul_adde_m,
                nmul_adde_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                nmul_sube,
                nmul_sube_c,
                nmul_sube_m,
                nmul_sube_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                mul_add,
                mul_add_c,
                mul_add_m,
                mul_add_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                mul_sub,
                mul_sub_c,
                mul_sub_m,
                mul_sub_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                nmul_add,
                nmul_add_c,
                nmul_add_m,
                nmul_add_z,
                Tol::Rel(0.0)
            );
            masked_tern!(
                $label,
                UT,
                E,
                nmul_sub,
                nmul_sub_c,
                nmul_sub_m,
                nmul_sub_z,
                Tol::Rel(0.0)
            );
        }
    };
}

macro_rules! int_mask_tests {
    ($name:ident, $backend:ty, $reg:ident, $label:expr, signed) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            type E = <UT as Register>::Element;
            numeric_common!(UT, E, $label, Tol::Exact);
            masked_bin!($label, UT, E, bitand, bitand_c, bitand_m, bitand_z, Tol::Exact);
            masked_bin!($label, UT, E, bitor, bitor_c, bitor_m, bitor_z, Tol::Exact);
            masked_bin!($label, UT, E, bitxor, bitxor_c, bitxor_m, bitxor_z, Tol::Exact);
            masked_un!($label, UT, E, neg, neg_c, neg_m, neg_z, Tol::Exact);
            masked_un!($label, UT, E, abs, abs_c, abs_m, abs_z, Tol::Exact);
            int_extra_masked!(UT, E, $label);
            masked_bin!(
                $label,
                UT,
                E,
                avg_floor,
                avg_floor_c,
                avg_floor_m,
                avg_floor_z,
                Tol::Exact
            );
            masked_bin!(
                $label,
                UT,
                E,
                avg_ceil,
                avg_ceil_c,
                avg_ceil_m,
                avg_ceil_z,
                Tol::Exact
            );
            masked_shift!($label, UT, E, sra, sra_c, sra_m, sra_z);
        }
    };
    ($name:ident, $backend:ty, $reg:ident, $label:expr, unsigned) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            type E = <UT as Register>::Element;
            numeric_common!(UT, E, $label, Tol::Exact);
            masked_bin!($label, UT, E, bitand, bitand_c, bitand_m, bitand_z, Tol::Exact);
            masked_bin!($label, UT, E, bitor, bitor_c, bitor_m, bitor_z, Tol::Exact);
            masked_bin!($label, UT, E, bitxor, bitxor_c, bitxor_m, bitxor_z, Tol::Exact);
            int_extra_masked!(UT, E, $label);
            masked_bin!($label, UT, E, avg, avg_c, avg_m, avg_z, Tol::Exact);
        }
    };
}

// --- X86V3 (AVX2 + FMA) ----------------------------------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    mod v3_float {
        use super::*;
        float_mask_tests!(f32x4, X86V3, f32x4, "x86_v3 f32x4");
        float_mask_tests!(f32x8, X86V3, f32x8, "x86_v3 f32x8");
        float_mask_tests!(f32x16, X86V3, f32x16, "x86_v3 f32x16");
        float_mask_tests!(f64x2, X86V3, f64x2, "x86_v3 f64x2");
        float_mask_tests!(f64x4, X86V3, f64x4, "x86_v3 f64x4");
        float_mask_tests!(f64x8, X86V3, f64x8, "x86_v3 f64x8");
    }
    mod v3_int {
        use super::*;
        int_mask_tests!(i32x4, X86V3, i32x4, "x86_v3 i32x4", signed);
        int_mask_tests!(i32x8, X86V3, i32x8, "x86_v3 i32x8", signed);
        int_mask_tests!(i64x2, X86V3, i64x2, "x86_v3 i64x2", signed);
        int_mask_tests!(i64x4, X86V3, i64x4, "x86_v3 i64x4", signed);
        int_mask_tests!(u32x4, X86V3, u32x4, "x86_v3 u32x4", unsigned);
        int_mask_tests!(u32x8, X86V3, u32x8, "x86_v3 u32x8", unsigned);
        int_mask_tests!(u64x2, X86V3, u64x2, "x86_v3 u64x2", unsigned);
        int_mask_tests!(u64x4, X86V3, u64x4, "x86_v3 u64x4", unsigned);
    }

    // --- X86V2 (SSE4.2) --------------------------------------------------------
    mod v2_float {
        use super::*;
        float_mask_tests!(f32x4, X86V2, f32x4, "x86_v2 f32x4");
        float_mask_tests!(f32x8, X86V2, f32x8, "x86_v2 f32x8");
        float_mask_tests!(f64x2, X86V2, f64x2, "x86_v2 f64x2");
        float_mask_tests!(f64x4, X86V2, f64x4, "x86_v2 f64x4");
    }
    mod v2_int {
        use super::*;
        int_mask_tests!(i32x4, X86V2, i32x4, "x86_v2 i32x4", signed);
        int_mask_tests!(i32x8, X86V2, i32x8, "x86_v2 i32x8", signed);
        int_mask_tests!(i64x2, X86V2, i64x2, "x86_v2 i64x2", signed);
        int_mask_tests!(u32x4, X86V2, u32x4, "x86_v2 u32x4", unsigned);
        int_mask_tests!(u64x2, X86V2, u64x2, "x86_v2 u64x2", unsigned);
    }

    // --- X86V1 (SSE2) -----------------------------------------------------------
    mod v1_float {
        use super::*;
        float_mask_tests!(f32x4, X86V1, f32x4, "x86_v1 f32x4");
        float_mask_tests!(f32x8, X86V1, f32x8, "x86_v1 f32x8");
        float_mask_tests!(f64x2, X86V1, f64x2, "x86_v1 f64x2");
        float_mask_tests!(f64x4, X86V1, f64x4, "x86_v1 f64x4");
    }
    mod v1_int {
        use super::*;
        int_mask_tests!(i32x4, X86V1, i32x4, "x86_v1 i32x4", signed);
        int_mask_tests!(i32x8, X86V1, i32x8, "x86_v1 i32x8", signed);
        int_mask_tests!(i64x2, X86V1, i64x2, "x86_v1 i64x2", signed);
        int_mask_tests!(u32x4, X86V1, u32x4, "x86_v1 u32x4", unsigned);
        int_mask_tests!(u64x2, X86V1, u64x2, "x86_v1 u64x2", unsigned);
    }
}

// --- Scalar (reference backend as the *subject*, vs the same Rust oracle) ---
// Covers the scalar register impls and `ArrayRegister` lane delegation, which
// the differential suites only ever exercise as the oracle, never as the UUT.
mod scalar_float {
    use super::*;
    float_mask_tests!(f32x4, Scalar, f32x4, "scalar f32x4");
    float_mask_tests!(f32x8, Scalar, f32x8, "scalar f32x8");
    float_mask_tests!(f64x2, Scalar, f64x2, "scalar f64x2");
    float_mask_tests!(f64x4, Scalar, f64x4, "scalar f64x4");
}
mod scalar_int {
    use super::*;
    int_mask_tests!(i32x4, Scalar, i32x4, "scalar i32x4", signed);
    int_mask_tests!(i32x8, Scalar, i32x8, "scalar i32x8", signed);
    int_mask_tests!(i64x2, Scalar, i64x2, "scalar i64x2", signed);
    int_mask_tests!(u32x4, Scalar, u32x4, "scalar u32x4", unsigned);
    int_mask_tests!(u64x2, Scalar, u64x2, "scalar u64x2", unsigned);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    mod wasm_float {
        use super::*;
        float_mask_tests!(f32x4, Wasm, f32x4, "wasm f32x4");
        float_mask_tests!(f32x8, Wasm, f32x8, "wasm f32x8");
        float_mask_tests!(f64x2, Wasm, f64x2, "wasm f64x2");
        float_mask_tests!(f64x4, Wasm, f64x4, "wasm f64x4");
    }
    mod wasm_int {
        use super::*;
        int_mask_tests!(i32x4, Wasm, i32x4, "wasm i32x4", signed);
        int_mask_tests!(i32x8, Wasm, i32x8, "wasm i32x8", signed);
        int_mask_tests!(i64x2, Wasm, i64x2, "wasm i64x2", signed);
        int_mask_tests!(u32x4, Wasm, u32x4, "wasm u32x4", unsigned);
        int_mask_tests!(u64x2, Wasm, u64x2, "wasm u64x2", unsigned);
    }
}
