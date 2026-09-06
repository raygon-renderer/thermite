//! Byte and word bit ops on every backend: `reverse_bits`, rotates (runtime,
//! const, per-lane, out-of-range), the immediate shifts, their masked
//! variants, and the masked byte permutes, at every `Simd` width and the
//! native width. AVX-512 tier 2 lowers these to GFNI affine transforms and
//! VBMI2 funnel shifts (`backend/x86_v4/polyfills/{gfni,funnel}.rs`). Below
//! that they are word-op polyfills, and every other backend takes the trait
//! defaults. Oracles are Rust's scalar ops. Masked variants are checked
//! against the same backend's unmasked op blended per a known mask.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use harness::Tol;
use thermite::register::{BitshiftRegister, CoreRegister, Register, SignedIntegerRegister};
use thermite::simd::{NativeSimd, Simd};

macro_rules! lanes_of {
    ($ut:ty) => {
        <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE
    };
}

/// `roli`/`rori`/`shli`/`shri` for each literal amount below the width.
macro_rules! check_const {
    ($l:expr, $ut:ty, $e:ty, $ue:ty, $input:expr, $regv:expr, $bits:expr; $($imm:literal),* $(,)?) => {$(
        if ($imm as u32) < $bits {
            let cases: [(&str, Vec<$e>, Vec<$e>); 4] = [
                ("roli", harness::read::<$ut>(&<$ut>::roli::<$imm>($regv)),
                    $input.iter().map(|&x| x.rotate_left($imm as u32)).collect()),
                ("rori", harness::read::<$ut>(&<$ut>::rori::<$imm>($regv)),
                    $input.iter().map(|&x| x.rotate_right($imm as u32)).collect()),
                ("shli", harness::read::<$ut>(&<$ut>::shli::<$imm>($regv)),
                    $input.iter().map(|&x| x.wrapping_shl($imm as u32)).collect()),
                ("shri", harness::read::<$ut>(&<$ut>::shri::<$imm>($regv)),
                    $input.iter().map(|&x| (x as $ue).wrapping_shr($imm as u32) as $e).collect()),
            ];
            for (name, got, want) in cases.iter() {
                harness::assert_lanes_eq(
                    &format!("{} [{}<{}>]", $l, name, stringify!($imm)),
                    &[$input.as_slice()], got, want, Tol::Exact);
            }
        }
    )*};
}

/// Unmasked ops vs Rust: reverse, runtime/const/per-lane/out-of-range rotates,
/// runtime and const shifts.
macro_rules! bitperm {
    ($l:expr, $ut:ty, $e:ty, $ue:ty) => {{
        let l = $l;
        oracle_unary!(l, $ut, $e, reverse_bits, |x| x.reverse_bits(), Tol::Exact);
        oracle_unary!(l, $ut, $e, swap_bytes, |x| x.swap_bytes(), Tol::Exact);
        oracle_shift!(l, $ut, $e, rol, |x, s| x.rotate_left(s));
        oracle_shift!(l, $ut, $e, ror, |x, s| x.rotate_right(s));
        oracle_shift!(l, $ut, $e, shl, |x, s| x.wrapping_shl(s));
        oracle_shift!(l, $ut, $e, shr, |x, s| ((x as $ue) >> s) as $e);

        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let bits = (core::mem::size_of::<$e>() * 8) as u32;
        for (i, input) in harness::corpus::<$e>(lanes, &mut rng).iter().enumerate() {
            let reg = harness::make_array::<$ut>(input);

            let shifts: Vec<$ue> = (0..lanes).map(|j| ((i * 7 + j * 3) as u32 % bits) as $ue).collect();
            let sv = harness::make_array::<<$ut as Register>::Unsigned>(&shifts);
            let got = harness::read::<$ut>(&<$ut>::rolv(reg, sv));
            let want: Vec<$e> = input.iter().zip(&shifts).map(|(&x, &s)| x.rotate_left(s as u32)).collect();
            harness::assert_lanes_eq(&format!("{l} [rolv]"), &[input.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::rorv(reg, sv));
            let want: Vec<$e> = input.iter().zip(&shifts).map(|(&x, &s)| x.rotate_right(s as u32)).collect();
            harness::assert_lanes_eq(&format!("{l} [rorv]"), &[input.as_slice()], &got, &want, Tol::Exact);

            for extra in [0u32, 1, 3, bits / 2] {
                let sh = bits + extra;
                let got = harness::read::<$ut>(&<$ut>::rol(reg, sh));
                let want: Vec<$e> = input.iter().map(|&x| x.rotate_left(sh)).collect();
                harness::assert_lanes_eq(&format!("{l} [rol {sh}]"), &[input.as_slice()], &got, &want, Tol::Exact);
                let got = harness::read::<$ut>(&<$ut>::ror(reg, sh));
                let want: Vec<$e> = input.iter().map(|&x| x.rotate_right(sh)).collect();
                harness::assert_lanes_eq(&format!("{l} [ror {sh}]"), &[input.as_slice()], &got, &want, Tol::Exact);
            }

            check_const!(l, $ut, $e, $ue, input, reg, bits; 0, 1, 3, 5, 7, 9, 12, 15);
        }
    }};
}

/// Arithmetic right shifts, runtime and const.
macro_rules! bitperm_signed {
    ($l:expr, $ut:ty, $e:ty) => {{
        let l = $l;
        oracle_shift!(l, $ut, $e, sra, |x, s| x >> s);
        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let bits = (core::mem::size_of::<$e>() * 8) as u32;
        for input in harness::corpus::<$e>(lanes, &mut rng) {
            let reg = harness::make_array::<$ut>(&input);
            check_srai!(l, $ut, $e, input, reg, bits; 0, 1, 3, 5, 7, 9, 15);
        }
    }};
}

macro_rules! check_srai {
    ($l:expr, $ut:ty, $e:ty, $input:expr, $regv:expr, $bits:expr; $($imm:literal),* $(,)?) => {$(
        if ($imm as u32) < $bits {
            let got = harness::read::<$ut>(&<$ut>::srai::<$imm>($regv));
            let want: Vec<$e> = $input.iter().map(|&x| x.wrapping_shr($imm as u32)).collect();
            harness::assert_lanes_eq(&format!("{} [srai<{}>]", $l, $imm), &[$input.as_slice()], &got, &want, Tol::Exact);
        }
    )*};
}

/// `_c`/`_m`/`_z` of a unary op vs the unmasked op blended per lane.
macro_rules! masked_un {
    ($l:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident $(, $imm:literal)?) => {{
        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for ((x, s), bits) in xs.iter().zip(ss.iter()).zip(pats.iter()) {
            let mask = harness::build_mask::<$ut>(bits);
            let base = harness::read::<$ut>(&<$ut>::$base$(::<$imm>)?(mk(x)));
            let got = harness::read::<$ut>(&<$ut>::$c$(::<$imm>)?(mask, mk(x)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { x[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($c)), &[x.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::$m$(::<$imm>)?(mk(s), mask, mk(x)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($m)), &[x.as_slice(), s.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::$z$(::<$imm>)?(mask, mk(x)));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($z)), &[x.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

/// `_c`/`_m`/`_z` of a runtime-count shift/rotate vs the unmasked op.
macro_rules! masked_shift {
    ($l:expr, $ut:ty, $e:ty, $base:ident, $c:ident, $m:ident, $z:ident) => {{
        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let bits = (core::mem::size_of::<$e>() * 8) as u32;
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for sh in [1u32, 5, bits - 1, bits + 2] {
            for ((x, s), bits) in xs.iter().zip(ss.iter()).zip(pats.iter()) {
                let mask = harness::build_mask::<$ut>(bits);
                let base = harness::read::<$ut>(&<$ut>::$base(mk(x), sh));
                let got = harness::read::<$ut>(&<$ut>::$c(mask, mk(x), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { x[i] }).collect();
                harness::assert_lanes_eq(&format!("{} [{} {sh}]", $l, stringify!($c)), &[x.as_slice()], &got, &want, Tol::Exact);
                let got = harness::read::<$ut>(&<$ut>::$m(mk(s), mask, mk(x), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
                harness::assert_lanes_eq(&format!("{} [{} {sh}]", $l, stringify!($m)), &[x.as_slice(), s.as_slice()], &got, &want, Tol::Exact);
                let got = harness::read::<$ut>(&<$ut>::$z(mask, mk(x), sh));
                let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
                harness::assert_lanes_eq(&format!("{} [{} {sh}]", $l, stringify!($z)), &[x.as_slice()], &got, &want, Tol::Exact);
            }
        }
    }};
}

/// `_c`/`_m`/`_z` of a per-lane-count rotate vs the unmasked op.
macro_rules! masked_rotv {
    ($l:expr, $ut:ty, $e:ty, $ue:ty, $base:ident, $c:ident, $m:ident, $z:ident) => {{
        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let width = (core::mem::size_of::<$e>() * 8) as u32;
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for (i, ((x, s), bits)) in xs.iter().zip(ss.iter()).zip(pats.iter()).enumerate() {
            let shifts: Vec<$ue> = (0..lanes).map(|j| ((i * 5 + j * 3) as u32 % width) as $ue).collect();
            let sv = harness::make_array::<<$ut as Register>::Unsigned>(&shifts);
            let mask = harness::build_mask::<$ut>(bits);
            let base = harness::read::<$ut>(&<$ut>::$base(mk(x), sv));
            let got = harness::read::<$ut>(&<$ut>::$c(mask, mk(x), sv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { x[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($c)), &[x.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::$m(mk(s), mask, mk(x), sv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($m)), &[x.as_slice(), s.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::$z(mask, mk(x), sv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(&format!("{} [{}]", $l, stringify!($z)), &[x.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

/// `permutev_m`/`permutev_z`/`swizzle_m`/`swizzle_z` vs the unmasked permutes.
macro_rules! masked_permute {
    ($l:expr, $ut:ty, $e:ty, $ue:ty) => {{
        let mut rng = harness::rng();
        let lanes = lanes_of!($ut);
        let xs = harness::corpus::<$e>(lanes, &mut rng);
        let ys = harness::corpus::<$e>(lanes, &mut rng);
        let ss = harness::corpus::<$e>(lanes, &mut rng);
        let pats = harness::mask_patterns(lanes, xs.len(), &mut rng);
        let zero = <$e as Default>::default();
        let mk = harness::make_array::<$ut>;
        for (i, (((x, y), s), bits)) in xs.iter().zip(ys.iter()).zip(ss.iter()).zip(pats.iter()).enumerate() {
            let idx: Vec<$ue> = (0..lanes).map(|j| ((i * 11 + j * 7) % (2 * lanes)) as $ue).collect();
            let iv = harness::make_array::<<$ut as Register>::Unsigned>(&idx);
            let mask = harness::build_mask::<$ut>(bits);

            let base = harness::read::<$ut>(&<$ut>::permutev(mk(x), iv));
            let got = harness::read::<$ut>(&<$ut>::permutev_m(mk(s), mask, mk(x), iv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [permutev_m]", $l), &[x.as_slice(), s.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::permutev_z(mask, mk(x), iv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(&format!("{} [permutev_z]", $l), &[x.as_slice()], &got, &want, Tol::Exact);

            let base = harness::read::<$ut>(&<$ut>::swizzle(mk(x), mk(y), iv));
            let got = harness::read::<$ut>(&<$ut>::swizzle_m(mk(s), mask, mk(x), mk(y), iv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { s[i] }).collect();
            harness::assert_lanes_eq(&format!("{} [swizzle_m]", $l), &[x.as_slice(), y.as_slice()], &got, &want, Tol::Exact);
            let got = harness::read::<$ut>(&<$ut>::swizzle_z(mask, mk(x), mk(y), iv));
            let want: Vec<$e> = (0..lanes).map(|i| if bits[i] { base[i] } else { zero }).collect();
            harness::assert_lanes_eq(&format!("{} [swizzle_z]", $l), &[x.as_slice(), y.as_slice()], &got, &want, Tol::Exact);
        }
    }};
}

/// Every masked form of the ops above.
macro_rules! masked_all {
    ($l:expr, $ut:ty, $e:ty, $ue:ty) => {{
        let l = $l;
        masked_un!(l, $ut, $e, reverse_bits, reverse_bits_c, reverse_bits_m, reverse_bits_z);
        masked_un!(l, $ut, $e, shli, shli_c, shli_m, shli_z, 3);
        masked_un!(l, $ut, $e, shri, shri_c, shri_m, shri_z, 2);
        masked_un!(l, $ut, $e, roli, roli_c, roli_m, roli_z, 3);
        masked_un!(l, $ut, $e, rori, rori_c, rori_m, rori_z, 5);
        masked_shift!(l, $ut, $e, shl, shl_c, shl_m, shl_z);
        masked_shift!(l, $ut, $e, shr, shr_c, shr_m, shr_z);
        masked_shift!(l, $ut, $e, rol, rol_c, rol_m, rol_z);
        masked_shift!(l, $ut, $e, ror, ror_c, ror_m, ror_z);
        masked_rotv!(l, $ut, $e, $ue, rolv, rolv_c, rolv_m, rolv_z);
        masked_rotv!(l, $ut, $e, $ue, rorv, rorv_c, rorv_m, rorv_z);
        masked_permute!(l, $ut, $e, $ue);
    }};
}

macro_rules! masked_signed {
    ($l:expr, $ut:ty, $e:ty) => {{
        let l = $l;
        masked_un!(l, $ut, $e, srai, srai_c, srai_m, srai_z, 2);
        masked_shift!(l, $ut, $e, sra, sra_c, sra_m, sra_z);
    }};
}

for_each_backend_concrete! {
    fn byte_ops() {
        bitperm!(harness::label::<S>("i8x2"), <S as Simd>::i8x2, i8, u8);
        bitperm!(harness::label::<S>("i8x4"), <S as Simd>::i8x4, i8, u8);
        bitperm!(harness::label::<S>("i8x8"), <S as Simd>::i8x8, i8, u8);
        bitperm!(harness::label::<S>("i8x16"), <S as Simd>::i8x16, i8, u8);
        bitperm!(harness::label::<S>("i8xN"), <S as NativeSimd>::i8xN, i8, u8);
        bitperm!(harness::label::<S>("u8x2"), <S as Simd>::u8x2, u8, u8);
        bitperm!(harness::label::<S>("u8x4"), <S as Simd>::u8x4, u8, u8);
        bitperm!(harness::label::<S>("u8x8"), <S as Simd>::u8x8, u8, u8);
        bitperm!(harness::label::<S>("u8x16"), <S as Simd>::u8x16, u8, u8);
        bitperm!(harness::label::<S>("u8xN"), <S as NativeSimd>::u8xN, u8, u8);
        bitperm_signed!(harness::label::<S>("i8x2"), <S as Simd>::i8x2, i8);
        bitperm_signed!(harness::label::<S>("i8x16"), <S as Simd>::i8x16, i8);
        bitperm_signed!(harness::label::<S>("i8xN"), <S as NativeSimd>::i8xN, i8);
    }

    fn word_ops() {
        bitperm!(harness::label::<S>("i16x2"), <S as Simd>::i16x2, i16, u16);
        bitperm!(harness::label::<S>("i16x4"), <S as Simd>::i16x4, i16, u16);
        bitperm!(harness::label::<S>("i16x8"), <S as Simd>::i16x8, i16, u16);
        bitperm!(harness::label::<S>("i16x16"), <S as Simd>::i16x16, i16, u16);
        bitperm!(harness::label::<S>("i16xN"), <S as NativeSimd>::i16xN, i16, u16);
        bitperm!(harness::label::<S>("u16x2"), <S as Simd>::u16x2, u16, u16);
        bitperm!(harness::label::<S>("u16x4"), <S as Simd>::u16x4, u16, u16);
        bitperm!(harness::label::<S>("u16x8"), <S as Simd>::u16x8, u16, u16);
        bitperm!(harness::label::<S>("u16x16"), <S as Simd>::u16x16, u16, u16);
        bitperm!(harness::label::<S>("u16xN"), <S as NativeSimd>::u16xN, u16, u16);
        bitperm_signed!(harness::label::<S>("i16x2"), <S as Simd>::i16x2, i16);
        bitperm_signed!(harness::label::<S>("i16x8"), <S as Simd>::i16x8, i16);
        bitperm_signed!(harness::label::<S>("i16xN"), <S as NativeSimd>::i16xN, i16);
    }

    fn byte_masked() {
        masked_all!(harness::label::<S>("i8x2"), <S as Simd>::i8x2, i8, u8);
        masked_all!(harness::label::<S>("i8x16"), <S as Simd>::i8x16, i8, u8);
        masked_all!(harness::label::<S>("i8xN"), <S as NativeSimd>::i8xN, i8, u8);
        masked_all!(harness::label::<S>("u8x8"), <S as Simd>::u8x8, u8, u8);
        masked_all!(harness::label::<S>("u8x16"), <S as Simd>::u8x16, u8, u8);
        masked_all!(harness::label::<S>("u8xN"), <S as NativeSimd>::u8xN, u8, u8);
        masked_signed!(harness::label::<S>("i8x16"), <S as Simd>::i8x16, i8);
        masked_signed!(harness::label::<S>("i8xN"), <S as NativeSimd>::i8xN, i8);
    }

    fn word_masked() {
        masked_all!(harness::label::<S>("i16x2"), <S as Simd>::i16x2, i16, u16);
        masked_all!(harness::label::<S>("i16x8"), <S as Simd>::i16x8, i16, u16);
        masked_all!(harness::label::<S>("i16x16"), <S as Simd>::i16x16, i16, u16);
        masked_all!(harness::label::<S>("i16xN"), <S as NativeSimd>::i16xN, i16, u16);
        masked_all!(harness::label::<S>("u16x8"), <S as Simd>::u16x8, u16, u16);
        masked_all!(harness::label::<S>("u16xN"), <S as NativeSimd>::u16xN, u16, u16);
        masked_signed!(harness::label::<S>("i16x8"), <S as Simd>::i16x8, i16);
        masked_signed!(harness::label::<S>("i16xN"), <S as NativeSimd>::i16xN, i16);
    }
}
