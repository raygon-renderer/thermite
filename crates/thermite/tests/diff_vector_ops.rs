//! Coverage for the `Vector`-layer operator / masked / assign delegations in
//! `vector/ops.rs` (was ~14%). These are thin wrappers over the register ops,
//! so correctness of the *plain* op is already covered at the register layer
//! (`diff_ops`/`diff_mask`). Here we only assert that, for each op, the masked
//! variants are consistent selects of the plain result:
//!
//!   op_c(mask, b)      == mask ? (a op b) : a
//!   op_m(src, mask, b) == mask ? (a op b) : src
//!   op_z(mask, b)      == mask ? (a op b) : 0
//!   (+ the `_assign` / `_assign_c/_m/_z` mutating equivalents)
//!
//! Inputs are finite and small (no NaN/Inf, no overflow even under debug
//! overflow checks), and f32 `==` treats ±0 as equal, so exact comparison is safe.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use core::ops::{
    AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, DivAssign, MulAssign, RemAssign, ShlAssign, ShrAssign,
    SubAssign,
};

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;
use thermite::vector::ops::*;

use thermite::backend::scalar::Scalar;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

fn rd<V: GenericVector>(v: V) -> Vec<V::Element>
where
    V::Element: Clone,
{
    v.into_array().as_slice().to_vec()
}

fn sel<E: Copy>(mb: &[bool], t: &[E], f: &[E]) -> Vec<E> {
    (0..mb.len()).map(|i| if mb[i] { t[i] } else { f[i] }).collect()
}

/// Binary op: plain `$plain` precomputed; checks the 7 masked/assign variants.
macro_rules! bin {
    ($a:expr, $b:expr, $src:expr, $zero:expr, $mask:expr, $mb:expr, $plain:expr,
     $c:ident, $m:ident, $z:ident, $as:ident, $ac:ident, $am:ident, $az:ident) => {{
        let a = $a;
        let b = $b;
        let src = $src;
        let mask = $mask;
        let mb: &[bool] = $mb;
        let plain: Vec<_> = $plain;
        let (aa, sa, za) = (rd(a), rd(src), rd($zero));
        assert_eq!(rd(a.$c(mask, b)), sel(mb, &plain, &aa), stringify!($c));
        assert_eq!(rd(a.$m(src, mask, b)), sel(mb, &plain, &sa), stringify!($m));
        assert_eq!(rd(a.$z(mask, b)), sel(mb, &plain, &za), stringify!($z));
        { let mut t = a; t.$as(b); assert_eq!(rd(t), plain, stringify!($as)); }
        { let mut t = a; t.$ac(mask, b); assert_eq!(rd(t), sel(mb, &plain, &aa), stringify!($ac)); }
        { let mut t = a; t.$am(src, mask, b); assert_eq!(rd(t), sel(mb, &plain, &sa), stringify!($am)); }
        { let mut t = a; t.$az(mask, b); assert_eq!(rd(t), sel(mb, &plain, &za), stringify!($az)); }
    }};
}

/// Unary op (no rhs, no assign variants): not / neg.
macro_rules! un {
    ($a:expr, $src:expr, $zero:expr, $mask:expr, $mb:expr, $plain:expr,
     $c:ident, $m:ident, $z:ident) => {{
        let a = $a;
        let src = $src;
        let mask = $mask;
        let mb: &[bool] = $mb;
        let plain: Vec<_> = $plain;
        let (aa, sa, za) = (rd(a), rd(src), rd($zero));
        assert_eq!(rd(a.$c(mask)), sel(mb, &plain, &aa), stringify!($c));
        assert_eq!(rd(a.$m(src, mask)), sel(mb, &plain, &sa), stringify!($m));
        assert_eq!(rd(a.$z(mask)), sel(mb, &plain, &za), stringify!($z));
    }};
}

/// Fused multiply-add family: `self.$op(a, b)`, plus masked + assign variants.
macro_rules! fma {
    ($s:expr, $a:expr, $b:expr, $src:expr, $zero:expr, $mask:expr, $mb:expr,
     $op:ident, $c:ident, $m:ident, $z:ident, $as:ident, $ac:ident, $am:ident, $az:ident) => {{
        let s = $s;
        let a = $a;
        let b = $b;
        let src = $src;
        let mask = $mask;
        let mb: &[bool] = $mb;
        let plain = rd(s.$op(a, b));
        let (ss, sa, za) = (rd(s), rd(src), rd($zero));
        assert_eq!(rd(s.$c(mask, a, b)), sel(mb, &plain, &ss), stringify!($c));
        assert_eq!(rd(s.$m(src, mask, a, b)), sel(mb, &plain, &sa), stringify!($m));
        assert_eq!(rd(s.$z(mask, a, b)), sel(mb, &plain, &za), stringify!($z));
        { let mut t = s; t.$as(a, b); assert_eq!(rd(t), plain, stringify!($as)); }
        { let mut t = s; t.$ac(mask, a, b); assert_eq!(rd(t), sel(mb, &plain, &ss), stringify!($ac)); }
        { let mut t = s; t.$am(src, mask, a, b); assert_eq!(rd(t), sel(mb, &plain, &sa), stringify!($am)); }
        { let mut t = s; t.$az(mask, a, b); assert_eq!(rd(t), sel(mb, &plain, &za), stringify!($az)); }
    }};
}

macro_rules! ops_suite {
    ($mod:ident, $backend:ty, $freg:ident, $ireg:ident) => {
        mod $mod {
            use super::*;
            type F = Vector<<$backend as Simd>::$freg>;
            type I = Vector<<$backend as Simd>::$ireg>;
            type IU = <I as GenericVector>::Unsigned;

            #[test]
            fn float_ops() {
                let h = (F::LANES / 2) as f32;
                let fa = F::indexed() - F::splat(h); // mixed sign
                let fb = F::splat(0.5); // nonzero (safe for div/rem)
                let fsrc = F::splat(7.0);
                let mbf: Vec<bool> = rd(fa).iter().map(|&x| x < 0.5).collect();
                let mf = fa.cmp_lt(fb);

                bin!(
                    fa,
                    fb,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa + fb),
                    add_c,
                    add_m,
                    add_z,
                    add_assign,
                    add_assign_c,
                    add_assign_m,
                    add_assign_z
                );
                bin!(
                    fa,
                    fb,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa - fb),
                    sub_c,
                    sub_m,
                    sub_z,
                    sub_assign,
                    sub_assign_c,
                    sub_assign_m,
                    sub_assign_z
                );
                bin!(
                    fa,
                    fb,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa * fb),
                    mul_c,
                    mul_m,
                    mul_z,
                    mul_assign,
                    mul_assign_c,
                    mul_assign_m,
                    mul_assign_z
                );
                bin!(
                    fa,
                    fb,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa / fb),
                    div_c,
                    div_m,
                    div_z,
                    div_assign,
                    div_assign_c,
                    div_assign_m,
                    div_assign_z
                );
                bin!(
                    fa,
                    fb,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa % fb),
                    rem_c,
                    rem_m,
                    rem_z,
                    rem_assign,
                    rem_assign_c,
                    rem_assign_m,
                    rem_assign_z
                );

                un!(fa, fsrc, F::ZERO, mf, &mbf, rd(-fa), neg_c, neg_m, neg_z);

                let (a2, b3) = (F::splat(2.0), F::splat(3.0));
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    mul_add,
                    mul_add_c,
                    mul_add_m,
                    mul_add_z,
                    mul_add_assign,
                    mul_add_assign_c,
                    mul_add_assign_m,
                    mul_add_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    mul_sub,
                    mul_sub_c,
                    mul_sub_m,
                    mul_sub_z,
                    mul_sub_assign,
                    mul_sub_assign_c,
                    mul_sub_assign_m,
                    mul_sub_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    nmul_add,
                    nmul_add_c,
                    nmul_add_m,
                    nmul_add_z,
                    nmul_add_assign,
                    nmul_add_assign_c,
                    nmul_add_assign_m,
                    nmul_add_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    nmul_sub,
                    nmul_sub_c,
                    nmul_sub_m,
                    nmul_sub_z,
                    nmul_sub_assign,
                    nmul_sub_assign_c,
                    nmul_sub_assign_m,
                    nmul_sub_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    mul_adde,
                    mul_adde_c,
                    mul_adde_m,
                    mul_adde_z,
                    mul_adde_assign,
                    mul_adde_assign_c,
                    mul_adde_assign_m,
                    mul_adde_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    mul_sube,
                    mul_sube_c,
                    mul_sube_m,
                    mul_sube_z,
                    mul_sube_assign,
                    mul_sube_assign_c,
                    mul_sube_assign_m,
                    mul_sube_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    nmul_adde,
                    nmul_adde_c,
                    nmul_adde_m,
                    nmul_adde_z,
                    nmul_adde_assign,
                    nmul_adde_assign_c,
                    nmul_adde_assign_m,
                    nmul_adde_assign_z
                );
                fma!(
                    fa,
                    a2,
                    b3,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    nmul_sube,
                    nmul_sube_c,
                    nmul_sube_m,
                    nmul_sube_z,
                    nmul_sube_assign,
                    nmul_sube_assign_c,
                    nmul_sube_assign_m,
                    nmul_sube_assign_z
                );

                // --- FloatVector unary conditional ops (`_c`/`_m`/`_z`, no assign) ---
                let fpos = fa.abs() + F::ONE; // positive (safe for sqrt/rcp/rsqrt)
                un!(
                    fpos,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fpos.sqrt()),
                    sqrt_c,
                    sqrt_m,
                    sqrt_z
                );
                un!(fpos, fsrc, F::ZERO, mf, &mbf, rd(fpos.rcp()), rcp_c, rcp_m, rcp_z);
                un!(
                    fpos,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fpos.rsqrt()),
                    rsqrt_c,
                    rsqrt_m,
                    rsqrt_z
                );
                un!(
                    fa,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa.floor()),
                    floor_c,
                    floor_m,
                    floor_z
                );
                un!(fa, fsrc, F::ZERO, mf, &mbf, rd(fa.ceil()), ceil_c, ceil_m, ceil_z);
                un!(
                    fa,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa.round()),
                    round_c,
                    round_m,
                    round_z
                );
                un!(
                    fa,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa.trunc()),
                    trunc_c,
                    trunc_m,
                    trunc_z
                );
                un!(
                    fa,
                    fsrc,
                    F::ZERO,
                    mf,
                    &mbf,
                    rd(fa.fract()),
                    fract_c,
                    fract_m,
                    fract_z
                );
                un!(fa, fsrc, F::ZERO, mf, &mbf, rd(fa.abs()), abs_c, abs_m, abs_z);

                // --- plain NumericVector/SignedVector binary ops vs per-lane oracle ---
                let (xa, xb) = (rd(fa), rd(fb));
                let want_min: Vec<_> = (0..F::LANES)
                    .map(|i| if xa[i] < xb[i] { xa[i] } else { xb[i] })
                    .collect();
                let want_max: Vec<_> = (0..F::LANES)
                    .map(|i| if xa[i] > xb[i] { xa[i] } else { xb[i] })
                    .collect();
                assert_eq!(rd(fa.min(fb)), want_min, "min");
                assert_eq!(rd(fa.max(fb)), want_max, "max");
                // clamp(lo=-1, hi=1)
                let want_cl: Vec<_> = xa.iter().map(|&v| v.max(-1.0 as f32).min(1.0 as f32)).collect();
                assert_eq!(rd(fa.clamp(F::splat(-1.0), F::splat(1.0))), want_cl, "clamp");
                // copysign(fpos, fa): magnitude of fpos, sign of fa (fa is mixed-sign)
                let fp = rd(fpos);
                let want_cs: Vec<_> = (0..F::LANES)
                    .map(|i| fp[i] * if xa[i].is_sign_negative() { -1.0 } else { 1.0 })
                    .collect();
                assert_eq!(rd(fpos.copysign(fa)), want_cs, "copysign");
            }

            #[test]
            fn int_ops() {
                let ih = (I::LANES / 2) as i32;
                let ia = I::indexed() - I::splat(ih); // mixed sign
                let ib = I::splat(3);
                let isrc = I::splat(7);
                let mbi: Vec<bool> = rd(ia).iter().map(|&x| x < 3).collect();
                let mi = ia.cmp_lt(ib);

                // arithmetic (NumericRegister, integer path)
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia + ib),
                    add_c,
                    add_m,
                    add_z,
                    add_assign,
                    add_assign_c,
                    add_assign_m,
                    add_assign_z
                );
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia * ib),
                    mul_c,
                    mul_m,
                    mul_z,
                    mul_assign,
                    mul_assign_c,
                    mul_assign_m,
                    mul_assign_z
                );

                // bitwise
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia & ib),
                    bitand_c,
                    bitand_m,
                    bitand_z,
                    bitand_assign,
                    bitand_assign_c,
                    bitand_assign_m,
                    bitand_assign_z
                );
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia | ib),
                    bitor_c,
                    bitor_m,
                    bitor_z,
                    bitor_assign,
                    bitor_assign_c,
                    bitor_assign_m,
                    bitor_assign_z
                );
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia ^ ib),
                    bitxor_c,
                    bitxor_m,
                    bitxor_z,
                    bitxor_assign,
                    bitxor_assign_c,
                    bitxor_assign_m,
                    bitxor_assign_z
                );
                bin!(
                    ia,
                    ib,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia.bitandnot(ib)),
                    bitandnot_c,
                    bitandnot_m,
                    bitandnot_z,
                    bitandnot_assign,
                    bitandnot_assign_c,
                    bitandnot_assign_m,
                    bitandnot_assign_z
                );

                // shifts — scalar (u32) rhs
                bin!(
                    ia,
                    2u32,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia << 2u32),
                    shl_c,
                    shl_m,
                    shl_z,
                    shl_assign,
                    shl_assign_c,
                    shl_assign_m,
                    shl_assign_z
                );
                bin!(
                    ia,
                    2u32,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia >> 2u32),
                    shr_c,
                    shr_m,
                    shr_z,
                    shr_assign,
                    shr_assign_c,
                    shr_assign_m,
                    shr_assign_z
                );

                // shifts — per-lane vector rhs
                let sv = IU::splat(2);
                bin!(
                    ia,
                    sv,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia << sv),
                    shl_c,
                    shl_m,
                    shl_z,
                    shl_assign,
                    shl_assign_c,
                    shl_assign_m,
                    shl_assign_z
                );
                bin!(
                    ia,
                    sv,
                    isrc,
                    I::ZERO,
                    mi,
                    &mbi,
                    rd(ia >> sv),
                    shr_c,
                    shr_m,
                    shr_z,
                    shr_assign,
                    shr_assign_c,
                    shr_assign_m,
                    shr_assign_z
                );

                // unary
                un!(ia, isrc, I::ZERO, mi, &mbi, rd(!ia), not_c, not_m, not_z);
                un!(ia, isrc, I::ZERO, mi, &mbi, rd(-ia), neg_c, neg_m, neg_z);
            }
        }
    };
}

ops_suite!(v3, X86V3, f32x8, i32x8);
ops_suite!(v2, X86V2, f32x4, i32x4);
ops_suite!(v1, X86V1, f32x4, i32x4);
ops_suite!(scalar, Scalar, f32x4, i32x4);
