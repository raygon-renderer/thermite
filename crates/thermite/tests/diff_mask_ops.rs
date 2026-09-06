//! Coverage for the `Mask<R>` wrapper and `GenericMask` API (`mask.rs`):
//! all/any/none, the bitwise operators (`&`/`|`/`^`/`!`/andnot + assign),
//! `select`/`cast`/`swap`/`ternlog`, `interleave`, `native_bitmask`, `From<bool>`/
//! `From<Vector>`, `Debug`, `Default`, and the `TRUTHY`/`FALSY` constants.
//!
//! Masks are built from comparisons (known per-lane bool patterns) and every
//! operation is checked against the booleans computed in plain Rust.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::mask::{GenericMask, Mask};
use thermite::prelude::*;
use thermite::register::{CoreRegister, MaskRegister, Register};
use thermite::simd::{Simd, Simd3A};
use thermite::vector::Interleave;
use thermite::vector::ops::{BitAndNot, BitAndNotAssign};

/// Lane-count-agnostic coverage for the **`MaskRegister`** primitives on the mask
/// register of `R` (`new_mask`/`test`/`set`/`all`/`any`/`none`/`native_bitmask`) plus
/// `into_mask` (via `From<Vector>`) and the `GenericMask` bitwise ops, for every
/// register width, not just the 4-lane `f32x4`/`i32x4` the bespoke suite uses.
/// Masks are built from the shared `mask_patterns` corpus and checked per-lane.
#[inline(always)]
fn check_mask_reg<R>(label: &str)
where
    R: Register,
    Mask<R>: GenericMask,
    Vector<R>: GenericVector,
{
    let n = <Vector<R> as GenericVector>::LANES;
    let lo_bits = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut rng = harness::rng();
    for pat in harness::mask_patterns(n, 24, &mut rng) {
        let m = Mask::<R>(harness::build_mask::<R>(&pat)); // new_mask
        assert_eq!(harness::read_mask::<R>(m.0, n), pat, "{label} new_mask/test"); // test
        assert_eq!(m.all(), pat.iter().all(|&b| b), "{label} all");
        assert_eq!(m.any(), pat.iter().any(|&b| b), "{label} any");
        assert_eq!(m.none(), !pat.iter().any(|&b| b), "{label} none");
        if let Some(bm) = m.native_bitmask() {
            let want = pat.iter().enumerate().fold(0u64, |a, (i, &b)| a | ((b as u64) << i));
            assert_eq!(bm & lo_bits, want, "{label} native_bitmask");
        }
        // set: flip each lane in turn and read it back
        for i in 0..n {
            let m2 = <R::Mask as MaskRegister>::set(m.0, i, !pat[i]);
            assert_eq!(<R::Mask as MaskRegister>::test(m2, i), !pat[i], "{label} set lane {i}");
        }
        // value<->mask conversions: from_mask gives all-1-bits where true (MSB set,
        // bitwise-nonzero), so into_mask and msb_to_mask both round-trip it back.
        let truthy_val = <R as CoreRegister>::from_mask(m.0); // from_mask
        assert_eq!(
            harness::read_mask::<R>(<R as Register>::into_mask(truthy_val), n),
            pat,
            "{label} into_mask"
        );
        assert_eq!(
            harness::read_mask::<R>(<R as Register>::msb_to_mask(truthy_val), n),
            pat,
            "{label} msb_to_mask"
        );

        // bitwise identities (GenericMask)
        assert_eq!(
            harness::read_mask::<R>((m & Mask::<R>::TRUTHY).0, n),
            pat,
            "{label} & TRUTHY"
        );
        assert_eq!(
            harness::read_mask::<R>((m | Mask::<R>::FALSY).0, n),
            pat,
            "{label} | FALSY"
        );
        assert!((m ^ m).none(), "{label} ^ self");
        assert_eq!(harness::read_mask::<R>((!!m).0, n), pat, "{label} double-not");
    }
}

macro_rules! reg {
    ($S:ty, $reg:ident) => {
        check_mask_reg::<<$S as Simd>::$reg>(&harness::label::<$S>(stringify!($reg)))
    };
}

/// The same `MaskRegister` primitives on the 3-lane `ReducedRegister` mask types
/// (`set`/`test`/`new_mask`/`native_bitmask`/`from_mask`/bitwise), which the
/// native-width suite above doesn't reach.
macro_rules! reg3 {
    ($S:ty, $reg:ident) => {
        check_mask_reg::<<$S as Simd3A>::$reg>(&harness::label::<$S>(stringify!($reg)))
    };
}

// ---------------------------------------------------------------------------
// The bespoke 4-lane `Mask` API suite, generic over the backend.
// ---------------------------------------------------------------------------

// read a float mask back into 4 bools
#[inline(always)]
fn bf<S: Simd>(m: Mask<S::f32x4>) -> [bool; 4] {
    let g = m.select(Vector::<S::f32x4>::ONE, Vector::<S::f32x4>::ZERO).into_array();
    [g[0] != 0.0, g[1] != 0.0, g[2] != 0.0, g[3] != 0.0]
}
#[inline(always)]
fn bi<S: Simd>(m: Mask<S::i32x4>) -> [bool; 4] {
    let g = m.select(Vector::<S::i32x4>::ONE, Vector::<S::i32x4>::ZERO).into_array();
    [g[0] != 0, g[1] != 0, g[2] != 0, g[3] != 0]
}
// masks with known patterns
#[inline(always)]
fn ma<S: Simd>() -> Mask<S::f32x4> {
    Vector::<S::f32x4>::new([1.0, 2.0, 3.0, 4.0]).cmp_lt(Vector::<S::f32x4>::new([4.0, 3.0, 2.0, 1.0]))
} // [T,T,F,F]
#[inline(always)]
fn mb<S: Simd>() -> Mask<S::f32x4> {
    Vector::<S::f32x4>::new([2.0, 2.0, 2.0, 2.0]).cmp_lt(Vector::<S::f32x4>::new([1.0, 3.0, 1.0, 3.0]))
} // [F,T,F,T]

const A: [bool; 4] = [true, true, false, false];
const B: [bool; 4] = [false, true, false, true];

for_each_backend! {
    fn reg_f32x4<S: Simd>() { reg!(S, f32x4) }
    fn reg_f32x8<S: Simd>() { reg!(S, f32x8) }
    fn reg_f32x16<S: Simd>() { reg!(S, f32x16) }
    fn reg_f64x2<S: Simd>() { reg!(S, f64x2) }
    fn reg_f64x4<S: Simd>() { reg!(S, f64x4) }
    fn reg_f64x8<S: Simd>() { reg!(S, f64x8) }
    fn reg_i32x4<S: Simd>() { reg!(S, i32x4) }
    fn reg_i32x8<S: Simd>() { reg!(S, i32x8) }
    fn reg_i32x16<S: Simd>() { reg!(S, i32x16) }
    fn reg_i64x2<S: Simd>() { reg!(S, i64x2) }
    fn reg_i64x4<S: Simd>() { reg!(S, i64x4) }
    fn reg_i64x8<S: Simd>() { reg!(S, i64x8) }
    fn reg_u32x4<S: Simd>() { reg!(S, u32x4) }
    fn reg_u32x8<S: Simd>() { reg!(S, u32x8) }
    fn reg_u32x16<S: Simd>() { reg!(S, u32x16) }
    fn reg_u64x2<S: Simd>() { reg!(S, u64x2) }
    fn reg_u64x4<S: Simd>() { reg!(S, u64x4) }
    fn reg_u64x8<S: Simd>() { reg!(S, u64x8) }
    fn reg_i16x8<S: Simd>() { reg!(S, i16x8) }
    fn reg_u16x16<S: Simd>() { reg!(S, u16x16) }
    fn reg_i8x16<S: Simd>() { reg!(S, i8x16) }
    fn reg_u8x8<S: Simd>() { reg!(S, u8x8) }

    fn reduced_f32x3a<S: Simd3A>() { reg3!(S, f32x3A) }
    fn reduced_i32x3a<S: Simd3A>() { reg3!(S, i32x3A) }
    fn reduced_u32x3a<S: Simd3A>() { reg3!(S, u32x3A) }
    fn reduced_f64x3a<S: Simd3A>() { reg3!(S, f64x3A) }
    fn reduced_i64x3a<S: Simd3A>() { reg3!(S, i64x3A) }
    fn reduced_u64x3a<S: Simd3A>() { reg3!(S, u64x3A) }

    fn patterns_and_reductions<S: Simd>() {
        assert_eq!(bf::<S>(ma::<S>()), A);
        assert_eq!(bf::<S>(mb::<S>()), B);
        // constants
        assert_eq!(bf::<S>(Mask::<S::f32x4>::TRUTHY), [true; 4]);
        assert_eq!(bf::<S>(Mask::<S::f32x4>::FALSY), [false; 4]);
        assert_eq!(bf::<S>(Mask::<S::f32x4>::default()), [false; 4]);
        // all / any / none
        assert!(Mask::<S::f32x4>::TRUTHY.all() && Mask::<S::f32x4>::TRUTHY.any() && !Mask::<S::f32x4>::TRUTHY.none());
        assert!(!Mask::<S::f32x4>::FALSY.all() && !Mask::<S::f32x4>::FALSY.any() && Mask::<S::f32x4>::FALSY.none());
        assert!(!ma::<S>().all() && ma::<S>().any() && !ma::<S>().none());
    }

    fn bitwise_ops<S: Simd>() {
        let f = |op: fn(bool, bool) -> bool| -> [bool; 4] { core::array::from_fn(|i| op(A[i], B[i])) };
        assert_eq!(bf::<S>(ma::<S>() & mb::<S>()), f(|a, b| a & b));
        assert_eq!(bf::<S>(ma::<S>() | mb::<S>()), f(|a, b| a | b));
        assert_eq!(bf::<S>(ma::<S>() ^ mb::<S>()), f(|a, b| a ^ b));
        assert_eq!(bf::<S>(!ma::<S>()), core::array::from_fn(|i| !A[i]));
        // andnot: self & !rhs
        assert_eq!(bf::<S>(ma::<S>().bitandnot(mb::<S>())), f(|a, b| a & !b));
        // assign forms
        let mut t = ma::<S>();
        t &= mb::<S>();
        assert_eq!(bf::<S>(t), f(|a, b| a & b));
        let mut t = ma::<S>();
        t |= mb::<S>();
        assert_eq!(bf::<S>(t), f(|a, b| a | b));
        let mut t = ma::<S>();
        t ^= mb::<S>();
        assert_eq!(bf::<S>(t), f(|a, b| a ^ b));
        let mut t = ma::<S>();
        t.bitandnot_assign(mb::<S>());
        assert_eq!(bf::<S>(t), f(|a, b| a & !b));
    }

    fn ternlog<S: Simd>() {
        // 0xFE => out unless a=b=c=0  => a|b|c ; 0x80 => only a=b=c=1 => a&b&c
        let (a, b, c) = (ma::<S>(), mb::<S>(), Mask::<S::f32x4>::TRUTHY);
        assert_eq!(
            bf::<S>(<Mask<S::f32x4> as GenericMask>::ternlog::<0xFE>(a, b, c)),
            core::array::from_fn(|i| A[i] | B[i] | true)
        );
        let c0 = Mask::<S::f32x4>::FALSY;
        assert_eq!(
            bf::<S>(<Mask<S::f32x4> as GenericMask>::ternlog::<0x80>(a, b, c0)),
            core::array::from_fn(|i| A[i] & B[i] & false)
        );
        assert_eq!(
            bf::<S>(<Mask<S::f32x4> as GenericMask>::ternlog::<0x80>(a, b, Mask::<S::f32x4>::TRUTHY)),
            core::array::from_fn(|i| A[i] & B[i])
        );
    }

    fn select_swap_cast<S: Simd>() {
        // select: mask ? t : f
        let sel = ma::<S>().select(
            Vector::<S::f32x4>::new([10.0, 20.0, 30.0, 40.0]),
            Vector::<S::f32x4>::new([1.0, 2.0, 3.0, 4.0]),
        );
        assert_eq!(sel.into_array().as_slice(), &[10.0, 20.0, 3.0, 4.0]);

        // swap: exchange lanes where mask is true
        let mut x = Vector::<S::f32x4>::new([1.0, 2.0, 3.0, 4.0]);
        let mut y = Vector::<S::f32x4>::new([5.0, 6.0, 7.0, 8.0]);
        ma::<S>().swap(&mut x, &mut y); // swap lanes 0,1
        assert_eq!(x.into_array().as_slice(), &[5.0, 6.0, 3.0, 4.0]);
        assert_eq!(y.into_array().as_slice(), &[1.0, 2.0, 7.0, 8.0]);

        // cast: f32 mask -> i32 mask preserves the booleans
        let mi: Mask<S::i32x4> = ma::<S>().cast();
        assert_eq!(bi::<S>(mi), A);
    }

    fn interleave_and_bitmask<S: Simd>() {
        // order-preserving interleave: lo=[a0,b0,a1,b1], hi=[a2,b2,a3,b3]
        let (lo, hi) = ma::<S>().interleave(mb::<S>());
        assert_eq!(bf::<S>(lo), [A[0], B[0], A[1], B[1]]);
        assert_eq!(bf::<S>(hi), [A[2], B[2], A[3], B[3]]);

        // native_bitmask (lane 0 = least significant), when available
        if let Some(bm) = ma::<S>().native_bitmask() {
            let want: u64 = A
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &t)| acc | ((t as u64) << i));
            assert_eq!(bm & 0xF, want);
        }
    }

    fn conversions_and_debug<S: Simd>() {
        // From<bool>
        assert!(Mask::<S::f32x4>::from(true).all());
        assert!(Mask::<S::f32x4>::from(false).none());
        // From<Vector>: nonzero lanes -> true
        let m: Mask<S::f32x4> = Mask::<S::f32x4>::from(Vector::<S::f32x4>::new([0.0, 1.0, 0.0, 2.0]));
        assert_eq!(bf::<S>(m), [false, true, false, true]);
        // Debug doesn't panic and reflects the lanes
        let s = format!("{:?}", ma::<S>());
        assert!(s.contains("Mask") && s.contains("true") && s.contains("false"));
    }
}
