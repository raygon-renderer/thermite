//! Coverage for the `Mask<R>` wrapper and `GenericMask` API (`mask.rs`, ~33%):
//! all/any/none, the bitwise operators (`&`/`|`/`^`/`!`/andnot + assign),
//! `select`/`cast`/`swap`/`ternlog`, `interleave`, `native_bitmask`, `From<bool>`/
//! `From<Vector>`, `Debug`, `Default`, and the `TRUTHY`/`FALSY` constants.
//!
//! Masks are built from comparisons (known per-lane bool patterns) and every
//! operation is checked against the booleans computed in plain Rust.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use thermite::Vector;
use thermite::mask::{GenericMask, Mask};
use thermite::prelude::*;
use thermite::register::{CoreRegister, MaskRegister, Register};
use thermite::simd::Simd;
use thermite::vector::Interleave;
use thermite::vector::ops::{BitAndNot, BitAndNotAssign};

use thermite::backend::scalar::Scalar;

/// Lane-count-agnostic coverage for the **`MaskRegister`** primitives on the mask
/// register of `R` (`new_mask`/`test`/`set`/`all`/`any`/`none`/`native_bitmask`) plus
/// `into_mask` (via `From<Vector>`) and the `GenericMask` bitwise ops — for every
/// register width, not just the 4-lane `f32x4`/`i32x4` the bespoke suite uses.
/// Masks are built from the shared `mask_patterns` corpus and checked per-lane.
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

macro_rules! reg_mask_suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;
            macro_rules! t {
                ($name:ident, $reg:ident) => {
                    #[test]
                    fn $name() {
                        check_mask_reg::<<$backend as Simd>::$reg>(concat!($bl, " ", stringify!($reg)));
                    }
                };
            }
            t!(f32x4, f32x4);
            t!(f32x8, f32x8);
            t!(f32x16, f32x16);
            t!(f64x2, f64x2);
            t!(f64x4, f64x4);
            t!(f64x8, f64x8);
            t!(i32x4, i32x4);
            t!(i32x8, i32x8);
            t!(i64x2, i64x2);
            t!(i64x4, i64x4);
            t!(u32x4, u32x4);
            t!(u32x8, u32x8);
            t!(u64x2, u64x2);
            t!(u64x4, u64x4);
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_reg {
use super::*;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
reg_mask_suite!(reg_v3, X86V3, "x86_v3");
reg_mask_suite!(reg_v2, X86V2, "x86_v2");
reg_mask_suite!(reg_v1, X86V1, "x86_v1");
}
reg_mask_suite!(reg_scalar, Scalar, "scalar");

#[cfg(target_arch = "wasm32")]
mod wasm_reg {
use super::*;
use thermite::backend::wasm::Wasm;
reg_mask_suite!(reg_wasm, Wasm, "wasm");
}

/// The same `MaskRegister` primitives on the 3-lane `ReducedRegister` mask types
/// (`set`/`test`/`new_mask`/`native_bitmask`/`from_mask`/bitwise), which the
/// native-width suite above doesn't reach.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod reduced_mask {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    use thermite::simd::Simd3A;

    macro_rules! t3 {
        ($name:ident, $backend:ty, $reg:ident, $bl:expr) => {
            #[test]
            fn $name() {
                check_mask_reg::<<$backend as Simd3A>::$reg>(concat!($bl, " ", stringify!($reg)));
            }
        };
    }
    t3!(v3_f32x3A, X86V3, f32x3A, "x86_v3");
    t3!(v3_i32x3A, X86V3, i32x3A, "x86_v3");
    t3!(v3_u64x3A, X86V3, u64x3A, "x86_v3");
    t3!(v2_f32x3A, X86V2, f32x3A, "x86_v2");
    t3!(v2_i64x3A, X86V2, i64x3A, "x86_v2");
    t3!(v1_f32x3A, X86V1, f32x3A, "x86_v1");
    t3!(v1_i32x3A, X86V1, i32x3A, "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod reduced_mask_wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    use thermite::simd::Simd3A;

    macro_rules! t3 {
        ($name:ident, $backend:ty, $reg:ident, $bl:expr) => {
            #[test]
            fn $name() {
                check_mask_reg::<<$backend as Simd3A>::$reg>(concat!($bl, " ", stringify!($reg)));
            }
        };
    }
    t3!(wasm_f32x3A, Wasm, f32x3A, "wasm");
    t3!(wasm_i64x3A, Wasm, i64x3A, "wasm");
}

macro_rules! mask_suite {
    ($mod:ident, $backend:ty) => {
        mod $mod {
            use super::*;
            type VF = Vector<<$backend as Simd>::f32x4>;
            type VI = Vector<<$backend as Simd>::i32x4>;
            type MF = Mask<<$backend as Simd>::f32x4>;
            type MI = Mask<<$backend as Simd>::i32x4>;

            // read a float mask back into 4 bools
            fn bf(m: MF) -> [bool; 4] {
                let g = m.select(VF::ONE, VF::ZERO).into_array();
                [g[0] != 0.0, g[1] != 0.0, g[2] != 0.0, g[3] != 0.0]
            }
            fn bi(m: MI) -> [bool; 4] {
                let g = m.select(VI::ONE, VI::ZERO).into_array();
                [g[0] != 0, g[1] != 0, g[2] != 0, g[3] != 0]
            }
            // masks with known patterns
            fn ma() -> MF {
                VF::new([1.0, 2.0, 3.0, 4.0]).cmp_lt(VF::new([4.0, 3.0, 2.0, 1.0]))
            } // [T,T,F,F]
            fn mb() -> MF {
                VF::new([2.0, 2.0, 2.0, 2.0]).cmp_lt(VF::new([1.0, 3.0, 1.0, 3.0]))
            } // [F,T,F,T]

            const A: [bool; 4] = [true, true, false, false];
            const B: [bool; 4] = [false, true, false, true];

            #[test]
            fn patterns_and_reductions() {
                assert_eq!(bf(ma()), A);
                assert_eq!(bf(mb()), B);
                // constants
                assert_eq!(bf(MF::TRUTHY), [true; 4]);
                assert_eq!(bf(MF::FALSY), [false; 4]);
                assert_eq!(bf(MF::default()), [false; 4]);
                // all / any / none
                assert!(MF::TRUTHY.all() && MF::TRUTHY.any() && !MF::TRUTHY.none());
                assert!(!MF::FALSY.all() && !MF::FALSY.any() && MF::FALSY.none());
                assert!(!ma().all() && ma().any() && !ma().none());
            }

            #[test]
            fn bitwise_ops() {
                let f = |op: fn(bool, bool) -> bool| -> [bool; 4] { core::array::from_fn(|i| op(A[i], B[i])) };
                assert_eq!(bf(ma() & mb()), f(|a, b| a & b));
                assert_eq!(bf(ma() | mb()), f(|a, b| a | b));
                assert_eq!(bf(ma() ^ mb()), f(|a, b| a ^ b));
                assert_eq!(bf(!ma()), core::array::from_fn(|i| !A[i]));
                // andnot: self & !rhs
                assert_eq!(bf(ma().bitandnot(mb())), f(|a, b| a & !b));
                // assign forms
                let mut t = ma();
                t &= mb();
                assert_eq!(bf(t), f(|a, b| a & b));
                let mut t = ma();
                t |= mb();
                assert_eq!(bf(t), f(|a, b| a | b));
                let mut t = ma();
                t ^= mb();
                assert_eq!(bf(t), f(|a, b| a ^ b));
                let mut t = ma();
                t.bitandnot_assign(mb());
                assert_eq!(bf(t), f(|a, b| a & !b));
            }

            #[test]
            fn ternlog() {
                let (a, b, c) = (ma(), mb(), MF::TRUTHY);
                // 0xFE => out unless a=b=c=0  => a|b|c ; 0x80 => only a=b=c=1 => a&b&c
                assert_eq!(
                    bf(<MF as GenericMask>::ternlog::<0xFE>(a, b, c)),
                    core::array::from_fn(|i| A[i] | B[i] | true)
                );
                let c0 = MF::FALSY;
                assert_eq!(
                    bf(<MF as GenericMask>::ternlog::<0x80>(a, b, c0)),
                    core::array::from_fn(|i| A[i] & B[i] & false)
                );
                assert_eq!(
                    bf(<MF as GenericMask>::ternlog::<0x80>(a, b, MF::TRUTHY)),
                    core::array::from_fn(|i| A[i] & B[i])
                );
            }

            #[test]
            fn select_swap_cast() {
                // select: mask ? t : f
                let sel = ma().select(VF::new([10.0, 20.0, 30.0, 40.0]), VF::new([1.0, 2.0, 3.0, 4.0]));
                assert_eq!(sel.into_array().as_slice(), &[10.0, 20.0, 3.0, 4.0]);

                // swap: exchange lanes where mask is true
                let mut x = VF::new([1.0, 2.0, 3.0, 4.0]);
                let mut y = VF::new([5.0, 6.0, 7.0, 8.0]);
                ma().swap(&mut x, &mut y); // swap lanes 0,1
                assert_eq!(x.into_array().as_slice(), &[5.0, 6.0, 3.0, 4.0]);
                assert_eq!(y.into_array().as_slice(), &[1.0, 2.0, 7.0, 8.0]);

                // cast: f32 mask -> i32 mask preserves the booleans
                let mi: MI = ma().cast();
                assert_eq!(bi(mi), A);
            }

            #[test]
            fn interleave_and_bitmask() {
                // order-preserving interleave: lo=[a0,b0,a1,b1], hi=[a2,b2,a3,b3]
                let (lo, hi) = ma().interleave(mb());
                assert_eq!(bf(lo), [A[0], B[0], A[1], B[1]]);
                assert_eq!(bf(hi), [A[2], B[2], A[3], B[3]]);

                // native_bitmask (lane 0 = least significant), when available
                if let Some(bm) = ma().native_bitmask() {
                    let want: u64 = A
                        .iter()
                        .enumerate()
                        .fold(0, |acc, (i, &t)| acc | ((t as u64) << i));
                    assert_eq!(bm & 0xF, want);
                }
            }

            #[test]
            fn conversions_and_debug() {
                // From<bool>
                assert!(MF::from(true).all());
                assert!(MF::from(false).none());
                // From<Vector>: nonzero lanes -> true
                let m: MF = MF::from(VF::new([0.0, 1.0, 0.0, 2.0]));
                assert_eq!(bf(m), [false, true, false, true]);
                // Debug doesn't panic and reflects the lanes
                let s = format!("{:?}", ma());
                assert!(s.contains("Mask") && s.contains("true") && s.contains("false"));
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
use super::*;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
mask_suite!(v3, X86V3);
mask_suite!(v2, X86V2);
}
mask_suite!(scalar, Scalar);

#[cfg(target_arch = "wasm32")]
mod wasm {
use super::*;
use thermite::backend::wasm::Wasm;
mask_suite!(wasm, Wasm);
}
