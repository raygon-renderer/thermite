//! Sum-of-absolute-differences (`Sad16`/`Sad32`/`Sad64`) correctness.
//!
//! The scalar backend is the oracle here, checked against a naive per-group sum. This
//! validates the generic SWAR cascade and the same-width reinterpret it is built on --
//! including that the transmute-based `ArrayRegister` reinterpret groups the same bytes
//! the native (identity-bitcast) backends do.

use thermite::prelude::*;
use thermite::register::array::ArrayRegister;
use thermite::vector::{Sad16Vector, Sad32Vector, Sad64Vector};

type U8x16 = Vector<ArrayRegister<u8, 16>>;
type U16x8 = Vector<ArrayRegister<u16, 8>>;
type U32x4 = Vector<ArrayRegister<u32, 4>>;
type U64x2 = Vector<ArrayRegister<u64, 2>>;

/// Naive reference: sum `|a - b|` over each aligned group of `GROUP` bytes.
fn reference<const GROUP: usize>(a: &[u8; 16], b: &[u8; 16]) -> Vec<u64> {
    a.chunks(GROUP)
        .zip(b.chunks(GROUP))
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| u64::from(p.abs_diff(*q))).sum())
        .collect()
}

fn vectors(a: &[u8; 16], b: &[u8; 16]) -> (U8x16, U8x16) {
    (U8x16::from_slice(a), U8x16::from_slice(b))
}

/// Deterministic pseudo-random byte patterns plus the saturating edge cases.
fn cases() -> Vec<([u8; 16], [u8; 16])> {
    let mut out = vec![
        ([0u8; 16], [0u8; 16]),
        ([255u8; 16], [0u8; 16]),   // every group at its maximum
        ([0u8; 16], [255u8; 16]),   // ... and with the operands swapped
        ([128u8; 16], [127u8; 16]), // adjacent values, |diff| == 1
    ];

    // A few scrambled patterns; a simple LCG keeps this reproducible without a dep.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..32 {
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        for i in 0..16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            a[i] = (state >> 33) as u8;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b[i] = (state >> 33) as u8;
        }
        out.push((a, b));
    }
    out
}

#[test]
fn sad64_matches_reference() {
    for (a, b) in cases() {
        let (va, vb) = vectors(&a, &b);
        let got: U64x2 = va.sad64(vb);
        let want = reference::<8>(&a, &b);
        for (lane, w) in want.iter().enumerate() {
            assert_eq!(got.extractv(lane), *w, "sad64 lane {lane} for {a:?} vs {b:?}");
        }
    }
}

#[test]
fn sad32_matches_reference() {
    for (a, b) in cases() {
        let (va, vb) = vectors(&a, &b);
        let got: U32x4 = va.sad32(vb);
        let want = reference::<4>(&a, &b);
        for (lane, w) in want.iter().enumerate() {
            assert_eq!(u64::from(got.extractv(lane)), *w, "sad32 lane {lane}");
        }
    }
}

#[test]
fn sad16_matches_reference() {
    for (a, b) in cases() {
        let (va, vb) = vectors(&a, &b);
        let got: U16x8 = va.sad16(vb);
        let want = reference::<2>(&a, &b);
        for (lane, w) in want.iter().enumerate() {
            assert_eq!(u64::from(got.extractv(lane)), *w, "sad16 lane {lane}");
        }
    }
}

/// Each grouping must total the same across the whole register -- they only differ in how
/// finely the sum is partitioned.
#[test]
fn groupings_agree_on_total() {
    for (a, b) in cases() {
        let (va, vb) = vectors(&a, &b);
        let total: u64 = reference::<1>(&a, &b).iter().sum();

        let s16: U16x8 = va.sad16(vb);
        let s32: U32x4 = va.sad32(vb);
        let s64: U64x2 = va.sad64(vb);

        assert_eq!((0..8).map(|i| u64::from(s16.extractv(i))).sum::<u64>(), total);
        assert_eq!((0..4).map(|i| u64::from(s32.extractv(i))).sum::<u64>(), total);
        assert_eq!((0..2).map(|i| s64.extractv(i)).sum::<u64>(), total);
    }
}

/// Exercise the *native* path (x86 `psadbw` on v2/v3, and whatever the running CPU
/// dispatches to) and check it against the same reference. The tests above only reach the
/// scalar backend, since native registers need a `target_feature` context.
#[test]
fn native_dispatch_matches_reference() {
    for (av, bv) in cases() {
        let want: u64 = reference::<1>(&av, &bv).iter().sum();
        // `dispatch_dyn!`'s closure form captures by name, so the bindings it names must
        // already be the parameter types.
        let a: &[u8] = &av;
        let b: &[u8] = &bv;

        let got64 = thermite::dispatch_dyn!(for<S> |a: &[u8], b: &[u8]| -> u64 {
            u8x16::from_slice(a).sad64(u8x16::from_slice(b)).sum_elements()
        });
        assert_eq!(got64, want, "native sad64 for {av:?} vs {bv:?}");

        let got32 = thermite::dispatch_dyn!(for<S> |a: &[u8], b: &[u8]| -> u64 {
            u64::from(u8x16::from_slice(a).sad32(u8x16::from_slice(b)).sum_elements())
        });
        assert_eq!(got32, want, "native sad32");

        let got16 = thermite::dispatch_dyn!(for<S> |a: &[u8], b: &[u8]| -> u64 {
            u64::from(u8x16::from_slice(a).sad16(u8x16::from_slice(b)).sum_elements())
        });
        assert_eq!(got16, want, "native sad16");
    }
}

/// The sub-native byte ladder (`u8x8`/`u8x4`/`u8x2`) takes the lane-wise path, including
/// the partial-group cases where the register holds fewer bytes than one group and the
/// single output lane must sum everything it has.
#[test]
fn sub_native_ladder() {
    type U8x8 = Vector<ArrayRegister<u8, 8>>;
    type U8x4 = Vector<ArrayRegister<u8, 4>>;
    type U8x2 = Vector<ArrayRegister<u8, 2>>;

    for (av, bv) in cases() {
        // 8 lanes: two 2-byte groups... and exactly one full 8-byte group.
        let (a8, b8) = (U8x8::from_slice(&av[..8]), U8x8::from_slice(&bv[..8]));
        let want8: u64 = reference::<1>(&av, &bv)[..8].iter().sum();
        let s16: Vector<ArrayRegister<u16, 4>> = a8.sad16(b8);
        let s32: Vector<ArrayRegister<u32, 2>> = a8.sad32(b8);
        assert_eq!((0..4).map(|i| u64::from(s16.extractv(i))).sum::<u64>(), want8);
        assert_eq!((0..2).map(|i| u64::from(s32.extractv(i))).sum::<u64>(), want8);
        // One lane, one full group.
        let s64_8: Vector<u64> = a8.sad64(b8);
        assert_eq!(s64_8.extractv(0), want8);

        // 4 lanes: sad64's group is wider than the register -> partial, sums all 4.
        let (a4, b4) = (U8x4::from_slice(&av[..4]), U8x4::from_slice(&bv[..4]));
        let want4: u64 = reference::<1>(&av, &bv)[..4].iter().sum();
        let s16: Vector<ArrayRegister<u16, 2>> = a4.sad16(b4);
        let s32: Vector<u32> = a4.sad32(b4);
        let s64: Vector<u64> = a4.sad64(b4);
        assert_eq!((0..2).map(|i| u64::from(s16.extractv(i))).sum::<u64>(), want4);
        assert_eq!(u64::from(s32.extractv(0)), want4);
        assert_eq!(s64.extractv(0), want4);

        // 2 lanes: below every grouping -> all three sum the whole register.
        let (a2, b2) = (U8x2::from_slice(&av[..2]), U8x2::from_slice(&bv[..2]));
        let want2: u64 = reference::<1>(&av, &bv)[..2].iter().sum();
        let t16: Vector<u16> = a2.sad16(b2);
        let t32: Vector<u32> = a2.sad32(b2);
        let t64: Vector<u64> = a2.sad64(b2);
        assert_eq!(u64::from(t16.extractv(0)), want2);
        assert_eq!(u64::from(t32.extractv(0)), want2);
        assert_eq!(t64.extractv(0), want2);
    }
}

/// SAD generalises past `u8`: `sad32` over `u16` pairs, `sad64` over `u16` quads and
/// `u32` pairs. Group size is always `output_bits / input_element_bits`.
#[test]
fn wider_element_inputs() {
    type U16x8 = Vector<ArrayRegister<u16, 8>>;
    type U32x4 = Vector<ArrayRegister<u32, 4>>;

    let a16: [u16; 8] = [0, 9, 40000, 7, 65535, 1, 300, 12];
    let b16: [u16; 8] = [5, 2, 100, 7, 0, 60000, 44, 12];
    let (va, vb) = (U16x8::from_slice(&a16), U16x8::from_slice(&b16));
    let d: Vec<u64> = a16.iter().zip(&b16).map(|(p, q)| u64::from(p.abs_diff(*q))).collect();

    for i in 0..4 {
        let r: Vector<ArrayRegister<u32, 4>> = va.sad32(vb);
        assert_eq!(u64::from(r.extractv(i)), d[2 * i] + d[2 * i + 1], "u16 sad32 lane {i}");
    }
    for i in 0..2 {
        let r: Vector<ArrayRegister<u64, 2>> = va.sad64(vb);
        assert_eq!(
            r.extractv(i),
            d[4 * i..4 * i + 4].iter().sum::<u64>(),
            "u16 sad64 lane {i}"
        );
    }

    let a32: [u32; 4] = [0, 4_000_000_000, 17, 1];
    let b32: [u32; 4] = [123, 5, 17, 4_294_967_295];
    let (wa, wb) = (U32x4::from_slice(&a32), U32x4::from_slice(&b32));
    let e: Vec<u64> = a32.iter().zip(&b32).map(|(p, q)| u64::from(p.abs_diff(*q))).collect();
    for i in 0..2 {
        let r: Vector<ArrayRegister<u64, 2>> = wa.sad64(wb);
        assert_eq!(r.extractv(i), e[2 * i] + e[2 * i + 1], "u32 sad64 lane {i}");
    }
}

/// A `u32` pair sum needs 33 bits, so the cheap add-then-mask fold used for the narrower
/// cascades would silently truncate here. Pin the widest possible input.
#[test]
fn u32_sad64_does_not_truncate() {
    type U32x4 = Vector<ArrayRegister<u32, 4>>;
    let a = U32x4::from_slice(&[u32::MAX, u32::MAX, u32::MAX, 0]);
    let b = U32x4::from_slice(&[0, 0, 0, u32::MAX]);

    let got: Vector<ArrayRegister<u64, 2>> = a.sad64(b);
    assert_eq!(got.extractv(0), 2 * u64::from(u32::MAX));
    assert_eq!(got.extractv(1), 2 * u64::from(u32::MAX));
}

/// The native paths for the wider inputs, through whichever backend this CPU dispatches to.
#[test]
fn wider_native_dispatch() {
    let a16: Vec<u16> = (0..8).map(|i| (i * 8191) as u16).collect();
    let b16: Vec<u16> = (0..8).map(|i| (i * 7919 + 3) as u16).collect();
    let want16: u64 = a16.iter().zip(&b16).map(|(p, q)| u64::from(p.abs_diff(*q))).sum();
    let (a, b) = (&a16[..], &b16[..]);

    let got = thermite::dispatch_dyn!(for<S> |a: &[u16], b: &[u16]| -> u64 {
        u64::from(u16x8::from_slice(a).sad32(u16x8::from_slice(b)).sum_elements())
    });
    assert_eq!(got, want16, "native u16 sad32");

    let got = thermite::dispatch_dyn!(for<S> |a: &[u16], b: &[u16]| -> u64 {
        u16x8::from_slice(a).sad64(u16x8::from_slice(b)).sum_elements()
    });
    assert_eq!(got, want16, "native u16 sad64");

    let a32: Vec<u32> = (0..4).map(|i| i * 1_000_000_007).collect();
    let b32: Vec<u32> = (0..4).map(|i| i * 17 + 5).collect();
    let want32: u64 = a32.iter().zip(&b32).map(|(p, q)| u64::from(p.abs_diff(*q))).sum();
    let (a, b) = (&a32[..], &b32[..]);

    let got = thermite::dispatch_dyn!(for<S> |a: &[u32], b: &[u32]| -> u64 {
        u32x4::from_slice(a).sad64(u32x4::from_slice(b)).sum_elements()
    });
    assert_eq!(got, want32, "native u32 sad64");
}

/// The accumulating forms are what a blocked SAD loop actually uses.
#[test]
fn accumulate_forms() {
    let (a, b) = (&[7u8; 16], &[3u8; 16]);
    let (va, vb) = vectors(a, b);

    let mut acc64 = U64x2::ZERO;
    let mut acc32 = U32x4::ZERO;
    for _ in 0..1000 {
        acc64 = va.sad64_accum(acc64, vb);
        acc32 = va.sad32_accum(acc32, vb);
    }

    // |7 - 3| == 4 per byte: 32 per 8-byte group, 16 per 4-byte group.
    assert_eq!(acc64.extractv(0), 32 * 1000);
    assert_eq!(acc32.extractv(0), 16 * 1000);
}

/// Native-width SAD via the `xN` ladder. There is no fixed-width `u8x32` slot, but on
/// AVX2 `u8xN` *is* the 256-bit byte register, so this is what reaches
/// `_mm256_sad_epu8`; on the other backends it is the 128-bit one, and on scalar it is a
/// single lane. The lane ratios (`u8xN` -> half/quarter/eighth) hold everywhere, so one
/// generic body covers them all.
#[test]
fn native_width_xn_ladder() {
    let a: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
    let b: Vec<u8> = (0..64u32).map(|i| (i * 91 + 5) as u8).collect();
    let want: u64 = a.iter().zip(&b).map(|(p, q)| u64::from(p.abs_diff(*q))).sum();
    let (a, b) = (&a[..], &b[..]);

    // Each closure consumes exactly one native register's worth of lanes, so compare
    // against the same prefix rather than the whole buffer.
    let (got, lanes) = thermite::dispatch_dyn!(for<S> |a: &[u8], b: &[u8]| -> (u64, usize) {
        let n = <u8xN as GenericVector>::LANES;
        (
            u8xN::from_slice(&a[..n])
                .sad64(u8xN::from_slice(&b[..n]))
                .sum_elements(),
            n,
        )
    });
    let want_n: u64 = a[..lanes]
        .iter()
        .zip(&b[..lanes])
        .map(|(p, q)| u64::from(p.abs_diff(*q)))
        .sum();
    assert_eq!(got, want_n, "u8xN sad64 over {lanes} lanes");

    let (got, lanes) = thermite::dispatch_dyn!(for<S> |a: &[u8], b: &[u8]| -> (u64, usize) {
        let n = <u8xN as GenericVector>::LANES;
        (
            u64::from(
                u8xN::from_slice(&a[..n])
                    .sad32(u8xN::from_slice(&b[..n]))
                    .sum_elements(),
            ),
            n,
        )
    });
    let want_n: u64 = a[..lanes]
        .iter()
        .zip(&b[..lanes])
        .map(|(p, q)| u64::from(p.abs_diff(*q)))
        .sum();
    assert_eq!(got, want_n, "u8xN sad32 over {lanes} lanes");

    let _ = want;
}
