//! `load_deinterleaved` / `store_interleaved`: the AoS <-> SoA memory ops.
//!
//! ARM lowers these to the structural `LD2`/`LD3`/`LD4` + `ST2`/`ST3`/`ST4`
//! instructions (the de-interleave happens in the load unit); every other
//! backend takes the portable default (contiguous loads + a cross-register
//! permute). Both must agree with the obvious scalar reference:
//!
//!   load_deinterleaved:  out[j][lane] == ptr[lane * N + j]
//!   store_interleaved:   ptr[lane * N + j] == values[j][lane]
//!
//! and the two must round-trip. Deliberately runs over UNALIGNED pointers too:
//! AArch64 structural loads have no alignment requirement (unlike ARMv7's
//! `:64`/`:128` qualifiers), and the portable default uses unaligned loads.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;

use thermite::register::{CoreRegister, Register};
use thermite::simd::Simd;

/// Round-trip + reference check for one register type at one `N`.
macro_rules! check {
    ($label:expr, $reg:ty, $e:ty, $n:literal) => {{
        let lanes = <<$reg as CoreRegister>::Lanes as Unsigned>::USIZE;
        let total = $n * lanes;

        // Deterministic, distinct values; +1 slot so we can also run misaligned.
        // Kept under 256 so every lane is exactly representable in u8 as well as
        // in f32/f64 - the point of the test is lane ROUTING, not arithmetic.
        let src: Vec<$e> = (0..total + 1).map(|i| ((i % 251) + 1) as $e).collect();

        for offset in [0usize, 1] {
            let base = unsafe { src.as_ptr().add(offset) };

            // --- load_deinterleaved vs the scalar reference ---
            let got = unsafe { <$reg as Register>::load_deinterleaved::<$n>(base) };

            for j in 0..$n {
                let lanes_got = <$reg>::as_slice(&got[j]);
                for lane in 0..lanes {
                    let want = src[offset + lane * $n + j];
                    assert_eq!(
                        lanes_got[lane], want,
                        "{} [load_deinterleaved::<{}> off={}]: stream {} lane {}",
                        $label, $n, offset, j, lane
                    );
                }
            }

            // --- store_interleaved is its exact inverse ---
            let mut out: Vec<$e> = vec![0 as $e; total + 1];
            unsafe {
                <$reg as Register>::store_interleaved::<$n>(out.as_mut_ptr().add(offset), got);
            }
            assert_eq!(
                &out[offset..offset + total],
                &src[offset..offset + total],
                "{} [store_interleaved::<{}> off={}]: round-trip mismatch",
                $label,
                $n,
                offset
            );
        }
    }};
}

/// Every `N` we care about for one register type. `N` is unbounded, and the
/// portable lowering factors it into radix-2/3/4 rounds plus a gather stage.
macro_rules! check_all_n {
    ($S:ty, $reg:ident, $e:ty) => {{
        let label = harness::label::<$S>(stringify!($reg));
        check!(label, <$S as Simd>::$reg, $e, 1);
        check!(label, <$S as Simd>::$reg, $e, 2);
        check!(label, <$S as Simd>::$reg, $e, 3); // one radix-3 round
        check!(label, <$S as Simd>::$reg, $e, 4);
        check!(label, <$S as Simd>::$reg, $e, 5); // pure gather (prime leftover)
        check!(label, <$S as Simd>::$reg, $e, 6); // mixed radix (3 * 2), past LD4
        check!(label, <$S as Simd>::$reg, $e, 7); // pure gather, prime > LANES/2
        check!(label, <$S as Simd>::$reg, $e, 8); // butterfly, past LD4
        check!(label, <$S as Simd>::$reg, $e, 9); // two radix-3 rounds (3 * 3)
        check!(label, <$S as Simd>::$reg, $e, 10); // gather stage + butterfly (5 * 2)
        check!(label, <$S as Simd>::$reg, $e, 12); // mixed radix (3 * 4), past LD4
        check!(label, <$S as Simd>::$reg, $e, 15); // gather stage + radix-3 round (5 * 3)
        check!(label, <$S as Simd>::$reg, $e, 20); // gather stage + two butterflies (5 * 4)
    }};
}

for_each_backend! {
    fn f32_streams<S: Simd>() {
        check_all_n!(S, f32x4, f32);
        check_all_n!(S, f32x8, f32);
        check_all_n!(S, f32x16, f32);
    }
    fn f64_streams<S: Simd>() {
        check_all_n!(S, f64x2, f64);
        check_all_n!(S, f64x4, f64);
        check_all_n!(S, f64x8, f64);
    }
    fn int_streams<S: Simd>() {
        check_all_n!(S, i32x4, i32);
        check_all_n!(S, i32x8, i32);
        check_all_n!(S, u16x8, u16);
        check_all_n!(S, u16x16, u16);
        check_all_n!(S, u8x16, u8);
    }
}
