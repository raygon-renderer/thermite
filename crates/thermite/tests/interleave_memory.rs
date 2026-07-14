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
    all(feature = "neon", target_arch = "aarch64")
))]

mod harness;

use generic_array::typenum::Unsigned;

use thermite::register::{CoreRegister, Register};
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

/// Round-trip + reference check for one register type at one `N`.
macro_rules! check {
    ($label:expr, $reg:ty, $e:ty, $n:literal) => {{
        type R = $reg;
        let lanes = <<R as CoreRegister>::Lanes as Unsigned>::USIZE;
        let total = $n * lanes;

        // Deterministic, distinct values; +1 slot so we can also run misaligned.
        // Kept under 256 so every lane is exactly representable in u8 as well as
        // in f32/f64 - the point of the test is lane ROUTING, not arithmetic.
        let src: Vec<$e> = (0..total + 1).map(|i| ((i % 251) + 1) as $e).collect();

        for offset in [0usize, 1] {
            let base = unsafe { src.as_ptr().add(offset) };

            // --- load_deinterleaved vs the scalar reference ---
            let got = unsafe { <R as Register>::load_deinterleaved::<$n>(base) };

            for j in 0..$n {
                let lanes_got = R::as_slice(&got[j]);
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
                <R as Register>::store_interleaved::<$n>(out.as_mut_ptr().add(offset), got);
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
/// portable path is a mixed-radix stage engine: a radix-2 butterfly and
/// radix-3 rounds cover the `2^a * 3^b` part of `N`, and a permute+blend
/// gather stage handles any leftover factor. NEON has native `LD2`/`LD3`/`LD4`
/// only for 2..=4, so the larger `N` also exercise the portable fallbacks
/// *behind* a backend that special-cases the small cases.
macro_rules! check_all_n {
    ($label:expr, $reg:ty, $e:ty) => {{
        check!($label, $reg, $e, 1);
        check!($label, $reg, $e, 2);
        check!($label, $reg, $e, 3); // one radix-3 round
        check!($label, $reg, $e, 4);
        check!($label, $reg, $e, 5); // pure gather (prime leftover)
        check!($label, $reg, $e, 6); // mixed radix (3 * 2), past LD4
        check!($label, $reg, $e, 8); // butterfly, past LD4
        check!($label, $reg, $e, 10); // gather stage + butterfly (5 * 2)
        check!($label, $reg, $e, 12); // mixed radix (3 * 4), past LD4
        check!($label, $reg, $e, 15); // gather stage + radix-3 round (5 * 3)
    }};
}

macro_rules! suite {
    ($modname:ident, $backend:ty, $bl:expr) => {
        mod $modname {
            use super::*;

            #[test]
            fn f32_streams() {
                check_all_n!(concat!($bl, " f32x4"), <$backend as Simd>::f32x4, f32);
                check_all_n!(concat!($bl, " f32x8"), <$backend as Simd>::f32x8, f32);
            }

            #[test]
            fn f64_streams() {
                check_all_n!(concat!($bl, " f64x2"), <$backend as Simd>::f64x2, f64);
            }

            #[test]
            fn int_streams() {
                check_all_n!(concat!($bl, " i32x4"), <$backend as Simd>::i32x4, i32);
                check_all_n!(concat!($bl, " u16x8"), <$backend as Simd>::u16x8, u16);
                check_all_n!(concat!($bl, " u8x16"), <$backend as Simd>::u8x16, u8);
            }
        }
    };
}

// The scalar backend is the oracle everything else must match.
suite!(scalar, Scalar, "scalar");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    suite!(v2, X86V2, "x86_v2");
    suite!(v3, X86V3, "x86_v3");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    suite!(wasm, Wasm, "wasm");
}

// NEON: LD2/LD3/LD4 + ST2/ST3/ST4.
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    suite!(neon, Neon, "neon");
}
