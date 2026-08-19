//! Morton (Z-order curve) encode/decode correctness.
//!
//! Checks [`UnsignedIntegerRegister::morton`] / `reverse_morton` for every
//! implemented backend and width against an independent pure-Rust per-lane
//! bit-interleave oracle, plus a `reverse(morton(x)) == x` roundtrip. This
//! exercises the generic shift/mask cascade, the CLMUL `N == 2` fast path on
//! u64-lane v3 registers (default `avx2-pclmul`), the x86 `pshufb` and wasm
//! `i8x16.swizzle` nibble-LUT paths on u16/u32 lanes.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use thermite::register::{CoreRegister, Storage, UnsignedIntegerRegister as _};
use thermite::simd::Simd;

use harness::{corpus, make_array, read, rng};

/// Test `morton::<N>` + `reverse_morton::<N>` for one register type against a
/// pure-Rust per-lane bit-interleave oracle (bit `i` of coord `d` -> output bit
/// `i*N + d`), plus a roundtrip check that decode recovers the masked inputs.
macro_rules! morton_check {
    ($label:expr, $ut:ty, $e:ty, $n:literal) => {{
        const N: usize = $n;
        let mut rng = rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let width = (core::mem::size_of::<$e>() * 8) as u32;
        let bpl = { let b = width / N as u32; if b == 0 { 1 } else { b } }; // usable bits per coord
        let mask: u64 = if bpl >= 64 { u64::MAX } else { (1u64 << bpl) - 1 };

        // N independent input corpora (one per coordinate axis).
        let corpora: Vec<Vec<Vec<$e>>> = (0..N).map(|_| corpus::<$e>(lanes, &mut rng)).collect();
        let len = corpora.iter().map(Vec::len).min().unwrap();

        for idx in 0..len {
            let inputs: [Storage<$ut>; N] = core::array::from_fn(|d| make_array::<$ut>(&corpora[d][idx]));
            let code = <$ut>::morton::<N>(inputs);
            let got = read::<$ut>(&code);

            for lane in 0..lanes {
                let mut want: u64 = 0;
                for d in 0..N {
                    let c = (corpora[d][idx][lane] as u64) & mask;
                    for i in 0..bpl {
                        let pos = i * N as u32 + d as u32;
                        if (c >> i) & 1 == 1 && pos < width {
                            want |= 1u64 << pos;
                        }
                    }
                }
                assert_eq!(got[lane] as u64, want, "{} morton lane {}", $label, lane);
            }

            // roundtrip: reverse_morton(morton(x)) == x masked to usable bits
            let back = <$ut>::reverse_morton::<N>(code);
            for d in 0..N {
                let backd = read::<$ut>(&back[d]);
                for lane in 0..lanes {
                    let want = (corpora[d][idx][lane] as u64) & mask;
                    assert_eq!(backd[lane] as u64, want, "{} reverse d{} lane {}", $label, d, lane);
                }
            }
        }
    }};
}

macro_rules! morton_suite {
    ($name:ident, $ut:ty, $e:ty, $label:expr) => {
        #[test]
        fn $name() {
            morton_check!(concat!($label, " N=1"), $ut, $e, 1);
            morton_check!(concat!($label, " N=2"), $ut, $e, 2);
            morton_check!(concat!($label, " N=3"), $ut, $e, 3);
            morton_check!(concat!($label, " N=4"), $ut, $e, 4);
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

    // u64 v3 carries the CLMUL `N == 2` fast path, u16/u32 use the pshufb LUT, and
    // u64x2/v1 and all N != 2 exercise the cascade.
    morton_suite!(v3_u64x2, <X86V3 as Simd>::u64x2, u64, "v3 u64x2");
    morton_suite!(v3_u64x4, <X86V3 as Simd>::u64x4, u64, "v3 u64x4");
    morton_suite!(v3_u32x4, <X86V3 as Simd>::u32x4, u32, "v3 u32x4");
    morton_suite!(v3_u32x8, <X86V3 as Simd>::u32x8, u32, "v3 u32x8");
    morton_suite!(v3_u16x8, <X86V3 as Simd>::u16x8, u16, "v3 u16x8");
    morton_suite!(v3_u16x16, <X86V3 as Simd>::u16x16, u16, "v3 u16x16");

    morton_suite!(v2_u64x2, <X86V2 as Simd>::u64x2, u64, "v2 u64x2");
    morton_suite!(v2_u32x4, <X86V2 as Simd>::u32x4, u32, "v2 u32x4");
    morton_suite!(v2_u16x8, <X86V2 as Simd>::u16x8, u16, "v2 u16x8");

    morton_suite!(v1_u64x2, <X86V1 as Simd>::u64x2, u64, "v1 u64x2");
    morton_suite!(v1_u32x4, <X86V1 as Simd>::u32x4, u32, "v1 u32x4");
    morton_suite!(v1_u16x8, <X86V1 as Simd>::u16x8, u16, "v1 u16x8");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    // u16/u32 use the `i8x16.swizzle` LUT; u64x2 and all N != 2 use the cascade.
    morton_suite!(wasm_u16x8, <Wasm as Simd>::u16x8, u16, "wasm u16x8");
    morton_suite!(wasm_u32x4, <Wasm as Simd>::u32x4, u32, "wasm u32x4");
    morton_suite!(wasm_u64x2, <Wasm as Simd>::u64x2, u64, "wasm u64x2");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    // u16/u32 use the `i8x16.swizzle` LUT; u64x2 and all N != 2 use the cascade.
    // (inherited from the wasm section, revisit for NEON)
    morton_suite!(neon_u16x8, <Neon as Simd>::u16x8, u16, "neon u16x8");
    morton_suite!(neon_u32x4, <Neon as Simd>::u32x4, u32, "neon u32x4");
    morton_suite!(neon_u64x2, <Neon as Simd>::u64x2, u64, "neon u64x2");
}

/// Exercise the user-facing `Vector<R>` layer (the `transmute_copy` delegation to
/// the register methods), independent of backend. Uses a scalar `Vector<u32>`.
#[test]
fn vector_layer() {
    use thermite::prelude::*;

    type V = Vector<u32>; // scalar, 1 lane; N=2 -> 16 usable bits per coordinate

    let x = V::splat(0x9ABC);
    let y = V::splat(0x1234);

    let code = V::morton::<2>([x, y]);

    // independent oracle for lane 0: bit i of x -> 2i, bit i of y -> 2i+1
    let mut want = 0u32;
    for i in 0..16u32 {
        want |= ((0x9ABCu32 >> i) & 1) << (2 * i);
        want |= ((0x1234u32 >> i) & 1) << (2 * i + 1);
    }
    assert_eq!(code.extract::<0>(), want, "vector morton encode");

    let [dx, dy] = code.reverse_morton::<2>();
    assert_eq!(dx.extract::<0>(), 0x9ABC, "vector reverse_morton x");
    assert_eq!(dy.extract::<0>(), 0x1234, "vector reverse_morton y");
}
