//! Morton (Z-order curve) encode/decode correctness.
//!
//! Checks [`UnsignedIntegerRegister::morton`] / `reverse_morton` for every
//! backend and width against an independent pure-Rust per-lane bit-interleave
//! oracle, plus a `reverse(morton(x)) == x` roundtrip. This exercises the
//! generic shift/mask cascade, the CLMUL `N == 2` fast path on u64-lane v3/v4
//! registers (default `avx2-pclmul`), the x86 `pshufb` and wasm
//! `i8x16.swizzle` nibble-LUT paths on u16/u32 lanes, the `ArrayRegister`
//! chunk delegation into those, and the reduced (half) registers' wide
//! delegation.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use thermite::register::array::ArrayRegister;
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
                assert_eq!(got[lane] as u64, want, "{} N={} morton lane {}", $label, N, lane);
            }

            // roundtrip: reverse_morton(morton(x)) == x masked to usable bits
            let back = <$ut>::reverse_morton::<N>(code);
            for d in 0..N {
                let backd = read::<$ut>(&back[d]);
                for lane in 0..lanes {
                    let want = (corpora[d][idx][lane] as u64) & mask;
                    assert_eq!(backd[lane] as u64, want, "{} N={} reverse d{} lane {}", $label, N, d, lane);
                }
            }
        }
    }};
}

/// N = 1..=4 for one register type.
macro_rules! morton4 {
    ($label:expr, $ut:ty, $e:ty) => {{
        let label = $label;
        morton_check!(label, $ut, $e, 1);
        morton_check!(label, $ut, $e, 2);
        morton_check!(label, $ut, $e, 3);
        morton_check!(label, $ut, $e, 4);
    }};
}

macro_rules! slot {
    ($S:ty, $reg:ident, $e:ty) => {
        morton4!(harness::label::<$S>(stringify!($reg)), <$S as Simd>::$reg, $e)
    };
}

for_each_backend! {
    fn u16<S: Simd>() {
        slot!(S, u16x8, u16);
        slot!(S, u16x16, u16);
        slot!(S, u16x4, u16); // reduced: wide-delegates
    }
    fn u32<S: Simd>() {
        slot!(S, u32x4, u32);
        slot!(S, u32x8, u32);
        slot!(S, u32x16, u32);
        slot!(S, u32x2, u32); // reduced: wide-delegates
    }
    fn u64<S: Simd>() {
        slot!(S, u64x2, u64);
        slot!(S, u64x4, u64);
        slot!(S, u64x8, u64);
    }
    /// Composite widths: `ArrayRegister` chunk-delegates into the native fast paths.
    fn arrays<S: Simd>() {
        morton4!(harness::label::<S>("ArrayRegister<u64x4, 2>"), ArrayRegister<<S as Simd>::u64x4, 2>, u64);
        morton4!(harness::label::<S>("ArrayRegister<u32x8, 2>"), ArrayRegister<<S as Simd>::u32x8, 2>, u32);
    }
}

#[test]
fn vector_layer() {
    use thermite::prelude::*;

    type V = Vector<u32>; // scalar, 1 lane; N=2 -> 16 usable bits per coordinate

    let x = V::splat(0x9ABC);
    let y = V::splat(0x1234);

    let code = V::morton::<2>([x, y]);

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
