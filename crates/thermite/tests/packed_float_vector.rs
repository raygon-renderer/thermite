//! Vector-layer `PackedFloatVector` smoke + differential test: drives the user-facing
//! `Vector<u16 register>::{pack, unpack}` API (the wrapper over `PackedFloatRegister`) and diffs
//! against the scalar `FloatSpec` oracle. The register layer is already exhaustively covered by
//! `packed_float{,_native,_f16c}.rs`; this confirms the `Vector` wrapper delegates correctly and
//! the `SimdVectors` bound resolves through real backend types.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::element::float::spec::{Bf16, FloatSpec, Fp16, Fp16Fast};
use thermite::register::{CoreRegister, Register};
use thermite::vector::{GenericVector, PackedFloatVector, Vector};

fn f32_eq(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn packed_eq<S: FloatSpec>(got: u16, want: u16) -> bool {
    got == want || (S::unpack(got as u32).is_nan() && S::unpack(want as u32).is_nan())
}

/// Build a `Vector<R>` from the first `LANES` elements of `vals`.
fn vec_of<R: Register>(vals: &[R::Element]) -> Vector<R>
where
    R::Element: Copy,
{
    Vector::from_slice(vals)
}

/// unpack (all code points) + pack (a structured sweep) of `U` <-> `F` through the *vector* API,
/// diffed against the `S` oracle. `U`/`F` are the u16 / f32 register types.
macro_rules! check_vector {
    ($spec:ty, $u:ty, $f:ty) => {{
        let l = <<$u as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;

        // unpack: every code point.
        let mut code: u32 = 0;
        while code < 0x1_0000 {
            let chunk: Vec<u16> = (0..l).map(|i| (code + i as u32) as u16).collect();
            let unpacked = <Vector<$u> as PackedFloatVector<$spec, Vector<$f>>>::unpack(vec_of::<$u>(&chunk));
            let got = unpacked.into_array();
            for k in 0..l {
                let want = <$spec>::unpack(chunk[k] as u32);
                assert!(
                    f32_eq(got[k], want),
                    "{} unpack({:#06x}): got {:#010x}, want {:#010x}",
                    stringify!($spec),
                    chunk[k],
                    got[k].to_bits(),
                    want.to_bits()
                );
            }
            code += l as u32;
        }

        // pack: a structured + random sweep.
        let mut inputs: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            1e30,
            -1e30,
            65504.0,
            65520.0,
            448.0,
            57344.0,
            6.1e-5,
            5.96e-8,
        ];
        let mut bits: u32 = 0x9E37_79B9;
        for _ in 0..60_000 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            inputs.push(f32::from_bits(bits));
        }
        for chunk in inputs.chunks(l) {
            let mut buf = vec![0.0f32; l];
            buf[..chunk.len()].copy_from_slice(chunk);
            let packed: Vector<$u> = <Vector<$u> as PackedFloatVector<$spec, Vector<$f>>>::pack(vec_of::<$f>(&buf));
            let got = packed.into_array();
            for k in 0..l {
                let want = <$spec>::pack(buf[k]) as u16;
                assert!(
                    packed_eq::<$spec>(got[k], want),
                    "{} pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                    stringify!($spec),
                    buf[k],
                    buf[k].to_bits(),
                    got[k],
                    want
                );
            }
        }
    }};
}

// fp16 + bf16, always (these match the oracle on both the generic default and the F16C path).
macro_rules! vector_suite {
    ($mod:ident, $u:ty, $f:ty) => {
        mod $mod {
            use super::*;
            #[test]
            fn fp16() {
                check_vector!(Fp16, $u, $f);
            }
            #[test]
            fn bf16() {
                check_vector!(Bf16, $u, $f);
            }
        }
    };
}

// fp16fast as a separate test fn, added only where the generic default is wired. (The F16C
// hardware fast unpack decodes the assumed-absent inf/NaN code points differently than the
// Unchecked oracle, so it is excluded for v3 under avx2-f16c.)
macro_rules! fast_test {
    ($u:ty, $f:ty) => {
        #[test]
        fn fp16fast() {
            check_vector!(Fp16Fast, $u, $f);
        }
    };
}

// 8-lane u16 <-> f32 on each backend (v3's f32x8 is native; v1/v2 it's an ArrayRegister - the
// `PackedFloatVector` blanket handles both transparently).
mod v1 {
    use super::*;
    use thermite::backend::x86_v1::registers::F32x4V1;
    use thermite::backend::x86_v1::registers::U16x8V1;
    use thermite::register::array::ArrayRegister;
    vector_suite!(x8, U16x8V1, ArrayRegister<F32x4V1, 2>);
    mod x8_fast {
        use super::*;
        fast_test!(U16x8V1, ArrayRegister<F32x4V1, 2>);
    }
}

mod v2 {
    use super::*;
    use thermite::backend::x86_v2::registers::F32x4V2;
    use thermite::backend::x86_v2::registers::U16x8V2;
    use thermite::register::array::ArrayRegister;
    vector_suite!(x8, U16x8V2, ArrayRegister<F32x4V2, 2>);
    mod x8_fast {
        use super::*;
        fast_test!(U16x8V2, ArrayRegister<F32x4V2, 2>);
    }
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use super::*;
    use thermite::backend::x86_v3::registers::{F32x8V3, U16x8V3};
    vector_suite!(x8, U16x8V3, F32x8V3);
    // Fast suite only when v3 uses the generic default (no F16C); the hardware fast unpack
    // diverges on reserved code points by design.
    #[cfg(not(feature = "avx2-f16c"))]
    mod x8_fast {
        use super::*;
        fast_test!(U16x8V3, F32x8V3);
    }
}
