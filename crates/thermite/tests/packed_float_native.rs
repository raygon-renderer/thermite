//! The generic branchless `PackedFloatRegister` defaults, validated on each backend's *native*
//! 16-bit registers (not just the emulated `ArrayRegister` path that `tests/packed_float.rs`
//! covers). This exercises the real backend `mulhi` / shift / blendv / `u16<->u32` cast impls
//! that the generic helpers call through, on every native u16 width (4 / 8 / 16 lanes).
//!
//! Diffed against the scalar [`FloatSpec`] oracle for `Fp16`, `Fp16Fast`, and `Bf16`:
//! - unpack exhaustively over all 65536 code points,
//! - pack over a structured + random f32 sweep.
//!
//! NaN comparisons: the f32 side is NaN-insensitive, and the packed side treats two NaN code points as equal.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::element::float::spec::{Bf16, FloatSpec, Fp16, Fp16Fast};
use thermite::register::{CoreRegister, PackedFloatRegister, Register, Storage};

fn make<R: Register>(vals: &[R::Element]) -> Storage<R>
where
    R::Element: Copy,
{
    use generic_array::sequence::GenericSequence;
    R::new(generic_array::GenericArray::generate(|i| vals[i]))
}

fn read<R: Register>(s: &Storage<R>) -> Vec<R::Element>
where
    R::Element: Copy,
{
    R::as_slice(s).to_vec()
}

fn lanes<R: CoreRegister>() -> usize {
    <R::Lanes as generic_array::typenum::Unsigned>::USIZE
}

fn f32_eq(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Two packed code points are equal, or both decode (under `S`) to NaN.
fn packed_eq<S: FloatSpec>(got: u16, want: u16) -> bool {
    got == want || (S::unpack(got as u32).is_nan() && S::unpack(want as u32).is_nan())
}

fn pack_inputs() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        f32::MAX,
        1e-30,
        1e30,
        -1e30,
        65504.0,
        65520.0,
        448.0,
        57344.0,
        0.015625,
    ];
    let mut bits: u32 = 0x1234_5678;
    for _ in 0..120_000 {
        bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(f32::from_bits(bits));
    }
    for e in 0..255u32 {
        for m in [0u32, 1, 0x3F_FFFF, 0x20_0000] {
            v.push(f32::from_bits((e << 23) | (m & 0x7F_FFFF)));
            v.push(f32::from_bits(0x8000_0000 | (e << 23) | (m & 0x7F_FFFF)));
        }
    }
    v
}

/// Exhaustive unpack (all 65536 code points) of `$u16` -> `$f32` via the generic default vs the
/// `$spec` oracle. Concrete in the types to sidestep generic trait-bound threading.
macro_rules! check_unpack {
    ($spec:ty, $u16:ty, $f32:ty) => {{
        let l = lanes::<$u16>();
        let mut code: u32 = 0;
        while code < 0x1_0000 {
            let chunk: Vec<u16> = (0..l).map(|i| (code + i as u32) as u16).collect();
            let got = read::<$f32>(&<$u16 as PackedFloatRegister<$spec, $f32>>::unpack(make::<$u16>(
                &chunk,
            )));
            for k in 0..l {
                let want = <$spec>::unpack(chunk[k] as u32);
                assert!(
                    f32_eq(got[k], want),
                    "unpack({:#06x}): got {:#010x}, want {:#010x}",
                    chunk[k],
                    got[k].to_bits(),
                    want.to_bits()
                );
            }
            code += l as u32;
        }
    }};
}

/// Pack sweep of `$f32` -> `$u16` via the generic default vs the `$spec` oracle.
macro_rules! check_pack {
    ($spec:ty, $u16:ty, $f32:ty) => {{
        let inputs = pack_inputs();
        let l = lanes::<$u16>();
        for chunk in inputs.chunks(l) {
            let mut buf = vec![0.0f32; l];
            buf[..chunk.len()].copy_from_slice(chunk);
            let got = read::<$u16>(&<$u16 as PackedFloatRegister<$spec, $f32>>::pack(make::<$f32>(&buf)));
            for k in 0..l {
                let want = <$spec>::pack(buf[k]) as u16;
                assert!(
                    packed_eq::<$spec>(got[k], want),
                    "pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                    buf[k],
                    buf[k].to_bits(),
                    got[k],
                    want
                );
            }
        }
    }};
}

/// The bf16 unpack+pack tests (bf16 always uses the generic default on every backend, including
/// v3 where F16C does not apply to it).
macro_rules! bf16_tests {
    ($u16:ty, $f32:ty) => {
        #[test]
        fn bf16_unpack() {
            check_unpack!(Bf16, $u16, $f32);
        }
        #[test]
        fn bf16_pack() {
            check_pack!(Bf16, $u16, $f32);
        }
    };
}

/// The binary16 (Fp16 / Fp16Fast) unpack+pack tests against the generic-default impls. Only valid
/// where the generic default is what's actually wired (v1/v2 always, v3 only when `avx2-f16c` is
/// off, since with F16C the hardware path is tested separately in `packed_float_f16c.rs`, and its fast
/// unpack intentionally diverges from the `Fp16Fast` oracle on the assumed-absent specials).
macro_rules! fp16_tests {
    ($u16:ty, $f32:ty) => {
        #[test]
        fn fp16_unpack() {
            check_unpack!(Fp16, $u16, $f32);
        }
        #[test]
        fn fp16_pack() {
            check_pack!(Fp16, $u16, $f32);
        }
        #[test]
        fn fp16fast_unpack() {
            check_unpack!(Fp16Fast, $u16, $f32);
        }
        #[test]
        fn fp16fast_pack() {
            check_pack!(Fp16Fast, $u16, $f32);
        }
    };
}

/// Full native suite (all three formats, generic defaults) for v1/v2.
macro_rules! native_suite {
    ($mod:ident, $u16:ty, $f32:ty) => {
        mod $mod {
            use super::*;
            bf16_tests!($u16, $f32);
            fp16_tests!($u16, $f32);
        }
    };
}

/// v3 native suite: bf16 always, and binary16 generic-default tests only when F16C is unavailable.
macro_rules! native_suite_v3 {
    ($mod:ident, $u16:ty, $f32:ty) => {
        mod $mod {
            use super::*;
            bf16_tests!($u16, $f32);
            #[cfg(not(feature = "avx2-f16c"))]
            fp16_tests!($u16, $f32);
        }
    };
}

mod v1 {
    use super::*;
    use thermite::backend::x86_v1::registers::half16::U16x4V1;
    use thermite::backend::x86_v1::registers::{F32x4V1, U16x8V1};
    use thermite::register::array::ArrayRegister;

    native_suite!(x4, U16x4V1, F32x4V1);
    native_suite!(x8, U16x8V1, ArrayRegister<F32x4V1, 2>);
    native_suite!(x16, ArrayRegister<U16x8V1, 2>, ArrayRegister<F32x4V1, 4>);
}

mod v2 {
    use super::*;
    use thermite::backend::x86_v2::registers::half16::U16x4V2;
    use thermite::backend::x86_v2::registers::{F32x4V2, U16x8V2};
    use thermite::register::array::ArrayRegister;

    native_suite!(x4, U16x4V2, F32x4V2);
    native_suite!(x8, U16x8V2, ArrayRegister<F32x4V2, 2>);
    native_suite!(x16, ArrayRegister<U16x8V2, 2>, ArrayRegister<F32x4V2, 4>);
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use super::*;
    use thermite::backend::x86_v3::registers::half16::U16x4V3;
    use thermite::backend::x86_v3::registers::{F32x4V3, F32x8V3, U16x8V3, U16x16V3};
    use thermite::register::array::ArrayRegister;

    // v3 has native 8- and 16-lane 16-bit registers, and f32x8 is native (F32x8V3).
    native_suite_v3!(x4, U16x4V3, F32x4V3);
    native_suite_v3!(x8, U16x8V3, F32x8V3);
    native_suite_v3!(x16, U16x16V3, ArrayRegister<F32x8V3, 2>);
}
