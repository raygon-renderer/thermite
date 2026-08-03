//! fp8 (`Fp8E4M3` / `Fp8E5M2`) pack/unpack on the native `u8` registers (the generic branchless
//! defaults - no hardware transcodes fp8), diffed against the scalar `FloatSpec` oracle. The
//! emulated `ArrayRegister<u8,16>` path is already covered by `packed_float.rs`; this exercises the
//! real backend `u8 <-> u32` casts the generic kernels call through, plus the `Vector` wrapper.
//!
//! - **unpack**: exhaustive over all 256 fp8 code points.
//! - **pack**: a structured + random f32 sweep.
//!
//! NaN: f32 side NaN-insensitive; packed side treats two NaN code points as equal (E4M3 has one
//! NaN, E5M2 is IEEE).
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::element::float::spec::{FloatSpec, Fp8E4M3, Fp8E5M2};
use thermite::register::{CoreRegister, PackedFloatRegister, Register, Storage};
use thermite::vector::{GenericVector, PackedFloatVector, Vector};

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

fn packed_eq<S: FloatSpec>(got: u8, want: u8) -> bool {
    got == want || (S::unpack(got as u32).is_nan() && S::unpack(want as u32).is_nan())
}

fn pack_inputs() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        f32::MAX,
        1e-30,
        1e30,
        -1e30,
        448.0,   // e4m3 max finite
        449.0,   // e4m3 saturates
        57344.0, // e5m2 max finite
        61440.0, // e5m2 rounds to inf
        0.015625,
    ];
    let mut bits: u32 = 0x1234_5678;
    for _ in 0..80_000 {
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

// ----- register layer -----

macro_rules! reg_unpack {
    ($spec:ty, $u8:ty, $f32:ty) => {{
        let l = lanes::<$u8>();
        let mut code: u32 = 0;
        while code < 0x100 {
            let chunk: Vec<u8> = (0..l).map(|i| (code + i as u32) as u8).collect();
            let got = read::<$f32>(&<$u8 as PackedFloatRegister<$spec, $f32>>::unpack(make::<$u8>(&chunk)));
            for k in 0..l {
                let want = <$spec>::unpack(chunk[k] as u32);
                assert!(
                    f32_eq(got[k], want),
                    "{} unpack({:#04x}): got {:#010x}, want {:#010x}",
                    stringify!($spec),
                    chunk[k],
                    got[k].to_bits(),
                    want.to_bits()
                );
            }
            code += l as u32;
        }
    }};
}

macro_rules! reg_pack {
    ($spec:ty, $u8:ty, $f32:ty) => {{
        let inputs = pack_inputs();
        let l = lanes::<$u8>();
        for chunk in inputs.chunks(l) {
            let mut buf = vec![0.0f32; l];
            buf[..chunk.len()].copy_from_slice(chunk);
            let got = read::<$u8>(&<$u8 as PackedFloatRegister<$spec, $f32>>::pack(make::<$f32>(&buf)));
            for k in 0..l {
                let want = <$spec>::pack(buf[k]) as u8;
                assert!(
                    packed_eq::<$spec>(got[k], want),
                    "{} pack({} = {:#010x}): got {:#04x}, want {:#04x}",
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

// ----- vector layer (the public Vector::pack/unpack wrapper) -----

macro_rules! vec_roundtrip {
    ($spec:ty, $u8:ty, $f32:ty) => {{
        let l = lanes::<$u8>();
        // unpack all 256 codes through Vector::unpack
        let mut code: u32 = 0;
        while code < 0x100 {
            let chunk: Vec<u8> = (0..l).map(|i| (code + i as u32) as u8).collect();
            let uv: Vector<$u8> = Vector::from_slice(&chunk);
            let fv = <Vector<$u8> as PackedFloatVector<$spec, Vector<$f32>>>::unpack(uv);
            let got = fv.into_array();
            for k in 0..l {
                let want = <$spec>::unpack(chunk[k] as u32);
                assert!(
                    f32_eq(got[k], want),
                    "{} vec unpack({:#04x})",
                    stringify!($spec),
                    chunk[k]
                );
            }
            code += l as u32;
        }
        // pack a sweep through Vector::pack
        let inputs = pack_inputs();
        for chunk in inputs.chunks(l) {
            let mut buf = vec![0.0f32; l];
            buf[..chunk.len()].copy_from_slice(chunk);
            let fv: Vector<$f32> = Vector::from_slice(&buf);
            let uv = <Vector<$u8> as PackedFloatVector<$spec, Vector<$f32>>>::pack(fv);
            let got = uv.into_array();
            for k in 0..l {
                let want = <$spec>::pack(buf[k]) as u8;
                assert!(
                    packed_eq::<$spec>(got[k], want),
                    "{} vec pack({})",
                    stringify!($spec),
                    buf[k]
                );
            }
        }
    }};
}

macro_rules! fp8_suite {
    ($mod:ident, $u8x4:ty, $u8x8:ty, $u8x16:ty, $f32x4:ty, $f32x8:ty, $f32x16:ty) => {
        mod $mod {
            use super::*;
            #[test]
            fn e4m3() {
                reg_unpack!(Fp8E4M3, $u8x4, $f32x4);
                reg_unpack!(Fp8E4M3, $u8x8, $f32x8);
                reg_unpack!(Fp8E4M3, $u8x16, $f32x16);
                reg_pack!(Fp8E4M3, $u8x4, $f32x4);
                reg_pack!(Fp8E4M3, $u8x8, $f32x8);
                reg_pack!(Fp8E4M3, $u8x16, $f32x16);
                vec_roundtrip!(Fp8E4M3, $u8x8, $f32x8);
            }
            #[test]
            fn e5m2() {
                reg_unpack!(Fp8E5M2, $u8x4, $f32x4);
                reg_unpack!(Fp8E5M2, $u8x8, $f32x8);
                reg_unpack!(Fp8E5M2, $u8x16, $f32x16);
                reg_pack!(Fp8E5M2, $u8x4, $f32x4);
                reg_pack!(Fp8E5M2, $u8x8, $f32x8);
                reg_pack!(Fp8E5M2, $u8x16, $f32x16);
                vec_roundtrip!(Fp8E5M2, $u8x8, $f32x8);
            }
        }
    };
}

mod v1 {
    use super::*;
    use thermite::backend::x86_v1::registers::half8::{U8x4V1, U8x8V1};
    use thermite::backend::x86_v1::registers::{F32x4V1, U8x16V1};
    use thermite::register::array::ArrayRegister;
    fp8_suite!(t, U8x4V1, U8x8V1, U8x16V1, F32x4V1, ArrayRegister<F32x4V1, 2>, ArrayRegister<F32x4V1, 4>);
}

mod v2 {
    use super::*;
    use thermite::backend::x86_v2::registers::half8::{U8x4V2, U8x8V2};
    use thermite::backend::x86_v2::registers::{F32x4V2, U8x16V2};
    use thermite::register::array::ArrayRegister;
    fp8_suite!(t, U8x4V2, U8x8V2, U8x16V2, F32x4V2, ArrayRegister<F32x4V2, 2>, ArrayRegister<F32x4V2, 4>);
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use super::*;
    use thermite::backend::x86_v3::registers::half8::{U8x4V3, U8x8V3};
    use thermite::backend::x86_v3::registers::{F32x4V3, F32x8V3, U8x16V3};
    use thermite::register::array::ArrayRegister;
    fp8_suite!(t, U8x4V3, U8x8V3, U8x16V3, F32x4V3, F32x8V3, ArrayRegister<F32x8V3, 2>);
}
