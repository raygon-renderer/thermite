//! F16C hardware path validation: the x86-v3 `vcvtph2ps`/`vcvtps2ph` overrides of
//! `PackedFloatRegister<Fp16, ...>` must agree with the scalar `Fp16` oracle (and hence with the
//! generic branchless fallback, which `tests/packed_float.rs` already pins to that oracle).
//!
//! Built only for x86_64 with the `avx2-f16c` feature (the gate the override impls live behind).
//! The test calls the native register methods directly, the same pattern the other x86
//! differential suites use, so the host must support F16C (every AVX2 CPU does).
//!
//! - **unpack**: exhaustive over all 65536 binary16 code points, for both the 8-lane
//!   (`U16x8V3` -> `f32x8`) and 16-lane (`U16x16V3` -> `f32x16`) registers.
//! - **pack**: a structured + random f32 sweep.
//!
//! NaN on the f32 side compares NaN-to-NaN (payload is not contractual); the packed code point
//! must match the oracle exactly.
#![cfg(all(target_arch = "x86_64", feature = "avx2-f16c"))]

use thermite::backend::x86_v3::registers::half16::U16x4V3;
use thermite::backend::x86_v3::registers::{F32x4V3, F32x8V3, U16x8V3, U16x16V3};
use thermite::element::float::spec::{FloatSpec, Fp16, Fp16Fast};
use thermite::register::array::ArrayRegister;
use thermite::register::{PackedFloatRegister, Register, Storage};

type F32x16 = ArrayRegister<F32x8V3, 2>;

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

fn f32_eq(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Two packed binary16 code points are equivalent if they are bit-identical, or if both decode to
/// NaN. F16C `vcvtps2ph` carries the source NaN payload down into the half, whereas the scalar
/// oracle canonicalizes to a fixed quiet NaN; both are valid IEEE NaN encodings, and the NaN
/// payload is explicitly not contractual (see the module / `packed.rs` docs).
fn fp16_eq(got: u16, want: u16) -> bool {
    got == want || (Fp16::unpack(got as u32).is_nan() && Fp16::unpack(want as u32).is_nan())
}

fn read_f32x16(s: &Storage<F32x16>) -> Vec<f32> {
    let mut out = read::<F32x8V3>(&s.0[0]);
    out.extend(read::<F32x8V3>(&s.0[1]));
    out
}

fn make_f32x16(b: &[f32]) -> Storage<F32x16> {
    ArrayRegister([make::<F32x8V3>(&b[..8]), make::<F32x8V3>(&b[8..16])])
}

// =============================== unpack: exhaustive ===================================

#[test]
fn f16c_unpack_4_exhaustive() {
    let mut code: u32 = 0;
    while code < 0x1_0000 {
        let chunk: Vec<u16> = (0..4).map(|i| (code + i) as u16).collect();
        let got = read::<F32x4V3>(&<U16x4V3 as PackedFloatRegister<Fp16, F32x4V3>>::unpack(
            make::<U16x4V3>(&chunk),
        ));
        for k in 0..4 {
            let want = Fp16::unpack(chunk[k] as u32);
            assert!(
                f32_eq(got[k], want),
                "u16x4(f16c) unpack({:#06x}): got {:#010x}, want {:#010x}",
                chunk[k],
                got[k].to_bits(),
                want.to_bits()
            );
        }
        code += 4;
    }
}

#[test]
fn f16c_unpack_8_exhaustive() {
    let mut code: u32 = 0;
    while code < 0x1_0000 {
        let chunk: Vec<u16> = (0..8).map(|i| (code + i) as u16).collect();
        let got = read::<F32x8V3>(&<U16x8V3 as PackedFloatRegister<Fp16, F32x8V3>>::unpack(
            make::<U16x8V3>(&chunk),
        ));
        for k in 0..8 {
            let want = Fp16::unpack(chunk[k] as u32);
            assert!(
                f32_eq(got[k], want),
                "u16x8(f16c) unpack({:#06x}): got {:#010x}, want {:#010x}",
                chunk[k],
                got[k].to_bits(),
                want.to_bits()
            );
        }
        code += 8;
    }
}

#[test]
fn f16c_unpack_16_exhaustive() {
    let mut code: u32 = 0;
    while code < 0x1_0000 {
        let chunk: Vec<u16> = (0..16).map(|i| (code + i) as u16).collect();
        let got = read_f32x16(&<U16x16V3 as PackedFloatRegister<Fp16, F32x16>>::unpack(
            make::<U16x16V3>(&chunk),
        ));
        for k in 0..16 {
            let want = Fp16::unpack(chunk[k] as u32);
            assert!(
                f32_eq(got[k], want),
                "u16x16(f16c) unpack({:#06x}): got {:#010x}, want {:#010x}",
                chunk[k],
                got[k].to_bits(),
                want.to_bits()
            );
        }
        code += 16;
    }
}

// =============================== pack: structured sweep ==============================

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
        f32::MIN,
        1e-30,
        1e30,
        -1e30,
        65504.0, // fp16 max finite
        65520.0, // rounds to inf
        65472.0, // largest that rounds down to max finite
        6.1035e-5,
        5.96e-8, // ~ smallest subnormal
        0.015625,
    ];

    let mut bits: u32 = 0x1234_5678;
    for _ in 0..200_000 {
        bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(f32::from_bits(bits));
    }
    for e in 0..255u32 {
        for m in [0u32, 1, 0x3F_FFFF, 0x40_0000 - 1, 0x20_0000, 1 << 12] {
            v.push(f32::from_bits((e << 23) | (m & 0x7F_FFFF)));
            v.push(f32::from_bits(0x8000_0000 | (e << 23) | (m & 0x7F_FFFF)));
        }
    }
    v
}

#[test]
fn f16c_pack_4_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(4) {
        let mut buf = vec![0.0f32; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x4V3>(&<U16x4V3 as PackedFloatRegister<Fp16, F32x4V3>>::pack(make::<F32x4V3>(
            &buf,
        )));
        for k in 0..4 {
            let want = Fp16::pack(buf[k]) as u16;
            assert!(
                fp16_eq(got[k], want),
                "u16x4(f16c) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}

#[test]
fn f16c_pack_8_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(8) {
        let mut buf = vec![0.0f32; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x8V3>(&<U16x8V3 as PackedFloatRegister<Fp16, F32x8V3>>::pack(make::<F32x8V3>(
            &buf,
        )));
        for k in 0..8 {
            let want = Fp16::pack(buf[k]) as u16;
            assert!(
                fp16_eq(got[k], want),
                "u16x8(f16c) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}

#[test]
fn f16c_pack_16_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(16) {
        let mut buf = vec![0.0f32; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x16V3>(&<U16x16V3 as PackedFloatRegister<Fp16, F32x16>>::pack(make_f32x16(
            &buf,
        )));
        for k in 0..16 {
            let want = Fp16::pack(buf[k]) as u16;
            assert!(
                fp16_eq(got[k], want),
                "u16x16(f16c) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}

// =============================== fast / unchecked (Fp16Fast) =========================
//
// The hardware fast `unpack` is the plain `vcvtph2ps`, so for the (assumed-absent) inf/NaN code
// points it decodes inf/NaN, *not* the `Unchecked` oracle's large normals, so this diffs it
// against the regular `Fp16` oracle (which is exactly `vcvtph2ps`). The fast `pack` flushes
// non-finite/overflowing inputs to signed zero and must match the `Fp16Fast` oracle exactly.

#[test]
fn f16c_fast_unpack_8_exhaustive() {
    let mut code: u32 = 0;
    while code < 0x1_0000 {
        let chunk: Vec<u16> = (0..8).map(|i| (code + i) as u16).collect();
        let got = read::<F32x8V3>(&<U16x8V3 as PackedFloatRegister<Fp16Fast, F32x8V3>>::unpack(make::<
            U16x8V3,
        >(
            &chunk
        )));
        for k in 0..8 {
            let want = Fp16::unpack(chunk[k] as u32); // hardware decode == regular fp16
            assert!(
                f32_eq(got[k], want),
                "u16x8(f16c,fast) unpack({:#06x}): got {:#010x}, want {:#010x}",
                chunk[k],
                got[k].to_bits(),
                want.to_bits()
            );
        }
        code += 8;
    }
}

#[test]
fn f16c_fast_pack_8_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(8) {
        let mut buf = vec![0.0f32; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x8V3>(&<U16x8V3 as PackedFloatRegister<Fp16Fast, F32x8V3>>::pack(make::<
            F32x8V3,
        >(
            &buf
        )));
        for k in 0..8 {
            let want = Fp16Fast::pack(buf[k]) as u16; // non-finite/overflow -> signed zero
            assert_eq!(
                got[k],
                want,
                "u16x8(f16c,fast) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}

#[test]
fn f16c_fast_pack_4_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(4) {
        let mut buf = vec![0.0f32; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x4V3>(&<U16x4V3 as PackedFloatRegister<Fp16Fast, F32x4V3>>::pack(make::<
            F32x4V3,
        >(
            &buf
        )));
        for k in 0..4 {
            let want = Fp16Fast::pack(buf[k]) as u16;
            assert_eq!(
                got[k],
                want,
                "u16x4(f16c,fast) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}

#[test]
fn f16c_fast_pack_16_sweep() {
    let inputs = pack_inputs();
    for chunk in inputs.chunks(16) {
        let mut buf = vec![0.0f32; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        let got = read::<U16x16V3>(&<U16x16V3 as PackedFloatRegister<Fp16Fast, F32x16>>::pack(make_f32x16(
            &buf,
        )));
        for k in 0..16 {
            let want = Fp16Fast::pack(buf[k]) as u16;
            assert_eq!(
                got[k],
                want,
                "u16x16(f16c,fast) pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want
            );
        }
    }
}
