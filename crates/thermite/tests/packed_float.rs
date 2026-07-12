//! Differential test of the generic branchless `PackedFloatRegister` defaults against the
//! scalar [`FloatSpec`] oracle.
//!
//! The vectorized `pack`/`unpack` (see `register::{pack_packed, unpack_packed}`) reconstruct
//! the same transcoding as `FloatSpec::pack`/`unpack` purely with register ops. Here we drive
//! the emulated `ArrayRegister` container (which takes those defaults verbatim - no hardware
//! override) and assert it matches the oracle lane-for-lane:
//!
//! - **unpack**: exhaustive over *every* code point (all 65536 for the 16-bit formats, all 256
//!   for the 8-bit ones).
//! - **pack**: a large structured + random `f32` sweep, plus all the interesting edges.
//!
//! NaN comparisons: on the `f32` side any two NaNs are treated as equal (payload is not
//! contractual); on the packed side the canonical NaN code point must match exactly.

use thermite::element::float::spec::{Bf16, FloatSpec, Fp8E4M3, Fp8E5M2, Fp16, Fp16Fast};
use thermite::register::array::ArrayRegister;
use thermite::register::{CoreRegister, PackedFloatRegister, Register, Storage};

type F8 = ArrayRegister<f32, 16>;
type F32x8 = ArrayRegister<f32, 8>;
type U16x8 = ArrayRegister<u16, 8>;
type U8x16 = ArrayRegister<u8, 16>;

fn lanes<R: CoreRegister>() -> usize {
    <R::Lanes as generic_array::typenum::Unsigned>::USIZE
}

fn make<R: Register>(vals: &[R::Element]) -> Storage<R>
where
    R::Element: Copy,
{
    use generic_array::sequence::GenericSequence;
    let arr = generic_array::GenericArray::generate(|i| vals[i]);
    R::new(arr)
}

fn read<R: Register>(s: &Storage<R>) -> Vec<R::Element>
where
    R::Element: Copy,
{
    R::as_slice(s).to_vec()
}

/// f32 bit-equality with NaN-insensitive payload.
fn f32_eq(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    a.to_bits() == b.to_bits()
}

// =============================== unpack: exhaustive ===================================

/// Exhaustively unpack every code point of a 16-bit format through the vectorized default and
/// compare against the scalar oracle.
fn check_unpack_16<S: FloatSpec>(name: &str)
where
    U16x8: PackedFloatRegister<S, F32x8>,
{
    let l = lanes::<U16x8>();
    let mut code: u32 = 0;
    while code < 0x1_0000 {
        let chunk: Vec<u16> = (0..l).map(|i| (code + i as u32) as u16).collect();
        let want: Vec<f32> = chunk.iter().map(|&c| S::unpack(c as u32)).collect();

        let packed = make::<U16x8>(&chunk);
        let got = read::<F32x8>(&<U16x8 as PackedFloatRegister<S, F32x8>>::unpack(packed));

        for k in 0..l {
            assert!(
                f32_eq(got[k], want[k]),
                "{name} unpack({:#06x}): got {:#010x} ({}), want {:#010x} ({})",
                chunk[k],
                got[k].to_bits(),
                got[k],
                want[k].to_bits(),
                want[k]
            );
        }
        code += l as u32;
    }
}

/// Exhaustively unpack every code point of an 8-bit format.
fn check_unpack_8<S: FloatSpec>(name: &str)
where
    U8x16: PackedFloatRegister<S, F8>,
{
    let l = lanes::<U8x16>();
    let mut code: u32 = 0;
    while code < 0x100 {
        let chunk: Vec<u8> = (0..l).map(|i| (code + i as u32) as u8).collect();
        let want: Vec<f32> = chunk.iter().map(|&c| S::unpack(c as u32)).collect();

        let packed = make::<U8x16>(&chunk);
        let got = read::<F8>(&<U8x16 as PackedFloatRegister<S, F8>>::unpack(packed));

        for k in 0..l {
            assert!(
                f32_eq(got[k], want[k]),
                "{name} unpack({:#04x}): got {:#010x} ({}), want {:#010x} ({})",
                chunk[k],
                got[k].to_bits(),
                got[k],
                want[k].to_bits(),
                want[k]
            );
        }
        code += l as u32;
    }
}

#[test]
fn fp16_unpack_exhaustive() {
    check_unpack_16::<Fp16>("fp16");
}
#[test]
fn fp16_fast_unpack_exhaustive() {
    check_unpack_16::<Fp16Fast>("fp16fast");
}
#[test]
fn bf16_unpack_exhaustive() {
    check_unpack_16::<Bf16>("bf16");
}
#[test]
fn fp8_e4m3_unpack_exhaustive() {
    check_unpack_8::<Fp8E4M3>("fp8e4m3");
}
#[test]
fn fp8_e5m2_unpack_exhaustive() {
    check_unpack_8::<Fp8E5M2>("fp8e5m2");
}

// =============================== pack: structured sweep ==============================

/// A broad set of f32 inputs covering every regime that `pack` distinguishes.
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
        65504.0,  // fp16 max finite
        65520.0,  // fp16 rounds to inf
        448.0,    // e4m3 max finite
        449.0,    // e4m3 saturates
        57344.0,  // e5m2 max finite
        61440.0,  // e5m2 rounds to inf
        0.015625, // exact small power of two
    ];

    // Walk a deterministic LCG over the full f32 bit space, sampling a dense, reproducible mix
    // of normals/subnormals/specials across both signs.
    let mut bits: u32 = 0x1234_5678;
    for _ in 0..200_000 {
        bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(f32::from_bits(bits));
    }

    // Dense sweep of every code point reachable by varying the low f32 bits around the
    // boundaries each format rounds at (subnormal/normal/overflow transitions).
    for e in 0..255u32 {
        for m in [0u32, 1, 0x3F_FFFF, 0x40_0000 - 1, 0x20_0000, 1 << 12] {
            v.push(f32::from_bits((e << 23) | (m & 0x7F_FFFF)));
            v.push(f32::from_bits(0x8000_0000 | (e << 23) | (m & 0x7F_FFFF)));
        }
    }
    v
}

fn check_pack_16<S: FloatSpec>(name: &str)
where
    U16x8: PackedFloatRegister<S, F32x8>,
{
    let inputs = pack_inputs();
    let l = lanes::<U16x8>();
    for chunk in inputs.chunks(l) {
        let mut buf = vec![0.0f32; l];
        buf[..chunk.len()].copy_from_slice(chunk);
        let want: Vec<u16> = buf.iter().map(|&x| S::pack(x) as u16).collect();

        let fv = make::<F32x8>(&buf);
        let got = read::<U16x8>(&<U16x8 as PackedFloatRegister<S, F32x8>>::pack(fv));

        for k in 0..l {
            assert_eq!(
                got[k],
                want[k],
                "{name} pack({} = {:#010x}): got {:#06x}, want {:#06x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want[k]
            );
        }
    }
}

fn check_pack_8<S: FloatSpec>(name: &str)
where
    U8x16: PackedFloatRegister<S, F8>,
{
    let inputs = pack_inputs();
    let l = lanes::<U8x16>();
    for chunk in inputs.chunks(l) {
        let mut buf = vec![0.0f32; l];
        buf[..chunk.len()].copy_from_slice(chunk);
        let want: Vec<u8> = buf.iter().map(|&x| S::pack(x) as u8).collect();

        let fv = make::<F8>(&buf);
        let got = read::<U8x16>(&<U8x16 as PackedFloatRegister<S, F8>>::pack(fv));

        for k in 0..l {
            assert_eq!(
                got[k],
                want[k],
                "{name} pack({} = {:#010x}): got {:#04x}, want {:#04x}",
                buf[k],
                buf[k].to_bits(),
                got[k],
                want[k]
            );
        }
    }
}

#[test]
fn fp16_pack_sweep() {
    check_pack_16::<Fp16>("fp16");
}
#[test]
fn fp16_fast_pack_sweep() {
    check_pack_16::<Fp16Fast>("fp16fast");
}
#[test]
fn bf16_pack_sweep() {
    check_pack_16::<Bf16>("bf16");
}
#[test]
fn fp8_e4m3_pack_sweep() {
    check_pack_8::<Fp8E4M3>("fp8e4m3");
}
#[test]
fn fp8_e5m2_pack_sweep() {
    check_pack_8::<Fp8E5M2>("fp8e5m2");
}

// =============================== round-trip closure =================================

/// For formats where every code point is the canonical encoding of its value (no redundant
/// representations), `pack(unpack(c)) == c` for all finite, non-NaN code points.
#[test]
fn fp16_round_trip() {
    for c in 0u32..0x1_0000 {
        let v = Fp16::unpack(c);
        if v.is_nan() {
            continue;
        }
        assert_eq!(Fp16::pack(v), c, "fp16 round-trip {c:#06x}");
    }
}
