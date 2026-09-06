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

mod harness;

use thermite::element::float::spec::{Bf16, FloatSpec, Fp16, Fp16Fast};
use thermite::register::{CoreRegister, PackedFloatRegister, Register, Storage};
use thermite::simd::Simd;

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

/// AVX2-and-up rows take the F16C hardware path when the feature is on. The binary16
/// generic-default tests are only valid where that default is what is wired (below AVX2
/// always, AVX2+ only with `avx2-f16c` off). The hardware path has its own suite in
/// `packed_float_f16c.rs`, and its fast unpack intentionally diverges from the `Fp16Fast`
/// oracle on the assumed-absent specials. bf16 always uses the generic default.
fn f16c_hardware<S: thermite::simd::HasIsa>() -> bool {
    cfg!(feature = "avx2-f16c") && S::ISA >= thermite::isa::InstructionSet::X86V3
}

/// Every backend's u16x4/u16x8/u16x16 <-> f32x4/f32x8/f32x16 slots, native or
/// `ArrayRegister` as the backend defines them.
macro_rules! native_tests {
    ($($sfx:ident: $u16:ty, $f32:ty);+ $(;)?) => { paste::paste! { for_each_backend_concrete! { $(
        fn [<bf16_unpack_ $sfx>]() {
            check_unpack!(Bf16, $u16, $f32);
        }
        fn [<bf16_pack_ $sfx>]() {
            check_pack!(Bf16, $u16, $f32);
        }
        fn [<fp16_unpack_ $sfx>]() {
            if f16c_hardware::<S>() {
                return;
            }
            check_unpack!(Fp16, $u16, $f32);
        }
        fn [<fp16_pack_ $sfx>]() {
            if f16c_hardware::<S>() {
                return;
            }
            check_pack!(Fp16, $u16, $f32);
        }
        fn [<fp16fast_unpack_ $sfx>]() {
            if f16c_hardware::<S>() {
                return;
            }
            check_unpack!(Fp16Fast, $u16, $f32);
        }
        fn [<fp16fast_pack_ $sfx>]() {
            if f16c_hardware::<S>() {
                return;
            }
            check_pack!(Fp16Fast, $u16, $f32);
        }
    )+ } } };
}

native_tests! {
    x4: <S as Simd>::u16x4, <S as Simd>::f32x4;
    x8: <S as Simd>::u16x8, <S as Simd>::f32x8;
    x16: <S as Simd>::u16x16, <S as Simd>::f32x16;
}
