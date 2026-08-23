//! Swizzle / permute coverage, checked against each register's own
//! `scalar_swizzle` / `scalar_permutev` ground truth.
//!
//! Two halves:
//!  - **Constant-index** paths (`swizzle_const` / `permutev_const`, reached via
//!    the public `swizzle!` macro) across native (`__m128`/`__m256`/`__m256d`)
//!    and emulated registers on every backend. The 4-lane batteries deliberately
//!    include the exact index patterns `impl_mat4_inverse!` relies on.
//!  - **Runtime** paths (`R::permutev` / `R::swizzle` with live index
//!    registers), with exhaustive O(N^2) single-lane routing and
//!    random fuzzing, the coverage formerly in `array_swizzle.rs`, broadened
//!    here from V3-emulated-only to native registers across v1/v2/v3.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use generic_array::{GenericArray, arr, typenum::Unsigned};
use rand::RngExt;
use thermite::Vector;
use thermite::register::array::ArrayRegister;
use thermite::register::{NumericRegister, Register, Storage};

use thermite::backend::scalar::Scalar;
use thermite::simd::{NativeSimd, Simd};

/// Build the live index register `permutev`/`swizzle` consume from a `u32`
/// index array (the test-side mirror of the crate's internal converter).
fn idx_reg<R: Register>(idxs: &GenericArray<u32, R::Lanes>) -> Storage<R::Unsigned> {
    use thermite::element::Element;

    let mut arr: GenericArray<<R::Unsigned as Register>::Element, R::Lanes> = GenericArray::default();
    for i in 0..<R::Lanes as Unsigned>::USIZE {
        arr[i] = Element::from_u16(idxs[i] as u16);
    }
    <R::Unsigned as Register>::new(arr)
}

/// `permutev_const` (single-register, indices `0..LANES`) vs `scalar_permutev`.
macro_rules! perm {
    ($R:ty, $a:expr, [$($i:literal),* $(,)?]) => {{
        let got = thermite::swizzle!(Vector::<$R>($a), [$($i),*]).0;
        let want = <$R>::scalar_permutev($a, idx_reg::<$R>(&arr![$($i as u32),*]));
        assert_eq!(
            <$R>::as_slice(&got), <$R>::as_slice(&want),
            "permute_const{:?} mismatch", [$($i),*]
        );
    }};
}

/// `swizzle_const` (two-register, indices `0..2*LANES`) vs `scalar_swizzle`.
macro_rules! swz {
    ($R:ty, $a:expr, $b:expr, [$($i:literal),* $(,)?]) => {{
        let got = thermite::swizzle!(Vector::<$R>($a), Vector::<$R>($b), [$($i),*]).0;
        let want = <$R>::scalar_swizzle($a, $b, idx_reg::<$R>(&arr![$($i as u32),*]));
        assert_eq!(
            <$R>::as_slice(&got), <$R>::as_slice(&want),
            "swizzle_const{:?} mismatch", [$($i),*]
        );
    }};
}

/// 4-lane battery (f32x4 / f64x4 / i32x4 / i64x2-as-4? no - 4-lane only).
macro_rules! battery4 {
    ($R:ty) => {{
        let a = <$R>::indexed();
        let b = <$R>::add(a, a); // distinct second operand (= 2*a)

        // --- permute (1-input, idx 0..3) ---
        perm!($R, a, [0, 1, 2, 3]); // identity
        perm!($R, a, [3, 2, 1, 0]); // reverse
        perm!($R, a, [0, 0, 0, 0]);
        perm!($R, a, [1, 1, 1, 1]);
        perm!($R, a, [2, 2, 2, 2]);
        perm!($R, a, [3, 3, 3, 3]);
        perm!($R, a, [0, 0, 0, 2]); // from impl_mat4_inverse
        perm!($R, a, [0, 2, 2, 2]); // from impl_mat4_inverse
        perm!($R, a, [1, 0, 3, 2]);
        perm!($R, a, [2, 3, 0, 1]);

        // --- swizzle (2-input, idx 0..7) ---
        swz!($R, a, b, [0, 1, 2, 3]); // identity (all a)
        swz!($R, a, b, [4, 5, 6, 7]); // all b
        swz!($R, a, b, [7, 6, 5, 4]); // b reversed
        swz!($R, a, b, [0, 4, 1, 5]); // interleave
        // exact patterns used by mat4_inverse:
        swz!($R, a, b, [3, 3, 7, 7]);
        swz!($R, a, b, [2, 2, 6, 6]);
        swz!($R, a, b, [1, 1, 5, 5]);
        swz!($R, a, b, [0, 0, 4, 4]);
        swz!($R, a, b, [0, 2, 4, 6]);
    }};
}

/// 8-lane battery (f32x8 / i32x8), stresses cross-128-bit-lane routing.
macro_rules! battery8 {
    ($R:ty) => {{
        let a = <$R>::indexed();
        let b = <$R>::add(a, a); // distinct second operand (= 2*a)

        perm!($R, a, [0, 1, 2, 3, 4, 5, 6, 7]); // identity
        perm!($R, a, [7, 6, 5, 4, 3, 2, 1, 0]); // reverse
        perm!($R, a, [4, 5, 6, 7, 0, 1, 2, 3]); // swap 128-bit halves
        perm!($R, a, [0, 0, 0, 0, 0, 0, 0, 0]);
        perm!($R, a, [7, 7, 7, 7, 7, 7, 7, 7]);
        perm!($R, a, [0, 4, 1, 5, 2, 6, 3, 7]); // cross-lane interleave

        swz!($R, a, b, [0, 1, 2, 3, 4, 5, 6, 7]); // all a
        swz!($R, a, b, [8, 9, 10, 11, 12, 13, 14, 15]); // all b
        swz!($R, a, b, [15, 14, 13, 12, 11, 10, 9, 8]); // b reversed
        swz!($R, a, b, [0, 8, 1, 9, 2, 10, 3, 11]); // interleave a/b
    }};
}

macro_rules! reg4 {
    ($name:ident, $backend:ty, $reg:ident) => {
        #[test]
        fn $name() {
            battery4!(<$backend as Simd>::$reg);
        }
    };
}
macro_rules! reg8 {
    ($name:ident, $backend:ty, $reg:ident) => {
        #[test]
        fn $name() {
            battery8!(<$backend as Simd>::$reg);
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_const {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    // Native 128-bit registers
    reg4!(v2_f32x4, X86V2, f32x4);
    reg4!(v2_i32x4, X86V2, i32x4);
    reg4!(v2_u32x4, X86V2, u32x4);
    reg4!(v3_f32x4, X86V3, f32x4);
    reg4!(v3_i32x4, X86V3, i32x4);

    // v1 (SSE2): no pshufb, so variable permutes/swizzles take the scalar
    // Register default, a distinct code path from v2/v3.
    reg4!(v1_f32x4, X86V1, f32x4);
    reg4!(v1_i32x4, X86V1, i32x4);
    reg4!(v1_u32x4, X86V1, u32x4);
    reg4!(v1_f64x4, X86V1, f64x4); // ArrayRegister-emulated on v1
    reg4!(v1_i64x4, X86V1, i64x4);
    reg8!(v1_f32x8, X86V1, f32x8);
    reg8!(v1_i32x8, X86V1, i32x8);

    // Native 256-bit registers (V3)
    reg4!(v3_f64x4, X86V3, f64x4);
    reg4!(v3_i64x4, X86V3, i64x4);
    reg8!(v3_f32x8, X86V3, f32x8);
    reg8!(v3_i32x8, X86V3, i32x8);
    reg8!(v3_u32x8, X86V3, u32x8);

    // Scalar reference path (1-lane "register" - trivial but exercises the generic glue)
    reg4!(scalar_f32x4, Scalar, f32x4);
    reg4!(scalar_f64x4, Scalar, f64x4);
}

// WASM: native 128-bit i8x16_swizzle const-index paths.
#[cfg(target_arch = "wasm32")]
mod wasm_const {
    use super::*;
    use thermite::backend::wasm::Wasm;

    reg4!(wasm_f32x4, Wasm, f32x4);
    reg4!(wasm_i32x4, Wasm, i32x4);
    reg4!(wasm_u32x4, Wasm, u32x4);
    reg4!(wasm_f64x4, Wasm, f64x4); // ArrayRegister-emulated
    reg8!(wasm_f32x8, Wasm, f32x8); // ArrayRegister-emulated
}

// NEON: native 128-bit const-index swizzle paths.
#[cfg(target_arch = "aarch64")]
mod neon_const {
    use super::*;
    use thermite::backend::neon::Neon;

    reg4!(neon_f32x4, Neon, f32x4);
    reg4!(neon_i32x4, Neon, i32x4);
    reg4!(neon_u32x4, Neon, u32x4);
    reg4!(neon_f64x4, Neon, f64x4); // ArrayRegister-emulated
    reg8!(neon_f32x8, Neon, f32x8); // ArrayRegister-emulated
}

// ===========================================================================
// Runtime swizzle / permute coverage (ported from the former array_swizzle.rs).
//
// The `swizzle!` batteries above only reach the *const-index* paths. These drive
// the runtime `R::permutev` / `R::swizzle` (live index registers) against each
// register's own `scalar_*` ground truth, with exhaustive single-lane routing
// and random fuzzing. Out-of-range indices produce UNSPECIFIED lane values
// (backend-dependent), so every index generated here stays in range.
//
// Broadened beyond the original (which was V3 + emulated `ArrayRegister` only)
// to native 128-/256-bit registers across v1/v2/v3, so the hardware permute
// paths (`pshufb` on v2, `vpermps` on v3) and the v1 scalar fallback all run.
// ===========================================================================

fn rt_permutev<R: Register>(input: Storage<R>, idxs: &GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    let ir = idx_reg::<R>(idxs);
    let want = R::scalar_permutev(input, ir);
    let got = R::permutev(input, ir);
    // Compare via black-boxed slices, not a direct array `assert_eq!`. For integer
    // element types at -O3 the wasm backend can't select the vectorized all-lanes-
    // equal reduction that array equality lowers to (LLVM "Cannot select ... setcc
    // seteq"); `black_box` on opaque slices forces a scalar compare. (Floats lower
    // via `f32x4.eq`, so this only bit the int register types.)
    let (wa, ga) = (R::as_slice(&want), R::as_slice(&got));
    assert_eq!(
        core::hint::black_box(wa),
        core::hint::black_box(ga),
        "permutev {idxs:?} vs scalar"
    );
}

fn rt_swizzle<R: Register>(a: Storage<R>, b: Storage<R>, idxs: &GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    let ir = idx_reg::<R>(idxs);
    let want = R::scalar_swizzle(a, b, ir);
    let got = R::swizzle(a, b, ir);
    // See rt_permutev: black-boxed slice compare avoids the int -O3 wasm "Cannot select".
    let (wa, ga) = (R::as_slice(&want), R::as_slice(&got));
    assert_eq!(
        core::hint::black_box(wa),
        core::hint::black_box(ga),
        "swizzle {idxs:?} vs scalar"
    );
}

fn run_runtime<R>()
where
    R: Register + NumericRegister,
    R::Element: PartialEq + core::fmt::Debug,
{
    let lanes = <R::Lanes as Unsigned>::USIZE;
    let a = R::indexed();
    let b = R::add(a, a); // distinct second operand for the swizzle source
    let mut idxs = GenericArray::<u32, R::Lanes>::default();

    // identity + reverse
    for i in 0..lanes {
        idxs[i] = i as u32;
    }
    rt_permutev::<R>(a, &idxs);
    rt_swizzle::<R>(a, b, &idxs);
    for i in 0..lanes {
        idxs[i] = (lanes - 1 - i) as u32;
    }
    rt_permutev::<R>(a, &idxs);
    rt_swizzle::<R>(a, b, &idxs);

    // every-lane broadcasts (permute: 0..N; swizzle: 0..2N)
    for t in 0..lanes {
        idxs.iter_mut().for_each(|x| *x = t as u32);
        rt_permutev::<R>(a, &idxs);
    }
    for t in 0..2 * lanes {
        idxs.iter_mut().for_each(|x| *x = t as u32);
        rt_swizzle::<R>(a, b, &idxs);
    }

    // exhaustive single-lane routing (O(N^2)) - catches off-by-one / chunk bugs
    for out in 0..lanes {
        for inl in 0..lanes {
            idxs.iter_mut().for_each(|x| *x = 0);
            idxs[out] = inl as u32;
            rt_permutev::<R>(a, &idxs);
        }
        for inl in 0..2 * lanes {
            idxs.iter_mut().for_each(|x| *x = 0);
            idxs[out] = inl as u32;
            rt_swizzle::<R>(a, b, &idxs);
        }
    }

    // dense random fuzzing (stresses the blendv accumulation)
    let mut prng: rand::rngs::SmallRng = rand::make_rng();
    for _ in 0..2000 {
        for i in 0..lanes {
            idxs[i] = prng.random_range(0..lanes as u32);
        }
        rt_permutev::<R>(a, &idxs);
        for i in 0..lanes {
            idxs[i] = prng.random_range(0..(2 * lanes) as u32);
        }
        rt_swizzle::<R>(a, b, &idxs);
    }
}

macro_rules! rt {
    ($name:ident, $R:ty) => {
        #[test]
        fn $name() {
            run_runtime::<$R>();
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_rt {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    // Emulated ArrayRegister (the original array_swizzle coverage).
    rt!(rt_v3_arr_f32x4x4, ArrayRegister<<X86V3 as Simd>::f32x4, 4>); // 16 lanes, 4 chunks
    rt!(rt_v3_arr_i64x2x4, ArrayRegister<<X86V3 as Simd>::i64x2, 4>); // 8 lanes, 4 chunks
    rt!(rt_v3_f32x16, <X86V3 as Simd>::f32x16);
    rt!(rt_v2_f32x16, <X86V2 as Simd>::f32x16);
    rt!(rt_v1_f32x16, <X86V1 as Simd>::f32x16);

    // Native hardware permute paths + the v1 scalar fallback.
    rt!(rt_v3_f32x4, <X86V3 as Simd>::f32x4);
    rt!(rt_v3_i32x4, <X86V3 as Simd>::i32x4);
    rt!(rt_v3_f32x8, <X86V3 as Simd>::f32x8);
    rt!(rt_v3_i32x8, <X86V3 as Simd>::i32x8);
    rt!(rt_v3_i64x4, <X86V3 as Simd>::i64x4);
    rt!(rt_v3_u64x4, <X86V3 as Simd>::u64x4);
    rt!(rt_v3_f64x2, <X86V3 as Simd>::f64x2);
    rt!(rt_v2_f32x4, <X86V2 as Simd>::f32x4);
    // The v2 64-bit registers gained pshufb-based `permutev` (2026-08-08).
    rt!(rt_v2_f64x2, <X86V2 as Simd>::f64x2);
    rt!(rt_v2_i64x2, <X86V2 as Simd>::i64x2);
    rt!(rt_v2_u64x2, <X86V2 as Simd>::u64x2);
    rt!(rt_v2_i32x4, <X86V2 as Simd>::i32x4);
    rt!(rt_v2_f32x8, <X86V2 as Simd>::f32x8); // ArrayRegister-emulated on v2
    // Chunked 2x4 integer forms: the cross-chunk blend path that thermite-bvh's
    // unmasked-index bug slipped through (2026-08-23), so keep these covered.
    rt!(rt_v2_u32x8, <X86V2 as Simd>::u32x8);
    rt!(rt_v2_i32x8, <X86V2 as Simd>::i32x8);
    rt!(rt_v1_f32x4, <X86V1 as Simd>::f32x4);
    rt!(rt_v1_f32x8, <X86V1 as Simd>::f32x8);

    // Native 16-bit pshufb permute paths: 128-bit (single pshufb) and 256-bit (cross-lane).
    rt!(rt_v2_i16x8, <X86V2 as Simd>::i16x8);
    rt!(rt_v2_u16x8, <X86V2 as Simd>::u16x8);
    rt!(rt_v3_i16x8, <X86V3 as Simd>::i16x8);
    rt!(rt_v3_u16x8, <X86V3 as Simd>::u16x8);
    rt!(rt_v3_i16x16, <X86V3 as Simd>::i16x16);
    rt!(rt_v3_u16x16, <X86V3 as Simd>::u16x16);
    // Reduced (i16x4) and ArrayRegister (i16x2/i16x16-on-v2) forms route through the native permutev.
    rt!(rt_v3_i16x4, <X86V3 as Simd>::i16x4);
    rt!(rt_v2_i16x4, <X86V2 as Simd>::i16x4);
    rt!(rt_v2_i16x16, <X86V2 as Simd>::i16x16);
    // v1 (SSE2): no pshufb, so 16-bit permutes take the scalar Register fallback.
    rt!(rt_v1_i16x8, <X86V1 as Simd>::i16x8);
    rt!(rt_v1_u16x8, <X86V1 as Simd>::u16x8);

    // Native 8-bit pshufb permute paths (the byte index IS the pshufb control): 128-bit on v2.
    rt!(rt_v2_i8x16, <X86V2 as NativeSimd>::i8xN);
    rt!(rt_v2_u8x16, <X86V2 as NativeSimd>::u8xN);
    // v1 (SSE2): no pshufb, so 8-bit permutes take the scalar Register fallback.
    rt!(rt_v1_i8x16, <X86V1 as NativeSimd>::i8xN);
    rt!(rt_v1_u8x16, <X86V1 as NativeSimd>::u8xN);
    // v3 (AVX2): native 256-bit, cross-lane byte permute (pshufb x2 + blend by bit4).
    rt!(rt_v3_i8x32, <X86V3 as NativeSimd>::i8xN);
    rt!(rt_v3_u8x32, <X86V3 as NativeSimd>::u8xN);
    // v3 fixed 128-bit i8x16 (single pshufb permute, distinct register from the 256-bit native).
    rt!(rt_v3_i8x16, <X86V3 as Simd>::i8x16);
    rt!(rt_v3_u8x16, <X86V3 as Simd>::u8x16);
}

// WASM: runtime permute/swizzle via `i8x16`/`u8x16_relaxed_swizzle` (and the scalar/array glue).
#[cfg(target_arch = "wasm32")]
mod wasm_rt {
    use super::*;
    use thermite::backend::wasm::Wasm;

    // native 128-bit + emulated array forms
    rt!(rt_wasm_f32x4, <Wasm as Simd>::f32x4);
    rt!(rt_wasm_i32x4, <Wasm as Simd>::i32x4);
    rt!(rt_wasm_f32x8, <Wasm as Simd>::f32x8); // ArrayRegister-emulated
    rt!(rt_wasm_f32x16, <Wasm as Simd>::f32x16); // ArrayRegister-emulated
    rt!(rt_wasm_arr_f32x4x4, ArrayRegister<<Wasm as Simd>::f32x4, 4>);
    // 16-bit: native i16x8 (byte-doubled relaxed_swizzle), reduced i16x4, array i16x16.
    rt!(rt_wasm_i16x8, <Wasm as Simd>::i16x8);
    rt!(rt_wasm_u16x8, <Wasm as Simd>::u16x8);
    rt!(rt_wasm_i16x4, <Wasm as Simd>::i16x4);
    rt!(rt_wasm_i16x16, <Wasm as Simd>::i16x16);
    // 8-bit: native i8x16 relaxed_swizzle (byte index is the control directly).
    rt!(rt_wasm_i8x16, <Wasm as NativeSimd>::i8xN);
    rt!(rt_wasm_u8x16, <Wasm as NativeSimd>::u8xN);
}

// NEON: runtime permute/swizzle on the native 128-bit registers (and the scalar/array glue).
#[cfg(target_arch = "aarch64")]
mod neon_rt {
    use super::*;
    use thermite::backend::neon::Neon;

    // native 128-bit + emulated array forms
    rt!(rt_neon_f32x4, <Neon as Simd>::f32x4);
    rt!(rt_neon_i32x4, <Neon as Simd>::i32x4);
    rt!(rt_neon_f32x8, <Neon as Simd>::f32x8); // ArrayRegister-emulated
    rt!(rt_neon_f32x16, <Neon as Simd>::f32x16); // ArrayRegister-emulated
    rt!(rt_neon_arr_f32x4x4, ArrayRegister<<Neon as Simd>::f32x4, 4>);
    // 16-bit: native i16x8 (byte-doubled relaxed_swizzle), reduced i16x4, array i16x16.
    // (inherited from the wasm section, revisit for NEON)
    rt!(rt_neon_i16x8, <Neon as Simd>::i16x8);
    rt!(rt_neon_u16x8, <Neon as Simd>::u16x8);
    rt!(rt_neon_i16x4, <Neon as Simd>::i16x4);
    rt!(rt_neon_i16x16, <Neon as Simd>::i16x16);
    // 8-bit: native i8x16 relaxed_swizzle (byte index is the control directly).
    // (inherited from the wasm section, revisit for NEON)
    rt!(rt_neon_i8x16, <Neon as NativeSimd>::i8xN);
    rt!(rt_neon_u8x16, <Neon as NativeSimd>::u8xN);
}
