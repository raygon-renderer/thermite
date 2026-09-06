//! Swizzle / permute coverage, checked against each register's own
//! `scalar_swizzle` / `scalar_permutev` ground truth.
//!
//! Two halves:
//!  - **Constant-index** paths (`swizzle_const` / `permutev_const`, reached via
//!    the public `swizzle!` macro) across native and emulated registers on every
//!    backend. The 4-lane batteries deliberately include the exact index
//!    patterns `impl_mat4_inverse!` relies on.
//!  - **Runtime** paths (`R::permutev` / `R::swizzle` with live index
//!    registers), with exhaustive O(N^2) single-lane routing and random fuzzing.
//!
//! Every shape runs on every backend, so a slot that is native on one backend
//! (`f64x4` on AVX2) is the `ArrayRegister` path on another (SSE, WASM, NEON).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::{GenericArray, arr, typenum::Unsigned};
use rand::RngExt;
use thermite::Vector;
use thermite::register::array::ArrayRegister;
use thermite::register::{NumericRegister, Register, Storage};

use thermite::simd::{NativeSimd, Simd};

/// Build the live index register `permutev`/`swizzle` consume from a `u32`
/// index array (the test-side mirror of the crate's internal converter).
#[inline(always)]
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

macro_rules! battery4 {
    ($R:ty) => {{
        let a = <$R>::indexed();
        let b = <$R>::add(a, a); // distinct second operand (= 2*a)

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

        swz!($R, a, b, [0, 1, 2, 3]); // identity (all a)
        swz!($R, a, b, [4, 5, 6, 7]); // all b
        swz!($R, a, b, [7, 6, 5, 4]); // b reversed
        swz!($R, a, b, [0, 4, 1, 5]); // interleave
        swz!($R, a, b, [3, 3, 7, 7]);
        swz!($R, a, b, [2, 2, 6, 6]);
        swz!($R, a, b, [1, 1, 5, 5]);
        swz!($R, a, b, [0, 0, 4, 4]);
        swz!($R, a, b, [0, 2, 4, 6]);
    }};
}

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

// ---------------------------------------------------------------------------
// Runtime (live index register) paths.
// ---------------------------------------------------------------------------

#[inline(always)]
fn rt_permutev<R: Register>(input: Storage<R>, idxs: &GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    let ir = idx_reg::<R>(idxs);
    let want = R::scalar_permutev(input, ir);
    let got = R::permutev(input, ir);
    let (wa, ga) = (R::as_slice(&want), R::as_slice(&got));
    assert_eq!(
        core::hint::black_box(wa),
        core::hint::black_box(ga),
        "permutev {idxs:?} vs scalar"
    );
}

#[inline(always)]
fn rt_swizzle<R: Register>(a: Storage<R>, b: Storage<R>, idxs: &GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    let ir = idx_reg::<R>(idxs);
    let want = R::scalar_swizzle(a, b, ir);
    let got = R::swizzle(a, b, ir);
    let (wa, ga) = (R::as_slice(&want), R::as_slice(&got));
    assert_eq!(
        core::hint::black_box(wa),
        core::hint::black_box(ga),
        "swizzle {idxs:?} vs scalar"
    );
}

/// Identity, reverse, every broadcast, exhaustive single-lane routing
/// (O(N^2)), then 2000 random index vectors, for both `permutev` and
/// `swizzle`.
#[inline(always)]
fn run_runtime<R>()
where
    R: Register + NumericRegister,
    R::Element: PartialEq + core::fmt::Debug,
{
    let lanes = <R::Lanes as Unsigned>::USIZE;
    let a = R::indexed();
    let b = R::add(a, a); // distinct second operand for the swizzle source
    let mut idxs = GenericArray::<u32, R::Lanes>::default();

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

    for t in 0..lanes {
        idxs.iter_mut().for_each(|x| *x = t as u32);
        rt_permutev::<R>(a, &idxs);
    }
    for t in 0..2 * lanes {
        idxs.iter_mut().for_each(|x| *x = t as u32);
        rt_swizzle::<R>(a, b, &idxs);
    }

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

for_each_backend! {
    fn const_4lane<S: Simd>() {
        battery4!(<S as Simd>::f32x4);
        battery4!(<S as Simd>::i32x4);
        battery4!(<S as Simd>::u32x4);
        battery4!(<S as Simd>::f64x4);
        battery4!(<S as Simd>::i64x4);
        battery4!(<S as Simd>::u64x4);
    }
    fn const_8lane<S: Simd>() {
        battery8!(<S as Simd>::f32x8);
        battery8!(<S as Simd>::i32x8);
        battery8!(<S as Simd>::u32x8);
        battery8!(<S as Simd>::f64x8);
        battery8!(<S as Simd>::i64x8);
    }

    fn rt_32bit<S: Simd>() {
        run_runtime::<<S as Simd>::f32x4>();
        run_runtime::<<S as Simd>::i32x4>();
        run_runtime::<<S as Simd>::u32x4>();
        run_runtime::<<S as Simd>::f32x8>();
        run_runtime::<<S as Simd>::i32x8>();
        run_runtime::<<S as Simd>::u32x8>();
        run_runtime::<<S as Simd>::f32x16>();
        run_runtime::<<S as Simd>::i32x16>();
    }
    fn rt_64bit<S: Simd>() {
        run_runtime::<<S as Simd>::f64x2>();
        run_runtime::<<S as Simd>::i64x2>();
        run_runtime::<<S as Simd>::u64x2>();
        run_runtime::<<S as Simd>::f64x4>();
        run_runtime::<<S as Simd>::i64x4>();
        run_runtime::<<S as Simd>::u64x4>();
        run_runtime::<<S as Simd>::f64x8>();
        run_runtime::<<S as Simd>::u64x8>();
    }
    /// Explicit `ArrayRegister` shapes: 4 chunks of a native register.
    fn rt_array<S: Simd>() {
        run_runtime::<ArrayRegister<<S as Simd>::f32x4, 4>>(); // 16 lanes, 4 chunks
        run_runtime::<ArrayRegister<<S as Simd>::i64x2, 4>>(); // 8 lanes, 4 chunks
    }
    fn rt_16bit<S: Simd>() {
        run_runtime::<<S as Simd>::i16x4>();
        run_runtime::<<S as Simd>::i16x8>();
        run_runtime::<<S as Simd>::u16x8>();
        run_runtime::<<S as Simd>::i16x16>();
        run_runtime::<<S as Simd>::u16x16>();
        run_runtime::<<S as NativeSimd>::i16xN>();
    }
    fn rt_8bit<S: Simd>() {
        run_runtime::<<S as Simd>::i8x16>();
        run_runtime::<<S as Simd>::u8x16>();
        run_runtime::<<S as NativeSimd>::i8xN>();
        run_runtime::<<S as NativeSimd>::u8xN>();
    }
}
