//! Gather / scatter / lookup, tested at the **`Vector` (public API) layer**.
//!
//! The safe wrappers (`gather`, `gather_or`, `gather_or_zero`, `lookup`,
//! `scatter`) live only at the `Vector` layer (the register `*_ptr` primitives
//! are unchecked), so their bounds-checking and masking logic is only covered here.
//! Each is oracled against plain Rust slice indexing.
//!
//! One generic `fn check_gather::<V>()` over any `GenericVector`, instantiated
//! per backend/width by `for_each_backend!`. Indices are the vector's own
//! `Unsigned` type. Values are bit-preserving, so NaN lanes must match too.
//! On wasm gather is the scalar-fallback IndexableRegister path (no hw gather),
//! which still validates the API.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use rand::RngExt;

use harness::{Diff, Tol};
use thermite::Vector;
use thermite::mask::CastMask;
use thermite::simd::Simd;
use thermite::vector::{GenericVector, VectorIndices};

const TRIALS: usize = 256;

/// Build an index vector (the vector's `Unsigned` companion) from `usize` lanes.
fn mk_indices<V>(us: &[usize]) -> V::Unsigned
where
    V: GenericVector,
    <V::Unsigned as GenericVector>::Element: TryFrom<usize>,
{
    let e: Vec<<V::Unsigned as GenericVector>::Element> = us
        .iter()
        .map(|&u| {
            <<V::Unsigned as GenericVector>::Element as TryFrom<usize>>::try_from(u)
                .ok()
                .unwrap()
        })
        .collect();
    <V::Unsigned as GenericVector>::from_slice(&e)
}

#[inline(always)]
fn check_gather<V>(label: &str)
where
    V: GenericVector<Element: Diff + Default>,
    V::Unsigned: VectorIndices<V>,
    V::Mask: CastMask<<V::Unsigned as GenericVector>::Mask>,
    <V::Unsigned as GenericVector>::Element: TryFrom<usize>,
{
    let mut rng = harness::rng();
    let lanes = V::LANES;
    let len = lanes * 4; // backing slice length
    let zero = <V::Element as Default>::default();
    let read = |v: V| v.into_array().as_slice().to_vec();
    let rand_vec = |rng: &mut _, n: usize| (0..n).map(|_| <V::Element as Diff>::rand(rng)).collect::<Vec<_>>();

    for _ in 0..TRIALS {
        let data = rand_vec(&mut rng, len);

        // --- gather: every index in bounds. result[lane] == data[idx[lane]] ---
        let idx: Vec<usize> = (0..lanes).map(|_| rng.random_range(0..len)).collect();
        let got = read(V::gather(&data, mk_indices::<V>(&idx)));
        let want: Vec<V::Element> = idx.iter().map(|&i| data[i]).collect();
        harness::assert_lanes_eq(&format!("{label} [gather]"), &[], &got, &want, Tol::Exact);

        // --- gather_or: ~half the indices out of bounds -> `or` lane ---
        let idx2: Vec<usize> = (0..lanes).map(|_| rng.random_range(0..len * 2)).collect();
        let or_e = rand_vec(&mut rng, lanes);
        let got = read(V::gather_or(&data, mk_indices::<V>(&idx2), V::from_slice(&or_e)));
        let want: Vec<V::Element> = idx2
            .iter()
            .enumerate()
            .map(|(ln, &i)| if i < len { data[i] } else { or_e[ln] })
            .collect();
        harness::assert_lanes_eq(&format!("{label} [gather_or]"), &[], &got, &want, Tol::Exact);

        // --- gather_or_zero: out of bounds -> 0 ---
        let got = read(V::gather_or_zero(&data, mk_indices::<V>(&idx2)));
        let want: Vec<V::Element> = idx2.iter().map(|&i| if i < len { data[i] } else { zero }).collect();
        harness::assert_lanes_eq(&format!("{label} [gather_or_zero]"), &[], &got, &want, Tol::Exact);

        // --- lookup: small table, out of bounds -> table[0] ---
        let table = rand_vec(&mut rng, lanes);
        let idxl: Vec<usize> = (0..lanes).map(|_| rng.random_range(0..lanes * 2)).collect();
        let got = read(V::lookup(&table, mk_indices::<V>(&idxl)));
        let want: Vec<V::Element> = idxl
            .iter()
            .map(|&i| if i < lanes { table[i] } else { table[0] })
            .collect();
        harness::assert_lanes_eq(&format!("{label} [lookup]"), &[], &got, &want, Tol::Exact);

        // --- scatter: unique in-bounds indices (a permutation), so the winning
        // lane is deterministic. out[idx[lane]] == value[lane] ---
        let mut perm: Vec<usize> = (0..lanes).collect();
        for i in (1..lanes).rev() {
            perm.swap(i, rng.random_range(0..i + 1));
        }
        let val_e = rand_vec(&mut rng, lanes);
        let mut out = vec![zero; lanes];
        V::from_slice(&val_e).scatter(&mut out, mk_indices::<V>(&perm));
        let mut want = vec![zero; lanes];
        for (ln, &dst) in perm.iter().enumerate() {
            want[dst] = val_e[ln];
        }
        harness::assert_lanes_eq(&format!("{label} [scatter]"), &[], &out, &want, Tol::Exact);
    }
}

macro_rules! gather {
    ($S:ty, $reg:ident) => {
        check_gather::<Vector<<$S as Simd>::$reg>>(&harness::label::<$S>(stringify!($reg)))
    };
}

for_each_backend! {
    fn f32x4<S: Simd>() { gather!(S, f32x4) }
    fn f32x8<S: Simd>() { gather!(S, f32x8) }
    fn f32x16<S: Simd>() { gather!(S, f32x16) }
    fn f64x2<S: Simd>() { gather!(S, f64x2) }
    fn f64x4<S: Simd>() { gather!(S, f64x4) }
    fn f64x8<S: Simd>() { gather!(S, f64x8) }
    fn i32x4<S: Simd>() { gather!(S, i32x4) }
    fn i32x8<S: Simd>() { gather!(S, i32x8) }
    fn i32x16<S: Simd>() { gather!(S, i32x16) }
    fn i64x2<S: Simd>() { gather!(S, i64x2) }
    fn i64x4<S: Simd>() { gather!(S, i64x4) }
    fn i64x8<S: Simd>() { gather!(S, i64x8) }
    fn u32x4<S: Simd>() { gather!(S, u32x4) }
    fn u32x8<S: Simd>() { gather!(S, u32x8) }
    fn u32x16<S: Simd>() { gather!(S, u32x16) }
    fn u64x2<S: Simd>() { gather!(S, u64x2) }
    fn u64x4<S: Simd>() { gather!(S, u64x4) }
    fn u64x8<S: Simd>() { gather!(S, u64x8) }
}
