//! `Register::compress` / `compress_z` / `expand` / `expand_z` and their `_n`
//! forms on every backend, checked against a scalar partition oracle. Exercises
//! the wired `compress_via_table!` (<= 8 lanes) and `compress_via_wide!`
//! (16/32/64-lane) overrides, the `ArrayRegister` merge trees, and the AVX-512
//! `vpcompress`/`vpexpand` forms.
//!
//! Every shape runs on every backend. Which lowering a shape takes is the
//! backend's business (an x2 slot is a table on x86 and scalar on `Scalar`).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use thermite::register::array::ArrayRegister;
use thermite::register::{Element, Register, Storage};
use thermite::simd::{NativeSimd, Simd};

/// Mask patterns for an `n`-lane register: exhaustive up to 16 lanes, a
/// deterministic xorshift sample above that.
fn patterns(n: usize) -> Box<dyn Iterator<Item = u64>> {
    if n <= 16 {
        Box::new(0..(1u64 << n))
    } else {
        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        Box::new((0..20000).map(move |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & mask
        }))
    }
}

/// Compare `R::compress` (stable partition) and `R::compress_z` (zeroing) to
/// scalar oracles over every mask pattern.
#[inline(always)]
fn check<R: Register>() {
    let n = <R::Lanes as Unsigned>::USIZE;

    // Distinct nonzero lane values so a dropped/misplaced lane is visible.
    let v: Storage<R> = R::new(GenericArray::generate(|i| {
        <R::Element as Element>::from_u8((i + 1) as u8)
    }));
    let vals: Vec<R::Element> = R::as_slice(&v).to_vec();

    for bits in patterns(n) {
        let sel: GenericArray<R::Element, R::Lanes> =
            GenericArray::generate(|i| <R::Element as Element>::from_u8(((bits >> i) & 1) as u8));
        let mask = R::into_mask(R::new(sel));

        // Non-zeroing oracle: selected in order, then unselected in order.
        // Zeroing oracle: selected in order, then zeros.
        let mut part = vec![R::Element::default(); n];
        let mut zero = vec![R::Element::default(); n];
        let mut pos = 0;
        for l in 0..n {
            if (bits >> l) & 1 == 1 {
                part[pos] = vals[l];
                zero[pos] = vals[l];
                pos += 1;
            }
        }
        for l in 0..n {
            if (bits >> l) & 1 == 0 {
                part[pos] = vals[l];
                pos += 1;
            }
        }

        let got = R::compress(v, mask);
        let got_z = R::compress_z(v, mask);
        assert_eq!(R::as_slice(&got), &part[..], "compress n={n} bits={bits:b}");
        assert_eq!(R::as_slice(&got_z), &zero[..], "compress_z n={n} bits={bits:b}");
    }
}

/// Compare `R::expand` (the inverse stable partition) and `R::expand_z`
/// (zeroing) to scalar oracles over every mask pattern.
///
/// The `ArrayRegister` shapes route both through the branchless wide scatter
/// (`expand_permute_wide`) above 8 lanes. Below that they take the scalar
/// default, which this test also covers.
#[inline(always)]
fn check_expand<R: Register>() {
    let n = <R::Lanes as Unsigned>::USIZE;

    let v: Storage<R> = R::new(GenericArray::generate(|i| {
        <R::Element as Element>::from_u8((i + 1) as u8)
    }));
    let vals: Vec<R::Element> = R::as_slice(&v).to_vec();

    for bits in patterns(n) {
        let sel: GenericArray<R::Element, R::Lanes> =
            GenericArray::generate(|i| <R::Element as Element>::from_u8(((bits >> i) & 1) as u8));
        let mask = R::into_mask(R::new(sel));

        // Non-zeroing oracle: selected lanes read the packed front in order,
        // unselected lanes read the tail in order.
        // Zeroing oracle: selected lanes only, everything else zero.
        let mut full = vec![R::Element::default(); n];
        let mut zero = vec![R::Element::default(); n];
        let mut pos = 0;
        for l in 0..n {
            if (bits >> l) & 1 == 1 {
                full[l] = vals[pos];
                zero[l] = vals[pos];
                pos += 1;
            }
        }
        for l in 0..n {
            if (bits >> l) & 1 == 0 {
                full[l] = vals[pos];
                pos += 1;
            }
        }

        let got = R::expand(v, mask);
        let got_z = R::expand_z(v, mask);
        assert_eq!(R::as_slice(&got), &full[..], "expand n={n} bits={bits:b}");
        assert_eq!(R::as_slice(&got_z), &zero[..], "expand_z n={n} bits={bits:b}");
    }
}

/// The `_n` family: `compress_n` / `compress_z_n` / `expand_n` / `expand_z_n`
/// must be **bit-identical** to `N` separate single-vector calls under the same
/// mask, for every mask pattern.
///
/// That is the whole contract. The `_n` forms only share the mask-derived plan
/// (movemask, table rows, index registers, merge controls), never change the
/// permutation, so comparing against the single-vector ops is a complete
/// check, and it transitively inherits those ops' own oracle coverage above.
///
/// The `N` value registers carry disjoint value ranges so a lane leaking
/// between values is visible, not just a lane leaking between positions.
#[inline(always)]
fn check_n<R: Register, const N: usize>() {
    let n = <R::Lanes as Unsigned>::USIZE;

    let mut vals = [R::EMPTY; N];
    for (k, slot) in vals.iter_mut().enumerate() {
        // Distinct per (value, lane), wrapped into the element's range.
        *slot = R::new(GenericArray::generate(|i| {
            <R::Element as Element>::from_u8(((k * n + i + 1) % 251 + 1) as u8)
        }));
    }

    for bits in patterns(n) {
        let sel: GenericArray<R::Element, R::Lanes> =
            GenericArray::generate(|i| <R::Element as Element>::from_u8(((bits >> i) & 1) as u8));
        let mask = R::into_mask(R::new(sel));

        let got_c = R::compress_n::<N>(vals, mask);
        let got_cz = R::compress_z_n::<N>(vals, mask);
        let got_e = R::expand_n::<N>(vals, mask);
        let got_ez = R::expand_z_n::<N>(vals, mask);

        for k in 0..N {
            assert_eq!(
                R::as_slice(&got_c[k]),
                R::as_slice(&R::compress(vals[k], mask)),
                "compress_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_cz[k]),
                R::as_slice(&R::compress_z(vals[k], mask)),
                "compress_z_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_e[k]),
                R::as_slice(&R::expand(vals[k], mask)),
                "expand_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_ez[k]),
                R::as_slice(&R::expand_z(vals[k], mask)),
                "expand_z_n N={N} k={k} lanes={n} bits={bits:b}"
            );
        }
    }
}

for_each_backend! {
    /// <= 8-lane shapes (`compress_via_table!` on x86). The integer 64x4
    /// registers route through their own `permutev` override (the
    /// doubled-index `vpermd`), not f64x4's, so cover both.
    fn compress_table<S: Simd>() {
        check::<<S as Simd>::f32x8>();
        check::<<S as Simd>::i32x8>();
        check::<<S as Simd>::u16x8>();
        check::<<S as Simd>::i16x8>();
        check::<<S as Simd>::f32x4>();
        check::<<S as Simd>::f64x4>();
        check::<<S as Simd>::i64x4>();
        check::<<S as Simd>::u64x4>();
        check::<<S as Simd>::i64x2>();
        check::<<S as Simd>::u64x2>();
        check::<<S as Simd>::f64x2>();
    }

    /// 16-lane and native-width byte shapes (`compress_via_wide!` on x86,
    /// 4-group routing at 32 lanes, 64 lanes on AVX-512).
    fn compress_wide<S: Simd>() {
        check::<<S as Simd>::i16x16>();
        check::<<S as Simd>::u16x16>();
        check::<<S as Simd>::i8x16>();
        check::<<S as Simd>::u8x16>();
        check::<<S as NativeSimd>::i8xN>();
        check::<<S as NativeSimd>::u8xN>();
        check::<<S as NativeSimd>::i16xN>();
    }

    /// Emulated-wide `ArrayRegister` shapes: `compress_z` takes the merge tree
    /// (chunk compress + count-indexed merges), `compress` the stable-partition
    /// default. f32x16 is 2x8 on AVX2 and 4x4 on the 128-bit backends (a
    /// two-level pairwise tree, so the M=2 chunk-shift stage as well as M=1).
    /// The byte arrays are built from the fixed 16-lane slot, not `u8xN`: a
    /// `2 * Native8Width` lane count is not provable for a generic `S`.
    fn compress_array<S: Simd>() {
        check::<<S as Simd>::f32x16>();
        check::<<S as Simd>::i32x16>();
        check::<ArrayRegister<<S as Simd>::u8x16,2>>();
    }

    /// `expand` / `expand_z` on the emulated-wide `ArrayRegister` shapes, whose
    /// >8-lane arms route onto `expand_permute_wide` (branchless).
    fn expand_array<S: Simd>() {
        check_expand::<<S as Simd>::f32x16>();
        check_expand::<<S as Simd>::i32x16>();
        check_expand::<<S as Simd>::i16x16>();
        check_expand::<ArrayRegister<<S as Simd>::u8x16,2>>();
    }

    /// `expand` / `expand_z` on the native wide registers (the
    /// `compress_via_wide!` shapes), where both arms are grouped kernels.
    fn expand_wide<S: Simd>() {
        check_expand::<<S as Simd>::u16x16>();
        check_expand::<<S as Simd>::i8x16>();
        check_expand::<<S as Simd>::u8x16>();
        check_expand::<<S as NativeSimd>::i8xN>();
        check_expand::<<S as NativeSimd>::u8xN>();
    }

    /// `_n` on the <= 8-lane table path (one row fetch, `N` `permutev_row`s).
    fn n_family_table<S: Simd>() {
        check_n::<<S as Simd>::i32x8, 2>();
        check_n::<<S as Simd>::i32x8, 4>();
    }

    /// `_n` on the native grouped kernels: one plan for the zeroing forms, two
    /// plus a shared shift control for the non-zeroing ones.
    fn n_family_grouped<S: Simd>() {
        check_n::<<S as Simd>::u16x16, 2>();
        check_n::<<S as Simd>::u16x16, 4>();
        check_n::<<S as NativeSimd>::u8xN, 2>();
        check_n::<<S as NativeSimd>::u8xN, 4>();
    }

    /// `_n` on the emulated `ArrayRegister` shapes.
    fn n_family_array<S: Simd>() {
        check_n::<<S as Simd>::f32x16, 2>();
        check_n::<<S as Simd>::f32x16, 4>();
        check_n::<ArrayRegister<<S as Simd>::u8x16,4>, 2>();
    }
}
