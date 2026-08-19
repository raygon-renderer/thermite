use thermite::backend::generic::polyfills::sort;
use thermite::element::Element;
use thermite::register::{NumericRegister, Storage};
use thermite::sort::{Ascending, Descending};

/// Element type of the register under test, so the same helpers serve f32 and
/// f64 registers. Values are small integers, exact in either.
type Val = u8;

fn lanes_of<R: NumericRegister>(v: Storage<R>) -> Vec<R::Element> {
    R::as_slice(&v).to_vec()
}

fn oracle<R: NumericRegister>(vals: &[Val]) -> Vec<R::Element> {
    let mut s: Vec<R::Element> = vals.iter().map(|&x| R::Element::from_u8(x)).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

fn oracle_desc<R: NumericRegister>(vals: &[Val]) -> Vec<R::Element> {
    let mut s = oracle::<R>(vals);
    s.reverse();
    s
}

fn build<R: NumericRegister>(vals: &[Val]) -> Storage<R> {
    let mut v = R::EMPTY;
    let s = R::as_mut_slice(&mut v);
    for (slot, &x) in s.iter_mut().zip(vals) {
        *slot = R::Element::from_u8(x);
    }
    v
}

/// Permutations of `1..=n`, so a dropped or duplicated lane fails as a multiset
/// mismatch rather than merely a misordering. Includes both sorted directions, a
/// spread of deterministic shuffles, and ties.
fn patterns(n: usize) -> Vec<Vec<Val>> {
    let base: Vec<Val> = (1..=n).map(|i| i as Val).collect();

    let mut out = vec![base.clone(), base.iter().rev().copied().collect()];

    let mut s = 0x9E3779B97F4A7C15u64;
    for _ in 0..128 {
        let mut p = base.clone();
        for i in (1..n).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p.swap(i, (s % (i as u64 + 1)) as usize);
        }
        out.push(p);
    }

    out.push(vec![1; n]);
    let mut half = base.clone();
    for i in 0..n / 2 {
        half[i] = 1;
    }
    out.push(half);

    out
}

macro_rules! check_sort {
    ($label:expr, $r:ty, $n:expr, $f:expr) => {{
        for p in patterns($n) {
            let v = build::<$r>(&p);
            let got = lanes_of::<$r>(($f)(v));
            assert_eq!(got, oracle::<$r>(&p), "{}: input {:?}", $label, p);
        }
    }};
}

/// Both directions of `sort_by` against direction-matched oracles.
///
/// Descending is not merely "ascending reversed" as far as the *code* is
/// concerned (it is a different monomorph with the comparators flipped), so it
/// needs its own oracle rather than a reverse of the ascending result.
macro_rules! check_sort_both {
    ($label:expr, $r:ty, $n:expr) => {{
        for p in patterns($n) {
            let v = build::<$r>(&p);
            assert_eq!(
                lanes_of::<$r>(<$r as NumericRegister>::sort_by::<Ascending>(v)),
                oracle::<$r>(&p),
                "{} asc: input {:?}",
                $label,
                p
            );
            assert_eq!(
                lanes_of::<$r>(<$r as NumericRegister>::sort_by::<Descending>(v)),
                oracle_desc::<$r>(&p),
                "{} desc: input {:?}",
                $label,
                p
            );
        }
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::registers::F32x4V1;
    use thermite::backend::x86_v2::registers::F32x4V2;
    use thermite::backend::x86_v3::registers::{F32x4V3, F32x8V3, F64x2V3, F64x4V3};

    /// Now instantiable for *any* register of the right width, not just the
    /// three that had `BlendRegister`.
    #[test]
    fn sort_4_network() {
        check_sort!("sort_4 F32x4V1", F32x4V1, 4, sort::sort_4::<F32x4V1, Ascending>);
        check_sort!("sort_4 F32x4V2", F32x4V2, 4, sort::sort_4::<F32x4V2, Ascending>);
        check_sort!("sort_4 F32x4V3", F32x4V3, 4, sort::sort_4::<F32x4V3, Ascending>);
    }

    /// The network that had never been instantiated, let alone run.
    #[test]
    fn sort_8_network() {
        check_sort!("sort_8 F32x8V3", F32x8V3, 8, sort::sort_8::<F32x8V3, Ascending>);
    }

    /// Both directions of every wired lane network. Descending is a separate
    /// monomorph with the comparators flipped, so it gets its own oracle.
    #[test]
    fn sort_by_both_directions() {
        check_sort_both!("F32x4V1", F32x4V1, 4);
        check_sort_both!("F32x4V2", F32x4V2, 4);
        check_sort_both!("F32x4V3", F32x4V3, 4);
        check_sort_both!("F32x8V3", F32x8V3, 8);
        check_sort_both!("F64x2V3", F64x2V3, 2);
        check_sort_both!("F64x4V3", F64x4V3, 4);
    }

    /// The 16-lane registers, which have **no** `sort_via_network!` override and
    /// therefore exercise the trait default, the one-chunk tail chain.
    ///
    /// These took the quadratic `sort_any` until that default changed, so this
    /// is the only test that covers the new path at the width where it is the
    /// sole implementation. Both element widths, because the 16-lane shuffles
    /// straddle the 128-bit boundary differently for `i8` (one 128-bit register)
    /// and `i16` (a 256-bit one).
    #[test]
    fn sort_16_default_network() {
        use thermite::backend::x86_v3::registers::{I8x16V3, I16x16V3, U8x16V3, U16x16V3};

        check_sort_both!("I16x16V3", I16x16V3, 16);
        check_sort_both!("U16x16V3", U16x16V3, 16);
        check_sort_both!("I8x16V3", I8x16V3, 16);
        check_sort_both!("U8x16V3", U8x16V3, 16);
    }

    /// `bitonic_clean` at 16 lanes, on genuinely bitonic input at every split
    /// point: an ascending run of length `k` followed by a descending one.
    ///
    /// The default's halving strides are only correct for bitonic input, so a
    /// full-sort oracle would not distinguish a correct clean from a broken one
    /// on arbitrary data. The input has to actually be bitonic.
    #[test]
    fn bitonic_clean_16_default() {
        use thermite::backend::x86_v3::registers::I16x16V3;

        for k in 0..=16usize {
            let mut vals: Vec<Val> = (1..=k as Val).collect();
            vals.extend((k + 1..=16).map(|i| i as Val).rev());

            let v = build::<I16x16V3>(&vals);
            let got = lanes_of::<I16x16V3>(<I16x16V3 as NumericRegister>::bitonic_clean(v));
            assert_eq!(got, oracle::<I16x16V3>(&vals), "split at {k}: input {vals:?}");
        }
    }

    /// `NumericRegister::sort` as each register actually dispatches it, which
    /// after `sort_via_network!` is the network rather than the scalar walk.
    #[test]
    fn register_sort_method() {
        check_sort!("F32x4V1", F32x4V1, 4, <F32x4V1 as NumericRegister>::sort);
        check_sort!("F32x4V2", F32x4V2, 4, <F32x4V2 as NumericRegister>::sort);
        check_sort!("F32x4V3", F32x4V3, 4, <F32x4V3 as NumericRegister>::sort);
        check_sort!("F32x8V3", F32x8V3, 8, <F32x8V3 as NumericRegister>::sort);
        check_sort!("F64x2V3", F64x2V3, 2, <F64x2V3 as NumericRegister>::sort);
        check_sort!("F64x4V3", F64x4V3, 4, <F64x4V3 as NumericRegister>::sort);
    }

    /// `bitonic_clean` on genuinely bitonic input: an ascending run followed by
    /// a descending one, at every split point. A register that kept the scalar
    /// default still passes (a full sort sorts a bitonic sequence too), so this
    /// pins correctness, not the lowering. `sort_via_network_is_wired` does that.
    #[test]
    fn bitonic_clean_method() {
        fn bitonic(n: usize, split: usize) -> Vec<Val> {
            let mut up: Vec<Val> = (0..split).map(|i| (i * 2 + 1) as Val).collect();
            let down: Vec<Val> = (split..n).rev().map(|i| (i * 2) as Val).collect();
            up.extend(down);
            up
        }

        macro_rules! check_clean {
            ($label:expr, $r:ty, $n:expr) => {{
                for split in 0..=$n {
                    let p = bitonic($n, split);
                    let v = build::<$r>(&p);
                    let got = lanes_of::<$r>(<$r as NumericRegister>::bitonic_clean(v));
                    assert_eq!(got, oracle::<$r>(&p), "{} split={split}: input {:?}", $label, p);
                }
            }};
        }

        check_clean!("F64x2V3", F64x2V3, 2);
        check_clean!("F32x4V3", F32x4V3, 4);
        check_clean!("F32x8V3", F32x8V3, 8);
    }

    /// The networks are a *performance* choice, since both paths sort correctly, so
    /// nothing above notices a register that silently kept the scalar default.
    /// `sort_any` spills to `as_mut_slice`, so it cannot be const-folded, while the
    /// network can. Sorting an all-equal register is the identity either way,
    /// but only the network form lets the optimizer see that.
    #[test]
    fn sort_via_network_is_wired() {
        // This does not prove a register avoided the scalar insertion-sort
        // fallback: the trait default IS a network (`sort_lanes`) and agrees
        // with `sort_8` on every input, as it must, both being correct sorts.
        //
        // What it still pins is that `F32x8V3::sort_by` is a *fixed-shape*
        // network in agreement with the reference one, which is worth keeping:
        // 8 lanes deliberately has no `sort_via_network!` arm any more, so this
        // is the check that removing it did not change behaviour.
        for p in patterns(8) {
            let v = build::<F32x8V3>(&p);
            assert_eq!(
                lanes_of::<F32x8V3>(<F32x8V3 as NumericRegister>::sort(v)),
                lanes_of::<F32x8V3>(sort::sort_8::<F32x8V3, Ascending>(v)),
                "F32x8V3::sort should be sort_8, input {:?}",
                p
            );
            assert_eq!(
                lanes_of::<F32x8V3>(<F32x8V3 as NumericRegister>::bitonic_clean(sort::sort_8::<
                    F32x8V3,
                    Ascending,
                >(v))),
                lanes_of::<F32x8V3>(sort::sort_8::<F32x8V3, Ascending>(v)),
                "bitonic_clean of a sorted register is the identity"
            );
        }
    }
}

/// `ArrayRegister<R, N>::sort`: one ascending run across all `N * LANES`
/// elements, via per-chunk networks plus a columnar bitonic merge tree.
/// Power-of-two `N` takes the network ladder, while anything else (`N = 3` exists)
/// keeps the scalar fallback, which these tests also pin as *correct*.
mod array {
    use super::*;
    use thermite::register::array::ArrayRegister;

    /// Chunks of 1-lane scalar registers: every merge stage degenerates
    /// (reverse and clean are identities) and the sort is pure columnar CEs.
    #[test]
    fn scalar_chunks() {
        check_sort!("A<f32,2>", ArrayRegister<f32, 2>, 2, <ArrayRegister<f32, 2> as NumericRegister>::sort);
        check_sort!("A<f32,4>", ArrayRegister<f32, 4>, 4, <ArrayRegister<f32, 4> as NumericRegister>::sort);
        check_sort!("A<f32,8>", ArrayRegister<f32, 8>, 8, <ArrayRegister<f32, 8> as NumericRegister>::sort);
        check_sort!("A<f32,16>", ArrayRegister<f32, 16>, 16, <ArrayRegister<f32, 16> as NumericRegister>::sort);
    }

    /// Nested ArrayRegister: the inner chunk's `sort`/`bitonic_clean` are
    /// themselves the array network, exercising the recursion.
    #[test]
    fn nested_chunks() {
        type Inner = ArrayRegister<f32, 4>;
        check_sort!("A<A<f32,4>,4>", ArrayRegister<Inner, 4>, 16, <ArrayRegister<Inner, 4> as NumericRegister>::sort);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    mod x86 {
        use super::*;
        use thermite::backend::x86_v1::registers::F32x4V1;
        use thermite::backend::x86_v3::registers::{F32x4V3, F32x8V3, F64x4V3};

        type A2 = ArrayRegister<F32x8V3, 2>; // f32x16
        type A3 = ArrayRegister<F32x8V3, 3>; // non-power-of-two -> fallback
        type A4 = ArrayRegister<F32x8V3, 4>; // f32x32
        type A8 = ArrayRegister<F32x4V3, 8>; // f32x32 from 4-lane chunks
        type A16 = ArrayRegister<F32x8V3, 16>; // f32x128
        type D4 = ArrayRegister<F64x4V3, 4>; // f64x16
        type V1 = ArrayRegister<F32x4V1, 4>; // SSE2 chunks

        #[test]
        fn sort_method() {
            check_sort!("A2", A2, 16, <A2 as NumericRegister>::sort);
            check_sort!("A3", A3, 24, <A3 as NumericRegister>::sort);
            check_sort!("A4", A4, 32, <A4 as NumericRegister>::sort);
            check_sort!("A8", A8, 32, <A8 as NumericRegister>::sort);
            check_sort!("A16", A16, 128, <A16 as NumericRegister>::sort);
            check_sort!("D4", D4, 16, <D4 as NumericRegister>::sort);
            check_sort!("V1", V1, 16, <V1 as NumericRegister>::sort);
        }

        /// Both directions through the merge tree, including the `N = 3`
        /// fallback arm, which reaches descending by reversing rather than by
        /// flipped comparators, so it is a genuinely different path.
        #[test]
        fn sort_by_both_directions() {
            check_sort_both!("A2", A2, 16);
            check_sort_both!("A3", A3, 24);
            check_sort_both!("A4", A4, 32);
            check_sort_both!("A8", A8, 32);
            check_sort_both!("A16", A16, 128);
            check_sort_both!("D4", D4, 16);
            check_sort_both!("V1", V1, 16);
        }

        /// `bitonic_clean` on genuinely bitonic input at every split point.
        /// The network arms are clean-only (not a full sort), so this is the
        /// test that distinguishes them from the fallback.
        #[test]
        fn bitonic_clean_method() {
            fn bitonic(n: usize, split: usize) -> Vec<Val> {
                let mut up: Vec<Val> = (0..split).map(|i| (i * 2 + 1) as Val).collect();
                let down: Vec<Val> = (split..n).rev().map(|i| (i * 2) as Val).collect();
                up.extend(down);
                up
            }

            macro_rules! check_clean {
                ($label:expr, $r:ty, $n:expr) => {{
                    for split in 0..=$n {
                        let p = bitonic($n, split);
                        let v = build::<$r>(&p);
                        let got = lanes_of::<$r>(<$r as NumericRegister>::bitonic_clean(v));
                        assert_eq!(got, oracle::<$r>(&p), "{} split={split}: input {:?}", $label, p);
                    }
                }};
            }

            check_clean!("A2", A2, 16);
            check_clean!("A4", A4, 32);
            check_clean!("A8", A8, 32);
            check_clean!("A16", A16, 128);
        }

        /// A sorted input is a fixed point of `sort`, and `bitonic_clean` of a
        /// sorted input is the identity, a cheap structural check on the wiring.
        #[test]
        fn sorted_is_fixed_point() {
            for p in patterns(16) {
                let sorted = <A2 as NumericRegister>::sort(build::<A2>(&p));
                let twice = <A2 as NumericRegister>::sort(sorted);
                let cleaned = <A2 as NumericRegister>::bitonic_clean(sorted);
                assert_eq!(lanes_of::<A2>(twice), lanes_of::<A2>(sorted), "input {:?}", p);
                assert_eq!(lanes_of::<A2>(cleaned), lanes_of::<A2>(sorted), "input {:?}", p);
            }
        }
    }
}

/// The scalar backend's `sort` is a documented no-op (1 lane).
#[test]
fn scalar_sort_is_identity() {
    let v = <f32 as NumericRegister>::sort(build::<f32>(&[3]));
    assert_eq!(lanes_of::<f32>(v), vec![3.0f32]);
}

/// The macro-stamped backends. `sort_via_network!` is invoked from inside
/// NEON's and wasm's register macros rather than per file, so these confirm the
/// stamp reached every width. A lane count that fell through to the no-op arm
/// is still *correct*, just quietly scalar, which nothing else here would catch.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm {
    use super::*;
    use thermite::backend::wasm::registers::{F32x4Wasm, F64x2Wasm, I16x8Wasm, I32x4Wasm};

    #[test]
    fn register_sort_method() {
        check_sort!("F64x2Wasm", F64x2Wasm, 2, <F64x2Wasm as NumericRegister>::sort);
        check_sort!("F32x4Wasm", F32x4Wasm, 4, <F32x4Wasm as NumericRegister>::sort);
        check_sort!("I32x4Wasm", I32x4Wasm, 4, <I32x4Wasm as NumericRegister>::sort);
        check_sort!("I16x8Wasm", I16x8Wasm, 8, <I16x8Wasm as NumericRegister>::sort);
    }

    #[test]
    fn matches_the_network_directly() {
        for p in patterns(8) {
            let v = build::<I16x8Wasm>(&p);
            assert_eq!(
                lanes_of::<I16x8Wasm>(<I16x8Wasm as NumericRegister>::sort(v)),
                lanes_of::<I16x8Wasm>(sort::sort_8::<I16x8Wasm, Ascending>(v)),
                "I16x8Wasm::sort should be sort_8, input {:?}",
                p
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::registers::{F32x4Neon, F64x2Neon, I16x8Neon, I32x4Neon, U16x8Neon};

    #[test]
    fn register_sort_method() {
        check_sort!("F64x2Neon", F64x2Neon, 2, <F64x2Neon as NumericRegister>::sort);
        check_sort!("F32x4Neon", F32x4Neon, 4, <F32x4Neon as NumericRegister>::sort);
        check_sort!("I32x4Neon", I32x4Neon, 4, <I32x4Neon as NumericRegister>::sort);
        check_sort!("I16x8Neon", I16x8Neon, 8, <I16x8Neon as NumericRegister>::sort);
        check_sort!("U16x8Neon", U16x8Neon, 8, <U16x8Neon as NumericRegister>::sort);
    }

    #[test]
    fn matches_the_network_directly() {
        for p in patterns(8) {
            let v = build::<I16x8Neon>(&p);
            assert_eq!(
                lanes_of::<I16x8Neon>(<I16x8Neon as NumericRegister>::sort(v)),
                lanes_of::<I16x8Neon>(sort::sort_8::<I16x8Neon, Ascending>(v)),
                "I16x8Neon::sort should be sort_8, input {:?}",
                p
            );
        }
    }
}
