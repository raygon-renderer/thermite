//! `load_deinterleaved` / `store_interleaved` for `Dual` vectors.
//!
//! A `Dual<E, N>` element is `#[repr(C)]` over `N + 1` floats (primal, then the
//! derivative parts), so an array-of-structures of `M` dual streams is really
//! `M * (N + 1)` interleaved float streams, and the vector impl de-interleaves
//! it by handing that flattened view to the inner vector. The contract is the
//! same as for a plain vector, just with a composite element:
//!
//!   load_deinterleaved:  out[j].extract(lane) == ptr[lane * M + j]
//!   store_interleaved:   ptr[lane * M + j] == values[j].extract(lane)
//!
//! The grouped register engine covers every `M` and `N` directly, with no dispatch
//! ladder and no scalar fallback, so `M` is swept generously here just to
//! exercise a range of stream counts, and every component of every element
//! gets a distinct value: a de-interleave that crossed two components or two
//! streams would still produce plausible-looking floats, so only exact
//! routing checks catch it.

use thermite::prelude::*;
use thermite_dual::Dual;

/// Distinct, exactly-representable value for (element index, component index).
fn tag(elem: usize, comp: usize) -> f32 {
    (elem * 100 + comp) as f32
}

/// Round-trip + reference check for one dual vector type at one stream count.
macro_rules! check {
    ($label:expr, $v:ty, $n:literal, $m:literal) => {{
        type D = Dual<$v, $n>;

        let lanes = <D as GenericVector>::LANES;
        let total = $m * lanes;

        // AoS source: element `e` carries `tag(e, 0)` in its primal and
        // `tag(e, 1 + k)` in derivative `k`.
        let src: Vec<Dual<f32, $n>> = (0..total)
            .map(|e| Dual {
                re: tag(e, 0),
                dual: core::array::from_fn(|k| tag(e, 1 + k)),
            })
            .collect();

        // --- load_deinterleaved against the reference ---
        let out: [D; $m] = unsafe { D::load_deinterleaved::<$m>(src.as_ptr()) };

        for j in 0..$m {
            for lane in 0..lanes {
                let got = out[j].extractv(lane);
                let want = src[lane * $m + j];

                assert_eq!(
                    got, want,
                    "{}: load_deinterleaved<{}> stream {j} lane {lane}",
                    $label, $m
                );
            }
        }

        // --- store_interleaved must invert it exactly ---
        let mut dst = vec![Dual::<f32, $n>::default(); total];
        unsafe { D::store_interleaved::<$m>(dst.as_mut_ptr(), out) };

        assert_eq!(dst, src, "{}: store_interleaved<{}> round-trip", $label, $m);
    }};
}

/// Sweep the stream count for one `(vector, N)` pair. The grouped engine
/// handles every `M * (N + 1)` product directly, so this just spans a range
/// of realistic stream counts.
macro_rules! check_all_m {
    ($label:expr, $v:ty, $n:literal) => {{
        check!($label, $v, $n, 1);
        check!($label, $v, $n, 2);
        check!($label, $v, $n, 3);
        check!($label, $v, $n, 4);
        check!($label, $v, $n, 6);
    }};
}

/// The 1-lane scalar backend: the oracle, and the only one that needs no
/// target features.
#[test]
fn scalar_streams() {
    check_all_m!("scalar Dual<f32, 0>", Vector<f32>, 0);
    check_all_m!("scalar Dual<f32, 1>", Vector<f32>, 1);
    check_all_m!("scalar Dual<f32, 2>", Vector<f32>, 2);
    check_all_m!("scalar Dual<f32, 3>", Vector<f32>, 3);
    check_all_m!("scalar Dual<f32, 4>", Vector<f32>, 4);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3_streams() {
        use thermite::backend::x86_v3::{f32x4, f32x8, f32x16};

        check_all_m!("x86_v3 Dual<f32x4, 1>", f32x4, 1);
        check_all_m!("x86_v3 Dual<f32x8, 1>", f32x8, 1);
        check_all_m!("x86_v3 Dual<f32x8, 2>", f32x8, 2);
        check_all_m!("x86_v3 Dual<f32x8, 3>", f32x8, 3);
        check_all_m!("x86_v3 Dual<f32x8, 4>", f32x8, 4);
        // f32x16 is an ArrayRegister on AVX2 - covers its per-chunk grouped override.
        check_all_m!("x86_v3 Dual<f32x16, 2>", f32x16, 2);
    }

    #[test]
    fn v2_streams() {
        use thermite::backend::x86_v2::f32x4;

        check_all_m!("x86_v2 Dual<f32x4, 2>", f32x4, 2);
    }
}
