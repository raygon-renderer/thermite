//! `load_deinterleaved` / `store_interleaved` for `Compensated` vectors.
//!
//! A `Compensated<E>` element is `#[repr(C)]` over two floats (value, error), so
//! an array-of-structures of `M` compensated streams is really `2 * M`
//! interleaved float streams, and the vector impl de-interleaves it by handing
//! that flattened view to the inner vector. The contract is the same as for a
//! plain vector, just with a composite element:
//!
//!   load_deinterleaved:  out[j].extract(lane) == ptr[lane * M + j]
//!   store_interleaved:   ptr[lane * M + j] == values[j].extract(lane)
//!
//! Value and error carry distinct tags, because a de-interleave that swapped a
//! pair's two halves (or crossed two streams) would still produce
//! plausible-looking floats. Only exact routing checks catch that.

use thermite::prelude::*;
use thermite_compensated::Compensated;

/// Distinct, exactly-representable value for (element index, component index).
fn tag(elem: usize, comp: usize) -> f32 {
    (elem * 100 + comp) as f32
}

macro_rules! check {
    ($label:expr, $v:ty, $m:literal) => {{
        type C = Compensated<$v>;

        let lanes = <C as GenericVector>::LANES;
        let total = $m * lanes;

        let src: Vec<Compensated<f32>> = (0..total)
            .map(|e| Compensated {
                value: tag(e, 0),
                error: tag(e, 1),
            })
            .collect();

        // --- load_deinterleaved against the reference ---
        let out: [C; $m] = unsafe { C::load_deinterleaved::<$m>(src.as_ptr()) };

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
        let mut dst = vec![Compensated::<f32>::default(); total];
        unsafe { C::store_interleaved::<$m>(dst.as_mut_ptr(), out) };

        assert_eq!(dst, src, "{}: store_interleaved<{}> round-trip", $label, $m);
    }};
}

/// The grouped register engine covers every `M` directly, with no dispatch
/// ladder and no scalar fallback, so this just spans a range of stream counts,
/// including `M = 9` (18 element streams) well past a ladder's reach.
macro_rules! check_all_m {
    ($label:expr, $v:ty) => {{
        check!($label, $v, 1);
        check!($label, $v, 2);
        check!($label, $v, 3);
        check!($label, $v, 4);
        check!($label, $v, 6);
        check!($label, $v, 9);
    }};
}

/// The 1-lane scalar backend: the oracle, and the only one that needs no
/// target features.
#[test]
fn scalar_streams() {
    check_all_m!("scalar Compensated<f32>", Vector<f32>);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3_streams() {
        use thermite::backend::x86_v3::{f32x4, f32x8, f32x16};

        check_all_m!("x86_v3 Compensated<f32x4>", f32x4);
        check_all_m!("x86_v3 Compensated<f32x8>", f32x8);
        // f32x16 is an ArrayRegister on AVX2 - covers its per-chunk grouped override.
        check_all_m!("x86_v3 Compensated<f32x16>", f32x16);
    }

    #[test]
    fn v2_streams() {
        use thermite::backend::x86_v2::f32x4;

        check_all_m!("x86_v2 Compensated<f32x4>", f32x4);
    }
}
