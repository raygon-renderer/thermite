//! AoS <-> SoA loads for the geometry primitives.
//!
//! `Point`/`Vector` are `N` components; a `Ray` is two records of `N` (origin,
//! direction). The contract, over a span of `LANES` records:
//!
//!   load_interleaved:   self.0[c].extract(lane) == ptr[lane * N + c]
//!   store_interleaved:  the exact inverse
//!
//! Every component of every record carries a distinct tag, because a transpose
//! that crossed two components - or, for a ray, swapped origin and direction -
//! would still produce perfectly plausible geometry. Only exact routing checks
//! catch it.

use thermite::prelude::*;
use thermite_geometry::prim::{Point, Ray, RayRecord, Vector as GVector};

/// Distinct, exactly-representable value for (record index, component index).
fn tag(record: usize, comp: usize) -> f32 {
    (record * 100 + comp) as f32
}

macro_rules! check_vector {
    ($label:expr, $v:ty, $n:literal) => {{
        type V = $v;
        let lanes = <V as GenericVector>::LANES;

        // AoS source: `[[f32; N]]`, record `r` component `c` tagged.
        let src: Vec<[f32; $n]> = (0..lanes).map(|r| core::array::from_fn(|c| tag(r, c))).collect();

        let loaded = unsafe { GVector::<V, $n>::load_interleaved(src.as_ptr() as *const f32) };

        for c in 0..$n {
            for lane in 0..lanes {
                assert_eq!(
                    loaded.0[c].extractv(lane),
                    src[lane][c],
                    "{}: Vector load_interleaved component {c} lane {lane}",
                    $label
                );
            }
        }

        // Points share the implementation; check one to pin the contract.
        let p = unsafe { Point::<V, $n>::load_interleaved(src.as_ptr() as *const f32) };
        for c in 0..$n {
            for lane in 0..lanes {
                assert_eq!(p.0[c].extractv(lane), src[lane][c], "{}: Point lane {lane}", $label);
            }
        }

        // Round-trip.
        let mut dst = vec![[0.0f32; $n]; lanes];
        unsafe { loaded.store_interleaved(dst.as_mut_ptr() as *mut f32) };
        assert_eq!(dst, src, "{}: Vector store_interleaved round-trip", $label);
    }};
}

macro_rules! check_ray {
    ($label:expr, $v:ty, $n:literal) => {{
        type V = $v;
        let lanes = <V as GenericVector>::LANES;

        // Origin components tagged 0..N, direction components N..2N, so a swap of
        // the two records - or of any pair of components - is caught exactly.
        let src: Vec<RayRecord<f32, $n>> = (0..lanes)
            .map(|r| RayRecord {
                origin: core::array::from_fn(|c| tag(r, c)),
                direction: core::array::from_fn(|c| tag(r, $n + c)),
            })
            .collect();

        let ray = unsafe { Ray::<V, $n>::load_interleaved(src.as_ptr()) };

        for c in 0..$n {
            for lane in 0..lanes {
                assert_eq!(
                    ray.origin.0[c].extractv(lane),
                    src[lane].origin[c],
                    "{}: Ray origin component {c} lane {lane}",
                    $label
                );
                assert_eq!(
                    ray.direction.0[c].extractv(lane),
                    src[lane].direction[c],
                    "{}: Ray direction component {c} lane {lane}",
                    $label
                );
            }
        }

        let mut dst = vec![
            RayRecord::<f32, $n> {
                origin: [0.0; $n],
                direction: [0.0; $n]
            };
            lanes
        ];
        unsafe { ray.store_interleaved(dst.as_mut_ptr()) };
        assert_eq!(dst, src, "{}: Ray store_interleaved round-trip", $label);
    }};
}

/// The 1-lane scalar backend: the oracle, and the only one needing no target
/// features.
#[test]
fn scalar() {
    check_vector!("scalar 2D", Vector<f32>, 2);
    check_vector!("scalar 3D", Vector<f32>, 3);
    check_vector!("scalar 4D", Vector<f32>, 4);

    check_ray!("scalar ray 2D", Vector<f32>, 2);
    check_ray!("scalar ray 3D", Vector<f32>, 3);
    check_ray!("scalar ray 4D", Vector<f32>, 4);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3() {
        use thermite::backend::x86_v3::{f32x4, f32x8, f32x16};

        check_vector!("v3 f32x8 3D", f32x8, 3);
        check_vector!("v3 f32x4 3D", f32x4, 3);
        check_vector!("v3 f32x8 4D", f32x8, 4);
        // f32x16 is an ArrayRegister on AVX2: exercises its per-chunk override.
        check_vector!("v3 f32x16 3D", f32x16, 3);

        check_ray!("v3 f32x8 ray 3D", f32x8, 3);
        check_ray!("v3 f32x4 ray 3D", f32x4, 3);
        check_ray!("v3 f32x16 ray 3D", f32x16, 3);
        check_ray!("v3 f32x8 ray 4D", f32x8, 4);
    }

    #[test]
    fn v2() {
        use thermite::backend::x86_v2::f32x4;

        check_vector!("v2 f32x4 3D", f32x4, 3);
        check_ray!("v2 f32x4 ray 3D", f32x4, 3);
    }
}
