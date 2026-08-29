//! SoA (Structure of Arrays) geometry: one geometric object *per lane*.
//!
//! Every component of a type is a separate SIMD vector, so a
//! [`Vector3<f32x8>`](prim::Vector3) holds the x/y/z of **eight** vectors at
//! once and a single call operates on all eight. Scalars come back as the
//! vector type `V` (one value per object) and predicates as `V::Mask`, so
//! there is no branching on a per-object result, so use
//! [`select`](thermite::mask::GenericMask::select) instead.
//!
//! This is the throughput layout: batches of rays, particles, or sample points,
//! and the layout [`thermite-sdf`](https://docs.rs/thermite-sdf) is built on.
//! Types are generic over the vector type `V: FloatVector` and the dimension
//! `N`, so the same code covers 2D, 3D, and beyond on any backend and lane
//! width.
//!
//! Data on disk is almost never SoA, so the primitives that correspond to a
//! record format ([`Vector`](prim::Vector), [`Point`](prim::Point),
//! [`Ray`](prim::Ray)) provide `load_interleaved`/`store_interleaved` to
//! transpose AoS <-> SoA on the way in and out.

pub mod algo;
pub mod prim;
