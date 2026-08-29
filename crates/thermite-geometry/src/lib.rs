#![cfg_attr(docsrs, feature(doc_cfg))]

//! SIMD geometry primitives built on Thermite.
//!
//! Geometry can be laid out two ways, and which one is right depends on whether
//! you are processing one object or many:
//!
//! - [`soa`] is **structure of arrays**, one object per SIMD lane. Each component
//!   lives in its own vector, so `Vector3<f32x8>` is eight 3D vectors. This is
//!   the throughput layout: ray batches, particles, sample grids.
//! - [`aos`] is **array of structures**, one object per SIMD register. The lanes
//!   *are* the components, so a 3-lane register is one 3D vector and `dot3` is a
//!   horizontal reduction returning a scalar. This is the latency layout for
//!   single-object work (camera transforms, scene-graph nodes, one ray walked
//!   against many boxes), and it rides the hardware `dot3`/`cross3`/`mat4_*`
//!   primitives the SoA layout cannot reach.
//!
//! Neither layout is the fast one in general: pick SoA when you have a batch and
//! want throughput, AoS when you have one object and want it now.

#![no_std]

pub mod aos;
pub mod soa;

pub mod error_bounds;

pub use error_bounds::{gamma, gamma_scalar};
