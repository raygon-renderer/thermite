//! AoS (Array of Structures) geometric primitives: **one object per SIMD
//! register**.
//!
//! The counterpart to [`soa`](crate::soa). There, a `Vector3<f32x8>` is eight
//! separate 3D vectors, one per lane, and every predicate answers per lane. Here
//! a single 3-lane register *is* one 3D vector: the lanes are the components,
//! `dot3` is a horizontal reduction, and every predicate answers with a plain
//! `bool` about the one object.
//!
//! Use this layout when you have **one** object at a time and the work is
//! inherently serial: a camera transform, a scene-graph node, a single ray being
//! traversed against many boxes, an editor gizmo. Use [`soa`](crate::soa) when
//! you have a batch and want throughput.
//!
//! ```text
//!   soa::Vector3<f32x8>          aos 3-lane register
//!   x: [x0 x1 x2 x3 ...]          [x y z _]
//!   y: [y0 y1 y2 y3 ...]          \___________/
//!   z: [z0 z1 z2 z3 ...)           one vector
//!   \_______________/
//!    8 vectors at once
//! ```
//!
//! # Why the lanes-are-components layout is not just "the slow one"
//!
//! It rides hardware that the SoA layout cannot reach. `dot3`, `cross3`,
//! `mat4_point3_product`, `quat4_product` and friends are single primitives in
//! [`LinAlg3Vector`](thermite::vector::LinAlg3Vector) /
//! [`LinAlg4Vector`](thermite::vector::LinAlg4Vector), lowered per backend to
//! shuffle networks (or to genuine `dot`/`vec3` instructions on GPU targets).
//! Transforming one point by an affine matrix here is three FMAs, because
//! [`mat4_point3_product`](thermite::vector::LinAlg4Vector::mat4_point3_product)
//! folds the `w = 1` translation column into the FMA chain and never computes
//! `w` at all.
//!
//! # Types
//!
//! Everything is generic over a backend `S: `[`Simd3`](thermite::simd::Simd3)
//! (via its vector mirror [`Simd3Vectors`]) and over the float element, so the
//! same code serves `f32` and `f64`. The 3-lane registers are the **true**
//! 3-lane family (`f32x3`, not the alpha-padded `f32x3A`), so a backend with
//! native 3-component vectors (SPIR-V's `vec3`) uses them directly rather than
//! carrying a wasted fourth lane.
//!
//! Vectors and points are both bare 3-lane registers rather than newtypes: the
//! whole point of this layout is that the register *is* the object, and wrapping
//! it would only obstruct the arithmetic operators the register already has. The
//! distinction between a direction and a position lives in which method you call
//! ([`transform_vector`](matrix::Matrix4::transform_vector) versus
//! [`transform_point`](matrix::Matrix4::transform_point)), exactly as it does in
//! shader code.

pub mod bounds;
pub mod matrix;
pub mod quaternion;
pub mod ray;
pub mod tangent;
pub mod transform;
pub mod vector;

pub use bounds::Bounds3;
pub use matrix::Matrix4;
pub use quaternion::Quaternion;
pub use ray::{Ray3, RayError3};
pub use tangent::TangentFrame;
pub use transform::Transform;
pub use vector::{Point3Ext, Vector3Ext};

use thermite::{
    element::FloatElement,
    generic_array::typenum::{U3, U4},
    math::SpatialMath,
    simd::Simd3Vectors,
    vector::{ExtendVector, LinAlg3Vector, LinAlg4Vector, SwizzleVector},
};

/// Selects one float family out of a backend, naming its 3-lane and 4-lane
/// registers together.
///
/// This is what keeps the AoS types generic over the backend **and** the element
/// with a single extra parameter: `Matrix4<S, f32>` and `Matrix4<S, f64>` are
/// both spelled the same way, and `E::V3`/`E::V4` resolve to `S::f32x3`/
/// `S::f32x4` or the `f64` pair.
///
/// The `V4: ExtendVector<V3>` bound lives here rather than on each method that
/// widens a point into a matrix column. Stated once, it never appears in a
/// `where` clause again.
pub trait AosFloat<S: Simd3Vectors>: FloatElement {
    /// The 3-lane register: a vector, point or normal.
    ///
    /// `SwizzleVector` is part of the bound because component shuffling is the
    /// basic move of this layout, since the lanes *are* the components, so a swizzle
    /// is how you reorder x/y/z without ever touching a scalar.
    type V3: LinAlg3Vector<Element = Self, Lanes = U3> + SpatialMath + SwizzleVector;

    /// The 4-lane register: a matrix column, a quaternion, or a homogeneous point.
    type V4: LinAlg4Vector<Element = Self, Lanes = U4>
        + ExtendVector<Self::V3, Element = Self>
        + SwizzleVector;
}

impl<S: Simd3Vectors<f32x3: SpatialMath>> AosFloat<S> for f32 {
    type V3 = S::f32x3;
    type V4 = S::f32x4;
}

impl<S: Simd3Vectors<f64x3: SpatialMath>> AosFloat<S> for f64 {
    type V3 = S::f64x3;
    type V4 = S::f64x4;
}

/// The 3-lane register for backend `S` and element `E`: a vector or a point.
pub type V3<S, E> = <E as AosFloat<S>>::V3;

/// The 4-lane register for backend `S` and element `E`: a matrix column or quaternion.
pub type V4<S, E> = <E as AosFloat<S>>::V4;

/// A backend usable by this module: true 3-lane registers whose float families
/// carry the spatial math (`sqrt`, `hypot`, reciprocals) the primitives need.
///
/// Every in-tree backend satisfies this. x86 v1/v2/v3, NEON, WASM and Scalar
/// all implement [`Simd3`](thermite::simd::Simd3).
pub trait AosSimd: Simd3Vectors<f32x3: SpatialMath, f64x3: SpatialMath> {}

impl<S> AosSimd for S where S: Simd3Vectors<f32x3: SpatialMath, f64x3: SpatialMath> {}

/// The working bound for a 3-lane geometric vector.
///
/// Nothing is added on top of [`LinAlg3Vector`](thermite::vector::LinAlg3Vector):
/// it already implies `FloatVector`, whose `Element` is a `FloatElement`, so the
/// scalars handed back by `dot3`/`norm`/`distance` come with `sqrt`, the float
/// constants and full arithmetic. The alias exists to name the concept once.
///
/// The few methods that need more than `FloatElement` offers (`atan2`, in
/// [`angle_between`](vector::Vector3Ext::angle_between)) carry that bound
/// themselves rather than imposing it on every user of the trait.
pub trait Vec3: thermite::vector::LinAlg3Vector {}

impl<V> Vec3 for V where V: thermite::vector::LinAlg3Vector {}

/// The working bound for a 4-lane register: matrix columns and quaternions.
pub trait Vec4: thermite::vector::LinAlg4Vector {}

impl<V> Vec4 for V where V: thermite::vector::LinAlg4Vector {}
