//! SoA (Structure of Arrays) geometric primitives and types.
//!
//! Each dimension of a geometric type is stored in a separate vector,
//! allowing for efficient SIMD operations on multiple instances of the type,
//! and potentially batch processing of geometric computations.
//!
//! Nothing is guaranteed to work when any dimension is zero, and you may
//! even get compilation errors in that case.

pub mod bounds;
pub mod matrix;
pub mod point;
pub mod ray;
pub mod vector;

pub use bounds::Bounds;
pub use matrix::Matrix;
pub use point::Point;
pub use ray::Ray;
pub use vector::Vector;

pub type Point2<V> = Point<V, 2>;
pub type Point3<V> = Point<V, 3>;
pub type Point4<V> = Point<V, 4>;

pub type Vector2<V> = Vector<V, 2>;
pub type Vector3<V> = Vector<V, 3>;
pub type Vector4<V> = Vector<V, 4>;

pub type Matrix2x2<V> = Matrix<V, 2, 2>;
pub type Matrix3x3<V> = Matrix<V, 3, 3>;
pub type Matrix4x4<V> = Matrix<V, 4, 4>;

pub type Bounds2<V> = Bounds<V, 2>;
pub type Bounds3<V> = Bounds<V, 3>;
pub type Bounds4<V> = Bounds<V, 4>;

pub type Ray2<V> = Ray<V, 2>;
pub type Ray3<V> = Ray<V, 3>;
pub type Ray4<V> = Ray<V, 4>;
