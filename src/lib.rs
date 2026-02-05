//! # Apollonius
//!
//! **Apollonius** is a lightweight, N-dimensional Euclidean geometry library for Rust. It provides
//! points, vectors, and geometric primitives (lines, segments, hyperspheres, hyperplanes, AABBs,
//! triangles) with a unified intersection API and spatial queries, all built with `const generics`
//! for type-safe 2D, 3D, or arbitrary dimension.
//!
//! ## Design goals
//!
//! * **Pure geometry** — No physics, no units; only shapes, distances, and intersections.
//! * **N-dimensional** — Same types and traits work in 2D, 3D, or N-D via `Point<T, N>` and
//!   `Vector<T, N>`.
//! * **Minimal dependencies** — Core logic depends only on `num-traits`; optional `serde` for
//!   serialization.
//!
//! ## Core types
//!
//! | Module      | Types / traits |
//! |-------------|----------------|
//! | **Points**  | [`Point`], [`Point2D`], [`Point3D`], [`MetricSquared`], [`EuclideanMetric`] |
//! | **Vectors** | [`Vector`], [`Vector2D`], [`Vector3D`], [`VectorMetricSquared`], [`EuclideanVector`] |
//! | **Primitives** | [`Line`], [`Segment`], [`Hypersphere`] (Circle, Sphere), [`Hyperplane`], [`AABB`], [`Triangle`] |
//! | **Traits**  | [`SpatialRelation`] (closest_point, distance_to_point, contains), [`Bounded`] (aabb) |
//! | **Results** | [`IntersectionResult`] (None, Tangent, Secant, Collinear, Single, HalfSpacePenetration) |
//! | **Utils**   | [`classify_to_zero`], [`FloatSign`] for robust float comparison |
//!
//! ## Quick example
//!
//! ```
//! use apollonius::{Point, Vector, Line, Hypersphere, IntersectionResult};
//!
//! let line = Line::new(Point::new([-5.0, 0.0]), Vector::new([1.0, 0.0]));
//! let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);
//!
//! match line.intersect_hypersphere(&sphere) {
//!     IntersectionResult::Secant(p1, p2) => { /* line pierces sphere at p1, p2 */ }
//!     IntersectionResult::Tangent(p) => { /* line touches sphere at p */ }
//!     _ => { /* no intersection */ }
//! }
//! ```
//!
//! ## Features
//!
//! * **`serde`** — Enables `Serialize` / `Deserialize` for points, vectors, and primitives.
//!   Use with `apollonius = { version = "...", features = ["serde"] }`.

pub mod algebra;
pub mod primitives;
pub mod utils;
pub mod space;

pub use crate::algebra::{
    points::{EuclideanMetric, MetricSquared, Point, Point2D, Point3D},
    vectors::{EuclideanVector, Vector, Vector2D, Vector3D, VectorMetricSquared},
    matrix::Matrix, angle::Angle
};
pub use crate::primitives::{
    Bounded, IntersectionResult, SpatialRelation, aabb::AABB, hyperplane::Hyperplane,
    hypersphere::Circle, hypersphere::Hypersphere, hypersphere::Sphere, line::Line,
    segment::Segment, triangle::Triangle,
};
pub use crate::utils::{FloatSign, classify_to_zero};
