pub mod points;
pub mod primitives;
pub mod vectors;
mod utils;

pub use crate::points::{EuclideanMetric, MetricSquared, Point, Point2D, Point3D};
pub use crate::primitives::{
    SpatialRelation, hypersphere::Circle, hypersphere::Hypersphere, hypersphere::Sphere,
    line::Line, segment::Segment, IntersectionResult
};
pub use crate::vectors::{EuclideanVector, Vector, Vector2D, Vector3D, VectorMetricSquared};
pub use crate::utils::{classify_to_zero, FloatSign};
