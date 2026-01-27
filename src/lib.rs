pub mod points;
pub mod primitives;
pub mod vectors;

pub use crate::points::{EuclideanMetric, MetricSquared, Point, Point2D, Point3D};
pub use crate::primitives::{SpatialRelation, line::Line, segment::Segment};
pub use crate::vectors::{EuclideanVector, Vector, Vector2D, Vector3D, VectorMetricSquared};
