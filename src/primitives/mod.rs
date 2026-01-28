pub mod aabb;
pub mod hypersphere;
pub mod line;
pub mod segment;
pub mod hyperplane;

use crate::{AABB, EuclideanVector, FloatSign, Point, VectorMetricSquared, classify_to_zero};
use num_traits::Float;

/// Defines spatial queries for geometric entities in N-dimensional space.
///
/// This trait provides a unified interface to calculate projections and
/// distances between points and various geometric primitives like lines,
/// segments, and circles.
pub trait SpatialRelation<T, const N: usize> {
    /// Projects a point onto the nearest location on the geometric entity.
    ///
    /// For a line, this is the orthogonal projection. For finite shapes
    /// like segments, the result is clamped to the boundaries.
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N>;

    /// Calculates the minimum Euclidean distance between the entity and a point.
    ///
    /// This method is provided by default and relies on [`Self::closest_point`].
    /// It is defined as the magnitude of the displacement vector between
    /// the point and its projection on the entity.
    ///
    /// # Constraints
    /// * `T` must implement [`Float`] as distances often involve non-integer results.
    fn distance_to_point(&self, p: &Point<T, N>) -> T
    where
        T: Float + std::iter::Sum,
    {
        (self.closest_point(p) - *p).magnitude()
    }

    /// Check if a point lies on the boundary/structure of the entity.
    #[inline]
    fn contains(&self, point: &Point<T, N>) -> bool
    where
        T: Float + std::iter::Sum,
    {
        match classify_to_zero(
            (self.closest_point(point) - *point).magnitude_squared(),
            None,
        ) {
            FloatSign::Zero => true,
            _ => false,
        }
    }

    /// Check if a point is within the volume or area defined by the entity.
    /// For 1D entities (Line, Segment), this defaults to `contains`.
    #[inline]
    fn is_inside(&self, point: &Point<T, N>) -> bool
    where
        T: Float + std::iter::Sum,
    {
        self.contains(point)
    }
}

pub trait Bounded<T, const N: usize> {
    /// Returns the minimum Axis-Aligned Bounding Box that encloses the entity.
    fn aabb(&self) -> AABB<T, N>;
}

/// Represents the outcome of an intersection query between geometric primitives.
///
/// This enum accounts for the different ways entities can interact in N-dimensional space,
/// distinguishing between points of contact, boundary crossings, and overlapping structures.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum IntersectionResult<T, const N: usize> {
    /// No intersection occurs between the entities.
    None,

    /// The entities touch at exactly one point without crossing boundaries.
    ///
    /// In physics, this often represents a grazing contact or a perfect bounce.
    Tangent(Point<T, N>),

    /// The entities intersect at two distinct points.
    ///
    /// Typical of a line or segment that enters and then exits a volume (like a hypersphere).
    Secant(Point<T, N>, Point<T, N>),

    /// The entities are coincident or overlap over an infinite or continuous range.
    ///
    /// *Note: Reserved for future implementation of collinear lines and overlapping segments.*
    Collinear,

    /// Exactly one point of intersection that is not a tangency.
    ///
    /// This occurs when a finite primitive (like a segment) crosses a boundary exactly once,
    /// usually because one of its endpoints is contained within the other entity's volume.
    Single(Point<T, N>),
}
