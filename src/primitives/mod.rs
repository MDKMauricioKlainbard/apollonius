pub mod line;
pub mod segment;

use crate::{EuclideanVector, Point};
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
}
