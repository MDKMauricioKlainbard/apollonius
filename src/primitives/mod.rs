pub mod aabb;
pub mod hyperplane;
pub mod hypersphere;
pub mod line;
pub mod segment;

use crate::{AABB, EuclideanVector, FloatSign, Point, VectorMetricSquared, classify_to_zero};
use num_traits::Float;

/// Defines spatial queries for geometric entities in N-dimensional space.
///
/// This trait provides a unified interface to calculate projections, distances,
/// and containment checks between points and various geometric primitives.
pub trait SpatialRelation<T, const N: usize> {
    /// Projects a point onto the nearest location on the geometric entity.
    ///
    /// For infinite entities like lines, this is the orthogonal projection.
    /// For finite shapes like segments or spheres, the result is clamped
    /// to the boundaries or surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Circle, SpatialRelation};
    ///
    /// let circle = Circle::new(Point::new([0.0, 0.0]), 1.0);
    /// let p = Point::new([2.0, 0.0]);
    ///
    /// // Projects the point onto the circle's boundary
    /// let closest = circle.closest_point(&p);
    /// assert_eq!(closest.coords[0], 1.0);
    /// ```
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N>;

    /// Calculates the minimum Euclidean distance between the entity and a point.
    ///
    /// This method is provided by default and relies on [`Self::closest_point`].
    ///
    /// # Constraints
    /// * `T` must implement [`Float`] and [`std::iter::Sum`].
    fn distance_to_point(&self, p: &Point<T, N>) -> T
    where
        T: Float + std::iter::Sum,
    {
        (self.closest_point(p) - *p).magnitude()
    }

    /// Checks if a point lies on the boundary or structure of the entity.
    ///
    /// This uses the engine's internal epsilon tolerance to account for
    /// floating-point inaccuracies.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Circle, SpatialRelation};
    ///
    /// let circle = Circle::new(Point::new([0.0, 0.0]), 1.0);
    /// assert!(circle.contains(&Point::new([1.0, 0.0])));
    /// ```
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

    /// Checks if a point is within the volume or area defined by the entity.
    ///
    /// For 1D entities (Line, Segment) or boundaries (Hyperplane), this
    /// usually defaults to [`Self::contains`]. For volumes (Hypersphere),
    /// it includes the interior.
    #[inline]
    fn is_inside(&self, point: &Point<T, N>) -> bool
    where
        T: Float + std::iter::Sum,
    {
        self.contains(point)
    }
}

/// Represents an entity that can be enclosed within an Axis-Aligned Bounding Box.
pub trait Bounded<T, const N: usize> {
    /// Returns the minimum Axis-Aligned Bounding Box that encloses the entity.
    ///
    /// Used for broad-phase collision detection to quickly prune non-colliding objects.
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
    /// In physics simulations, this typically represents a grazing contact.
    Tangent(Point<T, N>),

    /// The entities intersect at two distinct points.
    ///
    /// Typical of a line or segment that enters and then exits a volume
    /// (e.g., a secant line through a hypersphere).
    Secant(Point<T, N>, Point<T, N>),

    /// The entities are coincident or overlap over a continuous range.
    ///
    /// This occurs when two lines are identical or a segment lies
    /// entirely within a hyperplane.
    Collinear,

    /// Exactly one point of intersection that is not a tangency.
    ///
    /// Usually occurs when a finite primitive (like a segment) starts
    /// outside and ends inside another entity's volume.
    Single(Point<T, N>),
}
