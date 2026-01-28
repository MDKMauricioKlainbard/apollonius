use crate::{EuclideanVector, Point, SpatialRelation, Vector, VectorMetricSquared};
use num_traits::Float;

/// Represents an infinite line in N-dimensional space.
///
/// A line is defined by an origin point and a direction vector.
/// It extends infinitely in both directions.
pub struct Line<T, const N: usize> {
    origin: Point<T, N>,
    direction: Vector<T, N>,
}

impl<T, const N: usize> Line<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new Line from an origin and a direction.
    ///
    /// The direction vector is automatically normalized to ensure
    /// that parametric evaluations and projections are consistent.
    ///
    /// # Example
    /// ```
    /// use apollonius::{Point, Vector};
    /// use apollonius::primitives::line::Line;
    ///
    /// let origin = Point::new([0.0, 0.0]);
    /// let direction = Vector::new([10.0, 0.0]);
    /// let line = Line::new(origin, direction);
    /// ```
    pub fn new(origin: Point<T, N>, direction: Vector<T, N>) -> Self {
        let direction = direction.normalize().unwrap_or(direction);
        Self { origin, direction }
    }

    /// Evaluates the line at a given parameter t.
    ///
    /// Returns the point calculated as the origin plus the direction vector scaled by t.
    ///
    /// # Example
    /// ```
    /// # use apollonius::{Point, Vector};
    /// # use apollonius::primitives::line::Line;
    /// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
    /// let p = line.at(5.0);
    /// assert_eq!(p.coords, [5.0, 0.0]);
    /// ```
    pub fn at(&self, t: T) -> Point<T, N> {
        self.origin + self.direction * t
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Line<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point onto the line to find the nearest location.
    ///
    /// Since the direction vector is normalized, the projection is calculated
    /// using the dot product between the direction and the displacement vector
    /// from the origin to the target point.
    ///
    /// # Example
    /// ```
    /// use apollonius::{Point, Vector, Line, SpatialRelation};
    ///
    /// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
    /// let p = Point::new([5.0, 10.0]);
    /// let closest = line.closest_point(&p);
    ///
    /// assert_eq!(closest.coords, [5.0, 0.0]);
    /// ```
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let t = (*p - self.origin).dot(&self.direction);
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_at_parameter() {
        let origin = Point::new([0.0, 0.0]);
        let direction = Vector::new([1.0, 0.0]);
        let line = Line::new(origin, direction);

        // L(t) = origin + t * direction
        assert_relative_eq!(line.at(0.0).coords[0], 0.0);
        assert_relative_eq!(line.at(1.0).coords[0], 1.0);
        assert_relative_eq!(line.at(-1.0).coords[0], -1.0);
    }

    #[test]
    fn test_line_closest_point() {
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));

        let p = Point::new([5.0, 10.0]);

        let closest = line.closest_point(&p);
        assert_relative_eq!(closest.coords[0], 5.0);
        assert_relative_eq!(closest.coords[1], 0.0);
    }

    #[test]
    fn test_distance_to_point() {
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let p = Point::new([0.0, 10.0]);

        assert_relative_eq!(line.distance_to_point(&p), 10.0);
    }

    #[test]
    fn test_line_contains_points_between_references() {
        let p1 = Point::new([0.0, 0.0]);
        let p2 = Point::new([10.0, 0.0]);
        let line = Line::new(p1, p2 - p1); // The line passes through these two points

        let mid = Point::new([5.0, 0.0]);
        assert!(
            line.contains(&mid),
            "Points between the reference points should be contained"
        );
    }

    #[test]
    fn test_line_contains_points_to_infinity() {
        let p1 = Point::new([0.0, 0.0]);
        let p2 = Point::new([1.0, 1.0]); // 45-degree diagonal
        let line = Line::new(p1, p2 - p1);

        // Very distant points in both directions
        let far_positive = Point::new([1000.0, 1000.0]);
        let far_negative = Point::new([-500.0, -500.0]);

        assert!(
            line.contains(&far_positive),
            "The line extends infinitely forward"
        );
        assert!(
            line.contains(&far_negative),
            "The line extends infinitely backward"
        );
    }

    #[test]
    fn test_line_excludes_non_collinear() {
        let p1 = Point::new([0.0, 0.0]);
        let p2 = Point::new([10.0, 0.0]); // X-axis
        let line = Line::new(p1, p2 - p1);

        let off = Point::new([5.0, 0.1]); // Slightly offset
        assert!(
            !line.contains(&off),
            "A point not collinear with the line should be excluded"
        );
    }
}
