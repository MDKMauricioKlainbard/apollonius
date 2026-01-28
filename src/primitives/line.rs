use crate::{
    EuclideanVector, FloatSign, Hypersphere, IntersectionResult, Point, SpatialRelation, Vector,
    VectorMetricSquared, classify_to_zero,
};
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

    pub fn intersect_sphere(&self, sphere: &Hypersphere<T, N>) -> IntersectionResult<T, N> {
        let pc = self.closest_point(&sphere.center());
        let dist_sq = (pc - sphere.center()).magnitude_squared();
        let r_sq = sphere.radius() * sphere.radius();
        let diff = r_sq - dist_sq;

        match classify_to_zero(diff, None) {
            FloatSign::Negative => IntersectionResult::None,
            FloatSign::Zero => IntersectionResult::Tangent(pc),
            FloatSign::Positive => {
                let h = diff.sqrt();
                IntersectionResult::Secant(pc - self.direction * h, pc + self.direction * h)
            }
        }
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

    #[test]
    fn test_line_sphere_no_intersection() {
        let line = Line::new(Point::new([0.0, 11.0]), Vector::new([1.0, 0.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);

        match line.intersect_sphere(&sphere) {
            IntersectionResult::None => {}
            _ => panic!("Should not intersect: line is outside the radius"),
        }
    }

    #[test]
    fn test_line_sphere_tangent() {
        // Line passes exactly at y = 10, touching the sphere at (0, 10)
        let line = Line::new(Point::new([-5.0, 10.0]), Vector::new([1.0, 0.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);

        if let IntersectionResult::Tangent(p) = line.intersect_sphere(&sphere) {
            assert_relative_eq!(p.coords[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords[1], 10.0, epsilon = 1e-6);
        } else {
            panic!("Expected tangent intersection at (0, 10)");
        }
    }

    #[test]
    fn test_line_sphere_secant_centered() {
        // Line passes through the center along X axis
        let line = Line::new(Point::new([-20.0, 0.0]), Vector::new([1.0, 0.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);

        if let IntersectionResult::Secant(p1, p2) = line.intersect_sphere(&sphere) {
            // We expect (-10, 0) and (10, 0)
            let (min_x, max_x) = if p1.coords[0] < p2.coords[0] {
                (p1.coords[0], p2.coords[0])
            } else {
                (p2.coords[0], p1.coords[0])
            };
            assert_relative_eq!(min_x, -10.0, epsilon = 1e-6);
            assert_relative_eq!(max_x, 10.0, epsilon = 1e-6);
        } else {
            panic!("Expected two intersection points at -10 and 10");
        }
    }

    #[test]
    fn test_line_sphere_secant_off_center() {
        // Vertical line at x = 3, sphere radius 5 at origin
        // Intersection should be at y = sqrt(5^2 - 3^2) = 4
        let line = Line::new(Point::new([3.0, 0.0]), Vector::new([0.0, 1.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);

        if let IntersectionResult::Secant(p1, p2) = line.intersect_sphere(&sphere) {
            let (min_y, max_y) = if p1.coords[1] < p2.coords[1] {
                (p1.coords[1], p2.coords[1])
            } else {
                (p2.coords[1], p1.coords[1])
            };
            assert_relative_eq!(min_y, -4.0, epsilon = 1e-6);
            assert_relative_eq!(max_y, 4.0, epsilon = 1e-6);
        } else {
            panic!("Expected secant points at y = -4 and y = 4");
        }
    }
}
