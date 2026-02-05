use crate::{
    EuclideanVector, FloatSign, Hyperplane, Hypersphere, IntersectionResult, Point, Segment,
    SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// Represents an infinite line in N-dimensional space.
///
/// A line is defined by an origin point and a direction vector.
/// It extends infinitely in both positive and negative directions.
///
/// # Examples
///
/// ```
/// use apollonius::{Point, Vector, Line};
///
/// let origin = Point::new([0.0, 0.0, 0.0]);
/// let direction = Vector::new([1.0, 0.0, 0.0]);
/// let line = Line::new(origin, direction);
///
/// assert_eq!(line.at(10.0).coords()[0], 10.0);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de>"
    ))
)]
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
    /// The direction vector is automatically normalized. If a null vector is provided,
    /// the line retains the original vector, which may lead to undefined behavior
    /// in projections.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line};
    ///
    /// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([5.0, 0.0]));
    /// // The direction is normalized to unit length
    /// assert_eq!(line.at(1.0).coords()[0], 1.0);
    /// ```
    pub fn new(origin: Point<T, N>, direction: Vector<T, N>) -> Self {
        let direction = direction.normalize().unwrap_or(direction);
        Self { origin, direction }
    }

    /// Evaluates the line at a given parameter t.
    ///
    /// Returns the point P = origin + direction * t.
    #[inline]
    pub fn at(&self, t: T) -> Point<T, N> {
        self.origin + self.direction * t
    }

    /// Returns a mutable reference to the origin point.
    #[inline]
    pub fn origin_mut(&mut self) -> &mut Point<T, N> {
        &mut self.origin
    }

    /// Returns a mutable reference to the direction vector.
    ///
    /// Note: mutating the direction does not re-normalize it.
    #[inline]
    pub fn direction_mut(&mut self) -> &mut Vector<T, N> {
        &mut self.direction
    }

    /// Computes the intersection(s) between this line and a hypersphere.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Single`, `Collinear`, or `HalfSpacePenetration`):
    /// - [`None`](crate::IntersectionResult::None): The line does not intersect the sphere.
    /// - [`Tangent(p)`](crate::IntersectionResult::Tangent): The line is tangent to the sphere at point `p`.
    /// - [`Secant(p1, p2)`](crate::IntersectionResult::Secant): The line enters and leaves the sphere at `p1` and `p2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, Hypersphere, IntersectionResult};
    ///
    /// let line = Line::new(Point::new([-5.0, 0.0]), Vector::new([1.0, 0.0]));
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);
    ///
    /// let result = line.intersect_hypersphere(&sphere);
    /// if let IntersectionResult::Secant(p1, p2) = result {
    ///     assert_eq!(p1.coords()[0], -2.0);
    ///     assert_eq!(p2.coords()[0], 2.0);
    /// }
    /// ```
    pub fn intersect_hypersphere(&self, sphere: &Hypersphere<T, N>) -> IntersectionResult<T, N> {
        let (center, radius) = (sphere.center(), sphere.radius());
        let pc = self.closest_point(&center);
        let dist_sq = (pc - center).magnitude_squared();
        let r_sq = radius * radius;
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

    /// Calculates the intersection between this line and a hyperplane in N-dimensional space.
    ///
    /// The intersection is determined by substituting the parametric equation of the line
    /// `L(t) = P + tV` into the implicit equation of the hyperplane `(X - P0) · n = 0`.
    ///
    /// # Returns
    /// - [`None`](crate::IntersectionResult::None): The line is parallel to the plane and not on it.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The line lies entirely on the hyperplane.
    /// - [`Single(p)`](crate::IntersectionResult::Single): The line pierces the plane at exactly one point `p`.
    ///
    /// # Mathematical Approach
    /// Solving for `t` gives:
    /// `t = ((P0 - P) · n) / (V · n)`
    /// where:
    /// - `P` is the line origin.
    /// - `V` is the line direction.
    /// - `P0` is the hyperplane origin.
    /// - `n` is the hyperplane normal.
    ///
    /// # Precision and Edge Cases
    /// - If the denominator `(V · n)` is zero, the line is parallel to the plane.
    /// - If the numerator `((P0 - P) · n)` is also zero, the line is contained within the plane (Collinear).
    /// - Epsilon-based checks are used via `classify_to_zero` to ensure numerical stability.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, Hyperplane, IntersectionResult};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    /// let line = Line::new(Point::new([0.0, 0.0, 10.0]), Vector::new([0.0, 0.0, -1.0]));
    ///
    /// if let IntersectionResult::Single(p) = line.intersect_hyperplane(&plane) {
    ///     assert_eq!(p, Point::new([0.0, 0.0, 0.0]));
    /// }
    /// ```
    pub fn intersect_hyperplane(&self, hyperplane: &Hyperplane<T, N>) -> IntersectionResult<T, N> {
        let plane_normal = hyperplane.normal();
        let line_dir = self.direction;

        // Denominator: V · n (relative direction/slope)
        let dot_dir_normal = line_dir.dot(&plane_normal);

        // Numerator: (P0 - P) · n (projected distance from line origin to plane)
        let origin_offset = hyperplane.origin() - self.origin;
        let dot_offset_normal = origin_offset.dot(&plane_normal);

        match classify_to_zero(dot_dir_normal, None) {
            // Case: Line direction is perpendicular to the plane normal (Parallel)
            FloatSign::Zero => {
                // If the origin offset is also perpendicular to the normal,
                // the line origin lies on the plane.
                if let FloatSign::Zero = classify_to_zero(dot_offset_normal, None) {
                    IntersectionResult::Collinear
                } else {
                    IntersectionResult::None
                }
            }
            // Case: Line intersects the hyperplane at exactly one point
            _ => {
                let t = dot_offset_normal / dot_dir_normal;
                IntersectionResult::Single(self.at(t))
            }
        }
    }

    /// Calculates the intersection between this line and another line in N-dimensional space.
    ///
    /// This method finds the points of closest approach between two lines by solving a 2x2
    /// system of equations derived from the condition that the vector connecting the
    /// points must be orthogonal to both line directions.
    ///
    /// # Returns
    /// - [`None`](crate::IntersectionResult::None): The lines are parallel but distinct, or skew (non-parallel and non-intersecting in 3D+).
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The lines coincide.
    /// - [`Single(p)`](crate::IntersectionResult::Single): The lines intersect at exactly one point `p`.
    ///
    /// # Mathematical Approach
    /// Let the lines be defined as `L1(t) = P1 + tV1` and `L2(s) = P2 + sV2`.
    /// The vector connecting any two points on the lines is `W(t, s) = (P1 - P2) + tV1 - sV2`.
    /// For the distance to be minimal (and zero if they intersect), `W` must be orthogonal
    /// to both `V1` and `V2`:
    /// 1. `W(t, s) · V1 = 0`
    /// 2. `W(t, s) · V2 = 0`
    ///
    /// This leads to the following 2x2 system for `t` and `s`:
    /// `(V1 · V1)t - (V2 · V1)s = -(P1 - P2) · V1`
    /// `(V1 · V2)t - (V2 · V2)s = -(P1 - P2) · V2`
    ///
    /// # Precision and Edge Cases
    /// - **Parallel Lines**: If the lines are parallel, the system's determinant is zero.
    ///   The method checks for collinearity by verifying if the origin of the other line
    ///   lies on this line using `closest_point`.
    /// - **Skew Lines**: In 3D or higher, lines may be non-parallel but never intersect.
    ///   The method calculates the closest points `P1(t)` and `P2(s)` and returns `None`
    ///   if the squared distance between them exceeds the epsilon threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, IntersectionResult};
    ///
    /// let line1 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
    /// let line2 = Line::new(Point::new([0.0, 1.0, 0.0]), Vector::new([0.0, -1.0, 0.0]));
    ///
    /// // Lines intersect at (0, 0, 0)
    /// if let IntersectionResult::Single(p) = line1.intersect_line(&line2) {
    ///     assert_eq!(p.coords()[0], 0.0);
    ///     assert_eq!(p.coords()[1], 0.0);
    /// }
    /// ```
    pub fn intersect_line(&self, other: &Line<T, N>) -> IntersectionResult<T, N> {
        let v_diff = self.origin - other.origin;

        // Coefficients for the 2x2 system
        let a = T::one(); // self.direction.magnitude_squared(), but direction is normalized
        let b = self.direction.dot(&other.direction);
        let c = T::one(); // other.direction.magnitude_squared(), but direction is normalized
        let e = other.direction.dot(&v_diff);
        let f = self.direction.dot(&v_diff);

        let det = b * b - a * c;

        match classify_to_zero(det, None) {
            // Case: Lines are parallel (direction vectors are linearly dependent)
            FloatSign::Zero => {
                // Check if the lines are collinear by finding the distance from
                // the other line's origin to this line.
                let closest = self.closest_point(&other.origin);
                let dist_sq = (closest - other.origin).magnitude_squared();

                match classify_to_zero(dist_sq, None) {
                    FloatSign::Zero => IntersectionResult::Collinear,
                    _ => IntersectionResult::None,
                }
            }
            // Case: Lines are not parallel
            _ => {
                // Solve the system using Cramer's rule
                let t = (f * c - e * b) / det;
                let s = (b * f - a * e) / det;

                let p1 = self.at(t);
                let p2 = other.at(s);

                // Verify if the distance between the closest points is effectively zero
                let gap_sq = (p2 - p1).magnitude_squared();
                match classify_to_zero(gap_sq, None) {
                    FloatSign::Zero => IntersectionResult::Single(p1),
                    _ => IntersectionResult::None,
                }
            }
        }
    }

    /// Computes the intersection between this infinite line and a finite segment.
    ///
    /// The segment is treated as a bounded portion of a line; the result is a single
    /// point only if the line and the segment meet at a point that lies within the
    /// segment's parameter range `[0, 1]`. Uses the same 2×2 parametric system as
    /// [`Self::intersect_line`], with the segment parameter clamped to the finite range.
    ///
    /// # Returns
    /// - [`None`](crate::IntersectionResult::None): The line and segment are parallel and separated, or the intersection lies outside the segment.
    /// - [`Single(p)`](crate::IntersectionResult::Single): The line intersects the segment at exactly one point `p`.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The segment lies entirely on the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, Segment, IntersectionResult};
    ///
    /// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
    /// let segment = Segment::new(Point::new([5.0, -1.0]), Point::new([5.0, 1.0]));
    ///
    /// if let IntersectionResult::Single(p) = line.intersect_segment(&segment) {
    ///     assert_eq!(p.coords()[0], 5.0);
    ///     assert_eq!(p.coords()[1], 0.0);
    /// }
    /// ```
    pub fn intersect_segment(&self, segment: &Segment<T, N>) -> IntersectionResult<T, N> {
        let v_diff = self.origin - segment.start();

        let a = T::one(); // self.direction is normalized
        let b = self.direction.dot(&segment.delta());
        let c = segment.length_squared();
        let e = segment.delta().dot(&v_diff);
        let f = self.direction.dot(&v_diff);

        let det = b * b - a * c;

        match classify_to_zero(det, None) {
            // Case: Line and segment are parallel
            FloatSign::Zero => {
                let closest = self.closest_point(&segment.start());
                let dist_sq = (closest - segment.start()).magnitude_squared();
                match classify_to_zero(dist_sq, None) {
                    FloatSign::Zero => IntersectionResult::Collinear,
                    _ => IntersectionResult::None,
                }
            }
            // Case: Line and segment are not parallel
            _ => {
                let t = (f * c - e * b) / det;
                let s = (b * f - a * e) / det;

                let one = T::one();
                let s_in_range = !matches!(classify_to_zero(s, None), FloatSign::Negative)
                    && !matches!(classify_to_zero(s - one, None), FloatSign::Positive);

                if s_in_range {
                    let p_line = self.at(t);
                    let p_seg = segment.at(s);
                    let gap_sq = (p_seg - p_line).magnitude_squared();
                    match classify_to_zero(gap_sq, None) {
                        FloatSign::Zero => IntersectionResult::Single(p_seg),
                        _ => IntersectionResult::None,
                    }
                } else {
                    IntersectionResult::None
                }
            }
        }
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Line<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point `p` onto the line to find the nearest location.
    ///
    /// The projection is calculated by finding the scalar projection of the
    /// vector (p - origin) onto the line's unit direction vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, SpatialRelation};
    ///
    /// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
    /// let p = Point::new([5.0, 10.0]);
    /// let closest = line.closest_point(&p);
    ///
    /// assert_eq!(closest.coords()[0], 5.0);
/// assert_eq!(closest.coords()[1], 0.0);
    /// ```
    #[inline]
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let t = (*p - self.origin).dot(&self.direction);
        self.origin + self.direction * t
    }

    /// Checks if a point lies on the line within the engine's tolerance.
    fn contains(&self, p: &Point<T, N>) -> bool {
        let projected = self.closest_point(p);
        let dist_sq = (*p - projected).magnitude_squared();
        match classify_to_zero(dist_sq, None) {
            FloatSign::Zero => true,
            _ => false,
        }
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
        assert_relative_eq!(line.at(0.0).coords()[0], 0.0);
        assert_relative_eq!(line.at(1.0).coords()[0], 1.0);
        assert_relative_eq!(line.at(-1.0).coords()[0], -1.0);
    }

    #[test]
    fn test_line_closest_point() {
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));

        let p = Point::new([5.0, 10.0]);

        let closest = line.closest_point(&p);
        assert_relative_eq!(closest.coords()[0], 5.0);
        assert_relative_eq!(closest.coords()[1], 0.0);
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

        match line.intersect_hypersphere(&sphere) {
            IntersectionResult::None => {}
            _ => panic!("Should not intersect: line is outside the radius"),
        }
    }

    #[test]
    fn test_line_sphere_tangent() {
        // Line passes exactly at y = 10, touching the sphere at (0, 10)
        let line = Line::new(Point::new([-5.0, 10.0]), Vector::new([1.0, 0.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);

        if let IntersectionResult::Tangent(p) = line.intersect_hypersphere(&sphere) {
            assert_relative_eq!(p.coords()[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords()[1], 10.0, epsilon = 1e-6);
        } else {
            panic!("Expected tangent intersection at (0, 10)");
        }
    }

    #[test]
    fn test_line_sphere_secant_centered() {
        // Line passes through the center along X axis
        let line = Line::new(Point::new([-20.0, 0.0]), Vector::new([1.0, 0.0]));
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);

        if let IntersectionResult::Secant(p1, p2) = line.intersect_hypersphere(&sphere) {
            // We expect (-10, 0) and (10, 0)
            let (min_x, max_x) = if p1.coords()[0] < p2.coords()[0] {
                (p1.coords()[0], p2.coords()[0])
            } else {
                (p2.coords()[0], p1.coords()[0])
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

        if let IntersectionResult::Secant(p1, p2) = line.intersect_hypersphere(&sphere) {
            let (min_y, max_y) = if p1.coords()[1] < p2.coords()[1] {
                (p1.coords()[1], p2.coords()[1])
            } else {
                (p2.coords()[1], p1.coords()[1])
            };
            assert_relative_eq!(min_y, -4.0, epsilon = 1e-6);
            assert_relative_eq!(max_y, 4.0, epsilon = 1e-6);
        } else {
            panic!("Expected secant points at y = -4 and y = 4");
        }
    }

    #[test]
    fn test_line_hyperplane_intersection_single_point() {
        // Plane: XY plane at z=0 (normal points up along Z axis)
        let plane_point = Point::new([0.0, 0.0, 0.0]);
        let plane_normal = Vector::new([0.0, 0.0, 1.0]);
        let plane = Hyperplane::new(plane_point, plane_normal);

        // Line: Starts at (0, 0, 10), points diagonally down towards the origin
        let line_origin = Point::new([0.0, 0.0, 10.0]);
        let line_dir = Vector::new([0.0, 0.0, -1.0]);
        let line = Line::new(line_origin, line_dir);

        // Expect intersection at the origin (0, 0, 0)
        match line.intersect_hyperplane(&plane) {
            IntersectionResult::Single(p) => {
                assert!((p.coords()[0] - 0.0).abs() < 1e-6);
                assert!((p.coords()[1] - 0.0).abs() < 1e-6);
                assert!((p.coords()[2] - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected a single intersection point"),
        }
    }

    #[test]
    fn test_line_hyperplane_intersection_parallel() {
        // Plane: XY plane at z=5
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 5.0]), Vector::new([0.0, 0.0, 1.0]));

        // Line: At z=10, pointing along X axis (parallel to the plane)
        let line = Line::new(Point::new([0.0, 0.0, 10.0]), Vector::new([1.0, 0.0, 0.0]));

        assert_eq!(line.intersect_hyperplane(&plane), IntersectionResult::None);
    }

    #[test]
    fn test_line_hyperplane_intersection_collinear() {
        // Plane: XY plane at z=0
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));

        // Line: At z=0, pointing along Y axis (contained in the plane)
        let line = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));

        assert_eq!(
            line.intersect_hyperplane(&plane),
            IntersectionResult::Collinear
        );
    }

    #[test]
    fn test_line_hyperplane_intersection_oblique() {
        // Plane: XY plane at z=0
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));

        // Line: Starts at (1, 1, 1), points towards (0, 0, -1)
        // t = (Po - P) * n / (v * n)
        // t = (0 - 1) * 1 / (-1 * 1) = -1 / -1 = 1
        // Point = (1, 1, 1) + 1 * (0, 0, -1) = (1, 1, 0)
        let line = Line::new(Point::new([1.0, 1.0, 1.0]), Vector::new([0.0, 0.0, -1.0]));

        if let IntersectionResult::Single(p) = line.intersect_hyperplane(&plane) {
            assert!((p.coords()[0] - 1.0).abs() < 1e-6);
            assert!((p.coords()[1] - 1.0).abs() < 1e-6);
            assert!((p.coords()[2] - 0.0).abs() < 1e-6);
        } else {
            panic!("Expected single oblique intersection");
        }
    }

    #[test]
    fn test_line_line_intersection_single_point() {
        // Line 1: Along the X axis
        let line1 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
        // Line 2: Along the Y axis
        let line2 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));

        // They should intersect exactly at the origin
        match line1.intersect_line(&line2) {
            IntersectionResult::Single(p) => {
                assert!((p.coords()[0] - 0.0).abs() < 1e-6);
                assert!((p.coords()[1] - 0.0).abs() < 1e-6);
                assert!((p.coords()[2] - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected a single intersection point at the origin"),
        }
    }

    #[test]
    fn test_line_line_intersection_parallel() {
        // Line 1: Along X axis at y=0
        let line1 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
        // Line 2: Along X axis at y=5 (Parallel)
        let line2 = Line::new(Point::new([0.0, 5.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));

        assert_eq!(line1.intersect_line(&line2), IntersectionResult::None);
    }

    #[test]
    fn test_line_line_intersection_collinear() {
        // Line 1: Along X axis
        let line1 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
        // Line 2: Also along X axis, but starting at x=10
        let line2 = Line::new(Point::new([10.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));

        assert_eq!(line1.intersect_line(&line2), IntersectionResult::Collinear);
    }

    #[test]
    fn test_line_line_intersection_skew() {
        // In 3D+, lines can cross without touching.
        // Line 1: Along X axis at z=0
        let line1 = Line::new(Point::new([0.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
        // Line 2: Along Y axis at z=1
        let line2 = Line::new(Point::new([0.0, 0.0, 1.0]), Vector::new([0.0, 1.0, 0.0]));

        // They are not parallel, but they don't intersect (distance is 1.0)
        assert_eq!(line1.intersect_line(&line2), IntersectionResult::None);
    }

    #[test]
    fn test_line_line_intersection_oblique_3d() {
        // More complex intersection in 3D space
        let line1 = Line::new(Point::new([1.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));
        let line2 = Line::new(Point::new([0.0, 1.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));

        // Intersection should be at (1, 1, 0)
        if let IntersectionResult::Single(p) = line1.intersect_line(&line2) {
            assert!((p.coords()[0] - 1.0).abs() < 1e-6);
            assert!((p.coords()[1] - 1.0).abs() < 1e-6);
            assert!((p.coords()[2] - 0.0).abs() < 1e-6);
        } else {
            panic!("Expected single intersection point at (1, 1, 0)");
        }
    }

    // --- Line ∩ Segment ---

    #[test]
    fn test_line_segment_intersection_single_point() {
        // Line: X axis. Segment: vertical from (5, -1) to (5, 1). Intersection at (5, 0).
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let segment = Segment::new(Point::new([5.0, -1.0]), Point::new([5.0, 1.0]));

        if let IntersectionResult::Single(p) = line.intersect_segment(&segment) {
            assert_relative_eq!(p.coords()[0], 5.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords()[1], 0.0, epsilon = 1e-6);
        } else {
            panic!("Expected single intersection at (5, 0)");
        }
    }

    #[test]
    fn test_line_segment_parallel_no_intersection() {
        // Line: X axis at y=0. Segment: X axis at y=3 (parallel, separated).
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let segment = Segment::new(Point::new([0.0, 3.0]), Point::new([10.0, 3.0]));

        assert_eq!(
            line.intersect_segment(&segment),
            IntersectionResult::None,
            "Parallel line and segment should not intersect"
        );
    }

    #[test]
    fn test_line_segment_collinear() {
        // Line: X axis. Segment: from (2, 0) to (8, 0) — lies on the line.
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let segment = Segment::new(Point::new([2.0, 0.0]), Point::new([8.0, 0.0]));

        assert_eq!(
            line.intersect_segment(&segment),
            IntersectionResult::Collinear,
            "Segment on the line should be Collinear"
        );
    }

    #[test]
    fn test_line_segment_non_parallel_no_intersection() {
        // Line: through (0, 0) direction (1, 0). Segment: from (5, 1) to (5, 3) — vertical strip that the line would hit at (5, 0), but segment is only y in [1, 3], so no intersection.
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let segment = Segment::new(Point::new([5.0, 1.0]), Point::new([5.0, 3.0]));

        // Line hits the infinite line of the segment at (5, 0); segment has y in [1, 3], so (5, 0) is outside the segment.
        assert_eq!(
            line.intersect_segment(&segment),
            IntersectionResult::None,
            "Line hits segment's line outside segment bounds"
        );
    }

    #[test]
    fn test_line_segment_intersection_at_endpoint() {
        // Line: diagonal y = x. Segment: from (0, 0) to (2, 0). They meet at (0, 0) — segment start.
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 1.0]));
        let segment = Segment::new(Point::new([0.0, 0.0]), Point::new([2.0, 0.0]));

        if let IntersectionResult::Single(p) = line.intersect_segment(&segment) {
            assert_relative_eq!(p.coords()[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords()[1], 0.0, epsilon = 1e-6);
        } else {
            panic!("Expected single intersection at segment start (0, 0)");
        }
    }

    #[test]
    fn test_line_segment_oblique_intersection() {
        // Line: from (0, 0) direction (1, 1). Segment: from (2, 0) to (2, 4). Intersection at (2, 2).
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 1.0]));
        let segment = Segment::new(Point::new([2.0, 0.0]), Point::new([2.0, 4.0]));

        if let IntersectionResult::Single(p) = line.intersect_segment(&segment) {
            assert_relative_eq!(p.coords()[0], 2.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords()[1], 2.0, epsilon = 1e-6);
        } else {
            panic!("Expected single intersection at (2, 2)");
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_line_serialization_roundtrip() {
        use serde_json;

        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let json = serde_json::to_string(&line).unwrap();
        let restored: Line<f64, 2> = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(line.at(0.0).coords()[0], restored.at(0.0).coords()[0]);
        assert_relative_eq!(line.at(0.0).coords()[1], restored.at(0.0).coords()[1]);
        assert_relative_eq!(line.at(1.0).coords()[0], restored.at(1.0).coords()[0]);
        assert_relative_eq!(line.at(1.0).coords()[1], restored.at(1.0).coords()[1]);
    }
}
