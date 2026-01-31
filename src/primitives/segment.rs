use crate::{
    AABB, Bounded, EuclideanVector, FloatSign, Hyperplane, Hypersphere, IntersectionResult, Line,
    Point, SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// A finite line segment in N-dimensional space defined by two endpoints.
///
/// Derived quantities (displacement vector, length, direction, etc.) are
/// computed on demand to favor simulations that update endpoints often and
/// only need spatial queries (e.g. AABB overlap) in a subset of steps.
///
/// # Examples
///
/// ```
/// use apollonius::{Point, Segment};
///
/// let start = Point::new([0.0, 0.0]);
/// let end = Point::new([10.0, 0.0]);
/// let segment = Segment::new(start, end);
///
/// assert_eq!(segment.length(), 10.0);
/// assert_eq!(segment.midpoint().coords[0], 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de>"
    ))
)]
pub struct Segment<T, const N: usize> {
    start: Point<T, N>,
    end: Point<T, N>,
}

impl<T, const N: usize> Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new segment from two endpoints.
    #[inline]
    pub fn new(start: Point<T, N>, end: Point<T, N>) -> Self {
        Self { start, end }
    }

    /// Returns the start point of the segment.
    #[inline]
    pub fn start(&self) -> Point<T, N> {
        self.start
    }

    /// Returns the end point of the segment.
    #[inline]
    pub fn end(&self) -> Point<T, N> {
        self.end
    }

    /// Updates the start point.
    pub fn set_start(&mut self, new_start: Point<T, N>) {
        self.start = new_start;
    }

    /// Updates the end point.
    pub fn set_end(&mut self, new_end: Point<T, N>) {
        self.end = new_end;
    }

    /// Returns the squared length of the segment.
    #[inline]
    pub fn length_squared(&self) -> T {
        self.delta().magnitude_squared()
    }

    /// Returns the displacement vector (end - start).
    #[inline]
    pub fn delta(&self) -> Vector<T, N> {
        self.end - self.start
    }
}

impl<T, const N: usize> Bounded<T, N> for Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the Axis-Aligned Bounding Box enclosing the segment.
    ///
    /// Min and max coordinates are computed per axis using [`std::array::from_fn`],
    /// so the AABB is built in a single pass over the dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Segment, Bounded};
    ///
    /// let seg = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 0.0]));
    /// let aabb = seg.aabb();
    /// assert_eq!(aabb.min.coords, [0.0, 0.0]);
    /// assert_eq!(aabb.max.coords, [10.0, 10.0]);
    /// ```
    fn aabb(&self) -> AABB<T, N> {
        let (start, end) = (self.start().coords, self.end.coords);

        let min_coords = std::array::from_fn(|i| start[i].min(end[i]));
        let max_coords = std::array::from_fn(|i| start[i].max(end[i]));

        AABB::new(Point::new(min_coords), Point::new(max_coords))
    }
}

impl<T, const N: usize> Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the Euclidean length of the segment.
    #[inline]
    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    /// Returns the normalized direction vector of the segment.
    ///
    /// Returns `None` if the segment has zero length (start == end).
    pub fn direction(&self) -> Option<Vector<T, N>> {
        self.delta().normalize()
    }

    /// Returns the point at the exact center of the segment.
    pub fn midpoint(&self) -> Point<T, N> {
        let half = T::from(0.5).unwrap();
        self.at(half)
    }

    /// Returns a point along the segment at parameter `t`.
    ///
    /// * `t = 0.0` yields the start point.
    /// * `t = 1.0` yields the end point.
    ///
    /// Note: This method allows extrapolation if `t` is outside the [0, 1] range.
    #[inline]
    pub fn at(&self, t: T) -> Point<T, N> {
        self.start + self.delta() * t
    }

    /// Internal helper to find the parameter `t` for a point `p`.
    fn get_t(&self, p: Point<T, N>) -> T {
        let mag_sq = self.length_squared();
        if mag_sq <= T::zero() {
            return T::zero();
        }
        let delta = self.delta();
        (p - self.start).dot(&delta) / mag_sq
    }

    /// Calculates the intersection points between this segment and a hypersphere.
    ///
    /// This method identifies if the segment enters, leaves, or touches the sphere.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Collinear` or `HalfSpacePenetration`):
    /// - [`None`](crate::IntersectionResult::None): No intersection within segment bounds.
    /// - [`Tangent(p)`](crate::IntersectionResult::Tangent): The segment is tangent to the sphere at `p`.
    /// - [`Secant(p1, p2)`](crate::IntersectionResult::Secant): Both intersection points lie on the segment.
    /// - [`Single(p)`](crate::IntersectionResult::Single): Exactly one intersection point lies on the segment (e.g. segment enters or exits the sphere).
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Segment, Hypersphere, IntersectionResult};
    ///
    /// let circle = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);
    /// let segment = Segment::new(Point::new([-5.0, 0.0]), Point::new([0.0, 0.0]));
    ///
    /// if let IntersectionResult::Single(p) = segment.intersect_hypersphere(&circle) {
    ///     assert_eq!(p.coords[0], -2.0);
    /// }
    /// ```
    pub fn intersect_hypersphere(&self, sphere: &Hypersphere<T, N>) -> IntersectionResult<T, N> {
        let mag_sq = self.length_squared();
        if mag_sq <= T::zero() {
            return IntersectionResult::None;
        }

        let delta = self.delta();
        let t_line = (sphere.center() - self.start).dot(&delta) / mag_sq;
        let pc = self.at(t_line);

        let dist_sq = (pc - sphere.center()).magnitude_squared();
        let r_sq = sphere.radius() * sphere.radius();
        let diff = r_sq - dist_sq;

        match classify_to_zero(diff, None) {
            FloatSign::Negative => IntersectionResult::None,
            FloatSign::Zero => {
                if t_line >= -T::epsilon() && t_line <= T::one() + T::epsilon() {
                    IntersectionResult::Tangent(pc)
                } else {
                    IntersectionResult::None
                }
            }
            FloatSign::Positive => {
                let h = diff.sqrt();
                let dir = self.direction().unwrap_or_else(|| self.delta());

                let p1 = pc - dir * h;
                let p2 = pc + dir * h;

                let t1 = self.get_t(p1);
                let t2 = self.get_t(p2);

                let v1 = t1 >= -T::epsilon() && t1 <= T::one() + T::epsilon();
                let v2 = t2 >= -T::epsilon() && t2 <= T::one() + T::epsilon();

                match (v1, v2) {
                    (true, true) => IntersectionResult::Secant(p1, p2),
                    (true, false) => IntersectionResult::Single(p1),
                    (false, true) => IntersectionResult::Single(p2),
                    (false, false) => IntersectionResult::None,
                }
            }
        }
    }

    /// Computes the intersection between this segment and a hyperplane.
    ///
    /// Delegates to [`Hyperplane::intersect_segment`]. The segment is treated
    /// as a finite line; the result depends on whether it crosses the plane,
    /// is parallel to it, or lies entirely on it.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Tangent`, `Secant`, or `HalfSpacePenetration`):
    /// - [`Single(p)`](crate::IntersectionResult::Single): The segment crosses the hyperplane at point `p`.
    /// - [`None`](crate::IntersectionResult::None): The segment is parallel (and not on the plane) or does not reach it.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The entire segment lies on the hyperplane.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hyperplane, Segment, IntersectionResult};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]));
    /// let segment = Segment::new(Point::new([0.0, -1.0]), Point::new([0.0, 1.0]));
    ///
    /// if let IntersectionResult::Single(p) = segment.intersect_hyperplane(&plane) {
    ///     assert_eq!(p.coords[1], 0.0);
    /// }
    /// ```
    #[inline]
    pub fn intersect_hyperplane(&self, hyperplane: &Hyperplane<T, N>) -> IntersectionResult<T, N> {
        hyperplane.intersect_segment(self)
    }

    /// Computes the intersection between this segment and an infinite line.
    ///
    /// Delegates to [`Line::intersect_segment`]. The result is a point only if the
    /// line meets the segment within its finite bounds.
    ///
    /// # Returns
    /// - [`None`](crate::IntersectionResult::None): The line and segment are parallel and separated, or the intersection lies outside the segment.
    /// - [`Single(p)`](crate::IntersectionResult::Single): The line intersects the segment at exactly one point `p`.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The segment lies entirely on the line.
    #[inline]
    pub fn intersect_line(&self, line: &Line<T, N>) -> IntersectionResult<T, N> {
        line.intersect_segment(self)
    }

    /// Computes the intersection between this segment and another segment in N-dimensional space.
    ///
    /// Uses a parametric formulation and Cramer's rule for non-parallel segments;
    /// for parallel or collinear segments, the overlap is computed in parameter
    /// space and clamped to `[0, 1]`. The AABB broad-phase is applied only when
    /// segments are not parallel, to avoid false rejections for collinear segments
    /// with degenerate AABBs on the same line.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Tangent`, `Secant`, or `HalfSpacePenetration`):
    /// - [`None`](crate::IntersectionResult::None): The segments do not intersect (parallel and separated, or skew).
    /// - [`Single(p)`](crate::IntersectionResult::Single): The segments intersect at exactly one point `p` (crossing or touching at an endpoint).
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The segments are collinear and overlap over a positive length.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Segment, IntersectionResult};
    ///
    /// let s1 = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));
    /// let s2 = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 0.0]));
    ///
    /// if let IntersectionResult::Single(p) = s1.intersect_segment(&s2) {
    ///     assert_eq!(p.coords[0], 5.0);
    ///     assert_eq!(p.coords[1], 5.0);
    /// }
    /// ```
    pub fn intersect_segment(&self, other: &Segment<T, N>) -> IntersectionResult<T, N> {
        let v_diff = self.start - other.start; // P1 - P2
        let d_self = self.delta();
        let d_other = other.delta();
        let a = d_self.magnitude_squared(); // V1 · V1
        let b = d_self.dot(&d_other); // V1 · V2
        let c = d_other.magnitude_squared(); // V2 · V2
        let e = d_other.dot(&v_diff); // V2 · (P1 - P2)
        let f = d_self.dot(&v_diff); // V1 · (P1 - P2)

        let det = b * b - a * c;

        match classify_to_zero(det, None) {
            // Case: Segments are parallel (possibly collinear)
            // Do not use AABB broad-phase here: degenerate AABBs on the same line
            // (e.g. both y=0) make intersects() return false due to <= comparison.
            FloatSign::Zero => {
                // Correct projection: t = (P2 - P1) · V1 / |V1|² = -f / a
                let t_proj = if !matches!(classify_to_zero(a, None), FloatSign::Zero) {
                    -f / a
                } else {
                    T::zero()
                };
                let closest_on_line = self.start + d_self * t_proj;
                let dist_to_line_sq = (other.start - closest_on_line).magnitude_squared();

                if let FloatSign::Zero = classify_to_zero(dist_to_line_sq, None) {
                    // Lines are collinear. Find range of 'other' in 'self' space [t0, t1]
                    let t0 = -f / a;
                    let t1 = (b - f) / a;

                    let t_min = if t0 < t1 { t0 } else { t1 };
                    let t_max = if t0 > t1 { t0 } else { t1 };

                    // Clamp intersection to [0, 1] using Apollonius classification
                    let zero = T::zero();
                    let one = T::one();

                    let overlap_min = if let FloatSign::Negative = classify_to_zero(t_min, None) {
                        zero
                    } else {
                        t_min
                    };
                    let overlap_max =
                        if let FloatSign::Positive = classify_to_zero(t_max - one, None) {
                            one
                        } else {
                            t_max
                        };

                    // Compare clamped endpoints to determine result
                    match classify_to_zero(overlap_max - overlap_min, None) {
                        FloatSign::Positive => IntersectionResult::Collinear,
                        FloatSign::Zero => IntersectionResult::Single(self.at(overlap_min)),
                        FloatSign::Negative => IntersectionResult::None,
                    }
                } else {
                    IntersectionResult::None
                }
            }
            // Case: Segments are not parallel — safe to use AABB broad-phase
            _ => {
                if !self.aabb().intersects(&other.aabb()) {
                    return IntersectionResult::None;
                }
                // Solve 2x2 system using Cramer's rule
                let t = (f * c - e * b) / det;
                let s = (b * f - a * e) / det;

                let one = T::one();

                // Check if t and s are within [0, 1] using classify_to_zero
                let t_valid = !matches!(classify_to_zero(t, None), FloatSign::Negative)
                    && !matches!(classify_to_zero(t - one, None), FloatSign::Positive);
                let s_valid = !matches!(classify_to_zero(s, None), FloatSign::Negative)
                    && !matches!(classify_to_zero(s - one, None), FloatSign::Positive);

                if t_valid && s_valid {
                    let p1 = self.at(t);
                    let p2 = other.at(s);

                    if let FloatSign::Zero = classify_to_zero((p2 - p1).magnitude_squared(), None) {
                        return IntersectionResult::Single(p1);
                    }
                }
                IntersectionResult::None
            }
        }
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point `p` onto the segment and returns the closest point.
    ///
    /// The resulting point is clamped to the segment's endpoints.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Segment, SpatialRelation};
    ///
    /// let segment = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));
    /// let p = Point::new([-5.0, 5.0]);
    ///
    /// // Should clamp to the start point
    /// assert_eq!(segment.closest_point(&p).coords[0], 0.0);
    /// ```
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let mag_sq = self.length_squared();

        if mag_sq > T::zero() {
            let delta = self.delta();
            let t = ((*p - self.start).dot(&delta) / mag_sq)
                .max(T::zero())
                .min(T::one());

            self.at(t)
        } else {
            self.start
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_segment_closest_point() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 0.0]);
        let seg = Segment::new(start, end);

        let p_behind = Point::new([-5.0, 5.0]);
        assert_eq!(seg.closest_point(&p_behind).coords, [0.0, 0.0]);

        let p_ahead = Point::new([15.0, 5.0]);
        assert_eq!(seg.closest_point(&p_ahead).coords, [10.0, 0.0]);

        let p_mid = Point::new([5.0, 10.0]);
        assert_eq!(seg.closest_point(&p_mid).coords, [5.0, 0.0]);
    }

    #[test]
    fn test_segment_metrics() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([3.0, 4.0]);
        let seg = Segment::new(start, end);

        assert_relative_eq!(seg.length_squared(), 25.0);
        assert_relative_eq!(seg.length(), 5.0);
    }

    #[test]
    fn test_segment_midpoint() {
        let seg = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 20.0]));
        let mid = seg.midpoint();

        assert_relative_eq!(mid.coords[0], 5.0);
        assert_relative_eq!(mid.coords[1], 15.0);
    }

    #[test]
    fn test_segment_at_parameter() {
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));

        assert_relative_eq!(seg.at(0.2).coords[0], 2.0);
        assert_relative_eq!(seg.at(1.0).coords[0], 10.0);
        assert_relative_eq!(seg.at(1.5).coords[0], 15.0);
    }

    #[test]
    fn test_segment_direction() {
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([0.0, 10.0]));
        let dir = seg.direction().expect("Should have direction");

        assert_relative_eq!(dir.coords[0], 0.0);
        assert_relative_eq!(dir.coords[1], 1.0);

        let point_seg = Segment::new(Point::new([1.0, 1.0]), Point::new([1.0, 1.0]));
        assert!(point_seg.direction().is_none());
    }

    #[test]
    fn test_segment_zero_length_proximity() {
        let p = Point::new([1.0, 1.0]);
        let seg = Segment::new(p, p);

        let query = Point::new([5.0, 5.0]);
        assert_eq!(seg.closest_point(&query), p);
    }

    #[test]
    fn test_segment_contains_midpoint() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 0.0]);
        let segment = Segment::new(start, end);

        let midpoint = Point::new([5.0, 0.0]);

        assert!(
            segment.contains(&midpoint),
            "The midpoint should be contained within the segment"
        );
    }

    #[test]
    fn test_segment_contains_endpoints() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 10.0]);
        let segment = Segment::new(start, end);

        assert!(
            segment.contains(&start),
            "The start point is part of the segment"
        );
        assert!(
            segment.contains(&end),
            "The end point is part of the segment"
        );
    }

    #[test]
    fn test_segment_excludes_collinear_point_outside() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 0.0]);
        let segment = Segment::new(start, end);

        let outside_point = Point::new([11.0, 0.0]); // Collinear but out of range

        assert!(
            !segment.contains(&outside_point),
            "A collinear but distant point should NOT be contained"
        );
    }

    #[test]
    fn test_segment_excludes_point_off_line() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 0.0]);
        let segment = Segment::new(start, end);

        let off_point = Point::new([5.0, 1.0]); // "Above" the segment

        assert!(
            !segment.contains(&off_point),
            "A point off the line should not be contained"
        );
    }

    #[test]
    fn test_segment_sphere_no_intersection() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        let seg = Segment::new(Point::new([6.0, 0.0]), Point::new([10.0, 0.0]));

        assert!(matches!(
            seg.intersect_hypersphere(&sphere),
            IntersectionResult::None
        ));
    }

    #[test]
    fn test_segment_entirely_inside_sphere() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);
        let seg = Segment::new(Point::new([-2.0, 0.0]), Point::new([2.0, 0.0]));

        // No intersection with the BOUNDARY
        assert!(matches!(
            seg.intersect_hypersphere(&sphere),
            IntersectionResult::None
        ));
    }

    #[test]
    fn test_segment_piercing_one_side() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        // Starts inside (0,0), ends outside (10,0) -> Should hit boundary at (5,0)
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));

        if let IntersectionResult::Single(p) = seg.intersect_hypersphere(&sphere) {
            assert_relative_eq!(p.coords[0], 5.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords[1], 0.0, epsilon = 1e-6);
        } else {
            panic!("Expected single intersection point at (5, 0)");
        }
    }

    #[test]
    fn test_segment_piercing_both_sides() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        let seg = Segment::new(Point::new([-10.0, 0.0]), Point::new([10.0, 0.0]));

        if let IntersectionResult::Secant(p1, p2) = seg.intersect_hypersphere(&sphere) {
            let x1 = p1.coords[0];
            let x2 = p2.coords[0];
            assert!((x1.abs() - 5.0).abs() < 1e-6);
            assert!((x2.abs() - 5.0).abs() < 1e-6);
            assert!((x1 - x2).abs() > 9.0); // Points are far apart
        } else {
            panic!("Expected two intersection points at -5 and 5");
        }
    }

    #[test]
    fn test_segment_tangent_within_bounds() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        let seg = Segment::new(Point::new([-10.0, 5.0]), Point::new([10.0, 5.0]));

        if let IntersectionResult::Tangent(p) = seg.intersect_hypersphere(&sphere) {
            assert_relative_eq!(p.coords[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords[1], 5.0, epsilon = 1e-6);
        } else {
            panic!("Expected tangent intersection at (0, 5)");
        }
    }

    #[test]
    fn test_segment_sphere_broken_by_clamping() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        // Segment starts inside the sphere's X-range of intersection but the
        // orthogonal projection (0, 1) is outside the segment [4, 10].
        let seg = Segment::new(Point::new([4.0, 1.0]), Point::new([10.0, 1.0]));

        if let IntersectionResult::Single(p) = seg.intersect_hypersphere(&sphere) {
            // The real intersection is at x = sqrt(r^2 - y^2) = sqrt(25 - 1) = sqrt(24)
            let expected_x = 24.0f64.sqrt();

            assert_relative_eq!(p.coords[0], expected_x, epsilon = 1e-6);
            assert_relative_eq!(p.coords[1], 1.0, epsilon = 1e-6);

            // CRITICAL CHECK: The point must actually be on the sphere surface
            let dist_to_center = (p - sphere.center()).magnitude();
            assert_relative_eq!(dist_to_center, 5.0, epsilon = 1e-6);
        } else {
            panic!("Should have found a Single intersection at x ≈ 4.898");
        }
    }

    #[test]
    fn test_segment_initial_aabb() {
        // Diagonal segment from (0, 10) to (10, 0)
        let seg = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 0.0]));
        let aabb = seg.aabb();

        // Min should be (0, 0), Max should be (10, 10)
        assert_relative_eq!(aabb.min.coords[0], 0.0);
        assert_relative_eq!(aabb.min.coords[1], 0.0);
        assert_relative_eq!(aabb.max.coords[0], 10.0);
        assert_relative_eq!(aabb.max.coords[1], 10.0);
    }

    #[test]
    fn test_aabb_updates_on_set_start() {
        let mut seg = Segment::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));

        // Move start far to the negative side
        seg.set_start(Point::new([-10.0, -10.0]));
        let aabb = seg.aabb();

        assert_relative_eq!(aabb.min.coords[0], -10.0);
        assert_relative_eq!(aabb.min.coords[1], -10.0);
        assert_relative_eq!(aabb.max.coords[0], 5.0);
        assert_relative_eq!(aabb.max.coords[1], 5.0);
    }

    #[test]
    fn test_aabb_updates_on_set_end() {
        let mut seg = Segment::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));

        // Shrink end point
        seg.set_end(Point::new([1.0, 2.0]));
        let aabb = seg.aabb();

        assert_relative_eq!(aabb.min.coords[0], 0.0);
        assert_relative_eq!(aabb.max.coords[0], 1.0);
        assert_relative_eq!(aabb.max.coords[1], 2.0);
    }

    #[test]
    fn test_aabb_handles_endpoint_swap() {
        let mut seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));

        // Swap endpoints
        seg.set_start(Point::new([10.0, 10.0]));
        seg.set_end(Point::new([0.0, 0.0]));

        let aabb = seg.aabb();
        // AABB should still be (0,0) to (10,10)
        assert_relative_eq!(aabb.min.coords[0], 0.0);
        assert_relative_eq!(aabb.max.coords[0], 10.0);
    }

    #[test]
    fn test_segment_segment_intersection_cross() {
        let s1 = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));
        let s2 = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 0.0]));

        if let IntersectionResult::Single(p) = s1.intersect_segment(&s2) {
            assert_relative_eq!(p.coords[0], 5.0);
            assert_relative_eq!(p.coords[1], 5.0);
        } else {
            panic!("Expected single intersection at (5, 5)");
        }
    }

    #[test]
    fn test_segment_segment_collinear_overlap() {
        let s1 = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));
        let s2 = Segment::new(Point::new([5.0, 0.0]), Point::new([15.0, 0.0]));

        assert_eq!(s1.intersect_segment(&s2), IntersectionResult::Collinear);
    }

    #[test]
    fn test_segment_segment_touch_at_endpoint() {
        let s1 = Segment::new(Point::new([0.0, 0.0]), Point::new([5.0, 0.0]));
        let s2 = Segment::new(Point::new([5.0, 0.0]), Point::new([10.0, 0.0]));

        if let IntersectionResult::Single(p) = s1.intersect_segment(&s2) {
            assert_relative_eq!(p.coords[0], 5.0);
        } else {
            panic!("Expected single point intersection at endpoint (5, 0)");
        }
    }

    #[test]
    fn test_segment_segment_parallel_no_intersection() {
        let s1 = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));
        let s2 = Segment::new(Point::new([0.0, 1.0]), Point::new([10.0, 1.0]));

        assert_eq!(s1.intersect_segment(&s2), IntersectionResult::None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_segment_serialization_roundtrip() {
        use serde_json;

        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));
        let json = serde_json::to_string(&seg).unwrap();
        let restored: Segment<f64, 2> = serde_json::from_str(&json).unwrap();
        assert_eq!(seg.start, restored.start);
        assert_eq!(seg.end, restored.end);
    }
}
