use crate::{
    AABB, Bounded, EuclideanVector, FloatSign, Hypersphere, IntersectionResult, Point,
    SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// A finite line segment in N-dimensional space defined by two endpoints.
///
/// The segment pre-calculates its displacement vector (delta), squared
/// magnitude, and its Axis-Aligned Bounding Box (AABB) to optimize
/// frequent spatial queries and broad-phase collision detection.
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
pub struct Segment<T, const N: usize> {
    start: Point<T, N>,
    end: Point<T, N>,
    delta: Vector<T, N>,
    mag_sq: T,               // Cached value for optimization
    cached_aabb: AABB<T, N>, // Cached Bounding Box
}

impl<T, const N: usize> Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new Segment from two points and pre-calculates the internal state.
    #[inline]
    pub fn new(start: Point<T, N>, end: Point<T, N>) -> Self {
        let delta = end - start;
        let mag_sq = delta.magnitude_squared();
        let cached_aabb = Self::compute_aabb(start, end);
        Self {
            start,
            end,
            delta,
            mag_sq,
            cached_aabb,
        }
    }

    /// Internal helper to calculate the AABB from the two endpoints.
    fn compute_aabb(start: Point<T, N>, end: Point<T, N>) -> AABB<T, N> {
        let mut min_coords = [T::zero(); N];
        let mut max_coords = [T::zero(); N];

        for i in 0..N {
            let s = start.coords[i];
            let e = end.coords[i];
            if s < e {
                min_coords[i] = s;
                max_coords[i] = e;
            } else {
                min_coords[i] = e;
                max_coords[i] = s;
            }
        }

        AABB::new(Point::new(min_coords), Point::new(max_coords))
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

    /// Updates the start point and synchronizes the delta, magnitude, and AABB.
    pub fn set_start(&mut self, new_start: Point<T, N>) {
        self.start = new_start;
        self.delta = self.end - self.start;
        self.mag_sq = self.delta.magnitude_squared();
        self.cached_aabb = Self::compute_aabb(self.start, self.end);
    }

    /// Updates the end point and synchronizes the delta, magnitude, and AABB.
    pub fn set_end(&mut self, new_end: Point<T, N>) {
        self.end = new_end;
        self.delta = self.end - self.start;
        self.mag_sq = self.delta.magnitude_squared();
        self.cached_aabb = Self::compute_aabb(self.start, self.end);
    }

    /// Returns the cached squared length of the segment.
    #[inline]
    pub fn length_squared(&self) -> T {
        self.mag_sq
    }

    /// Returns the displacement vector (end - start).
    #[inline]
    pub fn delta(&self) -> Vector<T, N> {
        self.delta
    }
}

impl<T, const N: usize> Bounded<T, N> for Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the cached Axis-Aligned Bounding Box.
    #[inline]
    fn aabb(&self) -> AABB<T, N> {
        self.cached_aabb
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
        self.delta.normalize()
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
        self.start + self.delta * t
    }

    /// Internal helper to find the parameter `t` for a point `p`.
    fn get_t(&self, p: Point<T, N>) -> T {
        if self.mag_sq <= T::zero() {
            return T::zero();
        }
        (p - self.start).dot(&self.delta) / self.mag_sq
    }

    /// Calculates the intersection points between this segment and a hypersphere.
    ///
    /// This method identifies if the segment enters, leaves, or touches the sphere.
    ///
    /// # Returns
    /// - `IntersectionResult::None`: No intersection within segment bounds.
    /// - `IntersectionResult::Tangent(p)`: The segment is tangent to the sphere.
    /// - `IntersectionResult::Secant(p1, p2)`: Both intersection points lie on the segment.
    /// - `IntersectionResult::Single(p)`: Only one end of the segment is inside the sphere.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Segment, Hypersphere, IntersectionResult};
    ///
    /// let circle = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);
    /// let segment = Segment::new(Point::new([-5.0, 0.0]), Point::new([0.0, 0.0]));
    ///
    /// if let IntersectionResult::Single(p) = segment.intersect_sphere(&circle) {
    ///     assert_eq!(p.coords[0], -2.0);
    /// }
    /// ```
    pub fn intersect_sphere(&self, sphere: &Hypersphere<T, N>) -> IntersectionResult<T, N> {
        let mag_sq = self.length_squared();
        if mag_sq <= T::zero() {
            return IntersectionResult::None;
        }

        let t_line = (sphere.center() - self.start).dot(&self.delta) / mag_sq;
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
                let dir = self.direction().unwrap_or(self.delta);

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
            let t = ((*p - self.start).dot(&self.delta) / mag_sq)
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
    use crate::Point;
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
            seg.intersect_sphere(&sphere),
            IntersectionResult::None
        ));
    }

    #[test]
    fn test_segment_entirely_inside_sphere() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);
        let seg = Segment::new(Point::new([-2.0, 0.0]), Point::new([2.0, 0.0]));

        // No intersection with the BOUNDARY
        assert!(matches!(
            seg.intersect_sphere(&sphere),
            IntersectionResult::None
        ));
    }

    #[test]
    fn test_segment_piercing_one_side() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        // Starts inside (0,0), ends outside (10,0) -> Should hit boundary at (5,0)
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));

        if let IntersectionResult::Single(p) = seg.intersect_sphere(&sphere) {
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

        if let IntersectionResult::Secant(p1, p2) = seg.intersect_sphere(&sphere) {
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

        if let IntersectionResult::Tangent(p) = seg.intersect_sphere(&sphere) {
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

        if let IntersectionResult::Single(p) = seg.intersect_sphere(&sphere) {
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
}
