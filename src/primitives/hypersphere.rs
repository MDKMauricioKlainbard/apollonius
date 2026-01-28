use crate::{
    AABB, Bounded, EuclideanVector, FloatSign, IntersectionResult, Line, Point, Segment,
    SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// An N-dimensional hypersphere defined by a center point and a radius.
///
/// In 2D, this represents a circle. In 3D, a sphere. In higher dimensions,
/// it represents the set of all points at a fixed distance from a central point.
///
/// This structure maintains a cached `AABB` to optimize spatial queries.
/// #[derive(Debug, PartialEq, Clone, Copy)]
pub struct Hypersphere<T, const N: usize> {
    center: Point<T, N>,
    radius: T,
    cached_aabb: AABB<T, N>,
}
/// A 2-dimensional hypersphere.
pub type Circle<T> = Hypersphere<T, 2>;
/// A 3-dimensional hypersphere.
pub type Sphere<T> = Hypersphere<T, 3>;

impl<T, const N: usize> Hypersphere<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new hypersphere and pre-calculates its bounding box.
    #[inline]
    pub fn new(center: Point<T, N>, radius: T) -> Self {
        let cached_aabb = Self::compute_aabb(&center, radius);
        Self {
            center,
            radius,
            cached_aabb,
        }
    }

    /// Static helper to compute an AABB from a center and radius without instantiation.
    fn compute_aabb(center: &Point<T, N>, radius: T) -> AABB<T, N> {
        let mut min_coords = [T::zero(); N];
        let mut max_coords = [T::zero(); N];

        for i in 0..N {
            min_coords[i] = center.coords[i] - radius;
            max_coords[i] = center.coords[i] + radius;
        }

        AABB {
            min: Point::new(min_coords),
            max: Point::new(max_coords),
        }
    }

    /// Returns a reference to the hypersphere's center.
    #[inline]
    pub fn center(&self) -> Point<T, N> {
        self.center
    }

    /// Returns the current radius.
    #[inline]
    pub fn radius(&self) -> T {
        self.radius
    }

    /// Updates the center and performs an incremental O(N) translation of the cached AABB.
    pub fn set_center(&mut self, new_center: Point<T, N>) {
        let offset = new_center - self.center;
        self.center = new_center;

        self.cached_aabb.min = self.cached_aabb.min + offset;
        self.cached_aabb.max = self.cached_aabb.max + offset;
    }

    /// Updates the radius and expands or contracts the cached AABB radially in O(N).
    pub fn set_radius(&mut self, new_radius: T) {
        let delta_r = new_radius - self.radius;
        self.radius = new_radius;

        for i in 0..N {
            self.cached_aabb.min.coords[i] = self.cached_aabb.min.coords[i] - delta_r;
            self.cached_aabb.max.coords[i] = self.cached_aabb.max.coords[i] + delta_r;
        }
    }
}

impl<T, const N: usize> Bounded<T, N> for Hypersphere<T, N>
where
    T: Copy,
{
    /// Returns the cached Axis-Aligned Bounding Box.
    #[inline]
    fn aabb(&self) -> AABB<T, N> {
        self.cached_aabb
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Hypersphere<T, N>
where
    T: std::iter::Sum + Float,
{
    /// Finds the closest point on the hypersphere's surface to a given point `p`.
    ///
    /// If `p` is at the center of the hypersphere, the projection is undefined.
    /// In this specific case, the method defaults to projecting the point
    /// along the positive X-axis (canonical direction).
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let direction = (*p - self.center).normalize().unwrap_or_else(|| {
            let mut direction = Vector::new([T::zero(); N]);
            direction.coords[0] = T::one();
            direction
        });

        self.center + direction * self.radius
    }

    /// Checks if a point `p` lies exactly on the hypersphere's boundary (surface).
    ///
    /// This method uses `classify_to_zero` to account for floating-point inaccuracies.
    /// Points strictly inside or outside the surface will return `false`.
    fn contains(&self, p: &Point<T, N>) -> bool {
        let dist_sq = (*p - self.center).magnitude_squared();
        let radius_sq = self.radius * self.radius;
        match classify_to_zero((dist_sq - radius_sq).abs(), None) {
            FloatSign::Zero => true,
            _ => false,
        }
    }

    /// Checks if a point `p` is contained within the hypersphere's volume.
    ///
    /// Returns `true` if the distance from the center to `p` is less than or
    /// equal to the radius (including the boundary).
    fn is_inside(&self, p: &Point<T, N>) -> bool {
        let dist_sq = (*p - self.center).magnitude_squared();
        let radius_sq = self.radius * self.radius;

        // diff = distance^2 - radius^2
        // If Negative: inside volume
        // If Zero: on boundary
        // If Positive: outside
        match classify_to_zero(dist_sq - radius_sq, None) {
            FloatSign::Positive => false,
            _ => true,
        }
    }
}

impl<T, const N: usize> Hypersphere<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point `p` onto the hypersphere's surface.
    ///
    /// This is an alias for [`SpatialRelation::closest_point`].
    #[inline]
    pub fn project(&self, p: &Point<T, N>) -> Point<T, N> {
        self.closest_point(&p)
    }

    /// Computes the intersection(s) between this hypersphere and an infinite line.
    ///
    /// This is a convenience method that delegates the geometric calculation to
    /// [`Line::intersect_sphere`].
    #[inline]
    pub fn intersect_line(&self, line: &Line<T, N>) -> IntersectionResult<T, N> {
        line.intersect_sphere(self)
    }

    /// Computes the intersection(s) between this hypersphere and a finite line segment.
    ///
    /// This is a convenience method that delegates the geometric calculation to
    /// [`Segment::intersect_sphere`].
    #[inline]
    pub fn intersect_segment(&self, segment: &Segment<T, N>) -> IntersectionResult<T, N> {
        segment.intersect_sphere(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use approx::assert_relative_eq;

    #[test]
    fn test_sphere_boundary_contains_point() {
        // Circle at (0,0) with radius 10
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);
        let point_on_boundary = Point::new([10.0, 0.0]);

        assert!(
            sphere.contains(&point_on_boundary),
            "Point exactly on the radius should be contained in the boundary"
        );
    }

    #[test]
    fn test_sphere_boundary_excludes_interior_point() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);
        let point_inside = Point::new([5.0, 5.0]);

        assert!(
            !sphere.contains(&point_inside),
            "Interior points should not be part of the boundary (surface)"
        );
    }

    #[test]
    fn test_is_inside_volume_check() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 10.0);
        let point_inside = Point::new([5.0, 5.0]);
        let point_on_boundary = Point::new([0.0, 10.0]);
        let point_outside = Point::new([11.0, 0.0]);

        assert!(
            sphere.is_inside(&point_inside),
            "Point deep inside should be inside"
        );
        assert!(
            sphere.is_inside(&point_on_boundary),
            "Boundary point should be considered inside"
        );
        assert!(
            !sphere.is_inside(&point_outside),
            "Point beyond radius must be outside"
        );
    }

    #[test]
    fn test_spatial_consistency_3d() {
        // 3D Sphere at origin, radius 5
        let sphere = Hypersphere::new(Point::new([0.0, 0.0, 0.0]), 5.0);
        let point = Point::new([0.0, 3.0, 4.0]); // 3-4-5 triangle, distance is exactly 5

        assert!(
            sphere.contains(&point),
            "3D boundary check failed at exact radius"
        );
        assert!(
            sphere.is_inside(&point),
            "3D interior check failed at exact radius"
        );
    }

    #[test]
    fn test_negative_coordinates_handling() {
        let sphere = Hypersphere::new(Point::new([-10.0, -10.0]), 5.0);
        let point = Point::new([-12.0, -10.0]); // Distance is 2

        assert!(
            sphere.is_inside(&point),
            "Failed to handle negative coordinate space"
        );
    }

    #[test]
    fn test_closest_point_projection_from_outside() {
        let circle = Circle::new(Point::new([0.0, 0.0]), 5.0);
        let p = Point::new([10.0, 0.0]);
        let projected = circle.closest_point(&p);

        assert_relative_eq!(projected.coords[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(projected.coords[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_closest_point_projection_from_inside() {
        let circle = Circle::new(Point::new([0.0, 0.0]), 5.0);
        let p = Point::new([0.0, 2.0]);
        let projected = circle.closest_point(&p);

        assert_relative_eq!(projected.coords[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(projected.coords[1], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_closest_point_on_boundary_invariance() {
        let circle = Circle::new(Point::new([10.0, 10.0]), 5.0);
        let p = Point::new([15.0, 10.0]);
        let projected = circle.closest_point(&p);

        assert_relative_eq!(projected.coords[0], p.coords[0], epsilon = 1e-6);
        assert_relative_eq!(projected.coords[1], p.coords[1], epsilon = 1e-6);
    }

    #[test]
    fn test_projection_center_singularity_fallback() {
        let circle = Circle::new(Point::new([0.0, 0.0]), 10.0);
        let projected = circle.closest_point(&circle.center);

        // Fallback should project towards positive X-axis
        assert_relative_eq!(projected.coords[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(projected.coords[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sphere_line_secant_diagonal() {
        // Sphere at (0,0) radius 5
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        // Diagonal line y = x (normalized direction is [0.707, 0.707])
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 1.0]));

        if let IntersectionResult::Secant(p1, p2) = sphere.intersect_line(&line) {
            // Points should be at distance 5 from origin: 5 * cos(45) = 3.5355...
            let expected = 5.0 / 2.0f64.sqrt();
            assert_relative_eq!(p1.coords[0].abs(), expected, epsilon = 1e-6);
            assert_relative_eq!(p1.coords[1].abs(), expected, epsilon = 1e-6);
            assert_relative_eq!(p2.coords[0].abs(), expected, epsilon = 1e-6);
            assert_relative_eq!(p2.coords[1].abs(), expected, epsilon = 1e-6);
        } else {
            panic!("Expected diagonal secant intersection");
        }
    }

    #[test]
    fn test_sphere_line_tangent_top() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        let line = Line::new(Point::new([-10.0, 5.0]), Vector::new([1.0, 0.0]));

        if let IntersectionResult::Tangent(p) = sphere.intersect_line(&line) {
            assert_relative_eq!(p.coords[0], 0.0, epsilon = 1e-6);
            assert_relative_eq!(p.coords[1], 5.0, epsilon = 1e-6);
        } else {
            panic!("Expected tangent at (0, 5)");
        }
    }

    #[test]
    fn test_initial_aabb_calculation() {
        let circle = Circle::new(Point::new([10.0, 20.0]), 5.0);
        let aabb = circle.aabb();

        // Min: 10-5, 20-5 -> (5, 15)
        assert_relative_eq!(aabb.min.coords[0], 5.0);
        assert_relative_eq!(aabb.min.coords[1], 15.0);
        // Max: 10+5, 20+5 -> (15, 25)
        assert_relative_eq!(aabb.max.coords[0], 15.0);
        assert_relative_eq!(aabb.max.coords[1], 25.0);
    }

    #[test]
    fn test_aabb_update_after_moving_center() {
        let mut sphere = Sphere::new(Point::new([0.0, 0.0, 0.0]), 10.0);
        sphere.set_center(Point::new([100.0, 0.0, 0.0]));

        let aabb = sphere.aabb();
        // New Center 100, Radius 10 -> Min X: 90, Max X: 110
        assert_relative_eq!(aabb.min.coords[0], 90.0);
        assert_relative_eq!(aabb.max.coords[0], 110.0);
        // Y and Z should remain centered around 0 -> Min: -10, Max: 10
        assert_relative_eq!(aabb.min.coords[1], -10.0);
        assert_relative_eq!(aabb.max.coords[2], 10.0);
    }

    #[test]
    fn test_aabb_update_after_changing_radius() {
        let mut circle = Circle::new(Point::new([0.0, 0.0]), 5.0);
        // Expand radius to 15 (delta_r = 10)
        circle.set_radius(15.0);

        let aabb = circle.aabb();
        // Min should be 0 - 15 = -15
        assert_relative_eq!(aabb.min.coords[0], -15.0);
        assert_relative_eq!(aabb.min.coords[1], -15.0);
        // Max should be 0 + 15 = 15
        assert_relative_eq!(aabb.max.coords[0], 15.0);
        assert_relative_eq!(aabb.max.coords[1], 15.0);
    }

    #[test]
    fn test_aabb_shrinking_radius() {
        let mut circle = Circle::new(Point::new([10.0, 10.0]), 10.0);
        // Shrink radius to 2 (delta_r = -8)
        circle.set_radius(2.0);

        let aabb = circle.aabb();
        // Center 10, Radius 2 -> Min: 8, Max: 12
        assert_relative_eq!(aabb.min.coords[0], 8.0);
        assert_relative_eq!(aabb.max.coords[1], 12.0);
    }
}
