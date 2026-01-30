use crate::{
    AABB, Bounded, EuclideanVector, FloatSign, Hyperplane, IntersectionResult, Line, Point,
    Segment, SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// An N-dimensional hypersphere defined by a center point and a radius.
///
/// In 2D space, this represents a circle. In 3D, a sphere. In higher dimensions,
/// it represents the set of all points at a fixed distance (radius) from a central point.
///
/// This structure maintains a cached `AABB` to optimize spatial queries and
/// broad-phase collision detection.
///
/// # Examples
///
/// ```
/// use apollonius::{Point, Hypersphere};
///
/// // Create a 2D circle at (0, 0) with radius 1.0
/// let center = Point::new([0.0, 0.0]);
/// let circle = Hypersphere::new(center, 1.0);
///
/// assert_eq!(circle.radius(), 1.0);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Hypersphere<T, const N: usize> {
    center: Point<T, N>,
    radius: T,
    cached_aabb: AABB<T, N>,
}

/// A 2-dimensional hypersphere (Circle).
pub type Circle<T> = Hypersphere<T, 2>;
/// A 3-dimensional hypersphere (Sphere).
pub type Sphere<T> = Hypersphere<T, 3>;

impl<T, const N: usize> Hypersphere<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new hypersphere and pre-calculates its bounding box.
    ///
    /// # Arguments
    /// * `center` - The central point of the hypersphere.
    /// * `radius` - The distance from the center to the surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Hypersphere, Bounded};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 0.0]), 5.0);
    /// let aabb = sphere.aabb();
    ///
    /// assert_eq!(aabb.min.coords[0], -5.0);
    /// assert_eq!(aabb.max.coords[0], 5.0);
    /// ```
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

    /// Returns the hypersphere's center point.
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
    ///
    /// This is more efficient than recomputing the AABB from scratch as it only
    /// applies the displacement vector to the bounding box boundaries.
    pub fn set_center(&mut self, new_center: Point<T, N>) {
        let offset = new_center - self.center;
        self.center = new_center;

        self.cached_aabb.min = self.cached_aabb.min + offset;
        self.cached_aabb.max = self.cached_aabb.max + offset;
    }

    /// Updates the radius and expands or contracts the cached AABB radially in O(N).
    ///
    /// The update is performed incrementally by adjusting the AABB bounds
    /// based on the difference between the new and old radius.
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
    /// If `p` is exactly at the center of the hypersphere, the projection direction
    /// is mathematically undefined. In this specific case, the method projects the
    /// point along the positive X-axis (canonical direction).
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Hypersphere, SpatialRelation};
    ///
    /// let circle = Hypersphere::new(Point::new([0.0, 0.0]), 2.0);
    /// let p = Point::new([4.0, 0.0]);
    ///
    /// let closest = circle.closest_point(&p);
    /// assert_eq!(closest.coords[0], 2.0);
    /// assert_eq!(closest.coords[1], 0.0);
    /// ```
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let direction = (*p - self.center).normalize().unwrap_or_else(|| {
            let mut direction = Vector::new([T::zero(); N]);
            direction.coords[0] = T::one();
            direction
        });

        self.center + direction * self.radius
    }

    /// Checks if a point `p` lies exactly on the hypersphere's surface.
    ///
    /// Accounts for floating-point inaccuracies using the engine's internal
    /// epsilon tolerance via `classify_to_zero`.
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
    /// Returns `true` if the point is inside or exactly on the boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Hypersphere, SpatialRelation};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 0.0]), 1.0);
    ///
    /// assert!(sphere.is_inside(&Point::new([0.5, 0.0, 0.0])));
    /// assert!(!sphere.is_inside(&Point::new([1.5, 0.0, 0.0])));
    /// ```
    fn is_inside(&self, p: &Point<T, N>) -> bool {
        let dist_sq = (*p - self.center).magnitude_squared();
        let radius_sq = self.radius * self.radius;

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
    /// the line's specific intersection logic.
    #[inline]
    pub fn intersect_line(&self, line: &Line<T, N>) -> IntersectionResult<T, N> {
        line.intersect_hypersphere(self)
    }

    /// Computes the intersection(s) between this hypersphere and a finite line segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Segment, Hypersphere, IntersectionResult};
    ///
    /// let circle = Hypersphere::new(Point::new([0.0, 0.0]), 1.0);
    /// let seg = Segment::new(Point::new([-2.0, 0.0]), Point::new([2.0, 0.0]));
    ///
    /// let result = circle.intersect_segment(&seg);
    /// if let IntersectionResult::Secant(p1, p2) = result {
    ///     assert_eq!(p1.coords[0], -1.0);
    ///     assert_eq!(p2.coords[0], 1.0);
    /// }
    /// ```
    #[inline]
    pub fn intersect_segment(&self, segment: &Segment<T, N>) -> IntersectionResult<T, N> {
        segment.intersect_hypersphere(self)
    }

    /// Computes the intersection of this hypersphere with a hyperplane (plane).
    ///
    /// The result describes whether the sphere lies entirely on one side of the plane,
    /// is tangent to it (touching at exactly one point), or penetrates the negative
    /// half-space (the side opposite to the plane's normal).
    ///
    /// # Return semantics: `Tangent` vs `Single`
    ///
    /// When the sphere touches the plane at exactly one point (tangent contact),
    /// this method returns **[`IntersectionResult::Tangent`](crate::IntersectionResult::Tangent)(p)**,
    /// where `p` is that contact point (the orthogonal projection of the sphere's center
    /// onto the plane). It **never** returns [`Single`](crate::IntersectionResult::Single)
    /// for this query: `Single` is reserved for other primitives (e.g. segment crossing
    /// a boundary). Use `Tangent` to detect grazing contact between sphere and plane.
    ///
    /// # Returns
    ///
    /// - **[`None`](crate::IntersectionResult::None)**: The sphere lies entirely in the
    ///   positive half-space (on the side of the plane's normal). No contact with the plane.
    /// - **[`Tangent(p)`](crate::IntersectionResult::Tangent)**: The sphere is tangent to the
    ///   plane at point `p` (exactly one point of contact). `p` is the plane's closest point
    ///   to the sphere center.
    /// - **[`HalfSpacePenetration(depth)`](crate::IntersectionResult::HalfSpacePenetration)**: The
    ///   sphere crosses into or lies inside the negative half-space. `depth` is the penetration
    ///   depth along the plane normal (distance from the plane to the furthest point of the
    ///   sphere inside the half-space).
    ///
    /// # Examples
    ///
    /// Tangent contact (sphere touching the plane at one point):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane, IntersectionResult};
    ///
    /// // Sphere centered at (0, 0, 5) with radius 5 — touches plane z = 0 at (0, 0, 0)
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 5.0]), 5.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// match sphere.intersect_hyperplane(&plane) {
    ///     IntersectionResult::Tangent(p) => {
    ///         assert_eq!(p.coords[0], 0.0);
    ///         assert_eq!(p.coords[1], 0.0);
    ///         assert_eq!(p.coords[2], 0.0);
    ///     }
    ///     _ => panic!("expected Tangent for tangent sphere-plane contact"),
    /// }
    /// ```
    ///
    /// No intersection (sphere entirely above the plane):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane, IntersectionResult};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 10.0]), 2.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// assert!(matches!(sphere.intersect_hyperplane(&plane), IntersectionResult::None));
    /// ```
    ///
    /// Penetration (sphere crosses the plane):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane, IntersectionResult};
    ///
    /// // Sphere center at z = 2, radius 5 → penetrates plane z = 0 by depth 3
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 2.0]), 5.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// if let IntersectionResult::HalfSpacePenetration(depth) = sphere.intersect_hyperplane(&plane) {
    ///     assert!(((depth - 3.0) as f64).abs() < 1e-6);
    /// } else {
    ///     panic!("expected HalfSpacePenetration");
    /// }
    /// ```
    pub fn intersect_hyperplane(&self, plane: &Hyperplane<T, N>) -> IntersectionResult<T, N> {
        let d = plane.signed_distance(&self.center);
        let r = self.radius;

        // Check for tangency first
        if let FloatSign::Zero = classify_to_zero(d.abs() - r, None) {
            return IntersectionResult::Tangent(plane.closest_point(&self.center));
        }

        match classify_to_zero(d - r, None) {
            // Sphere is completely in the positive half-space (outside)
            FloatSign::Positive => IntersectionResult::None,
            // Sphere is at least partially in the negative half-space (inside/overlapping)
            _ => {
                // Penetration depth is the distance from the furthest point
                // inside the half-space to the plane boundary.
                IntersectionResult::HalfSpacePenetration(r - d)
            }
        }
    }

    /// Returns the fraction of the hypersphere's volume that lies in the plane's negative half-space.
    ///
    /// "Submerged" means the part of the sphere on the side **opposite** to the plane's normal
    /// (the negative signed-distance side). The ratio is in **[0.0, 1.0]**:
    /// - **0.0**: the sphere lies entirely in the positive half-space (none submerged).
    /// - **0.5**: the plane passes through the center (half the volume on each side).
    /// - **1.0**: the sphere lies entirely in the negative half-space (fully submerged).
    ///
    /// In 2D this is the area ratio of the circular segment; in 3D the volume ratio of the
    /// spherical cap. For N > 3 a linear height-based approximation is used.
    ///
    /// # Examples
    ///
    /// Center on the plane (half submerged):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 0.0]), 10.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// let ratio = sphere.submerged_ratio(&plane);
    /// assert!(((ratio - 0.5) as f64).abs() < 1e-6);
    /// ```
    ///
    /// Fully submerged (sphere entirely below the plane):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, -20.0]), 10.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// let ratio = sphere.submerged_ratio(&plane);
    /// assert!(((ratio - 1.0) as f64).abs() < 1e-6);
    /// ```
    ///
    /// Not submerged (sphere entirely above the plane):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane};
    ///
    /// let sphere = Hypersphere::new(Point::new([0.0, 0.0, 20.0]), 10.0);
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    ///
    /// let ratio = sphere.submerged_ratio(&plane);
    /// assert!(((ratio - 0.0) as f64).abs() < 1e-6);
    /// ```
    pub fn submerged_ratio(&self, plane: &Hyperplane<T, N>) -> T {
        let d = plane.signed_distance(&self.center);
        let r = self.radius;

        // Clamp signed distance to [-r, r] to handle fully submerged or fully outside cases
        let d_clamped = if let FloatSign::Positive = classify_to_zero(d - r, None) {
            r
        } else if let FloatSign::Negative = classify_to_zero(d + r, None) {
            -r
        } else {
            d
        };

        Self::compute_ratio(d_clamped, r)
    }
}

/// Internal trait to encapsulate N-dimensional volume ratio logic.
trait SubmergedVolumeScale<T> {
    fn compute_ratio(d: T, r: T) -> T;
}

impl<T: Float, const N: usize> SubmergedVolumeScale<T> for Hypersphere<T, N> {
    fn compute_ratio(d: T, r: T) -> T {
        let x = d / r; // Normalized distance to center [-1, 1]

        match N {
            2 => {
                // Area of a circular segment ratio (2D)
                let pi = T::from(std::f64::consts::PI).unwrap();
                (x.acos() - x * (T::one() - x * x).sqrt()) / pi
            }
            3 => {
                // Volume of a spherical cap ratio (3D)
                let three = T::from(3.0).unwrap();
                let two = T::from(2.0).unwrap();
                let four = T::from(4.0).unwrap();
                (x.powi(3) - three * x + two) / four
            }
            _ => {
                // General N-dimensional linear approximation.
                // As N increases, the volume concentrates near the equator.
                // This linear height approximation serves as a base fallback.
                (r - d) / (T::from(2.0).unwrap() * r)
            }
        }
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

    #[test]
    fn test_half_space_penetration_depth() {
        // Sphere of radius 5 at z=2. Plane at z=0 (normal [0,0,1])
        let sphere = Hypersphere::new(Point::new([0.0, 0.0, 2.0]), 5.0);
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));

        // Signed distance is 2.0. Depth = 5.0 - 2.0 = 3.0
        if let IntersectionResult::HalfSpacePenetration(depth) = sphere.intersect_hyperplane(&plane)
        {
            assert!((depth - 3.0).abs() < 1e-6);
        } else {
            panic!("Expected HalfSpacePenetration");
        }
    }

    #[test]
    fn test_submerged_ratio_3d() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0, 0.0]), 10.0);
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));

        // Case 1: Center exactly on plane -> 50% submerged
        let ratio_half = sphere.submerged_ratio(&plane);
        assert!((ratio_half - 0.5).abs() < 1e-6);

        // Case 2: Fully submerged (center at z = -20)
        let deep_sphere = Hypersphere::new(Point::new([0.0, 0.0, -20.0]), 10.0);
        assert!((deep_sphere.submerged_ratio(&plane) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tangency_case() {
        let sphere = Hypersphere::new(Point::new([0.0, 0.0, 5.0]), 5.0);
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));

        match sphere.intersect_hyperplane(&plane) {
            IntersectionResult::Tangent(p) => {
                assert_eq!(p, Point::new([0.0, 0.0, 0.0]));
            }
            _ => panic!("Expected Tangent point for tangency"),
        }
    }
}
