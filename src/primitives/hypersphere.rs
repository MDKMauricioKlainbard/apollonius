use crate::{EuclideanVector, Point, SpatialRelation, Vector, VectorMetricSquared};
use num_traits::Float;

/// An N-dimensional hypersphere defined by a center point and a radius.
///
/// In 2D, this represents a circle. In 3D, a sphere. In higher dimensions,
/// it represents the set of all points at a fixed distance from a central point.
pub struct Hypersphere<T, const N: usize> {
    /// The geometric center of the hypersphere.
    pub center: Point<T, N>,
    /// The distance from the center to the surface.
    pub radius: T,
}

/// A 2-dimensional hypersphere.
pub type Circle<T> = Hypersphere<T, 2>;
/// A 3-dimensional hypersphere.
pub type Sphere<T> = Hypersphere<T, 3>;

impl<T, const N: usize> Hypersphere<T, N> {
    /// Creates a new hypersphere with the given center and radius.
    ///
    /// # Examples
    /// ```
    /// use apollonius::Point;
    /// use apollonius::primitives::hypersphere::Circle;
    ///
    /// let center = Point::new([0.0, 0.0]);
    /// let circle = Circle::new(center, 5.0);
    /// ```
    pub fn new(center: Point<T, N>, radius: T) -> Self {
        Self { center, radius }
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
    /// This method uses `T::epsilon()` to account for floating-point inaccuracies.
    /// Points strictly inside or outside the surface will return `false`.
    fn contains(&self, p: &Point<T, N>) -> bool {
        let dist_sq = (*p - self.center).magnitude_squared();
        let radius_sq = self.radius * self.radius;
        (dist_sq - radius_sq).abs() <= T::epsilon()
    }

    /// Checks if a point `p` is contained within the hypersphere's volume.
    ///
    /// Returns `true` if the distance from the center to `p` is less than or
    /// equal to the radius (including the boundary).
    fn is_inside(&self, p: &Point<T, N>) -> bool {
        let dist_sq = (*p - self.center).magnitude_squared();
        dist_sq <= self.radius * self.radius + T::epsilon()
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
}
