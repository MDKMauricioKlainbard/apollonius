use crate::{
    EuclideanVector, FloatSign, Hypersphere, IntersectionResult, Line, Point, Segment,
    SpatialRelation, Vector, VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// An N-dimensional hyperplane defined by a point and a normal vector.
///
/// A hyperplane represents the set of all points P such that the dot product
/// of (P - origin) and the normal vector is zero. In 3D space, this defines
/// an infinite flat surface.
///
/// # Examples
/// ```
/// use apollonius::{Point, Vector, Hyperplane};
///
/// let origin = Point::new([0.0, 0.0, 0.0]);
/// let normal = Vector::new([0.0, 1.0, 0.0]); // Y-axis normal (horizontal plane)
/// let plane = Hyperplane::new(origin, normal);
///
/// assert_eq!(plane.normal().coords()[1], 1.0);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::Deserialize<'de>")))]
pub struct Hyperplane<T, const N: usize> {
    origin: Point<T, N>,
    normal: Vector<T, N>, // Always kept normalized
}

impl<T, const N: usize> Hyperplane<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Creates a new hyperplane.
    ///
    /// If the provided normal vector is null (zero length), the hyperplane
    /// defaults to a canonical unit vector [1, 0, ..., 0] to maintain
    /// geometric validity.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hyperplane};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 5.0]));
    /// assert_eq!(plane.normal().coords()[1], 1.0); // Normalized automatically
    /// ```
    pub fn new(origin: Point<T, N>, normal: Vector<T, N>) -> Self {
        Self {
            origin,
            normal: Self::normal_or_default(&normal),
        }
    }

    /// Returns the point of origin that lies on the hyperplane.
    #[inline]
    pub fn origin(&self) -> Point<T, N> {
        self.origin
    }

    /// Returns the normalized vector perpendicular to the hyperplane.
    #[inline]
    pub fn normal(&self) -> Vector<T, N> {
        self.normal
    }

    /// Updates the hyperplane's origin point.
    #[inline]
    pub fn set_origin(&mut self, p: &Point<T, N>) {
        self.origin = *p;
    }

    /// Updates the normal vector and ensures it remains a unit vector.
    ///
    /// If the provided vector is null, it defaults to the canonical X-axis.
    #[inline]
    pub fn set_normal(&mut self, v: &Vector<T, N>) {
        self.normal = Self::normal_or_default(v)
    }

    /// Returns a mutable reference to the origin point.
    #[inline]
    pub fn origin_mut(&mut self) -> &mut Point<T, N> {
        &mut self.origin
    }

    /// Returns a mutable reference to the normal vector.
    ///
    /// Note: mutating the normal does not re-normalize it; use [`set_normal`](Self::set_normal) to keep it unit length.
    #[inline]
    pub fn normal_mut(&mut self) -> &mut Vector<T, N> {
        &mut self.normal
    }

    /// Internal helper to ensure the normal vector is normalized.
    fn normal_or_default(v: &Vector<T, N>) -> Vector<T, N> {
        v.normalize().unwrap_or_else(|| {
            Vector::new(std::array::from_fn(|i| if i == 0 { T::one() } else { T::zero() }))
        })
    }

    /// Calculates the signed distance from a point to the hyperplane.
    ///
    /// The sign indicates which side of the plane the point is on relative
    /// to the normal vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hyperplane};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]));
    /// let p = Point::new([0.0, 5.0]);
    ///
    /// assert_eq!(plane.signed_distance(&p), 5.0);
    /// ```
    #[inline]
    pub fn signed_distance(&self, p: &Point<T, N>) -> T {
        (*p - self.origin).dot(&self.normal)
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Hyperplane<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point `p` onto the hyperplane.
    ///
    /// This returns the closest point on the plane to the given point.
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let d = self.signed_distance(p);
        *p - self.normal * d
    }

    /// Checks if a point lies on the hyperplane within the engine's tolerance.
    fn contains(&self, p: &Point<T, N>) -> bool {
        match classify_to_zero(self.signed_distance(p).abs(), None) {
            FloatSign::Zero => true,
            _ => false,
        }
    }

    /// Defines "inside" as the half-space opposite to the normal vector direction.
    ///
    /// Points with a negative or zero signed distance are considered inside.
    fn is_inside(&self, p: &Point<T, N>) -> bool {
        self.signed_distance(p) <= T::zero()
    }
}

impl<T, const N: usize> Hyperplane<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Intersects a line segment with the hyperplane.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Tangent`, `Secant`, or `HalfSpacePenetration`):
    /// - [`Single(p)`](crate::IntersectionResult::Single): The segment crosses the plane at point `p`.
    /// - [`None`](crate::IntersectionResult::None): The segment is parallel (and not on the plane) or does not reach it.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The entire segment lies on the plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hyperplane, Segment, IntersectionResult};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]));
    /// let segment = Segment::new(Point::new([0.0, -1.0]), Point::new([0.0, 1.0]));
    ///
    /// if let IntersectionResult::Single(p) = plane.intersect_segment(&segment) {
    ///     assert_eq!(p.coords()[1], 0.0);
    /// }
    /// ```
    pub fn intersect_segment(&self, segment: &Segment<T, N>) -> IntersectionResult<T, N> {
        let d_dot_n = segment.delta().dot(&self.normal);

        // Check if the segment is parallel to the plane
        if d_dot_n.abs() < T::epsilon() {
            return if self.contains(&segment.start()) {
                IntersectionResult::Collinear
            } else {
                IntersectionResult::None
            };
        }

        // Parametric intersection: t = (origin - start) · normal / (delta · normal)
        let t = (self.origin - segment.start()).dot(&self.normal) / d_dot_n;

        if t >= -T::epsilon() && t <= T::one() + T::epsilon() {
            IntersectionResult::Single(segment.at(t))
        } else {
            IntersectionResult::None
        }
    }

    /// Calculates the intersection between this hyperplane and a line.
    ///
    /// This method delegates to [`Line::intersect_hyperplane`], providing a symmetrical API.
    ///
    /// # Returns
    /// This method returns only the following variants (never `Tangent`, `Secant`, or `HalfSpacePenetration`):
    /// - [`None`](crate::IntersectionResult::None): The line is parallel to the plane and not on it.
    /// - [`Collinear`](crate::IntersectionResult::Collinear): The line lies entirely on the hyperplane.
    /// - [`Single(p)`](crate::IntersectionResult::Single): The line pierces the plane at exactly one point `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, Vector, Line, Hyperplane, IntersectionResult};
    ///
    /// let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
    /// let line = Line::new(Point::new([0.0, 0.0, 10.0]), Vector::new([0.0, 0.0, -1.0]));
    ///
    /// // Intersecting from the plane's perspective
    /// if let IntersectionResult::Single(p) = plane.intersect_line(&line) {
    ///     assert_eq!(p, Point::new([0.0, 0.0, 0.0]));
    /// }
    /// ```
    #[inline]
    pub fn intersect_line(&self, line: &Line<T, N>) -> IntersectionResult<T, N> {
        line.intersect_hyperplane(self)
    }

    /// Computes the intersection of this hyperplane with a hypersphere.
    ///
    /// This is the symmetric counterpart to [`Hypersphere::intersect_hyperplane`].
    /// It treats the hyperplane as a boundary where the direction of the normal
    /// vector defines the "outside" (positive half-space) and the opposite direction
    /// defines the "inside" (negative half-space).
    ///
    /// # Delegation and Symmetry
    ///
    /// This method delegates the calculation to the sphere's implementation to ensure
    /// mathematical consistency. It is provided here to allow for a more natural
    /// API when the hyperplane is the primary object of the query.
    ///
    /// # Returns
    ///
    /// This method returns only the following variants (never `Single`, `Secant`, or `Collinear`):
    /// - **[`None`](crate::IntersectionResult::None)**: The sphere is located entirely
    ///   in the positive half-space (the side the normal points towards).
    /// - **[`Tangent(p)`](crate::IntersectionResult::Tangent)**: The sphere's surface
    ///   grazes the hyperplane at exactly one point `p`.
    /// - **[`HalfSpacePenetration(depth)`](crate::IntersectionResult::HalfSpacePenetration)**:
    ///   The sphere has crossed the plane and is partially or fully submerged in the
    ///   negative half-space. `depth` is the distance from the plane to the furthest
    ///   point of the sphere inside the half-space.
    ///
    ///
    ///
    /// # Examples
    ///
    /// Detecting a sphere partially submerged in a 3D floor:
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane, IntersectionResult};
    ///
    /// // A floor plane at y = 0 with normal pointing up (y+)
    /// let floor = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));
    ///
    /// // A sphere with radius 10, centered at y = 4 (submerged by 6 units)
    /// let sphere = Hypersphere::new(Point::new([0.0, 4.0, 0.0]), 10.0);
    ///
    /// match floor.intersect_hypersphere(&sphere) {
    ///     IntersectionResult::HalfSpacePenetration(depth) => {
    ///         // depth = radius - signed_distance = 10 - 4 = 6
    ///         assert!(((depth - 6.0) as f64).abs() < 1e-6);
    ///     }
    ///     _ => panic!("Expected penetration for submerged sphere"),
    /// }
    /// ```
    ///
    /// Grazing contact (Tangent):
    ///
    /// ```
    /// use apollonius::{Point, Vector, Hypersphere, Hyperplane, IntersectionResult};
    ///
    /// let wall = Hyperplane::new(Point::new([10.0, 0.0, 0.0]), Vector::new([1.0, 0.0, 0.0]));
    /// let sphere = Hypersphere::new(Point::new([5.0, 0.0, 0.0]), 5.0);
    ///
    /// if let IntersectionResult::Tangent(p) = wall.intersect_hypersphere(&sphere) {
    ///     assert_eq!(p.coords()[0], 10.0);
    /// } else {
    ///     panic!("Expected Tangent contact at x = 10");
    /// }
    /// ```
    #[inline]
    pub fn intersect_hypersphere(&self, sphere: &Hypersphere<T, N>) -> IntersectionResult<T, N> {
        sphere.intersect_hyperplane(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, Segment, Vector};
    use approx::assert_relative_eq;

    #[test]
    fn test_hyperplane_closest_point() {
        // XY Plane in 3D (normal is Z-axis)
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 1.0]));
        let p = Point::new([10.0, 10.0, 5.0]);
        let projected = plane.closest_point(&p);

        assert_relative_eq!(projected.coords()[2], 0.0);
        assert_relative_eq!(projected.coords()[0], 10.0);
    }

    #[test]
    fn test_segment_piercing_plane() {
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));
        // Segment from y=-5 to y=5, should hit at y=0
        let seg = Segment::new(Point::new([0.0, -5.0, 0.0]), Point::new([0.0, 5.0, 0.0]));

        if let IntersectionResult::Single(p) = plane.intersect_segment(&seg) {
            assert_relative_eq!(p.coords()[1], 0.0);
        } else {
            panic!("Expected single intersection point");
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hyperplane_serialization_roundtrip() {
        use serde_json;

        let plane = Hyperplane::new(
            Point::new([0.0, 0.0, 0.0]),
            Vector::new([0.0, 0.0, 1.0]),
        );
        let json = serde_json::to_string(&plane).unwrap();
        let restored: Hyperplane<f64, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(plane.origin(), restored.origin());
        assert_eq!(plane.normal().coords, restored.normal().coords);
    }
}
