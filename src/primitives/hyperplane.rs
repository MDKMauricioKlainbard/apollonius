use crate::{
    EuclideanVector, FloatSign, IntersectionResult, Point, Segment, SpatialRelation, Vector,
    VectorMetricSquared, classify_to_zero,
};
use num_traits::Float;

/// An N-dimensional hyperplane defined by a point and a normal vector.
///
/// A hyperplane represents the set of all points P such that the dot product 
/// of (P - origin) and the normal vector is zero. In 3D space, this defines 
/// an infinite flat surface.
///
/// # Examples
///
/// ```
/// use apollonius::{Point, Vector, Hyperplane};
///
/// let origin = Point::new([0.0, 0.0, 0.0]);
/// let normal = Vector::new([0.0, 1.0, 0.0]); // Y-axis normal (horizontal plane)
/// let plane = Hyperplane::new(origin, normal);
///
/// assert_eq!(plane.normal().coords[1], 1.0);
/// ```
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
    /// assert_eq!(plane.normal().coords[1], 1.0); // Normalized automatically
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

    /// Internal helper to ensure the normal vector is normalized.
    fn normal_or_default(v: &Vector<T, N>) -> Vector<T, N> {
        v.normalize().unwrap_or_else(|| {
            let mut default = Vector {
                coords: [T::zero(); N],
            };
            default.coords[0] = T::one();
            default
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
    /// - `IntersectionResult::Single(p)`: The segment crosses the plane at point `p`.
    /// - `IntersectionResult::None`: The segment is parallel (and not on the plane) or does not reach it.
    /// - `IntersectionResult::Collinear`: The entire segment lies on the plane.
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
    ///     assert_eq!(p.coords[1], 0.0);
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

        assert_relative_eq!(projected.coords[2], 0.0);
        assert_relative_eq!(projected.coords[0], 10.0);
    }

    #[test]
    fn test_segment_piercing_plane() {
        let plane = Hyperplane::new(Point::new([0.0, 0.0, 0.0]), Vector::new([0.0, 1.0, 0.0]));
        // Segment from y=-5 to y=5, should hit at y=0
        let seg = Segment::new(Point::new([0.0, -5.0, 0.0]), Point::new([0.0, 5.0, 0.0]));

        if let IntersectionResult::Single(p) = plane.intersect_segment(&seg) {
            assert_relative_eq!(p.coords[1], 0.0);
        } else {
            panic!("Expected single intersection point");
        }
    }
}
