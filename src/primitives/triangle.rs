use crate::{AABB, Bounded, Point, VectorMetricSquared};
use num_traits::Float;

/// A triangle in N-dimensional space defined by three vertices.
///
/// Vertices are stored as `[a, b, c]`. In 2D the triangle is a planar polygon;
/// in 3D it is a planar patch (three points define a plane). Area is computed
/// in any dimension using Lagrange's identity (see [`area`](Triangle::area)).
///
/// # Examples
///
/// ```
/// use apollonius::{Point, Triangle, Bounded};
///
/// let tri = Triangle::new([
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ]);
/// assert_eq!(tri.a().coords, [0.0, 0.0]);
/// assert_eq!(tri.area(), 0.5);
/// let aabb = tri.aabb();
/// assert_eq!(aabb.min.coords[0], 0.0);
/// assert_eq!(aabb.max.coords[0], 1.0);
/// ```
pub struct Triangle<T, const N: usize> {
    vertices: [Point<T, N>; 3],
}

impl<T, const N: usize> Triangle<T, N>
where
    T: Copy,
{
    /// Creates a new triangle from three vertices.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// let a = Point::new([0.0, 0.0]);
    /// let b = Point::new([4.0, 0.0]);
    /// let c = Point::new([0.0, 3.0]);
    /// let tri = Triangle::new([a, b, c]);
    /// assert_eq!(tri.b().coords[0], 4.0);
    /// ```
    pub fn new(vertices: [Point<T, N>; 3]) -> Self {
        Self { vertices }
    }

    /// First vertex.
    #[inline]
    pub fn a(&self) -> Point<T, N> {
        self.vertices[0]
    }

    /// Second vertex.
    #[inline]
    pub fn b(&self) -> Point<T, N> {
        self.vertices[1]
    }

    /// Third vertex.
    #[inline]
    pub fn c(&self) -> Point<T, N> {
        self.vertices[2]
    }

    /// Returns the three vertices as a tuple `(a, b, c)`.
    ///
    /// Convenient when several operations need the same vertices; avoids
    /// repeated destructuring of [`a`](Self::a), [`b`](Self::b), [`c`](Self::c).
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// let tri = Triangle::new([
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.0, 1.0]),
    /// ]);
    /// let (a, b, c) = tri.vertices();
    /// assert_eq!(a.coords[0], 0.0);
    /// assert_eq!(b.coords[0], 1.0);
    /// assert_eq!(c.coords[1], 1.0);
    /// ```
    #[inline]
    pub fn vertices(&self) -> (Point<T, N>, Point<T, N>, Point<T, N>) {
        (self.a(), self.b(), self.c())
    }
}

impl<T, const N: usize> From<(Point<T, N>, Point<T, N>, Point<T, N>)> for Triangle<T, N>
where
    T: Copy,
{
    /// Converts a 3-tuple of points into a triangle.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// let tri: Triangle<f64, 2> = Triangle::from((
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.0, 1.0]),
    /// ));
    /// assert_eq!(tri.c().coords[1], 1.0);
    /// ```
    fn from(value: (Point<T, N>, Point<T, N>, Point<T, N>)) -> Self {
        Self {
            vertices: [value.0, value.1, value.2],
        }
    }
}

impl<T, const N: usize> From<[Point<T, N>; 3]> for Triangle<T, N>
where
    T: Copy,
{
    /// Converts an array of three points into a triangle.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// let tri = Triangle::from([
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.0, 1.0]),
    /// ]);
    /// assert_eq!(tri.a().coords[0], 0.0);
    /// ```
    fn from(value: [Point<T, N>; 3]) -> Self {
        Self { vertices: value }
    }
}

impl<T, const N: usize> Triangle<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the centroid (geometric center) of the triangle: (a + b + c) / 3.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// let tri: Triangle<f64, 2> = Triangle::new([
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([2.0, 0.0]),
    ///     Point::new([0.0, 2.0]),
    /// ]);
    /// let c = tri.centroid();
    /// let two_thirds = 2.0 / 3.0;
    /// assert!((c.coords[0] - two_thirds).abs() < 1e-10);
    /// assert!((c.coords[1] - two_thirds).abs() < 1e-10);
    /// ```
    pub fn centroid(&self) -> Point<T, N> {
        let (a, b, c) = self.vertices();
        let mut coords = [T::zero(); N];

        let three = T::from(3.0).unwrap();

        for i in 0..N {
            coords[i] = (a.coords[i] + b.coords[i] + c.coords[i]) / three;
        }

        Point { coords }
    }

    /// Returns the area of the triangle in N-dimensional space.
    ///
    /// The area is computed using **Lagrange's identity**: for edge vectors
    /// `u = b - a` and `v = c - a`, the squared area of the parallelogram
    /// spanned by `u` and `v` is `|u|²|v|² - (u·v)²`. The triangle area is
    /// half of the square root of that expression, so it is well-defined for
    /// any dimension N (e.g. 2D planar area, 3D planar patch area).
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle};
    ///
    /// // Right triangle (0,0), (1,0), (0,1) has area 0.5
    /// let tri: Triangle<f64, 2> = Triangle::new([
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.0, 1.0]),
    /// ]);
    /// assert!((tri.area() - 0.5).abs() < 1e-10);
    /// ```
    pub fn area(&self) -> T {
        let two = T::one() + T::one();
        let (a, b, c) = self.vertices();
        let (u, v) = (b - a, c - a);
        let u_dot_v = u.dot(&v);
        let mag_u = u.magnitude_squared();
        let mag_v = v.magnitude_squared();
        (mag_u * mag_v - u_dot_v * u_dot_v).sqrt() / two
    }
}

impl<T, const N: usize> Bounded<T, N> for Triangle<T, N>
where
    T: Float,
{
    /// Returns the Axis-Aligned Bounding Box enclosing the three vertices.
    ///
    /// Min and max are computed per axis using [`std::array::from_fn`] over the
    /// three vertex coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Triangle, Bounded};
    ///
    /// let tri = Triangle::new([
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([4.0, 0.0]),
    ///     Point::new([0.0, 3.0]),
    /// ]);
    /// let aabb = tri.aabb();
    /// assert_eq!(aabb.min.coords, [0.0, 0.0]);
    /// assert_eq!(aabb.max.coords, [4.0, 3.0]);
    /// ```
    fn aabb(&self) -> AABB<T, N> {
        let (a, b, c) = self.vertices();
        let min_coords = std::array::from_fn(|i| a.coords[i].min(b.coords[i]).min(c.coords[i]));
        let max_coords = std::array::from_fn(|i| a.coords[i].max(b.coords[i]).max(c.coords[i]));

        AABB {
            min: Point { coords: min_coords },
            max: Point { coords: max_coords },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bounded;
    use approx::assert_relative_eq;

    #[test]
    fn test_triangle_construction_and_accessors() {
        let a = Point::new([0.0, 0.0]);
        let b = Point::new([1.0, 0.0]);
        let c = Point::new([0.0, 1.0]);

        let tri = Triangle::new([a, b, c]);
        assert_eq!(tri.a(), a);
        assert_eq!(tri.b(), b);
        assert_eq!(tri.c(), c);

        let from_tuple: Triangle<f64, 2> = Triangle::from((a, b, c));
        assert_eq!(from_tuple.a(), a);
        assert_eq!(from_tuple.b(), b);
        assert_eq!(from_tuple.c(), c);

        let from_arr: Triangle<f64, 2> = Triangle::from([a, b, c]);
        assert_eq!(from_arr.a(), a);
    }

    #[test]
    fn test_triangle_aabb_2d() {
        // Triangle with vertices (0,0), (4,0), (0,3). AABB: min=(0,0), max=(4,3).
        let tri = Triangle::new([
            Point::new([0.0, 0.0]),
            Point::new([4.0, 0.0]),
            Point::new([0.0, 3.0]),
        ]);
        let aabb = tri.aabb();
        assert_eq!(aabb.min.coords[0], 0.0);
        assert_eq!(aabb.min.coords[1], 0.0);
        assert_eq!(aabb.max.coords[0], 4.0);
        assert_eq!(aabb.max.coords[1], 3.0);
    }

    #[test]
    fn test_triangle_aabb_3d() {
        // Triangle in 3D: (0,0,0), (2,0,0), (0,2,0). AABB: min=(0,0,0), max=(2,2,0).
        let tri = Triangle::new([
            Point::new([0.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([0.0, 2.0, 0.0]),
        ]);
        let aabb = tri.aabb();
        assert_eq!(aabb.min.coords, [0.0, 0.0, 0.0]);
        assert_eq!(aabb.max.coords, [2.0, 2.0, 0.0]);
    }

    #[test]
    fn test_triangle_centroid_2d() {
        // Right triangle (0,0), (2,0), (0,2). Centroid = (2/3, 2/3).
        let tri = Triangle::new([
            Point::new([0.0, 0.0]),
            Point::new([2.0, 0.0]),
            Point::new([0.0, 2.0]),
        ]);
        let c = tri.centroid();
        let two_thirds = 2.0 / 3.0;
        assert_relative_eq!(c.coords[0], two_thirds);
        assert_relative_eq!(c.coords[1], two_thirds);
    }

    #[test]
    fn test_triangle_centroid_3d() {
        // Triangle (1,0,0), (0,1,0), (0,0,1). Centroid = (1/3, 1/3, 1/3).
        let tri = Triangle::new([
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ]);
        let c = tri.centroid();
        let third = 1.0 / 3.0;
        assert_relative_eq!(c.coords[0], third);
        assert_relative_eq!(c.coords[1], third);
        assert_relative_eq!(c.coords[2], third);
    }

    #[test]
    fn test_triangle_area_2d() {
        // Right triangle (0,0), (1,0), (0,1). Area = 0.5.
        let tri = Triangle::new([
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ]);
        assert_relative_eq!(tri.area(), 0.5);
    }

    #[test]
    fn test_triangle_area_2d_general() {
        // Triangle (0,0), (4,0), (0,3). Area = 6.0 (base 4, height 3).
        let tri = Triangle::new([
            Point::new([0.0, 0.0]),
            Point::new([4.0, 0.0]),
            Point::new([0.0, 3.0]),
        ]);
        assert_relative_eq!(tri.area(), 6.0);
    }

    #[test]
    fn test_triangle_area_3d() {
        // Triangle in xy-plane: (0,0,0), (2,0,0), (0,2,0). Area = 2.0.
        let tri = Triangle::new([
            Point::new([0.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([0.0, 2.0, 0.0]),
        ]);
        assert_relative_eq!(tri.area(), 2.0);
    }
}
