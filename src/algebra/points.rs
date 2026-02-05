use num_traits::Float;
use std::ops::{Add, Sub};

use crate::{Vector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Represents a location in an N-dimensional affine space.
///
/// Unlike a [`Vector`], a [`Point`] represents a fixed position in space and
/// does not have direction or magnitude. Operations between points and vectors
/// follow the rules of affine geometry.
///
/// # Examples
///
/// ```
/// use apollonius::Point;
///
/// let p = Point::new([1.0, 2.0, 3.0]);
/// assert_eq!(p.coords(), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<T, const N: usize> {
    /// The coordinate values along each of the N axes.
    coords: [T; N],
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for Point<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.as_slice().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Point<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let coords_vec = Vec::<T>::deserialize(deserializer)?;
        let coords: [T; N] = coords_vec.try_into().map_err(|_| {
            serde::de::Error::custom(format!("Point dimension mismatch: expected {}", N))
        })?;
        Ok(Self { coords })
    }
}

/// A 2D point specialization.
pub type Point2D<T> = Point<T, 2>;

/// A 3D point specialization.
pub type Point3D<T> = Point<T, 3>;

/// Trait for types that can calculate the squared distance between each other.
///
/// Using squared distance is often preferred for performance-critical
/// comparisons to avoid the computational cost of the square root operation.
///
/// # Example
///
/// ```
/// use apollonius::{Point, MetricSquared};
///
/// let a = Point::new([0.0, 0.0]);
/// let b = Point::new([3.0, 4.0]);
/// assert_eq!(a.distance_squared(&b), 25.0);
/// ```
pub trait MetricSquared<T> {
    /// Calculates the squared Euclidean distance between `self` and `other`.
    fn distance_squared(&self, other: &Self) -> T;
}

/// Trait for types that can calculate the actual Euclidean distance.
///
/// # Example
///
/// ```
/// use apollonius::{Point, EuclideanMetric};
///
/// let a = Point::new([0.0, 0.0]);
/// let b = Point::new([3.0, 4.0]);
/// assert_eq!(a.distance(&b), 5.0);
/// ```
pub trait EuclideanMetric<T>: MetricSquared<T> {
    /// Calculates the Euclidean distance between `self` and `other`.
    fn distance(&self, other: &Self) -> T;
}

impl<T, const N: usize> Point<T, N>
where
    T: Float,
{
    /// Creates a new point from a fixed-size array of coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Point;
    ///
    /// let p = Point::new([1.0, 2.0, 3.0]);
    /// assert_eq!(p.coords(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn new(coords: [T; N]) -> Self {
        Self { coords }
    }

    /// Returns a reference to the coordinate array.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Point;
    ///
    /// let p = Point::new([1.0, 2.0]);
    /// assert_eq!(p.coords()[0], 1.0);
    /// assert_eq!(p.coords()[1], 2.0);
    /// ```
    #[inline]
    pub fn coords(&self) -> &[T; N] {
        &self.coords
    }

    /// Sets the coordinate array.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Point;
    ///
    /// let mut p = Point::new([0.0, 0.0]);
    /// p.set_coords([3.0, 4.0]);
    /// assert_eq!(p.coords(), &[3.0, 4.0]);
    /// ```
    #[inline]
    pub fn set_coords(&mut self, coords: [T; N]) {
        self.coords = coords;
    }

    /// Returns a mutable reference to the coordinate array.
    ///
    /// Useful in simulations for in-place updates (e.g. `p.coords_mut()[0] += dt * v.coords()[0]`)
    /// without copying.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Point;
    ///
    /// let mut p = Point::new([1.0, 2.0]);
    /// p.coords_mut()[0] += 1.0;
    /// assert_eq!(p.coords(), &[2.0, 2.0]);
    /// ```
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [T; N] {
        &mut self.coords
    }
}

impl<T> From<(T, T)> for Point2D<T> {
    /// Converts a 2-element tuple into a [`Point2D`].
    ///
    /// # Example
    /// ```
    /// use apollonius::Point2D;
    ///
    /// let p = Point2D::from((1.0, 2.0));
    /// assert_eq!(p.coords(), &[1.0, 2.0]);
    /// ```
    #[inline]
    fn from(tuple: (T, T)) -> Self {
        Self {
            coords: [tuple.0, tuple.1],
        }
    }
}


impl<T> From<(T, T, T)> for Point3D<T> {
    /// Converts a 3-element tuple into a [`Point3D`].
    ///
    /// # Example
    /// ```
    /// use apollonius::Point3D;
    ///
    /// let p = Point3D::from((1.0, 2.0, 3.0));
    /// assert_eq!(p.coords(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(tuple: (T, T, T)) -> Self {
        Self {
            coords: [tuple.0, tuple.1, tuple.2],
        }
    }
}

impl<T, const N: usize> From<Vector<T, N>> for Point<T, N>
where
    T: Float,
{
    /// Converts a [`Vector`] into a [`Point`], assuming the vector represents
    /// a position relative to the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// let p = Point::from(v);
    /// assert_eq!(p.coords(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(vector: Vector<T, N>) -> Self {
        Self {
            coords: *vector.coords(),
        }
    }
}

impl<T, const N: usize> MetricSquared<T> for Point<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Calculates the squared Euclidean distance between two points.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, MetricSquared};
    ///
    /// let p1 = Point::new([0.0, 0.0]);
    /// let p2 = Point::new([3.0, 4.0]);
    /// assert_eq!(p1.distance_squared(&p2), 25.0);
    /// ```
    #[inline]
    fn distance_squared(&self, other: &Self) -> T {
        self.coords
            .iter()
            .zip(other.coords().iter())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .sum()
    }
}

impl<T, const N: usize> EuclideanMetric<T> for Point<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Calculates the Euclidean distance between two points.
    ///
    /// # Examples
    /// ```
    /// use apollonius::{Point, EuclideanMetric};
    ///
    /// let p1 = Point::new([0.0, 0.0]);
    /// let p2 = Point::new([3.0, 4.0]);
    /// assert_eq!(p1.distance(&p2), 5.0);
    /// ```
    #[inline]
    fn distance(&self, other: &Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

impl<T, const N: usize> Sub for Point<T, N>
where
    T: Float,
{
    type Output = Vector<T, N>;

    /// Subtracting two points yields a displacement [`Vector`].
    ///
    /// # Examples
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let p1 = Point::new([10.0, 20.0]);
    /// let p2 = Point::new([15.0, 25.0]);
    /// let v = p2 - p1;
    ///
    /// assert_eq!(v.coords(), &[5.0, 5.0]);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        let coords = std::array::from_fn(|i| self.coords()[i] - rhs.coords()[i]);
        Vector::new(coords)
    }
}

impl<T, const N: usize> Add<Vector<T, N>> for Point<T, N>
where
    T: Float,
{
    type Output = Point<T, N>;

    /// Adding a [`Vector`] to a [`Point`] translates the point in space.
    ///
    /// # Examples
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let p = Point::new([1.0, 2.0]);
    /// let v = Vector::new([3.0, 4.0]);
    /// let p_prime = p + v;
    ///
    /// assert_eq!(p_prime.coords(), &[4.0, 6.0]);
    /// ```
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| self.coords()[i] + rhs.coords()[i]);
        Self { coords }
    }
}

impl<T, const N: usize> Sub<Vector<T, N>> for Point<T, N>
where
    T: Float,
{
    type Output = Point<T, N>;

    /// Subtracting a [`Vector`] from a [`Point`] translates the point in the opposite direction.
    ///
    /// # Examples
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let p = Point::new([1.0, 2.0]);
    /// let v = Vector::new([3.0, 4.0]);
    /// let p_prime = p - v;
    ///
    /// assert_eq!(p_prime.coords(), &[-2.0, -2.0]);
    /// ```
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| self.coords()[i] - rhs.coords()[i]);
        Self { coords }
    }
}

#[cfg(test)]
mod points_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_construction_and_conversions() {
        let p_gen = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(p_gen.coords(), &[1.0, 2.0, 3.0]);

        let p_from_arr = Point::new([1.5, 2.5]);
        assert_eq!(p_from_arr.coords(), &[1.5, 2.5]);

        let p2d: Point2D<f32> = Point2D::from((1.0, 2.0));
        assert_eq!(p2d.coords(), &[1.0, 2.0]);

        let p3d: Point3D<f32> = Point3D::from((1.0, 2.0, 3.0));
        assert_eq!(p3d.coords(), &[1.0, 2.0, 3.0]);

        let alias_2d: Point2D<f64> = Point::new([1.0, 2.0]);
        assert_eq!(alias_2d.coords(), &[1.0, 2.0]);
    }

    #[test]
    fn test_distance_operations_floats() {
        let p1 = Point::new([0.0_f32, 0.0, 0.0]);
        let p2 = Point::new([3.0, 4.0, 0.0]);

        assert_relative_eq!(p1.distance_squared(&p2), 25.0);
        assert_relative_eq!(p1.distance(&p2), 5.0);

        assert_relative_eq!(p1.distance_squared(&p1), 0.0);
        assert_relative_eq!(p1.distance(&p1), 0.0);
        let p1_f64 = Point::new([0.0_f64, 0.0]);
        let p2_f64 = Point::new([1.0, 1.0]);
        assert_relative_eq!(p1_f64.distance(&p2_f64), 2.0_f64.sqrt());
    }

    #[test]
    fn test_distance_squared_floats() {
        let p1 = Point::new([1.0, 2.0]);
        let p2 = Point::new([4.0, 6.0]);
        assert_relative_eq!(p1.distance_squared(&p2), 25.0);
    }

    #[test]
    fn test_properties_and_traits() {
        let p1 = Point::new([1.0, 2.0]);
        let p2 = p1;

        assert_eq!(p1.coords(), p2.coords());

        let p3 = Point::new([1.0, 2.0]);
        println!("{:?}", p3);
        assert!(p3 == Point::new([1.0, 2.0]));
        assert!(p3 != Point::new([1.0, 3.0]));
    }

    #[test]
    fn test_point_subtraction_to_vector() {
        use super::Vector;

        let p1 = Point::new([10.0_f64, 20.0, 30.0]);
        let p2 = Point::new([15.0_f64, 25.0, 35.0]);

        let v: Vector<f64, 3> = p2 - p1;
        assert_relative_eq!(v.coords()[0], 5.0);
        assert_relative_eq!(v.coords()[1], 5.0);
        assert_relative_eq!(v.coords()[2], 5.0);

        let p3 = Point::new([1.0, 2.0]);
        let p4 = Point::new([4.0, 6.0]);
        let v2: Vector<f64, 2> = p4 - p3;
        assert_eq!(v2.coords(), &[3.0, 4.0]);
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_point_serialization_roundtrip() {
        let p = Point::new([1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&p).unwrap();
        let q: Point<f64, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn test_point2d_serialization_roundtrip() {
        let p = Point2D::from((1.0, 2.0));
        let json = serde_json::to_string(&p).unwrap();
        let q: Point2D<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn test_point3d_serialization_roundtrip() {
        let p = Point3D::from((1.0, 2.0, 3.0));
        let json = serde_json::to_string(&p).unwrap();
        let q: Point3D<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(p, q);
    }
}
