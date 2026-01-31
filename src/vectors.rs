use crate::Point;
use num_traits::Float;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Trait for types that can calculate squared magnitude and dot products.
///
/// These operations are mathematically valid for any type that supports multiplication
/// and addition, including integers, without requiring square root operations.
pub trait VectorMetricSquared<T> {
    /// Calculates the squared magnitude (squared norm) of the vector.
    fn magnitude_squared(&self) -> T;

    /// Calculates the dot product between two vectors.
    fn dot(&self, other: &Self) -> T;
}

/// Trait for types that support Euclidean operations like magnitude and normalization.
///
/// These operations require floating-point numbers ([`Float`]).
pub trait EuclideanVector<T>: VectorMetricSquared<T>
where
    Self: Sized,
{
    /// Calculates the Euclidean magnitude (norm) of the vector.
    fn magnitude(&self) -> T;

    /// Returns a normalized version of the vector (unit vector).
    ///
    /// Returns `None` if the magnitude is zero or near epsilon.
    fn normalize(&self) -> Option<Self>;
}

/// Represents a displacement or direction in N-dimensional space.
///
/// Vectors represent relative direction and magnitude. Unlike [`Point`],
/// which represents an absolute position, vectors are origin-independent.
///
/// # Examples
///
/// ```
/// use apollonius::Vector;
///
/// let v = Vector::new([1.0, 2.0, 3.0]);
/// assert_eq!(v.coords, [1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<T, const N: usize> {
    /// The components of the vector along each of the N axes.
    pub coords: [T; N],
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for Vector<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coords.as_slice().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Vector<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let coords_vec = Vec::<T>::deserialize(deserializer)?;
        let coords: [T; N] = coords_vec.try_into().map_err(|_| {
            serde::de::Error::custom(format!("Vector dimension mismatch: expected {}", N))
        })?;
        Ok(Self { coords })
    }
}

/// A 2D vector specialization.
pub type Vector2D<T> = Vector<T, 2>;

/// A 3D vector specialization.
pub type Vector3D<T> = Vector<T, 3>;

impl<T, const N: usize> Vector<T, N>
where
    T: Copy,
{
    /// Creates a new vector from an array of coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(v.coords, [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn new(coords: [T; N]) -> Self {
        Self { coords }
    }
}

impl<T, const N: usize> From<(&Point<T, N>, &Point<T, N>)> for Vector<T, N>
where
    T: Sub<Output = T> + Copy,
{
    /// Creates a vector that represents the displacement from an initial to a final point.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let a = Point::new([1.0, 2.0]);
    /// let b = Point::new([4.0, 6.0]);
    /// let v: Vector<f64, 2> = Vector::from((&a, &b));
    /// assert_eq!(v.coords, [3.0, 4.0]);
    /// ```
    #[inline]
    fn from((initial, final_point): (&Point<T, N>, &Point<T, N>)) -> Self {
        let coords = std::array::from_fn(|i| final_point.coords[i] - initial.coords[i]);
        Vector { coords }
    }
}

impl<T, const N: usize> From<&Point<T, N>> for Vector<T, N>
where
    T: Sub<Output = T> + Copy,
{
    /// Converts a reference to a point into a vector from the origin to that point.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Point, Vector};
    ///
    /// let p = Point::new([2.0, 3.0]);
    /// let v = Vector::from(&p);
    /// assert_eq!(v.coords, [2.0, 3.0]);
    /// ```
    #[inline]
    fn from(final_point: &Point<T, N>) -> Self {
        Vector {
            coords: final_point.coords,
        }
    }
}

impl<T> From<(T, T)> for Vector2D<T> {
    /// Converts a 2-element tuple into a [`Vector2D`].
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Vector2D;
    ///
    /// let v = Vector2D::from((1.0, 2.0));
    /// assert_eq!(v.coords, [1.0, 2.0]);
    /// ```
    #[inline]
    fn from(value: (T, T)) -> Self {
        let (x, y) = value;
        Self { coords: [x, y] }
    }
}

impl<T> From<(T, T, T)> for Vector3D<T> {
    /// Converts a 3-element tuple into a [`Vector3D`].
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Vector3D;
    ///
    /// let v = Vector3D::from((1.0, 2.0, 3.0));
    /// assert_eq!(v.coords, [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(value: (T, T, T)) -> Self {
        let (x, y, z) = value;
        Self { coords: [x, y, z] }
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Vector<T, N>;

    /// Performs vector addition component-wise.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Vector;
    ///
    /// let u = Vector::new([1.0, 2.0]);
    /// let v = Vector::new([3.0, 4.0]);
    /// let w = u + v;
    /// assert_eq!(w.coords, [4.0, 6.0]);
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let coords = std::array::from_fn(|i| self.coords[i] + rhs.coords[i]);
        Self { coords }
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Vector<T, N>;

    /// Performs vector subtraction component-wise.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Vector;
    ///
    /// let u = Vector::new([4.0, 5.0]);
    /// let v = Vector::new([1.0, 2.0]);
    /// let w = u - v;
    /// assert_eq!(w.coords, [3.0, 3.0]);
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let coords = std::array::from_fn(|i| self.coords[i] - rhs.coords[i]);
        Self { coords }
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vector<T, N>;

    /// Performs scalar multiplication (Vector * Scalar).
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0]) * 2.0;
    /// assert_eq!(v.coords, [2.0, 4.0]);
    /// ```
    #[inline]
    fn mul(self, scalar: T) -> Self::Output {
        let coords = self.coords.map(|coord| coord * scalar);
        Self { coords }
    }
}

impl<T, const N: usize> AddAssign for Vector<T, N>
where
    T: AddAssign + Copy,
{
    /// Performs in-place vector addition.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coords
            .iter_mut()
            .zip(rhs.coords.iter())
            .for_each(|(coord, rhs_coord)| *coord += *rhs_coord);
    }
}

impl<T, const N: usize> SubAssign for Vector<T, N>
where
    T: SubAssign + Copy,
{
    /// Performs in-place vector subtraction.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coords
            .iter_mut()
            .zip(rhs.coords.iter())
            .for_each(|(coord, rhs_coord)| *coord -= *rhs_coord);
    }
}

impl<const N: usize> Mul<Vector<f32, N>> for f32 {
    type Output = Vector<f32, N>;
    /// Performs scalar multiplication (Scalar * Vector) for f32.
    #[inline]
    fn mul(self, vector: Vector<f32, N>) -> Vector<f32, N> {
        vector * self
    }
}

impl<const N: usize> Mul<Vector<f64, N>> for f64 {
    type Output = Vector<f64, N>;
    /// Performs scalar multiplication (Scalar * Vector) for f64.
    #[inline]
    fn mul(self, vector: Vector<f64, N>) -> Vector<f64, N> {
        vector * self
    }
}

impl<T, const N: usize> VectorMetricSquared<T> for Vector<T, N>
where
    T: Mul<Output = T> + std::iter::Sum + Copy,
{
    /// Returns the squared magnitude (squared length) of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Vector, VectorMetricSquared};
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// assert_eq!(v.magnitude_squared(), 25.0);
    /// ```
    #[inline]
    fn magnitude_squared(&self) -> T {
        self.coords.iter().map(|coord| *coord * *coord).sum()
    }

    /// Calculates the dot product between two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Vector, VectorMetricSquared};
    ///
    /// let v1 = Vector::new([1, 2]);
    /// let v2 = Vector::new([3, 4]);
    /// assert_eq!(v1.dot(&v2), 11); // (1*3) + (2*4)
    /// ```
    #[inline]
    fn dot(&self, other: &Self) -> T {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }
}

impl<T, const N: usize> EuclideanVector<T> for Vector<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the Euclidean magnitude (length) of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::{Vector, EuclideanVector};
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// assert_eq!(v.magnitude(), 5.0);
    /// ```
    #[inline]
    fn magnitude(&self) -> T {
        self.magnitude_squared().sqrt()
    }

    /// Returns a normalized unit vector.
    ///
    /// Returns `None` if the vector is too small to be safely normalized
    /// (magnitude <= epsilon * 10).
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Vector, EuclideanVector};
    ///
    /// let v = Vector::new([3.0, 0.0]);
    /// let unit = v.normalize().unwrap();
    /// assert_eq!(unit.coords[0], 1.0);
    /// ```
    #[inline]
    fn normalize(&self) -> Option<Self> {
        let mag = self.magnitude();
        if mag <= T::epsilon() * T::from(10.0).unwrap_or(T::one()) {
            None
        } else {
            Some(*self * (T::one() / mag))
        }
    }
}

impl<T> Vector<T, 3>
where
    T: Copy + Mul<Output = T> + Sub<Output = T>,
{
    /// Calculates the cross product (3D vectors only).
    ///
    /// The cross product returns a vector perpendicular to the plane
    /// formed by the two input vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::Vector3D;
    ///
    /// let v1 = Vector3D::from((1, 0, 0));
    /// let v2 = Vector3D::from((0, 1, 0));
    /// let cross = v1.cross(&v2);
    /// assert_eq!(cross.coords, [0, 0, 1]);
    /// ```
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        let [x1, y1, z1] = self.coords;
        let [x2, y2, z2] = other.coords;

        Self {
            coords: [y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2],
        }
    }
}

#[cfg(test)]
mod vectors_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_construction_and_conversions() {
        let v_gen: Vector<i32, 3> = Vector::new([1, 2, 3]);
        assert_eq!(v_gen.coords, [1, 2, 3]);

        let p1 = Point::new([1.0, 2.0, 3.0]);
        let p2 = Point::new([4.0, 6.0, 8.0]);
        let v_from_pts: Vector<f64, 3> = Vector::from((&p1, &p2));
        assert_eq!(v_from_pts.coords, [3.0, 4.0, 5.0]);

        let v2d: Vector2D<f32> = Vector2D::from((1.0, 2.0));
        assert_eq!(v2d.coords, [1.0, 2.0]);

        let v3d: Vector3D<f32> = Vector3D::from((1.0, 2.0, 3.0));
        assert_eq!(v3d.coords, [1.0, 2.0, 3.0]);

        let alias: Vector3D<f64> = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(alias.coords, [1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_arithmetic_operations() {
        let v1 = Vector::new([1.0, 2.0, 3.0]);
        let v2 = Vector::new([4.0, 5.0, 6.0]);

        let sum = v1 + v2;
        assert_relative_eq!(sum.coords[0], 5.0);
        assert_relative_eq!(sum.coords[1], 7.0);
        assert_relative_eq!(sum.coords[2], 9.0);

        let diff = v2 - v1;
        assert_relative_eq!(diff.coords[0], 3.0);
        assert_relative_eq!(diff.coords[1], 3.0);
        assert_relative_eq!(diff.coords[2], 3.0);

        let scaled_r = v1 * 2.0;
        assert_eq!(scaled_r.coords, [2.0, 4.0, 6.0]);

        let scaled_l_f32: Vector<f32, 2> = 3.0_f32 * Vector::new([1.0, 2.0]);
        assert_eq!(scaled_l_f32.coords, [3.0, 6.0]);

        let scaled_l_f64: Vector<f64, 2> = 3.0_f64 * Vector::new([1.0, 2.0]);
        assert_eq!(scaled_l_f64.coords, [3.0, 6.0]);

        let mut v = Vector::new([1.0, 2.0]);
        v += Vector::new([3.0, 4.0]);
        assert_eq!(v.coords, [4.0, 6.0]);

        v -= Vector::new([1.0, 2.0]);
        assert_eq!(v.coords, [3.0, 4.0]);
    }

    #[test]
    fn test_vector_properties() {
        let v = Vector::new([3.0_f64, 4.0, 0.0]);

        assert_relative_eq!(v.magnitude_squared(), 25.0);

        assert_relative_eq!(v.magnitude(), 5.0);

        let v1 = Vector::new([1.0, 2.0, 3.0]);
        let v2 = Vector::new([4.0, 5.0, 6.0]);

        assert_relative_eq!(v1.dot(&v2), 32.0);

        assert_relative_eq!(v1.dot(&v1), v1.magnitude_squared());

        let normalized = v.normalize().unwrap();
        assert_relative_eq!(normalized.magnitude(), 1.0);

        assert_relative_eq!(normalized.coords[0] / v.coords[0], 0.2);

        let zero_vec: Vector<f64, 3> = Vector::new([0.0, 0.0, 0.0]);
        assert!(zero_vec.normalize().is_none());
    }

    #[test]
    fn test_cross_product() {
        let v1 = Vector3D::from((1.0, 0.0, 0.0));
        let v2 = Vector3D::from((0.0, 1.0, 0.0));
        let cross = v1.cross(&v2);
        assert_relative_eq!(cross.coords[0], 0.0);
        assert_relative_eq!(cross.coords[1], 0.0);
        assert_relative_eq!(cross.coords[2], 1.0);

        let v = Vector3D::from((3.0, -2.0, 5.0));
        let self_cross = v.cross(&v);
        assert!(self_cross.coords.iter().all(|&c| c.abs() < 1e-10));

        let v = Vector3D::from((2.0, 3.0, 4.0));
        let w = Vector3D::from((5.0, 6.0, 7.0));
        let cross_vw = v.cross(&w);
        assert_relative_eq!(cross_vw.dot(&v), 0.0, epsilon = 1e-10);
        assert_relative_eq!(cross_vw.dot(&w), 0.0, epsilon = 1e-10);

        let v_int: Vector3D<i32> = Vector3D::from((1, 0, 0));
        let w_int = Vector3D::from((0, 1, 0));
        let cross_int = v_int.cross(&w_int);
        assert_eq!(cross_int.coords, [0, 0, 1]);
    }

    #[test]
    fn test_traits_and_types() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = v1;
        assert_eq!(v1.coords, v2.coords);
        println!("{:?}", v1);

        let v_f32: Vector<f32, 2> = Vector::new([1.5, 2.5]);
        let _ = v_f32 * 2.0_f32;

        let v_i32: Vector<i32, 3> = Vector::new([1, 2, 3]);
        let _sum_i32 = v_i32 + Vector::new([4, 5, 6]);
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_vector_serialization_roundtrip() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&v).unwrap();
        let w: Vector<f64, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(v, w);
    }

    #[test]
    fn test_vector2d_serialization_roundtrip() {
        let v = Vector2D::from((1.0, 2.0));
        let json = serde_json::to_string(&v).unwrap();
        let w: Vector2D<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(v, w);
    }

    #[test]
    fn test_vector3d_serialization_roundtrip() {
        let v = Vector3D::from((1.0, 2.0, 3.0));
        let json = serde_json::to_string(&v).unwrap();
        let w: Vector3D<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(v, w);
    }
}
