//! Square matrices over floating-point types with linear algebra operations.
//!
//! This module provides an N×N [`Matrix`] type parameterised by element type and dimension, with
//! support for identity, addition, subtraction, matrix multiplication, and multiplication by
//! [`Point`] and [`Vector`]. Element type is constrained by [`Float`](num_traits::Float) so that
//! zero and one are available for identity and products.

use crate::{Angle, Point, Vector, space::LinearMap};
use num_traits::Float;
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An N×N square matrix with floating-point elements.
///
/// Storage is row-major: `data[i][j]` is the element in row `i` and column `j`. Matrix × vector
/// and matrix × point products treat the right-hand side as a column vector.
///
/// # Type parameters
///
/// * `T` — element type, must implement [`Float`](num_traits::Float).
/// * `N` — dimension (matrix size is N×N).
///
/// # Serialization
///
/// With the **`serde`** feature enabled, `Matrix` implements `Serialize` and `Deserialize`.
/// The matrix is stored as a single row-major array of `N×N` elements (same pattern as [`Point`]
/// and [`Vector`]).
///
/// # Examples
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
///
/// let m = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// assert_eq!(m.data(), &[[1.0, 0.0], [0.0, 1.0]]);
/// ```
///
/// Matrix-vector product:
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
/// use apollonius::Vector;
///
/// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let v = Vector::new([1.0, 0.0]);
/// let out = m * v;
/// assert_eq!(out.coords(), &[1.0, 3.0]);
/// ```
///
/// Matrix-point product:
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
/// use apollonius::Point;
///
/// let m = Matrix::<f64, 2>::identity();
/// let p = Point::new([2.0, 3.0]);
/// assert_eq!(m * p, p);
/// ```
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Matrix<T, const N: usize> {
    /// Row-major storage: `data[i][j]` is row `i`, column `j`.
    data: [[T; N]; N],
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for Matrix<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let flat: Vec<&T> = self.data.iter().flat_map(|row| row.iter()).collect();
        flat.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Matrix<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let flat = Vec::<T>::deserialize(deserializer)?;
        let expected = N * N;
        if flat.len() != expected {
            return Err(serde::de::Error::custom(format!(
                "Matrix dimension mismatch: expected {} elements ({}×{}), got {}",
                expected,
                N,
                N,
                flat.len()
            )));
        }
        let mut it = flat.into_iter();
        let data =
            std::array::from_fn(|_| std::array::from_fn(|_| it.next().expect("N*N elements")));
        Ok(Self { data })
    }
}

impl<T, const N: usize> Matrix<T, N>
where
    T: Float,
{
    /// Creates a matrix from a row-major array of shape `N×N`.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::algebra::matrix::Matrix;
    ///
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.data()[0], [1.0, 2.0]);
    /// assert_eq!(m.data()[1], [3.0, 4.0]);
    /// ```
    #[inline]
    pub fn new(data: [[T; N]; N]) -> Self {
        Self { data }
    }

    /// Returns a reference to the inner row-major data array.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::algebra::matrix::Matrix;
    ///
    /// let m = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    /// assert_eq!(m.data(), &[[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn data(&self) -> &[[T; N]; N] {
        &self.data
    }

    /// Sets the inner row-major data array.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::algebra::matrix::Matrix;
    ///
    /// let mut m = Matrix::new([[0.0, 0.0], [0.0, 0.0]]);
    /// m.set_data([[1.0, 0.0], [0.0, 1.0]]);
    /// assert_eq!(m.data(), &[[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn set_data(&mut self, data: [[T; N]; N]) {
        self.data = data;
    }

    /// Sets the element at the given row and column (zero-based indices).
    ///
    /// # Panics
    ///
    /// Panics if `row` or `column` is not in `0..N`.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Matrix;
    ///
    /// let mut m = Matrix::<f64, 2>::identity();
    /// m.set_component(0, 1, 5.0);
    /// m.set_component(1, 0, -3.0);
    /// assert_eq!(m.data(), &[[1.0, 5.0], [-3.0, 1.0]]);
    /// ```
    pub fn set_component(&mut self, row: usize, column: usize, value: T) {
        assert!(
            row < N && column < N,
            "row and column indices must be in 0..{} for {0}×{0} matrix (got row={}, column={})",
            N,
            row,
            column
        );
        self.data[row][column] = value;
    }

    /// Returns a mutable reference to the inner row-major data array.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [[T; N]; N] {
        &mut self.data
    }

    /// Returns the N×N identity matrix (ones on the diagonal, zeros elsewhere).
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::algebra::matrix::Matrix;
    ///
    /// let i = Matrix::<f64, 3>::identity();
    /// assert_eq!(i.data()[0], [1.0, 0.0, 0.0]);
    /// assert_eq!(i.data()[1], [0.0, 1.0, 0.0]);
    /// assert_eq!(i.data()[2], [0.0, 0.0, 1.0]);
    /// ```
    pub fn identity() -> Self {
        let data = std::array::from_fn(|i| {
            let mut row = [T::zero(); N];
            row[i] = T::one();
            row
        });
        Self { data }
    }

    /// Builds a linear map that rotates by `angle` (radians) in the plane spanned by the two
    /// given axes. The rotation is counterclockwise when viewing from the positive side of the
    /// remaining axes (right-hand rule). Returns [`LinearMap::Rotation`](crate::space::LinearMap::Rotation).
    ///
    /// `axis_i` and `axis_j` must be distinct and in `0..N`; they define the 2D plane of rotation
    /// (all other coordinates are unchanged).
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range or if `axis_i == axis_j`.
    ///
    /// # Example
    ///
    /// 90° rotation in the plane of axes 0 and 1 (2D rotation in the first two coordinates):
    ///
    /// ```
    /// use apollonius::{Angle, Matrix, Vector, space::LinearMap};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
    /// let LinearMap::Rotation(m) = Matrix::<f64, 3>::rotation(angle, 0, 1);
    /// let v = Vector::new([1.0, 0.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
    /// assert_eq!(rotated.coords()[2], 0.0);
    /// ```
    pub fn rotation(angle: Angle<T>, axis_i: usize, axis_j: usize) -> LinearMap<T, N> {
        assert!(
            axis_i < N && axis_j < N,
            "rotation axis indices must be in 0..{} for dimension N={} (got axis_i={}, axis_j={})",
            N,
            N,
            axis_i,
            axis_j
        );
        assert_ne!(
            axis_i, axis_j,
            "rotation requires two distinct axes to define a plane (got axis_i={}, axis_j={})",
            axis_i, axis_j
        );

        let mut mat = Self::identity();
        let (sin, cos) = angle.sin_cos();

        mat.set_component(axis_i, axis_i, cos);
        mat.set_component(axis_i, axis_j, -sin);

        mat.set_component(axis_j, axis_i, sin);
        mat.set_component(axis_j, axis_j, cos);

        LinearMap::Rotation(mat)
    }
}

impl<T> Matrix<T, 2>
where
    T: Float,
{
    /// Builds a 2D rotation by `angle` (radians) in the plane of the two axes (0 and 1).
    /// Equivalent to [`rotation`](Matrix::rotation)(`angle`, 0, 1). Returns
    /// [`LinearMap::Rotation`](crate::space::LinearMap::Rotation).
    ///
    /// # Example
    ///
    /// Rotate the unit vector (1, 0) by 90° to get (0, 1):
    ///
    /// ```
    /// use apollonius::{Angle, Matrix, Vector, space::LinearMap};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
    /// let LinearMap::Rotation(m) = Matrix::rotation_2d(angle);
    /// let v = Vector::new([1.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn rotation_2d(angle: Angle<T>) -> LinearMap<T, 2> {
        Self::rotation(angle, 0, 1)
    }
}

impl<T> Matrix<T, 3>
where
    T: Float,
{
    /// Builds a 3D rotation by `angle` (radians) around a single axis. `axis` selects the
    /// rotation axis: 0 = X, 1 = Y, 2 = Z. Returns
    /// [`LinearMap::Rotation`](crate::space::LinearMap::Rotation).
    ///
    /// # Panics
    ///
    /// Panics if `axis` is not 0, 1, or 2.
    ///
    /// # Example
    ///
    /// Rotate the vector (1, 0, 0) by 90° around the Z axis to get (0, 1, 0):
    ///
    /// ```
    /// use apollonius::{Angle, Matrix, Vector, space::LinearMap};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
    /// let LinearMap::Rotation(m) = Matrix::rotation_3d(angle, 2);
    /// let v = Vector::new([1.0, 0.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
    /// assert_eq!(rotated.coords()[2], 0.0);
    /// ```
    pub fn rotation_3d(angle: Angle<T>, axis: usize) -> LinearMap<T, 3> {
        match axis {
            0 => Self::rotation(angle, 1, 2),
            1 => Self::rotation(angle, 2, 0), // Y: plane Z-X so that +90° maps (0,0,1) → (1,0,0)
            2 => Self::rotation(angle, 0, 1),
            _ => panic!(
                "rotation_3d axis must be 0 (X), 1 (Y), or 2 (Z), got {}",
                axis
            ),
        }
    }
}

/// Element-wise addition of two matrices of the same size.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
///
/// let a = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// let b = Matrix::new([[0.0, 1.0], [1.0, 0.0]]);
/// let sum = a + b;
/// assert_eq!(sum.data(), &[[1.0, 1.0], [1.0, 1.0]]);
/// ```
impl<T, const N: usize> Add for Matrix<T, N>
where
    T: Float,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let data = std::array::from_fn(|i| {
            let row = std::array::from_fn(|j| self.data[i][j] + rhs.data[i][j]);
            row
        });
        Self { data }
    }
}

/// Element-wise subtraction of two matrices of the same size.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
///
/// let a = Matrix::new([[2.0, 1.0], [1.0, 2.0]]);
/// let b = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
/// assert_eq!((a - b).data(), &[[1.0, 1.0], [1.0, 1.0]]);
/// ```
impl<T, const N: usize> Sub for Matrix<T, N>
where
    T: Float,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let data = std::array::from_fn(|i| {
            let row = std::array::from_fn(|j| self.data[i][j] - rhs.data[i][j]);
            row
        });
        Self { data }
    }
}

/// Matrix multiplication (row × column). Result element `(i, j)` is the dot product of row `i` of
/// `self` with column `j` of `rhs`.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
///
/// let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
/// let ab = a * b;
/// assert_eq!(ab.data(), &[[19.0, 22.0], [43.0, 50.0]]);
/// ```
impl<T, const N: usize> Mul for Matrix<T, N>
where
    T: Float,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let data = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let mut sum = T::zero();
                for k in 0..N {
                    sum = sum + self.data[i][k] * rhs.data[k][j];
                }
                sum
            })
        });
        Self { data }
    }
}

/// Matrix × vector product (vector as column). Returns a [`Vector`] of the same dimension.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
/// use apollonius::Vector;
///
/// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let v = Vector::new([1.0, 1.0]);
/// let w = m * v;
/// assert_eq!(w.coords(), &[3.0, 7.0]);
/// ```
impl<T, const N: usize> Mul<Vector<T, N>> for Matrix<T, N>
where
    T: Float,
{
    type Output = Vector<T, N>;
    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| {
            let mut sum = T::zero();
            for j in 0..N {
                sum = sum + self.data[i][j] * rhs.coords()[j];
            }
            sum
        });
        Vector::new(coords)
    }
}

/// Matrix × point product (point coordinates as column). Returns a [`Point`] of the same dimension.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::Matrix;
/// use apollonius::Point;
///
/// let m = Matrix::<f64, 2>::identity();
/// let p = Point::new([5.0, 10.0]);
/// assert_eq!(m * p, p);
/// ```
impl<T, const N: usize> Mul<Point<T, N>> for Matrix<T, N>
where
    T: Float,
{
    type Output = Point<T, N>;
    fn mul(self, rhs: Point<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| {
            let mut sum = T::zero();
            for j in 0..N {
                sum = sum + self.data[i][j] * rhs.coords()[j];
            }
            sum
        });
        Point::new(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use crate::space::LinearMap;
    use crate::{Angle, Point, Vector};
    use std::f64::consts::FRAC_PI_2;

    fn identity_2() -> Matrix<f64, 2> {
        Matrix::new([[1.0, 0.0], [0.0, 1.0]])
    }
    fn identity_3() -> Matrix<f64, 3> {
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[test]
    fn equality_same_data_are_equal() {
        let a = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(a, b);
    }

    #[test]
    fn equality_different_data_are_not_equal() {
        let a = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::new([[1.0, 1.0], [0.0, 1.0]]);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_identity_equals_self() {
        assert_eq!(Matrix::<f64, 2>::identity(), identity_2());
        assert_eq!(Matrix::<f64, 3>::identity(), identity_3());
    }

    #[test]
    fn identity_2x2_diagonal_ones_elsewhere_zero() {
        let i = Matrix::<f64, 2>::identity();
        assert_eq!(i, identity_2());
    }

    #[test]
    fn identity_3x3_diagonal_ones_elsewhere_zero() {
        let i = Matrix::<f64, 3>::identity();
        assert_eq!(i, identity_3());
    }

    #[test]
    fn identity_4x4() {
        let i = Matrix::<f64, 4>::identity();
        assert_eq!(
            i.data(),
            &[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn add_element_wise_2x2() {
        let a = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::new([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(a + b, Matrix::new([[1.0, 1.0], [1.0, 1.0]]));
    }

    #[test]
    fn add_commutative() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn add_identity_is_neutral() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m + identity_2(), Matrix::new([[2.0, 2.0], [3.0, 5.0]]));
    }

    #[test]
    fn sub_element_wise_2x2() {
        let a = Matrix::new([[2.0, 1.0], [1.0, 2.0]]);
        let b = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(a - b, Matrix::new([[1.0, 1.0], [1.0, 1.0]]));
    }

    #[test]
    fn sub_inverse_of_add() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[0.5, 1.0], [1.5, 2.0]]);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn mul_matrix_identity_left() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(Matrix::<f64, 2>::identity() * m, m);
    }

    #[test]
    fn mul_matrix_identity_right() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m * Matrix::<f64, 2>::identity(), m);
    }

    #[test]
    fn mul_matrix_2x2_diagonal() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let diag = Matrix::new([[2.0, 0.0], [0.0, 2.0]]);
        assert_eq!(a * diag, Matrix::new([[2.0, 4.0], [6.0, 8.0]]));
    }

    #[test]
    fn mul_matrix_2x2_general() {
        let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(a * b, Matrix::new([[19.0, 22.0], [43.0, 50.0]]));
    }

    #[test]
    fn mul_matrix_3x3() {
        let a = Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let b = Matrix::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(a * b, b);
    }

    #[test]
    fn mul_vector_identity_preserves_vector_2d() {
        let i = Matrix::<f64, 2>::identity();
        let v = Vector::new([3.0, 4.0]);
        assert_eq!(i * v, v);
    }

    #[test]
    fn mul_vector_identity_preserves_vector_3d() {
        let i = Matrix::<f64, 3>::identity();
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(i * v, v);
    }

    #[test]
    fn mul_vector_2x2_first_column() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([1.0, 0.0]);
        let out = m * v;
        assert_eq!(out.coords(), &[1.0, 3.0]);
    }

    #[test]
    fn mul_vector_2x2_second_column() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([0.0, 1.0]);
        let out = m * v;
        assert_eq!(out.coords(), &[2.0, 4.0]);
    }

    #[test]
    fn mul_vector_2x2_general() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([2.0, 1.0]);
        let out = m * v;
        assert_eq!(out.coords(), &[4.0, 10.0]);
    }

    #[test]
    fn mul_vector_3x3() {
        let m = Matrix::new([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]);
        let v = Vector::new([1.0, 2.0, 3.0]);
        let out = m * v;
        assert_eq!(out.coords(), &[4.0, 5.0, 3.0]);
    }

    #[test]
    fn mul_point_identity_preserves_point_2d() {
        let i = Matrix::<f64, 2>::identity();
        let p = Point::new([3.0, 4.0]);
        assert_eq!(i * p, p);
    }

    #[test]
    fn mul_point_identity_preserves_point_3d() {
        let i = Matrix::<f64, 3>::identity();
        let p = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(i * p, p);
    }

    #[test]
    fn mul_point_2x2() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let p = Point::new([1.0, 0.0]);
        let out = m * p;
        assert_eq!(out.coords(), &[1.0, 3.0]);
    }

    #[test]
    fn mul_point_2x2_general() {
        let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let p = Point::new([2.0, 1.0]);
        let out = m * p;
        assert_eq!(out.coords(), &[4.0, 10.0]);
    }

    #[test]
    fn mul_point_3x3() {
        let m = Matrix::new([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]);
        let p = Point::new([1.0, 2.0, 1.0]);
        let out = m * p;
        assert_eq!(out.coords(), &[3.0, 5.0, 1.0]);
    }

    #[test]
    fn rotation_2d_returns_linear_map_rotation() {
        let angle = Angle::<f64>::from_radians(0.0);
        let r = Matrix::rotation_2d(angle);
        let LinearMap::Rotation(_) = r;
    }

    #[test]
    fn rotation_2d_zero_angle_is_identity() {
        let angle = Angle::<f64>::from_radians(0.0);
        let LinearMap::Rotation(m) = Matrix::rotation_2d(angle);
        assert_eq!(m, Matrix::<f64, 2>::identity());
    }

    #[test]
    fn rotation_2d_90_degrees_maps_unit_x_to_unit_y() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::rotation_2d(angle);
        let v = Vector::new([1.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_2d_90_degrees_maps_unit_y_to_negative_x() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::rotation_2d(angle);
        let v = Vector::new([0.0, 1.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - (-1.0)).abs() < 1e-10);
        assert!((rotated.coords()[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_2d_is_orthogonal() {
        let angle = Angle::<f64>::from_radians(0.7);
        let LinearMap::Rotation(m) = Matrix::rotation_2d(angle);
        let mt_m = {
            let d = m.data();
            Matrix::new([
                [
                    d[0][0] * d[0][0] + d[1][0] * d[1][0],
                    d[0][0] * d[0][1] + d[1][0] * d[1][1],
                ],
                [
                    d[0][1] * d[0][0] + d[1][1] * d[1][0],
                    d[0][1] * d[0][1] + d[1][1] * d[1][1],
                ],
            ])
        };
        let i = Matrix::<f64, 2>::identity();
        assert!((mt_m.data()[0][0] - i.data()[0][0]).abs() < 1e-9);
        assert!((mt_m.data()[0][1] - i.data()[0][1]).abs() < 1e-9);
        assert!((mt_m.data()[1][0] - i.data()[1][0]).abs() < 1e-9);
        assert!((mt_m.data()[1][1] - i.data()[1][1]).abs() < 1e-9);
    }

    // --- Rotation 3D (ángulo + eje 0/1/2; devuelve LinearMap::Rotation) ----------------------

    #[test]
    fn rotation_3d_returns_linear_map_rotation() {
        let angle = Angle::<f64>::from_radians(0.0);
        let r = Matrix::rotation_3d(angle, 0);
        let LinearMap::Rotation(_) = r;
    }

    #[test]
    fn rotation_3d_zero_angle_is_identity() {
        for axis in [0usize, 1, 2] {
            let angle = Angle::<f64>::from_radians(0.0);
            let LinearMap::Rotation(m) = Matrix::rotation_3d(angle, axis);
            assert_eq!(m, Matrix::<f64, 3>::identity(), "axis = {}", axis);
        }
    }

    #[test]
    fn rotation_3d_around_z_90_degrees_same_as_2d_in_xy() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::rotation_3d(angle, 2);
        let v = Vector::new([1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords()[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_3d_around_x_90_degrees_maps_y_to_z() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::rotation_3d(angle, 0);
        let v = Vector::new([0.0, 1.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_3d_around_y_90_degrees_maps_z_to_x() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::rotation_3d(angle, 1);
        let v = Vector::new([0.0, 0.0, 1.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 1.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[2] - 0.0).abs() < 1e-10);
    }

    // --- Rotation N-D (ángulo + dos ejes; panic si eje < 0 o > N-1) --------------------------

    #[test]
    fn rotation_nd_returns_linear_map_rotation_2d() {
        let angle = Angle::<f64>::from_radians(0.0);
        let r = Matrix::<f64, 2>::rotation(angle, 0, 1);
        let LinearMap::Rotation(_) = r;
    }

    #[test]
    fn rotation_nd_2d_plane_zero_angle_is_identity() {
        let angle = Angle::<f64>::from_radians(0.0);
        let LinearMap::Rotation(m) = Matrix::<f64, 2>::rotation(angle, 0, 1);
        assert_eq!(m, Matrix::<f64, 2>::identity());
    }

    #[test]
    fn rotation_nd_3d_plane_01_90_degrees_same_as_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::<f64, 3>::rotation(angle, 0, 1);
        let v = Vector::new([1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords()[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_nd_4d_plane_02_identity_on_axes_1_and_3() {
        let angle = Angle::<f64>::from_radians(0.5);
        let LinearMap::Rotation(m) = Matrix::<f64, 4>::rotation(angle, 0, 2);
        let v = Vector::new([0.0, 1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert_eq!(rotated.coords(), &[0.0, 1.0, 0.0, 0.0]);
        let v = Vector::new([0.0, 0.0, 0.0, 1.0]);
        let rotated = m * v;
        assert_eq!(rotated.coords(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rotation_nd_4d_plane_01_rotates_first_two_coords() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let LinearMap::Rotation(m) = Matrix::<f64, 4>::rotation(angle, 0, 1);
        let v = Vector::new([1.0, 0.0, 5.0, 10.0]);
        let rotated = m * v;
        assert!((rotated.coords()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords()[2] - 5.0).abs() < 1e-10);
        assert!((rotated.coords()[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_first_axis_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 3>::rotation(angle, 3, 1);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_second_axis_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 3>::rotation(angle, 0, 3);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_both_axes_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 2>::rotation(angle, 2, 2);
    }
}
