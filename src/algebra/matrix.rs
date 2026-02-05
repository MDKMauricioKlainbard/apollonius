//! Square matrices over floating-point types with linear algebra operations.
//!
//! This module provides an N×N [`Matrix`] type parameterised by element type and dimension, with
//! support for identity, addition, subtraction, matrix multiplication, and multiplication by
//! [`Point`] and [`Vector`]. Element type is constrained by [`Float`](num_traits::Float) so that
//! zero and one are available for identity and products.

use crate::{Point, Vector};
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
        let data = std::array::from_fn(|_| {
            std::array::from_fn(|_| it.next().expect("N*N elements"))
        });
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
    use crate::{Point, Vector};

    fn identity_2() -> Matrix<f64, 2> {
        Matrix::new([[1.0, 0.0], [0.0, 1.0]])
    }
    fn identity_3() -> Matrix<f64, 3> {
        Matrix::new([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
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
        let a = Matrix::new([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let b = Matrix::new([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]);
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
        let m = Matrix::new([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]);
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
        let m = Matrix::new([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0],
        ]);
        let p = Point::new([1.0, 2.0, 1.0]);
        let out = m * p;
        assert_eq!(out.coords(), &[3.0, 5.0, 1.0]);
    }
}
