//! Linear actions of matrices on points and vectors.
//!
//! This module defines the product of a [`Matrix`] with a [`Point`] or [`Vector`]: the operand is
//! treated as a column vector, and the result has the same dimension. These operations are
//! available for any matrix tag ([`General`](crate::algebra::matrix::General),
//! [`Isometry`](crate::algebra::matrix::Isometry), [`Affine`](crate::algebra::matrix::Affine)).

mod hypersphere;
mod hyperplane;
mod line;
mod segment;
mod triangle;

use crate::algebra::matrix::MatrixTag;
use crate::{Matrix, Point, Vector};
use num_traits::Float;
use std::ops::Mul;

/// Matrix × point (point as column). Returns a [`Point`] of the same dimension.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::General;
/// use apollonius::{Matrix, Point};
///
/// let m = Matrix::<f64, 2, General>::identity();
/// let p = Point::new([5.0, 10.0]);
/// assert_eq!(m * p, p);
/// ```
impl<T, const N: usize, Tag> Mul<Point<T, N>> for Matrix<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    type Output = Point<T, N>;
    fn mul(self, rhs: Point<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| {
            let mut sum = T::zero();
            for j in 0..N {
                sum = sum + self.data_ref()[i][j] * rhs.coords_ref()[j];
            }
            sum
        });
        Point::new(coords)
    }
}

/// Matrix × vector (vector as column). Returns a [`Vector`] of the same dimension.
///
/// # Example
///
/// ```
/// use apollonius::algebra::matrix::General;
/// use apollonius::{Matrix, Vector};
///
/// let m = Matrix::<_, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
/// let v = Vector::new([1.0, 1.0]);
/// let w = m * v;
/// assert_eq!(w.coords_ref(), &[3.0, 7.0]);
/// ```
impl<T, const N: usize, Tag> Mul<Vector<T, N>> for Matrix<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    type Output = Vector<T, N>;
    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        let coords = std::array::from_fn(|i| {
            let mut sum = T::zero();
            for j in 0..N {
                sum = sum + self.data_ref()[i][j] * rhs.coords_ref()[j];
            }
            sum
        });
        Vector::new(coords)
    }
}

#[cfg(test)]
mod tests {
    use crate::algebra::matrix::General;
    use crate::{Matrix, Point, Vector};

    // --- Matrix × Vector ----------------------------------------------------

    #[test]
    fn mul_vector_identity_preserves_vector_2d() {
        let i = Matrix::<f64, 2, General>::identity();
        let v = Vector::new([3.0, 4.0]);
        assert_eq!(i * v, v);
    }

    #[test]
    fn mul_vector_identity_preserves_vector_3d() {
        let i = Matrix::<f64, 3, General>::identity();
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(i * v, v);
    }

    #[test]
    fn mul_vector_2x2_first_column() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([1.0, 0.0]);
        let out = m * v;
        assert_eq!(out.coords_ref(), &[1.0, 3.0]);
    }

    #[test]
    fn mul_vector_2x2_second_column() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([0.0, 1.0]);
        let out = m * v;
        assert_eq!(out.coords_ref(), &[2.0, 4.0]);
    }

    #[test]
    fn mul_vector_2x2_general() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vector::new([2.0, 1.0]);
        let out = m * v;
        assert_eq!(out.coords_ref(), &[4.0, 10.0]);
    }

    #[test]
    fn mul_vector_3x3() {
        let m = Matrix::<f64, 3, General>::new([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]);
        let v = Vector::new([1.0, 2.0, 3.0]);
        let out = m * v;
        assert_eq!(out.coords_ref(), &[4.0, 5.0, 3.0]);
    }

    // --- Matrix × Point -----------------------------------------------------

    #[test]
    fn mul_point_identity_preserves_point_2d() {
        let i = Matrix::<f64, 2, General>::identity();
        let p = Point::new([3.0, 4.0]);
        assert_eq!(i * p, p);
    }

    #[test]
    fn mul_point_identity_preserves_point_3d() {
        let i = Matrix::<f64, 3, General>::identity();
        let p = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(i * p, p);
    }

    #[test]
    fn mul_point_2x2() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let p = Point::new([1.0, 0.0]);
        let out = m * p;
        assert_eq!(out.coords_ref(), &[1.0, 3.0]);
    }

    #[test]
    fn mul_point_2x2_general() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let p = Point::new([2.0, 1.0]);
        let out = m * p;
        assert_eq!(out.coords_ref(), &[4.0, 10.0]);
    }

    #[test]
    fn mul_point_3x3() {
        let m = Matrix::<f64, 3, General>::new([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]);
        let p = Point::new([1.0, 2.0, 1.0]);
        let out = m * p;
        assert_eq!(out.coords_ref(), &[3.0, 5.0, 1.0]);
    }
}
