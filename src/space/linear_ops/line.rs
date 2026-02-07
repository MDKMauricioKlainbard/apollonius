//! Linear and affine actions on [`Line`].
//!
//! Operations follow the convention **transform × primitive** only (e.g. `Matrix * Line`,
//! `AffineTransform * Line`). The reverse order (e.g. `Line * Matrix`) is not implemented,
//! so algebra-style notation is consistent and the compiler rejects invalid usage.
//!
//! Matrices and affine transforms with any tag ([`General`](crate::algebra::matrix::General),
//! [`Affine`](crate::algebra::matrix::Affine), [`Isometry`](crate::algebra::matrix::Isometry)) may
//! act on lines: the origin is transformed as a point, and the direction as a vector (the
//! resulting direction is re-normalized by [`Line::new`](crate::Line::new)).

use std::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    algebra::matrix::MatrixTag,
    space::AffineTransform,
    Line, Matrix, Vector,
};

/// **Matrix × Line** (linear action).
///
/// The origin is transformed by the matrix; the direction is transformed as a vector (then
/// re-normalized). Defined for any matrix tag.
///
/// # Example
///
/// ```
/// use apollonius::{Angle, Isometry, Line, Matrix, Point, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let rot = Matrix::<f64, 2, Isometry>::rotation_2d(Angle::<f64>::from_radians(FRAC_PI_2));
/// let line = Line::new(Point::new([1.0, 0.0]), Vector::new([1.0, 0.0]));
/// let out = rot * line;
/// // origin (1,0) -> (0,1); direction (1,0) -> (0,1)
/// assert!((out.origin().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.origin().coords_ref()[1] - 1.0).abs() < 1e-10);
/// assert!((out.direction().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.direction().coords_ref()[1] - 1.0).abs() < 1e-10);
/// ```
impl<T, const N: usize, Tag> Mul<Line<T, N>> for Matrix<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Line<T, N>;
    fn mul(self, rhs: Line<T, N>) -> Self::Output {
        Line::new(self * rhs.origin(), self * rhs.direction())
    }
}

/// **AffineTransform × Line** (affine action).
///
/// The origin is mapped as `linear * origin + translation`; the direction is transformed by the
/// linear part only (then re-normalized).
///
/// # Example
///
/// ```
/// use apollonius::{AffineTransform, Isometry, Line, Matrix, Point, Vector};
///
/// let tr = AffineTransform::new(
///     Matrix::<f64, 2, Isometry>::identity(),
///     Vector::new([1.0, 2.0]),
/// );
/// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
/// let out = tr * line;
/// assert_eq!(out.origin().coords_ref(), &[1.0, 2.0]);
/// assert_eq!(out.direction().coords_ref(), &[1.0, 0.0]);
/// ```
impl<T, const N: usize, Tag> Mul<Line<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Line<T, N>;
    fn mul(self, rhs: Line<T, N>) -> Self::Output {
        Line::new(self * rhs.origin(), self * rhs.direction())
    }
}

/// **Line + Vector** (translation).
///
/// Adds the vector to the origin; the direction is unchanged. Equivalent to translating the line.
///
/// # Example
///
/// ```
/// use apollonius::{Line, Point, Vector};
///
/// let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
/// let out = line + Vector::new([3.0, 4.0]);
/// assert_eq!(out.origin().coords_ref(), &[3.0, 4.0]);
/// assert_eq!(out.direction().coords_ref(), &[1.0, 0.0]);
/// ```
impl<T, const N: usize> Add<Vector<T, N>> for Line<T, N>
where
    T: Float + std::iter::Sum,
{
    type Output = Line<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Line::new(self.origin() + rhs, self.direction())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use crate::algebra::matrix::{General, Isometry};
    use crate::space::AffineTransform;
    use crate::{Angle, Line, Matrix, Point, Vector};

    fn line_2d_x() -> Line<f64, 2> {
        Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]))
    }

    // --- Matrix × Line ------------------------------------------------------

    #[test]
    fn mul_matrix_identity_preserves_line_2d() {
        let i = Matrix::<f64, 2, Isometry>::identity();
        let line = line_2d_x();
        let out = i * line;
        assert_eq!(out.origin().coords_ref(), line.origin().coords_ref());
        assert_eq!(out.direction().coords_ref(), line.direction().coords_ref());
    }

    #[test]
    fn mul_matrix_identity_preserves_line_3d() {
        let i = Matrix::<f64, 3, Isometry>::identity();
        let line = Line::new(
            Point::new([1.0, 2.0, 3.0]),
            Vector::new([0.0, 1.0, 0.0]),
        );
        let out = i * line;
        assert_eq!(out.origin().coords_ref(), line.origin().coords_ref());
        assert_eq!(out.direction().coords_ref(), line.direction().coords_ref());
    }

    #[test]
    fn mul_matrix_rotation_2d_rotates_origin_and_direction() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let r = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let line = Line::new(Point::new([1.0, 0.0]), Vector::new([1.0, 0.0]));
        let out = r * line;
        // origin (1,0) -> (0,1); direction (1,0) -> (0,1)
        assert!((out.origin().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.origin().coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((out.direction().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.direction().coords_ref()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mul_matrix_general_2d() {
        let m = Matrix::<f64, 2, General>::new([[2.0, 0.0], [0.0, 2.0]]);
        let line = Line::new(Point::new([1.0, 0.0]), Vector::new([1.0, 0.0]));
        let out = m * line;
        assert_eq!(out.origin().coords_ref(), &[2.0, 0.0]);
        // direction (1,0) -> (2,0), then normalized -> (1,0)
        assert_eq!(out.direction().coords_ref(), &[1.0, 0.0]);
    }

    // --- Line + Vector ------------------------------------------------------

    #[test]
    fn add_vector_translates_origin_preserves_direction() {
        let line = line_2d_x();
        let v = Vector::new([5.0, 10.0]);
        let out = line + v;
        assert_eq!(out.origin().coords_ref(), &[5.0, 10.0]);
        assert_eq!(out.direction().coords_ref(), &[1.0, 0.0]);
    }

    #[test]
    fn add_vector_3d() {
        let line = Line::new(
            Point::new([0.0, 0.0, 0.0]),
            Vector::new([1.0, 0.0, 0.0]),
        );
        let v = Vector::new([1.0, 2.0, 3.0]);
        let out = line + v;
        assert_eq!(out.origin().coords_ref(), &[1.0, 2.0, 3.0]);
        assert_eq!(out.direction().coords_ref(), &[1.0, 0.0, 0.0]);
    }

    // --- AffineTransform × Line ---------------------------------------------

    #[test]
    fn mul_affine_identity_plus_translation_moves_origin() {
        let tr = AffineTransform::new(
            Matrix::<f64, 2, Isometry>::identity(),
            Vector::new([10.0, 20.0]),
        );
        let line = line_2d_x();
        let out = tr * line;
        assert_eq!(out.origin().coords_ref(), &[10.0, 20.0]);
        assert_eq!(out.direction().coords_ref(), &[1.0, 0.0]);
    }

    #[test]
    fn mul_affine_rotation_and_translation_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let linear = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tr = AffineTransform::new(linear, Vector::new([1.0, 0.0]));
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let out = tr * line;
        // origin (0,0) -> (0,0)+(1,0)=(1,0); direction (1,0) -> (0,1)
        assert!((out.origin().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.origin().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((out.direction().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.direction().coords_ref()[1] - 1.0).abs() < 1e-10);
    }
}
