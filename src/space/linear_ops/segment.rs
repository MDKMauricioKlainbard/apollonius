//! Linear and affine actions on [`Segment`].
//!
//! Operations follow the convention **transform × primitive** only (e.g. `Matrix * Segment`,
//! `AffineTransform * Segment`). The reverse order (e.g. `Segment * Matrix`) is not implemented,
//! so algebra-style notation is consistent and the compiler rejects invalid usage.
//!
//! Matrices and affine transforms with any tag ([`General`](crate::algebra::matrix::General),
//! [`Affine`](crate::algebra::matrix::Affine), [`Isometry`](crate::algebra::matrix::Isometry)) may
//! act on segments: both endpoints are transformed as points.

use std::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    algebra::matrix::MatrixTag,
    space::AffineTransform,
    Matrix, Segment, Vector,
};

/// **Matrix × Segment** (linear action).
///
/// Both endpoints are transformed by the matrix. Defined for any matrix tag.
///
/// # Example
///
/// ```
/// use apollonius::{Angle, Isometry, Matrix, Point, Segment, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let rot = Matrix::<f64, 2, Isometry>::rotation_2d(Angle::<f64>::from_radians(FRAC_PI_2));
/// let seg = Segment::new(Point::new([1.0, 0.0]), Point::new([2.0, 0.0]));
/// let out = rot * seg;
/// // (1,0) -> (0,1); (2,0) -> (0,2)
/// assert!((out.start().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.start().coords_ref()[1] - 1.0).abs() < 1e-10);
/// assert!((out.end().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.end().coords_ref()[1] - 2.0).abs() < 1e-10);
/// ```
impl<T, const N: usize, Tag> Mul<Segment<T, N>> for Matrix<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Segment<T, N>;
    fn mul(self, rhs: Segment<T, N>) -> Self::Output {
        Segment::new(self * rhs.start(), self * rhs.end())
    }
}

/// **AffineTransform × Segment** (affine action).
///
/// Both endpoints are mapped as `linear * point + translation`.
///
/// # Example
///
/// ```
/// use apollonius::{AffineTransform, Isometry, Matrix, Point, Segment, Vector};
///
/// let tr = AffineTransform::new(
///     Matrix::<f64, 2, Isometry>::identity(),
///     Vector::new([1.0, 2.0]),
/// );
/// let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([1.0, 0.0]));
/// let out = tr * seg;
/// assert_eq!(out.start().coords_ref(), &[1.0, 2.0]);
/// assert_eq!(out.end().coords_ref(), &[2.0, 2.0]);
/// ```
impl<T, const N: usize, Tag> Mul<Segment<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Segment<T, N>;
    fn mul(self, rhs: Segment<T, N>) -> Self::Output {
        Segment::new(self * rhs.start(), self * rhs.end())
    }
}

/// **Segment + Vector** (translation).
///
/// Adds the vector to both endpoints. Equivalent to translating the segment.
///
/// # Example
///
/// ```
/// use apollonius::{Point, Segment, Vector};
///
/// let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([1.0, 0.0]));
/// let out = seg + Vector::new([3.0, 4.0]);
/// assert_eq!(out.start().coords_ref(), &[3.0, 4.0]);
/// assert_eq!(out.end().coords_ref(), &[4.0, 4.0]);
/// ```
impl<T, const N: usize> Add<Vector<T, N>> for Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    type Output = Segment<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Segment::new(self.start() + rhs, self.end() + rhs)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use crate::algebra::matrix::{General, Isometry};
    use crate::space::AffineTransform;
    use crate::{Angle, Matrix, Point, Segment, Vector};

    fn seg_2d_unit_x() -> Segment<f64, 2> {
        Segment::new(Point::new([0.0, 0.0]), Point::new([1.0, 0.0]))
    }

    // --- Matrix × Segment ---------------------------------------------------

    #[test]
    fn mul_matrix_identity_preserves_segment_2d() {
        let i = Matrix::<f64, 2, Isometry>::identity();
        let seg = seg_2d_unit_x();
        let out = i * seg;
        assert_eq!(out.start().coords_ref(), seg.start().coords_ref());
        assert_eq!(out.end().coords_ref(), seg.end().coords_ref());
    }

    #[test]
    fn mul_matrix_identity_preserves_segment_3d() {
        let i = Matrix::<f64, 3, Isometry>::identity();
        let seg = Segment::new(
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 1.0, 1.0]),
        );
        let out = i * seg;
        assert_eq!(out.start().coords_ref(), seg.start().coords_ref());
        assert_eq!(out.end().coords_ref(), seg.end().coords_ref());
    }

    #[test]
    fn mul_matrix_rotation_2d_rotates_endpoints() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let r = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let seg = seg_2d_unit_x();
        let out = r * seg;
        // (0,0) -> (0,0); (1,0) -> (0,1)
        assert_eq!(out.start().coords_ref(), &[0.0, 0.0]);
        assert!((out.end().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.end().coords_ref()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mul_matrix_general_2d_scales_length() {
        let m = Matrix::<f64, 2, General>::new([[2.0, 0.0], [0.0, 2.0]]);
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([1.0, 0.0]));
        let out = m * seg;
        assert_eq!(out.start().coords_ref(), &[0.0, 0.0]);
        assert_eq!(out.end().coords_ref(), &[2.0, 0.0]);
        assert!((out.length() - 2.0).abs() < 1e-10);
    }

    // --- Segment + Vector ---------------------------------------------------

    #[test]
    fn add_vector_translates_both_endpoints() {
        let seg = seg_2d_unit_x();
        let v = Vector::new([5.0, 10.0]);
        let out = seg + v;
        assert_eq!(out.start().coords_ref(), &[5.0, 10.0]);
        assert_eq!(out.end().coords_ref(), &[6.0, 10.0]);
    }

    #[test]
    fn add_vector_3d() {
        let seg = Segment::new(
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
        );
        let v = Vector::new([1.0, 2.0, 3.0]);
        let out = seg + v;
        assert_eq!(out.start().coords_ref(), &[1.0, 2.0, 3.0]);
        assert_eq!(out.end().coords_ref(), &[2.0, 2.0, 3.0]);
    }

    // --- AffineTransform × Segment ------------------------------------------

    #[test]
    fn mul_affine_identity_plus_translation_moves_endpoints() {
        let tr = AffineTransform::new(
            Matrix::<f64, 2, Isometry>::identity(),
            Vector::new([10.0, 20.0]),
        );
        let seg = seg_2d_unit_x();
        let out = tr * seg;
        assert_eq!(out.start().coords_ref(), &[10.0, 20.0]);
        assert_eq!(out.end().coords_ref(), &[11.0, 20.0]);
    }

    #[test]
    fn mul_affine_rotation_and_translation_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let linear = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tr = AffineTransform::new(linear, Vector::new([1.0, 0.0]));
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([1.0, 0.0]));
        let out = tr * seg;
        // start (0,0) -> (1,0); end (1,0) -> (0,1)+(1,0)=(1,1)
        assert_eq!(out.start().coords_ref(), &[1.0, 0.0]);
        assert!((out.end().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.end().coords_ref()[1] - 1.0).abs() < 1e-10);
    }
}
