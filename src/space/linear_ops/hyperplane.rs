//! Linear and affine actions on [`Hyperplane`].
//!
//! Operations follow the convention **transform × primitive** only (e.g. `Matrix * Hyperplane`,
//! `AffineTransform * Hyperplane`). The reverse order (e.g. `Hyperplane * Matrix`) is not
//! implemented, so algebra-style notation is consistent and the compiler rejects invalid usage.
//!
//! Matrices and affine transforms with any tag ([`General`](crate::algebra::matrix::General),
//! [`Affine`](crate::algebra::matrix::Affine), [`Isometry`](crate::algebra::matrix::Isometry)) may
//! act on hyperplanes: the origin is transformed as a point, and the normal as a vector (the
//! resulting normal is re-normalized by [`Hyperplane::new`](crate::Hyperplane::new)).

use std::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    algebra::matrix::MatrixTag,
    space::AffineTransform,
    Hyperplane, Matrix, Vector,
};

/// **Matrix × Hyperplane** (linear action).
///
/// The origin is transformed by the matrix; the normal is transformed as a vector (then
/// re-normalized). Defined for any matrix tag.
///
/// # Example
///
/// ```
/// use apollonius::{Angle, Hyperplane, Isometry, Matrix, Point, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let rot = Matrix::<f64, 2, Isometry>::rotation_2d(Angle::<f64>::from_radians(FRAC_PI_2));
/// let plane = Hyperplane::new(Point::new([1.0, 0.0]), Vector::new([1.0, 0.0]));
/// let out = rot * plane;
/// // origin (1,0) -> (0,1); normal (1,0) -> (0,1)
/// assert!((out.origin().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.origin().coords_ref()[1] - 1.0).abs() < 1e-10);
/// assert!((out.normal().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.normal().coords_ref()[1] - 1.0).abs() < 1e-10);
/// ```
impl<T, const N: usize, Tag> Mul<Hyperplane<T, N>> for Matrix<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Hyperplane<T, N>;
    fn mul(self, rhs: Hyperplane<T, N>) -> Self::Output {
        Hyperplane::new(self * rhs.origin(), self * rhs.normal())
    }
}

/// **AffineTransform × Hyperplane** (affine action).
///
/// The origin is mapped as `linear * origin + translation`; the normal is transformed by the
/// linear part only (then re-normalized).
///
/// # Example
///
/// ```
/// use apollonius::{AffineTransform, Hyperplane, Isometry, Matrix, Point, Vector};
///
/// let tr = AffineTransform::new(
///     Matrix::<f64, 2, Isometry>::identity(),
///     Vector::new([1.0, 2.0]),
/// );
/// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]));
/// let out = tr * plane;
/// assert_eq!(out.origin().coords_ref(), &[1.0, 2.0]);
/// assert_eq!(out.normal().coords_ref(), &[0.0, 1.0]);
/// ```
impl<T, const N: usize, Tag> Mul<Hyperplane<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Hyperplane<T, N>;
    fn mul(self, rhs: Hyperplane<T, N>) -> Self::Output {
        Hyperplane::new(self * rhs.origin(), self * rhs.normal())
    }
}

/// **Hyperplane + Vector** (translation).
///
/// Adds the vector to the origin; the normal is unchanged. Equivalent to translating the hyperplane.
///
/// # Example
///
/// ```
/// use apollonius::{Hyperplane, Point, Vector};
///
/// let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]));
/// let out = plane + Vector::new([3.0, 4.0]);
/// assert_eq!(out.origin().coords_ref(), &[3.0, 4.0]);
/// assert_eq!(out.normal().coords_ref(), &[0.0, 1.0]);
/// ```
impl<T, const N: usize> Add<Vector<T, N>> for Hyperplane<T, N>
where
    T: Float + std::iter::Sum,
{
    type Output = Hyperplane<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Hyperplane::new(self.origin() + rhs, self.normal())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use crate::algebra::matrix::Isometry;
    use crate::space::AffineTransform;
    use crate::{Angle, Hyperplane, Matrix, Point, Vector};

    fn plane_2d_horizontal() -> Hyperplane<f64, 2> {
        Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([0.0, 1.0]))
    }

    // --- Matrix × Hyperplane ------------------------------------------------

    #[test]
    fn mul_matrix_identity_preserves_hyperplane_2d() {
        let i = Matrix::<f64, 2, Isometry>::identity();
        let plane = plane_2d_horizontal();
        let out = i * plane;
        assert_eq!(out.origin().coords_ref(), plane.origin().coords_ref());
        assert_eq!(out.normal().coords_ref(), plane.normal().coords_ref());
    }

    #[test]
    fn mul_matrix_identity_preserves_hyperplane_3d() {
        let i = Matrix::<f64, 3, Isometry>::identity();
        let plane = Hyperplane::new(
            Point::new([1.0, 2.0, 3.0]),
            Vector::new([0.0, 1.0, 0.0]),
        );
        let out = i * plane;
        assert_eq!(out.origin().coords_ref(), plane.origin().coords_ref());
        assert_eq!(out.normal().coords_ref(), plane.normal().coords_ref());
    }

    #[test]
    fn mul_matrix_rotation_2d_rotates_origin_and_normal() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let r = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let plane = Hyperplane::new(Point::new([1.0, 0.0]), Vector::new([1.0, 0.0]));
        let out = r * plane;
        // origin (1,0) -> (0,1); normal (1,0) -> (0,1)
        assert!((out.origin().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.origin().coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((out.normal().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.normal().coords_ref()[1] - 1.0).abs() < 1e-10);
    }

    // --- Hyperplane + Vector ------------------------------------------------

    #[test]
    fn add_vector_translates_origin_preserves_normal() {
        let plane = plane_2d_horizontal();
        let v = Vector::new([5.0, 10.0]);
        let out = plane + v;
        assert_eq!(out.origin().coords_ref(), &[5.0, 10.0]);
        assert_eq!(out.normal().coords_ref(), &[0.0, 1.0]);
    }

    #[test]
    fn add_vector_3d() {
        let plane = Hyperplane::new(
            Point::new([0.0, 0.0, 0.0]),
            Vector::new([1.0, 0.0, 0.0]),
        );
        let v = Vector::new([1.0, 2.0, 3.0]);
        let out = plane + v;
        assert_eq!(out.origin().coords_ref(), &[1.0, 2.0, 3.0]);
        assert_eq!(out.normal().coords_ref(), &[1.0, 0.0, 0.0]);
    }

    // --- AffineTransform × Hyperplane ---------------------------------------

    #[test]
    fn mul_affine_identity_plus_translation_moves_origin() {
        let tr = AffineTransform::new(
            Matrix::<f64, 2, Isometry>::identity(),
            Vector::new([10.0, 20.0]),
        );
        let plane = plane_2d_horizontal();
        let out = tr * plane;
        assert_eq!(out.origin().coords_ref(), &[10.0, 20.0]);
        assert_eq!(out.normal().coords_ref(), &[0.0, 1.0]);
    }

    #[test]
    fn mul_affine_rotation_and_translation_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let linear = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tr = AffineTransform::new(linear, Vector::new([1.0, 0.0]));
        let plane = Hyperplane::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let out = tr * plane;
        // origin (0,0) -> (0,0)+(1,0)=(1,0); normal (1,0) -> (0,1)
        assert!((out.origin().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.origin().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((out.normal().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.normal().coords_ref()[1] - 1.0).abs() < 1e-10);
    }
}
