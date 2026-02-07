//! Linear and affine actions on [`Hypersphere`].
//!
//! Operations follow the convention **transform × primitive** only (e.g. `Matrix * Hypersphere`,
//! `AffineTransform * Hypersphere`). The reverse order (e.g. `Hypersphere * Matrix`) is not
//! implemented, so algebra-style notation is consistent and the compiler rejects invalid usage.
//!
//! Only **isometric** matrices and affine transforms are allowed to act on hyperspheres, so that
//! radius is preserved. [`IsIsometry`] is required on the matrix/transform tag.

use std::ops::{Add, Mul};

use num_traits::Float;

use crate::{Hypersphere, Matrix, Vector, algebra::matrix::IsIsometry, space::AffineTransform};

/// **Matrix × Hypersphere** (linear action).
///
/// Defined only for matrices with tag implementing [`IsIsometry`] (e.g. [`Isometry`](crate::algebra::matrix::Isometry)).
/// The center is transformed by the matrix; the radius is unchanged (isometries preserve distance).
///
/// # Example
///
/// ```
/// use apollonius::{Angle, Hypersphere, Isometry, Matrix, Point};
///
/// let rot = Matrix::<f64, 2, Isometry>::rotation_2d(Angle::<f64>::from_radians(std::f64::consts::FRAC_PI_2));
/// let circle = Hypersphere::new(Point::new([1.0, 0.0]), 2.0);
/// let rotated = rot * circle;
/// assert_eq!(rotated.radius(), 2.0);
/// assert!((rotated.center().coords_ref()[0] - 0.0_f64).abs() < 1e-10);
/// assert!((rotated.center().coords_ref()[1] - 1.0_f64).abs() < 1e-10);
/// ```
impl<T, const N: usize, Tag> Mul<Hypersphere<T, N>> for Matrix<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: IsIsometry,
{
    type Output = Hypersphere<T, N>;
    fn mul(self, rhs: Hypersphere<T, N>) -> Self::Output {
        Hypersphere::new(self * rhs.center(), rhs.radius())
    }
}

/// **Hypersphere + Vector** (translation by a vector).
///
/// Adds the vector to the center; the radius is unchanged. Equivalent to translating the hypersphere.
///
/// # Example
///
/// ```
/// use apollonius::{Hypersphere, Point, Vector};
///
/// let s = Hypersphere::new(Point::new([0.0, 0.0]), 1.0);
/// let t = s + Vector::new([3.0, 4.0]);
/// assert_eq!(t.center().coords_ref(), &[3.0, 4.0]);
/// assert_eq!(t.radius(), 1.0);
/// ```
impl<T, const N: usize> Add<Vector<T, N>> for Hypersphere<T, N>
where
    T: Float + std::iter::Sum,
{
    type Output = Hypersphere<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Hypersphere::new(self.center() + rhs, self.radius())
    }
}

/// **AffineTransform × Hypersphere** (affine action).
///
/// Defined only for affine transforms whose linear part has tag implementing [`IsIsometry`].
/// The center is mapped as `linear * center + translation`; the radius is unchanged.
///
/// # Example
///
/// ```
/// use apollonius::{AffineTransform, Hypersphere, Isometry, Matrix, Point, Vector};
///
/// let linear = Matrix::<f64, 2, Isometry>::identity();
/// let tr = AffineTransform::new(linear, Vector::new([1.0, 2.0]));
/// let circle = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
/// let moved = tr * circle;
/// assert_eq!(moved.center().coords_ref(), &[1.0, 2.0]);
/// assert_eq!(moved.radius(), 5.0);
/// ```
impl<T, const N: usize, Tag> Mul<Hypersphere<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: IsIsometry,
{
    type Output = Hypersphere<T, N>;
    fn mul(self, rhs: Hypersphere<T, N>) -> Self::Output {
        Hypersphere::new(self * rhs.center(), rhs.radius())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use crate::algebra::matrix::Isometry;
    use crate::space::AffineTransform;
    use crate::{Angle, Hypersphere, Matrix, Point, Vector};

    // --- Matrix × Hypersphere (Isometry) ------------------------------------

    #[test]
    fn mul_matrix_identity_preserves_hypersphere_2d() {
        let i = Matrix::<f64, 2, Isometry>::identity();
        let s = Hypersphere::new(Point::new([3.0, 4.0]), 5.0);
        let out = i * s;
        assert_eq!(out.center().coords_ref(), s.center().coords_ref());
        assert_eq!(out.radius(), s.radius());
    }

    #[test]
    fn mul_matrix_identity_preserves_hypersphere_3d() {
        let i = Matrix::<f64, 3, Isometry>::identity();
        let s = Hypersphere::new(Point::new([1.0, 2.0, 3.0]), 1.5);
        let out = i * s;
        assert_eq!(out.center().coords_ref(), s.center().coords_ref());
        assert_eq!(out.radius(), s.radius());
    }

    #[test]
    fn mul_matrix_rotation_2d_preserves_radius() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let r = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let s = Hypersphere::new(Point::new([1.0, 0.0]), 2.0);
        let out = r * s;
        assert_eq!(out.radius(), 2.0);
        // (1,0) rotated 90° CCW -> (0, 1)
        assert!((out.center().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.center().coords_ref()[1] - 1.0).abs() < 1e-10);
    }

    // --- Hypersphere + Vector ------------------------------------------------

    #[test]
    fn add_vector_translates_center_preserves_radius() {
        let s = Hypersphere::new(Point::new([0.0, 0.0]), 3.0);
        let v = Vector::new([1.0, 2.0]);
        let out = s + v;
        assert_eq!(out.center().coords_ref(), &[1.0, 2.0]);
        assert_eq!(out.radius(), 3.0);
    }

    #[test]
    fn add_vector_3d() {
        let s = Hypersphere::new(Point::new([1.0, 0.0, -1.0]), 1.0);
        let v = Vector::new([0.0, 1.0, 1.0]);
        let out = s + v;
        assert_eq!(out.center().coords_ref(), &[1.0, 1.0, 0.0]);
        assert_eq!(out.radius(), 1.0);
    }

    // --- AffineTransform × Hypersphere --------------------------------------

    #[test]
    fn mul_affine_identity_plus_translation_moves_center() {
        let tr = AffineTransform::new(
            Matrix::<f64, 2, Isometry>::identity(),
            Vector::new([10.0, 20.0]),
        );
        let s = Hypersphere::new(Point::new([0.0, 0.0]), 5.0);
        let out = tr * s;
        assert_eq!(out.center().coords_ref(), &[10.0, 20.0]);
        assert_eq!(out.radius(), 5.0);
    }

    #[test]
    fn mul_affine_rotation_and_translation_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let linear = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tr = AffineTransform::new(linear, Vector::new([1.0, 0.0]));
        let s = Hypersphere::new(Point::new([0.0, 0.0]), 1.0);
        let out = tr * s;
        // center (0,0) -> rotate (0,0) -> (0,0) + translation (1,0) = (1, 0)
        assert!((out.center().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.center().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert_eq!(out.radius(), 1.0);
    }
}
