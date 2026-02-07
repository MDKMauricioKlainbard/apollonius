//! Linear and affine actions on [`Triangle`].
//!
//! Operations follow the convention **transform × primitive** only (e.g. `Matrix * Triangle`,
//! `AffineTransform * Triangle`). The reverse order (e.g. `Triangle * Matrix`) is not implemented,
//! so algebra-style notation is consistent and the compiler rejects invalid usage.
//!
//! Matrices and affine transforms with any tag ([`General`](crate::algebra::matrix::General),
//! [`Affine`](crate::algebra::matrix::Affine), [`Isometry`](crate::algebra::matrix::Isometry)) may
//! act on triangles; each vertex is transformed by the linear (or affine) map.

use std::ops::{Add, Mul};

use num_traits::Float;

use crate::{
    algebra::matrix::MatrixTag,
    space::AffineTransform,
    Matrix, Triangle, Vector,
};

/// **Matrix × Triangle** (linear action).
///
/// Each vertex is transformed by the matrix; the result is a triangle with vertices
/// `(matrix * a, matrix * b, matrix * c)`. Defined for any matrix tag
/// ([`General`](crate::algebra::matrix::General), [`Affine`](crate::algebra::matrix::Affine),
/// [`Isometry`](crate::algebra::matrix::Isometry)).
///
/// # Example
///
/// ```
/// use apollonius::{Angle, Isometry, Matrix, Point, Triangle};
/// use std::f64::consts::FRAC_PI_2;
///
/// let rot = Matrix::<f64, 2, Isometry>::rotation_2d(Angle::<f64>::from_radians(FRAC_PI_2));
/// let tri = Triangle::new([
///     Point::new([1.0, 0.0]),
///     Point::new([2.0, 0.0]),
///     Point::new([1.0, 1.0]),
/// ]);
/// let out = rot * tri;
/// // (1,0) -> (0,1); (2,0) -> (0,2); (1,1) -> (-1,1)
/// assert!((out.a().coords_ref()[0] - 0.0).abs() < 1e-10);
/// assert!((out.a().coords_ref()[1] - 1.0).abs() < 1e-10);
/// ```
impl<T, const N: usize, Tag> Mul<Triangle<T, N>> for Matrix<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Triangle<T, N>;
    fn mul(self, rhs: Triangle<T, N>) -> Self::Output {
        let (a, b, c) = rhs.vertices();
        Triangle::from((self * a, self * b, self * c))
    }
}

/// **AffineTransform × Triangle** (affine action).
///
/// Each vertex is mapped as `linear * vertex + translation`. Defined for any transform tag.
///
/// # Example
///
/// ```
/// use apollonius::{AffineTransform, Isometry, Matrix, Point, Triangle, Vector};
///
/// let tr = AffineTransform::new(
///     Matrix::<f64, 2, Isometry>::identity(),
///     Vector::new([1.0, 2.0]),
/// );
/// let tri = Triangle::new([
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ]);
/// let out = tr * tri;
/// assert_eq!(out.a().coords_ref(), &[1.0, 2.0]);
/// assert_eq!(out.b().coords_ref(), &[2.0, 2.0]);
/// assert_eq!(out.c().coords_ref(), &[1.0, 3.0]);
/// ```
impl<T, const N: usize, Tag> Mul<Triangle<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float + std::iter::Sum,
    Tag: MatrixTag,
{
    type Output = Triangle<T, N>;
    fn mul(self, rhs: Triangle<T, N>) -> Self::Output {
        let (a, b, c) = rhs.vertices();
        Triangle::from((self * a, self * b, self * c))
    }
}

/// **Triangle + Vector** (translation).
///
/// Adds the vector to each vertex; equivalent to translating the triangle.
///
/// # Example
///
/// ```
/// use apollonius::{Point, Triangle, Vector};
///
/// let tri = Triangle::new([
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ]);
/// let out = tri + Vector::new([5.0, 10.0]);
/// assert_eq!(out.a().coords_ref(), &[5.0, 10.0]);
/// assert_eq!(out.b().coords_ref(), &[6.0, 10.0]);
/// assert_eq!(out.c().coords_ref(), &[5.0, 11.0]);
/// ```
impl<T, const N: usize> Add<Vector<T, N>> for Triangle<T, N>
where
    T: Float + std::iter::Sum,
{
    type Output = Triangle<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        let (a, b, c) = self.vertices();
        Triangle::from((a + rhs, b + rhs, c + rhs))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use crate::algebra::matrix::Isometry;
    use crate::space::AffineTransform;
    use crate::{Angle, Matrix, Point, Triangle, Vector};

    fn tri_2d_01() -> Triangle<f64, 2> {
        Triangle::new([
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ])
    }

    // --- Matrix × Triangle (Isometry) ---------------------------------------

    #[test]
    fn mul_matrix_identity_preserves_triangle_2d() {
        let i = Matrix::<f64, 2, Isometry>::identity();
        let tri = tri_2d_01();
        let out = i * tri;
        assert_eq!(out.a().coords_ref(), tri.a().coords_ref());
        assert_eq!(out.b().coords_ref(), tri.b().coords_ref());
        assert_eq!(out.c().coords_ref(), tri.c().coords_ref());
    }

    #[test]
    fn mul_matrix_identity_preserves_triangle_3d() {
        let i = Matrix::<f64, 3, Isometry>::identity();
        let tri = Triangle::new([
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
        ]);
        let out = i * tri;
        assert_eq!(out.a().coords_ref(), tri.a().coords_ref());
        assert_eq!(out.b().coords_ref(), tri.b().coords_ref());
        assert_eq!(out.c().coords_ref(), tri.c().coords_ref());
    }

    #[test]
    fn mul_matrix_rotation_2d_rotates_vertices_preserves_area() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let r = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tri = tri_2d_01();
        let out = r * tri;
        // (0,0) -> (0,0); (1,0) -> (0,1); (0,1) -> (-1,0)
        assert!((out.a().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.a().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((out.b().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.b().coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((out.c().coords_ref()[0] - (-1.0)).abs() < 1e-10);
        assert!((out.c().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!(
            (out.area() - tri.area()).abs() < 1e-10,
            "isometry must preserve area"
        );
    }

    // --- Triangle + Vector --------------------------------------------------

    #[test]
    fn add_vector_translates_all_vertices() {
        let tri = tri_2d_01();
        let v = Vector::new([5.0, 10.0]);
        let out = tri + v;
        assert_eq!(out.a().coords_ref(), &[5.0, 10.0]);
        assert_eq!(out.b().coords_ref(), &[6.0, 10.0]);
        assert_eq!(out.c().coords_ref(), &[5.0, 11.0]);
    }

    #[test]
    fn add_vector_3d() {
        let tri = Triangle::new([
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ]);
        let v = Vector::new([0.0, 0.0, 1.0]);
        let out = tri + v;
        assert_eq!(out.a().coords_ref(), &[1.0, 0.0, 1.0]);
        assert_eq!(out.b().coords_ref(), &[0.0, 1.0, 1.0]);
        assert_eq!(out.c().coords_ref(), &[0.0, 0.0, 2.0]);
    }

    // --- AffineTransform × Triangle -----------------------------------------

    #[test]
    fn mul_affine_identity_plus_translation_moves_vertices() {
        let tr = AffineTransform::new(
            Matrix::<f64, 2, Isometry>::identity(),
            Vector::new([10.0, 20.0]),
        );
        let tri = tri_2d_01();
        let out = tr * tri;
        assert_eq!(out.a().coords_ref(), &[10.0, 20.0]);
        assert_eq!(out.b().coords_ref(), &[11.0, 20.0]);
        assert_eq!(out.c().coords_ref(), &[10.0, 21.0]);
    }

    #[test]
    fn mul_affine_rotation_and_translation_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let linear = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let tr = AffineTransform::new(linear, Vector::new([1.0, 0.0]));
        let tri = tri_2d_01();
        let out = tr * tri;
        // a (0,0) -> (0,0)+(1,0)=(1,0); b (1,0) -> (0,1)+(1,0)=(1,1); c (0,1) -> (-1,0)+(1,0)=(0,0)
        assert!((out.a().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.a().coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((out.b().coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((out.b().coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((out.c().coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((out.c().coords_ref()[1] - 0.0).abs() < 1e-10);
    }
}
