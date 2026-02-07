//! Affine transformations and their actions on geometry.
//!
//! * **[`AffineTransform`]** — Linear part plus translation; supports composition (`*`), addition,
//!   and subtraction (for `General` tag).
//! * **[`linear_ops`]** — Implements matrix × [`Point`](crate::Point) and matrix ×
//!   [`Vector`](crate::Vector) (linear action of a matrix on points and vectors).

mod linear_ops;

use std::ops::{Add, Mul, Sub};

use crate::algebra::matrix::{General, MatrixTag, MulOutput};
use crate::algebra::points::Point;
use crate::{Matrix, Vector};
use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct AffineTransform<T, const N: usize, Tag>
where
    Tag: MatrixTag,
{
    linear: Matrix<T, N, Tag>,
    translation: Vector<T, N>,
}

impl<T: Float, const N: usize, Tag> AffineTransform<T, N, Tag>
where
    Tag: MatrixTag,
{
    pub fn new(linear: Matrix<T, N, Tag>, translation: Vector<T, N>) -> Self {
        Self {
            linear,
            translation,
        }
    }

    pub fn linear(&self) -> &Matrix<T, N, Tag> {
        &self.linear
    }

    pub fn translation(&self) -> &Vector<T, N> {
        &self.translation
    }

    pub fn set_linear(&mut self, linear: Matrix<T, N, Tag>) {
        self.linear = linear;
    }

    pub fn set_translation(&mut self, translation: Vector<T, N>) {
        self.translation = translation;
    }

    pub fn linear_mut(&mut self) -> &mut Matrix<T, N, Tag> {
        &mut self.linear
    }

    pub fn translation_mut(&mut self) -> &mut Vector<T, N> {
        &mut self.translation
    }
}

impl<T, const N: usize> Add for AffineTransform<T, N, General>
where
    T: Float,
{
    type Output = AffineTransform<T, N, General>;
    fn add(self, rhs: Self) -> Self::Output {
        AffineTransform {
            linear: self.linear + rhs.linear,
            translation: self.translation + rhs.translation,
        }
    }
}

impl<T, const N: usize> Sub for AffineTransform<T, N, General>
where
    T: Float,
{
    type Output = AffineTransform<T, N, General>;
    fn sub(self, rhs: Self) -> Self::Output {
        AffineTransform {
            linear: self.linear - rhs.linear,
            translation: self.translation - rhs.translation,
        }
    }
}

impl<T, const N: usize, Tag> Mul<Point<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    type Output = Point<T, N>;

    fn mul(self, rhs: Point<T, N>) -> Self::Output {
        self.linear * rhs + self.translation
    }
}

impl<T, const N: usize, Tag> Mul<Vector<T, N>> for AffineTransform<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    type Output = Vector<T, N>;

    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        self.linear * rhs
    }
}

impl<T, const N: usize, LTag, RTag> Mul<AffineTransform<T, N, RTag>> for AffineTransform<T, N, LTag>
where
    T: Float,
    LTag: MatrixTag + MulOutput<RTag>,
    RTag: MatrixTag,
{
    type Output = AffineTransform<T, N, LTag::ResultTag>;
    fn mul(self, rhs: AffineTransform<T, N, RTag>) -> Self::Output {
        let linear = self.linear * rhs.linear;
        let translation = (self.linear * rhs.translation) + self.translation;
        AffineTransform {
            linear,
            translation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AffineTransform;
    use crate::algebra::matrix::General;
    use crate::{Matrix, Point, Vector};

    fn identity_affine_2d() -> AffineTransform<f64, 2, General> {
        AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([0.0, 0.0]),
        )
    }

    fn zero_affine_2d() -> AffineTransform<f64, 2, General> {
        AffineTransform::new(
            Matrix::<f64, 2, General>::new([[0.0, 0.0], [0.0, 0.0]]),
            Vector::new([0.0, 0.0]),
        )
    }

    // --- Addition ----------------------------------------------------------------------------

    #[test]
    fn add_zero_left_2d() {
        let z = zero_affine_2d();
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 2.0]),
        );
        let sum = z + a;
        assert_eq!(*sum.linear(), *a.linear());
        assert_eq!(sum.translation().coords_ref(), a.translation().coords_ref());
    }

    #[test]
    fn add_zero_right_2d() {
        let z = zero_affine_2d();
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 2.0]),
        );
        let sum = a + z;
        assert_eq!(*sum.linear(), *a.linear());
        assert_eq!(sum.translation().coords_ref(), a.translation().coords_ref());
    }

    #[test]
    fn add_two_translations_2d() {
        let t1 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 0.0]),
        );
        let t2 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([0.0, 2.0]),
        );
        let sum = t1 + t2;
        let expected_linear =
            Matrix::<f64, 2, General>::identity() + Matrix::<f64, 2, General>::identity();
        assert_eq!(*sum.linear(), expected_linear);
        assert_eq!(sum.translation().coords_ref(), &[1.0, 2.0]);
    }

    #[test]
    fn add_commutative_2d() {
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]),
            Vector::new([1.0, 2.0]),
        );
        let b = AffineTransform::new(
            Matrix::<f64, 2, General>::new([[0.0, 1.0], [1.0, 0.0]]),
            Vector::new([3.0, 4.0]),
        );
        assert_eq!((a + b).linear().data_ref(), (b + a).linear().data_ref());
        assert_eq!(
            (a + b).translation().coords_ref(),
            (b + a).translation().coords_ref()
        );
    }

    #[test]
    fn add_associative_2d() {
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 0.0]),
        );
        let b = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([0.0, 1.0]),
        );
        let c = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 1.0]),
        );
        let ab_c = (a + b) + c;
        let a_bc = a + (b + c);
        assert_eq!(ab_c.linear().data_ref(), a_bc.linear().data_ref());
        assert_eq!(ab_c.translation().coords_ref(), a_bc.translation().coords_ref());
    }

    // --- Subtraction -------------------------------------------------------------------------

    #[test]
    fn sub_self_is_zero_2d() {
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([5.0, -3.0]),
        );
        let diff = a - a;
        assert_eq!(
            *diff.linear(),
            Matrix::<f64, 2, General>::new([[0.0, 0.0], [0.0, 0.0]])
        );
        assert_eq!(diff.translation().coords_ref(), &[0.0, 0.0]);
    }

    #[test]
    fn sub_inverse_of_add_2d() {
        let a = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 2.0]),
        );
        let b = AffineTransform::new(
            Matrix::<f64, 2, General>::new([[0.5, 0.0], [0.0, 0.5]]),
            Vector::new([0.5, 1.0]),
        );
        let sum = a + b;
        let diff = sum - b;
        assert_eq!(diff.linear().data_ref(), a.linear().data_ref());
        assert_eq!(diff.translation().coords_ref(), a.translation().coords_ref());
    }

    #[test]
    fn sub_two_translations_2d() {
        let t1 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([3.0, 4.0]),
        );
        let t2 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 2.0]),
        );
        let diff = t1 - t2;
        assert_eq!(
            *diff.linear(),
            Matrix::<f64, 2, General>::new([[0.0, 0.0], [0.0, 0.0]])
        );
        assert_eq!(diff.translation().coords_ref(), &[2.0, 2.0]);
    }

    // --- Multiplication: identity ------------------------------------------------------------

    #[test]
    fn mul_identity_left_2d() {
        let id = identity_affine_2d();
        let r = Matrix::<f64, 2, General>::identity(); // placeholder: use rotation when Tag allows
        let t = Vector::new([1.0, 2.0]);
        let a = AffineTransform::new(r, t);
        let product = id * a;
        assert_eq!(product.linear().data_ref(), a.linear().data_ref());
        assert_eq!(product.translation().coords_ref(), a.translation().coords_ref());
    }

    #[test]
    fn mul_identity_right_2d() {
        let id = identity_affine_2d();
        let r = Matrix::<f64, 2, General>::identity();
        let t = Vector::new([3.0, -1.0]);
        let a = AffineTransform::new(r, t);
        let product = a * id;
        assert_eq!(product.linear().data_ref(), a.linear().data_ref());
        assert_eq!(product.translation().coords_ref(), a.translation().coords_ref());
    }

    // --- Multiplication: two translations (identity linear) ----------------------------------

    #[test]
    fn mul_two_translations_2d() {
        let t1 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([1.0, 0.0]),
        );
        let t2 = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([0.0, 2.0]),
        );
        let product = t1 * t2;
        assert_eq!(*product.linear(), Matrix::<f64, 2, General>::identity());
        assert_eq!(product.translation().coords_ref(), &[1.0, 2.0]);
    }

    // --- Multiplication: (L, t_a) * (I, t_b) => linear L, translation L*t_b + t_a -------------

    #[test]
    fn mul_affine_times_translation_translation_part_2d() {
        let r = Matrix::<f64, 2, General>::identity(); // use rotation matrix when available for General
        let t_a = Vector::new([10.0, 0.0]);
        let t_b = Vector::new([1.0, 0.0]);
        let a = AffineTransform::new(r, t_a);
        let b = AffineTransform::new(Matrix::<f64, 2, General>::identity(), t_b);
        let product = a * b;
        let expected_translation = r * t_b + t_a;
        assert_eq!(
            product.translation().coords_ref(),
            expected_translation.coords_ref()
        );
    }

    // --- Apply: (A * B)(p) == A(B(p)) -------------------------------------------------------

    #[test]
    fn apply_composition_equals_apply_apply_2d() {
        let r = Matrix::<f64, 2, General>::identity();
        let a = AffineTransform::new(r, Vector::new([1.0, 2.0]));
        let b = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([3.0, 4.0]),
        );
        let p = Point::new([0.0, 0.0]);
        let ab = a * b;
        let applied_twice = a * (b * p);
        let applied_product = ab * p;
        assert_eq!(applied_product.coords_ref(), applied_twice.coords_ref());
    }

    // --- Associativity of multiplication -----------------------------------------------------

    #[test]
    fn mul_associative_2d() {
        let r = Matrix::<f64, 2, General>::identity();
        let a = AffineTransform::new(r, Vector::new([1.0, 0.0]));
        let b = AffineTransform::new(
            Matrix::<f64, 2, General>::identity(),
            Vector::new([0.0, 1.0]),
        );
        let c = AffineTransform::new(r, Vector::new([0.0, 0.0]));
        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);
        assert_eq!(ab_c.linear().data_ref(), a_bc.linear().data_ref());
        assert_eq!(ab_c.translation().coords_ref(), a_bc.translation().coords_ref());
    }
}
