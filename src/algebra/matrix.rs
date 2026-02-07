//! N×N matrices with type-level tags for semantic kind (General, Isometry, Affine).
//!
//! This module provides [`Matrix<T, N, Tag>`], an N×N square matrix parameterised by element type
//! `T`, dimension `N`, and a *tag* `Tag` that indicates the kind of linear map it represents. Tags
//! determine which geometric primitives can safely be transformed by the matrix.
//!
//! # Tags and safe application
//!
//! | Tag | Meaning | Safe to apply to |
//! |-----|---------|------------------|
//! | [`General`] | User-defined custom matrix | Vectors, points, hyperplanes, triangles, segments, lines |
//! | [`Affine`] | Invertible affine map (e.g. scaling + rotation) | Vectors, points, triangles, segments, lines, hyperplanes |
//! | [`Isometry`] | Distance-preserving (e.g. rotation) | **All** primitives (vectors, points, lines, segments, hyperplanes, triangles, hyperspheres, AABBs, etc.) |
//!
//! # Operations and result tags
//!
//! * **Addition / subtraction:** Defined for any two matrices (any tags). Result is always
//!   `Matrix<T, N, General>`.
//! * **Matrix × matrix:** Result tag is given by [`MulOutput`]. For example: `Isometry * Isometry`
//!   → `Isometry`; `General * anything` → `General`; `Affine * Affine` → `Affine`.
//! * **Matrix × vector / point:** Implemented in [`space::linear_ops`](crate::space::linear_ops);
//!   defined for any tag; result is `Vector` or `Point`.
//!
//! # Construction
//!
//! * **General:** Use [`new`](Matrix::new) with a row-major array, or [`identity`](Matrix::identity).
//! * **Isometry:** Use [`rotation`](Matrix::rotation), [`rotation_2d`](Matrix::rotation_2d), or
//!   [`rotation_3d`](Matrix::rotation_3d), or [`identity`](Matrix::identity).

use crate::Angle;
use num_traits::Float;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

/// Tag for a user-defined custom matrix. Can be applied safely to vectors, points, hyperplanes,
/// triangles, segments, and lines.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct General;

/// Tag for distance-preserving linear maps (e.g. rotations). Can be applied safely to **all**
/// primitives: vectors, points, lines, segments, hyperplanes, triangles, hyperspheres, AABBs, etc.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Isometry;

/// Tag for invertible affine linear maps (e.g. scaling + rotation). Can be applied safely to
/// vectors, points, triangles, segments, lines, and hyperplanes.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Affine;

/// Marker trait for matrix tags. Implemented by [`General`], [`Isometry`], and [`Affine`].
pub trait MatrixTag: Copy {}

/// Marker for tags that represent affine (invertible) maps. Implemented by [`Affine`] and [`Isometry`].
pub trait IsAffine: MatrixTag {}

/// Marker for tags that represent isometries. Implemented by [`Isometry`].
pub trait IsIsometry: MatrixTag {}

impl MatrixTag for General {}
impl MatrixTag for Isometry {}
impl MatrixTag for Affine {}

impl IsAffine for Affine {}
impl IsAffine for Isometry {}

impl IsIsometry for Isometry {}

/// Determines the tag of the result of matrix × matrix multiplication, so that the result tag
/// reflects the "least safe" of the two operands (e.g. `General * Isometry` → `General`).
///
/// * `Isometry * Isometry` → `Isometry` (safe for all primitives).
/// * `Affine * Affine` → `Affine`; `Affine * General` → `General`.
/// * `General * anything` → `General`.
pub trait MulOutput<RhsTag> {
    /// Tag of `Self * Matrix<T, N, RhsTag>`.
    type ResultTag: MatrixTag;
}

impl<T: MatrixTag> MulOutput<T> for Isometry {
    type ResultTag = T;
}

impl<T: MatrixTag> MulOutput<T> for General {
    type ResultTag = General;
}

impl<T: IsAffine> MulOutput<T> for Affine {
    type ResultTag = Affine;
}

impl MulOutput<General> for Affine {
    type ResultTag = General;
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An N×N square matrix with floating-point elements and a type-level *tag* for semantic kind.
///
/// Storage is row-major: `data[i][j]` is the element in row `i` and column `j`. Matrix × vector
/// and matrix × point products (right-hand side as column vector) are implemented in
/// [`space::linear_ops`](crate::space::linear_ops). The tag determines which geometric primitives
/// can safely be transformed: see the [module-level documentation](self) for the safe-application
/// table ([`General`], [`Affine`], [`Isometry`]).
///
/// # Type parameters
///
/// * `T` — element type, must implement [`Float`](num_traits::Float).
/// * `N` — dimension (matrix size is N×N).
/// * `Tag` — semantic kind: [`General`] (custom), [`Affine`], or [`Isometry`] (safest for all primitives).
///
/// # Serialization
///
/// With the **`serde`** feature enabled, `Matrix` implements `Serialize` and `Deserialize`.
/// The matrix is stored as a single row-major array of `N×N` elements (same pattern as [`Point`]
/// and [`Vector`]).
///
/// # Examples
///
/// General 2×2 matrix:
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let m = Matrix::<_, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
/// assert_eq!(m.data_ref(), &[[1.0, 0.0], [0.0, 1.0]]);
/// ```
///
/// Matrix-vector product (works for any tag):
///
/// ```
/// use apollonius::{General, Matrix, Vector};
///
/// let m = Matrix::<_, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
/// let v = Vector::new([1.0, 0.0]);
/// let out = m * v;
/// assert_eq!(out.coords_ref(), &[1.0, 3.0]);
/// ```
///
/// Identity matrix (available for any tag):
///
/// ```
/// use apollonius::{General, Matrix, Point};
///
/// let m = Matrix::<f64, 2, General>::identity();
/// let p = Point::new([2.0, 3.0]);
/// assert_eq!(m * p, p);
/// ```
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Matrix<T, const N: usize, Tag>
where
    Tag: MatrixTag,
{
    /// Row-major storage: `data[i][j]` is row `i`, column `j`.
    data: [[T; N]; N],
    _marker: PhantomData<Tag>,
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize, Tag> Serialize for Matrix<T, N, Tag> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let flat: Vec<&T> = self.data.iter().flat_map(|row| row.iter()).collect();
        flat.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize, Tag> Deserialize<'de> for Matrix<T, N, Tag> {
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
        Ok(Self {
            data,
            _marker: PhantomData,
        })
    }
}

impl<T, const N: usize, Tag> Matrix<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    #[inline]
    pub(crate) fn from_raw(data: [[T; N]; N]) -> Self {
        Self {
            data,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn multiply_raw(lhs: &[[T; N]; N], rhs: &[[T; N]; N]) -> [[T; N]; N] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let mut sum = T::zero();
                for k in 0..N {
                    sum = sum + lhs[i][k] * rhs[k][j];
                }
                sum
            })
        })
    }
}

impl<T, const N: usize, Tag> Matrix<T, N, Tag>
where
    T: Float,
    Tag: MatrixTag,
{
    /// Returns a copy of the inner row-major data array (field name = by value).
    /// Requires `T: Copy` (e.g. `f32`, `f64`).
    #[inline]
    pub fn data(&self) -> [[T; N]; N]
    where
        T: Copy,
    {
        self.data
    }

    /// Returns a reference to the inner row-major data array.
    #[inline]
    pub fn data_ref(&self) -> &[[T; N]; N] {
        &self.data
    }

    /// Returns the N×N identity matrix (ones on the diagonal, zeros elsewhere).
    /// Available for any tag ([`General`], [`Isometry`], [`Affine`]).
    ///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let i = Matrix::<f64, 3, General>::identity();
    /// assert_eq!(i.data_ref()[0], [1.0, 0.0, 0.0]);
    /// assert_eq!(i.data_ref()[1], [0.0, 1.0, 0.0]);
    /// assert_eq!(i.data_ref()[2], [0.0, 0.0, 1.0]);
    /// ```
    pub fn identity() -> Self {
        let data = std::array::from_fn(|i| {
            let mut row = [T::zero(); N];
            row[i] = T::one();
            row
        });
        Self::from_raw(data)
    }
}

impl<T, const N: usize> Matrix<T, N, General>
where
    T: Float,
{
    /// Creates a general matrix from a row-major array of shape `N×N`.
    /// Only available for [`General`] matrices; use [`rotation`](Matrix::rotation) or
    /// [`rotation_2d`](Matrix::rotation_2d) / [`rotation_3d`](Matrix::rotation_3d) for [`Isometry`].
    ///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let m = Matrix::<_, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(m.data_ref()[0], [1.0, 2.0]);
    /// assert_eq!(m.data_ref()[1], [3.0, 4.0]);
    /// ```
    #[inline]
    pub fn new(data: [[T; N]; N]) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    /// Replaces the inner row-major data array. Only available for [`General`] matrices.
    ///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let mut m = Matrix::<_, 2, General>::new([[0.0, 0.0], [0.0, 0.0]]);
/// m.set_data([[1.0, 0.0], [0.0, 1.0]]);
    /// assert_eq!(m.data_ref(), &[[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn set_data(&mut self, data: [[T; N]; N]) {
        self.data = data;
    }

    /// Sets the element at the given row and column (zero-based indices).
    /// Only available for [`General`] matrices.
    ///
    /// # Panics
    ///
    /// Panics if `row` or `column` is not in `0..N`.
    ///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let mut m = Matrix::<f64, 2, General>::identity();
/// m.set_component(0, 1, 5.0);
    /// m.set_component(1, 0, -3.0);
    /// assert_eq!(m.data_ref(), &[[1.0, 5.0], [-3.0, 1.0]]);
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

    /// Returns a mutable reference to the inner row-major data array. Only available for [`General`] matrices.
    #[inline]
    pub fn data_ref_mut(&mut self) -> &mut [[T; N]; N] {
        &mut self.data
    }
}

impl<T, const N: usize> Matrix<T, N, Isometry>
where
    T: Float,
{
    /// Builds a rotation matrix by `angle` (radians) in the plane spanned by the two given axes.
    /// The rotation is counterclockwise when viewing from the positive side of the remaining axes
    /// (right-hand rule). Returns a matrix with tag [`Isometry`].
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
/// use apollonius::{Angle, Isometry, Matrix, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
/// let m = Matrix::<f64, 3, Isometry>::rotation(angle, 0, 1);
    /// let v = Vector::new([1.0, 0.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
    /// assert_eq!(rotated.coords_ref()[2], 0.0);
    /// ```
    pub fn rotation(angle: Angle<T>, axis_i: usize, axis_j: usize) -> Self {
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

        mat.data[axis_i][axis_i] = cos;
        mat.data[axis_i][axis_j] = -sin;
        mat.data[axis_j][axis_i] = sin;
        mat.data[axis_j][axis_j] = cos;

        mat
    }
}

impl<T> Matrix<T, 2, Isometry>
where
    T: Float,
{
    /// Builds a 2D rotation by `angle` (radians) in the plane of the two axes (0 and 1).
    /// Equivalent to [`rotation`](Matrix::rotation)(`angle`, 0, 1). Returns a matrix with tag [`Isometry`].
    ///
    /// # Example
    ///
    /// Rotate the unit vector (1, 0) by 90° to get (0, 1):
    ///
/// ```
/// use apollonius::{Angle, Isometry, Matrix, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
/// let m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
    /// let v = Vector::new([1.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
    /// ```
    pub fn rotation_2d(angle: Angle<T>) -> Self {
        Self::rotation(angle, 0, 1)
    }
}

impl<T> Matrix<T, 3, Isometry>
where
    T: Float,
{
    /// Builds a 3D rotation by `angle` (radians) around a single axis. `axis` selects the
    /// rotation axis: 0 = X, 1 = Y, 2 = Z. Returns a matrix with tag [`Isometry`].
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
/// use apollonius::{Angle, Isometry, Matrix, Vector};
/// use std::f64::consts::FRAC_PI_2;
///
/// let angle = Angle::<f64>::from_radians(FRAC_PI_2);
/// let m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 2);
    /// let v = Vector::new([1.0, 0.0, 0.0]);
    /// let rotated = m * v;
    /// assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
    /// assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
    /// assert_eq!(rotated.coords_ref()[2], 0.0);
    /// ```
    pub fn rotation_3d(angle: Angle<T>, axis: usize) -> Self {
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
/// The result is always a [`General`] matrix, regardless of the tags of the operands.
///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let a = Matrix::<_, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
/// let b = Matrix::<_, 2, General>::new([[0.0, 1.0], [1.0, 0.0]]);
/// let sum = a + b;
/// assert_eq!(sum.data_ref(), &[[1.0, 1.0], [1.0, 1.0]]);
/// ```
impl<T, const N: usize, TagL, TagR> Add<Matrix<T, N, TagR>> for Matrix<T, N, TagL>
where
    T: Float,
    TagL: MatrixTag,
    TagR: MatrixTag,
{
    type Output = Matrix<T, N, General>;
    fn add(self, rhs: Matrix<T, N, TagR>) -> Self::Output {
        let data = std::array::from_fn(|i| {
            let row = std::array::from_fn(|j| self.data[i][j] + rhs.data[i][j]);
            row
        });
        Matrix::<T, N, General>::from_raw(data)
    }
}

/// Element-wise subtraction of two matrices of the same size.
/// The result is always a [`General`] matrix, regardless of the tags of the operands.
///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let a = Matrix::<_, 2, General>::new([[2.0, 1.0], [1.0, 2.0]]);
/// let b = Matrix::<_, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
/// assert_eq!((a - b).data_ref(), &[[1.0, 1.0], [1.0, 1.0]]);
/// ```
impl<T, const N: usize, TagL, TagR> Sub<Matrix<T, N, TagR>> for Matrix<T, N, TagL>
where
    T: Float,
    TagL: MatrixTag,
    TagR: MatrixTag,
{
    type Output = Matrix<T, N, General>;

    fn sub(self, rhs: Matrix<T, N, TagR>) -> Self::Output {
        let data =
            std::array::from_fn(|i| std::array::from_fn(|j| self.data[i][j] - rhs.data[i][j]));

        Matrix::<T, N, General>::from_raw(data)
    }
}

/// Matrix multiplication (row × column). The tag of the result is given by
/// [`MulOutput`]: e.g. `Isometry * Isometry` → `Isometry`, `General * Any` → `General`,
/// `Affine * Affine` → `Affine`, `Affine * General` → `General`.
///
/// # Example
///
/// ```
/// use apollonius::{General, Matrix};
///
/// let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
/// let b = Matrix::<f64, 2, General>::new([[5.0, 6.0], [7.0, 8.0]]);
/// let ab = a * b;
/// assert_eq!(ab.data_ref(), &[[19.0, 22.0], [43.0, 50.0]]);
/// ```
impl<T, const N: usize, LTag, RTag> Mul<Matrix<T, N, RTag>> for Matrix<T, N, LTag>
where
    T: Float,
    LTag: MulOutput<RTag>,
    LTag: MatrixTag,
    RTag: MatrixTag,
{
    type Output = Matrix<T, N, LTag::ResultTag>;

    fn mul(self, rhs: Matrix<T, N, RTag>) -> Self::Output {
        let data = Matrix::<T, N, General>::multiply_raw(&self.data, &rhs.data);
        Matrix::<T, N, LTag::ResultTag>::from_raw(data)
    }
}

#[cfg(test)]
mod tests {
    use super::{General, Isometry, Matrix};
    use crate::{Angle, Vector};
    use std::f64::consts::FRAC_PI_2;

    // Type-level assertions: these functions only accept the stated matrix tag. Used in tests to
    // ensure that operations return the expected tag (test fails to compile if the tag changes).
    fn assert_tag_general_2d(_: Matrix<f64, 2, General>) {}
    fn assert_tag_isometry_2d(_: Matrix<f64, 2, Isometry>) {}
    fn assert_tag_isometry_3d(_: Matrix<f64, 3, Isometry>) {}

    fn identity_2() -> Matrix<f64, 2, General> {
        Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]])
    }
    fn identity_3() -> Matrix<f64, 3, General> {
        Matrix::<f64, 3, General>::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[test]
    fn equality_same_data_are_equal() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(a, b);
    }

    #[test]
    fn equality_different_data_are_not_equal() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::<f64, 2, General>::new([[1.0, 1.0], [0.0, 1.0]]);
        assert_ne!(a, b);
    }

    #[test]
    fn equality_identity_equals_self() {
        assert_eq!(Matrix::<f64, 2, General>::identity(), identity_2());
        assert_eq!(Matrix::<f64, 3, General>::identity(), identity_3());
    }

    #[test]
    fn identity_2x2_diagonal_ones_elsewhere_zero() {
        let i = Matrix::<f64, 2, General>::identity();
        assert_eq!(i, identity_2());
    }

    #[test]
    fn identity_3x3_diagonal_ones_elsewhere_zero() {
        let i = Matrix::<f64, 3, General>::identity();
        assert_eq!(i, identity_3());
    }

    #[test]
    fn identity_4x4() {
        let i = Matrix::<f64, 4, General>::identity();
        assert_eq!(
            i.data_ref(),
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
        let a = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::<f64, 2, General>::new([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(
            a + b,
            Matrix::<f64, 2, General>::new([[1.0, 1.0], [1.0, 1.0]])
        );
    }

    #[test]
    fn add_commutative() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f64, 2, General>::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn add_identity_is_neutral() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(
            m + identity_2(),
            Matrix::<f64, 2, General>::new([[2.0, 2.0], [3.0, 5.0]])
        );
    }

    #[test]
    fn sub_element_wise_2x2() {
        let a = Matrix::<f64, 2, General>::new([[2.0, 1.0], [1.0, 2.0]]);
        let b = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(
            a - b,
            Matrix::<f64, 2, General>::new([[1.0, 1.0], [1.0, 1.0]])
        );
    }

    #[test]
    fn sub_inverse_of_add() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f64, 2, General>::new([[0.5, 1.0], [1.5, 2.0]]);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn mul_matrix_identity_left() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(Matrix::<f64, 2, General>::identity() * m, m);
    }

    #[test]
    fn mul_matrix_identity_right() {
        let m = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m * Matrix::<f64, 2, General>::identity(), m);
    }

    #[test]
    fn mul_matrix_2x2_diagonal() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let diag = Matrix::<f64, 2, General>::new([[2.0, 0.0], [0.0, 2.0]]);
        assert_eq!(
            a * diag,
            Matrix::<f64, 2, General>::new([[2.0, 4.0], [6.0, 8.0]])
        );
    }

    #[test]
    fn mul_matrix_2x2_general() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f64, 2, General>::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(
            a * b,
            Matrix::<f64, 2, General>::new([[19.0, 22.0], [43.0, 50.0]])
        );
    }

    #[test]
    fn mul_matrix_3x3() {
        let a = Matrix::<f64, 3, General>::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let b = Matrix::<f64, 3, General>::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(a * b, b);
    }

    #[test]
    fn rotation_2d_returns_isometry() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
    }

    #[test]
    fn rotation_2d_zero_angle_is_identity() {
        let angle = Angle::<f64>::from_radians(0.0);
        let m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        assert_eq!(m, Matrix::<f64, 2, Isometry>::identity());
    }

    #[test]
    fn rotation_2d_90_degrees_maps_unit_x_to_unit_y() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let v = Vector::new([1.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_2d_90_degrees_maps_unit_y_to_negative_x() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let v = Vector::new([0.0, 1.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - (-1.0)).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_2d_is_orthogonal() {
        let angle = Angle::<f64>::from_radians(0.7);
        let m = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let mt_m = {
            let d = m.data_ref();
            Matrix::<f64, 2, General>::new([
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
        let i = Matrix::<f64, 2, General>::identity();
        assert!((mt_m.data_ref()[0][0] - i.data_ref()[0][0]).abs() < 1e-9);
        assert!((mt_m.data_ref()[0][1] - i.data_ref()[0][1]).abs() < 1e-9);
        assert!((mt_m.data_ref()[1][0] - i.data_ref()[1][0]).abs() < 1e-9);
        assert!((mt_m.data_ref()[1][1] - i.data_ref()[1][1]).abs() < 1e-9);
    }

    // --- Rotation 3D (angle + axis 0/1/2; returns Isometry) ----------------------------------

    #[test]
    fn rotation_3d_returns_isometry() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 0);
    }

    #[test]
    fn rotation_3d_zero_angle_is_identity() {
        for axis in [0usize, 1, 2] {
            let angle = Angle::<f64>::from_radians(0.0);
            let m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, axis);
            assert_eq!(m, Matrix::<f64, 3, Isometry>::identity(), "axis = {}", axis);
        }
    }

    #[test]
    fn rotation_3d_around_z_90_degrees_same_as_2d_in_xy() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 2);
        let v = Vector::new([1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_3d_around_x_90_degrees_maps_y_to_z() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 0);
        let v = Vector::new([0.0, 1.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_3d_around_y_90_degrees_maps_z_to_x() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 1);
        let v = Vector::new([0.0, 0.0, 1.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 1.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[2] - 0.0).abs() < 1e-10);
    }

    // --- Rotation N-D (angle + two axes; panic if axis out of 0..N) --------------------------

    #[test]
    fn rotation_nd_returns_isometry_2d() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _m = Matrix::<f64, 2, Isometry>::rotation(angle, 0, 1);
    }

    #[test]
    fn rotation_nd_2d_plane_zero_angle_is_identity() {
        let angle = Angle::<f64>::from_radians(0.0);
        let m = Matrix::<f64, 2, Isometry>::rotation(angle, 0, 1);
        assert_eq!(m, Matrix::<f64, 2, Isometry>::identity());
    }

    #[test]
    fn rotation_nd_3d_plane_01_90_degrees_same_as_2d() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 3, Isometry>::rotation(angle, 0, 1);
        let v = Vector::new([1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_nd_4d_plane_02_identity_on_axes_1_and_3() {
        let angle = Angle::<f64>::from_radians(0.5);
        let m = Matrix::<f64, 4, Isometry>::rotation(angle, 0, 2);
        let v = Vector::new([0.0, 1.0, 0.0, 0.0]);
        let rotated = m * v;
        assert_eq!(rotated.coords_ref(), &[0.0, 1.0, 0.0, 0.0]);
        let v = Vector::new([0.0, 0.0, 0.0, 1.0]);
        let rotated = m * v;
        assert_eq!(rotated.coords_ref(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rotation_nd_4d_plane_01_rotates_first_two_coords() {
        let angle = Angle::<f64>::from_radians(FRAC_PI_2);
        let m = Matrix::<f64, 4, Isometry>::rotation(angle, 0, 1);
        let v = Vector::new([1.0, 0.0, 5.0, 10.0]);
        let rotated = m * v;
        assert!((rotated.coords_ref()[0] - 0.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[1] - 1.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[2] - 5.0).abs() < 1e-10);
        assert!((rotated.coords_ref()[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_first_axis_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 3, Isometry>::rotation(angle, 3, 1);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_second_axis_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 3, Isometry>::rotation(angle, 0, 3);
    }

    #[test]
    #[should_panic(expected = "axis")]
    fn rotation_nd_panics_when_both_axes_out_of_range() {
        let angle = Angle::<f64>::from_radians(0.0);
        let _ = Matrix::<f64, 2, Isometry>::rotation(angle, 2, 2);
    }

    // --- Result tag of operations (compile-time: wrong tag => compile error) ----------------

    #[test]
    fn add_returns_general() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let b = Matrix::<f64, 2, Isometry>::identity();
        let sum = a + b;
        assert_tag_general_2d(sum);
    }

    #[test]
    fn sub_returns_general() {
        let a = Matrix::<f64, 2, General>::new([[2.0, 0.0], [0.0, 2.0]]);
        let b = Matrix::<f64, 2, Isometry>::identity();
        let diff = a - b;
        assert_tag_general_2d(diff);
    }

    #[test]
    fn mul_general_times_general_returns_general() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let product = a * b;
        assert_tag_general_2d(product);
    }

    #[test]
    fn mul_isometry_times_isometry_returns_isometry() {
        let angle = Angle::<f64>::from_radians(0.5);
        let a = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let b = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let product = a * b;
        assert_tag_isometry_2d(product);
    }

    #[test]
    fn mul_general_times_isometry_returns_general() {
        let a = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let angle = Angle::<f64>::from_radians(0.3);
        let b = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let product = a * b;
        assert_tag_general_2d(product);
    }

    #[test]
    fn mul_isometry_times_general_returns_general() {
        let angle = Angle::<f64>::from_radians(0.3);
        let a = Matrix::<f64, 2, Isometry>::rotation_2d(angle);
        let b = Matrix::<f64, 2, General>::new([[1.0, 0.0], [0.0, 1.0]]);
        let product = a * b;
        assert_tag_general_2d(product);
    }

    #[test]
    fn mul_isometry_3d_times_isometry_3d_returns_isometry() {
        let angle = Angle::<f64>::from_radians(0.2);
        let a = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 0);
        let b = Matrix::<f64, 3, Isometry>::rotation_3d(angle, 1);
        let product = a * b;
        assert_tag_isometry_3d(product);
    }
}
