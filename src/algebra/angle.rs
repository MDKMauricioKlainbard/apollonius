//! Angles in radians with conversion to/from degrees and trigonometry.
//!
//! This module provides an [`Angle`] type that stores an angle in radians and supports conversion
//! to/from degrees and direct access to sine and cosine via [`Float`](num_traits::Float).

use std::fmt;

use num_traits::Float;

use crate::{classify_to_zero, FloatSign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An angle stored in radians.
///
/// The internal value is always in radians. Use [`from_degrees`](Angle::from_degrees) to construct
/// from degrees and [`as_degrees`](Angle::as_degrees) to read back in degrees.
///
/// # Type parameter
///
/// * `T` — scalar type, must implement [`Float`](num_traits::Float).
///
/// # Serialization
///
/// With the **`serde`** feature enabled, `Angle` serializes as a single number (radians).
///
/// # Examples
///
/// ```
/// use apollonius::Angle;
///
/// let a = Angle::<f64>::from_radians(std::f64::consts::PI);
/// assert!((a.as_degrees() - 180.0).abs() < 1e-10);
/// ```
///
/// From degrees:
///
/// ```
/// use apollonius::Angle;
///
/// let a = Angle::<f64>::from_degrees(90.0);
/// assert!((a.as_radians() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
/// ```
///
/// Sine and cosine:
///
/// ```
/// use apollonius::Angle;
///
/// let a = Angle::<f64>::from_degrees(0.0);
/// let (s, c) = a.sin_cos();
/// assert!((s - 0.0).abs() < 1e-10);
/// assert!((c - 1.0).abs() < 1e-10);
/// ```
/// Error returned when [`Angle::tan`] is undefined (cosine is zero or near zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TanUndefined;

impl fmt::Display for TanUndefined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tan is undefined: cosine is zero or below epsilon")
    }
}

impl std::error::Error for TanUndefined {}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Angle<T: Float> {
    radians: T,
}

#[cfg(feature = "serde")]
impl<T: Float + Serialize> Serialize for Angle<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.radians.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Float + Deserialize<'de>> Deserialize<'de> for Angle<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let radians = T::deserialize(deserializer)?;
        Ok(Self { radians })
    }
}

/// Displays the angle as degrees and radians, e.g. `90° (1.5708 rad)`.
///
/// # Example
///
/// ```
/// use apollonius::Angle;
///
/// let a = Angle::<f64>::from_degrees(90.0);
/// let s = format!("{}", a);
/// assert!(s.starts_with("90"));
/// assert!(s.contains("°"));
/// assert!(s.contains("rad"));
/// ```
impl<T: Float + fmt::Display> fmt::Display for Angle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}° ({:.4} rad)", self.as_degrees(), self.radians)
    }
}

impl<T: Float> Angle<T> {
    /// Creates an angle from a value in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_radians(0.0);
    /// assert_eq!(a.as_radians(), 0.0);
    /// ```
    pub fn from_radians(radians: T) -> Self {
        Self { radians }
    }

    /// Creates an angle from a value in degrees.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(180.0);
    /// assert!((a.as_radians() - std::f64::consts::PI).abs() < 1e-10);
    /// ```
    pub fn from_degrees(deg: T) -> Self {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let deg_conv = T::from(180.0).unwrap();
        Self {
            radians: deg * (pi / deg_conv),
        }
    }

    /// Returns the angle in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(90.0);
    /// assert!((a.as_radians() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    /// ```
    pub fn as_radians(&self) -> T {
        self.radians
    }

    /// Returns the angle in degrees.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_radians(std::f64::consts::PI);
    /// assert!((a.as_degrees() - 180.0).abs() < 1e-10);
    /// ```
    pub fn as_degrees(&self) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let deg_conv = T::from(180.0).unwrap();
        self.radians * (deg_conv / pi)
    }

    /// Returns a mutable reference to the angle value in radians.
    #[inline]
    pub fn radians_mut(&mut self) -> &mut T {
        &mut self.radians
    }

    /// Returns the sine of the angle.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(90.0);
    /// assert!((a.sin() - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn sin(self) -> T {
        self.radians.sin()
    }

    /// Returns the cosine of the angle.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(0.0);
    /// assert!((a.cos() - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn cos(self) -> T {
        self.radians.cos()
    }

    /// Returns the tangent of the angle, or an error if the cosine is zero or below the
    /// classification epsilon (avoids division by zero and overflow).
    ///
    /// Uses [`classify_to_zero`] on the cosine; if classified as zero, returns [`TanUndefined`].
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(45.0);
    /// let t = a.tan().unwrap();
    /// assert!((t - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Error
    ///
    /// Returns `Err(TanUndefined)` when the angle is ±90° (or equivalent), where cos ≈ 0.
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(90.0);
    /// assert!(a.tan().is_err());
    /// ```
    pub fn tan(self) -> Result<T, TanUndefined> {
        let c = self.radians.cos();
        if classify_to_zero(c, None) == FloatSign::Zero {
            return Err(TanUndefined);
        }
        Ok(self.radians.tan())
    }

    /// Returns `(sin(angle), cos(angle))` using the stored radians.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let a = Angle::<f64>::from_degrees(90.0);
    /// let (s, c) = a.sin_cos();
    /// assert!((s - 1.0).abs() < 1e-10);
    /// assert!(c.abs() < 1e-10);
    /// ```
    pub fn sin_cos(self) -> (T, T) {
        self.radians.sin_cos()
    }

    /// Returns a new angle equivalent to `self` in the range **[0, 2π)**.
    ///
    /// Any angle (positive, negative, or beyond one turn) is mapped into [0, 2π) by adding or
    /// subtracting multiples of 2π.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let pi = std::f64::consts::PI;
    /// let a = Angle::<f64>::from_radians(3.0 * pi);
    /// let b = a.normalized_0_2pi();
    /// assert!((b.as_radians() - pi).abs() < 1e-10);
    ///
    /// let c = Angle::<f64>::from_radians(-pi / 2.0);
    /// let d = c.normalized_0_2pi();
    /// assert!((d.as_radians() - (3.0 * pi / 2.0)).abs() < 1e-10);
    /// ```
    pub fn normalized_0_2pi(self) -> Self {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = pi + pi;
        let r = self.radians - (self.radians / two_pi).floor() * two_pi;
        Self::from_radians(r)
    }

    /// Returns a new angle equivalent to `self` in the range **(-π, π]**.
    ///
    /// Any angle is mapped into (-π, π] by adding or subtracting multiples of 2π.
    ///
    /// # Example
    ///
    /// ```
    /// use apollonius::Angle;
    ///
    /// let pi = std::f64::consts::PI;
    /// let a = Angle::<f64>::from_radians(3.0 * pi / 2.0);
    /// let b = a.normalized_neg_pi_pi();
    /// assert!((b.as_radians() - (-pi / 2.0)).abs() < 1e-10);
    ///
    /// let c = Angle::<f64>::from_radians(-pi / 2.0);
    /// let d = c.normalized_neg_pi_pi();
    /// assert!((d.as_radians() - (-pi / 2.0)).abs() < 1e-10);
    /// ```
    pub fn normalized_neg_pi_pi(self) -> Self {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = pi + pi;
        let a = self.normalized_0_2pi();
        let r = a.as_radians();
        if r > pi {
            Self::from_radians(r - two_pi)
        } else {
            a
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Angle, TanUndefined};

    const PI: f64 = std::f64::consts::PI;

    #[test]
    fn from_radians_roundtrip() {
        let a = Angle::<f64>::from_radians(PI);
        assert!((a.as_radians() - PI).abs() < 1e-15);
    }

    #[test]
    fn from_degrees_180_is_pi_radians() {
        let a = Angle::<f64>::from_degrees(180.0);
        assert!((a.as_radians() - PI).abs() < 1e-10);
    }

    #[test]
    fn from_degrees_90_is_half_pi() {
        let a = Angle::<f64>::from_degrees(90.0);
        assert!((a.as_radians() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn from_degrees_360_is_two_pi() {
        let a = Angle::<f64>::from_degrees(360.0);
        assert!((a.as_radians() - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn as_degrees_roundtrip() {
        let a = Angle::<f64>::from_degrees(45.0);
        assert!((a.as_degrees() - 45.0).abs() < 1e-10);
    }

    #[test]
    fn sin_cos_zero() {
        let a = Angle::<f64>::from_radians(0.0);
        let (s, c) = a.sin_cos();
        assert!((s - 0.0).abs() < 1e-15);
        assert!((c - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sin_cos_half_pi() {
        let a = Angle::<f64>::from_radians(PI / 2.0);
        let (s, c) = a.sin_cos();
        assert!((s - 1.0).abs() < 1e-15);
        assert!(c.abs() < 1e-15);
    }

    #[test]
    fn sin_cos_pi() {
        let a = Angle::<f64>::from_radians(PI);
        let (s, c) = a.sin_cos();
        assert!(s.abs() < 1e-15);
        assert!((c - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn sin_isolated() {
        let a = Angle::<f64>::from_degrees(90.0);
        assert!((a.sin() - 1.0).abs() < 1e-10);
        let b = Angle::<f64>::from_radians(0.0);
        assert!(b.sin().abs() < 1e-15);
    }

    #[test]
    fn cos_isolated() {
        let a = Angle::<f64>::from_degrees(0.0);
        assert!((a.cos() - 1.0).abs() < 1e-10);
        let b = Angle::<f64>::from_degrees(90.0);
        assert!(b.cos().abs() < 1e-10);
    }

    #[test]
    fn tan_ok() {
        let a = Angle::<f64>::from_degrees(45.0);
        let t = a.tan().unwrap();
        assert!((t - 1.0).abs() < 1e-10);
        let b = Angle::<f64>::from_radians(0.0);
        assert!(b.tan().unwrap().abs() < 1e-15);
    }

    #[test]
    fn tan_err_at_90_degrees() {
        let a = Angle::<f64>::from_degrees(90.0);
        let r = a.tan();
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), TanUndefined);
    }

    #[test]
    fn tan_err_at_270_degrees() {
        let a = Angle::<f64>::from_degrees(270.0);
        assert!(a.tan().is_err());
    }

    #[test]
    fn equality_same_radians() {
        let a = Angle::<f64>::from_radians(1.0);
        let b = Angle::<f64>::from_radians(1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn equality_different_radians() {
        let a = Angle::<f64>::from_radians(0.0);
        let b = Angle::<f64>::from_radians(1.0);
        assert_ne!(a, b);
    }

    #[test]
    fn ordering() {
        let a = Angle::<f64>::from_radians(0.0);
        let b = Angle::<f64>::from_radians(PI);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn normalized_0_2pi_unchanged_when_in_range() {
        let a = Angle::<f64>::from_radians(PI / 2.0);
        let b = a.normalized_0_2pi();
        assert!((a.as_radians() - b.as_radians()).abs() < 1e-15);
    }

    #[test]
    fn normalized_0_2pi_wraps_positive() {
        let a = Angle::<f64>::from_radians(3.0 * PI);
        let b = a.normalized_0_2pi();
        assert!((b.as_radians() - PI).abs() < 1e-10);
    }

    #[test]
    fn normalized_0_2pi_wraps_negative() {
        let a = Angle::<f64>::from_radians(-PI / 2.0);
        let b = a.normalized_0_2pi();
        assert!((b.as_radians() - (3.0 * PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn normalized_0_2pi_zero() {
        let a = Angle::<f64>::from_radians(2.0 * PI);
        let b = a.normalized_0_2pi();
        assert!(b.as_radians().abs() < 1e-10);
    }

    #[test]
    fn normalized_neg_pi_pi_unchanged_when_in_range() {
        let a = Angle::<f64>::from_radians(PI / 2.0);
        let b = a.normalized_neg_pi_pi();
        assert!((a.as_radians() - b.as_radians()).abs() < 1e-15);
    }

    #[test]
    fn normalized_neg_pi_pi_above_pi() {
        let a = Angle::<f64>::from_radians(3.0 * PI / 2.0);
        let b = a.normalized_neg_pi_pi();
        assert!((b.as_radians() - (-PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn normalized_neg_pi_pi_negative() {
        let a = Angle::<f64>::from_radians(-PI / 2.0);
        let b = a.normalized_neg_pi_pi();
        assert!((b.as_radians() - (-PI / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn normalized_neg_pi_pi_exactly_pi() {
        let a = Angle::<f64>::from_radians(PI);
        let b = a.normalized_neg_pi_pi();
        assert!((b.as_radians() - PI).abs() < 1e-15);
    }

    #[test]
    fn display_shows_degrees_and_radians() {
        let a = Angle::<f64>::from_degrees(90.0);
        let s = format!("{}", a);
        assert!(s.starts_with("90"));
        assert!(s.contains("°"));
        assert!(s.contains("rad"));
        assert!(s.contains("1.57")); // pi/2 ≈ 1.57
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_radians() {
        let a = Angle::<f64>::from_radians(0.5);
        let json = serde_json::to_string(&a).unwrap();
        let b: Angle<f64> = serde_json::from_str(&json).unwrap();
        assert!((a.as_radians() - b.as_radians()).abs() < 1e-15);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_degrees() {
        let a = Angle::<f64>::from_degrees(30.0);
        let json = serde_json::to_string(&a).unwrap();
        let b: Angle<f64> = serde_json::from_str(&json).unwrap();
        assert!((a.as_degrees() - b.as_degrees()).abs() < 1e-10);
    }
}
