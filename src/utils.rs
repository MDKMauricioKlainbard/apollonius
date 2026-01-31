//! Utilities for robust floating-point comparisons in geometric code.
//!
//! This module provides [`FloatSign`] and [`classify_to_zero`] for classifying
//! values relative to zero with an epsilon tolerance, avoiding brittle `== 0`
//! checks in the presence of numerical error.

use num_traits::Float;

/// Classification of a value relative to zero using an epsilon tolerance.
///
/// Used throughout the library for robust geometric tests (e.g. whether a point
/// lies on a plane or a segment is parallel to another). Values whose absolute
/// value is below the tolerance are treated as [`Zero`](FloatSign::Zero).
///
/// # Example
///
/// ```
/// use apollonius::{classify_to_zero, FloatSign};
///
/// match classify_to_zero(1e-10, None) {
///     FloatSign::Positive => {}
///     FloatSign::Zero => {} // tiny values are treated as zero
///     FloatSign::Negative => {}
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FloatSign {
    /// The value is strictly greater than the positive epsilon.
    Positive,
    /// The absolute value is within the epsilon threshold (treated as zero).
    Zero,
    /// The value is strictly less than the negative epsilon.
    Negative,
}

/// Classifies a floating-point number relative to zero using an epsilon threshold.
///
/// This utility is essential for robust geometric calculations where precision
/// errors can cause strict equality checks (`val == 0.0`) to fail.
///
/// # Arguments
/// * `val` - The value to classify.
/// * `epsilon_override` - An optional custom threshold. Defaults to `1e-6` if `None`.
///
/// # Examples
///
/// ```
/// use apollonius::{classify_to_zero, FloatSign};
///
/// // Values smaller than epsilon are treated as Zero
/// let tiny = 1e-8;
/// assert_eq!(classify_to_zero(tiny, None), FloatSign::Zero);
///
/// // Values larger than epsilon are classified by their sign
/// let big = 10.0;
/// assert_eq!(classify_to_zero(big, None), FloatSign::Positive);
///
/// let neg = -5.0;
/// assert_eq!(classify_to_zero(neg, None), FloatSign::Negative);
/// ```
#[inline]
pub fn classify_to_zero<T: Float>(val: T, epsilon_override: Option<T>) -> FloatSign {
    // Defaulting to 1e-6 to account for common accumulation errors in
    // physics integration (like RK4).
    let epsilon = epsilon_override.unwrap_or_else(|| T::from(1e-6).unwrap());

    if val.abs() < epsilon {
        FloatSign::Zero
    } else if val > T::zero() {
        FloatSign::Positive
    } else {
        FloatSign::Negative
    }
}
