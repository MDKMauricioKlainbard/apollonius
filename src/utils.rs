use num_traits::Float;

/// Represents the classification of a floating-point number relative to a zero-tolerance (epsilon).
///
/// This is used throughout the engine to handle the inherent imprecision of
/// floating-point arithmetic in geometric tests (e.g., checking if a point
/// lies exactly on a plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
