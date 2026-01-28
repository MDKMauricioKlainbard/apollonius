use num_traits::Float;

/// Represents the result of a high-precision floating-point comparison with zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSign {
    Positive,
    Zero,
    Negative,
}

/// Classifies a floating-point number relative to zero.
///
/// The `epsilon_override` allows providing a custom precision threshold.
/// If `None` is provided, a default safe tolerance (1e-6) is used to account
/// for common floating-point accumulation errors.
#[inline]
pub fn classify_to_zero<T: Float>(val: T, epsilon_override: Option<T>) -> FloatSign {
    let epsilon = epsilon_override.unwrap_or_else(|| T::from(1e-6).unwrap());

    if val.abs() < epsilon {
        FloatSign::Zero
    } else if val > T::zero() {
        FloatSign::Positive
    } else {
        FloatSign::Negative
    }
}
