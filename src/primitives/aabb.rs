use crate::Point;
use num_traits::Float;

/// An Axis-Aligned Bounding Box (AABB) defined by its minimum and maximum coordinates.
///
/// AABBs are used as a first-pass (broad-phase) collision detection mechanism.
/// They provide a computationally cheap way to determine if two entities might be
/// intersecting before running more expensive primitive-specific intersection algorithms.
///
/// This implementation is N-dimensional, allowing for AABBs in 2D, 3D, or higher-dimensional spaces.
///
/// # Examples
///
/// ```
/// use apollonius::{Point, AABB};
///
/// let min = Point::new([0.0, 0.0]);
/// let max = Point::new([2.0, 2.0]);
/// let aabb = AABB::new(min, max);
///
/// assert_eq!(aabb.min_ref().coords_ref()[0], 0.0);
/// assert_eq!(aabb.max_ref().coords_ref()[1], 2.0);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::Deserialize<'de>")))]
pub struct AABB<T, const N: usize> {
    /// The component-wise minimum point (e.g., bottom-left-front).
    min: Point<T, N>,
    /// The component-wise maximum point (e.g., top-right-back).
    max: Point<T, N>,
}

impl<T, const N: usize> AABB<T, N>
where
    T: Float,
{
    /// Creates a new AABB from two points.
    ///
    /// # Arguments
    /// * `min` - The component-wise minimum point.
    /// * `max` - The component-wise maximum point.
    ///
    /// # Panics
    /// While this method does not currently panic, the caller must ensure that
    /// `min` components are less than or equal to `max` components for correct
    /// intersection logic.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, AABB};
    ///
    /// let aabb = AABB::new(Point::new([-1.0, -1.0]), Point::new([1.0, 1.0]));
    /// ```
    #[inline]
    pub fn new(min: Point<T, N>, max: Point<T, N>) -> Self {
        Self { min, max }
    }

    /// Returns the minimum point (by value).
    #[inline]
    pub fn min(&self) -> Point<T, N>
    where
        T: Copy,
    {
        self.min
    }

    /// Returns the maximum point (by value).
    #[inline]
    pub fn max(&self) -> Point<T, N>
    where
        T: Copy,
    {
        self.max
    }

    /// Returns a reference to the minimum point.
    #[inline]
    pub fn min_ref(&self) -> &Point<T, N> {
        &self.min
    }

    /// Returns a reference to the maximum point.
    #[inline]
    pub fn max_ref(&self) -> &Point<T, N> {
        &self.max
    }

    /// Sets the minimum point.
    #[inline]
    pub fn set_min(&mut self, min: Point<T, N>) {
        self.min = min;
    }

    /// Sets the maximum point.
    #[inline]
    pub fn set_max(&mut self, max: Point<T, N>) {
        self.max = max;
    }

    /// Returns a mutable reference to the minimum point.
    ///
    /// Useful for in-place updates (e.g. `aabb.min_ref_mut().coords_ref_mut()[0] = x`).
    #[inline]
    pub fn min_ref_mut(&mut self) -> &mut Point<T, N> {
        &mut self.min
    }

    /// Returns a mutable reference to the maximum point.
    ///
    /// Useful for in-place updates (e.g. `aabb.max_ref_mut().coords_ref_mut()[0] = x`).
    #[inline]
    pub fn max_ref_mut(&mut self) -> &mut Point<T, N> {
        &mut self.max
    }

    /// Determines if this AABB overlaps with another AABB.
    ///
    /// This implementation uses the Hyper-rectangle Overlap theorem. Two AABBs intersect
    /// if and only if they overlap on all N axes simultaneously.
    ///
    /// # Performance
    /// The method uses an early-exit strategy: as soon as a separation is found on
    /// any axis, it returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use apollonius::{Point, AABB};
    ///
    /// let box_a = AABB::new(Point::new([0.0, 0.0]), Point::new([2.0, 2.0]));
    /// let box_b = AABB::new(Point::new([1.0, 1.0]), Point::new([3.0, 3.0]));
    /// let box_c = AABB::new(Point::new([10.0, 10.0]), Point::new([12.0, 12.0]));
    ///
    /// assert!(box_a.intersects(&box_b));
    /// assert!(!box_a.intersects(&box_c));
    /// ```
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..N {
            // Hyper-rectangle overlap logic:
            // Two intervals [a, b] and [c, d] overlap if a < d and c < b.
            // We use the negation for early exit.
            if self.max_ref().coords_ref()[i] <= other.min_ref().coords_ref()[i]
                || other.max_ref().coords_ref()[i] <= self.min_ref().coords_ref()[i]
            {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    #[test]
    fn test_aabb_overlap_2d() {
        let box1 = AABB::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));
        let box2 = AABB::new(Point::new([5.0, 5.0]), Point::new([15.0, 15.0]));

        assert!(
            box1.intersects(&box2),
            "Boxes should overlap in the center region"
        );
    }

    #[test]
    fn test_aabb_no_overlap_separated() {
        let box1 = AABB::new(Point::new([0.0, 0.0]), Point::new([2.0, 2.0]));
        let box2 = AABB::new(Point::new([5.0, 5.0]), Point::new([7.0, 7.0]));

        assert!(!box1.intersects(&box2), "Boxes are clearly separated");
    }

    #[test]
    fn test_aabb_touching_edge_returns_false() {
        // In most physics engines, touching at the exact edge is not an overlap
        // to prevent sticking. Using '<=' logic as per your LeetCode solution.
        let box1 = AABB::new(Point::new([0.0, 0.0]), Point::new([5.0, 5.0]));
        let box2 = AABB::new(Point::new([5.0, 0.0]), Point::new([10.0, 5.0]));

        assert!(
            !box1.intersects(&box2),
            "Touching edges should not count as overlap"
        );
    }

    #[test]
    fn test_aabb_one_inside_another() {
        let large = AABB::new(
            Point::new([0.0, 0.0, 0.0]),
            Point::new([100.0, 100.0, 100.0]),
        );
        let small = AABB::new(
            Point::new([40.0, 40.0, 40.0]),
            Point::new([60.0, 60.0, 60.0]),
        );

        assert!(
            large.intersects(&small),
            "Contained box must trigger intersection"
        );
        assert!(small.intersects(&large), "Symmetry check failed");
    }

    #[test]
    fn test_aabb_3d_mismatch_on_one_axis() {
        // Overlap on X and Y, but separated on Z
        let box1 = AABB::new(Point::new([0.0, 0.0, 0.0]), Point::new([10.0, 10.0, 10.0]));
        let box2 = AABB::new(Point::new([5.0, 5.0, 20.0]), Point::new([15.0, 15.0, 30.0]));

        assert!(
            !box1.intersects(&box2),
            "Separation on Z axis should prevent overlap"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_aabb_serialization_roundtrip() {
        use serde_json;

        let aabb = AABB::new(Point::new([0.0, 0.0]), Point::new([10.0, 10.0]));
        let json = serde_json::to_string(&aabb).unwrap();
        let restored: AABB<f64, 2> = serde_json::from_str(&json).unwrap();
        assert_eq!(aabb.min_ref(), restored.min_ref());
        assert_eq!(aabb.max_ref(), restored.max_ref());
    }
}
