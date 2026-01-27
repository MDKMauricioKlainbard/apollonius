use crate::{EuclideanVector, Point, SpatialRelation, Vector, VectorMetricSquared};
use num_traits::Float;
use std::ops::{Mul, Sub};

/// A finite line segment in N-dimensional space defined by two endpoints.
///
/// The segment pre-calculates its displacement vector (delta) to optimize
/// frequent spatial queries and metric calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment<T, const N: usize> {
    start: Point<T, N>,
    end: Point<T, N>,
    delta: Vector<T, N>,
}

// Basic implementation for construction and data integrity
impl<T, const N: usize> Segment<T, N>
where
    T: Copy + Sub<Output = T>,
{
    /// Creates a new Segment from two points.
    ///
    /// # Examples
    /// ```
    /// use apollonius::{Point, Segment};
    /// let start = Point::new([0.0, 0.0]);
    /// let end = Point::new([10.0, 0.0]);
    /// let segment = Segment::new(start, end);
    /// ```
    #[inline]
    pub fn new(start: Point<T, N>, end: Point<T, N>) -> Self {
        let delta = end - start;
        Self { start, end, delta }
    }

    /// Returns the start point of the segment.
    #[inline]
    pub fn start(&self) -> Point<T, N> {
        self.start
    }

    /// Returns the end point of the segment.
    #[inline]
    pub fn end(&self) -> Point<T, N> {
        self.end
    }

    /// Updates the start point and synchronizes the internal state.
    pub fn set_start(&mut self, new_start: Point<T, N>) {
        self.start = new_start;
        self.delta = self.end - self.start;
    }

    /// Updates the end point and synchronizes the internal state.
    pub fn set_end(&mut self, new_end: Point<T, N>) {
        self.end = new_end;
        self.delta = self.end - self.start;
    }
}

// Metrics implementation for types that support basic arithmetic
impl<T, const N: usize> Segment<T, N>
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + std::iter::Sum,
{
    /// Returns the squared length of the segment.
    ///
    /// This is more efficient than length() as it avoids a square root operation.
    #[inline]
    pub fn length_squared(&self) -> T {
        self.delta.magnitude_squared()
    }
}

// Advanced geometric operations requiring floating-point numbers
impl<T, const N: usize> Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Returns the Euclidean length of the segment.
    #[inline]
    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    /// Returns the normalized direction vector of the segment.
    ///
    /// Returns None if the segment has zero length (start == end).
    pub fn direction(&self) -> Option<Vector<T, N>> {
        self.delta.normalize()
    }

    /// Returns the point at the exact center of the segment.
    pub fn midpoint(&self) -> Point<T, N> {
        let half = T::from(0.5).unwrap();
        self.at(half)
    }

    /// Returns a point along the segment at parameter t.
    ///
    /// * t = 0.0 yields the start point.
    /// * t = 1.0 yields the end point.
    ///
    /// Note: This method allows extrapolation if t is outside the [0, 1] range.
    #[inline]
    pub fn at(&self, t: T) -> Point<T, N> {
        self.start + self.delta * t
    }
}

impl<T, const N: usize> SpatialRelation<T, N> for Segment<T, N>
where
    T: Float + std::iter::Sum,
{
    /// Projects a point p onto the segment and returns the closest point.
    ///
    /// The projection is clamped to the segment's endpoints.
    ///
    /// Calculation logic:
    /// t = ((p - start) dot delta) / length_squared
    fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let mag_sq = self.length_squared();

        if mag_sq > T::zero() {
            let t = ((*p - self.start).dot(&self.delta) / mag_sq)
                .max(T::zero())
                .min(T::one());

            self.at(t)
        } else {
            self.start
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use approx::assert_relative_eq;

    #[test]
    fn test_segment_closest_point() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([10.0, 0.0]);
        let seg = Segment::new(start, end);

        let p_behind = Point::new([-5.0, 5.0]);
        assert_eq!(seg.closest_point(&p_behind).coords, [0.0, 0.0]);

        let p_ahead = Point::new([15.0, 5.0]);
        assert_eq!(seg.closest_point(&p_ahead).coords, [10.0, 0.0]);

        let p_mid = Point::new([5.0, 10.0]);
        assert_eq!(seg.closest_point(&p_mid).coords, [5.0, 0.0]);
    }

    #[test]
    fn test_segment_metrics() {
        let start = Point::new([0.0, 0.0]);
        let end = Point::new([3.0, 4.0]);
        let seg = Segment::new(start, end);

        assert_relative_eq!(seg.length_squared(), 25.0);
        assert_relative_eq!(seg.length(), 5.0);
    }

    #[test]
    fn test_segment_midpoint() {
        let seg = Segment::new(Point::new([0.0, 10.0]), Point::new([10.0, 20.0]));
        let mid = seg.midpoint();

        assert_relative_eq!(mid.coords[0], 5.0);
        assert_relative_eq!(mid.coords[1], 15.0);
    }

    #[test]
    fn test_segment_at_parameter() {
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([10.0, 0.0]));

        assert_relative_eq!(seg.at(0.2).coords[0], 2.0);
        assert_relative_eq!(seg.at(1.0).coords[0], 10.0);
        assert_relative_eq!(seg.at(1.5).coords[0], 15.0);
    }

    #[test]
    fn test_segment_direction() {
        let seg = Segment::new(Point::new([0.0, 0.0]), Point::new([0.0, 10.0]));
        let dir = seg.direction().expect("Should have direction");

        assert_relative_eq!(dir.coords[0], 0.0);
        assert_relative_eq!(dir.coords[1], 1.0);

        let point_seg = Segment::new(Point::new([1.0, 1.0]), Point::new([1.0, 1.0]));
        assert!(point_seg.direction().is_none());
    }

    #[test]
    fn test_segment_zero_length_proximity() {
        let p = Point::new([1.0, 1.0]);
        let seg = Segment::new(p, p);

        let query = Point::new([5.0, 5.0]);
        assert_eq!(seg.closest_point(&query), p);
    }
}
