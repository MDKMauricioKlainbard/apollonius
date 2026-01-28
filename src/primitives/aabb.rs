use crate::Point;
use num_traits::Float;

/// An Axis-Aligned Bounding Box (AABB) defined by its minimum and maximum coordinates.
///
/// AABBs are used as a first-pass (broad-phase) collision detection mechanism.
/// They provide a computationally cheap way to determine if two entities *might* be
/// intersecting before running more expensive primitive-specific intersection algorithms.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct AABB<T, const N: usize> {
    /// The component-wise minimum point (bottom-left-front).
    pub min: Point<T, N>,
    /// The component-wise maximum point (top-right-back).
    pub max: Point<T, N>,
}

impl<T, const N: usize> AABB<T, N>
where
    T: Float,
{
    /// Creates a new AABB from two points.
    ///
    /// Note: The caller must ensure that `min` components are less than or equal to `max` components.
    pub fn new(min: Point<T, N>, max: Point<T, N>) -> Self {
        Self { min, max }
    }

    /// Determines if this AABB overlaps with another AABB.
    ///
    /// This implementation uses the Hyper-rectangle Overlap theorem, an extension of the
    /// logic used in 2D rectangle collision. Two AABBs intersect if and only if they
    /// overlap on all N axes simultaneously.
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..N {
            // If there is a gap in any single dimension, the boxes cannot overlap.
            // Using the negation of the "no-overlap" condition for early exit.
            if self.max.coords[i] <= other.min.coords[i]
                || other.max.coords[i] <= self.min.coords[i]
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
}
