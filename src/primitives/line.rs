use num_traits::Float;

use crate::{EuclideanVector, Point, Vector, VectorMetricSquared};

pub struct Line<T, const N: usize> {
    origin: Point<T, N>,
    direction: Vector<T, N>,
}

impl<T, const N: usize> Line<T, N>
where
    T: Float + std::iter::Sum,
{
    pub fn new(origin: Point<T, N>, direction: Vector<T, N>) -> Self {
        let direction = direction.normalize().unwrap_or(direction);
        Self { origin, direction }
    }

    pub fn distance_to_point(&self, point: &Point<T, N>) -> T {
        (self.closest_point(point) - *point).magnitude()
    }

    pub fn closest_point(&self, p: &Point<T, N>) -> Point<T, N> {
        let t = (*p - self.origin).dot(&self.direction);
        self.origin + self.direction * t
    }

    pub fn at(&self, t: T) -> Point<T, N> {
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_line_at_parameter() {
        let origin = Point::new([0.0, 0.0]);
        let direction = Vector::new([1.0, 0.0]);
        let line = Line::new(origin, direction);

        // L(t) = origin + t * direction
        assert_relative_eq!(line.at(0.0).coords[0], 0.0);
        assert_relative_eq!(line.at(1.0).coords[0], 1.0);
        assert_relative_eq!(line.at(-1.0).coords[0], -1.0);
    }

    #[test]
    fn test_line_closest_point() {
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));

        let p = Point::new([5.0, 10.0]);

        let closest = line.closest_point(&p);
        assert_relative_eq!(closest.coords[0], 5.0);
        assert_relative_eq!(closest.coords[1], 0.0);
    }

    #[test]
    fn test_distance_to_point() {
        let line = Line::new(Point::new([0.0, 0.0]), Vector::new([1.0, 0.0]));
        let p = Point::new([0.0, 10.0]);

        assert_relative_eq!(line.distance_to_point(&p), 10.0);
    }
}
