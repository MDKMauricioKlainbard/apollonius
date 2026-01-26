use num_traits::Float;
use std::ops::{Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<T, const N: usize> {
    pub coords: [T; N],
}

pub type Point2D<T> = Point<T, 2>;
pub type Point3D<T> = Point<T, 3>;

pub trait MetricSquared<T> {
    fn distance_squared(&self, other: &Self) -> T;
}
pub trait EuclideanMetric<T>: MetricSquared<T> {
    fn distance(&self, other: &Self) -> T;
}

impl<T, const N: usize> Point<T, N>
where
    T: Copy,
{
    #[inline]
    pub fn new(coords: [T; N]) -> Self {
        Self { coords }
    }
}

impl<T> From<(T, T)> for Point2D<T> {
    #[inline]
    fn from(tuple: (T, T)) -> Self {
        Self {
            coords: [tuple.0, tuple.1],
        }
    }
}

impl<T> From<(T, T, T)> for Point3D<T> {
    #[inline]
    fn from(tuple: (T, T, T)) -> Self {
        Self {
            coords: [tuple.0, tuple.1, tuple.2],
        }
    }
}

impl<T, const N: usize> MetricSquared<T> for Point<T, N>
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + std::iter::Sum,
{
    #[inline]
    fn distance_squared(&self, other: &Self) -> T {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .sum()
    }
}

impl<T, const N: usize> EuclideanMetric<T> for Point<T, N>
where
    T: Float + std::iter::Sum,
{
    #[inline]
    fn distance(&self, other: &Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

#[cfg(test)]
mod points_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_construction_and_conversions() {
        let p_gen = Point::new([1, 2, 3]);
        assert_eq!(p_gen.coords, [1, 2, 3]);

        let p_from_arr = Point::new([1.5, 2.5]);
        assert_eq!(p_from_arr.coords, [1.5, 2.5]);

        let p2d: Point2D<f32> = Point2D::from((1.0, 2.0));
        assert_eq!(p2d.coords, [1.0, 2.0]);

        let p3d: Point3D<f32> = Point3D::from((1.0, 2.0, 3.0));
        assert_eq!(p3d.coords, [1.0, 2.0, 3.0]);

        // Type aliases funcionan
        let alias_2d: Point2D<f64> = Point::new([1.0, 2.0]);
        assert_eq!(alias_2d.coords, [1.0, 2.0]);
    }

    #[test]
    fn test_distance_operations_floats() {
        let p1 = Point::new([0.0_f32, 0.0, 0.0]);
        let p2 = Point::new([3.0, 4.0, 0.0]);

        // distance_squared (3^2 + 4^2 + 0^2 = 25)
        assert_relative_eq!(p1.distance_squared(&p2), 25.0);
        // distance (sqrt(25) = 5)
        assert_relative_eq!(p1.distance(&p2), 5.0);

        // Distancia a sí mismo debe ser 0
        assert_relative_eq!(p1.distance_squared(&p1), 0.0);
        assert_relative_eq!(p1.distance(&p1), 0.0);

        // Probando con f64
        let p1_f64 = Point::new([0.0_f64, 0.0]);
        let p2_f64 = Point::new([1.0, 1.0]);
        assert_relative_eq!(p1_f64.distance(&p2_f64), 2.0_f64.sqrt());
    }

    #[test]
    fn test_distance_squared_integers() {
        let p1: Point<i32, 2> = Point::new([1, 2]);
        let p2 = Point::new([4, 6]);
        assert_eq!(p1.distance_squared(&p2), 25);
    }

    #[test]
    fn test_properties_and_traits() {
        let p1 = Point::new([1, 2]);
        let p2 = p1;

        assert_eq!(p1.coords, p2.coords);

        let p3 = Point::new([1.0, 2.0]);
        println!("{:?}", p3);
        assert!(p3 == Point::new([1.0, 2.0]));
        assert!(p3 != Point::new([1.0, 3.0]));
    }
}
