pub mod line;
pub mod segment;
pub mod circle;

pub trait Area<T> {
    fn area(&self) -> T;
}
