use crate::{Matrix, Vector};

#[derive(Clone, Copy, Debug)]
pub enum LinearMap<T, const N: usize> {
    Rotation(Matrix<T, N>),
}

impl<T, const N: usize> LinearMap<T, N> {
    /// Returns a mutable reference to the inner matrix when this is a [`Rotation`](Self::Rotation).
    pub fn as_rotation_mut(&mut self) -> Option<&mut Matrix<T, N>> {
        match self {
            LinearMap::Rotation(m) => Some(m),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AffineTransform<T, const N: usize> {
    linear: LinearMap<T, N>,
    translation: Vector<T, N>,
}

impl<T, const N: usize> AffineTransform<T, N> {
    pub fn new(linear: LinearMap<T, N>, translation: Vector<T, N>) -> Self {
        Self {
            linear,
            translation,
        }
    }

    pub fn linear(&self) -> &LinearMap<T, N> {
        &self.linear
    }

    pub fn translation(&self) -> &Vector<T, N> {
        &self.translation
    }

    pub fn set_linear(&mut self, linear: LinearMap<T, N>) {
        self.linear = linear;
    }

    pub fn set_translation(&mut self, translation: Vector<T, N>) {
        self.translation = translation
    }

    pub fn linear_mut(&mut self) -> &mut LinearMap<T, N> {
        &mut self.linear
    }

    pub fn translation_mut(&mut self) -> &mut Vector<T, N> {
        &mut self.translation
    }
}
