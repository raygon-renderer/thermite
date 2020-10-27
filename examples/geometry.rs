use thermite::*;

#[derive(Debug, Clone, Copy)]
pub struct Vector3xN<S: Simd> {
    pub x: S::Vf32,
    pub y: S::Vf32,
    pub z: S::Vf32,
}

impl<S: Simd> Vector3xN<S> {
    pub fn dot(&self, other: &Self) -> S::Vf32 {
        self.x.mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y.mul_sub(other.z, self.z * other.y),
            y: self.z.mul_sub(other.x, self.x * other.z),
            z: self.x.mul_sub(other.y, self.y * other.x),
        }
    }

    pub fn norm_squared(&self) -> S::Vf32 {
        self.dot(self)
    }

    pub fn norm(&self) -> S::Vf32 {
        self.norm_squared().sqrt()
    }

    pub fn normalize(&self) -> Self {
        let inv_norm = self.norm_squared().rsqrt();

        Self {
            x: self.x * inv_norm,
            y: self.y * inv_norm,
            z: self.z * inv_norm,
        }
    }
}

fn main() {
    println!("Hello, world!");
}
