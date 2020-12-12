use thermite::*;

pub struct Vector2D<S: Simd> {
    pub x: VectorBuffer<S, Vf32<S>>,
    pub y: VectorBuffer<S, Vf32<S>>,
}

impl<S: Simd> Vector2D<S> {
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Vf32<S>, Vf32<S>)> + 'a {
        self.x.iter_vectors().zip(self.y.iter_vectors())
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a mut Vf32<S>, &'a mut Vf32<S>)> + 'a {
        self.x.iter_vectors_mut().zip(self.y.iter_vectors_mut())
    }
}

pub struct System<S: Simd> {
    pub pos: Vector2D<S>,
    pub vel: Vector2D<S>,
}

impl<S: Simd> System<S> {
    pub fn update(&mut self, dt: f32) {
        let dt = Vf32::<S>::splat(dt);

        debug_assert_eq!(self.pos.x.len(), self.pos.y.len());
        debug_assert_eq!(self.vel.x.len(), self.vel.y.len());
        debug_assert_eq!(self.pos.x.len(), self.vel.y.len());

        for ((px, py), (vx, vy)) in self.pos.iter_mut().zip(self.vel.iter()) {
            *px = dt.mul_add(vx, *px);
            *py = dt.mul_add(vy, *py);
        }
    }
}

fn main() {}
