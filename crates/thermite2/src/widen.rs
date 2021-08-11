use crate::*;

pub struct Widen<S: Simd, V, const N: usize> {
    vectors: [V; N],
    _simd: PhantomData<S>,
}
