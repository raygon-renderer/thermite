#![allow(unused)]

use crate::*;

use super::SimdRng;

const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdb;
const PCG32_MULT: u64 = 0x5851f42d4c957f2d;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PCG32<S: Simd> {
    state: S::Vu64,
    inc: S::Vu64,
}

impl<S: Simd> PCG32<S> {
    #[inline(always)]
    pub fn new(seed: Vu64<S>) -> Self {
        let mut rng = PCG32 {
            state: unsafe { Vu64::<S>::undefined() },
            inc: unsafe { Vu64::<S>::undefined() },
        };
        rng.reseed(seed);
        rng
    }
}

// TODO: #[dispatch]
impl<S: Simd> SimdRng<S> for PCG32<S> {
    #[inline]
    fn reseed(&mut self, seed: Vu64<S>) {
        self.state = Vu64::<S>::zero();
        self.inc = (seed << 1) | Vu64::<S>::one();

        let _ = self.next_u32();
        self.state += Vu64::<S>::splat(PCG32_DEFAULT_STATE);
        let _ = self.next_u32();
    }

    #[inline]
    fn next_u32(&mut self) -> Vu32<S> {
        let old_state = self.state;
        self.state = old_state * Vu64::<S>::splat(PCG32_MULT) + self.inc;
        let xorshifted = <Vu32<S> as SimdFromCast<S, Vu64<S>>>::from_cast(((old_state >> 18) ^ old_state) >> 27);
        let rot_offset = <Vu32<S> as SimdFromCast<S, Vu64<S>>>::from_cast(old_state >> 59);

        xorshifted.rorv(rot_offset)
    }
}
