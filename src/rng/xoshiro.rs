// TODO

use crate::*;

use super::SimdRng;

#[derive(Debug, Clone, PartialEq)]
pub struct SplitMix64<S: Simd> {
    x: Vu64<S>,
}

const PHI: u64 = 0x9e3779b97f4a7c15;

impl<S: Simd> SplitMix64<S> {
    #[inline(always)]
    pub fn new(seed: Vu64<S>) -> Self {
        SplitMix64 { x: seed }
    }
}

impl<S: Simd> SimdRng<S> for SplitMix64<S> {
    #[inline(always)]
    fn reseed(&mut self, seed: Vu64<S>) {
        self.x = seed;
    }

    #[inline(always)]
    fn next_u32(&mut self) -> Vu32<S> {
        self.x = self.x + Vu64::<S>::splat(PHI);
        let mut z = self.x;

        z = (z ^ (z >> 33)) * Vu64::<S>::splat(0x62A9D9ED799705F5);
        z = (z ^ (z >> 28)) * Vu64::<S>::splat(0xCB24D0A5C88C35B3);

        (z >> 32).cast()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> Vu64<S> {
        self.x = self.x + Vu64::<S>::splat(PHI);
        let mut z = self.x;

        z = (z ^ (z >> 30)) * Vu64::<S>::splat(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * Vu64::<S>::splat(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Xoshiro128Plus<S: Simd> {
    s0: Vu64<S>,
    s1: Vu64<S>,
}

impl<S: Simd> Xoshiro128Plus<S> {
    #[inline(always)]
    pub fn new(seed: Vu64<S>) -> Self {
        let mut rng = SplitMix64::<S>::new(seed);
        Xoshiro128Plus {
            s0: rng.next_u64(),
            s1: rng.next_u64(),
        }
    }
}

impl<S: Simd> SimdRng<S> for Xoshiro128Plus<S> {
    #[inline(always)]
    fn reseed(&mut self, seed: Vu64<S>) {
        *self = Self::new(seed);
    }

    #[inline(always)]
    fn next_u64(&mut self) -> Vu64<S> {
        let result = self.s0 + self.s1;

        self.s1 ^= self.s0;
        self.s0 = self.s0.rol(24) ^ self.s1 ^ (self.s1 << 16);
        self.s1 = self.s1.rol(37);

        result
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Xoshiro256Plus<S: Simd> {
    state: [Vu64<S>; 4],
}

impl<S: Simd> Xoshiro256Plus<S> {
    #[inline(always)]
    pub fn new(seed: Vu64<S>) -> Self {
        let mut rng = SplitMix64::<S>::new(seed);
        Xoshiro256Plus {
            state: [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()],
        }
    }
}

impl<S: Simd> SimdRng<S> for Xoshiro256Plus<S> {
    #[inline(always)]
    fn reseed(&mut self, seed: Vu64<S>) {
        *self = Self::new(seed);
    }

    #[inline(always)]
    fn next_u64(&mut self) -> Vu64<S> {
        let result = self.state[0] + self.state[3];

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;

        self.state[3] = self.state[3].rol(45);

        result
    }
}
