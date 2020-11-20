use crate::*;

pub mod pcg32;

pub trait SimdRng<S: Simd> {
    fn reseed(&mut self, seed: Vu64<S>);
    fn next_u32(&mut self) -> Vu32<S>;

    #[inline(always)]
    fn next_u64(&mut self) -> Vu64<S> {
        let low: Vu64<S> = self.next_u32().cast();
        let high: Vu64<S> = self.next_u32().cast();

        low | (high << 32)
    }

    #[inline(always)]
    fn next_f32(&mut self) -> Vf32<S> {
        Vf32::<S>::from_bits((self.next_u32() >> 9) | Vu32::<S>::splat(0x3f800000)) - Vf32::<S>::one()
    }

    #[inline(always)]
    fn next_f64(&mut self) -> Vf64<S> {
        Vf64::<S>::from_bits((self.next_u64() >> 20) | Vu64::<S>::splat(0x3ff0000000000000)) - Vf64::<S>::one()
    }
}
