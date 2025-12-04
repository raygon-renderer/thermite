use super::*;

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub fn convert_pd_epi64_limited<R: FloatRegister<Element = f64>>(x: Storage<R>) -> Storage<R::Signed> {
    // https://stackoverflow.com/a/41148578/2083075
    let m = R::splat(0x0018000000000000u64 as i64 as f64);
    <R::Signed as BitsRegister<R>>::from_bits(R::bitxor(R::add(x, m), m))
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub fn convert_pd_epu64_limited<R: FloatRegister<Element = f64>>(x: Storage<R>) -> Storage<R::Bits> {
    // https://stackoverflow.com/a/41148578/2083075
    let m = R::splat(0x0010000000000000u64 as i64 as f64);
    <R::Bits as BitsRegister<R>>::from_bits(R::bitxor(R::add(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub fn convert_epi64_pd_limited<R: FloatRegister<Element = f64>>(mut x: Storage<R::Signed>) -> Storage<R> {
    // https://stackoverflow.com/a/41223013/2083075
    let m = R::splat(0x0018000000000000u64 as i64 as f64);

    // _mm_add_epi64(x, _mm_castpd_si128(m))
    x = <R::Signed as NumericRegister>::add(x, <R::Signed as BitsRegister<R>>::from_bits(m));

    R::sub(<R as BitsRegister<R::Signed>>::from_bits(x), m) // _mm_sub_pd(_mm_castsi128_pd(...), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub fn convert_epu64_pd_limited<R: FloatRegister<Element = f64>>(mut x: Storage<R::Bits>) -> Storage<R> {
    // https://stackoverflow.com/a/41223013/2083075
    let m = R::splat(0x0010000000000000u64 as i64 as f64);

    // _mm_or_si128(x, _mm_castpd_si128(m))
    x = <R::Bits as Register>::bitor(x, <R::Bits as BitsRegister<R>>::from_bits(m));

    R::sub(<R as BitsRegister<R::Bits>>::from_bits(x), m) // _mm_sub_pd(_mm_castsi128_pd(...), m)
}
