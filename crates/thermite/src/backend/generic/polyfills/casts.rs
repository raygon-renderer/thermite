use super::*;

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub fn convert_pd_epi64_limited<R: FloatRegister<Element = f64>>(x: Storage<R>) -> Storage<R::SignedBits> {
    // https://stackoverflow.com/a/41148578/2083075
    //
    // Adding 1.5 * 2^52 parks the integer in the low 52 bits of the mantissa, biased
    // by 2^51: bits(x + m) == bits(m) + x for x in [-2^51, 2^51]. The unbias is
    // therefore an integer SUBTRACT, not an xor - xor only cancels the bias when it
    // does not borrow out of bit 51, i.e. only for x >= 0. With xor, x = -2 came back
    // as 2^52 - 2 (which is what made wasm `powf` see a bogus exponent and overflow).
    let m = R::splat(0x0018000000000000u64 as i64 as f64);

    <R::SignedBits as NumericRegister>::sub(
        <R::SignedBits as BitCastRegister<R>>::from_bits(R::add(x, m)),
        <R::SignedBits as BitCastRegister<R>>::from_bits(m),
    )
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub fn convert_pd_epu64_limited<R: FloatRegister<Element = f64>>(x: Storage<R>) -> Storage<R::Bits> {
    // https://stackoverflow.com/a/41148578/2083075
    let m = R::splat(0x0010000000000000u64 as i64 as f64);
    <R::Bits as BitCastRegister<R>>::from_bits(R::bitxor(R::add(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub fn convert_epi64_pd_limited<R: FloatRegister<Element = f64>>(mut x: Storage<R::SignedBits>) -> Storage<R> {
    // https://stackoverflow.com/a/41223013/2083075
    let m = R::splat(0x0018000000000000u64 as i64 as f64);

    // _mm_add_epi64(x, _mm_castpd_si128(m))
    x = <R::SignedBits as NumericRegister>::add(x, <R::SignedBits as BitCastRegister<R>>::from_bits(m));

    R::sub(<R as BitCastRegister<R::SignedBits>>::from_bits(x), m) // _mm_sub_pd(_mm_castsi128_pd(...), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub fn convert_epu64_pd_limited<R: FloatRegister<Element = f64>>(mut x: Storage<R::Bits>) -> Storage<R> {
    // https://stackoverflow.com/a/41223013/2083075
    let m = R::splat(0x0010000000000000u64 as i64 as f64);

    // _mm_or_si128(x, _mm_castpd_si128(m))
    x = <R::Bits as BitwiseRegister>::bitor(x, <R::Bits as BitCastRegister<R>>::from_bits(m));

    R::sub(<R as BitCastRegister<R::Bits>>::from_bits(x), m) // _mm_sub_pd(_mm_castsi128_pd(...), m)
}
