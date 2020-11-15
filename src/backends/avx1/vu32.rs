use super::*;

decl!(u32x8: u32 => __m256i);
impl<S: Simd> Default for u32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

#[rustfmt::skip]
macro_rules! log_reduce_epu32_avx1 {
    ($value:expr; $op:ident) => {unsafe {
        let ymm0 = $value;

        let xmm0 = _mm256_castsi256_si128($value);
        let xmm1 = _mm256_extractf128_si256($value, 1);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 78);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 229);
        let xmm0 = $op(xmm0, xmm1);

        _mm_cvtsi128_si32(xmm0) as u32
    }};
}

impl SimdVectorBase<AVX1> for u32x8<AVX1> {
    type Element = u32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_epi32(value as i32) })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new(_mm256_undefined_si256())
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_load_si256(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_loadu_si256(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_store_si256(ptr as *mut _, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_storeu_si256(ptr as *mut _, self.value)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
        *transmute::<&_, *const Self::Element>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
        *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
        self
    }
}

impl SimdBitwise<AVX1> for u32x8<AVX1> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_si256(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111_1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.value)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm256_and_si256x(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm256_or_si256x(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm256_xor_si256x(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: u32x8<AVX1>) -> Self {
        Self::new(_mm256_srlv_epi32(self.value, count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: u32x8<AVX1>) -> Self {
        Self::new(_mm256_sllv_epi32(self.value, count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(_mm256_sll_epi32(self.value, _mm_cvtsi32_si128(count as i32)))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm256_srl_epi32(self.value, _mm_cvtsi32_si128(count as i32)))
    }
}

impl PartialEq<Self> for u32x8<AVX1> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::ne(*self, *other).any()
    }
}

impl Eq for u32x8<AVX1> {}

impl SimdMask<AVX1> for u32x8<AVX1> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_epi32x(f.value, t.value, self.value))
    }
}

impl SimdVector<AVX1> for u32x8<AVX1> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
    }

    #[inline(always)]
    fn index() -> Self {
        unsafe { Self::new(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_min_epu32(self.value, other.value) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_max_epu32(self.value, other.value) })
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(u32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(u32::MAX)
    }

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        log_reduce_epu32_avx1!(self.value; _mm_min_epu32)
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        log_reduce_epu32_avx1!(self.value; _mm_max_epu32)
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX1, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpeq_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX1, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpgt_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm256_add_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm256_sub_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(_mm256_mullo_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Div::div)
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Rem::rem)
    }
}

impl SimdIntVector<AVX1> for u32x8<AVX1> {
    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        rhs + self.min(!rhs)
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        self.max(rhs) - rhs
    }

    #[inline(always)]
    fn wrapping_sum(self) -> Self::Element {
        log_reduce_epu32_avx1!(self.value; _mm_add_epi32)
    }

    #[inline(always)]
    fn wrapping_product(self) -> Self::Element {
        log_reduce_epu32_avx1!(self.value; _mm_mullo_epi32)
    }
}

impl_ops!(@UNARY  u32x8 AVX1 => Not::not);
impl_ops!(@BINARY u32x8 AVX1 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u32x8 AVX1 => Shr::shr, Shl::shl);

impl SimdFromCast<AVX1, Vi32> for u32x8<AVX1> {
    #[inline(always)]
    fn from_cast(from: Vi32) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vi32>) -> Mask<AVX1, Self> {
        Mask::new(from.value().cast()) // same width
    }
}

impl SimdFromCast<AVX1, Vf32> for u32x8<AVX1> {
    #[inline(always)]
    fn from_cast(from: Vf32) -> Self {
        Self::new(unsafe { _mm256_cvtps_epu32x(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vf32>) -> Mask<AVX1, Self> {
        // equal width mask, so zero-cost cast
        Mask::new(Self::new(unsafe { _mm256_castps_si256(from.value().value) }))
    }
}

impl SimdFromCast<AVX1, Vf64> for u32x8<AVX1> {
    #[inline(always)]
    fn from_cast(from: Vf64) -> Self {
        Self::new(unsafe {
            let low = _mm256_cvtpd_epu32x(from.value.0);
            let high = _mm256_cvtpd_epu32x(from.value.1);

            _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1)
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vf64>) -> Mask<AVX1, Self> {
        // cast to bits and truncate
        Mask::new(from.value().into_bits().cast())
    }
}

impl SimdFromCast<AVX1, Vu64> for u32x8<AVX1> {
    #[inline(always)]
    fn from_cast(from: Vu64) -> Self {
        Self::new(unsafe {
            let (ymm0, ymm1) = from.value;
            let xmm1 = _mm256_castsi256_si128(ymm1);

            _mm256_castps_si256(_mm256_shuffle_ps(
                _mm256_castsi256_ps(_mm256_inserti128_si256(ymm0, xmm1, 1)),
                _mm256_castsi256_ps(_mm256_permute2f128_si256(ymm0, ymm1, 49)),
                136,
            ))
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vu64>) -> Mask<AVX1, Self> {
        Mask::new(from.value().cast()) // truncate
    }
}

impl SimdFromCast<AVX1, Vi64> for u32x8<AVX1> {
    fn from_cast(from: Vi64) -> Self {
        from.into_bits().cast() // truncate
    }

    fn from_cast_mask(from: Mask<AVX1, Vi64>) -> Mask<AVX1, Self> {
        Mask::new(from.value().cast()) // truncate
    }
}
