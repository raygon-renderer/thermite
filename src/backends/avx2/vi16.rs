use super::*;

decl!(i16x8: i16 => __m128i);
impl<S: Simd> Default for i16x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm_setzero_si128() })
    }
}

impl SimdVectorBase<AVX2> for i16x8<AVX2> {
    type Element = i16;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm_set1_epi16(value) })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new(_mm_undefined_si128())
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(_mm_load_si128(src as *const _))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(_mm_loadu_si128(src as *const _))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        _mm_store_si128(dst as *mut _, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        _mm_storeu_si128(dst as *mut _, self.value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
        *transmute::<&_, *const Self::Element>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
        *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
        self
    }
}
