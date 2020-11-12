use super::*;

decl!(i32x8: i32 => [__m128i; 2]);
impl<S: Simd> Default for i32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new([unsafe { _mm_setzero_si128() }; 2])
    }
}

impl SimdVectorBase<AVX1> for i32x8<AVX1> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { [_mm_set1_epi32(value); 2] })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new([_mm_undefined_si128(); 2])
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::new([_mm_load_si128(src as *const _), _mm_load_si128(src.add(4) as *const _)])
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        let src = src as *const _;
        Self::new([_mm_load_si128(src), _mm_load_si128(src.add(1))])
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        let dst = dst as *mut _;
        _mm_store_si128(dst, self.value[0]);
        _mm_store_si128(dst.add(1), self.value[1]);
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        let dst = dst as *mut _;
        _mm_storeu_si128(dst, self.value[0]);
        _mm_storeu_si128(dst.add(1), self.value[1]);
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
