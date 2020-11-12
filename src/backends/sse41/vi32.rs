use super::*;

decl!(i32x4: i32 => __m128i);
impl<S: Simd> Default for i32x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm_setzero_si128() })
    }
}

impl SimdVectorBase<SSE41> for i32x4<SSE41> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm_set1_epi32(value) })
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
    #[target_feature(enable = "sse4.1")]
    unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
        *transmute::<&_, *const Self::Element>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
        *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
        self
    }
}

impl SimdBitwise<SSE41> for i32x4<SSE41> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm_andnot_si128(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.value)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm_and_si128(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm_or_si128(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm_xor_si128(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        Self::zip(self, count, |x, s| x >> s)
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        Self::zip(self, count, |x, s| x << s)
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(_mm_sll_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm_srl_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }
}
