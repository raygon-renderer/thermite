use super::*;

decl!(u32x4: u32 => __m128i);
impl<S: Simd> Default for u32x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm_setzero_si128() })
    }
}

impl SimdVectorBase<SSE42> for u32x4<SSE42> {
    type Element = u32;

    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm_set1_epi32(value as i32) })
    }

    unsafe fn undefined() -> Self {
        Self::new(_mm_undefined_si128())
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm_load_si128(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm_loadu_si128(ptr as *const _))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm_store_si128(ptr as *mut _, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, ptr: *mut Self::Element) {
        _mm_storeu_si128(ptr as *mut _, self.value)
    }

    decl_base_common!(#[target_feature(enable = "sse4.1")] u32x4: u32 => __m128i);
}

impl SimdBitwise<SSE42> for u32x4<SSE42> {
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm_andnot_si128(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    fn bitmask(self) -> u16 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.value)) }
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
        Self::zip(self, count, Shr::shr)
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        Self::zip(self, count, Shl::shl)
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(_mm_sll_epi32(self.value, _mm_cvtsi32_si128(count as i32)))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm_srl_epi32(self.value, _mm_cvtsi32_si128(count as i32)))
    }
}

impl PartialEq<Self> for u32x4<SSE42> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE42>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE42>>::ne(*self, *other).any()
    }
}

impl Eq for u32x4<SSE42> {}

impl SimdMask<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_epi8(f.value, t.value, self.value))
    }

    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        _mm_movemask_epi8(self.value) as u16 == u16::MAX
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        _mm_movemask_epi8(self.value) != 0
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        _mm_movemask_epi8(self.value) == 0
    }
}
