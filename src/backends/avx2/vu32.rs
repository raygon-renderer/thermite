use super::*;

decl!(u32x8: u32 => __m256i);
impl<S: Simd> Default for u32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

impl SimdVectorBase<AVX2> for u32x8<AVX2> {
    type Element = u32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_epi32(value as i32) })
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

impl SimdBitwise<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_si256(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111_1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_ps(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm256_and_si256(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm256_or_si256(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm256_xor_si256(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: u32x8<AVX2>) -> Self {
        Self::new(_mm256_srlv_epi32(self.value, count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: u32x8<AVX2>) -> Self {
        Self::new(_mm256_sllv_epi32(self.value, count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(_mm256_sll_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm256_srl_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }
}

impl PartialEq<Self> for u32x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl Eq for u32x8<AVX2> {}

impl SimdMask<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        _mm256_movemask_epi8(self.value) as u32 == 0xFFFF_FFFF
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        _mm256_movemask_epi8(self.value) as u32 != 0
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        _mm256_movemask_epi8(self.value) as u32 == 0
    }

    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_epi8(f.value, t.value, self.value))
    }
}

impl SimdVector<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
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

    #[inline]
    fn min_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpeq_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpgt_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        self.gt(other) ^ self.eq(other)
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

impl SimdIntVector<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        rhs + self.min(!rhs)
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        self.max(rhs) - rhs
    }

    fn wrapping_sum(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|sum, x| sum.wrapping_add(x)) }
    }

    fn wrapping_product(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|prod, x| x.wrapping_mul(prod)) }
    }
}

impl_ops!(@UNARY  u32x8 AVX2 => Not::not);
impl_ops!(@BINARY u32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u32x8 AVX2 => Shr::shr, Shl::shl);

impl SimdCastFrom<AVX2, Vi32> for u32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi32) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_cast(from.value())) // same width
    }
}

impl SimdCastFrom<AVX2, Vf32> for u32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf32) -> Self {
        Self::new(unsafe {
            // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
            // produces different results from `f32 as u32` with negaitve values and values larger than some value
            let xmm0 = from.value;
            let xmm1 = _mm256_set1_ps(f32::from_bits(0x4f000000));
            let xmm2 = _mm256_cmp_ps(from.value, xmm1, _CMP_LT_OQ);
            let xmm1 = _mm256_sub_ps(xmm0, xmm1);
            let xmm1 = _mm256_cvtps_epi32(xmm1);
            let xmm3 = _mm256_set1_epi32(0x80000000u32 as i32);
            let xmm1 = _mm256_xor_si256(xmm1, xmm3);
            let xmm0 = _mm256_cvtps_epi32(xmm0);
            let xmm0 = _mm256_blendv_ps(_mm256_castsi256_ps(xmm1), _mm256_castsi256_ps(xmm0), xmm2);

            _mm256_castps_si256(xmm0)
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_castps_si256(from.value().value) }))
    }
}
