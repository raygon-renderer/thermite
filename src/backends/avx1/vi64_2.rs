use super::*;

decl!(i64x8: i64 => [__m128i; 4]);
impl<S: Simd> Default for i64x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new([unsafe { _mm_setzero_si128() }; 4])
    }
}

impl<S: Simd> i64x8<S> {
    #[inline(always)]
    fn mapv<F>(mut self, f: F) -> Self
    where
        F: Fn(__m128i, usize) -> __m128i,
    {
        for i in 0..4 {
            self.value[i] = f(self.value[i], i);
        }
        self
    }

    #[inline(always)]
    fn zipv<F>(mut self, b: Self, f: F) -> Self
    where
        F: Fn(__m128i, __m128i) -> __m128i,
    {
        self.mapv(|a, i| f(a, b.value[i]))
    }
}

impl SimdVectorBase<AVX1> for i64x8<AVX1> {
    type Element = i64;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { [_mm_set1_epi64x(value); 4] })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new([_mm_undefined_si128(); 4])
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::undefined().mapv(|_, i| _mm_load_si128((src as *const __m128i).add(i)))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        Self::undefined().mapv(|_, i| _mm_loadu_si128((src as *const __m128i).add(i)))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        for i in 0..4 {
            _mm_store_si128((dst as *mut __m128i).add(i), self.value[i]);
        }
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        for i in 0..4 {
            _mm_storeu_si128((dst as *mut __m128i).add(i), self.value[i]);
        }
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

impl SimdBitwise<AVX1> for i64x8<AVX1> {
    fn and_not(self, other: Self) -> Self {
        self.zipv(other, |a, b| unsafe { _mm_andnot_si128(a, b) })
    }

    const FULL_BITMASK: u16 = 0b1111_1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        let mut bitmask = 0;
        for i in 0..4 {
            // shift mask by 2*i as each vector has 2 64-bit lanes
            bitmask |= unsafe { _mm_movemask_pd(_mm_castsi128_pd(self.value[i])) } << (2 * i);
        }
        bitmask as u16
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        self.zipv(rhs, |a, b| _mm_and_si128(a, b))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        self.zipv(rhs, |a, b| _mm_or_si128(a, b))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        self.zipv(rhs, |a, b| _mm_xor_si128(a, b))
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
        let count = _mm_cvtsi32_si128(count as i32);
        self.mapv(|a, _| _mm_sll_epi64(a, count))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        let count = _mm_cvtsi32_si128(count as i32);
        self.mapv(|a, _| _mm_srl_epi64(a, count))
    }
}

impl PartialEq<Self> for i64x8<AVX1> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::ne(*self, *other).any()
    }
}

impl Eq for i64x8<AVX1> {}

impl SimdMask<AVX1> for i64x8<AVX1> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        self.mapv(|m, i| _mm_blendv_epi8(f.value[i], t.value[i], m))
    }
}

impl SimdVector<AVX1> for i64x8<AVX1> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { [_mm_setzero_si128(); 4] })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(i64::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(i64::MAX)
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
    fn eq(self, other: Self) -> Mask<AVX1, Self> {
        Mask::new(self.zipv(other, |a, b| unsafe { _mm_cmpeq_epi64(a, b) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX1, Self> {
        Mask::new(self.zipv(other, |a, b| unsafe { _mm_cmpgt_epi64(a, b) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        self.zipv(rhs, |l, r| _mm_add_epi64(l, r))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        self.zipv(rhs, |l, r| _mm_sub_epi64(l, r))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        self.zipv(rhs, |l, r| _mm_mullo_epi64x(l, r))
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

impl SimdSignedVector<AVX1> for i64x8<AVX1> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.mapv(|x, _| unsafe { _mm256_abs_epi64x(x) })
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        (self ^ Self::neg_one()) + Self::one()
    }
}

impl_ops!(@UNARY i64x8 AVX1 => Not::not, Neg::neg);
impl_ops!(@BINARY i64x8 AVX1 => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@BINARY i64x8 AVX1 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);
impl_ops!(@SHIFTS i64x8 AVX1 => Shr::shr, Shl::shl);
