use super::*;

decl!(u64x8: u64 => (__m256i, __m256i));
impl<S: Simd> Default for u64x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm256_setzero_si256(), _mm256_setzero_si256()) })
    }
}

impl SimdVectorBase<AVX1> for u64x8<AVX1> {
    type Element = u64;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe {
            let value = transmute(value);
            (_mm256_set1_epi64x(value), _mm256_set1_epi64x(value))
        })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new((_mm256_undefined_si256(), _mm256_undefined_si256()))
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const __m256i;
        Self::new((_mm256_load_si256(ptr), _mm256_load_si256(ptr.add(1))))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        let ptr = ptr as *mut __m256i;

        _mm256_store_si256(ptr, self.value.0);
        _mm256_store_si256(ptr.add(1), self.value.1);
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

impl SimdBitwise<AVX1> for u64x8<AVX1> {
    fn and_not(self, other: Self) -> Self {
        unsafe {
            Self::new((
                _mm256_andnot_si256(self.value.0, other.value.0),
                _mm256_andnot_si256(self.value.1, other.value.1),
            ))
        }
    }

    const FULL_BITMASK: u16 = 0b11111111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe {
            let low = _mm256_movemask_pd(_mm256_castsi256_pd(self.value.0)) as u16;
            let high = _mm256_movemask_pd(_mm256_castsi256_pd(self.value.1)) as u16;

            low | (high << 4)
        }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new((
            _mm256_and_si256x(self.value.0, rhs.value.0),
            _mm256_and_si256x(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_or_si256x(self.value.0, rhs.value.0),
            _mm256_or_si256x(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_xor_si256x(self.value.0, rhs.value.0),
            _mm256_xor_si256x(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        let low = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(count.value));
        let high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(count.value, 1));

        Self::new((
            _mm256_srlv_epi64(self.value.0, low),
            _mm256_srlv_epi64(self.value.1, high),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        let low = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(count.value));
        let high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(count.value, 1));

        Self::new((
            _mm256_sllv_epi64(self.value.0, low),
            _mm256_sllv_epi64(self.value.1, high),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        let count = _mm_setr_epi32(count as i32, 0, 0, 0);

        Self::new((
            _mm256_sll_epi64(self.value.0, count),
            _mm256_sll_epi64(self.value.1, count),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        let count = _mm_setr_epi32(count as i32, 0, 0, 0);

        Self::new((
            _mm256_srl_epi64(self.value.0, count),
            _mm256_srl_epi64(self.value.1, count),
        ))
    }
}

impl PartialEq<Self> for u64x8<AVX1> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX1>>::ne(*self, *other).any()
    }
}

impl Eq for u64x8<AVX1> {}

impl SimdMask<AVX1> for u64x8<AVX1> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new((
            _mm256_blendv_epi64x(f.value.0, t.value.0, self.value.0),
            _mm256_blendv_epi64x(f.value.1, t.value.1, self.value.1),
        ))
    }
}

impl SimdVector<AVX1> for u64x8<AVX1> {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { Self::new((_mm256_setzero_si256(), _mm256_setzero_si256())) }
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
    }

    #[inline(always)]
    fn index() -> Self {
        unsafe { Self::new((_mm256_setr_epi64x(0, 1, 2, 3), _mm256_setr_epi64x(4, 5, 6, 7))) }
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(u64::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(u64::MAX)
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
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmpeq_epi64(self.value.0, other.value.0),
                _mm256_cmpeq_epi64(self.value.1, other.value.1),
            )
        }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX1, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmpgt_epi64(self.value.0, other.value.0),
                _mm256_cmpgt_epi64(self.value.1, other.value.1),
            )
        }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new((
            _mm256_add_epi32(self.value.0, rhs.value.0),
            _mm256_add_epi32(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new((
            _mm256_sub_epi32(self.value.0, rhs.value.0),
            _mm256_sub_epi32(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new((
            _mm256_mullo_epi32(self.value.0, rhs.value.0),
            _mm256_mullo_epi32(self.value.1, rhs.value.1),
        ))
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

impl SimdIntVector<AVX1> for u64x8<AVX1> {
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

impl_ops!(@UNARY u64x8 AVX1 => Not::not);
impl_ops!(@BINARY u64x8 AVX1 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u64x8 AVX1 => Shr::shr, Shl::shl);

impl SimdCastFrom<AVX1, Vu32> for u64x8<AVX1> {
    fn from_cast(from: Vu32) -> Self {
        Self::new(unsafe {
            (
                _mm256_cvtepu32_epi64(_mm256_castsi256_si128(from.value)),
                _mm256_cvtepu32_epi64(_mm256_extracti128_si256(from.value, 1)),
            )
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vu32>) -> Mask<AVX1, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX1, Vf32> for u64x8<AVX1> {
    fn from_cast(from: Vf32) -> Self {
        decl_brute_force_convert!(#[target_feature(enable = "avx")] f32 => u64);
        unsafe { do_convert(from) }
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vf32>) -> Mask<AVX1, Self> {
        Self::from_cast(from.value().into_bits()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX1, Vi32> for u64x8<AVX1> {
    fn from_cast(from: Vi32) -> Self {
        // zero extend
        from.cast_to::<Vu32>().cast()
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vi32>) -> Mask<AVX1, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX1, Vf64> for u64x8<AVX1> {
    fn from_cast(from: Vf64) -> Self {
        decl_brute_force_convert!(#[target_feature(enable = "avx")] f64 => u64);
        unsafe { do_convert(from) }
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vf64>) -> Mask<AVX1, Self> {
        // zero-cost transmute for same-width
        Mask::new(from.value().into_bits())
    }
}

impl SimdCastFrom<AVX1, Vi64> for u64x8<AVX1> {
    #[inline(always)]
    fn from_cast(from: Vi64) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX1, Vi64>) -> Mask<AVX1, Self> {
        Mask::new(Self::new(from.value().value))
    }
}

/////////////////////////////////////

impl SimdPtrInternal<AVX1, Vi32> for u64x8<AVX1> {}

impl SimdPtrInternal<AVX1, Vu32> for u64x8<AVX1> {}

impl SimdPtrInternal<AVX1, Vf32> for u64x8<AVX1> {}

impl SimdPtrInternal<AVX1, Vf64> for u64x8<AVX1> {}

impl SimdPtrInternal<AVX1, Vu64> for u64x8<AVX1> {}
