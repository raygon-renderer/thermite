use super::*;

decl!(u64x8: u64 => (__m256i, __m256i));
impl<S: Simd> Default for u64x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm256_setzero_si256(), _mm256_setzero_si256()) })
    }
}

impl SimdVectorBase<AVX2> for u64x8<AVX2> {
    type Element = u64;

    const ALIGNMENT: usize = mem::align_of::<__m256i>(); // allow half-alignment

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe {
            let value = transmute(value);
            (_mm256_set1_epi64x(value), _mm256_set1_epi64x(value))
        })
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

    #[inline(always)]
    unsafe fn gather(base_ptr: *const Self::Element, indices: Vi32) -> Self {
        let low_indices = _mm256_castsi256_si128(indices.value);
        let high_indices = _mm256_extracti128_si256(indices.value, 1);

        Self::new((
            _mm256_i32gather_epi64(base_ptr as _, low_indices, mem::size_of::<Self::Element>() as _),
            _mm256_i32gather_epi64(base_ptr as _, high_indices, mem::size_of::<Self::Element>() as _),
        ))
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base_ptr: *const Self::Element,
        indices: Vi32,
        mask: Mask<AVX2, Self>,
        default: Self,
    ) -> Self {
        let low_indices = _mm256_castsi256_si128(indices.value);
        let high_indices = _mm256_extracti128_si256(indices.value, 1);

        let mask = mask.value();

        Self::new((
            _mm256_mask_i32gather_epi64(
                default.value.0,
                base_ptr as _,
                low_indices,
                mask.value.0,
                mem::size_of::<Self::Element>() as _,
            ),
            _mm256_mask_i32gather_epi64(
                default.value.1,
                base_ptr as _,
                high_indices,
                mask.value.1,
                mem::size_of::<Self::Element>() as _,
            ),
        ))
    }
}

impl SimdBitwise<AVX2> for u64x8<AVX2> {
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
            let low = _mm256_movemask_pd(transmute(self.value.0)) as u16;
            let high = _mm256_movemask_pd(transmute(self.value.1)) as u16;

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
            _mm256_and_si256(self.value.0, rhs.value.0),
            _mm256_and_si256(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_or_si256(self.value.0, rhs.value.0),
            _mm256_or_si256(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_xor_si256(self.value.0, rhs.value.0),
            _mm256_xor_si256(self.value.1, rhs.value.1),
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

impl PartialEq<Self> for u64x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl Eq for u64x8<AVX2> {}

impl SimdMask<AVX2> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        let low = _mm256_movemask_epi8(self.value.0) as u32;
        let high = _mm256_movemask_epi8(self.value.1) as u32;
        (low & high) == 0xFFFF_FFFF
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        let low = _mm256_movemask_epi8(self.value.0) as u32;
        let high = _mm256_movemask_epi8(self.value.1) as u32;
        (low | high) != 0
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        let low = _mm256_movemask_epi8(self.value.0) as u32;
        let high = _mm256_movemask_epi8(self.value.1) as u32;
        (low | high) == 0
    }

    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new((
            _mm256_blendv_epi8(f.value.0, t.value.0, self.value.0),
            _mm256_blendv_epi8(f.value.1, t.value.1, self.value.1),
        ))
    }
}

impl SimdVector<AVX2> for u64x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { Self::new((_mm256_setzero_si256(), _mm256_setzero_si256())) }
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
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
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmpeq_epi64(self.value.0, other.value.0),
                _mm256_cmpeq_epi64(self.value.1, other.value.1),
            )
        }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmpgt_epi64(self.value.0, other.value.0),
                _mm256_cmpgt_epi64(self.value.1, other.value.1),
            )
        }))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        self.gt(other) ^ self.eq(other)
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

impl SimdIntVector<AVX2> for u64x8<AVX2> {
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

impl_ops!(@UNARY u64x8 AVX2 => Not::not);
impl_ops!(@BINARY u64x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u64x8 AVX2 => Shr::shr, Shl::shl);

impl SimdCastFrom<AVX2, Vu32> for u64x8<AVX2> {
    fn from_cast(from: Vu32) -> Self {
        brute_force_convert!(&from; u32 => u64)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX2, Vf32> for u64x8<AVX2> {
    fn from_cast(from: Vf32) -> Self {
        brute_force_convert!(&from; f32 => u64)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value().into_bits()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX2, Vi32> for u64x8<AVX2> {
    fn from_cast(from: Vi32) -> Self {
        brute_force_convert!(&from; i32 => u64)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        // convert as unsigned to avoid sign extend overhead
        from.value().into_bits().cast_to::<Vu64>().ne(Self::zero())
    }
}

impl SimdCastFrom<AVX2, Vf64> for u64x8<AVX2> {
    fn from_cast(from: Vf64) -> Self {
        brute_force_convert!(&from; f64 => u64)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // zero-cost transmute for same-width
        Mask::new(from.value().into_bits())
    }
}

/////////////////////////////////////

impl SimdPtrInternal<AVX2, Vi32> for u64x8<AVX2> {
    #[inline]
    unsafe fn _mm_gather_masked(self, mask: Mask<AVX2, Vi32>, default: Vi32) -> Vi32 {
        let mask = mask.value();
        let mask_low = _mm256_castsi256_si128(mask.value);
        let mask_high = _mm256_extracti128_si256(mask.value, 1);

        let default_low = _mm256_castsi256_si128(default.value);
        let default_high = _mm256_extracti128_si256(default.value, 1);

        let low = _mm256_mask_i64gather_epi32(default_low, ptr::null(), self.value.0, mask_low, 1);
        let high = _mm256_mask_i64gather_epi32(default_high, ptr::null(), self.value.1, mask_high, 1);

        Vi32::new(_mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1))
    }
}

impl SimdPtrInternal<AVX2, Vu32> for u64x8<AVX2> {}
impl SimdPtrInternal<AVX2, Vf32> for u64x8<AVX2> {}
impl SimdPtrInternal<AVX2, Vf64> for u64x8<AVX2> {}
impl SimdPtrInternal<AVX2, Vu64> for u64x8<AVX2> {}
