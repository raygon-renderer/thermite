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

    decl_base_common!(#[target_feature(enable = "avx2,fma")] u64x8: u64 => __m256i);

    #[inline(always)]
    unsafe fn gather_unchecked(base_ptr: *const Self::Element, indices: Vi32) -> Self {
        let low_indices = _mm256_castsi256_si128(indices.value);
        let high_indices = _mm256_extracti128_si256(indices.value, 1);

        Self::new((
            _mm256_i32gather_epi64(base_ptr as _, low_indices, mem::size_of::<Self::Element>() as _),
            _mm256_i32gather_epi64(base_ptr as _, high_indices, mem::size_of::<Self::Element>() as _),
        ))
    }

    #[inline(always)]
    unsafe fn gather_masked_unchecked(
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
        let count = _mm_cvtsi32_si128(count as i32);

        Self::new((
            _mm256_sll_epi64(self.value.0, count),
            _mm256_sll_epi64(self.value.1, count),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        let count = _mm_cvtsi32_si128(count as i32);

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
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new((
            _mm256_blendv_epi8(f.value.0, t.value.0, self.value.0),
            _mm256_blendv_epi8(f.value.1, t.value.1, self.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        _mm256_movemask_epi8(_mm256_and_si256(self.value.0, self.value.1)) as u32 == u32::MAX
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        _mm256_movemask_epi8(_mm256_or_si256(self.value.0, self.value.1)) != 0
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        _mm256_movemask_epi8(_mm256_or_si256(self.value.0, self.value.1)) == 0
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
    fn indexed() -> Self {
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
            _mm256_add_epi64(self.value.0, rhs.value.0),
            _mm256_add_epi64(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new((
            _mm256_sub_epi64(self.value.0, rhs.value.0),
            _mm256_sub_epi64(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new((
            _mm256_mullo_epi64x(self.value.0, rhs.value.0),
            _mm256_mullo_epi64x(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        rhs.eq(Self::zero()).select(
            Self::zero(),
            Self::zip(self, rhs, |lhs, rhs| match lhs.checked_div(rhs) {
                Some(value) => value,
                _ => std::hint::unreachable_unchecked(),
            }),
        )
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        rhs.eq(Self::zero()).select(
            Self::zero(),
            Self::zip(self, rhs, |lhs, rhs| match lhs.checked_rem(rhs) {
                Some(value) => value,
                _ => std::hint::unreachable_unchecked(),
            }),
        )
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

    #[inline(always)]
    fn rolv(self, cnt: Vu32) -> Self {
        unsafe { Self::zip(self, cnt, |x, r| x.rotate_left(r)) }
    }

    #[inline(always)]
    fn rorv(self, cnt: Vu32) -> Self {
        unsafe { Self::zip(self, cnt, |x, r| x.rotate_right(r)) }
    }

    #[inline(always)]
    fn reverse_bits(self) -> Self {
        let y0 = Vu64::splat(0x5555555555555555);
        let y1 = Vu64::splat(0x3333333333333333);
        let y2 = Vu64::splat(0x0f0f0f0f0f0f0f0f);
        let y3 = Vu64::splat(0x00ff00ff00ff00ff);
        let y4 = Vu64::splat(0x0000ffff0000ffff);

        let mut x = self;

        x = (((x >> 1) & y0) | ((x & y0) << 1));
        x = (((x >> 2) & y1) | ((x & y1) << 2));
        x = (((x >> 4) & y2) | ((x & y2) << 4));
        x = (((x >> 8) & y3) | ((x & y3) << 8));
        x = (((x >> 16) & y4) | ((x & y4) << 16));
        x = ((x >> 32) | (x << 32));

        x
    }

    #[inline(always)]
    fn count_ones(self) -> Self {
        Self::new(unsafe { (_mm256_popcnt_epi64x(self.value.0), _mm256_popcnt_epi64x(self.value.1)) })
    }

    #[inline(always)]
    fn leading_zeros(mut self) -> Self {
        Self::splat(Self::ELEMENT_SIZE as u64 * 8) - self.log2p1()
    }

    #[inline(always)]
    fn trailing_zeros(self) -> Self {
        Vi64::from_bits(self).trailing_zeros().into_bits()
    }
}

impl Div<Divider<u64>> for u64x8<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Divider<u64>) -> Self {
        let multiplier = rhs.multiplier();
        let shift = rhs.shift();

        Self::new(unsafe {
            (
                _mm256_div_epu64x(self.value.0, multiplier, shift),
                _mm256_div_epu64x(self.value.1, multiplier, shift),
            )
        })
    }
}

impl SimdUnsignedIntVector<AVX2> for u64x8<AVX2> {
    #[inline(always)]
    fn next_power_of_two_m1(mut self) -> Self {
        self |= (self >> 1);
        self |= (self >> 2);
        self |= (self >> 4);
        self |= (self >> 8);
        self |= (self >> 16);
        self |= (self >> 32);
        self
    }
}

impl_ops!(@UNARY u64x8 AVX2 => Not::not);
impl_ops!(@BINARY u64x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u64x8 AVX2 => Shr::shr, Shl::shl);

impl SimdFromCast<AVX2, Vu32> for u64x8<AVX2> {
    fn from_cast(from: Vu32) -> Self {
        Self::new(unsafe {
            (
                _mm256_cvtepu32_epi64(_mm256_castsi256_si128(from.value)),
                _mm256_cvtepu32_epi64(_mm256_extracti128_si256(from.value, 1)),
            )
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vf32> for u64x8<AVX2> {
    fn from_cast(from: Vf32) -> Self {
        decl_brute_force_convert!(#[target_feature(enable = "avx2")] f32 => u64);
        unsafe { do_convert(from) }
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value().into_bits()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vi32> for u64x8<AVX2> {
    fn from_cast(from: Vi32) -> Self {
        // zero extend
        from.cast_to::<Vu32>().cast()
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vf64> for u64x8<AVX2> {
    fn from_cast(from: Vf64) -> Self {
        decl_brute_force_convert!(#[target_feature(enable = "avx2")] f64 => u64);
        unsafe { do_convert(from) }
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // zero-cost transmute for same-width
        Mask::new(from.value().into_bits())
    }
}

impl SimdFromCast<AVX2, Vi64> for u64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi64) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi64>) -> Mask<AVX2, Self> {
        Mask::new(Self::new(from.value().value))
    }
}

/////////////////////////////////////

impl SimdPtrInternal<AVX2, Vi32> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> Vi32 {
        let low = _mm256_i64gather_epi32(ptr::null(), self.value.0, 1);
        let high = _mm256_i64gather_epi32(ptr::null(), self.value.1, 1);

        Vi32::new(_mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1))
    }

    #[inline(always)]
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

impl SimdPtrInternal<AVX2, Vu32> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> Vu32 {
        <Self as SimdPtrInternal<AVX2, Vi32>>::_mm_gather(self).into_bits()
    }

    #[inline(always)]
    unsafe fn _mm_gather_masked(self, mask: Mask<AVX2, Vu32>, default: Vu32) -> Vu32 {
        // just load as Vi32 and transmute
        self._mm_gather_masked(mask.cast_to::<Vi32>(), default.cast())
            .into_bits()
    }
}

impl SimdPtrInternal<AVX2, Vf32> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> Vf32 {
        let low = _mm256_i64gather_ps(ptr::null(), self.value.0, 1);
        let high = _mm256_i64gather_ps(ptr::null(), self.value.1, 1);

        Vf32::new(_mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1))
    }

    #[inline(always)]
    unsafe fn _mm_gather_masked(self, mask: Mask<AVX2, Vf32>, default: Vf32) -> Vf32 {
        let mask = mask.value();
        let mask_low = _mm256_castps256_ps128(mask.value);
        let mask_high = _mm256_extractf128_ps(mask.value, 1);

        let default_low = _mm256_castps256_ps128(default.value);
        let default_high = _mm256_extractf128_ps(default.value, 1);

        let low = _mm256_mask_i64gather_ps(default_low, ptr::null(), self.value.0, mask_low, 1);
        let high = _mm256_mask_i64gather_ps(default_high, ptr::null(), self.value.1, mask_high, 1);

        Vf32::new(_mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1))
    }
}

impl SimdPtrInternal<AVX2, Vf64> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> Vf64 {
        Vf64::new((
            _mm256_i64gather_pd(ptr::null(), self.value.0, 1),
            _mm256_i64gather_pd(ptr::null(), self.value.1, 1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_gather_masked(self, mask: Mask<AVX2, Vf64>, default: Vf64) -> Vf64 {
        let mask = mask.value();

        Vf64::new((
            _mm256_mask_i64gather_pd(default.value.0, ptr::null(), self.value.0, mask.value.0, 1),
            _mm256_mask_i64gather_pd(default.value.1, ptr::null(), self.value.1, mask.value.1, 1),
        ))
    }
}

impl SimdPtrInternal<AVX2, Vu64> for u64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_gather(self) -> Vu64 {
        Vu64::new((
            _mm256_i64gather_epi64(ptr::null(), self.value.0, 1),
            _mm256_i64gather_epi64(ptr::null(), self.value.1, 1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_gather_masked(self, mask: Mask<AVX2, Vu64>, default: Vu64) -> Vu64 {
        let mask = mask.value();

        Vu64::new((
            _mm256_mask_i64gather_epi64(default.value.0, ptr::null(), self.value.0, mask.value.0, 1),
            _mm256_mask_i64gather_epi64(default.value.1, ptr::null(), self.value.1, mask.value.1, 1),
        ))
    }
}
