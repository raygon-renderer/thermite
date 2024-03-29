use super::*;

decl!(u32x8: u32 => __m256i);
impl<S: Simd> Default for u32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

#[rustfmt::skip]
macro_rules! log_reduce_epu32_avx2 {
    ($value:expr; $op:ident) => {unsafe {
        let ymm0 = $value;

        let xmm0 = _mm256_castsi256_si128($value);
        let xmm1 = _mm256_extracti128_si256($value, 1);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 78);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 229);
        let xmm0 = $op(xmm0, xmm1);

        _mm_cvtsi128_si32(xmm0) as u32
    }};
}

impl SimdVectorBase<AVX2> for u32x8<AVX2> {
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

    decl_base_common!(#[target_feature(enable = "avx2,fma")] u32x8: u32 => __m256i);

    #[inline(always)]
    unsafe fn gather_unchecked(base_ptr: *const Self::Element, indices: Vi32) -> Self {
        Self::new(_mm256_i32gather_epi32(
            base_ptr as _,
            indices.value,
            mem::size_of::<Self::Element>() as _,
        ))
    }

    #[inline(always)]
    unsafe fn gather_masked_unchecked(
        base_ptr: *const Self::Element,
        indices: Vi32,
        mask: Mask<AVX2, Self>,
        default: Self,
    ) -> Self {
        Self::new(_mm256_mask_i32gather_epi32(
            default.value,
            base_ptr as _,
            indices.value,
            mask.value().value,
            mem::size_of::<Self::Element>() as _,
        ))
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
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.value)) as u16 }
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
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        Self::new(_mm256_srlv_epi32(self.value, count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
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

impl PartialEq<Self> for u32x8<AVX2> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // if equal, XOR cancels out and all bits are zero
        unsafe { (*self ^ *other)._mm_none() }
    }
}

impl Eq for u32x8<AVX2> {}

impl SimdMask<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_epi8(f.value, t.value, self.value))
    }

    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        let ones = Mask::<AVX2, Self>::truthy().value();
        0 != _mm256_testc_si256(self.value, ones.value)
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        0 == _mm256_testz_si256(self.value, self.value)
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        0 != _mm256_testz_si256(self.value, self.value)
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
    fn indexed() -> Self {
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
        log_reduce_epu32_avx2!(self.value; _mm_min_epu32)
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        log_reduce_epu32_avx2!(self.value; _mm_max_epu32)
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpeq_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmpgt_epu32x(self.value, other.value) }))
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
        rhs.eq(Self::zero()).select(
            Self::zero(),
            Self::zip(self, rhs, |lhs, rhs| match lhs.checked_div(rhs) {
                Some(value) => value,
                _ => core::hint::unreachable_unchecked(),
            }),
        )
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        rhs.eq(Self::zero()).select(
            Self::zero(),
            Self::zip(self, rhs, |lhs, rhs| match lhs.checked_rem(rhs) {
                Some(value) => value,
                _ => core::hint::unreachable_unchecked(),
            }),
        )
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

    #[inline(always)]
    fn wrapping_sum(self) -> Self::Element {
        log_reduce_epu32_avx2!(self.value; _mm_add_epi32)
    }

    #[inline(always)]
    fn wrapping_product(self) -> Self::Element {
        log_reduce_epu32_avx2!(self.value; _mm_mullo_epi32)
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
        let y0 = Vu32::splat(0x55555555);
        let y1 = Vu32::splat(0x33333333);
        let y2 = Vu32::splat(0x0f0f0f0f);
        let y3 = Vu32::splat(0x00ff00ff);

        let mut x = self;

        x = ((x >> 1) & y0) | ((x & y0) << 1);
        x = ((x >> 2) & y1) | ((x & y1) << 2);
        x = ((x >> 4) & y2) | ((x & y2) << 4);
        x = ((x >> 8) & y3) | ((x & y3) << 8);
        x = (x >> 16) | (x << 16);

        x
    }

    #[inline(always)]
    fn count_ones(self) -> Self {
        Self::new(unsafe { _mm256_popcnt_epi32x(self.value) })
    }

    #[inline(always)]
    fn leading_zeros(mut self) -> Self {
        Self::splat(Self::ELEMENT_SIZE as u32 * 8) - self.log2p1()
    }

    #[inline(always)]
    fn trailing_zeros(self) -> Self {
        Vi32::from_bits(self).trailing_zeros().into_bits()
    }
}

impl Div<Divider<u32>> for u32x8<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Divider<u32>) -> Self {
        Self::new(unsafe { _mm256_div_epu32x(self.value, rhs.multiplier(), rhs.shift()) })
    }
}

impl Div<BranchfreeDivider<u32>> for u32x8<AVX2> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: BranchfreeDivider<u32>) -> Self {
        Self::new(unsafe { _mm256_div_epu32x_bf(self.value, rhs.multiplier(), rhs.shift()) })
    }
}

impl SimdUnsignedIntVector<AVX2> for u32x8<AVX2> {
    #[inline(always)]
    fn next_power_of_two_m1(mut self) -> Self {
        self |= (self >> 1);
        self |= (self >> 2);
        self |= (self >> 4);
        self |= (self >> 8);
        self |= (self >> 16);
        self
    }
}

impl_ops!(@UNARY  u32x8 AVX2 => Not::not);
impl_ops!(@BINARY u32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u32x8 AVX2 => Shr::shr, Shl::shl);

impl SimdFromCast<AVX2, Vi32> for u32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi32) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        Mask::new(from.value().cast()) // same width
    }
}

impl SimdFromCast<AVX2, Vf32> for u32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf32) -> Self {
        Self::new(unsafe { _mm256_cvtps_epu32x(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        // equal width mask, so zero-cost cast
        Mask::new(Self::new(unsafe { _mm256_castps_si256(from.value().value) }))
    }
}

impl SimdFromCast<AVX2, Vf64> for u32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf64) -> Self {
        Self::new(unsafe {
            let low = _mm256_cvtpd_epu32x(from.value.0);
            let high = _mm256_cvtpd_epu32x(from.value.1);

            _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1)
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // cast to bits and truncate
        Mask::new(from.value().into_bits().cast())
    }
}

impl SimdFromCast<AVX2, Vu64> for u32x8<AVX2> {
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
    fn from_cast_mask(from: Mask<AVX2, Vu64>) -> Mask<AVX2, Self> {
        Mask::new(from.value().cast()) // truncate
    }
}

impl SimdFromCast<AVX2, Vi64> for u32x8<AVX2> {
    fn from_cast(from: Vi64) -> Self {
        from.into_bits().cast() // truncate
    }

    fn from_cast_mask(from: Mask<AVX2, Vi64>) -> Mask<AVX2, Self> {
        Mask::new(from.value().cast()) // truncate
    }
}
