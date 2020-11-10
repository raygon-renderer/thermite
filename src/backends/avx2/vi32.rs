use super::*;

decl!(i32x8: i32 => __m256i);
impl<S: Simd> Default for i32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

#[rustfmt::skip]
macro_rules! log_reduce_epi32_avx2 {
    ($value:expr; $op:ident) => {unsafe {
        let ymm0 = $value;

        let xmm0 = _mm256_castsi256_si128($value);
        let xmm1 = _mm256_extracti128_si256($value, 1);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 78);
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_shuffle_epi32(xmm0, 229);
        let xmm0 = $op(xmm0, xmm1);

        _mm_cvtsi128_si32(xmm0)
    }};
}

impl SimdVectorBase<AVX2> for i32x8<AVX2> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_epi32(value) })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new(_mm256_undefined_si256())
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(_mm256_load_si256(src as *const _))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(_mm256_loadu_si256(src as *const _))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        _mm256_store_si256(dst as *mut _, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        _mm256_storeu_si256(dst as *mut _, self.value)
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
        Self::new(_mm256_i32gather_epi32(
            base_ptr as _,
            indices.value,
            mem::size_of::<Self::Element>() as _,
        ))
    }

    #[inline(always)]
    unsafe fn gather_masked(
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

impl SimdBitwise<AVX2> for i32x8<AVX2> {
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
        Self::new(_mm256_sll_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm256_srl_epi32(self.value, _mm_setr_epi32(count as i32, 0, 0, 0)))
    }
}

impl PartialEq<Self> for i32x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl Eq for i32x8<AVX2> {}

impl SimdMask<AVX2> for i32x8<AVX2> {
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

impl SimdVector<AVX2> for i32x8<AVX2> {
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
        Self::new(unsafe { _mm256_min_epi32(self.value, other.value) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_max_epi32(self.value, other.value) })
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(i32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(i32::MAX)
    }

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        log_reduce_epi32_avx2!(self.value; _mm_min_epi32)
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        log_reduce_epi32_avx2!(self.value; _mm_max_epi32)
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

impl SimdIntoBits<AVX2, Vu32> for i32x8<AVX2> {
    #[inline(always)]
    fn into_bits(self) -> Vu32 {
        u32x8::new(self.value)
    }
}

impl SimdFromBits<AVX2, Vu32> for i32x8<AVX2> {
    #[inline(always)]
    fn from_bits(bits: Vu32) -> Self {
        Self::new(bits.value)
    }
}

impl SimdIntVector<AVX2> for i32x8<AVX2> {
    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        Self::new(unsafe { _mm256_adds_epi32(self.value, rhs.value) })
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(unsafe { _mm256_subs_epi32(self.value, rhs.value) })
    }

    #[inline(always)]
    fn wrapping_sum(self) -> Self::Element {
        log_reduce_epi32_avx2!(self.value; _mm_add_epi32)
    }

    #[inline(always)]
    fn wrapping_product(self) -> Self::Element {
        log_reduce_epi32_avx2!(self.value; _mm_mullo_epi32)
    }
}

impl SimdSignedVector<AVX2> for i32x8<AVX2> {
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
        Self::new(unsafe { _mm256_abs_epi32(self.value) })
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        Self::new(_mm256_sign_epi32(self.value, _mm256_set1_epi32(-1)))
    }
}

impl_ops!(@UNARY i32x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY i32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS i32x8 AVX2 => Shr::shr, Shl::shl);

impl SimdCastFrom<AVX2, Vf32> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf32) -> Self {
        Self::new(unsafe { _mm256_cvttps_epi32(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_castps_si256(from.value().value) }))
    }
}

impl SimdCastFrom<AVX2, Vu32> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vu32) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu32>) -> Mask<AVX2, Self> {
        Mask::new(from.value().cast())
    }
}

impl SimdCastFrom<AVX2, Vu64> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vu64) -> Self {
        Self::from_bits(from.cast()) // truncate
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu64>) -> Mask<AVX2, Self> {
        Mask::new(from.value().cast()) // truncate
    }
}

impl SimdCastFrom<AVX2, Vf64> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf64) -> Self {
        Self::new(unsafe {
            let low = _mm256_cvtpd_epi32(from.value.0);
            let high = _mm256_cvtpd_epi32(from.value.1);

            _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1)
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // skip float conversion and go through raw bits
        Mask::new(Vi32::from_bits(from.value().into_bits().cast_to::<Vu32>()))
    }
}

impl SimdCastFrom<AVX2, Vi64> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi64) -> Self {
        Self::from_bits(from.into_bits().cast()) // truncate
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi64>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_bits(from.value().into_bits().cast())) // truncate
    }
}
