use super::*;

decl!(i32x8: i32 => __m256i);
impl<S: Simd> Default for i32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_si256() })
    }
}

impl SimdVectorBase<AVX2> for i32x8<AVX2> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_epi32(value) })
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

impl SimdBitwise<AVX2> for i32x8<AVX2> {
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
        Self::new(_mm256_blendv_epi8(t.value, f.value, self.value))
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
    fn min_value() -> Self {
        Self::splat(i32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(i32::MAX)
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

impl SimdIntVector<AVX2> for i32x8<AVX2> {
    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        unsafe {
            let res = _mm256_add_epi32(self.value, rhs.value);

            // cheeky hack relying on only the highest significant bit, which is the effective "sign" bit
            let saturated = _mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN)),
                _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX)),
                _mm256_castsi256_ps(res),
            );

            let overflow = _mm256_xor_si256(rhs.value, _mm256_cmpgt_epi32(self.value, res));

            Self::new(_mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(res),
                saturated,
                _mm256_castsi256_ps(overflow),
            )))
        }
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        unsafe {
            let res = _mm256_sub_epi32(self.value, rhs.value);

            let overflow = _mm256_xor_si256(
                _mm256_cmpgt_epi32(rhs.value, _mm256_setzero_si256()),
                _mm256_cmpgt_epi32(self.value, rhs.value),
            );

            let saturated = _mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN)),
                _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX)),
                _mm256_castsi256_ps(res),
            );

            Self::new(_mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(res),
                saturated,
                _mm256_castsi256_ps(overflow),
            )))
        }
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

impl SimdCastFrom<AVX2, f32x8<AVX2>> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: f32x8<AVX2>) -> Self {
        Self::new(unsafe { _mm256_cvttps_epi32(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, f32x8<AVX2>>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_cast(from.value())) // same width
    }
}

impl SimdCastFrom<AVX2, u32x8<AVX2>> for i32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: u32x8<AVX2>) -> Self {
        Self::new(from.value)
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, u32x8<AVX2>>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_cast(from.value())) // same width
    }
}

impl SimdCastFrom<AVX2, u64x8<AVX2>> for i32x8<AVX2> {
    #[inline]
    fn from_cast(from: u64x8<AVX2>) -> Self {
        brute_force_convert!(&from; u64 => i32)
    }

    #[inline]
    fn from_cast_mask(from: Mask<AVX2, u64x8<AVX2>>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}
