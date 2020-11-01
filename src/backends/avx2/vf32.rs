use super::*;

decl!(f32x8: f32 => __m256);
impl<S: Simd> Default for f32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_ps() })
    }
}

impl SimdVectorBase<AVX2> for f32x8<AVX2> {
    type Element = f32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_ps(value) })
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_load_ps(ptr))
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new(_mm256_loadu_ps(ptr))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_store_ps(ptr, self.value)
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_storeu_ps(ptr, self.value)
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

impl SimdBitwise<AVX2> for f32x8<AVX2> {
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_ps(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b11111111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_ps(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(f32::from_bits(!0))
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm256_and_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm256_or_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm256_xor_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: u32x8<AVX2>) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_srlv_epi32(
            _mm256_castps_si256(self.value),
            count.value,
        )))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: u32x8<AVX2>) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_sllv_epi32(
            _mm256_castps_si256(self.value),
            count.value,
        )))
    }
}

impl PartialEq<Self> for f32x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl SimdMask<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_ps(t.value, f.value, self.value))
    }
}

impl SimdVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm256_setzero_ps() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(f32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(f32::MAX)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_min_ps(self.value, other.value) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_max_ps(self.value, other.value) })
    }

    #[inline]
    fn min_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_EQ_OQ) }))
    }

    #[inline(always)]
    fn ne(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            _mm256_cmp_ps(self.value, other.value, _CMP_NEQ_OQ)
        }))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_LT_OQ) }))
    }

    #[inline(always)]
    fn le(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_LE_OQ) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_GT_OQ) }))
    }

    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_cmp_ps(self.value, other.value, _CMP_GE_OQ) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm256_add_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm256_sub_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(_mm256_mul_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::new(_mm256_div_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        self - ((self / rhs).trunc() * rhs)
    }
}

impl SimdIntoBits<AVX2, u32x8<AVX2>> for f32x8<AVX2> {
    #[inline(always)]
    fn into_bits(self) -> u32x8<AVX2> {
        u32x8::new(unsafe { _mm256_castps_si256(self.value) })
    }
}

impl SimdFromBits<AVX2, u32x8<AVX2>> for f32x8<AVX2> {
    #[inline(always)]
    fn from_bits(bits: u32x8<AVX2>) -> Self {
        Self::new(unsafe { _mm256_castsi256_ps(bits.value) })
    }
}

impl SimdSignedVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1.0)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(f32::MIN_POSITIVE)
    }

    #[inline(always)]
    fn signum(self) -> Self {
        // copy sign bit from `self` onto 1.0
        Self::one() | (self & Self::neg_zero())
    }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        // clear sign bit, then copy sign bit from `sign`
        self.abs() | (sign & Self::neg_zero())
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // clear sign bit
        self & Self::splat(f32::from_bits(0x7fffffff))
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        // Xor sign bit using -0.0 as a shorthand for the sign bit
        self ^ Self::neg_zero()
    }
}

impl SimdFloatVector<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f32::EPSILON)
    }
    #[inline(always)]
    fn infinity() -> Self {
        Self::splat(f32::INFINITY)
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        Self::splat(f32::NEG_INFINITY)
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Self::splat(-0.0)
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::splat(f32::NAN)
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    unsafe fn load_half_unaligned_unchecked(src: *const f16) -> Self {
        Self::new(_mm256_cvtph_ps(_mm_loadu_si128(src as *const _)))
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_half_unaligned_unchecked(src: *const f16) -> Self {
        let mut dst = mem::MaybeUninit::uninit();
        for i in 0..Self::NUM_ELEMENTS {
            *(dst.as_mut_ptr() as *mut Self::Element).add(i) = (*src.add(i)).to_f32();
        }
        dst.assume_init()
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    unsafe fn store_half_unaligned_unchecked(&self, dst: *mut f16) {
        let mut dst_vec = _mm256_cvtps_ph(self.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        ptr::copy_nonoverlapping(&dst_vec as *const __m128i as *const f16, dst, Self::NUM_ELEMENTS);
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_half_unaligned_unchecked(&self, dst: *mut f16) {
        for i in 0..Self::NUM_ELEMENTS {
            *dst.add(i) = f16::from_f32(self.extract_unchecked(i));
        }
    }

    fn sum(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|sum, x| sum + x) }
    }

    fn product(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|prod, x| x * prod) }
    }

    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe { _mm256_fmadd_ps(self.value, m.value, a.value) })
    }

    #[inline(always)]
    fn mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe { _mm256_fmsub_ps(self.value, m.value, s.value) })
    }

    #[inline(always)]
    fn neg_mul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe { _mm256_fnmadd_ps(self.value, m.value, a.value) })
    }

    #[inline(always)]
    fn neg_mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe { _mm256_fnmsub_ps(self.value, m.value, s.value) })
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::new(unsafe { _mm256_floor_ps(self.value) })
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Self::new(unsafe { _mm256_ceil_ps(self.value) })
    }

    #[inline(always)]
    fn round(self) -> Self {
        Self::new(unsafe { _mm256_round_ps(self.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) })
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Self::new(unsafe { _mm256_round_ps(self.value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) })
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::new(unsafe { _mm256_sqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::new(unsafe { _mm256_rsqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt_precise(self) -> Self {
        let y = self.rsqrt();
        let nx2 = self * Self::splat(-0.5);
        let threehalfs = Self::splat(1.5);

        y * (y * y).mul_add(nx2, threehalfs)
    }

    #[inline(always)]
    fn recepr(self) -> Self {
        Self::new(unsafe { _mm256_rcp_ps(self.value) })
    }
}

impl_ops!(@UNARY f32x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY f32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS f32x8 AVX2 => Shr::shr, Shl::shl);

impl SimdCastFrom<AVX2, i32x8<AVX2>> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: i32x8<AVX2>) -> Self {
        Self::new(unsafe { _mm256_cvtepi32_ps(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, i32x8<AVX2>>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdCastFrom<AVX2, u64x8<AVX2>> for f32x8<AVX2> {
    #[inline]
    fn from_cast(from: u64x8<AVX2>) -> Self {
        brute_force_convert!(&from; u64 => f32)
    }

    #[inline]
    fn from_cast_mask(from: Mask<AVX2, u64x8<AVX2>>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}
