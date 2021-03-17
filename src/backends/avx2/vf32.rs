use super::*;

decl!(f32x8: f32 => __m256);
impl<S: Simd> Default for f32x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm256_setzero_ps() })
    }
}

#[rustfmt::skip]
macro_rules! log_reduce_ps_avx2 {
    ($value:expr; $op:ident $last:ident) => {unsafe {
        let ymm0 = $value;
        // split the 256-bit vector into two 128-bit vectors
        let xmm0 = _mm256_castps256_ps128(ymm0);
        let xmm1 = _mm256_extractf128_ps(ymm0, 1);
        let xmm0 = $op(xmm0, xmm1); // then run one regular op on the split vectors

        // shuffle the upper 2 floats to the front
        let xmm1 = _mm_castpd_ps(_mm_permute_pd(_mm_castps_pd(xmm0), 1));
        let xmm0 = $op(xmm0, xmm1);
        let xmm1 = _mm_movehdup_ps(xmm0); // interleave

        _mm_cvtss_f32($last(xmm0, xmm1))
    }};
}

impl SimdVectorBase<AVX2> for f32x8<AVX2> {
    type Element = f32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { _mm256_set1_ps(value) })
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new(_mm256_undefined_ps())
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

    decl_base_common!(#[target_feature(enable = "avx2,fma")] f32x8: f32 => __m256);

    #[inline(always)]
    unsafe fn gather_unchecked(base_ptr: *const Self::Element, indices: Vi32) -> Self {
        Self::new(_mm256_i32gather_ps(
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
        Self::new(_mm256_mask_i32gather_ps(
            default.value,
            base_ptr as _,
            indices.value,
            mask.value().value,
            mem::size_of::<Self::Element>() as _,
        ))
    }
}

impl SimdBitwise<AVX2> for f32x8<AVX2> {
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm256_andnot_ps(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b11111111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm256_movemask_ps(self.value) as u16 }
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
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_srlv_epi32(
            _mm256_castps_si256(self.value),
            count.value,
        )))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_sllv_epi32(
            _mm256_castps_si256(self.value),
            count.value,
        )))
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_sll_epi32(
            _mm256_castps_si256(self.value),
            _mm_cvtsi32_si128(count as i32),
        )))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(_mm256_castsi256_ps(_mm256_srl_epi32(
            _mm256_castps_si256(self.value),
            _mm_cvtsi32_si128(count as i32),
        )))
    }
}

impl PartialEq<Self> for f32x8<AVX2> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // shouldn't use XOR/ptest trick here because NaNs
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl SimdMask<AVX2> for f32x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new(_mm256_blendv_ps(f.value, t.value, self.value))
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
    fn indexed() -> Self {
        unsafe { Self::new(_mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)) }
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

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        log_reduce_ps_avx2!(self.value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        log_reduce_ps_avx2!(self.value; _mm_max_ps _mm_max_ss)
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
        // https://stackoverflow.com/a/26342944/2083075 + Bernard's comment
        (self / rhs).trunc().nmul_add(rhs, self)
    }
}

impl SimdIntoBits<AVX2, Vu32> for f32x8<AVX2> {
    #[inline(always)]
    fn into_bits(self) -> Vu32 {
        u32x8::new(unsafe { _mm256_castps_si256(self.value) })
    }
}

impl SimdFromBits<AVX2, Vu32> for f32x8<AVX2> {
    #[inline(always)]
    fn from_bits(bits: Vu32) -> Self {
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
    fn conditional_neg(self, mask: Mask<AVX2, impl SimdCastTo<AVX2, Self>>) -> Self {
        self ^ (SimdCastTo::cast_mask(mask).value() & Self::neg_zero())
    }

    #[inline(always)]
    fn select_negative(self, neg: Self, pos: Self) -> Self {
        unsafe { self._mm_blendv(neg, pos) }
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        // Xor sign bit using -0.0 as a shorthand for the sign bit
        self ^ Self::neg_zero()
    }
}

impl SimdFloatVector<AVX2> for f32x8<AVX2> {
    type Vu = Vu32;
    type Vi = Vi32;

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
    unsafe fn load_f16_unaligned_unchecked(src: *const f16) -> Self {
        Self::new(_mm256_cvtph_ps(_mm_loadu_si128(src as *const _)))
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_f16_unaligned_unchecked(src: *const f16) -> Self {
        let mut dst = mem::MaybeUninit::uninit();
        for i in 0..Self::NUM_ELEMENTS {
            *(dst.as_mut_ptr() as *mut Self::Element).add(i) = (*src.add(i)).to_f32();
        }
        dst.assume_init()
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    unsafe fn store_f16_unaligned_unchecked(&self, dst: *mut f16) {
        let mut dst_vec = _mm256_cvtps_ph(self.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        ptr::copy_nonoverlapping(&dst_vec as *const __m128i as *const f16, dst, Self::NUM_ELEMENTS);
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_f16_unaligned_unchecked(&self, dst: *mut f16) {
        for i in 0..Self::NUM_ELEMENTS {
            *dst.add(i) = f16::from_f32(self.extract_unchecked(i));
        }
    }

    #[inline(always)]
    unsafe fn to_int_fast(self) -> Self::Vi {
        self.cast()
    }

    #[inline(always)]
    unsafe fn to_uint_fast(self) -> Self::Vu {
        self.cast()
    }

    #[inline(always)]
    fn sum(self) -> Self::Element {
        log_reduce_ps_avx2!(self.value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn product(self) -> Self::Element {
        log_reduce_ps_avx2!(self.value; _mm_mul_ps _mm_mul_ss)
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
    fn nmul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe { _mm256_fnmadd_ps(self.value, m.value, a.value) })
    }

    #[inline(always)]
    fn nmul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe { _mm256_fnmsub_ps(self.value, m.value, s.value) })
    }

    #[inline(always)]
    fn mul_adde(self, m: Self, a: Self) -> Self {
        self.mul_add(m, a)
    }

    #[inline(always)]
    fn mul_sube(self, m: Self, s: Self) -> Self {
        self.mul_sub(m, s)
    }

    #[inline(always)]
    fn nmul_adde(self, m: Self, a: Self) -> Self {
        self.nmul_add(m, a)
    }

    #[inline(always)]
    fn nmul_sube(self, m: Self, s: Self) -> Self {
        self.nmul_sub(m, s)
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
    fn rcp(self) -> Self {
        Self::new(unsafe { _mm256_rcp_ps(self.value) })
    }

    #[inline(always)]
    fn rcp_precise(self) -> Self {
        let rcp = self.rcp();

        // one iteration of Newton's method
        rcp * self.nmul_add(rcp, Self::splat(2.0))
    }

    #[inline(always)]
    fn is_subnormal(self) -> Mask<AVX2, Self> {
        let m: Self = Self::splat(f32::from_bits(0xFF000000));
        let u: Self = self << 1;
        let zero = Self::zero();

        (u & m).eq(zero) & m.and_not(u).ne(zero)
    }

    #[inline(always)]
    fn is_zero_or_subnormal(self) -> Mask<AVX2, Self> {
        (self & Self::splat(f32::from_bits(0x7F800000))).eq(Self::zero())
    }
}

impl_ops!(@UNARY f32x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY f32x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS f32x8 AVX2 => Shr::shr, Shl::shl);

impl SimdFromCast<AVX2, Vi32> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi32) -> Self {
        Self::new(unsafe { _mm256_cvtepi32_ps(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe { _mm256_castsi256_ps(from.value().value) }))
    }
}

impl SimdFromCast<AVX2, Vu32> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vu32) -> Self {
        Self::new(unsafe { _mm256_cvtepu32_psx(from.value) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu32>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_bits(from.value()))
    }
}

impl SimdFromCast<AVX2, Vu64> for f32x8<AVX2> {
    #[inline]
    fn from_cast(from: Vu64) -> Self {
        decl_brute_force_convert!(#[target_feature(enable = "avx2")] u64 => f32);
        unsafe { do_convert(from) }
    }

    #[inline]
    fn from_cast_mask(from: Mask<AVX2, Vu64>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vf64> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf64) -> Self {
        Self::new(unsafe {
            _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm256_cvtpd_ps(from.value.0)),
                _mm256_cvtpd_ps(from.value.1),
                1,
            )
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // Cast u64 -> u32 with truncate, should be faster
        Mask::new(Vf32::from_bits(from.value().into_bits().cast()))
    }
}

impl SimdFromCast<AVX2, Vi64> for f32x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi64) -> Self {
        from.cast_to::<Vf64>().cast()
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi64>) -> Mask<AVX2, Self> {
        Mask::new(Self::from_bits(from.value().into_bits().cast()))
    }
}
