use super::*;

decl!(f64x8: f64 => (__m256d, __m256d));
impl<S: Simd> Default for f64x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm256_setzero_pd(), _mm256_setzero_pd()) })
    }
}

impl SimdVectorBase<AVX2> for f64x8<AVX2> {
    type Element = f64;

    const ALIGNMENT: usize = mem::align_of::<__m256d>(); // allow half-alignment

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe {
            let value = _mm256_set1_pd(value);
            (value, value)
        })
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(ptr: *const Self::Element) -> Self {
        Self::new((_mm256_load_pd(ptr), _mm256_load_pd(ptr.add(Self::NUM_ELEMENTS))))
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, ptr: *mut Self::Element) {
        _mm256_store_pd(ptr, self.value.0);
        _mm256_store_pd(ptr.add(Self::NUM_ELEMENTS), self.value.1);
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

impl SimdBitwise<AVX2> for f64x8<AVX2> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_andnot_pd(self.value.0, other.value.0),
                _mm256_andnot_pd(self.value.1, other.value.1),
            )
        })
    }

    const FULL_BITMASK: u16 = 0b11111111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe {
            let low = _mm256_movemask_pd(self.value.0) as u16;
            let high = _mm256_movemask_pd(self.value.1) as u16;

            low | (high << 4)
        }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(f64::from_bits(!0))
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new((
            _mm256_and_pd(self.value.0, rhs.value.0),
            _mm256_and_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_or_pd(self.value.0, rhs.value.0),
            _mm256_or_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new((
            _mm256_xor_pd(self.value.0, rhs.value.0),
            _mm256_xor_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: i32x8<AVX2>) -> Self {
        let low = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(count.value));
        let high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(count.value, 1));

        Self::new((
            _mm256_castsi256_pd(_mm256_srlv_epi64(_mm256_castpd_si256(self.value.0), low)),
            _mm256_castsi256_pd(_mm256_srlv_epi64(_mm256_castpd_si256(self.value.1), high)),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: i32x8<AVX2>) -> Self {
        let low = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(count.value));
        let high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(count.value, 1));

        Self::new((
            _mm256_castsi256_pd(_mm256_sllv_epi64(_mm256_castpd_si256(self.value.0), low)),
            _mm256_castsi256_pd(_mm256_sllv_epi64(_mm256_castpd_si256(self.value.1), high)),
        ))
    }
}

impl PartialEq<Self> for f64x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl SimdMask<AVX2> for f64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new((
            _mm256_blendv_pd(t.value.0, f.value.0, self.value.0),
            _mm256_blendv_pd(t.value.1, f.value.1, self.value.1),
        ))
    }
}

impl SimdVector<AVX2> for f64x8<AVX2> {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { Self::new((_mm256_setzero_pd(), _mm256_setzero_pd())) }
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(f64::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(f64::MAX)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_min_pd(self.value.0, other.value.0),
                _mm256_min_pd(self.value.1, other.value.1),
            )
        })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_max_pd(self.value.0, other.value.0),
                _mm256_max_pd(self.value.1, other.value.1),
            )
        })
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
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_EQ_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_EQ_OQ),
            )
        }))
    }

    #[inline(always)]
    fn ne(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_NEQ_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_NEQ_OQ),
            )
        }))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_LT_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_LT_OQ),
            )
        }))
    }

    #[inline(always)]
    fn le(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_LE_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_LE_OQ),
            )
        }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_GT_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_GT_OQ),
            )
        }))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<AVX2, Self> {
        Mask::new(Self::new(unsafe {
            (
                _mm256_cmp_pd(self.value.0, other.value.0, _CMP_GE_OQ),
                _mm256_cmp_pd(self.value.1, other.value.1, _CMP_GE_OQ),
            )
        }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new((
            _mm256_add_pd(self.value.0, rhs.value.0),
            _mm256_add_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new((
            _mm256_sub_pd(self.value.0, rhs.value.0),
            _mm256_sub_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new((
            _mm256_mul_pd(self.value.0, rhs.value.0),
            _mm256_mul_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::new((
            _mm256_div_pd(self.value.0, rhs.value.0),
            _mm256_div_pd(self.value.1, rhs.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        self - ((self / rhs).trunc() * rhs)
    }
}

impl SimdSignedVector<AVX2> for f64x8<AVX2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1.0)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(f64::MIN_POSITIVE)
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
        self & Self::splat(f64::from_bits(0x7fffffffffffffff))
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        // Xor sign bit using -0.0 as a shorthand for the sign bit
        self ^ Self::neg_zero()
    }
}

impl SimdFloatVector<AVX2> for f64x8<AVX2> {
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f64::EPSILON)
    }
    #[inline(always)]
    fn infinity() -> Self {
        Self::splat(f64::INFINITY)
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        Self::splat(f64::NEG_INFINITY)
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Self::splat(-0.0)
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::splat(f64::NAN)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_half_unaligned_unchecked(src: *const f16) -> Self {
        let mut dst = mem::MaybeUninit::uninit();
        for i in 0..Self::NUM_ELEMENTS {
            *(dst.as_mut_ptr() as *mut Self::Element).add(i) = (*src.add(i)).to_f64();
        }
        dst.assume_init()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_half_unaligned_unchecked(&self, dst: *mut f16) {
        for i in 0..Self::NUM_ELEMENTS {
            *dst.add(i) = f16::from_f64(self.extract_unchecked(i));
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
        Self::new(unsafe {
            (
                _mm256_fmadd_pd(self.value.0, m.value.0, a.value.0),
                _mm256_fmadd_pd(self.value.1, m.value.1, a.value.1),
            )
        })
    }

    #[inline(always)]
    fn mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_fmsub_pd(self.value.0, m.value.0, s.value.0),
                _mm256_fmsub_pd(self.value.1, m.value.1, s.value.1),
            )
        })
    }

    #[inline(always)]
    fn neg_mul_add(self, m: Self, a: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_fnmadd_pd(self.value.0, m.value.0, a.value.0),
                _mm256_fnmadd_pd(self.value.1, m.value.1, a.value.1),
            )
        })
    }

    #[inline(always)]
    fn neg_mul_sub(self, m: Self, s: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_fnmsub_pd(self.value.0, m.value.0, s.value.0),
                _mm256_fnmsub_pd(self.value.1, m.value.1, s.value.1),
            )
        })
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::new(unsafe { (_mm256_floor_pd(self.value.0), _mm256_floor_pd(self.value.1)) })
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Self::new(unsafe { (_mm256_ceil_pd(self.value.0), _mm256_ceil_pd(self.value.1)) })
    }

    #[inline(always)]
    fn round(self) -> Self {
        Self::new(unsafe {
            (
                _mm256_round_pd(self.value.0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                _mm256_round_pd(self.value.1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
            )
        })
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Self::new(unsafe {
            (
                _mm256_round_pd(self.value.0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC),
                _mm256_round_pd(self.value.1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC),
            )
        })
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::new(unsafe { (_mm256_sqrt_pd(self.value.0), _mm256_sqrt_pd(self.value.1)) })
    }
}

impl_ops!(@UNARY f64x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY f64x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS f64x8 AVX2 => Shr::shr, Shl::shl);
