use super::*;

decl!(i64x8: i64 => (__m256i, __m256i));
impl<S: Simd> Default for i64x8<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm256_setzero_si256(), _mm256_setzero_si256()) })
    }
}

impl SimdVectorBase<AVX2> for i64x8<AVX2> {
    type Element = i64;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(unsafe { (_mm256_set1_epi64x(value), _mm256_set1_epi64x(value)) })
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

impl SimdBitwise<AVX2> for i64x8<AVX2> {
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

impl PartialEq<Self> for i64x8<AVX2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<AVX2>>::ne(*self, *other).any()
    }
}

impl Eq for i64x8<AVX2> {}

impl SimdMask<AVX2> for i64x8<AVX2> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        Self::new((
            _mm256_blendv_epi8(f.value.0, t.value.0, self.value.0),
            _mm256_blendv_epi8(f.value.1, t.value.1, self.value.1),
        ))
    }

    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        _mm256_movemask_epi8(_mm256_and_si256(self.value.0, self.value.1)) == -1
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

impl SimdVector<AVX2> for i64x8<AVX2> {
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

impl SimdIntoBits<AVX2, Vu64> for i64x8<AVX2> {
    #[inline(always)]
    fn into_bits(self) -> Vu64 {
        u64x8::new(self.value)
    }
}

impl SimdFromBits<AVX2, Vu64> for i64x8<AVX2> {
    #[inline(always)]
    fn from_bits(bits: Vu64) -> Self {
        Self::new(bits.value)
    }
}

impl SimdIntVector<AVX2> for i64x8<AVX2> {
    #[inline(always)]
    fn saturating_add(self, rhs: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_adds_epi64x(self.value.0, rhs.value.0),
                _mm256_adds_epi64x(self.value.1, rhs.value.1),
            )
        })
    }

    #[inline(always)]
    fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(unsafe {
            (
                _mm256_subs_epi64x(self.value.0, rhs.value.0),
                _mm256_subs_epi64x(self.value.1, rhs.value.1),
            )
        })
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

impl SimdSignedVector<AVX2> for i64x8<AVX2> {
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
        Self::new(unsafe { (_mm256_abs_epi64x(self.value.0), _mm256_abs_epi64x(self.value.1)) })
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        self ^ Self::neg_one() + Self::one()
    }
}

impl_ops!(@UNARY i64x8 AVX2 => Not::not, Neg::neg);
impl_ops!(@BINARY i64x8 AVX2 => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@BINARY i64x8 AVX2 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);
impl_ops!(@SHIFTS i64x8 AVX2 => Shr::shr, Shl::shl);

impl SimdFromCast<AVX2, Vf32> for i64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf32) -> Self {
        Self::new(unsafe {
            let low = _mm256_castps256_ps128(from.value);
            let high = _mm256_extractf128_ps(from.value, 1);

            (_mm256_cvtps_epi64(low), _mm256_cvtps_epi64(high))
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf32>) -> Mask<AVX2, Self> {
        Vi64::from_cast_mask(Mask::new(from.value().into_bits()))
    }
}

impl SimdFromCast<AVX2, Vi32> for i64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vi32) -> Self {
        Self::new(unsafe {
            (
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(from.value)),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(from.value, 1)),
            )
        })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vi32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vu32> for i64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vu32) -> Self {
        // zero extend
        Self::from_bits(from.cast_to::<Vu64>())
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vu32>) -> Mask<AVX2, Self> {
        Self::from_cast(from.value()).ne(Self::zero())
    }
}

impl SimdFromCast<AVX2, Vu64> for i64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vu64) -> Self {
        Self::new(from.value)
    }

    fn from_cast_mask(from: Mask<AVX2, Vu64>) -> Mask<AVX2, Self> {
        Mask::new(Self::new(from.value().value))
    }
}

impl SimdFromCast<AVX2, Vf64> for i64x8<AVX2> {
    #[inline(always)]
    fn from_cast(from: Vf64) -> Self {
        Self::new(unsafe { (_mm256_cvtpd_epi64x(from.value.0), _mm256_cvtpd_epi64x(from.value.1)) })
    }

    #[inline(always)]
    fn from_cast_mask(from: Mask<AVX2, Vf64>) -> Mask<AVX2, Self> {
        // same width, cast through bits
        Mask::new(Self::from_bits(from.value().into_bits()))
    }
}
