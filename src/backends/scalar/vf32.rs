use super::*;

decl!(f32x1: f32 => f32);
impl<S: Simd> Default for f32x1<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl SimdVectorBase<Scalar> for f32x1<Scalar> {
    type Element = f32;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::new(value)
    }

    #[inline(always)]
    unsafe fn undefined() -> Self {
        Self::new(0.0)
    }

    #[inline(always)]
    unsafe fn load_aligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(*src)
    }

    #[inline(always)]
    unsafe fn load_unaligned_unchecked(src: *const Self::Element) -> Self {
        Self::new(src.read_unaligned())
    }

    #[inline(always)]
    unsafe fn store_aligned_unchecked(self, dst: *mut Self::Element) {
        *dst = self.value;
    }

    #[inline(always)]
    unsafe fn store_unaligned_unchecked(self, dst: *mut Self::Element) {
        dst.write_unaligned(self.value)
    }

    decl_base_common!(#[target_feature()] f32x1: f32 => f32);
}

impl SimdBitwise<Scalar> for f32x1<Scalar> {
    const FULL_BITMASK: u16 = 1;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        self.into_bits().bitmask()
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(f32::from_bits(!0))
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() & rhs.value.to_bits()))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() | rhs.value.to_bits()))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() ^ rhs.value.to_bits()))
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() << count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() >> count.value))
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() << count))
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(f32::from_bits(self.value.to_bits() >> count))
    }
}

impl PartialEq<Self> for f32x1<Scalar> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl SimdMask<Scalar> for f32x1<Scalar> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        if self.value.to_bits() != 0 {
            t
        } else {
            f
        }
    }
}

impl SimdVector<Scalar> for f32x1<Scalar> {
    fn zero() -> Self {
        Self::splat(0.0)
    }

    fn one() -> Self {
        Self::splat(1.0)
    }

    fn indexed() -> Self {
        Self::splat(0.0)
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
        Self::new(self.value.min(other.value))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(self.value.max(other.value))
    }

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        self.value
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        self.value
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(f32::from_bits(bool_to_u32(self.value == other.value)))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(f32::from_bits(bool_to_u32(self.value < other.value)))
    }

    #[inline(always)]
    fn le(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(f32::from_bits(bool_to_u32(self.value <= other.value)))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(f32::from_bits(bool_to_u32(self.value > other.value)))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(f32::from_bits(bool_to_u32(self.value >= other.value)))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(Add::add(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(Sub::sub(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(Mul::mul(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::new(Div::div(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        Self::new(Rem::rem(self.value, rhs.value))
    }
}

impl SimdIntoBits<Scalar, Vu32> for f32x1<Scalar> {
    fn into_bits(self) -> Vu32 {
        u32x1::new(self.value.to_bits())
    }
}

impl SimdFromBits<Scalar, Vu32> for f32x1<Scalar> {
    fn from_bits(bits: Vu32) -> Self {
        Self::new(f32::from_bits(bits.value))
    }
}

impl_ops!(@UNARY f32x1 Scalar => Not::not, Neg::neg);
impl_ops!(@BINARY f32x1 Scalar => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS f32x1 Scalar => Shr::shr, Shl::shl);
