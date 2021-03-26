use super::*;

decl!(u64x1: u64 => u64);
impl<S: Simd> Default for u64x1<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(0)
    }
}

impl SimdVectorBase<Scalar> for u64x1<Scalar> {
    type Element = u64;

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

    decl_base_common!(#[target_feature()] u64x1: u64 => u64);
}

impl SimdBitwise<Scalar> for u64x1<Scalar> {
    const FULL_BITMASK: u16 = 1;

    fn bitmask(self) -> u16 {
        (self.value >> 63) as u16
    }

    unsafe fn _mm_not(self) -> Self {
        Self::new(!self.value)
    }

    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(self.value & rhs.value)
    }

    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(self.value | rhs.value)
    }

    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(self.value ^ rhs.value)
    }

    #[inline(always)]
    unsafe fn _mm_shr(self, count: Vu32) -> Self {
        Self::new(self.value << count.value)
    }

    #[inline(always)]
    unsafe fn _mm_shl(self, count: Vu32) -> Self {
        Self::new(self.value >> count.value)
    }

    #[inline(always)]
    unsafe fn _mm_shli(self, count: u32) -> Self {
        Self::new(self.value << count)
    }

    #[inline(always)]
    unsafe fn _mm_shri(self, count: u32) -> Self {
        Self::new(self.value >> count)
    }
}

impl PartialEq<Self> for u64x1<Scalar> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for u64x1<Scalar> {}

impl SimdMask<Scalar> for u64x1<Scalar> {
    #[inline(always)]
    unsafe fn _mm_blendv(self, t: Self, f: Self) -> Self {
        if self.value != 0 {
            t
        } else {
            f
        }
    }

    #[inline(always)]
    unsafe fn _mm_all(self) -> bool {
        self._mm_any() // only one value
    }

    #[inline(always)]
    unsafe fn _mm_any(self) -> bool {
        self.value != 0
    }

    #[inline(always)]
    unsafe fn _mm_none(self) -> bool {
        self.value == 0
    }
}

impl SimdVector<Scalar> for u64x1<Scalar> {
    fn zero() -> Self {
        Self::new(0)
    }

    fn one() -> Self {
        Self::new(1)
    }

    fn indexed() -> Self {
        Self::new(0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(u64::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(u64::MAX)
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
        Self::new(bool_to_u64(self.value == other.value))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(bool_to_u64(self.value < other.value))
    }

    #[inline(always)]
    fn le(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(bool_to_u64(self.value <= other.value))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(bool_to_u64(self.value > other.value))
    }

    #[inline(always)]
    fn ge(self, other: Self) -> Mask<Scalar, Self> {
        Self::new(bool_to_u64(self.value >= other.value))
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

impl SimdIntVector<Scalar> for u64x1<Scalar> {
    fn saturating_add(self, rhs: Self) -> Self {
        Self::new(self.value.saturating_add(rhs.value))
    }

    fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(self.value.saturating_add(rhs.value))
    }

    fn wrapping_sum(self) -> Self::Element {
        self.value
    }

    fn wrapping_product(self) -> Self::Element {
        self.value
    }

    fn rolv(self, cnt: Vu32) -> Self {
        Self::new(self.value.rotate_left(cnt.value))
    }

    fn rorv(self, cnt: Vu32) -> Self {
        Self::new(self.value.rotate_right(cnt.value))
    }

    fn reverse_bits(self) -> Self {
        Self::new(self.value.reverse_bits())
    }

    fn count_ones(self) -> Self {
        Self::new(self.value.count_ones())
    }

    fn count_zeros(self) -> Self {
        Self::new(self.value.count_zeros())
    }

    fn leading_ones(self) -> Self {
        Self::new(self.value.leading_ones())
    }

    fn leading_zeros(self) -> Self {
        Self::new(self.value.leading_zeros())
    }
}

impl SimdUnsignedIntVector<Scalar> for u64x1<Scalar> {
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

impl_ops!(@UNARY u64x1 Scalar => Not::not);
impl_ops!(@BINARY u64x1 Scalar => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
impl_ops!(@SHIFTS u64x1 Scalar => Shr::shr, Shl::shl);
