//! Low-level SIMD Register interface

pub mod dp;

use generic_array::{ArrayLength, GenericArray, IntoArrayLength, typenum};

pub type DoublePump<V> = <V as dp::DoublePumpVector>::DoublePump;

#[inline(always)]
pub(crate) const fn reg<R: Register, const N: usize>(values: [R::Element; N]) -> R::Storage
where
    typenum::Const<N>: IntoArrayLength<ArrayLength = R::Lanes>,
{
    const {
        assert!(
            core::mem::size_of::<R::Storage>() == core::mem::size_of::<[R::Element; N]>(),
            "Size mismatch between register and array of elements"
        );
    }

    // SAFETY: The way const_transmute works
    // handles alignment automatically, so this
    // is valid so long as the size is the same.
    unsafe { generic_array::const_transmute(values) }
}

#[inline(always)]
pub(crate) const fn reg_splat<R: Register>(value: R::Element) -> R::Storage {
    let mut dst = R::EMPTY;

    // SAFETY: This is iterating over contiguous memory, just using a pointer
    unsafe {
        let dst = &mut dst as *mut R::Storage as *mut R::Element;

        let mut i = 0;
        while i < <R::Lanes as typenum::Unsigned>::USIZE {
            dst.add(i).write(value);

            i += 1;
        }
    }

    dst
}

#[inline(always)]
pub(crate) const fn empty_reg<R>() -> R::Storage
where
    R: Register,
{
    // SAFETY: Initialized memory but unset
    unsafe { core::mem::zeroed() }
}

pub trait Interoperable<A: MaskRegister<Lanes = Self::Lanes>, B: MaskRegister<Lanes = Self::Lanes>>: MaskRegister
// bits
+ BitsRegister<Self>
+ BitsRegister<A>
+ BitsRegister<B>
// casts
+ CastRegister<Self>
+ CastRegister<A>
+ CastRegister<B>
// masks
+ CastMaskRegister<Self>
+ CastMaskRegister<A>
+ CastMaskRegister<B>
where
A: BitsRegister<Self> + CastRegister<Self> + CastMaskRegister<Self>,
B: BitsRegister<Self> + CastRegister<Self> + CastMaskRegister<Self>
{}

impl<R, A, B> Interoperable<A, B> for R
where
    R: MaskRegister
        + BitsRegister<Self>
        + BitsRegister<A>
        + BitsRegister<B>
        + CastRegister<Self>
        + CastRegister<A>
        + CastRegister<B>
        + CastMaskRegister<Self>
        + CastMaskRegister<A>
        + CastMaskRegister<B>,
    A: MaskRegister<Lanes = R::Lanes>,
    B: MaskRegister<Lanes = R::Lanes>,
    A: BitsRegister<R> + CastRegister<R> + CastMaskRegister<R>,
    B: BitsRegister<R> + CastRegister<R> + CastMaskRegister<R>,
{
}

/// A trait for array length types representing the number of lanes in a SIMD register.
pub trait Lanes: ArrayLength + core::ops::Shl<typenum::B1> {}
impl<T> Lanes for T where T: ArrayLength + core::ops::Shl<typenum::B1> {}

pub trait Register: Sized + 'static {
    type Lanes: Lanes;
    type Element: Copy + Default + PartialEq + PartialOrd + core::fmt::Debug;
    type Storage: Copy + core::fmt::Debug;

    // Note: These don't require :Register because it would introduce recursive type bounds.
    type HalfRegister;
    type DoubleRegister;

    const EMPTY: Self::Storage;

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage;
    fn splat(value: Self::Element) -> Self::Storage;

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Self::Storage
    where
        Self::HalfRegister: Register,
    {
        let _ = (lo, hi);
        unimplemented!("Register::join() not implemented for this register type.")
    }

    #[inline(always)]
    fn split(
        value: Self::Storage,
    ) -> (
        <Self::HalfRegister as Register>::Storage,
        <Self::HalfRegister as Register>::Storage,
    )
    where
        Self::HalfRegister: Register,
    {
        let _ = value;
        unimplemented!("Register::split() not implemented for this register type.")
    }

    #[inline(always)]
    fn concat(lo: Self::Storage, hi: Self::Storage) -> <Self::DoubleRegister as Register>::Storage
    where
        Self::DoubleRegister: Register<HalfRegister = Self>,
    {
        <Self::DoubleRegister as Register>::join(lo, hi)
    }

    #[inline(always)]
    fn as_array(storage: &Self::Storage) -> &GenericArray<Self::Element, Self::Lanes> {
        unsafe { &*(storage as *const Self::Storage as *const GenericArray<Self::Element, Self::Lanes>) }
    }

    #[inline(always)]
    fn as_array_mut(storage: &mut Self::Storage) -> &mut GenericArray<Self::Element, Self::Lanes> {
        unsafe { &mut *(storage as *mut Self::Storage as *mut GenericArray<Self::Element, Self::Lanes>) }
    }

    #[inline(always)]
    fn iter(storage: &Self::Storage) -> core::slice::Iter<'_, Self::Element> {
        Self::as_array(storage).iter()
    }

    #[inline(always)]
    fn iter_mut(storage: &mut Self::Storage) -> core::slice::IterMut<'_, Self::Element> {
        Self::as_array_mut(storage).iter_mut()
    }

    #[inline(always)]
    fn extract<const I: usize>(value: Self::Storage) -> Self::Element {
        Self::as_array(&value)[I]
    }

    #[inline(always)]
    fn insert<const I: usize>(mut value: Self::Storage, element: Self::Element) -> Self::Storage {
        Self::as_array_mut(&mut value)[I] = element;
        value
    }

    #[inline(always)]
    fn map<F>(mut value: Self::Storage, mut f: F) -> Self::Storage
    where
        F: FnMut(Self::Element) -> Self::Element,
    {
        for v in Self::as_array_mut(&mut value) {
            *v = f(*v);
        }

        value
    }

    #[inline(always)]
    fn zip<F>(mut lhs: Self::Storage, rhs: Self::Storage, f: F) -> Self::Storage
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for (a, b) in Self::as_array_mut(&mut lhs).iter_mut().zip(Self::as_array(&rhs)) {
            *a = f(*a, *b);
        }

        lhs
    }

    fn xor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn and(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    /// !lhs & rhs
    #[inline(always)]
    fn andnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::and(Self::not(lhs), rhs)
    }

    fn or(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn not(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::or(Self::and(mask, lhs), Self::andnot(mask, rhs))
    }

    /// Indicates if blendv only cares about the most significant bit (MSB) of the mask.
    const HAS_MSB_BLENDV: bool;

    fn shr(value: Self::Storage, shift: u32) -> Self::Storage;
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage;

    fn shrv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage;
    fn shlv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage;
}

pub trait ShuffleRegister: Register {
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
}

pub trait PermuteRegister: Register {
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage;
}

pub trait BlendRegister: Register {
    fn blend<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
}

pub trait SwizzleRegister: Register {
    fn permutev(value: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage;

    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     a: Self::Storage,
    //     b: Self::Storage,
    // ) -> Self::Storage;

    #[inline(always)]
    fn swizzle(a: Self::Storage, b: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let idxs: GenericArray<u32, Self::Lanes> = idxs.into();

        use typenum::Unsigned;

        let mut a_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();
        let mut b_idxs: GenericArray<u32, Self::Lanes> = GenericArray::default();
        let mut blend: GenericArray<u32, Self::Lanes> = GenericArray::default();

        for (i, &idx) in idxs.iter().enumerate() {
            if idx < Self::Lanes::USIZE as u32 {
                a_idxs[i] = idx;
                b_idxs[i] = i as u32;
                blend[i] = 0;
            } else {
                a_idxs[i] = i as u32;
                b_idxs[i] = idx - Self::Lanes::USIZE as u32;
                blend[i] = !0;
            }
        }

        let tmp_a = Self::permutev(a, a_idxs);
        let tmp_b = Self::permutev(b, b_idxs);

        let blend = unsafe { generic_array::const_transmute(blend) };

        Self::blendv(blend, tmp_a, tmp_b)
    }
}

pub trait MaskElement: Sized + 'static {
    fn to_bool(self) -> bool;
    fn from_bool(value: bool) -> Self;
}

macro_rules! impl_mask_element {
    ($($t:ty),+) => {$(
        impl MaskElement for $t {
            #[inline(always)]
            fn to_bool(self) -> bool {
                self != 0
            }

            #[inline(always)]
            fn from_bool(value: bool) -> Self {
                if value { 1 } else { 0 }
            }
        }
    )+};
}

impl_mask_element!(u8, u16, u32, u64, i8, i16, i32, i64);

impl MaskElement for f32 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        f32::from_bits(if value { !0 } else { 0 })
    }
}

impl MaskElement for f64 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        f64::from_bits(if value { !0 } else { 0 })
    }
}

pub trait MaskRegister: Register<Element: MaskElement> {
    const TRUTHY: Self::Storage;
    const FALSY: Self::Storage;

    #[inline(always)]
    fn boolean(value: bool) -> Self::Storage {
        if value { Self::TRUTHY } else { Self::FALSY }
    }

    #[inline(always)]
    fn new_mask(value: impl Into<GenericArray<bool, Self::Lanes>>) -> Self::Storage {
        // NOTE: This is a fallback implementation.

        let value = value.into();
        let mut result = Self::EMPTY;

        {
            let result = Self::as_array_mut(&mut result);
            for (i, v) in value.into_iter().enumerate() {
                result[i] = Self::Element::from_bool(v);
            }
        }
        result
    }

    #[inline(always)]
    fn debug_iter_bool(value: &Self::Storage) -> impl Iterator<Item = bool> {
        Self::as_array(value).iter().map(|v| v.to_bool())
    }

    fn all(value: Self::Storage) -> bool;
    fn any(value: Self::Storage) -> bool;

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        !Self::any(value)
    }
}

/// A trait for registers that can be cast to/from other registers,
/// including of varying element types.
pub trait CastRegister<FROM: Register>: Register {
    /// Cast a register from another register type.
    fn cast_from(value: FROM::Storage) -> Self::Storage;

    /// Cast a register to another register type, potentially faster
    /// when the values are within a certain range, otherwise
    /// unspecified values are returned. This method is safe in the
    /// Rust sense, but may not be safe in the sense that values
    /// may not be preserved across the cast.
    #[inline(always)]
    fn fast_cast_from(value: FROM::Storage) -> Self::Storage {
        Self::cast_from(value)
    }
}

/// A trait for registers that can be reinterpreted as other registers,
/// though this is not a safe operation. This is only available for registers
/// of the same size in bytes. This is enforced simply by the fact that
/// it will only be implemented for registers of the same size.
pub trait BitsRegister<FROM: Register>: Register {
    fn from_bits(value: FROM::Storage) -> Self::Storage;
}

/// A trait for registers that can be reinterpreted as other registers, as masks,
/// such that the masks retain 0 or !0 values for the appropriate lanes.
pub trait CastMaskRegister<FROM: MaskRegister>: MaskRegister {
    fn mask_from(value: FROM::Storage) -> Self::Storage;
}

pub trait ShiftRegister: Register {
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage;
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage;
}

pub trait PartialOrdRegister: MaskRegister {
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        let gt = Self::gt(lhs, rhs);
        let eq = Self::eq(lhs, rhs);

        Self::or(gt, eq)
    }

    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::gt(rhs, lhs)
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::ge(rhs, lhs)
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::not(Self::eq(lhs, rhs))
    }
}

// TODO: Replace `: Register` with `: PartialOrdRegister` when
// it's implemented for all registers.
pub trait NumericRegister: PartialOrdRegister {
    const ZERO: Self::Storage;
    const ONE: Self::Storage;
    const TWO: Self::Storage;

    const MIN: Self::Storage;
    const MAX: Self::Storage;

    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    fn min_element(value: Self::Storage) -> Self::Element;
    fn max_element(value: Self::Storage) -> Self::Element;

    fn sum_elements(value: Self::Storage) -> Self::Element;
    fn prod_elements(value: Self::Storage) -> Self::Element;

    /// Effectively the number of lanes in the register, splatted across the lanes.
    fn offset() -> Self::Storage;
    /// 0, 1, 2, 3, 4, ... etc.
    fn indexed() -> Self::Storage;
}

pub trait SignedRegister: NumericRegister {
    fn neg(value: Self::Storage) -> Self::Storage;
    fn abs(value: Self::Storage) -> Self::Storage;

    fn signum(value: Self::Storage) -> Self::Storage;

    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    const NEG_ONE: Self::Storage;

    #[inline(always)]
    fn is_negative(value: Self::Storage) -> Self::Storage {
        Self::lt(value, Self::ZERO)
    }

    #[inline(always)]
    fn is_positive(value: Self::Storage) -> Self::Storage {
        Self::ge(value, Self::ZERO)
    }

    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage;
}

/// A trait for float element types that can be used in SIMD operations.
///
/// Notably, this trait provides scalar fallback methods for true fused multiply-add (FMA) operations,
/// when they aren't available in the target architecture. Sometimes it's essential to have these
/// fallbacks for correctness, given FMAs rounding behavior.
pub trait FloatElement: num_traits::float::FloatCore + From<i8> {
    type Bits: MaskElement;
    type Signed: MaskElement;

    fn from_f32(value: f32) -> Self;

    fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self;
    fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self;
}

#[rustfmt::skip]
const _: () = {
    impl FloatElement for f32 {
        type Bits = u32;
        type Signed = i32;
        #[inline(always)] fn from_f32(value: f32) -> Self { value }
        #[inline(always)] fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fmaf(lhs, rhs, acc) }
        #[inline(always)] fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fmaf(lhs, rhs, -acc) }
        #[inline(always)] fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fmaf(lhs, -rhs, acc) }
        #[inline(always)] fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fmaf(lhs, -rhs, -acc) }
    }
    impl FloatElement for f64 {
        type Bits = u64;
        type Signed = i64;
        #[inline(always)] fn from_f32(value: f32) -> Self { value as f64 }
        #[inline(always)] fn scalar_mul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fma(lhs, rhs, acc) }
        #[inline(always)] fn scalar_mul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fma(lhs, rhs, -acc) }
        #[inline(always)] fn scalar_nmul_add(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fma(lhs, -rhs, acc) }
        #[inline(always)] fn scalar_nmul_sub(lhs: Self, rhs: Self, acc: Self) -> Self { libm::fma(lhs, -rhs, -acc) }
    }
};

#[inline(always)]
fn zip_ternary<R: FloatRegister, F>(mut lhs: R::Storage, rhs: R::Storage, acc: R::Storage, f: F) -> R::Storage
where
    F: Fn(&mut R::Element, R::Element, R::Element),
{
    let rhs = R::iter(&rhs);
    let acc = R::iter(&acc);

    for ((lhs, rhs), acc) in R::iter_mut(&mut lhs).zip(rhs).zip(acc) {
        f(lhs, *rhs, *acc);
    }

    lhs
}

pub trait FloatRegister: SignedRegister<Element: FloatElement> + Interoperable<Self::Bits, Self::Signed> {
    type Bits: UnsignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElement>::Bits>
        + Interoperable<Self, Self::Signed>;
    type Signed: SignedIntegerRegister<Lanes = Self::Lanes, Element = <Self::Element as FloatElement>::Signed>
        + Interoperable<Self, Self::Bits>;

    const HAS_TRUE_FMA: bool;

    const NEG_ZERO: Self::Storage;
    const INFINITY: Self::Storage;
    const NEG_INFINITY: Self::Storage;
    const NAN: Self::Storage;
    const EPSILON: Self::Storage;

    #[inline(always)]
    fn is_nan(value: Self::Storage) -> Self::Storage {
        // easiest way to check for NaN is to check if it's not equal to itself
        Self::ne(value, value)
    }

    #[inline(always)]
    fn is_infinite(value: Self::Storage) -> Self::Storage {
        Self::eq(Self::abs(value), Self::INFINITY)
    }

    #[inline(always)]
    fn is_finite(value: Self::Storage) -> Self::Storage {
        Self::lt(Self::abs(value), Self::INFINITY)
    }

    fn is_subnormal(value: Self::Storage) -> Self::Storage;
    fn is_zero_or_subnormal(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn is_normal(value: Self::Storage) -> Self::Storage {
        // !is_zero_or_subnormal(value) && is_finite(value)
        Self::andnot(Self::is_zero_or_subnormal(value), Self::is_finite(value))
    }

    #[inline(always)]
    fn mul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::add(Self::mul(lhs, rhs), acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::sub(Self::mul(lhs, rhs), acc)
    }

    #[inline(always)]
    fn nmul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::sub(acc, Self::mul(lhs, rhs))
    }

    #[inline(always)]
    fn nmul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::neg(Self::mul_adde(lhs, rhs, acc))
    }

    #[inline]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_mul_add(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_mul_sub(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_nmul_add(*lhs, rhs, acc);
        })
    }

    #[inline]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        zip_ternary::<Self, _>(lhs, rhs, acc, |lhs, rhs, acc| {
            *lhs = FloatElement::scalar_nmul_sub(*lhs, rhs, acc);
        })
    }

    fn sqrt(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn rcp(value: Self::Storage) -> Self::Storage {
        Self::div(Self::ONE, value)
    }

    #[inline(always)]
    fn rsqrt(value: Self::Storage) -> Self::Storage {
        Self::rcp(Self::sqrt(value))
    }

    const HAS_APPROX_RSQRT: bool;
    const HAS_APPROX_RCP: bool;

    fn floor(value: Self::Storage) -> Self::Storage;
    fn ceil(value: Self::Storage) -> Self::Storage;
    fn round(value: Self::Storage) -> Self::Storage;
    fn trunc(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn fract(value: Self::Storage) -> Self::Storage {
        Self::sub(value, Self::trunc(value))
    }

    #[inline(always)]
    fn combine_sign(value: Self::Storage, sign: Self::Storage) -> Self::Storage {
        Self::xor(value, Self::and(sign, Self::NEG_ZERO))
    }

    fn next_up(value: Self::Storage) -> Self::Storage;
    fn next_down(value: Self::Storage) -> Self::Storage;
}

/// Extensions to the `FloatRegister` trait for the most common 3D linear algebra operations.
///
/// This is only available on 4-lane registers.
pub trait LinAlg3Register: FloatRegister {
    fn dot3(lhs: Self::Storage, rhs: Self::Storage) -> Self::Element;
    fn cross3(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn zero4(value: Self::Storage) -> Self::Storage;
    fn one4(value: Self::Storage) -> Self::Storage;

    fn min_element3(value: Self::Storage) -> Self::Element;
    fn max_element3(value: Self::Storage) -> Self::Element;
    fn sum_elements3(value: Self::Storage) -> Self::Element;
    fn prod_elements3(value: Self::Storage) -> Self::Element;
}

pub trait IntegerRegister: NumericRegister + ShiftRegister {
    fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    fn wrapping_sum(value: Self::Storage) -> Self::Element;
    fn wrapping_product(value: Self::Storage) -> Self::Element;

    /// Rotate bits left
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage;
    /// Rotate bits right
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage;

    /// Rotate bits left by a constant amount
    #[inline(always)]
    fn roli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self::rol(value, IMM8 as u32)
    }

    /// Rotate bits right by a constant amount
    #[inline(always)]
    fn rori<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self::ror(value, IMM8 as u32)
    }

    fn rolv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage;
    fn rorv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage;
    //fn rotatev(value: Self::Storage, shifts: impl Into<GenericArray<i32, Self::Lanes>>) -> Self::Storage;

    fn reverse_bits(value: Self::Storage) -> Self::Storage;
    fn count_ones(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn count_zeros(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Self::Storage) -> Self::Storage;
    fn trailing_zeros(value: Self::Storage) -> Self::Storage;

    #[inline(always)]
    fn leading_ones(value: Self::Storage) -> Self::Storage {
        Self::leading_zeros(Self::not(value))
    }

    #[inline(always)]
    fn trailing_ones(value: Self::Storage) -> Self::Storage {
        Self::trailing_zeros(Self::not(value))
    }
}

pub trait UnsignedIntegerRegister: IntegerRegister {
    /// Returns `floor(log2(x)) + 1`
    #[inline(always)]
    fn ilog2p1(value: Self::Storage) -> Self::Storage {
        Self::count_ones(Self::next_power_of_two_m1(value))
    }

    /// Next power of two minus 1
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage;

    fn is_power_of_two(value: Self::Storage) -> Self::Storage;

    fn parity(value: Self::Storage) -> Self::Storage;

    // TODO: Interleave bits?
}

pub trait SignedIntegerRegister: IntegerRegister + SignedRegister {}
impl<T> SignedIntegerRegister for T where T: IntegerRegister + SignedRegister {}
