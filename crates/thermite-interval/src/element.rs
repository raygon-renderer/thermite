//! The scalar element form [`IntervalElem<E>`]: a per-lane `[lo, hi]` pair
//! over a bare float element, so that `Interval<V, W>` can be a
//! `GenericVector` with `Element = IntervalElem<V::Element>`.
//!
//! This is a separate type (rather than `Interval<E, W>` reused at the scalar
//! layer, the way `Compensated` nests) for two reasons. First, coherence: the
//! vector ops are written against `IntervalFloatVector` and the scalar ops
//! against `FloatElement`, and one type cannot carry both impl families.
//! Second, the widening policy is meaningless here. This form exists for lane
//! bookkeeping (`extract`/`insert`, tables, constants), not performance, so
//! its arithmetic always uses bump widening (`next_down`/`next_up` on the
//! element). Containment holds regardless.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use thermite::LargeInt;
use thermite::element::{Element, FloatElement, SignedElement};
use thermite::tribool::{self, Tribool};

/// Bound shorthand for the scalar element form.
pub trait ScalarFloat:
    FloatElement + SignedElement + PartialOrd + num_traits::NumOps + Neg<Output = Self> + Copy
{
}
impl<E> ScalarFloat for E where
    E: FloatElement + SignedElement + PartialOrd + num_traits::NumOps + Neg<Output = Self> + Copy
{
}

/// A closed scalar interval `[lo, hi]`: the [`Element`] of an
/// [`Interval`](crate::Interval) vector. See the module docs.
///
/// `PartialOrd` is the lexicographic `(lo, hi)` order, a total-ish order for
/// sorting and table bookkeeping. It is NOT the certainly/possibly comparison
/// of interval semantics (those live on the vector type).
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct IntervalElem<E> {
    pub lo: E,
    pub hi: E,
}

impl<E> IntervalElem<E> {
    /// The degenerate scalar interval `[v, v]`.
    #[inline(always)]
    pub const fn degenerate(v: E) -> Self
    where
        E: Copy,
    {
        Self { lo: v, hi: v }
    }
}

#[inline(always)]
fn smin<E: PartialOrd>(a: E, b: E) -> E {
    if b < a { b } else { a }
}

#[inline(always)]
fn smax<E: PartialOrd>(a: E, b: E) -> E {
    if b > a { b } else { a }
}

/// NaN artifacts from `0 * inf` endpoint combinations: set limit is 0.
#[inline(always)]
fn unpoison_s<E: ScalarFloat>(p: E) -> E {
    if p.partial_cmp(&p).is_none() { E::ZERO } else { p }
}

impl<E: ScalarFloat> Neg for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl<E: ScalarFloat> Add for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: E::next_down(self.lo + rhs.lo),
            hi: E::next_up(self.hi + rhs.hi),
        }
    }
}

impl<E: ScalarFloat> Sub for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<E: ScalarFloat> Mul for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let p0 = unpoison_s(self.lo * rhs.lo);
        let p1 = unpoison_s(self.lo * rhs.hi);
        let p2 = unpoison_s(self.hi * rhs.lo);
        let p3 = unpoison_s(self.hi * rhs.hi);

        Self {
            lo: E::next_down(smin(smin(p0, p1), smin(p2, p3))),
            hi: E::next_up(smax(smax(p0, p1), smax(p2, p3))),
        }
    }
}

impl<E: ScalarFloat> Div for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Divisor containing zero: the entire line, matching the vector form.
        // (`ONE / ZERO` is the portable spelling of the element's infinity.)
        if rhs.lo <= E::ZERO && E::ZERO <= rhs.hi {
            let inf = E::ONE / E::ZERO;
            return Self { lo: -inf, hi: inf };
        }

        let q0 = unpoison_s(self.lo / rhs.lo);
        let q1 = unpoison_s(self.lo / rhs.hi);
        let q2 = unpoison_s(self.hi / rhs.lo);
        let q3 = unpoison_s(self.hi / rhs.hi);

        Self {
            lo: E::next_down(smin(smin(q0, q1), smin(q2, q3))),
            hi: E::next_up(smax(smax(q0, q1), smax(q2, q3))),
        }
    }
}

impl<E: ScalarFloat> Rem for IntervalElem<E> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let qt = Self {
            lo: E::trunc(q.lo),
            hi: E::trunc(q.hi),
        };
        self - qt * rhs
    }
}

macro_rules! elem_assign {
    ($($trait:ident::$method:ident, $op:tt;)*) => {$(
        impl<E: ScalarFloat> $trait for IntervalElem<E> {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    )*};
}

elem_assign! {
    AddAssign::add_assign, +;
    SubAssign::sub_assign, -;
    MulAssign::mul_assign, *;
    DivAssign::div_assign, /;
    RemAssign::rem_assign, %;
}

#[rustfmt::skip]
impl<E: ScalarFloat> Element for IntervalElem<E> {
    type Signed = <E as Element>::Signed;
    type Unsigned = <E as Element>::Unsigned;

    const ONE: Self = Self { lo: E::ONE, hi: E::ONE };
    const ZERO: Self = Self { lo: E::ZERO, hi: E::ZERO };

    const ORDER_MAX: Self = Self { lo: E::ORDER_MAX, hi: E::ORDER_MAX };
    const ORDER_MIN: Self = Self { lo: E::ORDER_MIN, hi: E::ORDER_MIN };
    const HAS_UNORDERED: bool = E::HAS_UNORDERED;
    const IS_FLOAT: bool = E::IS_FLOAT;

    fn from_i8(value: i8) -> Self { Self::degenerate(E::from_i8(value)) }
    fn from_u8(value: u8) -> Self { Self::degenerate(E::from_u8(value)) }
    fn from_u16(value: u16) -> Self { Self::degenerate(E::from_u16(value)) }
}

impl<E: ScalarFloat> SignedElement for IntervalElem<E> {
    #[inline(always)]
    fn abs(self) -> Self {
        if self.lo <= E::ZERO && E::ZERO <= self.hi {
            // Contains zero: [0, max(|lo|, |hi|)].
            Self {
                lo: E::ZERO,
                hi: smax(SignedElement::abs(self.lo), SignedElement::abs(self.hi)),
            }
        } else {
            let a = SignedElement::abs(self.lo);
            let b = SignedElement::abs(self.hi);
            Self {
                lo: smin(a, b),
                hi: smax(a, b),
            }
        }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self {
            lo: SignedElement::signum(self.lo),
            hi: SignedElement::signum(self.hi),
        }
    }
}

#[rustfmt::skip]
impl<E: ScalarFloat> FloatElement for IntervalElem<E> {
    #[inline(always)]
    fn sqrt(this: Self) -> Self {
        // Domain-clamped, with scalar bump widening.
        let lo = if this.lo < E::ZERO { E::ZERO } else { this.lo };
        Self {
            lo: smax(E::next_down(E::sqrt(lo)), E::ZERO),
            hi: E::next_up(E::sqrt(this.hi)),
        }
    }

    // Monotone step functions: per-endpoint, exact.
    #[inline(always)] fn floor(this: Self) -> Self { Self { lo: E::floor(this.lo), hi: E::floor(this.hi) } }
    #[inline(always)] fn ceil(this: Self) -> Self { Self { lo: E::ceil(this.lo), hi: E::ceil(this.hi) } }
    #[inline(always)] fn round(this: Self) -> Self { Self { lo: E::round(this.lo), hi: E::round(this.hi) } }
    #[inline(always)] fn trunc(this: Self) -> Self { Self { lo: E::trunc(this.lo), hi: E::trunc(this.hi) } }

    // Set-map semantics: both endpoints step, monotone, exact.
    #[inline(always)] fn next_up(this: Self) -> Self { Self { lo: E::next_up(this.lo), hi: E::next_up(this.hi) } }
    #[inline(always)] fn next_down(this: Self) -> Self { Self { lo: E::next_down(this.lo), hi: E::next_down(this.hi) } }

    #[inline(always)]
    fn try_from_int(value: LargeInt) -> Option<Self> {
        // E::try_from_int is exact when it succeeds: degenerate.
        E::try_from_int(value).map(Self::degenerate)
    }

    #[inline(always)]
    fn try_from_ratio(n: LargeInt, d: LargeInt) -> Option<Self> {
        // The quotient may round: widen one ulp each way. (Loses degeneracy
        // for exactly-representable ratios, which is acceptable for a constant seed.)
        let v = E::try_from_ratio(n, d)?;
        Some(Self {
            lo: E::next_down(v),
            hi: E::next_up(v),
        })
    }

    const HAS_INFINITY: bool = E::HAS_INFINITY;
    const HAS_SIGNED_ZERO: bool = E::HAS_SIGNED_ZERO;
    const HAS_SUBNORMALS: bool = E::HAS_SUBNORMALS;

    // CAVEAT (mirrors the FloatConsts limitation in the crate docs): const
    // ratio seeds are degenerate at the element's rounded value, NOT
    // enclosures. Rigorous constant tables are future work.
    type ConstInt<const N: LargeInt> = IntervalElemConst<E::ConstInt<N>>;
    type ConstRatio<const N: LargeInt, const D: LargeInt> = IntervalElemConst<E::ConstRatio<N, D>>;
}

/// Splat-carrier lifting an element constant to a degenerate interval-element
/// constant. See the `ConstRatio` caveat above.
#[doc(hidden)]
pub struct IntervalElemConst<Inner>(core::marker::PhantomData<Inner>);

impl<E, Inner> thermite::vector::SplatConst<IntervalElem<E>> for IntervalElemConst<Inner>
where
    E: ScalarFloat,
    Inner: thermite::vector::SplatConst<E>,
{
    const VALUE: IntervalElem<E> = IntervalElem::degenerate(Inner::VALUE);
}

// FloatElement supertraits: a scalar interval "FMA" is mul-then-add, both
// bump-widened. There is no fused form to be more accurate than.
#[rustfmt::skip]
impl<E: ScalarFloat> thermite::vector::ops::MulAddExt<Self, Self> for IntervalElem<E> {
    type Output = Self;

    const HAS_NATIVE_FMA: Tribool = tribool::False;

    #[inline(always)] fn mul_add(self, m: Self, a: Self) -> Self { self * m + a }
    #[inline(always)] fn mul_sub(self, m: Self, a: Self) -> Self { self * m - a }
    #[inline(always)] fn nmul_add(self, m: Self, a: Self) -> Self { a - self * m }
    #[inline(always)] fn nmul_sub(self, m: Self, a: Self) -> Self { -(self * m + a) }

    #[inline(always)] fn mul_adde(self, m: Self, a: Self) -> Self { self * m + a }
    #[inline(always)] fn mul_sube(self, m: Self, a: Self) -> Self { self * m - a }
    #[inline(always)] fn nmul_adde(self, m: Self, a: Self) -> Self { a - self * m }
    #[inline(always)] fn nmul_sube(self, m: Self, a: Self) -> Self { -(self * m + a) }
}

// CAVEAT: degenerate point constants, NOT enclosures (crate-docs limitation).
macro_rules! elem_degenerate_consts {
    ($($name:ident),* $(,)?) => {
        impl<E: ScalarFloat> thermite::math::FloatConsts for IntervalElem<E> {
            $(const $name: Self = Self { lo: <E as thermite::math::FloatConsts>::$name, hi: <E as thermite::math::FloatConsts>::$name };)*
        }
    };
}

thermite::for_each_float_const!(elem_degenerate_consts);
