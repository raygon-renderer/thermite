#![allow(unused)]

use crate::*;

pub trait SimdVectorizedMath<S: Simd>: SimdFloatVector<S> {
    #[inline]
    fn sin(self) -> Self {
        self.sin_cos().0
    }

    #[inline]
    fn cos(self) -> Self {
        self.sin_cos().1
    }

    #[inline]
    fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
    }

    fn sin_cos(self) -> (Self, Self);

    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;

    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn expm1(self) -> Self {
        self.exp() - Self::one()
    }

    fn powf(self, e: Self) -> Self;
    fn powi(self, e: S::Vi32) -> Self {
        let mut res = Self::one();
        let mut x = self;

        /*
        x = e.is_negative().select(Self::one() / x, x);
        e = e.abs();

        let zero = Vi32::<S>::zero();

        loop {
            res = (e & Vi32::<S>::one()).ne(zero).select(res * x, res);

            e >>= 1;

            let fin = e.eq(zero);

            x = fin.select(x, x * x);

            if fin.all() {
                break;
            }
        }
        */

        res
    }

    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;

    fn erf(self) -> Self;
    fn ierf(self) -> Self;

    #[inline(always)]
    fn erfcf(self) -> Self {
        Self::one() - self.erf()
    }
}

impl<S: Simd, T> SimdVectorizedMath<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdVectorizedMathInternal<S, Vf = T>,
{
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        <<Self as SimdVectorBase<S>>::Element as SimdVectorizedMathInternal<S>>::sin_cos(self)
    }
}

trait SimdVectorizedMathInternal<S: Simd>: SimdElement {
    type Vf: SimdFloatVector<S, Element = Self>;

    fn sin_cos(x: Self::Vf) -> (Self::Vf, Self::Vf);
}

impl<S: Simd> SimdVectorizedMathInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    type Vf = <S as Simd>::Vf32;

    #[inline(always)]
    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        let dp1f = Vf32::<S>::splat(0.78515625 * 2.0);
        let dp2f = Vf32::<S>::splat(2.4187564849853515625E-4 * 2.0);
        let dp3f = Vf32::<S>::splat(3.77489497744594108E-8 * 2.0);
        let p0sinf = Vf32::<S>::splat(-1.6666654611E-1);
        let p1sinf = Vf32::<S>::splat(8.3321608736E-3);
        let p2sinf = Vf32::<S>::splat(-1.9515295891E-4);
        let p0cosf = Vf32::<S>::splat(4.166664568298827E-2);
        let p1cosf = Vf32::<S>::splat(-1.388731625493765E-3);
        let p2cosf = Vf32::<S>::splat(2.443315711809948E-5);

        let xa = xx.abs();

        /*
        let y = (xa * Vf32::<S>::splat(2.0 / std::f32::consts::PI)).round();
        let q = Vi32::<S>::from_cast(y).into_bits(); // cast to signed (faster), then transmute to unsigned

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3f, y.nmul_add(dp2f, y.nmul_add(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2: Vf32<S> = x * x;
        let x3: Vf32<S> = x2 * x;
        let x4: Vf32<S> = x2 * x2;
        let mut s = x4.mul_add(p2sinf, x2.mul_add(p1sinf, p0sinf)).mul_add(x3, x);
        let mut c = x4
            .mul_add(p2cosf, x2.mul_add(p1cosf, p0cosf))
            .mul_add(x4, Vf32::<S>::splat(0.5).nmul_add(x2, Vf32::<S>::one()));

        // swap sin and cos if odd quadrant
        let swap = (q & Vu32::<S>::one()).ne(Vu32::zero());

        let mut overflow = q.gt(Vu32::<S>::splat(0x2000000)); // q big if overflow
        overflow &= Vu32::<S>::from_cast_mask(xa.is_finite());

        s = overflow.select(Vf32::<S>::zero(), s);
        c = overflow.select(Vf32::<S>::one(), c);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf32::<S>::from_bits((q << 30) ^ xx.into_bits());
        let signcos = Vf32::<S>::from_bits(((q + Vu32::<S>::one()) & Vu32::<S>::splat(2)) << 30);

        // combine signs
        (sin1 ^ (signsin & Vf32::<S>::neg_zero()), cos1 ^ signcos)
        */

        (xa, xa)
    }
}
