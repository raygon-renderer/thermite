use crate::*;

trait SimdMathSinglePrecision<S: Simd>: SimdFloatVector<S, Element = f32> {
    fn sin_cos(self) -> (Self, Self);
}

trait SimdMathDoublePrecision<S: Simd>: SimdFloatVector<S, Element = f64> {}

pub trait SimdMathExt<S: Simd>: SimdFloatVector<S> {
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
    fn expm1(self) -> Self;

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

impl<S: Simd, T> SimdMathSinglePrecision<S> for T
where
    T: SimdFloatVector<S, Element = f32>,
{
    fn sin_cos(self) -> (Self, Self) {
        type Vi32 = <S as Simd>::Vi32;
        type Vf32 = <S as Simd>::Vf32;

        let dp1f = Self::splat(0.78515625 * 2.0);
        let dp2f = Self::splat(2.4187564849853515625E-4 * 2.0);
        let dp3f = Self::splat(3.77489497744594108E-8 * 2.0);

        let p0sinf = Self::splat(-1.6666654611E-1);
        let p1sinf = Self::splat(8.3321608736E-3);
        let p2sinf = Self::splat(-1.9515295891E-4);

        let p0cosf = Self::splat(4.166664568298827E-2);
        let p1cosf = Self::splat(-1.388731625493765E-3);
        let p2cosf = Self::splat(2.443315711809948E-5);

        let xa = self.abs();

        let y: Self = (xa * Self::splat(2.0 / std::f32::consts::PI)).round();
        let q = Vi32::from_cast(y).into_bits(); // cast to signed (faster), then transmute

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3f, y.nmul_add(dp2f, y.nmul_add(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2: Self = x * x;
        let x3: Self = x2 * x;
        let x4: Self = x2 * x2;
        let mut s = x4.mul_add(p2sinf, x2.mul_add(p1sinf, p0sinf)).mul_add(x3, x);
        let mut c = x4
            .mul_add(p2cosf, x2.mul_add(p1cosf, p0cosf))
            .mul_add(x4, Self::splat(0.5).nmul_add(x2, Self::one()));

        // swap sin and cos if odd quadrant
        let swap = (q & Vu32::one()).ne(Vu32::zero());

        let mut overflow = q.gt(Vu32::splat(0x2000000)); // q big if overflow
        overflow &= Vu32::from_cast_mask(xa.is_finite());

        s = overflow.select(Self::zero(), s);
        c = overflow.select(Self::one(), c);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin: Vf32 = Vf32::from_bits((q << 30) ^ self.into_bits());
        let signcos: Vf32 = Vf32::from_bits(((q + Vu32::one()) & Vu32::splat(2)) << 30);

        // combine signs
        (sin1 ^ (signsin & Self::neg_zero()), cos1 ^ signcos)
    }
}
