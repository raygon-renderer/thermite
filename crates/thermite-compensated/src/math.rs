use super::{Compensated, CompensatedRegister};
use num_traits::MulAdd as _;
use thermite::{
    Vector,
    math::{
        FloatConsts, MathWithPolicy,
        policy::{Policy, PrecisionPolicy},
    },
    register::{Element, FloatElement, FloatRegister},
};

impl<R: CompensatedRegister> MathWithPolicy<R> for Compensated<R>
where
    Vector<R>: MathWithPolicy<R>,
{
    fn ldexp_p<P: Policy>(self, exp: Vector<<R as FloatRegister>::Signed>) -> Self {
        todo!()
    }

    fn frexp_p<P: Policy>(self) -> (Self, Vector<<R as FloatRegister>::Signed>) {
        todo!()
    }

    #[inline(always)]
    fn to_degrees_p<P: Policy>(self) -> Self {
        self * Self::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians_p<P: Policy>(self) -> Self {
        self * Self::FRAC_PI_180
    }

    fn tolerance_p<P: Policy>() -> Self {
        todo!()
    }

    #[inline(always)]
    fn poly_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_add(self, Self::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(self, |i| unsafe { Self::splat(*coeffs.get_unchecked(i)) })
    }

    #[inline(always)]
    fn poly_rev_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_add(self, Self::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(self, |i| unsafe { Self::splat(*coeffs.get_unchecked(N - 1 - i)) })
    }

    #[inline(always)]
    fn poly_rational_p<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self {
        self.poly_p::<P, N>(numerator) / self.poly_p::<P, D>(denominator)
    }

    fn sum_f_p<P: Policy, F>(start: i64, end: i64, f: F) -> Result<Self, Self>
    where
        F: FnMut(i64) -> Self,
    {
        todo!()
    }

    fn prod_f_p<P: Policy, F>(start: i64, end: i64, f: F) -> Result<Self, Self>
    where
        F: FnMut(i64) -> Self,
    {
        todo!()
    }

    fn newtons_method_p<P: Policy, F>(self, tolerance: Self, bounds: Option<(Self, Self)>, f: F) -> Self
    where
        F: FnMut(Self) -> (Self, Self),
    {
        todo!()
    }

    #[inline(always)]
    fn smoothstep_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        use thermite::math::internal::Smoothstep;

        let mut t = self;

        if let Some((a, b)) = edges {
            t = (t - a) / (b - a);
        }

        if P::POLICY.check_overflow {
            t = t.clamp(Self::ZERO, Self::ONE);
        }

        match N {
            0 => Self::new(t.value().step_p::<P>(Vector::HALF)),
            1 => t,
            _ => {
                t.powi_p::<P>(N as i32)
                    * const { Smoothstep::<R::Element, N>::COEFFICIENTS }
                        .into_iter()
                        .fold(Self::ZERO, |res, c| {
                            res.mul_add(t, Self::splat(FloatElement::from_i64(c)))
                        })
            }
        }
    }

    fn inverse_smoothstep_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        todo!()
    }

    fn smoothstep_derivative_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        todo!()
    }

    fn smooth_interpolator_p<P: Policy>(self, edges: Option<(Self, Self)>, k: Self) -> Self {
        todo!()
    }

    fn smooth_interpolator_inverse_p<P: Policy>(self, edges: Option<(Self, Self)>, k: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn step_p<P: Policy>(self, edge: Self) -> Self {
        // removes error component
        Self::new(self.value().step_p::<P>(edge.value()))
    }

    #[inline(always)]
    fn lerp_p<P: Policy>(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }

    #[inline(always)]
    fn scale_p<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        let in_range = in_max - in_min;
        let t = (self - in_min) / in_range;
        t.mul_add(out_max - out_min, out_min)
    }

    #[inline(always)]
    fn reciprocal_p<P: Policy>(self) -> Self {
        Self::ONE / self
    }

    #[inline(always)]
    fn inverse_sqrt_p<P: Policy>(self) -> Self {
        Self::ONE / self.sqrt()
    }

    #[inline(always)]
    fn powi_p<P: Policy>(self, mut e: i32) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        if e < 0 {
            e = -e;
            x = Self::reciprocal_p::<P>(x);
        }

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x *= x;
            e >>= 1;
        }

        res
    }

    fn powiv_p<P: Policy>(self, e: Vector<R::Signed>) -> Self {
        todo!()
    }

    #[inline(always)]
    fn hypot_p<P: Policy>(self, y: Self) -> Self {
        self.mul_add(self, y * y).sqrt() // TODO: Check for overflow/underflow
    }

    fn sin_cos_p<P: Policy>(self) -> (Self, Self) {
        todo!()
    }

    fn sin_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn cos_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn tan_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn sinc_p<P: Policy>(self) -> Self {
        self.sin_p::<P>() / self
    }

    fn sin_pix_p<P: Policy>(self) -> Self {
        self * (self * Self::PI).sin_p::<P>()
    }

    fn sinh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn cosh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn tanh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn asin_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn acos_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atan_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atan2_p<P: Policy>(self, x: Self) -> Self {
        todo!()
    }

    fn asinh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn acosh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atanh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn exp_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn exph_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn exp2_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn exp10_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn exp_m1_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn powf_p<P: Policy>(self, e: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn cbrt_p<P: Policy>(self) -> Self {
        let s = MathWithPolicy::cbrt_p::<P>(self.value);

        // s^2
        let (p2, e2) = super::two_prod(s, s);
        // s^3
        let (p3, e3_base) = super::two_prod(p2, s);

        let e3 = s.mul_adde(e2, e3_base); // (s * e2) + e3_base

        // residual
        let r = (self.value - p3) + (self.error - e3);

        // derivative = 3s^2
        let deriv = Vector::<R>::splat(Element::from_i8(3)) * p2;

        Self::renormalized(s, r / deriv)
    }

    fn ln_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn ln_1p_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn log2_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn log10_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn log_p<P: Policy>(self, base: Self) -> Self {
        todo!()
    }

    fn log_n_p<P: Policy, const N: usize>(self) -> Self {
        todo!()
    }

    fn ln1m_expnx_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn ln1m_expnx_ext_p<P: Policy>(self, lnx: Self) -> Self {
        todo!()
    }

    fn erf_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn erfc_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn erfinv_p<P: Policy>(self) -> Self {
        todo!()
    }
}
