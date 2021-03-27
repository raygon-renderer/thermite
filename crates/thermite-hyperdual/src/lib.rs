#![no_std]
use thermite::*;

use core::{
    fmt,
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

pub type Hyperdual<S, V, const N: usize> = HyperdualP<S, V, policies::Performance, N>;
pub type DualNumber<S, V> = Hyperdual<S, V, 1>;

pub struct HyperdualP<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> {
    /// Real part
    pub re: V,
    /// Dual parts
    pub du: [V; N],
    _simd: PhantomData<(S, P)>,
}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Clone for HyperdualP<S, V, P, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Copy for HyperdualP<S, V, P, N> {}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> fmt::Debug for HyperdualP<S, V, P, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HyperdualP")
            .field("re", &self.re)
            .field("du", &self.du)
            .finish()
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> HyperdualP<S, V, P, N> {
    #[inline(always)]
    pub fn new(re: V, du: [V; N]) -> Self {
        Self {
            re,
            du,
            _simd: PhantomData,
        }
    }

    #[inline(always)]
    pub fn real(re: V) -> Self {
        Self::new(re, [V::zero(); N])
    }

    #[inline(always)]
    pub fn one() -> Self {
        Self::real(V::one())
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::real(V::zero())
    }

    #[inline(always)]
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(V) -> V,
    {
        self.map_dual(f(self.re), f)
    }

    #[inline(always)]
    pub fn map_dual<F>(mut self, re: V, f: F) -> Self
    where
        F: Fn(V) -> V,
    {
        self.re = re;
        for dual in &mut self.du {
            *dual = f(*dual);
        }
        self
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> HyperdualP<S, V, P, N>
where
    V: SimdVectorizedMath<S>,
{
    #[inline(always)]
    fn div_dual(self, re: V, denom: V) -> Self {
        if N > 1 {
            let rcp = denom.reciprocal_p::<P>();
            self.map_dual(re, |x| x * rcp)
        } else {
            self.map_dual(re, |x| x / denom)
        }
    }

    #[inline(always)]
    pub fn fract(mut self) -> Self {
        self.re = self.re.fract();
        self
    }

    #[inline(always)]
    pub fn signum(self) -> Self {
        Self::real(self.re.signum())
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        let signum = self.re.signum();
        self.map(|x| x * signum)
    }

    #[inline(always)]
    pub fn select(mask: Mask<S, V>, t: Self, f: Self) -> Self {
        let mut t = t; // Weird compiler bug
        for i in 0..N {
            t.du[i] = mask.select(t.du[i], f.du[i]);
        }
        t.re = mask.select(t.re, f.re);
        t
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::select(self.re.lt(other.re), self, other)
    }

    #[inline(always)]
    pub fn max(mut self, other: Self) -> Self {
        Self::select(self.re.gt(other.re), self, other)
    }

    #[inline(always)]
    pub fn mul_add(mut self, m: Self, a: Self) -> Self {
        for i in 0..N {
            self.du[i] = self.du[i].mul_add(m.re, self.re.mul_add(m.du[i], a.du[i]));
        }
        self.re = self.re.mul_add(m.re, a.re);
        self
    }

    #[inline(always)]
    pub fn powi(self, n: i32) -> Self {
        let r = self.re.powi_p::<P>(n - 1);
        let nf = V::splat_as(n) * r;
        self.map_dual(self.re * r, |x| x * nf)
    }

    #[inline(always)]
    pub fn powf(mut self, n: Self) -> Self {
        let re_n1 = self.re.powf_p::<P>(n.re - V::one());

        let re = re_n1 * self.re; // re^n

        let a = n.re * re_n1; // n * re^(n-1)
        let b = re * self.re.ln_p::<P>();

        self.re = re;
        for i in 0..N {
            self.du[i] = a.mul_add(self.du[i], b * n.du[i]);
        }
        self
    }

    #[inline(always)]
    pub fn exp(self) -> Self {
        let re = self.re.exp_p::<P>();
        self.map_dual(re, |x| re * x)
    }

    #[inline(always)]
    pub fn exp2(self) -> Self {
        let re = self.re.exp2_p::<P>();
        let re_ln2 = V::LN_2() * re;
        self.map_dual(re, |x| x * re_ln2)
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        self.div_dual(self.re.ln_p::<P>(), self.re)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let re = self.re.sqrt();
        self.div_dual(re, re + re)
    }

    #[inline(always)]
    pub fn cbrt(self) -> Self {
        let re = self.re.cbrt();
        self.div_dual(re, re + re + re)
    }

    fn hypot(self, other: Self) -> Self {
        let c = self.re.hypot(other.re);
        let mut v = Self::real(c);

        let inv_c = c.reciprocal_p::<P>();
        for i in 0..N {
            let x = self.du[i];
            let y = other.du[i];

            v.du[i] = self.re.mul_add(x, other.re * y);

            if N > 1 {
                v.du[i] *= inv_c;
            } else {
                v.du[i] /= c;
            }
        }

        v
    }

    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();

        let mut sine = self;
        let mut cosi = self;

        sine.re = s;
        cosi.re = c;
        for i in 0..N {
            sine.du[i] *= c;
            cosi.du[i] *= s;
        }

        (sine, cosi)
    }

    #[inline(always)]
    pub fn tan(self) -> Self {
        let t = self.re.tan_p::<P>();
        let c = t.mul_add(t, V::one());
        self.map_dual(t, |x| x * c)
    }

    #[inline(always)]
    pub fn asin(self) -> Self {
        let c = self.re.nmul_adde(self.re, V::one()).invsqrt_p::<P>();
        self.map_dual(self.re.asin(), |x| x * c)
    }

    #[inline(always)]
    pub fn acos(self) -> Self {
        let c = self.re.nmul_adde(self.re, V::one()).invsqrt_p::<P>().neg();
        self.map_dual(self.re.acos(), |x| x * c)
    }

    #[inline(always)]
    pub fn atan(self) -> Self {
        let c = self.re.mul_adde(self.re, V::one());
        self.div_dual(self.re.atan(), c)
    }

    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        let c = y.re.mul_add(y.re, x.re * x.re);

        let mut v = Self::real(y.re.atan2(x.re));

        let inv_c = c.reciprocal_p::<P>();
        for i in 0..N {
            v.du[i] = x.re.mul_sub(y.du[i], y.re * x.du[i]) * c;

            if N > 1 {
                v.du[i] *= inv_c;
            } else {
                v.du[i] /= c;
            }
        }

        v
    }

    #[inline(always)]
    pub fn sinh_cosh(self) -> (Self, Self) {
        let s = self.re.sinh_p::<P>();
        let c = self.re.cosh_p::<P>();
        (self.map_dual(s, |x| x * c), self.map_dual(c, |x| x * s))
    }

    #[inline(always)]
    pub fn tanh(self) -> Self {
        let re = self.re.tanh_p::<P>();
        let c = re.nmul_add(re, V::one()); // 1 - r^2
        self.map_dual(re, |x| x * c)
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Add<Self> for HyperdualP<S, V, P, N> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        self.re += rhs.re;
        for i in 0..N {
            self.du[i] += rhs.du[i];
        }
        self
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Sub<Self> for HyperdualP<S, V, P, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self {
        self.re -= rhs.re;
        for i in 0..N {
            self.du[i] -= rhs.du[i];
        }
        self
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Mul<Self> for HyperdualP<S, V, P, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self {
        for i in 0..N {
            self.du[i] = self.re.mul_add(rhs.du[i], rhs.re * self.du[i]);
        }
        self.re *= rhs.re;
        self
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy, const N: usize> Div<Self> for HyperdualP<S, V, P, N>
where
    V: SimdVectorizedMath<S>,
{
    type Output = Self;

    #[inline(always)]
    fn div(mut self, rhs: Self) -> Self {
        let d = self.re * rhs.re;

        let inv_d = d.reciprocal_p::<P>();
        for i in 0..N {
            self.du[i] = rhs.re.mul_sub(self.du[i], self.re * rhs.du[i]) * d;

            if N > 1 {
                self.du[i] *= inv_d;
            } else {
                self.du[i] /= d;
            }
        }
        self.re /= rhs.re;
        self
    }
}
