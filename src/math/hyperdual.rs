use crate::*;

use std::marker::PhantomData;

pub type DuelNumber<S, V> = Hyperdual<S, V, 1>;

#[derive(Debug, Clone, Copy)]
pub struct Hyperdual<S: Simd, V: SimdFloatVector<S>, const N: usize> {
    pub re: V,
    pub du: [V; N],
    _simd: PhantomData<S>,
}

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Hyperdual<S, V, N> {
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

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Hyperdual<S, V, N>
where
    V: SimdVectorizedMath<S>,
{
    #[inline(always)]
    pub fn abs(self) -> Self {
        let signum = self.re.signum();
        self.map(|x| x * signum)
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
        let r = self.re.powi(n - 1);
        unimplemented!() // TODO
    }

    #[inline(always)]
    pub fn powf(mut self, n: Self) -> Self {
        let re = self.re.powf(n.re);
        let a = n.re * self.re.powf(n.re - V::one());
        let b = re * self.re.ln();
        self.re = re;
        for i in 0..N {
            self.du[i] = a.mul_add(self.du[i], b * n.du[i]);
        }
        self
    }

    #[inline(always)]
    pub fn exp(self) -> Self {
        let re = self.re.exp();
        self.map_dual(re, |x| re * x)
    }

    pub fn exp2(self) -> Self {
        let re = self.re.exp2();
        unimplemented!() // TODO

        //let ln2 =
        //self.map_dual(re, |x| )
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        self.map_dual(self.re.ln(), |x| x / self.re)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let re = self.re.sqrt();
        let re2 = re + re;
        if N > 1 {
            let d = V::one() / re2;
            self.map_dual(re, |x| x * d)
        } else {
            self.map_dual(re, |x| x / re2)
        }
    }

    #[inline(always)]
    pub fn cbrt(self) -> Self {
        let re = self.re.cbrt();
        let re3 = re + re + re;
        if N > 1 {
            let d = V::one() / re3;
            self.map_dual(re, |x| x * d)
        } else {
            self.map_dual(re, |x| x / re3)
        }
    }

    #[inline(always)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();

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
        let t = self.re.tan();
        let c = t.mul_add(t, V::one());
        self.map_dual(t, |x| x * c)
    }

    #[inline(always)]
    pub fn sinh_cosh(self) -> (Self, Self) {
        let s = self.re.sinh();
        let c = self.re.cosh();
        (self.map_dual(s, |x| x * c), self.map_dual(c, |x| x * s))
    }

    #[inline(always)]
    pub fn tanh(self) -> Self {
        let re = self.re.tanh();
        let c = re.nmul_add(re, V::one()); // 1 - r^2
        self.map_dual(re, |x| x * c)
    }
}

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Add<Self> for Hyperdual<S, V, N> {
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

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Sub<Self> for Hyperdual<S, V, N> {
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

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Mul<Self> for Hyperdual<S, V, N> {
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

#[dispatch(S, thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>, const N: usize> Div<Self> for Hyperdual<S, V, N> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, rhs: Self) -> Self {
        let d = self.re * rhs.re;
        if N > 1 {
            // precompute division
            let d = V::one() / d;

            for i in 0..N {
                self.du[i] = rhs.re.mul_sub(self.du[i], self.re * rhs.du[i]) * d;
            }
        } else {
            for i in 0..N {
                self.du[i] = rhs.re.mul_sub(self.du[i], self.re * rhs.du[i]) / d;
            }
        }
        self.re /= rhs.re;
        self
    }
}
