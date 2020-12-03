use crate::*;

use core::convert::TryFrom;

#[derive(Debug, Clone, Copy)]
pub struct Compensated<S: Simd, V: SimdFloatVector<S>> {
    pub val: V,
    pub err: V,
    _simd: PhantomData<S>,
}

#[dispatch(thermite = "crate")]
impl<S: Simd, V: SimdFloatVector<S>> Compensated<S, V> {
    #[inline(always)]
    fn from_parts(val: V, err: V) -> Self {
        Compensated {
            val,
            err,
            _simd: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new(val: V) -> Self {
        Self::from_parts(val, V::zero())
    }

    #[inline(always)]
    pub fn value(self) -> V {
        self.val + self.err
    }

    #[inline(always)]
    pub fn product(a: V, b: V) -> Self {
        let val = a * b;

        if S::INSTRSET.has_true_fma() {
            Compensated::from_parts(val, a.mul_sub(b, val))
        } else {
            // split into half-ish-precision
            let factor = match V::ELEMENT_SIZE {
                4 => V::splat_as::<u32>(1u32 << 13 + 1),
                8 => V::splat_as::<u32>(1u32 << 27 + 1),
                _ => unsafe { crate::unreachable_unchecked() },
            };

            let (a1, a2) = {
                let c = factor * a;
                let x = c - (c - a);
                (x, a - x)
            };

            let (b1, b2) = {
                let c = factor * b;
                let x = c - (c - b);
                (x, b - x)
            };

            let err = a2 * b2 - (((val - a1 * b1) - a2 * b1) - a1 * b2);

            Compensated::from_parts(val, err)
        }
    }

    #[inline(always)]
    pub fn sum(a: V, b: V) -> Self {
        let x = a + b;
        let z = x - a;
        let y = (a - (x - z)) + (b - z);

        Compensated::from_parts(x, y)
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Add<V> for Compensated<S, V> {
    type Output = Self;

    fn add(mut self, rhs: V) -> Self {
        let pi = Self::sum(self.val, rhs);
        self.val = pi.val;
        self.err += pi.err;
        self
    }
}

// Accurate Floating Point Product, Stef Graillat
//
// https://www-pequan.lip6.fr/~graillat/papers/REC08_Paper_Graillat.pdf
impl<S: Simd, V: SimdFloatVector<S>> Mul<V> for Compensated<S, V> {
    type Output = Self;

    fn mul(mut self, rhs: V) -> Self {
        let pi = Self::product(self.val, rhs);
        self.val = pi.val;
        self.err = self.err * rhs + pi.err;
        self
    }
}
