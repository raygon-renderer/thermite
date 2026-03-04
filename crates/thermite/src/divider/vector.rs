use generic_array::{GenericArray, typenum};

use crate::{
    Vector,
    divider::Denominator,
    register::{Element, IntegerRegister, Lanes, Register, dp::DoublePumpRegister},
    vector::GenericVector as _,
};

/// Precomputed multipliers and shifts for branchless vectorized division
pub struct VectorDivider<R: Register> {
    pub multipliers: Vector<R>,
    pub shifts: Vector<R>,
}

impl<R: Register> Clone for VectorDivider<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for VectorDivider<R> {}

impl<R: Register> VectorDivider<DoublePumpRegister<R>>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    pub(crate) fn split(self) -> (VectorDivider<R>, VectorDivider<R>) {
        let DoublePumpRegister(low_multipliers, high_multipliers) = self.multipliers.0;
        let DoublePumpRegister(low_shifts, high_shifts) = self.shifts.0;

        (
            VectorDivider {
                multipliers: Vector(low_multipliers),
                shifts: Vector(low_shifts),
            },
            VectorDivider {
                multipliers: Vector(high_multipliers),
                shifts: Vector(high_shifts),
            },
        )
    }
}

impl<R: IntegerRegister> VectorDivider<R>
where
    R::Element: Denominator,
{
    #[inline]
    pub fn new(divisor: Vector<R>) -> Self {
        Self::try_new(divisor).unwrap()
    }

    #[inline]
    pub fn try_new(divisor: Vector<R>) -> Result<Self, super::UnsupportedDivisor> {
        let mut multipliers = GenericArray::default();
        let mut shifts = GenericArray::default();

        for ((m, s), d) in multipliers.iter_mut().zip(shifts.iter_mut()).zip(divisor.as_slice()) {
            let divisor = d.try_to_branchfree_divider()?;

            *m = divisor.multiplier();
            *s = Element::from_i8(divisor.shift() as i8);
        }

        Ok(VectorDivider {
            multipliers: Vector(R::new(multipliers)),
            shifts: Vector(R::new(shifts)),
        })
    }
}
