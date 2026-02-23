use num_traits::{WrappingSub as _, Zero as _};

use crate::divider::Denominator;

use super::*;

#[inline(always)]
pub fn div_epi_bf<R: SignedIntegerRegister>(numers: Storage<R>, multiplier: R::Element, shift: u8) -> Storage<R> {
    // unpack sign and shift
    let sign: R::Element = Element::from_i8((shift as i8) >> 7); // sign extended to element size
    let shift = shift & <R::Element as Denominator>::SHIFT_MASK;

    let sign = R::splat(sign);

    // q = mulhi(numers, multiplier) + numers
    let mut q = R::add(numers, R::mulhi(numers, R::splat(multiplier)));

    // If q is non-negative, we have nothing to do.
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2.
    let is_power_of_two: R::Element = Element::from_u8(multiplier.is_zero() as u8);

    // arithmetic shift to get 0 or -1 (all bits set)
    let q_sign = R::sra(q, const { size_of::<R::Element>() as u32 * 8 - 1 });
    // mask = (1 << shift) - is_power_of_two, Element makes this ugly but should compile down fine,
    // and despite Element being signed, shl and wrapping_sub are invariant to sign.
    let mask = R::splat((R::Element::ONE << R::Element::from_u8(shift)).wrapping_sub(&is_power_of_two));

    q = R::add(q, R::bitand(q_sign, mask)); // q = q + (q_sign & mask)
    q = R::sra(q, shift as u32); // q >>= shift, arithmetic shift
    q = R::sub(R::bitxor(q, sign), sign); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub fn divv_epi_bf<R: SignedIntegerRegister>(
    numers: Storage<R>,
    multipliers: Storage<R>,
    shifts: Storage<R>,
) -> Storage<R> {
    // unpack sign and shift
    let shift_mask = R::Element::from_u8(<R::Element as Denominator>::SHIFT_MASK);
    let signs = R::sra(shifts, const { size_of::<R::Element>() as u32 * 8 - 1 }); // sign extended to element size

    // extract shifts and cast to unsigned for shifting
    let shifts = <R::Unsigned as CastRegister<R>>::cast_from(R::bitand(shifts, R::splat(shift_mask)));

    // q = mulhi(numers, multiplier) + numers
    let mut q = R::add(numers, R::mulhi(numers, multipliers));

    // If q is non-negative, we have nothing to do.
    // If q is negative, we want to add either (2**shift)-1 if d is
    // a power of 2, or (2**shift) if it is not a power of 2.
    let is_power_of_two = R::eq(multipliers, R::ZERO);

    // arithmetic shift to get 0 or -1 (all bits set)
    let q_sign = R::sra(q, const { size_of::<R::Element>() as u32 * 8 - 1 });
    // mask = (1 << shift) - is_power_of_two, Element makes this ugly but should compile down fine,
    // and despite Element being signed, shl and wrapping_sub are invariant to sign.
    // Note that R::from_mask should fill the entire element with 0s or 1s appropriately, to be either 0 or -1.
    let mask = R::sub(R::shlv(R::ONE, shifts), R::from_mask(is_power_of_two));

    q = R::add(q, R::bitand(q_sign, mask)); // q = q + (q_sign & mask)
    q = R::srav(q, shifts); // q >>= shift, arithmetic shift
    q = R::sub(R::bitxor(q, signs), signs); // q = (q ^ sign) - sign

    q
}

#[inline(always)]
pub fn div_epu_bf<R: UnsignedIntegerRegister>(numers: Storage<R>, multiplier: R::Element, shift: u8) -> Storage<R> {
    let mut q = R::mulhi(numers, R::splat(multiplier));

    // q += (numers - q) << 1
    q = R::add(q, R::shri::<1>(R::sub(numers, q)));

    R::shr(q, shift as u32) // q >>= shift
}

#[inline(always)]
pub fn divv_epu_bf<R: UnsignedIntegerRegister>(
    numers: Storage<R>,
    multipliers: Storage<R>,
    shifts: Storage<R>,
) -> Storage<R> {
    let mut q = R::mulhi(numers, multipliers);

    // q += (numers - q) << 1
    q = R::add(q, R::shri::<1>(R::sub(numers, q)));

    R::shrv(q, shifts) // q >>= shifts
}

#[inline(always)]
pub fn div_epi<R: SignedIntegerRegister>(numers: Storage<R>, multiplier: R::Element, shift: u8) -> Storage<R> {
    // unpack sign and shift
    let sign = R::splat(Element::from_i8((shift as i8) >> 7)); // sign extended to element size
    let masked_shift = shift & <R::Element as Denominator>::SHIFT_MASK;

    if multiplier.is_zero() {
        // mask = (1 << shift) - 1
        let mask = R::splat((R::Element::ONE << R::Element::from_u8(masked_shift)).wrapping_sub(&Element::ONE));

        // q = (numer >> (BITS - 1)) & mask
        let mut q = R::bitand(R::sra(numers, const { size_of::<R::Element>() as u32 * 8 - 1 }), mask);

        q = R::sra(R::add(numers, q), masked_shift as u32); // (q + numers) >>= shift

        // q = (q ^ sign) - sign;
        q = R::sub(R::bitxor(q, sign), sign);

        q
    } else {
        let mut q = R::mulhi(numers, R::splat(multiplier));

        if (shift & crate::divider::ADD_MARKER) != 0 {
            // q += ((numer ^ sign) - sign)
            q = R::add(q, R::sub(R::bitxor(numers, sign), sign));
        }

        q = R::sra(q, masked_shift as u32); // q >>= shift
        // q += (q < 0) by shifting sign bit to LSB position
        q = R::add(q, R::shr(q, const { size_of::<R::Element>() as u32 * 8 - 1 }));

        q
    }
}

#[inline(always)]
pub fn div_epu<R: UnsignedIntegerRegister>(numers: Storage<R>, multiplier: R::Element, shift: u8) -> Storage<R> {
    if multiplier.is_zero() {
        return R::shr(numers, shift as u32); // q >>= shift
    }

    let mut q = R::mulhi(numers, R::splat(multiplier));

    if (shift & crate::divider::ADD_MARKER) != 0 {
        q = R::add(R::shr(R::sub(numers, q), 1), q);
        q = R::shr(q, (shift & <R::Element as Denominator>::SHIFT_MASK) as u32);
    } else {
        q = R::shr(q, shift as u32);
    }

    q
}
