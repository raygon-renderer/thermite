// Derived from:
//
// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#![allow(unused)]

pub mod vector;

use core::ops::Deref;

macro_rules! decl_div_half {
    ($($t:ty => $dt:ty),*) => {
        paste::paste! {$(
            #[inline(always)]
            const fn [<div_ $dt _ $t _to_ $t>](u1: $t, u0: $t, v: $t) -> ($t, $t) {
                let v = v as $dt;
                let n = ((u1 as $dt) << <$t>::BITS) | (u0 as $dt);
                let res = (n / v) as $t; // truncate
                let rem = n.wrapping_sub((res as $dt).wrapping_mul(v));
                (res, rem as $t)
            }
        )*}
    };
}

decl_div_half!(u64 => u128, u32 => u64, u16 => u32, u8 => u16);

/// Trait for types that can be used as denominators in dividers.
pub trait Denominator: Sized {
    /// Create a divider for this denominator.
    fn to_divider(self) -> Divider<Self>;

    /// Create a branchfree divider for this denominator.
    fn to_branchfree_divider(self) -> BranchfreeDivider<Self>;

    /// Try to create a branchfree divider for this denominator.
    ///
    /// Branchfree dividers may not support all divisors, see the documentation of
    /// [`BranchfreeDivider`] for details.
    fn try_to_branchfree_divider(self) -> Result<BranchfreeDivider<Self>, UnsupportedDivisor>;

    /// Shift mask for this type.
    const SHIFT_MASK: u8;
}

/// Divider recommended for constant divisors.
///
/// When using constant divisors, divisions using this can remove extra branches
/// and generate ideal integer division code.
///
/// However, when used with dynamic input, the extra branches can be expensive,
/// therefore it is recommended to use the branchfree alternative for dynamic divisors.
#[repr(C, packed)]
pub struct Divider<T> {
    multiplier: T,
    shift: u8,
}

/// Divider without branching, useful for dynamic divisors where branches can be expensive,
/// at the cost of some extra work compared to the branching [`Divider`].
///
/// However, when used with constant input, this may perform extra unnecessary work that could
/// be removed in the branching [`Divider`].
///
/// Furthermore, the unsigned version of this divider does not support a divisor of 1,
/// due to the way the algorithm works.
#[repr(transparent)]
#[derive(Copy, PartialEq)]
pub struct BranchfreeDivider<T>(Divider<T>);

/// Error return by `TryFrom` implementations when the divisor is unsupported
/// by the branchfree divider. See the documentation of [`BranchfreeDivider`] for details.
#[derive(Debug, Clone, Copy)]
pub struct UnsupportedDivisor;

impl core::fmt::Display for UnsupportedDivisor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unsupported divisor")
    }
}

impl core::error::Error for UnsupportedDivisor {}

impl<T: Copy> Clone for BranchfreeDivider<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Deref for BranchfreeDivider<T> {
    type Target = Divider<T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Copy> Clone for Divider<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> Copy for Divider<T> {}

impl<T: PartialEq> PartialEq for Divider<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.multiplier() == other.multiplier() && self.shift == other.shift
    }
}

impl<T> Divider<T> {
    #[inline(always)]
    pub const fn multiplier(&self) -> T {
        // unaligned access needs `&raw const` and `read_unaligned` to be safe
        unsafe { (&raw const self.multiplier).read_unaligned() }
    }

    #[inline(always)]
    pub const fn shift(&self) -> u8 {
        // shift has an alignment of 1 byte anyway, so it's fine to read normally
        self.shift
    }
}

impl<T: Copy + core::fmt::Debug> core::fmt::Debug for Divider<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Divider")
            .field("multiplier", &self.multiplier())
            .field("shift", &self.shift())
            .finish()
    }
}

impl<T: Copy + core::fmt::Debug> core::fmt::Debug for BranchfreeDivider<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("BranchfreeDivider").field(&self.0).finish()
    }
}

pub(crate) const ADD_MARKER: u8 = 0x40;
pub(crate) const NEG_DIVISOR: u8 = 0x80;

macro_rules! impl_shift_mask {
    ($($ty:ty => $ut:ty),*) => {$(
        impl Divider<$ty> {
            const BITS: u32 = <$ty>::BITS as u32;

            /// !log2(N::BITS)
            pub(crate) const SHIFT_MASK: u8 = !(<$ut>::MAX << <$ty>::BITS.trailing_zeros()) as u8;
        }
    )*};
}

impl_shift_mask! {
    u8 => u8,
    u16 => u16,
    u32 => u32,
    u64 => u64,
    i8 => u8,
    i16 => u16,
    i32 => u32,
    i64 => u64
}

macro_rules! impl_unsigned_divider {
    ($($t:ty => $dt:ty),*) => {
        paste::paste! {$(
            impl BranchfreeDivider<$t> {
                /// Create a new branchfree divider for the given divisor.
                ///
                /// # Panics
                ///
                /// Panics if `d == 1`, as unsigned division by 1 is not supported in branchfree mode due
                /// to the way the algorithm works.
                #[inline(always)]
                pub const fn [<$t>](d: $t) -> Self {
                    BranchfreeDivider(Divider::[<$t _internal>](d, true))
                }

                /// Try to create a new branchfree divider for the given divisor.
                ///
                /// The unsigned branchfree divider does not support a divisor of 1, so this returns
                /// `None` in that case.
                #[inline(always)]
                pub const fn [<try_ $t>](d: $t) -> Option<Self> {
                    if d == 1 { None } else { Some(Self::[<$t>](d)) }
                }

                #[inline(always)]
                pub fn divide(self, x: $t) -> $t {
                    let q = Divider::<$t>::mullhi(x, self.multiplier());
                    let t = x.wrapping_sub(q) >> 1;
                    t.wrapping_add(q) >> self.shift()
                }
            }

            impl From<$t> for Divider<$t> {
                #[inline(always)]
                fn from(d: $t) -> Self {
                    Self::[<$t>](d)
                }
            }

            impl TryFrom<$t> for BranchfreeDivider<$t> {
                type Error = UnsupportedDivisor;

                #[inline(always)]
                fn try_from(d: $t) -> Result<Self, Self::Error> {
                    Self::[<try_ $t>](d).ok_or(UnsupportedDivisor)
                }
            }

            impl Denominator for $t {
                #[inline(always)]
                fn to_divider(self) -> Divider<Self> {
                    Divider::[<$t>](self)
                }

                #[inline(always)]
                fn to_branchfree_divider(self) -> BranchfreeDivider<Self> {
                    BranchfreeDivider::[<$t>](self)
                }

                #[inline(always)]
                fn try_to_branchfree_divider(self) -> Result<BranchfreeDivider<Self>, UnsupportedDivisor> {
                    BranchfreeDivider::[<try_ $t>](self).ok_or(UnsupportedDivisor)
                }

                const SHIFT_MASK: u8 = Divider::<$t>::SHIFT_MASK;
            }

            impl Divider<$t> {
                /// Create a new divider for the given divisor.
                #[inline(always)]
                pub const fn [<$t>](d: $t) -> Self {
                    Self::[<$t _internal>](d, false)
                }

                #[inline(always)]
                pub fn divide(self, x: $t) -> $t {
                    let multiplier = self.multiplier();
                    let shift = self.shift();

                    if multiplier == 0 {
                        return x >> shift;
                    }

                    let mut q = Self::mullhi(x, multiplier);

                    if (shift & ADD_MARKER) != 0 {
                        q = (x.wrapping_sub(q) >> 1).wrapping_add(q);
                    }

                    q >> (shift & Divider::<$t>::SHIFT_MASK)
                }

                #[inline(always)]
                const fn [<$t _internal>](d: $t, bf: bool) -> Self {
                    if d == 0 {
                        return Divider { multiplier: 0, shift: 0 };
                    }

                    if bf && d == 1 {
                        panic!("branchfree divider must be != 1");
                    }

                    let floor_log_2_d = Self::BITS - 1 - d.leading_zeros();

                    if d.is_power_of_two() {
                        return Divider {
                            multiplier: 0,
                            // We need to subtract 1 from the shift value in case of an unsigned
                            // branchfree divider because there is a hardcoded right shift by 1
                            // in its division algorithm.
                            shift: (floor_log_2_d - bf as u32) as u8,
                        };
                    }

                    let k = 1 << floor_log_2_d;
                    let (mut proposed_m, rem) = [<div_ $dt _ $t _to_ $t>](k, 0, d);

                    let e = d.wrapping_sub(rem);

                    let mut shift;

                    if !bf && e < k {
                        shift = floor_log_2_d as u8;
                    } else {
                        proposed_m = proposed_m.wrapping_add(proposed_m);
                        let rem2 = rem.wrapping_add(rem);

                        if rem2 >= d || rem2 < rem {
                            proposed_m = proposed_m.wrapping_add(1);
                        }

                        shift = floor_log_2_d as u8;

                        if !bf {
                            // instead of masking out the ADD_MARKER bit, we just don't set it
                            shift |= ADD_MARKER;
                        }
                    }

                    Divider { multiplier: proposed_m.wrapping_add(1), shift }
                }
            }
        )*}
    }
}

macro_rules! impl_signed_divider {
    ($($t:ty => $ut:ty => $udt:ty),*) => {
        paste::paste!{$(
            impl BranchfreeDivider<$t> {
                /// Create a new branchfree divider for the given divisor.
                ///
                /// Unlike the unsigned version, this does support a divisor of 1.
                #[inline(always)]
                pub const fn [<$t>](d: $t) -> Self {
                    BranchfreeDivider(Divider::[<$t _internal>](d, true))
                }

                #[inline(always)]
                pub fn divide(self, x: $t) -> $t {
                    let multiplier = self.multiplier();
                    let shift = self.shift();

                    let masked_shift = shift & Divider::<$t>::SHIFT_MASK;

                    let mut q = Divider::<$t>::mullhi(x, multiplier).wrapping_add(x);

                    let is_power_of_2: $ut = (multiplier == 0) as $ut;
                    let q_sign = q >> (Divider::<$t>::BITS - 1); // extends sign to fill bits

                    q = q.wrapping_add( q_sign & ((1 as $ut) << masked_shift).wrapping_sub(is_power_of_2) as $t );

                    let sign = ((shift as i8) >> 7) as $t; // take last bit as sign, to convert this to 0 or -1

                    ((q >> masked_shift) ^ sign) - sign
                }
            }

            impl From<$t> for BranchfreeDivider<$t> {
                #[inline(always)]
                fn from(d: $t) -> Self {
                    BranchfreeDivider::[<$t>](d)
                }
            }

            impl From<$t> for Divider<$t> {
                #[inline(always)]
                fn from(d: $t) -> Self {
                    Self::[<$t>](d)
                }
            }

            impl Denominator for $t {
                #[inline(always)]
                fn to_divider(self) -> Divider<Self> {
                    Divider::[<$t>](self)
                }

                #[inline(always)]
                fn to_branchfree_divider(self) -> BranchfreeDivider<Self> {
                    BranchfreeDivider::[<$t>](self)
                }

                #[inline(always)]
                fn try_to_branchfree_divider(self) -> Result<BranchfreeDivider<Self>, UnsupportedDivisor> {
                    Ok(BranchfreeDivider::[<$t>](self))
                }

                const SHIFT_MASK: u8 = Divider::<$t>::SHIFT_MASK;
            }

            impl Divider<$t> {
                /// Create a new divider for the given divisor.
                #[inline(always)]
                pub const fn [<$t>](d: $t) -> Self {
                    Self::[<$t _internal>](d, false)
                }

                pub fn divide(self, x: $t) -> $t {
                    let multiplier = self.multiplier();
                    let shift = self.shift();

                    let masked_shift = shift & Divider::<$t>::SHIFT_MASK;

                    // take last bit as sign, to convert this to 0 or -1
                    let sign = ((shift as i8) >> 7) as $t;

                    if multiplier == 0 {
                        let mask = ((1 as $ut) << masked_shift).wrapping_sub(1) as $t;
                        let uq = x.wrapping_add((x >> (Self::BITS - 1)) & mask);

                        return ((uq as $t >> masked_shift) ^ sign) - sign;
                    }

                    let mut uq = Self::mullhi(x, multiplier) as $ut;

                    if (shift & ADD_MARKER) != 0 {
                        uq = uq.wrapping_add(x as $ut ^ sign as $ut).wrapping_sub(sign as $ut);
                    }

                    let q = uq as $t >> masked_shift;

                    q + (q < 0) as $t
                }

                #[inline(always)]
                const fn [<$t _internal>](d: $t, bf: bool) -> Self {
                    if d == 0 {
                        return Divider { multiplier: 0, shift: 0 };
                    }

                    let abs_d = d.unsigned_abs();

                    let floor_log_2_d = Divider::<$ut>::BITS - 1 - abs_d.leading_zeros();

                    if abs_d.is_power_of_two() {
                        return Divider {
                            multiplier: 0,
                            shift: floor_log_2_d as u8 | if d < 0 { NEG_DIVISOR } else { 0 },
                        };
                    }

                    let (mut proposed_m, rem) = [<div_ $udt _ $ut _to_ $ut>](1 << (floor_log_2_d - 1), 0, abs_d);

                    let e = abs_d.wrapping_sub(rem);

                    let mut shift;

                    if !bf && e < (1 << floor_log_2_d) {
                        shift = (floor_log_2_d - 1) as u8;
                    } else {
                        proposed_m = proposed_m.wrapping_add(proposed_m);
                        let rem2 = rem.wrapping_add(rem);

                        if rem2 >= abs_d || rem2 < rem {
                            proposed_m = proposed_m.wrapping_add(1);
                        }

                        shift = floor_log_2_d as u8 | ADD_MARKER;
                    }

                    proposed_m = proposed_m.wrapping_add(1);

                    let mut multiplier = proposed_m as $t;

                    if d < 0 {
                        shift |= NEG_DIVISOR;

                        if !bf {
                            multiplier = -multiplier;
                        }
                    }

                    Divider { multiplier, shift }
                }
            }
        )*}
    }
}

macro_rules! impl_divider {
    ($($t:ty => $dt:ty),*) => {paste::paste! {$(
        impl Divider<$t> {
            #[inline(always)]
            pub(crate) const fn mullhi(x: $t, y: $t) -> $t {
                (((x as $dt) * (y as $dt)) >> <$t>::BITS) as $t
            }

            #[inline(always)]
            pub(crate) const fn new(m: $t, s: u8) -> Self {
                Divider { multiplier: m, shift: s }
            }
        }

        impl BranchfreeDivider<$t> {
            #[inline(always)]
            pub(crate) const fn new(m: $t, s: u8) -> Self {
                BranchfreeDivider(Divider { multiplier: m, shift: s })
            }
        }
    )*}};
}

impl_unsigned_divider! {
    u8 => u16,
    u16 => u32,
    u32 => u64,
    u64 => u128
}

impl_signed_divider! {
    i8 => u8 => u16,
    i16 => u16 => u32,
    i32 => u32 => u64,
    i64 => u64 => u128
}

impl_divider! {
    u8 => u16,
    u16 => u32,
    u32 => u64,
    u64 => u128,
    i8 => i16,
    i16 => i32,
    i32 => i64,
    i64 => i128
}
