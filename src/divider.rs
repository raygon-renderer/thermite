// Derived from:
//
// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#![allow(unused)]

use std::ops::Deref;

macro_rules! decl_div_half {
    ($($t:ty => $dt:ty),*) => {
        paste::paste! {$(
            #[inline(always)]
            const fn [<div_ $dt _ $t _to_ $t>](u1: $t, u0: $t, v: $t) -> ($t, $t) {
                let v = v as $dt;
                let n = ((u1 as $dt) << (std::mem::size_of::<$t>() * 8)) | (u0 as $dt);
                let res = (n / v) as $t; // truncate
                let rem = n.wrapping_sub((res as $dt).wrapping_mul(v));
                (res, rem as $t)
            }
        )*}
    };
}

decl_div_half!(u64 => u128, u32 => u64, u16 => u32, u8 => u16);

#[repr(C, packed)]
pub struct Divider<T> {
    multiplier: T,
    shift: u8,
}

#[repr(transparent)]
#[derive(Copy, PartialEq)]
pub struct BranchfreeDivider<T>(Divider<T>);

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
    pub fn multiplier(&self) -> T {
        // with repr(C), self points to first value
        unsafe { (self as *const Self as *const T).read_unaligned() }
    }

    #[inline(always)]
    pub fn shift(&self) -> u8 {
        // shift has an alignment of 1 byte anyway, so it's fine to read normally
        self.shift
    }
}

pub(crate) const ADD_MARKER: u8 = 0x40;
pub(crate) const NEG_DIVISOR: u8 = 0x80;

macro_rules! impl_shift_mask {
    ($($ty:ty),*) => {$(
        impl Divider<$ty> {
            const BITS: u32 = 8 * std::mem::size_of::<$ty>() as u32;
            /// !log2(N::BITS)
            pub(crate) const SHIFT_MASK: u8 = !(<$ty>::MAX << Self::BITS.trailing_zeros()) as u8;
        }
    )*};
}

impl_shift_mask!(u8, u16, u32, u64);

macro_rules! impl_unsigned_divider {
    ($($t:ty => $dt:ty),*) => {
        paste::paste! {$(
            impl Divider<$t> {
                #[inline(always)]
                pub const fn [<$t>](d: $t) -> Self {
                    Self::[<$t _internal>](d, false)
                }

                #[inline]
                pub const fn [<$t _branchfree>](d: $t) -> BranchfreeDivider<$t> {
                    let mut divider = Self::[<$t _internal>](d, true);
                    divider.shift &= Self::SHIFT_MASK;
                    BranchfreeDivider(divider)
                }

                #[inline]
                const fn [<$t _internal>](d: $t, bf: bool) -> Self {
                    if d == 0 {
                        return Divider {
                            multiplier: 0,
                            shift: Self::BITS as u8, // shift to zero
                        }
                    }

                    let floor_log_2_d = Self::BITS - 1 - d.leading_zeros();

                    if d.is_power_of_two() {
                        Divider {
                            multiplier: 0,
                            // We need to subtract 1 from the shift value in case of an unsigned
                            // branchfree divider because there is a hardcoded right shift by 1
                            // in its division algorithm.
                            shift: (floor_log_2_d - bf as u32) as u8,
                        }
                    } else {
                        let k = 1 << floor_log_2_d;
                        let (mut proposed_m, rem) = [<div_ $dt _ $t _to_ $t>](k, 0, d);

                        let e = d.wrapping_sub(rem);

                        let shift = if !bf && e < k {
                            floor_log_2_d as u8
                        } else {
                            proposed_m = proposed_m.wrapping_add(proposed_m);
                            let rem2 = rem.wrapping_add(rem);

                            if rem2 >= d || rem2 < rem {
                                proposed_m = proposed_m.wrapping_add(1);
                            }

                            floor_log_2_d as u8 | ADD_MARKER
                        };

                        Divider {
                            multiplier: proposed_m.wrapping_add(1),
                            shift,
                        }
                    }
                }
            }
        )*}
    }
}

macro_rules! impl_signed_divider {
    ($($t:ty => $ut:ty => $udt:ty),*) => {
        paste::paste!{$(
            impl Divider<$t> {
                #[inline(always)]
                const fn [<$t>](d: $t) -> Self {
                    Self::[<$t _internal>](d, false)
                }

                #[inline]
                const fn [<$t _branchfree>](d: $t) -> BranchfreeDivider<$t> {
                    let mut divider = Self::[<$t _internal>](d, true);
                    divider.shift &= Divider::<$ut>::SHIFT_MASK;
                    BranchfreeDivider(divider)
                }

                #[inline]
                const fn [<$t _internal>](d: $t, bf: bool) -> Self {
                    if d == 0 {
                        return Divider {
                            multiplier: 0,
                            shift: Divider::<$ut>::BITS as u8, // shift to zero
                        };
                    }

                    let abs_d = d.abs() as $ut;

                    let floor_log_2_d = Divider::<$ut>::BITS - 1 - d.leading_zeros();

                    if abs_d.is_power_of_two() {
                        Divider {
                            multiplier: 0,
                            shift: floor_log_2_d as u8 | if d < 0 { NEG_DIVISOR } else { 0 },
                        }
                    } else {
                        let (mut proposed_m, rem) = [<div_ $udt _ $ut _to_ $ut>](1 << (floor_log_2_d - 1), 0, abs_d);

                        let e = abs_d.wrapping_sub(rem);

                        let mut shift = if !bf && e < (1 << floor_log_2_d) {
                            (floor_log_2_d - 1) as u8
                        } else {
                            proposed_m = proposed_m.wrapping_add(proposed_m);
                            let rem2 = rem.wrapping_add(rem);

                            if rem2 >= abs_d || rem2 < rem {
                                proposed_m = proposed_m.wrapping_add(1);
                            }

                            floor_log_2_d as u8 | ADD_MARKER
                        };

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
            }
        )*}
    }
}

impl_unsigned_divider!(u8 => u16, u16 => u32, u32 => u64, u64 => u128);

impl_signed_divider! {
    i8 => u8 => u16,
    i16 => u16 => u32,
    i32 => u32 => u64,
    i64 => u64 => u128
}
