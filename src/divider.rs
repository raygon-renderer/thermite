// Derived from:
//
// libdivide.h - Optimized integer division
// https://libdivide.com
//
// Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

#![allow(unused)]

#[inline(always)]
const fn div_128_64_to_64(u1: u64, u0: u64, v: u64) -> (u64, u64) {
    let v = v as u128;
    let n = ((u1 as u128) << 64) | (u0 as u128);
    let res = (n / v) as u64; // truncate
    let rem = n.wrapping_sub((res as u128).wrapping_mul(v));
    (res, rem as u64)
}

#[inline(always)]
const fn div_64_32_to_32(u1: u32, u0: u32, v: u32) -> (u32, u32) {
    let v = v as u64;
    let n = ((u1 as u64) << 32) | (u0 as u64);
    let res = (n / v) as u32;
    let rem = n.wrapping_sub((res as u64).wrapping_mul(v));
    (res, rem as u32)
}

#[repr(C, packed)]
pub struct Divider<T> {
    multiplier: T,
    shift: u8,
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

impl Divider<u32> {
    #[inline(always)]
    pub const fn u32(d: u32) -> Self {
        if d == 0 {
            return Divider {
                multiplier: 0,
                shift: 32, // shift to zero
            };
        }

        let floor_log_2_d = 31 - d.leading_zeros();

        if (d & (d - 1)) == 0 {
            Divider {
                multiplier: 0,
                shift: floor_log_2_d as u8,
            }
        } else {
            let k = 1 << floor_log_2_d;
            let (mut proposed_m, rem) = div_64_32_to_32(k, 0, d);

            let e = d.wrapping_sub(rem);

            let shift = if e < k {
                floor_log_2_d as u8
            } else {
                proposed_m = proposed_m.wrapping_add(proposed_m);
                let rem2 = rem.wrapping_add(rem);

                if rem2 >= d || rem2 < rem {
                    proposed_m = proposed_m.wrapping_add(1);
                }

                floor_log_2_d as u8 | 0x40
            };

            Divider {
                multiplier: proposed_m.wrapping_add(1),
                shift,
            }
        }
    }
}

impl Divider<u64> {
    #[inline(always)]
    pub const fn u64(d: u64) -> Self {
        if d == 0 {
            return Divider {
                multiplier: 0,
                shift: 64, // shift to zero
            };
        }

        let floor_log_2_d = 63 - d.leading_zeros();

        if (d & (d - 1)) == 0 {
            Divider {
                multiplier: 0,
                shift: floor_log_2_d as u8,
            }
        } else {
            let k = 1 << floor_log_2_d;
            let (mut proposed_m, rem) = div_128_64_to_64(k, 0, d);

            let e = d.wrapping_sub(rem);

            let shift = if e < k {
                floor_log_2_d as u8
            } else {
                proposed_m = proposed_m.wrapping_add(proposed_m);
                let rem2 = rem.wrapping_add(rem);

                if rem2 >= d || rem2 < rem {
                    proposed_m = proposed_m.wrapping_add(1);
                }

                floor_log_2_d as u8 | 0x40
            };

            Divider {
                multiplier: proposed_m.wrapping_add(1),
                shift,
            }
        }
    }
}

impl Divider<i32> {
    pub fn i32(d: i32) -> Self {
        unimplemented!()
    }
}

impl Divider<i64> {
    pub fn i64(d: i64) -> Self {
        unimplemented!()
    }
}
