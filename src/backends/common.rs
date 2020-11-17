pub use divider::*;
mod divider {
    // libdivide.h - Optimized integer division
    // https://libdivide.com
    //
    // Copyright (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
    // Copyright (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>

    #[inline(always)]
    pub const fn div_128_64_to_64(u1: u64, u0: u64, v: u64) -> (u64, u64) {
        let v = v as u128;
        let n = ((u1 as u128) << 64) | (u0 as u128);
        let res = (n / v) as u64; // truncate
        let rem = n.wrapping_sub((res as u128).wrapping_mul(v));
        (res, rem as u64)
    }

    #[inline(always)]
    pub const fn div_64_32_to_32(u1: u32, u0: u32, v: u32) -> (u32, u32) {
        let v = v as u64;
        let n = ((u1 as u64) << 32) | (u0 as u64);
        let res = (n / v) as u32;
        let rem = n.wrapping_sub((res as u64).wrapping_mul(v));
        (res, rem as u32)
    }

    #[inline(always)]
    pub const fn gen_u32(d: u32) -> (u32, u8) {
        if d == 0 {
            return (0, 32); // shift to zero
        }

        let floor_log_2_d = 31 - d.leading_zeros();

        if (d & (d - 1)) == 0 {
            (0, floor_log_2_d as u8)
        } else {
            let k = 1 << floor_log_2_d;
            let (mut proposed_m, rem) = div_64_32_to_32(k, 0, d);

            let e = d.wrapping_sub(rem);

            let more = if e < k {
                floor_log_2_d as u8
            } else {
                proposed_m = proposed_m.wrapping_add(proposed_m);
                let rem2 = rem.wrapping_add(rem);

                if rem2 >= d || rem2 < rem {
                    proposed_m = proposed_m.wrapping_add(1);
                }

                floor_log_2_d as u8 | 0x40
            };

            (proposed_m.wrapping_add(1), more)
        }
    }

    #[inline(always)]
    pub const fn gen_u64(d: u64) -> (u64, u8) {
        if d == 0 {
            return (0, 64); // shift to zero
        }

        let floor_log_2_d = 63 - d.leading_zeros();

        if (d & (d - 1)) == 0 {
            (0, floor_log_2_d as u8)
        } else {
            let k = 1 << floor_log_2_d;
            let (mut proposed_m, rem) = div_128_64_to_64(k, 0, d);

            let e = d.wrapping_sub(rem);

            let more = if e < k {
                floor_log_2_d as u8
            } else {
                proposed_m = proposed_m.wrapping_add(proposed_m);
                let rem2 = rem.wrapping_add(rem);

                if rem2 >= d || rem2 < rem {
                    proposed_m = proposed_m.wrapping_add(1);
                }

                floor_log_2_d as u8 | 0x40
            };

            (proposed_m.wrapping_add(1), more)
        }
    }
}
