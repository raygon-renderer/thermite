#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// NOTE: Specifically using x86-v2 here so that i32x16 = ArrayRegister<i32x4, 4>
use generic_array::{GenericArray, sequence::GenericSequence};
use thermite::backend::x86_v2::prelude::*;

#[test]
fn test_array_register_interleave() {
    // A = [0, 1, 2, ..., 15]
    let a_arr = GenericArray::generate(|i| i as i32);
    // B = [100, 101, 102, ..., 115]
    let b_arr = GenericArray::generate(|i| (i + 100) as i32);

    let a = i32x16::from_array(a_arr);
    let b = i32x16::from_array(b_arr);

    // 1. Test Interleave
    let (lo, hi) = i32x16::interleave(a, b);

    let mut lo_out = [0i32; 16];
    let mut hi_out = [0i32; 16];

    unsafe {
        i32x16::store_unaligned(lo, lo_out.as_mut_ptr());
        i32x16::store_unaligned(hi, hi_out.as_mut_ptr());
    }

    let expected_lo = [0, 100, 1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106, 7, 107];
    let expected_hi = [8, 108, 9, 109, 10, 110, 11, 111, 12, 112, 13, 113, 14, 114, 15, 115];

    assert_eq!(lo_out, expected_lo, "Interleave LO did not match expected layout");
    assert_eq!(hi_out, expected_hi, "Interleave HI did not match expected layout");

    // 2. Test Deinterleave (which should perfectly invert the process)
    let (de_a, de_b) = i32x16::deinterleave(lo, hi);

    let mut a_out = [0i32; 16];
    let mut b_out = [0i32; 16];

    unsafe {
        i32x16::store_unaligned(de_a, a_out.as_mut_ptr());
        i32x16::store_unaligned(de_b, b_out.as_mut_ptr());
    }

    assert_eq!(a_out, a_arr.as_slice(), "Deinterleave A did not match original input A");
    assert_eq!(b_out, b_arr.as_slice(), "Deinterleave B did not match original input B");
}
