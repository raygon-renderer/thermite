/// Generates the imm8 constants for shuffling together two vectors
pub const fn double_swizzle<const N: usize>(indices: [u32; N]) -> (i32, i32, i32) {
    // not supported for vectors larger than 32 lanes,
    // elsewhere will handle fallbacks
    if N >= 16 {
        return (-1, 0, 0);
    }

    let mut imm_shuffle_a = 0;
    let mut imm_shuffle_b = 0;
    let mut imm_blend = 0;

    let mut i = 0;
    let n = N as i32;
    let l = N.ilog2() as i32;

    while i < N {
        let k = i as i32;
        let src_idx = indices[i] as i32;

        if src_idx < n {
            // pick idx from a
            imm_shuffle_a |= src_idx << (k * l);
        } else if i < (N * 2) {
            // pick idx from b
            imm_shuffle_b |= (src_idx - n) << (k * l);

            // make sure we blend from b
            imm_blend |= 1 << k;
        } else {
            panic!("Invalid vector shuffle index.");
        }

        i += 1;
    }

    (imm_shuffle_a, imm_shuffle_b, imm_blend)
}
