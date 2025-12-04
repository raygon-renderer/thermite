use super::*;

#[inline(always)]
fn flip(v: v128) -> v128 {
    v128_xor(v, i64x2_splat(i64::MIN)) // 0x8000000000000000
}

/// u64x2 Greater Than (a > b)
#[inline(always)]
pub fn u64x2_gt(a: v128, b: v128) -> v128 {
    i64x2_gt(flip(a), flip(b))
}

/// u64x2 Greater Than or Equal (a >= b)
#[inline(always)]
pub fn u64x2_ge(a: v128, b: v128) -> v128 {
    i64x2_ge(flip(a), flip(b))
}

/// u64x2 Less Than (a < b)
#[inline(always)]
pub fn u64x2_lt(a: v128, b: v128) -> v128 {
    i64x2_lt(flip(a), flip(b))
}

/// u64x2 Less Than or Equal (a <= b)
#[inline(always)]
pub fn u64x2_le(a: v128, b: v128) -> v128 {
    i64x2_le(flip(a), flip(b))
}

#[inline(always)]
pub fn u64x2_max(a: v128, b: v128) -> v128 {
    u8x16_relaxed_laneselect(a, b, u64x2_gt(a, b))
}

#[inline(always)]
pub fn u64x2_min(a: v128, b: v128) -> v128 {
    u8x16_relaxed_laneselect(a, b, u64x2_lt(a, b))
}

#[inline(always)]
pub fn i64x2_max(a: v128, b: v128) -> v128 {
    u8x16_relaxed_laneselect(a, b, i64x2_gt(a, b))
}

#[inline(always)]
pub fn i64x2_min(a: v128, b: v128) -> v128 {
    u8x16_relaxed_laneselect(a, b, i64x2_lt(a, b))
}
