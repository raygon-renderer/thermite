#[inline(always)]
pub const fn _mm_shuffle(w: i32, z: i32, y: i32, x: i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
}

#[inline(always)]
pub const fn _mm256_shuffle(d: i32, c: i32, b: i32, a: i32, w: i32, z: i32, y: i32, x: i32) -> i32 {
    _mm_shuffle(w, z, y, x) | (_mm_shuffle(d - 4, c - 4, b - 4, a - 4) << 8)
}
