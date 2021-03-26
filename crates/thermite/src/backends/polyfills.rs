#[inline(always)]
pub const fn _mm_shuffle(w: i32, z: i32, y: i32, x: i32) -> i32 {
    (w << 6) | (z << 4) | (y << 2) | x
}
