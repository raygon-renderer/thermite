use super::register::{BinaryRegisterOps, FloatRegisterOps};

#[inline(always)]
pub const fn _mm_shuffle(w: i32, z: i32, y: i32, x: i32) -> i32 {
    (w << 6) | (z << 4) | (y << 2) | x
}

// https://stackoverflow.com/a/26342944/2083075 + Bernard's comment
#[inline(always)]
pub unsafe fn float_rem<R>(lhs: R::Storage, rhs: R::Storage) -> R::Storage
where
    R: FloatRegisterOps + BinaryRegisterOps,
{
    R::nmul_add(R::trunc(R::div(lhs, rhs)), rhs, lhs)
}
