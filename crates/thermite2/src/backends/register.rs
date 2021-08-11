pub unsafe trait Register {
    type Element: Clone + Copy;
    type Storage: Clone + Copy;

    unsafe fn set1(x: Self::Element) -> Self::Storage;
}

pub unsafe trait SimpleRegister: Register {
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage;
}

pub unsafe trait FixedRegister<const N: usize>: Register {
    unsafe fn setr(values: [Self::Element; N]) -> Self::Storage;
}

pub unsafe trait UnaryRegisterOps: Register {
    unsafe fn bit_not(r: Self::Storage) -> Self::Storage;
}

pub unsafe trait BinaryRegisterOps: UnaryRegisterOps {
    unsafe fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;

    #[inline(always)]
    unsafe fn and_not(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self::bitand(Self::bit_not(lhs), rhs)
    }

    unsafe fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
    unsafe fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage;
}

pub unsafe trait FloatRegisterOps: SignedRegisterOps + BinaryRegisterOps {
    unsafe fn round(x: Self::Storage) -> Self::Storage;
    unsafe fn ceil(x: Self::Storage) -> Self::Storage;
    unsafe fn floor(x: Self::Storage) -> Self::Storage;
    unsafe fn trunc(x: Self::Storage) -> Self::Storage;

    #[inline(always)]
    unsafe fn fract(x: Self::Storage) -> Self::Storage {
        Self::sub(x, Self::trunc(x))
    }

    unsafe fn sqrt(x: Self::Storage) -> Self::Storage;
    unsafe fn rsqrt(x: Self::Storage) -> Self::Storage;
    unsafe fn rcp(x: Self::Storage) -> Self::Storage;

    unsafe fn mul_add(x: Self::Storage, m: Self::Storage, a: Self::Storage) -> Self::Storage;
    unsafe fn mul_sub(x: Self::Storage, m: Self::Storage, a: Self::Storage) -> Self::Storage;
    unsafe fn nmul_add(x: Self::Storage, m: Self::Storage, a: Self::Storage) -> Self::Storage;
    unsafe fn nmul_sub(x: Self::Storage, m: Self::Storage, a: Self::Storage) -> Self::Storage;
}

pub unsafe trait SignedRegisterOps: Register {
    unsafe fn neg(x: Self::Storage) -> Self::Storage;
    unsafe fn abs(x: Self::Storage) -> Self::Storage;
}

pub unsafe trait MaskRegisterOps: BinaryRegisterOps {
    #[inline(always)]
    unsafe fn blendv(mask: Self::Storage, t: Self::Storage, f: Self::Storage) -> Self::Storage {
        Self::bitor(Self::bitand(mask, t), Self::and_not(mask, f))
    }

    unsafe fn all(mask: Self::Storage) -> bool;
    unsafe fn any(mask: Self::Storage) -> bool;

    #[inline(always)]
    unsafe fn none(mask: Self::Storage) -> bool {
        !Self::any(mask)
    }
}
