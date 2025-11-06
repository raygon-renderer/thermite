use generic_array::GenericArray;

use crate::{
    Vector,
    mask::Mask,
    register::{MaskRegister, Register, SwizzleRegister},
};

/*
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
*/

pub trait Swizzle<R: Register> {
    fn swizzle(self, other: Self, indices: GenericArray<u32, R::Lanes>) -> Self;
    fn permutev(self, indices: GenericArray<u32, R::Lanes>) -> Self;
}

impl<R: Register> Swizzle<R> for Vector<R>
where
    R: SwizzleRegister,
{
    #[inline(always)]
    fn swizzle(self, other: Self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Vector(R::swizzle(self.0, other.0, indices))
    }

    #[inline(always)]
    fn permutev(self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Vector(R::permutev(self.0, indices))
    }
}

impl<R: MaskRegister> Swizzle<R> for Mask<R>
where
    R: SwizzleRegister,
{
    #[inline(always)]
    fn swizzle(self, other: Self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Mask(R::swizzle(self.0, other.0, indices))
    }

    #[inline(always)]
    fn permutev(self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Mask(R::permutev(self.0, indices))
    }
}

/// Swizzle lanes one or two vectors according to the given indices.
///
/// If compiling with optimizations enabled, this macro will usually generate
/// efficient swizzle/permutation instructions.
///
/// This works for any type that implements the [`Swizzle`] trait, such as [`Vector`] and [`Mask`].
///
/// The indices can either be given as a constant literal array, or an expression that evaluates to
/// a `GenericArray<u32, R::Lanes>` for dynamic shuffling.
#[macro_export]
macro_rules! swizzle {
    ($a:expr, $b:expr, [$($i:literal),* $(,)?]) => {{
        #[inline(always)]
        fn __do_swizzle2<R: $crate::register::SwizzleRegister, S: $crate::swizzle::Swizzle<R>>(a: S, b: S) -> S {
            use $crate::{swizzle::Swizzle, generic_array::typenum::Unsigned};
            a.swizzle(b, const {
                let idxs = [$($i),*];
                assert!(R::Lanes::USIZE == idxs.len(), "Swizzle mask must be the same length of the vector");
                unsafe { $crate::generic_array::const_transmute::<_, $crate::generic_array::GenericArray<u32, R::Lanes>>(idxs) }
            })
        }

        __do_swizzle2($a, $b)
    }};

    ($a:expr, [$($i:literal),* $(,)?]) => {{
        #[inline(always)]
        fn __do_swizzle1<R: $crate::register::SwizzleRegister, S: $crate::swizzle::Swizzle<R>>(a: S) -> S {
            use $crate::{swizzle::Swizzle, generic_array::typenum::Unsigned};
            a.permutev(const {
                let idxs = [$($i),*];
                assert!(R::Lanes::USIZE == idxs.len(), "Swizzle mask must be the same length of the vector");
                unsafe { $crate::generic_array::const_transmute::<_, $crate::generic_array::GenericArray<u32, R::Lanes>>(idxs) }
            })
        }

        __do_swizzle1($a)
    }};

    ($a:expr, $b:expr, $idxs:expr) => { $crate::swizzle::Swizzle::swizzle($a, $b, $idxs) };
    ($a:expr, $idxs:expr) => { $crate::swizzle::Swizzle::permutev($a, $idxs) };
}
