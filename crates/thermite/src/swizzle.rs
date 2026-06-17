use generic_array::{ArrayLength, GenericArray};

use crate::{
    Vector,
    register::{Register, SwizzleRegister},
};

#[doc(hidden)]
pub use crate::register::SwizzleIndices;

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

/// Trait for swizzling and permuting vector types. Use the [`swizzle!`](crate::swizzle!) macro for convenient usage.
pub trait Swizzle<N: ArrayLength>: Sized {
    /// Swizzle lanes from two vectors according to the given indices.
    fn swizzle(self, other: Self, indices: GenericArray<u32, N>) -> Self;

    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<N>>(self, other: Self) -> Self {
        Self::swizzle(self, other, I::INDICES)
    }

    /// Permute lanes from a single vector according to the given indices.
    fn permute(self, indices: GenericArray<u32, N>) -> Self;

    #[inline(always)]
    fn permute_const<I: SwizzleIndices<N>>(self) -> Self {
        Self::permute(self, I::INDICES)
    }
}

impl<R: Register> Swizzle<R::Lanes> for Vector<R>
where
    R: SwizzleRegister,
{
    #[inline(always)]
    fn swizzle(self, other: Self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Vector(R::swizzle(self.0, other.0, indices))
    }

    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<R::Lanes>>(self, other: Self) -> Self {
        Vector(R::swizzle_const::<I>(self.0, other.0))
    }

    #[inline(always)]
    fn permute(self, indices: GenericArray<u32, R::Lanes>) -> Self {
        Vector(R::permutev(self.0, indices))
    }

    #[inline(always)]
    fn permute_const<I: SwizzleIndices<R::Lanes>>(self) -> Self {
        Vector(R::permutev_const::<I>(self.0))
    }
}

/// Swizzle lanes one or two vectors according to the given indices.
///
/// If compiling with optimizations enabled, this macro will usually generate
/// efficient swizzle/permutation instructions.
///
/// This works for any type that implements the [`Swizzle`] trait, such as [`Vector`].
///
/// The indices can either be given as a constant literal array, or an expression that evaluates to
/// a `GenericArray<u32, R::Lanes>` for dynamic shuffling. Either way the number of indices must
/// equal the lane count (this is checked at compile time).
///
/// # Index range
///
/// Each index selects a source lane: `0..LANES` for the single-vector form, and `0..2*LANES` for
/// the two-vector form (the second vector's lanes follow the first's). **Indices outside that range
/// are undefined behavior** - the resulting lane is unspecified and differs by backend (some mask
/// the index to the valid range, some do not). This is not checked, since the macro is intended for
/// fixed, known-good index sets; keep every index in range.
#[macro_export]
macro_rules! swizzle {
    ($a:expr, $b:expr, [$($i:expr),* $(,)?]) => {{
        #[inline(always)]
        fn __do_swizzle2<N: $crate::generic_array::ArrayLength, S: $crate::swizzle::Swizzle<N>>(a: S, b: S) -> S {
            use $crate::{swizzle::{Swizzle, SwizzleIndices}, generic_array::{GenericArray, typenum::Unsigned}};

            struct Indices<N: $crate::generic_array::ArrayLength>(core::marker::PhantomData<N>);

            impl<N: $crate::generic_array::ArrayLength> SwizzleIndices<N> for Indices<N> {
                const INDICES: GenericArray<u32, N> = {
                    let idxs = [$($i),*];
                    assert!(N::USIZE == idxs.len(), "Swizzle mask must be the same length of the vector");
                    unsafe { $crate::generic_array::const_transmute::<_, GenericArray<u32, N>>(idxs) }
                };
            }

            a.swizzle_const::<Indices::<N>>(b)
        }

        __do_swizzle2($a, $b)
    }};

    ($a:expr, [$($i:expr),* $(,)?]) => {{
        #[inline(always)]
        fn __do_swizzle1<N: $crate::generic_array::ArrayLength, S: $crate::swizzle::Swizzle<N>>(a: S) -> S {
            use $crate::{swizzle::{Swizzle, SwizzleIndices}, generic_array::{GenericArray, typenum::Unsigned}};

            struct Indices<N: $crate::generic_array::ArrayLength>(core::marker::PhantomData<N>);

            impl<N: $crate::generic_array::ArrayLength> SwizzleIndices<N> for Indices<N> {
                const INDICES: GenericArray<u32, N> = const {
                    let idxs = [$($i),*];
                    assert!(N::USIZE == idxs.len(), "Swizzle mask must be the same length of the vector");
                    unsafe { $crate::generic_array::const_transmute::<_, $crate::generic_array::GenericArray<u32, N>>(idxs) }
                };
            }

            a.permute_const::<Indices::<N>>()
        }

        __do_swizzle1($a)
    }};

    ($a:expr, $b:expr, $idxs:expr) => { $crate::swizzle::Swizzle::swizzle($a, $b, $idxs) };
    ($a:expr, $idxs:expr) => { $crate::swizzle::Swizzle::permute($a, $idxs) };
}
