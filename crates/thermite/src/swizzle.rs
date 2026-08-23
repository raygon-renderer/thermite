use core::marker::PhantomData;

use generic_array::{ArrayLength, GenericArray};

use crate::{Vector, register::Register};

#[doc(hidden)]
pub use crate::register::SwizzleIndices;

/// Compile-time [`SwizzleIndices`] for a two-register element align: lane `i`
/// maps to index `i + OFFSET` of the concatenation `[a, b]` (`a`'s lanes first,
/// then `b`'s).
///
/// Fed to [`swizzle_const`](Swizzle::swizzle_const) this produces a
/// `palignr`-style sliding window - `Self::Lanes` lanes starting `OFFSET` lanes
/// into `a` and spilling into `b`. `OFFSET == 0` yields `a`; `OFFSET == LANES`
/// yields `b`.
pub(crate) struct AlignIndices<const OFFSET: usize, N>(PhantomData<N>);

impl<const OFFSET: usize, N: ArrayLength> SwizzleIndices<N> for AlignIndices<OFFSET, N> {
    const INDICES: GenericArray<u32, N> = const {
        // `GenericArray<u32, N>` has no const literal constructor for a generic
        // `N`, so zero-initialize (an all-zero `u32` array is valid) and fill it
        // in place. Writing through a raw pointer avoids needing a const `IndexMut`
        // or `DerefMut` on `GenericArray`.
        let mut idxs: GenericArray<u32, N> = unsafe { core::mem::zeroed() };
        let ptr = &mut idxs as *mut GenericArray<u32, N> as *mut u32;
        let mut i = 0;
        while i < N::USIZE {
            unsafe { *ptr.add(i) = (i + OFFSET) as u32 };
            i += 1;
        }
        idxs
    };
}

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

/// Compile-time lane swizzles: the `_const` surface over a [`SwizzleIndices`]
/// implementor, where backends pattern-match immediate-encoded shuffles.
/// Use the [`swizzle!`](crate::swizzle!) macro for convenient usage.
///
/// This trait is exclusively the const-index entry point. Dynamic (live index
/// vector) permutes are ordinary vector methods:
/// [`GenericVector::permutev`](crate::vector::GenericVector::permutev) and
/// [`GenericVector::swizzle`](crate::vector::GenericVector::swizzle), taking
/// the vector's own `Unsigned` type as indices.
///
/// # Index range
///
/// `permutev_const` indices select from `0..LANES`, and `swizzle_const` indices
/// from `0..2*LANES` (the second vector's lanes follow the first's). An
/// out-of-range index yields an UNSPECIFIED value in that lane, memory-safe and
/// never UB, but backend-dependent. No release-mode range checks are
/// performed.
pub trait Swizzle<N: ArrayLength>: Sized {
    /// Swizzle lanes from two vectors according to compile-time indices.
    fn swizzle_const<I: SwizzleIndices<N>>(self, other: Self) -> Self;

    /// Permute lanes of a single vector according to compile-time indices.
    fn permutev_const<I: SwizzleIndices<N>>(self) -> Self;
}

impl<R: Register> Swizzle<R::Lanes> for Vector<R> {
    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<R::Lanes>>(self, other: Self) -> Self {
        Vector(R::swizzle_const::<I>(self.0, other.0))
    }

    #[inline(always)]
    fn permutev_const<I: SwizzleIndices<R::Lanes>>(self) -> Self {
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
/// The indices can either be given as a constant literal array (lowered through [`Swizzle`]'s
/// `_const` machinery), or an expression that evaluates to the type's live index vector
/// (its [`GenericVector::Unsigned`](crate::vector::GenericVector::Unsigned) type) for dynamic
/// shuffling. Either way the number of indices must equal the lane count (checked at compile
/// time for the literal form).
///
/// # Index range
///
/// Each index selects a source lane: `0..LANES` for the single-vector form, and `0..2*LANES` for
/// the two-vector form (the second vector's lanes follow the first's). **An index outside that
/// range yields an UNSPECIFIED value in that lane**, memory-safe and never UB, but backend-dependent
/// (some wrap, some zero, some clamp). This is not checked, so keep every index in range.
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

            a.permutev_const::<Indices::<N>>()
        }

        __do_swizzle1($a)
    }};

    ($a:expr, $b:expr, $idxs:expr) => { $crate::vector::GenericVector::swizzle($a, $b, $idxs) };
    ($a:expr, $idxs:expr) => { $crate::vector::GenericVector::permutev($a, $idxs) };
}
