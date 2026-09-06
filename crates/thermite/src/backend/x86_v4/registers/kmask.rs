//! Native AVX-512 opmask (`k`-register) mask types.
//!
//! One type per lane count, storage = the matching `__mmaskN` integer typedef
//! (`u8` for 2/4/8 lanes, then `u16`/`u32`/`u64`). All six exist
//! unconditionally: 32/64-lane opmasks need AVX512BW, which is part of this
//! backend's floor.
//!
//! Two deliberate design points:
//!
//! - **Every operation is a plain Rust integer op, not a `_kand_mask16`-style
//!   intrinsic.** The mask intrinsics are thin wrappers, and LLVM allocates
//!   masks to `k`-registers or GPRs as it sees fit. Plain ops give it the
//!   freedom and cost nothing. Nothing in this file emits an AVX-512
//!   instruction, which also means these types are fully testable on any
//!   host. Revisit only if inspected codegen disappoints (`sde-mix`).
//!
//! - **Invariant: storage bits at or above the lane count are ZERO.** The
//!   sub-8-lane masks share `u8` storage (`__mmask8`), so complement-shaped
//!   ops (`not`, and anything built on it) re-mask with [`VALID`] -- the same
//!   discipline `ReducedRegister` uses for its dead upper lanes. Compares and
//!   the masked-off `k` results of hardware instructions uphold it for free.
//!   Every constructor here (`from_native_bitmask`, `set`, `new_mask`)
//!   enforces it. Breaking it corrupts `all`/`count_set`/`native_bitmask` and
//!   every consumer that trusts a packed bitmask.
//!
//! [`VALID`]: MaskRegister::TRUTHY

use generic_array::GenericArray;
use generic_array::typenum::{U2, U4, U8, U16, U32, U64, Unsigned};

use crate::register::{
    BitwiseRegister, CastMaskRegister, ConcatRegister, CoreRegister, ExtendRegister, InterleaveRegister, MaskRegister,
    Storage, ZeroUpper,
};

use super::super::X86V4Default;

/// Bitmask of the low `n` bits (`n >= 64` yields all ones).
#[inline(always)]
const fn low_bits(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Spread the low 32 bits apart: bit `i` moves to bit `2i`, zeroes between.
///
/// BMI2 `pdep` on x86-64: masks live in GPRs, which is exactly where `pdep`
/// operates, one instruction instead of the five-step shift-mask cascade.
/// (The dev-guide "don't reach for PDEP" caveat is about _vector-lane_ Morton
/// codes, where no packed form exists, and it does not apply here.) Every
/// AVX-512 part has BMI2, and every AVX-512-capable AMD part is Zen 4+,
/// where `pdep`/`pext` are 3 cycles. The microcoded-PDEP trap is Zen 1/2
/// only, which never had AVX-512. See `Avx512Features::BMI2`.
#[inline(always)]
fn spread_bits(x: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: BMI2 is asserted for the x86-v4 backend (`Avx512Features::BMI2`,
    // present on every AVX-512 part).
    unsafe {
        core::arch::x86_64::_pdep_u64(x, 0x5555_5555_5555_5555)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // 32-bit x86 has no `_pdep_u64`, so the classic cascade.
        let mut x = x & 0x0000_0000_FFFF_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }
}

/// Inverse of [`spread_bits`]: keep even-position bits, bit `2i` moves to bit
/// `i`. BMI2 `pext`. See [`spread_bits`] for the reasoning.
#[inline(always)]
fn squash_bits(x: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as in `spread_bits`.
    unsafe {
        core::arch::x86_64::_pext_u64(x, 0x5555_5555_5555_5555)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut x = x & 0x5555_5555_5555_5555;
        x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
        x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
        x
    }
}

macro_rules! decl_kmask {
    ($($(#[$doc:meta])* $name:ident: $lanes:ty => $storage:ty;)*) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            /// The valid-lane window: low `LANES` bits set. Also
            /// [`MaskRegister::TRUTHY`]. Storage bits outside it are always
            /// zero (the module invariant).
            pub const VALID: $storage = low_bits(<$lanes as Unsigned>::USIZE) as $storage;
        }

        impl crate::simd::HasIsa for $name {
            type Native = X86V4Default;
        }

        #[thermite_macros::inline_always]
        impl CoreRegister for $name {
            type Lanes = $lanes;
            type Mask = Self;
            type Storage = $storage;

            const IS_EMULATED: bool = false;
            const EMPTY: Storage<Self> = 0;
            // A mask's mask is itself: identical storage, trivially convertible.
            const HAS_EQUAL_SIZE_MASK: bool = true;

            fn from_mask(mask: Storage<Self>) -> Storage<Self> {
                mask
            }

            fn blendv(mask: Storage<Self>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
                // `!mask` has dirty high bits, but `on_false` is invariant-clean,
                // so the AND scrubs them.
                (on_true & mask) | (on_false & !mask)
            }

            fn zz(mask: Storage<Self>, value: Storage<Self>) -> Storage<Self> {
                value & mask
            }

            fn nz(mask: Storage<Self>, value: Storage<Self>) -> Storage<Self> {
                value & !mask
            }

            fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                value & (low_bits(Z::N) as $storage)
            }
        }

        #[rustfmt::skip]
        #[thermite_macros::inline_always]
        impl BitwiseRegister for $name {
            fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs ^ rhs }
            fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs & rhs }
            fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs | rhs }

            // The one complement-shaped primitive: re-mask to keep the
            // high-bit invariant. `bitandnot`/`ternlog`/`bilog` defaults
            // compose `not` with AND/OR against clean operands, so they
            // inherit cleanliness from this single point.
            fn not(value: Storage<Self>) -> Storage<Self> { !value & Self::VALID }
        }

        #[thermite_macros::inline_always]
        impl InterleaveRegister for $name {
            fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                let (a, b) = (a as u64, b as u64);
                let half = <$lanes as Unsigned>::USIZE / 2;

                let lo = spread_bits(a & low_bits(half)) | (spread_bits(b & low_bits(half)) << 1);
                let hi = spread_bits(a >> half) | (spread_bits(b >> half) << 1);

                (lo as $storage, hi as $storage)
            }

            fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                let (a, b) = (a as u64, b as u64);
                let half = <$lanes as Unsigned>::USIZE / 2;

                let evens = squash_bits(a) | (squash_bits(b) << half);
                let odds = squash_bits(a >> 1) | (squash_bits(b >> 1) << half);

                (evens as $storage, odds as $storage)
            }
        }

        #[rustfmt::skip]
        #[thermite_macros::inline_always]
        impl MaskRegister for $name {
            const TRUTHY: Storage<Self> = Self::VALID;
            const FALSY: Storage<Self> = 0;

            fn set(mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
                let bit = (1 as $storage) << lane;
                if value { mask | bit } else { mask & !bit }
            }

            fn test(mask: Storage<Self>, lane: usize) -> bool {
                (mask >> lane) & 1 != 0
            }

            fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
                let mut bits: $storage = 0;
                let mut i = 0;
                while i < Self::lanes() {
                    bits |= (value[i] as $storage) << i;
                    i += 1;
                }
                bits
            }

            fn all(value: Storage<Self>) -> bool { value == Self::VALID }
            fn any(value: Storage<Self>) -> bool { value != 0 }

            fn native_bitmask(value: Storage<Self>) -> Option<u64> { Some(value as u64) }
            fn from_native_bitmask(bitmask: u64) -> Storage<Self> { (bitmask as $storage) & Self::VALID }

            #[cfg(feature = "bitvec")]
            fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
                let mut i = 0;
                while i < Self::lanes() {
                    view.set(i, Self::test(value, i));
                    i += 1;
                }
            }
        }

        #[thermite_macros::inline_always]
        impl CastMaskRegister<$name> for $name {
            fn mask_from(value: Storage<Self>) -> Storage<Self> {
                value
            }
        }
    )*};
}

decl_kmask! {
    /// 2-lane opmask (`f64x2`, `u64x2`, ...). `__mmask8` storage. Only the low
    /// 2 bits may be set.
    KMask2: U2 => u8;
    /// 4-lane opmask (`f32x4`, `f64x4`, ...). `__mmask8` storage. Only the low
    /// 4 bits may be set.
    KMask4: U4 => u8;
    /// 8-lane opmask (`f32x8`, `f64x8`, `i16x8`, ...). `__mmask8` storage.
    KMask8: U8 => u8;
    /// 16-lane opmask (`f32x16`, `i16x16`, `u8x16`, ...). `__mmask16` storage.
    KMask16: U16 => u16;
    /// 32-lane opmask (`i16x32`, `u8x32`, ...). `__mmask32` storage.
    KMask32: U32 => u32;
    /// 64-lane opmask (`i8x64`, `u8x64`). `__mmask64` storage.
    KMask64: U64 => u64;
}

/// Width ladder for opmasks: the wide mask is `lo | (hi << HALF_LANES)`.
///
/// `concat`/`extend` inputs are invariant-clean (bits above their lane count
/// are zero), so the shift-or produces a clean wide mask. `split`/`narrow`
/// re-scrub with `VALID` because a same-storage truncation (`u8 -> u8` for
/// `KMask4 -> KMask2`) keeps the sibling half's bits. Where the storage really
/// narrows the AND folds away.
macro_rules! impl_kmask_concat {
    ($($wide:ident: $half:ident;)*) => {$(
        #[thermite_macros::inline_always]
        impl ConcatRegister<$half> for $wide {
            fn concat(lo: Storage<$half>, hi: Storage<$half>) -> Storage<Self> {
                (lo as Storage<Self>) | ((hi as Storage<Self>) << <$half as CoreRegister>::Lanes::USIZE)
            }

            fn split(value: Storage<Self>) -> (Storage<$half>, Storage<$half>) {
                let lo = (value as Storage<$half>) & <$half>::VALID;
                let hi = ((value >> <$half as CoreRegister>::Lanes::USIZE) as Storage<$half>) & <$half>::VALID;
                (lo, hi)
            }
        }

        #[thermite_macros::inline_always]
        impl ExtendRegister<$half> for $wide {
            fn extend(value: Storage<$half>) -> Storage<Self> {
                value as Storage<Self>
            }

            fn narrow(value: Storage<Self>) -> Storage<$half> {
                (value as Storage<$half>) & <$half>::VALID
            }
        }
    )*};
}

impl_kmask_concat! {
    KMask4: KMask2;
    KMask8: KMask4;
    KMask16: KMask8;
    KMask32: KMask16;
    KMask64: KMask32;
}

// Bottom rung: two `bool` lanes make a `KMask2` (the scalar-concat mask side
// of `FullConcatRegister<f64>` on `f64x2` and friends).
#[thermite_macros::inline_always]
impl ConcatRegister<bool> for KMask2 {
    fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
        (lo as u8) | ((hi as u8) << 1)
    }

    fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
        (value & 0b01 != 0, value & 0b10 != 0)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<bool> for KMask2 {
    fn extend(value: Storage<bool>) -> Storage<Self> {
        value as u8
    }

    fn narrow(value: Storage<Self>) -> Storage<bool> {
        value & 0b01 != 0
    }
}

#[cfg(test)]
mod tests {
    use generic_array::sequence::GenericSequence;

    use super::*;

    struct KeepLow<const N: usize>;

    impl<const N: usize> ZeroUpper for KeepLow<N> {
        const N: usize = N;
    }

    fn lcg(x: u64) -> u64 {
        x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    }

    /// Every assertion routes through `native_bitmask`/`from_native_bitmask`
    /// so the suite stays generic over the six storage types, with plain u64
    /// arithmetic as the reference.
    fn suite<M: MaskRegister>() {
        let n = M::lanes();
        let valid = low_bits(n);
        let nb = |s: Storage<M>| M::native_bitmask(s).unwrap();

        // Constants and the high-bit invariant on the complement path.
        assert_eq!(nb(M::TRUTHY), valid);
        assert_eq!(nb(M::FALSY), 0);
        assert_eq!(nb(M::EMPTY), 0);
        assert_eq!(nb(M::not(M::FALSY)), valid, "not must set exactly the valid window");
        assert_eq!(nb(M::not(M::TRUTHY)), 0);
        assert!(M::all(M::TRUTHY) && !M::all(M::FALSY));
        assert!(M::any(M::TRUTHY) && !M::any(M::FALSY) && M::none(M::FALSY));
        assert_eq!(nb(M::boolean(true)), valid);
        assert_eq!(nb(M::boolean(false)), 0);

        // Constructors must scrub bits above the lane count.
        assert_eq!(nb(M::from_native_bitmask(u64::MAX)), valid);

        // zeroupper keeps only the requested prefix.
        assert_eq!(nb(M::zeroupper_z::<KeepLow<1>>(M::TRUTHY)), 1);
        assert_eq!(nb(M::zeroupper(M::TRUTHY)), valid);

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..64 {
            state = lcg(state);
            let am = state & valid;
            state = lcg(state);
            let bm = state & valid;
            state = lcg(state);
            let cm = state & valid;

            let a = M::from_native_bitmask(am);
            let b = M::from_native_bitmask(bm);
            let c = M::from_native_bitmask(cm);
            assert_eq!(nb(a), am, "from_native_bitmask/native_bitmask must round-trip");

            // Per-lane probes.
            for i in 0..n {
                assert_eq!(M::test(a, i), (am >> i) & 1 != 0);
            }

            // Bitwise against the u64 reference (results must stay clean).
            assert_eq!(nb(M::bitand(a, b)), am & bm);
            assert_eq!(nb(M::bitor(a, b)), am | bm);
            assert_eq!(nb(M::bitxor(a, b)), am ^ bm);
            assert_eq!(nb(M::not(a)), !am & valid);
            assert_eq!(nb(M::bitandnot(a, b)), am & !bm);
            assert_eq!(
                nb(M::ternlog::<0xCA>(a, b, c)),
                (bm & am) | (cm & !am),
                "select-form ternlog"
            );

            // Core select family (mask = a).
            assert_eq!(nb(M::blendv(a, b, c)), (cm & am) | (bm & !am));
            assert_eq!(nb(M::zz(a, b)), bm & am);
            assert_eq!(nb(M::nz(a, b)), bm & !am);
            assert_eq!(nb(M::from_mask(a)), am);

            // Scans/counts ride the native_bitmask defaults.
            assert_eq!(M::count_set_one(a), am.count_ones() as usize);
            assert_eq!(M::first_set_one(a), (am != 0).then(|| am.trailing_zeros() as usize));
            assert_eq!(M::last_set_one(a), (am != 0).then(|| 63 - am.leading_zeros() as usize));

            // Interleave against the lane-formula reference, then round-trip.
            let (lo, hi) = M::interleave(a, b);
            for i in 0..n {
                let src = if i % 2 == 0 { am } else { bm };
                assert_eq!(M::test(lo, i), (src >> (i / 2)) & 1 != 0, "interleave lo lane {i}");
                assert_eq!(
                    M::test(hi, i),
                    (src >> (n / 2 + i / 2)) & 1 != 0,
                    "interleave hi lane {i}"
                );
            }
            let (ra, rb) = M::deinterleave(lo, hi);
            assert_eq!(nb(ra), am, "deinterleave . interleave = id");
            assert_eq!(nb(rb), bm);

            // new_mask agrees with per-lane set().
            let arr = GenericArray::<bool, M::Lanes>::generate(|i| (am >> i) & 1 != 0);
            let built = M::new_mask(arr);
            assert_eq!(nb(built), am);
        }

        // set() builds and clears.
        let mut m = M::FALSY;
        for i in (0..n).step_by(2) {
            m = M::set(m, i, true);
        }
        assert_eq!(nb(m), 0x5555_5555_5555_5555 & valid);
        m = M::set(m, 0, false);
        assert_eq!(nb(m), 0x5555_5555_5555_5554 & valid);
    }

    macro_rules! kmask_tests {
        ($($test:ident => $ty:ident),* $(,)?) => {$(
            #[test]
            fn $test() {
                suite::<$ty>();
            }
        )*};
    }

    kmask_tests! {
        kmask2 => KMask2,
        kmask4 => KMask4,
        kmask8 => KMask8,
        kmask16 => KMask16,
        kmask32 => KMask32,
        kmask64 => KMask64,
    }
}
