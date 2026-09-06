//! Embedded-mask variant stampers for the v4 register files, plus the
//! stable-partition `compress`/`expand` built on the zeroing EVEX forms
//! ([`compress_expand_v4!`], [`compress_v4!`], [`expand_v4!`]).
//!
//! `register_trait` derives every `_c`/`_m`/`_z` sibling from `blendv`, which
//! on AVX-512 is a `vblendm` AFTER the op. Every EVEX instruction can take the
//! opmask itself (merge form `(src, k, operands..)`, zero form
//! `(k, operands..)`), so the derived defaults are one instruction too long
//! and put the mask on the result's critical path. The register files
//! override every variant explicitly. These macros only remove the
//! repetition of writing the three signatures. Each invocation still names
//! the base method and BOTH intrinsics, so the mapping is visible and
//! reviewable per register. Nothing about which instruction backs a method
//! is inferred.
//!
//! Shapes (the `_c` variant keeps its FIRST operand where the mask is false,
//! matching the derived default):
//!
//! - [`masked_binary_v4!`]: `f(lhs, rhs)` with `(src, k, a, b)` intrinsics.
//!   An optional `(Type)` after the name overrides the `rhs` type (variable
//!   shifts take `Storage<Self::Unsigned>`).
//! - [`masked_andnot_v4!`]: `bitandnot(lhs, rhs)` = `lhs & !rhs` with the
//!   `vandn` intrinsics, which negate their FIRST operand, so `lhs`/`rhs`
//!   reach the intrinsic swapped while `_c` still keeps `lhs`.
//! - [`masked_unary_v4!`]: `f(value)` with `(src, k, a)` intrinsics. An
//!   optional trailing `, IMM` literal appends a legacy-const immediate
//!   (`vrndscale`, `vpternlog` in its self-NOT form).
//! - [`masked_fma_v4!`]: the FMA family. The merge intrinsic keeps its first
//!   multiplicand (`_mm512_mask_fmadd(a, k, b, c)`), which is exactly `_c`.
//!   `_m` with an arbitrary `src` needs the op plus a merge move.
//! - [`masked_shift_v4!`] / [`masked_shifti_v4!`]: the count-in-xmm shift
//!   forms (`vpsll{d,q} zmm, zmm, xmm`), runtime `u32` and `const IMM8`
//!   respectively. The immediate forms cannot be used: `::<{ IMM8 as u32 }>`
//!   needs `generic_const_exprs`, and LLVM folds the xmm count back to the
//!   immediate encoding anyway.
//! - [`masked_via_mov_v4!`]: composed ops whose final step has no masked
//!   form. The op runs unmasked and one `vmovdq*`/`vmov*ps` merge does the
//!   select, the same instruction count as the derived default, written
//!   out so the file states its choice for every method rather than
//!   silently inheriting.
//!
//! All emit `#[inline(always)]` themselves: the `inline_always` proc macro
//! only sees fn items, not macro invocations.

macro_rules! masked_binary_v4 {
    ($($name:ident $(($rhs:ty))? => $mask:ident, $maskz:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: masked_binary_v4!(@rhs $($rhs)?)) -> Storage<Self> {
            unsafe { arch::$mask(lhs, mask, lhs, rhs) }
        }

        #[inline(always)]
        fn [<$name _m>](src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: masked_binary_v4!(@rhs $($rhs)?)) -> Storage<Self> {
            unsafe { arch::$mask(src, mask, lhs, rhs) }
        }

        #[inline(always)]
        fn [<$name _z>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: masked_binary_v4!(@rhs $($rhs)?)) -> Storage<Self> {
            unsafe { arch::$maskz(mask, lhs, rhs) }
        }
    )* } };

    (@rhs) => { Storage<Self> };
    (@rhs $rhs:ty) => { $rhs };
}

macro_rules! masked_andnot_v4 {
    ($($name:ident => $mask:ident, $maskz:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(lhs, mask, rhs, lhs) }
        }

        #[inline(always)]
        fn [<$name _m>](src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(src, mask, rhs, lhs) }
        }

        #[inline(always)]
        fn [<$name _z>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$maskz(mask, rhs, lhs) }
        }
    )* } };
}

macro_rules! masked_unary_v4 {
    ($($name:ident => $mask:ident, $maskz:ident $(, $imm:literal)?;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>](mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(value, mask, value $(, $imm)?) }
        }

        #[inline(always)]
        fn [<$name _m>](src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(src, mask, value $(, $imm)?) }
        }

        #[inline(always)]
        fn [<$name _z>](mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$maskz(mask, value $(, $imm)?) }
        }
    )* } };
}

macro_rules! masked_fma_v4 {
    (mov = $mov:ident; $($name:ident => $mask:ident, $maskz:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(lhs, mask, rhs, acc) }
        }

        #[inline(always)]
        fn [<$name _m>](src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mov(src, mask, Self::$name(lhs, rhs, acc)) }
        }

        #[inline(always)]
        fn [<$name _z>](mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$maskz(mask, lhs, rhs, acc) }
        }
    )* } };
}

macro_rules! masked_shift_v4 {
    ($($name:ident => $mask:ident, $maskz:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>](mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
            unsafe { arch::$mask(value, mask, value, arch::_mm_cvtsi32_si128(shift as i32)) }
        }

        #[inline(always)]
        fn [<$name _m>](src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
            unsafe { arch::$mask(src, mask, value, arch::_mm_cvtsi32_si128(shift as i32)) }
        }

        #[inline(always)]
        fn [<$name _z>](mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
            unsafe { arch::$maskz(mask, value, arch::_mm_cvtsi32_si128(shift as i32)) }
        }
    )* } };
}

macro_rules! masked_shifti_v4 {
    ($($name:ident => $mask:ident, $maskz:ident;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>]<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(value, mask, value, arch::_mm_cvtsi32_si128(IMM8)) }
        }

        #[inline(always)]
        fn [<$name _m>]<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$mask(src, mask, value, arch::_mm_cvtsi32_si128(IMM8)) }
        }

        #[inline(always)]
        fn [<$name _z>]<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
            unsafe { arch::$maskz(mask, value, arch::_mm_cvtsi32_si128(IMM8)) }
        }
    )* } };
}

macro_rules! masked_via_mov_v4 {
    (mov = $mov:ident, movz = $movz:ident;
     $($name:ident $(<$(const $g:ident : $gt:ty),+>)? ($this:ident : $this_ty:ty $(, $arg:ident : $arg_ty:ty)*);)*
    ) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>] $(<$(const $g: $gt),+>)? (mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { arch::$mov($this, mask, Self::$name $(::<$($g),+>)? ($this $(, $arg)*)) }
        }

        #[inline(always)]
        fn [<$name _m>] $(<$(const $g: $gt),+>)? (src: Storage<Self>, mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { arch::$mov(src, mask, Self::$name $(::<$($g),+>)? ($this $(, $arg)*)) }
        }

        #[inline(always)]
        fn [<$name _z>] $(<$(const $g: $gt),+>)? (mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { arch::$movz(mask, Self::$name $(::<$($g),+>)? ($this $(, $arg)*)) }
        }
    )* } };
}

/// `compress` with the trait's stable-partition contract (selected lanes
/// packed low, UNselected lanes packed after them, in order) from the zeroing
/// `vpcompress`: pack both halves, then merge the second above the first with
/// a masked `vpexpand`. `$k` is the raw opmask integer behind `Self::Mask`
/// (`u8` for every mask of 8 or fewer lanes). Sub-8-lane masks are scrubbed
/// to their valid bits before use.
///
/// Found by diff_compress under SDE: the plain `maskz_compress` zeroes the
/// tail, which is `compress_z`, not `compress`.
macro_rules! compress_v4 {
    ($k:ty, $maskz_compress:ident, $mask_expand:ident, $mask:expr, $value:expr) => {{
        let lanes =
            <<Self as $crate::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE as u32;
        let valid: $k = if lanes >= <$k>::BITS {
            <$k>::MAX
        } else {
            ((1u64 << lanes) - 1) as $k
        };
        let mask: $k = $mask & valid;
        let value = $value;
        let cnt = mask.count_ones();
        let hi: $k = if cnt >= lanes { 0 } else { (<$k>::MAX << cnt) & valid };
        unsafe {
            let sel = arch::$maskz_compress(mask, value);
            let rest = arch::$maskz_compress(!mask & valid, value);
            arch::$mask_expand(sel, hi, rest)
        }
    }};
}

/// The inverse permutation of [`compress_v4!`]: selected lanes read the packed
/// front in order (`vpexpand`), unselected lanes read the tail in order (pack
/// the tail low with `vpcompress`, scatter it into the unselected positions
/// with a masked `vpexpand`).
macro_rules! expand_v4 {
    ($k:ty, $maskz_compress:ident, $mask_expand:ident, $maskz_expand:ident, $mask:expr, $value:expr) => {{
        let lanes =
            <<Self as $crate::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE as u32;
        let valid: $k = if lanes >= <$k>::BITS {
            <$k>::MAX
        } else {
            ((1u64 << lanes) - 1) as $k
        };
        let mask: $k = $mask & valid;
        let value = $value;
        let cnt = mask.count_ones();
        let hi: $k = if cnt >= lanes { 0 } else { (<$k>::MAX << cnt) & valid };
        unsafe {
            let front = arch::$maskz_expand(mask, value);
            let tail = arch::$maskz_compress(hi, value);
            arch::$mask_expand(front, !mask & valid, tail)
        }
    }};
}

/// All four of `compress`/`compress_z`/`expand`/`expand_z` for a register whose
/// element size has unconditional EVEX compress/expand (32/64-bit at every
/// width). The 8/16-bit files use [`compress_v4!`]/[`expand_v4!`] inside their
/// VBMI2 arms instead.
macro_rules! compress_expand_v4 {
    ($k:ty, $maskz_compress:ident, $mask_expand:ident, $maskz_expand:ident) => {
        #[inline(always)]
        fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
            compress_v4!($k, $maskz_compress, $mask_expand, mask, value)
        }

        #[inline(always)]
        fn compress_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
            unsafe { arch::$maskz_compress(mask, value) }
        }

        #[inline(always)]
        fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
            expand_v4!($k, $maskz_compress, $mask_expand, $maskz_expand, mask, value)
        }

        #[inline(always)]
        fn expand_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
            unsafe { arch::$maskz_expand(mask, value) }
        }
    };
}

/// Masked variants backed by a polyfill pair that already forks on the tier
/// (`gfni`, `funnel`): `$mask(src, k, this, args..)` and
/// `$maskz(k, this, args..)`. `_c` is `$mask` with `src = this` unless a
/// dedicated keep-`this` form is named third.
macro_rules! masked_poly_v4 {
    ($($name:ident $(<$(const $g:ident : $gt:ty),+>)? ($this:ident : $this_ty:ty $(, $arg:ident : $arg_ty:ty)*)
        => $mask:ident, $maskz:ident $(, $maskc:ident)?;)*) => { paste::paste! { $(
        #[inline(always)]
        fn [<$name _c>] $(<$(const $g: $gt),+>)? (mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { masked_poly_v4!(@c $this, mask, $mask $(, $maskc)? [$($($g),+)?] ($($arg),*)) }
        }

        #[inline(always)]
        fn [<$name _m>] $(<$(const $g: $gt),+>)? (src: Storage<Self>, mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { arch::$mask $(::<$($g),+>)? (src, mask, $this $(, $arg)*) }
        }

        #[inline(always)]
        fn [<$name _z>] $(<$(const $g: $gt),+>)? (mask: Storage<Self::Mask>, $this: $this_ty $(, $arg: $arg_ty)*) -> Storage<Self> {
            unsafe { arch::$maskz $(::<$($g),+>)? (mask, $this $(, $arg)*) }
        }
    )* } };

    (@c $this:ident, $mask_v:ident, $mask:ident [$($g:ident),*] ($($arg:ident),*)) => {
        arch::$mask::<$($g),*>($this, $mask_v, $this $(, $arg)*)
    };
    (@c $this:ident, $mask_v:ident, $mask:ident, $maskc:ident [$($g:ident),*] ($($arg:ident),*)) => {
        arch::$maskc::<$($g),*>($mask_v, $this $(, $arg)*)
    };
}
