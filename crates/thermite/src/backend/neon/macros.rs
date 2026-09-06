//! Trait-impl stamping macros for the NEON register files.
//!
//! NEON intrinsic names are perfectly regular (`vaddq_f32` / `vaddq_s16` /
//! ...), so the bulk of every register's trait surface is generated here from
//! a type-suffix parameter, calling through the `polyfills` normalization
//! layer (`neon_and_f32`, `neon_movemask_u32`, ...) where NEON needs plumbing
//! or lacks an instruction. Register files invoke these macros and then write
//! only their genuinely type-specific impls (casts, LinAlg, 64-bit gaps).

/// `extract`/`insert` lane accessors: NEON `vgetq_lane`/`vsetq_lane` take an
/// `i32` const generic, while the trait's `I` is a `usize` const generic, so a
/// `match` (collapsed at monomorphization) bridges them.
macro_rules! neon_lane_accessors {
    ($get:ident, $set:ident; 2) => {
        #[inline(always)]
        fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
            unsafe {
                match I {
                    0 => arch::$get::<0>(value),
                    _ => arch::$get::<1>(value),
                }
            }
        }

        #[inline(always)]
        fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$set::<0>(element, value),
                    _ => arch::$set::<1>(element, value),
                }
            }
        }
    };
    ($get:ident, $set:ident; 4) => {
        #[inline(always)]
        fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
            unsafe {
                match I {
                    0 => arch::$get::<0>(value),
                    1 => arch::$get::<1>(value),
                    2 => arch::$get::<2>(value),
                    _ => arch::$get::<3>(value),
                }
            }
        }

        #[inline(always)]
        fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$set::<0>(element, value),
                    1 => arch::$set::<1>(element, value),
                    2 => arch::$set::<2>(element, value),
                    _ => arch::$set::<3>(element, value),
                }
            }
        }
    };
    ($get:ident, $set:ident; 8) => {
        #[inline(always)]
        fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
            unsafe {
                match I {
                    0 => arch::$get::<0>(value),
                    1 => arch::$get::<1>(value),
                    2 => arch::$get::<2>(value),
                    3 => arch::$get::<3>(value),
                    4 => arch::$get::<4>(value),
                    5 => arch::$get::<5>(value),
                    6 => arch::$get::<6>(value),
                    _ => arch::$get::<7>(value),
                }
            }
        }

        #[inline(always)]
        fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$set::<0>(element, value),
                    1 => arch::$set::<1>(element, value),
                    2 => arch::$set::<2>(element, value),
                    3 => arch::$set::<3>(element, value),
                    4 => arch::$set::<4>(element, value),
                    5 => arch::$set::<5>(element, value),
                    6 => arch::$set::<6>(element, value),
                    _ => arch::$set::<7>(element, value),
                }
            }
        }
    };
    ($get:ident, $set:ident; 16) => {
        #[inline(always)]
        fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
            unsafe {
                match I {
                    0 => arch::$get::<0>(value),
                    1 => arch::$get::<1>(value),
                    2 => arch::$get::<2>(value),
                    3 => arch::$get::<3>(value),
                    4 => arch::$get::<4>(value),
                    5 => arch::$get::<5>(value),
                    6 => arch::$get::<6>(value),
                    7 => arch::$get::<7>(value),
                    8 => arch::$get::<8>(value),
                    9 => arch::$get::<9>(value),
                    10 => arch::$get::<10>(value),
                    11 => arch::$get::<11>(value),
                    12 => arch::$get::<12>(value),
                    13 => arch::$get::<13>(value),
                    14 => arch::$get::<14>(value),
                    _ => arch::$get::<15>(value),
                }
            }
        }

        #[inline(always)]
        fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$set::<0>(element, value),
                    1 => arch::$set::<1>(element, value),
                    2 => arch::$set::<2>(element, value),
                    3 => arch::$set::<3>(element, value),
                    4 => arch::$set::<4>(element, value),
                    5 => arch::$set::<5>(element, value),
                    6 => arch::$set::<6>(element, value),
                    7 => arch::$set::<7>(element, value),
                    8 => arch::$set::<8>(element, value),
                    9 => arch::$set::<9>(element, value),
                    10 => arch::$set::<10>(element, value),
                    11 => arch::$set::<11>(element, value),
                    12 => arch::$set::<12>(element, value),
                    13 => arch::$set::<13>(element, value),
                    14 => arch::$set::<14>(element, value),
                    _ => arch::$set::<15>(element, value),
                }
            }
        }
    };
}

/// Multiplicative lane reduction (no NEON instruction exists): log2 fold via
/// `vextq` rotations, ending in a lane-0 extract.
macro_rules! neon_mul_reduce {
    ($mul:ident, $ext:ident, $get:ident, $v:expr; 2) => {{
        let v = $v;
        unsafe { arch::$get::<0>(arch::$mul(v, arch::$ext::<1>(v, v))) }
    }};
    ($mul:ident, $ext:ident, $get:ident, $v:expr; 4) => {{
        let v = $v;
        unsafe {
            let t = arch::$mul(v, arch::$ext::<2>(v, v));
            arch::$get::<0>(arch::$mul(t, arch::$ext::<1>(t, t)))
        }
    }};
    ($mul:ident, $ext:ident, $get:ident, $v:expr; 8) => {{
        let v = $v;
        unsafe {
            let t = arch::$mul(v, arch::$ext::<4>(v, v));
            let t = arch::$mul(t, arch::$ext::<2>(t, t));
            arch::$get::<0>(arch::$mul(t, arch::$ext::<1>(t, t)))
        }
    }};
    ($mul:ident, $ext:ident, $get:ident, $v:expr; 16) => {{
        let v = $v;
        unsafe {
            let t = arch::$mul(v, arch::$ext::<8>(v, v));
            let t = arch::$mul(t, arch::$ext::<4>(t, t));
            let t = arch::$mul(t, arch::$ext::<2>(t, t));
            arch::$get::<0>(arch::$mul(t, arch::$ext::<1>(t, t)))
        }
    }};
}

/// CoreRegister + BitwiseRegister + InterleaveRegister + MaskRegister +
/// self-identity mask/bit casts - the mask-capable core shared by every
/// native NEON register (`Mask = Self`, full-width lane masks).
macro_rules! neon_mask_core {
    (
        $reg:ty, lanes: $n:tt($lt:ty), storage: $st:ident, suffix: $s:ident,
        truthy: $truthy:expr, from_u: $from_u:ident
    ) => {
        paste::paste! {
            impl crate::simd::HasIsa for $reg {
                type Native = crate::backend::neon::Neon;
            }

            #[thermite_macros::inline_always]
            impl CoreRegister for $reg {
                type Lanes = $lt;
                type Storage = arch::$st;
                type Mask = Self;

                const IS_EMULATED: bool = false;
                const HAS_EQUAL_SIZE_MASK: bool = true;
                const EMPTY: Storage<Self> = empty_reg::<Self>();

                fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_bsl_ $s>](mask, on_true, on_false)
                }

                fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_and_ $s>](value, mask)
                }

                fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                    // keep value where mask is false: value & !mask
                    arch::[<neon_andnot_ $s>](value, mask)
                }

                fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                    if const { Z::N >= $n } {
                        value
                    } else {
                        arch::[<neon_and_ $s>](value, const { arch::[<neon_keep_mask_ $s>](Z::N) })
                    }
                }

                fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
                    mask
                }
            }

            #[rustfmt::skip] #[thermite_macros::inline_always]
            impl BitwiseRegister for $reg {
                fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_xor_ $s>](lhs, rhs)
                }

                fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_and_ $s>](lhs, rhs)
                }

                fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_andnot_ $s>](lhs, rhs)
                }

                fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_or_ $s>](lhs, rhs)
                }

                fn not(value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_not_ $s>](value)
                }
            }

            #[thermite_macros::inline_always]
            impl InterleaveRegister for $reg {
                fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                    unsafe { (arch::[<vzip1q_ $s>](a, b), arch::[<vzip2q_ $s>](a, b)) }
                }

                fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                    unsafe { (arch::[<vuzp1q_ $s>](a, b), arch::[<vuzp2q_ $s>](a, b)) }
                }
            }

            #[thermite_macros::inline_always]
            impl MaskRegister for $reg {
                const FALSY: Storage<Self> = empty_reg::<Self>();
                const TRUTHY: Storage<Self> = reg::<Self, $n>([$truthy; $n]);

                fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
                    Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
                    mask
                }

                fn test(mask: Storage<Self>, lane: usize) -> bool {
                    Self::as_slice(&mask)[lane].to_bool()
                }

                fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_bools_to_mask_x $n>](value)) }
                }

                fn all(value: Storage<Self>) -> bool {
                    arch::[<neon_mask_all_ $s>](value)
                }

                fn any(value: Storage<Self>) -> bool {
                    arch::[<neon_mask_any_ $s>](value)
                }

                fn native_bitmask(value: Storage<Self>) -> Option<u64> {
                    Some(arch::[<neon_movemask_ $s>](value))
                }

                fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_frombitmask_x $n>](bitmask)) }
                }

                // Negated horizontal sum of the mask lanes - no bitmask, no popcount.
                fn count_set<const N: usize>(values: [Storage<Self>; N]) -> usize {
                    arch::[<neon_count_mask_ $s>](values)
                }

                #[cfg(feature = "bitvec")]
                fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
                    let mask = arch::[<neon_movemask_ $s>](value) as u32;
                    let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
                    view.copy_from_bitslice(&mask[..<Self::Lanes as Unsigned>::USIZE]);
                }
            }

            impl CastMaskRegister<$reg> for $reg {
                #[inline(always)]
                fn mask_from(value: Storage<Self>) -> Storage<Self> {
                    value
                }
            }
        }
    };
}

/// The `Register` impl: memory ops, lane accessors, splats, permutes.
macro_rules! neon_register {
    (
        $reg:ty, elem: $e:ty, lanes: $n:tt, suffix: $s:ident, vec: $vt:ident,
        signed: $sg:ty, unsigned: $un:ty,
        compress: $compress:tt, bytes: ($to_b:ident, $from_b:ident)
        $(, extras: { $($extras:tt)* })?
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl Register for $reg {
                type Element = $e;
                type Signed = $sg;
                type Unsigned = $un;

                // `LD2`/`LD3`/`LD4` below are real load-unit transposes, so the
                // grouped memory ops should decompose into per-chunk structural
                // loads rather than take the flat shuffle engine.
                const HAS_STRUCTURAL_MEMOPS: bool = true;

                fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                    arch::[<neon_nonzero_mask_ $s>](value)
                }

                fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
                    value
                }

                fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                    arch::[<neon_msb_mask_ $s>](value)
                }

                fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
                    unsafe { arch::[<vld1q_ $s>](value.as_slice().as_ptr()) }
                }

                fn single(value: Self::Element) -> Storage<Self> {
                    unsafe { arch::[<vsetq_lane_ $s>]::<0>(value, Self::EMPTY) }
                }

                fn splat(value: Self::Element) -> Storage<Self> {
                    unsafe { arch::[<vdupq_n_ $s>](value) }
                }

                unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
                    unsafe { arch::[<vld1q_ $s>](ptr) }
                }

                unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
                    unsafe { arch::[<vst1q_ $s>](ptr, value) }
                }

                fn reverse(value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_reverse_ $s>](value)
                }

                fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_swap_bytes_ $s>](value)
                }

                neon_lane_accessors!([<vgetq_lane_ $s>], [<vsetq_lane_ $s>]; $n);

                neon_broadcast_align!([<vdupq_laneq_ $s>], [<vextq_ $s>]; $n);

                const HAS_PERMUTEV: bool = true;

                fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
                    arch::[<neon_tbl_ $s>](value, arch::[<neon_ctrl_x $n>](idxs))
                }

                // One TBL2 replaces the default's two-permute + blend lowering.
                // Byte indices >= 32 (lane index >= 2 * LANES) yield zero, matching
                // the permutev-based default's out-of-range behavior. The control
                // build is width-scaled, not range-limited, so the same builder
                // serves both forms.
                fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
                    arch::[<neon_tbl2_ $s>](a, b, arch::[<neon_ctrl_x $n>](idxs))
                }

                fn swizzle_const<I: crate::swizzle::SwizzleIndices<Self::Lanes>>(
                    a: Storage<Self>,
                    b: Storage<Self>,
                ) -> Storage<Self> {
                    arch::[<neon_tbl2_ $s>](a, b, const {
                        arch::neon_lane_table::<$n>(16 / $n, unsafe {
                            crate::generic_array::const_transmute(I::INDICES)
                        })
                    })
                }

                // Cross-chunk permute by LIVE index registers, one `TBL` per
                // output chunk (the M-chunk array IS a 16*M-byte table) against
                // the default's M permutes + M blends per chunk. The global
                // index scales to a global byte index by exactly the same
                // `neon_ctrl_x*` recipe as `permutev` (`M * 16 <= 64` bytes
                // keeps every in-range byte under 256), and an out-of-range
                // index lands past the table, which `TBL` zeroes (a legal
                // unspecified value). TBL tables cap at 4 registers, so M > 4
                // keeps the scalar path, as it did before.
                fn array_permutev<const M: usize>(
                    value: [Storage<Self>; M],
                    idxs: [Storage<Self::Unsigned>; M],
                ) -> [Storage<Self>; M] {
                    if const { M >= 1 && M <= 4 } {
                        unsafe {
                            let v = value.as_slice();
                            let mut table = [arch::vdupq_n_u8(0); 4];
                            let mut j = 0;
                            while j < M {
                                table[j] = arch::$to_b(v[j]);
                                j += 1;
                            }

                            let mut out = [Self::EMPTY; M];
                            let mut i = 0;
                            while i < M {
                                let ctrl = arch::[<neon_ctrl_x $n>](idxs[i]);
                                out[i] = arch::$from_b(arch::neon_tbl_n_u8::<M>(table, ctrl));
                                i += 1;
                            }
                            return out;
                        }
                    }

                    // M > 4: no single TBL spans the table, so gather scalar-side.
                    let l = $n;
                    let total = M * l;

                    let mut out = [Self::EMPTY; M];
                    for i in 0..M {
                        let mut arr: GenericArray<Self::Element, Self::Lanes> = GenericArray::default();
                        let row = <Self::Unsigned as Register>::as_slice(&idxs[i]);
                        for lane in 0..l {
                            let g = usize::try_from(row[lane]).unwrap_or(usize::MAX);
                            let g = if total.is_power_of_two() { g & (total - 1) } else { g.min(total - 1) };
                            arr[lane] = Self::as_slice(&value[g / l])[g % l];
                        }
                        out[i] = Self::new(arr);
                    }
                    out
                }

                // Compile-time-index companion: the scalar table build folds
                // away entirely when `idxs` is constant, and the runtime-row
                // callers (the compress merge shapes) still get the whole
                // cross-chunk permute in one TBL.
                //
                // Semantics: the generic default WRAPS an out-of-range index
                // (`g & (total-1)` when total is a power of two), while TBL
                // ZEROES an out-of-range byte, so indices are masked before
                // building the table, reproducing the default exactly.
                fn array_permutev_indices<const M: usize>(value: [Storage<Self>; M], idxs: &[u32]) -> [Storage<Self>; M] {
                    const ES: usize = 16 / $n; // element size in bytes
                    let l = $n;
                    let total = M * l;

                    if const { M >= 1 && M <= 4 } {
                        // one up-front bound so the per-lane index reads do not each check
                        let idxs = &idxs[..total];

                        unsafe {
                            let v = value.as_slice();
                            let mut table = [arch::vdupq_n_u8(0); 4];
                            let mut j = 0;
                            while j < M {
                                table[j] = arch::$to_b(v[j]);
                                j += 1;
                            }

                            let mut out = [Self::EMPTY; M];
                            let mut i = 0;
                            while i < M {
                                let mut bytes = [0u8; 16];
                                let mut lane = 0;
                                while lane < l {
                                    let g = idxs[i * l + lane] as usize;
                                    // mirror the default's normalization exactly
                                    let g = if const { (M * $n).is_power_of_two() } {
                                        g & (total - 1)
                                    } else {
                                        g.min(total - 1)
                                    };
                                    let mut b = 0;
                                    while b < ES {
                                        bytes[lane * ES + b] = (g * ES + b) as u8;
                                        b += 1;
                                    }
                                    lane += 1;
                                }
                                let idxv = arch::vld1q_u8(bytes.as_ptr());
                                out[i] = arch::$from_b(arch::neon_tbl_n_u8::<M>(table, idxv));
                                i += 1;
                            }
                            return out;
                        }
                    }

                    // M > 4 (or a non-power-of-two span): scalar gather.
                    let mut out = [Self::EMPTY; M];
                    for i in 0..M {
                        let mut arr: GenericArray<Self::Element, Self::Lanes> = GenericArray::default();
                        for lane in 0..l {
                            let g = idxs[i * l + lane] as usize;
                            let g = if total.is_power_of_two() { g & (total - 1) } else { g.min(total - 1) };
                            arr[lane] = Self::as_slice(&value[g / l])[g % l];
                        }
                        out[i] = Self::new(arr);
                    }
                    out
                }

                // Two-source companion of `array_permutev`: `a` then `b` is a
                // 32*M-byte table, so it fits TBL for M <= 2 (2 or 4 q-registers).
                fn array_swizzle<const M: usize>(
                    a: [Storage<Self>; M],
                    b: [Storage<Self>; M],
                    idxs: [Storage<Self::Unsigned>; M],
                ) -> [Storage<Self>; M] {
                    if const { M >= 1 && M <= 2 } {
                        unsafe {
                            let (av, bv) = (a.as_slice(), b.as_slice());
                            let mut table = [arch::vdupq_n_u8(0); 4];
                            let mut j = 0;
                            while j < M {
                                table[j] = arch::$to_b(av[j]);
                                table[M + j] = arch::$to_b(bv[j]);
                                j += 1;
                            }

                            let mut out = [Self::EMPTY; M];
                            let mut i = 0;
                            while i < M {
                                let ctrl = arch::[<neon_ctrl_x $n>](idxs[i]);
                                // `{ 2 * M }` in const-generic position needs
                                // `generic_const_exprs`. M is 1 or 2 here, so
                                // branch on it, `if const` folds the dead arm.
                                out[i] = arch::$from_b(if const { M == 1 } {
                                    arch::neon_tbl_n_u8::<2>(table, ctrl)
                                } else {
                                    arch::neon_tbl_n_u8::<4>(table, ctrl)
                                });
                                i += 1;
                            }
                            return out;
                        }
                    }

                    let l = $n;
                    let span = 2 * M * l;

                    let mut out = [Self::EMPTY; M];
                    for i in 0..M {
                        let mut arr: GenericArray<Self::Element, Self::Lanes> = GenericArray::default();
                        let row = <Self::Unsigned as Register>::as_slice(&idxs[i]);
                        for lane in 0..l {
                            let g = usize::try_from(row[lane]).unwrap_or(usize::MAX);
                            let g = if span.is_power_of_two() { g & (span - 1) } else { g.min(span - 1) };
                            let src = if g / l < M { &a[g / l] } else { &b[g / l - M] };
                            arr[lane] = Self::as_slice(src)[g % l];
                        }
                        out[i] = Self::new(arr);
                    }
                    out
                }

                // Compile-time-index companion of `array_swizzle`, see
                // `array_permutev_indices`.
                fn array_swizzle_indices<const M: usize>(
                    a: [Storage<Self>; M],
                    b: [Storage<Self>; M],
                    idxs: &[u32],
                ) -> [Storage<Self>; M] {
                    const ES: usize = 16 / $n;
                    let l = $n;
                    let total = M * l;
                    let span = 2 * total;

                    if const { M >= 1 && M <= 2 } {
                        let idxs = &idxs[..total];

                        unsafe {
                            let (av, bv) = (a.as_slice(), b.as_slice());
                            let mut table = [arch::vdupq_n_u8(0); 4];
                            let mut j = 0;
                            while j < M {
                                table[j] = arch::$to_b(av[j]);
                                table[M + j] = arch::$to_b(bv[j]);
                                j += 1;
                            }

                            let mut out = [Self::EMPTY; M];
                            let mut i = 0;
                            while i < M {
                                let mut bytes = [0u8; 16];
                                let mut lane = 0;
                                while lane < l {
                                    let g = idxs[i * l + lane] as usize;
                                    let g = if const { (2 * M * $n).is_power_of_two() } {
                                        g & (span - 1)
                                    } else {
                                        g.min(span - 1)
                                    };
                                    let mut bb = 0;
                                    while bb < ES {
                                        bytes[lane * ES + bb] = (g * ES + bb) as u8;
                                        bb += 1;
                                    }
                                    lane += 1;
                                }
                                let idxv = arch::vld1q_u8(bytes.as_ptr());
                                // `{ 2 * M }` in const-generic position needs
                                // `generic_const_exprs`. M is 1 or 2 here, so
                                // branch on it, `if const` folds the dead arm.
                                out[i] = arch::$from_b(if const { M == 1 } {
                                    arch::neon_tbl_n_u8::<2>(table, idxv)
                                } else {
                                    arch::neon_tbl_n_u8::<4>(table, idxv)
                                });
                                i += 1;
                            }
                            return out;
                        }
                    }

                    let mut out = [Self::EMPTY; M];
                    for i in 0..M {
                        let mut arr: GenericArray<Self::Element, Self::Lanes> = GenericArray::default();
                        for lane in 0..l {
                            let g = idxs[i * l + lane] as usize;
                            let g = if span.is_power_of_two() { g & (span - 1) } else { g.min(span - 1) };
                            let src = if g / l < M { &a[g / l] } else { &b[g / l - M] };
                            arr[lane] = Self::as_slice(src)[g % l];
                        }
                        out[i] = Self::new(arr);
                    }
                    out
                }

                // Radix-3 register de-interleave in three `TBL3`s: the three
                // registers ARE a 48-byte table, so each output stream is one
                // whole-table lookup (vs the default's 9 permute+blend pairs).
                // Index vectors are compile-time constants, so they fold to a
                // literal load. Only the `RADIX == 3` arm is native here; every
                // other radix takes the shared default (2-way `deinterleave` /
                // permute+blend gather).
                fn deinterleave_radix<const RADIX: usize>(
                    inputs: [Storage<Self>; RADIX],
                ) -> [Storage<Self>; RADIX] {
                    if const { RADIX == 3 } {
                        const ES: usize = 16 / $n;

                        // Byte table for output stream `r`: lane `l` wants flat
                        // element `l * 3 + r`, i.e. bytes `(l*3+r)*ES ..`.
                        const fn table<const R: usize>() -> arch::uint8x16_t {
                            let mut bytes = [0u8; 16];
                            let mut l = 0;
                            while l < $n {
                                let g = l * 3 + R;
                                let mut b = 0;
                                while b < ES {
                                    bytes[l * ES + b] = (g * ES + b) as u8;
                                    b += 1;
                                }
                                l += 1;
                            }
                            arch::cu8x16(bytes)
                        }

                        // SAFETY: `RADIX == 3` on this arm.
                        let (a, b, c) = unsafe {
                            (*inputs.get_unchecked(0), *inputs.get_unchecked(1), *inputs.get_unchecked(2))
                        };
                        let (o0, o1, o2) = unsafe {
                            let t = [arch::$to_b(a), arch::$to_b(b), arch::$to_b(c), arch::vdupq_n_u8(0)];
                            (
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<0>() })),
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<1>() })),
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<2>() })),
                            )
                        };
                        let mut out = [Self::EMPTY; RADIX];
                        // SAFETY: `RADIX == 3` on this arm.
                        unsafe {
                            *out.get_unchecked_mut(0) = o0;
                            *out.get_unchecked_mut(1) = o1;
                            *out.get_unchecked_mut(2) = o2;
                        }
                        out
                    } else {
                        crate::backend::generic::polyfills::deinterleave_radix_default::<Self, RADIX>(inputs)
                    }
                }

                fn interleave_radix<const RADIX: usize>(
                    inputs: [Storage<Self>; RADIX],
                ) -> [Storage<Self>; RADIX] {
                    if const { RADIX == 3 } {
                        const ES: usize = 16 / $n;

                        // Byte table for output register `i`: lane `l` is flat
                        // position `g = i * LANES + l`, which is element `g / 3` of
                        // stream `g % 3` - source flat index `(g % 3) * LANES + g / 3`.
                        const fn table<const I: usize>() -> arch::uint8x16_t {
                            let mut bytes = [0u8; 16];
                            let mut l = 0;
                            while l < $n {
                                let g = I * $n + l;
                                let src = (g % 3) * $n + (g / 3);
                                let mut b = 0;
                                while b < ES {
                                    bytes[l * ES + b] = (src * ES + b) as u8;
                                    b += 1;
                                }
                                l += 1;
                            }
                            arch::cu8x16(bytes)
                        }

                        // SAFETY: `RADIX == 3` on this arm.
                        let (x, y, z) = unsafe {
                            (*inputs.get_unchecked(0), *inputs.get_unchecked(1), *inputs.get_unchecked(2))
                        };
                        let (o0, o1, o2) = unsafe {
                            let t = [arch::$to_b(x), arch::$to_b(y), arch::$to_b(z), arch::vdupq_n_u8(0)];
                            (
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<0>() })),
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<1>() })),
                                arch::$from_b(arch::neon_tbl_n_u8::<3>(t, const { table::<2>() })),
                            )
                        };
                        let mut out = [Self::EMPTY; RADIX];
                        // SAFETY: `RADIX == 3` on this arm.
                        unsafe {
                            *out.get_unchecked_mut(0) = o0;
                            *out.get_unchecked_mut(1) = o1;
                            *out.get_unchecked_mut(2) = o2;
                        }
                        out
                    } else {
                        crate::backend::generic::polyfills::interleave_radix_default::<Self, RADIX>(inputs)
                    }
                }

                // Structural (de-interleaving) load/store: `LD2`/`LD3`/`LD4` read
                // N interleaved streams and hand back N de-interleaved registers
                // in ONE instruction - the AoS -> SoA transpose happens in the
                // load unit. `ST2`/`ST3`/`ST4` do the inverse. This is the NEON
                // feature x86 has no answer to (there, the default's load + TBL
                // permute is the best available).
                //
                // No alignment requirement on AArch64: unlike ARMv7's `:64`/`:128`
                // qualifiers, A64 structural loads take a plain address (a fault
                // needs SCTLR.A strict-alignment checking, which Linux leaves off).
                //
                // N == 1 and N > 4 fall back to the portable default's shape
                // (contiguous loads + a cross-register permute).
                unsafe fn load_deinterleaved<const N: usize>(ptr: *const Self::Element) -> [Storage<Self>; N] {
                    let mut out = [Self::EMPTY; N];
                    {
                        let o = out.as_mut_slice();
                        unsafe {
                            match N {
                                2 => {
                                    let v = arch::[<vld2q_ $s>](ptr);
                                    o[0] = v.0;
                                    o[1] = v.1;
                                }
                                3 => {
                                    let v = arch::[<vld3q_ $s>](ptr);
                                    o[0] = v.0;
                                    o[1] = v.1;
                                    o[2] = v.2;
                                }
                                4 => {
                                    let v = arch::[<vld4q_ $s>](ptr);
                                    o[0] = v.0;
                                    o[1] = v.1;
                                    o[2] = v.2;
                                    o[3] = v.3;
                                }
                                _ => {
                                    // N == 1, or beyond what LD4 covers: load
                                    // contiguously and use the portable
                                    // butterfly / gather.
                                    let mut src = [Self::EMPTY; N];
                                    for (i, s) in src.iter_mut().enumerate() {
                                        *s = Self::load_unaligned(ptr.add(i * $n));
                                    }
                                    return crate::backend::generic::polyfills::deinterleave_n::<Self, N>(src);
                                }
                            }
                        }
                    }
                    out
                }

                unsafe fn store_interleaved<const N: usize>(ptr: *mut Self::Element, values: [Storage<Self>; N]) {
                    let v = values.as_slice();
                    unsafe {
                        match N {
                            2 => arch::[<vst2q_ $s>](ptr, arch::[<$vt x2_t>](v[0], v[1])),
                            3 => arch::[<vst3q_ $s>](ptr, arch::[<$vt x3_t>](v[0], v[1], v[2])),
                            4 => arch::[<vst4q_ $s>](ptr, arch::[<$vt x4_t>](v[0], v[1], v[2], v[3])),
                            _ => {
                                let out = crate::backend::generic::polyfills::interleave_n::<Self, N>(values);
                                for (i, s) in out.iter().enumerate() {
                                    Self::store_unaligned(ptr.add(i * $n), *s);
                                }
                            }
                        }
                    }
                }

                neon_compress_sel!($compress, $n, $s);

                $($($extras)*)?
            }
        }
    };
}

/// The `table` arm also carries the byte-row overrides: the table path is the
/// only consumer of `widen_index_bytes`/`permutev_row`, so the two travel
/// together rather than being listed separately per register.
macro_rules! neon_compress_sel {
    (table, 2, $s:ident) => {
        compress_via_table!();
        impl_widen_index_bytes_neon!(x2, $s);
    };
    (table, 4, $s:ident) => {
        compress_via_table!();
        impl_widen_index_bytes_neon!(x4, $s);
    };
    (table, 8, $s:ident) => {
        compress_via_table!();
        impl_widen_index_bytes_neon!(x8, $s);
    };
    (wide, $n:tt, $s:ident) => {
        compress_via_wide!();
    };
}

/// PartialOrdRegister via native compares (all element types, including the
/// 64-bit ones SSE2/wasm have to emulate).
macro_rules! neon_partial_ord {
    ($reg:ty, suffix: $s:ident, from_u: $from_u:ident) => {
        paste::paste! {
            #[rustfmt::skip] #[thermite_macros::inline_always]
            impl PartialOrdRegister for $reg {
                fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<vceqq_ $s>](lhs, rhs)) }
                }

                fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<vcgtq_ $s>](lhs, rhs)) }
                }

                fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<vcgeq_ $s>](lhs, rhs)) }
                }

                fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<vcltq_ $s>](lhs, rhs)) }
                }

                fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<vcleq_ $s>](lhs, rhs)) }
                }
            }
        }
    };
}

/// x86-IMM8-encoded `ShuffleRegister` (per-lane blend) + `PermuteRegister`
/// (2-bit lane indices) for 4- and 2-lane registers.
macro_rules! neon_shuffle_permute {
    ($reg:ty, suffix: $s:ident, from_u: $from_u:ident; 4) => {
        paste::paste! {
            impl ShuffleRegister for $reg {
                #[inline(always)]
                fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::blendv(
                        unsafe { arch::$from_u(const { arch::neon_imm8x4_to_mask::<IMM8>() }) },
                        lhs,
                        rhs,
                    )
                }
            }

            impl PermuteRegister for $reg {
                #[inline(always)]
                fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_tbl_ $s>](value, const { arch::neon_imm8x4_to_table::<IMM8>() })
                }
            }
        }
    };
    ($reg:ty, suffix: $s:ident, from_u: $from_u:ident; 2) => {
        paste::paste! {
            impl ShuffleRegister for $reg {
                #[inline(always)]
                fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::blendv(
                        unsafe { arch::$from_u(const { arch::neon_imm8x2_to_mask::<IMM8>() }) },
                        lhs,
                        rhs,
                    )
                }
            }

            impl PermuteRegister for $reg {
                #[inline(always)]
                fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_tbl_ $s>](value, const { arch::neon_imm8x2_to_table::<IMM8>() })
                }
            }
        }
    };
}

/// NumericRegister for integer registers with native multiply and min/max
/// (8/16/32-bit lanes; the 64-bit registers hand-write their gaps).
macro_rules! neon_int_numeric {
    ($reg:ty, elem: $e:ty, lanes: $n:tt, suffix: $s:ident) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl NumericRegister for $reg {
                sort_via_network!($n);

                const ZERO: Storage<Self> = reg::<Self, $n>([0 as $e; $n]);
                const ONE: Storage<Self> = reg::<Self, $n>([1 as $e; $n]);
                const TWO: Storage<Self> = reg::<Self, $n>([2 as $e; $n]);

                const MIN: Storage<Self> = reg::<Self, $n>([<$e>::MIN; $n]);
                const MAX: Storage<Self> = reg::<Self, $n>([<$e>::MAX; $n]);

                // One across-vector reduce + scalar compare; the default is
                // all(eq(v, ZERO)) = CMEQ + UMINV. Exact for integers (every
                // zero lane is all-zero bits; floats cannot do this: -0.0).
                fn is_all_zero(value: Storage<Self>) -> bool {
                    !arch::[<neon_mask_any_ $s>](value)
                }

                fn min_element(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vminvq_ $s>](value) }
                }

                fn max_element(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vmaxvq_ $s>](value) }
                }

                fn sum_elements(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vaddvq_ $s>](value) }
                }

                fn prod_elements(value: Storage<Self>) -> Self::Element {
                    neon_mul_reduce!([<vmulq_ $s>], [<vextq_ $s>], [<vgetq_lane_ $s>], value; $n)
                }

                fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vpaddq_ $s>](lo, hi) }
                }

                fn offset() -> Storage<Self> {
                    Self::splat(<Self::Lanes as Unsigned>::USIZE as $e)
                }

                fn indexed() -> Storage<Self> {
                    const INDEXED: Storage<$reg> = {
                        let mut a = [0 as $e; $n];
                        let mut i = 0;
                        while i < $n {
                            a[i] = i as $e;
                            i += 1;
                        }
                        reg::<$reg, $n>(a)
                    };
                    INDEXED
                }

                fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vaddq_ $s>](lhs, rhs) }
                }

                fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vsubq_ $s>](lhs, rhs) }
                }

                fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vmulq_ $s>](lhs, rhs) }
                }

                fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
                }

                fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
                }

                fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vminq_ $s>](lhs, rhs) }
                }

                fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vmaxq_ $s>](lhs, rhs) }
                }
            }
        }
    };
}

/// BitshiftRegister: NEON's `vshlq` takes per-lane *signed* counts (negative
/// shifts right), giving native uniform and per-lane shifts in one
/// instruction. Logical right shifts run on the unsigned view, arithmetic on
/// the signed view; `$cs`/`$cu` name those views' suffixes and `$ce` the
/// count element type.
macro_rules! neon_bitshift {
    (
        $reg:ty, suffix: $s:ident, unsigned: $us:ident, count: ($ce:ty, $cs:ident),
        to_u: $to_u:ident, from_u: $from_u:ident, to_c: $to_c:ident,
        bytes: ($to_b:ident, $from_b:ident)
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl BitshiftRegister for $reg {
                const HAS_TRUE_SHIFTV: bool = true;
                const HAS_WIDE_BYTE_SHIFTS: bool = true;

                fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
                    unsafe { arch::[<vshlq_ $s>](value, arch::[<vdupq_n_ $cs>](shift as $ce)) }
                }

                fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
                    // logical: unsigned view, negated count
                    unsafe {
                        arch::$from_u(arch::[<vshlq_ $us>](
                            arch::$to_u(value),
                            arch::[<vdupq_n_ $cs>](-(shift as $ce)),
                        ))
                    }
                }

                // Counts are passed straight through: `vshlq_*` reads the low 8
                // bits of each lane as a signed value, so an out-of-range count
                // produces an unspecified result (a count of 256 reads as 0 and
                // leaves the value unshifted, and a count whose low byte is negative
                // shifts the other way). Clamping would cost a splat and a
                // `vmin` on every shift to tidy up input the contract does not
                // promise anything about (see `BitshiftVector::shrv`).
                fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                    unsafe { arch::[<vshlq_ $s>](value, arch::$to_c(shifts)) }
                }

                fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                    unsafe {
                        arch::$from_u(arch::[<vshlq_ $us>](
                            arch::$to_u(value),
                            arch::[<vnegq_ $cs>](arch::$to_c(shifts)),
                        ))
                    }
                }

                fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_b(arch::neon_bshli_u8x16::<IMM8>(arch::$to_b(value))) }
                }

                fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_b(arch::neon_bshri_u8x16::<IMM8>(arch::$to_b(value))) }
                }

                // Constant rotates fuse into SHL + SRI (2 instructions) instead
                // of the trait default's SHL + USHR + ORR. A rotate is a pure
                // bit operation, so both run on the unsigned view. See
                // `polyfills/bits.rs` for the immediate ranges and the `n == 0`
                // identity case.
                fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_roli_ $us>]::<IMM8>(arch::$to_u(value))) }
                }

                fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_rori_ $us>]::<IMM8>(arch::$to_u(value))) }
                }

                fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_bitrev_ $us>](arch::$to_u(value))) }
                }
            }
        }
    };
}

/// IntegerRegister for 8/16/32-bit lanes (native mul, saturating ops, popcnt,
/// clz; ctz via `rbit`).
macro_rules! neon_int_register {
    ($reg:ty, suffix: $s:ident, unsigned: $us:ident, to_u: $to_u:ident, from_u: $from_u:ident, div: $div:ident) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl IntegerRegister for $reg {
                fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::[<neon_mulhi_ $s>](lhs, rhs)
                }

                fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vmulq_ $s>](lhs, rhs) }
                }

                fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vqaddq_ $s>](lhs, rhs) }
                }

                fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vqsubq_ $s>](lhs, rhs) }
                }

                fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
                    arch::[<div_ $div>]::<Self>(value, divider.multiplier(), divider.shift())
                }

                fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
                    arch::[<div_ $div _bf>]::<Self>(value, divider.multiplier(), divider.shift())
                }

                fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
                    arch::[<divv_ $div _bf>]::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
                }

                const HAS_HARDWARE_POPCNT: bool = true;

                fn count_ones(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_popcnt_ $us>](arch::$to_u(value))) }
                }

                fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_clz_ $us>](arch::$to_u(value))) }
                }

                fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::[<neon_ctz_ $us>](arch::$to_u(value))) }
                }
            }
        }
    };
}

/// NumericRegister + SignedRegister + FloatRegister for the float registers.
macro_rules! neon_float_register {
    (
        $reg:ty, elem: $e:ty, lanes: $n:tt, suffix: $s:ident, from_u: $from_u:ident,
        bits: $bits:ty, signed_bits: $sbits:ty, extended: $ext:ty,
        exp_mask: $exp_mask:expr, approx: $approx:tt
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl NumericRegister for $reg {
                sort_via_network!($n);

                const ZERO: Storage<Self> = reg::<Self, $n>([0.0; $n]);
                const ONE: Storage<Self> = reg::<Self, $n>([1.0; $n]);
                const TWO: Storage<Self> = reg::<Self, $n>([2.0; $n]);

                const MIN: Storage<Self> = reg::<Self, $n>([<$e>::MIN; $n]);
                const MAX: Storage<Self> = reg::<Self, $n>([<$e>::MAX; $n]);

                fn min_element(value: Storage<Self>) -> Self::Element {
                    cfg_select! {
                        feature = "strict_ieee754" => {
                            unsafe { arch::[<vminnmvq_ $s>](value) }
                        }
                        _ => unsafe { arch::[<vminvq_ $s>](value) },
                    }
                }

                fn max_element(value: Storage<Self>) -> Self::Element {
                    cfg_select! {
                        feature = "strict_ieee754" => {
                            unsafe { arch::[<vmaxnmvq_ $s>](value) }
                        }
                        _ => unsafe { arch::[<vmaxvq_ $s>](value) },
                    }
                }

                fn sum_elements(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vaddvq_ $s>](value) }
                }

                fn prod_elements(value: Storage<Self>) -> Self::Element {
                    neon_mul_reduce!([<vmulq_ $s>], [<vextq_ $s>], [<vgetq_lane_ $s>], value; $n)
                }

                fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vpaddq_ $s>](lo, hi) }
                }

                fn offset() -> Storage<Self> {
                    Self::splat(<Self::Lanes as Unsigned>::USIZE as $e)
                }

                fn indexed() -> Storage<Self> {
                    const INDEXED: Storage<$reg> = {
                        let mut a = [0.0 as $e; $n];
                        let mut i = 0;
                        while i < $n {
                            a[i] = i as $e;
                            i += 1;
                        }
                        reg::<$reg, $n>(a)
                    };
                    INDEXED
                }

                fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vaddq_ $s>](lhs, rhs) }
                }

                fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vsubq_ $s>](lhs, rhs) }
                }

                fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vmulq_ $s>](lhs, rhs) }
                }

                fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vdivq_ $s>](lhs, rhs) }
                }

                fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
                }

                fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    cfg_select! {
                        // IEEE 754 minimumNumber (NaN yields the other operand),
                        // matching the scalar oracle's `f32::min`.
                        feature = "strict_ieee754" => {
                            unsafe { arch::[<vminnmq_ $s>](lhs, rhs) }
                        }
                        _ => unsafe { arch::[<vminq_ $s>](lhs, rhs) },
                    }
                }

                fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    cfg_select! {
                        feature = "strict_ieee754" => {
                            unsafe { arch::[<vmaxnmq_ $s>](lhs, rhs) }
                        }
                        _ => unsafe { arch::[<vmaxq_ $s>](lhs, rhs) },
                    }
                }
            }

            #[thermite_macros::inline_always]
            impl SignedRegister for $reg {
                const NEG_ONE: Storage<Self> = reg::<Self, $n>([-1.0; $n]);
                const MIN_POSITIVE: Storage<Self> = reg::<Self, $n>([<$e>::MIN_POSITIVE; $n]);

                fn neg(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vnegq_ $s>](value) }
                }

                fn abs(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vabsq_ $s>](value) }
                }

                fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    // (lhs & !(-0.0)) | (-0.0 & rhs): lhs's magnitude, rhs's sign
                    Self::bitor(Self::bitandnot(lhs, Self::NEG_ZERO), Self::bitand(Self::NEG_ZERO, rhs))
                }

                fn signum(value: Storage<Self>) -> Storage<Self> {
                    Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
                }

                fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                    Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
                }
            }

            #[thermite_macros::inline_always]
            impl FloatRegister for $reg {
                // AdvSIMD FMLA is a true fused multiply-add.
                const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

                type Bits = $bits;
                type SignedBits = $sbits;
                type ExtendedPrecision = $ext;

                const HALF: Storage<Self> = reg::<Self, $n>([0.5; $n]);
                const NEG_ZERO: Storage<Self> = reg::<Self, $n>([-0.0; $n]);
                const EPSILON: Storage<Self> = reg::<Self, $n>([<$e>::EPSILON; $n]);
                const INFINITY: Storage<Self> = reg::<Self, $n>([<$e>::INFINITY; $n]);
                const NEG_INFINITY: Storage<Self> = reg::<Self, $n>([<$e>::NEG_INFINITY; $n]);
                const NAN: Storage<Self> = reg::<Self, $n>([<$e>::NAN; $n]);

                const EXP_MASK: Storage<Self::Bits> = reg::<$bits, $n>([$exp_mask; $n]);

                // vfmaq(acc, a, b) = acc + a * b; vfmsq(acc, a, b) = acc - a * b.
                fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vfmaq_ $s>](acc, lhs, rhs) }
                }

                fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    // lhs * rhs - acc = -(acc - lhs * rhs)
                    unsafe { arch::[<vnegq_ $s>](arch::[<vfmsq_ $s>](acc, lhs, rhs)) }
                }

                fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vfmsq_ $s>](acc, lhs, rhs) }
                }

                fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    // -(lhs * rhs) - acc = -(acc + lhs * rhs)
                    unsafe { arch::[<vnegq_ $s>](arch::[<vfmaq_ $s>](acc, lhs, rhs)) }
                }

                // With true FMA the estimating forms are the fused forms.
                fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    Self::mul_add(lhs, rhs, acc)
                }

                fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    Self::mul_sub(lhs, rhs, acc)
                }

                fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    Self::nmul_add(lhs, rhs, acc)
                }

                fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                    Self::nmul_sub(lhs, rhs, acc)
                }

                fn sqrt(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vsqrtq_ $s>](value) }
                }

                // Absolute-compare instructions do classify in ONE op (the
                // defaults are abs + compare). FACGE/FACLT are IEEE unordered
                // compares: NaN operands yield false, exactly matching
                // is_infinite(NaN) = false / is_finite(NaN) = false.
                fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {
                    // |v| >= inf can only hold for |v| == inf
                    unsafe { arch::$from_u(arch::[<vcageq_ $s>](value, Self::INFINITY)) }
                }

                fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {
                    unsafe { arch::$from_u(arch::[<vcaltq_ $s>](value, Self::INFINITY)) }
                }

                neon_approx_recip!($approx, $s);

                fn floor(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vrndmq_ $s>](value) }
                }

                fn ceil(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vrndpq_ $s>](value) }
                }

                fn round(value: Storage<Self>) -> Storage<Self> {
                    // round-half-to-even, matching x86 `roundps` / wasm `nearest`
                    unsafe { arch::[<vrndnq_ $s>](value) }
                }

                fn trunc(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vrndq_ $s>](value) }
                }

                const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
            }
        }
    };
}

/// SignedRegister + SignedIntegerRegister for the signed integer registers
/// (native `vnegq`/`vabsq` on all four widths - aarch64 includes the 64-bit
/// forms - and `vcltzq`/`vcgezq` sign tests).
macro_rules! neon_signed_int {
    (
        $reg:ty, lanes: $n:tt, suffix: $s:ident, count: $ce:ty, from_u: $from_u:ident, to_c: $to_c:ident
        $(, signed_extras: { $($sx:tt)* })?
        $(, extras: { $($extras:tt)* })?
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl SignedRegister for $reg {
                const NEG_ONE: Storage<Self> = reg::<Self, $n>([-1; $n]);
                const MIN_POSITIVE: Storage<Self> = reg::<Self, $n>([1; $n]);

                fn neg(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vnegq_ $s>](value) }
                }

                fn abs(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vabsq_ $s>](value) }
                }

                fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
                    unsafe { arch::$from_u(arch::[<vcltzq_ $s>](value)) }
                }

                fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
                    unsafe { arch::$from_u(arch::[<vcgezq_ $s>](value)) }
                }

                $($($sx)*)?
            }

            #[thermite_macros::inline_always]
            impl SignedIntegerRegister for $reg {
                fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
                    unsafe { arch::[<vshlq_ $s>](value, arch::[<vdupq_n_ $s>](-(shift as $ce))) }
                }

                fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                    unsafe { arch::[<vshlq_ $s>](value, arch::[<vnegq_ $s>](arch::$to_c(shifts))) }
                }

                $($($extras)*)?
            }
        }
    };
}

/// UnsignedIntegerRegister for 8/16/32-bit lanes: native rounding/truncating
/// halving adds (`vrhaddq`/`vhaddq`) and absolute difference (`vabdq`), each a
/// single instruction where the defaults take three ops.
macro_rules! neon_unsigned_int {
    (
        $reg:ty, suffix: $s:ident, neg: ($ss:ident, $to_s:ident)
        $(, extras: { $($extras:tt)* })?
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl UnsignedIntegerRegister for $reg {
                fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vrhaddq_ $s>](a, b) }
                }

                fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vabdq_ $s>](a, b) }
                }

                // `!0 >> clz(v)`: 3 instructions (CLZ, NEG, USHL) vs the
                // default's log2(W)-step shift-or cascade. `v == 0` falls out
                // naturally: clz = W, and NEON logical shifts by >= W yield 0.
                fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {
                    unsafe {
                        let clz = arch::[<neon_clz_ $s>](value);
                        arch::[<vshlq_ $s>](Self::MAX, arch::[<vnegq_ $ss>](arch::$to_s(clz)))
                    }
                }

                // Per-lane bit width: `W - clz(v)`, 2 instructions (the default
                // composes popcount(next_power_of_two_m1(v)), ~7). `v == 0`
                // falls out: clz = W, so W - W = 0.
                fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
                    unsafe {
                        let w = Self::splat((core::mem::size_of::<<Self as Register>::Element>() * 8) as _);
                        arch::[<vsubq_ $s>](w, arch::[<neon_clz_ $s>](value))
                    }
                }

                $($($extras)*)?
            }
        }
    };
}

/// NumericRegister + IntegerRegister for the 64-bit integer registers. NEON
/// has no 64-bit lane multiply or min/max: multiply decomposes into 32x32
/// partials (`neon_mullo_u64`), min/max are compare+select (native 64-bit
/// compares exist on aarch64), `mulhi` is scalar 128-bit math on two lanes,
/// and the element reductions are two-lane extracts.
macro_rules! neon_int64_register {
    (
        $reg:ty, elem: $e:ty, suffix: $s:ident, wide: $w:ty,
        minmax: ($min:ident, $max:ident), to_u: $to_u:ident, from_u: $from_u:ident, div: $div:ident
    ) => {
        paste::paste! {
            #[thermite_macros::inline_always]
            impl NumericRegister for $reg {
                sort_via_network!(2); // i64x2 / u64x2

                const ZERO: Storage<Self> = reg::<Self, 2>([0; 2]);
                const ONE: Storage<Self> = reg::<Self, 2>([1; 2]);
                const TWO: Storage<Self> = reg::<Self, 2>([2; 2]);

                const MIN: Storage<Self> = reg::<Self, 2>([<$e>::MIN; 2]);
                const MAX: Storage<Self> = reg::<Self, 2>([<$e>::MAX; 2]);

                fn is_all_zero(value: Storage<Self>) -> bool {
                    !arch::[<neon_mask_any_ $s>](value)
                }

                fn min_element(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vgetq_lane_ $s>]::<0>(value).min(arch::[<vgetq_lane_ $s>]::<1>(value)) }
                }

                fn max_element(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vgetq_lane_ $s>]::<0>(value).max(arch::[<vgetq_lane_ $s>]::<1>(value)) }
                }

                fn sum_elements(value: Storage<Self>) -> Self::Element {
                    unsafe { arch::[<vaddvq_ $s>](value) }
                }

                fn prod_elements(value: Storage<Self>) -> Self::Element {
                    unsafe {
                        arch::[<vgetq_lane_ $s>]::<0>(value).wrapping_mul(arch::[<vgetq_lane_ $s>]::<1>(value))
                    }
                }

                fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vpaddq_ $s>](lo, hi) }
                }

                fn offset() -> Storage<Self> {
                    Self::splat(<Self::Lanes as Unsigned>::USIZE as $e)
                }

                fn indexed() -> Storage<Self> {
                    const INDEXED: Storage<$reg> = reg::<$reg, 2>([0, 1]);
                    INDEXED
                }

                fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vaddq_ $s>](lhs, rhs) }
                }

                fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vsubq_ $s>](lhs, rhs) }
                }

                fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    // sign-agnostic 32x32 partial-product decomposition
                    unsafe { arch::$from_u(arch::neon_mullo_u64(arch::$to_u(lhs), arch::$to_u(rhs))) }
                }

                fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
                }

                fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
                }

                fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::$min(lhs, rhs)
                }

                fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    arch::$max(lhs, rhs)
                }
            }

            #[thermite_macros::inline_always]
            impl IntegerRegister for $reg {
                fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe {
                        let a0 = arch::[<vgetq_lane_ $s>]::<0>(lhs);
                        let a1 = arch::[<vgetq_lane_ $s>]::<1>(lhs);
                        let b0 = arch::[<vgetq_lane_ $s>]::<0>(rhs);
                        let b1 = arch::[<vgetq_lane_ $s>]::<1>(rhs);

                        let r = [
                            (((a0 as $w) * (b0 as $w)) >> 64) as $e,
                            (((a1 as $w) * (b1 as $w)) >> 64) as $e,
                        ];
                        arch::[<vld1q_ $s>](r.as_ptr())
                    }
                }

                fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    <Self as NumericRegister>::mul(lhs, rhs)
                }

                fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vqaddq_ $s>](lhs, rhs) }
                }

                fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::[<vqsubq_ $s>](lhs, rhs) }
                }

                fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
                    arch::[<div_ $div>]::<Self>(value, divider.multiplier(), divider.shift())
                }

                fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
                    arch::[<div_ $div _bf>]::<Self>(value, divider.multiplier(), divider.shift())
                }

                fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
                    arch::[<divv_ $div _bf>]::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
                }

                const HAS_HARDWARE_POPCNT: bool = true;

                fn count_ones(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::neon_popcnt_u64(arch::$to_u(value))) }
                }

                fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::neon_clz_u64(arch::$to_u(value))) }
                }

                fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
                    unsafe { arch::$from_u(arch::neon_ctz_u64(arch::$to_u(value))) }
                }
            }
        }
    };
}

/// `ExtendRegister<scalar>` - the bottom rung of the sub-native width ladder:
/// a 1-lane scalar "register" extends into lane 0.
macro_rules! neon_extend_scalar {
    ($reg:ty, elem: $e:ty) => {
        impl crate::register::ExtendRegister<$e> for $reg {
            #[inline(always)]
            fn extend(value: Storage<$e>) -> Storage<Self> {
                Self::single(value)
            }

            #[inline(always)]
            fn narrow(value: Storage<Self>) -> Storage<$e> {
                Self::extract::<0>(value)
            }
        }
    };
}

/// Widen/narrow casts between a native register and the double-width
/// `ArrayRegister` pair: `vmovl`/`vmovl_high` widen, `vmovn` truncates
/// (wrapping, Rust `as`), `vqmovn` saturates.
macro_rules! neon_widen_casts {
    ($narrow_reg:ty => [$wide_reg:ty; 2], suffixes: $s:ident/$w:ident) => {
        paste::paste! {
            impl crate::register::CastRegister<$narrow_reg> for ArrayRegister<$wide_reg, 2> {
                #[inline(always)]
                fn cast_from(value: Storage<$narrow_reg>) -> Storage<Self> {
                    unsafe {
                        ArrayRegister([
                            arch::[<vmovl_ $s>](arch::[<vget_low_ $s>](value)),
                            arch::[<vmovl_high_ $s>](value),
                        ])
                    }
                }
            }

            // `vmovn` truncates and `vqmovn` saturates, so this pair is one of the
            // few where both strengths are a distinct single instruction.
            impl crate::register::CastRegister<ArrayRegister<$wide_reg, 2>> for $narrow_reg {
                #[inline(always)]
                fn cast_from(value: Storage<ArrayRegister<$wide_reg, 2>>) -> Storage<Self> {
                    unsafe { arch::[<vmovn_high_ $w>](arch::[<vmovn_ $w>](value.0[0]), value.0[1]) }
                }

                #[inline(always)]
                fn saturating_cast_from(value: Storage<ArrayRegister<$wide_reg, 2>>) -> Storage<Self> {
                    unsafe { arch::[<vqmovn_high_ $w>](arch::[<vqmovn_ $w>](value.0[0]), value.0[1]) }
                }
            }
        }
    };
}

/// `ConcatRegister<scalar>` for the 2-lane registers: a q-register is the
/// concatenation of two scalar "1-lane registers".
macro_rules! neon_concat_scalar2 {
    ($reg:ty, elem: $e:ty, suffix: $s:ident) => {
        paste::paste! {
            impl crate::register::ConcatRegister<$e> for $reg {
                #[inline(always)]
                fn concat(lo: Storage<$e>, hi: Storage<$e>) -> Storage<Self> {
                    unsafe {
                        arch::[<vsetq_lane_ $s>]::<1>(hi, arch::[<vsetq_lane_ $s>]::<0>(lo, Self::EMPTY))
                    }
                }

                #[inline(always)]
                fn split(value: Storage<Self>) -> (Storage<$e>, Storage<$e>) {
                    unsafe {
                        (
                            arch::[<vgetq_lane_ $s>]::<0>(value),
                            arch::[<vgetq_lane_ $s>]::<1>(value),
                        )
                    }
                }
            }
        }
    };
}

/// `broadcast` (single instruction `DUP Vd.T, Vn.T[lane]` - the trait default
/// round-trips through a GPR via `splat(extract(v))`) and `align` (single
/// instruction `EXT` - the trait default lowers to two `TBL`s + `BSL`).
/// Match arms collapse at monomorphization.
macro_rules! neon_broadcast_align {
    ($dup:ident, $ext:ident; 2) => {
        #[inline(always)]
        fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$dup::<0>(value),
                    _ => arch::$dup::<1>(value),
                }
            }
        }

        const HAS_NATIVE_ALIGN: bool = true;

        #[inline(always)]
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                match OFFSET {
                    0 => a,
                    1 => arch::$ext::<1>(a, b),
                    2 => b,
                    _ => Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
                }
            }
        }
    };
    ($dup:ident, $ext:ident; 4) => {
        #[inline(always)]
        fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$dup::<0>(value),
                    1 => arch::$dup::<1>(value),
                    2 => arch::$dup::<2>(value),
                    _ => arch::$dup::<3>(value),
                }
            }
        }

        const HAS_NATIVE_ALIGN: bool = true;

        #[inline(always)]
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                match OFFSET {
                    0 => a,
                    1 => arch::$ext::<1>(a, b),
                    2 => arch::$ext::<2>(a, b),
                    3 => arch::$ext::<3>(a, b),
                    4 => b,
                    _ => Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
                }
            }
        }
    };
    ($dup:ident, $ext:ident; 8) => {
        #[inline(always)]
        fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$dup::<0>(value),
                    1 => arch::$dup::<1>(value),
                    2 => arch::$dup::<2>(value),
                    3 => arch::$dup::<3>(value),
                    4 => arch::$dup::<4>(value),
                    5 => arch::$dup::<5>(value),
                    6 => arch::$dup::<6>(value),
                    _ => arch::$dup::<7>(value),
                }
            }
        }

        const HAS_NATIVE_ALIGN: bool = true;

        #[inline(always)]
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                match OFFSET {
                    0 => a,
                    1 => arch::$ext::<1>(a, b),
                    2 => arch::$ext::<2>(a, b),
                    3 => arch::$ext::<3>(a, b),
                    4 => arch::$ext::<4>(a, b),
                    5 => arch::$ext::<5>(a, b),
                    6 => arch::$ext::<6>(a, b),
                    7 => arch::$ext::<7>(a, b),
                    8 => b,
                    _ => Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
                }
            }
        }
    };
    ($dup:ident, $ext:ident; 16) => {
        #[inline(always)]
        fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
            unsafe {
                match I {
                    0 => arch::$dup::<0>(value),
                    1 => arch::$dup::<1>(value),
                    2 => arch::$dup::<2>(value),
                    3 => arch::$dup::<3>(value),
                    4 => arch::$dup::<4>(value),
                    5 => arch::$dup::<5>(value),
                    6 => arch::$dup::<6>(value),
                    7 => arch::$dup::<7>(value),
                    8 => arch::$dup::<8>(value),
                    9 => arch::$dup::<9>(value),
                    10 => arch::$dup::<10>(value),
                    11 => arch::$dup::<11>(value),
                    12 => arch::$dup::<12>(value),
                    13 => arch::$dup::<13>(value),
                    14 => arch::$dup::<14>(value),
                    _ => arch::$dup::<15>(value),
                }
            }
        }

        const HAS_NATIVE_ALIGN: bool = true;

        #[inline(always)]
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            unsafe {
                match OFFSET {
                    0 => a,
                    1 => arch::$ext::<1>(a, b),
                    2 => arch::$ext::<2>(a, b),
                    3 => arch::$ext::<3>(a, b),
                    4 => arch::$ext::<4>(a, b),
                    5 => arch::$ext::<5>(a, b),
                    6 => arch::$ext::<6>(a, b),
                    7 => arch::$ext::<7>(a, b),
                    8 => arch::$ext::<8>(a, b),
                    9 => arch::$ext::<9>(a, b),
                    10 => arch::$ext::<10>(a, b),
                    11 => arch::$ext::<11>(a, b),
                    12 => arch::$ext::<12>(a, b),
                    13 => arch::$ext::<13>(a, b),
                    14 => arch::$ext::<14>(a, b),
                    15 => arch::$ext::<15>(a, b),
                    16 => b,
                    _ => Self::swizzle_const::<crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
                }
            }
        }
    };
}

/// `rcp`/`rsqrt` + their capability flags.
///
/// The trait's contract is that `HAS_APPROX_RCP == false` means `rcp()` is
/// EXACT - generic code (`SpecializedCoreMath::reciprocal`) skips its
/// Newton-refinement step when the flag is false, and applies exactly ONE step
/// when it is true. That single step is calibrated for f32: an ~8-bit
/// `FRECPE` estimate plus our baked step gives ~16 bits, and one more Newton
/// doubling reaches ~32 - enough for f32's 24-bit mantissa, but 19 bits SHORT
/// of f64's 53.
///
/// So only the f32 registers advertise the approximation (`approx: yes`); the
/// f64 ones (`approx: no`) leave `rcp`/`rsqrt` at their exact trait defaults
/// (`1/x`, `1/sqrt(x)`), exactly as every x86 and wasm f64 register does -
/// none of those ISAs even has a packed-f64 reciprocal estimate. NEON does
/// (`FRECPE.2D`), and wiring it in naively silently cost 19 bits of f64
/// precision that the differential suite's 1e-6 tolerance could not see.
///
/// `strict_ieee754` disables the approximation entirely, matching x86.
macro_rules! neon_approx_recip {
    (yes, $s:ident) => {
        paste::paste! {
            const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));
            const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));

            // `vrecpe`/`vrsqrte` + one fused Newton step (see polyfills/math.rs).
            #[inline(always)]
            fn rcp(value: Storage<Self>) -> Storage<Self> {
                cfg_select! {
                    feature = "strict_ieee754" => {
                        Self::div(Self::ONE, value)
                    }
                    _ => arch::[<neon_rcp_ $s>](value),
                }
            }

            #[inline(always)]
            fn rsqrt(value: Storage<Self>) -> Storage<Self> {
                cfg_select! {
                    feature = "strict_ieee754" => {
                        Self::rcp(Self::sqrt(value))
                    }
                    _ => arch::[<neon_rsqrt_ $s>](value),
                }
            }
        }
    };
    (no, $s:ident) => {
        // No override: `rcp`/`rsqrt` keep the exact trait defaults.
        const HAS_APPROX_RCP: bool = false;
        const HAS_APPROX_RSQRT: bool = false;
    };
}

/// Emit the NEON overrides of
/// [`Register::widen_index_bytes`](crate::register::Register::widen_index_bytes)
/// and [`Register::permutev_row`](crate::register::Register::permutev_row):
/// widen the register's `LANES`-byte index array into the unsigned index
/// register via the `vmovl` ladder (`u8 -> u16 -> u32 -> u64`), stopping at the
/// register's own lane width. The ladder already produces a register, so
/// nothing is stored back out.
///
/// Shape tag is the lane count, `$sfx` the `neon_tbl_*` element suffix. Invoke
/// inside the register's own `impl Register` block, where `arch` is in scope.
/// Each `@body` load is exactly `LANES` bytes wide, since the argument is only
/// that wide, while `@row` keeps the 8-byte table row it is handed.
#[rustfmt::skip]
macro_rules! impl_widen_index_bytes_neon {
    ($shape:ident, $sfx:ident) => {
        #[inline(always)]
        fn widen_index_bytes(
            bytes: &generic_array::GenericArray<u8, <Self as $crate::register::CoreRegister>::Lanes>,
        ) -> $crate::register::Storage<<Self as $crate::register::Register>::Unsigned> {
            unsafe { impl_widen_index_bytes_neon!(@body bytes, $shape) }
        }

        // Straight from the byte row: no widen, no narrow, no clamp.
        #[inline(always)]
        fn permutev_row(
            value: $crate::register::Storage<Self>,
            row: &generic_array::GenericArray<u8, generic_array::typenum::U8>,
        ) -> $crate::register::Storage<Self> {
            paste::paste! {
                unsafe { arch::[<neon_tbl_ $sfx>](value, impl_widen_index_bytes_neon!(@row row, $shape)) }
            }
        }
    };

    (@row $row:ident, x2) => { arch::neon_lane_table_row::<2>($row.as_ptr()) };
    (@row $row:ident, x4) => { arch::neon_lane_table_row::<4>($row.as_ptr()) };
    (@row $row:ident, x8) => { arch::neon_lane_table_row::<8>($row.as_ptr()) };

    // 8 lanes of u16: one `vmovl` off the 8-byte index array.
    (@body $idxs:ident, x8) => {{
        arch::vmovl_u8(arch::vld1_u8($idxs.as_ptr()))
    }};

    // 4 lanes of u32: two rungs, taking the low half each time. Only four bytes
    // are readable, so the `d` register is built from a scalar `u32` load.
    (@body $idxs:ident, x4) => {{
        let lo = core::ptr::read_unaligned($idxs.as_ptr() as *const u32) as u64;
        let w16 = arch::vmovl_u8(arch::vreinterpret_u8_u32(arch::vcreate_u32(lo)));
        arch::vmovl_u16(arch::vget_low_u16(w16))
    }};

    // 2 lanes of u64: three rungs, off a two-byte scalar load.
    (@body $idxs:ident, x2) => {{
        let lo = core::ptr::read_unaligned($idxs.as_ptr() as *const u16) as u64;
        let w16 = arch::vmovl_u8(arch::vreinterpret_u8_u64(arch::vcreate_u64(lo)));
        let w32 = arch::vmovl_u16(arch::vget_low_u16(w16));
        arch::vmovl_u32(arch::vget_low_u32(w32))
    }};
}
