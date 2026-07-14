use super::*;

// ===========================================================================
// Unsigned base layer: ops NEON only spells one way (or not at all), written
// once per unsigned width and reused by the per-type layer below.
// ===========================================================================

/// Bitwise NOT. `vmvnq` has no 64-bit form, so `u64` routes through a `u32`
/// view (bitwise ops are width-agnostic).
#[inline(always)]
pub fn neon_not_u8(v: uint8x16_t) -> uint8x16_t {
    unsafe { vmvnq_u8(v) }
}

#[inline(always)]
pub fn neon_not_u16(v: uint16x8_t) -> uint16x8_t {
    unsafe { vmvnq_u16(v) }
}

#[inline(always)]
pub fn neon_not_u32(v: uint32x4_t) -> uint32x4_t {
    unsafe { vmvnq_u32(v) }
}

#[inline(always)]
pub fn neon_not_u64(v: uint64x2_t) -> uint64x2_t {
    unsafe { vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v))) }
}

/// Smear the sign (most significant) bit of each lane across the whole lane
/// via an arithmetic shift on the signed view.
#[inline(always)]
pub fn neon_msb_smear_u8(v: uint8x16_t) -> uint8x16_t {
    unsafe { vreinterpretq_u8_s8(vshrq_n_s8::<7>(vreinterpretq_s8_u8(v))) }
}

#[inline(always)]
pub fn neon_msb_smear_u16(v: uint16x8_t) -> uint16x8_t {
    unsafe { vreinterpretq_u16_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(v))) }
}

#[inline(always)]
pub fn neon_msb_smear_u32(v: uint32x4_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(v))) }
}

#[inline(always)]
pub fn neon_msb_smear_u64(v: uint64x2_t) -> uint64x2_t {
    unsafe { vreinterpretq_u64_s64(vshrq_n_s64::<63>(vreinterpretq_s64_u64(v))) }
}

// ---------------------------------------------------------------------------
// Mask reductions. Lane masks are all-ones / all-zeros, so `vminvq`/`vmaxvq`
// across-vector reductions answer all/any in one instruction + compare. There
// is no 64-bit across-vector min/max; a `u32` view is equivalent because each
// 64-bit mask lane has identical halves.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_mask_all_u8(m: uint8x16_t) -> bool {
    unsafe { vminvq_u8(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_any_u8(m: uint8x16_t) -> bool {
    unsafe { vmaxvq_u8(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_all_u16(m: uint16x8_t) -> bool {
    unsafe { vminvq_u16(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_any_u16(m: uint16x8_t) -> bool {
    unsafe { vmaxvq_u16(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_all_u32(m: uint32x4_t) -> bool {
    unsafe { vminvq_u32(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_any_u32(m: uint32x4_t) -> bool {
    unsafe { vmaxvq_u32(m) != 0 }
}

#[inline(always)]
pub fn neon_mask_all_u64(m: uint64x2_t) -> bool {
    unsafe { neon_mask_all_u32(vreinterpretq_u32_u64(m)) }
}

#[inline(always)]
pub fn neon_mask_any_u64(m: uint64x2_t) -> bool {
    unsafe { neon_mask_any_u32(vreinterpretq_u32_u64(m)) }
}

// ---------------------------------------------------------------------------
// Movemask (x86 `pmovmskb`-style bitmask extraction). NEON has no direct
// equivalent; AND each lane with a distinct power of two, then a horizontal
// add collects the bits. The 8-bit form sums the two halves separately since
// the powers only fit a byte.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_movemask_u32(m: uint32x4_t) -> u64 {
    const BITS: uint32x4_t = cu32x4([1, 2, 4, 8]);
    unsafe { vaddvq_u32(vandq_u32(m, BITS)) as u64 }
}

#[inline(always)]
pub fn neon_movemask_u64(m: uint64x2_t) -> u64 {
    const BITS: uint64x2_t = cu64x2([1, 2]);
    unsafe { vaddvq_u64(vandq_u64(m, BITS)) }
}

#[inline(always)]
pub fn neon_movemask_u16(m: uint16x8_t) -> u64 {
    const BITS: uint16x8_t = cu16x8([1, 2, 4, 8, 16, 32, 64, 128]);
    unsafe { vaddvq_u16(vandq_u16(m, BITS)) as u64 }
}

#[inline(always)]
pub fn neon_movemask_u8(m: uint8x16_t) -> u64 {
    #[rustfmt::skip]
    const BITS: uint8x16_t = cu8x16([
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128,
    ]);
    unsafe {
        let masked = vandq_u8(m, BITS);
        (vaddv_u8(vget_low_u8(masked)) as u64) | ((vaddv_u8(vget_high_u8(masked)) as u64) << 8)
    }
}

// ---------------------------------------------------------------------------
// Population count: native `vcntq_u8`, widened per level with pairwise adds.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_popcnt_u8(v: uint8x16_t) -> uint8x16_t {
    unsafe { vcntq_u8(v) }
}

#[inline(always)]
pub fn neon_popcnt_u16(v: uint16x8_t) -> uint16x8_t {
    unsafe { vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(v))) }
}

#[inline(always)]
pub fn neon_popcnt_u32(v: uint32x4_t) -> uint32x4_t {
    unsafe { vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(v)))) }
}

#[inline(always)]
pub fn neon_popcnt_u64(v: uint64x2_t) -> uint64x2_t {
    unsafe { vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(v))))) }
}

// ---------------------------------------------------------------------------
// Leading zeros: native for 8/16/32-bit lanes; composed from `u32` halves for
// 64-bit (no `vclzq_u64`).
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_clz_u8(v: uint8x16_t) -> uint8x16_t {
    unsafe { vclzq_u8(v) }
}

#[inline(always)]
pub fn neon_clz_u16(v: uint16x8_t) -> uint16x8_t {
    unsafe { vclzq_u16(v) }
}

#[inline(always)]
pub fn neon_clz_u32(v: uint32x4_t) -> uint32x4_t {
    unsafe { vclzq_u32(v) }
}

#[inline(always)]
pub fn neon_clz_u64(v: uint64x2_t) -> uint64x2_t {
    unsafe {
        // Per 64-bit lane [lo: u32, hi: u32] (little-endian lane order):
        // clz64 = if hi == 0 { 32 + clz32(lo) } else { clz32(hi) }.
        let clz = vclzq_u32(vreinterpretq_u32_u64(v)); // [clz(lo0), clz(hi0), clz(lo1), clz(hi1)]
        let lo = vuzp1q_u32(clz, clz); // [clz(lo0), clz(lo1), ...]
        let hi = vuzp2q_u32(clz, clz); // [clz(hi0), clz(hi1), ...]
        let hi_zero = vceqq_u32(hi, vdupq_n_u32(32));
        let r = vbslq_u32(hi_zero, vaddq_u32(lo, vdupq_n_u32(32)), hi); // [r0, r1, ...]
        // widen [r0, r1] to u64 lanes by zipping with zero
        vreinterpretq_u64_u32(vzip1q_u32(r, vdupq_n_u32(0)))
    }
}

// ---------------------------------------------------------------------------
// Whole-lane bit reversal: `vrbitq` reverses bits within each byte; a byte
// reversal within the lane completes it. This is the NEON specialization win
// for `reverse_bits` (x86 needs a LUT cascade), and also powers
// `trailing_zeros = clz(bitrev(x))`.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_bitrev_u8(v: uint8x16_t) -> uint8x16_t {
    unsafe { vrbitq_u8(v) }
}

#[inline(always)]
pub fn neon_bitrev_u16(v: uint16x8_t) -> uint16x8_t {
    unsafe { vreinterpretq_u16_u8(vrbitq_u8(vrev16q_u8(vreinterpretq_u8_u16(v)))) }
}

#[inline(always)]
pub fn neon_bitrev_u32(v: uint32x4_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u8(vrbitq_u8(vrev32q_u8(vreinterpretq_u8_u32(v)))) }
}

#[inline(always)]
pub fn neon_bitrev_u64(v: uint64x2_t) -> uint64x2_t {
    unsafe { vreinterpretq_u64_u8(vrbitq_u8(vrev64q_u8(vreinterpretq_u8_u64(v)))) }
}

#[inline(always)]
pub fn neon_ctz_u8(v: uint8x16_t) -> uint8x16_t {
    neon_clz_u8(neon_bitrev_u8(v))
}

#[inline(always)]
pub fn neon_ctz_u16(v: uint16x8_t) -> uint16x8_t {
    neon_clz_u16(neon_bitrev_u16(v))
}

#[inline(always)]
pub fn neon_ctz_u32(v: uint32x4_t) -> uint32x4_t {
    neon_clz_u32(neon_bitrev_u32(v))
}

#[inline(always)]
pub fn neon_ctz_u64(v: uint64x2_t) -> uint64x2_t {
    neon_clz_u64(neon_bitrev_u64(v))
}

// ===========================================================================
// Per-type normalization layer: uniformly-named ops for every register type,
// so the register-impl macros can be written once against
// `arch::[<neon_<op>_ $suffix>]`. All `vreinterpretq` plumbing lives here and
// compiles to nothing.
// ===========================================================================

macro_rules! stamp_type_layer {
    ($(
        $s:ident: $ty:ty, lanes: $n:literal,
        unsigned: ($us:ident, $ue:ty, $to_u:ident, $from_u:ident),
        bytes: ($to_b:ident, $from_b:ident)
    );* $(;)?) => {$(paste::paste! {
        #[inline(always)]
        pub fn [<neon_and_ $s>](a: $ty, b: $ty) -> $ty {
            unsafe { $from_u([<vandq_ $us>]($to_u(a), $to_u(b))) }
        }

        #[inline(always)]
        pub fn [<neon_or_ $s>](a: $ty, b: $ty) -> $ty {
            unsafe { $from_u([<vorrq_ $us>]($to_u(a), $to_u(b))) }
        }

        #[inline(always)]
        pub fn [<neon_xor_ $s>](a: $ty, b: $ty) -> $ty {
            unsafe { $from_u([<veorq_ $us>]($to_u(a), $to_u(b))) }
        }

        /// `!a & b` - the trait's `bitandnot(lhs, rhs)` operand order (x86
        /// `andnot` convention: the FIRST operand is inverted). NEON `vbic(x, y)`
        /// computes `x & !y`, so the operands swap here.
        #[inline(always)]
        pub fn [<neon_andnot_ $s>](a: $ty, b: $ty) -> $ty {
            unsafe { $from_u([<vbicq_ $us>]($to_u(b), $to_u(a))) }
        }

        /// Bitwise select in `vbsl` operand order: `(mask & on_true) | (!mask & on_false)`.
        /// The mask is self-typed (Thermite's `Mask = Self` model).
        #[inline(always)]
        pub fn [<neon_bsl_ $s>](mask: $ty, on_true: $ty, on_false: $ty) -> $ty {
            unsafe { [<vbslq_ $s>]($to_u(mask), on_true, on_false) }
        }

        /// `vqtbl1q_u8` byte-table lookup viewed at this type. Out-of-range
        /// byte indices produce zero (wasm `swizzle` semantics).
        #[inline(always)]
        pub fn [<neon_tbl_ $s>](v: $ty, table: uint8x16_t) -> $ty {
            unsafe { $from_b(vqtbl1q_u8($to_b(v), table)) }
        }

        /// `vqtbl2q_u8` two-register byte-table lookup viewed at this type:
        /// byte indices 0-15 select from `a`, 16-31 from `b`, >= 32 yield zero.
        /// One TBL instruction replaces the generic two-permute + blend
        /// `swizzle` lowering.
        #[inline(always)]
        pub fn [<neon_tbl2_ $s>](a: $ty, b: $ty, table: uint8x16_t) -> $ty {
            unsafe { $from_b(vqtbl2q_u8(uint8x16x2_t($to_b(a), $to_b(b)), table)) }
        }

        /// Smear each lane's sign/MSB across the lane (`msb_to_mask`).
        #[inline(always)]
        pub fn [<neon_msb_mask_ $s>](v: $ty) -> $ty {
            unsafe { $from_u([<neon_msb_smear_ $us>]($to_u(v))) }
        }

        /// All-ones where the lane is non-zero (`into_mask`).
        #[inline(always)]
        pub fn [<neon_nonzero_mask_ $s>](v: $ty) -> $ty {
            unsafe { $from_u([<neon_not_ $us>]([<vceqzq_ $s>](v))) }
        }

        /// Reverse the order of lanes.
        #[inline(always)]
        pub fn [<neon_reverse_ $s>](v: $ty) -> $ty {
            const TABLE: uint8x16_t = {
                let elem = 16 / $n;
                let mut idxs = [0u32; $n];
                let mut i = 0;
                while i < $n {
                    idxs[i] = ($n - 1 - i) as u32;
                    i += 1;
                }
                neon_lane_table::<$n>(elem, idxs)
            };
            [<neon_tbl_ $s>](v, TABLE)
        }

        /// All-ones in lanes `< n`, zero above - the `zeroupper_z` keep-mask.
        /// `const fn` so it folds in an inline-const at the call site.
        #[inline(always)]
        pub const fn [<neon_keep_mask_ $s>](n: usize) -> $ty {
            let mut a = [0 as $ue; $n];
            let mut i = 0;
            while i < n && i < $n {
                a[i] = !0;
                i += 1;
            }
            // SAFETY: same size, alignment handled by const_transmute.
            unsafe { crate::generic_array::const_transmute(a) }
        }
    })*};
}

stamp_type_layer! {
    f32: float32x4_t, lanes: 4,
        unsigned: (u32, u32, vreinterpretq_u32_f32, vreinterpretq_f32_u32),
        bytes: (vreinterpretq_u8_f32, vreinterpretq_f32_u8);
    f64: float64x2_t, lanes: 2,
        unsigned: (u64, u64, vreinterpretq_u64_f64, vreinterpretq_f64_u64),
        bytes: (vreinterpretq_u8_f64, vreinterpretq_f64_u8);
    s8: int8x16_t, lanes: 16,
        unsigned: (u8, u8, vreinterpretq_u8_s8, vreinterpretq_s8_u8),
        bytes: (vreinterpretq_u8_s8, vreinterpretq_s8_u8);
    s16: int16x8_t, lanes: 8,
        unsigned: (u16, u16, vreinterpretq_u16_s16, vreinterpretq_s16_u16),
        bytes: (vreinterpretq_u8_s16, vreinterpretq_s16_u8);
    s32: int32x4_t, lanes: 4,
        unsigned: (u32, u32, vreinterpretq_u32_s32, vreinterpretq_s32_u32),
        bytes: (vreinterpretq_u8_s32, vreinterpretq_s32_u8);
    s64: int64x2_t, lanes: 2,
        unsigned: (u64, u64, vreinterpretq_u64_s64, vreinterpretq_s64_u64),
        bytes: (vreinterpretq_u8_s64, vreinterpretq_s64_u8);
    u8: uint8x16_t, lanes: 16,
        unsigned: (u8, u8, identity, identity),
        bytes: (identity, identity);
    u16: uint16x8_t, lanes: 8,
        unsigned: (u16, u16, identity, identity),
        bytes: (vreinterpretq_u8_u16, vreinterpretq_u16_u8);
    u32: uint32x4_t, lanes: 4,
        unsigned: (u32, u32, identity, identity),
        bytes: (vreinterpretq_u8_u32, vreinterpretq_u32_u8);
    u64: uint64x2_t, lanes: 2,
        unsigned: (u64, u64, identity, identity),
        bytes: (vreinterpretq_u8_u64, vreinterpretq_u64_u8);
}

// ---------------------------------------------------------------------------
// Per-lane byte swap (`swap_bytes`): `vrev{16,32,64}q_u8` on a byte view;
// 8-bit lanes are a no-op.
// ---------------------------------------------------------------------------

macro_rules! stamp_swap_bytes {
    ($($s:ident: $ty:ty => ($to_b:ident, $rev:ident, $from_b:ident)),* $(,)?) => {$(paste::paste! {
        #[inline(always)]
        pub fn [<neon_swap_bytes_ $s>](v: $ty) -> $ty {
            unsafe { $from_b($rev($to_b(v))) }
        }
    })*};
}

stamp_swap_bytes! {
    f32: float32x4_t => (vreinterpretq_u8_f32, vrev32q_u8, vreinterpretq_f32_u8),
    f64: float64x2_t => (vreinterpretq_u8_f64, vrev64q_u8, vreinterpretq_f64_u8),
    s16: int16x8_t => (vreinterpretq_u8_s16, vrev16q_u8, vreinterpretq_s16_u8),
    s32: int32x4_t => (vreinterpretq_u8_s32, vrev32q_u8, vreinterpretq_s32_u8),
    s64: int64x2_t => (vreinterpretq_u8_s64, vrev64q_u8, vreinterpretq_s64_u8),
    u16: uint16x8_t => (vreinterpretq_u8_u16, vrev16q_u8, vreinterpretq_u16_u8),
    u32: uint32x4_t => (vreinterpretq_u8_u32, vrev32q_u8, vreinterpretq_u32_u8),
    u64: uint64x2_t => (vreinterpretq_u8_u64, vrev64q_u8, vreinterpretq_u64_u8),
    s8: int8x16_t => (identity, identity, identity),
    u8: uint8x16_t => (identity, identity, identity),
}

// ---------------------------------------------------------------------------
// Whole-register byte shifts (x86 `pslldq`/`psrldq`): `vextq_u8` against a
// zero register, with a `match` collapsing the const shift to one arm.
// ---------------------------------------------------------------------------

/// Shift the whole 128-bit register left by `IMM8` bytes, zero-filling.
#[rustfmt::skip]
#[inline(always)]
pub fn neon_bshli_u8x16<const IMM8: i32>(v: uint8x16_t) -> uint8x16_t {
    unsafe {
        let z = vdupq_n_u8(0);
        // result = concat(zero, v)[16 - IMM8 ..], i.e. vextq(z, v, 16 - IMM8)
        match IMM8 {
            0  => v,
            1  => vextq_u8::<15>(z, v),
            2  => vextq_u8::<14>(z, v),
            3  => vextq_u8::<13>(z, v),
            4  => vextq_u8::<12>(z, v),
            5  => vextq_u8::<11>(z, v),
            6  => vextq_u8::<10>(z, v),
            7  => vextq_u8::<9>(z, v),
            8  => vextq_u8::<8>(z, v),
            9  => vextq_u8::<7>(z, v),
            10 => vextq_u8::<6>(z, v),
            11 => vextq_u8::<5>(z, v),
            12 => vextq_u8::<4>(z, v),
            13 => vextq_u8::<3>(z, v),
            14 => vextq_u8::<2>(z, v),
            15 => vextq_u8::<1>(z, v),
            _  => z,
        }
    }
}

/// Shift the whole 128-bit register right by `IMM8` bytes, zero-filling.
#[rustfmt::skip]
#[inline(always)]
pub fn neon_bshri_u8x16<const IMM8: i32>(v: uint8x16_t) -> uint8x16_t {
    unsafe {
        let z = vdupq_n_u8(0);
        // result = concat(v, zero)[IMM8 ..], i.e. vextq(v, z, IMM8)
        match IMM8 {
            0  => v,
            1  => vextq_u8::<1>(v, z),
            2  => vextq_u8::<2>(v, z),
            3  => vextq_u8::<3>(v, z),
            4  => vextq_u8::<4>(v, z),
            5  => vextq_u8::<5>(v, z),
            6  => vextq_u8::<6>(v, z),
            7  => vextq_u8::<7>(v, z),
            8  => vextq_u8::<8>(v, z),
            9  => vextq_u8::<9>(v, z),
            10 => vextq_u8::<10>(v, z),
            11 => vextq_u8::<11>(v, z),
            12 => vextq_u8::<12>(v, z),
            13 => vextq_u8::<13>(v, z),
            14 => vextq_u8::<14>(v, z),
            15 => vextq_u8::<15>(v, z),
            _  => z,
        }
    }
}

// The four ops whose unsigned-suffix names ARE the base layer above; stamped
// here only for the float/signed views.

macro_rules! stamp_mask_view_layer {
    ($($s:ident: $ty:ty => ($us:ident, $to_u:ident, $from_u:ident)),* $(,)?) => {$(paste::paste! {
        #[inline(always)]
        pub fn [<neon_not_ $s>](v: $ty) -> $ty {
            unsafe { $from_u([<neon_not_ $us>]($to_u(v))) }
        }

        #[inline(always)]
        pub fn [<neon_mask_all_ $s>](m: $ty) -> bool {
            unsafe { [<neon_mask_all_ $us>]($to_u(m)) }
        }

        #[inline(always)]
        pub fn [<neon_mask_any_ $s>](m: $ty) -> bool {
            unsafe { [<neon_mask_any_ $us>]($to_u(m)) }
        }

        #[inline(always)]
        pub fn [<neon_movemask_ $s>](m: $ty) -> u64 {
            unsafe { [<neon_movemask_ $us>]($to_u(m)) }
        }
    })*};
}

stamp_mask_view_layer! {
    f32: float32x4_t => (u32, vreinterpretq_u32_f32, vreinterpretq_f32_u32),
    f64: float64x2_t => (u64, vreinterpretq_u64_f64, vreinterpretq_f64_u64),
    s8: int8x16_t => (u8, vreinterpretq_u8_s8, vreinterpretq_s8_u8),
    s16: int16x8_t => (u16, vreinterpretq_u16_s16, vreinterpretq_s16_u16),
    s32: int32x4_t => (u32, vreinterpretq_u32_s32, vreinterpretq_s32_u32),
    s64: int64x2_t => (u64, vreinterpretq_u64_s64, vreinterpretq_s64_u64),
}

// ===========================================================================
// 2D Morton (Z-order) encode/decode via `vqtbl1q_u8` as a nibble LUT - the NEON
// port of the wasm SIMD128 path (backend/wasm/polyfills/bits.rs), itself the
// analogue of the x86 `pshufb` method (see backend/x86_v2/polyfills/bits.rs for
// the bit-algebra). `vqtbl1q_u8(table, indices)` matches wasm `i8x16.swizzle`:
// the table is the first operand and any index byte >= 16 yields 0, so the
// compress path's nibble-masking logic ports 1:1. There is no NEON carry-less
// multiply, so the u64 2D path and all N != 2 stay on the generic cascade.
//
// NOTE the operand roles: the `neon_tbl_*` helpers take (value, table) where the
// VALUE is looked up per lane; the Morton LUTs invert this - the LUT constant is
// the table and the data supplies the indices - so we call `vqtbl1q_u8` directly
// via the small `neon_morton_lut_*` helpers below rather than reusing `neon_tbl_*`.
// ===========================================================================

/// Nibble -> spread-by-1 byte LUT (`b3 b2 b1 b0 -> 0 b3 0 b2 0 b1 0 b0`).
#[rustfmt::skip]
const MORTON2_SPREAD_LUT: uint8x16_t = cu8x16([
    0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
]);
/// Nibble -> 2-bit compress LUT (gather bits 0 and 2: `i -> bit0(i) | bit2(i)<<1`).
const MORTON2_COMPRESS_LUT: uint8x16_t = cu8x16([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]);

/// Morton nibble-LUT lookup for u16 lanes: view the data as bytes to use as
/// `vqtbl1q_u8` indices into `lut`, then view the result back as u16. (Roles are
/// inverted vs. `neon_tbl_u16`: the LUT is the table, the data supplies indices.)
#[inline(always)]
fn neon_morton_lut_u16(indices: uint16x8_t, lut: uint8x16_t) -> uint16x8_t {
    unsafe { vreinterpretq_u16_u8(vqtbl1q_u8(lut, vreinterpretq_u8_u16(indices))) }
}

/// Morton nibble-LUT lookup for u32 lanes (see [`neon_morton_lut_u16`]).
#[inline(always)]
fn neon_morton_lut_u32(indices: uint32x4_t, lut: uint8x16_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u8(vqtbl1q_u8(lut, vreinterpretq_u8_u32(indices))) }
}

/// Spread the low 8 bits of each 16-bit lane by one (2D Morton).
#[inline(always)]
pub fn neon_morton2_spread_u16(v: uint16x8_t) -> uint16x8_t {
    const LOW: uint16x8_t = cu16x8([0x00FF; 8]);
    const NIB: uint16x8_t = cu16x8([0x0F0F; 8]);
    unsafe {
        let c = vandq_u16(v, LOW);
        let n = vandq_u16(vorrq_u16(c, vshlq_n_u16::<4>(c)), NIB);
        neon_morton_lut_u16(n, MORTON2_SPREAD_LUT) // n bytes are already in 0..=15
    }
}

/// Spread the low 16 bits of each 32-bit lane by one (2D Morton).
#[inline(always)]
pub fn neon_morton2_spread_u32(v: uint32x4_t) -> uint32x4_t {
    const LOW: uint32x4_t = cu32x4([0x0000_FFFF; 4]);
    const M1: uint32x4_t = cu32x4([0x00FF_00FF; 4]);
    const NIB: uint32x4_t = cu32x4([0x0F0F_0F0F; 4]);
    unsafe {
        let c = vandq_u32(v, LOW);
        let c = vandq_u32(vorrq_u32(c, vshlq_n_u32::<8>(c)), M1);
        let n = vandq_u32(vorrq_u32(c, vshlq_n_u32::<4>(c)), NIB);
        neon_morton_lut_u32(n, MORTON2_SPREAD_LUT)
    }
}

/// Per-16-bit-lane 2D Morton encode: low 8 bits of `x` (even) with `y` (odd).
#[inline(always)]
pub fn neon_morton2_u16(x: uint16x8_t, y: uint16x8_t) -> uint16x8_t {
    unsafe { vorrq_u16(neon_morton2_spread_u16(x), vshlq_n_u16::<1>(neon_morton2_spread_u16(y))) }
}

/// Per-32-bit-lane 2D Morton encode: low 16 bits of `x` (even) with `y` (odd).
#[inline(always)]
pub fn neon_morton2_u32(x: uint32x4_t, y: uint32x4_t) -> uint32x4_t {
    unsafe { vorrq_u32(neon_morton2_spread_u32(x), vshlq_n_u32::<1>(neon_morton2_spread_u32(y))) }
}

/// Compress the even bits of each 16-bit lane back to a contiguous low 8 bits -
/// the inverse of [`neon_morton2_spread_u16`].
#[inline(always)]
pub fn neon_morton2_compress_u16(v: uint16x8_t) -> uint16x8_t {
    const NIB: uint16x8_t = cu16x8([0x0F0F; 8]);
    const EVEN: uint16x8_t = cu16x8([0x5555; 8]);
    const LOW: uint16x8_t = cu16x8([0x00FF; 8]);
    unsafe {
        let e = vandq_u16(v, EVEN);
        // indices masked to a nibble (`vqtbl1q_u8` zeroes lanes whose index is >= 16)
        let lo = neon_morton_lut_u16(vandq_u16(e, NIB), MORTON2_COMPRESS_LUT);
        let hi = neon_morton_lut_u16(vandq_u16(vshrq_n_u16::<4>(e), NIB), MORTON2_COMPRESS_LUT);
        let n = vandq_u16(vorrq_u16(lo, vshlq_n_u16::<2>(hi)), NIB);
        vandq_u16(vorrq_u16(n, vshrq_n_u16::<4>(n)), LOW)
    }
}

/// Compress the even bits of each 32-bit lane back to a contiguous low 16 bits -
/// the inverse of [`neon_morton2_spread_u32`].
#[inline(always)]
pub fn neon_morton2_compress_u32(v: uint32x4_t) -> uint32x4_t {
    const NIB: uint32x4_t = cu32x4([0x0F0F_0F0F; 4]);
    const EVEN: uint32x4_t = cu32x4([0x5555_5555; 4]);
    const M1: uint32x4_t = cu32x4([0x00FF_00FF; 4]);
    const LOW: uint32x4_t = cu32x4([0x0000_FFFF; 4]);
    unsafe {
        let e = vandq_u32(v, EVEN);
        let lo = neon_morton_lut_u32(vandq_u32(e, NIB), MORTON2_COMPRESS_LUT);
        let hi = neon_morton_lut_u32(vandq_u32(vshrq_n_u32::<4>(e), NIB), MORTON2_COMPRESS_LUT);
        let n = vandq_u32(vorrq_u32(lo, vshlq_n_u32::<2>(hi)), NIB);
        let c = vandq_u32(vorrq_u32(n, vshrq_n_u32::<4>(n)), M1);
        vandq_u32(vorrq_u32(c, vshrq_n_u32::<8>(c)), LOW)
    }
}

// ---------------------------------------------------------------------------
// Multi-register byte-table lookup (TBL1/2/3/4).
//
// `vqtbl{1,2,3,4}q_u8` index a table of 1-4 *consecutive* q-registers (16-64
// bytes) with one byte index per output lane, zeroing any lane whose index is
// out of range. This is what powers `array_permutev`/`array_swizzle` on
// `ArrayRegister` chunks: an N-chunk array is exactly a 16*N-byte table, so a
// whole cross-chunk permute is ONE instruction per output chunk instead of the
// generic default's N permutes + N blends per output chunk.
//
// `chunks.len()` must be in `1..=4` (the caller guarantees it).
// ---------------------------------------------------------------------------

/// `N` is a const generic (not `chunks.len()`) so the arity selection folds at
/// monomorphization into a single `TBL` - passing a slice leaves a runtime
/// branch over all four forms.
#[inline(always)]
pub fn neon_tbl_n_u8<const N: usize>(t: [uint8x16_t; 4], idx: uint8x16_t) -> uint8x16_t {
    unsafe {
        match N {
            1 => vqtbl1q_u8(t[0], idx),
            2 => vqtbl2q_u8(uint8x16x2_t(t[0], t[1]), idx),
            3 => vqtbl3q_u8(uint8x16x3_t(t[0], t[1], t[2]), idx),
            _ => vqtbl4q_u8(uint8x16x4_t(t[0], t[1], t[2], t[3]), idx),
        }
    }
}

// ---------------------------------------------------------------------------
// Constant-amount bit rotates via SRI (shift-right-and-insert).
//
// `vsriq_n_u32::<N>(dst, src)` shifts each `src` lane right by `N` and inserts
// it into `dst`, KEEPING `dst`'s top `N` bits. So the two halves of a rotate
// fuse into two instructions instead of the trait default's three (SHL, USHR,
// ORR):
//
//   ror(x, n) = (x >> n) | (x << (W - n))
//             = vsriq_n::<n>(vshlq_n::<W - n>(x), x)   // SHL, SRI
//   rol(x, n) = ror(x, W - n)
//             = vsriq_n::<W - n>(vshlq_n::<n>(x), x)   // SHL, SRI
//
// The SRI keeps the top `n` bits of its `dst` operand - exactly the
// `x << (W - n)` term - and fills the low `W - n` bits with `x >> n`. The two
// bit ranges are disjoint and together cover the lane, so the insert IS the OR.
//
// Immediate ranges: `vshlq_n` requires `0..=W-1` and `vsriq_n` requires
// `1..=W`, so `n == 0` (the identity rotate) cannot be expressed and is
// special-cased to return the input. The trait's `IMM8` is an `i32` the caller
// may set to anything, so it is normalized with `& (W - 1)` before dispatch -
// well-defined for every input, and it agrees with the generic default on the
// default's own well-defined domain (`0..=W`, where both `0` and `W` are the
// identity).
//
// NEON const-generic immediates must be constants, so one `match` arm per
// rotate amount is generated below; the arms collapse at monomorphization since
// `IMM8` is a constant.
// ---------------------------------------------------------------------------

macro_rules! neon_rotate_imm {
    (
        $rori:ident, $roli:ident, $ty:ty, width: $w:literal,
        shl: $shl:ident, sri: $sri:ident, amounts: [$($n:literal),*]
    ) => {
        /// Rotate each lane right by `IMM8 & (W - 1)` bits (SHL + SRI).
        #[inline(always)]
        pub fn $rori<const IMM8: i32>(v: $ty) -> $ty {
            unsafe {
                match (IMM8 as u32) & ($w - 1) {
                    $($n => $sri::<$n>($shl::<{ $w - $n }>(v), v),)*
                    // rotate by 0 is the identity (and is not an encodable
                    // SRI/SHL immediate)
                    _ => v,
                }
            }
        }

        /// Rotate each lane left by `IMM8 & (W - 1)` bits (SHL + SRI).
        #[inline(always)]
        pub fn $roli<const IMM8: i32>(v: $ty) -> $ty {
            unsafe {
                match (IMM8 as u32) & ($w - 1) {
                    $($n => $sri::<{ $w - $n }>($shl::<$n>(v), v),)*
                    _ => v,
                }
            }
        }
    };
}

neon_rotate_imm!(
    neon_rori_u8, neon_roli_u8, uint8x16_t, width: 8,
    shl: vshlq_n_u8, sri: vsriq_n_u8,
    amounts: [1, 2, 3, 4, 5, 6, 7]
);

neon_rotate_imm!(
    neon_rori_u16, neon_roli_u16, uint16x8_t, width: 16,
    shl: vshlq_n_u16, sri: vsriq_n_u16,
    amounts: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
);

neon_rotate_imm!(
    neon_rori_u32, neon_roli_u32, uint32x4_t, width: 32,
    shl: vshlq_n_u32, sri: vsriq_n_u32,
    amounts: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ]
);

neon_rotate_imm!(
    neon_rori_u64, neon_roli_u64, uint64x2_t, width: 64,
    shl: vshlq_n_u64, sri: vsriq_n_u64,
    amounts: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ]
);
