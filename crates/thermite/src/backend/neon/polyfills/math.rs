use super::*;

// ---------------------------------------------------------------------------
// Approximate reciprocal / reciprocal-sqrt.
//
// `vrecpeq`/`vrsqrteq` are only ~8-bit estimates - noticeably coarser than
// x86's ~12-bit `rcpps`/`rsqrtps` that Thermite's policy kernels calibrate
// their refinement around. One fused Newton step (`vrecpsq`/`vrsqrtsq`, a
// single instruction each) lifts them to ~16 bits, comfortably above x86
// parity, so `FloatRegister::rcp`/`rsqrt` on NEON includes that step.
// ---------------------------------------------------------------------------

/// ~16-bit reciprocal estimate: `vrecpe` + one `vrecps` Newton step.
#[inline(always)]
pub fn neon_rcp_f32(v: float32x4_t) -> float32x4_t {
    unsafe {
        let e = vrecpeq_f32(v);
        vmulq_f32(e, vrecpsq_f32(v, e))
    }
}

/// ~16-bit reciprocal estimate for `f64` lanes.
#[inline(always)]
pub fn neon_rcp_f64(v: float64x2_t) -> float64x2_t {
    unsafe {
        let e = vrecpeq_f64(v);
        vmulq_f64(e, vrecpsq_f64(v, e))
    }
}

/// ~16-bit reciprocal-sqrt estimate: `vrsqrte` + one `vrsqrts` Newton step.
///
/// Edge-case divergence from x86 `rsqrtps`: `rsqrt(0)` is `NaN` here (the
/// refinement computes `0 * inf`), where x86 returns `inf`. This is the same
/// acknowledged-unfixable divergence sse2neon documents (DLTcollab/sse2neon
/// issue #526); approximate ops are compared with relative tolerance and the
/// math kernels guard specials behind policy checks, so it is tolerated
/// rather than patched (masking it would cost 2+ ops on every call).
#[inline(always)]
pub fn neon_rsqrt_f32(v: float32x4_t) -> float32x4_t {
    unsafe {
        let e = vrsqrteq_f32(v);
        vmulq_f32(e, vrsqrtsq_f32(vmulq_f32(v, e), e))
    }
}

/// ~16-bit reciprocal-sqrt estimate for `f64` lanes.
#[inline(always)]
pub fn neon_rsqrt_f64(v: float64x2_t) -> float64x2_t {
    unsafe {
        let e = vrsqrteq_f64(v);
        vmulq_f64(e, vrsqrtsq_f64(vmulq_f64(v, e), e))
    }
}

// ---------------------------------------------------------------------------
// Widening-multiply high halves (x86 `pmulh`-style). NEON has no direct
// "high half of product" instruction; `vmull`/`vmull_high` produce the full
// double-width products and `vuzp2` collects the high halves.
// ---------------------------------------------------------------------------

macro_rules! stamp_mulhi {
    ($($name:ident: $s:ident/$w:ident, $ty:ty, $get_lo:ident, $uzp:ident, $re:ident);* $(;)?) => {$(paste::paste! {
        #[inline(always)]
        pub fn $name(a: $ty, b: $ty) -> $ty {
            unsafe {
                let lo = [<vmull_ $s>]($get_lo(a), $get_lo(b));
                let hi = [<vmull_high_ $s>](a, b);
                $uzp($re(lo), $re(hi))
            }
        }
    })*};
}

stamp_mulhi! {
    neon_mulhi_s8:  s8/s16,  int8x16_t,  vget_low_s8,  vuzp2q_s8,  vreinterpretq_s8_s16;
    neon_mulhi_s16: s16/s32, int16x8_t,  vget_low_s16, vuzp2q_s16, vreinterpretq_s16_s32;
    neon_mulhi_s32: s32/s64, int32x4_t,  vget_low_s32, vuzp2q_s32, vreinterpretq_s32_s64;
    neon_mulhi_u8:  u8/u16,  uint8x16_t, vget_low_u8,  vuzp2q_u8,  vreinterpretq_u8_u16;
    neon_mulhi_u16: u16/u32, uint16x8_t, vget_low_u16, vuzp2q_u16, vreinterpretq_u16_u32;
    neon_mulhi_u32: u32/u64, uint32x4_t, vget_low_u32, vuzp2q_u32, vreinterpretq_u32_u64;
}

// ---------------------------------------------------------------------------
// 64-bit lane integer multiply: no `vmulq_s64`/`vmulq_u64` exists. This is the
// canonical 7-instruction decomposition (the one LLVM/v8 emit for wasm
// `i64x2.mul`; see https://blog.ngzhian.com/i64x2mul.html): swap the 32-bit
// halves of one operand, a plain 32-bit multiply then pairwise-add-long forms
// the cross-term sum, and a widening multiply-accumulate adds `lo * lo`.
// Truncating each cross term to 32 bits is harmless - the discarded bits
// would be shifted past bit 63 anyway. Sign-agnostic, so it serves both
// signed and unsigned lanes.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neon_mullo_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let a32 = vreinterpretq_u32_u64(a);
        let b32 = vreinterpretq_u32_u64(b);

        // per 64-bit lane: [a_lo * b_hi, a_hi * b_lo] (each mod 2^32)
        let cross = vmulq_u32(a32, vrev64q_u32(b32));
        // (a_lo * b_hi + a_hi * b_lo) mod 2^32, widened per lane, shifted into place
        let cross = vshlq_n_u64::<32>(vpaddlq_u32(cross));
        // + full 64-bit a_lo * b_lo
        vmlal_u32(cross, vmovn_u64(a), vmovn_u64(b))
    }
}
