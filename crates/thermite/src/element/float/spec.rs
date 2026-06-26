//! [`FloatSpec`]: a compile-time description of a *packed* floating-point format
//! (IEEE binary16, bfloat16, the OCP FP8 variants, ...) - everything the generic
//! pack/unpack routines need to transcode a value between that format and `f32`.
//!
//! A packed value is just the low [`FloatSpec::BITS`] bits of a `u32` working register; the
//! container the bits actually live in (`u16`, `u8`, ...) is the register layer's concern, not
//! this descriptor's. Every quantity is given relative to `f32` (IEEE binary32), the format we
//! widen to and narrow from.
//!
//! The trait is **values only**: each impl states a handful of primitives (bit widths, bias,
//! special-value scheme) and the rest is derived with provided defaults. The scalar
//! [`FloatSpec::unpack`] / [`FloatSpec::pack`] reference transcoders are written purely in
//! terms of those values - they are the canonical definition of "correct" (the oracle the
//! SIMD backends are differentially tested against), and their existence is the proof that the
//! value set below is actually complete.

#![allow(clippy::unusual_byte_groupings)]

/// Layout of the `f32` (IEEE binary32) we transcode to and from.
pub const F32_EXP_BITS: u32 = 8;
pub const F32_MANTISSA_BITS: u32 = 23;
pub const F32_EXP_BIAS: i32 = 127;
/// All-ones `f32` exponent field (the inf/NaN exponent).
pub const F32_EXP_FIELD_MAX: u32 = (1 << F32_EXP_BITS) - 1; // 255
/// `f32`'s implicit-leading-one significand, i.e. `1.0` in fixed point (`1 << 23`).
pub const F32_IMPLICIT: u32 = 1 << F32_MANTISSA_BITS;
/// Canonical quiet-NaN `f32` (exponent all ones, top mantissa bit set).
pub const F32_QUIET_NAN: u32 = (F32_EXP_FIELD_MAX << F32_MANTISSA_BITS) | (1 << (F32_MANTISSA_BITS - 1));

/// How a format uses its maximum-exponent code points for the non-finite values.
///
/// This is the one axis where the small float formats genuinely disagree, so it must be part
/// of the descriptor rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialEncoding {
    /// Standard IEEE-754: an all-ones exponent field means `±inf` when the mantissa is zero and
    /// NaN otherwise. (binary16, bfloat16, FP8 **E5M2**, and `f32`/`f64` themselves.)
    Ieee,
    /// No infinities. The single code point `S 1..1 1..1` (all-ones exponent **and** all-ones
    /// mantissa) is the only NaN; every other all-ones-exponent pattern is a finite normal.
    /// (OCP FP8 **E4M3** - largest finite magnitude is 448; values above it saturate.)
    FiniteNanOnly,
    /// No infinities and no NaNs - the whole exponent range encodes finite values, and inputs
    /// outside the representable range saturate. (range-only / block-scaled mini-floats.)
    Finite,
    /// Fast/unchecked: assume inputs are well-formed and skip all inf/NaN handling, trading
    /// correctness on non-finite values for speed. For performance-critical paths where the data
    /// is known finite.
    ///
    /// - **unpack** decodes every code point as if finite (no runtime branch) - the all-ones
    ///   exponent patterns that other schemes treat as inf/NaN are decoded as ordinary large
    ///   normals.
    /// - **pack** flushes any non-finite or out-of-range input (`f32` inf/NaN, or a magnitude that
    ///   overflows the format) to signed zero instead of inf / NaN / saturation.
    ///
    /// Same field layout as the IEEE scheme; only the special-value treatment differs. On the
    /// generic fallback this drops the inf/NaN selects from both directions; F16C keeps its
    /// single-instruction decode and applies the flush-to-zero on pack.
    Unchecked,
}

/// Compile-time description of a packed float format, sufficient to pack/unpack against `f32`.
///
/// # The value set, and where each is used
///
/// **Primitives** (every impl specifies these):
/// - [`BITS`](Self::BITS) - total significant width (the container may be wider).
/// - [`EXP_BITS`](Self::EXP_BITS) / [`MANTISSA_BITS`](Self::MANTISSA_BITS) - field widths.
/// - [`EXP_BIAS`](Self::EXP_BIAS) - exponent bias.
/// - [`SPECIAL`](Self::SPECIAL) - the inf/NaN scheme above.
/// - [`HAS_SIGN`](Self::HAS_SIGN) - whether a sign bit is present (defaults to `true`).
///
/// **Derived layout** (provided; an impl overrides only for an exotic format):
/// field shifts/masks, the all-ones exponent value, the mantissa alignment shift to/from
/// `f32`, the exponent rebias, and the canonical inf / NaN / max-finite bit patterns.
pub trait FloatSpec: Copy + 'static {
    // ---- primitives ----------------------------------------------------------------------

    /// Total number of significant bits (e.g. `16` for fp16/bf16, `8` for fp8).
    const BITS: u32;
    /// Width of the exponent field in bits.
    const EXP_BITS: u32;
    /// Width of the (stored, fractional) mantissa field in bits.
    const MANTISSA_BITS: u32;
    /// Exponent bias (subtracted from the stored exponent to get the true exponent).
    const EXP_BIAS: i32;
    /// How the maximum-exponent code points encode inf / NaN.
    const SPECIAL: SpecialEncoding;
    /// Whether the format has a sign bit. Almost always `true`.
    const HAS_SIGN: bool = true;

    // ---- derived layout (provided defaults) ----------------------------------------------

    /// Bit position of the sign (and of the field block: `exp` occupies
    /// `MANTISSA_BITS .. SIGN_SHIFT`).
    const SIGN_SHIFT: u32 = Self::EXP_BITS + Self::MANTISSA_BITS;
    /// Mask of the significant bits within the `u32` working register (`(1 << BITS) - 1`).
    const STORAGE_MASK: u32 = (1u32 << Self::BITS) - 1;
    /// Mask isolating the sign bit (`0` when [`HAS_SIGN`](Self::HAS_SIGN) is false).
    const SIGN_MASK: u32 = (Self::HAS_SIGN as u32) << Self::SIGN_SHIFT;
    /// All-ones exponent field value (`(1 << EXP_BITS) - 1`).
    const EXP_FIELD_MAX: u32 = (1u32 << Self::EXP_BITS) - 1;
    /// Mask isolating the exponent field (already shifted into place).
    const EXP_MASK: u32 = Self::EXP_FIELD_MAX << Self::MANTISSA_BITS;
    /// Mask isolating the mantissa field.
    const MANTISSA_MASK: u32 = (1u32 << Self::MANTISSA_BITS) - 1;

    /// Bits the mantissa shifts when aligning to / from `f32`'s 23-bit field
    /// (`F32_MANTISSA_BITS - MANTISSA_BITS`). Requires `MANTISSA_BITS <= 23` (true for every
    /// sub-`f32` format).
    const MANTISSA_SHIFT: u32 = F32_MANTISSA_BITS - Self::MANTISSA_BITS;
    /// Added to the stored exponent going packed -> `f32` (subtracted the other way):
    /// `F32_EXP_BIAS - EXP_BIAS`.
    const EXP_REBIAS: i32 = F32_EXP_BIAS - Self::EXP_BIAS;

    /// Largest exponent field that still denotes a *finite* number on the *pack* (encode) side.
    /// For [`SpecialEncoding::Ieee`] and [`Unchecked`](SpecialEncoding::Unchecked) the all-ones
    /// field is the reserved inf/NaN code point, so this is `EXP_FIELD_MAX - 1`; for the
    /// no-reserved-code-point schemes it is `EXP_FIELD_MAX`. (Note `Unchecked` *unpack* still
    /// decodes the all-ones exponent as a large normal - it only matters here that an input which
    /// would round up onto that reserved code point overflows, and so flushes to zero, matching
    /// the hardware `vcvtps2ph` + flush path.)
    const MAX_FINITE_EXP_FIELD: u32 = match Self::SPECIAL {
        SpecialEncoding::Ieee | SpecialEncoding::Unchecked => Self::EXP_FIELD_MAX - 1,
        SpecialEncoding::FiniteNanOnly | SpecialEncoding::Finite => Self::EXP_FIELD_MAX,
    };

    /// Bit pattern (sign clear) of the largest finite magnitude - the saturation target.
    const MAX_FINITE_BITS: u32 = match Self::SPECIAL {
        // exp = max - 1, mantissa all ones
        SpecialEncoding::Ieee => ((Self::EXP_FIELD_MAX - 1) << Self::MANTISSA_BITS) | Self::MANTISSA_MASK,
        // exp = max, mantissa all-ones-minus-one (all-ones mantissa at max exp is the NaN)
        SpecialEncoding::FiniteNanOnly => Self::EXP_MASK | (Self::MANTISSA_MASK - 1),
        // exp = max, mantissa all ones (no reserved code points)
        SpecialEncoding::Finite | SpecialEncoding::Unchecked => Self::EXP_MASK | Self::MANTISSA_MASK,
    };

    /// Bit pattern (sign clear) of `+inf`. Only meaningful when [`SPECIAL`](Self::SPECIAL) is
    /// [`Ieee`](SpecialEncoding::Ieee).
    const INFINITY_BITS: u32 = Self::EXP_MASK;

    /// Canonical quiet-NaN bit pattern (sign clear). Meaningful unless the format has no NaN
    /// ([`SpecialEncoding::Finite`]).
    const NAN_BITS: u32 = match Self::SPECIAL {
        SpecialEncoding::Ieee => Self::EXP_MASK | (1 << (Self::MANTISSA_BITS - 1)),
        SpecialEncoding::FiniteNanOnly => Self::EXP_MASK | Self::MANTISSA_MASK,
        // no NaN code point (Unchecked never emits NaN - it flushes to zero)
        SpecialEncoding::Finite | SpecialEncoding::Unchecked => 0,
    };

    // ---- encode-direction targets (pre-resolved so `pack` needs no per-format branch) -----

    /// Result (sign clear) when a magnitude exceeds the finite range: `±inf` for IEEE formats,
    /// saturation to the largest finite for the no-inf schemes, and signed zero for `Unchecked`
    /// (which flushes overflow). Used for both overflow of a finite input and for an `f32`
    /// infinity input.
    const OVERFLOW_BITS: u32 = match Self::SPECIAL {
        SpecialEncoding::Ieee => Self::INFINITY_BITS,
        SpecialEncoding::FiniteNanOnly | SpecialEncoding::Finite => Self::MAX_FINITE_BITS,
        SpecialEncoding::Unchecked => 0,
    };

    /// Result (sign clear) for an `f32` NaN input: the canonical NaN for formats that have one,
    /// saturation for [`SpecialEncoding::Finite`] (no NaN code point), and signed zero for
    /// `Unchecked` (which flushes NaN).
    const NAN_OUT_BITS: u32 = match Self::SPECIAL {
        SpecialEncoding::Ieee | SpecialEncoding::FiniteNanOnly => Self::NAN_BITS,
        SpecialEncoding::Finite => Self::MAX_FINITE_BITS,
        SpecialEncoding::Unchecked => 0,
    };

    // ---- scalar reference transcoders (the oracle; built only from the values above) -----

    /// Decode the low [`BITS`](Self::BITS) bits of `packed` (one value of this format) into the
    /// `f32` it represents. Exact: every value of these sub-`f32` formats is representable in
    /// `f32`.
    #[inline]
    fn unpack(packed: u32) -> f32 {
        let sign_bit = if Self::HAS_SIGN {
            (packed >> Self::SIGN_SHIFT) & 1
        } else {
            0
        };
        let exp = (packed >> Self::MANTISSA_BITS) & Self::EXP_FIELD_MAX;
        let mant = packed & Self::MANTISSA_MASK;
        let f32_sign = sign_bit << 31;

        // Non-finite code points.
        match Self::SPECIAL {
            SpecialEncoding::Ieee if exp == Self::EXP_FIELD_MAX => {
                let f32_mant = if mant == 0 {
                    0 // inf
                } else {
                    // NaN: keep it quiet, carry the payload up into f32's wider mantissa.
                    (1 << (F32_MANTISSA_BITS - 1)) | (mant << Self::MANTISSA_SHIFT)
                };
                return f32::from_bits(f32_sign | (F32_EXP_FIELD_MAX << F32_MANTISSA_BITS) | f32_mant);
            }
            SpecialEncoding::FiniteNanOnly if exp == Self::EXP_FIELD_MAX && mant == Self::MANTISSA_MASK => {
                return f32::from_bits(f32_sign | F32_QUIET_NAN);
            }
            _ => {}
        }

        // Finite (normal, subnormal, or zero). Reconstruct the exact real value in `f64`
        // (which has the range and precision to hold every such value exactly, including
        // bf16 subnormals), then narrow losslessly to `f32`.
        //   value = (-1)^s * (mant + implicit * 2^MANTISSA_BITS) * 2^(e_eff - BIAS - MANTISSA_BITS)
        let (implicit, e_eff): (u64, i32) = if exp == 0 { (0, 1) } else { (1, exp as i32) };
        let significand = (mant as u64) + (implicit << Self::MANTISSA_BITS);
        let pow = e_eff - Self::EXP_BIAS - Self::MANTISSA_BITS as i32;
        // 2^pow as an exact f64 (pow stays well within f64's exponent range for all formats).
        let scale = f64::from_bits(((F64_EXP_BIAS + pow) as u64) << F64_MANTISSA_BITS);
        let magnitude = significand as f64 * scale;
        let value = if sign_bit != 0 { -magnitude } else { magnitude };
        value as f32
    }

    /// Encode `value` (an `f32`) into the low [`BITS`](Self::BITS) bits of the returned `u32`,
    /// rounding the mantissa to nearest, ties to even. Overflow becomes `±inf` (or saturates,
    /// for the non-IEEE schemes); tiny values become subnormals or signed zero.
    ///
    /// This is the canonical (RNE, full special-case handling) reference; policy-driven
    /// variants (truncation, flush-to-zero, ignore-special) live with the `PackingPolicy` layer.
    #[inline]
    fn pack(value: f32) -> u32 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let psign = if Self::HAS_SIGN { sign << Self::SIGN_SHIFT } else { 0 };
        let f32_exp = ((bits >> F32_MANTISSA_BITS) & F32_EXP_FIELD_MAX) as i32;
        let f32_mant = bits & (F32_IMPLICIT - 1);

        // f32 NaN / Inf -> the pre-resolved target for this scheme (inf/saturate/canonical-NaN,
        // or signed zero for `Unchecked`). `psign` is `0` for `Unchecked`'s zero, so this stays a
        // signed zero regardless.
        if f32_exp == F32_EXP_FIELD_MAX as i32 {
            if f32_mant != 0 {
                return psign | Self::NAN_OUT_BITS;
            }
            return psign | Self::OVERFLOW_BITS;
        }

        // f32 subnormals are far below the range of any sub-f32 format -> signed zero.
        if f32_exp == 0 {
            return psign;
        }

        // 24-bit significand with the implicit leading one; target (pre-round) biased exponent.
        let significand = F32_IMPLICIT | f32_mant;
        let mut e = f32_exp - Self::EXP_REBIAS;

        // Number of low significand bits to discard. Subnormal target adds (1 - e) more.
        let mut shift = Self::MANTISSA_SHIFT as i32;
        if e <= 0 {
            shift += 1 - e;
            e = 0;
        }
        if shift >= 32 {
            return psign; // underflows below half the smallest subnormal -> +/- 0
        }
        let shift = shift as u32;

        // Round to nearest, ties to even, on the discarded `shift` bits.
        let keep = significand >> shift;
        let rem = significand & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = rem > halfway || (rem == halfway && (keep & 1) == 1);
        let q = keep + round_up as u32;

        if e == 0 {
            // Subnormal result. If rounding carried `q` up into the implicit-bit position the
            // layout promotes it to the smallest normal automatically.
            return psign | q;
        }

        // Normal result: a rounding carry out of the implicit bit bumps the exponent.
        let (mut e, mut frac) = (e, q & Self::MANTISSA_MASK);
        if q >> Self::MANTISSA_BITS >= 2 {
            e += 1;
            frac = 0;
        }

        // Overflow -> inf / saturate / flush-to-zero, per the scheme (see `OVERFLOW_BITS`).
        if e > Self::MAX_FINITE_EXP_FIELD as i32 {
            return psign | Self::OVERFLOW_BITS;
        }

        let magnitude = ((e as u32) << Self::MANTISSA_BITS) | frac;
        // For the no-infinity schemes, a finite input must never land on the reserved NaN
        // code point (E4M3's `S.1111.111`); saturate to the largest finite instead.
        let magnitude = match Self::SPECIAL {
            SpecialEncoding::FiniteNanOnly if magnitude > Self::MAX_FINITE_BITS => Self::MAX_FINITE_BITS,
            _ => magnitude,
        };
        psign | magnitude
    }
}

// `f64` layout constants used by the exact-reconstruction path in `unpack`.
const F64_EXP_BIAS: i32 = 1023;
const F64_MANTISSA_BITS: u32 = 52;

// =====================================================================================
// Concrete formats. Each states only the primitives; everything else is derived above.
// =====================================================================================

/// IEEE-754 binary16 ("half"): 1 sign, 5 exponent, 10 mantissa, bias 15. Has inf and NaN.
///
/// For a faster, inf/NaN-unchecked variant of the same layout see [`Fp16Fast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp16;
impl FloatSpec for Fp16 {
    const BITS: u32 = 16;
    const EXP_BITS: u32 = 5;
    const MANTISSA_BITS: u32 = 10;
    const EXP_BIAS: i32 = 15;
    const SPECIAL: SpecialEncoding = SpecialEncoding::Ieee;
}

/// Fast, unchecked binary16: same layout as [`Fp16`] but with [`SpecialEncoding::Unchecked`] -
/// inf/NaN handling is elided for speed. unpack decodes the all-ones-exponent code points as
/// ordinary large normals (no runtime branch); pack flushes non-finite/overflowing inputs to
/// signed zero. Use only when the data is known to be finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp16Fast;
impl FloatSpec for Fp16Fast {
    const BITS: u32 = 16;
    const EXP_BITS: u32 = 5;
    const MANTISSA_BITS: u32 = 10;
    const EXP_BIAS: i32 = 15;
    const SPECIAL: SpecialEncoding = SpecialEncoding::Unchecked;
}

/// bfloat16: 1 sign, 8 exponent, 7 mantissa, bias 127 - exactly the top 16 bits of an `f32`.
/// Has inf and NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bf16;
impl FloatSpec for Bf16 {
    const BITS: u32 = 16;
    const EXP_BITS: u32 = 8;
    const MANTISSA_BITS: u32 = 7;
    const EXP_BIAS: i32 = 127;
    const SPECIAL: SpecialEncoding = SpecialEncoding::Ieee;
}

/// OCP FP8 **E4M3**: 1 sign, 4 exponent, 3 mantissa, bias 7. No infinities; the only NaN is
/// `S.1111.111`. Largest finite magnitude is 448; out-of-range values saturate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp8E4M3;
impl FloatSpec for Fp8E4M3 {
    const BITS: u32 = 8;
    const EXP_BITS: u32 = 4;
    const MANTISSA_BITS: u32 = 3;
    const EXP_BIAS: i32 = 7;
    const SPECIAL: SpecialEncoding = SpecialEncoding::FiniteNanOnly;
}

/// OCP FP8 **E5M2**: 1 sign, 5 exponent, 2 mantissa, bias 15. IEEE-style - has inf and NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp8E5M2;
impl FloatSpec for Fp8E5M2 {
    const BITS: u32 = 8;
    const EXP_BITS: u32 = 5;
    const MANTISSA_BITS: u32 = 2;
    const EXP_BIAS: i32 = 15;
    const SPECIAL: SpecialEncoding = SpecialEncoding::Ieee;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trip and known-value checks for the reference transcoders, which double as the
    // proof that the descriptor's value set is sufficient for both directions.
    #[test]
    fn fp16_known_values() {
        assert_eq!(Fp16::unpack(0x3C00), 1.0); // 1.0
        assert_eq!(Fp16::unpack(0xC000), -2.0); // -2.0
        assert_eq!(Fp16::unpack(0x7BFF), 65504.0); // max finite
        assert!(Fp16::unpack(0x7C00).is_infinite() && Fp16::unpack(0x7C00) > 0.0);
        assert!(Fp16::unpack(0xFC00).is_infinite() && Fp16::unpack(0xFC00) < 0.0);
        assert!(Fp16::unpack(0x7E00).is_nan());
        assert_eq!(Fp16::unpack(0x0001), 2.0f32.powi(-24)); // min subnormal
        assert_eq!(Fp16::unpack(0x0000), 0.0);

        assert_eq!(Fp16::pack(1.0), 0x3C00);
        assert_eq!(Fp16::pack(-2.0), 0xC000);
        assert_eq!(Fp16::pack(65504.0), 0x7BFF);
        assert_eq!(Fp16::pack(f32::INFINITY), 0x7C00);
        assert_eq!(Fp16::pack(1e30), 0x7C00); // overflow -> inf
    }

    #[test]
    fn fp16_fast_flushes_and_decodes_unchecked() {
        // unpack: the all-ones-exponent code points are decoded as ordinary large normals, not
        // inf/NaN. 0x7C00 = 2^16 = 65536 (1.0 * 2^(31-15)); 0x7E00 = 1.5 * 2^16 = 98304.
        assert_eq!(Fp16Fast::unpack(0x7C00), 65536.0);
        assert_eq!(Fp16Fast::unpack(0x7E00), 98304.0);
        // finite values still decode normally.
        assert_eq!(Fp16Fast::unpack(0x3C00), 1.0);

        // pack: non-finite / overflowing inputs flush to signed zero.
        assert_eq!(Fp16Fast::pack(f32::INFINITY), 0x0000);
        assert_eq!(Fp16Fast::pack(f32::NEG_INFINITY), 0x8000); // signed zero
        assert_eq!(Fp16Fast::pack(f32::NAN), 0x0000);
        assert_eq!(Fp16Fast::pack(1e30), 0x0000); // overflow -> zero (not inf)
        // in-range finite values pack identically to the checked form.
        assert_eq!(Fp16Fast::pack(1.0), 0x3C00);
        assert_eq!(Fp16Fast::pack(-2.0), 0xC000);
        assert_eq!(Fp16Fast::pack(65504.0), 0x7BFF);
    }

    #[test]
    fn bf16_is_f32_high_bits() {
        // bf16 unpack of the top 16 bits of an f32 returns (a rounded) f32; for exactly
        // representable bf16 values the low 16 f32 bits are zero.
        for &v in &[1.0f32, -2.5, 100.0, 0.0, 0.015625] {
            let packed = (v.to_bits() >> 16) & 0xFFFF;
            assert_eq!(Bf16::unpack(packed).to_bits(), v.to_bits() & 0xFFFF_0000);
        }
        assert!(Bf16::unpack(0x7F80).is_infinite());
        assert!(Bf16::unpack(0x7FC0).is_nan());
    }

    #[test]
    fn fp8_e4m3_saturates_and_has_no_inf() {
        assert_eq!(Fp8E4M3::unpack(0x70), 128.0); // exp field 14 -> 2^(14-7)
        assert_eq!(Fp8E4M3::unpack(0x78), 256.0); // exp field 15 is a normal here -> 2^(15-7)
        assert_eq!(Fp8E4M3::unpack(0x7E), 448.0); // max finite (1.110b * 2^8)
        assert!(Fp8E4M3::unpack(0x7F).is_nan()); // S.1111.111
        assert_eq!(Fp8E4M3::pack(1000.0), 0x7E); // saturates to 448, never inf/NaN
        assert_eq!(Fp8E4M3::pack(f32::INFINITY), 0x7E); // no inf -> saturate
    }

    #[test]
    fn fp8_e5m2_round_trips() {
        for byte in 0u32..256 {
            let v = Fp8E5M2::unpack(byte);
            if v.is_nan() {
                continue;
            }
            // re-pack returns the same code point (canonicalizing -0/0 aside)
            assert_eq!(Fp8E5M2::pack(v), byte, "e5m2 round-trip failed for {byte:#04x}");
        }
    }
}
