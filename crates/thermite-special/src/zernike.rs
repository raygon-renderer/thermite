//! Zernike ordering conventions: normalization flags and the three single-index schemes.
//!
//! The polynomials themselves are [`SpecialMath::zernike`](crate::SpecialMath::zernike)
//! and [`zernike_r`](crate::SpecialMath::zernike_r). What lives here is everything
//! *around* them, which in practice is where the errors are.
//!
//! A Zernike mode is named by two integers: the radial degree `n >= 0` and the
//! azimuthal frequency `m`, with `|m| <= n` and `n - |m|` even. Every application
//! flattens that pair into a single running index, and there are three incompatible
//! ways to do it, all in current use:
//!
//! | Scheme | First index | Ordering within a degree | Found in |
//! |---|---|---|---|
//! | ANSI Z80.28 / OSA | 0 | `m` ascending from `-n` to `+n` | ophthalmology, most Python tooling |
//! | Noll | 1 | `\|m\|` ascending, sign alternating by `n mod 4` | astronomy, Zemax "Standard" coefficients |
//! | Fringe (Air Force / Arizona) | 1 | by spatial frequency `n + \|m\|`, cosine before sine | interferometry, Zemax "Fringe" coefficients |
//!
//! Handing a Noll-indexed coefficient vector to ANSI-indexed code produces a
//! plausible-looking wavefront that is wrong from the second term on, and nothing in
//! the numbers announces it. Converting explicitly at the boundary is the fix, which is
//! why these are here rather than left to the caller.
//!
//! Normalization is the second, independent axis. [`ZERNIKE_UNIT_PEAK`] leaves the
//! radial polynomial alone, so every mode has `R_n^m(1) = 1` and coefficients read as
//! peak wavefront amplitude. [`ZERNIKE_ORTHONORMAL`] applies
//! `$N_n^m = \sqrt{2(n+1)/(1 + \delta_{m,0})}$`, making the modes orthonormal on the
//! unit disc under the `$1/\pi$`-weighted inner product, so a coefficient is an RMS
//! contribution and the total wavefront RMS is the root-sum-square of them. Both the
//! ANSI and Noll standards specify the orthonormal form, while unit-peak is what most
//! hand-rolled shader and interferometer code produces. There is no safe default, so
//! the choice is a required const generic rather than a flag with an opinion.
//!
//! All conversions here are `const fn` over plain integers. They are configuration,
//! evaluated once per mode and not per sample, and never belong in a vector loop.

/// Leave the radial polynomial unnormalized: `$R_n^m(1) = 1$` for every mode.
///
/// Coefficients then read as peak wavefront amplitude in whatever unit `rho` is
/// measured against. See the [module docs](self) for the trade against
/// [`ZERNIKE_ORTHONORMAL`].
pub const ZERNIKE_UNIT_PEAK: u8 = 0;

/// Scale each mode by `$\sqrt{2(n+1)/(1 + \delta_{m,0})}$`, the ANSI Z80.28 and Noll
/// normalization.
///
/// The modes are then orthonormal on the unit disc, so a coefficient is that mode's RMS
/// contribution and the wavefront RMS is the root-sum-square of the coefficients.
pub const ZERNIKE_ORTHONORMAL: u8 = 1;

/// Whether `(n, m)` names a real Zernike mode: `$|m| \le n$` with `$n - |m|$` even.
///
/// Every evaluator here returns zero for a pair that fails this, rather than an
/// arbitrary value from a recurrence run outside its range.
#[inline]
pub const fn is_valid(n: u32, m: i32) -> bool {
    let am = m.unsigned_abs();
    am <= n && (n - am).is_multiple_of(2)
}

/// The number of Zernike modes with radial degree at most `n`, i.e. `(n+1)(n+2)/2`.
///
/// This is the length of a full ANSI-indexed coefficient vector truncated at degree `n`,
/// and one past the largest valid ANSI index.
#[inline]
pub const fn count_up_to_degree(n: u32) -> u32 {
    (n + 1) * (n + 2) / 2
}

// --- ANSI Z80.28 / OSA, zero-based ---

/// The ANSI Z80.28 / OSA single index of mode `(n, m)`: `$j = (n(n+2) + m)/2$`, from 0.
#[inline]
pub const fn ansi_index(n: u32, m: i32) -> u32 {
    ((n * (n + 2)) as i32 + m) as u32 / 2
}

/// The mode `(n, m)` carrying ANSI index `j`. Inverse of [`ansi_index`].
#[inline]
pub const fn ansi_to_nm(j: u32) -> (u32, i32) {
    // Degree n occupies j in [n(n+1)/2, n(n+3)/2], so n is the largest degree whose
    // block starts at or before j. Counted rather than solved: the closed form needs a
    // square root, and an integer loop of at most O(sqrt(j)) steps runs at compile time.
    let mut n = 0;
    while (n + 1) * (n + 2) / 2 <= j {
        n += 1;
    }

    (n, 2 * j as i32 - (n * (n + 2)) as i32)
}

// --- Noll, one-based ---

/// The Noll single index of mode `(n, m)`, from 1.
///
/// `$j = n(n+1)/2 + |m| + c$`, where the parity correction `c` alternates which of the
/// `$\pm m$` pair comes first with `n mod 4`. That alternation is the whole reason Noll
/// indexing cannot be computed from `|m|` alone, and the usual place conversions go
/// wrong.
#[inline]
pub const fn noll_index(n: u32, m: i32) -> u32 {
    let am = m.unsigned_abs();

    // Cosine-first for n mod 4 in {0, 1}, sine-first for {2, 3}.
    let cosine_first = n % 4 <= 1;
    let is_cosine = m >= 0;

    let c = if is_cosine == cosine_first && m != 0 { 0 } else { 1 };

    n * (n + 1) / 2 + am + c
}

/// The mode `(n, m)` carrying Noll index `j`. Inverse of [`noll_index`].
///
/// # Panics
/// If `j` is zero. Noll indices start at 1, and there is no mode 0 to return.
#[inline]
pub const fn noll_to_nm(j: u32) -> (u32, i32) {
    assert!(j > 0, "Noll indices are one-based; there is no mode 0");

    // Degree n owns the n+1 indices starting at n(n+1)/2 + 1.
    let mut n = 0;
    while (n + 1) * (n + 2) / 2 < j {
        n += 1;
    }

    // Position within the degree block. Within it |m| ascends by 2 from n's parity,
    // each nonzero |m| appearing twice (once per sign).
    let r = j - n * (n + 1) / 2 - 1;

    let am = if n.is_multiple_of(2) {
        2 * r.div_ceil(2)
    } else {
        2 * (r / 2) + 1
    };

    // The sign is decided by the same parity rule as `noll_index`, so ask it rather
    // than restate it: whichever sign round-trips is the answer.
    let m = am as i32;

    if noll_index(n, m) == j { (n, m) } else { (n, -m) }
}

/// The ANSI index of the mode carrying Noll index `j`.
///
/// The gather a Noll-indexed caller needs against an ANSI-laid-out basis buffer, which
/// is what [`zernike_basis`](crate::SpecialMath::zernike_basis) fills. Exists as one
/// function because the composition `ansi_index(noll_to_nm(j))` is short enough to write
/// by hand at every call site and exactly the kind of thing that gets written backwards.
///
/// # Panics
/// If `j` is zero. Noll indices start at 1.
#[inline]
pub const fn noll_to_ansi(j: u32) -> u32 {
    let (n, m) = noll_to_nm(j);
    ansi_index(n, m)
}

// --- Fringe / Air Force / University of Arizona, one-based ---

/// The Fringe (Air Force / Arizona) single index of mode `(n, m)`, from 1.
///
/// `$j = (1 + (n + |m|)/2)^2 - 2|m| + [m < 0]$`. Ordering is by spatial frequency
/// `$n + |m|$` rather than by radial degree, which is why Fringe truncations (the
/// classic 37-term set) keep low-frequency high-degree terms that an ANSI truncation at
/// the same count would drop.
#[inline]
pub const fn fringe_index(n: u32, m: i32) -> u32 {
    let am = m.unsigned_abs();
    let s = 1 + (n + am) / 2;

    s * s - 2 * am + if m < 0 { 1 } else { 0 }
}

/// The mode `(n, m)` carrying Fringe index `j`. Inverse of [`fringe_index`].
///
/// # Panics
/// If `j` is zero. Fringe indices start at 1.
#[inline]
pub const fn fringe_to_nm(j: u32) -> (u32, i32) {
    assert!(j > 0, "Fringe indices are one-based; there is no mode 0");

    // Frequency group q = (n + |m|)/2 owns exactly j in [q^2 + 1, (q+1)^2], so the group
    // falls straight out of an integer square root and the rest is arithmetic.
    let mut q = 0;
    while (q + 1) * (q + 1) < j {
        q += 1;
    }

    // Within the group, j = (q+1)^2 - 2|m| + [m < 0], so the offset from the group's top
    // carries both |m| and the sign in its low bit.
    let t = (q + 1) * (q + 1) - j;

    let am = t.div_ceil(2);
    let m = if t.is_multiple_of(2) { am as i32 } else { -(am as i32) };

    (2 * q - am, m)
}

/// The ANSI index of the mode carrying Fringe index `j`.
///
/// The Fringe counterpart of [`noll_to_ansi`], and the more valuable of the two: Fringe
/// orders by spatial frequency rather than radial degree, so the mapping reorders modes
/// rather than merely renumbering them, and no amount of staring at a coefficient vector
/// reveals a missing conversion.
///
/// Note that a Fringe index can name a mode of higher radial degree than an ANSI
/// truncation of the same length contains (Fringe 9 is `(4, 0)`, ANSI index 12), so
/// check the result against the basis length rather than assuming it fits.
///
/// # Panics
/// If `j` is zero. Fringe indices start at 1.
#[inline]
pub const fn fringe_to_ansi(j: u32) -> u32 {
    let (n, m) = fringe_to_nm(j);
    ansi_index(n, m)
}
