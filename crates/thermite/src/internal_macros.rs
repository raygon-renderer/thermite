/// `a*b - c*d`, evaluated so the two products cannot cancel catastrophically.
///
/// Reads `FAST` from the enclosing function's const generic, so the whole
/// 4x4 det/inverse picks one form at monomorphization.
///
/// The accurate form recovers the rounding that `c * d` discarded
/// (`mul_sube(c, d, cd)` = `c*d - round(cd)`, subtracted, where pbrt computes the
/// opposite sign and adds it) and folds it back in. Same algorithm as
/// `LinAlg3Register::cross3<FAST = false>`. It costs two extra ops per site and
/// buys the property the whole thing exists for: a 2x2 minor of a
/// rank-deficient matrix comes back as exactly zero.
///
/// Only the estimating madd-family `_e` ops appear in the accurate arm, so it
/// NEVER lowers to the emulated FMA: where they fuse the residual is exact,
/// and where a wasm relaxed madd turns out unfused the residual computes as
/// fl(cd) - fl(cd) = exactly 0, degrading to the naive difference, which
/// keeps the exact self-minor for free. Keeping both ops on `mul_sube` (one
/// wasm instruction) matters: `nmul_adde` is a different relaxed instruction
/// the spec would let an engine fuse differently, breaking the cancellation.
///
/// `FAST` (and any register whose `HAS_NATIVE_FMA` is definitely unfused) takes
/// `mul_sube` instead. Which of the two lowerings that picks does not matter
/// here, because both are acceptable under `FAST` and on a definitely-unfused
/// register the naive one is the only option:
///
/// - unfused -> `a*b - c*d`, two roundings that cancel, so a self-minor is still
///   exactly zero. Only general cancellation suffers.
/// - fused -> `fma(a, b, -cd)`, one op cheaper again, but `a*b` stays exact while
///   `c*d` rounds, so a self-minor comes back as the discarded rounding rather
///   than zero. That is the edge case `FAST` buys its performance with.
///
/// Runtime-decided fusing (`HAS_NATIVE_FMA` = `Indeterminate`, the wasm
/// relaxed-madd canary) must NOT take the single-`mul_sube` arm at `FAST = false`:
/// a fusing engine turns it into the broken mixed case above. It takes the
/// accurate arm, which is exact whichever way the engine resolves (see above).
macro_rules! dop {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let (a, b, c, d) = ($a, $b, $c, $d);
        let cd = Self::mul(c, d);

        if const { !FAST && !matches!(Self::HAS_NATIVE_FMA, tribool::False) } {
            Self::sub(Self::mul_sube(a, b, cd), Self::mul_sube(c, d, cd))
        } else {
            Self::mul_sube(a, b, cd)
        }
    }};
}

/// Scale an unscaled adjugate by `1/det`, writing into `$out`.
///
/// `FAST` reciprocates once and multiplies four times: two roundings per
/// entry, the first shared by all sixteen. Otherwise four true divisions, one
/// correctly-rounded division per entry.
///
/// The division looks 4x worse in isolation (12.0 against 3.0 RThroughput on
/// znver3) and is not, because `vdivps` occupies FP1 alone and the caller arrives
/// here ~75% data-dependency bound with the divider idle. Cost of choosing it,
/// measured on `mat4_inverse` in cycles per call against the same body with the
/// reciprocal:
///
/// | | reciprocal | four divisions |
/// |---|---|---|
/// | f32x4 | 34.0 / lat 81 | 35.7 / lat 84 (+5%) |
/// | f64x4 | 56.0 / lat 96 | 64.3 / lat 104 (+15%) |
///
/// **Scale at the widest register available.** `vmulpd`/`vdivpd` cost the same at
/// 128 and 256 bits on znver3, so scaling a 2x128 pair issues every op twice for
/// nothing. That is why [`LinAlg4Register::mat4_adjugate`] hands the adjugate back
/// unscaled: `F64x4V3` computes it paired and recombines before arriving here.
/// Doing so took `f64x4` from 8 `vdivpd` to 4 and improved both settings of
/// `FAST`, 49.0 cycles down to 46.8 at `true` and 73.0 down to 64.3 at `false`,
/// because the pairing had been silently doubling the multiplies too.
macro_rules! mat4_scale {
    ($out:ident, $adj:expr, $det:expr) => {{
        let adj = $adj;

        if const { FAST } {
            let rcp = Self::div(Self::ONE, Self::splat($det));

            $out[0] = Self::mul(adj[0], rcp);
            $out[1] = Self::mul(adj[1], rcp);
            $out[2] = Self::mul(adj[2], rcp);
            $out[3] = Self::mul(adj[3], rcp);
        } else {
            let det = Self::splat($det);

            $out[0] = Self::div(adj[0], det);
            $out[1] = Self::div(adj[1], det);
            $out[2] = Self::div(adj[2], det);
            $out[3] = Self::div(adj[3], det);
        }
    }};
}

macro_rules! impl_mat4_inverse {
    (DET_ONLY $input:ident, $swizzle:ident) => {{
        let [x_axis, y_axis, z_axis, w_axis] = *$input;

        // Determinant only: cofactor expansion along the first column using
        // just the first-row cofactor vector - `det = dot4(col0, C)` - so we
        // never build the full adjugate the inverse path needs.
        use crate::{math::FloatConsts as C, register::Element as E};

        // Three packings of the six 2x2 minors of the (z, w) columns:
        //   minor_a = [a2323, a2323, a1323, a1223]
        //   minor_b = [a1323, a0323, a0323, a0223]
        //   minor_c = [a1223, a0223, a0123, a0123]
        let z_hi = $swizzle!(Self: z_axis, [2, 2, 1, 1]);
        let w_hi = $swizzle!(Self: w_axis, [3, 3, 3, 2]);
        let z_lo = $swizzle!(Self: z_axis, [3, 3, 3, 2]);
        let w_lo = $swizzle!(Self: w_axis, [2, 2, 1, 1]);
        let z_b = $swizzle!(Self: z_axis, [1, 0, 0, 0]);
        let w_b = $swizzle!(Self: w_axis, [1, 0, 0, 0]);

        let minor_a = dop!(z_hi, w_hi, z_lo, w_lo);
        let minor_b = dop!(z_b, w_hi, z_lo, w_b);
        let minor_c = dop!(z_b, w_lo, z_hi, w_b);

        // y-column coefficients with alternat... ing signs:
        //   coef_a = [ y1,-y0, y0,-y0]  coef_b = [-y2, y2,-y1, y1]  coef_c = [ y3,-y3, y3,-y2]
        let pnpn = const { reg::<Self, 4>([E::ZERO, C::NEG_ZERO, E::ZERO, C::NEG_ZERO]) };
        let npnp = const { reg::<Self, 4>([C::NEG_ZERO, E::ZERO, C::NEG_ZERO, E::ZERO]) };

        let coef_a = Self::bitxor($swizzle!(Self: y_axis, [1, 0, 0, 0]), pnpn);
        let coef_b = Self::bitxor($swizzle!(Self: y_axis, [2, 2, 1, 1]), npnp);
        let coef_c = Self::bitxor($swizzle!(Self: y_axis, [3, 3, 3, 2]), pnpn);

        let cof = Self::mul_adde(coef_c, minor_c, Self::mul_adde(coef_b, minor_b, Self::mul(coef_a, minor_a)));

        Self::dot4(x_axis, cof)
    }};

    ($input:ident, $swizzle:ident) => {{
        // Based on glam and https://github.com/g-truc/glm `glm_mat4_inverse`
        let [x_axis, y_axis, z_axis, w_axis] = *$input;

        let fac0 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            dop!(swp00, swp01, swp02, swp03)
        };

        let fac1 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            dop!(swp00, swp01, swp02, swp03)
        };

        let fac2 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);

            dop!(swp00, swp01, swp02, swp03)
        };

        let fac3 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            dop!(swp00, swp01, swp02, swp03)
        };

        let fac4 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);

            dop!(swp00, swp01, swp02, swp03)
        };

        let fac5 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);

            dop!(swp00, swp01, swp02, swp03)
        };

        use crate::{math::FloatConsts as C, register::Element as E};

        let sign_a = Self::new(GenericArray::from_array([C::NEG_ZERO, E::ZERO, C::NEG_ZERO, E::ZERO]));
        let sign_b = Self::new(GenericArray::from_array([E::ZERO, C::NEG_ZERO, E::ZERO, C::NEG_ZERO]));

        let temp0 = $swizzle!(Self: y_axis, x_axis, [0, 0, 4, 4]);
        let vec0 = $swizzle!(Self: temp0, [0, 2, 2, 2]);

        let temp1 = $swizzle!(Self: y_axis, x_axis, [1, 1, 5, 5]);
        let vec1 = $swizzle!(Self: temp1, [0, 2, 2, 2]);

        let temp2 = $swizzle!(Self: y_axis, x_axis, [2, 2, 6, 6]);
        let vec2 = $swizzle!(Self: temp2, [0, 2, 2, 2]);

        let temp3 = $swizzle!(Self: y_axis, x_axis, [3, 3, 7, 7]);
        let vec3 = $swizzle!(Self: temp3, [0, 2, 2, 2]);

        let sub00 = dop!(vec1, fac0, vec2, fac1);
        let add00 = Self::mul_adde(vec3, fac2, sub00);
        let inv0 = Self::bitxor(sign_b, add00);

        let sub01 = dop!(vec0, fac0, vec2, fac3);
        let add01 = Self::mul_adde(vec3, fac4, sub01);
        let inv1 = Self::bitxor(sign_a, add01);

        let sub02 = dop!(vec0, fac1, vec1, fac3);
        let add02 = Self::mul_adde(vec3, fac5, sub02);
        let inv2 = Self::bitxor(sign_b, add02);

        let sub03 = dop!(vec0, fac2, vec1, fac4);
        let add03 = Self::mul_adde(vec2, fac5, sub03);
        let inv3 = Self::bitxor(sign_a, add03);

        let row0 = $swizzle!(Self: inv0, inv1, [0, 0, 4, 4]);
        let row1 = $swizzle!(Self: inv2, inv3, [0, 0, 4, 4]);
        let row2 = $swizzle!(Self: row0, row1, [0, 2, 4, 6]);

        let dot0 = Self::dot4(x_axis, row2);

        ([inv0, inv1, inv2, inv3], dot0)
    }}
}
