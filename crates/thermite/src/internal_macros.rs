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

        let minor_a = Self::mul_sube(z_hi, w_hi, Self::mul(z_lo, w_lo));
        let minor_b = Self::mul_sube(z_b, w_hi, Self::mul(z_lo, w_b));
        let minor_c = Self::mul_sube(z_b, w_lo, Self::mul(z_hi, w_b));

        // y-column coefficients with alternating signs:
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
        use num_traits::Zero as _;

        // Based on glam and https://github.com/g-truc/glm `glm_mat4_inverse`
        let [x_axis, y_axis, z_axis, w_axis] = *$input;

        let fac0 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
        };

        let fac1 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
        };

        let fac2 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
        };

        let fac3 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [3, 3, 7, 7]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [3, 3, 7, 7]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
        };

        let fac4 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [2, 2, 6, 6]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [2, 2, 6, 6]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
        };

        let fac5 = {
            let swp0a = $swizzle!(Self: w_axis, z_axis, [1, 1, 5, 5]);
            let swp0b = $swizzle!(Self: w_axis, z_axis, [0, 0, 4, 4]);

            let swp00 = $swizzle!(Self: z_axis, y_axis, [0, 0, 4, 4]);
            let swp01 = $swizzle!(Self: swp0a, [0, 0, 0, 2]);
            let swp02 = $swizzle!(Self: swp0b, [0, 0, 0, 2]);
            let swp03 = $swizzle!(Self: z_axis, y_axis, [1, 1, 5, 5]);

            Self::mul_sube(swp00, swp01, Self::mul(swp02, swp03))
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

        let sub00 = Self::mul_sube(vec1, fac0, Self::mul(vec2, fac1));
        let add00 = Self::mul_adde(vec3, fac2, sub00);
        let inv0 = Self::bitxor(sign_b, add00);

        let sub01 = Self::mul_sube(vec0, fac0, Self::mul(vec2, fac3));
        let add01 = Self::mul_adde(vec3, fac4, sub01);
        let inv1 = Self::bitxor(sign_a, add01);

        let sub02 = Self::mul_sube(vec0, fac1, Self::mul(vec1, fac3));
        let add02 = Self::mul_adde(vec3, fac5, sub02);
        let inv2 = Self::bitxor(sign_b, add02);

        let sub03 = Self::mul_sube(vec0, fac2, Self::mul(vec1, fac4));
        let add03 = Self::mul_adde(vec2, fac5, sub03);
        let inv3 = Self::bitxor(sign_a, add03);

        let row0 = $swizzle!(Self: inv0, inv1, [0, 0, 4, 4]);
        let row1 = $swizzle!(Self: inv2, inv3, [0, 0, 4, 4]);
        let row2 = $swizzle!(Self: row0, row1, [0, 2, 4, 6]);

        let dot0 = Self::dot4(x_axis, row2);

        // Leave the matrix untouched for an exactly-singular determinant (a
        // well-predicted branch); otherwise scale the adjugate by 1/det.
        if crate::likely(!dot0.is_zero()) {
            let rcp = Self::div(Self::ONE, Self::splat(dot0));

            $input[0] = Self::mul(inv0, rcp);
            $input[1] = Self::mul(inv1, rcp);
            $input[2] = Self::mul(inv2, rcp);
            $input[3] = Self::mul(inv3, rcp);
        }

        dot0
    }}
}
