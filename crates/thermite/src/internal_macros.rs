macro_rules! impl_mat4_inverse {
    ($input:ident, $det:ident, $swizzle:ident, $det_only:expr) => {{
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

        *$det = dot0;

        if const { $det_only } || dot0.is_zero() {
            return false;
        }

        let rcp = Self::div(Self::ONE, Self::splat(dot0));

        $input[0] = Self::mul(inv0, rcp);
        $input[1] = Self::mul(inv1, rcp);
        $input[2] = Self::mul(inv2, rcp);
        $input[3] = Self::mul(inv3, rcp);

        true
    }}
}
