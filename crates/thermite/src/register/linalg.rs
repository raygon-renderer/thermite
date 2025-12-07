use super::*;

/// Useful swizzle indices for 3D linear algebra operations, such as cross products,
/// for both 3-lane and 4-lane registers.4
pub trait ValidLinAlg3Length<R: FloatRegister<Lanes = Self>>: Lanes {
    const ZXYW: GenericArray<u32, R::Lanes>;
    const YZXW: GenericArray<u32, R::Lanes>;
}

impl<R> ValidLinAlg3Length<R> for typenum::U4
where
    R: FloatRegister<Lanes = Self>,
{
    const ZXYW: GenericArray<u32, R::Lanes> = GenericArray::from_array([2, 0, 1, 3]);
    const YZXW: GenericArray<u32, R::Lanes> = GenericArray::from_array([1, 2, 0, 3]);
}

// Certain GPU vectors may actually have 3-lane registers, but this will probably
// never be implemented for CPU SIMD.
impl<R> ValidLinAlg3Length<R> for typenum::U3
where
    R: FloatRegister<Lanes = Self>,
{
    const ZXYW: GenericArray<u32, R::Lanes> = GenericArray::from_array([2, 0, 1]);
    const YZXW: GenericArray<u32, R::Lanes> = GenericArray::from_array([1, 2, 0]);
}

/// Extensions to the `FloatRegister` trait for the most common 3D linear algebra operations.
///
/// This is only available on 3 or 4-lane registers.
pub trait LinAlg3Register: FloatRegister<Lanes: ValidLinAlg3Length<Self>> + SwizzleRegister {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements3(Self::mul(lhs, rhs))
    }

    #[inline(always)]
    fn cross3<const DOP: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if DOP {
            // More accurate cross product using the "accurate difference of sums" method, but
            // requires fused multiply-add/subtract operations for best accuracy.
            let a = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::YZXW); // [y, z, x]
            let b = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW); // [z, x, y]
            let c = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW); // [z, x, y]
            let d = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::YZXW); // [y, z, x]

            let cd = Self::mul(c, d);

            let err = Self::nmul_add(c, d, cd);
            let dop = Self::mul_sub(a, b, cd);

            Self::add(dop, err)
        } else {
            let lhszxy = Self::permutev(lhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW);
            let rhszxy = Self::permutev(rhs, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW);

            let lhszxy_rhs = Self::mul(lhszxy, rhs);
            let rhszxy_lhs = Self::mul(rhszxy, lhs);

            let sub = Self::sub(lhszxy_rhs, rhszxy_lhs);

            Self::permutev(sub, <Self::Lanes as ValidLinAlg3Length<Self>>::ZXYW)
        }
    }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        if Self::Lanes::USIZE == 4 {
            Self::insert::<3>(value, Element::ZERO)
        } else {
            value
        }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        if Self::Lanes::USIZE == 4 {
            Self::insert::<3>(value, Element::ONE)
        } else {
            value
        }
    }

    fn min_element3(value: Storage<Self>) -> Self::Element;
    fn max_element3(value: Storage<Self>) -> Self::Element;
    fn sum_elements3(value: Storage<Self>) -> Self::Element;
    fn prod_elements3(value: Storage<Self>) -> Self::Element;
}

/// Extensions to the `FloatRegister` trait for 4D linear algebra operations,
/// including quaternion operations.
pub trait LinAlg4Register: LinAlg3Register<Lanes = typenum::U4> {
    #[inline(always)]
    fn dot4(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements(Self::mul(lhs, rhs))
    }

    /// Quaternion multiplication.
    ///
    /// Corresponds to `lhs * rhs`.
    ///
    /// Method:
    /// 1. Calculate T1 = (lhs.w * rhs)
    /// 2. Calculate T2 = (lhs.x * rhs.wzyx) * {+,-,+,-}
    /// 3. Calculate T3 = (lhs.y * rhs.zwxy) * {+,+,-,-}
    /// 4. Calculate T4 = (lhs.z * rhs.yxwz) * {-,+,+,-}
    /// 5. Sum T1 + T2 + T3 + T4
    #[inline(always)]
    fn quat4_product(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        use crate::math::FloatConsts as C;

        let w = Self::broadcast::<3>(lhs);
        let x = Self::broadcast::<0>(lhs);
        let y = Self::broadcast::<1>(lhs);
        let z = Self::broadcast::<2>(lhs);

        // TODO: Alternative implementation when permutev is not available?
        let rhs_x = s!(Self: rhs, [3, 2, 1, 0]);
        let rhs_y = s!(Self: rhs, [2, 3, 0, 1]);
        let rhs_z = s!(Self: rhs, [1, 0, 3, 2]);

        // T2 Signs: (+, -, +, -) -> Negate indices 1 and 3
        let rhs_x_signed = Self::bitxor(
            rhs_x,
            const { reg::<Self, 4>([C::ZERO, C::NEG_ZERO, C::ZERO, C::NEG_ZERO]) },
        );

        // T3 Signs: (+, +, -, -) -> Negate indices 2 and 3
        let rhs_y_signed = Self::bitxor(
            rhs_y,
            const { reg::<Self, 4>([C::ZERO, C::ZERO, C::NEG_ZERO, C::NEG_ZERO]) },
        );

        // T4 Signs: (-, +, +, -) -> Negate indices 0 and 3
        let rhs_z_signed = Self::bitxor(
            rhs_z,
            const { reg::<Self, 4>([C::NEG_ZERO, C::ZERO, C::ZERO, C::NEG_ZERO]) },
        );

        // Pair 1: (w * rhs) + (x * rhs_x_signed)
        let sum12 = Self::mul_adde(x, rhs_x_signed, Self::mul(w, rhs));

        // Pair 2: (y * rhs_y_signed) + (z * rhs_z_signed)
        let sum34 = Self::mul_adde(z, rhs_z_signed, Self::mul(y, rhs_y_signed));

        Self::add(sum12, sum34)
    }

    #[inline(always)]
    fn quat4_vec3_product<const DOP: bool>(q: Storage<Self>, v: Storage<Self>) -> Storage<Self> {
        if const { Self::HAS_PERMUTEV } {
            // --- Fast SIMD Path (Giesen) ---
            // Formula: v + 2w(q x v) + 2(q x (q x v))

            let w = Self::broadcast::<3>(q);
            let q_xyz = q;

            // t = 2 * cross(q, v)
            let t = Self::cross3::<DOP>(q_xyz, v);
            let t = Self::add(t, t); // multiply by 2

            // result = v + w*t + cross(q, t)
            let w_t = Self::mul(w, t);
            let cross_q_t = Self::cross3::<DOP>(q_xyz, t);

            // compute wt + v first to allow for better instruction level parallelism,
            // while waiting on the cross product to complete
            Self::add(cross_q_t, Self::add(w_t, v))
        } else {
            // --- Scalar/Fallback Path (Textbook) ---
            // Formula: 2(q.v)q + (w^2 - q.q)v + 2w(q x v)
            //
            // When shuffles are expensive (emulated), Cross Products are expensive.
            // This variant only uses 1 Cross Product, substituting the other with
            // 2 Dot Products (which are cheap purely vertical/scalar math).

            let u = q; // Vector part
            let s = Self::broadcast::<3>(q);

            // Term 1: 2 * dot(u, v) * u
            let dot_uv = Self::splat(Self::dot3(u, v));
            let t1 = Self::mul(u, Self::add(dot_uv, dot_uv));

            // Term 2: v * (s*s - dot(u, u))
            let t2 = Self::mul(v, Self::sub(Self::mul(s, s), Self::splat(Self::dot3(u, u))));

            // Term 3: 2s * cross(u, v)
            let t3 = Self::mul(Self::add(s, s), Self::cross3::<DOP>(u, v));

            // Summation: (Term 1 + Term 2) + Term 3
            Self::add(Self::add(t1, t2), t3)
        }
    }

    /// 4x4 Matrix Transpose
    #[inline(always)]
    fn mat4_transpose(m: &[Storage<Self>; 4]) -> [Storage<Self>; 4] {
        // Stage 1: Interleave Low and High halves
        // r0: [00, 01, 02, 03]
        // r1: [10, 11, 12, 13]
        // tmp0 (UnpackLo) -> [00, 10, 01, 11] (Rows 0+1 mixed lower)
        // tmp1 (UnpackHi) -> [02, 12, 03, 13] (Rows 0+1 mixed upper)
        let (tmp0, tmp1) = Self::unpack(m[0], m[1]);

        // r2: [20, 21, 22, 23]
        // r3: [30, 31, 32, 33]
        // tmp2 (UnpackLo) -> [20, 30, 21, 31] (Rows 2+3 mixed lower)
        // tmp3 (UnpackHi) -> [22, 32, 23, 33] (Rows 2+3 mixed upper)
        let (tmp2, tmp3) = Self::unpack(m[2], m[3]);

        // Stage 2: Swap 64-bit blocks (mixing the results of Stage 1)
        // Final columns are created by unpacking the results of Stage 1.

        // Col0 = UnpackLo(tmp0, tmp2) -> [00, 10, 20, 30]
        // Col1 = UnpackHi(tmp0, tmp2) -> [01, 11, 21, 31]
        let (c0, c1) = Self::unpack(tmp0, tmp2);

        // Col2 = UnpackLo(tmp1, tmp3) -> [02, 12, 22, 32]
        // Col3 = UnpackHi(tmp1, tmp3) -> [03, 13, 23, 33]
        let (c2, c3) = Self::unpack(tmp1, tmp3);

        [c0, c1, c2, c3]
    }

    /// 4x4 Matrix multiplied by 4D Vector
    #[inline(always)]
    fn mat4_vec4_product<const COLUMN_MAJOR: bool>(cols: &[Storage<Self>; 4], vector: Storage<Self>) -> Storage<Self> {
        if const { !COLUMN_MAJOR } {
            // transpose and treat as column-major
            return Self::mat4_vec4_product::<true>(&Self::mat4_transpose(cols), vector);
        }

        let x = Self::broadcast::<0>(vector);
        let y = Self::broadcast::<1>(vector);
        let z = Self::broadcast::<2>(vector);
        let w = Self::broadcast::<3>(vector);

        // Run two fused multiply-add operations in parallel using instruction-level parallelism
        let sum_ab = Self::mul_adde(cols[1], y, Self::mul(cols[0], x));
        let sum_cd = Self::mul_adde(cols[3], w, Self::mul(cols[2], z));

        // Final merge
        Self::add(sum_ab, sum_cd)
    }

    /// Multiplies two 4x4 Matrices.
    #[inline(always)]
    fn mat4_product<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 4],
        rhs: &[Storage<Self>; 4],
    ) -> [Storage<Self>; 4] {
        // swap operands if not column-major
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };

        [
            Self::mat4_vec4_product::<true>(lhs, rhs[0]),
            Self::mat4_vec4_product::<true>(lhs, rhs[1]),
            Self::mat4_vec4_product::<true>(lhs, rhs[2]),
            Self::mat4_vec4_product::<true>(lhs, rhs[3]),
        ]
    }

    #[inline(always)]
    fn mat4_product_wide<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 4],
        rhs: &[Storage<Self>; 4],
    ) -> [Storage<Self>; 4]
    where
        Self::DoubleRegister: FloatRegister<Element = Self::Element, HalfRegister = Self>,
    {
        // swap operands if not column-major
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };

        const {
            assert!(
                !<Self::DoubleRegister as Register>::IS_EMULATED,
                "Wide register matrix multiplication requires true wide registers."
            );
        }

        // 1. Double-Pump the LHS (Basis Vectors)
        // We concat each column with itself so it exists in both the low and high lanes.
        // a0_wide = (Col0, Col0)
        let a0 = Self::concat(lhs[0], lhs[0]);
        let a1 = Self::concat(lhs[1], lhs[1]);
        let a2 = Self::concat(lhs[2], lhs[2]);
        let a3 = Self::concat(lhs[3], lhs[3]);

        // Define the macro locally to handle the "Broadcast -> Concat -> FMA" pipeline.
        // passing types ($Wide, $Narrow) explicitly avoids ambiguity.
        #[rustfmt::skip]
        macro_rules! compute_pair {
            ($rhs_a:expr, $rhs_b:expr) => {{
                // A. Prepare Coefficients
                // Broadcast on narrow registers first (cheap), then concat into wide (cheap).

                // x = (b_left.x ... | b_right.x ...)
                let x = Self::concat(Self::broadcast::<0>($rhs_a), Self::broadcast::<0>($rhs_b));
                let y = Self::concat(Self::broadcast::<1>($rhs_a), Self::broadcast::<1>($rhs_b));
                let z = Self::concat(Self::broadcast::<2>($rhs_a), Self::broadcast::<2>($rhs_b));
                let w = Self::concat(Self::broadcast::<3>($rhs_a), Self::broadcast::<3>($rhs_b));

                // B. Wide FMA Chain (Pairwise Optimization)
                // We perform the math for Col N and Col N+1 simultaneously, then add the results together.
                Self::DoubleRegister::add(
                    // (Col0 * x) + (Col1 * y)
                    Self::DoubleRegister::mul_adde(a1, y, Self::DoubleRegister::mul(a0, x)),
                    // (Col2 * z) + (Col3 * w)
                    Self::DoubleRegister::mul_adde(a3, w, Self::DoubleRegister::mul(a2, z)),
                )
            }};
        }

        // 2. Compute First Half (Result Cols 0 and 1)
        let wide_res_01 = compute_pair!(rhs[0], rhs[1]);

        // 3. Compute Second Half (Result Cols 2 and 3)
        let wide_res_23 = compute_pair!(rhs[2], rhs[3]);

        // 4. Split and Return
        let (c0, c1) = <Self::DoubleRegister as Register>::split(wide_res_01);
        let (c2, c3) = <Self::DoubleRegister as Register>::split(wide_res_23);

        [c0, c1, c2, c3]
    }

    #[inline(always)]
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> bool {
        // standard implementation using swizzle macro that
        // invokes permutev/swizzle meta-instructions
        impl_mat4_inverse!(m, s)
    }
}
