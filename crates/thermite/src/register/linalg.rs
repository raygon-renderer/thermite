use super::*;

// Concrete SwizzleIndices types for the two 3D cross-product permutations,
// defined for both supported lane counts (U3 and U4).

/// [z, x, y, w] permutation for 4-lane registers.
pub struct Zxyw4;
impl SwizzleIndices<typenum::U4> for Zxyw4 {
    const INDICES: GenericArray<u32, typenum::U4> = GenericArray::from_array([2, 0, 1, 3]);
}

/// [y, z, x, w] permutation for 4-lane registers.
pub struct Yzxw4;
impl SwizzleIndices<typenum::U4> for Yzxw4 {
    const INDICES: GenericArray<u32, typenum::U4> = GenericArray::from_array([1, 2, 0, 3]);
}

/// [z, x, y] permutation for 3-lane registers.
pub struct Zxyw3;
impl SwizzleIndices<typenum::U3> for Zxyw3 {
    const INDICES: GenericArray<u32, typenum::U3> = GenericArray::from_array([2, 0, 1]);
}

/// [y, z, x] permutation for 3-lane registers.
pub struct Yzxw3;
impl SwizzleIndices<typenum::U3> for Yzxw3 {
    const INDICES: GenericArray<u32, typenum::U3> = GenericArray::from_array([1, 2, 0]);
}

pub trait ValidLinAlg3Length<R: FloatRegister<Lanes = Self>>: Lanes {
    type ZXYW: SwizzleIndices<Self>;
    type YZXW: SwizzleIndices<Self>;
}

impl<R> ValidLinAlg3Length<R> for typenum::U4
where
    R: FloatRegister<Lanes = Self>,
{
    type ZXYW = Zxyw4;
    type YZXW = Yzxw4;
}

// Certain GPU vectors may actually have 3-lane registers, but this will probably
// never be implemented for CPU SIMD.
impl<R> ValidLinAlg3Length<R> for typenum::U3
where
    R: FloatRegister<Lanes = Self>,
{
    type ZXYW = Zxyw3;
    type YZXW = Yzxw3;
}

/// Extensions to the `FloatRegister` trait for the most common 3D linear algebra operations.
///
/// This is only available on 3 or 4-lane registers.
pub trait LinAlg3Register: FloatRegister<Lanes: ValidLinAlg3Length<Self>> + SwizzleRegister {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements3(Self::mul(lhs, rhs))
    }

    #[allow(clippy::upper_case_acronyms)]
    #[inline(always)]
    fn cross3<const DOP: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        type ZXYW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::ZXYW;
        type YZXW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::YZXW;

        if DOP {
            // More accurate cross product using the "accurate difference of sums" method, but
            // requires fused multiply-add/subtract operations for best accuracy.
            let a = Self::permutev_const::<YZXW<Self>>(lhs); // [y, z, x]
            let b = Self::permutev_const::<ZXYW<Self>>(rhs); // [z, x, y]
            let c = Self::permutev_const::<ZXYW<Self>>(lhs); // [z, x, y]
            let d = Self::permutev_const::<YZXW<Self>>(rhs); // [y, z, x]

            let cd = Self::mul(c, d);

            let err = Self::nmul_add(c, d, cd);
            let dop = Self::mul_sub(a, b, cd);

            Self::add(dop, err)
        } else {
            let lhszxy = Self::permutev_const::<ZXYW<Self>>(lhs);
            let rhszxy = Self::permutev_const::<ZXYW<Self>>(rhs);

            let lhszxy_rhs = Self::mul(lhszxy, rhs);
            let rhszxy_lhs = Self::mul(rhszxy, lhs);

            let sub = Self::sub(lhszxy_rhs, rhszxy_lhs);

            Self::permutev_const::<ZXYW<Self>>(sub)
        }
    }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        if const { Self::Lanes::USIZE == 4 } {
            Self::insert::<3>(value, Element::ZERO)
        } else {
            value
        }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        if const { Self::Lanes::USIZE == 4 } {
            Self::insert::<3>(value, Element::ONE)
        } else {
            value
        }
    }

    /// 3x3 Matrix Transpose
    ///
    /// Each register holds one column (first 3 lanes). On 4-lane registers the
    /// 4th lane of every output row is **unspecified** - only the first three
    /// lanes are meaningful. Use [`LinAlg3Vector::zero4`] if you need it cleared.
    #[inline(always)]
    fn mat3_transpose(cols: &[Storage<Self>; 3]) -> [Storage<Self>; 3] {
        if const { Self::Lanes::USIZE == 4 } {
            let (lo, hi) = Self::interleave(cols[0], cols[1]);
            // lo = [col0.x, col1.x, col0.y, col1.y]
            // hi = [col0.z, col1.z, col0.w, col1.w]  (col?.w = unused lane)

            let c0 = Self::extract::<0>(cols[2]);
            let c1 = Self::extract::<1>(cols[2]);
            let c2 = Self::extract::<2>(cols[2]);

            [
                // row0 = [col0.x, col1.x, col2.x, 0]
                Self::insert::<2>(lo, c0),
                // row1 = [col0.y, col1.y, col2.y, 0]: move lo's upper pair to lanes 0,1
                Self::insert::<2>(s!(Self: lo, [2, 3, 2, 3]), c1),
                // row2 = [col0.z, col1.z, col2.z, 0]
                Self::insert::<2>(hi, c2),
            ]
        } else {
            // U3: scalar extract + insert (GPU/SPIRV backends)
            // this is a mess but I'm too tired to figure it out right now.
            [
                Self::insert::<2>(
                    Self::insert::<1>(Self::splat(Self::extract::<0>(cols[0])), Self::extract::<0>(cols[1])),
                    Self::extract::<0>(cols[2]),
                ),
                Self::insert::<2>(
                    Self::insert::<1>(Self::splat(Self::extract::<1>(cols[0])), Self::extract::<1>(cols[1])),
                    Self::extract::<1>(cols[2]),
                ),
                Self::insert::<2>(
                    Self::insert::<1>(Self::splat(Self::extract::<2>(cols[0])), Self::extract::<2>(cols[1])),
                    Self::extract::<2>(cols[2]),
                ),
            ]
        }
    }

    /// 3x3 Matrix multiplied by 3D Vector
    ///
    /// Each column/row is a register whose first 3 lanes hold the matrix data.
    /// If the register has 4 lanes the 4th lane of the result is always zeroed.
    #[inline(always)]
    fn mat3_vec3_product<const COLUMN_MAJOR: bool>(cols: &[Storage<Self>; 3], vector: Storage<Self>) -> Storage<Self> {
        let mut result = Self::EMPTY;

        if const { !COLUMN_MAJOR } {
            // Row-major: result[i] = dot3(row[i], vector).
            let x = Self::dot3(cols[0], vector);
            let y = Self::dot3(cols[1], vector);
            let z = Self::dot3(cols[2], vector);
            result = Self::insert::<0>(result, x);
            result = Self::insert::<1>(result, y);
            result = Self::insert::<2>(result, z);
        } else {
            let x = Self::broadcast::<0>(vector);
            let y = Self::broadcast::<1>(vector);
            let z = Self::broadcast::<2>(vector);

            // (cols[0] * x) + (cols[1] * y) then FMA cols[2] * z on top.
            result = Self::mul_adde(cols[2], z, Self::mul_adde(cols[1], y, Self::mul(cols[0], x)));
        }

        result
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
        use crate::{math::FloatConsts as C, register::Element as E};

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
            const { reg::<Self, 4>([E::ZERO, C::NEG_ZERO, E::ZERO, C::NEG_ZERO]) },
        );

        // T3 Signs: (+, +, -, -) -> Negate indices 2 and 3
        let rhs_y_signed = Self::bitxor(
            rhs_y,
            const { reg::<Self, 4>([E::ZERO, E::ZERO, C::NEG_ZERO, C::NEG_ZERO]) },
        );

        // T4 Signs: (-, +, +, -) -> Negate indices 0 and 3
        let rhs_z_signed = Self::bitxor(
            rhs_z,
            const { reg::<Self, 4>([C::NEG_ZERO, E::ZERO, E::ZERO, C::NEG_ZERO]) },
        );

        // Pair 1: (w * rhs) + (x * rhs_x_signed)
        let sum12 = Self::mul_adde(x, rhs_x_signed, Self::mul(w, rhs));

        // Pair 2: (y * rhs_y_signed) + (z * rhs_z_signed)
        let sum34 = Self::mul_adde(z, rhs_z_signed, Self::mul(y, rhs_y_signed));

        Self::add(sum12, sum34)
    }

    #[inline(always)]
    fn quat4_vec3_product<const DOP: bool>(q: Storage<Self>, v: Storage<Self>) -> Storage<Self> {
        // --- Fast method by Giesen ---
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
    }

    /// 4x4 Matrix Transpose
    #[inline(always)]
    fn mat4_transpose(m: &[Storage<Self>; 4]) -> [Storage<Self>; 4] {
        // Two-stage interleave transpose. NOTE: thermite's `interleave` is
        // order-preserving (`interleave(a, b) = ([a0,b0,a1,b1], [a2,b2,a3,b3])`),
        // NOT raw `unpcklps`/`movelh`. With that semantic the stage-1 pairing
        // must be (r0,r2) and (r1,r3) - pairing (r0,r1)/(r2,r3) instead swaps
        // lanes 1 and 2 of every output row.
        //
        // r0=[00,01,02,03] r2=[20,21,22,23]
        //   i0 = [00,20,01,21]   i1 = [02,22,03,23]
        let (i0, i1) = Self::interleave(m[0], m[2]);
        // r1=[10,11,12,13] r3=[30,31,32,33]
        //   i2 = [10,30,11,31]   i3 = [12,32,13,33]
        let (i2, i3) = Self::interleave(m[1], m[3]);

        // c0 = [00,10,20,30]  c1 = [01,11,21,31]
        let (c0, c1) = Self::interleave(i0, i2);
        // c2 = [02,12,22,32]  c3 = [03,13,23,33]
        let (c2, c3) = Self::interleave(i1, i3);

        [c0, c1, c2, c3]
    }

    /// 4x4 Matrix multiplied by 3D Vector
    #[inline(always)]
    fn mat4_vec3_product<const COLUMN_MAJOR: bool>(cols: &[Storage<Self>; 4], vector: Storage<Self>) -> Storage<Self> {
        if const { !COLUMN_MAJOR } {
            // transpose and treat as column-major
            return Self::mat4_vec3_product::<true>(&Self::mat4_transpose(cols), vector);
        }

        let x = Self::broadcast::<0>(vector);
        let y = Self::broadcast::<1>(vector);
        let z = Self::broadcast::<2>(vector);

        Self::mul_adde(cols[2], z, Self::mul_adde(cols[1], y, Self::mul(cols[0], x)))
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
        Self: WideRegister<Wide: FloatRegister>,
    {
        // swap operands if not column-major
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };

        const {
            assert!(
                !<Self::Wide as CoreRegister>::IS_EMULATED,
                "Wide register matrix multiplication requires true wide registers."
            );
        }

        // 1. Double-Pump the LHS (Basis Vectors)
        // We concat each column with itself so it exists in both the low and high lanes.
        // a0_wide = (Col0, Col0)
        let a0 = Self::Wide::concat(lhs[0], lhs[0]);
        let a1 = Self::Wide::concat(lhs[1], lhs[1]);
        let a2 = Self::Wide::concat(lhs[2], lhs[2]);
        let a3 = Self::Wide::concat(lhs[3], lhs[3]);

        // Define the macro locally to handle the "Broadcast -> Concat -> FMA" pipeline.
        // passing types ($Wide, $Narrow) explicitly avoids ambiguity.
        #[rustfmt::skip]
        macro_rules! compute_pair {
            ($rhs_a:expr, $rhs_b:expr) => {{
                // A. Prepare Coefficients
                // Broadcast on narrow registers first (cheap), then concat into wide (cheap).

                // x = (b_left.x ... | b_right.x ...)
                let x = Self::Wide::concat(Self::broadcast::<0>($rhs_a), Self::broadcast::<0>($rhs_b));
                let y = Self::Wide::concat(Self::broadcast::<1>($rhs_a), Self::broadcast::<1>($rhs_b));
                let z = Self::Wide::concat(Self::broadcast::<2>($rhs_a), Self::broadcast::<2>($rhs_b));
                let w = Self::Wide::concat(Self::broadcast::<3>($rhs_a), Self::broadcast::<3>($rhs_b));

                // TODO: optimize this to use a single concat, then 4 shuffles, since 8-wide
                // registers will reuse the immediate shuffle value for each 128-bit lane.

                // B. Wide FMA Chain (Pairwise Optimization)
                // We perform the math for Col N and Col N+1 simultaneously, then add the results together.
                Self::Wide::add(
                    // (Col0 * x) + (Col1 * y)
                    Self::Wide::mul_adde(a1, y, Self::Wide::mul(a0, x)),
                    // (Col2 * z) + (Col3 * w)
                    Self::Wide::mul_adde(a3, w, Self::Wide::mul(a2, z)),
                )
            }};
        }

        // 2. Compute First Half (Result Cols 0 and 1)
        let wide_res_01 = compute_pair!(rhs[0], rhs[1]);

        // 3. Compute Second Half (Result Cols 2 and 3)
        let wide_res_23 = compute_pair!(rhs[2], rhs[3]);

        // 4. Split and Return
        let (c0, c1) = Self::Wide::split(wide_res_01);
        let (c2, c3) = Self::Wide::split(wide_res_23);

        [c0, c1, c2, c3]
    }

    #[inline(always)]
    fn mat4_inverse<const DET_ONLY: bool>(m: &mut [Storage<Self>; 4], det: &mut Self::Element) -> bool {
        // standard implementation using swizzle macro that
        // invokes permutev/swizzle meta-instructions
        impl_mat4_inverse!(m, det, s, DET_ONLY)
    }
}
