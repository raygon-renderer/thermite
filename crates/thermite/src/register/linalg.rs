use num_traits::Zero;

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
pub trait LinAlg3Register: FloatRegister<Lanes: ValidLinAlg3Length<Self>> {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        Self::sum_elements3(Self::mul(lhs, rhs))
    }

    #[allow(clippy::upper_case_acronyms)]
    #[inline(always)]
    fn cross3<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        type ZXYW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::ZXYW;
        type YZXW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::YZXW;

        if !FAST {
            // Kahan's compensated difference of products: `err` recovers the rounding
            // that `c * d` discarded and adds it back, so the two sides cancel exactly.
            // Needs a real FMA to be worth anything, and degrades to the naive form
            // without one, which has the same exactness property for free.
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

    /// Refraction of incident vector `i` through a surface with normal `n` and
    /// relative index of refraction `eta` (`$\eta = \eta_i/\eta_t$`). `i` and `n` are assumed unit length.
    ///
    /// `$k = 1 - \eta^2(1 - (n \cdot i)^2)$`; total internal reflection (`k < 0`) yields
    /// the zero vector, otherwise `$\eta\, i - (\eta\,(n \cdot i) + \sqrt{k})\, n$`.
    #[allow(clippy::upper_case_acronyms)]
    #[inline(always)]
    fn refract(i: Storage<Self>, n: Storage<Self>, eta: Self::Element) -> Storage<Self> {
        type ZXYW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::ZXYW;
        type YZXW<R> = <<R as CoreRegister>::Lanes as ValidLinAlg3Length<R>>::YZXW;

        // d = dot3(n, i) replicated to every lane: sum the three lane products
        // via the two cross-product rotations (no extract-to-scalar + splat).
        let prod = Self::mul(n, i);
        let d = Self::add(
            Self::add(prod, Self::permutev_const::<YZXW<Self>>(prod)),
            Self::permutev_const::<ZXYW<Self>>(prod),
        );

        // k = 1 - eta^2*(1 - d^2)
        let etav = Self::splat(eta);
        let omd2 = Self::nmul_adde(d, d, Self::ONE); // 1 - d^2
        let k = Self::nmul_adde(Self::mul(etav, etav), omd2, Self::ONE); // 1 - eta^2*(1 - d^2)

        // r = eta*i - (eta*d + sqrt(k))*n
        let coef = Self::mul_adde(etav, d, Self::sqrt(k));
        let r = Self::nmul_adde(coef, n, Self::mul(etav, i));

        // Total internal reflection (k < 0, including the NaN from sqrt(negative)) -> 0.
        Self::select_negative(k, Self::ZERO, r)
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
    /// lanes are meaningful. Use [`LinAlg3Vector::zero4`](crate::vector::LinAlg3Vector::zero4) if you need it cleared.
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

    /// 3x3 matrix times `N` 3D vectors.
    ///
    /// Each column/row is a register whose first 3 lanes hold the matrix data.
    /// If the register has 4 lanes the 4th lane of each result is always zeroed.
    /// Same small-`N` semantics as [`mat4_vec4_product`](LinAlg4Register::mat4_vec4_product).
    #[inline(always)]
    fn mat3_vec3_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 3],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        let mut out = [Self::EMPTY; N];
        let mut i = 0;
        while i < N {
            let v = vectors[i];
            out[i] = if const { COLUMN_MAJOR } {
                let x = Self::broadcast::<0>(v);
                let y = Self::broadcast::<1>(v);
                let z = Self::broadcast::<2>(v);
                // (cols[0] * x) + (cols[1] * y) then FMA cols[2] * z on top.
                Self::mul_adde(cols[2], z, Self::mul_adde(cols[1], y, Self::mul(cols[0], x)))
            } else {
                // Row-major: result[j] = dot3(row[j], vector).
                let mut r = Self::EMPTY;
                r = Self::insert::<0>(r, Self::dot3(cols[0], v));
                r = Self::insert::<1>(r, Self::dot3(cols[1], v));
                Self::insert::<2>(r, Self::dot3(cols[2], v))
            };
            i += 1;
        }
        out
    }

    /// Column-major 3x3 * vec3 evaluated through the double-width register.
    ///
    /// Packs `[c0 | c1]` and `[c2 | 0]` (the third column zero-extended) so the
    /// three column-scales collapse into a single wide multiply + a single wide
    /// FMA, with the coefficients built by in-lane shuffles; the two halves of
    /// the product `[c0*x + c2*z | c1*y]` are then summed. Requires a true wide
    /// register.
    #[inline(always)]
    fn mat3_vec3_product_wide(cols: &[Storage<Self>; 3], vector: Storage<Self>) -> Storage<Self>
    where
        Self: WideRegister<Wide: FloatRegister>,
        typenum::Double<Self::Lanes>: Lanes,
    {
        const {
            assert!(
                !<Self::Wide as CoreRegister>::IS_EMULATED,
                "Wide matrix-vector multiplication requires true wide registers."
            );
        }

        // a = [c0 | c1] (contiguous -> folds into a 256-bit load); b = [c2 | 0].
        let a = Self::Wide::concat(cols[0], cols[1]);
        let b = <Self::Wide as ExtendRegister<Self>>::extend(cols[2]);

        // coef_ab = [x x x x | y y y y]; coef_c's high half is irrelevant (b is 0 there).
        let coef_ab = Self::Wide::concat(Self::broadcast::<0>(vector), Self::broadcast::<1>(vector));
        let coef_c = Self::Wide::concat(Self::broadcast::<2>(vector), Self::broadcast::<2>(vector));

        // [c0*x + c2*z | c1*y + 0]
        let prod = Self::Wide::mul_adde(b, coef_c, Self::Wide::mul(a, coef_ab));

        let (lo, hi) = Self::Wide::split(prod);
        Self::add(lo, hi)
    }

    /// Multiplies two 3x3 matrices (each stored as 3 column registers).
    ///
    /// This is "transform each of `rhs`'s 3 columns by `lhs`", so it inherits the
    /// (wide) batching of [`mat3_vec3_product`](Self::mat3_vec3_product). If
    /// `COLUMN_MAJOR` is `false` the operands are swapped (`rhs * lhs`) to account
    /// for row-major storage, mirroring [`mat4_product`](LinAlg4Register::mat4_product).
    #[inline(always)]
    fn mat3_product<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 3],
        rhs: &[Storage<Self>; 3],
    ) -> [Storage<Self>; 3] {
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };
        Self::mat3_vec3_product::<true, 3>(lhs, rhs)
    }

    /// Determinant of a column-major 3x3 matrix: the scalar triple product
    /// `c0 . (c1 x c2)`.
    ///
    /// In lieu of a full policy system, `FAST` is used to pick the internal behavior.
    /// It is forwarded to [`cross3`](Self::cross3), which this is built from. See
    /// there for what it does.
    #[inline(always)]
    fn mat3_det<const FAST: bool>(cols: &[Storage<Self>; 3]) -> Self::Element {
        Self::dot3(cols[0], Self::cross3::<FAST>(cols[1], cols[2]))
    }

    /// In-place inverse of a column-major 3x3 matrix; **returns the determinant**.
    ///
    /// Uses the cofactor/cross-product form: the rows of the inverse are
    /// `$c_1 \times c_2$`, `$c_2 \times c_0$`, `$c_0 \times c_1$`, each divided by the determinant.
    ///
    /// An **exactly-zero determinant leaves the matrix untouched**; a near-zero
    /// (ill-conditioned) determinant produces a finite but unreliable result, so
    /// inspect the returned determinant before trusting the matrix.
    ///
    /// In lieu of a full policy system, `FAST` is used to pick the internal behavior.
    /// It is forwarded to [`cross3`](Self::cross3), which the cofactors are built
    /// from, and also picks how the adjugate is scaled: `FAST=true` takes one
    /// reciprocal and multiplies, `FAST=false` divides once per column for a single
    /// rounding per entry.
    #[inline(always)]
    fn mat3_inverse<const FAST: bool>(cols: &mut [Storage<Self>; 3]) -> Self::Element {
        let [c0, c1, c2] = *cols;

        // Cofactor rows; these transpose into the inverse's columns.
        let r0 = Self::cross3::<FAST>(c1, c2);
        let r1 = Self::cross3::<FAST>(c2, c0);
        let r2 = Self::cross3::<FAST>(c0, c1);

        let d = Self::dot3(c0, r0);

        if crate::likely(!d.is_zero()) {
            let t = Self::mat3_transpose(&[r0, r1, r2]);

            if const { FAST } {
                let rcp = Self::div(Self::ONE, Self::splat(d));
                cols[0] = Self::mul(t[0], rcp);
                cols[1] = Self::mul(t[1], rcp);
                cols[2] = Self::mul(t[2], rcp);
            } else {
                let dv = Self::splat(d);
                cols[0] = Self::div(t[0], dv);
                cols[1] = Self::div(t[1], dv);
                cols[2] = Self::div(t[2], dv);
            }
        }

        d
    }

    /// "Normal matrix" for transforming normals/directions under non-uniform
    /// scale, built from the cofactor cross-products `$(c_1 \times c_2,\ c_2 \times c_0,\ c_0 \times c_1)$` of a
    /// column-major 3x3 (stored as its columns).
    ///
    /// `DIVIDE` selects the variant:
    /// - `true` -> the true inverse-transpose `$(M^{-1})^{T}$` (cofactors divided by the
    ///   determinant). A singular input yields non-finite results.
    /// - `false` -> the cofactor (adjugate-transpose) matrix, **un-divided**. This
    ///   skips the determinant and division entirely (and is never singular);
    ///   it transforms normals to the *same direction*, so it's the cheaper
    ///   choice whenever you re-normalize the result.
    ///
    /// Either way this is cheaper than [`mat3_inverse`](Self::mat3_inverse), it
    /// skips that method's transpose step.
    ///
    /// In lieu of a full policy system, `FAST` is used to pick the internal behavior.
    /// It is forwarded to [`cross3`](Self::cross3), which the cofactors are built
    /// from, and also picks how the cofactor matrix is scaled: `FAST=true` takes one
    /// reciprocal and multiplies, `FAST=false` divides once per column for a single
    /// rounding per entry. With `DIVIDE=false` there is no scale, so
    /// `FAST` only reaches `cross3`.
    #[inline(always)]
    fn mat3_normal<const DIVIDE: bool, const FAST: bool>(cols: &[Storage<Self>; 3]) -> [Storage<Self>; 3] {
        let [c0, c1, c2] = *cols;
        let a = Self::cross3::<FAST>(c1, c2);
        let b = Self::cross3::<FAST>(c2, c0);
        let c = Self::cross3::<FAST>(c0, c1);

        if const { DIVIDE } {
            let d = Self::dot3(c0, a);

            if const { FAST } {
                let rcp = Self::div(Self::ONE, Self::splat(d));
                [Self::mul(a, rcp), Self::mul(b, rcp), Self::mul(c, rcp)]
            } else {
                let dv = Self::splat(d);
                [Self::div(a, dv), Self::div(b, dv), Self::div(c, dv)]
            }
        } else {
            [a, b, c]
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
    ///
    /// In lieu of a full policy system, `FAST` is used to pick the internal
    /// behavior.
    ///
    /// `FAST = true` is the decomposition above with one product of each pair
    /// fused (5 FP ops). Good enough for almost every case, but the vector lanes
    /// of `q * conj(q)` retain the rounding error of the products that were
    /// supposed to cancel, because each cancelling pair compares an exact (fused)
    /// product against a rounded one. Measured on `f64` with
    /// `q = (0.1, 0.2, 0.3, 0.927...)`: `(1.5e-19, 0, -5.6e-17, 1.0)`.
    ///
    /// `FAST = false` regroups the terms geometrically (the vector part becomes
    /// `(w1*v2 + w2*v1) + (v1 x v2)`), so that every cancelling pair of the
    /// conjugate identity falls inside one compensated two-product group
    /// (9 FP ops, `cross3::<false>`'s scheme). `q * conj(q)` then comes out as
    /// exactly `(0, 0, 0, |q|^2)`, and away from cancellation it is
    /// *also* the more accurate arm: over 200k random f32 pairs, mean error 0.324
    /// against `FAST`'s 0.337 in units of the natural bound `u * sum|a_i b_i|`
    /// (p99 1.22 vs 1.23, max 2.20 vs 2.43). Cost on AVX2+FMA: 30 instructions
    /// against 23.
    ///
    /// On backends without a true FMA, `FAST = false` routes to the fast arm:
    /// the compensation residuals would be identically zero there, and the
    /// fast arm's unfused degradation already cancels the conjugate exactly
    /// (all products rounded), so the identity holds on every backend.
    ///
    /// The regrouping is the load-bearing part. In the component-broadcast
    /// decomposition the three vector lanes cancel across three *different*
    /// pairings of its four term-vectors, so exactness there demands treating
    /// every product identically. All rounded costs 7 ops and forfeits the
    /// FMAs, and with them the accuracy. All error-free-transformed costs 15
    /// ops, 41 instrs. Partial fusing and pair-wise Kahan both break a lane
    /// (the `y` lane surviving `FAST` is an accident of the arrangement).
    /// Re-shuffling the operands so the cancelling pairings become
    /// per-lane-local is what lets FMA compensation work at 9 ops. See the
    /// comments in the body.
    #[inline(always)]
    fn quat4_product<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        use crate::{math::FloatConsts as C, register::Element as E};

        let w = Self::broadcast::<3>(lhs);

        if const { FAST || !Self::HAS_TRUE_FMA } {
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
        } else {
            // Regrouped so every cancelling pair of `q * conj(q)` sits INSIDE one
            // compensated two-product group. The component-broadcast decomposition
            // above cannot do this at any fusing (its three vector lanes cancel
            // across three different pairings of its four term-vectors). These
            // shuffles instead realize the geometric split
            //
            //   vector = (w1*v2 + w2*v1) + (v1 x v2),   scalar = w1*w2 - v1.v2
            //
            // where the conjugate cancels within each parenthesis. Each group is a
            // sum of two products, so TwoProd-style compensation applies: one
            // product exact inside the FMA, the other's rounding residual
            // recovered exactly by a second FMA. For the conjugate the cancelling
            // terms then meet as `r` against `-r` (both exact residuals), and each
            // group comes out exactly zero. This is `cross3::<false>`'s scheme,
            // extended to the w-mixing group.
            //
            // This branch requires HAS_TRUE_FMA (gated above). Without it the
            // residuals would compute as fl(ab) - fl(ab) = exactly zero, leaving
            // only this arm's addition tree over the same four rounded products
            // the FAST arm degrades to. Measured a dead tie on random inputs
            // and ~13% better mean error near cancellation (tails identical),
            // so it is not worth the extra shuffles on legacy backends. The conjugate
            // identity stays exact on the non-FMA route regardless, because
            // all-rounded products cancel as fl(t) vs fl(-t) in any tree.

            // [ x1,  y1, z1, -x1]
            let a1 = Self::bitxor(
                s!(Self: lhs, [0, 1, 2, 0]),
                const { reg::<Self, 4>([E::ZERO, E::ZERO, E::ZERO, C::NEG_ZERO]) },
            );

            // [ y1, z1, x1, -y1]
            let a2 = Self::bitxor(
                s!(Self: lhs, [1, 2, 0, 1]),
                const { reg::<Self, 4>([E::ZERO, E::ZERO, E::ZERO, C::NEG_ZERO]) },
            );

            let b1 = s!(Self: rhs, [3, 3, 3, 0]); // [w2, w2, w2, x2]
            let b2 = s!(Self: rhs, [2, 0, 1, 1]); // [z2, x2, y2, y2]
            let c2 = s!(Self: lhs, [2, 0, 1, 2]); // [z1, x1, y1, z1]
            let d2 = s!(Self: rhs, [1, 2, 0, 2]); // [y2, z2, x2, z2]

            // Group 1: w1*rhs + a1*b1.
            let t1 = Self::mul(a1, b1);
            let e1 = Self::mul_sub(a1, b1, t1); // a1*b1 - fl(a1*b1), exact
            let s1 = Self::mul_add(w, rhs, t1);

            // Group 2: a2*b2 - c2*d2 (Kahan difference of products).
            let t2 = Self::mul(c2, d2);
            let e2 = Self::nmul_add(c2, d2, t2); // fl(c2*d2) - c2*d2, exact
            let s2 = Self::mul_sub(a2, b2, t2);

            // Group-wise so each (s, e) pair's exact zero survives the final add.
            Self::add(Self::add(s1, e1), Self::add(s2, e2))
        }
    }

    #[inline(always)]
    fn quat4_vec3_product<const FAST: bool>(q: Storage<Self>, v: Storage<Self>) -> Storage<Self> {
        // --- Fast method by Giesen ---
        // Formula: v + 2w(q x v) + 2(q x (q x v))

        let w = Self::broadcast::<3>(q);
        let q_xyz = q;

        // t = 2 * cross(q, v)
        let t = Self::cross3::<FAST>(q_xyz, v);
        let t = Self::add(t, t); // multiply by 2

        // result = v + w*t + cross(q, t)
        let w_t = Self::mul(w, t);
        let cross_q_t = Self::cross3::<FAST>(q_xyz, t);

        // compute wt + v first to allow for better instruction level parallelism,
        // while waiting on the cross product to complete
        Self::add(cross_q_t, Self::add(w_t, v))
    }

    /// Rotation matrix of a **unit** quaternion as 3 registers; the 4th lane of
    /// each is unspecified.
    ///
    /// `COLUMN_MAJOR` selects the storage (the registers are the rotation's
    /// columns when `true`, its rows when `false`) - i.e. `false` yields the
    /// transpose. It's free: only the sign masks differ at compile time.
    ///
    /// Trig-free: the entries are pairwise products of `{x, y, z, w}` assembled
    /// with FMAs - no `sin`/`cos`/`sqrt`. The quaternion is assumed normalized;
    /// normalize first if unsure.
    ///
    /// For rotating many vectors by one quaternion, prefer converting once here
    /// and batching through [`mat3_vec3_product`](LinAlg3Register::mat3_vec3_product)
    /// with the matching `COLUMN_MAJOR` - cheaper than a per-vector
    /// [`quat4_vec3_product`](Self::quat4_vec3_product) for large `N`.
    #[inline(always)]
    fn quat_to_mat3<const COLUMN_MAJOR: bool>(q: Storage<Self>) -> [Storage<Self>; 3] {
        use crate::{math::FloatConsts as C, register::Element as E};

        // 2q, with each component broadcast across all lanes.
        let q2 = Self::add(q, q);
        let x2 = Self::broadcast::<0>(q2);
        let y2 = Self::broadcast::<1>(q2);
        let z2 = Self::broadcast::<2>(q2);
        let w2 = Self::broadcast::<3>(q2);

        // The `2w` terms reuse the three `quat4_product` permutes. Only the sign
        // masks distinguish column-major (forward R) from row-major (R^T) - the
        // off-diagonal cross terms flip under transpose - so the choice is free.
        //   column: p0 = ( w,  z, -y, -x)  p1 = (-z,  w,  x, -y)  p2 = ( y, -x,  w, -z)
        //   row:    p0 = ( w, -z,  y, -x)  p1 = ( z,  w, -x, -y)  p2 = (-y,  x,  w, -z)
        let (m0, m1, m2) = if const { COLUMN_MAJOR } {
            (
                const { reg::<Self, 4>([E::ZERO, E::ZERO, C::NEG_ZERO, C::NEG_ZERO]) },
                const { reg::<Self, 4>([C::NEG_ZERO, E::ZERO, E::ZERO, C::NEG_ZERO]) },
                const { reg::<Self, 4>([E::ZERO, C::NEG_ZERO, E::ZERO, C::NEG_ZERO]) },
            )
        } else {
            (
                const { reg::<Self, 4>([E::ZERO, C::NEG_ZERO, E::ZERO, C::NEG_ZERO]) },
                const { reg::<Self, 4>([E::ZERO, E::ZERO, C::NEG_ZERO, C::NEG_ZERO]) },
                const { reg::<Self, 4>([C::NEG_ZERO, E::ZERO, E::ZERO, C::NEG_ZERO]) },
            )
        };

        let p0 = Self::bitxor(s!(Self: q, [3, 2, 1, 0]), m0);
        let p1 = Self::bitxor(s!(Self: q, [2, 3, 0, 1]), m1);
        let p2 = Self::bitxor(s!(Self: q, [1, 0, 3, 2]), m2);

        // out_j = (2*comp_j)*q + (2w*p_j - e_j). The trailing -e_j (the -1 on the
        // diagonal) folds into a fused multiply-sub, so no `-1` constant is needed.
        let e0 = const { reg::<Self, 4>([E::ONE, E::ZERO, E::ZERO, E::ZERO]) };
        let e1 = const { reg::<Self, 4>([E::ZERO, E::ONE, E::ZERO, E::ZERO]) };
        let e2 = const { reg::<Self, 4>([E::ZERO, E::ZERO, E::ONE, E::ZERO]) };

        let r0 = Self::mul_adde(x2, q, Self::mul_sube(w2, p0, e0));
        let r1 = Self::mul_adde(y2, q, Self::mul_sube(w2, p1, e1));
        let r2 = Self::mul_adde(z2, q, Self::mul_sube(w2, p2, e2));

        [r0, r1, r2]
    }

    /// Homogeneous 4x4 rotation matrix of a **unit** quaternion: the
    /// [`quat_to_mat3`](Self::quat_to_mat3) rotation in the upper-left 3x3 with
    /// each rotation register's 4th lane zeroed, plus a `[0, 0, 0, 1]` 4th
    /// register. `COLUMN_MAJOR` is forwarded to `quat_to_mat3`; the 4th register
    /// is identical either way since the translation is zero. Same trig-free,
    /// unit-quaternion assumptions as `quat_to_mat3`.
    #[inline(always)]
    fn quat_to_mat4<const COLUMN_MAJOR: bool>(q: Storage<Self>) -> [Storage<Self>; 4] {
        use crate::register::Element as E;
        let [c0, c1, c2] = Self::quat_to_mat3::<COLUMN_MAJOR>(q);
        let w_col = const { reg::<Self, 4>([E::ZERO, E::ZERO, E::ZERO, E::ONE]) };
        [Self::zero4(c0), Self::zero4(c1), Self::zero4(c2), w_col]
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

    /// 4x4 matrix times `N` 3D vectors (the 4th column is ignored).
    ///
    /// Same small-`N`, transpose-once semantics as [`mat4_vec4_product`](Self::mat4_vec4_product).
    #[inline(always)]
    fn mat4_vec3_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        let m = if const { COLUMN_MAJOR } {
            *cols
        } else {
            Self::mat4_transpose(cols)
        };

        let mut out = [Self::EMPTY; N];
        let mut i = 0;
        while i < N {
            let v = vectors[i];
            let x = Self::broadcast::<0>(v);
            let y = Self::broadcast::<1>(v);
            let z = Self::broadcast::<2>(v);
            out[i] = Self::mul_adde(m[2], z, Self::mul_adde(m[1], y, Self::mul(m[0], x)));
            i += 1;
        }
        out
    }

    /// 4x4 matrix times `N` 3D _points_. The 4th column is assumed to be multiplied with 1.0.
    ///
    /// Same small-`N`, transpose-once semantics as [`mat4_vec4_product`](Self::mat4_vec4_product).
    #[inline(always)]
    fn mat4_point3_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        let m = if const { COLUMN_MAJOR } {
            *cols
        } else {
            Self::mat4_transpose(cols)
        };

        let mut out = [Self::EMPTY; N];
        let mut i = 0;
        while i < N {
            let v = vectors[i];
            let x = Self::broadcast::<0>(v);
            let y = Self::broadcast::<1>(v);
            let z = Self::broadcast::<2>(v);
            out[i] = Self::mul_adde(m[2], z, Self::mul_adde(m[1], y, Self::mul_adde(m[0], x, m[3])));
            i += 1;
        }
        out
    }

    /// 4x4 matrix times `N` 4D vectors, returning the transformed array.
    ///
    /// Intended for **small** `N`: the array is taken/returned **by value** and
    /// the loop fully unrolls, so a large `N` bloats code/stack. Row-major
    /// matrices are transposed **once** up front (amortized over `N`). Backends
    /// with a true double-width register override this to process two vectors
    /// per wide pass; the [`Vector`](crate::Vector) layer exposes a single-vector
    /// convenience over this (`N == 1`).
    #[inline(always)]
    fn mat4_vec4_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        let m = if const { COLUMN_MAJOR } {
            *cols
        } else {
            Self::mat4_transpose(cols)
        };

        let mut out = [Self::EMPTY; N];
        let mut i = 0;
        while i < N {
            let v = vectors[i];
            let x = Self::broadcast::<0>(v);
            let y = Self::broadcast::<1>(v);
            let z = Self::broadcast::<2>(v);
            let w = Self::broadcast::<3>(v);

            // Two fused multiply-adds in parallel (ILP), then merge.
            out[i] = Self::add(
                Self::mul_adde(m[1], y, Self::mul(m[0], x)),
                Self::mul_adde(m[3], w, Self::mul(m[2], z)),
            );
            i += 1;
        }
        out
    }

    /// Multiplies two 4x4 Matrices.
    #[inline(always)]
    fn mat4_product<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 4],
        rhs: &[Storage<Self>; 4],
    ) -> [Storage<Self>; 4] {
        // swap operands if not column-major
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };

        // Multiplying is transforming each of `rhs`'s 4 columns by `lhs`.
        Self::mat4_vec4_product::<true, 4>(lhs, rhs)
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

    /// Column-major 4x4 * vec4, evaluated through the double-width register.
    ///
    /// Packs two basis columns per 256-bit lane (`[c0|c1]`, `[c2|c3]`) so the
    /// four column-scales collapse into a single wide multiply + a single wide
    /// FMA (instead of two 128-bit FMAs), with the coefficients built by in-lane
    /// shuffles rather than four separate broadcasts. The two halves of the
    /// product are then summed to give `c0*x + c1*y + c2*z + c3*w`.
    ///
    /// Assumes **column-major**. Requires a true wide register.
    #[inline(always)]
    fn mat4_vec4_product_wide(cols: &[Storage<Self>; 4], vector: Storage<Self>) -> Storage<Self>
    where
        Self: WideRegister<Wide: FloatRegister>,
    {
        const {
            assert!(
                !<Self::Wide as CoreRegister>::IS_EMULATED,
                "Wide matrix-vector multiplication requires true wide registers."
            );
        }

        // Pack columns two-per-lane: a = [c0 | c1], b = [c2 | c3].
        // (Contiguous in the source array, so these fold into 256-bit loads.)
        let a = Self::Wide::concat(cols[0], cols[1]);
        let b = Self::Wide::concat(cols[2], cols[3]);

        // Coefficients, built from in-lane shuffles of `vector` (the compiler
        // lowers concat(broadcast(i), broadcast(j)) to vbroadcastf128 + vpermilps):
        //   coef_ab = [x x x x | y y y y],  coef_cd = [z z z z | w w w w]
        let coef_ab = Self::Wide::concat(Self::broadcast::<0>(vector), Self::broadcast::<1>(vector));
        let coef_cd = Self::Wide::concat(Self::broadcast::<2>(vector), Self::broadcast::<3>(vector));

        // [c0*x + c2*z | c1*y + c3*w] in one wide multiply + one wide FMA.
        let prod = Self::Wide::mul_adde(b, coef_cd, Self::Wide::mul(a, coef_ab));

        // Fold the two halves: (c0*x + c2*z) + (c1*y + c3*w).
        let (lo, hi) = Self::Wide::split(prod);

        Self::add(lo, hi)
    }

    /// Column-major 4x4 * vec3 evaluated through the double-width register.
    ///
    /// A vec3 only uses the first three columns, so this is the 3-term sibling of
    /// [`mat4_vec4_product_wide`](Self::mat4_vec4_product_wide): it packs `[c0 | c1]` and `[c2 | 0]` (the third
    /// column zero-extended), collapsing the column-scales into one wide multiply
    /// plus one wide FMA, then sums the halves. Assumes **column-major**. Requires a
    /// true wide register.
    #[inline(always)]
    fn mat4_vec3_product_wide(cols: &[Storage<Self>; 4], vector: Storage<Self>) -> Storage<Self>
    where
        Self: WideRegister<Wide: FloatRegister>,
    {
        const {
            assert!(
                !<Self::Wide as CoreRegister>::IS_EMULATED,
                "Wide matrix-vector multiplication requires true wide registers."
            );
        }

        // a = [c0 | c1] (folds into a 256-bit load); b = [c2 | 0] (zero-extended).
        let a = Self::Wide::concat(cols[0], cols[1]);
        let b = <Self::Wide as ExtendRegister<Self>>::extend(cols[2]);

        // coef_ab = [x x x x | y y y y]; coef_c's high half is irrelevant (b is 0 there).
        let coef_ab = Self::Wide::concat(Self::broadcast::<0>(vector), Self::broadcast::<1>(vector));
        let coef_c = Self::Wide::concat(Self::broadcast::<2>(vector), Self::broadcast::<2>(vector));

        // [c0*x + c2*z | c1*y]
        let prod = Self::Wide::mul_adde(b, coef_c, Self::Wide::mul(a, coef_ab));

        let (lo, hi) = Self::Wide::split(prod);
        Self::add(lo, hi)
    }

    /// The UNSCALED adjugate of a column-major 4x4 matrix, plus its determinant.
    ///
    /// The override point for the 4x4 inverse: a backend supplies its own swizzle
    /// here and inherits [`mat4_inverse`](Self::mat4_inverse) unchanged.
    #[inline(always)]
    fn mat4_adjugate<const FAST: bool>(m: &[Storage<Self>; 4]) -> ([Storage<Self>; 4], Self::Element) {
        // standard implementation using swizzle macro that
        // invokes permutev/swizzle meta-instructions
        impl_mat4_inverse!(m, s)
    }

    /// Full in-place 4x4 inverse; **returns the determinant**.
    ///
    /// An exactly-zero determinant leaves the matrix untouched; a near-zero
    /// (ill-conditioned) determinant produces a finite but unreliable result, so
    /// inspect the returned determinant before trusting the matrix.
    ///
    /// Backends override [`mat4_adjugate`](Self::mat4_adjugate), not this.
    #[inline(always)]
    fn mat4_inverse<const FAST: bool>(m: &mut [Storage<Self>; 4]) -> Self::Element {
        use num_traits::Zero as _;

        let (adj, det) = Self::mat4_adjugate::<FAST>(m);

        // Leave the matrix untouched for an exactly-singular determinant (a
        // well-predicted branch).
        if crate::likely(!det.is_zero()) {
            mat4_scale!(m, adj, det);
        }

        det
    }

    /// Determinant of a column-major 4x4 matrix.
    #[inline(always)]
    fn mat4_det<const FAST: bool>(m: &[Storage<Self>; 4]) -> Self::Element {
        impl_mat4_inverse!(DET_ONLY m, s)
    }
}
