#[rustfmt::skip]
macro_rules! decl_binary_ops {
    ($kind:ident $(: $unary:ident)?; $($trait_name:ident::$method_name:ident),*) => {paste::paste! {
        #[doc = "Combination of masked binary operation traits for " $kind " operations."]
        pub trait [<Masked $kind Ops>]<Mask, Rhs>: $($unary<Mask> +)? $([<$trait_name Masked>]<Mask, Rhs> + )* {}

        #[doc = "Combination of masked assignment binary operation traits for " $kind " operations."]
        pub trait [<AssignMasked $kind Ops>]<Mask, Rhs>: $([<$trait_name AssignMasked>]<Mask, Rhs> + )* {}

        impl<V, Mask, Rhs> [<Masked $kind Ops>]<Mask, Rhs> for V
        where
            $(V: $unary<Mask>,)?
            V: $([<$trait_name Masked>]<Mask, Rhs> + )*
        {}

        impl<V, Mask, Rhs> [<AssignMasked $kind Ops>]<Mask, Rhs> for V
        where
            $(V: [<$trait_name AssignMasked>]<Mask, Rhs>),*
        {}

        $(
            #[doc = "Masked variants of the [`" $trait_name "`] trait."]
            pub trait [<$trait_name Masked>]<Mask, Rhs = Self>: $trait_name<Rhs> {
                #[doc = "Computes [`" $trait_name "`] with `rhs` where `mask` is true."]
                fn [<$method_name _c>](self, mask: Mask, rhs: Rhs) -> Self::Output;

                #[doc = "Merges [`" $trait_name "`] with `src` using `mask`, returning `src` where mask is false."]
                fn [<$method_name _m>](self, src: Self, mask: Mask, rhs: Rhs) -> Self::Output;

                #[doc = "Computes [`" $trait_name "`] masked (zeroed where mask is false)."]
                fn [<$method_name _z>](self, mask: Mask, rhs: Rhs) -> Self::Output;
            }

            #[doc = "Masked assignment variants of the [`" $trait_name "`] trait."]
            pub trait [<$trait_name AssignMasked>]<Mask, Rhs = Self>: [<$trait_name Assign>]<Rhs> {
                #[doc = "Computes [`" $trait_name "Assign`] with `rhs` where `mask` is true."]
                fn [<$method_name _assign_c>](&mut self, mask: Mask, rhs: Rhs);

                #[doc = "Merges [`" $trait_name "Assign`] with `src` using `mask`, assigning `src` where mask is false."]
                fn [<$method_name _assign_m>](&mut self, src: Self, mask: Mask, rhs: Rhs);

                #[doc = "Computes [`" $trait_name "Assign`] masked (zeroed where mask is false)."]
                fn [<$method_name _assign_z>](&mut self, mask: Mask, rhs: Rhs);
            }
        )*
    }};
}

macro_rules! decl_unary_ops {
    ($($trait_name:ident::$method_name:ident),*) => {paste::paste! {$(
        #[doc = "Masked variants of the [`" $trait_name "`] trait."]
        pub trait [<$trait_name Masked>]<Mask>: $trait_name {
            #[doc = "Computes [`" $trait_name "`] where `mask` is true, does nothing where false."]
            fn [<$method_name _c>](self, mask: Mask) -> Self::Output;
            #[doc = "Merges [`" $trait_name "`] with `src` using `mask`, returning `src` where mask is false."]
            fn [<$method_name _m>](self, src: Self, mask: Mask) -> Self::Output;
            #[doc = "Computes [`" $trait_name "`] masked (zeroed where mask is false)."]
            fn [<$method_name _z>](self, mask: Mask) -> Self::Output;
        }
    )*}};
}

macro_rules! impl_binary_op {
    ($trait_name:ident::$method_name:ident for $reg:ident, $rhs:ty) => {
        paste::paste! {
            impl<R: $reg + Register> $trait_name<$rhs> for Vector<R> {
                type Output = Self;

                #[inline(always)]
                fn $method_name(self, rhs: $rhs) -> Self::Output {
                    Vector(R::$method_name(self.0, rhs.0))
                }
            }

            impl<R: $reg + Register> [<$trait_name Masked>]<Mask<R>, $rhs> for Vector<R> {
                #[inline(always)]
                fn [<$method_name _c>](self, mask: Mask<R>, rhs: $rhs) -> Self::Output {
                    Vector(R::[<$method_name _c>](mask.0, self.0, rhs.0))
                }

                #[inline(always)]
                fn [<$method_name _m>](self, src: Self, mask: Mask<R>, rhs: $rhs) -> Self::Output {
                    Vector(R::[<$method_name _m>](src.0, mask.0, self.0, rhs.0))
                }

                #[inline(always)]
                fn [<$method_name _z>](self, mask: Mask<R>, rhs: $rhs) -> Self::Output {
                    Vector(R::[<$method_name _z>](mask.0, self.0, rhs.0))
                }
            }

            impl<R: $reg + Register> [<$trait_name Assign>]<$rhs> for Vector<R> {
                #[inline(always)]
                fn [<$method_name _assign>](&mut self, rhs: $rhs) {
                    self.0 = R::$method_name(self.0, rhs.0);
                }
            }

            impl<R: $reg + Register> [<$trait_name AssignMasked>]<Mask<R>, $rhs> for Vector<R> {
                #[inline(always)]
                fn [<$method_name _assign_c>](&mut self, mask: Mask<R>, rhs: $rhs) {
                    self.0 = R::[<$method_name _c>](mask.0, self.0, rhs.0);
                }

                #[inline(always)]
                fn [<$method_name _assign_m>](&mut self, src: Self, mask: Mask<R>, rhs: $rhs) {
                    self.0 = R::[<$method_name _m>](src.0, mask.0, self.0, rhs.0);
                }

                #[inline(always)]
                fn [<$method_name _assign_z>](&mut self, mask: Mask<R>, rhs: $rhs) {
                    self.0 = R::[<$method_name _z>](mask.0, self.0, rhs.0);
                }
            }
        }
    };
}

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign,
    Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use crate::{
    Mask, Vector,
    register::{BitshiftRegister, BitwiseRegister, FloatRegister, NumericRegister, Register, SignedRegister},
};

/// Trait for squaring a value: `self * self`
///
/// Some types are able to provide optimized implementations of squaring that are
/// faster or more accurate than a simple multiplication with itself.
pub trait Square {
    /// The squared value. Not always `Self`: a type may widen to hold the product.
    type Output;

    /// Computes `self * self`.
    fn square(self) -> Self::Output;
}

decl_binary_ops!(Num;
    Add::add,
    Sub::sub,
    Mul::mul,
    Div::div,
    Rem::rem
);

impl_binary_op!(Add::add for NumericRegister, Self);
impl_binary_op!(Sub::sub for NumericRegister, Self);
impl_binary_op!(Mul::mul for NumericRegister, Self);
impl_binary_op!(Div::div for NumericRegister, Self);
impl_binary_op!(Rem::rem for NumericRegister, Self);

/// Trait for the bitwise AND NOT operation: `self & !rhs`
pub trait BitAndNot<Rhs = Self> {
    /// The result of the AND NOT.
    type Output;

    /// Computes `self & !rhs`, one instruction on every SIMD backend.
    ///
    /// The **second** operand is the negated one here.
    #[must_use]
    fn bitandnot(self, rhs: Rhs) -> Self::Output;
}

/// Trait for the bitwise AND NOT assignment operation: `self &= !rhs`
pub trait BitAndNotAssign<Rhs = Self> {
    /// Assigns `self & !rhs` into `self`.
    fn bitandnot_assign(&mut self, rhs: Rhs);
}

decl_binary_ops!(Bitwise;
    BitAnd::bitand,
    BitAndNot::bitandnot,
    BitOr::bitor,
    BitXor::bitxor
);

impl_binary_op!(BitAnd::bitand for BitwiseRegister, Self);
impl_binary_op!(BitOr::bitor for BitwiseRegister, Self);
impl_binary_op!(BitXor::bitxor for BitwiseRegister, Self);

decl_binary_ops!(Bitshift;
    Shl::shl,
    Shr::shr
);

decl_unary_ops!(Not::not, Neg::neg, Square::square);

impl<R: BitwiseRegister + Register> Not for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Vector(R::not(self.0))
    }
}

impl<R: BitwiseRegister + Register> NotMasked<Mask<R>> for Vector<R> {
    #[inline(always)]
    fn not_c(self, mask: Mask<R>) -> Self::Output {
        Vector(R::not_c(mask.0, self.0))
    }

    #[inline(always)]
    fn not_m(self, src: Self, mask: Mask<R>) -> Self::Output {
        Vector(R::not_m(src.0, mask.0, self.0))
    }

    #[inline(always)]
    fn not_z(self, mask: Mask<R>) -> Self::Output {
        Vector(R::not_z(mask.0, self.0))
    }
}

impl<R: SignedRegister> Neg for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vector(R::neg(self.0))
    }
}

impl<R: SignedRegister> NegMasked<Mask<R>> for Vector<R> {
    #[inline(always)]
    fn neg_c(self, mask: Mask<R>) -> Self::Output {
        Vector(R::neg_c(mask.0, self.0))
    }

    #[inline(always)]
    fn neg_m(self, src: Self, mask: Mask<R>) -> Self::Output {
        Vector(R::neg_m(src.0, mask.0, self.0))
    }

    #[inline(always)]
    fn neg_z(self, mask: Mask<R>) -> Self::Output {
        Vector(R::neg_z(mask.0, self.0))
    }
}

// NOTE: BitAndNot is unique in that the BitwiseRegister trait expects !lhs & rhs,
// but since we want lhs & !rhs, the order of parameters is reversed here.
impl<R: BitwiseRegister + Register> BitAndNot<Self> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn bitandnot(self, rhs: Self) -> Self::Output {
        Vector(R::bitandnot(rhs.0, self.0))
    }
}

impl<R: BitwiseRegister + Register> BitAndNotMasked<Mask<R>, Self> for Vector<R> {
    #[inline(always)]
    fn bitandnot_c(self, mask: Mask<R>, rhs: Self) -> Self::Output {
        Vector(R::bitandnot_c(mask.0, rhs.0, self.0))
    }

    #[inline(always)]
    fn bitandnot_m(self, src: Self, mask: Mask<R>, rhs: Self) -> Self::Output {
        Vector(R::bitandnot_m(src.0, mask.0, rhs.0, self.0))
    }

    #[inline(always)]
    fn bitandnot_z(self, mask: Mask<R>, rhs: Self) -> Self::Output {
        Vector(R::bitandnot_z(mask.0, rhs.0, self.0))
    }
}

impl<R: BitwiseRegister + Register> BitAndNotAssign<Self> for Vector<R> {
    #[inline(always)]
    fn bitandnot_assign(&mut self, rhs: Self) {
        self.0 = R::bitandnot(rhs.0, self.0);
    }
}

impl<R: BitwiseRegister + Register> BitAndNotAssignMasked<Mask<R>, Self> for Vector<R> {
    #[inline(always)]
    fn bitandnot_assign_c(&mut self, mask: Mask<R>, rhs: Self) {
        self.0 = R::bitandnot_c(mask.0, rhs.0, self.0);
    }

    #[inline(always)]
    fn bitandnot_assign_m(&mut self, src: Self, mask: Mask<R>, rhs: Self) {
        self.0 = R::bitandnot_m(src.0, mask.0, rhs.0, self.0);
    }

    #[inline(always)]
    fn bitandnot_assign_z(&mut self, mask: Mask<R>, rhs: Self) {
        self.0 = R::bitandnot_z(mask.0, rhs.0, self.0);
    }
}

macro_rules! mul_add_ext {
    ($($(#[$meta:meta])* $name:ident),*) => {paste::paste! {
        /// Trait for fused multiply-add operations. This contains all variants of
        /// fused multiply-add, including guaranteed FMA and maybe FMA versions.
        ///
        /// If the platform does _not_ have native FMA support, the guaranteed FMA
        /// will either use a fast compensated arithmetic algorithm, or fall back to
        /// slow scalar evaluation to ensure correctness.
        ///
        /// This trait is superior to `num_traits::MulAdd`, but does not include
        /// it as a supertrait due to the potential for multiple
        /// conflicting implementation warnings.
        pub trait MulAddExt<A = Self, B = Self> {
            /// The result of the fused operation.
            type Output;

            /// Indicates whether the implementation uses true fused-multiply-add instructions.
            ///
            /// Non-`e` variants will always be accurate, regardless of this flag, but the `e` variants
            /// will fallback to separate multiply and add operations if this is false.
            const HAS_TRUE_FMA: bool;

            $(
                $(#[$meta])*
                fn $name(self, a: A, b: B) -> Self::Output;
            )*
        }

        impl<R: FloatRegister> MulAddExt<Self, Self> for Vector<R> {
            type Output = Self;

            const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;

            $(#[inline(always)] fn $name(self, a: Self, b: Self) -> Self::Output { Vector(R::$name(self.0, a.0, b.0)) } )*
        }

        /// Provides assignment variants of the fused multiply-add operations from [`MulAddExt`].
        ///
        /// Unlike regular assignment traits, this does require `MulAddExt` as a supertrait,
        /// so we can access the associated `HAS_TRUE_FMA` constant.
        pub trait MulAddAssignExt<A = Self, B = Self>: MulAddExt<A, B> {
            $(
                $(#[$meta])*
                fn [<$name _assign>](&mut self, a: A, b: B);
            )*
        }

        impl<R: FloatRegister> MulAddAssignExt<Self, Self> for Vector<R> {
            $(#[inline(always)] fn [<$name _assign>](&mut self, a: Self, b: Self) { self.0 = R::$name(self.0, a.0, b.0); } )*
        }

        /// Provides masked variants of the fused multiply-add operations from [`MulAddExt`].
        pub trait MulAddExtMasked<Mask, A = Self, B = Self>: MulAddExt<A, B> {
            $(
                $(#[$meta])*
                #[doc = "\n\nThis variant computes [`" $name "`](MulAddExt::" $name ") with `a` and `b` where `mask` is true."]
                fn [<$name _c>](self, mask: Mask, a: A, b: B) -> Self::Output;

                $(#[$meta])*
                #[doc = "\n\nThis variant merges [`" $name "`](MulAddExt::" $name ") with `src` using `mask`, returning `src` where mask is false."]
                fn [<$name _m>](self, src: Self, mask: Mask, a: A, b: B) -> Self::Output;

                $(#[$meta])*
                #[doc = "\n\nThis variant computes [`" $name "`](MulAddExt::" $name ") masked (zeroed where mask is false)."]
                fn [<$name _z>](self, mask: Mask, a: A, b: B) -> Self::Output;
            )*
        }

        impl<R: FloatRegister> MulAddExtMasked<Mask<R>, Self, Self> for Vector<R> {
            $(
                #[inline(always)] fn [<$name _c>](self, mask: Mask<R>, a: Self, b: Self) -> Self::Output { Vector(R::[<$name _c>](mask.0, self.0, a.0, b.0)) }
                #[inline(always)] fn [<$name _m>](self, src: Self, mask: Mask<R>, a: Self, b: Self) -> Self::Output { Vector(R::[<$name _m>](src.0, mask.0, self.0, a.0, b.0)) }
                #[inline(always)] fn [<$name _z>](self, mask: Mask<R>, a: Self, b: Self) -> Self::Output { Vector(R::[<$name _z>](mask.0, self.0, a.0, b.0)) }
            )*
        }

        /// Provides masked assignment variants of the fused multiply-add operations from [`MulAddExt`].
        pub trait MulAddAssignExtMasked<Mask, A = Self, B = Self>: MulAddAssignExt<A, B> {
            $(
                $(#[$meta])*
                #[doc = "\n\nThis variant computes [`" $name "_assign`](MulAddAssignExt::" $name "_assign) with `a` and `b` where `mask` is true."]
                fn [<$name _assign_c>](&mut self, mask: Mask, a: A, b: B);

                $(#[$meta])*
                #[doc = "\n\nThis variant merges [`" $name "_assign`](MulAddAssignExt::" $name "_assign) with `src` using `mask`, assigning `src` where mask is false."]
                fn [<$name _assign_m>](&mut self, src: Self, mask: Mask, a: A, b: B);

                $(#[$meta])*
                #[doc = "\n\nThis variant computes [`" $name "_assign`](MulAddAssignExt::" $name "_assign) masked (zeroed where mask is false)."]
                fn [<$name _assign_z>](&mut self, mask: Mask, a: A, b: B);
            )*
        }

        impl<R: FloatRegister> MulAddAssignExtMasked<Mask<R>, Self, Self> for Vector<R> {
            $(
                #[inline(always)] fn [<$name _assign_c>](&mut self, mask: Mask<R>, a: Self, b: Self) { self.0 = R::[<$name _c>](mask.0, self.0, a.0, b.0); }
                #[inline(always)] fn [<$name _assign_m>](&mut self, src: Self, mask: Mask<R>, a: Self, b: Self) { self.0 = R::[<$name _m>](src.0, mask.0, self.0, a.0, b.0); }
                #[inline(always)] fn [<$name _assign_z>](&mut self, mask: Mask<R>, a: Self, b: Self) { self.0 = R::[<$name _z>](mask.0, self.0, a.0, b.0); }
            )*
        }
    }};
}

mul_add_ext! {
    /// Guaranteed fused-multiply-add operation.
    ///
    /// If the target architecture does not support native FMA, this will use
    /// either compensated arithmetic or slow scalar evaluation to ensure correctness.
    mul_add,

    /// Guaranteed fused-multiply-subtract operation.
    ///
    /// If the target architecture does not support native FMA, this will use
    /// either compensated arithmetic or slow scalar evaluation to ensure correctness.
    mul_sub,

    /// Guaranteed fused-negated-multiply-add operation.
    ///
    /// If the target architecture does not support native FMA, this will use
    /// either compensated arithmetic or slow scalar evaluation to ensure correctness.
    nmul_add,

    /// Guaranteed fused-negated-multiply-subtract operation.
    ///
    /// If the target architecture does not support native FMA, this will use
    /// either compensated arithmetic or slow scalar evaluation to ensure correctness.
    nmul_sub,

    /// Fused-multiply-add operation where possible. May gracefully degrade to separate multiply and add
    /// if the target architecture does not support native FMA.
    mul_adde,

    /// Fused-multiply-subtract operation where possible. May gracefully degrade to separate multiply and subtract
    /// if the target architecture does not support native FMA.
    mul_sube,

    /// Fused-negated-multiply-add operation where possible. May gracefully degrade to separate multiply and add
    /// if the target architecture does not support native FMA.
    nmul_adde,

    /// Fused-negated-multiply-subtract operation where possible. May gracefully degrade to separate multiply and subtract
    /// if the target architecture does not support native FMA.
    nmul_sube
}

/// Lane-alternating add/subtract operations for interleaved data (e.g. complex
/// numbers packed as `[re, im, re, im, ...]`).
///
/// All three follow the x86 `ADDSUB`/`FMADDSUB` convention: **even lanes
/// subtract, odd lanes add** (`fmsubadd` is the opposite parity). See
/// [`FloatRegister::addsub`] for the
/// interleaved complex-multiply lowering these are built for.
pub trait AddSubExt: Sized {
    /// The result of the lane-alternating operation.
    type Output;

    /// `[a0 - b0, a1 + b1, a2 - b2, ...]` - even lanes subtract, odd lanes add.
    fn addsub(self, b: Self) -> Self::Output;

    /// `[a0*b0 - c0, a1*b1 + c1, ...]` - fused multiply then [`addsub`](Self::addsub).
    fn fmaddsub(self, b: Self, c: Self) -> Self::Output;

    /// `[a0*b0 + c0, a1*b1 - c1, ...]` - fused multiply then subadd (opposite parity).
    fn fmsubadd(self, b: Self, c: Self) -> Self::Output;
}

impl<R: FloatRegister> AddSubExt for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn addsub(self, b: Self) -> Self {
        Vector(R::addsub(self.0, b.0))
    }

    #[inline(always)]
    fn fmaddsub(self, b: Self, c: Self) -> Self {
        Vector(R::fmaddsub(self.0, b.0, c.0))
    }

    #[inline(always)]
    fn fmsubadd(self, b: Self, c: Self) -> Self {
        Vector(R::fmsubadd(self.0, b.0, c.0))
    }
}

/// Masked variants of [`AddSubExt`] (mask-first argument order, matching the rest
/// of Thermite's `_c`/`_m`/`_z` surface).
pub trait AddSubExtMasked<Mask>: AddSubExt {
    /// [`addsub`](AddSubExt::addsub) where `mask` is true, else `self`.
    fn addsub_c(self, mask: Mask, b: Self) -> Self::Output;
    /// [`addsub`](AddSubExt::addsub) where `mask` is true, else `src`.
    fn addsub_m(self, src: Self, mask: Mask, b: Self) -> Self::Output;
    /// [`addsub`](AddSubExt::addsub) where `mask` is true, else zero.
    fn addsub_z(self, mask: Mask, b: Self) -> Self::Output;

    /// [`fmaddsub`](AddSubExt::fmaddsub) where `mask` is true, else `self`.
    fn fmaddsub_c(self, mask: Mask, b: Self, c: Self) -> Self::Output;
    /// [`fmaddsub`](AddSubExt::fmaddsub) where `mask` is true, else `src`.
    fn fmaddsub_m(self, src: Self, mask: Mask, b: Self, c: Self) -> Self::Output;
    /// [`fmaddsub`](AddSubExt::fmaddsub) where `mask` is true, else zero.
    fn fmaddsub_z(self, mask: Mask, b: Self, c: Self) -> Self::Output;

    /// [`fmsubadd`](AddSubExt::fmsubadd) where `mask` is true, else `self`.
    fn fmsubadd_c(self, mask: Mask, b: Self, c: Self) -> Self::Output;
    /// [`fmsubadd`](AddSubExt::fmsubadd) where `mask` is true, else `src`.
    fn fmsubadd_m(self, src: Self, mask: Mask, b: Self, c: Self) -> Self::Output;
    /// [`fmsubadd`](AddSubExt::fmsubadd) where `mask` is true, else zero.
    fn fmsubadd_z(self, mask: Mask, b: Self, c: Self) -> Self::Output;
}

impl<R: FloatRegister> AddSubExtMasked<Mask<R>> for Vector<R> {
    #[inline(always)]
    fn addsub_c(self, mask: Mask<R>, b: Self) -> Self {
        Vector(R::addsub_c(mask.0, self.0, b.0))
    }
    #[inline(always)]
    fn addsub_m(self, src: Self, mask: Mask<R>, b: Self) -> Self {
        Vector(R::addsub_m(src.0, mask.0, self.0, b.0))
    }
    #[inline(always)]
    fn addsub_z(self, mask: Mask<R>, b: Self) -> Self {
        Vector(R::addsub_z(mask.0, self.0, b.0))
    }

    #[inline(always)]
    fn fmaddsub_c(self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmaddsub_c(mask.0, self.0, b.0, c.0))
    }
    #[inline(always)]
    fn fmaddsub_m(self, src: Self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmaddsub_m(src.0, mask.0, self.0, b.0, c.0))
    }
    #[inline(always)]
    fn fmaddsub_z(self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmaddsub_z(mask.0, self.0, b.0, c.0))
    }

    #[inline(always)]
    fn fmsubadd_c(self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmsubadd_c(mask.0, self.0, b.0, c.0))
    }
    #[inline(always)]
    fn fmsubadd_m(self, src: Self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmsubadd_m(src.0, mask.0, self.0, b.0, c.0))
    }
    #[inline(always)]
    fn fmsubadd_z(self, mask: Mask<R>, b: Self, c: Self) -> Self {
        Vector(R::fmsubadd_z(mask.0, self.0, b.0, c.0))
    }
}

// Vector shifts

impl<R: BitshiftRegister> Shl<Vector<R::Unsigned>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shlv(self.0, rhs.0))
    }
}

impl<R: BitshiftRegister> ShlMasked<Mask<R>, Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shl_c(self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shlv_c(mask.0, self.0, rhs.0))
    }

    #[inline(always)]
    fn shl_m(self, src: Self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shlv_m(src.0, mask.0, self.0, rhs.0))
    }

    #[inline(always)]
    fn shl_z(self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shlv_z(mask.0, self.0, rhs.0))
    }
}

impl<R: BitshiftRegister> Shr<Vector<R::Unsigned>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shrv(self.0, rhs.0))
    }
}

impl<R: BitshiftRegister> ShrMasked<Mask<R>, Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shr_c(self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shrv_c(mask.0, self.0, rhs.0))
    }

    #[inline(always)]
    fn shr_m(self, src: Self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shrv_m(src.0, mask.0, self.0, rhs.0))
    }

    #[inline(always)]
    fn shr_z(self, mask: Mask<R>, rhs: Vector<R::Unsigned>) -> Self::Output {
        Vector(R::shrv_z(mask.0, self.0, rhs.0))
    }
}

impl<R: BitshiftRegister> ShlAssign<Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: Vector<R::Unsigned>) {
        self.0 = R::shlv(self.0, rhs.0);
    }
}

impl<R: BitshiftRegister> ShlAssignMasked<Mask<R>, Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shl_assign_c(&mut self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shlv_c(mask.0, self.0, rhs.0);
    }

    #[inline(always)]
    fn shl_assign_m(&mut self, src: Self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shlv_m(src.0, mask.0, self.0, rhs.0);
    }

    #[inline(always)]
    fn shl_assign_z(&mut self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shlv_z(mask.0, self.0, rhs.0);
    }
}

impl<R: BitshiftRegister> ShrAssign<Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: Vector<R::Unsigned>) {
        self.0 = R::shrv(self.0, rhs.0);
    }
}

impl<R: BitshiftRegister> ShrAssignMasked<Mask<R>, Vector<R::Unsigned>> for Vector<R> {
    #[inline(always)]
    fn shr_assign_c(&mut self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shrv_c(mask.0, self.0, rhs.0);
    }

    #[inline(always)]
    fn shr_assign_m(&mut self, src: Self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shrv_m(src.0, mask.0, self.0, rhs.0);
    }

    #[inline(always)]
    fn shr_assign_z(&mut self, mask: Mask<R>, rhs: Vector<R::Unsigned>) {
        self.0 = R::shrv_z(mask.0, self.0, rhs.0);
    }
}

// Scalar shifts

impl<R: BitshiftRegister> Shl<u32> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Vector(R::shl(self.0, rhs))
    }
}

impl<R: BitshiftRegister> ShlMasked<Mask<R>, u32> for Vector<R> {
    #[inline(always)]
    fn shl_c(self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shl_c(mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn shl_m(self, src: Self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shl_m(src.0, mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn shl_z(self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shl_z(mask.0, self.0, rhs))
    }
}

impl<R: BitshiftRegister> Shr<u32> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Vector(R::shr(self.0, rhs))
    }
}

impl<R: BitshiftRegister> ShrMasked<Mask<R>, u32> for Vector<R> {
    #[inline(always)]
    fn shr_c(self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shr_c(mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn shr_m(self, src: Self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shr_m(src.0, mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn shr_z(self, mask: Mask<R>, rhs: u32) -> Self::Output {
        Vector(R::shr_z(mask.0, self.0, rhs))
    }
}

impl<R: BitshiftRegister> ShlAssign<u32> for Vector<R> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 = R::shl(self.0, rhs);
    }
}

impl<R: BitshiftRegister> ShlAssignMasked<Mask<R>, u32> for Vector<R> {
    #[inline(always)]
    fn shl_assign_c(&mut self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shl_c(mask.0, self.0, rhs);
    }

    #[inline(always)]
    fn shl_assign_m(&mut self, src: Self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shl_m(src.0, mask.0, self.0, rhs);
    }

    #[inline(always)]
    fn shl_assign_z(&mut self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shl_z(mask.0, self.0, rhs);
    }
}

impl<R: BitshiftRegister> ShrAssign<u32> for Vector<R> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 = R::shr(self.0, rhs);
    }
}

impl<R: BitshiftRegister> ShrAssignMasked<Mask<R>, u32> for Vector<R> {
    #[inline(always)]
    fn shr_assign_c(&mut self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shr_c(mask.0, self.0, rhs);
    }

    #[inline(always)]
    fn shr_assign_m(&mut self, src: Self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shr_m(src.0, mask.0, self.0, rhs);
    }

    #[inline(always)]
    fn shr_assign_z(&mut self, mask: Mask<R>, rhs: u32) {
        self.0 = R::shr_z(mask.0, self.0, rhs);
    }
}

macro_rules! impl_square {
    ($($ty:ty),*) => {$(
        impl Square for $ty {
            type Output = Self;

            #[inline(always)]
            fn square(self) -> Self::Output {
                self * self
            }
        }
    )*};
}

impl_square!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
