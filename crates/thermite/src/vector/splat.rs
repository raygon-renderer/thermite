use core::ops::Add;

use generic_array::{ArrayLength, GenericArray};

use super::{Element, GenericVector, Vector};

/// Macro to splat a compile-time constant value into all lanes of a generic vector.
///
/// There are five forms of this macro:
/// ```ignore
/// // 1. Generic type parameters with bounds
/// // used when the type depends on generic parameters, and especially `Self`
/// let exp_lsb_mask: Self::Bits = crate::const_splat!(
///     <Self> = <S: FloatVector>
///     <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
/// );
///
/// // 2. Static type, direct value
/// // used when the type is known and the value is a literal or const expression
/// let zero: Vector<u32x4> = crate::const_splat!(u32: 0);
///
/// // 3. Associated const value
/// // used when the value is an associated constant of a type
/// let infinity: Vector<f32x4> = crate::const_splat!(<f32>::INFINITY);
///
/// // 4. Compile-time integer constant cast to a generic float element type E
/// // Produces a const-folded OpConstantComposite in SPIR-V (no runtime OpCompositeConstruct).
/// // E must implement FloatElement.
/// let n_vec: V = crate::const_splat!(int <E>: 7i64);
///
/// // 5. Compile-time rational constant (N/D) cast to a generic float element type E
/// // E must implement FloatElement.
/// let inv3: V = crate::const_splat!(ratio <E>: 1i64, 3i64);
/// ```
#[macro_export]
macro_rules! const_splat {
    (
        <$($real_param:ident),+> = <$($gen_param:ident $(: $bound:path)?),+ $(,)?>
        $ty:ty : $value:expr
    ) => {{
        use core::marker::PhantomData;

        struct __GenericSplatValue<$($gen_param $(: $bound)?),+>(
            PhantomData<($($gen_param),+)>
        );

        impl<$($gen_param $(: $bound)?),+> $crate::vector::SplatConst<$ty>
        for __GenericSplatValue<$($gen_param),+> {
            const VALUE: $ty = const { $value };
        }

        const { $crate::vector::const_splat::<_,
            __GenericSplatValue<$($real_param),+>
        >() }
    }};

    // Compile-time integer constant cast to a generic float element E.
    // N must be a const expression of type i64.
    // Requires E: FloatElement (provides E::ConstInt<N> implementing SplatConst<E>).
    //
    // NOTE: this arm and the `ratio` arm below must come before the generic
    // `($ty:ty: $value:expr)` arm — `int <E>` is syntactically a valid type
    // (`int<E>`), so the `$ty:ty` matcher would otherwise swallow it.
    (int <$E:ty>: $n:expr) => {
        const { $crate::vector::const_splat::<_, <$E as $crate::register::FloatElement>::ConstInt<{$n}>>() }
    };

    // Compile-time rational constant N/D cast to a generic float element E.
    // N and D must be const expressions of type i64.
    // Requires E: FloatElement (provides E::ConstRatio<N, D> implementing SplatConst<E>).
    (ratio <$E:ty>: $n:expr, $d:expr) => {
        const { $crate::vector::const_splat::<_, <$E as $crate::register::FloatElement>::ConstRatio<{$n}, {$d}>>() }
    };

    // Static type, direct value
    ($ty:ty: $value:expr) => {{
        struct __ConstSplatValue;
        impl $crate::vector::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { $value };
        }
        const { $crate::vector::const_splat::<_, __ConstSplatValue>() }
    }};

    // Associated const value
    (<$ty:ty $(as $trait:path)?>::$associated:ident) => {{
        struct __ConstSplatValue;
        impl $crate::vector::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { <$ty $(as $trait)?>::$associated };
        }
        const { $crate::vector::const_splat::<_, __ConstSplatValue>() }
    }};
}

/// Type-level addition of two [`ArrayLength`] typenums.
///
/// Helper trait used by the `const_new!` machinery to compute the lane count of
/// a vector built from a literal array: each element bumps a running length by
/// one via [`Inc`]. Blanket-implemented for any `N1 + N2` whose sum is itself a
/// valid `ArrayLength`.
pub trait AddLength<N: ArrayLength>: ArrayLength {
    /// The summed length, `N1 + N2`.
    type Output: ArrayLength;
}

impl<N1, N2> AddLength<N2> for N1
where
    N1: ArrayLength + Add<N2>,
    N2: ArrayLength,
    <N1 as Add<N2>>::Output: ArrayLength,
{
    type Output = <N1 as Add<N2>>::Output;
}

/// Type-level increment: the [`ArrayLength`] one greater than `U`.
///
/// Shorthand for `AddLength<U1>::Output`, used to count array elements one at a
/// time in [`const_new_impl!`](crate::const_new_impl).
pub type Inc<U> = <U as AddLength<generic_array::typenum::U1>>::Output;

#[doc(hidden)]
#[macro_export]
macro_rules! const_new_impl {
    ($N:ty, [$($x:expr),*], []) => ( $N );

    ($N:ty, [], [$x1:expr]) => (
        $crate::const_new_impl!($crate::vector::splat::Inc<$N>, [$x1], [])
    );
    ($N:ty, [], [$x1:expr, $($x:expr),+]) => (
        $crate::const_new_impl!($crate::vector::splat::Inc<$N>, [$x1], [$($x),+])
    );
    ($N:ty, [$($y:expr),+], [$x1:expr]) => (
        $crate::const_new_impl!($crate::vector::splat::Inc<$N>, [$($y),+, $x1], [])
    );
    ($N:ty, [$($y:expr),+], [$x1:expr, $($x:expr),+]) => (
        $crate::const_new_impl!($crate::vector::splat::Inc<$N>, [$($y),+, $x1], [$($x),+])
    );
}

/// Macro to create a vector from a compile-time constant array of per-lane values.
///
/// ```ignore
/// // 1. Static element type, array literal
/// let v: Vector<f32x4> = crate::const_new!(f32: [1.0, 2.0, 3.0, 4.0]);
///
/// // 2. Generic type parameters with bounds
/// // used when the element type depends on generic parameters
/// let v: V = crate::const_new!(
///     <V> = <V: FloatVector>
///     <V::Element>: [1.0, 2.0, 3.0, 4.0]
/// );
/// ```
#[macro_export]
macro_rules! const_new {
    // Static element type, array literal
    ($ty:ty: [$($value:expr),+ $(,)?]) => {{
        type N = $crate::const_new_impl!($crate::generic_array::typenum::U0, [], [$($value),+]);

        struct __ConstNewValue;
        impl $crate::vector::NewConst<$ty, N> for __ConstNewValue {
            const VALUES: $crate::generic_array::GenericArray<$ty, N> =
                const { $crate::generic_array::GenericArray::from_array([$($value),+]) };
        }
        const { $crate::vector::const_new::<_, N, __ConstNewValue>() }
    }};

    // Generic type parameters with bounds
    (
        <$($real_param:ident),+> = <$($gen_param:ident $(: $bound:path)?),+ $(,)?>
        <$ty:ty>: [$($value:expr),+ $(,)?]
    ) => {{
        use core::marker::PhantomData;

        type N = $crate::const_new_impl!($crate::generic_array::typenum::U0, [], [$($value),+]);

        struct __GenericNewValue<$($gen_param $(: $bound)?),+>(
            PhantomData<($($gen_param),+)>
        );

        impl<$($gen_param $(: $bound)?),+> $crate::vector::NewConst<$ty, N>
        for __GenericNewValue<$($gen_param),+> {
            const VALUES: $crate::generic_array::GenericArray<$ty, N> =
                const { $crate::generic_array::GenericArray::from_array([$($value),+]) };
        }

        const { $crate::vector::const_new::<_, N, __GenericNewValue<$($real_param),+>>() }
    }};
}

/// Build a vector `V` with every lane set to the compile-time constant carried
/// by `E`.
///
/// This is the runtime entry point that the [`const_splat!`](crate::const_splat)
/// macro expands to (inside a `const {}` block). `E` is a zero-sized
/// [`SplatConst`] carrier holding the scalar value; the result is produced
/// entirely at compile time. Prefer the macro over calling this directly.
#[inline(never)]
pub const fn const_splat<V: GenericVector, E: SplatConst<V::Element>>() -> V {
    <<V as SplatVector<V::Element>>::Splat<E> as VectorValue<E, V>>::VALUE
}

/// Build a vector `V` from the compile-time constant per-lane array carried by
/// `C`.
///
/// This is the runtime entry point that the [`const_new!`](crate::const_new)
/// macro expands to (inside a `const {}` block). `C` is a zero-sized
/// [`NewConst`] carrier holding the `N`-element array, where `N` must equal
/// `V`'s lane count. Prefer the macro over calling this directly.
pub const fn const_new<V: GenericVector<Lanes = N>, N: ArrayLength, C: NewConst<V::Element, N>>() -> V {
    <<V as NewVector<V::Element, N>>::New<C> as VectorValue<C, V>>::VALUE
}

/// Associates a constant carrier `C` with the concrete vector constant `V` it
/// produces.
///
/// The final link in the const-construction chain: a [`SplatVector::Splat`] /
/// [`NewVector::New`] type implements this to expose the actual `const VALUE: V`
/// for a given carrier. This indirection is what lets a generic vector type
/// turn a compile-time constant into an instance of itself without const
/// generics over arbitrary types.
pub trait VectorValue<C, V: Sized>: Sized {
    /// The materialized vector constant.
    const VALUE: V;
}

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

/// Carrier for a compile-time constant array of `N` per-lane values of element
/// type `E`.
///
/// The array analogue of [`SplatConst`]: used with
/// [`NewVector`] / [`const_new()`] to construct a vector from a literal array at
/// compile time. Typically implemented by an anonymous zero-sized type emitted
/// by the [`const_new!`](crate::const_new) macro.
pub trait NewConst<E, N: ArrayLength> {
    /// The per-lane constant values.
    const VALUES: GenericArray<E, N>;
}

/// A vector type that can splat a compile-time [`SplatConst`] carrier into a
/// constant of itself.
///
/// Implemented by every [`GenericVector`]. Given a carrier `T: SplatConst<E>`,
/// the [`Splat`](Self::Splat) associated type names a [`VectorValue`] whose
/// `VALUE` is `Self` with all lanes set to `T::VALUE`. Drives
/// [`const_splat()`].
pub trait SplatVector<E>: Sized {
    /// For a given constant carrier `T`, the type exposing the splatted vector
    /// constant via [`VectorValue`].
    type Splat<T: SplatConst<E>>: VectorValue<T, Self>;
}

/// A vector type that can build a constant of itself from a compile-time
/// [`NewConst`] array carrier.
///
/// The array analogue of [`SplatVector`]. Given a carrier `T: NewConst<E, N>`,
/// the [`New`](Self::New) associated type names a [`VectorValue`] whose `VALUE`
/// is `Self` with its lanes set from `T::VALUES`. Drives [`const_new()`].
pub trait NewVector<E, N: ArrayLength>: Sized {
    /// For a given array carrier `T`, the type exposing the constructed vector
    /// constant via [`VectorValue`].
    type New<T: NewConst<E, N>>: VectorValue<T, Self>;
}
