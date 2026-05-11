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

    // Compile-time integer constant cast to a generic float element E.
    // N must be a const expression of type i64.
    // Requires E: FloatElement (provides E::IntSplat<N> implementing SplatConst<E>).
    (int <$E:ty>: $n:expr) => {
        const { $crate::vector::const_splat::<_, <$E as $crate::register::FloatElement>::IntSplat<{$n}>>() }
    };

    // Compile-time rational constant N/D cast to a generic float element E.
    // N and D must be const expressions of type i64.
    // Requires E: FloatElement (provides E::RatioSplat<N, D> implementing SplatConst<E>).
    (ratio <$E:ty>: $n:expr, $d:expr) => {
        const { $crate::vector::const_splat::<_, <$E as $crate::register::FloatElement>::RatioSplat<{$n}, {$d}>>() }
    };
}

/// Helper trait for `arr!` macro
pub trait AddLength<N: ArrayLength>: ArrayLength {
    /// Resulting length
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

#[inline(never)]
pub const fn const_splat<V: GenericVector, E: SplatConst<V::Element>>() -> V {
    <<V as SplatVector<V::Element>>::Splat<E> as VectorValue<E, V>>::VALUE
}

pub const fn const_new<V: GenericVector<Lanes = N>, N: ArrayLength, C: NewConst<V::Element, N>>() -> V {
    <<V as NewVector<V::Element, N>>::New<C> as VectorValue<C, V>>::VALUE
}

pub trait VectorValue<C, V: Sized>: Sized {
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

pub trait NewConst<E, N: ArrayLength> {
    const VALUES: GenericArray<E, N>;
}

pub trait SplatVector<E>: Sized {
    type Splat<T: SplatConst<E>>: VectorValue<T, Self>;
}

pub trait NewVector<E, N: ArrayLength>: Sized {
    type New<T: NewConst<E, N>>: VectorValue<T, Self>;
}
