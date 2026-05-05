use super::{Element, GenericVector, Vector};

/// Macro to splat a compile-time constant value into all lanes of a generic vector.
///
/// There are five forms of this macro:
/// ```ignore
/// // 1. Generic type parameters with bounds
/// // used when the type depends on generic parameters, and especially `Self`
/// let exp_lsb_mask: Self::Bits = crate::generic_splat!(
///     <Self> = <S: FloatVector>
///     <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
/// );
///
/// // 2. Static type, direct value
/// // used when the type is known and the value is a literal or const expression
/// let zero: Vector<u32x4> = crate::generic_splat!(u32: 0);
///
/// // 3. Associated const value
/// // used when the value is an associated constant of a type
/// let infinity: Vector<f32x4> = crate::generic_splat!(<f32>::INFINITY);
///
/// // 4. Compile-time integer constant cast to a generic float element type E
/// // Produces a const-folded OpConstantComposite in SPIR-V (no runtime OpCompositeConstruct).
/// // E must implement FloatElement.
/// let n_vec: V = crate::generic_splat!(int <E>: 7i64);
///
/// // 5. Compile-time rational constant (N/D) cast to a generic float element type E
/// // E must implement FloatElement.
/// let inv3: V = crate::generic_splat!(ratio <E>: 1i64, 3i64);
/// ```
#[macro_export]
macro_rules! generic_splat {
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

        const { $crate::vector::splat::<_,
            __GenericSplatValue<$($real_param),+>
        >() }
    }};

    // Static type, direct value
    ($ty:ty: $value:expr) => {{
        struct __ConstSplatValue;
        impl $crate::vector::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { $value };
        }
        const { $crate::vector::splat::<_, __ConstSplatValue>() }
    }};

    // Associated const value
    (<$ty:ty $(as $trait:path)?>::$associated:ident) => {{
        struct __ConstSplatValue;
        impl $crate::vector::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { <$ty $(as $trait)?>::$associated };
        }
        const { $crate::vector::splat::<_, __ConstSplatValue>() }
    }};

    // Compile-time integer constant cast to a generic float element E.
    // N must be a const expression of type i64.
    // Requires E: FloatElement (provides E::IntSplat<N> implementing SplatConst<E>).
    (int <$E:ty>: $n:expr) => {
        const { $crate::vector::splat::<_, <$E as $crate::register::FloatElement>::IntSplat<{$n}>>() }
    };

    // Compile-time rational constant N/D cast to a generic float element E.
    // N and D must be const expressions of type i64.
    // Requires E: FloatElement (provides E::RatioSplat<N, D> implementing SplatConst<E>).
    (ratio <$E:ty>: $n:expr, $d:expr) => {
        const { $crate::vector::splat::<_, <$E as $crate::register::FloatElement>::RatioSplat<{$n}, {$d}>>() }
    };
}

#[inline(never)]
pub const fn splat<V: GenericVector, E: SplatConst<V::Element>>() -> V {
    <<V as SplatVector<V::Element>>::Splat<E> as SplatVectorValue<E, V>>::VALUE
}

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

pub trait SplatVector<E>: Sized {
    type Splat<T: SplatConst<E>>: SplatVectorValue<T, Self>;
}

pub trait SplatVectorValue<C, V: Sized>: Sized {
    const VALUE: V;
}
