//! Hand-written machinery behind the generated tables in `mod.rs`.
//!
//! The tables reach every lane through the `SplatConst`/`const_splat` carrier path
//! rather than the deprecated `Vector::splat_const`.
//!
//! `const_splat::<V, C>()` is purely type-level: it wants a carrier _type_ `C` exposing
//! one `const VALUE: V::Element`. A `while`-loop index inside a const initializer cannot
//! become a type, so the carriers below take the table index as a `const I: usize`
//! parameter and the loop is unrolled into one carrier instantiation per entry. The
//! element stays generic (`E = R::Element`), which is what the `const_splat!` macro's
//! generic arm cannot express here. Its carrier takes a single path bound and these
//! need `where E: CompensatedLogTable<E>`.

use core::marker::PhantomData;

use thermite::vector::SplatConst;

use super::CompensatedLogTable;

/// Carrier for the high limb of `LOG_TABLE[I]`.
pub(crate) struct LogTableValue<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for LogTableValue<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LOG_TABLE[I].value;
}

/// Carrier for the low (error) limb of `LOG_TABLE[I]`.
pub(crate) struct LogTableError<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for LogTableError<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LOG_TABLE[I].error;
}

/// Carrier for `LN_2_EXTENDED[I]`.
pub(crate) struct Ln2Extended<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for Ln2Extended<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LN_2_EXTENDED[I];
}

/// Unrolls the log table: one `Compensated` entry per index literal.
///
/// `Self` and `R` resolve at the expansion site (inside the impl block in `mod.rs`).
/// The declared array length `LOG_TABLE_SIZE` is what checks the index list is
/// complete. A missing or extra literal is a compile error, not a silent truncation.
macro_rules! log_table {
    ($($i:literal),* $(,)?) => {
        [$(Compensated {
            value: const_splat::<Self, LogTableValue<R::Element, $i>>(),
            error: const_splat::<Self, LogTableError<R::Element, $i>>(),
        }),*]
    };
}

/// The three shapes the generated tables come in.
///
/// * `impl_consts!(NAME, ...)`: invoked through `thermite::for_each_float_const!`,
///   declares `SplitFloatConsts` and lifts it to `Compensated<V>` and `Vector<R>`.
/// * `impl_consts!(f32 { NAME = (hi, lo), ... })`: the element-level split table.
/// * `impl_consts!(LOG f32 [...], [...])`: the reciprocal-log and Cody-Waite tables.
macro_rules! impl_consts {
    ($($const:ident),*) => {
        pub trait SplitFloatConsts<T = Self>: CompensatedLogTable<T> {
            $(const $const: Compensated<T>; )*
        }

        impl<V: ScalarValue> FloatConsts for Compensated<V> {
            $(const $const: Self = {
                let c = <V as SplitFloatConsts<V>>::$const;
                Compensated { value: c.value, error: c.error }
            };)*
        }

        // One `SplatConst` carrier pair per constant name, so the `Vector` impl below can
        // go through `const_splat` instead of the deprecated `Vector::splat_const`. These
        // are indexed by NAME rather than by a const-generic, so unlike the log table
        // there is nothing to unroll, but they still cannot use the `const_splat!` macro,
        // whose generated carrier has no way to carry `where E: SplitFloatConsts<E>`.
        //
        // The names are SCREAMING_CASE constants, so the generated carrier idents are too.
        paste::paste! {$(
            #[allow(non_camel_case_types)]
            struct [<$const _Value>]<E>(PhantomData<E>);

            impl<E: SplitFloatConsts<E> + Copy> SplatConst<E> for [<$const _Value>]<E> {
                const VALUE: E = <E as SplitFloatConsts<E>>::$const.value;
            }

            #[allow(non_camel_case_types)]
            struct [<$const _Error>]<E>(PhantomData<E>);

            impl<E: SplitFloatConsts<E> + Copy> SplatConst<E> for [<$const _Error>]<E> {
                const VALUE: E = <E as SplitFloatConsts<E>>::$const.error;
            }
        )*}

        impl<R: thermite::register::FloatRegister> SplitFloatConsts<Self> for Vector<R>
            where R::Element: SplitFloatConsts<R::Element>,
        {
            $(const $const: Compensated<Self> = paste::paste! {
                Compensated {
                    value: const_splat::<Self, [<$const _Value>]<R::Element>>(),
                    error: const_splat::<Self, [<$const _Error>]<R::Element>>(),
                }
            };)*
        }
    };

    ($t:ty { $($const:ident = ($high:literal, $low:literal)),* $(,)? }) => {paste::paste! {
        impl SplitFloatConsts<$t> for $t {
            $(const $const: Compensated<$t> = Compensated { value: hexf::[<hex $t>]!($high), error: hexf::[<hex $t>]!($low) };)*
        }
    }};

    (LOG $ty:ty [ $(($hi:literal, $low:literal),)* $(,)? ], [$($ln2_third:literal),*] ) => {paste::paste! {
        impl CompensatedLogTable<$ty> for $ty {
            const LOG_TABLE: [Compensated<$ty>; LOG_TABLE_SIZE] = [
                $( Compensated { value: hexf::[<hex $ty>]!($hi), error: hexf::[<hex $ty>]!($low) }, )*
            ];

            const LN_2_EXTENDED: [$ty; 3] = [ $( hexf::[<hex $ty>]!($ln2_third), )* ];
        }
    }};
}
