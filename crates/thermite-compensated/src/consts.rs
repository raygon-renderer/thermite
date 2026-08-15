#![allow(clippy::approx_constant)]

use core::marker::PhantomData;

use thermite::{
    Vector,
    math::FloatConsts,
    vector::{SplatConst, const_splat},
};

use super::{Compensated, ScalarValue};

pub const LOG_TABLE_SIZE: usize = 30;

/// Helper trait to store precomputed compensated logarithm table for small integer bases.
pub trait CompensatedLogTable<T = Self>: FloatConsts {
    /// The log table entries for bases 3..32 as (high, low) pairs.
    const LOG_TABLE: [Compensated<T>; LOG_TABLE_SIZE];

    /// Third part of extended precision ln(2), where the first two parts are provided by `LN_2`.
    const LN_2_EXTENDED: [T; 3];
}

// The tables below reach every lane through the `SplatConst`/`const_splat` carrier
// path rather than the deprecated `Vector::splat_const`.
//
// `const_splat::<V, C>()` is purely type-level: it wants a carrier *type* `C` exposing
// one `const VALUE: V::Element`. A `while`-loop index inside a const initializer cannot
// become a type, so the carriers below take the table index as a `const I: usize`
// parameter and the loop is unrolled into one carrier instantiation per entry. The
// element stays generic (`E = R::Element`), which is what the `const_splat!` macro's
// generic arm cannot express here - its carrier takes a single path bound and these
// need `where E: CompensatedLogTable<E>`.

/// Carrier for the high limb of `LOG_TABLE[I]`.
struct LogTableValue<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for LogTableValue<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LOG_TABLE[I].value;
}

/// Carrier for the low (error) limb of `LOG_TABLE[I]`.
struct LogTableError<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for LogTableError<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LOG_TABLE[I].error;
}

/// Carrier for `LN_2_EXTENDED[I]`.
struct Ln2Extended<E, const I: usize>(PhantomData<E>);

impl<E: CompensatedLogTable<E> + Copy, const I: usize> SplatConst<E> for Ln2Extended<E, I> {
    const VALUE: E = <E as CompensatedLogTable<E>>::LN_2_EXTENDED[I];
}

/// Unrolls the log table: one `Compensated` entry per index literal.
///
/// `Self` and `R` resolve at the expansion site (inside the impl block below).
/// The declared array length `LOG_TABLE_SIZE` is what checks the index list is
/// complete - a missing or extra literal is a compile error, not a silent truncation.
macro_rules! log_table {
    ($($i:literal),* $(,)?) => {
        [$(Compensated {
            value: const_splat::<Self, LogTableValue<R::Element, $i>>(),
            error: const_splat::<Self, LogTableError<R::Element, $i>>(),
        }),*]
    };
}

impl<R: thermite::register::FloatRegister> CompensatedLogTable<Self> for Vector<R>
where
    R::Element: CompensatedLogTable<R::Element>,
{
    #[rustfmt::skip]
    const LOG_TABLE: [Compensated<Self>; LOG_TABLE_SIZE] = log_table!(
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    );

    const LN_2_EXTENDED: [Self; 3] = [
        const_splat::<Self, Ln2Extended<R::Element, 0>>(),
        const_splat::<Self, Ln2Extended<R::Element, 1>>(),
        const_splat::<Self, Ln2Extended<R::Element, 2>>(),
    ];
}

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
        // are indexed by *name* rather than by a const-generic, so unlike the log table
        // there is nothing to unroll - but they still cannot use the `const_splat!` macro,
        // whose generated carrier has no way to carry
        // `where E: SplitFloatConsts<E>`.
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

#[rustfmt::skip]
impl_consts!(
    NEG_ZERO,E,EULER_GAMMA,FRAC_1_PI,FRAC_1_SQRT_2,FRAC_1_SQRT_3,FRAC_2_PI,FRAC_1_SQRT_PI,
    FRAC_2_SQRT_PI,FRAC_SQRT_PI_2,FRAC_1_SQRT_TAU,FRAC_PI_2,FRAC_PI_3,FRAC_PI_4,FRAC_PI_6,FRAC_PI_8,
    FRAC_PI_180,FRAC_180_PI,LN_2,LN_10,LN_PI,FRAC_LN_PI_2,LOG2_10,LOG2_E,LOG10_2,LOG10_E,
    PI,PI_SQUARED,PI_CUBED,PI_FOURTH,SQRT_2,SQRT_3,SQRT_E,TAU,SQRT_FRAC_PI_2,SQRT_TAU,PHI,
    FRAC_1_3,FRAC_2_3,FRAC_1_4,FRAC_1_6,FRAC_NEG_1_E,
    EPSILON,SQRT_EPSILON,FOURTH_ROOT_EPSILON);

impl_consts!(f32 {
    NEG_ZERO = ("-0x0.0p+0", "0x0.0p+0"),
    E = ("0x1.5bf0a80000000p+1", "0x1.628aee0000000p-24"),
    EULER_GAMMA = ("0x1.2788d00000000p-1", "-0x1.c824f40000000p-28"),
    FRAC_1_PI = ("0x1.45f3060000000p-2", "0x1.b939100000000p-27"),
    FRAC_1_SQRT_2 = ("0x1.6a09e60000000p-1", "0x1.9fcef40000000p-27"),
    FRAC_1_SQRT_3 = ("0x1.279a740000000p-1", "0x1.640cc80000000p-27"),
    FRAC_2_PI = ("0x1.45f3060000000p-1", "0x1.b939100000000p-26"),
    FRAC_1_SQRT_PI = ("0x1.20dd760000000p-1", "-0x1.f7ac920000000p-26"),
    FRAC_2_SQRT_PI = ("0x1.20dd760000000p+0", "-0x1.f7ac920000000p-25"),
    FRAC_SQRT_PI_2 = ("0x1.c5bf8a0000000p-1", "-0x1.c962120000000p-26"),
    FRAC_1_SQRT_TAU = ("0x1.9884540000000p-2", "-0x1.8579360000000p-27"),
    FRAC_PI_2 = ("0x1.921fb60000000p+0", "-0x1.777a5c0000000p-25"),
    FRAC_PI_3 = ("0x1.0c15240000000p+0", "-0x1.f4a3260000000p-26"),
    FRAC_PI_4 = ("0x1.921fb60000000p-1", "-0x1.777a5c0000000p-26"),
    FRAC_PI_6 = ("0x1.0c15240000000p-1", "-0x1.f4a3260000000p-27"),
    FRAC_PI_8 = ("0x1.921fb60000000p-2", "-0x1.777a5c0000000p-27"),
    FRAC_PI_180 = ("0x1.1df46a0000000p-6", "0x1.294e9c0000000p-33"),
    FRAC_180_PI = ("0x1.ca5dc20000000p+5", "-0x1.670f820000000p-21"),
    LN_2 = ("0x1.62e4300000000p-1", "-0x1.05c6100000000p-29"),
    LN_10 = ("0x1.26bb1c0000000p+1", "-0x1.12aaba0000000p-25"),
    LN_PI = ("0x1.250d040000000p+0", "0x1.1cf4380000000p-25"),
    FRAC_LN_PI_2 = ("0x1.250d040000000p-1", "0x1.1cf4380000000p-26"),
    LOG2_10 = ("0x1.a934f00000000p+1", "0x1.2f346e0000000p-24"),
    LOG2_E = ("0x1.7154760000000p+0", "0x1.4ae0c00000000p-26"),
    LOG10_2 = ("0x1.3441360000000p-2", "-0x1.ec10c00000000p-27"),
    LOG10_E = ("0x1.bcb7b20000000p-2", "-0x1.5b235e0000000p-27"),
    PI = ("0x1.921fb60000000p+1", "-0x1.777a5c0000000p-24"),
    PI_SQUARED = ("0x1.3bd3cc0000000p+3", "0x1.37c8bc0000000p-22"),
    PI_CUBED = ("0x1.f019b60000000p+4", "-0x1.b1d8a00000000p-22"),
    PI_FOURTH = ("0x1.85a2e80000000p+6", "0x1.8521040000000p-19"),
    SQRT_2 = ("0x1.6a09e60000000p+0", "0x1.9fcef40000000p-26"),
    SQRT_3 = ("0x1.bb67ae0000000p+0", "0x1.0b09960000000p-25"),
    SQRT_E = ("0x1.a612980000000p+0", "0x1.c3c0d40000000p-25"),
    TAU = ("0x1.921fb60000000p+2", "-0x1.777a5c0000000p-23"),
    SQRT_FRAC_PI_2 = ("0x1.40d9320000000p+0", "-0x1.3b1f4e0000000p-33"),
    SQRT_TAU = ("0x1.40d9320000000p+1", "-0x1.3b1f4e0000000p-32"),
    PHI = ("0x1.9e377a0000000p+0", "-0x1.1a02d60000000p-26"),
    FRAC_1_3 = ("0x1.5555560000000p-2", "-0x1.5555560000000p-27"),
    FRAC_2_3 = ("0x1.5555560000000p-1", "-0x1.5555560000000p-26"),
    FRAC_1_4 = ("0x1.0000000000000p-2", "0x0.0p+0"),
    FRAC_1_6 = ("0x1.5555560000000p-3", "-0x1.5555560000000p-28"),
    FRAC_NEG_1_E = ("-0x1.78b5640000000p-2", "0x1.3a621a0000000p-27"),
    EPSILON = ("0x1.0000000000000p-47", "0x1.02f4fe0000000p-74"),
    SQRT_EPSILON = ("0x1.6a09e60000000p-24", "0x1.fb5d100000000p-50"),
    FOURTH_ROOT_EPSILON = ("0x1.306fe00000000p-12", "0x1.5976240000000p-37"),
});

impl_consts!(f64 {
    NEG_ZERO = ("-0x0.0p+0", "0x0.0p+0"),
    E = ("0x1.5bf0a8b145769p+1", "0x1.4d57ee2b1013ap-53"),
    EULER_GAMMA = ("0x1.2788cfc6fb619p-1", "-0x1.6cb90701fbfabp-58"),
    FRAC_1_PI = ("0x1.45f306dc9c883p-2", "-0x1.6b01ec5417056p-56"),
    FRAC_1_SQRT_2 = ("0x1.6a09e667f3bcdp-1", "-0x1.bdd3413b26456p-55"),
    FRAC_1_SQRT_3 = ("0x1.279a74590331cp-1", "0x1.34863e0792bedp-55"),
    FRAC_2_PI = ("0x1.45f306dc9c883p-1", "-0x1.6b01ec5417056p-55"),
    FRAC_1_SQRT_PI = ("0x1.20dd750429b6dp-1", "0x1.1ae3a914fed80p-57"),
    FRAC_2_SQRT_PI = ("0x1.20dd750429b6dp+0", "0x1.1ae3a914fed80p-56"),
    FRAC_SQRT_PI_2 = ("0x1.c5bf891b4ef6bp-1", "-0x1.618f13eb7ca89p-55"),
    FRAC_1_SQRT_TAU = ("0x1.9884533d43651p-2", "-0x1.cbc0d30ebfd15p-56"),
    FRAC_PI_2 = ("0x1.921fb54442d18p+0", "0x1.1a62633145c07p-54"),
    FRAC_PI_3 = ("0x1.0c152382d7366p+0", "-0x1.ee6913347c2a6p-54"),
    FRAC_PI_4 = ("0x1.921fb54442d18p-1", "0x1.1a62633145c07p-55"),
    FRAC_PI_6 = ("0x1.0c152382d7366p-1", "-0x1.ee6913347c2a6p-55"),
    FRAC_PI_8 = ("0x1.921fb54442d18p-2", "0x1.1a62633145c07p-56"),
    FRAC_PI_180 = ("0x1.1df46a2529d39p-6", "0x1.5c1d8becdd291p-62"),
    FRAC_180_PI = ("0x1.ca5dc1a63c1f8p+5", "-0x1.1e7ab456405f9p-49"),
    LN_2 = ("0x1.62e42fefa39efp-1", "0x1.abc9e3b39803fp-56"),
    LN_10 = ("0x1.26bb1bbb55516p+1", "-0x1.f48ad494ea3e9p-53"),
    LN_PI = ("0x1.250d048e7a1bdp+0", "0x1.7abf2ad8d5088p-57"),
    FRAC_LN_PI_2 = ("0x1.250d048e7a1bdp-1", "0x1.7abf2ad8d5088p-58"),
    LOG2_10 = ("0x1.a934f0979a371p+1", "0x1.7f2495fb7fa6dp-53"),
    LOG2_E = ("0x1.71547652b82fep+0", "0x1.777d0ffda0d24p-56"),
    LOG10_2 = ("0x1.34413509f79ffp-2", "-0x1.9dc1da994fd21p-59"),
    LOG10_E = ("0x1.bcb7b1526e50ep-2", "0x1.95355baaafad3p-57"),
    PI = ("0x1.921fb54442d18p+1", "0x1.1a62633145c07p-53"),
    PI_SQUARED = ("0x1.3bd3cc9be45dep+3", "0x1.692b71366cc04p-51"),
    PI_CUBED = ("0x1.f019b59389d7cp+4", "0x1.e019558e5380dp-52"),
    PI_FOURTH = ("0x1.85a2e8c290826p+6", "-0x1.cc0cdf4bfa1e7p-48"),
    SQRT_2 = ("0x1.6a09e667f3bcdp+0", "-0x1.bdd3413b26456p-54"),
    SQRT_3 = ("0x1.bb67ae8584caap+0", "0x1.cec95d0b5c1e3p-54"),
    SQRT_E = ("0x1.a61298e1e069cp+0", "-0x1.b4690082a4906p-55"),
    TAU = ("0x1.921fb54442d18p+2", "0x1.1a62633145c07p-52"),
    SQRT_FRAC_PI_2 = ("0x1.40d931ff62706p+0", "-0x1.a6a0d6f814637p-54"),
    SQRT_TAU = ("0x1.40d931ff62706p+1", "-0x1.a6a0d6f814637p-53"),
    PHI = ("0x1.9e3779b97f4a8p+0", "-0x1.f506319fcfd19p-55"),
    FRAC_1_3 = ("0x1.5555555555555p-2", "0x1.5555555555555p-56"),
    FRAC_2_3 = ("0x1.5555555555555p-1", "0x1.5555555555555p-55"),
    FRAC_1_4 = ("0x1.0000000000000p-2", "0x0.0p+0"),
    FRAC_1_6 = ("0x1.5555555555555p-3", "0x1.5555555555555p-57"),
    FRAC_NEG_1_E = ("-0x1.78b56362cef38p-2", "0x1.ca8a4270fadf5p-57"),
    EPSILON = ("0x1.0000000000000p-105", "0x1.946811deb71cap-160"),
    SQRT_EPSILON = ("0x1.6a09e667f3bccp-53", "0x1.4b76de2ebce1ap-107"),
    FOURTH_ROOT_EPSILON = ("0x1.ae89f995ad3adp-27", "0x1.133c04b31ca20p-83"),
});

impl_consts!(LOG f32 [
    ("0x1.d20ae00000000p-1", "0x1.de60aa0000000p-28"),
    ("0x1.7154760000000p-1", "0x1.4ae0c00000000p-27"),
    ("0x1.3e1f9c0000000p-1", "0x1.9f2eee0000000p-26"),
    ("0x1.1dc0ae0000000p-1", "-0x1.dda6380000000p-26"),
    ("0x1.071dae0000000p-1", "0x1.f7c96a0000000p-26"),
    ("0x1.ec709e0000000p-2", "-0x1.e2fe020000000p-29"),
    ("0x1.d20ae00000000p-2", "0x1.de60aa0000000p-29"),
    ("0x1.bcb7b20000000p-2", "-0x1.5b235e0000000p-27"),
    ("0x1.ab0a8a0000000p-2", "0x1.4518740000000p-31"),
    ("0x1.9c16820000000p-2", "-0x1.a3cddc0000000p-28"),
    ("0x1.8f3a680000000p-2", "0x1.8014a80000000p-28"),
    ("0x1.8404700000000p-2", "0x1.10dfaa0000000p-28"),
    ("0x1.7a21c00000000p-2", "0x1.17d8500000000p-29"),
    ("0x1.7154760000000p-2", "0x1.4ae0c00000000p-28"),
    ("0x1.696d540000000p-2", "0x1.0760680000000p-27"),
    ("0x1.62479a0000000p-2", "-0x1.e2a68a0000000p-28"),
    ("0x1.5bc6340000000p-2", "-0x1.1fffc20000000p-31"),
    ("0x1.55d1d20000000p-2", "-0x1.b703f60000000p-27"),
    ("0x1.50577c0000000p-2", "0x1.a83cf40000000p-27"),
    ("0x1.4b47a20000000p-2", "0x1.1831800000000p-27"),
    ("0x1.4695520000000p-2", "0x1.22d24e0000000p-29"),
    ("0x1.4235b40000000p-2", "-0x1.8912be0000000p-28"),
    ("0x1.3e1f9c0000000p-2", "0x1.9f2eee0000000p-27"),
    ("0x1.3a4b400000000p-2", "-0x1.37ecc00000000p-28"),
    ("0x1.36b1ea0000000p-2", "0x1.a5101c0000000p-27"),
    ("0x1.334dd80000000p-2", "-0x1.40ea420000000p-27"),
    ("0x1.301a020000000p-2", "-0x1.03afa00000000p-27"),
    ("0x1.2d12080000000p-2", "0x1.02e7c00000000p-27"),
    ("0x1.2a32160000000p-2", "-0x1.8a11340000000p-27"),
    ("0x1.2776c60000000p-2", "-0x1.e20c800000000p-27"),
],
// LN_2_EXTENDED: Cody-Waite pieces, NOT simply ln(2) to ever more digits.
//
// `exp_internal` reduces with `r -= k * LN_2_EXTENDED[i]`, where both `k` and the piece
// are plain (uncompensated) vectors - so each product is a single rounded multiply. The
// pieces therefore have to be narrow enough that those products are *exact*, which is
// what buys the reduction its accuracy; the compensated subtraction around them is
// already exact on its own.
//
// f32 carries 24 mantissa bits and `exp` overflows near 88.7, so |k| <= 128 needs 8 of
// them: the leading pieces get 24 - 8 = 16 bits each and the tail takes the remainder.
// Verified exact for |k| <= 160; worst-case reduction error 5.2e-17, against the ~3.6e-15
// that double-single needs.
//
// Widening these to "more accurate" full-precision values silently makes `exp` *worse* -
// it was previously ~21 bits per piece, which is inexact past about k = 16.
["0x1.62e4000000000p-1", "0x1.7f7e000000000p-20", "-0x1.c610ca0000000p-37"]);

impl_consts!(LOG f64 [
    ("0x1.d20ae03bcc153p-1", "-0x1.3a34bf2f1ab83p-55"),
    ("0x1.71547652b82fep-1", "0x1.777d0ffda0d24p-57"),
    ("0x1.3e1f9ccf97777p-1", "-0x1.db618df721f98p-55"),
    ("0x1.1dc0ad112ce3ep-1", "0x1.769f645267d4ap-55"),
    ("0x1.071daefbe4b4ap-1", "-0x1.9ea6c1f1794eep-55"),
    ("0x1.ec709dc3a03fdp-2", "0x1.d27f05548af0cp-56"),
    ("0x1.d20ae03bcc153p-2", "-0x1.3a34bf2f1ab83p-56"),
    ("0x1.bcb7b1526e50ep-2", "0x1.95355baaafad3p-57"),
    ("0x1.ab0a8a0a28c3ap-2", "0x1.b6455c79fed99p-56"),
    ("0x1.9c1681970c88fp-2", "0x1.e3221f6298af5p-59"),
    ("0x1.8f3a6860052a1p-2", "-0x1.afe7dc91a78f0p-56"),
    ("0x1.8404704437eabp-2", "0x1.ac5dd927e6112p-56"),
    ("0x1.7a21c022fb0a1p-2", "0x1.a9f5bd6a60428p-56"),
    ("0x1.71547652b82fep-2", "0x1.777d0ffda0d24p-58"),
    ("0x1.696d5483b0344p-2", "0x1.78ec9931dd399p-58"),
    ("0x1.62479987565dap-2", "0x1.2708bb7e5784cp-56"),
    ("0x1.5bc633f70001fp-2", "-0x1.1e2e054e1a1f4p-57"),
    ("0x1.55d1d1247e049p-2", "-0x1.09a5cb7ff50e7p-56"),
    ("0x1.50577cd41e7a0p-2", "-0x1.795d4f1c45fa7p-56"),
    ("0x1.4b47a28c18bfbp-2", "-0x1.a4ad57d5c1ac2p-58"),
    ("0x1.469552245a49dp-2", "-0x1.5b806338b1165p-59"),
    ("0x1.4235b39dbb506p-2", "0x1.15529d613d714p-56"),
    ("0x1.3e1f9ccf97777p-2", "-0x1.db618df721f98p-56"),
    ("0x1.3a4b3fb204cfep-2", "-0x1.431b7d1b937b5p-60"),
    ("0x1.36b1ead2880e2p-2", "-0x1.a2f0fee978f59p-57"),
    ("0x1.334dd75f8adeep-2", "-0x1.c3950ba6ceb12p-56"),
    ("0x1.301a017e282fcp-2", "0x1.dbdacfa0d9b0fp-57"),
    ("0x1.2d12088173e01p-2", "0x1.bffac2f932436p-57"),
    ("0x1.2a32153af765ep-2", "0x1.c22c8956209efp-56"),
    ("0x1.2776c50ef9bfep-2", "0x1.e4b29ccc535d4p-56"),
],
// LN_2_EXTENDED: Cody-Waite pieces - see the f32 table above for why these are narrow.
//
// f64 carries 53 mantissa bits and `exp` overflows near 709.8, so |k| <= 1024 needs 11
// of them: 53 - 11 = 42 bits per leading piece, tail takes the remainder. Verified exact
// for |k| <= 1100; worst-case reduction error 2.9e-42, against the ~1.2e-32 that
// double-double needs.
//
// These were previously full-precision f64 values, which made every `k * piece` a rounded
// multiply and capped `exp` at about 50 bits - 1.1e-15 relative at x = 29, growing with
// |k|. `ln` inherited that ceiling through its Halley step, and everything built on the
// pair (`powf`, the gamma family) inherited it in turn.
["0x1.62e42fefa3800p-1", "0x1.ef35793c76800p-45", "-0x1.9ff0342542fc3p-90"]);
