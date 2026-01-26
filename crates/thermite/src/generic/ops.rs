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
            #[doc = "Masked variants of the [`" $trait_name "`](core::ops::" $trait_name ") trait."]
            pub trait [<$trait_name Masked>]<Mask, Rhs = Self>: core::ops::$trait_name<Rhs> {
                #[doc = "Computes [`" $trait_name "`](core::ops::" $trait_name ") with `rhs` where `mask` is true."]
                fn [<$method_name _c>](self, mask: Mask, rhs: Rhs) -> Self::Output;

                #[doc = "Merges [`" $trait_name "`](core::ops::" $trait_name ") with `src` using `mask`, returning `src` where mask is false."]
                fn [<$method_name _m>](self, src: Self, mask: Mask, rhs: Rhs) -> Self::Output;

                #[doc = "Computes [`" $trait_name "`](core::ops::" $trait_name ") masked (zeroed where mask is false)."]
                fn [<$method_name _z>](self, mask: Mask, rhs: Rhs) -> Self::Output;
            }

            #[doc = "Masked assignment variants of the [`" $trait_name "`](core::ops::" $trait_name "Assign) trait."]
            pub trait [<$trait_name AssignMasked>]<Mask, Rhs = Self>: core::ops::[<$trait_name Assign>]<Rhs> {
                #[doc = "Computes [`" $trait_name "Assign`](core::ops::" $trait_name "Assign) with `rhs` where `mask` is true."]
                fn [<$method_name _assign_c>](&mut self, mask: Mask, rhs: Rhs);

                #[doc = "Merges [`" $trait_name "Assign`](core::ops::" $trait_name "Assign) with `src` using `mask`, assigning `src` where mask is false."]
                fn [<$method_name _assign_m>](&mut self, src: Self, mask: Mask, rhs: Rhs);

                #[doc = "Computes [`" $trait_name "Assign`](core::ops::" $trait_name "Assign) masked (zeroed where mask is false)."]
                fn [<$method_name _assign_z>](&mut self, mask: Mask, rhs: Rhs);
            }
        )*
    }};
}

macro_rules! decl_unary_ops {
    ($($trait_name:ident::$method_name:ident),*) => {paste::paste! {$(
        #[doc = "Masked variants of the [`" $trait_name "`](core::ops::" $trait_name ") trait."]
        pub trait [<$trait_name Masked>]<Mask>: core::ops::$trait_name {
            #[doc = "Computes [`" $trait_name "`](core::ops::" $trait_name ") where `mask` is true, does nothing where false."]
            fn [<$method_name _c>](self, mask: Mask) -> Self::Output;
            #[doc = "Merges [`" $trait_name "`](core::ops::" $trait_name ") with `src` using `mask`, returning `src` where mask is false."]
            fn [<$method_name _m>](self, src: Self, mask: Mask) -> Self::Output;
            #[doc = "Computes [`" $trait_name "`](core::ops::" $trait_name ") masked (zeroed where mask is false)."]
            fn [<$method_name _z>](self, mask: Mask) -> Self::Output;
        }
    )*}};
}

decl_binary_ops!(Num;
    Add::add,
    Sub::sub,
    Mul::mul,
    Div::div,
    Rem::rem
);

decl_binary_ops!(Bitwise: NotMasked;
    BitAnd::bitand,
    BitOr::bitor,
    BitXor::bitxor
);

decl_binary_ops!(Bitshift;
    Shl::shl,
    Shr::shr
);

decl_unary_ops!(Not::not, Neg::neg);
