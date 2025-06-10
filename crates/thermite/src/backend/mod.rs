macro_rules! decl_vectors {
    ($($name:ident = $ty:ty),* $(,)?) => {
        $(
            #[allow(private_interfaces, non_camel_case_types)]
            pub type $name = $ty;
        )*
    };
}

pub mod scalar;

pub mod x86;

pub mod x86_v1;
pub mod x86_v2;
pub mod x86_v3;
