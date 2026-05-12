use generic_array::{GenericArray, typenum};

use crate::{
    isa::InstructionSet,
    register::{BitwiseRegister, CastMaskRegister, CoreRegister, InterleaveRegister, MaskRegister, Storage, ZeroUpper},
};

use super::arch;

macro_rules! decl_MxN {
    ($name:ident x $N:literal { $($f:ident : $idx:literal),* }) => {paste::paste! {
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        #[derive(Debug, Clone, Copy, const_default::ConstDefault, PartialEq)]
        pub struct $name {
            $(pub $f: bool,)*
        }

        #[thermite_macros::inline_always]
        impl CoreRegister for $name {
            type Lanes = typenum::[<U $N>];
            type Storage = Self;
            type Mask = Self;

            const IS_EMULATED: bool = false;
            const ISA: InstructionSet = InstructionSet::SPIRV;
            const EMPTY: Self = <Self as const_default::ConstDefault>::DEFAULT;
            const HAS_EQUAL_SIZE_MASK: bool = true;

            fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Self {
                // OpSelect: component-wise select — on_true where mask is true, on_false otherwise.
                unsafe { arch::op_opselect::<Self, Self>(mask, on_true, on_false) }
            }

            fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicaland::<Self>(mask, value) }
            }

            fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicaland::<Self>(arch::op_oplogicalnot::<Self>(mask), value) }
            }

            fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                Self { $($f: if const { Z::N > $idx } { value.$f } else { false },)* }
            }
        }

        #[thermite_macros::inline_always]
        impl BitwiseRegister for $name {
            fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicalnotequal::<Self>(lhs, rhs) }
            }
            fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicaland::<Self>(lhs, rhs) }
            }
            fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicalor::<Self>(lhs, rhs) }
            }
            fn not(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_oplogicalnot::<Self>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl InterleaveRegister for $name {
            fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_interleave $N>](a, b) }
            }
            fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_deinterleave $N>](a, b) }
            }
        }

        #[thermite_macros::inline_always]
        impl CastMaskRegister<$name> for $name {
            fn mask_from(value: Storage<$name>) -> Storage<Self> { value }
        }

        impl MaskRegister for $name {
            const TRUTHY: Storage<Self> = Self { $($f: true,)* };
            const FALSY:  Storage<Self> = Self { $($f: false,)* };

            fn set(mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
                unsafe { arch::op_opvectorinsertdynamic::<Self, bool, usize>(mask, value, lane) }
            }

            fn test(mask: Storage<Self>, lane: usize) -> bool {
                unsafe { arch::op_opvectorextractdynamic::<bool, Self, usize>(mask, lane) }
            }

            fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
                Self { $($f: value[$idx],)* }
            }

            fn all(value: Storage<Self>) -> bool {
                unsafe { arch::op_opall::<bool, Self>(value) }
            }

            fn any(value: Storage<Self>) -> bool {
                unsafe { arch::op_opany::<bool, Self>(value) }
            }

            fn native_bitmask(value: Storage<Self>) -> Option<u64> {
                Some($(((value.$f as u64) << $idx)|)* 0u64)
            }

            // NOTE: bitvec isn't available on SPIR-V targets
            // #[cfg(feature = "bitvec")]
 //
            // fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
            //     $(view.set($idx, value.$f);)*
            // }
        }
    }};
}

decl_MxN!(Mx2 x 2 { x:0, y:1 });
decl_MxN!(Mx3 x 3 { x:0, y:1, z:2 });
decl_MxN!(Mx4 x 4 { x:0, y:1, z:2, w:3 });
