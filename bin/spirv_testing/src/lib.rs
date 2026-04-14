#![allow(unused, non_camel_case_types)]
#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(asm_experimental_arch)]

use spirv_std_macros::spirv;

use thermite::{
    backend::spirv,
    generic_array,
    math::TranscendentalMath,
    prelude::*,
    register::{Storage, SwizzleIndices},
    vector::{UnsignedIntegerVectorWithRegister, VectorWithRegister},
};

use thermite_special::{RealSpecialMath, SpecialMath, SpecialMathWithPolicy};

use spirv::registers::{F32x2x2, F32x3x3, F32x4x4};

type f32x2 = thermite::Vector<spirv::registers::F32x2>;
type f32x3 = thermite::Vector<spirv::registers::F32x3>;
type f32x4 = thermite::Vector<spirv::registers::F32x4>;
type i32x2 = thermite::Vector<spirv::registers::I32x2>;
type i32x3 = thermite::Vector<spirv::registers::I32x3>;
type i32x4 = thermite::Vector<spirv::registers::I32x4>;
type u32x2 = thermite::Vector<spirv::registers::U32x2>;
type u32x3 = thermite::Vector<spirv::registers::U32x3>;
type u32x4 = thermite::Vector<spirv::registers::U32x4>;

pub type RegisterOf<V> = Storage<<V as thermite::vector::FloatVectorWithRegister>::Register>;

#[cfg(all(not(test), target_arch = "spirv"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[inline(never)]
fn do_thing(x: f32x4) -> f32x4 {
    x.hermite_p::<thermite::math::policy::policies::UltraPerformance, 4>()
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: Storage<<u32x3 as UnsignedIntegerVectorWithRegister>::Register>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32x4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32x4],
) {
    let i = id.x as usize;
    if i < output.len() {
        output[i] = do_thing(input[i]);
    }
}
