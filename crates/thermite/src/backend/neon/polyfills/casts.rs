use generic_array::{GenericArray, typenum};

use super::*;

// Fast `bool` array -> lane-mask conversions (MaskRegister::new_mask).
// `bool` is guaranteed to be one byte of 0x00/0x01, so the bytes are loaded
// into a vector, widened to the lane width, and compared against zero.

#[inline(always)]
pub fn neon_bools_to_mask_x16(value: GenericArray<bool, typenum::U16>) -> uint8x16_t {
    unsafe {
        let bytes: [u8; 16] = core::mem::transmute(value);
        neon_not_u8(vceqzq_u8(vld1q_u8(bytes.as_ptr())))
    }
}

#[inline(always)]
pub fn neon_bools_to_mask_x8(value: GenericArray<bool, typenum::U8>) -> uint16x8_t {
    unsafe {
        let bytes: [u8; 8] = core::mem::transmute(value);
        let w16 = vmovl_u8(vcreate_u8(u64::from_le_bytes(bytes)));
        neon_not_u16(vceqzq_u16(w16))
    }
}

#[inline(always)]
pub fn neon_bools_to_mask_x4(value: GenericArray<bool, typenum::U4>) -> uint32x4_t {
    unsafe {
        let bytes: [u8; 4] = core::mem::transmute(value);
        let w16 = vmovl_u8(vcreate_u8(u32::from_le_bytes(bytes) as u64));
        let w32 = vmovl_u16(vget_low_u16(w16));
        neon_not_u32(vceqzq_u32(w32))
    }
}

#[inline(always)]
pub fn neon_bools_to_mask_x2(value: GenericArray<bool, typenum::U2>) -> uint64x2_t {
    unsafe {
        let bytes: [u8; 2] = core::mem::transmute(value);
        let w16 = vmovl_u8(vcreate_u8(u16::from_le_bytes(bytes) as u64));
        let w32 = vmovl_u16(vget_low_u16(w16));
        let w64 = vmovl_u32(vget_low_u32(w32));
        neon_not_u64(vceqzq_u64(w64))
    }
}
