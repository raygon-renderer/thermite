#[inline(always)]
pub fn bool_to_u32(value: bool) -> u32 {
    //if value { 0xFFFF_FFFF } else { 0 }
    -(value as i32) as u32
}

#[inline(always)]
pub fn bool_to_u64(value: bool) -> u32 {
    //if value { 0xFFFF_FFFF_FFFF_FFFF } else { 0 }
    -(value as i64) as u64
}
