var searchIndex = JSON.parse('{\
"thermite":{"doc":"","i":[[3,"Vptr","thermite","",null,null],[3,"Mask","","",null,null],[0,"backends","","",null,null],[0,"avx2","thermite::backends","",null,null],[3,"AVX2","thermite::backends::avx2","",null,null],[3,"i32x8","","",null,null],[3,"u64x8","","",null,null],[3,"f32x8","","",null,null],[11,"splat","thermite","",0,[[]]],[11,"add","","",0,[[]]],[11,"value","","",1,[[]]],[11,"from_value","","",1,[[]]],[11,"truthy","","Mask vector containing all true/non-zero lanes.",1,[[]]],[11,"falsey","","Mask vector containing all zero/false lanes.",1,[[]]],[11,"all","","",1,[[]]],[11,"any","","",1,[[]]],[11,"none","","",1,[[]]],[11,"count","","",1,[[]]],[11,"select","","",1,[[]]],[8,"SimdFloatVectorExt","","",null,null],[11,"approx_eq","","",2,[[],["mask",3]]],[11,"clamp","","",2,[[]]],[11,"saturate","","Clamps self to between 0 and 1",2,[[]]],[11,"scale","","Scales values between `in_min` and `in_max`, to between…",2,[[]]],[11,"lerp","","Linearly interpolates between `a` and `b` using `self`",2,[[]]],[11,"safe_sqrt","","Clamps input to positive numbers before calling `sqrt`",2,[[]]],[8,"SimdCastFrom","","Describes casting from one SIMD vector type to another",null,null],[10,"from_cast","","",3,[[]]],[10,"from_cast_mask","","",3,[[["mask",3]],["mask",3]]],[8,"SimdCastTo","","Describes casting to one SIMD vector type from another",null,null],[10,"cast","","",4,[[]]],[10,"cast_mask","","",4,[[["mask",3]],["mask",3]]],[8,"SimdCasts","","List of valid casts between SIMD types in an instruction set",null,null],[11,"cast_to","","",5,[[],["simdcastfrom",8]]],[8,"SimdVectorBase","","Basic shared vector interface",null,null],[16,"Element","","",6,null],[18,"ELEMENT_SIZE","","Size of element type in bytes",6,null],[18,"NUM_ELEMENTS","","",6,null],[18,"ALIGNMENT","","",6,null],[10,"splat","","",6,[[]]],[11,"splat_any","","",6,[[]]],[11,"load_aligned","","",6,[[]]],[11,"load_unaligned","","",6,[[]]],[11,"store_aligned","","",6,[[]]],[11,"store_unaligned","","",6,[[]]],[10,"load_aligned_unchecked","","",6,[[]]],[10,"store_aligned_unchecked","","",6,[[]]],[11,"load_unaligned_unchecked","","",6,[[]]],[11,"store_unaligned_unchecked","","",6,[[]]],[11,"extract","","",6,[[]]],[11,"replace","","",6,[[]]],[10,"extract_unchecked","","",6,[[]]],[10,"replace_unchecked","","",6,[[]]],[8,"SimdBitwise","","Defines bitwise operations on vectors",null,null],[11,"and_not","","Computes `!self & other`, may be more performant than the…",7,[[]]],[18,"FULL_BITMASK","","",7,null],[10,"bitmask","","",7,[[]]],[8,"SimdVector","","Alias for vector mask type Defines common operations on…",null,null],[10,"zero","","",8,[[]]],[10,"one","","",8,[[]]],[10,"min_value","","Maximum representable valid value",8,[[]]],[10,"max_value","","Minimum representable valid value (may be negative)",8,[[]]],[11,"min","","Per-lane, select the minimum value",8,[[]]],[11,"max","","Per-lane, select the maximum value",8,[[]]],[10,"min_element","","Find the minimum value across all lanes",8,[[]]],[10,"max_element","","Find the maximum value across all lanes",8,[[]]],[10,"eq","","",8,[[],["mask",3]]],[11,"ne","","",8,[[],["mask",3]]],[11,"lt","","",8,[[],["mask",3]]],[11,"le","","",8,[[],["mask",3]]],[11,"gt","","",8,[[],["mask",3]]],[11,"ge","","",8,[[],["mask",3]]],[8,"SimdIntVector","","Integer SIMD vectors",null,null],[11,"saturating_add","","Saturating addition, will not wrap",9,[[]]],[11,"saturating_sub","","Saturating subtraction, will not wrap",9,[[]]],[11,"wrapping_sum","","Sum all lanes together, wrapping the result if it can\'t…",9,[[]]],[11,"wrapping_product","","Multiple all lanes together, wrapping the result if it…",9,[[]]],[8,"SimdSignedVector","","Signed SIMD vector, with negative numbers",null,null],[10,"neg_one","","",10,[[]]],[10,"min_positive","","Minimum positive number",10,[[]]],[11,"abs","","Absolute value",10,[[]]],[11,"copysign","","Copies the sign from `sign` to `self`",10,[[]]],[11,"signum","","Returns `-1` is less than zero, `+1` otherwise.",10,[[]]],[11,"is_positive","","Test if positive, greater or equal to zero",10,[[],["mask",3]]],[11,"is_negative","","Test if negative, less than zero",10,[[],["mask",3]]],[8,"SimdFloatVector","","Floating point SIMD vectors",null,null],[10,"epsilon","","",11,[[]]],[10,"infinity","","",11,[[]]],[10,"neg_infinity","","",11,[[]]],[10,"neg_zero","","",11,[[]]],[10,"nan","","",11,[[]]],[10,"sum","","Compute the horizontal sum of all elements",11,[[]]],[10,"product","","Compute the horizontal product of all elements",11,[[]]],[11,"mul_add","","Fused multiply-add",11,[[]]],[11,"mul_sub","","Fused multiply-subtract",11,[[]]],[11,"neg_mul_add","","Fused negated multiple-add",11,[[]]],[11,"neg_mul_sub","","Fused negated multiple-subtract",11,[[]]],[10,"round","","",11,[[]]],[10,"ceil","","",11,[[]]],[10,"floor","","",11,[[]]],[10,"sqrt","","",11,[[]]],[11,"rsqrt","","Compute the reciprocal of the square root `(1 / sqrt(x))`",11,[[]]],[11,"rsqrt_precise","","A more precise `1 / sqrt(x)` variation, which may use…",11,[[]]],[11,"recepr","","Computes the approximate reciprocal/inverse of each value",11,[[]]],[11,"is_finite","","",11,[[],["mask",3]]],[11,"is_infinite","","",11,[[],["mask",3]]],[11,"is_normal","","",11,[[],["mask",3]]],[11,"is_nan","","",11,[[],["mask",3]]],[8,"Simd","","SIMD Instruction set",null,null],[16,"Vi32","","",12,null],[16,"Vu64","","",12,null],[16,"Vf32","","",12,null],[16,"Vusize","","",12,null],[11,"from_cast","","",0,[[]]],[11,"from_cast_mask","","",0,[[["mask",3]],["mask",3]]],[11,"cast","","",0,[[]]],[11,"cast_mask","","",0,[[["mask",3]],["mask",3]]],[11,"from","","",0,[[]]],[11,"try_from","","",0,[[],["result",4]]],[11,"into","","",0,[[]]],[11,"try_into","","",0,[[],["result",4]]],[11,"borrow","","",0,[[]]],[11,"borrow_mut","","",0,[[]]],[11,"type_id","","",0,[[],["typeid",3]]],[11,"from_cast","","",1,[[]]],[11,"from_cast_mask","","",1,[[["mask",3]],["mask",3]]],[11,"cast","","",1,[[]]],[11,"cast_mask","","",1,[[["mask",3]],["mask",3]]],[11,"from","","",1,[[]]],[11,"try_from","","",1,[[],["result",4]]],[11,"into","","",1,[[]]],[11,"try_into","","",1,[[],["result",4]]],[11,"borrow","","",1,[[]]],[11,"borrow_mut","","",1,[[]]],[11,"type_id","","",1,[[],["typeid",3]]],[11,"from_cast","thermite::backends::avx2","",13,[[]]],[11,"from_cast_mask","","",13,[[["mask",3]],["mask",3]]],[11,"cast","","",13,[[]]],[11,"cast_mask","","",13,[[["mask",3]],["mask",3]]],[11,"from","","",13,[[]]],[11,"try_from","","",13,[[],["result",4]]],[11,"into","","",13,[[]]],[11,"try_into","","",13,[[],["result",4]]],[11,"borrow","","",13,[[]]],[11,"borrow_mut","","",13,[[]]],[11,"type_id","","",13,[[],["typeid",3]]],[11,"from_cast","","",14,[[]]],[11,"from_cast_mask","","",14,[[["mask",3]],["mask",3]]],[11,"cast","","",14,[[]]],[11,"cast_mask","","",14,[[["mask",3]],["mask",3]]],[11,"from","","",14,[[]]],[11,"try_from","","",14,[[],["result",4]]],[11,"into","","",14,[[]]],[11,"try_into","","",14,[[],["result",4]]],[11,"borrow","","",14,[[]]],[11,"borrow_mut","","",14,[[]]],[11,"type_id","","",14,[[],["typeid",3]]],[11,"from_cast","","",15,[[]]],[11,"from_cast_mask","","",15,[[["mask",3]],["mask",3]]],[11,"cast","","",15,[[]]],[11,"cast_mask","","",15,[[["mask",3]],["mask",3]]],[11,"from","","",15,[[]]],[11,"try_from","","",15,[[],["result",4]]],[11,"into","","",15,[[]]],[11,"try_into","","",15,[[],["result",4]]],[11,"borrow","","",15,[[]]],[11,"borrow_mut","","",15,[[]]],[11,"type_id","","",15,[[],["typeid",3]]],[11,"from_cast","","",16,[[]]],[11,"from_cast_mask","","",16,[[["mask",3]],["mask",3]]],[11,"cast","","",16,[[]]],[11,"cast_mask","","",16,[[["mask",3]],["mask",3]]],[11,"from","","",16,[[]]],[11,"try_from","","",16,[[],["result",4]]],[11,"into","","",16,[[]]],[11,"try_into","","",16,[[],["result",4]]],[11,"borrow","","",16,[[]]],[11,"borrow_mut","","",16,[[]]],[11,"type_id","","",16,[[],["typeid",3]]],[11,"from_cast","","",16,[[["i32x8",3],["avx2",3]]]],[11,"from_cast_mask","","",16,[[["mask",3],["i32x8",3],["avx2",3]],[["mask",3],["avx2",3]]]],[11,"from_cast","","",16,[[["avx2",3],["u64x8",3]]]],[11,"from_cast_mask","","",16,[[["avx2",3],["mask",3],["u64x8",3]],[["mask",3],["avx2",3]]]],[11,"from_cast","","",14,[[["f32x8",3],["avx2",3]]]],[11,"from_cast_mask","","",14,[[["mask",3],["f32x8",3],["avx2",3]],[["mask",3],["avx2",3]]]],[11,"from_cast","","",14,[[["avx2",3],["u64x8",3]]]],[11,"from_cast_mask","","",14,[[["avx2",3],["mask",3],["u64x8",3]],[["mask",3],["avx2",3]]]],[11,"splat","","",14,[[]]],[11,"load_aligned_unchecked","","",14,[[]]],[11,"load_unaligned_unchecked","","",14,[[]]],[11,"store_aligned_unchecked","","",14,[[]]],[11,"store_unaligned_unchecked","","",14,[[]]],[11,"extract_unchecked","","",14,[[]]],[11,"replace_unchecked","","",14,[[]]],[11,"splat","","",16,[[]]],[11,"load_aligned_unchecked","","",16,[[]]],[11,"load_unaligned_unchecked","","",16,[[]]],[11,"store_aligned_unchecked","","",16,[[]]],[11,"store_unaligned_unchecked","","",16,[[]]],[11,"extract_unchecked","","",16,[[]]],[11,"replace_unchecked","","",16,[[]]],[11,"splat","","",15,[[]]],[11,"load_aligned_unchecked","","",15,[[]]],[11,"store_aligned_unchecked","","",15,[[]]],[11,"extract_unchecked","","",15,[[]]],[11,"replace_unchecked","","",15,[[]]],[11,"splat","thermite","",1,[[]]],[11,"load_aligned_unchecked","","",1,[[]]],[11,"store_aligned_unchecked","","",1,[[]]],[11,"extract_unchecked","","",1,[[]]],[11,"replace_unchecked","","",1,[[]]],[11,"and_not","thermite::backends::avx2","",14,[[]]],[11,"bitmask","","",14,[[]]],[11,"_mm_not","","",14,[[]]],[11,"_mm_bitand","","",14,[[]]],[11,"_mm_bitor","","",14,[[]]],[11,"_mm_bitxor","","",14,[[]]],[11,"and_not","","",16,[[]]],[11,"bitmask","","",16,[[]]],[11,"_mm_not","","",16,[[]]],[11,"_mm_bitand","","",16,[[]]],[11,"_mm_bitor","","",16,[[]]],[11,"_mm_bitxor","","",16,[[]]],[11,"and_not","","",15,[[]]],[11,"bitmask","","",15,[[]]],[11,"_mm_not","","",15,[[]]],[11,"_mm_bitand","","",15,[[]]],[11,"_mm_bitor","","",15,[[]]],[11,"_mm_bitxor","","",15,[[]]],[11,"and_not","thermite","",1,[[]]],[11,"bitmask","","",1,[[]]],[11,"_mm_not","","",1,[[]]],[11,"_mm_bitand","","",1,[[]]],[11,"_mm_bitor","","",1,[[]]],[11,"_mm_bitxor","","",1,[[]]],[11,"zero","thermite::backends::avx2","",14,[[]]],[11,"one","","",14,[[]]],[11,"min_value","","",14,[[]]],[11,"max_value","","",14,[[]]],[11,"min_element","","",14,[[]]],[11,"max_element","","",14,[[]]],[11,"eq","","",14,[[],[["mask",3],["avx2",3]]]],[11,"gt","","",14,[[],[["mask",3],["avx2",3]]]],[11,"ge","","",14,[[],[["mask",3],["avx2",3]]]],[11,"_mm_add","","",14,[[]]],[11,"_mm_sub","","",14,[[]]],[11,"_mm_mul","","",14,[[]]],[11,"_mm_div","","",14,[[]]],[11,"_mm_rem","","",14,[[]]],[11,"zero","","",16,[[]]],[11,"one","","",16,[[]]],[11,"min_value","","",16,[[]]],[11,"max_value","","",16,[[]]],[11,"min","","",16,[[]]],[11,"max","","",16,[[]]],[11,"min_element","","",16,[[]]],[11,"max_element","","",16,[[]]],[11,"eq","","",16,[[],[["mask",3],["avx2",3]]]],[11,"ne","","",16,[[],[["mask",3],["avx2",3]]]],[11,"lt","","",16,[[],[["mask",3],["avx2",3]]]],[11,"le","","",16,[[],[["mask",3],["avx2",3]]]],[11,"gt","","",16,[[],[["mask",3],["avx2",3]]]],[11,"ge","","",16,[[],[["mask",3],["avx2",3]]]],[11,"_mm_add","","",16,[[]]],[11,"_mm_sub","","",16,[[]]],[11,"_mm_mul","","",16,[[]]],[11,"_mm_div","","",16,[[]]],[11,"_mm_rem","","",16,[[]]],[11,"zero","","",15,[[]]],[11,"one","","",15,[[]]],[11,"min_value","","",15,[[]]],[11,"max_value","","",15,[[]]],[11,"min_element","","",15,[[]]],[11,"max_element","","",15,[[]]],[11,"eq","","",15,[[],[["mask",3],["avx2",3]]]],[11,"gt","","",15,[[],[["mask",3],["avx2",3]]]],[11,"ge","","",15,[[],[["mask",3],["avx2",3]]]],[11,"_mm_add","","",15,[[]]],[11,"_mm_sub","","",15,[[]]],[11,"_mm_mul","","",15,[[]]],[11,"_mm_div","","",15,[[]]],[11,"_mm_rem","","",15,[[]]],[11,"neg_one","","",14,[[]]],[11,"min_positive","","",14,[[]]],[11,"abs","","",14,[[]]],[11,"_mm_neg","","",14,[[]]],[11,"neg_one","","",16,[[]]],[11,"min_positive","","",16,[[]]],[11,"signum","","",16,[[]]],[11,"copysign","","",16,[[]]],[11,"abs","","",16,[[]]],[11,"_mm_neg","","",16,[[]]],[11,"epsilon","","",16,[[]]],[11,"infinity","","",16,[[]]],[11,"neg_infinity","","",16,[[]]],[11,"neg_zero","","",16,[[]]],[11,"nan","","",16,[[]]],[11,"sum","","",16,[[]]],[11,"product","","",16,[[]]],[11,"mul_add","","",16,[[]]],[11,"mul_sub","","",16,[[]]],[11,"neg_mul_add","","",16,[[]]],[11,"neg_mul_sub","","",16,[[]]],[11,"floor","","",16,[[]]],[11,"ceil","","",16,[[]]],[11,"round","","",16,[[]]],[11,"sqrt","","",16,[[]]],[11,"rsqrt","","",16,[[]]],[11,"rsqrt_precise","","",16,[[]]],[11,"recepr","","",16,[[]]],[11,"fmt","","",13,[[["formatter",3]],["result",6]]],[11,"fmt","","",14,[[["formatter",3]],["result",6]]],[11,"fmt","","",15,[[["formatter",3]],["result",6]]],[11,"fmt","","",16,[[["formatter",3]],["result",6]]],[11,"fmt","thermite","",0,[[["formatter",3]],["result",6]]],[11,"fmt","","",1,[[["formatter",3]],["result",6]]],[11,"div","thermite::backends::avx2","",14,[[]]],[11,"div","","",16,[[]]],[11,"div","","",15,[[]]],[11,"rem","","",14,[[]]],[11,"rem","","",16,[[]]],[11,"rem","","",15,[[]]],[11,"sub","","",14,[[]]],[11,"sub","","",16,[[]]],[11,"sub","","",15,[[]]],[11,"eq","","",13,[[["avx2",3]]]],[11,"eq","","",14,[[]]],[11,"ne","","",14,[[]]],[11,"eq","","",16,[[]]],[11,"ne","","",16,[[]]],[11,"eq","","",15,[[]]],[11,"ne","","",15,[[]]],[11,"eq","thermite","",0,[[["vptr",3]]]],[11,"ne","","",0,[[["vptr",3]]]],[11,"add","thermite::backends::avx2","",14,[[]]],[11,"add","","",16,[[]]],[11,"add","","",15,[[]]],[11,"mul","","",14,[[]]],[11,"mul","","",16,[[]]],[11,"mul","","",15,[[]]],[11,"neg","","",14,[[]]],[11,"neg","","",16,[[]]],[11,"add_assign","","",14,[[]]],[11,"add_assign","","",16,[[]]],[11,"add_assign","","",15,[[]]],[11,"sub_assign","","",14,[[]]],[11,"sub_assign","","",16,[[]]],[11,"sub_assign","","",15,[[]]],[11,"mul_assign","","",14,[[]]],[11,"mul_assign","","",16,[[]]],[11,"mul_assign","","",15,[[]]],[11,"div_assign","","",14,[[]]],[11,"div_assign","","",16,[[]]],[11,"div_assign","","",15,[[]]],[11,"rem_assign","","",14,[[]]],[11,"rem_assign","","",16,[[]]],[11,"rem_assign","","",15,[[]]],[11,"not","","",14,[[]]],[11,"not","","",16,[[]]],[11,"not","","",15,[[]]],[11,"not","thermite","",1,[[]]],[11,"bitand","thermite::backends::avx2","",14,[[]]],[11,"bitand","","",16,[[]]],[11,"bitand","","",15,[[]]],[11,"bitand","thermite","",1,[[]]],[11,"bitor","thermite::backends::avx2","",14,[[]]],[11,"bitor","","",16,[[]]],[11,"bitor","","",15,[[]]],[11,"bitor","thermite","",1,[[]]],[11,"bitxor","thermite::backends::avx2","",14,[[]]],[11,"bitxor","","",16,[[]]],[11,"bitxor","","",15,[[]]],[11,"bitxor","thermite","",1,[[]]],[11,"bitand_assign","thermite::backends::avx2","",14,[[]]],[11,"bitand_assign","","",16,[[]]],[11,"bitand_assign","","",15,[[]]],[11,"bitand_assign","thermite","",1,[[]]],[11,"bitor_assign","thermite::backends::avx2","",14,[[]]],[11,"bitor_assign","","",16,[[]]],[11,"bitor_assign","","",15,[[]]],[11,"bitor_assign","thermite","",1,[[]]],[11,"bitxor_assign","thermite::backends::avx2","",14,[[]]],[11,"bitxor_assign","","",16,[[]]],[11,"bitxor_assign","","",15,[[]]],[11,"bitxor_assign","thermite","",1,[[]]],[11,"hash","thermite::backends::avx2","",13,[[]]],[11,"clone","","",13,[[],["avx2",3]]],[11,"clone","","",14,[[],["i32x8",3]]],[11,"clone","","",15,[[],["u64x8",3]]],[11,"clone","","",16,[[],["f32x8",3]]],[11,"clone","thermite","",0,[[],["vptr",3]]],[11,"clone","","",1,[[],["mask",3]]],[11,"default","thermite::backends::avx2","",14,[[]]],[11,"default","","",15,[[]]],[11,"default","","",16,[[]]],[11,"default","thermite","",1,[[]]],[11,"approx_eq","","",2,[[],["mask",3]]],[11,"clamp","","",2,[[]]],[11,"saturate","","Clamps self to between 0 and 1",2,[[]]],[11,"scale","","Scales values between `in_min` and `in_max`, to between…",2,[[]]],[11,"lerp","","Linearly interpolates between `a` and `b` using `self`",2,[[]]],[11,"safe_sqrt","","Clamps input to positive numbers before calling `sqrt`",2,[[]]]],"p":[[3,"Vptr"],[3,"Mask"],[8,"SimdFloatVectorExt"],[8,"SimdCastFrom"],[8,"SimdCastTo"],[8,"SimdCasts"],[8,"SimdVectorBase"],[8,"SimdBitwise"],[8,"SimdVector"],[8,"SimdIntVector"],[8,"SimdSignedVector"],[8,"SimdFloatVector"],[8,"Simd"],[3,"AVX2"],[3,"i32x8"],[3,"u64x8"],[3,"f32x8"]]}\
}');
addSearchOptions(searchIndex);initSearch(searchIndex);