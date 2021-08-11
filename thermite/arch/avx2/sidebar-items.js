initSidebarItems({"fn":[["_mm256_abs_epi16","Computes the absolute values of packed 16-bit integers in `a`."],["_mm256_abs_epi32","Computes the absolute values of packed 32-bit integers in `a`."],["_mm256_abs_epi8","Computes the absolute values of packed 8-bit integers in `a`."],["_mm256_add_epi16","Adds packed 16-bit integers in `a` and `b`."],["_mm256_add_epi32","Adds packed 32-bit integers in `a` and `b`."],["_mm256_add_epi64","Adds packed 64-bit integers in `a` and `b`."],["_mm256_add_epi8","Adds packed 8-bit integers in `a` and `b`."],["_mm256_adds_epi16","Adds packed 16-bit integers in `a` and `b` using saturation."],["_mm256_adds_epi8","Adds packed 8-bit integers in `a` and `b` using saturation."],["_mm256_adds_epu16","Adds packed unsigned 16-bit integers in `a` and `b` using saturation."],["_mm256_adds_epu8","Adds packed unsigned 8-bit integers in `a` and `b` using saturation."],["_mm256_alignr_epi8","Concatenates pairs of 16-byte blocks in `a` and `b` into a 32-byte temporary result, shifts the result right by `n` bytes, and returns the low 16 bytes."],["_mm256_and_si256","Computes the bitwise AND of 256 bits (representing integer data) in `a` and `b`."],["_mm256_andnot_si256","Computes the bitwise NOT of 256 bits (representing integer data) in `a` and then AND with `b`."],["_mm256_avg_epu16","Averages packed unsigned 16-bit integers in `a` and `b`."],["_mm256_avg_epu8","Averages packed unsigned 8-bit integers in `a` and `b`."],["_mm256_blend_epi16","Blends packed 16-bit integers from `a` and `b` using control mask `IMM8`."],["_mm256_blend_epi32","Blends packed 32-bit integers from `a` and `b` using control mask `IMM8`."],["_mm256_blendv_epi8","Blends packed 8-bit integers from `a` and `b` using `mask`."],["_mm256_broadcastb_epi8","Broadcasts the low packed 8-bit integer from `a` to all elements of the 256-bit returned value."],["_mm256_broadcastd_epi32","Broadcasts the low packed 32-bit integer from `a` to all elements of the 256-bit returned value."],["_mm256_broadcastq_epi64","Broadcasts the low packed 64-bit integer from `a` to all elements of the 256-bit returned value."],["_mm256_broadcastsd_pd","Broadcasts the low double-precision (64-bit) floating-point element from `a` to all elements of the 256-bit returned value."],["_mm256_broadcastsi128_si256","Broadcasts 128 bits of integer data from a to all 128-bit lanes in the 256-bit returned value."],["_mm256_broadcastss_ps","Broadcasts the low single-precision (32-bit) floating-point element from `a` to all elements of the 256-bit returned value."],["_mm256_broadcastw_epi16","Broadcasts the low packed 16-bit integer from a to all elements of the 256-bit returned value"],["_mm256_bslli_epi128","Shifts 128-bit lanes in `a` left by `imm8` bytes while shifting in zeros."],["_mm256_bsrli_epi128","Shifts 128-bit lanes in `a` right by `imm8` bytes while shifting in zeros."],["_mm256_cmpeq_epi16","Compares packed 16-bit integers in `a` and `b` for equality."],["_mm256_cmpeq_epi32","Compares packed 32-bit integers in `a` and `b` for equality."],["_mm256_cmpeq_epi64","Compares packed 64-bit integers in `a` and `b` for equality."],["_mm256_cmpeq_epi8","Compares packed 8-bit integers in `a` and `b` for equality."],["_mm256_cmpgt_epi16","Compares packed 16-bit integers in `a` and `b` for greater-than."],["_mm256_cmpgt_epi32","Compares packed 32-bit integers in `a` and `b` for greater-than."],["_mm256_cmpgt_epi64","Compares packed 64-bit integers in `a` and `b` for greater-than."],["_mm256_cmpgt_epi8","Compares packed 8-bit integers in `a` and `b` for greater-than."],["_mm256_cvtepi16_epi32","Sign-extend 16-bit integers to 32-bit integers."],["_mm256_cvtepi16_epi64","Sign-extend 16-bit integers to 64-bit integers."],["_mm256_cvtepi32_epi64","Sign-extend 32-bit integers to 64-bit integers."],["_mm256_cvtepi8_epi16","Sign-extend 8-bit integers to 16-bit integers."],["_mm256_cvtepi8_epi32","Sign-extend 8-bit integers to 32-bit integers."],["_mm256_cvtepi8_epi64","Sign-extend 8-bit integers to 64-bit integers."],["_mm256_cvtepu16_epi32","Zeroes extend packed unsigned 16-bit integers in `a` to packed 32-bit integers, and stores the results in `dst`."],["_mm256_cvtepu16_epi64","Zero-extend the lower four unsigned 16-bit integers in `a` to 64-bit integers. The upper four elements of `a` are unused."],["_mm256_cvtepu32_epi64","Zero-extend unsigned 32-bit integers in `a` to 64-bit integers."],["_mm256_cvtepu8_epi16","Zero-extend unsigned 8-bit integers in `a` to 16-bit integers."],["_mm256_cvtepu8_epi32","Zero-extend the lower eight unsigned 8-bit integers in `a` to 32-bit integers. The upper eight elements of `a` are unused."],["_mm256_cvtepu8_epi64","Zero-extend the lower four unsigned 8-bit integers in `a` to 64-bit integers. The upper twelve elements of `a` are unused."],["_mm256_extract_epi16","Extracts a 16-bit integer from `a`, selected with `INDEX`. Returns a 32-bit integer containing the zero-extended integer data."],["_mm256_extract_epi8","Extracts an 8-bit integer from `a`, selected with `INDEX`. Returns a 32-bit integer containing the zero-extended integer data."],["_mm256_extracti128_si256","Extracts 128 bits (of integer data) from `a` selected with `IMM1`."],["_mm256_hadd_epi16","Horizontally adds adjacent pairs of 16-bit integers in `a` and `b`."],["_mm256_hadd_epi32","Horizontally adds adjacent pairs of 32-bit integers in `a` and `b`."],["_mm256_hadds_epi16","Horizontally adds adjacent pairs of 16-bit integers in `a` and `b` using saturation."],["_mm256_hsub_epi16","Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b`."],["_mm256_hsub_epi32","Horizontally subtract adjacent pairs of 32-bit integers in `a` and `b`."],["_mm256_hsubs_epi16","Horizontally subtract adjacent pairs of 16-bit integers in `a` and `b` using saturation."],["_mm256_i32gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i32gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 and 8."],["_mm256_i32gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i32gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i64gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i64gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i64gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_i64gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm256_inserti128_si256","Copies `a` to `dst`, then insert 128 bits (of integer data) from `b` at the location specified by `IMM1`."],["_mm256_madd_epi16","Multiplies packed signed 16-bit integers in `a` and `b`, producing intermediate signed 32-bit integers. Horizontally add adjacent pairs of intermediate 32-bit integers."],["_mm256_maddubs_epi16","Vertically multiplies each unsigned 8-bit integer from `a` with the corresponding signed 8-bit integer from `b`, producing intermediate signed 16-bit integers. Horizontally add adjacent pairs of intermediate signed 16-bit integers"],["_mm256_mask_i32gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i32gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i32gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i32gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i64gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i64gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i64gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_mask_i64gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm256_maskload_epi32","Loads packed 32-bit integers from memory pointed by `mem_addr` using `mask` (elements are zeroed out when the highest bit is not set in the corresponding element)."],["_mm256_maskload_epi64","Loads packed 64-bit integers from memory pointed by `mem_addr` using `mask` (elements are zeroed out when the highest bit is not set in the corresponding element)."],["_mm256_maskstore_epi32","Stores packed 32-bit integers from `a` into memory pointed by `mem_addr` using `mask` (elements are not stored when the highest bit is not set in the corresponding element)."],["_mm256_maskstore_epi64","Stores packed 64-bit integers from `a` into memory pointed by `mem_addr` using `mask` (elements are not stored when the highest bit is not set in the corresponding element)."],["_mm256_max_epi16","Compares packed 16-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_max_epi32","Compares packed 32-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_max_epi8","Compares packed 8-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_max_epu16","Compares packed unsigned 16-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_max_epu32","Compares packed unsigned 32-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_max_epu8","Compares packed unsigned 8-bit integers in `a` and `b`, and returns the packed maximum values."],["_mm256_min_epi16","Compares packed 16-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_min_epi32","Compares packed 32-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_min_epi8","Compares packed 8-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_min_epu16","Compares packed unsigned 16-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_min_epu32","Compares packed unsigned 32-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_min_epu8","Compares packed unsigned 8-bit integers in `a` and `b`, and returns the packed minimum values."],["_mm256_movemask_epi8","Creates mask from the most significant bit of each 8-bit element in `a`, return the result."],["_mm256_mpsadbw_epu8","Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a` compared to those in `b`, and stores the 16-bit results in dst. Eight SADs are performed for each 128-bit lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is selected from `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from sequential 8-bit integers selected from `a` starting at the offset specified in `imm8`."],["_mm256_mul_epi32","Multiplies the low 32-bit integers from each packed 64-bit element in `a` and `b`"],["_mm256_mul_epu32","Multiplies the low unsigned 32-bit integers from each packed 64-bit element in `a` and `b`"],["_mm256_mulhi_epi16","Multiplies the packed 16-bit integers in `a` and `b`, producing intermediate 32-bit integers and returning the high 16 bits of the intermediate integers."],["_mm256_mulhi_epu16","Multiplies the packed unsigned 16-bit integers in `a` and `b`, producing intermediate 32-bit integers and returning the high 16 bits of the intermediate integers."],["_mm256_mulhrs_epi16","Multiplies packed 16-bit integers in `a` and `b`, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and return bits `[16:1]`."],["_mm256_mullo_epi16","Multiplies the packed 16-bit integers in `a` and `b`, producing intermediate 32-bit integers, and returns the low 16 bits of the intermediate integers"],["_mm256_mullo_epi32","Multiplies the packed 32-bit integers in `a` and `b`, producing intermediate 64-bit integers, and returns the low 32 bits of the intermediate integers"],["_mm256_or_si256","Computes the bitwise OR of 256 bits (representing integer data) in `a` and `b`"],["_mm256_packs_epi16","Converts packed 16-bit integers from `a` and `b` to packed 8-bit integers using signed saturation"],["_mm256_packs_epi32","Converts packed 32-bit integers from `a` and `b` to packed 16-bit integers using signed saturation"],["_mm256_packus_epi16","Converts packed 16-bit integers from `a` and `b` to packed 8-bit integers using unsigned saturation"],["_mm256_packus_epi32","Converts packed 32-bit integers from `a` and `b` to packed 16-bit integers using unsigned saturation"],["_mm256_permute2x128_si256","Shuffles 128-bits of integer data selected by `imm8` from `a` and `b`."],["_mm256_permute4x64_epi64","Permutes 64-bit integers from `a` using control mask `imm8`."],["_mm256_permute4x64_pd","Shuffles 64-bit floating-point elements in `a` across lanes using the control in `imm8`."],["_mm256_permutevar8x32_epi32","Permutes packed 32-bit integers from `a` according to the content of `b`."],["_mm256_permutevar8x32_ps","Shuffles eight 32-bit foating-point elements in `a` across lanes using the corresponding 32-bit integer index in `idx`."],["_mm256_sad_epu8","Computes the absolute differences of packed unsigned 8-bit integers in `a` and `b`, then horizontally sum each consecutive 8 differences to produce four unsigned 16-bit integers, and pack these unsigned 16-bit integers in the low 16 bits of the 64-bit return value"],["_mm256_shuffle_epi32","Shuffles 32-bit integers in 128-bit lanes of `a` using the control in `imm8`."],["_mm256_shuffle_epi8","Shuffles bytes from `a` according to the content of `b`."],["_mm256_shufflehi_epi16","Shuffles 16-bit integers in the high 64 bits of 128-bit lanes of `a` using the control in `imm8`. The low 64 bits of 128-bit lanes of `a` are copied to the output."],["_mm256_shufflelo_epi16","Shuffles 16-bit integers in the low 64 bits of 128-bit lanes of `a` using the control in `imm8`. The high 64 bits of 128-bit lanes of `a` are copied to the output."],["_mm256_sign_epi16","Negates packed 16-bit integers in `a` when the corresponding signed 16-bit integer in `b` is negative, and returns the results. Results are zeroed out when the corresponding element in `b` is zero."],["_mm256_sign_epi32","Negates packed 32-bit integers in `a` when the corresponding signed 32-bit integer in `b` is negative, and returns the results. Results are zeroed out when the corresponding element in `b` is zero."],["_mm256_sign_epi8","Negates packed 8-bit integers in `a` when the corresponding signed 8-bit integer in `b` is negative, and returns the results. Results are zeroed out when the corresponding element in `b` is zero."],["_mm256_sll_epi16","Shifts packed 16-bit integers in `a` left by `count` while shifting in zeros, and returns the result"],["_mm256_sll_epi32","Shifts packed 32-bit integers in `a` left by `count` while shifting in zeros, and returns the result"],["_mm256_sll_epi64","Shifts packed 64-bit integers in `a` left by `count` while shifting in zeros, and returns the result"],["_mm256_slli_epi16","Shifts packed 16-bit integers in `a` left by `IMM8` while shifting in zeros, return the results;"],["_mm256_slli_epi32","Shifts packed 32-bit integers in `a` left by `IMM8` while shifting in zeros, return the results;"],["_mm256_slli_epi64","Shifts packed 64-bit integers in `a` left by `IMM8` while shifting in zeros, return the results;"],["_mm256_slli_si256","Shifts 128-bit lanes in `a` left by `imm8` bytes while shifting in zeros."],["_mm256_sllv_epi32","Shifts packed 32-bit integers in `a` left by the amount specified by the corresponding element in `count` while shifting in zeros, and returns the result."],["_mm256_sllv_epi64","Shifts packed 64-bit integers in `a` left by the amount specified by the corresponding element in `count` while shifting in zeros, and returns the result."],["_mm256_sra_epi16","Shifts packed 16-bit integers in `a` right by `count` while shifting in sign bits."],["_mm256_sra_epi32","Shifts packed 32-bit integers in `a` right by `count` while shifting in sign bits."],["_mm256_srai_epi16","Shifts packed 16-bit integers in `a` right by `IMM8` while shifting in sign bits."],["_mm256_srai_epi32","Shifts packed 32-bit integers in `a` right by `IMM8` while shifting in sign bits."],["_mm256_srav_epi32","Shifts packed 32-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in sign bits."],["_mm256_srl_epi16","Shifts packed 16-bit integers in `a` right by `count` while shifting in zeros."],["_mm256_srl_epi32","Shifts packed 32-bit integers in `a` right by `count` while shifting in zeros."],["_mm256_srl_epi64","Shifts packed 64-bit integers in `a` right by `count` while shifting in zeros."],["_mm256_srli_epi16","Shifts packed 16-bit integers in `a` right by `IMM8` while shifting in zeros"],["_mm256_srli_epi32","Shifts packed 32-bit integers in `a` right by `IMM8` while shifting in zeros"],["_mm256_srli_epi64","Shifts packed 64-bit integers in `a` right by `IMM8` while shifting in zeros"],["_mm256_srli_si256","Shifts 128-bit lanes in `a` right by `imm8` bytes while shifting in zeros."],["_mm256_srlv_epi32","Shifts packed 32-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in zeros,"],["_mm256_srlv_epi64","Shifts packed 64-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in zeros,"],["_mm256_sub_epi16","Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`"],["_mm256_sub_epi32","Subtract packed 32-bit integers in `b` from packed 32-bit integers in `a`"],["_mm256_sub_epi64","Subtract packed 64-bit integers in `b` from packed 64-bit integers in `a`"],["_mm256_sub_epi8","Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`"],["_mm256_subs_epi16","Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a` using saturation."],["_mm256_subs_epi8","Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a` using saturation."],["_mm256_subs_epu16","Subtract packed unsigned 16-bit integers in `b` from packed 16-bit integers in `a` using saturation."],["_mm256_subs_epu8","Subtract packed unsigned 8-bit integers in `b` from packed 8-bit integers in `a` using saturation."],["_mm256_unpackhi_epi16","Unpacks and interleave 16-bit integers from the high half of each 128-bit lane of `a` and `b`."],["_mm256_unpackhi_epi32","Unpacks and interleave 32-bit integers from the high half of each 128-bit lane of `a` and `b`."],["_mm256_unpackhi_epi64","Unpacks and interleave 64-bit integers from the high half of each 128-bit lane of `a` and `b`."],["_mm256_unpackhi_epi8","Unpacks and interleave 8-bit integers from the high half of each 128-bit lane in `a` and `b`."],["_mm256_unpacklo_epi16","Unpacks and interleave 16-bit integers from the low half of each 128-bit lane of `a` and `b`."],["_mm256_unpacklo_epi32","Unpacks and interleave 32-bit integers from the low half of each 128-bit lane of `a` and `b`."],["_mm256_unpacklo_epi64","Unpacks and interleave 64-bit integers from the low half of each 128-bit lane of `a` and `b`."],["_mm256_unpacklo_epi8","Unpacks and interleave 8-bit integers from the low half of each 128-bit lane of `a` and `b`."],["_mm256_xor_si256","Computes the bitwise XOR of 256 bits (representing integer data) in `a` and `b`"],["_mm_blend_epi32","Blends packed 32-bit integers from `a` and `b` using control mask `IMM4`."],["_mm_broadcastb_epi8","Broadcasts the low packed 8-bit integer from `a` to all elements of the 128-bit returned value."],["_mm_broadcastd_epi32","Broadcasts the low packed 32-bit integer from `a` to all elements of the 128-bit returned value."],["_mm_broadcastq_epi64","Broadcasts the low packed 64-bit integer from `a` to all elements of the 128-bit returned value."],["_mm_broadcastsd_pd","Broadcasts the low double-precision (64-bit) floating-point element from `a` to all elements of the 128-bit returned value."],["_mm_broadcastss_ps","Broadcasts the low single-precision (32-bit) floating-point element from `a` to all elements of the 128-bit returned value."],["_mm_broadcastw_epi16","Broadcasts the low packed 16-bit integer from a to all elements of the 128-bit returned value"],["_mm_i32gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i32gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i32gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i32gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i64gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i64gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i64gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_i64gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8."],["_mm_mask_i32gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i32gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i32gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i32gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i64gather_epi32","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i64gather_epi64","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i64gather_pd","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_mask_i64gather_ps","Returns values from `slice` at offsets determined by `offsets * scale`, where `scale` should be 1, 2, 4 or 8. If mask is set, load the value from `src` in that position instead."],["_mm_maskload_epi32","Loads packed 32-bit integers from memory pointed by `mem_addr` using `mask` (elements are zeroed out when the highest bit is not set in the corresponding element)."],["_mm_maskload_epi64","Loads packed 64-bit integers from memory pointed by `mem_addr` using `mask` (elements are zeroed out when the highest bit is not set in the corresponding element)."],["_mm_maskstore_epi32","Stores packed 32-bit integers from `a` into memory pointed by `mem_addr` using `mask` (elements are not stored when the highest bit is not set in the corresponding element)."],["_mm_maskstore_epi64","Stores packed 64-bit integers from `a` into memory pointed by `mem_addr` using `mask` (elements are not stored when the highest bit is not set in the corresponding element)."],["_mm_sllv_epi32","Shifts packed 32-bit integers in `a` left by the amount specified by the corresponding element in `count` while shifting in zeros, and returns the result."],["_mm_sllv_epi64","Shifts packed 64-bit integers in `a` left by the amount specified by the corresponding element in `count` while shifting in zeros, and returns the result."],["_mm_srav_epi32","Shifts packed 32-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in sign bits."],["_mm_srlv_epi32","Shifts packed 32-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in zeros,"],["_mm_srlv_epi64","Shifts packed 64-bit integers in `a` right by the amount specified by the corresponding element in `count` while shifting in zeros,"]]});