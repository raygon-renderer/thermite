initSidebarItems({"fn":[["_mm_abs_epi16","Computes the absolute value of each of the packed 16-bit signed integers in `a` and return the 16-bit unsigned integer"],["_mm_abs_epi32","Computes the absolute value of each of the packed 32-bit signed integers in `a` and return the 32-bit unsigned integer"],["_mm_abs_epi8","Computes the absolute value of packed 8-bit signed integers in `a` and return the unsigned results."],["_mm_alignr_epi8","Concatenate 16-byte blocks in `a` and `b` into a 32-byte temporary result, shift the result right by `n` bytes, and returns the low 16 bytes."],["_mm_hadd_epi16","Horizontally adds the adjacent pairs of values contained in 2 packed 128-bit vectors of `[8 x i16]`."],["_mm_hadd_epi32","Horizontally adds the adjacent pairs of values contained in 2 packed 128-bit vectors of `[4 x i32]`."],["_mm_hadds_epi16","Horizontally adds the adjacent pairs of values contained in 2 packed 128-bit vectors of `[8 x i16]`. Positive sums greater than 7FFFh are saturated to 7FFFh. Negative sums less than 8000h are saturated to 8000h."],["_mm_hsub_epi16","Horizontally subtract the adjacent pairs of values contained in 2 packed 128-bit vectors of `[8 x i16]`."],["_mm_hsub_epi32","Horizontally subtract the adjacent pairs of values contained in 2 packed 128-bit vectors of `[4 x i32]`."],["_mm_hsubs_epi16","Horizontally subtract the adjacent pairs of values contained in 2 packed 128-bit vectors of `[8 x i16]`. Positive differences greater than 7FFFh are saturated to 7FFFh. Negative differences less than 8000h are saturated to 8000h."],["_mm_maddubs_epi16","Multiplies corresponding pairs of packed 8-bit unsigned integer values contained in the first source operand and packed 8-bit signed integer values contained in the second source operand, add pairs of contiguous products with signed saturation, and writes the 16-bit sums to the corresponding bits in the destination."],["_mm_mulhrs_epi16","Multiplies packed 16-bit signed integer values, truncate the 32-bit product to the 18 most significant bits by right-shifting, round the truncated value by adding 1, and write bits `[16:1]` to the destination."],["_mm_shuffle_epi8","Shuffles bytes from `a` according to the content of `b`."],["_mm_sign_epi16","Negates packed 16-bit integers in `a` when the corresponding signed 16-bit integer in `b` is negative, and returns the results. Elements in result are zeroed out when the corresponding element in `b` is zero."],["_mm_sign_epi32","Negates packed 32-bit integers in `a` when the corresponding signed 32-bit integer in `b` is negative, and returns the results. Element in result are zeroed out when the corresponding element in `b` is zero."],["_mm_sign_epi8","Negates packed 8-bit integers in `a` when the corresponding signed 8-bit integer in `b` is negative, and returns the result. Elements in result are zeroed out when the corresponding element in `b` is zero."]]});