macro_rules! import_intrinsics {
    ($($name:ident),+) => {
        #[cfg(target_arch = "x86_64")]
        pub use std::arch::x86_64::{$($name),+};

        #[cfg(target_arch = "x86")]
        pub use std::arch::x86::{$($name),+};
    };
}

pub mod sse {
    import_intrinsics! { __m128, __m128d, __m128i }

    import_intrinsics! {
        _CMP_EQ_OQ, _CMP_EQ_OS, _CMP_EQ_UQ, _CMP_EQ_US, _CMP_FALSE_OQ, _CMP_FALSE_OS, _CMP_GE_OQ, _CMP_GE_OS,
        _CMP_GT_OQ, _CMP_GT_OS, _CMP_LE_OQ, _CMP_LE_OS, _CMP_LT_OQ, _CMP_LT_OS, _CMP_NEQ_OQ, _CMP_NEQ_OS,
        _CMP_NEQ_UQ, _CMP_NEQ_US, _CMP_NGE_UQ, _CMP_NGE_US, _CMP_NGT_UQ, _CMP_NGT_US, _CMP_NLE_UQ, _CMP_NLE_US,
        _CMP_NLT_UQ, _CMP_NLT_US, _CMP_ORD_Q, _CMP_ORD_S, _CMP_TRUE_UQ, _CMP_TRUE_US, _CMP_UNORD_Q, _CMP_UNORD_S,

        _MM_FROUND_CEIL, _MM_FROUND_CUR_DIRECTION, _MM_FROUND_FLOOR, _MM_FROUND_NEARBYINT, _MM_FROUND_NINT,
        _MM_FROUND_NO_EXC, _MM_FROUND_RAISE_EXC, _MM_FROUND_RINT, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_TO_NEG_INF,
        _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_ZERO, _MM_FROUND_TRUNC
    }

    import_intrinsics! {
        _mm_add_ps, _mm_add_ss, _mm_and_ps, _mm_andnot_ps, _mm_cmpeq_ps, _mm_cmpeq_ss, _mm_cmpge_ps, _mm_cmpge_ss,
        _mm_cmpgt_ps, _mm_cmpgt_ss, _mm_cmple_ps, _mm_cmple_ss, _mm_cmplt_ps, _mm_cmplt_ss, _mm_cmpneq_ps,
        _mm_cmpneq_ss, _mm_cmpnge_ps, _mm_cmpnge_ss, _mm_cmpngt_ps, _mm_cmpngt_ss, _mm_cmpnle_ps, _mm_cmpnle_ss,
        _mm_cmpnlt_ps, _mm_cmpnlt_ss, _mm_cmpord_ps, _mm_cmpord_ss, _mm_cmpunord_ps, _mm_cmpunord_ss, _mm_comieq_ss,
        _mm_comige_ss, _mm_comigt_ss, _mm_comile_ss, _mm_comilt_ss, _mm_comineq_ss, _mm_cvt_si2ss, _mm_cvt_ss2si,
        _mm_cvtsi32_ss, _mm_cvtss_f32, _mm_cvtss_si32, _mm_cvtt_ss2si, _mm_cvttss_si32, _mm_div_ps, _mm_div_ss,
        _mm_load_ps, _mm_load_ps1, _mm_load_ss, _mm_load1_ps, _mm_loadr_ps, _mm_loadu_ps, _mm_max_ps, _mm_max_ss,
        _mm_min_ps, _mm_min_ss, _mm_move_ss, _mm_movehl_ps, _mm_movelh_ps, _mm_movemask_ps, _mm_mul_ps, _mm_mul_ss,
        _mm_or_ps, _mm_prefetch, _mm_rcp_ps, _mm_rcp_ss, _mm_rsqrt_ps, _mm_rsqrt_ss, _mm_set_ps, _mm_set_ps1,
        _mm_set_ss, _mm_set1_ps, _mm_setcsr, _mm_setr_ps, _mm_setzero_ps, _mm_sfence, _mm_shuffle_ps, _mm_sqrt_ps,
        _mm_sqrt_ss, _mm_store_ps, _mm_store_ps1, _mm_store_ss, _mm_store1_ps, _mm_storer_ps, _mm_storeu_ps,
        _mm_stream_ps, _mm_sub_ps, _mm_sub_ss, _mm_ucomieq_ss, _mm_ucomige_ss, _mm_ucomigt_ss, _mm_ucomile_ss,
        _mm_ucomilt_ss, _mm_ucomineq_ss, _mm_undefined_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm_xor_ps, _mm_cvtsi64_ss,
        _mm_cvtss_si64, _mm_cvttss_si64, _mm_loadu_si64
    }
}

pub mod sse2 {
    #[doc(hidden)]
    pub use super::sse::*;
    import_intrinsics! {
        _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_add_epi8, _mm_add_pd, _mm_add_sd, _mm_adds_epi16, _mm_adds_epi8,
        _mm_adds_epu16, _mm_adds_epu8, _mm_and_pd, _mm_and_si128, _mm_andnot_pd, _mm_andnot_si128, _mm_avg_epu16,
        _mm_avg_epu8, _mm_bslli_si128, _mm_bsrli_si128, _mm_castpd_ps, _mm_castpd_si128, _mm_castps_pd, _mm_castps_si128,
        _mm_castsi128_pd, _mm_castsi128_ps, _mm_clflush, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpeq_epi8, _mm_cmpeq_pd,
        _mm_cmpeq_sd, _mm_cmpge_pd, _mm_cmpge_sd, _mm_cmpgt_epi16, _mm_cmpgt_epi32, _mm_cmpgt_epi8, _mm_cmpgt_pd,
        _mm_cmpgt_sd, _mm_cmple_pd, _mm_cmple_sd, _mm_cmplt_epi16, _mm_cmplt_epi32, _mm_cmplt_epi8, _mm_cmplt_pd,
        _mm_cmplt_sd, _mm_cmpneq_pd, _mm_cmpneq_sd, _mm_cmpnge_pd, _mm_cmpnge_sd, _mm_cmpngt_pd, _mm_cmpngt_sd,
        _mm_cmpnle_pd, _mm_cmpnle_sd, _mm_cmpnlt_pd, _mm_cmpnlt_sd, _mm_cmpord_pd, _mm_cmpord_sd, _mm_cmpunord_pd,
        _mm_cmpunord_sd, _mm_comieq_sd, _mm_comige_sd, _mm_comigt_sd, _mm_comile_sd, _mm_comilt_sd, _mm_comineq_sd,
        _mm_cvtepi32_pd, _mm_cvtepi32_ps, _mm_cvtpd_epi32, _mm_cvtpd_ps, _mm_cvtps_epi32, _mm_cvtps_pd, _mm_cvtsd_f64,
        _mm_cvtsd_si32, _mm_cvtsd_ss, _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_cvtsi128_si64x, _mm_cvtsi32_sd,
        _mm_cvtsi32_si128, _mm_cvtsi64x_si128, _mm_cvtss_sd, _mm_cvttpd_epi32, _mm_cvttps_epi32, _mm_cvttsd_si32,
        _mm_cvttsd_si64, _mm_cvttsd_si64x, _mm_div_pd, _mm_div_sd, _mm_extract_epi16, _mm_insert_epi16, _mm_lfence,
        _mm_load_pd, _mm_load_pd1, _mm_load_sd, _mm_load_si128, _mm_load1_pd, _mm_loadh_pd, _mm_loadl_epi64, _mm_loadl_pd,
        _mm_loadr_pd, _mm_loadu_pd, _mm_loadu_si128, _mm_madd_epi16, _mm_maskmoveu_si128, _mm_max_epi16, _mm_max_epu8,
        _mm_max_pd, _mm_max_sd, _mm_mfence, _mm_min_epi16, _mm_min_epu8, _mm_min_pd, _mm_min_sd, _mm_move_epi64,
        _mm_move_sd, _mm_movemask_epi8, _mm_movemask_pd, _mm_mul_epu32, _mm_mul_pd, _mm_mul_sd, _mm_mulhi_epi16,
        _mm_mulhi_epu16, _mm_mullo_epi16, _mm_or_pd, _mm_or_si128, _mm_packs_epi16, _mm_packs_epi32, _mm_packus_epi16,
        _mm_pause, _mm_sad_epu8, _mm_set_epi16, _mm_set_epi32, _mm_set_epi64x, _mm_set_epi8, _mm_set_pd, _mm_set_pd1,
        _mm_set_sd, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_set1_epi8, _mm_set1_pd, _mm_setr_epi16,
        _mm_setr_epi32, _mm_setr_epi8, _mm_setr_pd, _mm_setzero_pd, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shuffle_pd,
        _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_sll_epi16, _mm_sll_epi32, _mm_sll_epi64, _mm_slli_epi16,
        _mm_slli_epi32, _mm_slli_epi64, _mm_slli_si128, _mm_sqrt_pd, _mm_sqrt_sd, _mm_sra_epi16, _mm_sra_epi32,
        _mm_srai_epi16, _mm_srai_epi32, _mm_srl_epi16, _mm_srl_epi32, _mm_srl_epi64, _mm_srli_epi16, _mm_srli_epi32,
        _mm_srli_epi64, _mm_srli_si128, _mm_store_pd, _mm_store_pd1, _mm_store_sd, _mm_store_si128, _mm_store1_pd,
        _mm_storeh_pd, _mm_storel_epi64, _mm_storel_pd, _mm_storer_pd, _mm_storeu_pd, _mm_storeu_si128, _mm_stream_pd,
        _mm_stream_si128, _mm_stream_si32, _mm_stream_si64, _mm_sub_epi16, _mm_sub_epi32, _mm_sub_epi64, _mm_sub_epi8,
        _mm_sub_pd, _mm_sub_sd, _mm_subs_epi16, _mm_subs_epi8, _mm_subs_epu16, _mm_subs_epu8, _mm_ucomieq_sd,
        _mm_ucomige_sd, _mm_ucomigt_sd, _mm_ucomile_sd, _mm_ucomilt_sd, _mm_ucomineq_sd, _mm_undefined_pd,
        _mm_undefined_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpackhi_epi8,
        _mm_unpackhi_pd, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8, _mm_unpacklo_pd,
        _mm_xor_pd, _mm_xor_si128
    }
}

pub mod sse3 {
    #[doc(hidden)]
    pub use super::sse2::*;
    import_intrinsics! {
        _mm_addsub_pd, _mm_addsub_ps, _mm_hadd_pd, _mm_hadd_ps, _mm_hsub_pd, _mm_hsub_ps, _mm_lddqu_si128,
        _mm_loaddup_pd, _mm_movedup_pd, _mm_movehdup_ps, _mm_moveldup_ps
    }
}

pub mod ssse3 {
    #[doc(hidden)]
    pub use super::sse3::*;
    import_intrinsics! {
        _mm_abs_epi16, _mm_abs_epi32, _mm_abs_epi8, _mm_alignr_epi8, _mm_hadd_epi16, _mm_hadd_epi32,
        _mm_hadds_epi16, _mm_hsub_epi16, _mm_hsub_epi32, _mm_hsubs_epi16, _mm_maddubs_epi16, _mm_mulhrs_epi16,
        _mm_shuffle_epi8, _mm_sign_epi16, _mm_sign_epi32, _mm_sign_epi8
    }
}

pub mod sse41 {
    #[doc(hidden)]
    pub use super::ssse3::*;
    import_intrinsics! {
        _mm_blend_epi16, _mm_blend_pd, _mm_blend_ps, _mm_blendv_epi8, _mm_blendv_pd, _mm_blendv_ps, _mm_ceil_pd,
        _mm_ceil_ps, _mm_ceil_sd, _mm_ceil_ss, _mm_cmpeq_epi64, _mm_cvtepi16_epi32, _mm_cvtepi16_epi64,
        _mm_cvtepi32_epi64, _mm_cvtepi8_epi16, _mm_cvtepi8_epi32, _mm_cvtepi8_epi64, _mm_cvtepu16_epi32,
        _mm_cvtepu16_epi64, _mm_cvtepu32_epi64, _mm_cvtepu8_epi16, _mm_cvtepu8_epi32, _mm_cvtepu8_epi64, _mm_dp_pd,
        _mm_dp_ps, _mm_extract_epi32, _mm_extract_epi64, _mm_extract_epi8, _mm_extract_ps, _mm_floor_pd, _mm_floor_ps,
        _mm_floor_sd, _mm_floor_ss, _mm_insert_epi32, _mm_insert_epi64, _mm_insert_epi8, _mm_insert_ps, _mm_max_epi32,
        _mm_max_epi8, _mm_max_epu16, _mm_max_epu32, _mm_min_epi32, _mm_min_epi8, _mm_min_epu16, _mm_min_epu32,
        _mm_minpos_epu16, _mm_mpsadbw_epu8, _mm_mul_epi32, _mm_mullo_epi32, _mm_packus_epi32, _mm_round_pd,
        _mm_round_ps, _mm_round_sd, _mm_round_ss, _mm_test_all_ones, _mm_test_all_zeros, _mm_test_mix_ones_zeros,
        _mm_testc_si128, _mm_testnzc_si128, _mm_testz_si128
    }
}

pub mod sse42 {
    #[doc(hidden)]
    pub use super::sse41::*;

    import_intrinsics! {
        _mm_cmpestra, _mm_cmpestrc, _mm_cmpestri, _mm_cmpestrm, _mm_cmpestro, _mm_cmpestrs, _mm_cmpestrz,
        _mm_cmpgt_epi64, _mm_cmpistra, _mm_cmpistrc, _mm_cmpistri, _mm_cmpistrm, _mm_cmpistro, _mm_cmpistrs,
        _mm_cmpistrz, _mm_crc32_u16, _mm_crc32_u32, _mm_crc32_u64, _mm_crc32_u8
    }
}

pub mod avx {
    #[doc(hidden)]
    pub use super::sse42::*;

    import_intrinsics! { __m256, __m256d, __m256i }

    import_intrinsics! {
        _mm256_add_pd, _mm256_add_ps, _mm256_addsub_pd, _mm256_addsub_ps, _mm256_and_pd, _mm256_and_ps,
        _mm256_andnot_pd, _mm256_andnot_ps, _mm256_blend_pd, _mm256_blend_ps, _mm256_blendv_pd, _mm256_blendv_ps,
        _mm256_broadcast_pd, _mm256_broadcast_ps, _mm256_broadcast_sd, _mm_broadcast_ss, _mm256_broadcast_ss,
        _mm256_castpd_ps, _mm256_castpd_si256, _mm256_castpd128_pd256, _mm256_castpd256_pd128, _mm256_castps_pd,
        _mm256_castps_si256, _mm256_castps128_ps256, _mm256_castps256_ps128, _mm256_castsi128_si256,
        _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_ceil_pd, _mm256_ceil_ps,
        _mm_cmp_pd, _mm256_cmp_pd, _mm_cmp_ps, _mm256_cmp_ps, _mm_cmp_sd, _mm_cmp_ss, _mm256_cvtepi32_pd,
        _mm256_cvtepi32_ps, _mm256_cvtpd_epi32, _mm256_cvtpd_ps, _mm256_cvtps_epi32, _mm256_cvtps_pd,
        _mm256_cvtsd_f64, _mm256_cvtsi256_si32, _mm256_cvtss_f32, _mm256_cvttpd_epi32, _mm256_cvttps_epi32,
        _mm256_div_pd, _mm256_div_ps, _mm256_dp_ps, _mm256_extract_epi32, _mm256_extract_epi64,
        _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_extractf128_si256, _mm256_floor_pd, _mm256_floor_ps,
        _mm256_hadd_pd, _mm256_hadd_ps, _mm256_hsub_pd, _mm256_hsub_ps, _mm256_insert_epi16, _mm256_insert_epi32,
        _mm256_insert_epi64, _mm256_insert_epi8, _mm256_insertf128_pd, _mm256_insertf128_ps,
        _mm256_insertf128_si256, _mm256_lddqu_si256, _mm256_load_pd, _mm256_load_ps, _mm256_load_si256,
        _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_loadu2_m128, _mm256_loadu2_m128d,
        _mm256_loadu2_m128i, _mm_maskload_pd, _mm256_maskload_pd, _mm_maskload_ps, _mm256_maskload_ps,
        _mm_maskstore_pd, _mm256_maskstore_pd, _mm_maskstore_ps, _mm256_maskstore_ps, _mm256_max_pd, _mm256_max_ps,
        _mm256_min_pd, _mm256_min_ps, _mm256_movedup_pd, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_movemask_pd,
        _mm256_movemask_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_or_pd, _mm256_or_ps, _mm_permute_pd,
        _mm256_permute_pd, _mm_permute_ps, _mm256_permute_ps, _mm256_permute2f128_pd, _mm256_permute2f128_ps,
        _mm256_permute2f128_si256, _mm_permutevar_pd, _mm256_permutevar_pd, _mm_permutevar_ps, _mm256_permutevar_ps,
        _mm256_rcp_ps, _mm256_round_pd, _mm256_round_ps, _mm256_rsqrt_ps, _mm256_set_epi16, _mm256_set_epi32,
        _mm256_set_epi64x, _mm256_set_epi8, _mm256_set_m128, _mm256_set_m128d, _mm256_set_m128i, _mm256_set_pd,
        _mm256_set_ps, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_set1_pd,
        _mm256_set1_ps, _mm256_setr_epi16, _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_setr_epi8, _mm256_setr_m128,
        _mm256_setr_m128d, _mm256_setr_m128i, _mm256_setr_pd, _mm256_setr_ps, _mm256_setzero_pd, _mm256_setzero_ps,
        _mm256_setzero_si256, _mm256_shuffle_pd, _mm256_shuffle_ps, _mm256_sqrt_pd, _mm256_sqrt_ps, _mm256_store_pd,
        _mm256_store_ps, _mm256_store_si256, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256,
        _mm256_storeu2_m128, _mm256_storeu2_m128d, _mm256_storeu2_m128i, _mm256_stream_pd, _mm256_stream_ps,
        _mm256_stream_si256, _mm256_sub_pd, _mm256_sub_ps, _mm_testc_pd, _mm256_testc_pd, _mm_testc_ps,
        _mm256_testc_ps, _mm256_testc_si256, _mm_testnzc_pd, _mm256_testnzc_pd, _mm_testnzc_ps, _mm256_testnzc_ps,
        _mm256_testnzc_si256, _mm_testz_pd, _mm256_testz_pd, _mm_testz_ps, _mm256_testz_ps, _mm256_testz_si256,
        _mm256_undefined_pd, _mm256_undefined_ps, _mm256_undefined_si256, _mm256_unpackhi_pd, _mm256_unpackhi_ps,
        _mm256_unpacklo_pd, _mm256_unpacklo_ps, _mm256_xor_pd, _mm256_xor_ps, _mm256_zeroall, _mm256_zeroupper,
        _mm256_zextpd128_pd256, _mm256_zextps128_ps256, _mm256_zextsi128_si256
    }
}

pub mod fma {
    import_intrinsics! {
        _mm_fmadd_pd, _mm256_fmadd_pd, _mm_fmadd_ps, _mm256_fmadd_ps, _mm_fmadd_sd, _mm_fmadd_ss, _mm_fmaddsub_pd,
        _mm256_fmaddsub_pd, _mm_fmaddsub_ps, _mm256_fmaddsub_ps, _mm_fmsub_pd, _mm256_fmsub_pd, _mm_fmsub_ps,
        _mm256_fmsub_ps, _mm_fmsub_sd, _mm_fmsub_ss, _mm_fmsubadd_pd, _mm256_fmsubadd_pd, _mm_fmsubadd_ps,
        _mm256_fmsubadd_ps, _mm_fnmadd_pd, _mm256_fnmadd_pd, _mm_fnmadd_ps, _mm256_fnmadd_ps, _mm_fnmadd_sd,
        _mm_fnmadd_ss, _mm_fnmsub_pd, _mm256_fnmsub_pd, _mm_fnmsub_ps, _mm256_fnmsub_ps, _mm_fnmsub_sd, _mm_fnmsub_ss
    }
}

pub mod avx2 {
    #[doc(hidden)]
    pub use super::avx::*;
    #[doc(hidden)]
    pub use super::fma::*;

    import_intrinsics! {
        _mm256_abs_epi16, _mm256_abs_epi32, _mm256_abs_epi8, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64,
        _mm256_add_epi8, _mm256_adds_epi16, _mm256_adds_epi8, _mm256_adds_epu16, _mm256_adds_epu8, _mm256_alignr_epi8,
        _mm256_and_si256, _mm256_andnot_si256, _mm256_avg_epu16, _mm256_avg_epu8, _mm256_blend_epi16, _mm_blend_epi32,
        _mm256_blend_epi32, _mm256_blendv_epi8, _mm_broadcastb_epi8, _mm256_broadcastb_epi8, _mm_broadcastd_epi32,
        _mm256_broadcastd_epi32, _mm_broadcastq_epi64, _mm256_broadcastq_epi64, _mm_broadcastsd_pd, _mm256_broadcastsd_pd,
        _mm256_broadcastsi128_si256, _mm_broadcastss_ps, _mm256_broadcastss_ps, _mm_broadcastw_epi16,
        _mm256_broadcastw_epi16, _mm256_bslli_epi128, _mm256_bsrli_epi128, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32,
        _mm256_cmpeq_epi64, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi64,
        _mm256_cmpgt_epi8, _mm256_cvtepi16_epi32, _mm256_cvtepi16_epi64, _mm256_cvtepi32_epi64, _mm256_cvtepi8_epi16,
        _mm256_cvtepi8_epi32, _mm256_cvtepi8_epi64, _mm256_cvtepu16_epi32, _mm256_cvtepu16_epi64, _mm256_cvtepu32_epi64,
        _mm256_cvtepu8_epi16, _mm256_cvtepu8_epi32, _mm256_cvtepu8_epi64, _mm256_extract_epi16, _mm256_extract_epi8,
        _mm256_extracti128_si256, _mm256_hadd_epi16, _mm256_hadd_epi32, _mm256_hadds_epi16, _mm256_hsub_epi16,
        _mm256_hsub_epi32, _mm256_hsubs_epi16, _mm_i32gather_epi32, _mm_mask_i32gather_epi32, _mm256_i32gather_epi32,
        _mm256_mask_i32gather_epi32, _mm_i32gather_epi64, _mm_mask_i32gather_epi64, _mm256_i32gather_epi64,
        _mm256_mask_i32gather_epi64, _mm_i32gather_pd, _mm_mask_i32gather_pd, _mm256_i32gather_pd,
        _mm256_mask_i32gather_pd, _mm_i32gather_ps, _mm_mask_i32gather_ps, _mm256_i32gather_ps, _mm256_mask_i32gather_ps,
        _mm_i64gather_epi32, _mm_mask_i64gather_epi32, _mm256_i64gather_epi32, _mm256_mask_i64gather_epi32,
        _mm_i64gather_epi64, _mm_mask_i64gather_epi64, _mm256_i64gather_epi64, _mm256_mask_i64gather_epi64,
        _mm_i64gather_pd, _mm_mask_i64gather_pd, _mm256_i64gather_pd, _mm256_mask_i64gather_pd, _mm_i64gather_ps,
        _mm_mask_i64gather_ps, _mm256_i64gather_ps, _mm256_mask_i64gather_ps, _mm256_inserti128_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm_maskload_epi32, _mm256_maskload_epi32, _mm_maskload_epi64, _mm256_maskload_epi64,
        _mm_maskstore_epi32, _mm256_maskstore_epi32, _mm_maskstore_epi64, _mm256_maskstore_epi64, _mm256_max_epi16,
        _mm256_max_epi32, _mm256_max_epi8, _mm256_max_epu16, _mm256_max_epu32, _mm256_max_epu8, _mm256_min_epi16,
        _mm256_min_epi32, _mm256_min_epi8, _mm256_min_epu16, _mm256_min_epu32, _mm256_min_epu8, _mm256_movemask_epi8,
        _mm256_mpsadbw_epu8, _mm256_mul_epi32, _mm256_mul_epu32, _mm256_mulhi_epi16, _mm256_mulhi_epu16, _mm256_mulhrs_epi16,
        _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256, _mm256_packs_epi16, _mm256_packs_epi32, _mm256_packus_epi16,
        _mm256_packus_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_permute4x64_pd,
        _mm256_permutevar8x32_epi32, _mm256_permutevar8x32_ps, _mm256_sad_epu8, _mm256_shuffle_epi32, _mm256_shuffle_epi8,
        _mm256_shufflehi_epi16, _mm256_shufflelo_epi16, _mm256_sign_epi16, _mm256_sign_epi32, _mm256_sign_epi8,
        _mm256_sll_epi16, _mm256_sll_epi32, _mm256_sll_epi64, _mm256_slli_epi16, _mm256_slli_epi32, _mm256_slli_epi64,
        _mm256_slli_si256, _mm_sllv_epi32, _mm256_sllv_epi32, _mm_sllv_epi64, _mm256_sllv_epi64, _mm256_sra_epi16,
        _mm256_sra_epi32, _mm256_srai_epi16, _mm256_srai_epi32, _mm_srav_epi32, _mm256_srav_epi32, _mm256_srl_epi16,
        _mm256_srl_epi32, _mm256_srl_epi64, _mm256_srli_epi16, _mm256_srli_epi32, _mm256_srli_epi64, _mm256_srli_si256,
        _mm_srlv_epi32, _mm256_srlv_epi32, _mm_srlv_epi64, _mm256_srlv_epi64, _mm256_sub_epi16, _mm256_sub_epi32,
        _mm256_sub_epi64, _mm256_sub_epi8, _mm256_subs_epi16, _mm256_subs_epi8, _mm256_subs_epu16, _mm256_subs_epu8,
        _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpackhi_epi8, _mm256_unpacklo_epi16,
        _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_unpacklo_epi8, _mm256_xor_si256
    }
}
