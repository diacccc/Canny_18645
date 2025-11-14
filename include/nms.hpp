#ifndef NMS_HPP
#define NMS_HPP

#include <cstdint>
#include <immintrin.h>

// void non_max_suppression(const int16_t* gx, const int16_t* gy, 
//                         const int16_t* mag, int16_t *res, int M, int N);
#define NMS_KERNEL_REF(gx, gy, mag, res, M, N) \
do { \
    const int TG22 = 13573; /* tg22 scaled constant */ \
    for(int ii = 0; ii < 16; ++ii) \
    { \
        int16_t gx_val = *(gx + ii); \
        int16_t gy_val = *(gy + ii); \
        int16_t mag_val = *(mag + ii); \
        *(res + ii) = 0; \
        int x = (int)std::abs(gx_val); \
        int y = (int)std::abs(gy_val) << 15; \
        int tg22x = x * TG22; \
        if (y < tg22x) \
        { \
            if (mag_val > *(mag + ii - 1) && mag_val >= *(mag + ii + 1)) \
            { \
                *(res + ii) = mag_val; \
            } \
        } else \
        { \
            int tg67x = tg22x + (x << 16); \
            if (y > tg67x) \
            { \
                if (mag_val > *(mag + ii - N) && mag_val >= *(mag + ii + N)) \
                { \
                    *(res + ii) = mag_val; \
                } \
            } else \
            { \
                int s = (gx_val ^ gy_val) < 0 ? 1 : -1; \
                if(mag_val > *(mag + ii - N - s) && mag_val > *(mag + ii + N + s)) \
                { \
                    *(res + ii) = mag_val; \
                } \
            } \
        } \
    } \
} while(0)
/**
 * NMS_KERNEL_1x16
 * Performs non-maximum suppression for a 1x16 block of pixels using AVX2.
 * Expects loop variables 'i' (row) and 'j' (starting column) to be defined in scope,
 * with j chosen so that accesses j-1 and j+1 are valid and j+15 < N.
 * All pointers must reference at least M*N int16_t elements.
 * NOTE: YMM3 was previously used uninitialized (andnot). We set it to zero for deterministic behavior.
 */



#define NMS_KERNEL_1x16(gx, gy, mag, res, M, N) \
do{ \
    __m256i tg22x, one, v_gx, v_gy; \
    __m256i dir0, dir45, dir90, dir135; \
    __m256i v_mag, v_res, v_temp0, v_temp1; \
    __m256i v_magl, v_magr, v_magu, v_magd; \
    \
    v_gx = _mm256_loadu_si256((const __m256i*)gx); /* gx */ \
    v_gy = _mm256_loadu_si256((const __m256i*)gy); /* gy */ \
    \
    dir135 = _mm256_xor_si256(v_gx, v_gy); /* gx ^ gy */ \
    v_res = _mm256_setzero_si256(); \
    one = _mm256_cmpeq_epi16(v_res, v_res); \
    dir135 = _mm256_cmpgt_epi16(v_res, dir135); /* gx ^ gy < 0 <=> 135 deg */ \
    \
    v_gx = _mm256_abs_epi16(v_gx); /*|gx|*/  \
    v_gy = _mm256_abs_epi16(v_gy); /*|gy|*/  \
    \
    tg22x = _mm256_broadcastw_epi16(_mm_cvtsi32_si128(27146));/* tg22 scaled constant */ \
    tg22x = _mm256_mulhi_epi16(v_gx, tg22x); /* tg22x = |gx| * tg22 */ \
    dir0 = _mm256_min_epi16(tg22x, v_gy); /* min(tg22x, |gy|) */ \
    dir0 = _mm256_cmpeq_epi16(v_gy, dir0); /* |gy| <= tg22x <=> 0 degree */ \
    \
    v_gx = _mm256_slli_epi16(v_gx, 1); /* |gx| << 1 */ \
    tg22x = _mm256_add_epi16(tg22x, v_gx); /* tg67x = tg22x + (|gx| << 1) */ \
    dir45 = _mm256_min_epi16(tg22x, v_gy); /* min(tg67x, |gy|) */ \
    dir45 = _mm256_cmpeq_epi16(v_gy, dir45); /* |gy| <= tg67x <=> 45 degree */ \
    v_gx = _mm256_subs_epi16(one, dir0); /* |gy| > tg22x */ \
    dir90 = _mm256_subs_epi16(one, dir45); /* |gy| > tg67x */ \
    dir45 = _mm256_and_si256(dir45, v_gx); /* direction 45 deg */ \
    dir135 = _mm256_and_si256(dir45, dir135); /* direction 135 deg */ \
    dir45 = _mm256_subs_epi16(dir45, dir135); /* direction 45 deg */ \
    \
    v_mag = _mm256_loadu_si256((__m256i const*)mag); /* mag[i, j] */ \
    v_magl = _mm256_loadu_si256((__m256i const*)(mag - 1)); /* mag[i, j - 1] */ \
    v_magr = _mm256_loadu_si256((__m256i const*)(mag + 1)); /* mag[i, j + 1] */ \
    v_magu = _mm256_loadu_si256((__m256i const*)(mag - N)); /* mag[i - 1, j] */ \
    v_magd = _mm256_loadu_si256((__m256i const*)(mag + N)); /* mag[i + 1, j] */ \
    v_magl = _mm256_cmpgt_epi16(v_mag, v_magl); /* mag[i, j] > mag[i, j - 1] */ \
    v_magr = _mm256_cmpgt_epi16(v_mag, v_magr); /* mag[i, j] > mag[i, j + 1] */ \
    v_magu = _mm256_cmpgt_epi16(v_mag, v_magu); /* mag[i, j] > mag[i - 1, j] */ \
    v_magd = _mm256_cmpgt_epi16(v_mag, v_magd); /* mag[i, j] > mag[i + 1, j] */ \
    v_temp0 = _mm256_and_si256(v_magl, v_magr); /* mag[i, j] is local max in 0 deg direction */ \
    v_temp0 = _mm256_and_si256(v_temp0, dir0); /* direction 0 deg */ \
    v_temp1 = _mm256_and_si256(v_magu, v_magd); /* mag[i, j] is local max in 90 deg direction */ \
    v_temp1 = _mm256_and_si256(v_temp1, dir90); /* direction 90 deg */ \
    \
    v_res = _mm256_or_si256(v_temp0, v_temp1); \
    \
    v_magl = _mm256_loadu_si256((__m256i const*)(mag - N - 1)); /* mag[i - 1, j - 1] */ \
    v_magr = _mm256_loadu_si256((__m256i const*)(mag + N + 1)); /* mag[i + 1, j + 1] */ \
    v_magu = _mm256_loadu_si256((__m256i const*)(mag - N + 1)); /* mag[i - 1, j + 1] */ \
    v_magd = _mm256_loadu_si256((__m256i const*)(mag + N - 1)); /* mag[i + 1, j - 1] */ \
    v_magl = _mm256_cmpgt_epi16(v_mag, v_magl); /* mag[i, j] > mag[i - 1, j - 1] */ \
    v_magr = _mm256_cmpgt_epi16(v_mag, v_magr); /* mag[i, j] > mag[i + 1, j + 1] */ \
    v_temp0 = _mm256_and_si256(v_magl, v_magr); /* mag[i, j] is local max in 135 deg direction */ \
    v_temp0 = _mm256_and_si256(v_temp0, dir135); /* direction 135 deg */ \
    v_res = _mm256_or_si256(v_res, v_temp0); \
    \
    v_magu = _mm256_cmpgt_epi16(v_mag, v_magu); /* mag[i, j] > mag[i - 1, j + 1] */ \
    v_magd = _mm256_cmpgt_epi16(v_mag, v_magd); /* mag[i, j] > mag[i + 1, j - 1] */ \
    v_temp1 = _mm256_and_si256(v_magu, v_magd); /* mag[i, j] is local max in 45 deg direction */ \
    v_temp1 = _mm256_and_si256(v_temp1, dir45); /* direction 45 deg */ \
    v_res = _mm256_or_si256(v_res, v_temp1); \
    \
    v_res = _mm256_and_si256(v_mag, v_res); /* keep mag where local maxima */ \
    _mm256_storeu_si256((__m256i*)res, v_res); \
}while(0)


#if 0

#define NMS_KERNEL_2x16(gx, gy, mag, res, M, N) \
do{ \
    __m256i ymm0, ymm1, ymm2, ymm3; \
    __m256i ymm4, ymm5, ymm6, ymm7; \
    __m256i ymm8, ymm9, ymm10, ymm11; \
    __m256i ymm12, ymm13, ymm14, ymm15; \
    \
    ymm2 = _mm256_loadu_si256((const __m256i*)gx); /* gx */ \
    ymm3 = _mm256_loadu_si256((const __m256i*)gy); /* gy */ \
    ymm10 = _mm256_loadu_si256((const __m256i*)(gx + N)); /* gx next row */ \
    ymm11 = _mm256_loadu_si256((const __m256i*)(gy + N)); /* gy next row */ \
    \
    ymm7 = _mm256_xor_si256(ymm2, ymm3); /* gx ^ gy */ \
    ymm15 = _mm256_xor_si256(ymm10, ymm11); /* gx ^ gy next row */ \
    \
    ymm9 = _mm256_setzero_si256(); \
    ymm1 = _mm256_cmpeq_epi16(ymm9, ymm9); \
    ymm7 = _mm256_cmpgt_epi16(ymm9, ymm7); /* gx ^ gy < 0 <=> 135 deg */ \
    ymm15 = _mm256_cmpgt_epi16(ymm9, ymm15); /* gx ^ gy < 0 <=> 135 deg next row */ \
    \
    ymm2 = _mm256_abs_epi16(ymm2); /*|gx|*/  \
    ymm10 = _mm256_abs_epi16(ymm10); /*|gx| next row*/  \
    ymm3 = _mm256_abs_epi16(ymm3); /*|gy|*/  \
    ymm11 = _mm256_abs_epi16(ymm11); /*|gy| next row*/  \
    \
    ymm0 = _mm256_broadcastw_epi16(_mm_cvtsi32_si128(27146));/* tg22 scaled constant */ \
    ymm8 = ymm0; \
    ymm0 = _mm256_mulhi_epi16(ymm2, ymm0); /* ymm0 = |gx| * tg22 */ \
    ymm8 = _mm256_mulhi_epi16(ymm10, ymm8); /* ymm8 = |gx| next row * tg22 */ \
    ymm4 = _mm256_min_epi16(ymm0, ymm3); /* min(ymm0, |gy|) */ \
    ymm12 = _mm256_min_epi16(ymm8, ymm11); /* min(ymm0 next row, |gy| next row) */ \
    ymm4 = _mm256_cmpeq_epi16(ymm3, ymm4); /* |gy| <= ymm0 <=> 0 degree */ \
    ymm12 = _mm256_cmpeq_epi16(ymm11, ymm12); /* |gy| <= ymm0 next row <=> 0 degree */ \
    \
    ymm2 = _mm256_slli_epi16(ymm2, 1); /* |gx| << 1 */ \
    ymm10 = _mm256_slli_epi16(ymm10, 1); /* |gx| next row << 1 */ \
    ymm0 = _mm256_add_epi16(ymm0, ymm2); /* tg67x = ymm0 + (|gx| << 1) */ \
    ymm8 = _mm256_add_epi16(ymm8, ymm10); /* tg67x next row = ymm0 next row + (|gx| next row << 1) */ \
    ymm5 = _mm256_min_epi16(ymm0, ymm3); /* min(tg67x, |gy|) */ \
    ymm13 = _mm256_min_epi16(ymm8, ymm11); /* min(tg67x next row, |gy| next row) */ \
    ymm5 = _mm256_cmpeq_epi16(ymm3, ymm5); /* |gy| <= tg67x <=> 45 degree */ \
    ymm13 = _mm256_cmpeq_epi16(ymm11, ymm13); /* |gy| <= tg67x next row <=> 45 degree */ \
    ymm2 = _mm256_subs_epi16(ymm1, ymm4); /* |gy| > ymm0 */ \
    ymm10 = _mm256_subs_epi16(ymm1, ymm12); /* |gy| > ymm0 next row */ \
    ymm6 = _mm256_subs_epi16(ymm1, ymm5); /* |gy| > tg67x */ \
    ymm14 = _mm256_subs_epi16(ymm1, ymm13); /* |gy| > tg67x next row */ \
    ymm5 = _mm256_and_si256(ymm5, ymm2); /* direction 45 deg */ \
    ymm13 = _mm256_and_si256(ymm13, ymm10); /* direction 45 deg next row */ \
    ymm7 = _mm256_and_si256(ymm5, ymm7); /* direction 135 deg */ \
    ymm15 = _mm256_and_si256(ymm13, ymm15); /* direction 135 deg next row */ \
    ymm5 = _mm256_subs_epi16(ymm5, ymm7); /* direction 45 deg */ \
    ymm13 = _mm256_subs_epi16(ymm13, ymm15); /* direction 45 deg next row */ \
    \
    ymm0 = _mm256_loadu_si256((__m256i const*)mag); /* mag[i, j] */ \
    ymm1 = _mm256_loadu_si256((__m256i const*)(mag - N)); /* mag[i - 1, j] */ \
    ymm2 = _mm256_loadu_si256((__m256i const*)(mag + N)); /* mag[i + 1, j] */ \
    ymm3 = _mm256_loadu_si256((__m256i const*)(mag + 2 * N)); /* mag[i + 2, j] */ \
    ymm1 = _mm256_cmpgt_epi16(ymm0, ymm1); /* mag[i, j] > mag[i - 1, j] */ \
    ymm8 = _mm256_cmpgt_epi16(ymm0, ymm2); /* mag[i, j] > mag[i + 1, j] */ \
    ymm9 = _mm256_cmpgt_epi16(ymm2, ymm0); /* mag[i + 1, j] > mag[i, j] */ \
    ymm10 = _mm256_cmpgt_epi16(ymm2, ymm3); /* mag[i + 1, j] > mag[i + 2, j] */ \
    ymm1 = _mm256_and_si256(ymm1, ymm8); /* mag[i, j] is local max in 90 deg direction */ \
    ymm6 = _mm256_and_si256(ymm1, ymm6); /* direction 90 deg */ \
    ymm9 = _mm256_and_si256(ymm9, ymm10); /* mag[i + 1, j] is local max in 90 deg direction */ \
    ymm14 = _mm256_and_si256(ymm9, ymm14); /* direction 90 deg next row */ \
    \
    ymm8 = _mm256_loadu_si256((__m256i const*)(mag - 1)); /* mag[i, j - 1] */ \
    ymm9 = _mm256_loadu_si256((__m256i const*)(mag + 1)); /* mag[i, j + 1] */ \
    ymm10 = _mm256_loadu_si256((__m256i const*)(mag - 1 + 2 * N)); /* mag[i + 2, j - 1] */ \
    ymm11 = _mm256_loadu_si256((__m256i const*)(mag + 1 + 2 * N)); /* mag[i + 2, j + 1] */ \
    ymm10 = _mm256_cmpgt_epi16(ymm2, ymm10); /* mag[i + 1, j] > mag[i + 2, j - 1] */ \
    ymm1 = _mm256_cmpgt_epi16(ymm2, ymm9); /* mag[i + 1, j] > mag[i, j + 1] */ \
    ymm11 = _mm256_cmpgt_epi16(ymm2, ymm11); /* mag[i + 1, j] > mag[i + 2, j + 1] */ \
    ymm3 = _mm256_cmpgt_epi16(ymm2, ymm8); /* mag[i + 1, j] > mag[i, j - 1] */ \
    ymm8 = _mm256_cmpgt_epi16(ymm0, ymm8); /* mag[i, j] > mag[i, j - 1] */ \
    ymm9 = _mm256_cmpgt_epi16(ymm0, ymm9); /* mag[i, j] > mag[i, j + 1] */ \
    ymm10 = _mm256_and_si256(ymm10, ymm1); /* mag[i + 1, j] is local max in 45 deg direction */ \
    ymm13 = _mm256_and_si256(ymm10, ymm13); /* direction 45 deg */ \
    ymm11 = _mm256_and_si256(ymm11, ymm3); /* mag[i + 1, j] is local max in 135 deg direction */ \
    ymm15 = _mm256_and_si256(ymm11, ymm15); /* direction 135 deg next row */ \
    ymm8 = _mm256_and_si256(ymm8, ymm9); /* mag[i, j] is local max in 0 deg direction */ \
    ymm4 = _mm256_and_si256(ymm8, ymm4); /* direction 0 deg */ \
    ymm4 = _mm256_or_si256(ymm4, ymm6); \
    ymm13 = _mm256_or_si256(ymm13, ymm14); \
    ymm13 = _mm256_or_si256(ymm13, ymm15); \
    \
    ymm8 = _mm256_loadu_si256((__m256i const*)(mag - N - 1)); /* mag[i - 1, j - 1] */ \
    ymm9 = _mm256_loadu_si256((__m256i const*)(mag - N + 1)); /* mag[i - 1, j + 1] */ \
    ymm10 = _mm256_loadu_si256((__m256i const*)(mag + N - 1)); /* mag[i + 1, j - 1] */ \
    ymm11 = _mm256_loadu_si256((__m256i const*)(mag + N + 1)); /* mag[i + 1, j + 1] */ \
    ymm8 = _mm256_cmpgt_epi16(ymm0, ymm8); /* mag[i, j] > mag[i - 1, j - 1] */ \
    ymm1 = _mm256_cmpgt_epi16(ymm0, ymm11); /* mag[i, j] > mag[i + 1, j + 1] */ \
    ymm9 = _mm256_cmpgt_epi16(ymm0, ymm9); /* mag[i, j] > mag[i - 1, j + 1] */ \
    ymm3 = _mm256_cmpgt_epi16(ymm0, ymm10); /* mag[i, j] > mag[i + 1, j - 1] */ \
    ymm10 = _mm256_cmpgt_epi16(ymm2, ymm10); /* mag[i + 1, j] > mag[i + 1, j - 1] */ \
    ymm11 = _mm256_cmpgt_epi16(ymm2, ymm11); /* mag[i + 1, j] > mag[i + 1, j + 1] */ \
    ymm8 = _mm256_and_si256(ymm8, ymm1); /* mag[i, j] is local max in 135 deg direction */ \
    ymm7 = _mm256_and_si256(ymm8, ymm7); /* direction 135 deg */ \
    ymm9 = _mm256_and_si256(ymm9, ymm3); /* mag[i, j] is local max in 45 deg direction */ \
    ymm5 = _mm256_and_si256(ymm9, ymm5); /* direction 45 deg */ \
    ymm10 = _mm256_and_si256(ymm10, ymm11); /* mag[i + 1, j] is local max in 0 deg direction */ \
    ymm12 = _mm256_and_si256(ymm10, ymm12); /* direction 0 deg next row */ \
    ymm4 = _mm256_or_si256(ymm4, ymm5); \
    ymm4 = _mm256_or_si256(ymm4, ymm7); \
    ymm12 = _mm256_or_si256(ymm12, ymm13); \
    \
    ymm0 = _mm256_and_si256(ymm0, ymm4); /* keep mag where local maxima */ \
    _mm256_storeu_si256((__m256i*)res, ymm0); \
    ymm2 = _mm256_and_si256(ymm2, ymm12); /* keep mag where local maxima next row */ \
    _mm256_storeu_si256((__m256i*)(res + N), ymm2); \
}while(0)
#endif

#endif
