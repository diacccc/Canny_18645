#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <cstdint>
#include <immintrin.h>

#define THRESHOLDING_TILE(map0, map1, map2, mag, low_threshold, res) \
do {\
	__m256i v_up0, v_up1, v_up2; \
	__m256i v_mid0, v_mid1, v_mid2; \
	__m256i v_lo0, v_lo1, v_lo2; \
	__m256i v_thresh; \
	__m256i v_res0, v_res1, v_res2; \
	__m256i v_mag0, v_mag1, v_mag2; \
	v_thresh = _mm256_broadcastw_epi16(_mm_cvtsi32_si128(low_threshold)); \
	v_up0 = _mm256_loadu_si256((__m256i*) (map0)); \
	v_up1 = _mm256_loadu_si256((__m256i*) (map0 + 16)); \
	v_up2 = _mm256_loadu_si256((__m256i*) (map0 + 32)); \
	v_mid0 = _mm256_loadu_si256((__m256i*) (map1)); \
	v_mid1 = _mm256_loadu_si256((__m256i*) (map1 + 16)); \
	v_mid2 = _mm256_loadu_si256((__m256i*) (map1 + 32)); \
	v_lo0 = _mm256_loadu_si256((__m256i*) (map2)); \
	v_lo1 = _mm256_loadu_si256((__m256i*) (map2 + 16)); \
	v_lo2 = _mm256_loadu_si256((__m256i*) (map2 + 32)); \
	v_mag0 = _mm256_loadu_si256((__m256i*) (mag)); \
	v_mag1 = _mm256_loadu_si256((__m256i*) (mag + 16)); \
	v_mag2 = _mm256_loadu_si256((__m256i*) (mag + 32)); \
	v_res0 = _mm256_or_si256(v_up0, v_mid0); \
	v_res1 = _mm256_or_si256(v_up1, v_mid1); \
	v_res2 = _mm256_or_si256(v_up2, v_mid2); \
	v_res0 = _mm256_or_si256(v_res0, v_lo0); \
	v_res1 = _mm256_or_si256(v_res1, v_lo1); \
	v_res2 = _mm256_or_si256(v_res2, v_lo2); \
	v_up0 = _mm256_loadu_si256((__m256i*) (map0 + 1)); \
	v_up1 = _mm256_loadu_si256((__m256i*) (map0 + 17)); \
	v_up2 = _mm256_loadu_si256((__m256i*) (map0 + 33)); \
	v_mid0 = _mm256_loadu_si256((__m256i*) (map1 + 1)); \
	v_mid1 = _mm256_loadu_si256((__m256i*) (map1 + 17)); \
	v_mid2 = _mm256_loadu_si256((__m256i*) (map1 + 33)); \
	v_lo0 = _mm256_loadu_si256((__m256i*) (map2 + 1)); \
	v_lo1 = _mm256_loadu_si256((__m256i*) (map2 + 17)); \
	v_lo2 = _mm256_loadu_si256((__m256i*) (map2 + 33)); \
	v_res0 = _mm256_or_si256(v_res0, v_up0); \
	v_res1 = _mm256_or_si256(v_res1, v_up1); \
	v_res2 = _mm256_or_si256(v_res2, v_up2); \
	v_res0 = _mm256_or_si256(v_res0, v_mid0); \
	v_res1 = _mm256_or_si256(v_res1, v_mid1); \
	v_res2 = _mm256_or_si256(v_res2, v_mid2); \
	v_res0 = _mm256_or_si256(v_res0, v_lo0); \
	v_res1 = _mm256_or_si256(v_res1, v_lo1); \
	v_res2 = _mm256_or_si256(v_res2, v_lo2); \
	v_up0 = _mm256_loadu_si256((__m256i*) (map0 + 2)); \
	v_up1 = _mm256_loadu_si256((__m256i*) (map0 + 18)); \
	v_up2 = _mm256_loadu_si256((__m256i*) (map0 + 34)); \
	v_mid0 = _mm256_loadu_si256((__m256i*) (map1 + 2)); \
	v_mid1 = _mm256_loadu_si256((__m256i*) (map1 + 18)); \
	v_mid2 = _mm256_loadu_si256((__m256i*) (map1 + 34)); \
	v_lo0 = _mm256_loadu_si256((__m256i*) (map2 + 2)); \
	v_lo1 = _mm256_loadu_si256((__m256i*) (map2 + 18)); \
	v_lo2 = _mm256_loadu_si256((__m256i*) (map2 + 34)); \
	v_res0 = _mm256_or_si256(v_res0, v_up0); \
	v_res1 = _mm256_or_si256(v_res1, v_up1); \
	v_res2 = _mm256_or_si256(v_res2, v_up2); \
	v_res0 = _mm256_or_si256(v_res0, v_mid0); \
	v_res1 = _mm256_or_si256(v_res1, v_mid1); \
	v_res2 = _mm256_or_si256(v_res2, v_mid2); \
	v_res0 = _mm256_or_si256(v_res0, v_lo0); \
	v_res1 = _mm256_or_si256(v_res1, v_lo1); \
	v_res2 = _mm256_or_si256(v_res2, v_lo2); \
	v_mag0 = _mm256_cmpgt_epi16(v_mag0, v_thresh); \
	v_mag1 = _mm256_cmpgt_epi16(v_mag1, v_thresh); \
	v_mag2 = _mm256_cmpgt_epi16(v_mag2, v_thresh); \
	v_res0 = _mm256_and_si256(v_res0, v_mag0); \
	v_res1 = _mm256_and_si256(v_res1, v_mag1); \
	v_res2 = _mm256_and_si256(v_res2, v_mag2); \
	_mm256_storeu_si256((__m256i*) (res), v_res0); \
	_mm256_storeu_si256((__m256i*) (res + 16), v_res1); \
	_mm256_storeu_si256((__m256i*) (res + 32), v_res2); \
} while (0)

#endif
