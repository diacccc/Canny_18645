#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <cstdint>
#include <immintrin.h>

#define SOBEL_3x3(gx, gy, src, step, idx) \
    gx[idx] = (int16_t) src[idx - step - 1] + 2*(int16_t) src[idx - step] + (int16_t) src[idx - step + 1] - \
               (int16_t) src[idx + step - 1] - 2*(int16_t) src[idx + step] - (int16_t) src[idx + step + 1]; \
    gy[idx] = (int16_t) src[idx - step + 1] + 2*(int16_t) src[idx + 1] + (int16_t) src[idx + step + 1] - \
               (int16_t) src[idx - step - 1] - 2*(int16_t) src[idx - 1] - (int16_t) src[idx + step - 1]; \

// void SOBEL_TILE(int16_t* gx1, int16_t* gx2, int16_t* gx3, int16_t* gx4, int16_t* gy1, int16_t* gy2, int16_t* gy3, int16_t* gy4, uint16_t* src0, uint16_t* src1, uint16_t* src2, uint16_t* src3, uint16_t* src4, uint16_t* src5)
#define SOBEL_TILE(gx1, gx2, gx3, gx4, gy1, gy2, gy3, gy4, src0, src1, src2, src3, src4, src5) \
do { \
	__m256i v_gx1, v_gx2, v_gx3, v_gx4; \
	__m256i v_gy1, v_gy2, v_gy3, v_gy4; \
	__m256i v_src0, v_src1, v_src2, v_src3, v_src4, v_src5; \

	v_src0 = _mm256_loadu_si256((const __m256i*) (src0 + 2)); \
	v_src1 = _mm256_loadu_si256((const __m256i*) (src1 + 2)); \
	v_src2 = _mm256_loadu_si256((const __m256i*) (src2 + 2)); \
	v_src3 = _mm256_loadu_si256((const __m256i*) (src3 + 2)); \
	v_src4 = _mm256_loadu_si256((const __m256i*) (src4 + 2)); \
	v_src5 = _mm256_loadu_si256((const __m256i*) (src5 + 2)); \

	v_gx1 = _mm256_add_epi16(src0, src2); \
	v_gx2 = _mm256_add_epi16(src1, src3); \
	v_gx3 = _mm256_add_epi16(src2, src4); \
	v_gx4 = _mm256_add_epi16(src3, src5); \

	v_gy1 = _mm256_sub_epi16(src2, src0); \
	v_gy2 = _mm256_sub_epi16(src3, src1); \
	v_gy3 = _mm256_sub_epi16(src4, src2); \
	v_gy4 = _mm256_sub_epi16(src5, src3); \

	v_gx1 = _mm256_add_epi16(gx1, src1); \
	v_gx2 = _mm256_add_epi16(gx2, src2); \
	v_gx3 = _mm256_add_epi16(gx3, src3); \
	v_gx4 = _mm256_add_epi16(gx4, src4); \

	v_gx1 = _mm256_add_epi16(gx1, src1); \
	v_gx2 = _mm256_add_epi16(gx2, src2); \
	v_gx3 = _mm256_add_epi16(gx3, src3); \
	v_gx4 = _mm256_add_epi16(gx4, src4); \

	v_src0 = _mm256_loadu_si256((const __m256i*) (src0)); \
	v_src1 = _mm256_loadu_si256((const __m256i*) (src1)); \
	v_src2 = _mm256_loadu_si256((const __m256i*) (src2)); \
	v_src3 = _mm256_loadu_si256((const __m256i*) (src3)); \
	v_src4 = _mm256_loadu_si256((const __m256i*) (src4)); \
	v_src5 = _mm256_loadu_si256((const __m256i*) (src5)); \

	v_gx1 = _mm256_sub_epi16(v_gx1, src0); \
	v_gx2 = _mm256_sub_epi16(v_gx2, src1); \
	v_gx3 = _mm256_sub_epi16(v_gx3, src2); \
	v_gx4 = _mm256_sub_epi16(v_gx4, src3); \

	v_gy1 = _mm256_sub_epi16(v_gy1, src0); \
	v_gy2 = _mm256_sub_epi16(v_gy2, src1); \
	v_gy3 = _mm256_sub_epi16(v_gy3, src2); \
	v_gy4 = _mm256_sub_epi16(v_gy4, src3); \

	v_gx1 = _mm256_sub_epi16(v_gx1, src2); \
	v_gx2 = _mm256_sub_epi16(v_gx2, src3); \
	v_gx3 = _mm256_sub_epi16(v_gx3, src4); \
	v_gx4 = _mm256_sub_epi16(v_gx4, src5); \

	v_gy1 = _mm256_add_epi16(v_gy1, src2); \
	v_gy2 = _mm256_add_epi16(v_gy2, src3); \
	v_gy3 = _mm256_add_epi16(v_gy3, src4); \
	v_gy4 = _mm256_add_epi16(v_gy4, src5); \

	v_src0 = _mm256_loadu_si256((const __m256i*) (src0 + 1)); \
	v_src1 = _mm256_loadu_si256((const __m256i*) (src1 + 1)); \
	v_src2 = _mm256_loadu_si256((const __m256i*) (src2 + 1)); \
	v_src3 = _mm256_loadu_si256((const __m256i*) (src3 + 1)); \
	v_src4 = _mm256_loadu_si256((const __m256i*) (src4 + 1)); \
	v_src5 = _mm256_loadu_si256((const __m256i*) (src5 + 1)); \

	v_gy1 = _mm256_sub_epi16(v_gy1, src0); \
	v_gy2 = _mm256_sub_epi16(v_gy2, src1); \
	v_gy3 = _mm256_sub_epi16(v_gy3, src2); \
	v_gy4 = _mm256_sub_epi16(v_gy4, src3); \

	v_gy1 = _mm256_add_epi16(v_gy1, src2); \
	v_gy2 = _mm256_add_epi16(v_gy2, src3); \
	v_gy3 = _mm256_add_epi16(v_gy3, src4); \
	v_gy4 = _mm256_add_epi16(v_gy4, src5); \

	_mm256_storeu_si256((__m256i*) (gx1), v_gx1); \
	_mm256_storeu_si256((__m256i*) (gx2), v_gx2); \
	_mm256_storeu_si256((__m256i*) (gx3), v_gx3); \
	_mm256_storeu_si256((__m256i*) (gx4), v_gx4); \
	_mm256_storeu_si256((__m256i*) (gy1), v_gy1); \
	_mm256_storeu_si256((__m256i*) (gy2), v_gy2); \
	_mm256_storeu_si256((__m256i*) (gy3), v_gy3); \
	_mm256_storeu_si256((__m256i*) (gy4), v_gy4); \
} while (0)

#endif
