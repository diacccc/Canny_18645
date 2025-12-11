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
#define SOBEL_TILE(gx1, gx2, gx3, gy1, gy2, gy3, src0, src1, src2, src3, src4) \
do { \
	__m256i v_gx1, v_gx2, v_gx3; \
	__m256i v_gy1, v_gy2, v_gy3; \
	__m256i v_src0, v_src1, v_src2, v_src3, v_src4; \
    __m256i v_t0, v_t1, v_t2, v_t3, v_t4;\
    v_src0 = _mm256_lddqu_si256((__m256i*)(src0 + 2)); \
	v_src1 = _mm256_lddqu_si256((__m256i*)(src1 + 2)); \
	v_src2 = _mm256_lddqu_si256((__m256i*)(src2 + 2)); \
	v_src3 = _mm256_lddqu_si256((__m256i*)(src3 + 2)); \
	v_src4 = _mm256_lddqu_si256((__m256i*)(src4 + 2)); \
    v_t0 = _mm256_lddqu_si256((__m256i*)(src0));\
    v_t1 = _mm256_lddqu_si256((__m256i*)(src1));\
    v_t2 = _mm256_lddqu_si256((__m256i*)(src2));\
    v_t3 = _mm256_lddqu_si256((__m256i*)(src3));\
    v_t4 = _mm256_lddqu_si256((__m256i*)(src4));\
	v_gx1 = _mm256_add_epi16(v_src0, v_src2); \
	v_gx2 = _mm256_add_epi16(v_src1, v_src3); \
	v_gx3 = _mm256_add_epi16(v_src2, v_src4); \
	v_gy1 = _mm256_sub_epi16(v_src2, v_src0); \
	v_gy2 = _mm256_sub_epi16(v_src3, v_src1); \
	v_gy3 = _mm256_sub_epi16(v_src4, v_src2); \
	v_gx1 = _mm256_add_epi16(v_gx1, v_src1); \
	v_gx2 = _mm256_add_epi16(v_gx2, v_src2); \
	v_gx3 = _mm256_add_epi16(v_gx3, v_src3); \
	v_gx1 = _mm256_add_epi16(v_gx1, v_src1); \
	v_gx2 = _mm256_add_epi16(v_gx2, v_src2); \
	v_gx3 = _mm256_add_epi16(v_gx3, v_src3); \
	v_gx1 = _mm256_sub_epi16(v_gx1, v_t0); \
	v_gx2 = _mm256_sub_epi16(v_gx2, v_t1); \
	v_gx3 = _mm256_sub_epi16(v_gx3, v_t2); \
	v_gx1 = _mm256_sub_epi16(v_gx1, v_t2); \
	v_gx2 = _mm256_sub_epi16(v_gx2, v_t3); \
	v_gx3 = _mm256_sub_epi16(v_gx3, v_t4); \
	_mm256_storeu_si256((__m256i*)(gx1), v_gx1); \
	_mm256_storeu_si256((__m256i*)(gx2), v_gx2); \
	_mm256_storeu_si256((__m256i*)(gx3), v_gx3); \
	v_gy1 = _mm256_sub_epi16(v_gy1, v_t0); \
	v_gy2 = _mm256_sub_epi16(v_gy2, v_t1); \
	v_gy3 = _mm256_sub_epi16(v_gy3, v_t2); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_t2); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_t3); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_t4); \
	v_src0 = _mm256_lddqu_si256((__m256i*)(src0 + 1)); \
	v_src1 = _mm256_lddqu_si256((__m256i*)(src1 + 1)); \
	v_src2 = _mm256_lddqu_si256((__m256i*)(src2 + 1)); \
	v_src3 = _mm256_lddqu_si256((__m256i*)(src3 + 1)); \
	v_src4 = _mm256_lddqu_si256((__m256i*)(src4 + 1)); \
	v_gy1 = _mm256_sub_epi16(v_gy1, v_src0); \
	v_gy2 = _mm256_sub_epi16(v_gy2, v_src1); \
	v_gy3 = _mm256_sub_epi16(v_gy3, v_src2); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src2); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src3); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src4); \
	_mm256_storeu_si256((__m256i*)(gy1), v_gy1); \
	_mm256_storeu_si256((__m256i*)(gy2), v_gy2); \
	_mm256_storeu_si256((__m256i*)(gy3), v_gy3); \
} while (0)

#endif
