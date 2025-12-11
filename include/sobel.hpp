#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <cstdint>
#include <immintrin.h>

#define SOBEL_3x3(gx, gy, src, step, idx) \
    gx[idx] = (int16_t) src[idx - step - 1] + 2*(int16_t) src[idx - step] + (int16_t) src[idx - step + 1] - \
               (int16_t) src[idx + step - 1] - 2*(int16_t) src[idx + step] - (int16_t) src[idx + step + 1]; \
    gy[idx] = (int16_t) src[idx - step + 1] + 2*(int16_t) src[idx + 1] + (int16_t) src[idx + step + 1] - \
               (int16_t) src[idx - step - 1] - 2*(int16_t) src[idx - 1] - (int16_t) src[idx + step - 1]; \

#define SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_row, curr_row, next_row, LEN, POS) \
do { \
    for (int ii = 0; ii < LEN; ++ii) \
    { \
        *(mag_ptr + ii) = 0; \
        int gx = 0; \
        int gy = 0; \
\
        const int has_left  = !(POS == 1 && ii == 0); \
        const int has_right = !(POS == -1 && ii == LEN - 1); \
\
        int16_t p_ul = has_left  ? *(prev_row + ii - 1) : 0; \
        int16_t p_u  = *(prev_row + ii); \
        int16_t p_ur = has_right ? *(prev_row + ii + 1) : 0; \
\
        int16_t p_ml = has_left  ? *(curr_row + ii - 1) : 0; \
        int16_t p_m  = *(curr_row + ii); \
        int16_t p_mr = has_right ? *(curr_row + ii + 1) : 0; \
\
        int16_t p_ll = has_left  ? *(next_row + ii - 1) : 0; \
        int16_t p_l  = *(next_row + ii); \
        int16_t p_lr = has_right ? *(next_row + ii + 1) : 0; \
\
        /* Sobel Gx */ \
        int gx_val = -p_ul + p_ur \
             -2*p_ml + 2*p_mr \
             -p_ll + p_lr; \
        *(gx_ptr + ii) = (int16_t)gx_val; \
\
        /* Sobel Gy */ \
        int gy_val = -p_ul - 2*p_u - p_ur + \
             p_ll + 2*p_l + p_lr; \
        *(gy_ptr + ii) = (int16_t)gy_val; \
\
        int mag_val = (int)std::abs(gx_val) + (int)std::abs(gy_val); \
        *(mag_ptr + ii) = (int16_t) mag_val; \
} \
} while(0)

#define SOBEL_TILE(gx, gy, mag, src0, src1, src2) \
do { \
	__m256i v_gx1, v_gx2, v_gx3, v_gx4; \
	__m256i v_gy1, v_gy2, v_gy3, v_gy4; \
	__m256i v_src0, v_src1, v_src2, v_src3, v_src4, v_src5, v_src6, v_src7; \
	v_src0 = _mm256_loadu_si256((__m256i*)((src0) - 1)); \
	v_src1 = _mm256_loadu_si256((__m256i*)((src0) + 15)); \
	v_src2 = _mm256_loadu_si256((__m256i*)((src0) + 31)); \
	v_src3 = _mm256_loadu_si256((__m256i*)((src0) + 47)); \
	v_src4 = _mm256_loadu_si256((__m256i*)((src0) + 1)); \
	v_src5 = _mm256_loadu_si256((__m256i*)((src0) + 17)); \
	v_src6 = _mm256_loadu_si256((__m256i*)((src0) + 33)); \
	v_src7 = _mm256_loadu_si256((__m256i*)((src0) + 49)); \
	v_gx1 = _mm256_sub_epi16(v_src4, v_src0); \
	v_gx2 = _mm256_sub_epi16(v_src5, v_src1); \
	v_gx3 = _mm256_sub_epi16(v_src6, v_src2); \
	v_gx4 = _mm256_sub_epi16(v_src7, v_src3); \
	v_gy1 = _mm256_add_epi16(v_src0, v_src4); \
	v_gy2 = _mm256_add_epi16(v_src1, v_src5); \
	v_gy3 = _mm256_add_epi16(v_src2, v_src6); \
	v_gy4 = _mm256_add_epi16(v_src3, v_src7); \
	v_src0 = _mm256_loadu_si256((__m256i*)((src0))); \
	v_src1 = _mm256_loadu_si256((__m256i*)((src0) + 16)); \
	v_src2 = _mm256_loadu_si256((__m256i*)((src0) + 32)); \
	v_src3 = _mm256_loadu_si256((__m256i*)((src0) + 48)); \
	v_src4 = _mm256_loadu_si256((__m256i*)((src1) - 1)); \
	v_src5 = _mm256_loadu_si256((__m256i*)((src1) + 15)); \
	v_src6 = _mm256_loadu_si256((__m256i*)((src1) + 31)); \
	v_src7 = _mm256_loadu_si256((__m256i*)((src1) + 47)); \
	v_gx1 = _mm256_sub_epi16(v_gx1, v_src4); \
	v_gx2 = _mm256_sub_epi16(v_gx2, v_src5); \
	v_gx3 = _mm256_sub_epi16(v_gx3, v_src6); \
	v_gx4 = _mm256_sub_epi16(v_gx4, v_src7); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src0); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src1); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src2); \
	v_gy4 = _mm256_add_epi16(v_gy4, v_src3); \
	v_gx1 = _mm256_sub_epi16(v_gx1, v_src4); \
	v_gx2 = _mm256_sub_epi16(v_gx2, v_src5); \
	v_gx3 = _mm256_sub_epi16(v_gx3, v_src6); \
	v_gx4 = _mm256_sub_epi16(v_gx4, v_src7); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src0); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src1); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src2); \
	v_gy4 = _mm256_add_epi16(v_gy4, v_src3); \
	v_src0 = _mm256_loadu_si256((__m256i*)((src1) + 1)); \
	v_src1 = _mm256_loadu_si256((__m256i*)((src1) + 17)); \
	v_src2 = _mm256_loadu_si256((__m256i*)((src1) + 33)); \
	v_src3 = _mm256_loadu_si256((__m256i*)((src1) + 49)); \
	v_src4 = _mm256_loadu_si256((__m256i*)((src2))); \
	v_src5 = _mm256_loadu_si256((__m256i*)((src2) + 16)); \
	v_src6 = _mm256_loadu_si256((__m256i*)((src2) + 32)); \
	v_src7 = _mm256_loadu_si256((__m256i*)((src2) + 48)); \
	v_gx1 = _mm256_add_epi16(v_gx1, v_src0); \
	v_gx2 = _mm256_add_epi16(v_gx2, v_src1); \
	v_gx3 = _mm256_add_epi16(v_gx3, v_src2); \
	v_gx4 = _mm256_add_epi16(v_gx4, v_src3); \
	v_gy1 = _mm256_sub_epi16(v_src4, v_gy1); \
	v_gy2 = _mm256_sub_epi16(v_src5, v_gy2); \
	v_gy3 = _mm256_sub_epi16(v_src6, v_gy3); \
	v_gy4 = _mm256_sub_epi16(v_src7, v_gy4); \
	v_gx1 = _mm256_add_epi16(v_gx1, v_src0); \
	v_gx2 = _mm256_add_epi16(v_gx2, v_src1); \
	v_gx3 = _mm256_add_epi16(v_gx3, v_src2); \
	v_gx4 = _mm256_add_epi16(v_gx4, v_src3); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src4); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src5); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src6); \
	v_gy4 = _mm256_add_epi16(v_gy4, v_src7); \
	v_src0 = _mm256_loadu_si256((__m256i*)((src2) - 1));\
	v_src1 = _mm256_loadu_si256((__m256i*)((src2) + 15)); \
	v_src2 = _mm256_loadu_si256((__m256i*)((src2) + 31)); \
	v_src3 = _mm256_loadu_si256((__m256i*)((src2) + 47)); \
	v_src4 = _mm256_loadu_si256((__m256i*)((src2) + 1)); \
	v_src5 = _mm256_loadu_si256((__m256i*)((src2) + 17)); \
	v_src6 = _mm256_loadu_si256((__m256i*)((src2) + 33)); \
	v_src7 = _mm256_loadu_si256((__m256i*)((src2) + 49)); \
	v_gx1 = _mm256_sub_epi16(v_gx1, v_src0); \
	v_gx2 = _mm256_sub_epi16(v_gx2, v_src1); \
	v_gx3 = _mm256_sub_epi16(v_gx3, v_src2); \
	v_gx4 = _mm256_sub_epi16(v_gx4, v_src3); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src0); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src1); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src2); \
	v_gy4 = _mm256_add_epi16(v_gy4, v_src3); \
	v_gx1 = _mm256_add_epi16(v_gx1, v_src4); \
	v_gx2 = _mm256_add_epi16(v_gx2, v_src5); \
	v_gx3 = _mm256_add_epi16(v_gx3, v_src6); \
	v_gx4 = _mm256_add_epi16(v_gx4, v_src7); \
	v_gy1 = _mm256_add_epi16(v_gy1, v_src4); \
	v_gy2 = _mm256_add_epi16(v_gy2, v_src5); \
	v_gy3 = _mm256_add_epi16(v_gy3, v_src6); \
	v_gy4 = _mm256_add_epi16(v_gy4, v_src7); \
	_mm256_storeu_si256((__m256i*)((gx)), v_gx1); \
	_mm256_storeu_si256((__m256i*)((gx) + 16), v_gx2); \
	_mm256_storeu_si256((__m256i*)((gx) + 32), v_gx3); \
	_mm256_storeu_si256((__m256i*)((gx) + 48), v_gx4); \
	_mm256_storeu_si256((__m256i*)((gy)), v_gy1); \
	_mm256_storeu_si256((__m256i*)((gy) + 16), v_gy2); \
	_mm256_storeu_si256((__m256i*)((gy) + 32), v_gy3); \
	_mm256_storeu_si256((__m256i*)((gy) + 48), v_gy4); \
	/* Magnitude Calculation */ \
	v_src0 = _mm256_abs_epi16(v_gx1); \
	v_src1 = _mm256_abs_epi16(v_gy1); \
	v_src2 = _mm256_abs_epi16(v_gx2); \
	v_src3 = _mm256_abs_epi16(v_gy2); \
	v_src4 = _mm256_abs_epi16(v_gx3); \
	v_src5 = _mm256_abs_epi16(v_gy3); \
	v_src6 = _mm256_abs_epi16(v_gx4); \
	v_src7 = _mm256_abs_epi16(v_gy4); \
	v_src0 = _mm256_add_epi16(v_src0, v_src1); \
	v_src2 = _mm256_add_epi16(v_src2, v_src3); \
	v_src4 = _mm256_add_epi16(v_src4, v_src5); \
	v_src6 = _mm256_add_epi16(v_src6, v_src7); \
	_mm256_storeu_si256((__m256i*)((mag)), v_src0); \
	_mm256_storeu_si256((__m256i*)((mag) + 16), v_src2); \
	_mm256_storeu_si256((__m256i*)((mag) + 32), v_src4); \
	_mm256_storeu_si256((__m256i*)((mag) + 48), v_src6); \
} while (0)



void sobel(const int16_t* src, int16_t* gx, int16_t* gy,
                        int16_t* mag, int M, int N)
{
    const int BLOCK_WIDTH = 2032; // 2032 pixels per block to fit in L1 cache
    std::vector<int16_t> zero_buf(80, 0);
    for (int j = 1; j < N; j += BLOCK_WIDTH) {
        int block_width = std::min(BLOCK_WIDTH, N - j - 1);
        block_width = block_width & (~63); // make it multiple of 64
        // Initialize pointers for the first two rows
        // Process each row
        const int16_t* prev_src_ptr = zero_buf.data(); // zero for the first row
        const int16_t* curr_src_ptr = src + ((0 + 0) * N);
        const int16_t* next_src_ptr = src + ((0 + 1) * N);
        int16_t* gx_ptr = gx + (0 * N);
        int16_t* gy_ptr = gy + (0 * N);
        int16_t* mag_ptr = mag + (0 * N);
        if (j == 1) {
            SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, block_width, 1);
            gx_ptr += 1;
            gy_ptr += 1;
            curr_src_ptr += 1;
            next_src_ptr += 1;
            mag_ptr += 1;
        } else {
            gx_ptr += j;
            gy_ptr += j;
            curr_src_ptr += j;
            next_src_ptr += j;
            mag_ptr += j;
        }
        for (int k = 0; k < block_width; k += 64) {
            SOBEL_TILE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr);
            gx_ptr += 64;
            gy_ptr += 64;
            curr_src_ptr += 64;
            next_src_ptr += 64;
            mag_ptr += 64;
        }
        if (block_width != BLOCK_WIDTH &&  N - j - block_width > 0) {
            printf("Processing first row last edge: j=%d, block_width=%d, N=%d\n", j, block_width, N);
            // Process the last column separately to handle right border
            const int LEN = N - j - block_width;
            SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, LEN, -1);
        }

        for (int i = 1; i < M - 1; ++i) {
            prev_src_ptr = src + ((i - 1) * N);
            curr_src_ptr = src + ((i + 0) * N);
            next_src_ptr = src + ((i + 1) * N);
            gx_ptr = gx + (i * N);
            gy_ptr = gy + (i * N);
            mag_ptr = mag + (i * N);

            if (j == 1) {
                SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, block_width, 1);
                gx_ptr += 1;
                gy_ptr += 1;
                prev_src_ptr += 1;
                curr_src_ptr += 1;
                next_src_ptr += 1;
                mag_ptr += 1;
            } else {
                gx_ptr += j;
                gy_ptr += j;
                prev_src_ptr += j;
                curr_src_ptr += j;
                next_src_ptr += j;
                mag_ptr += j;
            }
            for (int k = 0; k < block_width; k += 64) {
                if (k == 0 && i == 1) {
                    // For debugging: print the first 64 pixels of the current row being processed
                    std::cout << "First Tile of row " << i << " :" << std::endl;
                    for (int i = -1; i < 65; ++i) {
                        std::cout << (int)*(prev_src_ptr + i) << " ";
                    }
                    std::cout << std::endl;
                    for (int i = -1; i < 65; ++i) {
                        std::cout << (int)*(curr_src_ptr + i) << " ";
                    }
                    std::cout << std::endl;
                    for (int i = -1; i < 65; ++i) {
                        std::cout << (int)*(next_src_ptr + i) << " ";
                    }
                    std::cout << std::endl;
                }
                SOBEL_TILE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr);
                gx_ptr += 64;
                gy_ptr += 64;
                prev_src_ptr += 64;
                curr_src_ptr += 64;
                next_src_ptr += 64;
                mag_ptr += 64;
            }
            if (block_width != BLOCK_WIDTH &&  N - j - block_width > 0) {
                // Process the last column separately to handle right border
                const int LEN = N - j - block_width;
                SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, LEN, -1);
            }
        }

        prev_src_ptr = src + ((M - 2) * N);
        curr_src_ptr = src + ((M - 1) * N);
        next_src_ptr = zero_buf.data();
        gx_ptr = gx + ((M - 1) * N);
        gy_ptr = gy + ((M - 1) * N);
        mag_ptr = mag + ((M - 1) * N);

        if (j == 1) {
            SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, block_width, 1);
            gx_ptr += 1;
            gy_ptr += 1;
            prev_src_ptr += 1;
            curr_src_ptr += 1;
            mag_ptr += 1;
        } else {
            gx_ptr += j;
            gy_ptr += j;
            prev_src_ptr += j;
            curr_src_ptr += j;
            mag_ptr += j;
        }
        for (int k = 0; k < block_width; k += 64) {
            SOBEL_TILE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr);
            gx_ptr += 64;
            gy_ptr += 64;
            prev_src_ptr += 64;
            curr_src_ptr += 64;
            mag_ptr += 64;
        }
        if (block_width != BLOCK_WIDTH &&  N - j - block_width > 0) {
            // Process the last column separately to handle right border
            const int LEN = N - j - block_width;
            SOBEL_EDGE(gx_ptr, gy_ptr, mag_ptr, prev_src_ptr, curr_src_ptr, next_src_ptr, LEN, -1);
        }

    }
    
}

#endif
