#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <cstdint>

#define SOBEL_3x3(gx, gy, src, step, idx) \
    gx[idx] = (int16_t) src[idx - step - 1] + 2*(int16_t) src[idx - step] + (int16_t) src[idx - step + 1] - \
               (int16_t) src[idx + step - 1] - 2*(int16_t) src[idx + step] - (int16_t) src[idx + step + 1]; \
    gy[idx] = (int16_t) src[idx - step + 1] + 2*(int16_t) src[idx + 1] + (int16_t) src[idx + step + 1] - \
               (int16_t) src[idx - step - 1] - 2*(int16_t) src[idx - 1] - (int16_t) src[idx + step - 1]; \

#endif

#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>

void Sobel_5x1(uint16_t* Img, int M, int N, int row, int col, int16_t* Gx, int16_t* Gy
        // uint16_t* Mag
) {
    __m256i a0, a1, a2, a3, a4, a5;
    __m256i gx1, gx2, gx3, gx4, gx5;
    __m256i gy1, gy2, gy3, gy4, gy5;

    // L0 - L5
    a0 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * row + col));
    a1 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 1) + col));
    a2 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 2) + col));
    a3 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 3) + col));
    a4 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 4) + col));
    a5 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 5) + col));

    gx1 = _mm256_sub_epi16(gx1, a0);
    gy1 = _mm256_sub_epi16(gy1, a0);
    gx2 = _mm256_sub_epi16(gx2, a1);
    gy2 = _mm256_sub_epi16(gy2, a1);
    gx3 = _mm256_sub_epi16(gx3, a2);
    gy3 = _mm256_sub_epi16(gy3, a2);
    gx4 = _mm256_sub_epi16(gx4, a3);
    gy4 = _mm256_sub_epi16(gy4, a3);
    gx5 = _mm256_sub_epi16(gx5, a4);
    gy5 = _mm256_sub_epi16(gy5, a4);

    gx1 = _mm256_sub_epi16(gx1, a1);
    gx2 = _mm256_sub_epi16(gx2, a2);
    gx3 = _mm256_sub_epi16(gx3, a3);
    gx4 = _mm256_sub_epi16(gx4, a4);
    gx5 = _mm256_sub_epi16(gx5, a5);

    gx1 = _mm256_sub_epi16(gx1, a1);
    gx2 = _mm256_sub_epi16(gx2, a2);
    gx3 = _mm256_sub_epi16(gx3, a3);
    gx4 = _mm256_sub_epi16(gx4, a4);
    gx5 = _mm256_sub_epi16(gx5, a5);

    // L1 - L6
    a0 = _mm256_load_si256(reinterpret_cast<__m256i*>(Img + N * (row + 6) + col));

    gx1 = _mm256_sub_epi16(gx1, a2);
    gy1 = _mm256_add_epi16(gy1, a2);
    gx2 = _mm256_sub_epi16(gx2, a3);
    gy2 = _mm256_add_epi16(gy2, a3);
    gx3 = _mm256_sub_epi16(gx3, a4);
    gy3 = _mm256_add_epi16(gy3, a4);
    gx4 = _mm256_sub_epi16(gx4, a5);
    gy4 = _mm256_add_epi16(gy4, a5);
    gx5 = _mm256_sub_epi16(gx5, a0);
    gy5 = _mm256_add_epi16(gy5, a0);

    // R0 - R5
    a0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * row + (col + 2)));
    a1 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 1) + (col + 2)));
    a2 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 2) + (col + 2)));
    a3 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 3) + (col + 2)));
    a4 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 4) + (col + 2)));
    a5 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 5) + (col + 2)));

    gx1 = _mm256_add_epi16(gx1, a0);
    gy1 = _mm256_sub_epi16(gy1, a0);
    gx2 = _mm256_add_epi16(gx2, a1);
    gy2 = _mm256_sub_epi16(gy2, a1);
    gx3 = _mm256_add_epi16(gx3, a2);
    gy3 = _mm256_sub_epi16(gy3, a2);
    gx4 = _mm256_add_epi16(gx4, a3);
    gy4 = _mm256_sub_epi16(gy4, a3);
    gx5 = _mm256_add_epi16(gx5, a4);
    gy5 = _mm256_sub_epi16(gy5, a4);

    gx1 = _mm256_add_epi16(gx1, a1);
    gx2 = _mm256_add_epi16(gx2, a2);
    gx3 = _mm256_add_epi16(gx3, a3);
    gx4 = _mm256_add_epi16(gx4, a4);
    gx5 = _mm256_add_epi16(gx5, a5);

    gx1 = _mm256_add_epi16(gx1, a1);
    gx2 = _mm256_add_epi16(gx2, a2);
    gx3 = _mm256_add_epi16(gx3, a3);
    gx4 = _mm256_add_epi16(gx4, a4);
    gx5 = _mm256_add_epi16(gx5, a5);

    // R1 - R6
    a0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 6) + (col + 2)));

    gx1 = _mm256_add_epi16(gx1, a2);
    gy1 = _mm256_add_epi16(gy1, a2);
    gx2 = _mm256_add_epi16(gx2, a3);
    gy2 = _mm256_add_epi16(gy2, a3);
    gx3 = _mm256_add_epi16(gx3, a4);
    gy3 = _mm256_add_epi16(gy3, a4);
    gx4 = _mm256_add_epi16(gx4, a5);
    gy4 = _mm256_add_epi16(gy4, a5);
    gx5 = _mm256_add_epi16(gx5, a0);
    gy5 = _mm256_add_epi16(gy5, a0);

    // C0 - C5
    a0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * row + (col + 1)));
    a1 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 1) + (col + 1)));
    a2 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 2) + (col + 1)));
    a3 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 3) + (col + 1)));
    a4 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 4) + (col + 1)));
    a5 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 5) + (col + 1)));

    gy1 = _mm256_sub_epi16(gy1, a0);
    gy2 = _mm256_sub_epi16(gy2, a1);
    gy3 = _mm256_sub_epi16(gy3, a2);
    gy4 = _mm256_sub_epi16(gy4, a3);
    gy5 = _mm256_sub_epi16(gy5, a4);

    gy1 = _mm256_sub_epi16(gy1, a0);
    gy2 = _mm256_sub_epi16(gy2, a1);
    gy3 = _mm256_sub_epi16(gy3, a2);
    gy4 = _mm256_sub_epi16(gy4, a3);
    gy5 = _mm256_sub_epi16(gy5, a4);

    // C1 - C6
    a0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(Img + N * (row + 6) + (col + 1)));

    gy1 = _mm256_sub_epi16(gy1, a2);
    gy2 = _mm256_sub_epi16(gy2, a3);
    gy3 = _mm256_sub_epi16(gy3, a4);
    gy4 = _mm256_sub_epi16(gy4, a5);
    gy5 = _mm256_sub_epi16(gy5, a0);

    gy1 = _mm256_sub_epi16(gy1, a2);
    gy2 = _mm256_sub_epi16(gy2, a3);
    gy3 = _mm256_sub_epi16(gy3, a4);
    gy4 = _mm256_sub_epi16(gy4, a5);
    gy5 = _mm256_sub_epi16(gy5, a0);

    // Store

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gx + N * row + col), gx1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gx + N * (row + 1) + col), gx2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gx + N * (row + 2) + col), gx3);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gx + N * (row + 3) + col), gx4);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gx + N * (row + 4) + col), gx5);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gy + N * row + col), gy1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gy + N * (row + 1) + col), gy2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gy + N * (row + 2) + col), gy3);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gy + N * (row + 3) + col), gy4);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Gy + N * (row + 4) + col), gy5);


    // Magnitude
    // a0 = _mm256_abs_epi16(gx1);
    // a1 = _mm256_abs_epi16(gx2);
    // a2 = _mm256_abs_epi16(gx3);
    // a3 = _mm256_abs_epi16(gy1);
    // a4 = _mm256_abs_epi16(gy2);
    // a5 = _mm256_abs_epi16(gy3);

    // a0 = _mm256_add_epi16(a0, a3);
    // a1 = _mm256_add_epi16(a1, a4);
    // a2 = _mm256_add_epi16(a2, a5);

    // a3 = _mm256_abs_epi16(gx4);
    // a4 = _mm256_abs_epi16(gx5);
    // a5 = _mm256_abs_epi16(gy4);

    // a3 = _mm256_add_epi16(a3, a5);

    // a5 = _mm256_abs_epi16(gy5);
    // a4 = _mm256_add_epi16(a4, a5);
}