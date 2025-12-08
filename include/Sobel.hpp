#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <cstdint>
#include <immintrin.h>

#define SOBEL_3x3(gx, gy, src, step, idx) \
    gx[idx] = (int16_t) src[idx - step - 1] + 2*(int16_t) src[idx - step] + (int16_t) src[idx - step + 1] - \
               (int16_t) src[idx + step - 1] - 2*(int16_t) src[idx + step] - (int16_t) src[idx + step + 1]; \
    gy[idx] = (int16_t) src[idx - step + 1] + 2*(int16_t) src[idx + 1] + (int16_t) src[idx + step + 1] - \
               (int16_t) src[idx - step - 1] - 2*(int16_t) src[idx - 1] - (int16_t) src[idx + step - 1]; \

#endif

// Question: Is tiling horizontal or vertical?

void sobel(uint16_t* Img, int M, int N, int TILE_ROWS, int TILE_COLS, uint16_t* Gx, uint16_t* Gy) {
	__m256i vGx1, vGx2, vGx3;
	__m256i vGy1, vGy2, vGy3;
	__m256i vA0, vA1, vA2, vA3, vA4;
	__m256i vB0, vB1, vB2, vB3, vB4;
	for (int i = 0; i < M; i += TILE_ROWS) {
		for (int j = 0; j < N; j += TILE_COLS) {
			for (int x = 0; x < TILE_ROWS - 4; x += 3) {
				for (int y = 0; y < TILE_COLS; y += 16) {
    				vB0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 0) * TILE_COLS + y + 2]));
    				vB1 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 1) * TILE_COLS + y + 2]));
    				vB2 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 2) * TILE_COLS + y + 2]));
    				vB3 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 3) * TILE_COLS + y + 2]));
    				vB4 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 4) * TILE_COLS + y + 2]));
    
    				// vGx1 = _mm256_add_epi16(vB1, vB1);
    				// vGx2 = _mm256_add_epi16(vB2, vB2);
    				// vGx3 = _mm256_add_epi16(vB3, vB3);
    
    				// vGy1 = _mm256_sub_epi16(vB2, vB0);
    				// vGy2 = _mm256_sub_epi16(vB3, vB1);
    				// vGy3 = _mm256_sub_epi16(vB4, vB2);
    
    				// vGx1 = _mm256_add_epi16(vGx1, vB0);
    				// vGx2 = _mm256_add_epi16(vGx2, vB1);
    				// vGx3 = _mm256_add_epi16(vGx3, vB2);
    				// vGx1 = _mm256_add_epi16(vGx1, vB2);
    				// vGx2 = _mm256_add_epi16(vGx2, vB3);
    				// vGx3 = _mm256_add_epi16(vGx3, vB4);
    
    				__asm__ volatile (
    					"vpaddw %[vB1], %[vB1], %[vGx1]\n\t"
    					"vpaddw %[vB2], %[vB2], %[vGx2]\n\t"
    					"vpaddw %[vB3], %[vB3], %[vGx3]\n\t"
    
    					"vpsubw %[vB0], %[vB2], %[vGy1]\n\t"
    					"vpsubw %[vB1], %[vB3], %[vGy2]\n\t"
    					"vpsubw %[vB2], %[vB4], %[vGy3]\n\t"
    
    					"vpaddw %[vB0], %[vGx1], %[vGx1]\n\t"
    					"vpaddw %[vB1], %[vGx2], %[vGx2]\n\t"
    					"vpaddw %[vB2], %[vGx3], %[vGx3]\n\t"
    					"vpaddw %[vB2], %[vGx1], %[vGx1]\n\t"
    					"vpaddw %[vB3], %[vGx2], %[vGx2]\n\t"
    					"vpaddw %[vB4], %[vGx3], %[vGx3]\n\t"
    					:
    					    [vGx1] "+x"(vGx1),
    					    [vGx2] "+x"(vGx2),
    					    [vGx3] "+x"(vGx3),
    					    [vGy1] "+x"(vGy1),
    					    [vGy2] "+x"(vGy2),
    					    [vGy3] "+x"(vGy3)
    
    					:
    					    [vA0] "x"(vA0),
    					    [vA1] "x"(vA1),
    					    [vA2] "x"(vA2),
    					    [vA3] "x"(vA3),
    					    [vA4] "x"(vA4),
    
    					    [vB0] "x"(vB0),
    					    [vB1] "x"(vB1),
    					    [vB2] "x"(vB2),
    					    [vB3] "x"(vB3),
    					    [vB4] "x"(vB4)
    				);
    
    				vA0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 0) * TILE_COLS + y]));
    				vA1 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 1) * TILE_COLS + y]));
    				vA2 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 2) * TILE_COLS + y]));
    				vA3 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 3) * TILE_COLS + y]));
    				vA4 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 4) * TILE_COLS + y]));
    
    				// vGx1 = _mm256_sub_epi16(vGx1, vA0);
    				// vGx2 = _mm256_sub_epi16(vGx2, vA1);
    				// vGx3 = _mm256_sub_epi16(vGx3, vA2);
    
    				// vGy1 = _mm256_sub_epi16(vGy1, vA0);
    				// vGy2 = _mm256_sub_epi16(vGy2, vA1);
    				// vGy3 = _mm256_sub_epi16(vGy3, vA2);
    
    				// vGx1 = _mm256_sub_epi16(vGx1, vA2);
    				// vGx2 = _mm256_sub_epi16(vGx2, vA3);
    				// vGx3 = _mm256_sub_epi16(vGx3, vA4);
    
    				// vGy1 = _mm256_add_epi16(vGy1, vA2);
    				// vGy2 = _mm256_add_epi16(vGy2, vA3);
    				// vGy3 = _mm256_add_epi16(vGy3, vA4);
    
    				// vGx1 = _mm256_sub_epi16(vGx1, vA1);
    				// vGx2 = _mm256_sub_epi16(vGx2, vA2);
    				// vGx3 = _mm256_sub_epi16(vGx3, vA3);
    				// vGx1 = _mm256_sub_epi16(vGx1, vA1);
    				// vGx2 = _mm256_sub_epi16(vGx2, vA2);
    				// vGx3 = _mm256_sub_epi16(vGx3, vA3);
    
    				__asm__ volatile (
    					"vpsubw %[vA0], %[vGx1], %[vGx1]\n\t"
    					"vpsubw %[vA1], %[vGx2], %[vGx2]\n\t"
    					"vpsubw %[vA2], %[vGx3], %[vGx3]\n\t"
    
    					"vpsubw %[vA0], %[vGy1], %[vGy1]\n\t"
    					"vpsubw %[vA1], %[vGy2], %[vGy2]\n\t"
    					"vpsubw %[vA2], %[vGy3], %[vGy3]\n\t"
    
    					"vpsubw %[vA2], %[vGx1], %[vGx1]\n\t"
    					"vpsubw %[vA3], %[vGx2], %[vGx2]\n\t"
    					"vpsubw %[vA4], %[vGx3], %[vGx3]\n\t"
    
    
    					"vpaddw %[vA2], %[vGy1], %[vGy1]\n\t"
    					"vpaddw %[vA3], %[vGy2], %[vGy2]\n\t"
    					"vpaddw %[vA4], %[vGy3], %[vGy3]\n\t"
    
    
    					"vpsubw %[vA1], %[vGx1], %[vGx1]\n\t"
    					"vpsubw %[vA2], %[vGx2], %[vGx2]\n\t"
    					"vpsubw %[vA3], %[vGx3], %[vGx3]\n\t"
    					"vpsubw %[vA1], %[vGx1], %[vGx1]\n\t"
    					"vpsubw %[vA2], %[vGx2], %[vGx2]\n\t"
    					"vpsubw %[vA3], %[vGx3], %[vGx3]\n\t"
    					:
    					    [vGx1] "+x"(vGx1),
    					    [vGx2] "+x"(vGx2),
    					    [vGx3] "+x"(vGx3),
    					    [vGy1] "+x"(vGy1),
    					    [vGy2] "+x"(vGy2),
    					    [vGy3] "+x"(vGy3)
    
    					:
    					    [vA0] "x"(vA0),
    					    [vA1] "x"(vA1),
    					    [vA2] "x"(vA2),
    					    [vA3] "x"(vA3),
    					    [vA4] "x"(vA4),
    
    					    [vB0] "x"(vB0),
    					    [vB1] "x"(vB1),
    					    [vB2] "x"(vB2),
    					    [vB3] "x"(vB3),
    					    [vB4] "x"(vB4)
    				);
					    
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gx[i * N + j * TILE_ROWS + (x + 0) * TILE_COLS + y + 1]), vGx1);
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gx[i * N + j * TILE_ROWS + (x + 1) * TILE_COLS + y + 1]), vGx2);
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gx[i * N + j * TILE_ROWS + (x + 2) * TILE_COLS + y + 1]), vGx3);

    				vB0 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 0) * TILE_COLS + y + 1]));
    				vB1 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 1) * TILE_COLS + y + 1]));
    				vB2 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 2) * TILE_COLS + y + 1]));
    				vB3 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 3) * TILE_COLS + y + 1]));
    				vB4 = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&Img[i * N + j * TILE_ROWS + (x + 4) * TILE_COLS + y + 1]));
    
    				// vGy1 = _mm256_sub_epi16(vGy1, vB0);
    				// vGy2 = _mm256_sub_epi16(vGy2, vB1);
    				// vGy3 = _mm256_sub_epi16(vGy3, vB2);
    				// vGy1 = _mm256_add_epi16(vGy1, vB2);
    				// vGy2 = _mm256_add_epi16(vGy2, vB3);
    				// vGy3 = _mm256_add_epi16(vGy3, vB4);
					// vGy1 = _mm256_sub_epi16(vGy1, vB0);
    				// vGy2 = _mm256_sub_epi16(vGy2, vB1);
    				// vGy3 = _mm256_sub_epi16(vGy3, vB2);
    				// vGy1 = _mm256_add_epi16(vGy1, vB2);
    				// vGy2 = _mm256_add_epi16(vGy2, vB3);
    				// vGy3 = _mm256_add_epi16(vGy3, vB4);
    
    				__asm__ volatile (
    					"vpsubw %[vB0], %[vGy1], %[vGy1]\n\t"
    					"vpsubw %[vB1], %[vGy2], %[vGy2]\n\t"
    					"vpsubw %[vB2], %[vGy3], %[vGy3]\n\t"
    					"vpaddw %[vB2], %[vGy1], %[vGy1]\n\t"
    					"vpaddw %[vB3], %[vGy2], %[vGy2]\n\t"
    					"vpaddw %[vB4], %[vGy3], %[vGy3]\n\t"
						"vpsubw %[vB0], %[vGy1], %[vGy1]\n\t"
    					"vpsubw %[vB1], %[vGy2], %[vGy2]\n\t"
    					"vpsubw %[vB2], %[vGy3], %[vGy3]\n\t"
    					"vpaddw %[vB2], %[vGy1], %[vGy1]\n\t"
    					"vpaddw %[vB3], %[vGy2], %[vGy2]\n\t"
    					"vpaddw %[vB4], %[vGy3], %[vGy3]\n\t"
    					:
    					    [vGx1] "+x"(vGx1),
    					    [vGx2] "+x"(vGx2),
    					    [vGx3] "+x"(vGx3),
    					    [vGy1] "+x"(vGy1),
    					    [vGy2] "+x"(vGy2),
    					    [vGy3] "+x"(vGy3)
    
    					:
    					    [vA0] "x"(vA0),
    					    [vA1] "x"(vA1),
    					    [vA2] "x"(vA2),
    					    [vA3] "x"(vA3),
    					    [vA4] "x"(vA4),
    
    					    [vB0] "x"(vB0),
    					    [vB1] "x"(vB1),
    					    [vB2] "x"(vB2),
    					    [vB3] "x"(vB3),
    					    [vB4] "x"(vB4)
    				);
    
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gy[i * N + j * TILE_ROWS + (x + 0) * TILE_COLS + y + 1]), vGy1);
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gy[i * N + j * TILE_ROWS + (x + 1) * TILE_COLS + y + 1]), vGy2);
    				_mm256_storeu_epi16(reinterpret_cast<__m256i*>(&Gy[i * N + j * TILE_ROWS + (x + 2) * TILE_COLS + y + 1]), vGy3);
    			}
			}
		}
	}
}
