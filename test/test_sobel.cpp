#include "../include/Sobel.hpp"
#include <cstdio>
#include <vector>

#define MAX_FREQ 3.2
#define BASE_FREQ 2.1

#define MM 130
#define NN 130

#define NUM_ITER 1e6

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


int main() {
    std::vector<int16_t> Gx(MM * NN);
    std::vector<int16_t> Gy(MM * NN);
    std::vector<uint16_t> Src(MM * NN);
    
    for (int i = 0; i < MM * NN; i++) {
    	Src[i] = (uint16_t)((i * 37 + 13) & 0xFF); // pseudo-random pattern
    }

    uint64_t r0, r1;
    uint64_t sum = 0;
    for (int i = 0; i < NUM_ITER; i++) {
        for (int col = 0; col < NN - 17; col += 16) {
            for (int row = 0; row < MM - 5; row += 4) {
                r0 = rdtsc();
                uint16_t* Img = Src.data() + NN * row + col;
                int16_t* gx = Gx.data() + NN * row + col;
                int16_t* gy = Gy.data() + NN * row + col;
                SOBEL_TILE(gx + NN, gx + NN * 2, gx + NN * 3, gx + NN * 4, gy + NN, gy + NN * 2, gy + NN * 3, gy + NN * 4, Img, Img + NN, Img + NN * 2, Img + NN * 3, Img + NN * 4, Img + NN * 5);
                r1 = rdtsc();
                sum += (r1 - r0);
            }
        }
    }
    printf("Throughput: %lf\n", ((double)(16.0 * NUM_ITER * (MM - 2) * (NN - 2)) / ((double)(sum)*MAX_FREQ/BASE_FREQ)));

}
